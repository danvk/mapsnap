"""Cache each page's road-UNet P(road) map as a ``<stem>.roadprob.jpg`` sidecar.

``mapsnap roadprob <image>...`` is the road-UNet counterpart of ``mapsnap craft``:
a stateless pass over *parent* page images whose output depends only on the image
and the model, so it can run on a GPU instance before ``mapsnap split`` and never
again. Where CRAFT's boxes for a panel are derived from its parent's boxes (#361),
a panel's P(road) map is derived by cropping its parent's with the panel polygon,
which is why this can precede the split at all: predicting on the parent and
cropping matches predicting on the panel to within the UNet's own tiling noise
(measured over 30 panels: median mean |difference| 0.013 and road-mask IoU 0.93,
against 0.014 / 0.92 for the same panel with the tile grid shifted 32 px). The two
differ only within ~64 px of a cut, where the parent-based map has the neighbouring
panel's real ink in its receptive field and the panel-based one sees the white fill
that ``split`` painted there -- so the derived map is the better of the two.

Maps are stored as quality-90 JPEG, 158 KB for a 25%-scale page against 684 KB for
the equivalent PNG, at a mean error of 0.0008 (0.2 of one 8-bit level) -- far below
anything the chamfer/NCC matching in ``snap`` responds to, and a 220 GB saving over
the corpus.

Readers go through :func:`load_roadprob`, which falls back to the pre-#354 location
(``artifacts/edge_join/roadprob/<stem>.png``) so volumes fitted before this change
keep working without a re-run.

    mapsnap roadprob data/hudson_co_nj_1950_vol_9/p*.jpg --resume
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Beside the image, like <stem>.boxes.json and <stem>.streets.json.
SUFFIX = "roadprob.jpg"
# See the module docstring: 4.3x smaller than PNG at 0.2/255 mean error.
QUALITY = 90
# Where P(road) maps lived before #354, still read so old volumes keep working.
LEGACY_DIR = ("artifacts", "edge_join", "roadprob")


def roadprob_path(image_path: Path | str) -> Path:
    """Path of the P(road) sidecar for ``image_path`` (``<stem>.roadprob.jpg``)."""
    path = Path(image_path)
    return path.parent / f"{path.stem}.{SUFFIX}"


def legacy_roadprob_path(image_path: Path | str) -> Path:
    """Pre-#354 path of a page's P(road) map, under the volume's artifacts/."""
    path = Path(image_path)
    return path.parent.joinpath(*LEGACY_DIR, f"{path.stem}.png")


def save_roadprob(path: Path, probability: np.ndarray) -> None:
    """Write a float P(road) map in [0,1] to ``path`` as an 8-bit JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = (np.clip(probability, 0.0, 1.0) * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), scaled, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])


def load_roadprob(image_path: Path | str) -> np.ndarray | None:
    """An image's cached P(road) map as float32 in [0,1], or None if there is none.

    Prefers the sidecar beside the image and falls back to the pre-#354 artifacts
    location, so a volume cached by an older run still loads.
    """
    for path in (roadprob_path(image_path), legacy_roadprob_path(image_path)):
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            return raw.astype(np.float32) / 255.0
    return None


def crop_to_panel(
    probability: np.ndarray, ring: list[tuple[float, float]]
) -> np.ndarray:
    """``probability`` cropped to one panel: its bounding box, zero outside the ring.

    Deliberately the same geometry ``split.write_panels`` uses to cut the panel's
    JPEG -- mask from the rounded ring, bounding box truncated at the low corner and
    rounded at the high one -- so the map lands in exactly the panel's pixel frame;
    a pixel of disagreement would offset everything ``snap`` matches against it. The
    only difference is the fill outside the ring: 0 (no road) rather than white paper.
    """
    height, width = probability.shape[:2]
    points = np.array([[round(x), round(y)] for x, y in ring], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
    x1, y1 = min(width, round(max(xs))), min(height, round(max(ys)))
    cropped = probability[y0:y1, x0:x1].copy()
    cropped[mask[y0:y1, x0:x1] == 0] = 0.0
    return cropped


def derive_panel_roadprob(
    image_path: Path, rings: list[list[tuple[float, float]]], base: str
) -> list[Path]:
    """Cut the parent's P(road) map into ``<base>__N.roadprob.jpg``, one per panel.

    A no-op returning [] when the parent has no cached map. ``rings`` are the panel
    polygons in the parent's pixel frame, in panel-number order (as written to
    ``<stem>.panels.json``).
    """
    parent = load_roadprob(image_path)
    if parent is None:
        return []
    written = []
    for index, ring in enumerate(rings, start=1):
        out_path = image_path.parent / f"{base}__{index}.{SUFFIX}"
        save_roadprob(out_path, crop_to_panel(parent, ring))
        written.append(out_path)
    return written


def volume_roadprob_images(volume: Path) -> list[str]:
    """Every image in a volume needing its own inference: the parent sheets.

    Split panels are excluded because ``split`` (and this module's own CLI) cuts
    them out of their parent's map, and key-map sheets because their P(road) comes
    from the colour model in ``mapsnap.keymap.road_prob``.
    """
    from mapsnap.page_adjacency import volume_page_images

    return [str(path) for path in volume_page_images(volume)]


def pending_images(images: list[str], resume: bool) -> list[str]:
    """Images still needing inference: all of them, or those without a fresh map."""
    if not resume:
        return images
    pending = []
    for image in images:
        path = roadprob_path(image)
        if not path.exists() or path.stat().st_mtime < Path(image).stat().st_mtime:
            pending.append(image)
    return pending


def panel_rings(image_path: Path) -> list[list[tuple[float, float]]]:
    """Panel polygons from ``<stem>.panels.json`` beside the image, or [] if unsplit."""
    from mapsnap.split import panels_json_path, read_panels_json

    path = panels_json_path(image_path)
    if not path.exists():
        return []
    return [
        [(x, y) for x, y in ring] for ring in read_panels_json(path).get("panels", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict each page's P(road) map and write <stem>.roadprob.jpg."
    )
    parser.add_argument(
        "images", nargs="+", metavar="IMAGE", help="Image paths or globs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose .roadprob.jpg is newer than the image.",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU.")
    args = parser.parse_args()

    import torch

    from mapsnap.craft import expand_images
    from mapsnap.keymap.number_model import select_device
    from mapsnap.road_model import ROAD_MODEL_PATH, load_model, predict_page

    images = expand_images(args.images)
    if not images:
        sys.exit(f"No images matched: {' '.join(args.images)}")
    missing = [image for image in images if not Path(image).exists()]
    if missing:
        sys.exit(f"Image not found: {missing[0]}")

    todo = pending_images(images, args.resume)
    if args.resume and len(todo) < len(images):
        print(
            f"Resuming: {len(todo)}/{len(images)} image(s) need P(road).",
            file=sys.stderr,
        )
    if not todo:
        print("All images already have a current P(road) map.", file=sys.stderr)
        return

    device = torch.device("cpu") if args.no_gpu else select_device()
    print(f"device: {device}", file=sys.stderr)
    model = load_model(ROAD_MODEL_PATH, device)
    predicted = derived = 0
    for image in todo:
        image_path = Path(image)
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip (unreadable): {image}", file=sys.stderr)
            continue
        save_roadprob(roadprob_path(image_path), predict_page(model, gray, device))
        predicted += 1
        # A page split before this ran (or re-run after a split) gets its panels'
        # maps here, so the two commands can be ordered either way.
        derived += len(
            derive_panel_roadprob(image_path, panel_rings(image_path), image_path.stem)
        )
    print(
        f"Wrote {predicted} P(road) map(s) (+{derived} derived for panels).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
