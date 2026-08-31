"""Run CRAFT text detection and cache the boxes, without recognizing anything.

CRAFT detection is the slowest stage of the pipeline, it depends on none of the
recognition parameters (vocabulary, min-short-side, centerlines), and it now has
several consumers: ``mapsnap ocr`` recognizes text inside these boxes,
``mapsnap adjacency`` reads adjacent-sheet numbers from them, and the key-map
passes read page numbers. Splitting it out (issue #132) lets adjacency run
*before* OCR, which is what allows the key-map assignment repair to inform the
OCR vocabulary rather than the other way round.

Every consumer requires ``<stem>.boxes.json`` to exist and fails with the
command to run; this is the only command that runs CRAFT.

    mapsnap craft 'data/detroit_mich_1929_vol_11/p*.jpg'
    mapsnap craft data/<volume>/raw/p0.jpg          # key-map sheets too
"""

import argparse
import glob
import sys
from pathlib import Path

from tqdm import tqdm

from mapsnap.detect_text import boxes_path, write_craft_boxes
from mapsnap.panel_boxes import derive_boxes_for_panel_image


def expand_images(patterns: list[str]) -> list[str]:
    """Shell-style globs expanded and de-duplicated, in input order.

    The pipelines quote their globs so a volume with hundreds of pages does not
    overflow the command line, so expansion happens here as well as in the shell.
    """
    seen: dict[str, None] = {}
    for pattern in patterns:
        matches = (
            sorted(glob.glob(pattern))
            if any(ch in pattern for ch in "*?[")
            else [pattern]
        )
        for match in matches:
            seen.setdefault(match, None)
    return list(seen)


def volume_craft_images(
    volume: Path, keymap_keys: list[str] | None = None
) -> list[str]:
    """Every image in a volume that some later step recognizes inside.

    The union matters: ``mapsnap ocr`` reads the *effective* pages (split
    panels supersede their parent), while ``mapsnap adjacency`` reads the
    *parent* sheets, because the printed margin references live on the parent
    even when panels supersede it downstream. Crafting only one of the two
    lists leaves a split volume's parents (Champaign: p2, p4, p13, p20, p21,
    p23) without boxes, and adjacency then refuses to run.
    """
    from mapsnap.page_adjacency import volume_page_images
    from mapsnap.utils import list_pages

    seen: dict[str, None] = {}
    for path in [*list_pages(volume), *volume_page_images(volume)]:
        seen.setdefault(str(path), None)
    for key in keymap_keys or []:
        raw = volume / "raw" / f"{key}.jpg"
        if raw.exists():
            seen.setdefault(str(raw), None)
    return list(seen)


def pending_images(images: list[str], resume: bool) -> list[str]:
    """Images still needing CRAFT: all of them, or those without fresh boxes."""
    if not resume:
        return images
    return [
        image
        for image in images
        if not Path(boxes_path(image)).exists()
        or Path(boxes_path(image)).stat().st_mtime < Path(image).stat().st_mtime
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CRAFT text detection and write <stem>.boxes.json."
    )
    parser.add_argument(
        "images", nargs="+", metavar="IMAGE", help="Image paths or globs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose .boxes.json is newer than the image.",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU.")
    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        metavar="PX",
        help="CRAFT minimum text-box size (default: %(default)s).",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        metavar="T",
        help="CRAFT link threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--craft-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Detect at this fraction of full resolution (default: %(default)s).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2560,
        metavar="PX",
        help=(
            "Tile images whose long side exceeds this, so small labels survive "
            "EasyOCR's canvas downscaling (default: %(default)s; 0 disables)."
        ),
    )
    args = parser.parse_args()

    images = expand_images(args.images)
    if not images:
        sys.exit(f"No images matched: {' '.join(args.images)}")
    missing = [image for image in images if not Path(image).exists()]
    if missing:
        sys.exit(f"Image not found: {missing[0]}")

    todo = pending_images(images, args.resume)
    if args.resume and len(todo) < len(images):
        print(
            f"Resuming: {len(todo)}/{len(images)} image(s) need CRAFT.",
            file=sys.stderr,
        )
    if not todo:
        print("All images already have current boxes.", file=sys.stderr)
        return

    # Parents before panels, so a panel's derivation (#361) can see boxes its
    # parent wrote earlier in this same invocation.
    todo.sort(key=lambda image: ("__" in Path(image).stem, image))
    derived = 0
    detect: list[str] = []
    for image in todo:
        if derive_boxes_for_panel_image(image):
            derived += 1
        else:
            detect.append(image)
    if derived:
        print(
            f"Derived {derived} panel box file(s) from parents (#361).",
            file=sys.stderr,
        )
    if detect:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)
        for image in tqdm(detect, smoothing=0):
            write_craft_boxes(
                image,
                reader,
                min_size=args.min_size,
                link_threshold=args.link_threshold,
                craft_scale=args.craft_scale,
                tile_size=args.tile_size,
            )
    print(
        f"Wrote boxes for {len(todo)} image(s) ({derived} derived).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
