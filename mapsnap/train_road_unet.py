"""Train the road-corridor UNet on auto-labels from already-georeferenced pages.

Every fitted page provides a free (image, road-mask) training pair by projecting OSM
centerlines through its georef transform (mapsnap.road_model.rasterize_road_mask). Train on
several volumes and validate on a held-out volume to measure generalization to unseen
lithography — the val volume is exactly the kind of volume the model will be used on.

    uv run python -m mapsnap.train_road_unet \\
        data/chicago_il_1950_vol_1 data/washington_dc_1916_vol_2 \\
        data/new_orleans_la_1896_vol_2 data/kansas_city_mo_1951_vol_4 \\
        --val data/hudson_co_nj_1950_vol_9
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from mapsnap.keymap.number_model import select_device
from mapsnap.road_model import (
    PATCH,
    UNet,
    effective_gcp_count,
    normalize_patch,
    rasterize_road_mask,
)
from mapsnap.utils import image_stem

# Random patches sampled from each page per epoch.
PATCHES_PER_PAGE = 6
# Fraction of training patches constrained to the page-margin band, where the
# seam strips the edge-join matcher depends on live.
EDGE_FRACTION = 0.5
# Width of that band in pixels (~75m at the 25%-scale ~0.24 m/px).
EDGE_BAND_PX = 320


def volume_pages(volume: Path, min_effective_gcps: int) -> list[tuple[Path, dict]]:
    """(image path, georef) for each fitted, non-split, well-fit page.

    Pages below the effective-GCP gate are dropped: their OSM projection is
    the training label, and a fragile fit paints roads in the wrong place.
    """
    pages = []
    for path in sorted(volume.glob("p*.georef.json")):
        stem = image_stem(str(path))
        if "__" in stem:
            continue
        image = volume / f"{stem}.jpg"
        if not image.exists():
            continue
        georef = json.load(open(path))
        if effective_gcp_count(georef) < min_effective_gcps:
            continue
        pages.append((image, georef))
    return pages


def cached_mask(
    image_path: Path, georef: dict, features: list[dict], cache_dir: Path
) -> np.ndarray:
    """The page's auto-label mask, rasterized once and cached as PNG."""
    cache = cache_dir / f"{image_path.parent.name}_{image_path.stem}.png"
    if cache.exists():
        cached = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
        if cached is not None:
            return cached
    mask = rasterize_road_mask(georef, features)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), mask)
    return mask


def sample_patch(
    gray: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One augmented training patch: random crop + random 90-degree rotation and flip.

    The rotation augmentation matters here: adjacent Sanborn sheets are drawn grid-aligned
    rather than north-up, so at inference the model sees corridors at arbitrary angles.
    EDGE_FRACTION of patches are pinned to the page-margin band: the edge-join matcher
    reads the model exclusively in seam strips, where content is sketchier (duplicated
    margin blocks, big sheet numbers) than in the page interior.
    """
    height, width = gray.shape
    y_max, x_max = max(1, height - PATCH), max(1, width - PATCH)
    if rng.random() < EDGE_FRACTION:
        side = int(rng.integers(0, 4))
        band_y = min(EDGE_BAND_PX, y_max)
        band_x = min(EDGE_BAND_PX, x_max)
        if side == 0:  # top
            y, x = rng.integers(0, band_y), rng.integers(0, x_max)
        elif side == 1:  # bottom
            y, x = rng.integers(y_max - band_y, y_max), rng.integers(0, x_max)
        elif side == 2:  # left
            y, x = rng.integers(0, y_max), rng.integers(0, band_x)
        else:  # right
            y, x = rng.integers(0, y_max), rng.integers(x_max - band_x, x_max)
    else:
        y = rng.integers(0, y_max)
        x = rng.integers(0, x_max)
    patch = gray[y : y + PATCH, x : x + PATCH]
    label = mask[y : y + PATCH, x : x + PATCH]
    k = int(rng.integers(0, 4))
    if k:
        patch = np.rot90(patch, k)
        label = np.rot90(label, k)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
        label = np.fliplr(label)
    return np.ascontiguousarray(patch), np.ascontiguousarray(label)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss: overlap-based, robust to the road/background class imbalance."""
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * intersection + 1.0) / (denominator + 1.0)).mean()


def batch_tensors(
    patches: list[tuple[np.ndarray, np.ndarray]], device
) -> tuple[torch.Tensor, torch.Tensor]:
    images = np.stack([normalize_patch(p) for p, _ in patches])[:, None]
    labels = np.stack([(label > 127).astype(np.float32) for _, label in patches])[
        :, None
    ]
    return (
        torch.from_numpy(images).to(device),
        torch.from_numpy(labels).to(device),
    )


def validation_iou(
    model: nn.Module,
    pages: list[tuple[np.ndarray, np.ndarray]],
    device,
) -> tuple[float, float]:
    """(overall IoU, edge-band IoU) at threshold 0.5 over a fixed patch grid.

    The edge-band figure counts only patches touching the outer EDGE_BAND_PX
    of the page — the seam strips the edge-join matcher actually reads.
    """
    model.eval()
    intersection = union = 0.0
    edge_intersection = edge_union = 0.0

    def grid_positions(extent: int) -> list[int]:
        # Cover the full extent: regular stride plus a final patch flush with
        # the far edge (the edge band is exactly what the edge IoU measures).
        positions = list(range(0, extent - PATCH + 1, PATCH))
        if positions and positions[-1] != extent - PATCH:
            positions.append(extent - PATCH)
        return positions

    with torch.no_grad():
        for gray, mask in pages:
            height, width = gray.shape
            for y in grid_positions(height):
                for x in grid_positions(width):
                    patch = gray[y : y + PATCH, x : x + PATCH]
                    label = mask[y : y + PATCH, x : x + PATCH] > 127
                    tensor = (
                        torch.from_numpy(normalize_patch(patch))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    predicted = torch.sigmoid(model(tensor))[0, 0].cpu().numpy() > 0.5
                    patch_intersection = float(np.logical_and(predicted, label).sum())
                    patch_union = float(np.logical_or(predicted, label).sum())
                    intersection += patch_intersection
                    union += patch_union
                    on_edge = (
                        y < EDGE_BAND_PX
                        or x < EDGE_BAND_PX
                        or y + PATCH > height - EDGE_BAND_PX
                        or x + PATCH > width - EDGE_BAND_PX
                    )
                    if on_edge:
                        edge_intersection += patch_intersection
                        edge_union += patch_union
    model.train()
    return (
        intersection / union if union else 0.0,
        edge_intersection / edge_union if edge_union else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "volumes", nargs="+", type=Path, help="Training volume directories."
    )
    parser.add_argument(
        "--val", type=Path, required=True, help="Held-out validation volume."
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=24, help="UNet channel width.")
    parser.add_argument(
        "--min-effective-gcps",
        type=int,
        default=3,
        help="Drop training pages with fewer distinct inlier intersections.",
    )
    parser.add_argument(
        "--limit-pages", type=int, default=0, help="Per volume, for smoke tests."
    )
    parser.add_argument(
        "--val-pages", type=int, default=10, help="Validation pages sampled."
    )
    parser.add_argument("--output", type=Path, default=Path("models/road_unet.pt"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/roadlabel_cache"),
        help="Auto-label PNG cache (safe to delete).",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}", file=sys.stderr)

    train_set: list[tuple[Path, dict, list[dict]]] = []
    for volume in args.volumes:
        features = json.load(open(volume / "centerlines.geojson"))["features"]
        pages = volume_pages(volume, args.min_effective_gcps)
        if args.limit_pages:
            pages = pages[: args.limit_pages]
        train_set.extend((image, georef, features) for image, georef in pages)
        print(f"  {volume.name}: {len(pages)} pages", file=sys.stderr)
    print(f"training pages: {len(train_set)}", file=sys.stderr)

    val_features = json.load(open(args.val / "centerlines.geojson"))["features"]
    all_val_pages = volume_pages(args.val, args.min_effective_gcps)
    val_pages_meta = all_val_pages[:: max(1, len(all_val_pages) // args.val_pages)]
    val_pages = []
    for image, georef in val_pages_meta[: args.val_pages]:
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        mask = cached_mask(image, georef, val_features, args.cache_dir)
        val_pages.append((gray, mask))
    # An empty validation set would train to completion but never save a
    # checkpoint (0 IoU never beats 0); fail before spending hours.
    assert val_pages, (
        f"no usable validation pages in {args.val} "
        f"(min_effective_gcps={args.min_effective_gcps})"
    )
    print(f"validation pages: {len(val_pages)} from {args.val}", file=sys.stderr)

    model = UNet(base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        order = rng.permutation(len(train_set))
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        losses = []
        for index in order:
            image, georef, features = train_set[index]
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            mask = cached_mask(image, georef, features, args.cache_dir)
            for _ in range(PATCHES_PER_PAGE):
                pending.append(sample_patch(gray, mask, rng))
                if len(pending) == args.batch_size:
                    images, labels = batch_tensors(pending, device)
                    pending = []
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = bce(logits, labels) + dice_loss(logits, labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
        iou, edge_iou = validation_iou(model, val_pages, device)
        marker = ""
        # The edge band is where the matcher reads the model; select on it.
        if edge_iou > best_iou:
            best_iou = edge_iou
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            marker = "  (saved)"
        print(
            f"epoch {epoch:2d}: loss {np.mean(losses):.4f}  val IoU {iou:.3f}"
            f"  edge IoU {edge_iou:.3f}  [{time.time() - start:.0f}s]{marker}",
            file=sys.stderr,
        )
    print(f"best val edge IoU {best_iou:.3f} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
