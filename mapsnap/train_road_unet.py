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
from mapsnap.road_model import PATCH, UNet, normalize_patch, rasterize_road_mask
from mapsnap.utils import image_stem

# Random patches sampled from each page per epoch.
PATCHES_PER_PAGE = 6


def volume_pages(volume: Path) -> list[tuple[Path, dict]]:
    """(image path, georef) for each fitted, non-split page of a volume."""
    pages = []
    for path in sorted(volume.glob("p*.georef.json")):
        stem = image_stem(str(path))
        if "__" in stem:
            continue
        image = volume / f"{stem}.jpg"
        if image.exists():
            pages.append((image, json.load(open(path))))
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
    """
    height, width = gray.shape
    y = rng.integers(0, max(1, height - PATCH))
    x = rng.integers(0, max(1, width - PATCH))
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
) -> float:
    """Mean IoU at threshold 0.5 over a fixed grid of patches from the val pages."""
    model.eval()
    intersection = union = 0.0
    with torch.no_grad():
        for gray, mask in pages:
            height, width = gray.shape
            for y in range(0, height - PATCH, PATCH):
                for x in range(0, width - PATCH, PATCH):
                    patch = gray[y : y + PATCH, x : x + PATCH]
                    label = mask[y : y + PATCH, x : x + PATCH] > 127
                    tensor = (
                        torch.from_numpy(normalize_patch(patch))
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    predicted = torch.sigmoid(model(tensor))[0, 0].cpu().numpy() > 0.5
                    intersection += float(np.logical_and(predicted, label).sum())
                    union += float(np.logical_or(predicted, label).sum())
    model.train()
    return intersection / union if union else 0.0


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
        pages = volume_pages(volume)
        if args.limit_pages:
            pages = pages[: args.limit_pages]
        train_set.extend((image, georef, features) for image, georef in pages)
    print(f"training pages: {len(train_set)}", file=sys.stderr)

    val_features = json.load(open(args.val / "centerlines.geojson"))["features"]
    val_pages_meta = volume_pages(args.val)[
        :: max(1, len(volume_pages(args.val)) // args.val_pages)
    ]
    val_pages = []
    for image, georef in val_pages_meta[: args.val_pages]:
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        mask = cached_mask(image, georef, val_features, args.cache_dir)
        val_pages.append((gray, mask))
    print(f"validation pages: {len(val_pages)} from {args.val}", file=sys.stderr)

    model = UNet().to(device)
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
        iou = validation_iou(model, val_pages, device)
        marker = ""
        if iou > best_iou:
            best_iou = iou
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            marker = "  (saved)"
        print(
            f"epoch {epoch:2d}: loss {np.mean(losses):.4f}  val IoU {iou:.3f}"
            f"  [{time.time() - start:.0f}s]{marker}",
            file=sys.stderr,
        )
    print(f"best val IoU {best_iou:.3f} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
