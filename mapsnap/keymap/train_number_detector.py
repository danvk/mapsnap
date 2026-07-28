"""Train the tiny page-number localizer on labeled key maps.

Builds a binary patch dataset from every ``<stem>.labels.json`` in a directory (positives
centered on labels, negatives sampled away from them; see mapsnap.keymap.keymap_patches), holds
out one whole image for validation (so the metric reflects generalization to an unseen
scan), fine-tunes a pretrained MobileNetV3-small, and saves the best weights.

    uv run python -m mapsnap.keymap.train_number_detector --val-image chicago_il_1950_vol_1/p0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mapsnap.keymap.keymap_patches import (
    build_image_patches,
    keymap_key,
    labelled_keymaps,
    labels_path_for,
    load_label_points,
    scale_points,
    split_train_val,
    working_scale,
)
from mapsnap.keymap.number_model import (
    average_precision,
    build_model,
    eval_transform,
    select_device,
    train_transform,
)


def load_scaled_image(image_path: str, scale: float) -> np.ndarray:
    """Load an image and downscale it by ``scale`` (RGB uint8)."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        new_size = (round(rgb.width * scale), round(rgb.height * scale))
        rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        return np.asarray(rgb)


class PatchDataset(Dataset):
    """In-memory patch dataset; applies a transform per item."""

    def __init__(self, patches: list[np.ndarray], labels: list[int], transform) -> None:
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int):
        patch = self.transform(self.patches[index])
        return patch, torch.tensor([float(self.labels[index])])


def build_split(
    image_paths: list[Path], rng: np.random.Generator
) -> tuple[list[np.ndarray], list[int]]:
    """Concatenate patches/labels across the given images (each downscaled by SCALE)."""
    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    for image_path in image_paths:
        width, height, points = load_label_points(str(labels_path_for(str(image_path))))
        if not points:
            continue
        factor = working_scale(width, height)
        image = load_scaled_image(str(image_path), factor)
        scaled = scale_points(points, factor)
        patches, labels = build_image_patches(image, scaled, rng=rng)
        all_patches.extend(patches)
        all_labels.extend(labels)
    return all_patches, all_labels


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Average precision of the model over a loader."""
    model.eval()
    scores: list[float] = []
    targets: list[float] = []
    for patches, labels in loader:
        logits = model(patches.to(device)).squeeze(1)
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        targets.extend(labels.squeeze(1).numpy().tolist())
    return average_precision(np.array(scores), np.array(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the page-number localizer CNN.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Corpus root; truth is found at <volume>/raw/truth/*.labels.json.",
    )
    parser.add_argument(
        "--val-image",
        default="chicago_il_1950_vol_1/p0",
        help=(
            "Key map (<volume>/<stem>) held out for validation and "
            "best-checkpoint selection (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        metavar="KEYS",
        help=(
            "Comma-separated <volume>/<stem> key maps to exclude from training "
            "entirely (a leave-one-volume-out fold; distinct from --val-image, "
            "which drives best-checkpoint selection)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models/number_detector.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device()

    images = [image for image, _ in labelled_keymaps(args.data_dir)]
    excluded = {k.strip() for k in args.exclude.split(",") if k.strip()}
    try:
        train_images, val_images = split_train_val(images, args.val_image, excluded)
    except ValueError as exc:
        sys.exit(str(exc))
    print(
        f"device={device}  train={[keymap_key(p) for p in train_images]}  "
        f"val={args.val_image}  excluded={sorted(excluded)}",
        file=sys.stderr,
    )

    train_patches, train_labels = build_split(train_images, rng)
    val_patches, val_labels = build_split(val_images, rng)
    print(
        f"train patches={len(train_patches)} (pos={sum(train_labels)})  "
        f"val patches={len(val_patches)} (pos={sum(val_labels)})",
        file=sys.stderr,
    )

    train_loader = DataLoader(
        PatchDataset(train_patches, train_labels, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PatchDataset(val_patches, val_labels, eval_transform()),
        batch_size=args.batch_size,
    )

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for patches, labels in train_loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(patches), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * patches.size(0)
        val_ap = evaluate(model, val_loader, device)
        marker = ""
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(model.state_dict(), args.out)
            marker = " *saved"
        print(
            f"epoch {epoch:2d}  train_loss={epoch_loss / len(train_patches):.4f}  "
            f"val_AP={val_ap:.4f}{marker}",
            file=sys.stderr,
        )

    print(f"best val AP={best_ap:.4f}  weights -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
