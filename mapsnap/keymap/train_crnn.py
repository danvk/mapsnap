"""Train the CRNN page-number recognizer on labeled key maps.

For every ``<stem>.labels.json`` it crops a fixed strip around each labeled center (at the
image's native resolution; see mapsnap.crnn_model.number_strip) paired with that number's
digit string, holds out one whole image for validation, and trains the CRNN with CTC.
Runs on CPU (the model and data are small, and CTC loss is most portable there).

    uv run python -m mapsnap.keymap.train_crnn --val-image chicago-p0b
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mapsnap.keymap.crnn_model import (
    BLANK_INDEX,
    BOX_HALF_H_WORKING,
    BOX_HALF_W_WORKING,
    build_crnn,
    decode_batch,
    encode_text,
    eval_transform,
    number_strip,
    train_transform,
)
from mapsnap.keymap.keymap_patches import (
    crop_excludes_numbers,
    labels_path_for,
    load_label_points,
    working_scale,
)
from mapsnap.utils import image_stem

# Default empty-target negative strips sampled per positive, so the CRNN learns to emit
# nothing (-> rejected) on the localizer's false positives instead of inventing a number.
DEFAULT_NEGATIVE_RATIO = 0.5

# Fraction of negatives drawn from the most colorful (high-saturation) safe locations —
# the pastel gaps between numbers where the localizer tends to false-positive.
COLORFUL_FRAC = 0.6


def labeled_images(data_dir: Path) -> list[Path]:
    """Image files in ``data_dir`` that have a sibling .labels.json."""
    images = []
    for label_file in sorted(data_dir.glob("*.labels.json")):
        stem = label_file.name.split(".")[0]
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = data_dir / (stem + ext)
            if candidate.exists():
                images.append(candidate)
                break
    return images


def sample_negative_strips(
    image: np.ndarray,
    points: list[tuple[float, float, str]],
    factor: float,
    count: int,
    rng: np.random.Generator,
    *,
    colorful_frac: float = COLORFUL_FRAC,
) -> list[np.ndarray]:
    """Empty-target negative strips at safe centers, biased toward colorful (pastel) areas.

    Safe = the crop contains no part of a real page number (crop_excludes_numbers). Among
    safe candidates, the most saturated locations are preferred since that is where the
    localizer false-positives, with the remainder filled randomly for coverage.
    """
    if count <= 0:
        return []
    height, width = image.shape[:2]
    label_centers = [(x * factor, y * factor) for x, y, _ in points]
    crop_half_w = BOX_HALF_W_WORKING
    crop_half_h = BOX_HALF_H_WORKING

    candidates: list[tuple[int, int]] = []
    for _ in range(count * 40):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        if crop_excludes_numbers(
            cx * factor,
            cy * factor,
            label_centers,
            crop_half_w=crop_half_w,
            crop_half_h=crop_half_h,
        ):
            candidates.append((cx, cy))
        if len(candidates) >= count * 8:
            break
    if not candidates:
        return []

    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]

    def colorfulness(center: tuple[int, int]) -> float:
        cx, cy = center
        x0, x1 = max(0, cx - 10), min(width, cx + 10)
        y0, y1 = max(0, cy - 10), min(height, cy + 10)
        return float(saturation[y0:y1, x0:x1].mean())

    candidates.sort(key=colorfulness, reverse=True)
    n_colorful = min(len(candidates), round(count * colorful_frac))
    picked = candidates[:n_colorful]
    rest = candidates[n_colorful:]
    rng.shuffle(rest)
    picked += rest[: count - len(picked)]
    return [number_strip(image, cx, cy, factor) for cx, cy in picked]


def build_split(
    image_paths: list[Path],
    *,
    negative_ratio: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Strips and digit-string labels for every page number across the given images.

    With ``negative_ratio`` > 0, also adds that fraction (per image) of empty-target ("")
    negative strips so the CRNN learns to reject non-numbers (see sample_negative_strips).
    """
    strips: list[np.ndarray] = []
    texts: list[str] = []
    for image_path in image_paths:
        width, height, points = load_label_points(str(labels_path_for(str(image_path))))
        if not points:
            continue
        factor = working_scale(width, height)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        positives = 0
        for px, py, text in points:
            if not encode_text(text):
                continue
            strips.append(number_strip(image, px, py, factor))
            texts.append("".join(c for c in text if c.isdigit()))
            positives += 1
        if negative_ratio > 0 and rng is not None:
            negatives = sample_negative_strips(
                image, points, factor, round(negative_ratio * positives), rng
            )
            strips.extend(negatives)
            texts.extend([""] * len(negatives))
    return strips, texts


class StripDataset(Dataset):
    """Grayscale strips with their digit-string labels; applies a transform per item."""

    def __init__(self, strips: list[np.ndarray], texts: list[str], transform) -> None:
        self.strips = strips
        self.texts = texts
        self.transform = transform

    def __len__(self) -> int:
        return len(self.strips)

    def __getitem__(self, index: int):
        image = self.transform(self.strips[index])
        target = torch.tensor(encode_text(self.texts[index]), dtype=torch.long)
        return image, target, self.texts[index]


def collate(batch):
    """Stack images; concatenate CTC targets with their lengths; keep gt strings."""
    images = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.tensor([b[1].numel() for b in batch], dtype=torch.long)
    texts = [b[2] for b in batch]
    return images, targets, target_lengths, texts


@torch.no_grad()
def exact_match(model: nn.Module, loader: DataLoader) -> float:
    """Fraction of validation strips whose greedy-CTC decode equals the ground truth."""
    model.eval()
    correct = total = 0
    for images, _, _, texts in loader:
        preds = decode_batch(model(images))
        correct += sum(p == t for p, t in zip(preds, texts))
        total += len(texts)
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CRNN page-number recognizer."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/keymaps"))
    parser.add_argument("--val-image", default="chicago-p0b")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=DEFAULT_NEGATIVE_RATIO,
        help="Empty-target negatives per positive, for reject (default %(default)s).",
    )
    parser.add_argument("--out", type=Path, default=Path("models/number_crnn.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")

    images = labeled_images(args.data_dir)
    val_images = [p for p in images if image_stem(str(p)) == args.val_image]
    train_images = [p for p in images if image_stem(str(p)) != args.val_image]
    if not val_images:
        sys.exit(f"--val-image {args.val_image!r} not found")

    train_strips, train_texts = build_split(
        train_images, negative_ratio=args.negative_ratio, rng=rng
    )
    val_strips, val_texts = build_split(
        val_images, negative_ratio=args.negative_ratio, rng=rng
    )
    neg_train = sum(1 for t in train_texts if not t)
    print(
        f"train strips={len(train_strips)} ({neg_train} neg) "
        f"val strips={len(val_strips)} (val={args.val_image})",
        file=sys.stderr,
    )

    train_loader = DataLoader(
        StripDataset(train_strips, train_texts, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        StripDataset(val_strips, val_texts, eval_transform()),
        batch_size=args.batch_size,
        collate_fn=collate,
    )

    model = build_crnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images_b, targets, target_lengths, _ in train_loader:
            images_b = images_b.to(device)
            log_probs = model(images_b)  # (T, N, C)
            input_lengths = torch.full(
                (images_b.size(0),), log_probs.size(0), dtype=torch.long
            )
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images_b.size(0)
        acc = exact_match(model, val_loader)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            marker = " *saved"
        print(
            f"epoch {epoch:2d}  loss={epoch_loss / len(train_strips):.4f}  "
            f"val_exact={acc:.3f}{marker}",
            file=sys.stderr,
        )

    print(
        f"best val exact-match={best_acc:.3f}  weights -> {args.out}", file=sys.stderr
    )


if __name__ == "__main__":
    main()
