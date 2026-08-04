"""Train the key-map road UNet on OSM auto-labels (#211).

Mirrors mapsnap.train_road_unet's loop, adapted to key maps: colour input,
per-sheet pixel stroke widths, soft labels, a validity mask confining the loss
to where supervision means anything (see mapsnap.keymap.road_prob), and
checkpoint selection on the buffered F1 the project's success criteria are
written against -- selecting on plain IoU would select for overfitting the
labels' own georef misalignment.

    uv run python -m mapsnap.keymap.train_road_prob \\
        --holdout detroit_mich_1929_vol_11 miami_fl_1950_vol_1 \\
                  grand_rapids_mi_1953_vol7 brooklyn_ny_1939_vol_1

Holdouts are whole VOLUMES: several volumes carry two near-identical key-map
sheets, so a sheet-level split leaks style and inflates the score.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from mapsnap.keymap.number_model import select_device
from mapsnap.keymap.road_prob import (
    KeymapSheet,
    buffered_scores,
    keymap_sheets,
    measure_stroke_px,
    normalize_bgr,
    predict_sheet,
    sheet_label,
)
from mapsnap.road_model import PATCH, ROAD_MODEL_PATH, UNet

PATCHES_PER_SHEET = 120
"""Patches sampled per sheet per epoch. Sheets are ~25x a page's area, and the
page model saw 6 patches per page; 120 gives a from-scratch run a comparable
optimizer-step budget (~2900 steps over 20 epochs on 19 sheets)."""

JITTER_GAIN = 0.08
JITTER_BRIGHTNESS = 0.10
"""Photometric augmentation: per-channel gain and overall brightness factors.
Covers paper-tint and wash-intensity variation between volumes without making
inference stateful the way per-sheet standardization would."""


@dataclass
class TrainSheet:
    """One sheet fully materialized in memory for patch sampling."""

    key: str
    image: np.ndarray  # HWC uint8 BGR
    label: np.ndarray  # HW uint8, soft target * 255
    valid: np.ndarray  # HW uint8 0/1
    centers_y: np.ndarray  # candidate patch-centre coordinates (valid pixels)
    centers_x: np.ndarray


def cached_label(
    sheet: KeymapSheet, stroke: int, cache_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    """(soft label, valid mask), from the PNG cache or rendered fresh.

    Cache keys include the stroke so a changed width measurement cannot serve
    stale labels. Rendering needs the volume's centerlines (LA's is 138 MB),
    so hits matter.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    label_path = cache_dir / f"keymap_{sheet.key()}_w{stroke}.png"
    valid_path = cache_dir / f"keymap_{sheet.key()}_w{stroke}_valid.png"
    if label_path.exists() and valid_path.exists():
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
        if label is not None and valid is not None:
            return label, valid
    features = json.loads(sheet.centerlines_path.read_text())["features"]
    label, valid = sheet_label(sheet, features, stroke)
    label8 = (np.clip(label, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(label_path), label8)
    cv2.imwrite(str(valid_path), valid * 255)
    return label8, (valid > 0).astype(np.uint8) * 255


def load_train_sheet(sheet: KeymapSheet, cache_dir: Path) -> TrainSheet | None:
    """Materialize one sheet, or None when it cannot contribute."""
    image = cv2.imread(str(sheet.image_path))
    if image is None:
        return None
    stroke = measure_stroke_px(image)
    label, valid = cached_label(sheet, stroke, cache_dir)
    valid01 = (valid > 0).astype(np.uint8)
    margin = PATCH // 2
    interior = np.zeros_like(valid01)
    interior[margin:-margin, margin:-margin] = valid01[margin:-margin, margin:-margin]
    ys, xs = np.nonzero(interior)
    if ys.size < 100:
        return None
    # Keep a manageable candidate pool; sampling hits it with replacement.
    keep = np.random.default_rng(0).choice(
        ys.size, size=min(ys.size, 200_000), replace=False
    )
    return TrainSheet(
        key=sheet.key(),
        image=image,
        label=label,
        valid=valid01,
        centers_y=ys[keep],
        centers_x=xs[keep],
    )


def sample_batch(
    sheets: list[TrainSheet], batch_size: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(images, soft labels, valid weights) for one optimizer step."""
    images = np.empty((batch_size, 3, PATCH, PATCH), np.float32)
    labels = np.empty((batch_size, 1, PATCH, PATCH), np.float32)
    weights = np.empty((batch_size, 1, PATCH, PATCH), np.float32)
    half = PATCH // 2
    for i in range(batch_size):
        sheet = sheets[int(rng.integers(len(sheets)))]
        pick = int(rng.integers(sheet.centers_y.size))
        y, x = int(sheet.centers_y[pick]), int(sheet.centers_x[pick])
        y0, x0 = y - half, x - half
        patch = sheet.image[y0 : y0 + PATCH, x0 : x0 + PATCH]
        label = sheet.label[y0 : y0 + PATCH, x0 : x0 + PATCH]
        valid = sheet.valid[y0 : y0 + PATCH, x0 : x0 + PATCH]

        rotations = int(rng.integers(4))
        if rotations:
            patch = np.rot90(patch, rotations)
            label = np.rot90(label, rotations)
            valid = np.rot90(valid, rotations)
        if rng.random() < 0.5:
            patch = np.fliplr(patch)
            label = np.fliplr(label)
            valid = np.fliplr(valid)

        patch = patch.astype(np.float32)
        gains = rng.uniform(1 - JITTER_GAIN, 1 + JITTER_GAIN, size=3)
        brightness = rng.uniform(1 - JITTER_BRIGHTNESS, 1 + JITTER_BRIGHTNESS)
        patch = np.clip(patch * gains[None, None, :] * brightness, 0, 255)

        images[i] = normalize_bgr(patch.astype(np.uint8))
        labels[i, 0] = label.astype(np.float32) / 255.0
        weights[i, 0] = valid.astype(np.float32)
    return (
        torch.from_numpy(images),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
    )


def masked_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """BCE + soft dice, confined to the valid supervision area."""
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, reduction="sum"
    ) / weights.sum().clamp(min=1.0)
    probabilities = torch.sigmoid(logits) * weights
    targets = targets * weights
    intersection = (probabilities * targets).sum()
    dice = 1 - (2 * intersection + 1) / (probabilities.sum() + targets.sum() + 1)
    return bce + dice


def warm_start_state(device) -> dict:
    """The page checkpoint adapted to colour input.

    conv1 saw one grayscale channel; replicating its weights across three
    channels at a third of the magnitude reproduces the grayscale response on
    a gray image, so training starts from the page model's behaviour exactly.
    """
    state = torch.load(str(ROAD_MODEL_PATH), map_location=device)
    conv1 = state["enc1.block.0.weight"]
    state["enc1.block.0.weight"] = conv1.repeat(1, 3, 1, 1) / 3.0
    return state


def evaluate(
    model: nn.Module,
    holdout: list[tuple[KeymapSheet, np.ndarray, np.ndarray, np.ndarray]],
    device,
) -> tuple[float, list[str]]:
    """(mean buffered F1, per-sheet report lines) over the holdout sheets."""
    scores = []
    lines = []
    for sheet, image, label, valid in holdout:
        probability = predict_sheet(model, image, device)
        result = buffered_scores(
            probability, label.astype(np.float32) / 255.0, valid, image
        )
        completeness, correctness = result["completeness"], result["correctness"]
        f1 = (
            2 * completeness * correctness / (completeness + correctness)
            if completeness + correctness
            else 0.0
        )
        scores.append(f1)
        lines.append(
            f"    {sheet.key():40} F1 {f1:.3f}  compl {completeness:.3f} "
            f"(fill {result.get('completeness_fill', 0):.3f} / "
            f"paper {result.get('completeness_paper', 0):.3f})  "
            f"corr {correctness:.3f}  iou {result['iou']:.3f}"
        )
    return float(np.mean(scores)) if scores else 0.0, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--holdout",
        nargs="+",
        default=[],
        metavar="VOLUME",
        help="Volume directory names whose sheets are held out for evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=24, help="UNet channel width.")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Initialize from the page road UNet (forces --base 32).",
    )
    parser.add_argument("--patches-per-sheet", type=int, default=PATCHES_PER_SHEET)
    parser.add_argument(
        "--eval-every", type=int, default=2, help="Epoch stride for holdout eval."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/keymap_road_unet.pt")
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/roadlabel_cache"),
        help="Auto-label PNG cache (safe to delete).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device()
    print(f"device: {device}", file=sys.stderr)

    all_sheets = keymap_sheets(args.data)
    holdout_names = set(args.holdout)
    unknown = holdout_names - {sheet.volume.name for sheet in all_sheets}
    if unknown:
        sys.exit(f"--holdout names no usable volume: {sorted(unknown)}")
    train_sheets_meta = [
        sheet for sheet in all_sheets if sheet.volume.name not in holdout_names
    ]
    holdout_meta = [sheet for sheet in all_sheets if sheet.volume.name in holdout_names]
    print(
        f"sheets: {len(train_sheets_meta)} train, {len(holdout_meta)} holdout",
        file=sys.stderr,
    )

    start = time.time()
    train_sheets = []
    for sheet in train_sheets_meta:
        loaded = load_train_sheet(sheet, args.cache_dir)
        if loaded is None:
            print(f"  skipping {sheet.key()}: no usable patches", file=sys.stderr)
            continue
        train_sheets.append(loaded)
        print(f"  loaded {loaded.key}", file=sys.stderr)
    assert train_sheets, "no training sheets"
    print(
        f"loaded {len(train_sheets)} sheets in {time.time() - start:.0f}s",
        file=sys.stderr,
    )

    holdout = []
    for sheet in holdout_meta:
        image = cv2.imread(str(sheet.image_path))
        if image is None:
            continue
        stroke = measure_stroke_px(image)
        label, valid = cached_label(sheet, stroke, args.cache_dir)
        holdout.append((sheet, image, label, (valid > 0).astype(np.uint8)))
    # Without a holdout no checkpoint would ever be saved; fail fast, like the
    # page trainer does.
    assert holdout, "no holdout sheets"

    base = 32 if args.warm_start else args.base
    model = UNet(base=base, in_channels=3)
    if args.warm_start:
        model.load_state_dict(warm_start_state(device))
        print("warm-started from the page road UNet", file=sys.stderr)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = max(
        1, len(train_sheets) * args.patches_per_sheet // args.batch_size
    )
    print(f"{steps_per_epoch} steps/epoch", file=sys.stderr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total = 0.0
        for _step in range(steps_per_epoch):
            images, labels, weights = sample_batch(train_sheets, args.batch_size, rng)
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = masked_loss(logits, labels.to(device), weights.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())

        line = (
            f"epoch {epoch:2d}  loss {total / steps_per_epoch:.4f}  "
            f"[{time.time() - epoch_start:.0f}s]"
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            f1, lines = evaluate(model, holdout, device)
            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output)
                marker = "  *saved*"
            print(f"{line}  holdout F1 {f1:.3f}{marker}", file=sys.stderr)
            for detail in lines:
                print(detail, file=sys.stderr)
        else:
            print(line, file=sys.stderr)

    print(f"best holdout buffered F1: {best_f1:.3f} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
