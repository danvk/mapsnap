"""Train the cutline UNet (mapsnap.cutline_model) on OIM volunteers' cutlines.

Labels are free: every page OldInsuranceMaps volunteers split comes with the
cutlines they drew (``labels/<image stem>.cutlines.json``), rasterized here as
LINE_PX-wide strokes on the page letterboxed to the model's input size. Pages
volunteers marked as unsplit are all-zero masks.

The data is a cutline benchmark directory, as ``score_splits_oim.py --manifest``
reads it: ``manifest.tsv`` (image, item, label, fold, ...), ``images/`` and
``labels/``. The shipped weights were trained on the 4,000-page benchmark in
``~/Documents/mapsnap/cutline-training``: 2,000 split pages whose cuts follow
printed ink and 2,000 pages volunteers marked unsplit, from complete and
unfinished OIM volumes, San Francisco excluded; one volume in five is the
``test`` fold. Training never sees the test fold, and holds out one train-fold
volume in eight (``val``) to pick the best epoch.

    uv run python -m mapsnap.train_cutline_unet prep ~/Documents/mapsnap/cutline-training
    uv run python -m mapsnap.train_cutline_unet train ~/Documents/mapsnap/cutline-training

Prep letterboxes every page and rasterizes its mask once (``<benchmark>/model/
data-768/``); training then reads those. 15 epochs at base width 16 take about
five hours on an M2's GPU. Score the result with
``MAPSNAP_CUTLINE_WEIGHTS=<weights> scripts/score_splits_oim.py --manifest ...``.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from mapsnap.cutline_model import INPUT_SIZE, MODEL_PATH, letterbox
from mapsnap.keymap.number_model import select_device
from mapsnap.road_model import UNet

LINE_PX = 3  # stroke width of a cutline in the letterboxed label
BASE_WIDTH = 16  # UNet base channels: 2.0M parameters, ~1 s/page on one CPU thread
POS_WEIGHT = 10.0  # BCE weight on cutline pixels, which are ~1% of a split page


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread that fails loudly instead of returning None."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return image


def read_manifest(benchmark: Path) -> list[dict[str, str]]:
    """The benchmark manifest's rows."""
    with (benchmark / "manifest.tsv").open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def training_split(row: dict[str, str]) -> str:
    """train, val or test: val is one train-fold volume in eight, hashed apart from the fold."""
    if row["fold"] == "test":
        return "test"
    digest = int(hashlib.md5(row["item"].encode()).hexdigest(), 16)
    return "val" if digest % 8 == 0 else "train"


def cutline_mask(cutlines: dict, image_size: tuple[int, int]) -> np.ndarray:
    """A cutlines.json's polylines as LINE_PX strokes on the letterboxed page.

    ``image_size`` is the page image's (width, height); the cutlines are in the
    frame their own width/height record.
    """
    width, height = image_size
    scale = INPUT_SIZE / max(width, height)
    sx = width / cutlines["width"] * scale
    sy = height / cutlines["height"] * scale
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
    for line in cutlines["cutlines"]:
        points = np.round(np.array([[x * sx, y * sy] for x, y in line])).astype(
            np.int32
        )
        cv2.polylines(mask, [points], False, 255, LINE_PX)
    return mask


def prepare_page(row: dict[str, str], benchmark: Path, out_dir: Path) -> None:
    """Write one page's letterboxed image and mask into out_dir, unless already there."""
    name = Path(row["image"]).stem
    if (out_dir / f"{name}.mask.png").exists():
        return
    image = read_image(benchmark / row["image"])
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
    if row["label"] == "split":
        cutlines = json.loads(
            (benchmark / "labels" / f"{name}.cutlines.json").read_text()
        )
        mask = cutline_mask(cutlines, (image.shape[1], image.shape[0]))
    cv2.imwrite(
        str(out_dir / f"{name}.jpg"), letterbox(image), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    cv2.imwrite(str(out_dir / f"{name}.mask.png"), mask)


def standardize(image: np.ndarray) -> np.ndarray:
    """Per-image zero-mean/unit-variance floats, as cutline_probability feeds the model."""
    x = image.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Flips and mild brightness/contrast jitter.

    No 90-degree turns: a page's layout (margins, title block) is a cue.
    """
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1], mask[::-1]
    jittered = image.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-20, 20)
    return np.ascontiguousarray(np.clip(jittered, 0, 255)), np.ascontiguousarray(mask)


def to_tensors(
    pairs: list[tuple[np.ndarray, np.ndarray]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of (image, mask) pairs as model input and 0/1 target tensors."""
    images = np.stack([standardize(image) for image, _ in pairs]).transpose(0, 3, 1, 2)
    masks = np.stack([(mask > 127).astype(np.float32) for _, mask in pairs])[:, None]
    return torch.from_numpy(images).to(device), torch.from_numpy(masks).to(device)


def cutline_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Soft Dice plus cutline-weighted BCE: Dice for the thin lines, BCE for empty pages."""
    prob = torch.sigmoid(logits)
    intersection = (prob * masks).sum(dim=(1, 2, 3))
    denominator = prob.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, masks, pos_weight=torch.tensor(POS_WEIGHT, device=logits.device)
    )
    return dice + bce


def load_pair(data_dir: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """A prepared (letterboxed image, mask) pair."""
    name = Path(row["image"]).stem
    image = read_image(data_dir / f"{name}.jpg")
    mask = read_image(data_dir / f"{name}.mask.png", cv2.IMREAD_GRAYSCALE)
    return image, mask


def validation_dice(
    model: UNet, data_dir: Path, rows: list[dict[str, str]], batch_size: int
) -> float:
    """Mean soft Dice over the split pages among rows."""
    device = next(model.parameters()).device
    model.eval()
    dices = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            pairs = [
                load_pair(data_dir, row) for row in rows[start : start + batch_size]
            ]
            images, masks = to_tensors(pairs, device)
            prob = torch.sigmoid(model(images))
            for k in range(len(pairs)):
                if masks[k].sum() > 0:
                    intersection = float((prob[k] * masks[k]).sum())
                    denominator = float(prob[k].sum() + masks[k].sum())
                    dices.append(2 * intersection / denominator if denominator else 1.0)
    return float(np.mean(dices)) if dices else 0.0


def cmd_prep(args: argparse.Namespace) -> None:
    data_dir = args.data_dir or args.benchmark / "model" / f"data-{INPUT_SIZE}"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.benchmark)
    for i, row in enumerate(rows, 1):
        prepare_page(row, args.benchmark, data_dir)
        if i % 500 == 0:
            print(f"{i}/{len(rows)} pages prepared", flush=True)


def cmd_train(args: argparse.Namespace) -> None:
    data_dir = args.data_dir or args.benchmark / "model" / f"data-{INPUT_SIZE}"
    device = select_device()
    rows = read_manifest(args.benchmark)
    train = [row for row in rows if training_split(row) == "train"]
    val = [row for row in rows if training_split(row) == "val"]
    print(f"train {len(train)} pages, val {len(val)} (test fold untouched)", flush=True)
    model = UNet(base=args.base, in_channels=3, norm="group").to(device)
    with torch.no_grad():  # start the mask at a small positive prior, not at 0.5
        bias = model.head.bias
        assert bias is not None
        bias.fill_(math.log(0.01 / 0.99))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    rng = np.random.default_rng(0)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            pairs = [
                augment(*load_pair(data_dir, train[i]), rng)
                for i in order[start : start + args.batch_size]
            ]
            images, masks = to_tensors(pairs, device)
            loss = cutline_loss(model(images), masks)
            optimizer.zero_grad()
            loss.backward()
            # Whole pale pages starve some normalization channels of variance;
            # clipping keeps one bad step from collapsing the trunk (as in
            # region_model's training).
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        dice = validation_dice(model, data_dir, val, args.batch_size)
        saved = ""
        if dice > best:
            best = dice
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            saved = "  (saved)"
        print(
            f"epoch {epoch:2d}: loss {np.mean(losses):.4f}  val dice {dice:.3f}"
            f"  [{time.time() - started:.0f}s]{saved}",
            flush=True,
        )
    print(f"best val dice {best:.3f} -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("command", choices=["prep", "train"])
    parser.add_argument("benchmark", type=Path, help="Cutline benchmark directory")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Prepared pages (default: <benchmark>/model/data-768)",
    )
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--base", type=int, default=BASE_WIDTH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    {"prep": cmd_prep, "train": cmd_train}[args.command](args)


if __name__ == "__main__":
    main()
