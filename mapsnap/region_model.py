"""Page content-region model (#226): a whole-page UNet trained on OIM multimasks.

A page's content region is the part holding rich content exclusive to that page —
what the IIIF mask should clip to, and what a key map's region polygon depicts.
Every georeferenced truth page carries free labels: the SvgSelector polygon(s) of
its annotation(s) (one per panel on split sheets), in source-pixel coordinates.

Unlike the road UNet this trains on the DOWNSCALED WHOLE PAGE, not patches:
content-region-ness is a global property — duplicated margin content from a
neighboring sheet looks locally identical to interior content, and what
distinguishes it is where it sits on the sheet. Input is RGB (colour is
load-bearing: coloured blocks vs white margins), letterboxed to a fixed square.

    uv run python -m mapsnap.region_model train data/chicago_il_1950_vol_1 ... \
        --val data/hudson_co_nj_1950_vol_9
    uv run python -m mapsnap.region_model predict IMG [IMG ...] --out-dir DIR
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mapsnap.compare_iiif_georef import (
    annotations_by_source,
    label_split_index,
    parse_svg_polygon,
)
from mapsnap.fix_truth_splits import gcp_containment
from mapsnap.road_model import UNet
from mapsnap.train_road_unet import dice_loss

INPUT_SIZE = 512
REGION_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "region_unet.pt"


def truth_region_mask(
    items: list[dict], jpg_size: tuple[int, int], *, require_all: bool = False
) -> np.ndarray | None:
    """The union of a page's truth selector polygons as a uint8 mask in jpg pixels.

    Split-panel selectors are trusted only when they contain at least half of
    their own annotation's GCPs — OIM writes some split selectors in the crop's
    frame (OIM#402), and an unguarded crop-frame polygon would paint the wrong
    part of the sheet. Whole-page selectors (the paper mask) carry no such
    hazard. Returns None when no usable selector exists.

    ``require_all`` rejects a page when ANY of its annotations failed to paint:
    a split sheet whose second panel was gated out yields a mask covering a
    fraction of its real content (hudson p92: 5.8% of the page), which is a
    wrong label rather than a hard one — it teaches the model to under-predict
    and then scores a correct prediction at IoU 0.06. 4.4% of labeled pages
    corpus-wide are partial this way.
    """
    width, height = jpg_size
    mask = np.zeros((height, width), np.uint8)
    painted = 0
    for item in items:
        selector = (item.get("target") or {}).get("selector") or {}
        if selector.get("type") != "SvgSelector":
            continue
        points = parse_svg_polygon(selector.get("value", ""))
        if len(points) < 3:
            continue
        if label_split_index(item) is not None and gcp_containment(item, points) < 0.5:
            continue
        source = item["target"]["source"]
        sw, sh = float(source.get("width") or 0), float(source.get("height") or 0)
        if sw <= 0 or sh <= 0:
            continue
        ring = np.array(
            [[x * width / sw, y * height / sh] for x, y in points], np.int32
        )
        cv2.fillPoly(mask, [ring], 255)
        painted += 1
    if not painted or (require_all and painted < len(items)):
        return None
    return mask


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize preserving aspect onto a white size x size canvas (top-left anchored)."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale)))
    )
    if resized.ndim == 2:
        canvas = np.full((size, size), 255, resized.dtype)
    else:
        canvas = np.full((size, size, resized.shape[2]), 255, resized.dtype)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas


def volume_examples(volume: Path) -> list[tuple[Path, np.ndarray]]:
    """(jpg path, label mask at jpg resolution) for each truth page with a jpg.

    Page keys are matched to disk stems case-insensitively (lettered sheets are
    uppercase on some volumes).
    """
    truth = volume / "main.iiif.json"
    if not truth.exists():
        return []
    stems = {p.stem.lower(): p for p in volume.glob("p*.jpg") if "__" not in p.stem}
    examples = []
    for key, items in annotations_by_source(truth).items():
        jpg = stems.get(key.lower())
        if jpg is None:
            continue
        image = cv2.imread(str(jpg))
        if image is None:
            continue
        mask = truth_region_mask(
            items, (image.shape[1], image.shape[0]), require_all=True
        )
        if mask is not None:
            examples.append((jpg, mask))
    return examples


def standardize(image: np.ndarray) -> np.ndarray:
    """Per-image zero-mean/unit-variance floats: white-dominant pages give raw
    0-1 inputs almost no dynamic range, which starves the spatial gradients and
    feeds the all-positive attractor this task is prone to."""
    x = image.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def to_tensors(
    pairs: list[tuple[np.ndarray, np.ndarray]], device
) -> tuple[torch.Tensor, torch.Tensor]:
    images = np.stack([standardize(p) for p, _ in pairs]).transpose(0, 3, 1, 2)
    labels = np.stack([(m > 127).astype(np.float32) for _, m in pairs])[:, None]
    return torch.from_numpy(images).to(device), torch.from_numpy(labels).to(device)


def augment(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random flips only: whole pages have a canonical layout (margins, sheet
    numbers), so 90-degree rotations fight the global cues this model needs."""
    if rng.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if rng.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def predict_region(model, image: np.ndarray, device) -> np.ndarray:
    """P(content region) in [0,1] at the image's own resolution."""
    boxed = letterbox(image)
    tensor = torch.from_numpy(standardize(boxed).transpose(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    height, width = image.shape[:2]
    scale = INPUT_SIZE / max(height, width)
    crop = prob[: max(1, round(height * scale)), : max(1, round(width * scale))]
    return cv2.resize(crop, (width, height))


def page_iou(model, examples, device) -> float:
    """Mean whole-page IoU at threshold 0.5 over (image, mask) pairs."""
    total = 0.0
    for image, mask in examples:
        prob = predict_region(model, image, device)
        predicted = prob >= 0.5
        actual = mask > 127
        union = np.logical_or(predicted, actual).sum()
        total += np.logical_and(predicted, actual).sum() / union if union else 1.0
    return total / max(1, len(examples))


def cmd_train(args: argparse.Namespace) -> None:
    from mapsnap.keymap.number_model import select_device

    device = select_device()
    rng = np.random.default_rng(0)
    train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for volume in args.volumes:
        pairs = volume_examples(Path(volume))
        print(f"  {Path(volume).name}: {len(pairs)} labeled pages", flush=True)
        for jpg, mask in pairs:
            image = cv2.imread(str(jpg))
            if image is None:
                continue
            train_pairs.append((letterbox(image), letterbox(mask)))
    val_pairs = [
        (cv2.imread(str(jpg)), mask) for jpg, mask in volume_examples(Path(args.val))
    ]
    print(f"training pages: {len(train_pairs)}; validation: {len(val_pairs)}")
    if not val_pairs:
        sys.exit("empty validation set")

    model = UNet(base=args.base, in_channels=3, norm="group").to(device)
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"initialized from {args.init}")
    # Start at the positive-class prior instead of either constant attractor:
    # this task has two degenerate poles (all-region scores ~0.5 IoU by area,
    # all-background scores 0) and both trap a zero-initialized output. The
    # prior is measured from the training labels themselves.
    if args.init is None:
        prior = float(np.mean([float((m > 127).mean()) for _, m in train_pairs]) or 0.5)
        prior = min(max(prior, 0.05), 0.95)
        with torch.no_grad():
            bias = model.head.bias
            assert bias is not None
            bias.fill_(math.log(prior / (1 - prior)))
        print(f"output bias initialized to prior {prior:.3f}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Cosine decay to zero over the run: the plateau this model reaches is a
    # refinement plateau (boundaries, not blobs), where a decaying step is
    # what buys the last points.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(len(train_pairs))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [
                augment(train_pairs[i][0], train_pairs[i][1], rng)
                for i in order[start : start + args.batch_size]
            ]
            images, labels = to_tensors(batch, device)
            logits = model(images)
            loss = dice_loss(
                logits, labels
            ) + torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # Whole pale pages are mostly near-constant paper, so some
            # BatchNorm channels see tiny batch variance and their 1/sigma
            # amplifies first-layer gradients by ~1e6 (measured) -- one step
            # at any usable LR destroys the encoder and the model collapses
            # to a constant. Clipping keeps the trunk alive; road-model
            # patches never hit this because crops have variance everywhere.
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        schedule.step()
        model.eval()
        iou = page_iou(model, val_pairs, device)
        marker = ""
        if iou > best:
            best = iou
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            marker = "  (saved)"
        print(
            f"epoch {epoch:2d}: loss {np.mean(losses):.4f}  val IoU {iou:.3f}"
            f"  [{time.time() - started:.0f}s]{marker}",
            flush=True,
        )
    print(f"best val IoU {best:.3f} -> {args.output}")


def cmd_predict(args: argparse.Namespace) -> None:
    from mapsnap.keymap.number_model import select_device

    device = select_device()
    state = torch.load(args.model, map_location=device)
    base = state["enc1.block.0.weight"].shape[0]
    model = UNet(base=base, in_channels=3, norm="group").to(device)
    model.load_state_dict(state)
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip (unreadable): {path}", file=sys.stderr)
            continue
        prob = predict_region(model, image, device)
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.55, heat, 0.45, 0)
        out = (
            args.out_dir
            / f"{Path(path).parent.parent.name}_{Path(path).stem}.region.png"
        )
        cv2.imwrite(str(out), overlay)
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("volumes", nargs="+")
    train.add_argument("--val", required=True)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--base", type=int, default=24)
    train.add_argument("--output", type=Path, default=REGION_MODEL_PATH)
    train.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Continue from these weights (skips the prior bias init).",
    )
    predict = sub.add_parser("predict")
    predict.add_argument("images", nargs="+")
    predict.add_argument("--model", type=Path, default=REGION_MODEL_PATH)
    predict.add_argument("--out-dir", type=Path, default=Path("region_preview"))
    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    else:
        cmd_predict(args)


if __name__ == "__main__":
    main()
