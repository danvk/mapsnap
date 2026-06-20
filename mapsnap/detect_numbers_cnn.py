"""Locate page numbers with the trained CNN, tighten with CRAFT, read with constrained OCR.

Three stages:

1. CNN localizer — slide the page-number classifier over a downscaled scan, NMS the
   high-scoring windows into candidate centers (this finds page numbers CRAFT alone
   misses, and rejects the lot numbers / labels CRAFT alone keeps).
2. CRAFT fusion — around each candidate, run EasyOCR's CRAFT detector on a small crop and
   take the largest text box containing the candidate center (else the nearest box). This
   converts the CNN's loose, grid-quantized location into a tight box, which is what the
   recognizer needs. Only if CRAFT finds nothing does it fall back to a candidate-centered
   box, so no CNN detection is lost.
3. Recognition — the tight boxes go through the constrained-CTC decoder
   (mapsnap.ctc_vocab_decode) with the volume's page-number vocabulary, written to the
   same ``<stem>.streets.json`` schema as mapsnap.detect_keymap_numbers.

Images are brought to a common working resolution via keymap_patches.working_scale (full
scans downscaled by 0.25; scans already at ~25% used as-is); detections are emitted in
the input image's own pixel space.

    uv run python -m mapsnap.detect_numbers_cnn --pages 1-111 data/keymaps/chicago-p0b.jpg
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import easyocr
import numpy as np
import torch
from PIL import Image

from mapsnap.ctc_vocab_decode import patch_easyocr_reader
from mapsnap.detect_keymap_numbers import (
    DIGIT_ALLOWLIST,
    detection_record,
    filter_args,
    parse_page_spec,
    streets_path,
)
from mapsnap.keymap_patches import PATCH_SIZE, crop_patch, working_scale
from mapsnap.number_model import build_model, eval_transform, select_device

# Sliding-window stride (px, at working scale). Smaller = denser scan + better-centered
# candidates (so CRAFT boxes contain the center more often), at more compute.
DEFAULT_STRIDE = 16

# Probability above which a window is a page-number candidate.
DEFAULT_THRESHOLD = 0.5

# Min distance (px, at working scale) between kept candidate peaks (~ one block).
DEFAULT_NMS_DIST = 38

# Half-size (px, at working scale) of the crop CRAFT runs on around each candidate.
REGION_HALF_WORKING = 60

# Half-size (px, at working scale) of the fallback box centered on the candidate when no
# CRAFT box contains the center. Number-sized so it still reads, and small enough to keep
# the truth point inside for matching.
FALLBACK_HALF_WORKING = 40

# CRAFT min text size (px) at working scale; converted to the image's own scale per run.
CRAFT_MIN_WORKING = 15

# CRAFT detection thresholds, lower than EasyOCR's defaults (0.7 / 0.4) so the bold page
# numbers — which CRAFT otherwise sometimes fails to box inside a candidate crop — are
# detected. The CNN has already localized the region, so being sensitive here is safe.
CRAFT_TEXT_THRESHOLD = 0.5
CRAFT_LOW_TEXT = 0.3

# Tight boxes whose centers fall within this (px, working scale) are de-duplicated.
DEDUP_WORKING = 20


def window_centers(width: int, height: int, stride: int) -> list[tuple[int, int]]:
    """Grid of (cx, cy) window centers covering a width x height image at ``stride``."""
    half = stride // 2
    xs = list(range(half, width, stride))
    ys = list(range(half, height, stride))
    return [(x, y) for y in ys for x in xs]


def nms_peaks(
    centers: list[tuple[float, float]], scores: list[float], min_dist: float
) -> list[int]:
    """Greedy non-max suppression; return kept indices (highest score first)."""
    order = sorted(range(len(centers)), key=lambda i: scores[i], reverse=True)
    min_sq = min_dist * min_dist
    kept: list[int] = []
    for i in order:
        cx, cy = centers[i]
        if all(
            (cx - centers[k][0]) ** 2 + (cy - centers[k][1]) ** 2 >= min_sq
            for k in kept
        ):
            kept.append(i)
    return kept


def region_bounds(
    cx: float, cy: float, half: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamped (x0, y0, x1, y1) of a 2*half square around (cx, cy)."""
    x0 = max(0, round(cx) - half)
    y0 = max(0, round(cy) - half)
    x1 = min(width, round(cx) + half)
    y1 = min(height, round(cy) + half)
    return x0, y0, x1, y1


def box_center(box: list[int]) -> tuple[float, float]:
    """Center (x, y) of an [x_min, x_max, y_min, y_max] box."""
    return (box[0] + box[1]) / 2, (box[2] + box[3]) / 2


def select_tight_box(boxes: list[list[int]], cx: float, cy: float) -> int:
    """Index of the CRAFT box to tighten onto ([x_min,x_max,y_min,y_max]); -1 if empty.

    The candidate is centered on the page number, so prefer boxes that contain (cx, cy)
    and take the largest by area (the bold page number rather than an overlapping small lot
    number). If none contains the center, take the box whose center is nearest — a tight
    box reads far better than a candidate-centered crop even when slightly off. Returns -1
    only when there are no boxes at all.
    """
    if not boxes:
        return -1
    containing = [
        i for i, b in enumerate(boxes) if b[0] <= cx <= b[1] and b[2] <= cy <= b[3]
    ]
    if containing:
        return max(
            containing,
            key=lambda i: (boxes[i][1] - boxes[i][0]) * (boxes[i][3] - boxes[i][2]),
        )
    return min(
        range(len(boxes)),
        key=lambda i: (
            (box_center(boxes[i])[0] - cx) ** 2 + (box_center(boxes[i])[1] - cy) ** 2
        ),
    )


@torch.no_grad()
def score_windows(
    model, image_scaled: np.ndarray, device, *, stride: int, batch_size: int = 256
) -> tuple[list[tuple[int, int]], list[float]]:
    """Page-number probability for every sliding window over a downscaled image."""
    height, width = image_scaled.shape[:2]
    centers = window_centers(width, height, stride)
    transform = eval_transform()
    scores: list[float] = []
    model.eval()
    for start in range(0, len(centers), batch_size):
        batch = centers[start : start + batch_size]
        tensors = torch.stack(
            [
                cast(
                    torch.Tensor,
                    transform(crop_patch(image_scaled, cx, cy, PATCH_SIZE)),
                )
                for cx, cy in batch
            ]
        ).to(device)
        probs = torch.sigmoid(model(tensors).squeeze(1)).cpu().numpy()
        scores.extend(probs.tolist())
    return centers, scores


def detect_candidate_centers(
    image: np.ndarray,
    model,
    device,
    *,
    stride: int,
    threshold: float,
    nms_dist: float,
) -> tuple[list[tuple[float, float]], float]:
    """Candidate page-number centers in the image's own coords, plus the working factor."""
    height, width = image.shape[:2]
    factor = working_scale(width, height)
    if factor != 1.0:
        scaled = np.asarray(
            Image.fromarray(image).resize(
                (round(width * factor), round(height * factor)),
                Image.Resampling.LANCZOS,
            )
        )
    else:
        scaled = image

    centers, scores = score_windows(model, scaled, device, stride=stride)
    hot = [i for i, s in enumerate(scores) if s >= threshold]
    hot_centers = [(float(centers[i][0]), float(centers[i][1])) for i in hot]
    hot_scores = [scores[i] for i in hot]
    keep = nms_peaks(hot_centers, hot_scores, nms_dist)
    image_centers = [
        (hot_centers[k][0] / factor, hot_centers[k][1] / factor) for k in keep
    ]
    return image_centers, factor


def tight_box_for_candidate(
    reader: easyocr.Reader,
    image: np.ndarray,
    cx: float,
    cy: float,
    *,
    region_half: int,
    craft_min: int,
    text_threshold: float = CRAFT_TEXT_THRESHOLD,
    low_text: float = CRAFT_LOW_TEXT,
) -> list[int] | None:
    """CRAFT-tight box (image coords) nearest a candidate, or None if CRAFT finds none."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = region_bounds(cx, cy, region_half, width, height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    horizontal_agg, _ = reader.detect(
        crop, min_size=craft_min, text_threshold=text_threshold, low_text=low_text
    )
    boxes = [[int(v) for v in b] for b in horizontal_agg[0]]
    if not boxes:
        return None
    i = select_tight_box(boxes, (x1 - x0) / 2, (y1 - y0) / 2)
    if i < 0:
        return None
    b = boxes[i]
    return [x0 + b[0], x0 + b[1], y0 + b[2], y0 + b[3]]


def detect_and_read(
    image_path: str,
    model,
    device,
    reader: easyocr.Reader,
    *,
    stride: int,
    threshold: float,
    nms_dist: float,
    beam_width: int,
    text_threshold: float = CRAFT_TEXT_THRESHOLD,
    low_text: float = CRAFT_LOW_TEXT,
) -> list[dict]:
    """Full pipeline for one image: CNN candidates -> CRAFT-tight boxes -> read."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]

    centers, factor = detect_candidate_centers(
        image, model, device, stride=stride, threshold=threshold, nms_dist=nms_dist
    )
    region_half = round(REGION_HALF_WORKING / factor)
    craft_min = max(1, round(CRAFT_MIN_WORKING / factor))
    fallback_half = round(FALLBACK_HALF_WORKING / factor)

    tight_boxes: list[list[int]] = []
    for cx, cy in centers:
        box = tight_box_for_candidate(
            reader,
            image,
            cx,
            cy,
            region_half=region_half,
            craft_min=craft_min,
            text_threshold=text_threshold,
            low_text=low_text,
        )
        if box is None:
            # No CRAFT box covered the center: a number-sized box on the candidate keeps
            # the truth point inside while staying tight enough to read.
            fx0, fy0, fx1, fy1 = region_bounds(cx, cy, fallback_half, width, height)
            box = [fx0, fx1, fy0, fy1]  # [x_min, x_max, y_min, y_max]
        tight_boxes.append(box)

    # De-duplicate boxes that landed on the same number; keep the larger box.
    if tight_boxes:
        box_centers = [box_center(b) for b in tight_boxes]
        areas = [float((b[1] - b[0]) * (b[3] - b[2])) for b in tight_boxes]
        keep = nms_peaks(box_centers, areas, round(DEDUP_WORKING / factor))
        tight_boxes = [tight_boxes[k] for k in keep]

    results = cast(
        list,
        reader.recognize(
            image,
            tight_boxes,
            [],
            decoder="wordbeamsearch",
            beamWidth=beam_width,
            allowlist=DIGIT_ALLOWLIST,
        ),
    )
    detections = [
        detection_record(bbox, text, confidence)
        for bbox, text, confidence in results
        if str(text).strip()
    ]

    doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": detections,
    }
    with open(streets_path(image_path), "w") as f:
        json.dump(doc, f, indent=2)
    print(
        f"{Path(image_path).name}: {len(centers)} candidates -> "
        f"{len(tight_boxes)} boxes -> {len(detections)} read",
        file=sys.stderr,
    )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate page numbers with the CNN, tighten with CRAFT, read with OCR."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument("--pages", required=True, metavar="SPEC", help="e.g. '1-111'.")
    parser.add_argument(
        "--weights", type=Path, default=Path("models/number_detector.pt")
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--nms-dist", type=float, default=DEFAULT_NMS_DIST)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=CRAFT_TEXT_THRESHOLD,
        help="CRAFT text threshold for local detection (default %(default)s).",
    )
    parser.add_argument(
        "--low-text",
        type=float,
        default=CRAFT_LOW_TEXT,
        help="CRAFT low-text threshold for local detection (default %(default)s).",
    )
    args = parser.parse_args()

    device = select_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    vocab = [str(n) for n in parse_page_spec(args.pages)]
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    patch_easyocr_reader(reader, vocab, args.beam_width)

    for image_path in args.images:
        detect_and_read(
            image_path,
            model,
            device,
            reader,
            stride=args.stride,
            threshold=args.threshold,
            nms_dist=args.nms_dist,
            beam_width=args.beam_width,
            text_threshold=args.text_threshold,
            low_text=args.low_text,
        )


if __name__ == "__main__":
    main()
