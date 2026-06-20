"""Locate page numbers with the trained CNN, then read them with constrained OCR.

Stage 1 (this CNN): slide the page-number classifier over a downscaled scan to get a
"page-number-here" score per window, NMS the high-scoring windows into candidate centers,
and map them back to full-resolution boxes.

Stage 2 (reuse): hand those boxes to the constrained-CTC EasyOCR recognizer
(mapsnap.ctc_vocab_decode) with the volume's page-number vocabulary, and write the same
``<stem>.streets.json`` schema as mapsnap.detect_keymap_numbers.

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
from mapsnap.keymap_patches import PATCH_SIZE, SCALE, crop_patch
from mapsnap.number_model import build_model, eval_transform, select_device

# Sliding-window stride (px, at SCALE). Smaller = denser scan, more compute.
DEFAULT_STRIDE = 24

# Probability above which a window is a page-number candidate.
DEFAULT_THRESHOLD = 0.5

# Min distance (px, at SCALE) between kept peaks, so one number yields one detection but
# adjacent blocks stay separate. ~ one block is ~60px at SCALE.
DEFAULT_NMS_DIST = 38

# Full-resolution side length of the emitted boxes.
DEFAULT_BOX_SIZE = 220


def window_centers(width: int, height: int, stride: int) -> list[tuple[int, int]]:
    """Grid of (cx, cy) window centers covering a width x height image at ``stride``."""
    half = stride // 2
    xs = list(range(half, width, stride))
    ys = list(range(half, height, stride))
    return [(x, y) for y in ys for x in xs]


def nms_peaks(
    centers: list[tuple[int, int]], scores: list[float], min_dist: float
) -> list[int]:
    """Greedy non-max suppression; return kept indices (highest score first).

    Keeps the highest-scoring center, suppresses any remaining center within
    ``min_dist`` of a kept one, and repeats.
    """
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


def boxes_from_centers(
    centers: list[tuple[float, float]], box_size: int, width: int, height: int
) -> list[list[list[int]]]:
    """Axis-aligned square polygons of ``box_size`` around each center, clamped to image."""
    half = box_size // 2
    boxes = []
    for cx, cy in centers:
        x0 = max(0, round(cx) - half)
        y0 = max(0, round(cy) - half)
        x1 = min(width, round(cx) + half)
        y1 = min(height, round(cy) + half)
        boxes.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return boxes


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


def detect_candidate_boxes(
    image_path: str,
    model,
    device,
    *,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD,
    nms_dist: float = DEFAULT_NMS_DIST,
    box_size: int = DEFAULT_BOX_SIZE,
) -> tuple[list[list[list[int]]], int, int]:
    """Full-resolution candidate boxes for one image; returns (boxes, width, height)."""
    with Image.open(image_path) as img:
        full = img.convert("RGB")
        full_w, full_h = full.size
        scaled = np.asarray(
            full.resize(
                (round(full_w * SCALE), round(full_h * SCALE)), Image.Resampling.LANCZOS
            )
        )

    centers, scores = score_windows(model, scaled, device, stride=stride)
    hot = [i for i, s in enumerate(scores) if s >= threshold]
    hot_centers = [centers[i] for i in hot]
    hot_scores = [scores[i] for i in hot]
    keep = nms_peaks(hot_centers, hot_scores, nms_dist)

    full_centers = [
        (hot_centers[k][0] / SCALE, hot_centers[k][1] / SCALE) for k in keep
    ]
    boxes = boxes_from_centers(full_centers, box_size, full_w, full_h)
    return boxes, full_w, full_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate page numbers with the CNN, then read them with constrained OCR."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument("--pages", required=True, metavar="SPEC", help="e.g. '1-111'.")
    parser.add_argument(
        "--weights", type=Path, default=Path("models/number_detector.pt")
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--nms-dist", type=float, default=DEFAULT_NMS_DIST)
    parser.add_argument("--box-size", type=int, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--beam-width", type=int, default=20)
    args = parser.parse_args()

    device = select_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    vocab = [str(n) for n in parse_page_spec(args.pages)]
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    patch_easyocr_reader(reader, vocab, args.beam_width)

    for image_path in args.images:
        boxes, width, height = detect_candidate_boxes(
            image_path,
            model,
            device,
            stride=args.stride,
            threshold=args.threshold,
            nms_dist=args.nms_dist,
            box_size=args.box_size,
        )
        # Read each candidate box with the constrained recognizer.
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        horizontal_list = [[b[0][0], b[1][0], b[0][1], b[2][1]] for b in boxes]
        results = cast(
            list,
            reader.recognize(
                rgb,
                horizontal_list,
                [],
                decoder="wordbeamsearch",
                beamWidth=args.beam_width,
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
            f"{Path(image_path).name}: {len(boxes)} candidates -> {len(detections)} read",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
