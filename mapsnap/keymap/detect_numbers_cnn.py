"""CNN page-number localizer for key maps: slide the classifier, NMS into candidate centers.

A MobileNetV3 patch classifier (mapsnap.keymap.number_model) is slid over a downscaled
scan and the high-scoring windows are non-max-suppressed into candidate page-number centers.
This finds the page numbers free OCR misses and rejects the lot numbers / labels it keeps;
the CRNN recognizer (mapsnap.keymap.detect_numbers_crnn) then reads each candidate.

Images are brought to a common working resolution via keymap_patches.working_scale (full
scans downscaled by 0.25; scans already at ~25% used as-is); candidates are returned in the
input image's own pixel space.

Run directly to write CNN-only debug artifacts (a heatmap PNG and a no-text candidate JSON):

    uv run python -m mapsnap.keymap.detect_numbers_cnn data/keymaps/chicago-p0b.jpg
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image

from mapsnap.keymap.keymap_patches import PATCH_SIZE, crop_patch, working_scale
from mapsnap.keymap.number_model import build_model, eval_transform, select_device
from mapsnap.keymap.records import detection_record, filter_args

# Sliding-window stride (px, at working scale). Smaller = denser scan + better-centered
# candidates, at more compute.
DEFAULT_STRIDE = 16

# Probability above which a window is a page-number candidate.
DEFAULT_THRESHOLD = 0.5

# Min distance (px, at working scale) between kept candidate peaks (~ one block).
DEFAULT_NMS_DIST = 38

# Half-size (px, at working scale) of the number-sized box drawn around a candidate in the
# debug output (and the fallback box elsewhere).
FALLBACK_HALF_WORKING = 40

# Detections whose centers fall within this (px, working scale) are de-duplicated downstream.
DEDUP_WORKING = 20


def window_centers(width: int, height: int, stride: int) -> list[tuple[int, int]]:
    """Grid of (cx, cy) window centers covering a width x height image at ``stride``."""
    half = stride // 2
    xs = list(range(half, width, stride))
    ys = list(range(half, height, stride))
    return [(x, y) for y in ys for x in xs]


def nms_peaks(
    centers: Sequence[tuple[float, float]], scores: Sequence[float], min_dist: float
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


def scores_to_grid(scores: list[float], n_cols: int, n_rows: int) -> np.ndarray:
    """Reshape row-major window scores into an (n_rows, n_cols) float32 heatmap grid."""
    return np.array(scores, dtype=np.float32).reshape(n_rows, n_cols)


def heatmap_overlay(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Blend a JET colormap of the score grid (0..1) over the image (RGB uint8)."""
    height, width = image.shape[:2]
    big = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.45, colored, 0.55, 0.0)


def write_cnn_debug(
    image_path: str,
    model,
    device,
    *,
    stride: int,
    threshold: float,
    nms_dist: float,
) -> None:
    """Write CNN-only debug artifacts (no recognition): a heatmap PNG and a candidate JSON.

    ``<stem>.heatmap.png`` overlays the dense window-probability heatmap on the scan, and
    ``<stem>.cnn.json`` holds the NMS'd candidate boxes (text empty, confidence = the CNN
    peak score) so localization can be inspected in the debugger or scored on its own.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    factor = working_scale(width, height)
    scaled = (
        image
        if factor == 1.0
        else np.asarray(
            Image.fromarray(image).resize(
                (round(width * factor), round(height * factor)),
                Image.Resampling.LANCZOS,
            )
        )
    )

    centers, scores = score_windows(model, scaled, device, stride=stride)
    n_cols = len(range(stride // 2, scaled.shape[1], stride))
    n_rows = len(range(stride // 2, scaled.shape[0], stride))
    grid = scores_to_grid(scores, n_cols, n_rows)

    overlay = heatmap_overlay(image, grid)
    stem = Path(image_path).name.split(".")[0]
    heatmap_path = Path(image_path).parent / (stem + ".heatmap.png")
    Image.fromarray(overlay).save(heatmap_path)

    hot = [i for i, s in enumerate(scores) if s >= threshold]
    hot_centers = [(float(centers[i][0]), float(centers[i][1])) for i in hot]
    hot_scores = [scores[i] for i in hot]
    keep = nms_peaks(hot_centers, hot_scores, nms_dist)

    box_half = round(FALLBACK_HALF_WORKING / factor)
    detections = []
    for k in keep:
        cx, cy = hot_centers[k][0] / factor, hot_centers[k][1] / factor
        x0, y0, x1, y1 = region_bounds(cx, cy, box_half, width, height)
        polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        detections.append(detection_record(cast(list, polygon), "", hot_scores[k]))

    doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": detections,
    }
    cnn_path = Path(image_path).parent / (stem + ".cnn.json")
    with open(cnn_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(
        f"{Path(image_path).name}: {len(detections)} CNN candidates "
        f"-> {heatmap_path.name}, {cnn_path.name}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write CNN-only key-map localizer debug artifacts (heatmap + candidates)."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument(
        "--weights", type=Path, default=Path("models/number_detector.pt")
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--nms-dist", type=float, default=DEFAULT_NMS_DIST)
    args = parser.parse_args()

    device = select_device()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    for image_path in args.images:
        write_cnn_debug(
            image_path,
            model,
            device,
            stride=args.stride,
            threshold=args.threshold,
            nms_dist=args.nms_dist,
        )


if __name__ == "__main__":
    main()
