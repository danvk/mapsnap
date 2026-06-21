"""Read key-map page numbers with the CNN localizer + CRNN recognizer (no CRAFT/EasyOCR).

The CNN localizer (mapsnap.detect_numbers_cnn) proposes page-number centers at ~99%
recall; the CRNN (mapsnap.crnn_model) reads the digit string from a crop around each
center. Because the box stays centered on the candidate (no CRAFT box-tightening to drift
off), recall tracks the localizer and recognition is the learned CRNN — which handles the
ornate / low-resolution fonts that defeated CRAFT+EasyOCR.

Writes the same ``<stem>.streets.json`` schema as the other detectors. ``--pages`` is
optional: if given, each decode is snapped to the nearest valid page number within edit
distance 1 (a light constraint); otherwise the raw CRNN output is kept.

    uv run python -m mapsnap.detect_numbers_crnn data/keymaps/chicago-p0b.jpg
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from mapsnap.crnn_model import (
    BOX_HALF_H_WORKING,
    BOX_HALF_W_WORKING,
    build_crnn,
    decode_batch,
    eval_transform,
    number_strip,
)
from mapsnap.detect_keymap_numbers import (
    detection_record,
    filter_args,
    parse_page_spec,
    streets_path,
)
from mapsnap.detect_numbers_cnn import (
    DEFAULT_NMS_DIST,
    DEFAULT_STRIDE,
    DEFAULT_THRESHOLD,
    detect_candidate_centers,
)
from mapsnap.number_model import build_model, select_device


def levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def snap_to_pages(text: str, pages: list[str], max_distance: int = 1) -> str:
    """Nearest valid page number to ``text`` within ``max_distance``, else ``text``."""
    if not text or not pages:
        return text
    best = min(pages, key=lambda p: levenshtein(text, p))
    return best if levenshtein(text, best) <= max_distance else text


@torch.no_grad()
def read_candidates(
    image: np.ndarray,
    centers: list[tuple[float, float]],
    factor: float,
    crnn: torch.nn.Module,
    device,
    *,
    batch_size: int = 256,
) -> list[tuple[str, float]]:
    """Decode (digit string, confidence) for each candidate center via the CRNN."""
    transform = eval_transform()
    results: list[tuple[str, float]] = []
    crnn.eval()
    for start in range(0, len(centers), batch_size):
        batch = centers[start : start + batch_size]
        strips = torch.stack(
            [
                cast(torch.Tensor, transform(number_strip(image, cx, cy, factor)))
                for cx, cy in batch
            ]
        ).to(device)
        log_probs = crnn(strips)  # (T, N, C)
        texts = decode_batch(log_probs)
        confidences = log_probs.exp().max(dim=2).values.mean(dim=0).cpu().numpy()
        results.extend(zip(texts, (float(c) for c in confidences)))
    return results


def detect_and_read(
    image_path: str,
    cnn: torch.nn.Module,
    crnn: torch.nn.Module,
    device,
    *,
    stride: int,
    threshold: float,
    nms_dist: float,
    pages: list[str],
) -> list[dict]:
    """CNN-localize then CRNN-read one image; write <stem>.streets.json."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    centers, factor = detect_candidate_centers(
        image, cnn, device, stride=stride, threshold=threshold, nms_dist=nms_dist
    )
    reads = read_candidates(image, centers, factor, crnn, device)

    half_w = round(BOX_HALF_W_WORKING / factor)
    half_h = round(BOX_HALF_H_WORKING / factor)
    detections: list[dict] = []
    for (cx, cy), (text, confidence) in zip(centers, reads):
        text = snap_to_pages(text, pages)
        if not text:
            continue
        x0 = max(0, round(cx) - half_w)
        y0 = max(0, round(cy) - half_h)
        x1 = min(width, round(cx) + half_w)
        y1 = min(height, round(cy) + half_h)
        polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        detections.append(detection_record(polygon, text, confidence))

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
        f"{Path(image_path).name}: {len(centers)} candidates -> {len(detections)} read",
        file=sys.stderr,
    )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read page numbers with the CNN localizer + CRNN recognizer."
    )
    parser.add_argument("images", nargs="+", metavar="IMAGE")
    parser.add_argument(
        "--pages",
        metavar="SPEC",
        help="Optional valid page set (e.g. '1-111'); snaps decodes within edit distance 1.",
    )
    parser.add_argument(
        "--cnn-weights", type=Path, default=Path("models/number_detector.pt")
    )
    parser.add_argument(
        "--crnn-weights", type=Path, default=Path("models/number_crnn.pt")
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--nms-dist", type=float, default=DEFAULT_NMS_DIST)
    args = parser.parse_args()

    device = select_device()
    cnn = build_model(pretrained=False)
    cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
    cnn.to(device)
    crnn = build_crnn()
    crnn.load_state_dict(torch.load(args.crnn_weights, map_location=device))
    crnn.to(device)

    pages = [str(n) for n in parse_page_spec(args.pages)] if args.pages else []
    for image_path in args.images:
        detect_and_read(
            image_path,
            cnn,
            crnn,
            device,
            stride=args.stride,
            threshold=args.threshold,
            nms_dist=args.nms_dist,
            pages=pages,
        )


if __name__ == "__main__":
    main()
