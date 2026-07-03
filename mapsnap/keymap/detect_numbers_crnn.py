"""Read key-map page numbers with the CNN localizer + CRNN recognizer (no CRAFT/EasyOCR).

The CNN localizer (mapsnap.keymap.detect_numbers_cnn) proposes page-number centers at ~99%
recall; the CRNN (mapsnap.keymap.crnn_model) reads the digit string from a crop around each
center. Because the box stays centered on the candidate (no CRAFT box-tightening to drift
off), recall tracks the localizer and recognition is the learned CRNN — which handles the
ornate / low-resolution fonts that defeated CRAFT+EasyOCR.

Writes the same ``<stem>.keymap.json`` schema as the other detectors. ``--pages`` is
optional: if given, each decode is snapped to the nearest valid page number within edit
distance 1 (a light constraint); otherwise the raw CRNN output is kept.

    uv run python -m mapsnap.keymap.detect_numbers_crnn data/keymaps/chicago-p0b.jpg
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


from mapsnap.keymap.crnn_model import (
    build_crnn,
    central_group,
    ctc_greedy_decode,
    eval_transform,
    greedy_paths,
    locate_number,
    number_strip,
    strip_crop_box,
)
from mapsnap.keymap.detect_numbers_cnn import (
    DEDUP_WORKING,
    DEFAULT_NMS_DIST,
    DEFAULT_STRIDE,
    DEFAULT_THRESHOLD,
    detect_candidate_centers,
    nms_peaks,
)
from mapsnap.keymap.number_model import build_model, select_device
from mapsnap.keymap.records import (
    detection_record,
    filter_args,
    keymap_path,
    parse_page_spec,
)


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
) -> tuple[list[tuple[float, list[int]]], list[np.ndarray]]:
    """Per candidate: (confidence, CTC path) plus the grayscale strips.

    The path is segmented into number clusters downstream (central_group); decoding and
    boxing use only the cluster nearest the strip center, so a neighbor caught in the wide
    crop is dropped.
    """
    transform = eval_transform()
    strips = [number_strip(image, cx, cy, factor) for cx, cy in centers]
    results: list[tuple[float, list[int]]] = []
    crnn.eval()
    for start in range(0, len(strips), batch_size):
        batch = strips[start : start + batch_size]
        tensors = torch.stack(
            [cast(torch.Tensor, transform(strip)) for strip in batch]
        ).to(device)
        log_probs = crnn(tensors)  # (T, N, C)
        paths = greedy_paths(log_probs)
        confidences = log_probs.exp().max(dim=2).values.mean(dim=0).cpu().numpy()
        results.extend((float(conf), path) for conf, path in zip(confidences, paths))
    return results, strips


# Half-widths (working px) for the narrow-detection re-read, tighter than the default
# BOX_HALF_W_WORKING=55 so a squished multi-digit number resolves into separate digits.
REREAD_HALF_WIDTHS_WORKING = [30.0, 38.0]


@torch.no_grad()
def reread_narrow(
    image: np.ndarray,
    crnn: torch.nn.Module,
    device,
    center: tuple[float, float],
    factor: float,
    pages: list[str],
    half_w_working: float,
) -> tuple[list[list[int]], str, float] | None:
    """Re-read the central number around ``center`` from a tighter, minimum-width crop.

    The default strip is wide enough that a multi-digit number is squished until the CRNN
    fires only its central digit; a tighter crop resolves the full number. Returns the same
    (polygon, text, confidence) triple as the main pass, or None if nothing decodes.
    """
    height, width = image.shape[:2]
    cx, cy = center
    strip = number_strip(image, cx, cy, factor, half_w_working=half_w_working)
    tensor = cast(torch.Tensor, eval_transform()(strip)).unsqueeze(0).to(device)
    log_probs = crnn(tensor)
    path = greedy_paths(log_probs)[0]
    confidence = float(log_probs.exp().max(dim=2).values.mean())
    group = central_group(path)
    if group is None:
        return None
    text = snap_to_pages(ctc_greedy_decode(path[group[0] : group[1] + 1]), pages)
    if not text:
        return None
    crop_box = strip_crop_box(
        width, height, cx, cy, factor, half_w_working=half_w_working
    )
    polygon = locate_number(strip, group, len(path), crop_box)
    return polygon, text, confidence


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
    """CNN-localize then CRNN-read one image; write <stem>.keymap.json."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    centers, factor = detect_candidate_centers(
        image, cnn, device, stride=stride, threshold=threshold, nms_dist=nms_dist
    )
    reads, strips = read_candidates(image, centers, factor, crnn, device)

    # Decode and box only the central number of each crop, dropping a neighbor caught in the
    # wide window.
    found: list[tuple[list[list[int]], str, float]] = []
    for (cx, cy), (confidence, path), strip in zip(centers, reads, strips):
        group = central_group(path)
        if group is None:
            continue  # CRNN emitted nothing -> no number here
        raw = ctc_greedy_decode(path[group[0] : group[1] + 1])
        text = snap_to_pages(raw, pages)
        if not text:
            continue
        if text != raw:
            print(
                f"{Path(image_path).name}: snapped {raw!r} -> {text!r} "
                f"(edit distance {levenshtein(raw, text)})",
                file=sys.stderr,
            )
        crop_box = strip_crop_box(width, height, cx, cy, factor)
        polygon = locate_number(strip, group, len(path), crop_box)
        # A single-digit read is often a multi-digit number the wide strip squished until the
        # CRNN fired only its central digit (e.g. the "0" of "105", the "1" of "61"); re-read
        # tighter, minimum-width crops and take the longest strictly-longer valid number.
        if len(text) == 1:
            longer = [
                r
                for hw in REREAD_HALF_WIDTHS_WORKING
                if (
                    r := reread_narrow(image, crnn, device, (cx, cy), factor, pages, hw)
                )
                and len(r[1]) > len(text)
            ]
            if longer:
                polygon, widened, confidence = max(longer, key=lambda r: len(r[1]))
                print(
                    f"{Path(image_path).name}: re-read narrow {text!r} -> {widened!r}",
                    file=sys.stderr,
                )
                text = widened
        found.append((polygon, text, confidence))

    # Trimming a between-two-numbers candidate can duplicate a neighbor's detection; keep the
    # higher-confidence box of any that now sit on the same number.
    if found:
        box_centers = [
            (sum(p[0] for p in poly) / 4, sum(p[1] for p in poly) / 4)
            for poly, _, _ in found
        ]
        scores = [confidence for _, _, confidence in found]
        keep = nms_peaks(box_centers, scores, round(DEDUP_WORKING / factor))
        found = [found[k] for k in keep]

    detections = [
        detection_record(cast(list, polygon), text, confidence)
        for polygon, text, confidence in found
    ]

    doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": detections,
    }
    with open(keymap_path(image_path), "w") as f:
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
