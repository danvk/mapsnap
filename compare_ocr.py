"""Compare EasyOCR vs PaddleOCR on Sanborn map images.

For each image, runs both engines (with the same 3-rotation strategy used in
detect_text.py) and reports: total detections, kept after georef_from_labels
filtering, and how many match known street names from the accompanying centerlines.

Usage:
    uv run python compare_ocr.py IMAGE CENTERLINES [IMAGE CENTERLINES ...]

Each IMAGE/CENTERLINES pair is listed on the command line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image

from mapsnap.detect_text import (
    DIRECTION_WORDS,
    canonical_street_matches,
    deduplicate_detections,
    normalize_street,
    polygon_side_lengths,
)
from mapsnap.georef_from_labels import build_block_index, is_number_only


# ---------------------------------------------------------------------------
# PaddleOCR helpers
# ---------------------------------------------------------------------------


def _paddle_readtext(ocr, img_array: np.ndarray) -> list[dict]:
    """Run PaddleOCR on one numpy image and return normalised detection dicts."""
    result = ocr.ocr(img_array, cls=True)
    detections: list[dict] = []
    if not result or result[0] is None:
        return detections
    for line in result[0]:
        bbox_raw, (text, confidence) = line
        polygon = [[int(x), int(y)] for x, y in bbox_raw]
        sides = polygon_side_lengths(polygon)
        detections.append(
            {
                "polygon": polygon,
                "text": text,
                "confidence": round(float(confidence), 4),
                "long_side": round(max(sides), 1),
                "short_side": round(min(sides), 1),
            }
        )
    return detections


def run_paddle(image_path: str) -> list[dict]:
    """Run PaddleOCR at 0°, 90°, 270° and return all raw detections."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    all_detections: list[dict] = []
    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        arr = np.array(rotated)
        dets = _paddle_readtext(ocr, arr)
        for det in dets:
            polygon = det["polygon"]
            if angle == 90:
                polygon = [[orig_width - 1 - y, x] for x, y in polygon]
            elif angle == 270:
                polygon = [[y, orig_height - 1 - x] for x, y in polygon]
            det["polygon"] = polygon
            det["angle"] = angle
            # Recompute sides after coordinate transform
            sides = polygon_side_lengths(polygon)
            det["long_side"] = round(max(sides), 1)
            det["short_side"] = round(min(sides), 1)
        all_detections.extend(dets)
    return all_detections


# ---------------------------------------------------------------------------
# EasyOCR helpers
# ---------------------------------------------------------------------------


def run_easyocr(image_path: str) -> list[dict]:
    """Run EasyOCR at 0°, 90°, 270° (same as detect_text.py defaults)."""
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ."
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    all_detections: list[dict] = []
    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        results = reader.readtext(
            np.array(rotated),
            paragraph=False,
            min_size=15,
            allowlist=allowlist,
        )
        for bbox, text, confidence in results:
            polygon = [[int(x), int(y)] for x, y in bbox]
            if angle == 90:
                polygon = [[orig_width - 1 - y, x] for x, y in polygon]
            elif angle == 270:
                polygon = [[y, orig_height - 1 - x] for x, y in polygon]
            sides = polygon_side_lengths(polygon)
            all_detections.append(
                {
                    "polygon": polygon,
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "angle": angle,
                    "long_side": round(max(sides), 1),
                    "short_side": round(min(sides), 1),
                }
            )
    return all_detections


# ---------------------------------------------------------------------------
# Filtering / scoring (mirrors georef_from_labels.py process_image)
# ---------------------------------------------------------------------------


def filter_and_match(
    detections: list[dict],
    block_index: dict,
    min_confidence: float = 0.5,
    min_long_side: float = 60.0,
    min_short_side: float = 12.0,
    min_aspect_ratio: float = 2.0,
    fuzzy_threshold: float = 0.20,
) -> tuple[list[dict], list[str]]:
    """Apply georef_from_labels size/confidence filters, then match against centerlines.

    Returns (kept_detections, matched_canonical_street_names).
    """
    normalized_streets = set(block_index.keys())
    deduped = deduplicate_detections(detections, normalized_streets=normalized_streets)
    kept: list[dict] = []
    for det in deduped:
        if not (
            det["confidence"] >= min_confidence
            and det.get("long_side", 0) >= min_long_side
            and det.get("short_side", 0) >= min_short_side
            and det.get("long_side", 0) >= min_aspect_ratio * det.get("short_side", 1)
            and not is_number_only(det["text"])
            and normalize_street(det["text"]) not in DIRECTION_WORDS
        ):
            continue
        matches = canonical_street_matches(
            det["text"], normalized_streets, fuzzy_threshold
        )
        if matches:
            det["_matches"] = matches
            kept.append(det)

    matched_streets: list[str] = []
    seen: set[str] = set()
    for det in kept:
        for m in det.get("_matches", []):
            if m not in seen:
                seen.add(m)
                matched_streets.append(m)

    return kept, matched_streets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compare_image(
    image_path: str,
    centerlines_path: str,
    easyocr_dets: list[dict],
    paddle_dets: list[dict],
) -> None:
    geojson = json.load(open(centerlines_path))
    block_index = build_block_index(geojson)

    easy_kept, easy_streets = filter_and_match(easyocr_dets, block_index)
    paddle_kept, paddle_streets = filter_and_match(paddle_dets, block_index)

    easy_set = set(easy_streets)
    paddle_set = set(paddle_streets)
    only_easy = sorted(easy_set - paddle_set)
    only_paddle = sorted(paddle_set - easy_set)
    both = sorted(easy_set & paddle_set)

    print(f"\n{'=' * 60}")
    print(f"Image: {Path(image_path).name}")
    print(f"{'=' * 60}")
    print(
        f"  EasyOCR:   {len(easyocr_dets):4d} raw → {len(easy_kept):3d} kept → {len(easy_streets):3d} streets matched"
    )
    print(
        f"  PaddleOCR: {len(paddle_dets):4d} raw → {len(paddle_kept):3d} kept → {len(paddle_streets):3d} streets matched"
    )
    print()
    if both:
        print(f"  Both detect ({len(both)}): {', '.join(both)}")
    if only_easy:
        print(f"  EasyOCR only ({len(only_easy)}): {', '.join(only_easy)}")
    if only_paddle:
        print(f"  PaddleOCR only ({len(only_paddle)}): {', '.join(only_paddle)}")
    print()
    print("  EasyOCR kept detections:")
    for d in easy_kept:
        print(f"    {d['text']!r:35s} conf={d['confidence']:.3f}  →  {d['_matches']}")
    print()
    print("  PaddleOCR kept detections:")
    for d in paddle_kept:
        print(f"    {d['text']!r:35s} conf={d['confidence']:.3f}  →  {d['_matches']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs",
        nargs="+",
        metavar="IMAGE_OR_CENTERLINES",
        help="Alternating IMAGE CENTERLINES paths",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory for caching engine results as JSON. If a cache file exists for "
            "an engine+image, it is loaded instead of re-running OCR. "
            "Pass --no-easy or --no-paddle to skip an engine entirely."
        ),
    )
    parser.add_argument(
        "--no-easy", action="store_true", help="Skip EasyOCR (load from cache only)"
    )
    parser.add_argument(
        "--no-paddle", action="store_true", help="Skip PaddleOCR (load from cache only)"
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("Arguments must be IMAGE CENTERLINES pairs (even count).")

    image_centerlines = [
        (args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)
    ]

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(image_path: str, engine: str) -> Path | None:
        if cache_dir is None:
            return None
        stem = Path(image_path).name.split(".")[0]
        return cache_dir / f"{stem}.{engine}.json"

    def load_cache(image_path: str, engine: str) -> list[dict] | None:
        p = cache_path(image_path, engine)
        if p and p.exists():
            print(f"  Loading {engine} from cache: {p.name}", file=sys.stderr)
            return json.load(p.open())
        return None

    def save_cache(image_path: str, engine: str, dets: list[dict]) -> None:
        p = cache_path(image_path, engine)
        if p:
            p.write_text(json.dumps(dets))

    # Run both engines on each image, collecting timing info.
    all_results = []
    for image_path, centerlines_path in image_centerlines:
        print(f"\nProcessing {Path(image_path).name} ...", file=sys.stderr)

        easy_dets = load_cache(image_path, "easy")
        easy_time = 0.0
        if easy_dets is None and not args.no_easy:
            print("  Running EasyOCR ...", file=sys.stderr)
            t0 = time.time()
            easy_dets = run_easyocr(image_path)
            easy_time = time.time() - t0
            print(
                f"  EasyOCR done in {easy_time:.0f}s ({len(easy_dets)} detections)",
                file=sys.stderr,
            )
            save_cache(image_path, "easy", easy_dets)
        easy_dets = easy_dets or []

        paddle_dets = load_cache(image_path, "paddle")
        paddle_time = 0.0
        if paddle_dets is None and not args.no_paddle:
            print("  Running PaddleOCR ...", file=sys.stderr)
            t0 = time.time()
            paddle_dets = run_paddle(image_path)
            paddle_time = time.time() - t0
            print(
                f"  PaddleOCR done in {paddle_time:.0f}s ({len(paddle_dets)} detections)",
                file=sys.stderr,
            )
            save_cache(image_path, "paddle", paddle_dets)
        paddle_dets = paddle_dets or []

        all_results.append(
            (
                image_path,
                centerlines_path,
                easy_dets,
                paddle_dets,
                easy_time,
                paddle_time,
            )
        )

    # Print comparison report.
    for (
        image_path,
        centerlines_path,
        easy_dets,
        paddle_dets,
        easy_time,
        paddle_time,
    ) in all_results:
        compare_image(image_path, centerlines_path, easy_dets, paddle_dets)

    # Timing summary.
    print("\nTiming summary:")
    for image_path, _, _, _, easy_time, paddle_time in all_results:
        name = Path(image_path).name
        print(f"  {name}: EasyOCR {easy_time:.0f}s  PaddleOCR {paddle_time:.0f}s")


if __name__ == "__main__":
    main()
