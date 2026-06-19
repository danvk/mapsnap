"""Detect page numbers on Sanborn key maps with vanilla EasyOCR.

Each pastel block on a key map holds the page number of the detailed sheet that
covers it. This script runs EasyOCR with default settings — default alphabet, greedy
decoder, no rotation (the numbers are printed upright) — over each key map and writes
a ``<stem>.streets.json`` sidecar in the same format as mapsnap.detect_text, so the
results can be inspected in the debugger app.

The page numbers are relatively large, so a large detector ``--min-size`` (default 60
px) skips small text and speeds detection up.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.streets import polygon_side_lengths
from mapsnap.utils import image_stem


def streets_path(image_path: str) -> str:
    """Return the path for the OCR results file (<stem>.streets.json)."""
    return str(Path(image_path).parent / (image_stem(image_path) + ".streets.json"))


def filter_args(argv: list[str], image: str) -> list[str]:
    """Compact a long argv by dropping every .jpg argument except ``image``."""
    return [arg for arg in argv if not arg.endswith(".jpg") or arg == image]


def detection_record(bbox: list[list[float]], text: str, confidence: float) -> dict:
    """Build one detection dict in the .streets.json format from an EasyOCR result.

    ``bbox`` is EasyOCR's quadrilateral (four [x, y] corners). The record matches the
    fields mapsnap.detect_text writes: polygon, text, confidence, angle (always 0 here
    since no rotation is used), long_side, short_side, and dir_pix (orientation of the
    longer side, radians in [0, pi)).
    """
    polygon = [[int(x), int(y)] for x, y in bbox]
    sides = polygon_side_lengths(polygon)
    pts = np.array(polygon, dtype=float)
    edge_vecs = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    long_vec = max(edge_vecs, key=np.linalg.norm)
    return {
        "polygon": polygon,
        "text": text,
        "confidence": round(float(confidence), 4),
        "angle": 0,
        "long_side": round(max(sides), 1),
        "short_side": round(min(sides), 1),
        "dir_pix": round(float(np.arctan2(long_vec[1], long_vec[0])) % np.pi, 4),
    }


def detect_page_numbers(
    image_path: str,
    reader: easyocr.Reader,
    min_size: int = 60,
) -> list[dict]:
    """Run vanilla EasyOCR over one key map and write its <stem>.streets.json sidecar.

    Detection and recognition both use EasyOCR defaults (no allowlist, greedy decoder,
    upright text only); only the detector ``min_size`` is raised to skip small text.
    Returns the list of detection records and writes them alongside the image.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    results = cast(list[tuple], reader.readtext(np.array(img), min_size=min_size))
    detections = [
        detection_record(bbox, text, confidence) for bbox, text, confidence in results
    ]

    streets_doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": detections,
    }
    with open(streets_path(image_path), "w") as f:
        json.dump(streets_doc, f, indent=2)

    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect page numbers on key maps with vanilla EasyOCR."
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Input key map image(s). Detections are written to <stem>.streets.json.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=60,
        metavar="PX",
        help="Minimum text size passed to the EasyOCR detector (default: %(default)s).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )
    args = parser.parse_args()

    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)
    for image_path in tqdm(args.images, smoothing=0):
        detect_page_numbers(image_path, reader, min_size=args.min_size)


if __name__ == "__main__":
    main()
