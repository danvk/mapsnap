"""Detect page numbers on Sanborn key maps with vocabulary-constrained EasyOCR.

Each pastel block on a key map holds the page number of the detailed sheet that covers
it, and we know up front exactly which page numbers a volume contains (e.g. Brooklyn
vol. 2 is pages 1-64). Rather than free-OCR the page and clean up afterward, this
constrains the recognizer to that closed vocabulary using the prefix-constrained CTC
beam search in mapsnap.ctc_vocab_decode: every detected box decodes directly to a valid
page number, and its confidence is the constrained path probability — so a box that
doesn't actually contain a page number (a lot number, a stray glyph) scores low instead
of confidently producing the wrong string.

Detection uses EasyOCR defaults with no rotation (the numbers are printed upright) and a
large ``--min-size`` (default 60 px), since the page numbers are relatively large.
Recognition is restricted to digits and to the trie of valid page-number strings.

Writes a ``<stem>.streets.json`` sidecar in the same format as mapsnap.detect_text (so
it loads in the debugger app); boxes that decode to no valid page number are dropped.
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

from mapsnap.ctc_vocab_decode import patch_easyocr_reader
from mapsnap.streets import polygon_side_lengths
from mapsnap.utils import image_stem

# EasyOCR's default beam width for the constrained CTC decoder.
DEFAULT_BEAM_WIDTH = 20

# Only digits can appear in a page number; restricting the recognizer's alphabet zeros
# out letter probabilities so the constrained digit paths are cleaner.
DIGIT_ALLOWLIST = "0123456789"


def parse_page_spec(spec: str) -> list[int]:
    """Parse a page-number spec like '1-64', '451-577', or '1,3,5-8' into a sorted list."""
    numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            numbers.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            numbers.add(int(part))
    return sorted(numbers)


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
    *,
    min_size: int = 60,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> list[dict]:
    """Run vocabulary-constrained EasyOCR over one key map and write its sidecar.

    ``reader`` must already be patched (via patch_easyocr_reader) with the volume's
    page-number vocabulary; this runs upright detection at ``min_size`` and the
    constrained ``wordbeamsearch`` decoder. Boxes that decode to no valid page number
    (empty text) are dropped. Returns the kept detection records and writes them to
    ``<stem>.streets.json``.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    results = cast(
        list[tuple],
        reader.readtext(
            np.array(img),
            decoder="wordbeamsearch",
            beamWidth=beam_width,
            allowlist=DIGIT_ALLOWLIST,
            min_size=min_size,
        ),
    )
    detections = [
        detection_record(bbox, text, confidence)
        for bbox, text, confidence in results
        if str(text).strip()
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
        description="Detect page numbers on key maps with vocabulary-constrained EasyOCR."
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Input key map image(s). Detections are written to <stem>.streets.json.",
    )
    parser.add_argument(
        "--pages",
        required=True,
        metavar="SPEC",
        help="Valid page numbers for the volume, e.g. '1-64' or '451-577' or '1,3,5-8'.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=60,
        metavar="PX",
        help="Minimum text size passed to the EasyOCR detector (default: %(default)s).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        metavar="N",
        help="Beam width for the constrained CTC decoder (default: %(default)s).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration.",
    )
    args = parser.parse_args()

    page_numbers = parse_page_spec(args.pages)
    vocab_strings = [str(n) for n in page_numbers]
    print(
        f"Constraining recognition to {len(vocab_strings)} page numbers "
        f"({page_numbers[0]}-{page_numbers[-1]})",
        file=sys.stderr,
    )

    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)
    patch_easyocr_reader(reader, vocab_strings, args.beam_width)

    for image_path in tqdm(args.images, smoothing=0):
        detect_page_numbers(
            image_path, reader, min_size=args.min_size, beam_width=args.beam_width
        )


if __name__ == "__main__":
    main()
