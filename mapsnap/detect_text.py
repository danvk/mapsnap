"""Detect text regions in insurance map images using EasyOCR (CRAFT detector)."""

import argparse
import json
import sys
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.ctc_vocab_decode import generate_vocab_strings
from mapsnap.streets import build_block_index, polygon_side_lengths
from mapsnap.utils import image_stem


def detect_text(
    image_path: str,
    vocab_strings: list[str],
    min_size: int = 15,
    allowlist: str | None = None,
    link_threshold: float = 0.4,
    reader: easyocr.Reader | None = None,
    beam_width: int = 20,
) -> list[dict]:
    """Run CRAFT-based text detection at 0°, 90°, and 270° and return all results.

    Runs three passes to catch both horizontal and vertical text. Polygons from
    rotated passes are mapped back to original image coordinates. Returns all raw
    detections — deduplication (NMS) is left to the caller, which has access to
    the street name list needed for street-aware NMS ordering.

    Note: EasyOCR's rotation_info parameter only rotates already-detected crops
    for the recognition stage, so it does not help detect vertical text regions.
    Running the full image at multiple angles is required.

    link_threshold controls how aggressively CRAFT merges adjacent text regions
    (EasyOCR default 0.4). Lower values (e.g. 0.1) prevent adjacent street labels
    from being concatenated into a single detection.

    vocab_strings enables prefix-constrained CTC decoding: the recognizer is
    restricted to outputting strings that are prefixes of known street-name forms.
    This substantially improves recall on abbreviated and direction-prefixed labels
    at the cost of ~25% slower recognition.

    Each returned detection is a dict with:
      - polygon: list of 4 [x, y] corners in original image coordinates
      - text: recognized text string
      - confidence: float in [0, 1]
      - angle: rotation pass (0, 90, or 270) that produced this detection
      - long_side: length of the longer pair of polygon sides (pixels)
      - short_side: length of the shorter pair of polygon sides (pixels)
    """
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    from mapsnap.ctc_vocab_decode import patch_easyocr_reader

    patch_easyocr_reader(reader, vocab_strings, beam_width)

    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    all_detections: list[dict] = []
    readtext_kwargs: dict = {
        "paragraph": False,
        "min_size": min_size,
        "link_threshold": link_threshold,
        "decoder": "wordbeamsearch",
        "beamWidth": beam_width,
    }
    if allowlist is not None:
        readtext_kwargs["allowlist"] = allowlist

    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        results = reader.readtext(np.array(rotated), **readtext_kwargs)
        for bbox, text, confidence in results:
            # Reject boxes that are taller than wide in rotated-image coordinates.
            # Valid text is always wider than tall in the rotated image; a tall box
            # means the detection is at the wrong angle (e.g. MONTCLAIR at angle=0
            # instead of 270, or RIVER at angle=90 instead of 0).
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            if (max(ys) - min(ys)) > (max(xs) - min(xs)):
                continue
            polygon = [[int(x), int(y)] for x, y in bbox]
            if angle == 90:
                # PIL rotate(90) is CCW; inverse: (rx, ry) -> (W-1-ry, rx)
                polygon = [[orig_width - 1 - y, x] for x, y in polygon]
            elif angle == 270:
                # PIL rotate(270) is CW; inverse: (rx, ry) -> (ry, H-1-rx)
                polygon = [[y, orig_height - 1 - x] for x, y in polygon]
            all_detections.append(
                {
                    "polygon": polygon,
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "angle": angle,
                }
            )

    for det in all_detections:
        sides = polygon_side_lengths(det["polygon"])
        det["long_side"] = round(max(sides), 1)
        det["short_side"] = round(min(sides), 1)
    return all_detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect text regions in insurance map images using EasyOCR (CRAFT)."
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Input image file(s). Detections are written to <stem>.streets.json.",
    )
    parser.add_argument(
        "--allowlist",
        metavar="CHARS",
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .",
        help=(
            "Restrict OCR recognition to these characters. Defaults to letters, space, "
            "and period (period separates direction abbreviations like 'E.' from the "
            "street name so normalize_street can expand them)."
        ),
    )
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=15,
        metavar="PX",
        help="Minimum short side passed to the EasyOCR detector (default: 15)",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        metavar="T",
        help=(
            "CRAFT link threshold controlling how aggressively adjacent text regions "
            "are merged. Lower values (e.g. 0.1) prevent adjacent street labels from "
            "being concatenated. EasyOCR default is 0.4."
        ),
    )
    parser.add_argument(
        "--centerlines",
        metavar="GEOJSON",
        required=True,
        help=(
            "Centerlines GeoJSON file. Used to build a vocabulary of known street-name "
            "forms for prefix-constrained CTC decoding, which substantially improves "
            "recall on abbreviated and direction-prefixed labels."
        ),
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=20,
        metavar="N",
        help="Beam width for constrained CTC decoder (default: 20)",
    )
    args = parser.parse_args()

    geojson = json.load(open(args.centerlines))
    block_index = build_block_index(geojson)
    vocab_strings = generate_vocab_strings(set(block_index.keys()))
    print(
        f"Constrained vocab: {len(vocab_strings)} forms from {len(block_index)} streets",
        file=sys.stderr,
    )

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    for image_path in tqdm(args.images, smoothing=0):
        stem = image_stem(image_path)
        output_path = str(Path(image_path).parent / (stem + ".streets.json"))
        detections = detect_text(
            image_path,
            vocab_strings=vocab_strings,
            min_size=args.min_short_side,
            allowlist=args.allowlist,
            link_threshold=args.link_threshold,
            reader=reader,
            beam_width=args.beam_width,
        )
        with open(output_path, "w") as f:
            f.write(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
