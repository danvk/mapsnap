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

# Non-street text that appears on Sanborn maps and should be recognized but
# excluded from georeferencing.
#
# Water pipe labels like '6" W. PIPE' are tricky: the Sanborn font renders '6"'
# as a tight glyph that the OCR model commonly misreads. The constrained CTC
# decoder then maps to "EMPIRE" (a real Queens street) instead.
#
# Two independent sources of variation compound each other:
#
# (1) How '6"' is read:  'E' (tight glyph, horizontal text),
#                        'S' (compressed, vertical text),
#                        '5' / 'G' (other instances)
#
# (2) How 'W' is read:   depends on the exact crop height fed to the recognition
#     model. EasyOCR upscales crops to a fixed 32px height, so a 25px-tall
#     bounding box is upscaled 1.28×, a 29px box 1.10×, etc. At different
#     scales the same 'W' glyph reads as W / X / Y / M / K — the dominant
#     letter can shift by a single pixel of crop height. All must be covered.
#
# The double-dashed underline is also captured inside the CRAFT bounding box,
# making the box ~40% taller than the text alone; this is why vertical-text
# instances read '6' as 'S' rather than 'E' (the glyph is compressed).
NON_STREET_TEXT: frozenset[str] = frozenset(
    # Exact forms with inch mark (legible in higher-quality scans)
    {f'{size}" W. PIPE' for size in ("2", "4", "6", "8", "10", "12", "16", "20")}
    # '6"' → 'E'; cover W → W / X / Y / M (scale-dependent)
    | {"EWPIPE", "EXPIPE", "EYPIPE", "EMPIPE"}
    | {"EW PIPE", "EX PIPE", "EY PIPE", "EM PIPE"}
    | {"EW. PIPE", "EX. PIPE", "EY. PIPE", "EM. PIPE"}
    # vertical text: '6' → 'S', '"' visible; W → W / X / Y / M
    | {'S"WPIPE', 'S"XPIPE', 'S"YPIPE', 'S"MPIPE'}
    | {'S" W. PIPE', 'S" W PIPE'}
    # vertical text: '6' → 'S', '"' dropped; W → W / M
    | {"SM PIPE", "S W PIPE", "S W. PIPE"}
)


def _erase_underlines(
    img_array: np.ndarray,
    boxes: list,
    dark_coverage: float = 0.25,
    scan_fraction: float = 0.25,
) -> np.ndarray:
    """Return a copy of img_array with dashed underlines painted white.

    Sanborn maps often print dashed underlines below street labels. CRAFT captures
    these in the bounding box, causing near-baseline characters (e.g. 'D') to be
    misread (e.g. as 'P'). Painting them out preserves the full bounding-box height
    (avoiding character clipping) while removing the noise from the recognizer.

    Scans the bottom scan_fraction of each ``[x_min, x_max, y_min, y_max]`` box.
    Rows in that region with >= dark_coverage fraction of dark pixels
    (grayscale < 128) are overwritten with white (255) within the box column range.
    """
    img_out = img_array.copy()
    gray = img_array.mean(axis=2)  # (H, W) grayscale
    H, W = gray.shape
    for x_min, x_max, y_min, y_max in boxes:
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        crop_h = y_max - y_min
        scan_start = y_max - max(1, int(crop_h * scan_fraction))
        col_start, col_end = max(0, x_min), min(W, x_max)
        for row in range(scan_start, y_max):
            if row >= H:
                break
            row_pixels = gray[row, col_start:col_end]
            if len(row_pixels) and (row_pixels < 128).mean() >= dark_coverage:
                img_out[row, col_start:col_end] = 255
    return img_out


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
      - ignore: True if the text matches a NON_STREET_TEXT pattern (absent otherwise)
    """
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    from mapsnap.ctc_vocab_decode import patch_easyocr_reader

    # Include non-street labels in the trie so they decode correctly rather
    # than being forced to a random street name.
    all_vocab = sorted(set(vocab_strings) | NON_STREET_TEXT)
    patch_easyocr_reader(reader, all_vocab, beam_width)

    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    all_detections: list[dict] = []
    recognize_kwargs: dict = {
        "paragraph": False,
        "decoder": "wordbeamsearch",
        "beamWidth": beam_width,
    }
    if allowlist is not None:
        recognize_kwargs["allowlist"] = allowlist

    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)
        horizontal_list_agg, free_list_agg = reader.detect(
            rotated_array, min_size=min_size, link_threshold=link_threshold
        )
        horizontal_list = horizontal_list_agg[0]
        free_list = free_list_agg[0]
        rotated_clean = _erase_underlines(rotated_array, horizontal_list)
        results = reader.recognize(
            rotated_clean, horizontal_list, free_list, **recognize_kwargs
        )
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
        if det["text"].upper() in NON_STREET_TEXT:
            det["ignore"] = True
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
        default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ."',
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have a .streets.json output file.",
    )
    args = parser.parse_args()

    geojson = json.load(open(args.centerlines))
    block_index = build_block_index(geojson)
    vocab_strings = generate_vocab_strings(set(block_index.keys()))
    print(
        f"Constrained vocab: {len(vocab_strings)} forms from {len(block_index)} streets",
        file=sys.stderr,
    )

    images = args.images
    if args.resume:
        images = [
            p
            for p in images
            if not (Path(p).parent / (image_stem(p) + ".streets.json")).exists()
        ]
        print(
            f"Resuming: {len(images)}/{len(args.images)} remaining images to process.",
            file=sys.stderr,
        )

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    for image_path in tqdm(images, smoothing=0):
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
