"""Detect text regions in insurance map images using EasyOCR (CRAFT detector)."""

import argparse
import json
import multiprocessing
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.ctc_vocab_decode import _HINT_STRINGS, generate_vocab_strings
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


def _boxes_path(image_path: str) -> str:
    """Return the path for the CRAFT boxes cache file (<stem>.boxes.json)."""
    stem = image_stem(image_path)
    return str(Path(image_path).parent / (stem + ".boxes.json"))


def _streets_path(image_path: str) -> str:
    """Return the path for the OCR results file (<stem>.streets.json)."""
    stem = image_stem(image_path)
    return str(Path(image_path).parent / (stem + ".streets.json"))


def _craft_detect_all_angles(
    img: Image.Image,
    reader: easyocr.Reader,
    min_size: int,
    link_threshold: float,
    craft_scale: float,
) -> list[dict]:
    """Run CRAFT detection at 0°, 90°, and 270° and return per-angle box data.

    Each element of the returned list is a dict with:
      - angle: rotation in degrees (0, 90, or 270)
      - horizontal_list: list of [x_min, x_max, y_min, y_max] in rotated-image coords
      - free_list: list of [[x, y], ...] polygon lists in rotated-image coords
    """
    craft_min_size = max(1, int(min_size * craft_scale))
    result = []
    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)
        if craft_scale != 1.0:
            rot_h, rot_w = rotated_array.shape[:2]
            small_w = max(1, int(rot_w * craft_scale))
            small_h = max(1, int(rot_h * craft_scale))
            detect_array = np.array(
                Image.fromarray(rotated_array).resize(
                    (small_w, small_h), Image.Resampling.LANCZOS
                )
            )
        else:
            detect_array = rotated_array
        horizontal_list_agg, free_list_agg = reader.detect(
            detect_array, min_size=craft_min_size, link_threshold=link_threshold
        )
        horizontal_list = horizontal_list_agg[0]
        free_list = free_list_agg[0]
        inv = 1.0 / craft_scale
        horizontal_list = [
            [int(b[0] * inv), int(b[1] * inv), int(b[2] * inv), int(b[3] * inv)]
            for b in horizontal_list
        ]
        free_list = [
            [[int(c[0] * inv), int(c[1] * inv)] for c in box] for box in free_list
        ]
        result.append(
            {
                "angle": angle,
                "horizontal_list": horizontal_list,
                "free_list": free_list,
            }
        )
    return result


def filter_args(argv: list[str], image: str) -> list[str]:
    """If this script is run with *.jpg, then argv can be very long. This compacts it.

    Specifically, it removes images from the command line other than the one of interest.
    """
    return [arg for arg in argv if not arg.endswith(".jpg") or arg == image]


def _box_center_in_original(
    box: list[int], angle: int, orig_width: int, orig_height: int
) -> tuple[float, float]:
    """Convert a horizontal_list box [x_min, x_max, y_min, y_max] in rotated image
    coordinates to its center in original (unrotated) image coordinates."""
    rx = (box[0] + box[1]) / 2.0
    ry = (box[2] + box[3]) / 2.0
    if angle == 0:
        return rx, ry
    elif angle == 90:
        # PIL rotate(90) CCW; inverse: (rx, ry) → original (W-1-ry, rx)
        return orig_width - 1 - ry, rx
    else:  # 270
        # PIL rotate(270) CW; inverse: (rx, ry) → original (ry, H-1-rx)
        return ry, orig_height - 1 - rx


def _recognize_near_hints(
    hint_detections: list[dict],
    skipped_by_angle: dict[int, list[list[int]]],
    img: Image.Image,
    reader: easyocr.Reader,
    recognize_kwargs: dict,
    orig_width: int,
    orig_height: int,
    max_gap_px: float = 80.0,
    perp_tolerance_px: float = 20.0,
) -> list[dict]:
    """Run recognition on small boxes (skipped by min_long_side) adjacent to hints.

    For each hint detection, finds horizontal_list boxes that were below the
    min_long_side threshold and lie within max_gap_px along the hint's direction
    and perp_tolerance_px perpendicular to it. Runs recognition only on those
    candidates, avoiding a full re-scan of all skipped boxes.

    Returns new detections (already post-processed with long_side, dir_pix, hint flag)
    that should be appended to all_detections before saving streets.json.
    """
    new_detections: list[dict] = []

    for angle in (0, 90, 270):
        skipped_h = skipped_by_angle.get(angle, [])
        angle_hints = [d for d in hint_detections if d.get("angle") == angle]
        if not skipped_h or not angle_hints:
            continue

        box_orig_centers = [
            _box_center_in_original(box, angle, orig_width, orig_height)
            for box in skipped_h
        ]

        # Find boxes adjacent to any hint in this angle pass.
        candidate_indices: set[int] = set()
        for hint in angle_hints:
            hint_cx = sum(p[0] for p in hint["polygon"]) / 4.0
            hint_cy = sum(p[1] for p in hint["polygon"]) / 4.0
            dir_pix = hint.get("dir_pix", 0.0)
            cos_d = float(np.cos(dir_pix))
            sin_d = float(np.sin(dir_pix))

            for i, (box_cx, box_cy) in enumerate(box_orig_centers):
                dx = box_cx - hint_cx
                dy = box_cy - hint_cy
                parallel = abs(dx * cos_d + dy * sin_d)
                perp = abs(-dx * sin_d + dy * cos_d)
                if parallel <= max_gap_px and perp <= perp_tolerance_px:
                    candidate_indices.add(i)

        if not candidate_indices:
            continue

        candidate_h = [skipped_h[i] for i in sorted(candidate_indices)]
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)
        rotated_clean = _erase_underlines(rotated_array, candidate_h)

        results = reader.recognize(rotated_clean, candidate_h, [], **recognize_kwargs)
        for bbox, text, confidence in results:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            if (max(ys) - min(ys)) > (max(xs) - min(xs)):
                continue
            polygon = [[int(x), int(y)] for x, y in bbox]
            if angle == 90:
                polygon = [[orig_width - 1 - y, x] for x, y in polygon]
            elif angle == 270:
                polygon = [[y, orig_height - 1 - x] for x, y in polygon]

            pts = np.array(polygon, dtype=float)
            sides = polygon_side_lengths(polygon)
            edge_vecs = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
            long_vec = max(edge_vecs, key=np.linalg.norm)
            det: dict = {
                "polygon": polygon,
                "text": text,
                "confidence": round(float(confidence), 4),
                "angle": angle,
                "long_side": round(max(sides), 1),
                "short_side": round(min(sides), 1),
                "dir_pix": round(
                    float(np.arctan2(long_vec[1], long_vec[0])) % np.pi, 4
                ),
                "second_pass": True,
            }
            if text.upper() in NON_STREET_TEXT:
                det["ignore"] = True
            elif text.upper() in _HINT_STRINGS:
                det["hint"] = True
            new_detections.append(det)

    return new_detections


def detect_text(
    image_path: str,
    vocab_strings: list[str],
    min_size: int = 15,
    min_long_side: int = 0,
    allowlist: str | None = None,
    link_threshold: float = 0.4,
    reader: easyocr.Reader | None = None,
    beam_width: int = 20,
    craft_scale: float = 1.0,
    reuse_boxes: bool = False,
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

    min_long_side skips recognition for boxes whose long side (max of width and
    height in rotated-image coordinates) is below this threshold. Boxes are still
    detected by CRAFT but never passed to the recognizer, so they do not appear in
    the output. Set this to match the --min-long-side used by georef_from_labels.py
    to avoid spending time recognizing text that will be filtered downstream.

    craft_scale downsizes the image before CRAFT detection (e.g. 0.5 = half
    resolution). CRAFT CNN cost scales quadratically with image area, so 0.5
    gives ~4× faster detection. Detected bounding boxes are scaled back up to
    original coordinates before recognition, which always runs at full resolution.
    min_size is also scaled proportionally so the same physical text size threshold
    applies. Ignored when reuse_boxes=True.

    reuse_boxes loads CRAFT bounding boxes from the existing <stem>.boxes.json
    file instead of re-running CRAFT. Useful for iterating on recognition
    parameters without redoing the (slower) detection step. The caller is
    responsible for ensuring the file exists before calling with reuse_boxes=True.

    Writes <stem>.boxes.json alongside the image whenever CRAFT is run (i.e.
    when reuse_boxes=False). The file records the image dimensions, a timestamp,
    the command line, and the per-angle box data.

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

    if reuse_boxes:
        with open(_boxes_path(image_path)) as f:
            angle_boxes: list[dict] = json.load(f)["boxes"]
    else:
        angle_boxes = _craft_detect_all_angles(
            img, reader, min_size, link_threshold, craft_scale
        )
        boxes_doc = {
            "width": orig_width,
            "height": orig_height,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": filter_args(sys.argv[:], image_path),
            "boxes": angle_boxes,
        }
        with open(_boxes_path(image_path), "w") as f:
            json.dump(boxes_doc, f, indent=2)

    all_detections: list[dict] = []
    skipped_by_angle: dict[int, list[list[int]]] = {}
    recognize_kwargs: dict = {
        "paragraph": False,
        "decoder": "wordbeamsearch",
        "beamWidth": beam_width,
    }
    if allowlist is not None:
        recognize_kwargs["allowlist"] = allowlist

    for angle_data in angle_boxes:
        angle = angle_data["angle"]
        horizontal_list = list(angle_data["horizontal_list"])
        free_list = list(angle_data["free_list"])

        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)

        if min_long_side > 0:
            skipped_by_angle[angle] = [
                b for b in horizontal_list if (b[1] - b[0]) <= min_long_side
            ]
            horizontal_list = [
                b for b in horizontal_list if (b[1] - b[0]) > min_long_side
            ]
            free_list = [
                b
                for b in free_list
                if max(
                    max(c[0] for c in b) - min(c[0] for c in b),
                    max(c[1] for c in b) - min(c[1] for c in b),
                )
                > min_long_side
            ]
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
        pts = np.array(det["polygon"], dtype=float)
        sides = polygon_side_lengths(det["polygon"])
        det["long_side"] = round(max(sides), 1)
        det["short_side"] = round(min(sides), 1)
        edge_vecs = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
        long_vec = max(edge_vecs, key=np.linalg.norm)
        det["dir_pix"] = round(float(np.arctan2(long_vec[1], long_vec[0])) % np.pi, 4)
        if det["text"].upper() in NON_STREET_TEXT:
            det["ignore"] = True
        elif det["text"].upper() in _HINT_STRINGS:
            det["hint"] = True

    # Second pass: recognize small boxes adjacent to hint detections.
    hint_dets = [d for d in all_detections if d.get("hint")]
    if hint_dets and min_long_side > 0:
        second_pass = _recognize_near_hints(
            hint_dets,
            skipped_by_angle,
            img,
            reader,
            recognize_kwargs,
            orig_width,
            orig_height,
        )
        all_detections.extend(second_pass)

    streets_doc = {
        "width": orig_width,
        "height": orig_height,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "streets": all_detections,
    }
    with open(_streets_path(image_path), "w") as f:
        json.dump(streets_doc, f, indent=2)

    return all_detections


# Module-level state populated by _worker_init in each worker process.
_worker_state: dict[str, Any] = {}


def _worker_init(
    vocab_strings: list[str],
    min_size: int,
    min_long_side: int,
    allowlist: str | None,
    link_threshold: float,
    beam_width: int,
    craft_scale: float,
    reuse_boxes: bool,
    gpu: bool,
) -> None:
    """Initialize per-worker state once per process: create the EasyOCR reader."""
    _worker_state["reader"] = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    _worker_state["vocab_strings"] = vocab_strings
    _worker_state["min_size"] = min_size
    _worker_state["min_long_side"] = min_long_side
    _worker_state["allowlist"] = allowlist
    _worker_state["link_threshold"] = link_threshold
    _worker_state["beam_width"] = beam_width
    _worker_state["craft_scale"] = craft_scale
    _worker_state["reuse_boxes"] = reuse_boxes


def _process_image(image_path: str) -> str:
    """Process one image in a worker process, writing output to <stem>.streets.json."""
    detect_text(
        image_path,
        vocab_strings=_worker_state["vocab_strings"],
        min_size=_worker_state["min_size"],
        min_long_side=_worker_state["min_long_side"],
        allowlist=_worker_state["allowlist"],
        link_threshold=_worker_state["link_threshold"],
        reader=_worker_state["reader"],
        beam_width=_worker_state["beam_width"],
        craft_scale=_worker_state["craft_scale"],
        reuse_boxes=_worker_state["reuse_boxes"],
    )
    return image_path


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes (default: 1). Each worker loads its "
            "own EasyOCR reader. With GPU enabled, workers share the GPU via CUDA "
            "context switching; each worker requires ~500 MB of VRAM for model weights."
        ),
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (recommended with --num-workers > 1).",
    )
    parser.add_argument(
        "--min-long-side",
        type=int,
        default=0,
        metavar="PX",
        help=(
            "Skip recognition for CRAFT detections whose long side is below this "
            "threshold (default: 0, no filtering). Set to match the --min-long-side "
            "used by georef_from_labels.py to avoid recognizing boxes that will be "
            "filtered downstream."
        ),
    )
    parser.add_argument(
        "--craft-scale",
        type=float,
        default=1.0,
        metavar="S",
        help=(
            "Scale factor applied to images before CRAFT detection (default: 1.0). "
            "0.5 halves each dimension, reducing CRAFT CNN cost ~4×. Detected boxes "
            "are scaled back to original coordinates; recognition always runs at full "
            "resolution."
        ),
    )
    parser.add_argument(
        "--reuse-boxes",
        action="store_true",
        help=(
            "Reuse CRAFT bounding boxes from existing <stem>.boxes.json files instead "
            "of re-running CRAFT detection. Useful for iterating on recognition "
            "parameters without redoing detection. All images must already have a "
            ".boxes.json file; this flag aborts with an error if any are missing."
        ),
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

    if args.reuse_boxes:
        missing = [p for p in images if not Path(_boxes_path(p)).exists()]
        if missing:
            for p in missing:
                print(f"Missing boxes file: {_boxes_path(p)}", file=sys.stderr)
            sys.exit(
                f"--reuse-boxes: {len(missing)} image(s) have no .boxes.json file."
            )

    gpu = not args.no_gpu

    if args.num_workers > 1:
        initargs = (
            vocab_strings,
            args.min_short_side,
            args.min_long_side,
            args.allowlist,
            args.link_threshold,
            args.beam_width,
            args.craft_scale,
            args.reuse_boxes,
            gpu,
        )
        with multiprocessing.Pool(
            args.num_workers,
            initializer=_worker_init,
            initargs=initargs,
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_image, images),
                total=len(images),
                smoothing=0,
            ):
                pass
    else:
        reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        for image_path in tqdm(images, smoothing=0):
            detect_text(
                image_path,
                vocab_strings=vocab_strings,
                min_size=args.min_short_side,
                min_long_side=args.min_long_side,
                allowlist=args.allowlist,
                link_threshold=args.link_threshold,
                reader=reader,
                beam_width=args.beam_width,
                craft_scale=args.craft_scale,
                reuse_boxes=args.reuse_boxes,
            )


if __name__ == "__main__":
    main()
