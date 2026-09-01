"""Detect text regions in insurance map images using EasyOCR (CRAFT detector)."""

import argparse
import json
import math
import multiprocessing
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.ctc_vocab_decode import HINT_STRINGS, generate_vocab_strings
from mapsnap.keymap.locate import KeymapLocator, page_key, resolve_keymaps
from mapsnap.panel_boxes import rotate_point, unrotate_point
from mapsnap.streets import build_block_index, polygon_side_lengths
from mapsnap.utils import default_centerlines, image_stem

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

SCALE_NOTE_TEXT: frozenset[str] = frozenset(
    # The printed scale note on oddly-scaled sheets ("Scale 100 Ft. to One
    # Inch.") and the scale-bar tick numbers (a 50ft bar reads 50-25-0-50-100,
    # a 100ft bar 100-50-0-100-200). Only unioned into the trie for the
    # rotation-0 pass: the note and bar are always horizontal, and keeping the
    # rotated passes' vocabulary unchanged avoids collateral on vertical
    # street labels. Reads matching these are tagged ignore, like the pipe
    # annotations, so they never enter street matching.
    {"SCALE", "INCH", "ONE INCH", "FT", "FT.", "25", "50", "100", "200", "300"}
    | {
        f"SCALE {n} FT. TO ONE INCH"
        for n in ("50", "100", "IOO", "I00", "200", "2OO", "300", "3OO")
    }
    | {f"SCALE {n} FT TO ONE INCH" for n in ("50", "100", "200", "300")}
    | {f"{n} FT. TO ONE INCH" for n in ("50", "100", "200", "300")}
)

# How much a detection's box must exceed the page's own paper chroma (CIELAB) before its
# background colour is recorded rather than treated as paper. 4.0 separates Nashville's building
# labels (p51 "REP" at 9-14 chroma over 2.2 paper) from its street labels (1.0-3.6).
BACKGROUND_CHROMA_MARGIN = 4.0


def page_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB image from an RGB array, with a/b re-centered on 0.

    cv2 stores 8-bit Lab with a/b offset by 128; this subtracts that so hue and chroma are
    ordinary polar coordinates of (a, b). skimage's rgb2lab is not an option here: it segfaults
    once scipy's MINPACK has run in-process, which it has by the time georeferencing calls back.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float64)
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def lab_to_hex(lightness: float, a: float, b: float) -> str:
    """sRGB hex string ('#rrggbb') for a single CIELAB colour with a/b centered on 0."""
    values = np.clip([lightness, a + 128.0, b + 128.0], 0, 255)
    pixel = np.array([[values]], dtype=np.uint8)
    red, green, blue = cv2.cvtColor(pixel, cv2.COLOR_Lab2RGB)[0, 0]
    return f"#{red:02x}{green:02x}{blue:02x}"


def region_color(lab: np.ndarray, polygon: list) -> dict:
    """The median colour under a polygon's bounding box, as {color, hue, chroma}.

    A detection's box holds the glyphs (near-neutral ink) plus whatever they sit on, so the
    median reports the *background*: paper for a street label, the fill colour for a label
    printed on a building. ``a`` and ``b`` are medianed separately and combined afterwards,
    because hue is circular and a median over it is not meaningful.
    """
    pts = np.asarray(polygon, dtype=float)
    x0, x1 = max(0, int(pts[:, 0].min())), int(pts[:, 0].max())
    y0, y1 = max(0, int(pts[:, 1].min())), int(pts[:, 1].max())
    region = lab[y0 : y1 + 1, x0 : x1 + 1]
    if not region.size:
        return {"color": "#000000", "hue": 0.0, "chroma": 0.0}
    lightness = float(np.median(region[:, :, 0]))
    a = float(np.median(region[:, :, 1]))
    b = float(np.median(region[:, :, 2]))
    return {
        "color": lab_to_hex(lightness, a, b),
        "hue": round(math.degrees(math.atan2(b, a)) % 360.0, 1),
        "chroma": round(math.hypot(a, b), 1),
    }


def annotate_backgrounds(
    detections: list[dict],
    rgb: np.ndarray,
    margin: float = BACKGROUND_CHROMA_MARGIN,
) -> dict:
    """Record each detection's background colour when it differs from the page's paper.

    Sanborn sheets print street names on the paper background, so text sitting inside a coloured
    block is a *building* label. Sets ``background`` ({color, hue, chroma}) on every detection
    whose box is more saturated than the paper by more than ``margin``, and leaves it unset
    otherwise — the property's presence alone means "this label is not on paper". Deciding which
    of those colours disqualify a label is left to the reader of the file (see
    ``georef_from_labels.drop_labels_on_fill``), so re-OCR is not needed to change that policy.

    The chroma reference is the page's own because paper varies enormously between volumes:
    near-white in Nashville (chroma 1-5), heavily yellowed in Chicago and New Orleans (13-20).

    Returns the paper colour itself, for the caller to record alongside the detections.
    """
    lab = page_lab(rgb)
    lab_a, lab_b = lab[:, :, 1], lab[:, :, 2]
    # Chroma is medianed per pixel (the typical saturation), while the paper's displayed colour
    # comes from the median a/b — the right reduction for each, so they differ slightly.
    paper_chroma = float(np.median(np.hypot(lab_a, lab_b)))
    paper_a, paper_b = float(np.median(lab_a)), float(np.median(lab_b))
    for det in detections:
        background = region_color(lab, det["polygon"])
        if background["chroma"] > paper_chroma + margin:
            det["background"] = background
    return {
        "color": lab_to_hex(float(np.median(lab[:, :, 0])), paper_a, paper_b),
        "hue": round(math.degrees(math.atan2(paper_b, paper_a)) % 360.0, 1),
        "chroma": round(paper_chroma, 1),
    }


def backfill_backgrounds(image_path: str) -> int:
    """Add ``background`` to an existing <stem>.streets.json in place, without re-running OCR.

    Background colour is a pure function of the image and the detection boxes, so a file OCR'd
    before the property existed can be brought up to date far more cheaply than by re-reading the
    page. Returns the number of detections found to be on a coloured fill.
    """
    path = _streets_path(image_path)
    with open(path) as f:
        streets_doc = json.load(f)
    detections = streets_doc["streets"]
    for det in detections:
        det.pop("background", None)
    with Image.open(image_path) as img:
        streets_doc["paper"] = annotate_backgrounds(
            detections, np.array(img.convert("RGB"))
        )
    with open(path, "w") as f:
        json.dump(streets_doc, f, indent=2)
    return sum(1 for det in detections if "background" in det)


def boxes_path(image_path: str) -> str:
    """Return the path for the CRAFT boxes cache file (<stem>.boxes.json)."""
    stem = image_stem(image_path)
    return str(Path(image_path).parent / (stem + ".boxes.json"))


# Backwards-compatible alias for the pre-`mapsnap craft` private name.
_boxes_path = boxes_path


def craft_hint(image_paths: list[str] | list[Path]) -> str:
    """The `mapsnap craft` command that would produce the missing boxes.

    CRAFT detection is its own step (`mapsnap craft`), so every consumer of
    <stem>.boxes.json fails with an actionable message rather than silently
    re-running the slowest stage of the pipeline under a different command.
    """
    parents = {str(Path(p).parent) for p in image_paths}
    target = f"{next(iter(parents))}/p*.jpg" if len(parents) == 1 else "<images...>"
    return f"mapsnap craft '{target}'"


def missing_boxes(image_paths: list[str]) -> list[str]:
    """Images with no CRAFT boxes cache, in input order."""
    return [path for path in image_paths if not Path(boxes_path(path)).exists()]


def require_boxes(image_paths: list[str]) -> None:
    """Exit with the craft command to run when any image lacks its boxes."""
    absent = missing_boxes(image_paths)
    if not absent:
        return
    names = ", ".join(Path(p).name for p in absent[:5])
    more = f" (+{len(absent) - 5} more)" if len(absent) > 5 else ""
    sys.exit(
        f"No CRAFT boxes for {len(absent)} image(s): {names}{more}\n"
        f"Run: {craft_hint(absent)}"
    )


def write_craft_boxes(
    image_path: str,
    reader: easyocr.Reader,
    *,
    min_size: int = 15,
    link_threshold: float = 0.4,
    craft_scale: float = 1.0,
    tile_size: int = 2560,
) -> dict:
    """Run CRAFT at 0/90/270 on one image and write <stem>.boxes.json."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    angle_boxes = _craft_detect_all_angles(
        img, reader, min_size, link_threshold, craft_scale, tile_size
    )
    doc = {
        "width": width,
        "height": height,
        "timestamp": datetime.now(UTC).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "boxes": angle_boxes,
    }
    with open(boxes_path(image_path), "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def _streets_path(image_path: str) -> str:
    """Return the path for the OCR results file (<stem>.streets.json)."""
    stem = image_stem(image_path)
    return str(Path(image_path).parent / (stem + ".streets.json"))


def has_split_panels(image_path: str) -> bool:
    """True if image_path is a whole page already split into <stem>__N.jpg panels.

    Such a parent page is superseded by its panels and should not be OCR'd; the panels
    are processed instead. Returns False for the panels themselves (stems with '__').
    """
    stem = image_stem(image_path)
    if "__" in stem:
        return False
    return any(Path(image_path).parent.glob(f"{stem}__*.jpg"))


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile-start offsets covering [0, length]; the final tile sits flush with the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _iter_tiles(
    width: int, height: int, tile_size: int, overlap: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (x0, y0, x1, y1) tiles of at most tile_size px, overlapping by ``overlap``."""
    stride = max(1, tile_size - overlap)
    for y0 in _axis_starts(height, tile_size, stride):
        for x0 in _axis_starts(width, tile_size, stride):
            yield x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)


def _iou_xxyy(a: list[float], b: list[float]) -> float:
    """IoU of two [x_min, x_max, y_min, y_max] boxes."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[2], b[2])
    )
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[1] - a[0]) * max(0.0, a[3] - a[2])
    area_b = max(0.0, b[1] - b[0]) * max(0.0, b[3] - b[2])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_bboxes(bboxes: list[list[float]], iou_threshold: float) -> list[int]:
    """Greedy NMS over [x_min,x_max,y_min,y_max] boxes; returns kept indices (larger first).

    Used to drop near-duplicate detections produced where adjacent tiles overlap: the
    larger (more complete) box wins over a copy clipped at a tile seam.
    """
    order = sorted(
        range(len(bboxes)),
        key=lambda i: (bboxes[i][1] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][2]),
        reverse=True,
    )
    kept: list[int] = []
    for i in order:
        if all(_iou_xxyy(bboxes[i], bboxes[j]) <= iou_threshold for j in kept):
            kept.append(i)
    return kept


def _detect_frame(
    reader: easyocr.Reader,
    array: np.ndarray,
    min_size: int,
    link_threshold: float,
    craft_scale: float,
    tile_size: int,
) -> tuple[list, list]:
    """Run CRAFT on one image frame, tiling frames larger than ``tile_size``.

    EasyOCR caps the CRAFT input's long side at canvas_size (2560px), so a frame larger
    than that is downscaled before detection — shrinking small labels below the
    detector's resolution (an 8422px key map is seen at ~0.30×, turning a 40px label
    into ~12px). When tile_size > 0 and the frame's long side exceeds it, the frame is
    split into overlapping tiles detected at native resolution and their boxes merged
    with NMS. Only detection is affected: recognition always crops from the full-
    resolution image regardless of how boxes were found.

    Returns (horizontal_list, free_list) in ``array`` coordinates. tile_size <= 0
    disables tiling: the frame is detected in one pass, optionally downscaled by
    craft_scale.
    """
    height, width = array.shape[:2]

    if tile_size <= 0 or max(width, height) <= tile_size:
        craft_min_size = max(1, int(min_size * craft_scale))
        if craft_scale != 1.0:
            detect_array = np.array(
                Image.fromarray(array).resize(
                    (
                        max(1, int(width * craft_scale)),
                        max(1, int(height * craft_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            )
        else:
            detect_array = array
        horizontal_agg, free_agg = reader.detect(
            detect_array, min_size=craft_min_size, link_threshold=link_threshold
        )
        inv = 1.0 / craft_scale
        horizontal = [
            [int(b[0] * inv), int(b[1] * inv), int(b[2] * inv), int(b[3] * inv)]
            for b in horizontal_agg[0]
        ]
        free = [
            [[int(c[0] * inv), int(c[1] * inv)] for c in poly] for poly in free_agg[0]
        ]
        return horizontal, free

    # Tiled detection at native resolution: every tile is <= tile_size on its long side
    # so reader.detect does not downscale it.
    craft_min_size = max(1, min_size)
    overlap = tile_size // 5
    horizontal: list = []
    free: list = []
    tiles = [*_iter_tiles(width, height, tile_size, overlap)]
    print(
        f"Tiling large image ({width}x{height}) into {len(tiles)} tiles.",
        file=sys.stderr,
    )
    for x0, y0, x1, y1 in tiles:
        tile = array[y0:y1, x0:x1]
        horizontal_agg, free_agg = reader.detect(
            tile, min_size=craft_min_size, link_threshold=link_threshold
        )
        for b in horizontal_agg[0]:
            horizontal.append(
                [int(b[0]) + x0, int(b[1]) + x0, int(b[2]) + y0, int(b[3]) + y0]
            )
        for poly in free_agg[0]:
            free.append([[int(c[0]) + x0, int(c[1]) + y0] for c in poly])

    keep_h = _nms_bboxes(horizontal, 0.4)
    free_bboxes = [
        [
            min(c[0] for c in p),
            max(c[0] for c in p),
            min(c[1] for c in p),
            max(c[1] for c in p),
        ]
        for p in free
    ]
    keep_f = _nms_bboxes(free_bboxes, 0.4)
    return [horizontal[i] for i in keep_h], [free[i] for i in keep_f]


def _craft_detect_all_angles(
    img: Image.Image,
    reader: easyocr.Reader,
    min_size: int,
    link_threshold: float,
    craft_scale: float,
    tile_size: int,
) -> list[dict]:
    """Run CRAFT detection at 0°, 90°, and 270° and return per-angle box data.

    Each element of the returned list is a dict with:
      - angle: rotation in degrees (0, 90, or 270)
      - horizontal_list: list of [x_min, x_max, y_min, y_max] in rotated-image coords
      - free_list: list of [[x, y], ...] polygon lists in rotated-image coords

    Frames larger than tile_size are detected in overlapping native-resolution tiles
    (see _detect_frame) so small labels survive canvas_size downscaling.
    """
    result = []
    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)
        horizontal_list, free_list = _detect_frame(
            reader, rotated_array, min_size, link_threshold, craft_scale, tile_size
        )
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


DEFAULT_RECOGNIZER_WEIGHTS = (
    Path(__file__).resolve().parent.parent / "models" / "street_recognizer.pt"
)
"""The fine-tuned street recognizer, used unless --stock-recognizer is passed.

Measured against the stock EasyOCR model on held-out volumes: fargo +13.4 and
richmond +8.5 land-weighted points, neither in training. Shipping it as the
default means a run has to opt OUT of the better reads rather than remember to
opt in -- 15 pages of the corpus were still on stock weights months after the
model landed, purely because a re-OCR had been run without the flag.
"""


def cached_recognizer(streets_path: Path) -> str | None:
    """Which recognizer produced a cached read, by weights filename, or None.

    ``streets.json`` records the command that wrote it, so the weights are
    recoverable. None means EasyOCR's stock model -- either no flag was passed
    or the file predates the flag.
    """
    try:
        command = json.loads(streets_path.read_text()).get("command") or []
    except (OSError, ValueError):
        return None
    for i, token in enumerate(command):
        if token == "--recognizer-weights" and i + 1 < len(command):
            return Path(command[i + 1]).name
    return None


def reads_are_current(streets_path: Path, weights: str | None) -> bool:
    """Whether a cached read exists AND came from the recognizer now in use.

    ``--resume`` used to test only that the file existed, so a re-run with
    different weights silently kept the old reads. That is how 15 pages across
    8 volumes stayed on EasyOCR's stock model for a month after the fine-tuned
    one shipped: each was skipped by a --resume run whose weights differed.
    """
    if not streets_path.exists():
        return False
    return cached_recognizer(streets_path) == (Path(weights).name if weights else None)


def load_recognizer_weights(reader: easyocr.Reader, weights_path: str) -> None:
    """Swap fine-tuned recognizer weights (#265) into an EasyOCR reader.

    The checkpoint is a plain state_dict for easyocr's recognition model
    (written by mapsnap.train_street_recognizer), so the architecture, charset,
    and the constrained CTC decoder downstream are all unchanged — only the
    learned weights differ.
    """
    import torch

    state = torch.load(weights_path, map_location="cpu")
    model = reader.recognizer
    inner = model.module if hasattr(model, "module") else model
    inner.load_state_dict(state)
    print(f"Recognizer weights: {weights_path}", file=sys.stderr)


def _recognize_pass(
    reader: easyocr.Reader,
    img: Image.Image,
    angle_boxes: list[dict],
    *,
    vocab: list[str],
    beam_width: int,
    allowlist: str | None,
    min_long_side: int,
    orig_width: int,
    orig_height: int,
) -> list[dict]:
    """Recognize the cached CRAFT boxes at all angles with one vocabulary.

    Returns one {polygon, text, confidence, angle} detection per surviving box (mapped back to
    original-image coordinates). The vocabulary is applied via the prefix-constrained decoder,
    so a smaller vocabulary yields more confident, less ambiguous reads.
    """
    from mapsnap.ctc_vocab_decode import patch_easyocr_reader

    recognize_kwargs: dict = {
        "paragraph": False,
        "decoder": "wordbeamsearch",
        "beamWidth": beam_width,
    }
    if allowlist is not None:
        recognize_kwargs["allowlist"] = allowlist

    detections: list[dict] = []
    for angle_data in angle_boxes:
        angle = angle_data["angle"]
        # Include non-street labels in the trie so they decode correctly rather
        # than being forced to a random street name. The scale-note terms join
        # only the rotation-0 pass (the note and bar are always horizontal).
        extra = NON_STREET_TEXT | (SCALE_NOTE_TEXT if angle == 0 else frozenset())
        patch_easyocr_reader(reader, sorted(set(vocab) | extra), beam_width)
        horizontal_list = list(angle_data["horizontal_list"])
        free_list = list(angle_data["free_list"])

        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        rotated_array = np.array(rotated)

        if min_long_side > 0:
            horizontal_list = [
                b for b in horizontal_list if (b[1] - b[0]) >= min_long_side
            ]
            free_list = [
                b
                for b in free_list
                if max(
                    max(c[0] for c in b) - min(c[0] for c in b),
                    max(c[1] for c in b) - min(c[1] for c in b),
                )
                >= min_long_side
            ]
        results = reader.recognize(
            rotated_array, horizontal_list, free_list, **recognize_kwargs
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
            detections.append(
                {
                    "polygon": polygon,
                    "text": text,
                    "confidence": round(float(confidence), 4),
                    "angle": angle,
                }
            )
    return detections


def _merge_vocab_passes(primary: list[dict], fallback: list[dict]) -> list[dict]:
    """Merge a restricted-vocab pass with a broader fallback pass, box by box.

    Both passes recognize the same CRAFT boxes, so they align by polygon. For each box the
    higher-confidence read wins; when the broader fallback vocabulary wins with a *different*
    text, that street lies outside the page's own neighborhood, so the detection is marked
    ``fallback: True`` (a lower-location-confidence label for downstream georeferencing).
    """
    fallback_by_polygon = {
        tuple(tuple(point) for point in d["polygon"]): d for d in fallback
    }
    merged: list[dict] = []
    for detection in primary:
        alt = fallback_by_polygon.get(tuple(tuple(p) for p in detection["polygon"]))
        if (
            alt is not None
            and alt["confidence"] > detection["confidence"]
            and alt["text"] != detection["text"]
        ):
            merged.append({**alt, "fallback": True})
        else:
            merged.append(detection)
    return merged


# --- Cross-orientation box transfer (#144) -----------------------------------
#
# CRAFT segments the same text differently at 0/90/270: the natural orientation
# can cut a word in half (brooklyn v2 p19 TILLARY), glom it with an adjacency
# stamp (richmond p365 "373 HAMMOND"), or attach a prefix that poisons the
# constrained decode (detroit p85 "S. TENNESSEE"). When a LARGE box fails to
# read, the other orientations' differently-segmented footprints are re-read in
# the frame where the text is upright, and the failing footprint itself is
# re-read in the other frames. Transferred reads are flagged ``transferred``
# (with ``transfer_kind`` and ``transfer_source_angle``) for downstream use.

# A trigger box is at least this long and this wide-to-tall in its own frame
# (TILLARY's polygon is 111x55, "S. TENNESSEE" 92x24).
TRANSFER_MIN_LONG = 80
TRANSFER_MIN_ASPECT = 2.0
# A read at or above this confidence needs no help.
TRANSFER_CONFIDENT = 0.3
# Transferred reads below this are junk splits of titles/legends; drop them.
TRANSFER_FLOOR = 0.2
# Same-text overlap dedupe: within this confidence delta the EXISTING read wins,
# so a transferred twin of a read we already had is never flagged as new.
TRANSFER_TIE = 0.02


def reading_order(quad: list) -> list[list[int]]:
    """Order a quad [tl, tr, br, bl] with tl->tr along its long axis, left to right.

    EasyOCR's four_point_transform takes the FIRST edge as the crop width, so a
    quad rotated into another frame must be re-ordered or the warped crop comes
    out sideways/mirrored (TILLARY read 'E' until this).
    """
    pts = [(float(x), float(y)) for x, y in quad]
    cx = sum(q[0] for q in pts) / 4
    cy = sum(q[1] for q in pts) / 4
    e0 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    e1 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    ux, uy = e0 if math.hypot(*e0) >= math.hypot(*e1) else e1
    norm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm
    if ux < 0 or (ux == 0 and uy < 0):
        ux, uy = -ux, -uy
    nx, ny = -uy, ux
    proj = [
        ((q[0] - cx) * ux + (q[1] - cy) * uy, (q[0] - cx) * nx + (q[1] - cy) * ny, q)
        for q in pts
    ]
    top = sorted([q for q in proj if q[1] < 0], key=lambda q: q[0])
    bottom = sorted([q for q in proj if q[1] >= 0], key=lambda q: q[0])
    if len(top) != 2:
        return [[int(v) for v in q] for q in pts]
    return [
        [int(v) for v in top[0][2]],
        [int(v) for v in top[1][2]],
        [int(v) for v in bottom[1][2]],
        [int(v) for v in bottom[0][2]],
    ]


def _page_bbox(pts: list, angle: int, width: int, height: int) -> tuple:
    rw, rh = (width, height) if angle == 0 else (height, width)
    cs = [unrotate_point(x, y, angle, rw, rh) for x, y in pts]
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_inter(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _bbox_area(a: tuple) -> float:
    return max(1.0, (a[2] - a[0]) * (a[3] - a[1]))


def _bbox_iou(a: tuple, b: tuple) -> float:
    i = _bbox_inter(a, b)
    return i / (_bbox_area(a) + _bbox_area(b) - i) if i else 0.0


def transfer_candidates(
    angle_boxes: list[dict], detections: list[dict], width: int, height: int
) -> tuple[list[dict], list[dict]]:
    """Cross-orientation footprints to re-read, as angle_boxes-shaped additions.

    A trigger is a large wide box in its own frame whose best read is below
    TRANSFER_CONFIDENT. For each trigger the OTHER orientations' clean
    split/merge footprints join the trigger's frame, and the trigger's own
    footprint joins the other frames (horizontal boxes as rotated bboxes,
    free polygons as reading-ordered quads). Returns (additions, provenance);
    each provenance record carries the target angle, page-frame bbox,
    transfer kind and source angle.
    """
    best_conf: dict[tuple, float] = {}
    for det in detections:
        xs = [p[0] for p in det["polygon"]]
        ys = [p[1] for p in det["polygon"]]
        key = (
            det["angle"],
            round(min(xs)),
            round(min(ys)),
            round(max(xs)),
            round(max(ys)),
        )
        best_conf[key] = max(best_conf.get(key, 0.0), det["confidence"])
    items: dict[int, list] = {}
    for angle_data in angle_boxes:
        angle = angle_data["angle"]
        rw, rh = (width, height) if angle == 0 else (height, width)
        entries = []
        for x0, x1, y0, y1 in angle_data["horizontal_list"]:
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            entries.append(
                (
                    "h",
                    _page_bbox(pts, angle, width, height),
                    [unrotate_point(x, y, angle, rw, rh) for x, y in pts],
                    x1 - x0,
                    y1 - y0,
                )
            )
        for poly in angle_data["free_list"]:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            entries.append(
                (
                    "f",
                    _page_bbox(poly, angle, width, height),
                    [unrotate_point(x, y, angle, rw, rh) for x, y in poly],
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                )
            )
        items[angle] = entries
    additions: dict[int, dict] = {
        a: {"angle": a, "horizontal_list": [], "free_list": []} for a in items
    }
    provenance: list[dict] = []

    def add(
        target_angle: int,
        kind: str,
        page_box: tuple,
        page_pts: list,
        why: str,
        source_angle: int,
    ) -> None:
        for _, q, _, _, _ in items[target_angle]:
            if _bbox_iou(page_box, q) > 0.8:
                return
        for record in provenance:
            if (
                record["angle"] == target_angle
                and _bbox_iou(page_box, record["page_bbox"]) > 0.8
            ):
                return
        if kind == "h":
            cs = [
                rotate_point(x, y, target_angle, width, height)
                for x, y in (
                    (page_box[0], page_box[1]),
                    (page_box[2], page_box[1]),
                    (page_box[2], page_box[3]),
                    (page_box[0], page_box[3]),
                )
            ]
            xs = [c[0] for c in cs]
            ys = [c[1] for c in cs]
            additions[target_angle]["horizontal_list"].append(
                [int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))]
            )
        else:
            additions[target_angle]["free_list"].append(
                reading_order(
                    [
                        rotate_point(x, y, target_angle, width, height)
                        for x, y in page_pts
                    ]
                )
            )
        provenance.append(
            {
                "angle": target_angle,
                "page_bbox": tuple(page_box),
                "kind": why,
                "source_angle": source_angle,
            }
        )

    for angle, entries in items.items():
        for kind, page_box, page_pts, w, h in entries:
            if w < TRANSFER_MIN_LONG or w < TRANSFER_MIN_ASPECT * h:
                continue
            key = (
                angle,
                round(page_box[0]),
                round(page_box[1]),
                round(page_box[2]),
                round(page_box[3]),
            )
            if best_conf.get(key, 0.0) >= TRANSFER_CONFIDENT:
                continue
            for other in items:
                if other != angle:
                    add(other, kind, page_box, page_pts, "footprint", angle)
            for other, other_entries in items.items():
                if other == angle:
                    continue
                for kind2, q, pts2, _, _ in other_entries:
                    overlap = _bbox_inter(page_box, q)
                    if not overlap:
                        continue
                    if (
                        overlap >= 0.9 * _bbox_area(q)
                        and 0.25 <= _bbox_area(q) / _bbox_area(page_box) <= 0.75
                    ):
                        add(angle, kind2, q, pts2, "split", other)
                    elif (
                        overlap >= 0.9 * _bbox_area(page_box)
                        and 0.4 <= _bbox_area(page_box) / _bbox_area(q) <= 0.9
                    ):
                        add(angle, kind2, q, pts2, "merge", other)
    filled = [
        additions[a]
        for a in additions
        if additions[a]["horizontal_list"] or additions[a]["free_list"]
    ]
    return filled, provenance


def merge_transferred(
    detections: list[dict], transferred: list[dict], provenance: list[dict]
) -> list[dict]:
    """Merge transfer-pass reads into the main detections with dedupe.

    Every transfer-pass read is flagged and given its provenance (matched by
    page-frame IoU). Reads below TRANSFER_FLOOR are dropped. Same-text
    overlapping pairs involving a transferred read keep the higher confidence,
    with a TRANSFER_TIE margin preferring the existing read.
    """

    def bbox(det: dict) -> tuple:
        xs = [p[0] for p in det["polygon"]]
        ys = [p[1] for p in det["polygon"]]
        return (min(xs), min(ys), max(xs), max(ys))

    kept = []
    for det in transferred:
        if det["confidence"] < TRANSFER_FLOOR:
            continue
        det["transferred"] = True
        rb = bbox(det)
        for record in provenance:
            if (
                record["angle"] == det["angle"]
                and _bbox_iou(rb, record["page_bbox"]) >= 0.5
            ):
                det["transfer_kind"] = record["kind"]
                det["transfer_source_angle"] = record["source_angle"]
                break
        kept.append(det)
    merged = detections + kept
    drop: set[int] = set()
    for i, a in enumerate(merged):
        for j in range(i + 1, len(merged)):
            b = merged[j]
            if not a.get("transferred") and not b.get("transferred"):
                continue
            text = a["text"].strip().upper()
            if not text or text != b["text"].strip().upper():
                continue
            ba, bb = bbox(a), bbox(b)
            ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2)
            cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            touching = (
                _bbox_iou(ba, bb) >= 0.2
                or (bb[0] <= ca[0] <= bb[2] and bb[1] <= ca[1] <= bb[3])
                or (ba[0] <= cb[0] <= ba[2] and ba[1] <= cb[1] <= ba[3])
            )
            if not touching:
                continue
            if abs(a["confidence"] - b["confidence"]) < TRANSFER_TIE and a.get(
                "transferred"
            ) != b.get("transferred"):
                drop.add(i if a.get("transferred") else j)
            else:
                drop.add(j if a["confidence"] >= b["confidence"] else i)
    return [d for k, d in enumerate(merged) if k not in drop]


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
    tile_size: int = 2560,
    fallback_vocab: list[str] | None = None,
    transfer: bool = True,
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
    applies. Applied by `mapsnap craft`, not here.

    tile_size splits images whose long side exceeds it into overlapping tiles that CRAFT
    detects at native resolution, avoiding EasyOCR's canvas_size (2560px) downscaling
    that shrinks small labels on oversized sheets (e.g. key maps). Only detection is
    tiled; recognition is unchanged. Set to 0 to disable (single-pass detection, the
    prior behavior). Default 2560 matches EasyOCR's canvas_size. Applied by
    `mapsnap craft`, not here.

    CRAFT detection is NOT run here: <stem>.boxes.json must already exist,
    written by `mapsnap craft` (see write_craft_boxes). Detection is the
    slowest stage and is shared by several consumers (`mapsnap adjacency`,
    the keymap passes), so it is its own pipeline step; a missing boxes file
    raises rather than silently re-running it under another command.

    fallback_vocab, if given, recognizes each box a second time with a broader vocabulary and
    keeps, per box, the higher-confidence read. When the fallback vocabulary wins with a
    different text (a street outside the page's own restricted neighborhood), the detection is
    marked ``fallback: True``. Used with a key-map-restricted vocab_strings so a page whose real
    streets fell outside its neighborhood still gets read, tagged for downstream to trust less.

    Each returned detection is a dict with:
      - polygon: list of 4 [x, y] corners in original image coordinates
      - text: recognized text string
      - confidence: float in [0, 1]
      - angle: rotation pass (0, 90, or 270) that produced this detection
      - long_side: length of the longer pair of polygon sides (pixels)
      - short_side: length of the shorter pair of polygon sides (pixels)
      - ignore: True if the text matches a NON_STREET_TEXT pattern (absent otherwise)
      - fallback: True if the read came from fallback_vocab, not vocab_strings (absent otherwise)
      - background: {color, hue, chroma} of the box's background when it is more saturated than
        the page's paper, i.e. the label is printed on a coloured building fill (absent when the
        label sits on paper, which is where street names belong)

    The written <stem>.streets.json also records the page's own ``paper`` colour in the same
    {color, hue, chroma} form, which is the reference the ``background`` property is relative to.
    """
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    cached = Path(boxes_path(image_path))
    if not cached.exists():
        raise FileNotFoundError(
            f"No CRAFT boxes for {image_path}. Run: {craft_hint([image_path])}"
        )
    angle_boxes: list[dict] = json.loads(cached.read_text())["boxes"]

    def recognize(vocab: list[str]) -> list[dict]:
        return _recognize_pass(
            reader,
            img,
            angle_boxes,
            vocab=vocab,
            beam_width=beam_width,
            allowlist=allowlist,
            min_long_side=min_long_side,
            orig_width=orig_width,
            orig_height=orig_height,
        )

    all_detections = recognize(vocab_strings)
    if fallback_vocab is not None:
        all_detections = _merge_vocab_passes(all_detections, recognize(fallback_vocab))
    if transfer:
        # Cross-orientation transfer (#144): re-read failed large boxes with the
        # other orientations' segmentation, through the same vocab passes.
        additions, provenance = transfer_candidates(
            angle_boxes, all_detections, orig_width, orig_height
        )
        if additions:

            def recognize_added(vocab: list[str]) -> list[dict]:
                return _recognize_pass(
                    reader,
                    img,
                    additions,
                    vocab=vocab,
                    beam_width=beam_width,
                    allowlist=allowlist,
                    min_long_side=min_long_side,
                    orig_width=orig_width,
                    orig_height=orig_height,
                )

            extra = recognize_added(vocab_strings)
            if fallback_vocab is not None:
                extra = _merge_vocab_passes(extra, recognize_added(fallback_vocab))
            all_detections = merge_transferred(all_detections, extra, provenance)

    for det in all_detections:
        pts = np.array(det["polygon"], dtype=float)
        sides = polygon_side_lengths(det["polygon"])
        det["long_side"] = round(max(sides), 1)
        det["short_side"] = round(min(sides), 1)
        edge_vecs = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
        long_vec = max(edge_vecs, key=np.linalg.norm)
        det["dir_pix"] = round(float(np.arctan2(long_vec[1], long_vec[0])) % np.pi, 4)
        if det["text"].upper() in (NON_STREET_TEXT | SCALE_NOTE_TEXT):
            det["ignore"] = True
        elif det["text"].upper() in HINT_STRINGS:
            det["hint"] = True

    paper = annotate_backgrounds(all_detections, np.array(img))

    streets_doc = {
        "width": orig_width,
        "height": orig_height,
        "timestamp": datetime.now(UTC).isoformat(),
        "command": filter_args(sys.argv[:], image_path),
        "paper": paper,
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
    tile_size: int,
    gpu: bool,
    recognizer_weights: str | None,
    transfer: bool = True,
) -> None:
    """Initialize per-worker state once per process: create the EasyOCR reader."""
    _worker_state["reader"] = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    if recognizer_weights:
        load_recognizer_weights(_worker_state["reader"], recognizer_weights)
    _worker_state["vocab_strings"] = vocab_strings
    _worker_state["min_size"] = min_size
    _worker_state["min_long_side"] = min_long_side
    _worker_state["allowlist"] = allowlist
    _worker_state["link_threshold"] = link_threshold
    _worker_state["beam_width"] = beam_width
    _worker_state["craft_scale"] = craft_scale
    _worker_state["tile_size"] = tile_size
    _worker_state["transfer"] = transfer


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
        tile_size=_worker_state["tile_size"],
        transfer=_worker_state["transfer"],
    )
    return image_path


def page_vocabs(
    image_path: str,
    locator: KeymapLocator | None,
    geojson_features: list[dict],
    vocab_strings: list[str],
    rectangle_vocab: list[str],
) -> tuple[list[str], list[str] | None]:
    """(primary, fallback) vocab for one page: neighborhood + rectangle if placed, else rectangle.

    With no key map (``locator is None``) returns the full ``vocab_strings`` and no fallback.
    For a page the key map places, the primary vocab is its key-map neighborhood — falling back
    to the whole key-map ``rectangle_vocab`` when the neighborhood holds no street names — and the
    fallback pass uses the rectangle vocab. An unplaced or empty-neighborhood page uses the
    rectangle vocab alone (no second pass).
    """
    if locator is None:
        return vocab_strings, None
    restricted = locator.restricted_features(
        page_key(image_stem(image_path)), geojson_features
    )
    if not restricted:
        # Unplaced (None) or placed with no nearby features ([]): rectangle is the tightest.
        return rectangle_vocab, None
    near = build_block_index({"type": "FeatureCollection", "features": restricted})
    primary = generate_vocab_strings(set(near.keys())) or rectangle_vocab
    return primary, rectangle_vocab


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
        help="Minimum short side passed to the EasyOCR detector (default: %(default)s)",
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
        help=(
            "Centerlines GeoJSON file. Used to build a vocabulary of known street-name "
            "forms for prefix-constrained CTC decoding, which substantially improves "
            "recall on abbreviated and direction-prefixed labels. Defaults to a "
            "centerlines.geojson next to the input images (or their parent directory)."
        ),
    )
    parser.add_argument(
        "--keymap",
        nargs="+",
        metavar="JSON",
        help=(
            "One or more georeferenced key-map detections files (e.g. raw/p0.keymap.json, each "
            "with a sibling <stem>.georef.json); pass several for a volume with multiple key "
            "maps. Each page a key map places is OCR'd with only the streets within "
            "--keymap-radius of that page's key-map location — a much smaller vocabulary that "
            "raises recognizer confidence and drops far-away same-name streets. Pages no key "
            "map places fall back to the full vocabulary."
        ),
    )
    parser.add_argument(
        "--keymap-radius",
        type=float,
        default=None,
        metavar="M",
        help=(
            "Radius in metres around a page's key-map location for the restricted vocabulary "
            "(default: auto, ~2x the key map's page-to-page spacing)."
        ),
    )
    parser.add_argument(
        "--ignore-keymap",
        action="store_true",
        help=(
            "Do not use key maps. By default, when --keymap is not given, key-map detections "
            "files (<stem>.keymap.json with a sibling .georef.json) next to the images or under "
            "raw/ are discovered and used automatically; this flag turns that off."
        ),
    )
    parser.add_argument(
        "--no-transfer",
        action="store_true",
        help=(
            "Disable the cross-orientation box transfer (#144): failed large boxes "
            "are normally re-read using the other CRAFT orientations' segmentation. "
            "For A/B controls and debugging."
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
        "--backfill-background",
        action="store_true",
        help=(
            "Do not run OCR. Instead, add the 'background' colour property to the detections in "
            "each image's existing <stem>.streets.json, which is all that pages OCR'd before that "
            "property existed need. Images with no .streets.json are skipped."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes (default: %(default)s). Each worker loads its "
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
        default=20,
        metavar="PX",
        help=(
            "Skip recognition for CRAFT detections whose long side is below this "
            "threshold (default: %(default)s). Set to match the --min-long-side "
            "used by georef_from_labels.py to avoid recognizing boxes that will be "
            "filtered downstream."
        ),
    )
    parser.add_argument(
        "--recognizer-weights",
        default=str(DEFAULT_RECOGNIZER_WEIGHTS),
        metavar="PT",
        help=(
            "Fine-tuned recognizer weights (a state_dict from "
            "mapsnap.train_street_recognizer) to load in place of EasyOCR's "
            "stock model (default: %(default)s). Detection, vocabulary, and "
            "decoding are unchanged. Pass --stock-recognizer for EasyOCR's."
        ),
    )
    parser.add_argument(
        "--stock-recognizer",
        action="store_true",
        help="Use EasyOCR's own recognizer instead of the fine-tuned weights.",
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
        "--tile-size",
        type=int,
        default=2560,
        metavar="PX",
        help=(
            "Detect images larger than this in overlapping native-resolution tiles "
            "(default: 2560, matching EasyOCR's canvas_size). This avoids downscaling "
            "small labels on oversized sheets such as key maps. Only CRAFT detection is "
            "tiled; recognition is unchanged. Set to 0 to disable (single-pass detection)."
        ),
    )
    args = parser.parse_args()

    if args.backfill_background:
        pending = [p for p in args.images if Path(_streets_path(p)).exists()]
        print(
            f"Backfilling background colour for {len(pending)}/{len(args.images)} image(s) with "
            "existing detections.",
            file=sys.stderr,
        )
        on_fill = 0
        for image_path in tqdm(pending, smoothing=0):
            on_fill += backfill_backgrounds(image_path)
        print(
            f"Marked {on_fill} detection(s) as sitting on a coloured fill.",
            file=sys.stderr,
        )
        return

    if args.centerlines is None:
        centerlines = default_centerlines(Path(args.images[0]).parent)
        if centerlines is None:
            sys.exit(
                "No --centerlines given and no centerlines.geojson found next to the "
                "input images."
            )
        args.centerlines = str(centerlines)
        print(f"Using centerlines: {args.centerlines}", file=sys.stderr)

    geojson = json.loads(Path(args.centerlines).read_text())
    block_index = build_block_index(geojson)
    vocab_strings = generate_vocab_strings(set(block_index.keys()))
    print(
        f"Constrained vocab: {len(vocab_strings)} forms from {len(block_index)} streets",
        file=sys.stderr,
    )

    # With a georeferenced key map, restrict each placed page's vocabulary to nearby streets,
    # falling back to the streets within the whole key map's rectangle (a volume-wide box every
    # page sits inside) for pages the neighborhood misses or the key map does not place.
    keymap_files = resolve_keymaps(args.keymap, args.ignore_keymap, args.images)
    locator = None
    rectangle_vocab = vocab_strings
    if keymap_files:
        print(
            "Using key map(s): " + ", ".join(str(path) for path in keymap_files),
            file=sys.stderr,
        )
        locator = KeymapLocator.from_keymaps(keymap_files, args.keymap_radius)
        rectangle = locator.rectangle_features(geojson["features"])
        if rectangle:
            rectangle_index = build_block_index(
                {"type": "FeatureCollection", "features": rectangle}
            )
            rectangle_vocab = (
                generate_vocab_strings(set(rectangle_index.keys())) or vocab_strings
            )
        print(
            f"Key map places {len(locator.located_keys())} page numbers; restricting vocab "
            f"to streets within {locator.radius_m:.0f} m of each, with a "
            f"{len(rectangle_vocab)}-form key-map-rectangle fallback (vs {len(vocab_strings)} "
            "full).",
            file=sys.stderr,
        )

    # Never OCR a page that has been split into panels; OCR its panels instead. This
    # mirrors mapsnap.utils.list_pages so the rule holds however ocr is invoked (pipeline
    # or a raw shell glob that happens to include the parent).
    images = args.images
    superseded = [p for p in images if has_split_panels(p)]
    if superseded:
        images = [p for p in images if p not in superseded]
        print(
            f"Skipping {len(superseded)} split parent page(s): "
            + ", ".join(Path(p).name for p in superseded),
            file=sys.stderr,
        )

    if args.resume:
        images = [
            p
            for p in images
            if not reads_are_current(
                Path(p).parent / (image_stem(p) + ".streets.json"),
                args.recognizer_weights,
            )
        ]
        print(
            f"Resuming: {len(images)}/{len(args.images)} remaining images to process.",
            file=sys.stderr,
        )

    require_boxes(images)

    gpu = not args.no_gpu

    if locator is not None and args.num_workers > 1:
        print(
            "--keymap restricts the vocabulary per page; running with a single worker.",
            file=sys.stderr,
        )
        args.num_workers = 1

    if args.stock_recognizer:
        args.recognizer_weights = None
    elif args.recognizer_weights and not Path(args.recognizer_weights).exists():
        sys.exit(
            f"Recognizer weights not found: {args.recognizer_weights}\n"
            "Pass --stock-recognizer to use EasyOCR's model instead."
        )

    if args.num_workers > 1:
        initargs = (
            vocab_strings,
            args.min_short_side,
            args.min_long_side,
            args.allowlist,
            args.link_threshold,
            args.beam_width,
            args.craft_scale,
            args.tile_size,
            gpu,
            args.recognizer_weights,
            not args.no_transfer,
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
        if args.recognizer_weights:
            load_recognizer_weights(reader, args.recognizer_weights)
        for image_path in tqdm(images, smoothing=0):
            primary_vocab, fallback_vocab = page_vocabs(
                image_path, locator, geojson["features"], vocab_strings, rectangle_vocab
            )
            detect_text(
                image_path,
                vocab_strings=primary_vocab,
                min_size=args.min_short_side,
                min_long_side=args.min_long_side,
                allowlist=args.allowlist,
                link_threshold=args.link_threshold,
                reader=reader,
                beam_width=args.beam_width,
                craft_scale=args.craft_scale,
                tile_size=args.tile_size,
                fallback_vocab=fallback_vocab,
                transfer=not args.no_transfer,
            )


if __name__ == "__main__":
    main()
