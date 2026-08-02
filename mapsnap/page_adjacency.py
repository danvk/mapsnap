"""Detect printed adjacent-sheet numbers on page margins and build a volume adjacency graph.

Sanborn sheets print the neighboring sheet's number on the margin (the key legend calls it a
"reference to adjoining sheet"): a big numeral across the shared boundary road on the side that
sheet continues. Reading these gives page adjacency that is independent of the key map — and a
detected neighbor's edge (top/left/bottom/right of the image) pins the page's orientation.

Raw digit reads are noisy (house/block/dimension numbers collide with valid page numbers), so a
detection only becomes a *claim* when it resolves to a valid page key, near the page edge,
printed large, and axis-aligned. Resolution is letter-aware (a "1499G" read claims p1499g, not
p1499) and repairs the two systematic read failures of long page numbers — leading digits lost
against the sheet trim ("409" -> 1409) and neighboring ink absorbed into the box ("14027" ->
1402) — but only on a unique match against keys of >= MIN_REPAIR_DIGITS digits, so volumes
with 1- and 2-digit page numbers resolve exact reads only. A claim only becomes an adjacency
*edge* when it is reciprocated —
page A claims B AND page B claims A. Reciprocity alone is not proof: junk claims of small
numbers are common enough to reciprocate by chance (and page numbering correlates with
adjacency, so such accidents are often "correct"), so single-digit claims — where any tall
narrow ink can read as a "1" — additionally face a high confidence floor and a height band
calibrated from the volume's own confirmed multi-digit references (their printed size is very
consistent: Hudson County median 46 px with p10-p90 of 42-48). Measured on Hudson County the
result is 167 mutual edges with no known-false edge, reaching 30/33 pages that street matching
failed to georeference.

This is a whole-volume step: it scans every non-split page image (skipping key-map sheets,
whose faces are covered in page numbers) and writes a single ``<volume>/adjacency.json`` with
each page's number detections and the mutual-edge adjacency graph.

    uv run mapsnap adjacency data/hudson_co_nj_1950_vol_9
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.keymap.locate import page_key, page_number
from mapsnap.utils import image_stem

# A detection only counts as an adjacency claim when it sits in the outer EDGE_BAND of the
# page, is at least MIN_HEIGHT px tall (the printed sheet references are large numerals; this
# excludes house/dimension numbers), and clears a confidence floor. Measured on Hudson and
# Brooklyn: 0.3 loses no true edge on either volume while killing junk a permissive floor
# admits (a forced-transcription street name reads at ~0.1; genuine degraded references read
# above 0.3); 0.5 would kill a further junk edge but costs Hudson four real ones.
EDGE_BAND = 0.25
MIN_HEIGHT = 28.0
MIN_CONF = 0.3

# A rotated detection quad is never a sheet reference: every rotated candidate inspected on
# Hudson County (52 of them) was a misread street name ("WESTSIDE", "Mc ADOO") or pipe
# annotation ('10" W Pipe'). References are printed upright, so reject quads off-axis by more
# than this tolerance.
ROTATION_TOLERANCE_DEG = 5.0

# Single-digit reads are held to a stricter standard than reciprocity alone can provide:
# a hard confidence floor, a height band of these fractions around the median height of the
# volume's multi-digit mutual claims (see single_digit_height_band), and an isolation floor —
# a genuine reference is printed in open whitespace, while a junk single digit is typically a
# fragment of a block/house number whose sibling box it touches (gap ~0), so require clear
# space of at least this fraction of the same median height around it.
SINGLE_DIGIT_MIN_CONF = 0.9
SIZE_BAND_FRACTION = (0.65, 1.4)
SINGLE_DIGIT_MIN_GAP_FRACTION = 0.3


def volume_page_images(volume: Path) -> list[Path]:
    """The whole-page images to scan: top-level ``p*.jpg``, skipping split panels and key maps.

    Split panels (``__`` stems) are cuts of a parent sheet, so the printed margin references
    live on the parent; the parent is scanned even when panels supersede it for OCR. Key-map
    sheets are excluded: their faces are covered in page numbers, which would all read as
    spurious claims.
    """
    images = []
    for image in sorted(volume.glob("p*.jpg")):
        stem = image_stem(str(image))
        if "__" in stem:
            continue
        if (volume / f"{stem}.keymap.json").exists() or (
            volume / "raw" / f"{stem}.keymap.json"
        ).exists():
            continue
        images.append(image)
    return images


def classify_edge(x_frac: float, y_frac: float, band: float = EDGE_BAND) -> str:
    """Which page edge(s) a point (as fractions of width/height) is near: "T", "BL", "center"...

    Sides are image-relative — "T"op, "B"ottom, "L"eft, "R"ight, with corners "TL", "TR",
    "BL", "BR" — not compass directions: the page's orientation in the world is unknown at
    this stage (finding it is what adjacency is *for*).
    """
    edges = ""
    if y_frac < band:
        edges += "T"
    if y_frac > 1 - band:
        edges += "B"
    if x_frac < band:
        edges += "L"
    if x_frac > 1 - band:
        edges += "R"
    return edges or "center"


def bbox_gap(a: list[list[float]], b: list[list[float]]) -> float:
    """Gap in pixels between two boxes' axis-aligned bounds (0 when they touch/overlap)."""
    ax0, ax1 = min(p[0] for p in a), max(p[0] for p in a)
    ay0, ay1 = min(p[1] for p in a), max(p[1] for p in a)
    bx0, bx1 = min(p[0] for p in b), max(p[0] for p in b)
    by0, by1 = min(p[1] for p in b), max(p[1] for p in b)
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return math.hypot(dx, dy)


MIN_REPAIR_DIGITS = 3
"""Fewest digits a page key must have before read-repair may target it.

Repair (suffix completion, embedded extraction) exists for volumes like LA 1949
whose 4-digit '1'-leading references truncate against the sheet trim ("1409"
reads "409" at conf 1.0) or absorb neighboring ink ("14027" for 1402) — the
LA truth labels put 49% of all printed references in this boxed-but-unread
class. For volumes with 1- and 2-digit page numbers, short reads are
substrings of everything, so repair against short keys would be guesswork:
below this floor only exact and lettered matches fire, leaving those volumes'
behavior untouched (Hudson reads 96% of its references fine without repair).
"""

LETTERED_KEY_PATTERN = re.compile(r"([0-9IO]+)\s*([A-Z]{1,2})$")
"""A letters-pass read that ends in a short letter suffix ("1499G", "I33S")."""


def resolve_page_key(
    digit_text: str,
    letter_text: str,
    valid_keys: set[str],
    min_repair_digits: int = MIN_REPAIR_DIGITS,
) -> tuple[str, str] | None:
    """Resolve raw OCR reads of a margin reference to a valid page key.

    ``digit_text`` is the digits-allowlist read, ``letter_text`` the
    letters-allowed read of the same box ("" when none). Returns
    ``(key, method)`` or None, trying in order:

      lettered   the letter read shows a digit run plus a letter suffix that
                 forms a valid key ("1499G" -> 1499G); checked first because
                 the digits pass renders the same ink as a bare "1499" -- a
                 different, wrong page.
      exact      the digit read is a valid key as-is (the only method that
                 applies to keys shorter than ``min_repair_digits``).
      suffix     the read lost its leading digit against the sheet trim:
                 exactly one valid key ends with it ("409" -> 1409). Requires
                 a read of >= 3 digits: 2-digit remnants are house-number
                 fragments that happen to end a page number, measured 8%
                 precise on the LA truth labels (13/164) — including a
                 symmetric pair ("85" on p1475, "75" on p1485) confident
                 enough to reciprocate into a false edge. 3-digit remnants
                 measure 71% precise and reciprocity removes the rest.
      embedded   the read absorbed extra digits from neighboring ink: exactly
                 one valid key appears inside it ("14027" -> 1402).

    Repairs require a UNIQUE match among keys of >= ``min_repair_digits``
    digits; any ambiguity resolves to None, never a guess.
    """
    digits = "".join(c for c in digit_text if c.isdigit())
    match = LETTERED_KEY_PATTERN.search(letter_text.replace(" ", "").upper())
    if match:
        base = match.group(1).replace("I", "1").replace("O", "0")
        lettered = base + match.group(2)
        if lettered in valid_keys:
            return lettered, "lettered"
    if not digits:
        return None
    if digits in valid_keys:
        return digits, "exact"
    repairable = [
        key for key in valid_keys if key.isdigit() and len(key) >= min_repair_digits
    ]
    if len(digits) >= 3:
        suffixes = [
            key for key in repairable if len(key) > len(digits) and key.endswith(digits)
        ]
        if len(suffixes) == 1:
            return suffixes[0], "suffix"
    embedded = [key for key in repairable if len(key) < len(digits) and key in digits]
    if len(embedded) == 1:
        return embedded[0], "embedded"
    return None


def is_text_veto(
    letter_text: str, letter_confidence: float, digit_confidence: float
) -> bool:
    """Whether a letters-allowlist read reveals the box to be text rather than a number.

    Under the digits-only allowlist the recognizer is *forced* to transcribe street names as
    numbers ("SMITH" came out as a confident "55" on Brooklyn p56); re-reading the same box
    with letters available lets such text identify itself. A veto needs both a clear textual
    reveal — three or more letters, or a letter-dominated read — and the recognizer to
    *prefer* that reading over the digit one: a genuine "41" also comes back letter-dominated
    ("AL", look-alike glyphs) but at lower confidence than its digit reading, while SMITH
    reads as text far more confidently than as "55".
    """
    compact = letter_text.replace(" ", "")
    if not compact or letter_confidence <= digit_confidence:
        return False
    letters = sum(1 for c in compact if c.isalpha())
    return letters >= 3 or letters / len(compact) >= 0.7


def box_center(bbox: list[list[float]]) -> tuple[float, float]:
    """Center of an OCR quad."""
    return (sum(p[0] for p in bbox) / 4, sum(p[1] for p in bbox) / 4)


def cached_craft_boxes(image_path: Path) -> tuple[list, list] | None:
    """The angle-0 CRAFT boxes from ``<stem>.boxes.json`` (written by ``mapsnap ocr``), or None.

    Returns (horizontal_list, free_list) in the same shape ``reader.detect`` yields, so
    recognition can skip the CRAFT pass entirely. Returns None when the file is absent, records
    different image dimensions (boxes from another resolution would misplace every read), or has
    no angle-0 entry.
    """
    boxes_path = image_path.parent / f"{image_stem(str(image_path))}.boxes.json"
    if not boxes_path.exists():
        return None
    doc = json.loads(boxes_path.read_text())
    with Image.open(image_path) as image:
        if (doc.get("width"), doc.get("height")) != image.size:
            return None
    for entry in doc.get("boxes", []):
        if entry.get("angle") == 0:
            return entry.get("horizontal_list", []), entry.get("free_list", [])
    return None


def digit_detections(
    image_path: Path,
    reader,
    valid_keys: set[str],
    craft_boxes: tuple[list, list] | None = None,
) -> list[dict]:
    """All digit reads on one page that resolve to a valid volume page key.

    Two recognition passes over the same image: the digits-only pass supplies the
    transcription (immune to look-alike substitutions like 0 -> O), and a letters-allowed
    pass over the same boxes vetoes the ones that are actually street names or other text
    (see is_text_veto) and supplies letter suffixes for lettered keys ("1499G").
    Reads that are not a valid key verbatim go through ``resolve_page_key`` repair.
    ``craft_boxes`` (from cached_craft_boxes) skips the CRAFT detection pass — by far
    the most expensive step — reusing the boxes ``mapsnap ocr`` already computed for
    the same image.

    Each detection records the resolved ``key`` and how it resolved (``via``), the
    digit part as ``number`` (for size rules and older readers), the raw OCR ``text``,
    EasyOCR's polygon and confidence, the glyph ``height`` in pixels, position as
    width/height fractions, the ``edge`` classification, and ``nearest_box`` — the gap
    to the nearest other OCR box, since a genuine sheet reference is printed in open
    whitespace while a junk single digit is usually a fragment of a larger number with
    sibling text right beside it. Filtering into claims is left to the caller so the
    JSON keeps the full picture.
    """
    from easyocr.utils import reformat_input

    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    # Detect once, recognize twice: CRAFT detection dominates the cost and is
    # allowlist-independent, so the two passes share its boxes (readtext = detect +
    # recognize with these same defaults).
    img, img_grey = reformat_input(image)
    if craft_boxes is not None:
        horizontal_lists, free_lists = [craft_boxes[0]], [craft_boxes[1]]
    else:
        horizontal_lists, free_lists = reader.detect(img)
    results = reader.recognize(
        img_grey, horizontal_lists[0], free_lists[0], allowlist="0123456789"
    )
    letter_results = reader.recognize(
        img_grey,
        horizontal_lists[0],
        free_lists[0],
        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    letter_centers = [box_center(bbox) for bbox, _, _ in letter_results]
    detections = []
    for index, (bbox, text, confidence) in enumerate(results):
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            continue
        # Veto boxes whose letters-pass read reveals text. Pair by box center; the two
        # passes share the same detector so boxes normally coincide.
        cx, cy = box_center(bbox)
        box_height = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        nearest_letter = min(
            (
                (math.hypot(lx - cx, ly - cy), i)
                for i, (lx, ly) in enumerate(letter_centers)
            ),
            default=(math.inf, -1),
        )
        letter_text = ""
        if nearest_letter[0] < box_height:
            letter_read = letter_results[nearest_letter[1]]
            if is_text_veto(letter_read[1], float(letter_read[2]), float(confidence)):
                continue
            letter_text = str(letter_read[1])
        resolved = resolve_page_key(text, letter_text, valid_keys)
        if resolved is None:
            continue
        key, via = resolved
        x_frac = sum(p[0] for p in bbox) / 4 / width
        y_frac = sum(p[1] for p in bbox) / 4 / height
        glyph_height = float(max(p[1] for p in bbox) - min(p[1] for p in bbox))
        nearest_box = min(
            (
                bbox_gap(bbox, other_bbox)
                for other_index, (other_bbox, _, _) in enumerate(results)
                if other_index != index
            ),
            default=math.inf,
        )
        detections.append(
            {
                "key": key,
                "via": via,
                "number": int("".join(c for c in key if c.isdigit())),
                "text": text,
                "confidence": round(float(confidence), 4),
                "polygon": [[int(p[0]), int(p[1])] for p in bbox],
                "height": round(glyph_height, 1),
                "x_frac": round(x_frac, 4),
                "y_frac": round(y_frac, 4),
                "edge": classify_edge(x_frac, y_frac),
                "nearest_box": round(nearest_box, 1)
                if nearest_box != math.inf
                else None,
            }
        )
    return detections


def polygon_rotation_deg(polygon: list[list[float]]) -> float:
    """Rotation of a detection quad off the image axes, in degrees folded to [0, 45].

    EasyOCR returns an axis-aligned box for upright text and a rotated quad otherwise; this is
    the angle of the quad's top edge, folded so 0 means axis-aligned in either orientation.
    """
    (x0, y0), (x1, y1) = polygon[0], polygon[1]
    angle = abs(math.degrees(math.atan2(y1 - y0, x1 - x0))) % 90.0
    return min(angle, 90.0 - angle)


def single_digit_height_band(
    confirmed_heights: list[float],
) -> tuple[float, float] | None:
    """Height band for single-digit claims, calibrated from confirmed reference heights.

    ``confirmed_heights`` are the heights of multi-digit claims that ended up in mutual edges —
    numerals long enough that a junk read is unlikely. Their printed size is very consistent
    (Hudson County: median 46 px, p10-p90 of 42-48), so a single-digit read far from the median
    is junk: frame rules and stray strokes read as tall "1"s, house numbers as short ones.
    Returns None when there are no confirmed heights (e.g. a volume with only one-digit pages).
    """
    if not confirmed_heights:
        return None
    ordered = sorted(confirmed_heights)
    median = ordered[len(ordered) // 2]
    return (SIZE_BAND_FRACTION[0] * median, SIZE_BAND_FRACTION[1] * median)


def is_claim(
    detection: dict,
    own_key: str | None,
    *,
    min_height: float = MIN_HEIGHT,
    min_confidence: float = MIN_CONF,
    single_digit_min_confidence: float = SINGLE_DIGIT_MIN_CONF,
    height_band: tuple[float, float] | None = None,
    min_gap: float | None = None,
) -> bool:
    """Whether a detection qualifies as an adjacency claim.

    Every claim must be a valid page key other than the page's own, near a page edge, tall
    enough, confident enough, and axis-aligned (rotated quads are always misread street names
    or pipe annotations). Single-digit numbers face three stricter tests — the high
    ``single_digit_min_confidence`` floor, the ``height_band`` around the volume's
    printed-reference size, and the ``min_gap`` isolation floor (a digit torn off a larger
    number touches its sibling box, so it can be a perfect glyph at the right size and still
    be junk) — because junk single-digit claims are common enough to reciprocate by
    coincidence, so reciprocity alone cannot vouch for them.
    """
    if (
        detection["key"] == own_key
        or detection["edge"] == "center"
        or detection["height"] < min_height
        or detection["confidence"] < min_confidence
        or polygon_rotation_deg(detection["polygon"]) > ROTATION_TOLERANCE_DEG
    ):
        return False
    if detection["number"] < 10:
        if detection["confidence"] < single_digit_min_confidence:
            return False
        if height_band is not None and not (
            height_band[0] <= detection["height"] <= height_band[1]
        ):
            return False
        nearest_box = detection.get("nearest_box")
        if min_gap is not None and nearest_box is not None and nearest_box < min_gap:
            return False
    return True


def claim_resolver(claims_by_page: dict[str, set[str]]):
    """resolve(claim key) -> stems carrying it, shared by both edge builders.

    A claimed key resolves to the stem(s) carrying exactly that key; a
    bare-number claim in a volume whose sheet has a lettered stem (Chicago
    prints "60" for p60w) falls back to the number, but only when it names a
    single stem — a number shared by many lettered sheets (LA's p1499a-r) is
    ambiguous and resolves to nothing.
    """
    stems_by_key: dict[str, list[str]] = defaultdict(list)
    stems_by_number: dict[int, list[str]] = defaultdict(list)
    for stem in claims_by_page:
        key = page_key(stem)
        if key is not None:
            stems_by_key[key].append(stem)
        number = page_number(stem)
        if number is not None:
            stems_by_number[number].append(stem)

    def resolve(claim: str) -> list[str]:
        stems = stems_by_key.get(claim.upper(), [])
        if stems:
            return stems
        if claim.isdigit():
            fallback = stems_by_number.get(int(claim), [])
            if len(fallback) == 1:
                return fallback
        return []

    return resolve


def mutual_edges(claims_by_page: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Reciprocated adjacency edges: A claims B's key and B claims A's key.

    ``claims_by_page`` maps page stems to claimed page keys (see claim_resolver
    for how claims resolve to stems). The edge exists only if each side's
    claims resolve to the other. Returns sorted (stem, stem) pairs, each once.
    """
    resolve = claim_resolver(claims_by_page)
    edges: set[tuple[str, str]] = set()
    for stem, claimed in claims_by_page.items():
        for claim in claimed:
            for other in resolve(claim):
                if other == stem:
                    continue
                if any(
                    stem in resolve(back) for back in claims_by_page.get(other, set())
                ):
                    first, second = sorted((stem, other))
                    edges.add((first, second))
    return sorted(edges)


def one_sided_edges(claims_by_page: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Directed claims that resolve to a real page but are not reciprocated.

    The lower-trust adjacency tier: measured against the hand-labelled truth,
    88% of Hudson's and 89% of LA's unreciprocated-but-resolvable claims are
    genuine printed references whose reciprocal side failed to read (truth
    itself reciprocates 91-93%). Consumers must treat these as candidates to
    verify, not facts — a junk read that resolves to a real page lands here.
    Returns sorted (claimer, target) pairs, excluding mutual edges.
    """
    resolve = claim_resolver(claims_by_page)
    mutual = {frozenset(edge) for edge in mutual_edges(claims_by_page)}
    edges: set[tuple[str, str]] = set()
    for stem, claimed in claims_by_page.items():
        for claim in claimed:
            for other in resolve(claim):
                if other != stem and frozenset((stem, other)) not in mutual:
                    edges.add((stem, other))
    return sorted(edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect printed adjacent-sheet numbers and build a volume adjacency graph."
    )
    parser.add_argument(
        "volume", type=Path, help="Volume directory holding p*.jpg pages."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <volume>/adjacency.json).",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=MIN_HEIGHT,
        metavar="PX",
        help="Minimum glyph height for a claim (default: %(default)s, tuned for 25%%-scale pages).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONF,
        metavar="C",
        help="Minimum OCR confidence for a claim (default: %(default)s).",
    )
    parser.add_argument(
        "--single-digit-min-confidence",
        type=float,
        default=SINGLE_DIGIT_MIN_CONF,
        metavar="C",
        help=(
            "Stricter confidence floor for single-digit claims (default: %(default)s); "
            "junk single-digit reads are common enough to reciprocate by coincidence."
        ),
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration for EasyOCR."
    )
    parser.add_argument(
        "--reuse-boxes",
        action="store_true",
        help=(
            "Reuse CRAFT boxes from each page's <stem>.boxes.json (written by mapsnap ocr) "
            "instead of re-running detection; pages without one (or whose boxes were computed "
            "at different image dimensions) still run CRAFT."
        ),
    )
    args = parser.parse_args()

    import easyocr

    images = volume_page_images(args.volume)
    if not images:
        sys.exit(f"No page images found in {args.volume}.")
    valid_keys = {
        key for image in images if (key := page_key(image_stem(str(image)))) is not None
    }
    print(
        f"Scanning {len(images)} pages ({len(valid_keys)} page keys)...",
        file=sys.stderr,
    )
    reader = easyocr.Reader(["en"], gpu=not args.no_gpu, verbose=False)

    pages: dict[str, dict] = {}
    reused = 0
    for image in tqdm(images, smoothing=0):
        stem = image_stem(str(image))
        # Record the scanned image's dimensions so a viewer can rescale detection
        # polygons when it loads the page at a different resolution.
        width, height = Image.open(image).size
        craft_boxes = cached_craft_boxes(image) if args.reuse_boxes else None
        reused += craft_boxes is not None
        pages[stem] = {
            "key": page_key(stem),
            "number": page_number(stem),
            "width": width,
            "height": height,
            "detections": digit_detections(image, reader, valid_keys, craft_boxes),
        }
    if args.reuse_boxes:
        print(f"Reused CRAFT boxes for {reused}/{len(images)} pages.", file=sys.stderr)

    def compute_claims(
        height_band: tuple[float, float] | None,
        min_gap: float | None,
    ) -> dict[str, set[str]]:
        return {
            stem: {
                d["key"]
                for d in page["detections"]
                if is_claim(
                    d,
                    page["key"],
                    min_height=args.min_height,
                    min_confidence=args.min_confidence,
                    single_digit_min_confidence=args.single_digit_min_confidence,
                    height_band=height_band,
                    min_gap=min_gap,
                )
            }
            for stem, page in pages.items()
        }

    # Two passes: provisional mutual edges (no height band or isolation floor) calibrate the
    # printed-reference height from their multi-digit claims; single-digit claims are then
    # re-filtered against the derived band and isolation floor and the graph rebuilt.
    provisional_edges = mutual_edges(compute_claims(None, None))
    mutual_keys: dict[str, set[str]] = defaultdict(set)
    for first, second in provisional_edges:
        mutual_keys[first].add(pages[second]["key"])
        mutual_keys[second].add(pages[first]["key"])
    confirmed_heights = [
        d["height"]
        for stem, page in pages.items()
        for d in page["detections"]
        if d["number"] >= 10
        and d["key"] in mutual_keys[stem]
        and is_claim(
            d,
            page["key"],
            min_height=args.min_height,
            min_confidence=args.min_confidence,
            single_digit_min_confidence=args.single_digit_min_confidence,
        )
    ]
    height_band = single_digit_height_band(confirmed_heights)
    min_gap = None
    if height_band and confirmed_heights:
        ordered = sorted(confirmed_heights)
        median_height = ordered[len(ordered) // 2]
        min_gap = SINGLE_DIGIT_MIN_GAP_FRACTION * median_height
        print(
            f"Single-digit height band: [{height_band[0]:.0f}, {height_band[1]:.0f}]px, "
            f"isolation floor: {min_gap:.0f}px "
            f"(from {len(confirmed_heights)} confirmed multi-digit references).",
            file=sys.stderr,
        )
    else:
        print(
            "No confirmed multi-digit references; single-digit height band and "
            "isolation floor disabled.",
            file=sys.stderr,
        )

    claims_by_page = compute_claims(height_band, min_gap)
    for stem, page in pages.items():
        for detection in page["detections"]:
            detection["claim"] = is_claim(
                detection,
                page["key"],
                min_height=args.min_height,
                min_confidence=args.min_confidence,
                single_digit_min_confidence=args.single_digit_min_confidence,
                height_band=height_band,
                min_gap=min_gap,
            )

    edges = mutual_edges(claims_by_page)
    one_sided = one_sided_edges(claims_by_page)
    doc = {
        "timestamp": datetime.now(UTC).isoformat(),
        "command": sys.argv[:],
        "edge_band": EDGE_BAND,
        "min_height": args.min_height,
        "min_confidence": args.min_confidence,
        "single_digit_min_confidence": args.single_digit_min_confidence,
        "single_digit_height_band": list(height_band) if height_band else None,
        "single_digit_min_gap": round(min_gap, 1) if min_gap is not None else None,
        "pages": pages,
        "adjacency": [list(edge) for edge in edges],
        "one_sided": [list(edge) for edge in one_sided],
    }
    output = args.output or (args.volume / "adjacency.json")
    output.write_text(json.dumps(doc, indent=2))
    total_claims = sum(len(c) for c in claims_by_page.values())
    print(
        f"Wrote {output}: {len(pages)} pages, {total_claims} directed claims, "
        f"{len(edges)} mutual edges, {len(one_sided)} one-sided.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
