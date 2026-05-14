"""Detect text regions in insurance map images using EasyOCR (CRAFT detector)."""

import argparse
import json
import re
import sys
from pathlib import Path

import Levenshtein
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Map common street-type abbreviations to their full forms for normalization.
STREET_ABBREVS = {
    "ST": "STREET",
    "AVE": "AVENUE",
    "BLVD": "BOULEVARD",
    "PL": "PLACE",
    "DR": "DRIVE",
    "RD": "ROAD",
    "CT": "COURT",
    "LN": "LANE",
    "TER": "TERRACE",
    "TERR": "TERRACE",
    "HWY": "HIGHWAY",
    "PKWY": "PARKWAY",
    "CIR": "CIRCLE",
    "EXPY": "EXPRESSWAY",
}

# Direction prefix abbreviations expanded only when they appear as the first word.
DIRECTION_ABBREVS = {
    "N": "NORTH",
    "S": "SOUTH",
    "E": "EAST",
    "W": "WEST",
    "NE": "NORTHEAST",
    "NW": "NORTHWEST",
    "SE": "SOUTHEAST",
    "SW": "SOUTHWEST",
}

# Other prefix abbreviations expanded only when they appear as the first word.
# Applied after DIRECTION_ABBREVS so direction expansion doesn't interfere,
# and before STREET_ABBREVS so "ST" is treated as "Saint", not "Street".
PREFIX_ABBREVS = {
    "ST": "SAINT",
}

# Full direction words, used to strip leading direction prefixes from known street names.
DIRECTION_WORDS: frozenset[str] = frozenset(DIRECTION_ABBREVS.values())

# Prefixes that may be stripped from the start of a normalized street key when building
# match candidates. Extends DIRECTION_WORDS with "SAINT", which map labels often omit
# (e.g. the label "CHARLES" should still match "SAINT CHARLES AVENUE").
STRIPPABLE_PREFIXES: frozenset[str] = DIRECTION_WORDS | {"SAINT"}

# Full street-type words (STREET, AVENUE, …); used to strip trailing type suffixes when
# building fuzzy-match candidates so a misread name like "JOSEBH" can match
# "JOSEPH STREET" without paying for the full " STREET" edit cost.
STREET_TYPES: frozenset[str] = frozenset(STREET_ABBREVS.values())


def normalize_street(text: str) -> str:
    """Uppercase, strip punctuation, expand direction prefixes and street-type abbreviations.

    Direction abbreviations (N, S, E, W, etc.) are expanded only when they are the
    first word, since they appear as prefixes in street names (e.g. "N RAMPART ST"
    → "NORTH RAMPART STREET"). Street-type abbreviations (ST, AVE, …) are expanded
    wherever they appear.
    """
    text = re.sub(r"[^\w\s]", " ", text.upper())
    words = text.split()
    if not words:
        return ""
    words[0] = DIRECTION_ABBREVS.get(words[0], words[0])
    words[0] = PREFIX_ABBREVS.get(words[0], words[0])
    words = [STREET_ABBREVS.get(w, w) for w in words]
    return " ".join(words)


def is_number_only(text: str) -> bool:
    """Return True if text contains no alphabetic characters (e.g. block numbers)."""
    return not bool(re.search(r"[a-zA-Z]", text))


def _strip_street_type(s: str) -> str | None:
    """Return s without its trailing street-type word, or None if s has no type suffix."""
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] in STREET_TYPES:
        return parts[0]
    return None


def _match_candidates(s: str) -> list[str]:
    """Return the candidate forms to compare against when prefix-matching street key s.

    If s starts with a strippable prefix (a direction word or SAINT), also adds the
    prefix-stripped form so that a label like "CHARLES" can match "SAINT CHARLES AVE".
    """
    parts = s.split(" ", 1)
    if len(parts) == 2 and parts[0] in STRIPPABLE_PREFIXES:
        return [s, parts[1]]
    return [s]


def matches_any_street(text: str, normalized_streets: set[str]) -> bool:
    """Return True if text is a prefix-compatible match with any known street name.

    Handles labels that omit or include the street-type suffix ("COLUMBIA" matches
    "COLUMBIA PLACE"), omit a direction prefix ("LIBERTY" matches "SOUTH LIBERTY
    STREET"), or omit a SAINT prefix ("CHARLES" matches "SAINT CHARLES AVENUE").
    """
    normalized = normalize_street(text)
    for s in normalized_streets:
        for candidate in _match_candidates(s):
            if (
                normalized == candidate
                or (
                    len(candidate.split()) >= 2
                    and normalized.startswith(candidate + " ")
                )
                or candidate.startswith(normalized + " ")
            ):
                return True
    return False


def canonical_street_match(
    text: str,
    normalized_streets: set[str],
    fuzzy_threshold: float = 0.0,
    min_fuzzy_len: int = 5,
) -> str | None:
    """Return the key from normalized_streets that best matches text.

    Phase 1 — exact/prefix match: iterates normalized_streets and returns the first
    key whose prefix-compatible candidates (including direction- and SAINT-stripped
    forms) match normalize_street(text). Always returns the actual key so downstream
    block_index lookups succeed even when the label omits a prefix.

    Phase 2 — fuzzy match: if no exact match and fuzzy_threshold > 0, finds the key
    whose normalized Levenshtein distance to normalize_street(text) is smallest and
    below the threshold. Each key is compared against both its full form and a
    type-suffix-stripped form (e.g. "JOSEPH" from "JOSEPH STREET"), so that a
    one-letter OCR error like "JOSEBH" can match without paying for " STREET".
    Both strings must be at least min_fuzzy_len characters. Returns None if no match.
    """
    normalized = normalize_street(text)

    # Phase 1: exact/prefix match — return the actual key from normalized_streets.
    for s in normalized_streets:
        for candidate in _match_candidates(s):
            if (
                normalized == candidate
                or (
                    len(candidate.split()) >= 2
                    and normalized.startswith(candidate + " ")
                )
                or candidate.startswith(normalized + " ")
            ):
                return s

    # Phase 2: fuzzy match against full key and type-stripped form.
    if fuzzy_threshold <= 0 or len(normalized) < min_fuzzy_len:
        return None
    best_key: str | None = None
    best_norm_dist = fuzzy_threshold
    for s in normalized_streets:
        forms = [s]
        stripped = _strip_street_type(s)
        if stripped is not None and len(stripped) >= min_fuzzy_len:
            forms.append(stripped)
        for form in forms:
            if len(form) < min_fuzzy_len:
                continue
            dist = Levenshtein.distance(normalized, form)
            norm_dist = dist / max(len(normalized), len(form))
            if norm_dist < best_norm_dist:
                best_norm_dist = norm_dist
                best_key = s
    return best_key


def canonical_street_matches(
    text: str,
    normalized_streets: set[str],
    fuzzy_threshold: float = 0.0,
    min_fuzzy_len: int = 5,
) -> list[str]:
    """Return all keys from normalized_streets that match text.

    Phase 1 — exact/prefix match: returns every key with at least one
    prefix-compatible candidate matching normalize_street(text). Unlike
    canonical_street_match, this collects all matches rather than returning
    whichever the set iteration yields first, eliminating nondeterminism when
    a bare label (e.g. "CLINTON") could match multiple full names
    (e.g. "CLINTON AVENUE" and "CLINTON STREET").

    Phase 2 — fuzzy match: if Phase 1 finds nothing and fuzzy_threshold > 0,
    returns the single best fuzzy key (OCR errors typically have one target).
    """
    normalized = normalize_street(text)

    # Phase 1: collect every key that has at least one matching candidate form.
    matches: list[str] = []
    for s in normalized_streets:
        for candidate in _match_candidates(s):
            if (
                normalized == candidate
                or (
                    len(candidate.split()) >= 2
                    and normalized.startswith(candidate + " ")
                )
                or candidate.startswith(normalized + " ")
            ):
                matches.append(s)
                break  # count each key at most once

    if matches:
        return matches

    # Phase 2: single best fuzzy key (ambiguous OCR errors are rare).
    if fuzzy_threshold <= 0 or len(normalized) < min_fuzzy_len:
        return []
    best_key: str | None = None
    best_norm_dist = fuzzy_threshold
    for s in normalized_streets:
        forms = [s]
        stripped = _strip_street_type(s)
        if stripped is not None and len(stripped) >= min_fuzzy_len:
            forms.append(stripped)
        for form in forms:
            if len(form) < min_fuzzy_len:
                continue
            dist = Levenshtein.distance(normalized, form)
            norm_dist = dist / max(len(normalized), len(form))
            if norm_dist < best_norm_dist:
                best_norm_dist = norm_dist
                best_key = s
    return [best_key] if best_key is not None else []


def polygon_side_lengths(polygon: list[list[int]]) -> list[float]:
    """Return the four side lengths of a quadrilateral polygon."""
    pts = np.array(polygon, dtype=float)
    return [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]


def polygon_bbox(polygon: list[list[int]]) -> tuple[int, int, int, int]:
    """Return axis-aligned bounding box (x1, y1, x2, y2) of a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union of two axis-aligned bounding boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def deduplicate_detections(
    detections: list[dict],
    iou_threshold: float = 0.3,
    normalized_streets: set[str] | None = None,
) -> list[dict]:
    """Remove duplicate detections with greedy NMS.

    When normalized_streets is provided, detections whose text matches a known
    street name are ranked before non-matches regardless of EasyOCR confidence.
    Within each group, higher confidence wins. This prevents a high-confidence
    misread from suppressing a lower-confidence but correctly spelled street label.
    """

    def sort_key(d: dict) -> tuple[bool, float]:
        street_match = normalized_streets is not None and matches_any_street(
            d["text"], normalized_streets
        )
        return (street_match, d["confidence"])

    sorted_dets = sorted(detections, key=sort_key, reverse=True)
    kept: list[dict] = []
    kept_bboxes: list[tuple[int, int, int, int]] = []
    for det in sorted_dets:
        bbox = polygon_bbox(det["polygon"])
        if not any(bbox_iou(bbox, kb) > iou_threshold for kb in kept_bboxes):
            kept.append(det)
            kept_bboxes.append(bbox)
    return kept


def detect_text(
    image_path: str,
    min_size: int = 15,
    allowlist: str | None = None,
    reader: easyocr.Reader | None = None,
) -> list[dict]:
    """Run CRAFT-based text detection at 0°, 90°, and 270° and return all results.

    Runs three passes to catch both horizontal and vertical text. Polygons from
    rotated passes are mapped back to original image coordinates. Returns all raw
    detections — deduplication (NMS) is left to the caller, which has access to
    the street name list needed for street-aware NMS ordering.

    Note: EasyOCR's rotation_info parameter only rotates already-detected crops
    for the recognition stage, so it does not help detect vertical text regions.
    Running the full image at multiple angles is required.

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
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    all_detections: list[dict] = []
    readtext_kwargs: dict = {"paragraph": False, "min_size": min_size}
    if allowlist is not None:
        readtext_kwargs["allowlist"] = allowlist

    for angle in (0, 90, 270):
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        results = reader.readtext(np.array(rotated), **readtext_kwargs)
        for bbox, text, confidence in results:
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


def visualize_detections(
    image_path: str,
    accepted: list[dict],
    rejected: list[dict],
    output_path: str,
) -> None:
    """Draw detection polygons on the image: accepted in red, rejected in yellow.

    Rejected detections are those filtered out by the number or street-name filters.
    Both sets are drawn so the visualization shows where all candidate regions are.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
    except OSError:
        font = ImageFont.load_default()

    for color, detections in [((255, 200, 0), rejected), ((255, 0, 0), accepted)]:
        for det in detections:
            polygon = [tuple(pt) for pt in det["polygon"]]
            draw.polygon(polygon, outline=color, width=2)
            label = f"{det['text']} ({det['confidence']:.2f})"
            draw.text(polygon[0], label, fill=color, font=font)

    img.save(output_path)
    print(f"Saved annotated image to {output_path}", file=sys.stderr)


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
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
        help=(
            "Restrict OCR recognition to these characters. Defaults to letters + space."
        ),
    )
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=15,
        metavar="PX",
        help="Minimum short side passed to the EasyOCR detector (default: 15)",
    )
    args = parser.parse_args()

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    for image_path in args.images:
        if len(args.images) > 1:
            print(f"\n--- {image_path} ---", file=sys.stderr)
        stem = Path(image_path).name.split(".")[0]
        output_path = str(Path(image_path).parent / (stem + ".streets.json"))
        detections = detect_text(
            image_path,
            min_size=args.min_short_side,
            allowlist=args.allowlist,
            reader=reader,
        )
        with open(output_path, "w") as f:
            f.write(json.dumps(detections, indent=2))
        print(f"Wrote {len(detections)} detections to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
