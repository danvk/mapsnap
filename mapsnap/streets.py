"""Shared street-name constants, normalization, and matching utilities.

Also contains the Block dataclass and build_block_index function used by both
detect_text.py and georef_from_labels.py to build a centerlines index.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field

import numpy as np

# Map common street-type abbreviations to their full forms for normalization.
STREET_ABBREVS = {
    "ST": "STREET",
    "AVE": "AVENUE",
    "AV": "AVENUE",
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

# Full street-type words (STREET, AVENUE, …); used to identify bare-name aliases in build_block_index.
STREET_TYPES: frozenset[str] = frozenset(STREET_ABBREVS.values())

# Irregular ordinal words for 1–19.
_ORDINAL_ONES = [
    "",
    "FIRST",
    "SECOND",
    "THIRD",
    "FOURTH",
    "FIFTH",
    "SIXTH",
    "SEVENTH",
    "EIGHTH",
    "NINTH",
    "TENTH",
    "ELEVENTH",
    "TWELFTH",
    "THIRTEENTH",
    "FOURTEENTH",
    "FIFTEENTH",
    "SIXTEENTH",
    "SEVENTEENTH",
    "EIGHTEENTH",
    "NINETEENTH",
]

# Ordinal tens words (exact multiples of 10).
_ORDINAL_TENS = [
    "",
    "",
    "TWENTIETH",
    "THIRTIETH",
    "FORTIETH",
    "FIFTIETH",
    "SIXTIETH",
    "SEVENTIETH",
    "EIGHTIETH",
    "NINETIETH",
]

# Cardinal tens words used to form compound ordinals like "TWENTY FIRST".
_CARDINAL_TENS = [
    "",
    "",
    "TWENTY",
    "THIRTY",
    "FORTY",
    "FIFTY",
    "SIXTY",
    "SEVENTY",
    "EIGHTY",
    "NINETY",
]


def _num_to_ordinal_word(n: int) -> str:
    """Convert an integer 1–99 to its ordinal word form (e.g. 4 → "FOURTH", 21 → "TWENTY FIRST")."""
    if 1 <= n <= 19:
        return _ORDINAL_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _ORDINAL_TENS[tens]
    return f"{_CARDINAL_TENS[tens]} {_ORDINAL_ONES[ones]}"


# Map any ordinal token (with any numeric suffix) to its word form.
# Keys cover all four suffixes (ST/ND/RD/TH) for robustness against OCR suffix errors.
_ORDINALS: dict[str, str] = {
    f"{n}{suffix}": _num_to_ordinal_word(n)
    for n in range(1, 100)
    for suffix in ("ST", "ND", "RD", "TH")
}


def _canonical_ordinal_suffix(n: int) -> str:
    """Return the correct ordinal suffix (ST/ND/RD/TH) for integer n."""
    last_two = n % 100
    last_one = n % 10
    if 11 <= last_two <= 13:
        return "TH"
    if last_one == 1:
        return "ST"
    if last_one == 2:
        return "ND"
    if last_one == 3:
        return "RD"
    return "TH"


# Inverse of _ORDINALS: ordinal word/phrase → canonical numeric string.
# e.g., "FOURTH" → "4TH", "TWENTY FIRST" → "21ST".
# Used by generate_vocab_strings to add numeric label variants to the CTC trie.
ORDINAL_WORD_TO_NUM: dict[str, str] = {
    _num_to_ordinal_word(n): f"{n}{_canonical_ordinal_suffix(n)}" for n in range(1, 100)
}


def normalize_street(text: str) -> str:
    """Uppercase, strip punctuation, expand direction prefixes and street-type abbreviations.

    Direction abbreviations (N, S, E, W, etc.) are expanded only when they are the
    first word, since they appear as prefixes in street names (e.g. "N RAMPART ST"
    → "NORTH RAMPART STREET"). Street-type abbreviations (ST, AVE, …) are expanded
    wherever they appear. Numeric ordinals (e.g. "4TH", "21ST") are expanded to word
    form ("FOURTH", "TWENTY FIRST") so that "S. 4TH ST" matches "South Fourth Street".
    """
    text = re.sub(r"[^\w\s]", " ", text.upper())
    words = text.split()
    if not words:
        return ""
    # Expand numeric ordinals; may expand one token to two ("21ST" → ["TWENTY", "FIRST"]).
    words = [w for token in words for w in _ORDINALS.get(token, token).split()]
    words[0] = DIRECTION_ABBREVS.get(words[0], words[0])
    words[0] = PREFIX_ABBREVS.get(words[0], words[0])
    words = [STREET_ABBREVS.get(w, w) for w in words]
    return " ".join(words)


def is_number_only(text: str) -> bool:
    """Return True if text contains no alphabetic characters (e.g. block numbers)."""
    return not bool(re.search(r"[a-zA-Z]", text))


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
) -> str | None:
    """Return the key from normalized_streets that best matches text.

    Iterates normalized_streets and returns the first key whose prefix-compatible
    candidates (including direction- and SAINT-stripped forms) match
    normalize_street(text). Always returns the actual key so downstream block_index
    lookups succeed even when the label omits a prefix. Returns None if no match.

    When normalization expands a direction abbreviation (e.g. "W"→"WEST") and the
    raw text is itself a known alias (e.g. "W" for "AVENUE W"), the alias is returned
    directly and the expansion path is skipped — preventing "W" from matching all
    "WEST X" streets when it should match only "AVENUE W".
    """
    normalized = normalize_street(text)
    raw = text.upper().strip()
    if raw != normalized and raw in normalized_streets:
        return raw
    if normalized in DIRECTION_WORDS:
        # Bare direction word (e.g. "EAST", "WEST"): only match exact key aliases
        # or two-word "DIRECTION TYPE" keys (e.g. "EAST STREET", "WEST AVENUE").
        # Prevents "EAST" from prefix-matching all "EAST 103RD STREET",
        # "EAST FOURTH STREET", etc. via candidate.startswith("EAST ").
        for s in normalized_streets:
            if s == normalized:
                return s
            if (
                s.startswith(normalized + " ")
                and s[len(normalized) + 1 :] in STREET_TYPES
            ):
                return s
        return None
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
    return None


def canonical_street_matches(
    text: str,
    normalized_streets: set[str],
) -> list[str]:
    """Return all keys from normalized_streets that match text.

    Returns every key with at least one prefix-compatible candidate matching
    normalize_street(text). Collects all matches rather than returning whichever
    the set iteration yields first, eliminating nondeterminism when a bare label
    (e.g. "CLINTON") could match multiple full names (e.g. "CLINTON AVENUE" and
    "CLINTON STREET").

    When normalization expands a direction abbreviation (e.g. "W"→"WEST") and the
    raw text is itself a known alias (e.g. "W" for "AVENUE W"), only that alias is
    returned — preventing "W" from matching all 70+ "WEST X" streets.

    Bare direction words (e.g. "EAST", "WEST") only match exact key aliases or
    two-word "DIRECTION TYPE" keys (e.g. "EAST STREET") — not direction-prefixed
    named/numbered streets like "EAST 103RD STREET".
    """
    normalized = normalize_street(text)
    raw = text.upper().strip()
    if raw != normalized and raw in normalized_streets:
        return [raw]
    if normalized in DIRECTION_WORDS:
        matches = []
        for s in normalized_streets:
            if s == normalized:
                matches.append(s)
            elif (
                s.startswith(normalized + " ")
                and s[len(normalized) + 1 :] in STREET_TYPES
            ):
                matches.append(s)
        return matches
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
    return matches


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
    """Remove duplicate detections with greedy non-maximum suppression (NMS).

    When normalized_streets is provided, detections whose text matches a known
    street name are ranked before non-matches regardless of EasyOCR confidence.
    Within each group, higher confidence wins. This prevents a high-confidence
    misread from suppressing a lower-confidence but correctly spelled street label.

    Assembled (Phase 3) and promoted (Phase 4) detections always rank above plain
    detections so that synthetic or rescued detections are not suppressed by a
    higher-confidence misread at the same position.
    """

    def sort_key(d: dict) -> tuple[bool, bool, float]:
        street_match = normalized_streets is not None and matches_any_street(
            d["text"], normalized_streets
        )
        is_priority = bool(d.get("assembled") or d.get("promoted"))
        return (street_match, is_priority, d["confidence"])

    sorted_dets = sorted(detections, key=sort_key, reverse=True)
    kept: list[dict] = []
    kept_bboxes: list[tuple[int, int, int, int]] = []
    for det in sorted_dets:
        bbox = polygon_bbox(det["polygon"])
        if not any(bbox_iou(bbox, kb) > iou_threshold for kb in kept_bboxes):
            kept.append(det)
            kept_bboxes.append(bbox)
    return kept


# ---------------------------------------------------------------------------
# Centerlines index
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """One line segment from centerlines.geojson."""

    street_name: str
    coords: np.ndarray = dataclass_field(repr=False)  # shape (N, 2), columns [lon, lat]


def build_block_index(geojson: dict) -> dict[str, list[Block]]:
    """Index GeoJSON centerline features by normalized street_name.

    Also adds unambiguous base-name aliases so that bare labels (e.g. "MAGAZINE")
    match full centerline names (e.g. "MAGAZINE STREET"), direction-stripped aliases
    (e.g. "LIBERTY" for "SOUTH LIBERTY STREET"), and leading-type-stripped aliases
    (e.g. "X" for "AVENUE X"). Each alias is only added when unambiguous (exactly
    one full name maps to that stripped form).
    """
    index: dict[str, list[Block]] = {}
    for feature in geojson["features"]:
        raw_name = feature["properties"].get("street_name", "")
        if not raw_name:
            continue
        name = normalize_street(raw_name)
        geom = feature["geometry"]
        lines = (
            geom["coordinates"]
            if geom["type"] == "MultiLineString"
            else [geom["coordinates"]]
        )
        for line in lines:
            if len(line) < 2:
                continue
            coords = np.array([[c[0], c[1]] for c in line], dtype=float)
            index.setdefault(name, []).append(Block(street_name=name, coords=coords))

    # Build aliases from bare name → full name for unambiguous cases.
    # e.g. "MAGAZINE STREET" → also index under "MAGAZINE".
    base_to_full: dict[str, list[str]] = defaultdict(list)
    for key in index:
        parts = key.rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in STREET_TYPES:
            base_to_full[parts[0]].append(key)
    for base, full_names in base_to_full.items():
        if len(full_names) == 1 and base not in index:
            # Alias shares the same list object as the full name entry so that
            # id()-based deduplication in callers can detect and collapse aliases.
            index[base] = index[full_names[0]]

    # Build aliases with direction prefix stripped for unambiguous cases.
    # e.g. "SOUTH LIBERTY STREET" → also index under "LIBERTY STREET" and "LIBERTY"
    # (the latter via the bare-name alias already added above).
    # Iterating list(index.keys()) captures both original keys and the aliases just added.
    dir_stripped_to_full: dict[str, list[str]] = defaultdict(list)
    for key in list(index.keys()):
        parts = key.split(" ", 1)
        if len(parts) == 2 and parts[0] in DIRECTION_WORDS and parts[1] not in index:
            dir_stripped_to_full[parts[1]].append(key)
    for stripped, full_names in dir_stripped_to_full.items():
        if len(full_names) == 1:
            # Same list object as the full-name entry; see comment above.
            index[stripped] = index[full_names[0]]

    # Build aliases with direction suffix stripped for unambiguous cases.
    # e.g. "MAIN STREET NORTHEAST" → also index under "MAIN STREET" (then "MAIN" via
    # bare-name alias) when only one direction of that street exists.
    # Iterating list(index.keys()) captures both original keys and aliases added above.
    dir_suffix_stripped_to_full: dict[str, list[str]] = defaultdict(list)
    for key in list(index.keys()):
        parts = key.rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in DIRECTION_WORDS and parts[0] not in index:
            dir_suffix_stripped_to_full[parts[0]].append(key)
    for stripped, full_names in dir_suffix_stripped_to_full.items():
        if len(full_names) == 1:
            index[stripped] = index[full_names[0]]

    # Build aliases with leading type word stripped for unambiguous cases.
    # e.g. "AVENUE X" → also index under "X".
    # Iterating list(index.keys()) captures all aliases added so far.
    leading_type_stripped: dict[str, list[str]] = defaultdict(list)
    for key in list(index.keys()):
        parts = key.split(" ", 1)
        if len(parts) == 2 and parts[0] in STREET_TYPES and parts[1] not in index:
            leading_type_stripped[parts[1]].append(key)
    for stripped, full_names in leading_type_stripped.items():
        if len(full_names) == 1:
            index[stripped] = index[full_names[0]]

    return index
