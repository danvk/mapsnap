"""Unit tests for street-name helpers (now in streets.py)."""

import numpy as np

from mapsnap.detect_text import NON_STREET_TEXT, _trim_underlines
from mapsnap.streets import (
    canonical_street_match,
    matches_any_street,
    normalize_street,
)

# ---------------------------------------------------------------------------
# normalize_street
# ---------------------------------------------------------------------------


def test_normalize_uppercase():
    assert normalize_street("Magazine St") == "MAGAZINE STREET"


def test_normalize_direction_prefix():
    assert normalize_street("N Rampart St") == "NORTH RAMPART STREET"


def test_normalize_direction_not_expanded_mid_word():
    # "S" in the middle of a name is not a direction prefix
    assert normalize_street("Canal St") == "CANAL STREET"


def test_normalize_strips_punctuation():
    assert normalize_street("St. Charles Ave.") == "SAINT CHARLES AVENUE"


def test_normalize_empty():
    assert normalize_street("") == ""


def test_normalize_multi_word_direction():
    assert normalize_street("NE Broad Ave") == "NORTHEAST BROAD AVENUE"


def test_normalize_saint_prefix():
    assert normalize_street("St. Charles Ave") == "SAINT CHARLES AVENUE"


def test_normalize_saint_prefix_without_period():
    assert normalize_street("St Charles Ave") == "SAINT CHARLES AVENUE"


def test_normalize_st_suffix_still_street():
    # "ST" at the end (not first word) is still "STREET"
    assert normalize_street("Magazine St") == "MAGAZINE STREET"


# ---------------------------------------------------------------------------
# matches_any_street
# ---------------------------------------------------------------------------

STREETS = {
    "MAGAZINE STREET",
    "COLUMBIA PLACE",
    "CANAL STREET",
    "NORTH RAMPART STREET",
    "ESPLANADE AVENUE",
}


def test_exact_match_full_name():
    assert matches_any_street("Magazine Street", STREETS)


def test_exact_match_abbreviated():
    # "Magazine St" normalizes to "MAGAZINE STREET" — exact match
    assert matches_any_street("Magazine St", STREETS)


def test_bare_name_matches_full():
    # "COLUMBIA" is a prefix of "COLUMBIA PLACE "
    assert matches_any_street("Columbia", STREETS)


def test_full_name_matches_bare_label():
    # Detection says "Columbia Place" but street set has "COLUMBIA PLACE"
    assert matches_any_street("Columbia Place", STREETS)


def test_no_match_unknown_street():
    assert not matches_any_street("Bourbon Street", STREETS)


def test_no_match_partial_word():
    # "CANAL" should not match "CANAL STREET" as a suffix test — but it does
    # because "CANAL" is a prefix of "CANAL STREET "
    assert matches_any_street("Canal", STREETS)


def test_no_match_random_word():
    assert not matches_any_street("HELLO", STREETS)


def test_direction_expanded_before_matching():
    # "N Rampart St" → "NORTH RAMPART STREET" which is in the set
    assert matches_any_street("N Rampart St", STREETS)


def test_empty_streets_set():
    assert not matches_any_street("Canal Street", set())


def test_empty_text():
    assert not matches_any_street("", STREETS)


def test_number_only_does_not_match():
    # Block numbers should not match any street
    assert not matches_any_street("1200", STREETS)


def test_prefix_of_detection_longer_than_street():
    # Detection reads "Esplanade Ave Ext" — "ESPLANADE AVENUE" is a prefix of the normalized text
    assert matches_any_street("Esplanade Ave Ext", STREETS)


def test_ignores_north_south():
    assert matches_any_street("rampart", STREETS)


def test_saint_prefix_stripped():
    # "CHARLES" should match "SAINT CHARLES AVENUE" once "SAINT" is stripped from the key
    assert matches_any_street("Charles", {"SAINT CHARLES AVENUE"})


def test_saint_prefix_stripped_with_type():
    assert matches_any_street("Charles Ave", {"SAINT CHARLES AVENUE"})


def test_saint_prefix_no_false_positive():
    assert not matches_any_street("Joseph", {"SAINT CHARLES AVENUE"})


def test_no_match_short_street_prefix():
    # "W A R E" normalizes to "WEST A R E" which starts with "WEST " — but a 1-word
    # street name ("WEST") should not match any text that happens to start with it.
    # This prevents WARE HOUSE waterfront labels from matching the street "WEST".
    assert not matches_any_street("W A R E", {"WEST", "WEST ROAD"})


def test_multi_word_street_prefix_still_matches():
    # "Esplanade Ave Ext" → "ESPLANADE AVENUE EXT"; "ESPLANADE AVENUE" (2 words) is
    # a valid prefix of the longer detection.
    assert matches_any_street("Esplanade Ave Ext", STREETS)


# ---------------------------------------------------------------------------
# canonical_street_match
# ---------------------------------------------------------------------------

CANONICAL_STREETS = {
    "TCHOUPITOULAS STREET",
    "FRONT STREET",
    "CANAL STREET",
}


def test_canonical_exact_returns_key():
    # Exact match via normalize_street — returned value is the actual block_index key.
    result = canonical_street_match("Tchoupitoulas St", CANONICAL_STREETS)
    assert result == "TCHOUPITOULAS STREET"


def test_canonical_saint_prefix_returns_key():
    # "CHARLES" (no SAINT prefix) matches "SAINT CHARLES AVENUE" via prefix stripping.
    # The returned value is the actual key, not the bare normalized text.
    result = canonical_street_match("Charles", {"SAINT CHARLES AVENUE"})
    assert result == "SAINT CHARLES AVENUE"


def test_canonical_direction_prefix_returns_key():
    # "RAMPART" matches "NORTH RAMPART STREET" via direction prefix stripping.
    # The returned key is the full name so block_index lookups succeed.
    result = canonical_street_match("RAMPART", {"NORTH RAMPART STREET"})
    assert result == "NORTH RAMPART STREET"


def test_canonical_no_match():
    assert canonical_street_match("ZZZZZZZZZ", CANONICAL_STREETS) is None


# ---------------------------------------------------------------------------
# NON_STREET_TEXT
# ---------------------------------------------------------------------------


def test_non_street_text_contains_pipe_sizes():
    # Exact forms with inch mark for higher-quality scans.
    assert '6" W. PIPE' in NON_STREET_TEXT
    assert '12" W. PIPE' in NON_STREET_TEXT


def test_non_street_text_contains_confusable_forms():
    # Horizontal: "6"" → "E"; W reads as W, X, Y, or M depending on crop scale.
    assert "EWPIPE" in NON_STREET_TEXT
    assert "EXPIPE" in NON_STREET_TEXT
    assert "EYPIPE" in NON_STREET_TEXT
    # Vertical text: '6' → 'S', '"' visible
    assert 'S"WPIPE' in NON_STREET_TEXT
    # Vertical text: '6' → 'S', '"' dropped, 'W' → 'M'
    assert "SM PIPE" in NON_STREET_TEXT


def test_non_street_text_all_uppercase():
    assert all(s == s.upper() for s in NON_STREET_TEXT)


# ---------------------------------------------------------------------------
# _trim_underlines
# ---------------------------------------------------------------------------


def _make_img(height: int, width: int, dark_rows: list[int]) -> np.ndarray:
    """White (H, W, 3) image with fully dark (0) rows at the given indices."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in dark_rows:
        img[row, :, :] = 0
    return img


def test_trim_underlines_trims_dark_bottom_row():
    # Box spanning rows 0-19; dark row at 17 (within bottom 25%).
    img = _make_img(50, 100, dark_rows=[17])
    result = _trim_underlines(img, [[0, 100, 0, 20]])
    assert result == [[0, 100, 0, 17]]


def test_trim_underlines_no_underline_unchanged():
    # No dark rows — box should come back untouched.
    img = _make_img(50, 100, dark_rows=[])
    result = _trim_underlines(img, [[0, 100, 0, 20]])
    assert result == [[0, 100, 0, 20]]


def test_trim_underlines_dark_row_outside_scan_window_unchanged():
    # Dark row at row 2 is outside the bottom 25% of a 0-20 box (scan starts at 15).
    img = _make_img(50, 100, dark_rows=[2])
    result = _trim_underlines(img, [[0, 100, 0, 20]])
    assert result == [[0, 100, 0, 20]]


def test_trim_underlines_enforces_minimum_height():
    # If trimming would produce a box shorter than 4px, clamp to y_min+4.
    img = _make_img(50, 100, dark_rows=[1])
    result = _trim_underlines(img, [[0, 100, 0, 4]])
    # scan_start = 4 - max(1, int(4*0.25)) = 3; row 1 is outside → no trim
    assert result[0][3] >= result[0][2] + 4
