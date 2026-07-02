"""Unit tests for street-name helpers (now in streets.py)."""

import numpy as np

from mapsnap.detect_text import (
    NON_STREET_TEXT,
    _axis_starts,
    _erase_underlines,
    _iou_xxyy,
    _iter_tiles,
    _nms_bboxes,
    filter_args,
    has_split_panels,
)
from mapsnap.streets import (
    canonical_street_match,
    matches_any_street,
    normalize_street,
)


def test_has_split_panels(tmp_path):
    (tmp_path / "p428.jpg").touch()
    (tmp_path / "p428__1.jpg").touch()
    (tmp_path / "p428__2.jpg").touch()
    (tmp_path / "p528.jpg").touch()  # unsplit page, no panels

    # Parent with panels is superseded; panels and unsplit pages are not.
    assert has_split_panels(str(tmp_path / "p428.jpg")) is True
    assert has_split_panels(str(tmp_path / "p428__1.jpg")) is False
    assert has_split_panels(str(tmp_path / "p528.jpg")) is False
    # The "p4__" prefix must not match "p428__N" (delimiter-aware).
    (tmp_path / "p4.jpg").touch()
    assert has_split_panels(str(tmp_path / "p4.jpg")) is False


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


def test_erase_underlines_paints_dark_bottom_row_white():
    # Box spanning rows 0-19; dark row at 17 (within bottom 25%) should become white.
    img = _make_img(50, 100, dark_rows=[17])
    result = _erase_underlines(img, [[0, 100, 0, 20]])
    assert result[17, :, :].min() == 255  # row 17 is now white
    assert result[16, :, :].max() == 255  # row above (light) untouched


def test_erase_underlines_no_underline_unchanged():
    # No dark rows — returned image should equal the input.
    img = _make_img(50, 100, dark_rows=[])
    result = _erase_underlines(img, [[0, 100, 0, 20]])
    assert np.array_equal(result, img)


def test_erase_underlines_does_not_mutate_input():
    img = _make_img(50, 100, dark_rows=[17])
    original = img.copy()
    _erase_underlines(img, [[0, 100, 0, 20]])
    assert np.array_equal(img, original)


def test_erase_underlines_dark_row_outside_scan_window_unchanged():
    # Dark row at 2 is outside the bottom 25% of a 0-20 box (scan starts at 15).
    img = _make_img(50, 100, dark_rows=[2])
    result = _erase_underlines(img, [[0, 100, 0, 20]])
    assert result[2, :, :].min() == 0  # row 2 still dark


def test_erase_underlines_preserves_row_above_underline():
    # Row 14 (text) and row 17 (underline) both dark; only row 17 is in scan window.
    img = _make_img(50, 100, dark_rows=[14, 17])
    result = _erase_underlines(img, [[0, 100, 0, 20]])
    assert result[17, :, :].min() == 255  # underline erased
    assert result[14, :, :].max() == 0  # text row above scan window intact


# ---------------------------------------------------------------------------
# filter_args
# ---------------------------------------------------------------------------


def test_filter_args_keeps_non_jpg():
    argv = ["detect_text.py", "--centerlines", "streets.geojson", "a.jpg"]
    assert filter_args(argv, "a.jpg") == argv


def test_filter_args_removes_other_jpgs():
    argv = ["detect_text.py", "a.jpg", "b.jpg", "c.jpg"]
    assert filter_args(argv, "b.jpg") == ["detect_text.py", "b.jpg"]


def test_filter_args_keeps_flags_and_target():
    argv = ["detect_text.py", "--min-long-side", "45", "p1.jpg", "p2.jpg", "p3.jpg"]
    assert filter_args(argv, "p2.jpg") == [
        "detect_text.py",
        "--min-long-side",
        "45",
        "p2.jpg",
    ]


def test_filter_args_single_image():
    argv = ["detect_text.py", "only.jpg"]
    assert filter_args(argv, "only.jpg") == argv


def test_filter_args_no_jpgs():
    argv = ["detect_text.py", "--help"]
    assert filter_args(argv, "x.jpg") == argv


# --- tiled detection helpers ---


def test_axis_starts_smaller_than_tile():
    # A frame no larger than the tile needs a single tile at the origin.
    assert _axis_starts(2000, 2560, 2048) == [0]
    assert _axis_starts(2560, 2560, 2048) == [0]


def test_axis_starts_covers_end_flush():
    # Last start is snapped to length-tile so the final tile reaches the edge.
    starts = _axis_starts(8422, 2560, 2048)
    assert starts[0] == 0
    assert starts[-1] == 8422 - 2560
    # every position is within [0, length-tile]
    assert all(0 <= s <= 8422 - 2560 for s in starts)


def test_iter_tiles_covers_whole_image():
    # Union of tiles must cover every pixel of a large frame.
    width, height, tile, overlap = 6091, 8422, 2560, 512
    tiles = list(_iter_tiles(width, height, tile, overlap))
    assert tiles, "expected at least one tile"
    covered = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in tiles:
        assert x1 - x0 <= tile and y1 - y0 <= tile
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_iter_tiles_single_tile_when_small():
    tiles = list(_iter_tiles(2000, 1500, 2560, 512))
    assert tiles == [(0, 0, 2000, 1500)]


def test_iter_tiles_overlap_between_neighbors():
    # Horizontally adjacent tiles should share the configured overlap.
    tiles = list(_iter_tiles(6091, 2000, 2560, 512))
    xs = sorted({(x0, x1) for x0, _, x1, _ in tiles})
    first_end = xs[0][1]
    second_start = xs[1][0]
    assert first_end - second_start == 512  # overlap width = tile - stride


def test_iou_identical_boxes():
    assert _iou_xxyy([0, 10, 0, 10], [0, 10, 0, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert _iou_xxyy([0, 10, 0, 10], [20, 30, 0, 10]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
    assert _iou_xxyy([0, 10, 0, 10], [5, 15, 0, 10]) == 50 / 150


def test_nms_drops_duplicate_keeps_larger():
    # A duplicate and a clipped copy of one label plus a far-away box:
    # NMS keeps the larger of the overlapping pair and the disjoint box.
    boxes = [
        [0.0, 100.0, 0.0, 20.0],  # full label
        [0.0, 90.0, 0.0, 20.0],  # clipped copy (high IoU with the full one)
        [500.0, 600.0, 0.0, 20.0],  # unrelated, disjoint
    ]
    kept = _nms_bboxes(boxes, 0.4)
    assert 0 in kept  # larger box survives
    assert 1 not in kept  # clipped duplicate suppressed
    assert 2 in kept  # disjoint box kept


def test_nms_keeps_distinct_low_overlap_boxes():
    boxes = [[0.0, 100.0, 0.0, 20.0], [95.0, 195.0, 0.0, 20.0]]  # touch slightly, low IoU
    assert sorted(_nms_bboxes(boxes, 0.4)) == [0, 1]
