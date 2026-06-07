"""Tests for georef_from_labels helpers."""

import math

import numpy as np

from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    _cluster_geo_coords,
    assemble_hint_groups,
    correct_square_feature_dirs,
    label_features,
    promote_avenue_letters,
    project_to_polyline,
)
from mapsnap.streets import Block


# ---------------------------------------------------------------------------
# assemble_hint_groups
# ---------------------------------------------------------------------------


def _make_det(
    text: str,
    cx: float,
    cy: float,
    long_side: float = 60.0,
    short_side: float = 18.0,
    confidence: float = 0.9,
    dir_pix: float = 0.0,
    angle: int = 0,
    hint: bool = False,
) -> dict:
    """Build a mock detection dict with a horizontal bounding polygon."""
    half_long = long_side / 2.0
    half_short = short_side / 2.0
    return {
        "polygon": [
            [int(cx - half_long), int(cy - half_short)],
            [int(cx + half_long), int(cy - half_short)],
            [int(cx + half_long), int(cy + half_short)],
            [int(cx - half_long), int(cy + half_short)],
        ],
        "text": text,
        "confidence": confidence,
        "angle": angle,
        "long_side": long_side,
        "short_side": short_side,
        "dir_pix": dir_pix,
        **({"hint": True} if hint else {}),
    }


def test_assemble_three_token_label():
    # "EAST SEVENTH ST" split into three adjacent boxes; should assemble.
    # Positions: EAST (long=40, cx=20), SEVENTH (long=80, cx=80), ST. (long=30, cx=135)
    # Edge gaps: 20+20=40 right of EAST; 80-40=40 left of SEVENTH → edge gap = 0px ✓
    #            80+40=120 right of SEVENTH; 135-15=120 left of ST. → edge gap = 0px ✓
    streets = {"EAST SEVENTH STREET"}
    dets = [_make_det("SEVENTH", cx=80, cy=100, long_side=80)]
    hints = [
        _make_det("EAST", cx=20, cy=100, long_side=40, hint=True),
        _make_det("ST.", cx=135, cy=100, long_side=30, hint=True),
    ]
    result = assemble_hint_groups(dets, hints, streets)
    assert len(result) == 1
    assert result[0]["text"] == "EAST SEVENTH STREET"
    assert result[0]["assembled"] is True
    assert result[0]["confidence"] == 0.9  # min of non-hint confidences
    # Assembled polygon spans from EAST left edge (0) to ST. right edge (150)
    xs = [p[0] for p in result[0]["polygon"]]
    assert min(xs) <= 0 and max(xs) >= 150


def test_assemble_non_adjacent_groups_not_merged():
    # "EAST" (long_side=60) and "SEVENTH" (long_side=60) with a 200px edge gap.
    # EAST center=50 (edges 20-80), SEVENTH center=350 (edges 320-380).
    # Edge gap = 350-50 - (60+60)/2 = 300-60 = 240 > max_gap_px.
    streets = {"EAST SEVENTH STREET"}
    dets = [_make_det("SEVENTH", cx=350, cy=100)]
    hints = [_make_det("EAST", cx=50, cy=100, hint=True)]
    result = assemble_hint_groups(dets, hints, streets)
    assert result == []


def test_assemble_no_hints_returns_empty():
    streets = {"SEVENTH STREET"}
    dets = [_make_det("SEVENTH", cx=100, cy=100)]
    result = assemble_hint_groups(dets, [], streets)
    assert result == []


def test_assemble_ambiguous_match_skipped():
    # "SEVENTH ST" could match SEVENTH STREET but also an unrelated match —
    # here we use a streets set where "SEVENTH" alone matches two streets.
    streets = {"EAST SEVENTH STREET", "WEST SEVENTH STREET"}
    dets = [_make_det("SEVENTH", cx=100, cy=100)]
    hints = [_make_det("ST.", cx=200, cy=100, hint=True)]
    # "SEVENTH STREET" normalized → matches both EAST and WEST variants?
    # canonical_street_matches("SEVENTH STREET", ...) returns keys that
    # prefix-match "SEVENTH STREET"; both "EAST SEVENTH STREET" and
    # "WEST SEVENTH STREET" have "SEVENTH STREET" as a suffix-match candidate,
    # so two matches → ambiguous → no assembly.
    result = assemble_hint_groups(dets, hints, streets)
    assert result == []


def test_assemble_hint_only_run_skipped():
    # A run with only hints (no non-hint name) should not produce a detection.
    streets = {"EAST STREET"}
    hints = [
        _make_det("EAST", cx=50, cy=100, hint=True),
        _make_det("ST.", cx=120, cy=100, hint=True),
    ]
    result = assemble_hint_groups([], hints, streets)
    assert result == []


def test_assemble_perpendicular_gap_too_large():
    # "EAST" and "SEVENTH" are at same parallel position but far apart perpendicularly.
    streets = {"EAST SEVENTH STREET"}
    dets = [_make_det("SEVENTH", cx=150, cy=200)]  # cy differs by 100
    hints = [_make_det("EAST", cx=50, cy=100, hint=True)]
    result = assemble_hint_groups(dets, hints, streets, perp_tolerance_px=15.0)
    assert result == []


# ---------------------------------------------------------------------------
# promote_avenue_letters
# ---------------------------------------------------------------------------


def test_promote_single_char_near_avenue_hint():
    # "X" in the same vertical column as an "AVENUE" hint → promoted with corrected dir_pix.
    # Both share dir_pix=π/2 (vertical), as in the real p88 data where detect_text.py
    # already computes dir_pix correctly from the CRAFT polygon even for square boxes.
    streets = {"AVENUE X", "X"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det("X", cx=192, cy=274, long_side=24, short_side=24, dir_pix=math.pi / 2)
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1
    assert result[0]["text"] == "X"
    assert result[0]["promoted"] is True
    assert abs(result[0]["dir_pix"] - math.pi / 2) < 0.01


def test_promote_wrong_dir_bucket_not_promoted():
    # "X" has dir_pix=0.0 (wrong bucket, different from AVENUE hint's π/2) → no promotion.
    # promote_avenue_letters requires detection and hint in the same direction bucket.
    streets = {"AVENUE X", "X"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [_make_det("X", cx=192, cy=274, long_side=24, short_side=24, dir_pix=0.0)]
    result = promote_avenue_letters(hints, dets, streets)
    assert result == []


def test_promote_no_matching_street():
    # "Q" is not in the streets set → canonical_street_matches returns [] → no promotion.
    streets = {"AVENUE X", "X"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det("Q", cx=192, cy=274, long_side=24, short_side=24, dir_pix=math.pi / 2)
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert result == []


def test_promote_different_column_not_promoted():
    # "X" and "AVENUE" hint are in different columns (|perp diff| >> tolerance).
    streets = {"AVENUE X", "X"}
    hints = [
        _make_det(
            "AVENUE", cx=500, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det("X", cx=192, cy=274, long_side=24, short_side=24, dir_pix=math.pi / 2)
    ]
    result = promote_avenue_letters(hints, dets, streets, perp_tolerance_px=20.0)
    assert result == []


def test_promote_multi_char_not_promoted():
    # Multi-character detection is not eligible for promotion.
    streets = {"AVENUE XX", "XX"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det(
            "XX", cx=192, cy=274, long_side=40, short_side=20, dir_pix=math.pi / 2
        )
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert result == []


def test_promote_direction_hint_not_used():
    # Direction hints (EAST, NORTH) are not type-word hints → no promotion.
    streets = {"AVENUE X", "X"}
    hints = [_make_det("EAST", cx=193, cy=300, long_side=80, hint=True, dir_pix=0.0)]
    dets = [_make_det("X", cx=192, cy=274, long_side=24, short_side=24, dir_pix=0.0)]
    result = promote_avenue_letters(hints, dets, streets)
    assert result == []


# ---------------------------------------------------------------------------
# correct_square_feature_dirs
# ---------------------------------------------------------------------------


def test_correct_square_feature_dir():
    # A square-ish feature near an AVENUE hint gets dir_pix corrected to the hint's direction.
    # Hint at cx=270 (same cy), feature at cx=192 — distance 78px < default 200px.
    hints = [
        _make_det(
            "AVENUE", cx=270, cy=300, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    det = _make_det("X", cx=192, cy=300, long_side=24, short_side=24, dir_pix=0.0)
    features = label_features([det])
    assert features[0].long_side / features[0].short_side < 2.0  # confirms square-ish
    correct_square_feature_dirs(features, hints)
    assert abs(features[0].dir_pix - math.pi / 2) < 0.01


def test_correct_square_feature_too_far_unchanged():
    # Hint is within search_radius_px but on a different avenue — hint is > 200px away.
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    det = _make_det("X", cx=192, cy=300, long_side=24, short_side=24, dir_pix=0.0)
    features = label_features([det])
    original_dir = features[0].dir_pix
    correct_square_feature_dirs(features, hints, search_radius_px=50.0)
    assert features[0].dir_pix == original_dir


def test_correct_non_square_feature_unchanged():
    # Long, narrow feature (aspect ≫ 2) is not corrected even with a nearby hint.
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=300, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    det = _make_det("SEVENTH", cx=192, cy=300, long_side=80, short_side=18, dir_pix=0.0)
    features = label_features([det])
    original_dir = features[0].dir_pix
    correct_square_feature_dirs(features, hints)
    assert features[0].dir_pix == original_dir


# ---------------------------------------------------------------------------
# _cluster_geo_coords
# ---------------------------------------------------------------------------


def test_cluster_empty():
    assert _cluster_geo_coords([]) == []


def test_cluster_single_point():
    result = _cluster_geo_coords([(-90.0, 30.0)])
    assert result == [[(-90.0, 30.0)]]


def test_cluster_two_nearby_points():
    # Two points ~30 ft apart should merge into one cluster.
    deg_30ft = 30.0 / _FT_PER_DEG_LAT
    p1 = (-90.0, 30.0)
    p2 = (-90.0, 30.0 + deg_30ft)
    result = _cluster_geo_coords([p1, p2])
    assert len(result) == 1
    assert set(result[0]) == {p1, p2}


def test_cluster_two_far_points():
    # Two points ~200 ft apart should remain separate clusters.
    deg_200ft = 200.0 / _FT_PER_DEG_LAT
    p1 = (-90.0, 30.0)
    p2 = (-90.0, 30.0 + deg_200ft)
    result = _cluster_geo_coords([p1, p2])
    assert len(result) == 2


def test_cluster_jogging_street():
    # Simulate two intersections ~115 ft apart (the Newport St case): should be separate.
    deg_115ft = 115.0 / _FT_PER_DEG_LAT
    p1 = (-83.0, 42.35)
    p2 = (-83.0, 42.35 + deg_115ft)
    result = _cluster_geo_coords([p1, p2])
    assert len(result) == 2


def test_cluster_same_intersection_nearby_nodes():
    # Two nodes within the same OSM intersection (~10 ft) should merge.
    deg_10ft = 10.0 / _FT_PER_DEG_LAT
    p1 = (-90.0, 30.0)
    p2 = (-90.0, 30.0 + deg_10ft)
    result = _cluster_geo_coords([p1, p2])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# project_to_polyline — extrapolation cap
# ---------------------------------------------------------------------------


def _make_block(coords: list[tuple[float, float]]) -> Block:
    return Block(street_name="TEST STREET", coords=np.array(coords))


def test_project_no_extrapolation_clamps_to_endpoint():
    # A simple N-S segment from lat=30.0 to lat=30.001. Query point is past the end.
    block = _make_block([(-90.0, 30.0), (-90.0, 30.001)])
    result = project_to_polyline(-90.0, 30.002, [block], extrapolate=False)
    assert result is not None
    lon, lat, _ = result
    assert abs(lat - 30.001) < 1e-9  # clamped to the endpoint


def test_project_extrapolation_reaches_within_500ft():
    # Query point 300 ft past the terminal end — should snap to 300 ft out.
    deg_300ft = 300.0 / _FT_PER_DEG_LAT
    end_lat = 30.001
    block = _make_block([(-90.0, 30.0), (-90.0, end_lat)])
    result = project_to_polyline(-90.0, end_lat + deg_300ft, [block], extrapolate=True)
    assert result is not None
    lon, lat, _ = result
    assert abs(lat - (end_lat + deg_300ft)) < 1e-7  # projected to the query point


def test_project_extrapolation_capped_at_500ft():
    # Query point 800 ft past the terminal end — should be capped at 500 ft out.
    deg_500ft = 500.0 / _FT_PER_DEG_LAT
    deg_800ft = 800.0 / _FT_PER_DEG_LAT
    end_lat = 30.001
    block = _make_block([(-90.0, 30.0), (-90.0, end_lat)])
    result = project_to_polyline(-90.0, end_lat + deg_800ft, [block], extrapolate=True)
    assert result is not None
    lon, lat, _ = result
    # Should be at most 500 ft past the endpoint, not 800 ft.
    assert lat < end_lat + deg_800ft - 1e-8
    assert abs(lat - (end_lat + deg_500ft)) < 1e-7


def test_project_extrapolation_start_terminal_capped():
    # Query point before the start terminal — cap applies at start too.
    deg_500ft = 500.0 / _FT_PER_DEG_LAT
    deg_800ft = 800.0 / _FT_PER_DEG_LAT
    start_lat = 30.0
    block = _make_block([(-90.0, start_lat), (-90.0, 30.001)])
    result = project_to_polyline(
        -90.0, start_lat - deg_800ft, [block], extrapolate=True
    )
    assert result is not None
    lon, lat, _ = result
    assert lat > start_lat - deg_800ft + 1e-8
    assert abs(lat - (start_lat - deg_500ft)) < 1e-7


def test_project_returns_none_for_empty_blocks():
    assert project_to_polyline(-90.0, 30.0, []) is None
