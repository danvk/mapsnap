"""Tests for georef_from_labels helpers."""

import json
import math

import numpy as np

from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    Thresholds,
    _angle_diff_abs,
    _cluster_geo_coords,
    _rotation_from_neighbors,
    calibrate_thresholds,
    candidate_gcps_for_page,
    correct_square_feature_dirs,
    label_features,
    promote_avenue_letters,
    project_to_polyline,
    thresholds_at_admit_fraction,
)
from mapsnap.streets import Block


# ---------------------------------------------------------------------------
# promote_avenue_letters
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


def test_promote_correct_letter_wins_over_misread():
    # "W" and "M" (competing OCR readings of the same physical box) at essentially the
    # same position near an AVENUE hint. "W" is first in all_detections so it is
    # promoted first; "M" is then suppressed by center-based dedup (within 10px).
    streets = {"AVENUE W", "W", "AVENUE M", "M"}
    hints = [
        _make_det(
            "AVENUE", cx=1164, cy=1566, long_side=120, hint=True, dir_pix=math.pi / 2
        ),
    ]
    dets = [
        _make_det(
            "W", cx=1163, cy=268, long_side=28, short_side=26, dir_pix=math.pi / 2
        ),
        _make_det(
            "M", cx=1162, cy=270, long_side=28, short_side=26, dir_pix=math.pi / 2
        ),
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1
    assert result[0]["text"] == "W"
    assert result[0]["promoted"] is True


def test_promote_letter_in_column_with_avenue():
    # "W" as a regular detection in column with AVENUE hint → promoted.
    streets = {"AVENUE W", "W"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=1544, long_side=120, hint=True, dir_pix=math.pi / 2
        ),
    ]
    dets = [
        _make_det(
            "W", cx=192, cy=274, long_side=18, short_side=12, dir_pix=math.pi / 2
        ),
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1
    assert result[0]["text"] == "W"
    assert result[0]["promoted"] is True


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


# ---------------------------------------------------------------------------
# _angle_diff_abs
# ---------------------------------------------------------------------------


def test_angle_diff_abs_same():
    assert _angle_diff_abs(1.0, 1.0) == 0.0


def test_angle_diff_abs_opposite():
    assert abs(_angle_diff_abs(0.0, math.pi) - math.pi) < 1e-10


def test_angle_diff_abs_wraparound():
    # 10° and 350° are 20° apart
    assert (
        abs(_angle_diff_abs(math.radians(10), math.radians(350)) - math.radians(20))
        < 1e-10
    )


def test_angle_diff_abs_symmetric():
    a, b = math.radians(30), math.radians(200)
    assert abs(_angle_diff_abs(a, b) - _angle_diff_abs(b, a)) < 1e-10


# ---------------------------------------------------------------------------
# _rotation_from_neighbors
# ---------------------------------------------------------------------------

_PAGE_DIM = (0.01, 0.01)  # 0.01° in each direction


def _neighbors(angles_deg: list[float], center: tuple[float, float] = (0.0, 0.0)):
    return [(center, math.radians(a)) for a in angles_deg]


def test_rotation_from_neighbors_two_agree_candidate():
    # Two neighbours match the candidate (5°); confirmed, rotation ≈ 5°.
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=_neighbors([4, 6, 180 + 90]),
    )
    assert result.confirmed is True
    assert result.method == "neighbor_confirmed"
    assert abs(math.degrees(result.rotation) - 5) < 1
    assert result.n_agree == 2
    assert len(result.adjacent_deg) == 3


def test_rotation_from_neighbors_two_agree_flipped():
    # Two neighbours match candidate+180 (185°); confirmed, rotation ≈ 185°.
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=_neighbors([184, 186, 5]),
    )
    assert result.confirmed is True
    assert result.method == "neighbor_confirmed"
    assert abs(math.degrees(result.rotation) - 185) < 1


def test_rotation_from_neighbors_plurality():
    # Only 1 neighbour agrees with candidate — unconfirmed plurality.
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=_neighbors([4, 90, 135]),
    )
    assert result.confirmed is False
    assert result.method == "neighbor_plurality"
    assert abs(math.degrees(result.rotation) - 5) < 1
    assert result.n_agree == 1
    assert result.n_other == 0


def test_rotation_from_neighbors_no_adjacent_north_up_fallback():
    # No adjacent neighbours → north-up fallback.
    far = [((-1.0, -1.0), math.radians(5)), ((-1.0, -1.0), math.radians(6))]
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=far,
    )
    assert result.confirmed is False
    assert result.method == "north_up_fallback"
    assert result.n_agree == 0
    assert result.n_other == 0
    assert result.adjacent_deg == []


def test_rotation_from_neighbors_tie_uses_north_up():
    # Equal supporters (1 vs 1) → north-up fallback, not plurality.
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=_neighbors([4, 184]),
    )
    assert result.confirmed is False
    assert result.method == "north_up_fallback"


def test_rotation_from_neighbors_adjacency_boundary():
    # Neighbours exactly at 1.5 × page_dim should be included and confirm.
    on_edge = [
        ((1.5 * _PAGE_DIM[0], 1.5 * _PAGE_DIM[1]), math.radians(a)) for a in [4, 6]
    ]
    result = _rotation_from_neighbors(
        candidate=math.radians(5),
        approx_center=(0.0, 0.0),
        page_dim_deg=_PAGE_DIM,
        neighbor_rotations=on_edge,
    )
    assert result.confirmed is True
    assert result.n_agree == 2


# ---------------------------------------------------------------------------
# thresholds_at_admit_fraction
# ---------------------------------------------------------------------------


def test_thresholds_at_admit_fraction_independent_percentiles():
    confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
    long_sides = [10.0, 20.0, 30.0, 40.0, 50.0]
    short_sides = [1.0, 2.0, 3.0, 4.0, 5.0]
    # 80% admit fraction should keep roughly the top 80%, i.e. drop the single
    # smallest value from each independently-sorted list.
    result = thresholds_at_admit_fraction(confidences, long_sides, short_sides, 80)
    assert result.min_confidence == 0.2
    assert result.min_long_side == 20.0
    assert result.min_short_side == 2.0


def test_thresholds_at_admit_fraction_full_admit():
    result = thresholds_at_admit_fraction([0.1, 0.5], [10.0, 50.0], [1.0, 5.0], 100)
    assert result.min_confidence == 0.1
    assert result.min_long_side == 10.0
    assert result.min_short_side == 1.0


def test_thresholds_at_admit_fraction_empty_lists():
    result = thresholds_at_admit_fraction([], [], [], 50)
    assert result == Thresholds(
        min_confidence=0.0, min_long_side=0.0, min_short_side=0.0
    )


# ---------------------------------------------------------------------------
# candidate_gcps_for_page
# ---------------------------------------------------------------------------


def _make_vertical_det(
    text: str,
    cx: float,
    cy: float,
    long_side: float = 60.0,
    short_side: float = 18.0,
    confidence: float = 0.9,
) -> dict:
    """Build a mock detection dict with a vertical (tall) bounding polygon."""
    half_long = long_side / 2.0
    half_short = short_side / 2.0
    return {
        "polygon": [
            [int(cx - half_short), int(cy - half_long)],
            [int(cx + half_short), int(cy - half_long)],
            [int(cx + half_short), int(cy + half_long)],
            [int(cx - half_short), int(cy + half_long)],
        ],
        "text": text,
        "confidence": confidence,
        "angle": 90,
        "long_side": long_side,
        "short_side": short_side,
        "dir_pix": math.pi / 2,
    }


def test_candidate_gcps_for_page_resolves_ambiguous_canonical_name(tmp_path):
    # Regression test for a bug where the simulation used to validate this feature
    # skipped canonical_street_matches: a bare label like "CHURCH" can be ambiguous
    # (matching both "EAST CHURCH STREET" and "WEST CHURCH STREET"), but should still
    # resolve to whichever one actually shares a coordinate with another detected
    # street, rather than being silently dropped because "CHURCH" itself isn't a key.
    shared_point = (-90.0, 30.0)
    block_index = {
        "MAIN": [
            Block(street_name="MAIN", coords=np.array([shared_point, (-90.0, 30.001)]))
        ],
        "EAST CHURCH STREET": [
            Block(
                street_name="EAST CHURCH STREET",
                coords=np.array([shared_point, (-90.001, 30.0)]),
            )
        ],
        "WEST CHURCH STREET": [
            Block(
                street_name="WEST CHURCH STREET",
                coords=np.array([(-89.0, 29.0), (-89.001, 29.0)]),
            )
        ],
    }
    normalized_streets = set(block_index.keys())

    labels_path = tmp_path / "p1.streets.json"
    labels_path.write_text(
        json.dumps(
            {
                "streets": [
                    _make_det("MAIN", cx=100, cy=100),
                    _make_vertical_det("CHURCH", cx=100, cy=200),
                ]
            }
        )
    )

    _features, gcps = candidate_gcps_for_page(
        str(labels_path),
        Thresholds(min_confidence=0.15, min_long_side=40.0, min_short_side=10.0),
        min_aspect_ratio=1.75,
        block_index=block_index,
        normalized_streets=normalized_streets,
    )
    assert len(gcps) == 1
    assert {gcps[0].label_a, gcps[0].label_b} == {"MAIN", "EAST CHURCH STREET"}


def test_candidate_gcps_for_page_no_match_below_threshold(tmp_path):
    block_index = {
        "MAIN": [
            Block(street_name="MAIN", coords=np.array([(-90.0, 30.0), (-90.0, 30.001)]))
        ],
    }
    labels_path = tmp_path / "p1.streets.json"
    labels_path.write_text(
        json.dumps({"streets": [_make_det("MAIN", cx=100, cy=100, confidence=0.05)]})
    )
    _features, gcps = candidate_gcps_for_page(
        str(labels_path),
        Thresholds(min_confidence=0.15, min_long_side=40.0, min_short_side=10.0),
        min_aspect_ratio=1.75,
        block_index=block_index,
        normalized_streets=set(block_index.keys()),
    )
    assert gcps == []


# ---------------------------------------------------------------------------
# calibrate_thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = Thresholds(
    min_confidence=0.15, min_long_side=40.0, min_short_side=20.0
)


def _write_page(
    tmp_path, name: str, main_confidence: float, church_confidence: float
) -> str:
    """Write a <name>.streets.json with a MAIN/CHURCH intersection pair and return
    the (non-existent, but path-derivable) image path for it."""
    labels_path = tmp_path / f"{name}.streets.json"
    labels_path.write_text(
        json.dumps(
            {
                "streets": [
                    _make_det(
                        "MAIN",
                        cx=100,
                        cy=100,
                        short_side=22.0,
                        confidence=main_confidence,
                    ),
                    _make_vertical_det(
                        "CHURCH",
                        cx=100,
                        cy=200,
                        short_side=22.0,
                        confidence=church_confidence,
                    ),
                ]
            }
        )
    )
    return str(tmp_path / f"{name}.jpg")


def _church_block_index() -> dict[str, list[Block]]:
    shared_point = (-90.0, 30.0)
    return {
        "MAIN": [
            Block(street_name="MAIN", coords=np.array([shared_point, (-90.0, 30.001)]))
        ],
        "EAST CHURCH STREET": [
            Block(
                street_name="EAST CHURCH STREET",
                coords=np.array([shared_point, (-90.001, 30.0)]),
            )
        ],
    }


def test_calibrate_thresholds_keeps_default_when_already_good(tmp_path):
    # Every page already qualifies at the default thresholds (confidence 0.9), so
    # calibration must not loosen anything even though looser rungs would also work.
    # Each page has exactly one MAIN/CHURCH intersection pair, so min_gcps=1 here.
    images = [
        _write_page(tmp_path, f"p{i}", main_confidence=0.9, church_confidence=0.9)
        for i in range(4)
    ]
    result = calibrate_thresholds(
        images,
        _church_block_index(),
        min_aspect_ratio=1.75,
        default_thresholds=_DEFAULT_THRESHOLDS,
        min_gcps=1,
    )
    assert result == _DEFAULT_THRESHOLDS


def test_calibrate_thresholds_relaxes_for_low_confidence_volume(tmp_path):
    # Every page has confidence 0.05, well below the 0.15 default, so 0 pages qualify
    # at default but all pages qualify once thresholds relax to admit them.
    images = [
        _write_page(tmp_path, f"p{i}", main_confidence=0.05, church_confidence=0.05)
        for i in range(4)
    ]
    result = calibrate_thresholds(
        images,
        _church_block_index(),
        min_aspect_ratio=1.75,
        default_thresholds=_DEFAULT_THRESHOLDS,
        min_gcps=1,
    )
    assert result.min_confidence <= 0.05


def test_calibrate_thresholds_all_zero_gcps_returns_default(tmp_path):
    # No page can ever produce a GCP (block_index has no shared coordinates), so no
    # rung helps; calibration should fall back to the default rather than erroring.
    block_index = {
        "MAIN": [
            Block(street_name="MAIN", coords=np.array([(-90.0, 30.0), (-90.0, 30.001)]))
        ],
        "EAST CHURCH STREET": [
            Block(
                street_name="EAST CHURCH STREET",
                coords=np.array([(-80.0, 20.0), (-80.001, 20.0)]),
            )
        ],
    }
    images = [
        _write_page(tmp_path, f"p{i}", main_confidence=0.9, church_confidence=0.9)
        for i in range(3)
    ]
    result = calibrate_thresholds(
        images,
        block_index,
        min_aspect_ratio=1.75,
        default_thresholds=_DEFAULT_THRESHOLDS,
        min_gcps=2,
    )
    assert result == _DEFAULT_THRESHOLDS
