"""Tests for georef_from_labels helpers."""

import json
import math

import numpy as np

from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    _angle_diff_abs,
    _cluster_geo_coords,
    _distinct_pixel_gcps,
    _robust_affine_inlier_indices,
    _rotation_from_neighbors,
    assemble_multiword_streets,
    compute_auto_min_short_side,
    confidence_relaxed_threshold,
    correct_square_feature_dirs,
    dominant_axis_near,
    IntersectionGCP,
    is_rotation_outlier,
    label_features,
    LabelFeature,
    project_to_polyline,
    promote_avenue_letters,
    ransac_hybrid,
    straight_intersection_geo,
    street_segments_near_vertex,
)
from mapsnap.streets import Block


def _make_gcp(pixel: tuple[float, float], geo: tuple[float, float]) -> IntersectionGCP:
    """Minimal IntersectionGCP for grouping tests (only pixel/geo matter)."""
    feat = LabelFeature("A", "A", (0.0, 0.0), 0.0, 1.0, 1.0)
    return IntersectionGCP("A", "B", pixel, geo, 0.0, feat, feat)


def test_distinct_pixel_gcps_groups_coincident_pixels():
    # Two GCPs at the same pixel (a jog: same image crossing, two world points) -> one group.
    gcps = [
        _make_gcp((100.0, 200.0), (-83.0, 42.0)),
        _make_gcp((101.0, 201.0), (-83.0005, 42.0)),  # within 5px tolerance
    ]
    groups = _distinct_pixel_gcps(gcps)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_distinct_pixel_gcps_separates_distinct_pixels():
    # Two GCPs at far-apart pixels are independent control points -> two groups.
    gcps = [
        _make_gcp((100.0, 200.0), (-83.0, 42.0)),
        _make_gcp((900.0, 800.0), (-82.9, 42.1)),
    ]
    assert len(_distinct_pixel_gcps(gcps)) == 2


def test_distinct_pixel_gcps_mixed():
    # A jog pair at one pixel plus one distinct pixel -> two groups (RANSAC still has a pair).
    gcps = [
        _make_gcp((100.0, 200.0), (-83.0, 42.0)),
        _make_gcp((102.0, 200.0), (-83.0005, 42.0)),
        _make_gcp((900.0, 800.0), (-82.9, 42.1)),
    ]
    groups = _distinct_pixel_gcps(gcps)
    assert sorted(len(g) for g in groups) == [1, 2]


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
    """Build a mock detection whose polygon is oriented along ``dir_pix``.

    The first edge (polygon[0] → polygon[1]) points in the reading direction (+dir_pix), as a
    real detect_text polygon does, so reading_vector / the letter-ordering check behave as in
    production.
    """
    ux, uy = math.cos(dir_pix), math.sin(dir_pix)  # reading (long-axis) direction
    px, py = -uy, ux  # perpendicular
    hl, hs = long_side / 2.0, short_side / 2.0
    corners = [
        (cx - hl * ux - hs * px, cy - hl * uy - hs * py),  # start, near edge
        (cx + hl * ux - hs * px, cy + hl * uy - hs * py),  # end, near edge
        (cx + hl * ux + hs * px, cy + hl * uy + hs * py),  # end, far edge
        (cx - hl * ux + hs * px, cy - hl * uy + hs * py),  # start, far edge
    ]
    return {
        "polygon": [[round(x), round(y)] for x, y in corners],
        "text": text,
        "confidence": confidence,
        "angle": angle,
        "long_side": long_side,
        "short_side": short_side,
        "dir_pix": dir_pix,
        **({"hint": True} if hint else {}),
    }


# ---------------------------------------------------------------------------
# assemble_multiword_streets
# ---------------------------------------------------------------------------

_VAN_STREETS = {"VAN BRUNT STREET", "VAN BUREN STREET"}


def test_assemble_adjacent_words_into_street():
    # "VAN" and "BRUNT" detected as separate, collinear, adjacent boxes -> "VAN BRUNT".
    van = _make_det("VAN", cx=700, cy=1000, long_side=50, short_side=22)
    brunt = _make_det("BRUNT", cx=760, cy=1000, long_side=80, short_side=22)
    assembled, consumed = assemble_multiword_streets([van, brunt], _VAN_STREETS)
    assert len(assembled) == 1
    assert assembled[0]["text"] == "VAN BRUNT"
    assert assembled[0]["assembled"] is True
    assert (
        len(consumed) == 2
    )  # both parts consumed so a lone ambiguous "VAN" doesn't remain


def test_assemble_resolves_reading_order():
    # Spatial order doesn't matter: only "VAN BRUNT" matches a street, not "BRUNT VAN".
    brunt = _make_det("BRUNT", cx=700, cy=1000, long_side=80, short_side=22)
    van = _make_det("VAN", cx=770, cy=1000, long_side=50, short_side=22)
    assembled, _ = assemble_multiword_streets([brunt, van], _VAN_STREETS)
    assert len(assembled) == 1
    assert assembled[0]["text"] == "VAN BRUNT"


def test_assemble_skips_far_apart_words():
    van = _make_det("VAN", cx=700, cy=1000, long_side=50, short_side=22)
    brunt = _make_det("BRUNT", cx=1300, cy=1000, long_side=80, short_side=22)
    assembled, consumed = assemble_multiword_streets([van, brunt], _VAN_STREETS)
    assert assembled == [] and consumed == []


def test_assemble_skips_non_street_combination():
    # Adjacent words whose concatenation is not a street are left alone.
    van = _make_det("VAN", cx=700, cy=1000, long_side=50, short_side=22)
    other = _make_det("PARK", cx=760, cy=1000, long_side=70, short_side=22)
    assembled, _ = assemble_multiword_streets([van, other], _VAN_STREETS)
    assert assembled == []


def test_assemble_skips_different_orientation():
    # Words on perpendicular lines (different dir_pix bucket) are not combined.
    van = _make_det("VAN", cx=700, cy=1000, long_side=50, short_side=22, dir_pix=0.0)
    brunt = _make_det(
        "BRUNT", cx=760, cy=1000, long_side=80, short_side=22, dir_pix=math.pi / 2
    )
    assembled, _ = assemble_multiword_streets([van, brunt], _VAN_STREETS)
    assert assembled == []


# ---------------------------------------------------------------------------
# promote_avenue_letters
# ---------------------------------------------------------------------------


def test_promote_single_char_near_avenue_hint():
    # "X" in the same vertical column as an "AVENUE" hint → promoted with corrected dir_pix.
    # Both share dir_pix=π/2 (vertical, reading downward); for "AVENUE X" the AVENUE hint must
    # precede the letter, so the hint is above (smaller cy) and "X" below.
    streets = {"AVENUE X", "X"}
    hints = [
        _make_det(
            "AVENUE", cx=193, cy=274, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det(
            "X", cx=192, cy=1544, long_side=24, short_side=24, dir_pix=math.pi / 2
        )
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
            "AVENUE", cx=193, cy=274, long_side=120, hint=True, dir_pix=math.pi / 2
        )
    ]
    dets = [
        _make_det(
            "Q", cx=192, cy=1544, long_side=24, short_side=24, dir_pix=math.pi / 2
        )
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
            "AVENUE", cx=1164, cy=268, long_side=120, hint=True, dir_pix=math.pi / 2
        ),
    ]
    dets = [
        _make_det(
            "W", cx=1163, cy=1566, long_side=28, short_side=26, dir_pix=math.pi / 2
        ),
        _make_det(
            "M", cx=1162, cy=1568, long_side=28, short_side=26, dir_pix=math.pi / 2
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
            "AVENUE", cx=193, cy=274, long_side=120, hint=True, dir_pix=math.pi / 2
        ),
    ]
    dets = [
        _make_det(
            "W", cx=192, cy=1544, long_side=18, short_side=12, dir_pix=math.pi / 2
        ),
    ]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1
    assert result[0]["text"] == "W"
    assert result[0]["promoted"] is True


def test_promote_street_letter_must_follow_hint():
    # "K STREET": reading left→right, the "ST" hint must come AFTER the K. K left, ST right → OK.
    streets = {"K STREET", "K"}
    hints = [_make_det("ST", cx=200, cy=300, hint=True, dir_pix=0.0)]
    dets = [_make_det("K", cx=100, cy=300, long_side=24, short_side=24, dir_pix=0.0)]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1 and result[0]["text"] == "K"


def test_promote_street_letter_before_hint_rejected():
    # K to the RIGHT of "ST" (reads "ST K") → wrong order for "K STREET" → not promoted.
    streets = {"K STREET", "K"}
    hints = [_make_det("ST", cx=100, cy=300, hint=True, dir_pix=0.0)]
    dets = [_make_det("K", cx=200, cy=300, long_side=24, short_side=24, dir_pix=0.0)]
    assert promote_avenue_letters(hints, dets, streets) == []


def test_promote_avenue_letter_must_follow_hint():
    # "AVENUE Q": the "AV" hint must come BEFORE the Q. AV left, Q right → OK.
    streets = {"AVENUE Q", "Q"}
    hints = [_make_det("AV", cx=100, cy=300, hint=True, dir_pix=0.0)]
    dets = [_make_det("Q", cx=200, cy=300, long_side=24, short_side=24, dir_pix=0.0)]
    result = promote_avenue_letters(hints, dets, streets)
    assert len(result) == 1 and result[0]["text"] == "Q"


def test_promote_avenue_letter_after_hint_rejected():
    # Q to the LEFT of "AV" (reads "Q AV") → wrong order for "AVENUE Q" → not promoted.
    streets = {"AVENUE Q", "Q"}
    hints = [_make_det("AV", cx=200, cy=300, hint=True, dir_pix=0.0)]
    dets = [_make_det("Q", cx=100, cy=300, long_side=24, short_side=24, dir_pix=0.0)]
    assert promote_avenue_letters(hints, dets, streets) == []


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
# compute_auto_min_short_side
# ---------------------------------------------------------------------------


def _write_streets_json(path, detections: list[dict]) -> None:
    """Write a minimal <stem>.streets.json fixture file."""
    path.write_text(json.dumps({"streets": detections}))


def test_compute_auto_min_short_side_basic_percentile(tmp_path):
    dets = [_make_det(f"MAIN ST {i}", 0, 0, short_side=float(i)) for i in range(1, 101)]
    _write_streets_json(tmp_path / "p1.streets.json", dets)
    image_path = str(tmp_path / "p1.2048px.jpg")

    result = compute_auto_min_short_side(
        [image_path], min_confidence=0.3, percentile=25
    )
    assert result == 25.75


def test_compute_auto_min_short_side_excludes_hints_and_low_confidence(tmp_path):
    dets = [
        _make_det("MAIN STREET", 0, 0, short_side=10.0, confidence=0.9),
        _make_det("LOW CONFIDENCE", 0, 0, short_side=1000.0, confidence=0.1),
        _make_det("N", 0, 0, short_side=1000.0, confidence=0.9, hint=True),
        _make_det("A", 0, 0, short_side=1000.0, confidence=0.9),  # < 2 letters
    ]
    _write_streets_json(tmp_path / "p1.streets.json", dets)
    image_path = str(tmp_path / "p1.2048px.jpg")

    result = compute_auto_min_short_side(
        [image_path], min_confidence=0.3, percentile=25
    )
    assert result == 10.0


def test_compute_auto_min_short_side_include_hints(tmp_path):
    dets = [
        _make_det("MAIN STREET", 0, 0, short_side=10.0, confidence=0.9),
        _make_det("EAST", 0, 0, short_side=1000.0, confidence=0.9, hint=True),
    ]
    _write_streets_json(tmp_path / "p1.streets.json", dets)
    image_path = str(tmp_path / "p1.2048px.jpg")

    result = compute_auto_min_short_side(
        [image_path], min_confidence=0.3, percentile=25, include_hints=True
    )
    assert result == 257.5


def test_compute_auto_min_short_side_no_qualifying_detections(tmp_path):
    dets = [_make_det("A", 0, 0, confidence=0.9)]  # < 2 letters
    _write_streets_json(tmp_path / "p1.streets.json", dets)
    image_path = str(tmp_path / "p1.2048px.jpg")

    result = compute_auto_min_short_side(
        [image_path], min_confidence=0.3, percentile=25
    )
    assert result is None


def test_compute_auto_min_short_side_skips_missing_labels_file(tmp_path):
    image_path = str(tmp_path / "p1.2048px.jpg")  # no p1.streets.json written
    result = compute_auto_min_short_side(
        [image_path], min_confidence=0.3, percentile=25
    )
    assert result is None


# ---------------------------------------------------------------------------
# confidence_relaxed_threshold
# ---------------------------------------------------------------------------


def test_confidence_relaxed_threshold_at_min_confidence():
    # At min_confidence, the full base_threshold is required (no relaxation).
    result = confidence_relaxed_threshold(
        confidence=0.15,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=12.6,
    )
    assert result == 18.0


def test_confidence_relaxed_threshold_below_min_confidence():
    # Below min_confidence, still returns base_threshold (this detection wouldn't
    # pass the confidence check anyway).
    result = confidence_relaxed_threshold(
        confidence=0.1,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=12.6,
    )
    assert result == 18.0


def test_confidence_relaxed_threshold_at_max_confidence():
    # At confidence 1.0, the relaxed floor applies exactly.
    result = confidence_relaxed_threshold(
        confidence=1.0,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=12.6,
    )
    assert math.isclose(result, 12.6)


def test_confidence_relaxed_threshold_interpolates_between_endpoints():
    # Matches the schedule from issue #78: as confidence rises from min_confidence
    # towards 1.0, the required short side relaxes from 18px towards ~12.6px (0.7x).
    result_mid_low = confidence_relaxed_threshold(
        confidence=0.3,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=12.6,
    )
    result_mid_high = confidence_relaxed_threshold(
        confidence=0.6,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=12.6,
    )
    assert 12.6 < result_mid_high < result_mid_low < 18.0
    assert math.isclose(result_mid_low, 15.8, abs_tol=0.1)
    assert math.isclose(result_mid_high, 13.9, abs_tol=0.1)


def test_confidence_relaxed_threshold_disabled_when_floor_not_below_base():
    # high_confidence_floor >= base_threshold disables relaxation entirely.
    result = confidence_relaxed_threshold(
        confidence=1.0,
        min_confidence=0.15,
        base_threshold=18.0,
        high_confidence_floor=18.0,
    )
    assert result == 18.0


# ---------------------------------------------------------------------------
# straight-axis intersection correction (curved-junction GCPs)
# ---------------------------------------------------------------------------


def _ll_block(coords: list[tuple[float, float]]) -> Block:
    """A Block from (lon, lat) tuples."""
    return Block("S", np.array(coords, dtype=float))


# A vertical (north-south) street through the shared vertex V = (-90, 30).
_V = (-90.0, 30.0)
_VERTICAL = _ll_block([(-90.0, 29.997), (-90.0, 30.0), (-90.0, 30.003)])
# A street whose straight east-west axis runs ~110 ft north of V, dropping through a short
# elbow to reach V at its east end (a miniature of New Orleans p161's curving S. Claiborne).
_CURVED = [
    _ll_block(
        [
            (-90.0009, 30.0003),
            (-90.0007, 30.0003),
            (-90.0005, 30.0003),
            (-90.0003, 30.0003),
            (-90.0001, 30.0003),
        ]
    ),
    _ll_block([(-90.0001, 30.0003), (-90.00005, 30.00015), (-90.0, 30.0)]),
]


def test_dominant_axis_near_straight_street_returns_its_bearing():
    horizontal = _ll_block([(-90.002, 30.0), (-90.0, 30.0), (-89.998, 30.0)])
    anchor, bearing = dominant_axis_near(_V, [horizontal])
    assert abs(bearing % math.pi) < math.radians(1)  # east-west
    axis = dominant_axis_near(_V, [_VERTICAL])
    assert abs(axis[1] % math.pi - math.pi / 2) < math.radians(1)  # north-south


def test_dominant_axis_near_ignores_the_curved_elbow():
    # The straight east-west run dominates; the short steep elbow is filtered out, so the
    # bearing stays near horizontal despite the elbow dipping to V.
    _, bearing = dominant_axis_near(_V, _CURVED)
    off_horizontal = min(bearing % math.pi, math.pi - bearing % math.pi)
    assert off_horizontal < math.radians(15)


# A street that forks past node F=(-89.9994, 30.0): the main branch into V runs east-west,
# but two arms beyond F bend northeast (a miniature of Brooklyn p14's two COLUMBIA branches).
_FORK_MAIN = _ll_block([(-90.0, 30.0), (-89.9997, 30.0), (-89.9994, 30.0)])
_FORK_ARMS = [
    _ll_block([(-89.9994, 30.0), (-89.9992, 30.0004), (-89.999, 30.0008)]),
    _ll_block([(-89.9994, 30.0), (-89.9993, 30.0003), (-89.9992, 30.0006)]),
]


def test_street_segments_near_vertex_stops_at_self_intersection():
    # Walking from V stops at the fork F, keeping only the main branch (2 segments), not the
    # forked-off arms — even though the arms have segments within radius.
    blocks = [_FORK_MAIN, *_FORK_ARMS]
    assert len(street_segments_near_vertex(_V, blocks, 350.0)) == 2


def test_dominant_axis_near_ignores_forked_off_branch():
    # With the arms excluded the bearing follows the main branch (~0 deg, east-west); including
    # them would drag it toward the northeast arms (~64 deg).
    _, bearing = dominant_axis_near(_V, [_FORK_MAIN, *_FORK_ARMS])
    off_horizontal = min(bearing % math.pi, math.pi - bearing % math.pi)
    assert off_horizontal < math.radians(5)


def test_street_segments_near_vertex_no_fork_keeps_all_in_radius():
    # Without a self-intersection every within-radius segment is kept (unchanged behavior).
    straight = _ll_block([(-90.001, 30.0), (-90.0, 30.0), (-89.999, 30.0)])
    assert len(street_segments_near_vertex(_V, [straight], 350.0)) == 2


def test_straight_intersection_geo_none_for_straight_junction():
    # Two straight streets crossing squarely at V: the straight-axis crossing is V itself, so
    # the shift is below the deadband and no correction is made.
    horizontal = _ll_block([(-90.001, 30.0), (-90.0, 30.0), (-89.999, 30.0)])
    assert straight_intersection_geo(_V, [_VERTICAL], [horizontal]) is None


def test_straight_intersection_geo_none_for_near_parallel():
    almost_parallel = _ll_block([(-90.0002, 29.998), (-90.0002, 30.002)])
    assert straight_intersection_geo(_V, [_VERTICAL], [almost_parallel]) is None


def test_straight_intersection_geo_corrects_curved_junction():
    # With a low floor the correction moves the vertex north toward the curved street's
    # straight axis (~lat 30.0003), off the on-curve OSM vertex at lat 30.0.
    corrected = straight_intersection_geo(_V, [_VERTICAL], _CURVED, min_shift_ft=20.0)
    assert corrected is not None
    lon, lat = corrected
    assert abs(lon - (-90.0)) < 1e-4  # stays on the vertical street
    assert 30.0001 < lat < 30.0004  # shifted north toward the straight axis


def test_straight_intersection_geo_default_ignores_mild_curve():
    # The synthetic junction's ~84 ft shift is below the default floor, so only sharp curves
    # are corrected: with defaults it is left at the raw vertex.
    assert straight_intersection_geo(_V, [_VERTICAL], _CURVED) is None


def test_straight_intersection_geo_respects_shift_bounds():
    # The curved junction is suppressed when its ~84 ft shift falls outside the given bounds.
    assert (
        straight_intersection_geo(
            _V, [_VERTICAL], _CURVED, min_shift_ft=20.0, max_shift_ft=50.0
        )
        is None
    )
    assert (
        straight_intersection_geo(_V, [_VERTICAL], _CURVED, min_shift_ft=100.0) is None
    )


# ---------------------------------------------------------------------------
# _kink_block (shared geometry helper)
# ---------------------------------------------------------------------------


def _kink_block() -> Block:
    """A long east-west run ending in a short, steep kink segment."""
    return Block(
        street_name="MAIN",
        coords=np.array(
            [
                [-89.991, 29.995],  # long horizontal run ...
                [-89.989, 29.995],  # ... ~190 m of it
                [-89.9889, 29.9952],  # short ~25 m kink, ~63° off horizontal
            ]
        ),
    )


# ---------------------------------------------------------------------------
# ransac_hybrid: reject fits whose seed street is a *rotation* outlier (the model's
# orientation contradicts a street it was built from), while keeping fits whose seed
# is only a *position* outlier (right bearing, off a since-shortened centerline)
# ---------------------------------------------------------------------------

# Shared frame for the tests below: an orientation-reversing similarity at 1e-5 deg/px,
# north-up, where px=500 maps to lon=-90.0 and py=200 to lat=30.0. Two GCPs at
# (500, 200)->(-90, 30) and (500, 700)->(-90, 29.995) recover it exactly.
_RH_COS_PHI = math.cos(math.radians(30.0))


def _vertical_block(name: str, lon: float) -> Block:
    """A north-south street segment at the given longitude."""
    return Block(name, np.array([[lon, 29.99], [lon, 30.01]]))


def _horizontal_block(name: str, lat: float) -> Block:
    """An east-west street segment at the given latitude."""
    return Block(name, np.array([[-90.02, lat], [-89.98, lat]]))


def _rh_feat(text: str, center: tuple[float, float], dir_pix: float) -> LabelFeature:
    """A label feature for the ransac_hybrid frame (sizes are immaterial here)."""
    return LabelFeature(text, text, center, dir_pix, 100.0, 20.0)


def test_ransac_hybrid_rejects_rotation_outlier_seed():
    # Both GCPs are seeded by COREY (cf. Detroit p28). COREY's label maps to a bearing ~90°
    # off its (vertical) street, so it is a rotation outlier: the model's orientation
    # contradicts a street it was built from. The only candidate pair is disqualified and no
    # model is returned.
    kerch = _rh_feat("KERCH", (500.0, 200.0), 0.0)
    jeff = _rh_feat("JEFF", (500.0, 700.0), 0.0)
    corey = _rh_feat(
        "COREY", (500.0, 450.0), 0.0
    )  # horizontal label on a vertical street
    features = [kerch, jeff, corey]
    block_index = {
        "KERCH": [_horizontal_block("KERCH", 30.0)],
        "JEFF": [_horizontal_block("JEFF", 29.995)],
        "COREY": [_vertical_block("COREY", -90.0)],
    }
    gcps = [
        IntersectionGCP(
            "KERCH", "COREY", (500.0, 200.0), (-90.0, 30.0), 0.0, kerch, corey
        ),
        IntersectionGCP(
            "JEFF", "COREY", (500.0, 700.0), (-90.0, 29.995), 0.0, jeff, corey
        ),
    ]
    model, inliers, pair = ransac_hybrid(gcps, features, block_index, _RH_COS_PHI)
    assert model is None
    assert inliers == []
    assert pair is None


def test_ransac_hybrid_keeps_position_only_outlier_seed():
    # Same COREY-seeded frame, but COREY's label keeps its street's (vertical) bearing and is
    # merely displaced ~4300 ft east of the centerline -- a position-only outlier, as when a
    # historical street ran past where today's OSM line ends (cf. Brooklyn p7's VAN BRUNT).
    # The pair is kept even though COREY is not a position inlier.
    kerch = _rh_feat("KERCH", (500.0, 200.0), 0.0)
    jeff = _rh_feat("JEFF", (500.0, 700.0), 0.0)
    corey = _rh_feat(
        "COREY", (1500.0, 450.0), math.pi / 2
    )  # right bearing, off the line
    features = [kerch, jeff, corey]
    block_index = {
        "KERCH": [_horizontal_block("KERCH", 30.0)],
        "JEFF": [_horizontal_block("JEFF", 29.995)],
        "COREY": [_vertical_block("COREY", -90.0)],
    }
    gcps = [
        IntersectionGCP(
            "KERCH", "COREY", (500.0, 200.0), (-90.0, 30.0), 0.0, kerch, corey
        ),
        IntersectionGCP(
            "JEFF", "COREY", (500.0, 700.0), (-90.0, 29.995), 0.0, jeff, corey
        ),
    ]
    model, inliers, pair = ransac_hybrid(gcps, features, block_index, _RH_COS_PHI)
    assert model is not None
    assert set(inliers) == {0, 1}  # COREY is a position outlier, but the pair survives
    assert pair == (0, 1)


def test_ransac_hybrid_keeps_fit_when_seed_labels_are_inliers():
    # Same frame, but COREY's label now sits on its own centerline, so every seed label is an
    # inlier and the pair is accepted.
    kerch = _rh_feat("KERCH", (500.0, 200.0), 0.0)
    jeff = _rh_feat("JEFF", (500.0, 700.0), 0.0)
    corey = _rh_feat("COREY", (500.0, 450.0), math.pi / 2)  # on COREY's line -> inlier
    features = [kerch, jeff, corey]
    block_index = {
        "KERCH": [_horizontal_block("KERCH", 30.0)],
        "JEFF": [_horizontal_block("JEFF", 29.995)],
        "COREY": [_vertical_block("COREY", -90.0)],
    }
    gcps = [
        IntersectionGCP(
            "KERCH", "COREY", (500.0, 200.0), (-90.0, 30.0), 0.0, kerch, corey
        ),
        IntersectionGCP(
            "JEFF", "COREY", (500.0, 700.0), (-90.0, 29.995), 0.0, jeff, corey
        ),
    ]
    model, inliers, pair = ransac_hybrid(gcps, features, block_index, _RH_COS_PHI)
    assert model is not None
    assert set(inliers) == {0, 1, 2}
    assert pair == (0, 1)


def test_ransac_hybrid_accepts_ambiguous_label_seeding_two_streets():
    # A single physical "KORTE" label (one center) is ambiguous between KORTE STREET and
    # KORTE AVENUE and legitimately seeds a GCP under each name (cf. Detroit p94). Only the
    # real match, KORTE STREET, is an inlier; label_inliers dedups the center to it. Because
    # the seed check is by center, the pair is accepted -- a name-based check would wrongly
    # reject it, faulting the dropped KORTE AVENUE expansion as an outlier.
    drexel = _rh_feat("DREXEL", (500.0, 200.0), 0.0)
    piper = _rh_feat("PIPER", (500.0, 700.0), 0.0)
    korte_center = (500.0, 450.0)
    korte_st = _rh_feat("KORTE STREET", korte_center, math.pi / 2)  # on its line
    korte_ave = _rh_feat("KORTE AVENUE", korte_center, math.pi / 2)  # wrong street
    features = [drexel, piper, korte_st, korte_ave]
    block_index = {
        "DREXEL": [_horizontal_block("DREXEL", 30.0)],
        "PIPER": [_horizontal_block("PIPER", 29.995)],
        "KORTE STREET": [_vertical_block("KORTE STREET", -90.0)],
        "KORTE AVENUE": [_vertical_block("KORTE AVENUE", -89.0)],
    }
    gcps = [
        IntersectionGCP(
            "DREXEL",
            "KORTE AVENUE",
            (500.0, 200.0),
            (-90.0, 30.0),
            0.0,
            drexel,
            korte_ave,
        ),
        IntersectionGCP(
            "PIPER",
            "KORTE STREET",
            (500.0, 700.0),
            (-90.0, 29.995),
            0.0,
            piper,
            korte_st,
        ),
    ]
    model, inliers, pair = ransac_hybrid(gcps, features, block_index, _RH_COS_PHI)
    assert model is not None
    assert set(inliers) == {0, 1, 2}  # korte_ave (idx 3) deduped out by center
    assert pair == (0, 1)


# is_rotation_outlier ------------------------------------------------------

# Axis-aligned similarity: pixel (x, y) -> (lon, lat) = (-90 + 1e-5·x, 30 - 1e-5·y). A
# horizontal pixel label (dir_pix=0) maps to an east-west bearing, a vertical one (π/2) to
# north-south.
_AXIS_AFFINE = np.array([[1e-5, 0.0, -90.0], [0.0, -1e-5, 30.0]])


def test_is_rotation_outlier_true_when_perpendicular():
    # A horizontal label on a vertical (north-south) street is ~90° off its bearing.
    feat = _rh_feat("MAIN", (500.0, 500.0), 0.0)
    block_index = {"MAIN": [_vertical_block("MAIN", -89.995)]}
    assert is_rotation_outlier(feat, block_index, _AXIS_AFFINE)


def test_is_rotation_outlier_false_when_aligned_even_if_off_position():
    # A vertical label far (4300 ft) east of a vertical street: wrong position, right bearing.
    feat = _rh_feat("MAIN", (1500.0, 500.0), math.pi / 2)
    block_index = {"MAIN": [_vertical_block("MAIN", -89.995)]}
    assert not is_rotation_outlier(feat, block_index, _AXIS_AFFINE)


def test_is_rotation_outlier_false_when_no_street():
    feat = _rh_feat("MISSING", (500.0, 500.0), 0.0)
    assert not is_rotation_outlier(feat, {}, _AXIS_AFFINE)


def test_is_rotation_outlier_uses_dominant_bearing_not_kink():
    # The label runs along the street's long east-west body; the nearest segment is a short
    # steep kink. The dominant bearing keeps it an inlier, so it is not a rotation outlier.
    feat = LabelFeature("MAIN", "MAIN", (1110.0, 480.0), 0.0, 100.0, 20.0)
    block_index = {"MAIN": [_kink_block()]}
    affine = np.array([[1e-5, 0.0, -90.0], [0.0, -1e-5, 30.0]])
    assert not is_rotation_outlier(feat, block_index, affine)


# ---------------------------------------------------------------------------
# _robust_affine_inlier_indices — many-GCP pre-filter
# ---------------------------------------------------------------------------


def test_robust_affine_inlier_indices_drops_gross_outliers():
    # 120 GCPs consistent with a known affine (pixel → lon/lat) plus 8 gross outliers.
    # The robust affine RANSAC should keep the inliers and reject the outliers.
    import numpy as np

    rng = np.random.default_rng(0)
    inliers = []
    for _ in range(120):
        px, py = rng.uniform(0, 4000), rng.uniform(0, 4000)
        lon = -90.0 + 1e-4 * px
        lat = 30.0 - 1e-4 * py
        inliers.append(_make_gcp((px, py), (lon, lat)))
    outliers = [
        _make_gcp((rng.uniform(0, 4000), rng.uniform(0, 4000)), (-95.0, 35.0))
        for _ in range(8)
    ]
    gcps = inliers + outliers
    outlier_indices = set(range(120, 128))

    kept = set(_robust_affine_inlier_indices(gcps))
    # No gross outlier survives, and the large majority of true inliers are kept.
    assert not (kept & outlier_indices)
    assert len(kept & set(range(120))) >= 110
