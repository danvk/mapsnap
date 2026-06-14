"""Tests for georef_from_labels helpers."""

import math

import numpy as np

import mapsnap.georef_from_labels as gfl
from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    AcceptanceRef,
    PageFit,
    ProcessResult,
    Thresholds,
    _angle_diff_abs,
    _cluster_geo_coords,
    _rotation_from_neighbors,
    build_baseline_ladder,
    consistent_reference_fits,
    correct_square_feature_dirs,
    deg_per_px_to_px_per_ft,
    escalate_page,
    fit_is_neighbor_consistent,
    label_features,
    promote_avenue_letters,
    project_to_polyline,
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
# build_baseline_ladder
# ---------------------------------------------------------------------------


def test_build_baseline_ladder_keeps_all_by_default():
    # Default floors (matching the most permissive ladder row) keep every relaxation.
    strict = Thresholds(0.15, 45, 20)
    floors = Thresholds(0.07, 24, 11)
    ladder = build_baseline_ladder(strict, floors)
    assert ladder[0] == strict
    assert len(ladder) == 1 + len(gfl.RELAXATION_LADDER)


def test_build_baseline_ladder_floors_clamp():
    # Raising the floors drops every relaxation step, leaving only the strict baseline.
    strict = Thresholds(0.15, 45, 20)
    ladder = build_baseline_ladder(strict, Thresholds(0.3, 40, 20))
    assert ladder == [strict]


# ---------------------------------------------------------------------------
# fit_is_neighbor_consistent
# ---------------------------------------------------------------------------

_CENTER = (-90.0, 30.0)


def test_neighbor_consistent_in_band():
    assert fit_is_neighbor_consistent(10.0, _CENTER, 10.0, [_CENTER], 0.25, 1.5)


def test_neighbor_inconsistent_scale():
    # 50% larger scale than reference exceeds the 25% band.
    assert not fit_is_neighbor_consistent(15.0, _CENTER, 10.0, [_CENTER], 0.25, 1.5)


def test_neighbor_inconsistent_location():
    # ~900 km from the only reference center is far outside 1.5 km.
    far = (-80.0, 30.0)
    assert not fit_is_neighbor_consistent(10.0, far, 10.0, [_CENTER], 0.25, 1.5)


def test_neighbor_no_reference_scale_skips_scale_check():
    # With ref_scale=None, only the location check applies (and here it passes).
    assert fit_is_neighbor_consistent(999.0, _CENTER, None, [_CENTER], 0.25, 1.5)


def test_neighbor_empty_centers_skips_location_check():
    assert fit_is_neighbor_consistent(10.0, _CENTER, 10.0, [], 0.25, 1.5)


# ---------------------------------------------------------------------------
# consistent_reference_fits
# ---------------------------------------------------------------------------


def test_consistent_reference_drops_scale_outlier():
    fits = [
        PageFit("a", 10.0, _CENTER),
        PageFit("b", 10.0, _CENTER),
        PageFit("c", 10.0, _CENTER),
        PageFit("d", 20.0, _CENTER),  # 2× the median scale → dropped
    ]
    result = consistent_reference_fits(fits, 0.25, 1.5)
    assert {f.image_path for f in result} == {"a", "b", "c"}


def test_consistent_reference_picks_largest_location_cluster():
    near = _CENTER
    far = (-80.0, 30.0)  # ~900 km away
    fits = [
        PageFit("a", 10.0, near),
        PageFit("b", 10.0, near),
        PageFit("c", 10.0, near),
        PageFit("d", 10.0, far),
    ]
    result = consistent_reference_fits(fits, 0.25, 1.5)
    assert {f.image_path for f in result} == {"a", "b", "c"}


def test_consistent_reference_sparse_returns_subset():
    # A single fit is "consistent" with itself; the caller decides if it meets the minimum.
    result = consistent_reference_fits([PageFit("a", 10.0, _CENTER)], 0.25, 1.5)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# escalate_page
# ---------------------------------------------------------------------------

_SCALE_DEG_PER_PX = 1e-5
_REF_PX_PER_FT = deg_per_px_to_px_per_ft(_SCALE_DEG_PER_PX)


def _success(n_gcps: int, n_inliers: int, center=_CENTER) -> ProcessResult:
    return ProcessResult(
        success=True,
        scale_deg_per_px=_SCALE_DEG_PER_PX,
        center=center,
        rotation=0.0,
        n_gcps=n_gcps,
        n_inliers=n_inliers,
    )


_FAIL = ProcessResult(success=False, n_gcps=0)

# escalate_page only forwards config to the (patched-out) fit_image, so its contents are
# irrelevant in these tests; this placeholder just satisfies the type signature.
_DUMMY_CONFIG = gfl.GeorefConfig(
    block_index={},
    cos_phi=0.76,
    centerlines_path="",
    min_aspect_ratio=1.75,
    edge_margin=0.0,
    force_intersection=None,
    one_gcp_fits=True,
    debug=False,
)


def _reliable_acceptance(centers=None) -> AcceptanceRef:
    return AcceptanceRef(
        ref_scale_px_per_ft=_REF_PX_PER_FT,
        centers=[_CENTER] if centers is None else centers,
        reliable=True,
        scale_outlier_threshold=0.25,
        max_dist_km=1.5,
    )


def _patch_fit_image(monkeypatch, by_thresholds: dict[Thresholds, ProcessResult]):
    def fake(image_path, thresholds, config):
        return by_thresholds[thresholds]

    monkeypatch.setattr(gfl, "fit_image", fake)


def test_escalate_rescues_failed_page(monkeypatch):
    strict = Thresholds(0.15, 45, 20)
    step_a = Thresholds(0.10, 45, 20)
    step_b = Thresholds(0.07, 45, 20)
    ladder = [strict, step_a, step_b]
    _patch_fit_image(
        monkeypatch,
        {strict: _FAIL, step_a: _FAIL, step_b: _success(3, 5)},
    )
    result = escalate_page(
        "p1.jpg", _FAIL, 0, ladder, _DUMMY_CONFIG, _reliable_acceptance()
    )
    assert result.success
    assert result.n_gcps == 3


def test_escalate_improves_weak_page(monkeypatch):
    strict = Thresholds(0.15, 45, 20)
    step_a = Thresholds(0.10, 45, 20)
    ladder = [strict, step_a]
    weak = _success(2, 2)
    _patch_fit_image(monkeypatch, {strict: weak, step_a: _success(4, 6)})
    result = escalate_page(
        "p1.jpg", weak, 0, ladder, _DUMMY_CONFIG, _reliable_acceptance()
    )
    assert result.n_gcps == 4


def test_escalate_keeps_weak_when_relaxed_is_inconsistent(monkeypatch):
    strict = Thresholds(0.15, 45, 20)
    step_a = Thresholds(0.10, 45, 20)
    ladder = [strict, step_a]
    weak = _success(2, 2)
    far = _success(5, 8, center=(-80.0, 30.0))  # strong but ~900 km away → rejected
    _patch_fit_image(monkeypatch, {strict: weak, step_a: far})
    result = escalate_page(
        "p1.jpg", weak, 0, ladder, _DUMMY_CONFIG, _reliable_acceptance()
    )
    # Falls back to the strict baseline (re-fit at index 0).
    assert result.n_gcps == 2
    assert result.center == _CENTER


def test_escalate_internal_quality_fallback_when_unreliable(monkeypatch):
    # No trustworthy reference: a relaxed 2-GCP / 2-inlier fit is accepted on its own merits.
    strict = Thresholds(0.15, 45, 20)
    step_a = Thresholds(0.10, 45, 20)
    ladder = [strict, step_a]
    _patch_fit_image(monkeypatch, {strict: _FAIL, step_a: _success(2, 2)})
    unreliable = AcceptanceRef(None, [], False, 0.25, 1.5)
    result = escalate_page("p1.jpg", _FAIL, 0, ladder, _DUMMY_CONFIG, unreliable)
    assert result.success
    assert result.n_gcps == 2
