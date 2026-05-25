"""Tests for georef_from_labels helpers."""

import numpy as np

from mapsnap.georef_from_labels import (
    _FT_PER_DEG_LAT,
    _cluster_geo_coords,
    project_to_polyline,
)
from mapsnap.streets import Block


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
