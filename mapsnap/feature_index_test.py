"""Tests for the GeoJSON feature spatial index.

The index only ever *culls*, so the property that matters is that it never drops
a feature an exhaustive scan would have kept — every test below compares against
the brute-force answer rather than asserting an exact hit count.
"""

import math

from mapsnap.feature_index import (
    M_PER_DEG_LAT,
    Bounds,
    FeatureIndex,
    geometry_bbox,
    point_bounds,
)

LON0, LAT0 = -74.0, 40.0
KX = 111_320.0 * math.cos(math.radians(LAT0))


def line(name: str, points: list[tuple[float, float]]) -> dict:
    """A LineString feature from (lon, lat) degree offsets about the test origin."""
    return {
        "properties": {"street_name": name},
        "geometry": {
            "type": "LineString",
            "coordinates": [[LON0 + dx, LAT0 + dy] for dx, dy in points],
        },
    }


def scan_bbox(features: list[dict], bounds: Bounds) -> list[dict]:
    """Brute-force: features with any vertex inside the lon/lat box."""
    min_lon, max_lon, min_lat, max_lat = bounds
    return [
        feature
        for feature in features
        if any(
            min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
            for lon, lat in feature["geometry"]["coordinates"]
        )
    ]


def test_geometry_bbox_covers_every_geometry_kind() -> None:
    assert geometry_bbox({"type": "Point", "coordinates": [1.0, 2.0]}) == (
        1.0,
        1.0,
        2.0,
        2.0,
    )
    assert geometry_bbox(
        {"type": "LineString", "coordinates": [[1.0, 2.0], [-1.0, 5.0]]}
    ) == (-1.0, 1.0, 2.0, 5.0)
    assert geometry_bbox(
        {
            "type": "MultiLineString",
            "coordinates": [[[1.0, 2.0], [3.0, 2.0]], [[0.0, 9.0], [0.0, 8.0]]],
        }
    ) == (0.0, 3.0, 2.0, 9.0)
    assert geometry_bbox(
        {"type": "Polygon", "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]]]}
    ) == (0.0, 2.0, 0.0, 3.0)
    assert geometry_bbox({"type": "LineString", "coordinates": []}) is None
    assert geometry_bbox({}) is None


def test_near_bbox_matches_a_full_scan() -> None:
    features = [
        line("A", [(0.00, 0.00), (0.00, 0.02)]),
        line("B", [(0.01, 0.01), (0.02, 0.01)]),
        line("C", [(0.50, 0.50), (0.51, 0.51)]),
    ]
    index = FeatureIndex(features)
    bounds = (LON0 - 0.005, LON0 + 0.005, LAT0 - 0.005, LAT0 + 0.005)
    assert index.near_bbox(bounds) == scan_bbox(features, bounds)
    far = (LON0 + 10.0, LON0 + 11.0, LAT0, LAT0 + 1.0)
    assert index.near_bbox(far) == []


def test_near_bbox_keeps_a_long_line_with_no_vertex_inside() -> None:
    """A street crossing the box without a vertex in it must survive the cull."""
    through = line("THROUGH", [(-1.0, 0.0), (1.0, 0.0)])
    index = FeatureIndex([through])
    bounds = (LON0 - 0.001, LON0 + 0.001, LAT0 - 0.001, LAT0 + 0.001)
    assert index.near_bbox(bounds) == [through]
    assert scan_bbox([through], bounds) == []  # the exact vertex test would miss it


def test_results_come_back_in_feature_order() -> None:
    features = [line(str(i), [(0.0, 0.0), (0.001, 0.001)]) for i in range(5)]
    index = FeatureIndex(features)
    hit = index.near_bbox((LON0 - 1, LON0 + 1, LAT0 - 1, LAT0 + 1))
    assert [f["properties"]["street_name"] for f in hit] == ["0", "1", "2", "3", "4"]


def test_features_without_geometry_are_skipped() -> None:
    features = [
        {"properties": {"street_name": "NO GEOM"}},
        line("REAL", [(0.0, 0.0), (0.001, 0.0)]),
    ]
    index = FeatureIndex(features)
    hit = index.near_bbox((LON0 - 1, LON0 + 1, LAT0 - 1, LAT0 + 1))
    assert [f["properties"]["street_name"] for f in hit] == ["REAL"]


def test_empty_index_answers_without_error() -> None:
    assert FeatureIndex([]).near_bbox((0.0, 1.0, 0.0, 1.0)) == []


def test_point_bounds_reaches_the_radius_in_both_axes() -> None:
    min_lon, max_lon, min_lat, max_lat = point_bounds((LON0, LAT0), 100.0)
    assert abs((max_lat - LAT0) * M_PER_DEG_LAT - 100.0) < 1e-6
    assert abs((max_lon - LON0) * KX - 100.0) < 1e-6
    # Longitude degrees are worth less than latitude degrees away from the
    # equator, so the box must be wider than it is tall.
    assert (max_lon - min_lon) > (max_lat - min_lat)


def test_near_points_unions_the_query_points() -> None:
    near_a = line("A", [(0.0, 0.0), (0.0005, 0.0)])
    near_b = line("B", [(0.5, 0.0), (0.5005, 0.0)])
    far = line("FAR", [(9.0, 0.0), (9.0005, 0.0)])
    index = FeatureIndex([near_a, near_b, far])
    hit = index.near_points([(LON0, LAT0), (LON0 + 0.5, LAT0)], 200.0)
    assert [f["properties"]["street_name"] for f in hit] == ["A", "B"]
