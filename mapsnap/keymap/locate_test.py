import math

from mapsnap.keymap.locate import (
    KeymapLocator,
    bilinear_pixel_to_world,
    estimate_radius,
    geometry_vertices,
    meters_between,
    page_number,
)

# A key map georeferenced to an axis-aligned box: 1000x500 px over lon 0..1, lat 2..3
# (corners TL, TR, BR, BL; latitude decreases top to bottom).
CORNERS = [(0.0, 3.0), (1.0, 3.0), (1.0, 2.0), (0.0, 2.0)]


def test_bilinear_pixel_to_world_corners_and_center():
    assert bilinear_pixel_to_world(CORNERS, 1000, 500, (0, 0)) == (0.0, 3.0)
    assert bilinear_pixel_to_world(CORNERS, 1000, 500, (1000, 500)) == (1.0, 2.0)
    lon, lat = bilinear_pixel_to_world(CORNERS, 1000, 500, (500, 250))
    assert math.isclose(lon, 0.5) and math.isclose(lat, 2.5)


def test_geometry_vertices_line_and_multiline():
    assert geometry_vertices(
        {"type": "LineString", "coordinates": [[0, 1], [2, 3]]}
    ) == [
        (0, 1),
        (2, 3),
    ]
    multi = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2]]]}
    assert geometry_vertices(multi) == [(0, 0), (1, 1), (2, 2)]
    assert geometry_vertices({"type": "GeometryCollection"}) == []


def test_meters_between_is_symmetric_and_scaled():
    # ~0.001 deg latitude is ~111 m; longitude is shorter by cos(lat).
    assert math.isclose(meters_between((0.0, 0.0), (0.0, 0.001)), 110.54, rel_tol=1e-3)
    assert meters_between((0.0, 45.0), (0.001, 45.0)) < meters_between(
        (0.0, 0.0), (0.001, 0.0)
    )


def test_estimate_radius_is_twice_page_spacing():
    # Three pages spaced 0.001 deg lat (~110.5 m) apart in a line -> radius ~2 * 110.5.
    locations = {1: [(0.0, 0.0)], 2: [(0.0, 0.001)], 3: [(0.0, 0.002)]}
    assert math.isclose(estimate_radius(locations), 2 * 110.54, rel_tol=1e-2)


def test_restricted_features_none_when_unplaced_else_nearby():
    locator = KeymapLocator(locations={61: [(0.0, 0.0)]}, radius_m=150.0)
    features = [
        {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "id": "near"},
        {
            "geometry": {"type": "Point", "coordinates": [0.0, 0.01]},
            "id": "far",
        },  # ~1.1 km
    ]
    assert locator.restricted_features(999, features) is None  # unplaced page
    kept = locator.restricted_features(61, features)
    assert kept is not None and [f["id"] for f in kept] == ["near"]


def test_located_numbers_and_page_number():
    locator = KeymapLocator(
        locations={1: [(0.0, 0.0)], 61: [(1.0, 1.0)]}, radius_m=100.0
    )
    assert locator.located_numbers() == {1, 61}
    assert page_number("p61w") == 61 and page_number("p9n") == 9


def test_rectangle_features_covers_whole_keymap_box():
    # Key map spanning lon 0..0.01, lat 0..0.01 (~1.1 km); tiny margin from radius_m.
    locator = KeymapLocator(
        locations={1: [(0.0, 0.0)]},
        radius_m=50.0,
        corners=[(0.0, 0.01), (0.01, 0.01), (0.01, 0.0), (0.0, 0.0)],
    )
    features = [
        {"geometry": {"type": "Point", "coordinates": [0.005, 0.005]}, "id": "inside"},
        {"geometry": {"type": "Point", "coordinates": [0.5, 0.5]}, "id": "far_outside"},
    ]
    kept = locator.rectangle_features(features)
    assert kept is not None and [f["id"] for f in kept] == ["inside"]
    # No corners -> None (caller falls back to full vocab).
    assert (
        KeymapLocator(locations={}, radius_m=50.0).rectangle_features(features) is None
    )
