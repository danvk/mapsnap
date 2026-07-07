import math
from pathlib import Path

from mapsnap.keymap.locate import (
    KeymapLocator,
    bilinear_pixel_to_world,
    discover_keymaps,
    estimate_radius,
    geometry_segments,
    geometry_vertices,
    meters_between,
    page_number,
    resolve_keymaps,
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


def test_geometry_segments_line_multiline_and_point():
    assert geometry_segments(
        {"type": "LineString", "coordinates": [[0, 0], [1, 0], [2, 0]]}
    ) == [((0, 0), (1, 0)), ((1, 0), (2, 0))]
    multi = {
        "type": "MultiLineString",
        "coordinates": [[[0, 0], [1, 1]], [[5, 5], [6, 6]]],
    }
    assert geometry_segments(multi) == [((0, 0), (1, 1)), ((5, 5), (6, 6))]
    # A Point yields one degenerate segment so isolated points still register.
    assert geometry_segments({"type": "Point", "coordinates": [3, 4]}) == [
        ((3, 4), (3, 4))
    ]


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


def test_restricted_features_keeps_through_street_with_no_vertex_inside():
    # A street whose endpoints (~222 m away) both fall outside the 150 m radius but whose
    # segment crosses the page center: segment distance keeps it; a vertex test would drop it.
    locator = KeymapLocator(locations={61: [(0.0, 0.0)]}, radius_m=150.0)
    features = [
        {
            "geometry": {
                "type": "LineString",
                "coordinates": [[-0.002, 0.0], [0.002, 0.0]],
            },
            "id": "through",
        },
    ]
    assert all(
        meters_between((0.0, 0.0), v) > 150.0
        for v in geometry_vertices(features[0]["geometry"])
    )  # both endpoints are outside the radius
    kept = locator.restricted_features(61, features)
    assert kept is not None and [f["id"] for f in kept] == ["through"]


def test_located_numbers_and_page_number():
    locator = KeymapLocator(
        locations={1: [(0.0, 0.0)], 61: [(1.0, 1.0)]}, radius_m=100.0
    )
    assert locator.located_numbers() == {1, 61}
    assert page_number("p61w") == 61 and page_number("p9n") == 9


def test_page_keymap_entry():
    # Two detections of page 61 (e.g. a split sheet, one block per panel): lat/lon is
    # their mean (compatibility), centers carries each detection.
    locator = KeymapLocator(
        locations={61: [(-87.5, 41.9), (-87.6, 41.7)]}, radius_m=300.0
    )
    assert locator.page_keymap(61) == {
        "lat": 41.8,
        "lon": -87.55,
        "radius_m": 300.0,
        "centers": [[-87.5, 41.9], [-87.6, 41.7]],
    }
    assert locator.page_keymap(999) is None  # unplaced
    assert locator.page_keymap(None) is None


def test_page_keymap_includes_regions_as_lon_lat_rings():
    locator = KeymapLocator(
        locations={7: [(0.5, 0.5)], 8: [(0.9, 0.9)]},
        radius_m=100.0,
        regions={7: [[(0.4, 0.6), (0.6, 0.6), (0.6, 0.4), (0.4, 0.4)]]},
    )
    entry = locator.page_keymap(7)
    assert entry is not None
    assert entry["regions"] == [[[0.4, 0.6], [0.6, 0.6], [0.6, 0.4], [0.4, 0.4]]]
    # A placed page with no segmented region omits the key entirely.
    entry8 = locator.page_keymap(8)
    assert entry8 is not None and "regions" not in entry8


def test_region_scale_m_per_px():
    from mapsnap.keymap.locate import region_scale_m_per_px

    # A 0.001 x 0.001 deg square at the equator is ~110.5 x 111.3 m. On a 100 x 100 px
    # page that's sqrt(110.54 * 111.32 / 1e4) ~ 1.109 m/px.
    square = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    scale = region_scale_m_per_px([square], 100, 100)
    assert scale is not None and math.isclose(scale, 1.109, rel_tol=1e-2)
    # Two half-squares sum back to the full block (watershed-split duplicate detections).
    left = [(0.0, 0.0), (0.0005, 0.0), (0.0005, 0.001), (0.0, 0.001)]
    right = [(0.0005, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0005, 0.001)]
    both = region_scale_m_per_px([left, right], 100, 100)
    assert both is not None and math.isclose(both, scale, rel_tol=1e-6)
    # Degenerate rings and empty input give None.
    assert region_scale_m_per_px([], 100, 100) is None
    assert region_scale_m_per_px([[(0, 0), (1, 1)]], 100, 100) is None


def test_load_regions_maps_pixels_to_world(tmp_path):
    import json

    from mapsnap.keymap.locate import load_regions

    # Regions computed at half resolution (500x250) of the 1000x500 georeferenced image;
    # the pixel ring must be rescaled before the bilinear mapping. Non-numeric labels skipped.
    regions_doc = {
        "image": "km.jpg",
        "width": 500,
        "height": 250,
        "panels": [
            [[0, 0], [250, 0], [250, 125], [0, 125]],  # NW quarter of the key map
            [[0, 0], [10, 0], [10, 10]],
        ],
        "labels": ["61", "?"],
    }
    keymap_json = tmp_path / "km.keymap.json"
    (tmp_path / "km.regions.panels.json").write_text(json.dumps(regions_doc))
    regions = load_regions(keymap_json, CORNERS, 1000, 500)
    assert set(regions) == {61}
    ring = regions[61][0]
    assert ring[0] == (0.0, 3.0)  # top-left corner
    lon, lat = ring[2]
    assert math.isclose(lon, 0.5) and math.isclose(lat, 2.5)  # image center
    # No sidecar -> empty dict.
    assert load_regions(tmp_path / "other.keymap.json", CORNERS, 1000, 500) == {}


def test_rectangle_features_covers_whole_keymap_box():
    # Key map spanning lon 0..0.01, lat 0..0.01 (~1.1 km); tiny margin from radius_m.
    locator = KeymapLocator(
        locations={1: [(0.0, 0.0)]},
        radius_m=50.0,
        rectangles=[[(0.0, 0.01), (0.01, 0.01), (0.01, 0.0), (0.0, 0.0)]],
    )
    features = [
        {"geometry": {"type": "Point", "coordinates": [0.005, 0.005]}, "id": "inside"},
        {"geometry": {"type": "Point", "coordinates": [0.5, 0.5]}, "id": "far_outside"},
    ]
    kept = locator.rectangle_features(features)
    assert kept is not None and [f["id"] for f in kept] == ["inside"]
    # No rectangles -> None (caller falls back to full vocab).
    assert (
        KeymapLocator(locations={}, radius_m=50.0).rectangle_features(features) is None
    )


def test_rectangle_features_unions_multiple_keymaps():
    # Two disjoint key-map rectangles (SW box near origin, NE box near lon/lat 1).
    locator = KeymapLocator(
        locations={},
        radius_m=50.0,
        rectangles=[
            [(0.0, 0.01), (0.01, 0.01), (0.01, 0.0), (0.0, 0.0)],
            [(1.0, 1.01), (1.01, 1.01), (1.01, 1.0), (1.0, 1.0)],
        ],
    )
    features = [
        {"geometry": {"type": "Point", "coordinates": [0.005, 0.005]}, "id": "in_a"},
        {"geometry": {"type": "Point", "coordinates": [1.005, 1.005]}, "id": "in_b"},
        {"geometry": {"type": "Point", "coordinates": [0.5, 0.5]}, "id": "between"},
    ]
    kept = locator.rectangle_features(features)
    assert kept is not None
    assert {f["id"] for f in kept} == {"in_a", "in_b"}  # union of both rectangles


def make_keymap(directory: Path, stem: str, *, with_georef: bool = True) -> Path:
    """Create a <stem>.keymap.json (and optionally its .georef.json sibling) in directory."""
    directory.mkdir(parents=True, exist_ok=True)
    keymap = directory / f"{stem}.keymap.json"
    keymap.write_text("{}")
    if with_georef:
        (directory / f"{stem}.georef.json").write_text("{}")
    return keymap


def test_discover_keymaps_finds_under_raw(tmp_path: Path):
    # ocr/georef run on top-level pages; the key map's sidecars live under raw/.
    raw = tmp_path / "raw"
    make_keymap(raw, "p0")
    found = discover_keymaps([str(tmp_path / "p5.jpg"), str(tmp_path / "p6.jpg")])
    assert found == [raw / "p0.keymap.json"]


def test_discover_keymaps_finds_in_same_directory(tmp_path: Path):
    make_keymap(tmp_path, "p1b")
    assert discover_keymaps([str(tmp_path / "p1b.jpg")]) == [
        tmp_path / "p1b.keymap.json"
    ]


def test_discover_keymaps_skips_keymap_without_georef(tmp_path: Path):
    # A key map whose georeferencing failed has no .georef.json; a locator can't use it.
    make_keymap(tmp_path / "raw", "p0", with_georef=False)
    assert discover_keymaps([str(tmp_path / "p5.jpg")]) == []


def test_discover_keymaps_dedups_across_images(tmp_path: Path):
    make_keymap(tmp_path / "raw", "p0")
    images = [str(tmp_path / "p5.jpg"), str(tmp_path / "p6.jpg")]
    assert discover_keymaps(images) == [tmp_path / "raw" / "p0.keymap.json"]


def test_resolve_keymaps_ignore_beats_everything(tmp_path: Path):
    make_keymap(tmp_path / "raw", "p0")
    assert (
        resolve_keymaps(["explicit.keymap.json"], True, [str(tmp_path / "p5.jpg")])
        == []
    )


def test_resolve_keymaps_explicit_wins_over_discovery(tmp_path: Path):
    make_keymap(tmp_path / "raw", "p0")
    resolved = resolve_keymaps(["given.keymap.json"], False, [str(tmp_path / "p5.jpg")])
    assert resolved == [Path("given.keymap.json")]


def test_resolve_keymaps_falls_back_to_discovery(tmp_path: Path):
    keymap = make_keymap(tmp_path / "raw", "p0")
    assert resolve_keymaps(None, False, [str(tmp_path / "p5.jpg")]) == [keymap]
