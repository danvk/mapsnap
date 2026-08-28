import json
import math
from pathlib import Path

import pytest

from mapsnap.keymap.locate import (
    MIN_KEYMAP_MEGAPIXELS,
    KeymapLocator,
    above_megapixel_floor,
    bilinear_pixel_to_world,
    discover_keymaps,
    estimate_radius,
    geometry_segments,
    geometry_vertices,
    keymap_megapixels,
    meters_between,
    page_key,
    page_number,
    redundant_keymaps,
    resolve_keymaps,
    usable_keymaps,
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
    locations = {"1": [(0.0, 0.0)], "2": [(0.0, 0.001)], "3": [(0.0, 0.002)]}
    assert math.isclose(estimate_radius(locations), 2 * 110.54, rel_tol=1e-2)


def test_restricted_features_none_when_unplaced_else_nearby():
    locator = KeymapLocator(locations={"61": [(0.0, 0.0)]}, radius_m=150.0)
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
    locator = KeymapLocator(locations={"61": [(0.0, 0.0)]}, radius_m=150.0)
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


def test_feature_index_is_cached_per_feature_list():
    """The cull's index is reused for the same list and rebuilt for a different one."""
    locator = KeymapLocator(locations={"61": [(0.0, 0.0)]}, radius_m=150.0)
    features = [{"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "id": "a"}]
    first = locator.feature_index(features)
    assert locator.feature_index(features) is first
    other = [{"geometry": {"type": "Point", "coordinates": [1.0, 1.0]}, "id": "b"}]
    rebuilt = locator.feature_index(other)
    assert rebuilt is not first
    assert [f["id"] for f in rebuilt.near_bbox((0.9, 1.1, 0.9, 1.1))] == ["b"]


def test_located_keys_and_page_keys():
    locator = KeymapLocator(
        locations={"1": [(0.0, 0.0)], "61": [(1.0, 1.0)]}, radius_m=100.0
    )
    assert locator.located_keys() == {"1", "61"}
    assert page_number("p61w") == 61 and page_number("p9n") == 9
    assert page_key("p61w") == "61W" and page_key("p1499a") == "1499A"


def test_string_key_lookup_prefers_exact_then_family():
    locator = KeymapLocator(
        locations={"35A": [(0.0, 0.0)], "35B": [(1.0, 1.0)], "36": [(2.0, 2.0)]},
        radius_m=100.0,
    )
    # exact key -> only that entry; unknown member or bare int -> whole family
    assert locator.centers_for("35A") == [(0.0, 0.0)]
    assert sorted(locator.centers_for(35)) == [(0.0, 0.0), (1.0, 1.0)]
    assert sorted(locator.centers_for("35")) == [(0.0, 0.0), (1.0, 1.0)]
    # a lettered lookup against a bare-printed key map falls back to the stem
    bare = KeymapLocator(locations={"51": [(3.0, 3.0)]}, radius_m=100.0)
    assert bare.centers_for("51N") == [(3.0, 3.0)]
    assert locator.centers_for(None) == []


def test_regions_by_number_merges_families():
    ring_a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    ring_b = [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)]
    locator = KeymapLocator(
        locations={},
        radius_m=100.0,
        regions={"35A": [ring_a], "35B": [ring_b]},
    )
    assert locator.regions_by_number() == {35: [ring_a, ring_b]}
    assert locator.regions_for(35) == [ring_a, ring_b]
    assert locator.regions_for("35A") == [ring_a]


def test_page_keymap_entry():
    # Two detections of page 61 (e.g. a split sheet, one block per panel): lat/lon is
    # their mean (compatibility), centers carries each detection.
    locator = KeymapLocator(
        locations={"61": [(-87.5, 41.9), (-87.6, 41.7)]}, radius_m=300.0
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
        locations={"7": [(0.5, 0.5)], "8": [(0.9, 0.9)]},
        radius_m=100.0,
        regions={"7": [[(0.4, 0.6), (0.6, 0.6), (0.6, 0.4), (0.4, 0.4)]]},
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
    assert set(regions) == {"61"}
    ring = regions["61"][0]
    assert ring[0] == (0.0, 3.0)  # top-left corner
    lon, lat = ring[2]
    assert math.isclose(lon, 0.5) and math.isclose(lat, 2.5)  # image center
    # No sidecar -> empty dict.
    assert load_regions(tmp_path / "other.keymap.json", CORNERS, 1000, 500) == {}


def test_rectangle_features_covers_whole_keymap_box():
    # Key map spanning lon 0..0.01, lat 0..0.01 (~1.1 km); tiny margin from radius_m.
    locator = KeymapLocator(
        locations={"1": [(0.0, 0.0)]},
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
        # A georeferenced key map: usable_keymaps tests the recorded pose, not
        # merely the sidecar's existence (a demoted sidecar is still on disk).
        (directory / f"{stem}.georef.json").write_text(
            json.dumps(
                {
                    "corners": [
                        [-81.0, 34.1],
                        [-80.9, 34.1],
                        [-80.9, 34.0],
                        [-81.0, 34.0],
                    ]
                }
            )
        )
    return keymap


def test_usable_keymaps_keeps_only_georeferenced(tmp_path: Path):
    # Atlanta 1911 has two index sheets and only one georeferenced; the volume
    # scans that glob raw/*.keymap.json must skip the other, not open a georef
    # that isn't there.
    make_keymap(tmp_path, "p0a")
    make_keymap(tmp_path, "p0b", with_georef=False)
    assert usable_keymaps(tmp_path) == [tmp_path / "p0a.keymap.json"]


def test_usable_keymaps_empty_directory(tmp_path: Path):
    assert usable_keymaps(tmp_path) == []


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


# Volume pages 1-20, as volume_page_keys reports them.
VOLUME_KEYS = {str(n) for n in range(1, 21)}


def test_redundant_keymaps_drops_the_more_foreign_of_two_duplicates():
    # Atlanta's shape: both sheets index this volume's pages, but one is a
    # multi-volume index that also carries page numbers from other volumes.
    own = {"1", "2", "3", "4", "5", "6"}
    multi = {"1", "2", "3", "4", "5", "6", "154", "823", "7217", "195"}
    assert redundant_keymaps([own, multi], VOLUME_KEYS) == {1}
    assert redundant_keymaps([multi, own], VOLUME_KEYS) == {0}


def test_redundant_keymaps_keeps_complementary_halves():
    # The healthy multi-key-map shape: each sheet indexes its own half, so neither
    # is redundant however different their page counts or foreign-key rates.
    left = {"1", "2", "3", "4", "5"}
    right = {"11", "12", "13", "14", "15", "99"}
    assert redundant_keymaps([left, right], VOLUME_KEYS) == set()


def test_redundant_keymaps_keeps_duplicates_that_look_equally_good():
    # Redundancy alone is not grounds to drop: with nothing to separate them, a
    # wrong guess would cost half the volume's placements.
    a = {"1", "2", "3", "4", "5", "6"}
    b = {"1", "2", "3", "4", "5", "7"}
    assert redundant_keymaps([a, b], VOLUME_KEYS) == set()


def test_redundant_keymaps_needs_the_volume_page_set():
    # Without it nothing can be called foreign, so there is no evidence to act on.
    own = {"1", "2", "3", "4", "5", "6"}
    multi = own | {"154", "823", "7217"}
    assert redundant_keymaps([own, multi], set()) == set()


def test_redundant_keymaps_handles_a_single_or_empty_sheet():
    assert redundant_keymaps([{"1", "2"}], VOLUME_KEYS) == set()
    assert redundant_keymaps([], VOLUME_KEYS) == set()
    assert redundant_keymaps([set(), {"1", "2"}], VOLUME_KEYS) == set()


def test_redundant_keymaps_drops_only_one_of_three_duplicates():
    # Two good sheets and one multi-volume index: the index goes, both others stay.
    a = {"1", "2", "3", "4", "5", "6"}
    b = {"1", "2", "3", "4", "5", "6"}
    multi = a | {"154", "823", "7217", "911"}
    assert redundant_keymaps([a, b, multi], VOLUME_KEYS) == {2}


def write_keymap_pair(directory, stem, width, height):
    """A <stem>.keymap.json with the georef sidecar usable_keymaps requires."""
    (directory / f"{stem}.keymap.json").write_text(json.dumps({"streets": []}))
    (directory / f"{stem}.georef.json").write_text(
        json.dumps(
            {
                "width": width,
                "height": height,
                # A real pose: usable_keymaps requires a georeferenced key map,
                # and a cornerless sidecar is a page georef could not place.
                "corners": [[-81.0, 34.1], [-80.9, 34.1], [-80.9, 34.0], [-81.0, 34.0]],
            }
        )
    )


def test_keymap_megapixels_reads_the_sidecar(tmp_path):
    write_keymap_pair(tmp_path, "p0", 6447, 7795)
    assert keymap_megapixels(tmp_path / "p0.georef.json") == pytest.approx(
        50.3, abs=0.1
    )


def test_keymap_megapixels_returns_none_without_dimensions(tmp_path):
    (tmp_path / "p0.georef.json").write_text(json.dumps({"corners": []}))
    assert keymap_megapixels(tmp_path / "p0.georef.json") is None


def test_above_megapixel_floor_splits_the_two_populations(tmp_path):
    """Columbia's 4400x5517 is under the floor; Detroit's 6447x7795 is over it."""
    write_keymap_pair(tmp_path, "columbia", 4400, 5517)
    write_keymap_pair(tmp_path, "detroit", 6447, 7795)
    assert not above_megapixel_floor(tmp_path / "columbia.keymap.json")
    assert above_megapixel_floor(tmp_path / "detroit.keymap.json")


def test_above_megapixel_floor_passes_a_sheet_with_no_recorded_size(tmp_path):
    """An unreadable size is not evidence of a bad scan, so the sheet passes."""
    (tmp_path / "p0.keymap.json").write_text(json.dumps({"streets": []}))
    (tmp_path / "p0.georef.json").write_text(json.dumps({"corners": []}))
    assert above_megapixel_floor(tmp_path / "p0.keymap.json")


def test_usable_keymaps_ignores_resolution(tmp_path):
    """The locator itself keeps low-resolution sheets.

    osm-snap and street-solve build locators through usable_keymaps and get real
    value from a sheet whose page numbers are unreliable -- its georeferenced
    rectangle alone cuts Columbia's candidate streets to 22%. Only the
    vocabulary restriction, which needs each page's number read correctly,
    applies the floor.
    """
    write_keymap_pair(tmp_path, "columbia", 4400, 5517)
    assert [p.name for p in usable_keymaps(tmp_path)] == ["columbia.keymap.json"]


def test_usable_keymaps_still_requires_a_georef(tmp_path):
    """The pre-existing rule is unchanged: no sidecar, no locator."""
    (tmp_path / "p0.keymap.json").write_text(json.dumps({"streets": []}))
    assert usable_keymaps(tmp_path) == []


def test_resolve_keymaps_applies_the_floor_to_discovery(tmp_path):
    """The vocabulary path drops a sub-floor sheet; the locator path kept it."""
    write_keymap_pair(tmp_path, "p0", 4400, 5517)
    (tmp_path / "p0.jpg").write_bytes(b"")
    assert usable_keymaps(tmp_path) != []
    assert resolve_keymaps(None, False, [str(tmp_path / "p0.jpg")]) == []


def test_resolve_keymaps_honours_an_explicit_keymap(tmp_path):
    """--keymap is the caller overriding the defaults, floor included."""
    write_keymap_pair(tmp_path, "p0", 4400, 5517)
    explicit = [str(tmp_path / "p0.keymap.json")]
    assert resolve_keymaps(explicit, False, []) == [tmp_path / "p0.keymap.json"]


def test_resolve_keymaps_still_honours_ignore(tmp_path):
    write_keymap_pair(tmp_path, "p0", 6447, 7795)
    (tmp_path / "p0.jpg").write_bytes(b"")
    assert resolve_keymaps(None, True, [str(tmp_path / "p0.jpg")]) == []


def test_megapixel_floor_sits_in_the_corpus_gap():
    """The two populations are 23.8-24.3 MP and 43.0-53.7 MP; the floor is between.

    Pinned because the value's justification is that gap, not the number itself:
    a future sheet landing near the floor means the assumption needs rechecking.
    """
    assert 25.0 < MIN_KEYMAP_MEGAPIXELS < 42.0


def test_resolve_keymaps_apply_floor_false_keeps_sub_floor_sheets(tmp_path: Path):
    """The sidecar-embed resolver must see sub-floor keymaps.

    The floor guards only the vocabulary restriction; the georef sidecar's
    keymap field is transport to osm-snap and street-solve. Dropping the
    locator entirely nulled every asheville/columbia sidecar's keymap and
    silently severed snap's search centers and region priors.
    """
    (tmp_path / "raw").mkdir(exist_ok=True)
    write_keymap_pair(tmp_path / "raw", "p0", 4400, 5517)  # columbia-size: sub-floor
    floored = resolve_keymaps(None, False, [str(tmp_path / "p5.jpg")])
    unfloored = resolve_keymaps(
        None, False, [str(tmp_path / "p5.jpg")], apply_floor=False
    )
    assert floored == []
    assert [p.name for p in unfloored] == ["p0.keymap.json"]
