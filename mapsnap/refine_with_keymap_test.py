import math

from mapsnap.georef_from_labels import deg_per_px_to_px_per_ft
from mapsnap.keymap.fit_keymap import Detection, GeorefPage
from mapsnap.refine_with_keymap import (
    KeymapGeoref,
    expected_world_centers,
    filter_centerlines,
    frame_diagonal,
    geometry_vertices,
    keymap_pixel_to_world,
    near_any,
    scale_outlier_indices,
    unfit_located_numbers,
)

# A key map georeferenced to a simple axis-aligned box: 1000x500 px spanning lon 0..1, lat 2..3
# (corners in TL, TR, BR, BL order; note lat decreases from top to bottom).
BOX_KEYMAP = KeymapGeoref(
    corners=[(0.0, 3.0), (1.0, 3.0), (1.0, 2.0), (0.0, 2.0)],
    width=1000,
    height=500,
)


def test_keymap_pixel_to_world_corners_and_center():
    assert keymap_pixel_to_world(BOX_KEYMAP, (0, 0)) == (0.0, 3.0)  # top-left
    assert keymap_pixel_to_world(BOX_KEYMAP, (1000, 0)) == (1.0, 3.0)  # top-right
    assert keymap_pixel_to_world(BOX_KEYMAP, (1000, 500)) == (1.0, 2.0)  # bottom-right
    # Center pixel maps to the box center.
    lon, lat = keymap_pixel_to_world(BOX_KEYMAP, (500, 250))
    assert math.isclose(lon, 0.5) and math.isclose(lat, 2.5)


def test_expected_world_centers_unions_all_occurrences():
    detections = [
        Detection(79, (250, 250)),
        Detection(79, (750, 250)),
        Detection(61, (500, 125)),
    ]
    centers = expected_world_centers(79, detections, BOX_KEYMAP)
    assert len(centers) == 2  # page 79 appears twice on the key map
    assert {round(c[0], 3) for c in centers} == {0.25, 0.75}
    assert expected_world_centers(3, detections, BOX_KEYMAP) == []


def test_near_any():
    assert near_any((0.0, 0.0), [(3.0, 4.0)], radius=5.0)
    assert not near_any((0.0, 0.0), [(3.0, 4.0)], radius=4.9)
    assert not near_any((0.0, 0.0), [], radius=100.0)


def test_frame_diagonal():
    assert math.isclose(frame_diagonal([(0, 0), (3, 0), (3, 4), (0, 4)]), 5.0)


def test_geometry_vertices_handles_line_and_multiline():
    assert geometry_vertices(
        {"type": "LineString", "coordinates": [[0, 1], [2, 3]]}
    ) == [
        (0, 1),
        (2, 3),
    ]
    multi = {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]], [[2, 2]]]}
    assert geometry_vertices(multi) == [(0, 0), (1, 1), (2, 2)]
    assert geometry_vertices({"type": "GeometryCollection"}) == []


def test_filter_centerlines_keeps_only_near_features():
    # Two point features ~0m and ~far from a center at the projection origin (lon0, lat0).
    geojson = {
        "features": [
            {"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "id": "near"},
            {"geometry": {"type": "Point", "coordinates": [1.0, 1.0]}, "id": "far"},
        ]
    }
    kept = filter_centerlines(
        geojson, centers=[(0.0, 0.0)], radius=100.0, lon0=0.0, lat0=0.0
    )
    assert [f["id"] for f in kept["features"]] == ["near"]


def test_unfit_located_numbers():
    pages = [GeorefPage(2, (0, 0), [], ["p2"]), GeorefPage(5, (0, 0), [], ["p5"])]
    detections = [Detection(2, (0, 0)), Detection(5, (0, 0)), Detection(61, (0, 0))]
    assert unfit_located_numbers(pages, detections) == {61}


def test_scale_outlier_indices():
    # deg_per_px scales; the reference is the px/ft implied by the first (nominal) scale, so the
    # 2x-off scale (half the px/ft) exceeds the 0.25 threshold and the matching ones do not.
    scales = [1e-6, 1e-6, 2e-6]
    reference = deg_per_px_to_px_per_ft(1e-6)
    assert scale_outlier_indices(scales, reference, threshold=0.25) == {2}
    assert scale_outlier_indices(scales, reference, threshold=0) == set()
    assert scale_outlier_indices(scales, None, threshold=0.25) == set()
