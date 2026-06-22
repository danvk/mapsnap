import json

from mapsnap.keymap.fit_keymap import Detection
from mapsnap.refine_with_keymap import (
    expected_centers,
    filter_centerlines,
    frame_diagonal,
    georef_rotation,
    georef_scale_deg_per_px,
    keymap_centroid_error,
    near_any,
    pages_to_refine,
    refit_accepted,
    scale_outlier_indices,
)


def test_frame_diagonal():
    assert frame_diagonal([(0, 0), (3, 0), (3, 4), (0, 4)]) == 5.0


def test_near_any():
    centers = [(0.0, 0.0), (100.0, 0.0)]
    assert near_any((3.0, 4.0), centers, radius=5.0)  # exactly 5 from origin
    assert near_any((100.0, 10.0), centers, radius=15.0)  # near the second center
    assert not near_any((50.0, 50.0), centers, radius=10.0)


def test_expected_centers_applies_model_to_every_occurrence():
    # Model (1, 0, 0, 0): similarity_apply -> (x, -y) (a reflection).
    model = (1.0, 0.0, 0.0, 0.0)
    detections = [
        Detection(5, (2.0, 3.0)),
        Detection(6, (9.0, 9.0)),
        Detection(5, (4.0, 1.0)),
    ]
    assert expected_centers(5, model, detections) == [(2.0, -3.0), (4.0, -1.0)]


def _line(lon: float, lat: float) -> dict:
    return {
        "type": "Feature",
        "properties": {"street_name": f"S{lon}"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat], [lon, lat + 0.0001]],
        },
    }


def test_filter_centerlines_keeps_only_nearby_features():
    # lon0/lat0 = 0,0 -> project(lon,lat) = (lon*111320*cos0, lat*111320). 0.001 deg ~= 111 m.
    geojson = {
        "type": "FeatureCollection",
        "features": [_line(0.001, 0.0), _line(1.0, 0.0)],
    }
    subset = filter_centerlines(
        geojson, centers=[(0.0, 0.0)], radius=200.0, lon0=0.0, lat0=0.0
    )
    assert len(subset["features"]) == 1
    assert subset["features"][0]["properties"]["street_name"] == "S0.001"


class _Page:
    def __init__(self, number: int, centroid: tuple[float, float] = (0.0, 0.0)) -> None:
        self.number = number
        self.centroid = centroid


def test_pages_to_refine_picks_outliers_and_unfit():
    pages = [_Page(1), _Page(2), _Page(3)]
    detections = [Detection(1, (0, 0)), Detection(2, (0, 0)), Detection(4, (0, 0))]
    inliers = {0}  # page index 0 (number 1) is an inlier
    # number 2 is matched but not an inlier -> outlier; number 4 has no georef page -> unfit;
    # number 3 is georeferenced but absent from the key map -> left alone.
    assert pages_to_refine(pages, detections, inliers) == {2, 4}


def test_georef_scale_deg_per_px():
    # 0.001 deg of latitude over 100 px vertically (corners TL,TR,BR,BL), no horizontal tilt.
    corners = [[0.0, 0.0], [0.0, 0.0], [0.0, -0.001], [0.0, -0.001]]
    assert georef_scale_deg_per_px(corners, width=100, height=100) == 0.00001


def test_georef_rotation_north_up():
    # North-up page: lat decreases down the image, no horizontal lat change -> rotation 0.
    corners = [[0.0, 0.0], [0.0, 0.0], [0.0, -0.001], [0.0, -0.001]]
    assert georef_rotation(corners, width=100, height=100) == 0.0


def test_scale_outlier_indices():
    from mapsnap.georef_from_labels import deg_per_px_to_px_per_ft

    # Reference is the px/ft for a 1e-6 deg/px page; the 4x-deg/px page (index 3) is 0.25x it.
    ref = deg_per_px_to_px_per_ft(1e-6)
    scales = [1e-6, 1e-6, 1e-6, 4e-6]
    assert scale_outlier_indices(scales, ref, threshold=0.25) == {3}
    assert scale_outlier_indices(scales, ref, threshold=0.0) == set()  # disabled
    assert scale_outlier_indices(scales, None, threshold=0.25) == set()  # no reference
    assert scale_outlier_indices([], ref, threshold=0.25) == set()


def test_refit_accepted_strict_vs_loose(tmp_path):
    # Square frame lon 0..0.001, lat 0..-0.001 about origin (0,0) -> metres ~ 0..111 / 0..-111.
    g = tmp_path / "p1.georef2.json"
    g.write_text(
        json.dumps({"corners": [[0, 0], [0.001, 0], [0.001, -0.001], [0, -0.001]]})
    )
    origin = (0.0, 0.0)
    inside = (55.0, -55.0)  # metres, inside the frame
    far = (5000.0, 5000.0)
    # strict (1-GCP): the expected point must land inside the refined frame.
    assert refit_accepted(g, [inside], origin, 1000.0, strict=True)
    assert not refit_accepted(g, [far], origin, 1000.0, strict=True)
    # loose (multi-GCP): centroid (~55,-55) within radius of an expected center.
    assert refit_accepted(g, [(60.0, -60.0)], origin, 50.0, strict=False)
    assert not refit_accepted(g, [far], origin, 50.0, strict=False)


def test_keymap_centroid_error():
    # Model (1,0,0,0) maps (x, y) -> (x, -y); detection (2, 3) -> (2, -3), centroid (2, 0).
    pages = [_Page(5, centroid=(2.0, 0.0))]
    detections = [Detection(5, (2.0, 3.0))]
    assert keymap_centroid_error((1.0, 0.0, 0.0, 0.0), pages, {0}, detections) == 3.0
