"""Unit tests for the page focus-area / key-map region alignment helpers."""

import json
import math

import numpy as np
from shapely.geometry import Polygon

from mapsnap.keymap.align_page_region import (
    FocusParams,
    PageResult,
    align_focus_to_region,
    angle_difference_mod180,
    angle_wrap,
    focus_footprint,
    image_neighbor_directions,
    implied_reflected_rotation,
    init_pose_from_model,
    invert_similarity,
    page_axes,
    page_corners_world,
    point_to_segments,
    polygon_exterior,
    polygon_iou,
    pose_residuals,
    pose_world_of,
    reciprocated_neighbors,
    resample_ring,
    robust_rotation,
    similarity_from_pose,
    street_soup_metres,
    write_georef,
)
from mapsnap.keymap.fit_keymap import similarity_apply
from mapsnap.streets import Block

UNIT_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_angle_wrap_folds_to_half_open_range():
    assert angle_wrap(190.0) == -170.0
    assert angle_wrap(-190.0) == 170.0
    assert angle_wrap(45.0) == 45.0


def test_implied_reflected_rotation_identity_and_quarter_turn():
    # Image "up" (0, -1) should map to world "north" (0, 1) with zero reflected rotation.
    assert abs(implied_reflected_rotation((0.0, -1.0), (0.0, 1.0))) < 1e-9
    # A correspondence needing a +90 deg reflected rotation.
    assert abs(implied_reflected_rotation((1.0, 0.0), (0.0, 1.0)) - 90.0) < 1e-9


def test_robust_rotation_rejects_outliers():
    # Three pairs agree on ~identity; one is a wild outlier that must be excluded.
    pairs = [
        ((0.0, -1.0), (0.0, 1.0), 1.0),
        ((1.0, 0.0), (1.0, 0.0), 1.0),
        ((-1.0, 0.0), (-1.0, 0.0), 1.0),
        ((0.0, 1.0), (1.0, 0.0), 1.0),  # outlier: implies a different rotation
    ]
    (a, b), inliers = robust_rotation(pairs)
    assert inliers == 3
    assert abs(a - 1.0) < 1e-6 and abs(b) < 1e-6


def test_robust_rotation_too_few_pairs():
    (a, b), inliers = robust_rotation([((1.0, 0.0), (1.0, 0.0), 1.0)])
    assert inliers == 0
    assert (a, b) == (1.0, 0.0)


def test_similarity_from_pose_maps_centroids():
    model = similarity_from_pose((1.0, 0.0), 2.0, (10.0, 10.0), (0.0, 0.0))
    mapped = similarity_apply(model, (10.0, 10.0))
    assert abs(mapped[0]) < 1e-9 and abs(mapped[1]) < 1e-9


def test_invert_similarity_round_trips():
    model = (0.7, -0.4, 12.0, -3.0)
    for point in [(0.0, 0.0), (5.0, 9.0), (-3.0, 2.0)]:
        forward = similarity_apply(model, point)
        back = invert_similarity(model, forward)
        assert math.isclose(back[0], point[0], abs_tol=1e-6)
        assert math.isclose(back[1], point[1], abs_tol=1e-6)


def test_resample_ring_returns_requested_count_on_perimeter():
    points = resample_ring(UNIT_SQUARE, 8)
    assert len(points) == 8
    for x, y in points:
        on_edge = (
            math.isclose(x, 0.0, abs_tol=1e-6)
            or math.isclose(x, 10.0, abs_tol=1e-6)
            or math.isclose(y, 0.0, abs_tol=1e-6)
            or math.isclose(y, 10.0, abs_tol=1e-6)
        )
        assert on_edge


def test_polygon_iou_identical_and_disjoint():
    a = Polygon(UNIT_SQUARE)
    b = Polygon([(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)])
    assert math.isclose(polygon_iou(a, a), 1.0, abs_tol=1e-9)
    assert polygon_iou(a, b) == 0.0


def test_polygon_exterior_of_polygon_and_multipolygon():
    ring = polygon_exterior(Polygon(UNIT_SQUARE))
    assert len(ring) == 4
    small = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    big = Polygon([(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])
    largest = polygon_exterior(small.union(big))
    assert Polygon(largest).area > 50.0  # the larger part, not the small one


def test_align_focus_to_region_recovers_identity():
    # A square page outline aligned to the same square region (in metres) should score IoU ~ 1.
    region = Polygon(UNIT_SQUARE)
    model, iou = align_focus_to_region(UNIT_SQUARE, region, (1.0, 0.0), FocusParams())
    assert iou > 0.95
    assert model is not None


def test_reciprocated_neighbors():
    adjacency = {"adjacency": [["p45", "p43"], ["p44", "p45"], ["p1", "p2"]]}
    assert reciprocated_neighbors(adjacency, 45) == {43, 44}
    assert reciprocated_neighbors(adjacency, 1) == {2}
    assert reciprocated_neighbors(adjacency, 99) == set()


def test_image_neighbor_directions_filters_and_normalizes():
    adjacency = {
        "pages": {
            "p45": {
                "detections": [
                    {
                        "number": 43,
                        "claim": True,
                        "x_frac": 0.5,
                        "y_frac": 0.9,
                        "confidence": 0.9,
                    },
                    {
                        "number": 2,
                        "claim": False,
                        "x_frac": 0.9,
                        "y_frac": 0.4,
                        "confidence": 0.8,
                    },
                ]
            }
        }
    }
    directions = image_neighbor_directions(adjacency, "p45")
    assert set(directions) == {43}
    (dx, dy), confidence = directions[43]
    assert math.isclose(math.hypot(dx, dy), 1.0, abs_tol=1e-9)
    assert dy > 0 and confidence == 0.9  # printed toward the bottom edge


def test_focus_footprint_hulls_a_colored_rectangle():
    # White page with a saturated red rectangle; the hull should enclose that rectangle.
    rgb = np.ones((200, 300, 3), dtype=np.float64)
    rgb[40:160, 60:240] = (1.0, 0.0, 0.0)
    polygon = focus_footprint(rgb, FocusParams(target_long_side=300))
    assert len(polygon) >= 3
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert min(xs) < 75 and max(xs) > 225
    assert min(ys) < 55 and max(ys) > 145


def test_focus_footprint_detects_subtle_dark_fill():
    # A dark-brown block (low chroma, but darker than the tan paper) must be detected even though
    # a chroma-only rule would miss it — this is the Chicago case.
    rgb = np.full((200, 300, 3), 0.9, dtype=np.float64)  # light tan paper
    rgb[40:160, 60:240] = (0.40, 0.37, 0.30)  # dark brown, near-paper chroma
    polygon = focus_footprint(rgb, FocusParams(target_long_side=300))
    assert len(polygon) >= 3
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert min(xs) < 75 and max(xs) > 225 and min(ys) < 55 and max(ys) > 145
    # With the dark-fill rule disabled (floor above all lightness), chroma alone misses it.
    chroma_only = FocusParams(target_long_side=300, dark_fill_floor=300.0)
    assert focus_footprint(rgb, chroma_only) == []


def test_focus_footprint_ignores_edge_artifacts():
    # A colored blob only in the top-left corner (a scan artifact) is inside the 4% trim band and
    # must be ignored, leaving no focus area.
    rgb = np.ones((200, 300, 3), dtype=np.float64)
    rgb[0:6, 0:10] = (
        1.0,
        0.0,
        0.0,
    )  # within trim (round(0.04*200)=8 rows, round(0.04*300)=12 cols)
    assert focus_footprint(rgb, FocusParams(target_long_side=300)) == []
    # Disabling the trim exposes the artifact again.
    no_trim = FocusParams(target_long_side=300, edge_trim_frac=0.0)
    assert len(focus_footprint(rgb, no_trim)) >= 3


def test_page_corners_world_orders_four_corners():
    model = similarity_from_pose((1.0, 0.0), 1.0, (50.0, 50.0), (0.0, 0.0))
    corners = page_corners_world((100, 100), model, (-73.99, 40.67))
    assert len(corners) == 4
    assert all(len(corner) == 2 for corner in corners)


# --- street-constrained refinement helpers ---


def test_page_axes_north_up():
    up, right = page_axes(0.0)
    assert np.allclose(up, [0.0, 1.0])
    assert np.allclose(right, [1.0, 0.0])


def test_angle_difference_mod180_is_undirected():
    assert math.isclose(angle_difference_mod180(10.0, 0.0), 10.0)
    # 170 and 10 are 20 apart as undirected bearings (they wrap through 180).
    assert math.isclose(angle_difference_mod180(170.0, 10.0), -20.0)


def test_point_to_segments_distance_and_bearing():
    starts = np.array([[0.0, 0.0]])
    ends = np.array([[10.0, 0.0]])  # a due-east segment
    distance, bearing = point_to_segments(np.array([5.0, 3.0]), starts, ends)
    assert math.isclose(distance, 3.0, abs_tol=1e-9)
    assert math.isclose(bearing, 90.0, abs_tol=1e-9)


def test_pose_world_of_matches_the_model_it_was_derived_from():
    # init_pose_from_model must re-express the similarity exactly: same pixel -> same world point.
    model = similarity_from_pose((0.6, 0.3), 2.0, (40.0, 60.0), (5.0, -7.0))
    size = (120, 80)
    pose = init_pose_from_model(model, size)
    for pixel in [(60.0, 40.0), (10.0, 5.0), (119.0, 79.0)]:
        expected = similarity_apply(model, pixel)
        actual = pose_world_of(pose, pixel, size)
        assert np.allclose(actual, expected, atol=1e-6)


def test_street_soup_metres_splits_into_segments():
    block = Block("TEST", np.array([[-74.0, 40.7], [-74.0, 40.701], [-73.999, 40.701]]))
    starts, ends = street_soup_metres([block], (-74.0, 40.7))
    assert starts.shape == (2, 2) and ends.shape == (2, 2)
    # The first vertex projects to the origin.
    assert np.allclose(starts[0], [0.0, 0.0], atol=1e-6)


def test_pose_residuals_length_and_finite():
    region = np.array(UNIT_SQUARE)
    pose = (0.0, 0.0, 0.0, math.log(4.0))
    residuals = pose_residuals(
        pose,
        [],
        [],
        region_vertices=region,
        size=(100, 100),
        prior_log_scale=math.log(4.0),
    )
    # One residual per region vertex + 2 centroid + 1 scale prior, all finite.
    assert len(residuals) == len(region) + 3
    assert np.all(np.isfinite(residuals))
    # Scale prior is satisfied at the prior, so its residual is zero.
    assert math.isclose(residuals[-1], 0.0, abs_tol=1e-9)


def test_write_georef_includes_keymap_truth_and_focus_hull(tmp_path):
    focus_hull = [[-73.99, 40.67], [-73.98, 40.67], [-73.985, 40.68]]
    result = PageResult(
        45, "ok", stem="p45", corners=[[0.0, 0.0]] * 4, focus_hull=focus_hull
    )
    keymap = {
        "lat": 40.67,
        "lon": -73.99,
        "centers": [[-73.99, 40.67]],
        "regions": [[[0, 0]]],
    }
    truth = [[[-73.99, 40.67], [-73.98, 40.67], [-73.98, 40.68]]]
    write_georef(tmp_path, "p45", (100, 200), result, keymap=keymap, truth=truth)
    written = json.loads((tmp_path / "p45.georef-region.json").read_text())
    assert written["page"] == 45 and written["accepted"] is True
    assert written["keymap"]["lat"] == 40.67 and "regions" in written["keymap"]
    assert written["truth"] == truth
    assert written["focus_hull"] == focus_hull


def test_write_georef_omits_missing_keymap_truth_and_focus_hull(tmp_path):
    result = PageResult(45, "ok", stem="p45", corners=[[0.0, 0.0]] * 4)
    write_georef(tmp_path, "p45", (100, 200), result)
    written = json.loads((tmp_path / "p45.georef-region.json").read_text())
    assert "keymap" not in written
    assert "truth" not in written
    assert "focus_hull" not in written
