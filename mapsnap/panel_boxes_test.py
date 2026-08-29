"""Tests for parent-to-panel CRAFT box derivation (#361)."""

import json

from PIL import Image
from shapely.geometry import Polygon

from mapsnap.panel_boxes import (
    derive_boxes_for_panel_image,
    derive_panel_boxes,
    rotate_point,
    unrotate_point,
)


def parent_doc(width=200, height=100, boxes=None):
    return {
        "width": width,
        "height": height,
        "timestamp": "t",
        "boxes": boxes
        or [
            {"angle": 0, "horizontal_list": [], "free_list": []},
            {"angle": 90, "horizontal_list": [], "free_list": []},
            {"angle": 270, "horizontal_list": [], "free_list": []},
        ],
    }


def test_rotate_unrotate_round_trip_all_angles():
    for angle in (0, 90, 270):
        for x, y in [(0, 0), (37, 12), (199, 99)]:
            rw, rh = (200, 100) if angle == 0 else (100, 200)
            rx, ry = rotate_point(x, y, angle, 200, 100)
            assert unrotate_point(rx, ry, angle, rw, rh) == (x, y)


def test_boxes_assigned_to_their_majority_panel():
    # Two side-by-side panels; one rect in each, and a seam straddler whose
    # majority (60%) lies in the right panel -- it must appear there only.
    polys = [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
             Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])]
    doc = parent_doc(boxes=[
        {"angle": 0,
         "horizontal_list": [[10, 40, 10, 20], [110, 140, 10, 20], [80, 130, 50, 60]],
         "free_list": []},
    ])
    left = derive_panel_boxes(doc, polys, 1, (200, 100))
    right = derive_panel_boxes(doc, polys, 2, (100, 100))
    assert left["boxes"][0]["horizontal_list"] == [[10, 40, 10, 20]]
    assert left["derived_from_parent"] is True
    # Right panel is cropped (offset 100); the straddler clamps at the seam.
    assert right["boxes"][0]["horizontal_list"] == [[10, 40, 10, 20], [0, 30, 50, 60]]


def test_quads_transform_point_by_point_not_bboxed():
    # A 45-degree quad must keep its shape (the A/B's 3-point lesson).
    polys = [Polygon([(0, 0), (200, 0), (200, 100), (0, 100)]),
             Polygon([(0, 0), (0, 0), (0, 0)])]
    quad = [[50, 40], [70, 20], [80, 30], [60, 50]]
    doc = parent_doc(boxes=[{"angle": 0, "horizontal_list": [], "free_list": [quad]}])
    out = derive_panel_boxes(doc, [polys[0]], 1, (200, 100))
    assert out["boxes"][0]["free_list"] == [[[50.0, 40.0], [70.0, 20.0], [80.0, 30.0], [60.0, 50.0]]]


def test_rotated_angle_boxes_map_through_both_frames():
    # angle-90 rect in the parent's rotated frame (100x200 for a 200x100 page)
    # must land in the panel's rotated frame with the same ground truth.
    polys = [Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])]
    # Original-frame target: x 20..60, y 30..70. Forward-rotate corners to build input.
    corners = [(20, 30), (60, 30), (60, 70), (20, 70)]
    rot = [rotate_point(x, y, 90, 200, 100) for x, y in corners]
    rx0, rx1 = min(p[0] for p in rot), max(p[0] for p in rot)
    ry0, ry1 = min(p[1] for p in rot), max(p[1] for p in rot)
    doc = parent_doc(boxes=[{"angle": 90, "horizontal_list": [[rx0, rx1, ry0, ry1]], "free_list": []}])
    out = derive_panel_boxes(doc, polys, 1, (200, 100))
    # Full-canvas panel: same frame, so the box must round-trip exactly.
    assert out["boxes"][0]["horizontal_list"] == [[rx0, rx1, ry0, ry1]]


def test_derive_boxes_for_panel_image_end_to_end(tmp_path):
    Image.new("RGB", (200, 100), "white").save(tmp_path / "p7__1.jpg")
    Image.new("RGB", (100, 100), "white").save(tmp_path / "p7__2.jpg")
    (tmp_path / "p7.panels.json").write_text(json.dumps({
        "image": "p7.jpg", "width": 200, "height": 100,
        "panels": [[[0, 0], [100, 0], [100, 100], [0, 100]],
                   [[100, 0], [200, 0], [200, 100], [100, 100]]],
    }))
    (tmp_path / "p7.boxes.json").write_text(json.dumps(parent_doc(boxes=[
        {"angle": 0, "horizontal_list": [[110, 150, 40, 60]], "free_list": []}])))
    assert derive_boxes_for_panel_image(tmp_path / "p7__2.jpg") is True
    derived = json.loads((tmp_path / "p7__2.boxes.json").read_text())
    assert derived["derived_from_parent"] is True
    assert derived["boxes"][0]["horizontal_list"] == [[10, 50, 40, 60]]
    # Not a panel, or missing inputs: no derivation.
    Image.new("RGB", (10, 10), "white").save(tmp_path / "p8.jpg")
    assert derive_boxes_for_panel_image(tmp_path / "p8.jpg") is False
    Image.new("RGB", (10, 10), "white").save(tmp_path / "p9__1.jpg")
    assert derive_boxes_for_panel_image(tmp_path / "p9__1.jpg") is False
