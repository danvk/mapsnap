"""Unit tests for the geometry helpers in mapsnap.split."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import box

from mapsnap.split import (
    BORDER_PX,
    assemble_panels,
    crop_border,
    invalidate_changed_panels,
    is_keymap_sheet,
    keymap_split_rejection,
    merge_collinear,
    order_panels,
    panel_basename,
    panel_compactness,
    panels_json_path,
    read_panels_json,
    remove_panel_sidecars,
    remove_split_outputs,
    rings_match,
    seg_angle_deg,
    segment_thickness,
    write_panels_json,
)

# --- panel_basename ---


def test_panel_basename_strips_raw_and_scaled():
    assert panel_basename(Path("p45.raw.jpg")) == "p45"
    assert panel_basename(Path("p195.scaled.jpg")) == "p195"
    assert panel_basename(Path("dir/champaign-p20.jpg")) == "champaign-p20"
    assert panel_basename(Path("p8.jpg")) == "p8"


# --- panels.json round-trip and cleanup ---


def test_write_panels_json_records_ordered_rings(tmp_path):
    image_path = tmp_path / "p7.jpg"
    panels = [box(0, 0, 50, 100), box(50, 0, 120, 100)]
    out_path = write_panels_json(image_path, panels, width=120, height=100)

    assert out_path == panels_json_path(image_path) == tmp_path / "p7.panels.json"
    data = read_panels_json(out_path)
    assert data["image"] == "p7.jpg"
    assert (data["width"], data["height"]) == (120, 100)
    assert len(data["panels"]) == 2
    # First ring is the left panel; rings are closed (first point repeats).
    xs = [pt[0] for pt in data["panels"][0]]
    assert min(xs) == 0 and max(xs) == 50
    assert data["panels"][0][0] == data["panels"][0][-1]


def test_remove_split_outputs_deletes_panels_and_json(tmp_path):
    (tmp_path / "p2__1.jpg").touch()
    (tmp_path / "p2__2.jpg").touch()
    (tmp_path / "p2.panels.json").touch()
    # An unrelated page must be left alone.
    (tmp_path / "p3__1.jpg").touch()

    remove_split_outputs(tmp_path / "p2.jpg")

    assert not (tmp_path / "p2__1.jpg").exists()
    assert not (tmp_path / "p2__2.jpg").exists()
    assert not (tmp_path / "p2.panels.json").exists()
    assert (tmp_path / "p3__1.jpg").exists()


# --- panel ordering (#379) and sidecar invalidation ---


def test_order_panels_two_panels_bottom_left_first():
    """OIM numbers the panel holding the sheet's bottom-left corner first (200/200)."""
    top, bottom = box(0, 0, 100, 50), box(0, 50, 100, 100)
    assert order_panels([top, bottom], height=100) == [bottom, top]
    left, right = box(0, 0, 50, 100), box(50, 0, 100, 100)
    assert order_panels([right, left], height=100) == [left, right]
    # A bottom-left inset is numbered first; an inset in any other corner is
    # second, behind the big panel that owns the bottom-left corner.
    big_with_notch = box(0, 0, 100, 100).difference(box(0, 70, 30, 100))
    inset_bottom_left = box(0, 70, 30, 100)
    assert order_panels([big_with_notch, inset_bottom_left], height=100) == [
        inset_bottom_left,
        big_with_notch,
    ]
    big_notch_top_right = box(0, 0, 100, 100).difference(box(70, 0, 100, 30))
    inset_top_right = box(70, 0, 100, 30)
    assert order_panels([inset_top_right, big_notch_top_right], height=100) == [
        big_notch_top_right,
        inset_top_right,
    ]


def test_order_panels_reading_order_otherwise():
    # Three or more panels: OIM's order is unpredictable, so reading order stays.
    a, b, c = box(0, 0, 50, 50), box(50, 0, 100, 50), box(0, 50, 100, 100)
    assert order_panels([c, b, a], height=100) == [a, b, c]
    # Without a height, two panels also fall back to reading order.
    top, bottom = box(0, 0, 100, 50), box(0, 50, 100, 100)
    assert order_panels([bottom, top]) == [top, bottom]


def test_rings_match_compares_polygons_not_vertex_lists():
    ring = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0]]
    nudged = [[0.3, 0.0], [100.0, 0.4], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0]]
    rotated_start = [[100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0], [100.0, 0.0]]
    reversed_ring = list(reversed(ring))
    moved = [[0.0, 0.0], [100.0, 0.0], [100.0, 60.0], [0.0, 60.0], [0.0, 0.0]]
    assert rings_match(ring, nudged)
    assert rings_match(ring, rotated_start)
    assert rings_match(ring, reversed_ring)
    assert not rings_match(ring, moved)
    assert not rings_match(ring, ring[:2])


def test_remove_panel_sidecars_takes_every_derived_file_of_one_panel(tmp_path):
    for name in (
        "p5__1.boxes.json",
        "p5__1.streets.json",
        "p5__1.txt",
        "p5__1.georef.json",
        "p5__1.georef-final.json",
        "p5__1.georef-snap.json",
        "p5__1.jpg",  # the image is remove_split_outputs' job
        "p5__2.streets.json",  # the sibling panel is untouched
        "p50__1.streets.json",  # a different page sharing the prefix is untouched
    ):
        (tmp_path / name).touch()
    removed = {path.name for path in remove_panel_sidecars(tmp_path / "p5.jpg", 1)}
    assert removed == {
        "p5__1.boxes.json",
        "p5__1.streets.json",
        "p5__1.txt",
        "p5__1.georef.json",
        "p5__1.georef-final.json",
        "p5__1.georef-snap.json",
    }
    assert (tmp_path / "p5__1.jpg").exists()
    assert (tmp_path / "p5__2.streets.json").exists()
    assert (tmp_path / "p50__1.streets.json").exists()


def test_invalidate_changed_panels_keeps_unchanged_reads(tmp_path):
    """A re-split drops the reads of panels whose ring changed and only those.

    `mapsnap ocr --resume` keys on a streets.json existing, so a panel that was
    renumbered (#379) or re-cut would otherwise keep reads taken from a
    different picture.
    """
    old: list[list[list[float]]] = [
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0]],
        [[0.0, 50.0], [100.0, 50.0], [100.0, 100.0], [0.0, 100.0], [0.0, 50.0]],
        [[0.0, 100.0], [100.0, 100.0], [100.0, 150.0], [0.0, 150.0], [0.0, 100.0]],
    ]
    # Panels 1 and 2 swap (renumbering); panel 3 disappears.
    new = [old[1], old[0]]
    for index in (1, 2, 3):
        (tmp_path / f"p9__{index}.streets.json").touch()
    (tmp_path / "p9__2.georef.json").touch()
    changed = invalidate_changed_panels(tmp_path / "p9.jpg", old, new)
    assert changed == [1, 2, 3]
    assert not any((tmp_path / f"p9__{i}.streets.json").exists() for i in (1, 2, 3))
    assert not (tmp_path / "p9__2.georef.json").exists()

    # Identical rings keep everything.
    (tmp_path / "p9__1.streets.json").touch()
    assert invalidate_changed_panels(tmp_path / "p9.jpg", new, new) == []
    assert (tmp_path / "p9__1.streets.json").exists()


# --- key-map sheets refuse bad cuts (#276) ---


def test_is_keymap_sheet_mirrors_the_key_map_candidates():
    from mapsnap.keymap.identify import is_letter_page

    for stem in ("p0", "p0b", "p0L", "p1", "p1N", "p1a", "pa", "pb"):
        assert is_keymap_sheet(stem), stem
    for stem in ("p2", "p57", "p1499m", "p125", "covr", "ind1"):
        assert not is_keymap_sheet(stem), stem
    assert is_letter_page("pa") and is_keymap_sheet("pa")


def test_keymap_split_rejection_accepts_edge_boxes_and_refuses_notches():
    sheet = box(0, 0, 1000, 2000)
    key_box = box(770, 1640, 1000, 2000)  # bottom-right corner: two edges
    inset = box(0, 1360, 320, 2000)  # bottom-left corner: two edges
    notch = box(140, 0, 320, 580)  # hangs off the top edge only
    assert (
        keymap_split_rejection([sheet.difference(key_box), key_box], 1000, 2000) is None
    )
    assert keymap_split_rejection([sheet.difference(inset), inset], 1000, 2000) is None
    reason = keymap_split_rejection(
        [sheet.difference(notch).difference(key_box), notch, key_box], 1000, 2000
    )
    assert reason is not None and "panel 2" in reason and "1 sheet edge" in reason
    # A panel the size of the sheet is never a cut-away, whatever its edges say.
    assert keymap_split_rejection([sheet], 1000, 2000) is None


# --- crop_border ---


def test_crop_border_removes_border():
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    cropped = crop_border(arr, border=BORDER_PX)
    assert cropped.shape == (300 - 2 * BORDER_PX, 400 - 2 * BORDER_PX, 3)


def test_crop_border_raises_when_too_small():
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        crop_border(arr, border=BORDER_PX)


# --- seg_angle_deg ---


def test_seg_angle_deg():
    assert seg_angle_deg(np.array([0, 0, 10, 0])) == pytest.approx(0.0)
    assert seg_angle_deg(np.array([0, 0, 0, 10])) == pytest.approx(90.0)
    assert seg_angle_deg(np.array([0, 0, 10, 10])) == pytest.approx(45.0)
    # Angle is folded into [0, 180), so direction doesn't matter.
    assert seg_angle_deg(np.array([10, 0, 0, 0])) == pytest.approx(0.0)


# --- panel_compactness ---


def test_panel_compactness_square_vs_sliver():
    square = box(0, 0, 100, 100)
    # A square's Polsby-Popper score is pi/4 ~ 0.785.
    assert panel_compactness(square) == pytest.approx(np.pi / 4, abs=1e-6)
    sliver = box(0, 0, 100, 4)
    assert panel_compactness(sliver) < panel_compactness(square)


def test_panel_compactness_zero_perimeter():
    assert panel_compactness(box(0, 0, 0, 0)) == 0.0


# --- segment_thickness ---


def _line_distance_transform(thickness: int, size: int = 100) -> np.ndarray:
    """Distance transform of a horizontal ink band `thickness` px tall, centered."""
    binary = np.zeros((size, size), dtype=np.uint8)
    top = size // 2 - thickness // 2
    binary[top : top + thickness, 10 : size - 10] = 255
    return cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)


def test_segment_thickness_thick_vs_thin():
    size = 100
    seg = (10.0, size / 2, size - 10.0, size / 2)  # along the band's center
    thick = segment_thickness(_line_distance_transform(8, size), seg)
    thin = segment_thickness(_line_distance_transform(2, size), seg)
    assert thick >= 5.0  # a real divider passes the thickness filter
    assert thin < 5.0  # a grid/lot line does not


def test_segment_thickness_degenerate_segment():
    dist = _line_distance_transform(8)
    assert segment_thickness(dist, (5.0, 5.0, 5.0, 5.0)) == 0.0


# --- merge_collinear ---


def test_merge_collinear_joins_collinear_with_gap():
    lines = np.array([[0, 10, 40, 10], [50, 10, 90, 10]], dtype=float)
    merged = merge_collinear(lines, gap_tol_px=100.0)
    assert len(merged) == 1
    x0, y0, x1, y1 = merged[0]
    assert min(x0, x1) == pytest.approx(0.0)
    assert max(x0, x1) == pytest.approx(90.0)
    assert y0 == pytest.approx(10.0) and y1 == pytest.approx(10.0)


def test_merge_collinear_keeps_parallel_offset_segments_separate():
    # Same direction but ~70px apart perpendicular: beyond MERGE_PERP_PX, not merged.
    lines = np.array([[0, 10, 40, 10], [0, 80, 40, 80]], dtype=float)
    assert len(merge_collinear(lines, gap_tol_px=100.0)) == 2


# --- assemble_panels ---


def test_assemble_panels_single_real_panel_covers_whole_page():
    # One real panel (97%) plus a sub-threshold sliver → fall back to a full-page panel.
    faces = [box(0, 0, 100, 97), box(0, 97, 100, 100)]
    panels = assemble_panels(faces, 100, 100)
    assert len(panels) == 1
    assert panels[0].area == pytest.approx(100 * 100)


def test_assemble_panels_glues_sliver_and_tiles_page():
    # Two real panels with a thin sliver between them: 100% coverage, sliver absorbed.
    faces = [box(0, 0, 48, 100), box(52, 0, 100, 100), box(48, 0, 52, 100)]
    panels = assemble_panels(faces, 100, 100)
    assert len(panels) == 2
    assert all(p.geom_type == "Polygon" for p in panels)
    assert sum(p.area for p in panels) == pytest.approx(100 * 100)


def test_assemble_panels_over_fragmented_falls_back_to_single():
    # Two real panels cover only 80%; the rest is sub-threshold slivers → single panel.
    faces = [box(0, 0, 100, 40), box(0, 40, 100, 80)]
    faces += [box(0, 80 + 4 * i, 100, 84 + 4 * i) for i in range(5)]
    panels = assemble_panels(faces, 100, 100)
    assert len(panels) == 1
    assert panels[0].area == pytest.approx(100 * 100)


def test_assemble_panels_keeps_a_panel_shaped_small_inset():
    # A compact corner inset in the small band (3% of the page, aspect 1.2,
    # min dimension 16% of the page side) is a real panel, not a sliver --
    # the panel-billed disaster class the 0.05 hard floor used to glue away.
    faces = [
        box(0, 0, 100, 84),
        box(0, 84, 18, 100),
        box(18, 84, 100, 100),
    ]
    panels = assemble_panels(faces, 100, 100)
    assert len(panels) == 3


def test_assemble_panels_glues_an_edge_strip_in_the_small_band():
    # Same area as a keepable inset, but a 4x100 edge strip (aspect 25):
    # the nashville over-split class. Shape guard glues it.
    faces = [box(0, 0, 4, 100), box(4, 0, 100, 100)]
    panels = assemble_panels(faces, 100, 100)
    assert len(panels) == 1
    assert panels[0].area == pytest.approx(100 * 100)
