"""Unit tests for the geometry helpers in mapsnap.split."""

import cv2
import numpy as np
import pytest
from shapely.geometry import box

from mapsnap.split import (
    BORDER_PX,
    assemble_panels,
    crop_border,
    merge_collinear,
    panel_basename,
    panel_compactness,
    seg_angle_deg,
    segment_thickness,
)

# --- panel_basename ---


def test_panel_basename_strips_raw_and_scaled():
    from pathlib import Path

    assert panel_basename(Path("p45.raw.jpg")) == "p45"
    assert panel_basename(Path("p195.scaled.jpg")) == "p195"
    assert panel_basename(Path("dir/champaign-p20.jpg")) == "champaign-p20"
    assert panel_basename(Path("p8.jpg")) == "p8"


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
