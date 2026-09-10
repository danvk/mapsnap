"""Tests for the content-region overlap geometry."""

import math
from pathlib import Path

import cv2
import numpy as np

from mapsnap.region_overlap import (
    overlap_over_min,
    posed_region_metres,
    region_polygon_px,
)
from mapsnap.region_predict import region_prob_path

LAT0 = 40.0
KX = 111_320.0 * math.cos(math.radians(LAT0))
KY = 110_540.0


def write_map(volume: Path, stem: str, width: int, height: int, box) -> None:
    prob = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = box
    prob[y0:y1, x0:x1] = 255
    path = region_prob_path(volume, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), prob)


def north_up(shift_east_m: float, m_per_px: float = 0.5) -> np.ndarray:
    return np.array(
        [[m_per_px / KX, 0.0, -74.0 + shift_east_m / KX], [0.0, -m_per_px / KY, LAT0]]
    )


def test_region_polygon_reads_content_and_guards_tiny_maps(tmp_path: Path) -> None:
    write_map(tmp_path, "p3", 200, 100, (20, 10, 180, 90))
    region = region_polygon_px(tmp_path, "p3", (200, 100))
    assert region is not None
    assert abs(region.area - 160 * 80) < 0.05 * 160 * 80
    # 5% of the page: no tiling signal.
    write_map(tmp_path, "p4", 200, 100, (0, 0, 20, 50))
    assert region_polygon_px(tmp_path, "p4", (200, 100)) is None
    # Missing map, and a map at the wrong size.
    assert region_polygon_px(tmp_path, "p5", (200, 100)) is None
    assert region_polygon_px(tmp_path, "p3", (400, 200)) is None


def test_posed_regions_overlap_by_displacement(tmp_path: Path) -> None:
    write_map(tmp_path, "p3", 200, 100, (0, 0, 200, 100))
    region = region_polygon_px(tmp_path, "p3", (200, 100))
    assert region is not None
    origin = (-74.0, LAT0)
    at_home = posed_region_metres(region, north_up(0.0), origin)
    # 200 px * 0.5 m = 100 m wide; shifted half a width east -> 50% overlap.
    shifted = posed_region_metres(region, north_up(50.0), origin)
    # Contours trace pixel centres, so the polygon is half a pixel short per side.
    assert abs(at_home.area - 100 * 50) < 0.03 * 100 * 50
    assert abs(overlap_over_min(at_home, shifted) - 0.5) < 0.02
    apart = posed_region_metres(region, north_up(500.0), origin)
    assert overlap_over_min(at_home, apart) == 0.0
