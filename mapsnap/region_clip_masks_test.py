import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from mapsnap.region_clip_masks import (
    CELL_M,
    GroundGrid,
    Ownership,
    PageWindow,
    assign_units,
    clip_units_to_scans,
    compute_region_clip_masks,
    dissolve_pages,
    ensure_region_maps,
    fill_enclosed_holes,
    fit_line,
    interior_score,
    owned_cells_hull,
    page_footprint,
    panel_outline_mask,
    polygons_of,
    region_map_path,
    region_ownership,
    without_holes,
    without_near_duplicates,
)

# Near the equator: 1 px = 1e-5 degrees = 1.11 m, the same east and north.
DEG_PER_PX = 1e-5
M_PER_PX = 1.11195


def georef(x0: float, width: int = 200, height: int = 200) -> dict:
    """A north-up page of width x height px, x0 px east of the origin."""
    x1 = x0 + width
    corners = [
        [x0 * DEG_PER_PX, 0.0],
        [x1 * DEG_PER_PX, 0.0],
        [x1 * DEG_PER_PX, -height * DEG_PER_PX],
        [x0 * DEG_PER_PX, -height * DEG_PER_PX],
    ]
    return {"width": width, "height": height, "corners": corners}


def write_region_map(image: Path, prob: np.ndarray) -> None:
    """Write a P(region) map where ``mapsnap region`` would."""
    path = region_map_path(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(prob * 255, 0, 255).astype(np.uint8))


def street(points_px: list[tuple[float, float]]) -> dict:
    """A centerline feature through pixel positions of the pages' shared frame."""
    coords = [[x * DEG_PER_PX, -y * DEG_PER_PX] for x, y in points_px]
    return {
        "type": "Feature",
        "properties": {"name": "STREET"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def grid_of(rows: int, cols: int) -> GroundGrid:
    return GroundGrid(0.0, 0.0, 0.0, 0.0, rows, cols)


def test_ground_grid_round_trips_and_rasterizes():
    grid = GroundGrid(-73.9, 40.7, 0.0, 0.0, 10, 10)
    lon, lat = grid.to_lonlat(*grid.to_m(-73.899, 40.701))
    assert (lon, lat) == pytest.approx((-73.899, 40.701))
    cells = grid.cells_of(box(0, 0, 3 * CELL_M, 3 * CELL_M))
    assert cells[:3, :3].all() and not cells[5:, 5:].any()
    assert grid.cell_points(cells)[0] == pytest.approx([0, 0])


def test_panel_outline_mask_blanks_the_crop_outside_the_panel(tmp_path: Path):
    # An L-shaped panel: its crop's top-right quarter is another panel's.
    ring = [[0, 0], [50, 0], [50, 50], [100, 50], [100, 100], [0, 100]]
    panels = {"width": 100, "height": 100, "panels": [ring]}
    (tmp_path / "p3.panels.json").write_text(json.dumps(panels))
    mask = panel_outline_mask(tmp_path / "p3__1.jpg", (100, 100))
    assert mask[75, 75] == 1 and mask[25, 25] == 1
    assert mask[25, 75] == 0
    assert panel_outline_mask(tmp_path / "p3.jpg", (4, 4)).all()


def test_interior_score_rises_with_depth_inside_the_region():
    prob = np.zeros((50, 50), np.float32)
    prob[:, 10:] = 1.0
    score = interior_score(prob, m_per_px=100.0)
    assert score[25, 5] == 0.0
    assert 0.5 < score[25, 11] < score[25, 30] <= 1.0


def test_fit_line_follows_a_point_cloud():
    points = np.array([[0.0, float(y)] for y in range(10)])
    line = fit_line(points, extent=100)
    assert line is not None
    assert line.distance(LineString([(0, -50), (0, 50)])) == pytest.approx(0, abs=1e-9)
    assert fit_line(points[:3], extent=100) is None


def test_without_holes_fills_small_holes_and_slits_big_ones():
    small = Polygon(box(0, 0, 100, 100).exterior, [box(10, 10, 20, 20).exterior.coords])
    assert not without_holes(small).interiors
    assert without_holes(small).area == pytest.approx(10_000)
    big = Polygon(box(0, 0, 300, 300).exterior, [box(50, 50, 250, 250).exterior.coords])
    opened = without_holes(big)
    assert not opened.interiors
    assert opened.area == pytest.approx(300**2 - 200**2, rel=0.01)


def test_polygons_of_flattens_collections():
    two = box(0, 0, 1, 1).union(box(5, 5, 6, 6))
    assert len(polygons_of(two)) == 2
    assert polygons_of(None) == []
    assert polygons_of(LineString([(0, 0), (1, 1)])) == []


def test_page_footprint_is_a_panels_outline(tmp_path: Path):
    grid = GroundGrid(0.0, 0.0, 0.0, 0.0, 0, 0)
    panels = {
        "width": 200,
        "height": 200,
        "panels": [[[0, 0], [100, 0], [100, 50], [0, 50]]],
    }
    (tmp_path / "p2.panels.json").write_text(json.dumps(panels))
    whole = page_footprint(georef(0), tmp_path / "p1.jpg", grid)
    assert whole.area == pytest.approx((200 * M_PER_PX) ** 2, rel=1e-3)
    # The crop is the panel's 100 x 50 px bounding box.
    panel = page_footprint(georef(0, 100, 50), tmp_path / "p2__1.jpg", grid)
    assert panel.area == pytest.approx(100 * 50 * M_PER_PX**2, rel=1e-3)


def ownership_of(owner: np.ndarray) -> Ownership:
    """An Ownership over a 1-cell-per-CELL_M grid from an owner raster."""
    rows, cols = owner.shape
    best = np.where(owner >= 0, 1.0, 0.0).astype(np.float32)
    return Ownership(grid_of(rows, cols), owner, best, {})


def test_assign_units_cuts_a_shared_block_along_its_owners_boundary():
    owner = np.full((40, 40), 0, np.int32)
    owner[:, 20:] = 1  # page 1 owns the east half: far more than MINORITY_M2
    block = box(0, 0, 40 * CELL_M, 40 * CELL_M)
    units = assign_units([block], ownership_of(owner), extent=1000)
    by_page = {page: poly for poly, page in units}
    assert set(by_page) == {0, 1}
    assert by_page[0].bounds[2] == pytest.approx(20 * CELL_M, abs=CELL_M)


def test_assign_units_gives_a_block_whole_to_its_owner():
    owner = np.zeros((10, 10), np.int32)
    owner[0, 0] = 1  # a sliver of another page: under MINORITY_M2
    units = assign_units(
        [box(0, 0, 10 * CELL_M, 10 * CELL_M)], ownership_of(owner), 100
    )
    assert [page for _, page in units] == [0]


def test_clip_units_to_scans_hands_the_overhang_to_the_page_that_shows_it():
    owner = np.zeros((10, 20), np.int32)
    owner[:, 10:] = 1
    unit = box(0, 0, 20 * CELL_M, 10 * CELL_M)
    footprints = [
        box(0, 0, 10 * CELL_M, 10 * CELL_M),
        box(0, 0, 20 * CELL_M, 10 * CELL_M),
    ]
    clipped = clip_units_to_scans([(unit, 0)], footprints, ownership_of(owner))
    by_page = {page: poly for poly, page in clipped}
    assert by_page[0].area == pytest.approx(100 * CELL_M**2)
    assert by_page[1].area == pytest.approx(100 * CELL_M**2)


def test_dissolve_pages_moves_a_stray_to_the_neighbour_that_shows_it():
    main, stray, neighbour = (
        box(0, 0, 50, 50),
        box(100, 0, 110, 50),
        box(60, 0, 100, 50),
    )
    footprints = [box(0, 0, 60, 50), box(55, 0, 120, 50)]
    shapes = dissolve_pages([(main, 0), (stray, 0), (neighbour, 1)], footprints)
    assert shapes[0].area == pytest.approx(main.area)
    assert shapes[1].area == pytest.approx(neighbour.area + stray.area)


def test_dissolve_pages_keeps_a_stray_off_a_neighbour_that_cannot_show_it():
    main, stray, neighbour = (
        box(0, 0, 50, 50),
        box(100, 0, 110, 50),
        box(60, 0, 100, 50),
    )
    footprints = [box(0, 0, 120, 50), box(55, 0, 100, 50)]
    shapes = dissolve_pages([(main, 0), (stray, 0), (neighbour, 1)], footprints)
    assert shapes[1].area == pytest.approx(neighbour.area)


def test_fill_enclosed_holes_gives_a_hole_to_the_best_covering_page():
    size = 60
    grid = grid_of(size, size)
    ring = box(0, 0, size * CELL_M, size * CELL_M).difference(
        box(20 * CELL_M, 20 * CELL_M, 40 * CELL_M, 40 * CELL_M)
    )
    shapes = {0: ring}
    footprints = [box(0, 0, size * CELL_M, size * CELL_M)] * 2
    windows = {
        0: PageWindow(0, size, 0, size, np.full((size, size), 0.2, np.float32)),
        1: PageWindow(0, size, 0, size, np.full((size, size), 0.9, np.float32)),
    }
    owner = np.zeros((size, size), np.int32)
    ownership = Ownership(grid, owner, np.ones((size, size), np.float32), windows)
    fill_enclosed_holes(shapes, footprints, ownership)
    assert shapes[1].area == pytest.approx((20 * CELL_M) ** 2, rel=0.01)


def test_owned_cells_hull_covers_a_pages_cells():
    owner = np.full((20, 20), -1, np.int32)
    owner[5:10, 5:10] = 3
    hull = owned_cells_hull(3, ownership_of(owner), box(0, 0, 100, 100))
    assert hull is not None
    assert hull.bounds == pytest.approx(
        (5 * CELL_M, 5 * CELL_M, 9 * CELL_M, 9 * CELL_M)
    )
    assert owned_cells_hull(7, ownership_of(owner), box(0, 0, 100, 100)) is None


def two_page_volume(tmp_path: Path) -> tuple[list[dict], list[Path]]:
    """Pages at x 0..200 and 150..350 px; each map claims its side of x = 175."""
    georefs = [georef(0), georef(150)]
    paths = [tmp_path / "p1.jpg", tmp_path / "p2.jpg"]
    west = np.zeros((200, 200), np.float32)
    west[:, :175] = 1.0
    east = np.zeros((200, 200), np.float32)
    east[:, 25:] = 1.0
    write_region_map(paths[0], west)
    write_region_map(paths[1], east)
    return georefs, paths


def test_region_ownership_splits_overlap_where_the_maps_do(tmp_path: Path):
    georefs, paths = two_page_volume(tmp_path)
    ownership = region_ownership(georefs, paths)
    grid = ownership.grid
    row = ownership.grid.rows // 2

    def owner_at(x_px: float) -> int:
        x, _ = grid.to_m(x_px * DEG_PER_PX, -100 * DEG_PER_PX)
        return int(ownership.owner[row, int((x - grid.x_min) / CELL_M)])

    assert [owner_at(x) for x in (50, 160, 190, 300)] == [0, 0, 1, 1]


def test_ensure_region_maps_leaves_existing_maps_alone(tmp_path: Path):
    _, paths = two_page_volume(tmp_path)
    before = [region_map_path(p).stat().st_mtime_ns for p in paths]
    ensure_region_maps(paths)  # would load the model if any map were missing
    assert [region_map_path(p).stat().st_mtime_ns for p in paths] == before


def test_compute_region_clip_masks_meet_on_the_street_between_the_pages(tmp_path: Path):
    georefs, paths = two_page_volume(tmp_path)
    streets = [
        street([(175, -50), (175, 250)]),
        street([(-50, -10), (400, -10)]),
        street([(-50, 210), (400, 210)]),
        street([(-10, -50), (-10, 250)]),
        street([(360, -50), (360, 250)]),
    ]
    masks = compute_region_clip_masks(
        georefs, {"type": "FeatureCollection", "features": streets}, raw_paths=paths
    )
    west, east = masks
    assert west is not None and east is not None
    seam = 175 * DEG_PER_PX
    assert west.bounds[2] == pytest.approx(seam, abs=5 * DEG_PER_PX)
    assert east.bounds[0] == pytest.approx(seam, abs=5 * DEG_PER_PX)
    assert west.intersection(east).area < 0.01 * west.area
    # Each mask stays on its own scan.
    assert west.bounds[0] >= -1e-9 and east.bounds[2] <= 350 * DEG_PER_PX + 1e-9


def test_compute_region_clip_masks_needs_images():
    assert compute_region_clip_masks([georef(0)], {}, raw_paths=None) == [None]


def test_without_near_duplicates_drops_centimetre_edges():
    square = Polygon([(0, 0), (10, 0), (10, 0.01), (10, 10), (0, 10)])
    cleaned = without_near_duplicates(square)
    assert len(cleaned.exterior.coords) == 5
    assert cleaned.area == pytest.approx(square.area, rel=1e-3)


def test_without_near_duplicates_keeps_a_polygon_it_would_break():
    tiny = Polygon([(0, 0), (0.03, 0), (0.03, 0.03), (0, 0.03)])
    assert without_near_duplicates(tiny).equals(tiny)
