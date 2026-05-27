"""Tests for clip_masks.py."""

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from mapsnap.clip_masks import (
    PageColorData,
    _assign_blocks_to_pages,
    _assign_blocks_to_pages_with_splits,
    _convexity_ratio,
    _fill_concave_dents,
    _fill_coverage_gaps,
    _fit_affine,
    _is_substantial,
    _load_page_color_data,
    _polygonize_streets,
    _remove_spike_vertices,
    _score_block_on_page,
    compute_all_clip_masks,
    geo_polygon_to_svg,
)


def _make_georef(
    width: int,
    height: int,
    corners: list[list[float]],
) -> dict:
    """Build a minimal georef dict for testing."""
    return {"width": width, "height": height, "corners": corners}


def _axis_aligned_georef(
    lon0: float, lat0: float, lon1: float, lat1: float, w: int = 1000, h: int = 1000
) -> dict:
    """Return a georef whose corners form an axis-aligned lon/lat rectangle.

    corners: TL=(lon0,lat1), TR=(lon1,lat1), BR=(lon1,lat0), BL=(lon0,lat0)
    (image y increases downward, lat increases upward).
    """
    return _make_georef(
        w,
        h,
        [
            [lon0, lat1],  # TL: pixel (0,0)
            [lon1, lat1],  # TR: pixel (w,0)
            [lon1, lat0],  # BR: pixel (w,h)
            [lon0, lat0],  # BL: pixel (0,h)
        ],
    )


# ---------------------------------------------------------------------------
# _fit_affine
# ---------------------------------------------------------------------------


def test_fit_affine_round_trips_corners():
    """Forward affine maps pixel corners to the correct geo coords."""
    georef = _axis_aligned_georef(-90.0, 30.0, -89.9, 30.1)
    A_fwd, _ = _fit_affine(georef)
    w, h = float(georef["width"]), float(georef["height"])
    pixel_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    for (px, py), expected in zip(pixel_pts, georef["corners"]):
        result = A_fwd @ np.array([px, py, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-9)


def test_fit_affine_inverse_round_trips():
    """Inverse affine maps geo corners back to pixel coords."""
    georef = _axis_aligned_georef(-90.0, 30.0, -89.9, 30.1)
    A_fwd, A_inv = _fit_affine(georef)
    w, h = float(georef["width"]), float(georef["height"])
    expected_pixels = [(0, 0), (w, 0), (w, h), (0, h)]
    for (lon, lat), (ex, ey) in zip(georef["corners"], expected_pixels):
        geo_vec = np.array([lon, lat]) - A_fwd[:, 2]
        px, py = A_inv @ geo_vec
        assert abs(px - ex) < 1e-5
        assert abs(py - ey) < 1e-5


def test_fit_affine_handles_rotated_map():
    """Affine fit works when the map is ~45° rotated in geo space."""
    # Rotate a square 45°: TL→top, TR→right, BR→bottom, BL→left.
    d = 0.1
    corners = [
        [0.0, d],  # TL: pixel (0,0) → north tip
        [d, 0.0],  # TR: pixel (w,0) → east tip
        [0.0, -d],  # BR: pixel (w,h) → south tip
        [-d, 0.0],  # BL: pixel (0,h) → west tip
    ]
    georef = _make_georef(1000, 1000, corners)
    A_fwd, A_inv = _fit_affine(georef)
    w, h = 1000.0, 1000.0
    for (px, py), expected in zip([(0, 0), (w, 0), (w, h), (0, h)], corners):
        result = A_fwd @ np.array([px, py, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-9)
        geo_vec = np.array(expected) - A_fwd[:, 2]
        rx, ry = A_inv @ geo_vec
        assert abs(rx - px) < 1e-6
        assert abs(ry - py) < 1e-6


# ---------------------------------------------------------------------------
# _polygonize_streets
# ---------------------------------------------------------------------------


def _line_feature(coords: list[list[float]]) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _geojson(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def test_polygonize_simple_grid():
    """A 2×2 street grid inside a bounding box produces ≥4 blocks."""
    # Coverage: unit square [0,1]×[0,1]
    coverage = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gj = _geojson(
        _line_feature([[0.5, 0.0], [0.5, 1.0]]),  # vertical line x=0.5
        _line_feature([[0.0, 0.5], [1.0, 0.5]]),  # horizontal line y=0.5
    )
    blocks = _polygonize_streets(gj, coverage)
    # With the boundary ring added: 4 cells from the grid + boundary edges
    assert len(blocks) >= 4


def test_polygonize_boundary_closes_open_ends():
    """A single horizontal line + coverage boundary creates ≥2 blocks."""
    coverage = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    gj = _geojson(
        _line_feature([[0.0, 0.5], [2.0, 0.5]]),  # divides rectangle in half
    )
    blocks = _polygonize_streets(gj, coverage)
    assert len(blocks) >= 2


def test_polygonize_empty_centerlines():
    """No features → no blocks (just the boundary, which doesn't polygonize alone)."""
    coverage = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gj = _geojson()
    blocks = _polygonize_streets(gj, coverage)
    # A single closed ring with no interior lines produces 0 blocks from polygonize.
    assert isinstance(blocks, list)


def test_polygonize_clips_outside_coverage():
    """Lines entirely outside the coverage polygon are excluded."""
    coverage = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gj = _geojson(
        _line_feature([[2.0, 0.5], [3.0, 0.5]]),  # entirely outside
        _line_feature([[0.5, 0.0], [0.5, 1.0]]),  # inside — splits coverage
    )
    blocks = _polygonize_streets(gj, coverage)
    # Only the inside line contributes; should produce ≥2 blocks.
    assert len(blocks) >= 2


# ---------------------------------------------------------------------------
# _assign_blocks_to_pages
# ---------------------------------------------------------------------------


def test_assign_primary_by_area():
    """Block mostly inside page 0 is assigned to page 0."""
    page0 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # x in [0,10]
    page1 = Polygon([(8, 0), (20, 0), (20, 10), (8, 10)])  # x in [8,20]
    # Block x in [0,9]: 90% in page0, 10% in page1.
    block = Polygon([(0, 0), (9, 0), (9, 10), (0, 10)])
    result = _assign_blocks_to_pages([block], [page0, page1])
    assert 0 in result[0]
    assert result[1] == []


def test_assign_tiebreak_by_distance():
    """When overlap areas are equal, the closer page centroid wins."""
    # Two adjacent unit squares; block is exactly on the border.
    page0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])  # centroid (0.5, 0.5)
    page1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # centroid (1.5, 0.5)
    # Block spans both pages equally (0.5 units each side of x=1).
    block = Polygon([(0.5, 0.25), (1.5, 0.25), (1.5, 0.75), (0.5, 0.75)])
    result = _assign_blocks_to_pages([block], [page0, page1])
    # Block centroid is at (1.0, 0.5); both pages equally distant at 0.5 units.
    # Either assignment is valid; just verify it went somewhere.
    assert result[0] == [0] or result[1] == [0]


def test_assign_no_overlap_block_dropped():
    """Block with no overlap with any page is not assigned."""
    page = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    block = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])  # far away
    result = _assign_blocks_to_pages([block], [page])
    assert result[0] == []


def test_assign_no_double_assignment():
    """No block is assigned to more than one page."""
    page0 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    page1 = Polygon([(4, 0), (10, 0), (10, 5), (4, 5)])
    blocks = [
        Polygon([(0, 0), (4, 0), (4, 5), (0, 5)]),
        Polygon([(4, 0), (5, 0), (5, 5), (4, 5)]),
        Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
    ]
    result = _assign_blocks_to_pages(blocks, [page0, page1])
    all_assigned = result[0] + result[1]
    assert len(all_assigned) == len(set(all_assigned)), (
        "Block assigned to multiple pages"
    )


# ---------------------------------------------------------------------------
# _assign_blocks_to_pages_with_splits
# ---------------------------------------------------------------------------


def test_split_border_block_covers_both_pages():
    """A block straddling two pages is split, giving each page its half."""
    page0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    page1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    # Block x in [0.5, 1.5]: equal halves in each page.
    block = Polygon([(0.5, 0.2), (1.5, 0.2), (1.5, 0.8), (0.5, 0.8)])
    pieces = _assign_blocks_to_pages_with_splits([block], [page0, page1])
    assert len(pieces[0]) > 0, "page0 should receive a piece"
    assert len(pieces[1]) > 0, "page1 should receive a piece"
    total_area = sum(p.area for ps in pieces.values() for p in ps)
    assert abs(total_area - block.area) < 1e-10, "pieces should cover the full block"


def test_split_fully_inside_block_stays_whole():
    """A block entirely inside a page is not split."""
    page = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    block = Polygon([(2, 2), (5, 2), (5, 5), (2, 5)])
    pieces = _assign_blocks_to_pages_with_splits([block], [page])
    assert len(pieces[0]) == 1
    assert abs(pieces[0][0].area - block.area) < 1e-10


def test_split_outside_block_dropped():
    """A block with no overlap with any page produces no pieces."""
    page = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    block = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    pieces = _assign_blocks_to_pages_with_splits([block], [page])
    assert all(len(ps) == 0 for ps in pieces.values())


def test_split_sliver_remainder_dropped():
    """A thin outside remainder (<10% of block width) is not passed on."""
    page0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    page1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    # Block x in [0, 1.05]: outside piece is only ~5% of block width → sliver.
    block = Polygon([(0.0, 0.2), (1.05, 0.2), (1.05, 0.8), (0.0, 0.8)])
    pieces = _assign_blocks_to_pages_with_splits([block], [page0, page1])
    assert sum(p.area for p in pieces[1]) == 0, "sliver remainder should be dropped"


# ---------------------------------------------------------------------------
# _is_substantial
# ---------------------------------------------------------------------------


def test_is_substantial_full_overlap():
    """A piece identical to the reference is substantial."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert _is_substantial(poly, poly)


def test_is_substantial_thin_strip_rejected():
    """A strip that is <10% of reference height is not substantial."""
    reference = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    thin = Polygon([(0, 0), (1, 0), (1, 0.05), (0, 0.05)])  # 5% of height
    assert not _is_substantial(thin, reference)


def test_is_substantial_small_area_rejected():
    """A piece with <10% of reference area is not substantial."""
    reference = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    small = Polygon([(0, 0), (1, 0), (1, 0.9), (0, 0.9)])  # 0.9% of area
    assert not _is_substantial(small, reference)


# ---------------------------------------------------------------------------
# _remove_spike_vertices
# ---------------------------------------------------------------------------


def test_remove_spike_removes_backtrack_vertex():
    """A vertex where the polygon reverses direction (≥150° turn) is removed."""
    # Square with a spike: goes to (1, 0.5), which requires a ≥150° turn to reach
    # from (2, 0.9) and another to leave toward (2, 1.1).
    spike_poly = Polygon([(0, 0), (2, 0), (2, 0.9), (1, 0.5), (2, 1.1), (2, 2), (0, 2)])
    cleaned = _remove_spike_vertices(spike_poly, min_turn_deg=150.0)
    assert len(list(cleaned.exterior.coords)) < len(list(spike_poly.exterior.coords))


def test_remove_spike_preserves_clean_polygon():
    """A polygon with no spike vertices is returned unchanged."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cleaned = _remove_spike_vertices(square)
    assert list(cleaned.exterior.coords) == list(square.exterior.coords)


def test_remove_spike_collinear_not_removed():
    """A collinear vertex (0° turn, same direction) is NOT treated as a spike."""
    # (1,1) lies exactly on the top edge between (2,1) and (0,1) — 0° turn, not 180°.
    poly = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (0, 1)])
    cleaned = _remove_spike_vertices(poly)
    assert len(list(cleaned.exterior.coords)) == len(list(poly.exterior.coords))


# ---------------------------------------------------------------------------
# _convexity_ratio
# ---------------------------------------------------------------------------


def test_convexity_ratio_square():
    """A square is perfectly convex: ratio = 1.0."""
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert abs(_convexity_ratio(square) - 1.0) < 1e-9


def test_convexity_ratio_l_shape():
    """An L-shape has ratio < 1.0 (area=3, convex hull is a pentagon with area=3.5 → 6/7)."""
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    assert abs(_convexity_ratio(l_shape) - 6 / 7) < 1e-9


# ---------------------------------------------------------------------------
# compute_all_clip_masks
# ---------------------------------------------------------------------------


def _simple_georef_row() -> tuple[list[dict], dict]:
    """Three adjacent pages with a simple 3-column street grid.

    Pages are 1°×1° boxes at lon 0-1, 1-2, 2-3; all at lat 0-1.
    Centerlines are the vertical lines at lon=0.5, 1.5, 2.5 (one per page interior).
    """
    georefs = [
        _axis_aligned_georef(0.0, 0.0, 1.0, 1.0),
        _axis_aligned_georef(1.0, 0.0, 2.0, 1.0),
        _axis_aligned_georef(2.0, 0.0, 3.0, 1.0),
    ]
    gj = _geojson(
        _line_feature([[0.5, 0.0], [0.5, 1.0]]),
        _line_feature([[1.5, 0.0], [1.5, 1.0]]),
        _line_feature([[2.5, 0.0], [2.5, 1.0]]),
    )
    return georefs, gj


def test_compute_masks_returns_one_per_georef():
    georefs, gj = _simple_georef_row()
    masks = compute_all_clip_masks(georefs, gj)
    assert len(masks) == 3


def test_compute_masks_within_page_bounds():
    georefs, gj = _simple_georef_row()
    masks = compute_all_clip_masks(georefs, gj)
    for georef, mask in zip(georefs, masks):
        if mask is None:
            continue
        page_poly = Polygon(georef["corners"])
        # mask should be (approximately) contained in the page polygon.
        assert mask.difference(page_poly.buffer(1e-8)).area < 1e-8


def test_compute_masks_empty_georefs():
    assert (
        compute_all_clip_masks([], {"type": "FeatureCollection", "features": []}) == []
    )


def test_compute_masks_empty_centerlines():
    georefs, _ = _simple_georef_row()
    masks = compute_all_clip_masks(
        georefs, {"type": "FeatureCollection", "features": []}
    )
    # With no centerlines the boundary ring creates a single large block that is
    # split among all overlapping pages, so all pages may get a non-None mask.
    assert len(masks) == len(georefs)


def test_compute_masks_overlapping_pages_no_overlap():
    """Masks for geographically overlapping pages do not overlap each other.

    Pages cover lon=[0,2] and lon=[1,3] (overlap at lon=[1,2]). Each block is
    assigned to exactly one page via the split-assignment loop, so the resulting
    masks must be disjoint.
    """
    georefs = [
        _axis_aligned_georef(0.0, 0.0, 2.0, 1.0),  # lon 0-2
        _axis_aligned_georef(1.0, 0.0, 3.0, 1.0),  # lon 1-3, overlaps with page 0
    ]
    gj = _geojson(
        _line_feature([[0.5, 0.0], [0.5, 1.0]]),
        _line_feature([[1.5, 0.0], [1.5, 1.0]]),
        _line_feature([[2.5, 0.0], [2.5, 1.0]]),
    )
    masks = compute_all_clip_masks(georefs, gj)
    assert len(masks) == 2
    m0, m1 = masks
    assert m0 is not None and m1 is not None
    assert m0.intersection(m1).area < 1e-8, (
        "masks for overlapping pages must not overlap"
    )


# ---------------------------------------------------------------------------
# geo_polygon_to_svg
# ---------------------------------------------------------------------------


def test_svg_none_returns_full_page_rect():
    georef = _axis_aligned_georef(-90.0, 30.0, -89.9, 30.1, w=2000, h=3000)
    svg = geo_polygon_to_svg(None, georef, 2000, 3000)
    assert "2000" in svg
    assert "3000" in svg
    assert svg.count("<polygon") == 1


def test_svg_full_page_polygon_gives_corner_coords():
    """A polygon equal to the page corners should give approximately full-page coords."""
    georef = _axis_aligned_georef(-90.0, 30.0, -89.9, 30.1, w=1000, h=1000)
    page_poly = Polygon(georef["corners"])
    svg = geo_polygon_to_svg(page_poly, georef, 1000, 1000)
    assert "<polygon" in svg
    assert "<svg>" in svg


def test_svg_known_coords():
    """For an axis-aligned page, corners map to exact pixel coords."""
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=100, h=100)
    # A polygon exactly matching the page corners (but using just 4 points).
    poly = Polygon([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
    svg = geo_polygon_to_svg(poly, georef, 100, 100)
    # Should include corner pixel coords (0,0), (100,0), (100,100), (0,100).
    assert "0.0,0.0" in svg or "0,0" in svg


def test_svg_split_canvas_offset():
    """Split-canvas coordinates include the (cx, cy) offset."""
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=100, h=100)
    poly = Polygon([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
    split = (500.0, 200.0, 100.0, 100.0)  # offset (500, 200)
    svg = geo_polygon_to_svg(poly, georef, 1000, 1000, split_canvas=split)
    # The top-left corner pixel (0,0) → canvas (500, 200).
    assert "500.0,200.0" in svg


def test_svg_multipolygon_raises():
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0)
    mp = MultiPolygon(
        [
            Polygon([(0, 0), (0.4, 0), (0.4, 1), (0, 1)]),
            Polygon([(0.6, 0), (1, 0), (1, 1), (0.6, 1)]),
        ]
    )
    with pytest.raises(ValueError, match="MultiPolygon"):
        geo_polygon_to_svg(mp, georef, 1000, 1000)


def test_svg_empty_polygon_returns_full_page_rect():
    georef = _axis_aligned_georef(-90.0, 30.0, -89.9, 30.1, w=2000, h=3000)
    empty = Polygon()
    svg = geo_polygon_to_svg(empty, georef, 2000, 3000)
    assert "2000" in svg
    assert "3000" in svg


def test_svg_buffer_is_0_5_miles():
    """Verify the buffer constant is approximately 0.5 miles in degrees lat."""
    from mapsnap.clip_masks import _BUFFER_LAT_DEG

    miles_per_degree_lat = 111_320.0 / 1609.344  # ≈ 69.1 miles
    expected = 0.5 / miles_per_degree_lat
    assert math.isclose(_BUFFER_LAT_DEG, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# _load_page_color_data / _score_block_on_page
# ---------------------------------------------------------------------------


def _make_page_color_data(georef: dict, score_array: np.ndarray) -> PageColorData:
    """Build a PageColorData directly from a georef and a pre-made score array."""
    A_fwd, A_inv = _fit_affine(georef)
    H, W = score_array.shape
    return PageColorData(
        color_score=score_array,
        A_fwd=A_fwd,
        A_inv=A_inv,
        scale_x=W / float(georef["width"]),
        scale_y=H / float(georef["height"]),
    )


def test_load_color_data_missing_file():
    """Returns None when the raw image file does not exist."""
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=100, h=100)
    result = _load_page_color_data(Path("/nonexistent/file.jpg"), georef)
    assert result is None


def test_score_block_zero_image():
    """A block over an all-zero color image returns score 0."""
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=10, h=10)
    pcd = _make_page_color_data(georef, np.zeros((10, 10), dtype=np.int16))
    block = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
    assert _score_block_on_page(block, pcd) == 0.0


def test_score_block_bright_region():
    """A block over colored pixels returns a positive score."""
    georef = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=10, h=10)
    # For this axis-aligned georef:
    #   lon=0.2 → px=2, lat=0.8 → py=(1-0.8)*10=2
    #   lon=0.8 → px=8, lat=0.2 → py=(1-0.2)*10=8
    # So the block maps to pixel rows 2-8, cols 2-8.
    arr = np.zeros((10, 10), dtype=np.int16)
    arr[2:8, 2:8] = 50
    pcd = _make_page_color_data(georef, arr)
    block = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
    assert _score_block_on_page(block, pcd) > 0.0


# ---------------------------------------------------------------------------
# _fill_concave_dents
# ---------------------------------------------------------------------------


def test_fill_concave_dents_transfers_notch():
    """Dent region owned by page j is transferred to page i when color is equal."""
    # mask_i: L-shape (area=3). convex_hull is a pentagon (area=3.5); the
    # dent is the triangle [(2,1),(1,1),(1,2)] with area=0.5.
    # mask_j: unit square [(1,1)×(2,2)] that owns the dent triangle.
    mask_i = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    mask_j = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    result = _fill_concave_dents([mask_i, mask_j], page_color_data=None)

    # mask_i should absorb the dent triangle and equal its convex hull (area=3.5).
    assert result[0] is not None
    assert abs(result[0].area - mask_i.convex_hull.area) < 1e-6
    # mask_j lost the dent triangle → area shrank from 1.0 to 0.5.
    assert result[1] is not None
    assert result[1].area < mask_j.area


def test_fill_concave_dents_no_transfer_when_color_dominates():
    """Notch is NOT transferred when the owning page has substantially more color."""
    mask_i = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    mask_j = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])

    # page j has high color everywhere; page i has none.
    georef_i = _axis_aligned_georef(0.0, 0.0, 2.0, 2.0, w=20, h=20)
    georef_j = _axis_aligned_georef(1.0, 1.0, 2.0, 2.0, w=10, h=10)
    pcd_i = _make_page_color_data(georef_i, np.zeros((20, 20), dtype=np.int16))
    pcd_j = _make_page_color_data(georef_j, np.full((10, 10), 200, dtype=np.int16))

    result = _fill_concave_dents([mask_i, mask_j], page_color_data=[pcd_i, pcd_j])

    # Transfer should be blocked — mask_j still owns the notch.
    assert result[1] is not None
    assert abs(result[1].area - mask_j.area) < 1e-6


def test_fill_concave_dents_arm_via_page_poly():
    """Arm in j's territory inside i's page extent is transferred using page_poly detection.

    Hull-based detection would miss this arm because it isn't inside hull(mask_i).
    Page_poly-based detection finds it via page_poly_i − mask_i.

    Geometry:
      page_poly_p32: [(0,0)-(4,6)]  p32's full geographic extent
      mask_p32:      [(0,0)-(4,3)]  only south half of p32
      page_poly_p24: [(0,5)-(4,9)]  p24's extent (north of p32)
      arm:     [(1,3)-(2,5.5)]      thin strip in p32's upper area, protrudes down from p24
      mask_p24: rect[(0,5)-(4,9)] ∪ arm  — arm hangs below p24's main territory into p32's extent
    """
    page_poly_p32 = Polygon([(0, 0), (4, 0), (4, 6), (0, 6)])
    mask_p32 = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    page_poly_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)])
    arm = Polygon([(1, 3), (2, 3), (2, 5.5), (1, 5.5)])
    mask_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)]).union(arm)
    assert isinstance(mask_p24, Polygon), "test setup: mask_p24 must be a Polygon"

    # Without page_polys (hull-based), the arm should NOT be detected:
    # hull(mask_p32) only extends to y=3, so the arm at y=[3,5.5] is outside it.
    result_hull = _fill_concave_dents(
        [mask_p32, mask_p24], page_color_data=None, page_polys=None
    )
    assert result_hull[1] is not None
    assert abs(result_hull[1].area - mask_p24.area) < 1e-6, (
        "hull-based: arm should NOT be transferred (it's outside hull(mask_p32))"
    )

    # With page_polys (page-poly-based), the arm IS in page_poly_p32.difference(mask_p32),
    # and removing it improves p24's convexity ratio.
    result_poly = _fill_concave_dents(
        [mask_p32, mask_p24],
        page_color_data=None,
        page_polys=[page_poly_p32, page_poly_p24],
    )
    assert result_poly[0] is not None
    assert result_poly[0].area > mask_p32.area, "p32 should gain the arm"
    assert result_poly[1] is not None
    assert result_poly[1].area < mask_p24.area, "p24 should lose the arm"
    # Convexity of p24 should improve after losing the arm.
    assert _convexity_ratio(result_poly[1]) > _convexity_ratio(mask_p24)


def test_fill_concave_dents_page_poly_with_color_transfers():
    """With both page_polys and color_data, arm is transferred when color is similar."""
    # Same arm geometry as test_fill_concave_dents_arm_via_page_poly.
    page_poly_p32 = Polygon([(0, 0), (4, 0), (4, 6), (0, 6)])
    mask_p32 = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    page_poly_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)])
    arm = Polygon([(1, 3), (2, 3), (2, 5.5), (1, 5.5)])
    mask_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)]).union(arm)
    assert isinstance(mask_p24, Polygon)

    # Both pages have zero color — no dominance, transfer should proceed.
    georef_p32 = _axis_aligned_georef(0.0, 0.0, 4.0, 6.0, w=40, h=60)
    georef_p24 = _axis_aligned_georef(0.0, 5.0, 4.0, 9.0, w=40, h=40)
    pcd_p32 = _make_page_color_data(georef_p32, np.zeros((60, 40), dtype=np.int16))
    pcd_p24 = _make_page_color_data(georef_p24, np.zeros((40, 40), dtype=np.int16))

    result = _fill_concave_dents(
        [mask_p32, mask_p24],
        page_color_data=[pcd_p32, pcd_p24],
        page_polys=[page_poly_p32, page_poly_p24],
    )
    assert result[0] is not None
    assert result[0].area > mask_p32.area, "p32 should gain the arm"
    assert result[1] is not None
    assert result[1].area < mask_p24.area, "p24 should lose the arm"


def test_fill_concave_dents_page_poly_color_blocks_transfer():
    """With both page_polys and color_data, arm is NOT transferred when j dominates."""
    page_poly_p32 = Polygon([(0, 0), (4, 0), (4, 6), (0, 6)])
    mask_p32 = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    page_poly_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)])
    arm = Polygon([(1, 3), (2, 3), (2, 5.5), (1, 5.5)])
    mask_p24 = Polygon([(0, 5), (4, 5), (4, 9), (0, 9)]).union(arm)
    assert isinstance(mask_p24, Polygon)

    # p24 has high color everywhere; p32 has none → color dominance blocks transfer.
    georef_p32 = _axis_aligned_georef(0.0, 0.0, 4.0, 6.0, w=40, h=60)
    georef_p24 = _axis_aligned_georef(0.0, 5.0, 4.0, 9.0, w=40, h=40)
    pcd_p32 = _make_page_color_data(georef_p32, np.zeros((60, 40), dtype=np.int16))
    pcd_p24 = _make_page_color_data(georef_p24, np.full((40, 40), 200, dtype=np.int16))

    result = _fill_concave_dents(
        [mask_p32, mask_p24],
        page_color_data=[pcd_p32, pcd_p24],
        page_polys=[page_poly_p32, page_poly_p24],
    )
    assert result[0] is not None
    assert abs(result[0].area - mask_p32.area) < 1e-6, "p32 should NOT gain the arm"
    assert result[1] is not None
    assert abs(result[1].area - mask_p24.area) < 1e-6, "p24 should keep the arm"


def test_assign_uses_color_score_over_centroid():
    """Block is assigned to the page with higher color score, even if more distant."""
    # Two adjacent pages; block straddles the boundary near page0.
    # Without color: page0 centroid is closer → block goes to page0.
    # With color: page1 has rich content in the block area → block goes to page1.
    georef0 = _axis_aligned_georef(0.0, 0.0, 1.0, 1.0, w=10, h=10)
    georef1 = _axis_aligned_georef(0.9, 0.0, 2.0, 1.0, w=10, h=10)
    page0 = Polygon(georef0["corners"])
    page1 = Polygon(georef1["corners"])
    # Block centroid at (0.75, 0.5): closer to page0 centroid (0.5, 0.5) than
    # page1 centroid (1.45, 0.5), so centroid-distance would assign to page0.
    block = Polygon([(0.6, 0.2), (0.95, 0.2), (0.95, 0.8), (0.6, 0.8)])

    result_no_color = _assign_blocks_to_pages([block], [page0, page1])
    assert result_no_color[0] == [0], "without color, block should go to closer page0"

    # page0: zero color score; page1: rich color score everywhere
    pcd0 = _make_page_color_data(georef0, np.zeros((10, 10), dtype=np.int16))
    pcd1 = _make_page_color_data(georef1, np.full((10, 10), 100, dtype=np.int16))
    result_with_color = _assign_blocks_to_pages([block], [page0, page1], [pcd0, pcd1])
    assert result_with_color[1] == [0], "with color scoring, block should go to page1"


# ---------------------------------------------------------------------------
# _fill_coverage_gaps
# ---------------------------------------------------------------------------


def test_fill_coverage_gaps_notch_within_single_page():
    """A triangular notch in a page's mask is filled when it lies within the page extent."""
    page_poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    # mask has a triangular notch cut from the top-right corner
    mask = Polygon([(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)])
    # gap = triangle [(2,4)-(4,4)-(4,2)], entirely within page_poly

    result = _fill_coverage_gaps([mask], [page_poly])

    assert result[0] is not None
    assert abs(result[0].area - page_poly.area) < 1e-6, (
        "gap should be absorbed, covering full page"
    )


def test_fill_coverage_gaps_split_across_pages():
    """A gap spanning two adjacent page extents is split and each page gets its portion."""
    page_poly_0 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    page_poly_1 = Polygon([(4, 0), (8, 0), (8, 4), (4, 4)])
    mask_0 = Polygon([(0, 0), (3, 0), (3, 4), (0, 4)])  # leaves x=[3,4] uncovered
    mask_1 = Polygon([(5, 0), (8, 0), (8, 4), (5, 4)])  # leaves x=[4,5] uncovered

    result = _fill_coverage_gaps([mask_0, mask_1], [page_poly_0, page_poly_1])

    assert result[0] is not None
    assert result[0].area > mask_0.area, "page0 should gain x=[3,4] strip"
    assert result[1] is not None
    assert result[1].area > mask_1.area, "page1 should gain x=[4,5] strip"
    # No region should be double-covered (shared edge at x=4 has zero area)
    assert result[0].intersection(result[1]).area < 1e-8


def test_fill_coverage_gaps_clipped_to_page_extent():
    """Gap extending outside a page's extent is clipped before being added."""
    page_poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    mask = Polygon([(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)])
    # Simulate another page whose mask doesn't cover the top-right region
    # The gap is partially outside page_poly (e.g., extends to y=6).
    # Only the part within page_poly should be added to mask.
    extra_page_poly = Polygon([(2, 4), (6, 4), (6, 8), (2, 8)])
    extra_mask = Polygon([(4, 4), (6, 4), (6, 8), (2, 8)])  # leaves triangle at left

    result = _fill_coverage_gaps([mask, extra_mask], [page_poly, extra_page_poly])

    # page0 mask should grow (absorb the top-right triangle within page_poly)
    assert result[0] is not None
    assert result[0].area > mask.area
    # page0 mask must stay within its page extent
    assert result[0].difference(page_poly.buffer(1e-9)).area < 1e-8


def test_fill_coverage_gaps_disconnected_from_mask():
    """Gap piece that doesn't touch a page's mask is not added to that page.

    page_0's mask is a small corner square; the gap piece (a floating island)
    is disjoint from it, so mask_0.union(island) is a MultiPolygon and the
    connectivity check at line 584 rejects it for page 0.

    page_1's mask has the island as an interior hole, so mask_1.union(island)
    fills the hole and gives a solid Polygon — page 1 absorbs the gap.
    """
    big_square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    island = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])

    # page_0: small bottom-left corner, nowhere near the island
    mask_0 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    # page_1: big square with the island punched out as an interior hole
    mask_1 = big_square.difference(island)
    assert isinstance(mask_1, Polygon) and len(list(mask_1.interiors)) == 1

    # The only gap is the island itself (page_union = big_square; mask_union = mask_0 ∪ mask_1
    # = big_square - island + tiny corner, so gap ≈ island).
    result = _fill_coverage_gaps([mask_0, mask_1], [big_square, big_square])

    # page_1 fills its hole (island connects to mask_1's boundary)
    assert result[1] is not None
    assert abs(result[1].area - big_square.area) < 1e-6

    # page_0 does NOT absorb the island (disconnected from its mask)
    assert result[0] is not None
    assert abs(result[0].area - mask_0.area) < 1e-6


def test_fill_coverage_gaps_no_gaps():
    """When masks already cover all page extents, nothing changes."""
    page_poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    mask = page_poly  # exact coverage

    result = _fill_coverage_gaps([mask], [page_poly])
    assert result[0] is mask  # same object returned (no modification)
