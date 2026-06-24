import numpy as np

from mapsnap.keymap.keymap_geojson import (
    INLIER_COLOR,
    NEUTRAL_COLOR,
    OUTLIER_COLOR,
    page_color,
    parse_svg_polygon,
    transform_ring,
)


def test_parse_svg_polygon():
    svg = '<svg><polygon points="0,0 10,0 10,20 0,20 0,0" /></svg>'
    assert parse_svg_polygon(svg) == [(0, 0), (10, 0), (10, 20), (0, 20), (0, 0)]


def test_parse_svg_polygon_space_separated():
    # Some selectors separate x/y with spaces rather than commas.
    assert parse_svg_polygon('<polygon points="1 2 3 4" />') == [(1, 2), (3, 4)]


def test_parse_svg_polygon_no_polygon():
    assert parse_svg_polygon("<svg></svg>") == []


def test_transform_ring_identity():
    # [lon, lat] = A @ [px, py, 1]; identity affine leaves pixel coords unchanged.
    affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert transform_ring([(2.0, 3.0), (4.0, 5.0)], affine) == [[2.0, 3.0], [4.0, 5.0]]


def test_transform_ring_scale_and_offset():
    affine = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0]])
    assert transform_ring([(1.0, 1.0)], affine) == [[3.0, 2.0]]


def test_page_color():
    assert page_color(5, {5}, {5}) == INLIER_COLOR  # inlier wins even if also detected
    assert page_color(6, {5}, {6}) == OUTLIER_COLOR  # detected but not an inlier
    assert page_color(7, {5}, {6}) == NEUTRAL_COLOR  # not in the key map
    assert page_color(None, {5}, {6}) == NEUTRAL_COLOR
