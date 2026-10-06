"""Tests for mapsnap.cutline_model's mask-to-panels helpers."""

import numpy as np
from shapely.geometry import LineString, box

from mapsnap.cutline_model import boundary_support, merge_unsupported


def test_boundary_support_is_the_covered_share_of_a_line():
    near_line = np.zeros((100, 100), dtype=bool)
    near_line[:, 50] = True
    assert boundary_support(LineString([(50, 0), (50, 99)]), near_line) == 1.0
    assert boundary_support(LineString([(20, 0), (20, 99)]), near_line) == 0.0


def test_merge_unsupported_keeps_drawn_cuts_and_dissolves_the_rest():
    prob = np.zeros((100, 300), dtype=np.uint8)
    prob[:, 100] = 255  # the network drew the cut at x = 100 only
    left, middle, right = (
        box(0, 0, 100, 100),
        box(100, 0, 200, 100),
        box(200, 0, 300, 100),
    )
    panels = merge_unsupported([left, middle, right], prob)
    # The undrawn boundary at x = 200 is dissolved; the drawn one at x = 100 stays.
    assert sorted(p.bounds for p in panels) == [(0, 0, 100, 100), (100, 0, 300, 100)]


def test_merge_unsupported_leaves_a_single_panel_alone():
    whole = box(0, 0, 10, 10)
    assert merge_unsupported([whole], np.zeros((10, 10), dtype=np.uint8)) == [whole]
