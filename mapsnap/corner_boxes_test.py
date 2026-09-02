import numpy as np
import pytest
from shapely.geometry import Polygon

from mapsnap.corner_boxes import (
    chains_from_edge,
    corner_boxes,
    flush_edges,
    line_closure,
    panels_with_boxes,
)

H, W = 1000, 1000


def blank() -> np.ndarray:
    return np.zeros((H, W), dtype=np.uint8)


def test_flush_edges_uses_the_short_side_tolerance():
    assert flush_edges((5.0, 500.0), H, W) == {"left"}
    assert flush_edges((995.0, 990.0), H, W) == {"right", "bottom"}
    assert flush_edges((500.0, 500.0), H, W) == set()


def test_chains_join_endpoint_to_endpoint_only():
    segments = [
        (0.0, 600.0, 400.0, 600.0),
        (400.0, 600.0, 400.0, 1000.0),
        # Crosses the first segment mid-way: never part of a chain.
        (200.0, 300.0, 200.0, 900.0),
    ]
    chains = chains_from_edge(segments, "left", H, W)
    longest = max(chains, key=lambda chain: len(chain.used))
    assert longest.used == frozenset({0, 1})
    assert longest.free == (400.0, 1000.0)


def test_l_chain_closes_a_bottom_left_box():
    segments = [(2.0, 600.0, 400.0, 600.0), (402.0, 605.0, 400.0, 995.0)]
    boxes = corner_boxes(segments, blank(), H, W)
    assert len(boxes) == 1
    assert boxes[0].polygon.area == pytest.approx(400 * 400, rel=0.02)
    assert boxes[0].polygon.contains(
        Polygon([(1, 601), (399, 601), (399, 999), (1, 999)])
    )


def test_chamfer_is_a_diagonal_link_in_the_chain():
    segments = [
        (0.0, 700.0, 150.0, 700.0),
        (150.0, 700.0, 300.0, 550.0),
        (300.0, 550.0, 500.0, 550.0),
        (500.0, 550.0, 500.0, 1000.0),
    ]
    boxes = corner_boxes(segments, blank(), H, W)
    assert len(boxes) == 1
    # The 500x450 rectangle less the 150x150 block above the stub and the
    # chamfer triangle beside it.
    assert boxes[0].area == pytest.approx(
        500 * 450 - 150 * 150 - 150 * 150 / 2, rel=0.01
    )


def test_two_chains_meet_across_an_unseen_chamfer():
    # The diagonal itself was never vectorized; the free ends are 85 px apart,
    # beyond the join distance (50) but within the chamfer reach (120).
    segments = [(0.0, 600.0, 350.0, 600.0), (410.0, 660.0, 410.0, 1000.0)]
    boxes = corner_boxes(segments, blank(), H, W)
    assert len(boxes) == 1
    assert boxes[0].polygon.area == pytest.approx(410 * 400 - 60 * 60 / 2, rel=0.02)


def test_ink_line_closes_a_side_hough_missed():
    segments = [(0.0, 600.0, 400.0, 600.0)]
    binary = blank()
    assert corner_boxes(segments, binary, H, W) == []
    binary[600:1000, 401:405] = 255  # the box's right side, 4 px wide
    boxes = corner_boxes(segments, binary, H, W)
    assert len(boxes) == 1
    assert boxes[0].polygon.area == pytest.approx(400 * 400, rel=0.02)
    assert "ink line" in boxes[0].detail


def test_scattered_ink_is_not_a_line():
    binary = blank()
    rng = np.random.default_rng(0)
    # Dense content: 60% of rows carry ink somewhere in the strip, in no column.
    for y in range(600, 1000):
        if rng.random() < 0.6:
            x = int(rng.integers(392, 409))
            binary[y, x] = 255
    assert not line_closure(binary, (400.0, 600.0), "bottom", H, W)
    binary[600:1000, 399:403] = 255
    assert line_closure(binary, (400.0, 600.0), "bottom", H, W)


def test_a_street_thin_straight_line_is_not_a_box_side():
    binary = blank()
    binary[600:1000, 400:402] = 255  # map-line weight, 2 px
    assert not line_closure(binary, (400.0, 600.0), "bottom", H, W)


def test_a_box_reaching_a_third_edge_is_a_strip_not_a_corner_box():
    # A chain from the left edge that runs the full width to the right edge and
    # down: a bottom strip, the general splitter's business.
    strip = [(0.0, 700.0, 1000.0, 700.0), (1000.0, 700.0, 1000.0, 1000.0)]
    assert corner_boxes(strip, blank(), H, W) == []


def test_a_wandering_boundary_is_not_a_box():
    # A heavy volume boundary following a rotated street grid: every link is
    # off-axis, so however it reaches the edges it encloses no box.
    boundary = [
        (0.0, 500.0, 200.0, 420.0),
        (200.0, 420.0, 350.0, 560.0),
        (350.0, 560.0, 520.0, 700.0),
        (520.0, 700.0, 450.0, 1000.0),
    ]
    assert corner_boxes(boundary, blank(), H, W) == []


def test_a_pinched_ring_is_rejected_for_the_box_inside_it():
    # The chain runs up from the bottom edge and back left; an ink line from
    # its free end to the right edge passes 2 px above the chain's own side.
    # That ring is valid but pinched to a sliver; the box is the prefix chain's.
    segments = [(600.0, 1000.0, 600.0, 400.0), (600.0, 400.0, 300.0, 398.0)]
    binary = blank()
    binary[395:400, 300:1000] = 255
    boxes = corner_boxes(segments, binary, H, W)
    assert len(boxes) == 1
    assert boxes[0].polygon.area == pytest.approx(400 * 600, rel=0.02)


def test_a_flush_box_beats_an_overlapping_ink_closed_one():
    # A legend box in the top-left corner with a double-ruled right border
    # (Los Angeles pb): the outer rule closes the legend flush; from the inner
    # rule an ink line runs right along the legend's bottom, closing a second
    # box that overlaps the legend's border. The flush box stays.
    segments = [
        (470.0, 0.0, 470.0, 300.0),
        (470.0, 300.0, 0.0, 300.0),
        (455.0, 0.0, 455.0, 300.0),
    ]
    binary = blank()
    binary[298:303, 455:1000] = 255
    boxes = corner_boxes(segments, binary, H, W)
    assert [candidate.corner for candidate in boxes] == [("left", "top")]
    assert boxes[0].polygon.area == pytest.approx(470 * 300, rel=0.02)


def test_a_diagonal_reaching_the_edges_cuts_a_triangle_not_a_box():
    # A street running from the left edge to the bottom edge (Cincinnati p1R).
    assert corner_boxes([(0.0, 700.0, 300.0, 1000.0)], blank(), H, W) == []


def test_notch_and_sliver_are_not_boxes():
    notch = [
        (0.0, 300.0, 200.0, 300.0),
        (200.0, 300.0, 200.0, 500.0),
        (200.0, 500.0, 0.0, 500.0),
    ]
    assert corner_boxes(notch, blank(), H, W) == []
    sliver = [(0.0, 990.0, 300.0, 990.0), (300.0, 990.0, 300.0, 1000.0)]
    assert corner_boxes(sliver, blank(), H, W) == []


def test_largest_box_wins_over_its_interior_rule():
    segments = [
        (0.0, 600.0, 400.0, 600.0),
        (400.0, 600.0, 400.0, 1000.0),
        (200.0, 600.0, 200.0, 1000.0),  # a rule inside the box
    ]
    boxes = corner_boxes(segments, blank(), H, W)
    assert len(boxes) == 1
    assert boxes[0].polygon.area == pytest.approx(400 * 400, rel=0.02)


def test_panels_with_boxes_tile_the_sheet():
    segments = [
        (0.0, 600.0, 400.0, 600.0),
        (400.0, 600.0, 400.0, 1000.0),
        (1000.0, 200.0, 700.0, 200.0),
        (700.0, 200.0, 700.0, 0.0),
    ]
    boxes = corner_boxes(segments, blank(), H, W)
    assert len(boxes) == 2
    panels = panels_with_boxes([candidate.polygon for candidate in boxes], H, W)
    assert len(panels) == 3
    assert panels[0].area == pytest.approx(H * W - 400 * 400 - 300 * 200, rel=0.01)
    assert sum(panel.area for panel in panels) == pytest.approx(H * W, rel=1e-6)
    assert panels_with_boxes([], H, W) == []
