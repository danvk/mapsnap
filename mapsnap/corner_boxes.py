"""Corner boxes on key-map sheets (#276, step 5).

A key-map sheet gives up boxed regions in its corners: a KEY legend, a "graphic
map of volumes" inset, or a second key map (Los Angeles 1949 vol 14's Vol-13
continuation, Ellenville's Napanoch, Kingston's Port Ewen). The general
splitter finds the heavy ones (Kansas City, New Orleans 1951), but a key map is
dense with long linework: its divider candidates exceed the splitter's
MAX_DIVIDER_CANDIDATES safety gate, so the sheet stands whole, and a box border
drawn at 4 px sits below the divider thickness floor. This module looks for
the one shape a legitimate cut-away always has: a chain of axis-aligned
segments running from one sheet edge to an adjacent edge, enclosing the corner
between them and touching no third edge. A chamfered corner (Los Angeles) is
the one diagonal link a chain may carry; a side that Hough never vectorized
(Ellenville) is closed by tracing a straight, border-thick ink line from the
chain's free end to the edge.

Coordinates are the splitter's cropped detection frame; the caller expands the
resulting panels to the full sheet.
"""

from dataclasses import dataclass, field
from itertools import pairwise
from math import atan2, degrees, hypot
from statistics import median

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

Segment = tuple[float, float, float, float]
Point = tuple[float, float]

# A box border is drawn lighter than a sheet divider: Ellenville's Napanoch box is
# 4.0 px at the 25% working scale (the splitter's divider floor is 5.0 px), while
# key-map block and street lines measure 2-2.8 px.
BOX_MIN_THICK_PX = 3.5
# A cut-away smaller than this share of the sheet is a sliver along an edge, not
# a boxed region; larger than the max it is the sheet itself.
BOX_MIN_AREA_FRAC = 0.02
BOX_MAX_AREA_FRAC = 0.5  # the same bar as split.CUT_AWAY_MAX_AREA
# An endpoint within this fraction of the shorter sheet side counts as flush with
# a sheet edge (split.FLUSH_EDGE_TOLERANCE).
FLUSH_FRAC = 0.02
# Segment endpoints this close continue one chain.
JOIN_FRAC = 0.05
# Two chains, one from each edge of a corner, whose free ends are this close are
# closed with a straight link: a chamfer the detector saw both sides of.
CHAMFER_FRAC = 0.12
MAX_CHAIN = 6
# A box side runs parallel to a sheet edge; a chain may carry one link that does
# not (a chamfer). A heavy volume boundary wandering along a rotated street grid
# (Atlanta) carries many.
AXIS_TOLERANCE_DEG = 4.0
# A box is a rectangle or a chamfered one: convex. A ring whose closing link
# passes just beside one of its own vertices is a valid polygon pinched to a
# sliver, and its convex hull is far larger than it is.
BOX_MAX_HULL_RATIO = 1.10
# Ink closure: a straight run from a chain's free end to the edge, sampled in a
# strip this wide. A box side shows as one column dark along nearly the whole
# run (Ellenville's 4 px side scores 1.00; the false closures found on the
# corpus, straight streets and block edges, score 0.55-0.78) and border-thick
# in the typical row: a street through dense map content can be straight, but
# is drawn at map-line weight.
CLOSURE_HALF_WIDTH_PX = 8
CLOSURE_MIN_LINE = 0.8
MAX_CHAINS_PER_EDGE = 2000

CORNERS = (("left", "top"), ("top", "right"), ("right", "bottom"), ("bottom", "left"))
# How a box was closed, best first: the chain reached the other edge on its own,
# two chains met across a chamfer, or ink closed a side Hough never vectorized.
RANK_FLUSH, RANK_CHAMFER, RANK_INK = 0, 1, 2


@dataclass
class Chain:
    """A run of joined segments starting on a sheet edge; ``free`` is its far end."""

    points: list[Point]
    free: Point
    used: frozenset[int] = field(default_factory=frozenset)


@dataclass
class BoxCandidate:
    """A corner box with how it was closed (``rank``) and a line saying so."""

    rank: int
    polygon: Polygon
    corner: tuple[str, str]
    detail: str

    @property
    def area(self) -> float:
        return self.polygon.area


def flush_edges(point: Point, h: int, w: int) -> set[str]:
    """The sheet edges ``point`` is flush with (within FLUSH_FRAC of the short side)."""
    tol = FLUSH_FRAC * min(h, w)
    x, y = point
    edges = set()
    if x <= tol:
        edges.add("left")
    if x >= w - tol:
        edges.add("right")
    if y <= tol:
        edges.add("top")
    if y >= h - tol:
        edges.add("bottom")
    return edges


def project(point: Point, edge: str, h: int, w: int) -> Point:
    """``point`` moved perpendicularly onto ``edge``."""
    x, y = point
    if edge == "left":
        return (0.0, y)
    if edge == "right":
        return (float(w), y)
    if edge == "top":
        return (x, 0.0)
    return (x, float(h))


def corner(edge_a: str, edge_b: str, h: int, w: int) -> Point:
    """The sheet corner where two adjacent edges meet."""
    x = 0.0 if "left" in (edge_a, edge_b) else float(w)
    y = 0.0 if "top" in (edge_a, edge_b) else float(h)
    return (x, y)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def endpoints(segment: Segment) -> tuple[Point, Point]:
    return (segment[0], segment[1]), (segment[2], segment[3])


def off_axis(a: Point, b: Point) -> bool:
    """Whether the link a→b is neither horizontal nor vertical (a chamfer or worse)."""
    if distance(a, b) < 2.0:
        return False
    angle = abs(degrees(atan2(b[1] - a[1], b[0] - a[0]))) % 90.0
    return min(angle, 90.0 - angle) > AXIS_TOLERANCE_DEG


def chains_from_edge(segments: list[Segment], edge: str, h: int, w: int) -> list[Chain]:
    """Every chain of joined segments that starts flush with ``edge``.

    A chain begins at a segment endpoint on the edge (moved onto it) and grows by
    appending any unused segment whose endpoint lies within JOIN_FRAC of the
    chain's free end; each prefix is a chain in its own right, so a box closed
    early (by a flush end, a chamfer or ink) is found as well as the full run.
    Joins are endpoint to endpoint only: a line crossing a box side mid-way
    never continues the chain.
    """
    join = JOIN_FRAC * min(h, w)
    chains: list[Chain] = []

    def extend(chain: Chain) -> None:
        if len(chain.used) >= MAX_CHAIN or len(chains) >= MAX_CHAINS_PER_EDGE:
            return
        for index, segment in enumerate(segments):
            if index in chain.used:
                continue
            for near, far in (endpoints(segment), endpoints(segment)[::-1]):
                if distance(chain.free, near) <= join:
                    grown = Chain([*chain.points, far], far, chain.used | {index})
                    chains.append(grown)
                    extend(grown)

    for index, segment in enumerate(segments):
        for near, far in (endpoints(segment), endpoints(segment)[::-1]):
            if edge in flush_edges(near, h, w):
                chain = Chain([project(near, edge, h, w), far], far, frozenset({index}))
                chains.append(chain)
                extend(chain)
    return chains


def run_width(row: np.ndarray, column: int) -> int:
    """Length of the dark run through ``column`` (or a neighbor) in a strip row."""
    start = next(
        (c for c in (column, column - 1, column + 1) if 0 <= c < len(row) and row[c]),
        None,
    )
    if start is None:
        return 0
    left = start
    while left > 0 and row[left - 1]:
        left -= 1
    right = start
    while right < len(row) - 1 and row[right + 1]:
        right += 1
    return right - left + 1


def line_profile(
    binary: np.ndarray, point: Point, edge: str, h: int, w: int
) -> tuple[float, float]:
    """(coverage, width) of the best straight ink line from ``point`` to ``edge``.

    Samples a strip CLOSURE_HALF_WIDTH_PX either side of the run. Coverage is
    the share of the run along which the darkest column (allowing a pixel of
    wander) is inked; width is the typical dark run through that column, in
    pixels. A run shorter than the join distance reports full coverage and a
    border width: the end is as good as on the edge.
    """
    x, y = point
    half = CLOSURE_HALF_WIDTH_PX
    if edge in ("top", "bottom"):
        y0, y1 = (0, int(y)) if edge == "top" else (int(y), h)
        x0, x1 = max(0, int(x) - half), min(w, int(x) + half + 1)
        strip = binary[y0:y1, x0:x1] > 0
    else:
        x0, x1 = (0, int(x)) if edge == "left" else (int(x), w)
        y0, y1 = max(0, int(y) - half), min(h, int(y) + half + 1)
        strip = (binary[y0:y1, x0:x1] > 0).T  # rows along the run, columns across
    if strip.shape[0] <= JOIN_FRAC * min(h, w):
        return 1.0, float(BOX_MIN_THICK_PX)
    if strip.shape[1] == 0:
        return 0.0, 0.0
    wander = strip | np.roll(strip, 1, axis=1) | np.roll(strip, -1, axis=1)
    coverage = wander.mean(axis=0)
    column = int(coverage.argmax())
    widths = [run_width(row, column) for row in strip[wander[:, column]]]
    return float(coverage[column]), float(median(widths)) if widths else 0.0


def line_closure(binary: np.ndarray, point: Point, edge: str, h: int, w: int) -> bool:
    """Whether a straight, border-thick ink line runs from ``point`` to ``edge``.

    A box side is inked along at least CLOSURE_MIN_LINE of the run and at least
    BOX_MIN_THICK_PX wide; map content under the strip is spread across
    columns, and a straight street is drawn thinner than a box border.
    """
    coverage, width = line_profile(binary, point, edge, h, w)
    return coverage >= CLOSURE_MIN_LINE and width >= BOX_MIN_THICK_PX


def box_polygon(
    ring: list[Point], edge_a: str, edge_b: str, h: int, w: int
) -> Polygon | None:
    """The corner box enclosed by ``ring`` (edge A end first, edge B end last).

    Returns None unless the ring, closed through the corner, is a simple
    polygon whose share of the sheet lies between BOX_MIN_AREA_FRAC and
    BOX_MAX_AREA_FRAC, whose sides are axis-aligned but for one chamfer, which
    is convex (BOX_MAX_HULL_RATIO), and which touches no sheet edge but its
    corner's two: a region reaching a third edge is a strip across the sheet,
    the general splitter's business.
    """
    touched: set[str] = set()
    for point in ring:
        touched |= flush_edges(point, h, w)
    if touched - {edge_a, edge_b}:
        return None
    chamfers = 0
    for a, b in pairwise(ring):
        if not off_axis(a, b):
            continue
        # A diagonal that reaches a sheet edge is a street or a fold cutting
        # off a triangle, not a chamfer between two sides of a box.
        if flush_edges(a, h, w) or flush_edges(b, h, w):
            return None
        chamfers += 1
    if chamfers > 1:
        return None
    polygon = Polygon([corner(edge_a, edge_b, h, w), *ring])
    if not polygon.is_valid or polygon.is_empty:
        return None
    share = polygon.area / (h * w)
    if share < BOX_MIN_AREA_FRAC or share > BOX_MAX_AREA_FRAC:
        return None
    if polygon.convex_hull.area > BOX_MAX_HULL_RATIO * polygon.area:
        return None
    return polygon


def corner_box_candidates(
    segments: list[Segment],
    binary: np.ndarray,
    edges: tuple[str, str],
    h: int,
    w: int,
) -> list[BoxCandidate]:
    """Every box in the corner between two adjacent edges.

    Chains from either edge close a box three ways (RANK_FLUSH, RANK_CHAMFER,
    RANK_INK): the chain's free end is flush with the other edge; two chains'
    free ends meet across a chamfer; a straight ink line runs from the free end
    to the other edge.
    """
    edge_a, edge_b = edges
    chamfer = CHAMFER_FRAC * min(h, w)
    chains = {
        edge_a: chains_from_edge(segments, edge_a, h, w),
        edge_b: chains_from_edge(segments, edge_b, h, w),
    }
    candidates: list[BoxCandidate] = []
    for start, end in ((edge_a, edge_b), (edge_b, edge_a)):
        for chain in chains[start]:
            if end in flush_edges(chain.free, h, w):
                rank = RANK_FLUSH
                detail = f"{len(chain.used)}-segment chain from {start} to {end}"
            else:
                coverage, width = line_profile(binary, chain.free, end, h, w)
                if coverage < CLOSURE_MIN_LINE or width < BOX_MIN_THICK_PX:
                    continue
                rank = RANK_INK
                detail = (
                    f"{len(chain.used)}-segment chain from {start}, side to {end} "
                    f"closed by an ink line (coverage {coverage:.2f}, {width:.0f} px)"
                )
            ring = [*chain.points, project(chain.free, end, h, w)]
            polygon = box_polygon(ring, start, end, h, w)
            if polygon is not None:
                candidates.append(BoxCandidate(rank, polygon, edges, detail))
    for chain_a in chains[edge_a]:
        for chain_b in chains[edge_b]:
            if chain_a.used & chain_b.used:
                continue
            gap = distance(chain_a.free, chain_b.free)
            if gap > chamfer:
                continue
            ring = [*chain_a.points, *reversed(chain_b.points)]
            polygon = box_polygon(ring, edge_a, edge_b, h, w)
            if polygon is not None:
                detail = (
                    f"chains from {edge_a} ({len(chain_a.used)}) and {edge_b} "
                    f"({len(chain_b.used)}) meet across a {gap:.0f} px chamfer"
                )
                candidates.append(BoxCandidate(RANK_CHAMFER, polygon, edges, detail))
    return candidates


def corner_box(
    segments: list[Segment],
    binary: np.ndarray,
    edges: tuple[str, str],
    h: int,
    w: int,
) -> BoxCandidate | None:
    """The best box in a corner: best closure rank, then largest.

    Largest so that a box with an interior rule is not cut down to the rule.
    """
    candidates = corner_box_candidates(segments, binary, edges, h, w)
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate.rank, -candidate.area))


def corner_boxes(
    segments: list[Segment], binary: np.ndarray, h: int, w: int
) -> list[BoxCandidate]:
    """Non-overlapping corner boxes on a sheet, most trusted first.

    ``segments`` are the long, box-thick candidates in the cropped frame and
    ``binary`` the ink mask of that frame (for ink closure). Where two corners'
    boxes overlap, the one with the better closure rank stays, then the larger.
    """
    found: list[BoxCandidate] = []
    for edges in CORNERS:
        best = corner_box(segments, binary, edges, h, w)
        if best is not None:
            found.append(best)
    found.sort(key=lambda candidate: (candidate.rank, -candidate.area))
    kept: list[BoxCandidate] = []
    for candidate in found:
        if all(
            candidate.polygon.intersection(other.polygon).area < 0.01 * candidate.area
            for other in kept
        ):
            kept.append(candidate)
    return kept


def panels_with_boxes(boxes: list[Polygon], h: int, w: int) -> list[Polygon]:
    """The sheet cut into its remainder plus each box; empty if the cut is not clean.

    The remainder comes first. A remainder that is not one solid polygon (boxes
    that meet and sever it, or leave a hole) means the boxes do not describe a
    real layout, and no panels are returned.
    """
    if not boxes:
        return []
    remainder = box(0, 0, w, h).difference(unary_union(boxes))
    if not isinstance(remainder, Polygon) or remainder.interiors:
        return []
    return [remainder, *boxes]
