"""Clip masks from the content-region model, as a planar partition of street blocks (#544).

Each page's clip mask says which ground it shows in the mosaic. Two things decide
it: who owns the ground, and where the seams between owners run.

**Ownership.** Every page's P(region) map (``mapsnap region``: 0 on margins and
the strips a sheet shares with its neighbours, 1 on content exclusive to it) is
warped through the page's pose onto a shared ground grid of CELL_M cells. Where
several pages claim a cell, it goes to the one it lies deepest inside (distance
to that page's content edge), so contested ground splits along a midline rather
than along noise between two maps near 1. Cells below OWNER_THRESHOLD are
nobody's; unclaimed seams up to CLOSE_M wide go to the nearest owner.

**Seams.** No outline is traced from the grid, which gives jagged geometry that
Allmaps struggles with (#31, #224). The ground is cut into units instead, each
given whole to one page:

- an OSM street block with one dominant owner is one unit, so seams follow
  street centerlines;
- a superblock that another page owns at least MINORITY_M2 of is cut by a
  straight line fitted to the boundary between its two main owners (Sanborn
  sheet edges are straight), recursively for three or more;
- a unit reaching past its owner's scan is cut along the scan's edge, and the
  part beyond goes to the covering page that owns most of it.

Pages are dissolved from their units. A stray part joins the neighbour sharing
the longest boundary, if that neighbour's scan shows most of it. Holes the
mosaic encloses go to the covering page with the best P(region) there. Then
``shapely.coverage_simplify`` simplifies all pages together, so shared edges stay
shared, and each mask is clipped to what its page can show (its scan, or a split
panel's outline): a mask reaching past its image makes Allmaps fail to
triangulate the map.

Scored with ``mapsnap mask-eval`` / ``mask-score`` (#548): against OIM's masks on
OIM's own poses, IoU 0.785 → 0.880 on volumes the region model never saw, and
the share of each sheet drawn wrong 5.2% → 2.9%.
"""

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import shapely
from scipy import ndimage
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, split, transform, unary_union

from mapsnap.clip_masks import _fit_affine, _polygonize_streets, _remove_spike_vertices

M_PER_DEGREE = math.pi * 6_371_008.8 / 180
# Side of a ground-grid cell, in metres.
CELL_M = 1.5
# A cell is claimed by a page whose P(region) there is at least this.
OWNER_THRESHOLD = 0.05
# Unclaimed seams and holes inside the claimed area, up to about this wide, go to
# the nearest owner.
CLOSE_M = 12.0
# Depth inside a page's content region stops counting beyond this.
INTERIOR_CAP_M = 400.0
# A unit in which another page owns at least this much ground is cut (about a
# quarter of a median Manhattan block).
MINORITY_M2 = 1500.0
# A unit goes to a page only if it owns at least this share of its cells.
MIN_CLAIM = 0.5
# The claimed area's outline is traced at this stricter P(region), so a scan's
# black border admitted by the low ownership threshold is trimmed.
OUTLINE_THRESHOLD = 0.3
OUTLINE_SIMPLIFY_M = 8.0
COVERAGE_SIMPLIFY_M = 4.0
# Holes smaller than this are filled; bigger ones are slit open to the outside.
FILL_HOLE_M2 = 2500.0
MIN_UNIT_M2 = 20.0
# A page's units are merged on a grid this fine, closing float-noise seams between them.
UNIT_SNAP_M = 0.01
# Mask vertices this close to the one before are dropped.
REPEATED_POINT_M = 0.05
MAX_CUT_DEPTH = 12


def region_map_path(image_path: Path) -> Path:
    """Where ``mapsnap region`` writes an image's P(region) map."""
    # Imported here: region_predict loads torch, which the blocks masker never needs.
    from mapsnap.region_predict import region_prob_path

    return region_prob_path(image_path.parent, image_path.stem)


def ensure_region_maps(image_paths: list[Path]) -> None:
    """Predict the P(region) maps that are missing, beside each image's volume."""
    missing = [path for path in image_paths if not region_map_path(path).exists()]
    if not missing:
        return
    from mapsnap.region_model import load_region_model
    from mapsnap.region_predict import write_region_maps

    print(f"Predicting {len(missing)} P(region) map(s)...", file=sys.stderr)
    model, device = load_region_model()
    by_dir: dict[Path, list[Path]] = {}
    for path in missing:
        by_dir.setdefault(path.parent, []).append(path)
    for directory, paths in by_dir.items():
        write_region_maps(directory, paths, model=model, device=device)


def panel_outline_mask(image_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """1 inside a split panel's own outline and 0 in the whitened rest of its crop.

    A panel's crop is its bounding box painted white outside the (possibly
    L-shaped) outline, and the region model reads that blank paper as content.
    Whole pages get all ones.
    """
    ones = np.ones(shape, np.float32)
    if "__" not in image_path.stem:
        return ones
    base, index = image_path.stem.rsplit("__", 1)
    panels_path = image_path.parent / f"{base}.panels.json"
    if not panels_path.exists() or not index.isdigit():
        return ones
    ring = np.array(json.loads(panels_path.read_text())["panels"][int(index) - 1])
    origin = ring.min(axis=0)
    size = np.maximum(ring.max(axis=0) - origin, 1)
    scaled = (ring - origin) * [shape[1] / size[0], shape[0] / size[1]]
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(scaled).astype(np.int32)], 1)
    return mask.astype(np.float32)


def interior_score(prob: np.ndarray, m_per_px: float) -> np.ndarray:
    """P(region) below 0.5; inside the region, 0.5 rising to 1 with depth (m) to its edge."""
    inside = (prob >= 0.5).astype(np.uint8)
    depth = cv2.distanceTransform(inside, cv2.DIST_L2, 5) * m_per_px
    rising = 0.5 + 0.5 * np.minimum(depth, INTERIOR_CAP_M) / INTERIOR_CAP_M
    return np.where(inside > 0, rising, prob).astype(np.float32)


@dataclass
class GroundGrid:
    """A grid of CELL_M cells in a local metric frame about (lon0, lat0)."""

    lon0: float
    lat0: float
    x_min: float
    y_min: float
    rows: int
    cols: int

    @property
    def k(self) -> float:
        return math.cos(math.radians(self.lat0))

    def to_m(self, lon: float, lat: float) -> tuple[float, float]:
        return (
            (lon - self.lon0) * self.k * M_PER_DEGREE,
            (lat - self.lat0) * M_PER_DEGREE,
        )

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.lon0 + x / (self.k * M_PER_DEGREE),
            self.lat0 + y / M_PER_DEGREE,
        )

    def cells_of(self, geometry: BaseGeometry) -> np.ndarray:
        """Boolean raster of the cells inside a polygon (holes left out)."""
        mask = np.zeros((self.rows, self.cols), np.uint8)
        for polygon in polygons_of(geometry):
            rings = [
                np.round(
                    (np.asarray(ring.coords) - [self.x_min, self.y_min]) / CELL_M
                ).astype(np.int32)
                for ring in (polygon.exterior, *polygon.interiors)
            ]
            cv2.fillPoly(mask, rings, 1)
        return mask.astype(bool)

    def cell_points(self, cells: np.ndarray) -> np.ndarray:
        """Metric (x, y) of each True cell."""
        rows, cols = np.nonzero(cells)
        return np.c_[cols, rows] * CELL_M + [self.x_min, self.y_min]


@dataclass
class PageWindow:
    """One page's warped score, over the grid rows r0:r1 and columns c0:c1."""

    r0: int
    r1: int
    c0: int
    c1: int
    score: np.ndarray


@dataclass
class Ownership:
    """Which page owns each grid cell (-1: nobody), and each page's warped score."""

    grid: GroundGrid
    owner: np.ndarray
    best: np.ndarray
    windows: dict[int, PageWindow]

    @property
    def claimed(self) -> np.ndarray:
        return self.owner >= 0


def region_ownership(georefs: list[dict], image_paths: list[Path]) -> Ownership:
    """Warp every page's P(region) onto a ground grid; each cell to the page it lies deepest in."""
    corners = [c for georef in georefs for c in georef["corners"]]
    lon0 = float(np.mean([c[0] for c in corners]))
    lat0 = float(np.mean([c[1] for c in corners]))
    probe = GroundGrid(lon0, lat0, 0.0, 0.0, 0, 0)
    corners_m = np.array([[probe.to_m(*c) for c in g["corners"]] for g in georefs])
    x_min, y_min = corners_m[..., 0].min(), corners_m[..., 1].min()
    cols = math.ceil((corners_m[..., 0].max() - x_min) / CELL_M) + 1
    rows = math.ceil((corners_m[..., 1].max() - y_min) / CELL_M) + 1
    grid = GroundGrid(lon0, lat0, float(x_min), float(y_min), rows, cols)
    best = np.zeros((rows, cols), np.float32)
    owner = np.full((rows, cols), -1, np.int32)
    windows: dict[int, PageWindow] = {}
    k = grid.k
    for index, (georef, image_path) in enumerate(zip(georefs, image_paths)):
        prob = cv2.imread(str(region_map_path(image_path)), cv2.IMREAD_GRAYSCALE)
        if prob is None:
            print(f"region masks: no P(region) for {image_path}", file=sys.stderr)
            continue
        prob = prob.astype(np.float32) / 255.0
        prob *= panel_outline_mask(image_path, prob.shape)
        a_fwd, _ = _fit_affine(georef)  # georef px -> (lon, lat)
        m_per_px = math.hypot(
            a_fwd[0, 0] * k * M_PER_DEGREE, a_fwd[1, 0] * M_PER_DEGREE
        ) * (float(georef["width"]) / prob.shape[1])
        score = interior_score(prob, m_per_px)
        # Grid cell (col, row) -> metres -> georef px -> map px, as one affine.
        to_metric = np.array(
            [
                [
                    a_fwd[0, 0] * k * M_PER_DEGREE,
                    a_fwd[0, 1] * k * M_PER_DEGREE,
                    (a_fwd[0, 2] - lon0) * k * M_PER_DEGREE,
                ],
                [
                    a_fwd[1, 0] * M_PER_DEGREE,
                    a_fwd[1, 1] * M_PER_DEGREE,
                    (a_fwd[1, 2] - lat0) * M_PER_DEGREE,
                ],
                [0, 0, 1],
            ]
        )
        scale = np.diag(
            [
                prob.shape[1] / float(georef["width"]),
                prob.shape[0] / float(georef["height"]),
                1.0,
            ]
        )
        cell_to_metric = np.array([[CELL_M, 0, x_min], [0, CELL_M, y_min], [0, 0, 1]])
        cell_to_px = scale @ np.linalg.inv(to_metric) @ cell_to_metric
        page_cells = (corners_m[index] - [x_min, y_min]) / CELL_M
        c0, r0 = np.maximum(np.floor(page_cells.min(axis=0)).astype(int), 0)
        c1, r1 = np.ceil(page_cells.max(axis=0)).astype(int) + 1
        c1, r1 = min(c1, cols), min(r1, rows)
        shift = np.array([[1, 0, c0], [0, 1, r0], [0, 0, 1]])
        warped = cv2.warpAffine(
            score,
            (cell_to_px @ shift)[:2],
            (c1 - c0, r1 - r0),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        windows[index] = PageWindow(r0, r1, c0, c1, warped)
        window_best = best[r0:r1, c0:c1]
        wins = warped > window_best
        window_best[wins] = warped[wins]
        owner[r0:r1, c0:c1][wins] = index

    claimed = best >= OWNER_THRESHOLD
    radius = max(1, round(CLOSE_M / CELL_M))
    closed = np.asarray(
        ndimage.binary_fill_holes(
            ndimage.binary_closing(
                claimed, structure=np.ones((3, 3)), iterations=radius
            )
        ),
        bool,
    )
    gaps = closed & ~claimed
    if gaps.any():
        near_r, near_c = np.asarray(
            ndimage.distance_transform_edt(
                ~claimed, return_distances=False, return_indices=True
            )
        )
        owner = np.where(gaps, owner[near_r, near_c], owner)
        claimed = claimed | gaps
    owner = np.where(claimed, owner, -1)
    return Ownership(grid, owner, best, windows)


def polygons_of(geometry: BaseGeometry | None) -> list[Polygon]:
    """The non-empty polygons of any geometry, collections included."""
    if geometry is None or geometry.is_empty:
        return []
    polygons: list[Polygon] = []
    for part in getattr(geometry, "geoms", [geometry]):
        if isinstance(part, Polygon) and not part.is_empty:
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(p for p in part.geoms if not p.is_empty)
    return polygons


def fit_line(points: np.ndarray, extent: float) -> LineString | None:
    """A straight line of half-length ``extent`` through a point cloud (total least squares)."""
    if len(points) < 5:
        return None
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    return LineString([center - vt[0] * extent, center + vt[0] * extent])


def without_holes(polygon: Polygon, slit_m: float = 0.3) -> Polygon:
    """Fill a polygon's small holes and open its big ones to the outside with a thin slit.

    An SvgSelector is one simple polygon, so a hole (another page's ground inside
    this one) would otherwise be dropped and this page drawn over it.
    """
    big = [ring for ring in polygon.interiors if Polygon(ring).area >= FILL_HOLE_M2]
    polygon = Polygon(polygon.exterior, big)
    while polygon.interiors:
        a, b = nearest_points(polygon.interiors[0], polygon.exterior)
        cut = LineString([a, b]).buffer(slit_m, cap_style="square")
        parts = polygons_of(polygon.difference(cut))
        if not parts:
            break
        largest = max(parts, key=lambda p: p.area)
        if len(largest.interiors) >= len(polygon.interiors):
            break  # the slit did not open the hole; stop rather than loop
        polygon = largest
    return polygon


def page_footprint(georef: dict, image_path: Path, grid: GroundGrid) -> Polygon:
    """What a page can show, in metres: its scan, or for a split panel its own outline.

    A panel's crop is its bounding box, but the selector is clipped to the panel's
    outline, so ground outside it is never the panel's to show.
    """
    scan = Polygon([grid.to_m(*c) for c in georef["corners"]]).buffer(0)
    if "__" not in image_path.stem:
        return scan
    base, index = image_path.stem.rsplit("__", 1)
    panels_path = image_path.parent / f"{base}.panels.json"
    if not panels_path.exists() or not index.isdigit():
        return scan
    ring = np.array(json.loads(panels_path.read_text())["panels"][int(index) - 1])
    ring_px = ring - ring.min(axis=0)  # the crop's frame
    a_fwd, _ = _fit_affine(georef)
    lonlat = (a_fwd @ np.c_[ring_px, np.ones(len(ring_px))].T).T
    outline = Polygon([grid.to_m(lon, lat) for lon, lat in lonlat]).buffer(0)
    return outline if not outline.is_empty else scan


def claimed_outline(ownership: Ownership) -> BaseGeometry:
    """The claimed area's outline, holes filled, heavily simplified, in metres."""
    grid = ownership.grid
    interior = cv2.erode(
        ownership.claimed.astype(np.uint8), np.ones((21, 21), np.uint8)
    ).astype(bool)
    held = (ownership.best >= OUTLINE_THRESHOLD) | interior
    contours, _ = cv2.findContours(
        held.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return unary_union(
        [
            Polygon(c[:, 0, :] * CELL_M + [grid.x_min, grid.y_min]).buffer(0)
            for c in contours
            if len(c) >= 3
        ]
    ).simplify(OUTLINE_SIMPLIFY_M)


def street_blocks(
    centerlines_geojson: dict,
    grid: GroundGrid,
    hull: BaseGeometry,
    outline: BaseGeometry,
) -> list[Polygon]:
    """The OSM street blocks within ``hull`` and the claimed outline, holes kept, in metres."""
    hull_lonlat = MultiPolygon(
        polygons_of(transform(lambda x, y, z=None: grid.to_lonlat(x, y), hull))
    )
    blocks: list[Polygon] = []
    for block in _polygonize_streets(centerlines_geojson, hull_lonlat):
        polygon = Polygon(
            [grid.to_m(*c) for c in block.exterior.coords],
            [[grid.to_m(*c) for c in ring.coords] for ring in block.interiors],
        ).buffer(0)
        blocks.extend(
            p
            for p in polygons_of(polygon.intersection(outline))
            if p.area >= MIN_UNIT_M2
        )
    return blocks


def assign_units(
    blocks: list[Polygon], ownership: Ownership, extent: float
) -> list[tuple[Polygon, int]]:
    """Give each block to its owner, cutting shared superblocks with straight lines."""
    owner, claimed, grid = ownership.owner, ownership.claimed, ownership.grid
    units: list[tuple[Polygon, int]] = []
    kernel = np.ones((3, 3), np.uint8)

    def assign(polygon: Polygon, depth: int) -> None:
        inside = grid.cells_of(polygon)
        votes = Counter(owner[inside & claimed].tolist())
        total = int(inside.sum())
        if not votes or total == 0:
            return
        top, top_n = votes.most_common(1)[0]
        significant = [p for p, n in votes.items() if n * CELL_M**2 >= MINORITY_M2]
        if len(significant) <= 1 or depth >= MAX_CUT_DEPTH:
            if top_n >= MIN_CLAIM * total or depth > 0:
                units.append((polygon, top))
            return
        # Cut between the pair of significant owners sharing the longest boundary.
        grown = {
            p: cv2.dilate(((owner == p) & inside).astype(np.uint8), kernel) > 0
            for p in significant
        }
        best_touch = np.zeros_like(inside)
        for i, a in enumerate(significant):
            for b in significant[i + 1 :]:
                touch = (owner == a) & inside & grown[b]
                if touch.sum() > best_touch.sum():
                    best_touch = touch
        line = fit_line(grid.cell_points(best_touch), extent)
        pieces = polygons_of(split(polygon, line)) if line is not None else []
        if len(pieces) < 2:
            units.append((polygon, top))
            return
        for piece in pieces:
            if piece.area >= MIN_UNIT_M2:
                assign(piece, depth + 1)

    for block in blocks:
        assign(block, 0)
    return units


def clip_units_to_scans(
    units: list[tuple[Polygon, int]], footprints: list[Polygon], ownership: Ownership
) -> list[tuple[Polygon, int]]:
    """Cut each unit at its owner's footprint; the part beyond goes to the covering page owning most of it."""
    owner, claimed, grid = ownership.owner, ownership.claimed, ownership.grid
    result: list[tuple[Polygon, int]] = []
    for polygon, page in units:
        inside = polygon.intersection(footprints[page])
        result.extend((p, page) for p in polygons_of(inside) if p.area >= MIN_UNIT_M2)
        for part in polygons_of(polygon.difference(footprints[page])):
            if part.area < MIN_UNIT_M2:
                continue
            votes = Counter(owner[grid.cells_of(part) & claimed].tolist())
            votes.pop(page, None)
            if not votes:
                continue
            other = votes.most_common(1)[0][0]
            shown = part.intersection(footprints[other])
            result.extend(
                (p, other) for p in polygons_of(shown) if p.area >= MIN_UNIT_M2
            )
    return result


def dissolve_pages(
    units: list[tuple[Polygon, int]], footprints: list[Polygon]
) -> dict[int, BaseGeometry]:
    """Each page's units, merged; a stray part joins the neighbour sharing the longest edge.

    Only a neighbour whose footprint covers most of the stray can take it: a mask
    reaching past its own scan makes Allmaps fail to triangulate the map.

    A page's units are merged on a UNIT_SNAP_M grid. Two units meeting along the
    same cut, each clipped separately, can miss each other by float noise; a plain
    union would leave them as two pieces and the smaller one would be dropped as a
    stray no neighbour can show (Brooklyn 1951 p57 lost 33,126 m^2 this way).
    """
    by_page: dict[int, list[Polygon]] = {}
    for polygon, page in units:
        by_page.setdefault(page, []).append(polygon)
    shapes: dict[int, BaseGeometry] = {
        page: shapely.union_all(polygons, grid_size=UNIT_SNAP_M)
        for page, polygons in by_page.items()
    }
    for _ in range(3):
        moved = False
        for page in list(shapes):
            parts = sorted(polygons_of(shapes[page]), key=lambda p: -p.area)
            if len(parts) <= 1:
                continue
            shapes[page] = parts[0]
            for stray in parts[1:]:
                best_page, best_length = None, 0.0
                for other, geometry in shapes.items():
                    covered = footprints[other].intersection(stray).area
                    if other == page or covered < 0.5 * stray.area:
                        continue
                    shared = stray.boundary.intersection(geometry.buffer(0.5)).length
                    if shared > best_length:
                        best_page, best_length = other, shared
                if best_page is not None:
                    shapes[best_page] = unary_union(
                        [shapes[best_page], stray.intersection(footprints[best_page])]
                    )
                    moved = True
        if not moved:
            break
    return shapes


def fill_enclosed_holes(
    shapes: dict[int, BaseGeometry], footprints: list[Polygon], ownership: Ownership
) -> None:
    """Give each hole the mosaic encloses to the covering page with the best P(region) there.

    Holes on the volume's outer edge stay open. A thin channel to the outside does
    not make a hole open: the union is closed by a few metres first.
    """
    union = unary_union(list(shapes.values()))
    closed = union.buffer(5).buffer(-5)
    filled = unary_union([Polygon(part.exterior) for part in polygons_of(closed)])
    for hole in polygons_of(filled.difference(union)):
        if hole.area < MIN_UNIT_M2:
            continue
        hole_cells = ownership.grid.cells_of(hole)
        best_page, best_score = None, -1.0
        for page, footprint in enumerate(footprints):
            window = ownership.windows.get(page)
            if window is None or footprint.intersection(hole).area < 0.5 * hole.area:
                continue
            cells = hole_cells[window.r0 : window.r1, window.c0 : window.c1]
            if not cells.any():
                continue
            score = float(window.score[cells].mean())
            if score > best_score:
                best_page, best_score = page, score
        if best_page is not None:
            piece = hole.intersection(footprints[best_page])
            shapes[best_page] = unary_union([shapes.get(best_page, Polygon()), piece])


def without_near_duplicates(polygon: Polygon) -> Polygon:
    """The polygon without vertices within REPEATED_POINT_M of the one before.

    Snap-rounding units onto the UNIT_SNAP_M grid leaves pairs of vertices a
    centimetre apart, and Allmaps fails to triangulate masks with edges that short
    under some renderings. The polygon comes back unchanged if dropping them would
    make it invalid (or collapse a ring) or move its area by more than 0.1%.
    """
    try:
        cleaned = shapely.remove_repeated_points(polygon, REPEATED_POINT_M)
    except GEOSException:  # a ring collapsed below three points
        return polygon
    if (
        not isinstance(cleaned, Polygon)
        or cleaned.is_empty
        or not cleaned.is_valid
        or abs(cleaned.area - polygon.area) > 0.001 * polygon.area
    ):
        return polygon
    return cleaned


def owned_cells_hull(
    page: int, ownership: Ownership, footprint: BaseGeometry
) -> Polygon | None:
    """The hull of the cells a page owns, within what it can show: for a page no unit went to."""
    cells = ownership.owner == page
    if cells.sum() < 3:
        return None
    hull = MultiPoint(ownership.grid.cell_points(cells)).convex_hull
    parts = polygons_of(hull.intersection(footprint))
    return max(parts, key=lambda p: p.area) if parts else None


# The scanner bed around a sheet (New York 1923, #571) reads ~20-40 on 0-255;
# paper ~200+. It is told apart from ink, which is dark too, by being thick:
# an opening BED_MIN_WIDTH_PX wide erases ink lines (1-6 px at 25% scale) and
# keeps the bed's band. Only thick dark regions touching the image edge count,
# so a sheet's content running off a cropped edge (split panels) stays paper.
# (A first version took the largest light region as the paper; ink lines that
# reach the edge cut that into pieces, and Kansas City 1951 p527__2 kept 48%.)
BED_LUMINANCE = 60
BED_MIN_WIDTH_PX = 9
# The paper is inset by PAPER_INSET_PX so the outline's simplification can't
# reach back into the bed, then simplified by PAPER_SIMPLIFY_PX. Both are in the
# stored image's pixels: tracing a reduced copy blended a thin dark edge into
# the paper beside it.
PAPER_INSET_PX = 4
PAPER_SIMPLIFY_PX = 3.0


def scanner_bed(gray: np.ndarray) -> np.ndarray:
    """Pixels of a dark border around the sheet: thick dark regions touching the image edge."""
    dark = (gray < BED_LUMINANCE).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (BED_MIN_WIDTH_PX, BED_MIN_WIDTH_PX)
    )
    thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    _, labels = cv2.connectedComponents(thick, connectivity=8)
    edge = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    return np.isin(labels, edge[edge > 0])


def paper_outline(image_path: Path, georef: dict, grid: GroundGrid) -> Polygon | None:
    """The sheet's paper, in metres: the image less any dark border around it (``scanner_bed``).

    None when the image can't be read.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    paper = (~scanner_bed(gray)).astype(np.uint8)
    kernel = np.ones((2 * PAPER_INSET_PX + 1, 2 * PAPER_INSET_PX + 1), np.uint8)
    paper = cv2.erode(paper, kernel)
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = cv2.approxPolyDP(
        max(contours, key=cv2.contourArea), PAPER_SIMPLIFY_PX, True
    )
    if len(contour) < 3:
        return None
    # Back to the georef's pixel frame (the image may not be the size it records).
    sx = georef["width"] / gray.shape[1]
    sy = georef["height"] / gray.shape[0]
    a_fwd, _ = _fit_affine(georef)
    points = [
        grid.to_m(*(a_fwd @ np.array([x * sx, y * sy, 1.0])))
        for x, y in contour[:, 0, :]
    ]
    outline = Polygon(points).buffer(0)
    return outline if not outline.is_empty else None


def extend_into_free_margins(
    shapes: dict[int, BaseGeometry],
    footprints: list[Polygon],
    papers: list[Polygon | None],
) -> dict[int, float]:
    """Give each page the part of its paper no other page's scan reaches (#571).

    Ownership trims every page to its content region, which is right where
    sheets overlap but needlessly hides margins nothing else could show (Hot
    Wells 1919's lone sheet lost its title and borders). Ground inside a page's
    paper and outside every other page's footprint can't conflict, so it joins
    the page's mask; a dark border (``paper_outline``) never does. Returns the
    area added per page, in m^2.
    """
    added: dict[int, float] = {}
    for page, footprint in enumerate(footprints):
        paper = papers[page]
        if paper is None:
            continue
        others = unary_union([f for i, f in enumerate(footprints) if i != page])
        free = footprint.intersection(paper).difference(others)
        current = shapes.get(page, Polygon())
        gain = free.difference(current)
        if gain.area < 1.0:
            continue
        # On dissolve_pages' snapping grid: a plain union left near-duplicate
        # vertices that GEOS later failed on (Kansas City 1951, #571).
        merged = shapely.union_all([current, free], grid_size=UNIT_SNAP_M)
        shapes[page] = max(polygons_of(merged), key=lambda part: part.area)
        added[page] = shapes[page].area - current.area
    return added


def compute_region_clip_masks(
    georefs: list[dict],
    centerlines_geojson: dict,
    simplify_tolerance: float = 0.00005,
    debug_blocks_out: list[dict] | None = None,
    raw_paths: list[Path] | None = None,
    *,
    paper_clip: bool = False,
    free_margins: bool = False,
) -> list[Polygon | None]:
    """One clip polygon (lon, lat) per page, or None where the page gets none.

    The same signature as clip_masks.compute_all_clip_masks; ``raw_paths`` (the
    georeferenced images) are required, since P(region) is predicted per image.
    ``simplify_tolerance`` and ``debug_blocks_out`` are accepted for that
    compatibility and unused.

    ``paper_clip`` keeps every mask on its sheet's paper (``paper_outline``), so
    no page shows a dark scanner bed: seam and hole filling can otherwise hand a
    page ground its image only shows as black (New York 1923 p19-p21, #571).
    ``free_margins`` also gives each page the paper no other page's scan reaches
    (``extend_into_free_margins``), and implies ``paper_clip``.
    """
    if not georefs or raw_paths is None:
        return [None] * len(georefs)
    image_paths = [Path(path) for path in raw_paths]
    ensure_region_maps(image_paths)
    ownership = region_ownership(georefs, image_paths)
    grid = ownership.grid
    footprints = [
        page_footprint(g, path, grid) for g, path in zip(georefs, image_paths)
    ]
    outline = claimed_outline(ownership)
    hull = unary_union(footprints).convex_hull
    blocks = street_blocks(centerlines_geojson, grid, hull, outline)
    extent = max(footprint.length for footprint in footprints) * 2
    units = assign_units(blocks, ownership, extent)
    units = clip_units_to_scans(units, footprints, ownership)
    shapes = dissolve_pages(units, footprints)
    fill_enclosed_holes(shapes, footprints, ownership)
    # What each page may show: its scan (or panel outline), and with paper_clip
    # only the paper on it.
    visible: list[BaseGeometry] = list(footprints)
    if paper_clip or free_margins:
        papers = [paper_outline(path, g, grid) for g, path in zip(georefs, image_paths)]
        visible = [
            footprint if paper is None else footprint.intersection(paper)
            for footprint, paper in zip(footprints, papers)
        ]
        if free_margins:
            added = extend_into_free_margins(shapes, footprints, papers)
            print(
                f"Free margins: {len(added)} page(s) gained "
                f"{sum(added.values()):,.0f} m^2",
                file=sys.stderr,
            )

    pages = sorted(shapes)
    polygons = []
    for page in pages:
        parts = polygons_of(shapes[page])
        largest = max(parts, key=lambda p: p.area) if parts else Polygon()
        polygons.append(without_holes(largest) if parts else Polygon())
    # Simplify every page together, so shared edges stay shared.
    simplified = shapely.coverage_simplify(
        np.array(polygons, dtype=object), COVERAGE_SIMPLIFY_M
    )

    masks: list[Polygon | None] = [None] * len(georefs)
    for page, geometry in zip(pages, simplified):
        parts = polygons_of(geometry)
        if not parts:
            continue
        # Simplification, strays and filled holes must never carry a mask past
        # what its page can show.
        parts = polygons_of(
            max(parts, key=lambda p: p.area).intersection(visible[page])
        )
        if not parts:
            continue
        main = without_near_duplicates(
            _remove_spike_vertices(max(parts, key=lambda p: p.area))
        )
        if main.is_empty or len(main.exterior.coords) < 4:
            continue
        masks[page] = Polygon([grid.to_lonlat(x, y) for x, y in main.exterior.coords])
    # A page no unit went to would fall back to its whole sheet; give it the hull of
    # the cells it owns instead.
    for page, mask in enumerate(masks):
        if mask is None:
            hull_m = owned_cells_hull(page, ownership, visible[page])
            if hull_m is not None:
                masks[page] = Polygon(
                    [grid.to_lonlat(x, y) for x, y in hull_m.exterior.coords]
                )
    print(
        f"Region masks: {sum(m is not None for m in masks)}/{len(georefs)} pages; "
        f"{len(blocks)} blocks, {len(units)} units",
        file=sys.stderr,
    )
    return masks
