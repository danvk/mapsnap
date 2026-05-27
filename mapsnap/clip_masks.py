"""Compute block-based clipping polygons for Sanborn map pages.

Street network polygonization divides the mapped region into city blocks.
Each block is assigned to exactly one page (the one with the greatest
intersection area; tie-break: closest page centroid), producing gapless,
non-overlapping clipping masks.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry import mapping as geom_mapping
from shapely.geometry import shape as geom_shape
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import polygonize, unary_union

# ~0.5 miles in degrees latitude (constant regardless of location).
_BUFFER_LAT_DEG = 0.5 * 1609.344 / 111_320.0


@dataclass
class PageColorData:
    """Color-score array and affine for one page's raw image.

    color_score holds max(0, spread-20) per pixel where spread = max(R,G,B)-min(R,G,B).
    Background paper tint has spread ≈ 20, so this zeroes it out; real colored ink
    has spread > 20.

    A_fwd/A_inv operate in georef pixel space. scale_x/scale_y map from georef pixels
    to work-image pixels.
    """

    color_score: np.ndarray  # (H, W) int16
    A_fwd: np.ndarray  # (2, 3) pixel→geo at georef pixel scale
    A_inv: np.ndarray  # (2, 2) geo→pixel (inverse of A_fwd[:, :2])
    scale_x: float  # work_w / georef_w
    scale_y: float  # work_h / georef_h


def _fit_affine(georef: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2×3 pixel→geo affine from the 4 georef corners.

    Pixel positions are (0,0), (w,0), (w,h), (0,h) corresponding to
    corners[0..3] (TL, TR, BR, BL). Uses least-squares so the result is
    numerically stable even if the corners aren't perfectly affine.

    Returns (A_fwd, A_inv) where:
      [lon, lat] = A_fwd @ [px, py, 1]^T
      [px, py]   = A_inv @ ([lon, lat] - A_fwd[:, 2])
    """
    w = float(georef["width"])
    h = float(georef["height"])
    corners = georef["corners"]

    pixel_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    geo_pts = np.array(corners, dtype=float)  # shape (4, 2): [[lon, lat], ...]

    # Build overdetermined system: geo = [px, py, 1] @ A^T
    ones = np.ones((4, 1))
    X = np.hstack([pixel_pts, ones])  # (4, 3)
    A_T, _, _, _ = np.linalg.lstsq(X, geo_pts, rcond=None)
    A_fwd = A_T.T  # (2, 3): [lon, lat]^T = A_fwd @ [px, py, 1]^T

    A_inv = np.linalg.inv(A_fwd[:, :2])  # (2, 2)
    return A_fwd, A_inv


def _geo_to_pixel(
    lon: float, lat: float, A_fwd: np.ndarray, A_inv: np.ndarray
) -> tuple[float, float]:
    """Convert a single (lon, lat) point to pixel coordinates."""
    geo_vec = np.array([lon, lat]) - A_fwd[:, 2]
    px, py = A_inv @ geo_vec
    return float(px), float(py)


def _load_page_color_data(
    raw_path: Path,
    georef: dict,
    work_long_side: int = 512,
) -> PageColorData | None:
    """Load a raw JPEG and build a color-score array for block assignment scoring.

    Decodes the JPEG at reduced resolution using PIL's draft() hint (JPEG 1/8 DCT),
    then resizes to work_long_side on the long side. Computes per-pixel spread =
    max(R,G,B) - min(R,G,B), subtracts 20 (approximate background paper tint), and
    clamps to zero. Returns None if the file is missing or unreadable.

    Assumes georef["width"] / georef["height"] match the original raw image dimensions
    (true for full-canvas Sanborn pages).
    """
    if not raw_path.exists():
        return None
    try:
        img = Image.open(raw_path)
        img.draft("RGB", (work_long_side, work_long_side))
        img = img.convert("RGB")
        raw_w, raw_h = img.size
        scale = work_long_side / max(raw_w, raw_h)
        work_w = max(1, round(raw_w * scale))
        work_h = max(1, round(raw_h * scale))
        img = img.resize((work_w, work_h), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.int16)  # (H, W, 3)
        spread = arr.max(axis=2) - arr.min(axis=2)
        color_score = np.maximum(0, spread - 20).astype(np.int16)
        A_fwd, A_inv = _fit_affine(georef)
        georef_w = float(georef["width"])
        georef_h = float(georef["height"])
        return PageColorData(
            color_score=color_score,
            A_fwd=A_fwd,
            A_inv=A_inv,
            scale_x=work_w / georef_w,
            scale_y=work_h / georef_h,
        )
    except Exception:
        return None


def _score_block_on_page(block: Polygon, pcd: PageColorData) -> float:
    """Sum color-score values within the block polygon projected onto the page image.

    Projects the block exterior from geo coords to work-pixel coords using the page's
    affine, rasterizes with PIL.ImageDraw, then sums the color_score within the mask.
    """
    H, W = pcd.color_score.shape
    coords_px = []
    for lon, lat in block.exterior.coords:
        px, py = _geo_to_pixel(lon, lat, pcd.A_fwd, pcd.A_inv)
        coords_px.append((px * pcd.scale_x, py * pcd.scale_y))
    mask_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(coords_px, fill=1)
    mask = np.array(mask_img, dtype=bool)
    return float(pcd.color_score[mask].sum())


def _polygonize_streets(
    centerlines_geojson: dict,
    coverage: Polygon | MultiPolygon,
) -> list[Polygon]:
    """Polygonize street network clipped to the coverage region.

    Clips each LineString/MultiLineString feature to coverage, adds each
    coverage boundary ring as a closing edge, then polygonizes the union of
    all line segments. coverage may be a Polygon or MultiPolygon.
    """
    lines: list[LineString] = []
    for feat in centerlines_geojson.get("features", []):
        geom = geom_shape(feat["geometry"])
        clipped = geom.intersection(coverage)
        if clipped.is_empty:
            continue
        # intersection of a LineString with a Polygon can be a Point, LineString,
        # MultiLineString, or GeometryCollection; collect only linear parts.
        if hasattr(clipped, "geoms"):
            for part in clipped.geoms:
                if isinstance(part, LineString) and not part.is_empty:
                    lines.append(part)
        elif isinstance(clipped, LineString) and not clipped.is_empty:
            lines.append(clipped)

    # Add each coverage boundary exterior to close off edge/waterfront blocks.
    coverage_polys: list[Polygon] = (
        list(coverage.geoms)
        if isinstance(coverage, BaseMultipartGeometry)
        else [coverage]
    )
    for poly in coverage_polys:
        lines.append(LineString(list(poly.exterior.coords)))

    all_lines = unary_union(lines)
    return [p for p in polygonize(all_lines) if isinstance(p, Polygon)]


def _assign_blocks_to_pages(
    blocks: list[Polygon],
    page_polys: list[Polygon],
    page_color_data: list[PageColorData | None] | None = None,
) -> dict[int, list[int]]:
    """Assign each block to the page with the highest color score (or nearest centroid).

    When page_color_data is provided, each eligible (block, page) pair is scored by
    summing color-spread values within the block on that page's work image. The page
    with the highest score wins; centroid distance breaks ties, then intersection area.
    Pages without color data receive a score of 0.

    When page_color_data is None, falls back to centroid-distance assignment (Voronoi):
    the nearest page centroid wins with intersection area as tiebreak.

    Returns a dict mapping page_index → list of block indices.
    Blocks with zero intersection with every page are dropped.
    """
    page_centroids = [p.centroid for p in page_polys]
    assignment: dict[int, list[int]] = {i: [] for i in range(len(page_polys))}

    for block_idx, block in enumerate(blocks):
        block_centroid = block.centroid
        best_page = -1
        best_score = -1.0
        best_dist = float("inf")
        best_area = 0.0

        for page_idx, page_poly in enumerate(page_polys):
            inter = block.intersection(page_poly)
            area = inter.area
            if area <= 0:
                continue

            dist = block_centroid.distance(page_centroids[page_idx])

            if page_color_data is not None:
                pcd = page_color_data[page_idx]
                score = _score_block_on_page(block, pcd) if pcd is not None else 0.0
                if (
                    score > best_score
                    or (score == best_score and dist < best_dist)
                    or (score == best_score and dist == best_dist and area > best_area)
                ):
                    best_score = score
                    best_page = page_idx
                    best_dist = dist
                    best_area = area
            else:
                if dist < best_dist or (dist == best_dist and area > best_area):
                    best_dist = dist
                    best_page = page_idx
                    best_area = area

        if best_page >= 0:
            assignment[best_page].append(block_idx)

    return assignment


def _collect_polygons(geom: object) -> list[Polygon]:
    """Extract all non-empty Polygon parts from any Shapely geometry."""
    if isinstance(geom, Polygon):
        return [] if geom.is_empty else [geom]
    if isinstance(geom, BaseMultipartGeometry):
        return [p for p in geom.geoms if isinstance(p, Polygon) and not p.is_empty]
    return []


def _is_substantial(
    piece: Polygon, reference: Polygon, min_fraction: float = 0.10
) -> bool:
    """Return True if piece is substantial relative to reference.

    Requires the piece's area to be >= min_fraction of reference area AND
    its bounding-box extent in both dimensions to be >= min_fraction of
    reference extent. Rejects slivers that are narrow in any direction.
    """
    if piece.area < reference.area * min_fraction:
        return False
    rx_min, ry_min, rx_max, ry_max = reference.bounds
    px_min, py_min, px_max, py_max = piece.bounds
    ref_w, ref_h = rx_max - rx_min, ry_max - ry_min
    if ref_w > 0 and (px_max - px_min) < ref_w * min_fraction:
        return False
    if ref_h > 0 and (py_max - py_min) < ref_h * min_fraction:
        return False
    return True


def _remove_spike_vertices(polygon: Polygon, min_turn_deg: float = 170.0) -> Polygon:
    """Remove backtrack vertices from a polygon's exterior ring.

    A backtrack vertex is one where the signed turn angle is ≥ min_turn_deg,
    meaning the polygon boundary reverses direction and creates a spike or
    antenna. These vertices cause triangulation failures in viewers like Allmaps.

    Iterates until no more backtrack vertices remain (multi-vertex spikes may
    require more than one pass).
    """
    coords = list(polygon.exterior.coords[:-1])  # drop closing duplicate
    while True:
        n = len(coords)
        keep = []
        for i in range(n):
            prev = coords[(i - 1) % n]
            cur = coords[i]
            nxt = coords[(i + 1) % n]
            v1 = (cur[0] - prev[0], cur[1] - prev[1])
            v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            turn = math.degrees(math.atan2(cross, dot))
            if abs(turn) < min_turn_deg:
                keep.append(cur)
        if len(keep) == n or len(keep) < 3:
            break
        coords = keep
    return Polygon(coords)


def _assign_blocks_to_pages_with_splits(
    blocks: list[Polygon],
    page_polys: list[Polygon],
    page_color_data: list[PageColorData | None] | None = None,
) -> dict[int, list[Polygon]]:
    """Assign blocks to pages, splitting blocks that straddle page boundaries.

    Iteratively assigns the current pool of subblocks. For each subblock assigned
    to a page, the intersection with that page is added to that page's territory;
    the remainder (outside the page) is queued for the next round only if it is
    substantial relative to the subblock (≥10% of area and ≥10% of extent in both
    dimensions). Thin slivers are discarded to avoid invalid geometry.

    Convergence is guaranteed: each outside part is strictly smaller than the
    subblock it came from, and the set of eligible pages shrinks each round.

    Returns page_idx → list of Polygon pieces already clipped to the page boundary.
    """
    page_territory: dict[int, list[Polygon]] = {i: [] for i in range(len(page_polys))}
    remaining: list[Polygon] = list(blocks)

    while remaining:
        assignment = _assign_blocks_to_pages(remaining, page_polys, page_color_data)
        if not any(assignment.values()):
            break  # all remaining subblocks are outside every page

        next_remaining: list[Polygon] = []
        for page_idx, block_indices in assignment.items():
            page_poly = page_polys[page_idx]
            for block_idx in block_indices:
                subblock = remaining[block_idx]
                inside = subblock.intersection(page_poly)
                outside = subblock.difference(page_poly)
                page_territory[page_idx].extend(_collect_polygons(inside))
                for piece in _collect_polygons(outside):
                    if _is_substantial(piece, subblock):
                        next_remaining.append(piece)

        remaining = next_remaining

    return page_territory


def _convexity_ratio(poly: Polygon) -> float:
    """Return area/convex_hull.area (1.0 = convex, lower = more concave)."""
    hull = poly.convex_hull
    return poly.area / hull.area if hull.area > 0 else 1.0


def _fill_concave_dents(
    masks: list[Polygon | None],
    page_color_data: list[PageColorData | None] | None,
    page_polys: list[Polygon] | None = None,
    simplify_tolerance: float = 0.00005,
    color_dominance_threshold: float = 3.0,
    color_min_score: float = 500.0,
) -> list[Polygon | None]:
    """Fill concave dents in page masks when the neighboring page has no color advantage.

    When page_polys is provided, uses page_poly_i − mask_i to find candidate regions
    (areas within the page's geographic extent that aren't in its mask). This catches
    arms and protrusions at territory edges that don't fall within the convex hull.
    Without page_polys, falls back to hull(mask_i) − mask_i.

    For each candidate piece owned by exactly one other page j, the piece is transferred
    from j to i unless: j's color score substantially exceeds i's (color_dominance_threshold),
    or (when using page_polys) removing the piece from j would not improve j's convexity
    ratio (meaning it isn't a protrusion in j).

    After all transfers, re-applies simplification and spike removal to modified masks.
    """
    new_masks: list[Polygon | None] = list(masks)
    modified: set[int] = set()
    transfer_count = 0

    # Pre-compute all transfer candidates using the ORIGINAL masks so that dent
    # computations for all pages use a consistent baseline.  Applying transfers
    # incrementally would shrink a page's mask before later pages are inspected,
    # causing the dent region to change and hiding dents that should have been found.
    candidates: list[tuple[int, Polygon, int]] = []

    if page_polys is not None:
        # Page-poly-based detection: for each (i, j) pair, directly compute the
        # part of j's mask that lies within i's geographic extent but not in i's
        # mask.  This avoids the owner-check problem (the arm is guaranteed to
        # come from j) and catches arms at territory edges that hull(mask_i) misses.
        for i, mask_i in enumerate(masks):
            if mask_i is None:
                continue
            for j, mask_j in enumerate(masks):
                if j == i or mask_j is None:
                    continue
                arm_geo = mask_j.intersection(page_polys[i]).difference(mask_i)
                if arm_geo.is_empty:
                    continue
                for piece in _collect_polygons(arm_geo):
                    if piece.area < mask_i.area * 0.001:
                        continue  # ignore tiny fragments

                    # Compute color scores for both pages in this piece's region.
                    score_i, score_j = 0.0, 0.0
                    if page_color_data is not None:
                        pcd_i = page_color_data[i]
                        pcd_j = page_color_data[j]
                        if pcd_i is not None:
                            score_i = _score_block_on_page(piece, pcd_i)
                        if pcd_j is not None:
                            score_j = _score_block_on_page(piece, pcd_j)

                    # Skip if j has a clear, substantial color advantage.
                    if (
                        score_j
                        > max(score_i, color_min_score) * color_dominance_threshold
                    ):
                        continue

                    # Verify removing piece from j improves j's convexity ratio
                    # (confirms it's a protrusion in j, not a legitimate block
                    # assignment that happens to lie within i's page extent).
                    test_j = mask_j.difference(piece)
                    if not test_j.is_empty:
                        test_j_poly: Polygon | None = None
                        if isinstance(test_j, Polygon):
                            test_j_poly = test_j
                        elif isinstance(test_j, MultiPolygon):
                            parts = sorted(
                                test_j.geoms, key=lambda p: p.area, reverse=True
                            )
                            if parts[0].area >= test_j.area * 0.9:
                                test_j_poly = parts[0]
                        if test_j_poly is None:
                            continue  # too fragmented after removal; skip
                        if _convexity_ratio(test_j_poly) <= _convexity_ratio(mask_j):
                            continue  # removing piece doesn't improve j's convexity; skip

                    candidates.append((i, piece, j))
    else:
        # Hull-based detection: find pieces in hull(mask_i) − mask_i that are
        # fully owned by exactly one other page j.
        for i, mask_i in enumerate(masks):
            if mask_i is None:
                continue
            dent_i = mask_i.convex_hull.difference(mask_i)
            if dent_i.is_empty:
                continue
            for piece in _collect_polygons(dent_i):
                if piece.area < mask_i.area * 0.001:
                    continue  # ignore tiny fragments

                # Find the single page that fully owns this piece.
                owners = [
                    j
                    for j, mask_j in enumerate(masks)
                    if j != i
                    and mask_j is not None
                    and piece.intersection(mask_j).area > piece.area * 0.9
                ]
                if len(owners) != 1:
                    continue
                j = owners[0]

                # Compute color scores for both pages in this piece's region.
                score_i, score_j = 0.0, 0.0
                if page_color_data is not None:
                    pcd_i = page_color_data[i]
                    pcd_j = page_color_data[j]
                    if pcd_i is not None:
                        score_i = _score_block_on_page(piece, pcd_i)
                    if pcd_j is not None:
                        score_j = _score_block_on_page(piece, pcd_j)

                # Skip if j has a clear, substantial color advantage.
                if score_j > max(score_i, color_min_score) * color_dominance_threshold:
                    continue

                candidates.append((i, piece, j))

    # Apply transfers using incrementally updated masks.
    for i, piece, j in candidates:
        if new_masks[i] is None or new_masks[j] is None:
            continue
        # Verify j still owns the piece; an earlier transfer may have claimed it.
        if new_masks[j].intersection(piece).area <= piece.area * 0.5:
            continue

        candidate_i = new_masks[i].union(piece)
        candidate_j = new_masks[j].difference(piece)
        if not isinstance(candidate_i, Polygon):
            continue

        # For the source, allow a MultiPolygon if one component dominates (≥90%).
        # This handles the case where the arm bridges two parts of j's territory —
        # removing it leaves a main body plus a small disconnected remnant.
        new_j: Polygon | None
        if candidate_j.is_empty:
            new_j = None
        elif isinstance(candidate_j, Polygon):
            new_j = candidate_j
        elif isinstance(candidate_j, MultiPolygon):
            parts = sorted(candidate_j.geoms, key=lambda p: p.area, reverse=True)
            if parts[0].area >= candidate_j.area * 0.9:
                new_j = parts[0]
            else:
                continue  # too fragmented; skip
        else:
            continue

        new_masks[i] = candidate_i
        new_masks[j] = new_j
        modified.update([i, j])
        transfer_count += 1

    if transfer_count:
        print(
            f"Dent-filling: {transfer_count} transfer(s) across {len(modified)} page(s).",
            file=sys.stderr,
        )

    # Re-apply simplification and spike removal to modified masks.
    for i in modified:
        mask = new_masks[i]
        if mask is None:
            continue
        simplified = mask.simplify(simplify_tolerance, preserve_topology=True)
        if isinstance(simplified, Polygon):
            new_masks[i] = _remove_spike_vertices(simplified)
        elif isinstance(simplified, MultiPolygon):
            parts = sorted(simplified.geoms, key=lambda p: p.area, reverse=True)
            new_masks[i] = _remove_spike_vertices(parts[0])

    return new_masks


def _fill_coverage_gaps(
    masks: list[Polygon | None],
    page_polys: list[Polygon],
    simplify_tolerance: float = 0.00005,
    min_gap_fraction: float = 0.001,
) -> list[Polygon | None]:
    """Fill uncovered regions within page extents by adding them to adjacent masks.

    Computes union(page_polys) − union(masks) to find all gap areas, then for each
    gap piece tries to assign it to a page whose mask it connects to (mask.union(piece)
    is a Polygon). When the gap spans multiple page extents, the overlapping portion is
    split and each page gets the part within its own extent.

    Unlike dent-filling, no color check is applied — gap areas have no existing owner.
    Convexity improvement is used only to break ties when multiple pages could absorb
    the same overlapping gap area.
    """
    page_union = unary_union(page_polys)
    mask_union = unary_union([m for m in masks if m is not None])
    total_gaps = page_union.difference(mask_union)
    if total_gaps.is_empty:
        return masks

    gap_pieces = _collect_polygons(total_gaps)
    if not gap_pieces:
        return masks

    new_masks: list[Polygon | None] = list(masks)
    fill_count = 0
    modified: set[int] = set()

    for gap_piece in gap_pieces:
        claimed = Polygon()  # portion of this gap already assigned to some page

        # For each page, compute the clipped sub-piece and check connectivity.
        assignments: list[tuple[float, int, Polygon]] = []
        for j, (mask_j, page_poly_j) in enumerate(zip(masks, page_polys)):
            if mask_j is None:
                continue
            sub = gap_piece.intersection(page_poly_j)
            if sub.is_empty:
                continue
            for piece in _collect_polygons(sub):
                if piece.area < mask_j.area * min_gap_fraction:
                    continue  # ignore tiny fragments
                candidate = mask_j.union(piece)
                if not isinstance(candidate, Polygon):
                    continue  # piece doesn't connect to mask; skip
                improvement = _convexity_ratio(candidate) - _convexity_ratio(mask_j)
                assignments.append((improvement, j, piece))

        # Process in descending convexity-improvement order; track claimed area to
        # prevent the same gap region from being assigned to two overlapping pages.
        assignments.sort(key=lambda t: (-t[0], t[1]))

        for _, j, piece in assignments:
            unclaimed = piece.difference(claimed)
            if unclaimed.is_empty:
                continue
            for sub in _collect_polygons(unclaimed):
                cur = new_masks[j]
                if cur is None:
                    continue
                candidate = cur.union(sub)
                if not isinstance(candidate, Polygon):
                    continue
                new_masks[j] = candidate
                claimed = claimed.union(sub)
                modified.add(j)
                fill_count += 1

    if fill_count:
        print(
            f"Gap-filling: {fill_count} piece(s) added across {len(modified)} page(s).",
            file=sys.stderr,
        )

    # Re-apply simplification and spike removal to modified masks.
    for j in modified:
        mask = new_masks[j]
        if mask is None:
            continue
        simplified = mask.simplify(simplify_tolerance, preserve_topology=True)
        if isinstance(simplified, Polygon):
            new_masks[j] = _remove_spike_vertices(simplified)
        elif isinstance(simplified, MultiPolygon):
            parts = sorted(simplified.geoms, key=lambda p: p.area, reverse=True)
            new_masks[j] = _remove_spike_vertices(parts[0])

    return new_masks


def compute_all_clip_masks(
    georefs: list[dict],
    centerlines_geojson: dict,
    simplify_tolerance: float = 0.00005,
    debug_blocks_out: list[dict] | None = None,
    raw_paths: list[Path] | None = None,
) -> list[Polygon | None]:
    """Compute block-based clipping masks for all georeferenced pages.

    Returns one entry per georef: a Shapely Polygon in geo (lon/lat) space
    clipped to that page's boundary, or None if no street blocks were
    assigned to the page (caller should fall back to the full-page rectangle).

    Raises ValueError if any page's mask turns out to be a MultiPolygon
    (unexpected; indicates a problem with the street network or assignment
    algorithm worth investigating).

    Args:
        georefs: list of parsed georef.json dicts with 'corners', 'width', 'height'.
        centerlines_geojson: parsed GeoJSON FeatureCollection of LineStrings.
        simplify_tolerance: Douglas-Peucker tolerance in degrees (~0.00005 ≈ 5 m).
        debug_blocks_out: if provided, GeoJSON Feature dicts for each block are
            appended here (properties include 'page_idx': int | null).
        raw_paths: if provided, raw JPEG paths parallel to georefs; used to load
            color-score data for block assignment scoring.
    """
    if not georefs:
        return []

    page_polys = [Polygon(g["corners"]) for g in georefs]

    # Build coverage region: union of all pages, buffered by ~0.5 miles.
    # Scale lon coords by cos(lat) so the isotropic shapely buffer gives a
    # lat-equivalent distance in all directions, then undo the scaling.
    all_lats = [c[1] for g in georefs for c in g["corners"]]
    mean_lat = float(np.mean(all_lats))

    # Buffer each page polygon by ~0.5 miles, then union the results.
    # Scale lon by cos(lat) before buffering so the isotropic shapely buffer
    # gives equal metric distance in all directions; undo scaling afterward.
    cos_lat = math.cos(math.radians(mean_lat))
    buffered: list[Polygon] = []
    for poly in page_polys:
        scaled = Polygon([(lon * cos_lat, lat) for lon, lat in poly.exterior.coords])
        scaled_buf = scaled.buffer(_BUFFER_LAT_DEG)
        unbuf = Polygon(
            [(lon / cos_lat, lat) for lon, lat in scaled_buf.exterior.coords]
        )
        buffered.append(unbuf)
    coverage_buffered: Polygon | MultiPolygon = unary_union(buffered)

    blocks = _polygonize_streets(centerlines_geojson, coverage_buffered)
    if not blocks:
        return [None] * len(georefs)

    page_color_data: list[PageColorData | None] | None = None
    if raw_paths is not None:
        page_color_data = [
            _load_page_color_data(path, georef)
            for path, georef in zip(raw_paths, georefs)
        ]
        n_loaded = sum(pcd is not None for pcd in page_color_data)
        print(
            f"Loaded color data for {n_loaded}/{len(georefs)} pages.",
            file=sys.stderr,
        )

    page_pieces = _assign_blocks_to_pages_with_splits(
        blocks, page_polys, page_color_data
    )

    if debug_blocks_out is not None:
        for page_idx, pieces in page_pieces.items():
            for piece in pieces:
                debug_blocks_out.append(
                    {
                        "type": "Feature",
                        "properties": {"page_idx": page_idx},
                        "geometry": geom_mapping(piece),
                    }
                )

    masks: list[Polygon | None] = []
    for page_idx, georef in enumerate(georefs):
        pieces = page_pieces.get(page_idx, [])
        if not pieces:
            masks.append(None)
            continue

        mask_geo = unary_union(pieces)

        if mask_geo.is_empty:
            masks.append(None)
            continue

        mask_geo = mask_geo.simplify(simplify_tolerance, preserve_topology=True)

        if isinstance(mask_geo, MultiPolygon):
            # Filter out disconnected slivers (< 5% of the largest component).
            parts = sorted(mask_geo.geoms, key=lambda p: p.area, reverse=True)
            threshold = parts[0].area * 0.05
            substantial = [p for p in parts if p.area >= threshold]
            if len(substantial) == 1:
                sliver_pct = sum(p.area for p in parts[1:]) / parts[0].area * 100
                print(
                    f"Warning: page {page_idx} mask had {len(parts) - 1} sliver(s) "
                    f"({sliver_pct:.1f}% of main area) dropped. "
                    f"Corners: {georef['corners'][0]}",
                    file=sys.stderr,
                )
                mask_geo = substantial[0]
            else:
                # Multiple substantial parts — likely due to a rotated page whose diagonal
                # boundary clips the axis-aligned block grid into disconnected pieces.
                # Try the convex hull of the parts clipped to the page; since both the
                # convex hull and the page polygon (a convex quadrilateral) are convex,
                # their intersection is always a connected Polygon.
                hull_mask = mask_geo.convex_hull.intersection(page_polys[page_idx])
                if isinstance(hull_mask, Polygon) and not hull_mask.is_empty:
                    hull_pct = hull_mask.area / page_polys[page_idx].area * 100
                    print(
                        f"Warning: page {page_idx} mask had {len(substantial)} disconnected "
                        f"substantial parts; using convex hull "
                        f"({hull_pct:.1f}% of page). Corners: {georef['corners'][0]}",
                        file=sys.stderr,
                    )
                    mask_geo = hull_mask
                else:
                    raise ValueError(
                        f"Page {page_idx} (corners starting at {georef['corners'][0]}) "
                        f"produced a MultiPolygon clipping mask with {len(substantial)} "
                        f"substantial parts (each ≥5% of largest), and the convex hull "
                        f"fallback also failed. Investigate the street network or block "
                        f"assignment for this page."
                    )

        # Remove backtrack vertices from any path (direct Polygon or resolved
        # MultiPolygon). Simplification can introduce spikes when it removes
        # intermediate vertices from narrow arms.
        if isinstance(mask_geo, Polygon):
            mask_geo = _remove_spike_vertices(mask_geo)

        # Warn if interior holes are present (unexpected with city block geometry).
        if len(mask_geo.interiors) > 0:
            print(
                f"Warning: page {page_idx} mask has {len(mask_geo.interiors)} interior "
                f"hole(s); only the exterior will be used. Corners: {georef['corners'][0]}",
                file=sys.stderr,
            )

        masks.append(cast(Polygon, mask_geo))

    # Fill concave dents: transfer pieces from neighboring pages when there's no
    # strong color reason to keep them there.
    masks = _fill_concave_dents(masks, page_color_data, page_polys, simplify_tolerance)

    # Fill coverage gaps: add uncovered regions (within the page extent union but
    # absent from every mask) to adjacent pages based on connectivity.
    masks = _fill_coverage_gaps(masks, page_polys, simplify_tolerance)

    # Report what fraction of the original page coverage is still covered.
    # Pages without a computed mask fall back to their full page polygon.
    effective = [m if m is not None else p for m, p in zip(masks, page_polys)]
    pre_area = unary_union(page_polys).area
    post_area = unary_union(effective).area
    retention = post_area / pre_area if pre_area > 0 else 1.0
    print(
        f"Coverage retention: {retention:.1%} of pre-clip area remains covered.",
        file=sys.stderr,
    )

    return masks


def geo_polygon_to_svg(
    geo_polygon: Polygon | None,
    georef: dict,
    source_width: int,
    source_height: int,
    split_canvas: tuple[float, float, float, float] | None = None,
) -> str:
    """Convert a geo-space Shapely Polygon to an IIIF SvgSelector string.

    Canvas coordinate mapping:
      Full canvas (split_canvas=None):
        canvas_x = pixel_x * (source_width / georef_width)
        canvas_y = pixel_y * (source_height / georef_height)
      Split sub-image (split_canvas=(cx, cy, cw, ch)):
        canvas_x = cx + pixel_x * (cw / georef_width)
        canvas_y = cy + pixel_y * (ch / georef_height)

    Falls back to the standard full-page rectangle when geo_polygon is
    None or empty. Raises ValueError for MultiPolygon input.
    """
    georef_width = float(georef["width"])
    georef_height = float(georef["height"])

    def fallback_rect() -> str:
        return (
            f'<svg><polygon points="0,{source_height} 0,0 '
            f'{source_width},0 {source_width},{source_height} 0,{source_height}" /></svg>'
        )

    if geo_polygon is None or geo_polygon.is_empty:
        return fallback_rect()

    if isinstance(geo_polygon, MultiPolygon):
        raise ValueError(
            "geo_polygon_to_svg received a MultiPolygon; only Polygon is supported."
        )

    A_fwd, A_inv = _fit_affine(georef)

    if split_canvas is not None:
        cx, cy, cw, ch = split_canvas
        scale_x = cw / georef_width
        scale_y = ch / georef_height
    else:
        cx, cy = 0.0, 0.0
        scale_x = source_width / georef_width
        scale_y = source_height / georef_height

    def geo_to_canvas(lon: float, lat: float) -> tuple[float, float]:
        px, py = _geo_to_pixel(lon, lat, A_fwd, A_inv)
        return round(cx + px * scale_x, 1), round(cy + py * scale_y, 1)

    coords = list(geo_polygon.exterior.coords)
    points = " ".join(
        f"{x},{y}" for x, y in (geo_to_canvas(lon, lat) for lon, lat in coords)
    )
    return f'<svg><polygon points="{points}" /></svg>'
