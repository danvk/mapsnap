"""Compute block-based clipping polygons for Sanborn map pages.

Street network polygonization divides the mapped region into city blocks.
Each block is assigned to exactly one page (the one with the greatest
intersection area; tie-break: closest page centroid), producing gapless,
non-overlapping clipping masks.
"""

import math
import sys
from typing import cast

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry import shape as geom_shape
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import polygonize, unary_union

# ~0.5 miles in degrees latitude (constant regardless of location).
_BUFFER_LAT_DEG = 0.5 * 1609.344 / 111_320.0


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
) -> dict[int, list[int]]:
    """Assign each block to the page whose centroid is nearest (Voronoi assignment).

    Each block is assigned to the page with the minimum centroid-to-centroid
    distance. Only pages with non-zero intersection with the block are eligible.
    Tie-break: maximum intersection area.

    Distance-based (Voronoi) assignment naturally partitions the overlap zone
    along perpendicular bisectors between page centers, producing connected
    page territories. The earlier area-based approach let overlapping pages
    steal blocks from deep inside a page's interior, creating disconnected masks.

    Returns a dict mapping page_index → list of block indices.
    Blocks with zero intersection with every page are dropped.
    """
    page_centroids = [p.centroid for p in page_polys]
    assignment: dict[int, list[int]] = {i: [] for i in range(len(page_polys))}

    for block_idx, block in enumerate(blocks):
        block_centroid = block.centroid
        best_page = -1
        best_dist = float("inf")
        best_area = 0.0

        for page_idx, page_poly in enumerate(page_polys):
            inter = block.intersection(page_poly)
            area = inter.area
            if area <= 0:
                continue
            dist = block_centroid.distance(page_centroids[page_idx])
            if dist < best_dist or (dist == best_dist and area > best_area):
                best_dist = dist
                best_page = page_idx
                best_area = area

        if best_page >= 0:
            assignment[best_page].append(block_idx)

    return assignment


def compute_all_clip_masks(
    georefs: list[dict],
    centerlines_geojson: dict,
    simplify_tolerance: float = 0.00005,
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

    assignment = _assign_blocks_to_pages(blocks, page_polys)

    masks: list[Polygon | None] = []
    for page_idx, georef in enumerate(georefs):
        block_indices = assignment.get(page_idx, [])
        if not block_indices:
            masks.append(None)
            continue

        assigned = [blocks[i] for i in block_indices]
        mask_geo = unary_union(assigned).intersection(page_polys[page_idx])

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

        # Warn if interior holes are present (unexpected with city block geometry).
        if len(mask_geo.interiors) > 0:
            print(
                f"Warning: page {page_idx} mask has {len(mask_geo.interiors)} interior "
                f"hole(s); only the exterior will be used. Corners: {georef['corners'][0]}",
                file=sys.stderr,
            )

        masks.append(cast(Polygon, mask_geo))

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
