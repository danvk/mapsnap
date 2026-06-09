"""Georeference a map image from detected street label polygons.

Uses street-label line intersections as geographic control points (GCPs), then fits
a pixel→(lon,lat) affine transform via exhaustive 3-point sampling. Intersections are
found by detecting pairs of labeled streets that share a coordinate in the centerlines
GeoJSON (meaning they physically meet), then computing the pixel crossing of their label
direction lines.

Output is the georeference-branch web app format: {width, height, points: [{x, y, lat, lon}]},
which can be pasted into the textarea to preview the warped image on a MapLibre map.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from mapsnap.streets import (
    DIRECTION_WORDS,
    STREET_TYPES,
    Block,
    build_block_index,
    canonical_street_matches,
    deduplicate_detections,
    is_number_only,
    normalize_street,
)
from mapsnap.utils import image_stem


@dataclass
class LabelFeature:
    """Features derived from one detected street label polygon."""

    raw_text: str
    text: str  # normalize_street(raw_text)
    center: tuple[float, float]  # (px, py), mean of polygon corners
    dir_pix: float  # atan2 of long-axis edge mod π, in [0, π)
    long_side: float  # length of the longest polygon edge in pixels
    short_side: float  # length of the shortest polygon edge in pixels


@dataclass
class IntersectionGCP:
    """Geographic control point from two detected street labels whose streets cross."""

    label_a: str
    label_b: str
    pixel: tuple[float, float]  # pixel line-line crossing of the two label directions
    geo: tuple[float, float]  # shared GeoJSON endpoint (centroid if multiple)
    pixel_dist: float  # distance between label centers
    feat_a: LabelFeature
    feat_b: LabelFeature


@dataclass
class ProcessResult:
    """Outcome of processing a single map image."""

    success: bool
    scale_deg_per_px: float | None = None  # set on success
    center: tuple[float, float] | None = None  # (lon, lat) page center, set on success
    deferred: dict | None = None  # set when deferred (exactly 1 GCP, needs scale)


def label_features(labels: list[dict]) -> list[LabelFeature]:
    """Extract center and long-axis direction from each detected label polygon."""
    features = []
    for label in labels:
        pts = np.array(label["polygon"], dtype=float)
        center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        sides = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
        side_lengths = [float(np.linalg.norm(s)) for s in sides]
        long_side = max(side_lengths)
        short_side = min(side_lengths)
        long_vec = max(sides, key=np.linalg.norm)
        dir_pix_geom = float(np.arctan2(long_vec[1], long_vec[0])) % np.pi
        # For square polygons the geometry gives an unreliable direction; prefer the
        # stored dir_pix from detect_text.py (which uses the original CRAFT polygon).
        if "dir_pix" in label and short_side > 0 and long_side / short_side < 2.0:
            dir_pix = label["dir_pix"]
        else:
            dir_pix = dir_pix_geom
        features.append(
            LabelFeature(
                raw_text=label["text"],
                text=label.get("_canonical_text") or normalize_street(label["text"]),
                center=center,
                dir_pix=dir_pix,
                long_side=long_side,
                short_side=short_side,
            )
        )
    return features


def solve_affine_3pts(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray:
    """Solve for a 2×3 affine matrix from exactly 3 (pixel, geo) point pairs.

    Returns A where [lon, lat]^T = A @ [px, py, 1]^T.
    Raises np.linalg.LinAlgError if the 3 pixel points are collinear.
    """
    M = np.zeros((6, 6))
    b = np.zeros(6)
    for k, ((px, py), (lon, lat)) in enumerate(pairs):
        M[2 * k, :] = [px, py, 1, 0, 0, 0]
        b[2 * k] = lon
        M[2 * k + 1, :] = [0, 0, 0, px, py, 1]
        b[2 * k + 1] = lat
    x = np.linalg.solve(M, b)
    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]]])


def apply_affine(A: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Apply a 2×3 affine matrix to a pixel point, returning (lon, lat)."""
    result = A @ np.array([px, py, 1.0])
    return float(result[0]), float(result[1])


def compute_cos_phi(block_index: dict[str, list[Block]]) -> float:
    """Estimate cos(latitude) from the mean latitude of all block coordinates."""
    lats = []
    for blocks in block_index.values():
        for block in blocks:
            lats.extend(block.coords[:, 1].tolist())
    mean_lat = float(np.mean(lats)) if lats else 40.7
    return float(np.cos(np.radians(mean_lat)))


def solve_similarity_2pts(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    cos_phi: float,
) -> np.ndarray:
    """Solve for a similarity-constrained 2×3 affine from exactly 2 (pixel, geo) pairs.

    Enforces equal metric scale in x and y and orthogonality (no shear), parameterized
    as (α, β, tx, ty). Since pixel y increases downward while lat increases upward, the
    transform is orientation-reversing:
      lon = (α/cos_phi)·px + (β/cos_phi)·py + tx
      lat =  β·px          + (−α)·py         + ty

    Returns A where [lon, lat]^T = A @ [px, py, 1]^T.
    Raises np.linalg.LinAlgError if the pixel points are identical or collinear.
    """
    M = np.zeros((4, 4))
    b = np.zeros(4)
    for k, ((px, py), (lon, lat)) in enumerate(pairs):
        M[2 * k] = [px / cos_phi, py / cos_phi, 1, 0]
        b[2 * k] = lon
        M[2 * k + 1] = [-py, px, 0, 1]
        b[2 * k + 1] = lat
    x = np.linalg.solve(M, b)
    alpha, beta, tx, ty = x
    return np.array([[alpha / cos_phi, beta / cos_phi, tx], [beta, -alpha, ty]])


_FT_PER_DEG_LAT: float = math.pi * 20_925_524.0 / 180.0  # feet per degree latitude


def _dist_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate great-circle distance in km between two (lon, lat) points."""
    cos_lat = math.cos(math.radians((a[1] + b[1]) / 2))
    dlat_km = (b[1] - a[1]) * 111.0
    dlon_km = (b[0] - a[0]) * 111.0 * cos_lat
    return math.sqrt(dlat_km**2 + dlon_km**2)


def affine_scale_deg_per_px(A: np.ndarray) -> float:
    """Return the scale (degrees latitude per pixel) from a 2×3 similarity affine."""
    return float(math.sqrt(A[1, 0] ** 2 + A[1, 1] ** 2))


def deg_per_px_to_px_per_ft(scale: float) -> float:
    """Convert scale from degrees-per-pixel to pixels-per-foot."""
    return 1.0 / (scale * _FT_PER_DEG_LAT)


def px_per_ft_to_deg_per_px(px_per_ft: float) -> float:
    """Convert scale from pixels-per-foot to degrees-per-pixel."""
    return 1.0 / (px_per_ft * _FT_PER_DEG_LAT)


def build_affine_from_scale_rotation_gcp(
    scale: float,
    rotation: float,
    gcp: IntersectionGCP,
    cos_phi: float,
) -> np.ndarray:
    """Build a 2×3 similarity affine from scale, rotation, and one anchor GCP.

    With α = scale·cos(rotation) and β = scale·sin(rotation), the model is:
      lon = (α/cos_phi)·px + (β/cos_phi)·py + tx
      lat = β·px + (−α)·py + ty
    Translation (tx, ty) is solved so that the GCP pixel maps to its geo coordinate.
    """
    alpha = scale * math.cos(rotation)
    beta = scale * math.sin(rotation)
    px0, py0 = gcp.pixel
    lon0, lat0 = gcp.geo
    tx = lon0 - (alpha / cos_phi) * px0 - (beta / cos_phi) * py0
    ty = lat0 - beta * px0 + alpha * py0
    return np.array([[alpha / cos_phi, beta / cos_phi, tx], [beta, -alpha, ty]])


def pixel_line_intersection(
    ca: np.ndarray,
    dir_a: float,
    cb: np.ndarray,
    dir_b: float,
    max_dist_factor: float = 3.0,
) -> tuple[float, float] | None:
    """Find where two street label lines cross in pixel space.

    Each line passes through its label center along the label's long-axis direction.
    Returns None if lines are nearly parallel or if the crossing is implausibly far
    from either center (more than max_dist_factor × the center-to-center distance).
    """
    da = np.array([np.cos(dir_a), np.sin(dir_a)])
    db = np.array([np.cos(dir_b), np.sin(dir_b)])
    # Solve ca + t*da = cb + s*db  →  [da, -db] @ [t, s]^T = cb - ca
    M = np.array([[da[0], -db[0]], [da[1], -db[1]]])
    det = float(np.linalg.det(M))
    if abs(det) < 0.1:
        return None
    ts = np.linalg.solve(M, cb - ca)
    pt = ca + float(ts[0]) * da
    center_dist = float(np.linalg.norm(cb - ca))
    if center_dist < 1.0:
        return None
    if float(np.linalg.norm(pt - ca)) > max_dist_factor * center_dist:
        return None
    if float(np.linalg.norm(pt - cb)) > max_dist_factor * center_dist:
        return None
    return (float(pt[0]), float(pt[1]))


def _terminal_endpoints(blocks: list[Block]) -> set[tuple[float, float]]:
    """Return the set of block endpoints that connect to no other block.

    Each Block's first and last coordinate is counted. An endpoint is terminal
    when it appears in exactly one block (i.e. the street does not continue past
    that point in the OSM data).
    """
    count: dict[tuple[float, float], int] = {}
    for block in blocks:
        for ep in (
            (float(block.coords[0, 0]), float(block.coords[0, 1])),
            (float(block.coords[-1, 0]), float(block.coords[-1, 1])),
        ):
            count[ep] = count.get(ep, 0) + 1
    return {ep for ep, n in count.items() if n == 1}


def project_to_polyline(
    lon: float,
    lat: float,
    blocks: list[Block],
    extrapolate: bool = True,
    max_extrapolation_ft: float = 500.0,
) -> tuple[float, float, float] | None:
    """Project (lon, lat) onto the nearest segment across all blocks of a street.

    When extrapolate=True (default), terminal endpoints (block start/end that connects
    to no other block) allow the projection to extend beyond the endpoint along the
    segment direction, up to max_extrapolation_ft feet. This handles labels that lie
    past the end of an OSM segment because the historical street extended further than
    current data.

    When extrapolate=False, projection is always clamped to [0, 1] on each segment,
    so the returned point is guaranteed to lie on a true street segment.

    Returns (nearest_lon, nearest_lat, tangent_angle) where tangent_angle is in [0, π)
    (undirected — a street running NE and SW both have the same tangent angle).
    Returns None if blocks is empty.
    """
    q = np.array([lon, lat])
    best_dist = float("inf")
    best_pt: np.ndarray | None = None
    best_tangent: float = 0.0

    terminal = _terminal_endpoints(blocks) if extrapolate else set()

    for block in blocks:
        pts = block.coords  # shape (N, 2), columns [lon, lat]
        n_segs = len(pts) - 1
        start_terminal = (float(pts[0, 0]), float(pts[0, 1])) in terminal
        end_terminal = (float(pts[-1, 0]), float(pts[-1, 1])) in terminal
        for k in range(n_segs):
            p1, p2 = pts[k], pts[k + 1]
            seg = p2 - p1
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq < 1e-20:
                continue
            t_raw = float(np.dot(q - p1, seg) / seg_len_sq)
            is_start_terminal = k == 0 and start_terminal and extrapolate
            is_end_terminal = k == n_segs - 1 and end_terminal and extrapolate
            t_min = 0.0
            t_max = 1.0
            if is_start_terminal or is_end_terminal:
                mid_lat = float((p1[1] + p2[1]) / 2)
                ft_per_deg_lon = _FT_PER_DEG_LAT * math.cos(math.radians(mid_lat))
                seg_ft = math.sqrt(
                    (float(seg[0]) * ft_per_deg_lon) ** 2
                    + (float(seg[1]) * _FT_PER_DEG_LAT) ** 2
                )
                max_t = (max_extrapolation_ft / seg_ft) if seg_ft > 0 else 0.0
                if is_start_terminal:
                    t_min = -max_t
                if is_end_terminal:
                    t_max = 1.0 + max_t
            t = float(np.clip(t_raw, t_min, t_max))
            nearest = p1 + t * seg
            dist = float(np.linalg.norm(q - nearest))
            if dist < best_dist:
                best_dist = dist
                best_pt = nearest
                best_tangent = float(np.arctan2(seg[1], seg[0])) % np.pi

    if best_pt is None:
        return None
    return (float(best_pt[0]), float(best_pt[1]), best_tangent)


def label_inliers(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    affine: np.ndarray,
    pos_threshold: float = 0.001,
    dir_threshold: float = np.pi / 6,
    extrapolate: bool = True,
    debug: bool = False,
) -> tuple[list[int], float]:
    """Return indices of features whose label center is consistent with the affine.

    A label is an inlier when the affine maps its pixel center to a point that is:
      - within pos_threshold degrees of its street's polyline, AND
      - whose mapped direction agrees with the polyline tangent at that point within
        dir_threshold radians (undirected comparison).

    extrapolate is passed through to project_to_polyline; set False when scoring
    candidate orientations to avoid terminal-endpoint extrapolation inflating the fit.
    """
    A_linear = affine[:, :2]
    # Keyed on feat.center so that one raw detection (expanded to multiple canonical
    # street names, e.g. JEFFERSON → EAST JEFFERSON + WEST JEFFERSON) counts at most
    # once as an inlier, using the canonical match with the smallest positional error.
    best_by_center: dict[tuple[float, float], tuple[int, float]] = {}
    for i, feat in enumerate(features):
        blocks = block_index.get(feat.text)
        if not blocks:
            continue
        lon, lat = apply_affine(affine, *feat.center)
        snap = project_to_polyline(lon, lat, blocks, extrapolate=extrapolate)
        if snap is None:
            continue
        nearest_lon, nearest_lat, tangent_angle = snap
        pos_err = float(
            np.linalg.norm(np.array([lon - nearest_lon, lat - nearest_lat]))
        )
        if pos_err > pos_threshold:
            continue
        d_pixel = np.array([np.cos(feat.dir_pix), np.sin(feat.dir_pix)])
        d_geo = A_linear @ d_pixel
        d_geo_norm = d_geo / (float(np.linalg.norm(d_geo)) + 1e-10)
        tangent_vec = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
        # Undirected: accept if either direction aligns
        if abs(float(np.dot(d_geo_norm, tangent_vec))) < np.cos(dir_threshold):
            continue
        if (
            feat.center not in best_by_center
            or pos_err < best_by_center[feat.center][1]
        ):
            best_by_center[feat.center] = (i, pos_err)
    inliers = [idx for idx, _ in best_by_center.values()]
    inlier_err = sum(err for _, err in best_by_center.values())
    if debug:
        for idx, err in best_by_center.values():
            print(f"    {idx} {err:.06f} {features[idx].text}")
    return inliers, inlier_err


_CLUSTER_THRESHOLD_FT: float = 60.0


def _cluster_geo_coords(
    coords: list[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Group geographic coordinates into clusters by proximity (single-linkage, 60ft threshold).

    Points within 60ft of any point already in a cluster are merged into that cluster.
    Each returned cluster represents one distinct intersection event; callers should emit
    one GCP per cluster using the cluster centroid as the geo coordinate.

    A 60ft threshold separates same-intersection nodes (nearby GeoJSON vertices) from
    distinct intersection events such as jogging streets (Newport St jogs ~115ft along
    East Jefferson Ave) or divided-highway carriageway crossings.
    """
    if not coords:
        return []
    mean_lat = sum(p[1] for p in coords) / len(coords)
    ft_per_deg_lon = _FT_PER_DEG_LAT * math.cos(math.radians(mean_lat))
    clusters: list[list[tuple[float, float]]] = []
    for pt in coords:
        merged = False
        for cluster in clusters:
            for c in cluster:
                dist_ft = math.sqrt(
                    ((pt[0] - c[0]) * ft_per_deg_lon) ** 2
                    + ((pt[1] - c[1]) * _FT_PER_DEG_LAT) ** 2
                )
                if dist_ft < _CLUSTER_THRESHOLD_FT:
                    cluster.append(pt)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            clusters.append([pt])
    return clusters


def find_intersection_gcps(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
) -> list[IntersectionGCP]:
    """Find GCPs from pairs of detected labels whose streets share a GeoJSON coordinate.

    A shared coordinate means the streets physically meet at that point. The pixel
    coordinate is the crossing of the two label direction lines (extrapolated from each
    label center). Pairs are skipped if the lines are nearly parallel or if the crossing
    falls implausibly far from both labels.

    Shared coordinates are clustered by proximity (60ft threshold). Each cluster becomes
    a separate GCP candidate with its own centroid geo coordinate. This correctly handles
    jogging streets (e.g. Newport St crosses East Jefferson Ave at two points 115ft apart)
    and divided highways (two carriageway crossings) — both yield two candidate GCPs that
    RANSAC can evaluate independently.

    Returns GCPs sorted by pixel_dist ascending (closer labels = better pixel estimate).
    """
    street_coords: dict[str, set[tuple[float, float]]] = {}
    for feat in features:
        blocks = block_index.get(feat.text, [])
        if not blocks:
            continue
        pts: set[tuple[float, float]] = set()
        for block in blocks:
            for pt in block.coords:
                pts.add((round(float(pt[0]), 7), round(float(pt[1]), 7)))
        street_coords[feat.text] = pts

    # Group all detected label instances by normalized street name. Multiple
    # instances arise when a street bends (two labels for the same street) or
    # when two distinct historical streets share a modern name (e.g. Court
    # Street vs Court Square both detected as "COURT").
    feats_by_text: dict[str, list[LabelFeature]] = {}
    for f in features:
        if f.text in street_coords:
            feats_by_text.setdefault(f.text, []).append(f)
    texts = list(feats_by_text.keys())
    gcps: list[IntersectionGCP] = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            text_a, text_b = texts[i], texts[j]
            shared = street_coords[text_a] & street_coords[text_b]
            if not shared:
                continue
            geo_clusters = _cluster_geo_coords(sorted(shared))
            for cluster in geo_clusters:
                cluster_pts = np.array(cluster)
                geo = (float(cluster_pts[:, 0].mean()), float(cluster_pts[:, 1].mean()))
                # For each (text_a, text_b, cluster), keep only the best (fa, fb) pair
                # by pixel_dist. All pairs share the same geo centroid, so extra pairs
                # only add RANSAC iterations without adding independent intersection anchors.
                best: IntersectionGCP | None = None
                for fa in feats_by_text[text_a]:
                    for fb in feats_by_text[text_b]:
                        ca, cb = np.array(fa.center), np.array(fb.center)
                        pixel_dist = float(np.linalg.norm(ca - cb))
                        crossing = pixel_line_intersection(
                            ca, fa.dir_pix, cb, fb.dir_pix
                        )
                        if crossing is None:
                            continue
                        if best is None or pixel_dist < best.pixel_dist:
                            best = IntersectionGCP(
                                label_a=text_a,
                                label_b=text_b,
                                pixel=crossing,
                                geo=geo,
                                pixel_dist=pixel_dist,
                                feat_a=fa,
                                feat_b=fb,
                            )
                if best is not None:
                    gcps.append(best)

    return sorted(gcps, key=lambda g: g.pixel_dist)


def ransac_hybrid(
    gcps: list[IntersectionGCP],
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    cos_phi: float,
    pos_threshold: float = 0.001,
    dir_threshold: float = np.pi / 6,
    force_pair: tuple[int, int] | None = None,
    debug: bool = False,
) -> tuple[np.ndarray | None, list[int], tuple[int, int] | None]:
    """Find the best similarity affine using intersection GCPs as seeds, label-based inlier scoring.

    Generates candidate affines from all C(n, 2) pairs of intersection GCPs, solving for
    the 4-parameter similarity transform (equal metric scale + orthogonality) at each pair.
    Inlier counting uses point-to-polyline + direction for every label.

    A global rotation estimate R_est is derived from a histogram cross-correlation of label
    angles vs OSM tangents before the main loop. Each candidate is penalized by rot_penalty
    (inlier-equivalents per radian) for deviating from R_est, biasing selection toward
    geometrically consistent rotations without hard-fixing rotation or changing the solver.

    Returns the best affine and a list of inlier indices into `features`.
    If force_pair is set, all pairs are still evaluated and printed, but that
    pair is selected regardless of score.
    """
    n = len(gcps)
    if n < 2:
        return None, [], None

    best_A: np.ndarray | None = None
    best_inliers: list[int] = []
    best_score = -float("inf")
    best_pair: tuple[int, int] | None = None

    print("Candidate intersections:")
    for a in gcps:
        print(f"  {a.label_a} x {a.label_b}")

    for pair_idx in combinations(range(n), 2):
        i1, i2 = pair_idx
        a, b = gcps[i1], gcps[i2]
        # Skip pairs that map to the same OSM intersection — singular by construction.
        if abs(a.geo[0] - b.geo[0]) < 1e-6 and abs(a.geo[1] - b.geo[1]) < 1e-6:
            continue
        pairs = [(gcps[k].pixel, gcps[k].geo) for k in pair_idx]
        try:
            A = solve_similarity_2pts(pairs, cos_phi)
        except np.linalg.LinAlgError:
            continue
        inliers, err = label_inliers(
            features, block_index, A, pos_threshold, dir_threshold, debug=debug
        )

        # Each inlier contributes (pos_threshold - pos_err) to the score; an inlier at
        # exactly the threshold boundary contributes 0, so a marginal extra inlier cannot
        # blindly beat a tighter fit with one fewer inlier.
        score = float(len(inliers)) * pos_threshold - err

        should_print = debug
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_A = A
            best_pair = pair_idx
            should_print = True
        if should_print:
            print(
                f"  {i1} {i2} {score:.06f} ({len(inliers)}) {a.label_a} x {a.label_b} + {b.label_a} x {b.label_b}"
            )

    if force_pair is not None:
        i1, i2 = force_pair
        forced_pairs = [(gcps[i1].pixel, gcps[i1].geo), (gcps[i2].pixel, gcps[i2].geo)]
        try:
            forced_A = solve_similarity_2pts(forced_pairs, cos_phi)
            forced_inliers, _ = label_inliers(
                features, block_index, forced_A, pos_threshold, dir_threshold
            )
            print(f"Forcing pair {force_pair}.", file=sys.stderr)
            return forced_A, forced_inliers, force_pair
        except np.linalg.LinAlgError:
            print(
                f"Forced pair {force_pair} is degenerate; falling back to best.",
                file=sys.stderr,
            )

    return best_A, best_inliers, best_pair


def _finalize_georef(
    A: np.ndarray,
    features: list[LabelFeature],
    gcps: list[IntersectionGCP],
    inlier_feat_indices: list[int],
    residuals: list[float],
    image_path: str,
    output_path: str,
    labels_path: str,
    centerlines_path: str,
    initial_pair: tuple[int, int] | None = None,
) -> tuple[float, tuple[float, float]]:
    """Print fit stats, write georef JSON, and return (scale_deg_per_px, page_center)."""

    if residuals:
        mean_m = float(np.mean(residuals)) * 111_000
        max_m = float(max(residuals)) * 111_000
        print(f"  mean={mean_m:.1f}m  max={max_m:.1f}m", file=sys.stderr)

    a, b, tx = float(A[0, 0]), float(A[0, 1]), float(A[0, 2])
    c, d, ty = float(A[1, 0]), float(A[1, 1]), float(A[1, 2])
    print("\nAffine (pixel → lon/lat):", file=sys.stderr)
    print(f"  lon = {a:.8f}·px + {b:.8f}·py + {tx:.6f}", file=sys.stderr)
    print(f"  lat = {c:.8f}·px + {d:.8f}·py + {ty:.6f}", file=sys.stderr)
    print(
        f"GDAL geotransform: ({tx:.6f}, {a:.8f}, {b:.8f}, {ty:.6f}, {c:.8f}, {d:.8f})",
        file=sys.stderr,
    )

    with Image.open(image_path) as img:
        width, height = img.size

    corners = [
        list(apply_affine(A, 0, 0)),
        list(apply_affine(A, width, 0)),
        list(apply_affine(A, width, height)),
        list(apply_affine(A, 0, height)),
    ]

    A_linear = A[:, :2]
    inlier_feat_set = set(inlier_feat_indices)
    streets_out = []
    for i, feat in enumerate(features):
        lon, lat = apply_affine(A, *feat.center)
        d_pixel = np.array([np.cos(feat.dir_pix), np.sin(feat.dir_pix)])
        d_geo = A_linear @ d_pixel
        d_geo_norm = d_geo / np.linalg.norm(d_geo)
        streets_out.append(
            {
                "street": feat.raw_text,
                "x": round(feat.center[0]),
                "y": round(feat.center[1]),
                "lat": round(lat, 7),
                "lon": round(lon, 7),
                "dir_x": round(float(np.cos(feat.dir_pix)), 6),
                "dir_y": round(float(np.sin(feat.dir_pix)), 6),
                "dir_lon": round(float(d_geo_norm[0]), 6),
                "dir_lat": round(float(d_geo_norm[1]), 6),
                "inlier": i in inlier_feat_set,
            }
        )

    initial_set = set(initial_pair) if initial_pair is not None else set()
    intersections_out = [
        {
            "label_a": gcp.label_a,
            "label_b": gcp.label_b,
            "x": round(gcp.pixel[0]),
            "y": round(gcp.pixel[1]),
            "lat": round(gcp.geo[1], 7),
            "lon": round(gcp.geo[0], 7),
            "inlier": float(
                np.linalg.norm(
                    np.array(apply_affine(A, *gcp.pixel)) - np.array(gcp.geo)
                )
            )
            <= 0.001,
            "initial": i in initial_set,
        }
        for i, gcp in enumerate(gcps)
    ]

    result = {
        "width": width,
        "height": height,
        "corners": corners,
        "inputs": {
            "labels": labels_path,
            "centerlines": centerlines_path,
            "image": image_path,
        },
        "streets": streets_out,
        "intersections": intersections_out,
    }
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=2))
    print(
        f"Wrote {len(streets_out)} streets and {len(intersections_out)} intersection GCPs"
        f" to {output_path}",
        file=sys.stderr,
    )
    center = (
        sum(c[0] for c in corners) / 4,
        sum(c[1] for c in corners) / 4,
    )
    return affine_scale_deg_per_px(A), center


def assemble_hint_groups(
    detections: list[dict],
    hint_detections: list[dict],
    normalized_streets: set[str],
    block_index: dict[str, list[Block]] | None = None,
    min_confidence: float = 0.5,
    max_gap_px: float = 30.0,
    perp_tolerance_px: float = 15.0,
) -> list[dict]:
    """Assemble collinear hint+name token sequences into unambiguous street detections.

    When CRAFT splits one physical label (e.g. "EAST SEVENTH ST") into separate boxes,
    this groups spatially adjacent detections and checks whether their assembled text
    resolves to a single known street name. Returns one synthetic detection per
    unambiguous group.

    Only detections meeting min_confidence are considered. This filters low-quality
    noise detections (e.g. a misread street that overlaps a real one) that would
    otherwise contaminate runs with spurious non-hint tokens.

    Tokens are bucketed by dir_pix (15° buckets). Within each bucket, items are
    clustered into perp lanes by single-linkage on their perpendicular coordinate,
    then sweep-merged by par within each lane. Each run with exactly one non-hint
    token and at least one hint is tried for assembly. Both forward and reversed
    token orderings are tried so that direction prefixes (e.g. "N") that appear
    after the street name in par order (labels reading bottom-to-top) still match.
    """
    detections = [d for d in detections if d.get("confidence", 0.0) >= min_confidence]
    hint_detections = [
        d for d in hint_detections if d.get("confidence", 0.0) >= min_confidence
    ]
    all_candidates = detections + hint_detections
    if not all_candidates or not hint_detections:
        return []

    bucket_size = math.pi / 12.0  # 15° per bucket; 12 buckets cover [0, π)
    buckets: dict[int, list[dict]] = {}
    for det in all_candidates:
        dir_pix = det.get("dir_pix", 0.0)
        bucket_key = int(round(dir_pix / bucket_size)) % 12
        buckets.setdefault(bucket_key, []).append(det)

    assembled: list[dict] = []

    for group in buckets.values():
        if not any(d.get("hint") for d in group):
            continue

        mean_dir = sum(d.get("dir_pix", 0.0) for d in group) / len(group)
        cos_d = math.cos(mean_dir)
        sin_d = math.sin(mean_dir)

        # Project each detection center onto (parallel, perpendicular) axes.
        annotated: list[tuple[float, float, dict]] = []
        for det in group:
            poly = det["polygon"]
            cx = sum(p[0] for p in poly) / 4.0
            cy = sum(p[1] for p in poly) / 4.0
            par = cx * cos_d + cy * sin_d
            perp = -cx * sin_d + cy * cos_d
            annotated.append((par, perp, det))

        # Cluster into perp lanes using single-linkage on perp values (sorted first
        # so grouping is stable regardless of par order). Items from different text
        # rows are separated; the par sweep then runs within each lane only.
        annotated_by_perp = sorted(annotated, key=lambda t: t[1])
        perp_lanes: list[list[tuple[float, float, dict]]] = [[annotated_by_perp[0]]]
        for item in annotated_by_perp[1:]:
            if abs(item[1] - perp_lanes[-1][-1][1]) <= perp_tolerance_px:
                perp_lanes[-1].append(item)
            else:
                perp_lanes.append([item])

        # Sweep-merge within each perp lane: consecutive items (by par) join the
        # same run when edge-to-edge gap ≤ max_gap_px.
        # Gap is edge-to-edge: center_gap − (long_side_a + long_side_b) / 2.
        runs: list[list[tuple[float, float, dict]]] = []
        for lane in perp_lanes:
            lane.sort(key=lambda t: t[0])
            current_run = [lane[0]]
            for i in range(1, len(lane)):
                prev_par, _, prev_det = lane[i - 1]
                curr_par, _, curr_det = lane[i]
                center_gap = curr_par - prev_par
                half_extents = (
                    prev_det.get("long_side", 0.0) + curr_det.get("long_side", 0.0)
                ) / 2.0
                edge_gap = center_gap - half_extents
                if edge_gap <= max_gap_px:
                    current_run.append(lane[i])
                else:
                    runs.append(current_run)
                    current_run = [lane[i]]
            runs.append(current_run)

        for run in runs:
            hint_tokens = [d for _, _, d in run if d.get("hint")]
            non_hint_tokens = [d for _, _, d in run if not d.get("hint")]
            if not hint_tokens or len(non_hint_tokens) != 1:
                continue

            # Preserve par order for natural left-to-right / top-to-bottom assembly.
            tokens = [d for _, _, d in run]
            # Normalize the joined string, not each token independently.
            # normalize_street("ST.") alone gives "SAINT" (PREFIX_ABBREVS applies to
            # first word), but normalize_street("SEVENTH ST.") correctly gives
            # "SEVENTH STREET" (STREET_ABBREVS applies to non-first words).
            # Try both forward and reversed order: for vertical labels a direction
            # prefix (e.g. "N") may fall below the street name in par order
            # (labels reading bottom-to-top), so "STATE N" reversed gives "N STATE".
            canonical: str | None = None
            for ordering in [tokens, list(reversed(tokens))]:
                text = normalize_street(" ".join(d["text"] for d in ordering))
                raw_matches = canonical_street_matches(text, normalized_streets)
                # Deduplicate by block-list identity: "WEST COLUMBIA" and
                # "WEST COLUMBIA AVENUE" point to the same list object, so they
                # count as one street here just as they do in process_image.
                # When block_index is None (e.g. in tests), skip dedup.
                if block_index is not None:
                    seen_ids: set[int] = set()
                    deduped: list[str] = []
                    for m in raw_matches:
                        bid = id(block_index[m])
                        if bid not in seen_ids:
                            seen_ids.add(bid)
                            deduped.append(m)
                else:
                    deduped = list(raw_matches)
                if len(deduped) == 1:
                    canonical = deduped[0]
                    break
            if canonical is None:
                continue

            run_conf = min(d["confidence"] for d in non_hint_tokens)

            # Build bounding polygon aligned to dir_pix from all constituent corners.
            all_pars: list[float] = []
            all_perps: list[float] = []
            for d in tokens:
                for px, py in d["polygon"]:
                    all_pars.append(px * cos_d + py * sin_d)
                    all_perps.append(-px * sin_d + py * cos_d)
            par_min, par_max = min(all_pars), max(all_pars)
            perp_min, perp_max = min(all_perps), max(all_perps)

            # Convert (par, perp) corners back to (x, y) pixel coordinates.
            polygon = [
                [int(p * cos_d - q * sin_d), int(p * sin_d + q * cos_d)]
                for p, q in [
                    (par_min, perp_min),
                    (par_max, perp_min),
                    (par_max, perp_max),
                    (par_min, perp_max),
                ]
            ]

            assembled.append(
                {
                    "polygon": polygon,
                    "text": canonical,
                    "confidence": round(run_conf, 4),
                    "angle": non_hint_tokens[0].get("angle", 0),
                    "long_side": round(par_max - par_min, 1),
                    "short_side": round(perp_max - perp_min, 1),
                    "dir_pix": round(mean_dir % math.pi, 4),
                    "assembled": True,
                }
            )

    return assembled


def promote_avenue_letters(
    hint_detections: list[dict],
    all_detections: list[dict],
    normalized_streets: set[str],
    perp_tolerance_px: float = 20.0,
    center_dedup_px: float = 10.0,
) -> list[dict]:
    """Promote small single-char detections that sit in the same column as a type-word hint.

    When a type-word hint ('AVENUE', 'STREET') shares a direction column with a single-char
    detection ('X', 'J') that unambiguously matches a lettered avenue/street via bare alias,
    the detection is returned with `promoted=True` so the quality filter bypasses its size
    checks, and its dir_pix is corrected from the hint.

    Single-char direction-abbreviation hints ('W', 'N', 'E', 'S') in hint_detections are
    also eligible candidates. They are processed before non-hint detections so that e.g. 'W'
    (correctly read as Avenue W) takes priority over a same-position non-hint misread like 'M'.
    Once a detection is promoted at some center position, any other candidate within
    center_dedup_px is suppressed — this prevents both 'W' and 'M' from being promoted when
    they are different OCR readings of the same physical box.

    A 'column' means the same dir_pix bucket and a perpendicular offset ≤ perp_tolerance_px.
    The parallel (along-street) gap is unconstrained.
    """
    type_hints = [
        h for h in hint_detections if h.get("text", "").upper().strip() in STREET_TYPES
    ]
    if not type_hints:
        return []

    # Single-char direction-abbrev hints (e.g. "W" for Avenue W) are candidates too.
    # Prepend them so they win the center-based dedup over same-position non-hint misreads.
    single_char_dir_hints = [
        h
        for h in hint_detections
        if len(h.get("text", "").strip()) == 1
        and h.get("text", "").upper().strip() not in STREET_TYPES
    ]
    candidates = single_char_dir_hints + list(all_detections)

    if not candidates:
        return []

    bucket_size = math.pi / 12.0
    promoted: list[dict] = []
    promoted_centers: list[tuple[float, float]] = []

    for hint in type_hints:
        hint_dir = hint.get("dir_pix", 0.0)
        hint_bucket = int(round(hint_dir / bucket_size)) % 12
        cos_d = math.cos(hint_dir)
        sin_d = math.sin(hint_dir)

        hint_poly = hint["polygon"]
        hint_cx = sum(p[0] for p in hint_poly) / 4.0
        hint_cy = sum(p[1] for p in hint_poly) / 4.0
        hint_perp = -hint_cx * sin_d + hint_cy * cos_d

        for det in candidates:
            text = det.get("text", "").strip()
            if len(text) != 1:
                continue
            if det.get("promoted") or det.get("assembled"):
                continue

            det_dir = det.get("dir_pix", 0.0)
            det_bucket = int(round(det_dir / bucket_size)) % 12
            if det_bucket != hint_bucket:
                continue

            det_poly = det["polygon"]
            det_cx = sum(p[0] for p in det_poly) / 4.0
            det_cy = sum(p[1] for p in det_poly) / 4.0

            # Suppress candidates too close to an already-promoted detection.
            if any(
                math.hypot(det_cx - pc[0], det_cy - pc[1]) < center_dedup_px
                for pc in promoted_centers
            ):
                continue

            det_perp = -det_cx * sin_d + det_cy * cos_d
            if abs(det_perp - hint_perp) > perp_tolerance_px:
                continue

            matches = canonical_street_matches(text, normalized_streets)
            if len(matches) != 1:
                continue

            promoted_centers.append((det_cx, det_cy))
            # Strip "hint" so the promoted detection is treated as a regular detection
            # downstream (deduplicate_detections and the quality filter both skip hints).
            promoted_det = {k: v for k, v in det.items() if k != "hint"}
            promoted_det["dir_pix"] = round(hint_dir % math.pi, 4)
            promoted_det["promoted"] = True
            promoted.append(promoted_det)

    return promoted


def correct_square_feature_dirs(
    features: list[LabelFeature],
    hint_detections: list[dict],
    search_radius_px: float = 200.0,
    square_threshold: float = 2.0,
) -> None:
    """Correct dir_pix for square-ish features using nearby type-word hints.

    Modifies features in-place. For each feature whose long_side/short_side ratio is below
    square_threshold, the polygon long-axis direction is unreliable (a square has no
    dominant axis). The nearest type-word hint (AVENUE, STREET …) within search_radius_px
    is used to assign the direction. No bucket filtering is applied — for square features
    dir_pix is unreliable and cannot be used to pre-select hints.
    """
    type_hints = [
        h for h in hint_detections if h.get("text", "").upper().strip() in STREET_TYPES
    ]
    if not type_hints:
        return

    hint_info: list[tuple[float, float, float]] = []
    for hint in type_hints:
        poly = hint["polygon"]
        hcx = sum(p[0] for p in poly) / 4.0
        hcy = sum(p[1] for p in poly) / 4.0
        hdir = hint.get("dir_pix", 0.0)
        hint_info.append((hcx, hcy, hdir))

    for feat in features:
        if feat.short_side <= 0:
            continue
        if feat.long_side / feat.short_side >= square_threshold:
            continue

        feat_cx, feat_cy = feat.center

        best_dist = float("inf")
        best_dir: float | None = None
        for hcx, hcy, hdir in hint_info:
            dist = math.hypot(feat_cx - hcx, feat_cy - hcy)
            if dist < best_dist and dist <= search_radius_px:
                best_dist = dist
                best_dir = hdir

        if best_dir is not None:
            feat.dir_pix = round(best_dir % math.pi, 4)


def derive_paths(image_path: str) -> tuple[str, str]:
    """Derive labels and output paths from an image path.

    Strips everything after the first '.' in the filename to form the stem:
      p123.2048px.jpg → (p123.streets.json, p123.georef.json)
    """
    p = Path(image_path)
    stem = image_stem(image_path)
    base = p.parent / stem
    return str(base) + ".streets.json", str(base) + ".georef.json"


def process_image(
    image_path: str,
    labels_path: str,
    output_path: str,
    block_index: dict[str, list[Block]],
    cos_phi: float,
    centerlines_path: str,
    min_confidence: float = 0.5,
    min_long_side: float = 250.0,
    min_short_side: float = 60.0,
    min_aspect_ratio: float = 2.0,
    edge_margin: float = 0.02,
    force_intersection: tuple[int, int] | None = None,
    one_gcp_fits: bool = False,
    debug: bool = False,
) -> ProcessResult:
    """Fit a georeference model for one image and write GCPs to output_path.

    Returns a ProcessResult with success=True and scale_deg_per_px set on success.
    Returns success=False with deferred data when exactly 1 intersection GCP is found
    (needs median scale from other maps to complete). Returns success=False otherwise.
    """

    with Image.open(image_path) as pil_img:
        img_w, img_h = pil_img.size

    labels_raw = json.load(open(labels_path))
    if isinstance(labels_raw, dict):
        all_detections: list[dict] = labels_raw.get(
            "streets", labels_raw.get("detections", labels_raw.get("accepted", []))
        )
    else:
        all_detections = labels_raw

    print(f"All detections: {len(all_detections)}")
    all_detections = [d for d in all_detections if not d.get("ignore")]
    hint_detections = [d for d in all_detections if d.get("hint")]
    all_detections = [d for d in all_detections if not d.get("hint")]
    if hint_detections:
        print(
            f"Hint detections ({len(hint_detections)}): "
            + ", ".join(d["text"] for d in hint_detections),
            file=sys.stderr,
        )
    normalized_streets = set(block_index.keys())
    promoted = promote_avenue_letters(
        hint_detections, all_detections, normalized_streets
    )
    if promoted:
        print(
            f"Promoted detections ({len(promoted)}): "
            + ", ".join(d["text"] for d in promoted),
            file=sys.stderr,
        )
        all_detections = promoted + all_detections
    all_detections = deduplicate_detections(
        all_detections, normalized_streets=normalized_streets
    )
    print(f"Deduped detections: {len(all_detections)}")
    labels_data = []
    for det in all_detections:
        is_promoted = det.get("promoted")
        if is_promoted:
            # Promoted single-char avenue letters bypass size/aspect checks; confidence only.
            if det["confidence"] < min_confidence or is_number_only(det["text"]):
                continue
        elif not (
            det["confidence"] >= min_confidence
            and det.get("long_side", float("inf")) >= min_long_side
            and det.get("short_side", float("inf")) >= min_short_side
            and det.get("long_side", float("inf"))
            >= min_aspect_ratio * det.get("short_side", 1.0)
            and not is_number_only(det["text"])
            and (
                normalize_street(det["text"]) not in DIRECTION_WORDS
                or det["text"].upper().strip() in DIRECTION_WORDS
                or det["text"].upper().strip() in normalized_streets
            )
        ):
            continue
        if edge_margin > 0:
            poly = np.array(det["polygon"], dtype=float)
            cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
            if (
                cx < edge_margin * img_w
                or cx > (1 - edge_margin) * img_w
                or cy < edge_margin * img_h
                or cy > (1 - edge_margin) * img_h
            ):
                continue
        canonicals = canonical_street_matches(det["text"], normalized_streets)
        # Deduplicate by block-list identity: aliases like "HENRY" and "HENRY STREET"
        # map to the *same list object* in block_index (set by build_block_index), so
        # id() detects them as duplicates and keeps only the longest (most specific) name.
        seen_block_ids: set[int] = set()
        for canonical in sorted(canonicals, key=len, reverse=True):
            bid = id(block_index[canonical])
            if bid in seen_block_ids:
                continue
            seen_block_ids.add(bid)
            if canonical != normalize_street(det["text"]):
                entry = dict(det)
                entry["_canonical_text"] = canonical
            else:
                entry = det
            labels_data.append(entry)

    print(f"Filtered detections: {len(labels_data)}")

    features = label_features(labels_data)
    correct_square_feature_dirs(features, hint_detections)
    print(
        f"Labels ({len(features)}): {', '.join(f.text for f in features)}",
        file=sys.stderr,
    )

    gcps = find_intersection_gcps(features, block_index)
    print(f"Intersection GCPs: {len(gcps)}", file=sys.stderr)
    if len(gcps) == 0:
        print("No intersection GCPs found.", file=sys.stderr)
        return ProcessResult(success=False)
    if len(gcps) == 1:
        ix = f"{gcps[0].feat_a.raw_text} x {gcps[0].feat_b.raw_text}"
        if not one_gcp_fits:
            print(
                f"Only 1 intersection GCP: {ix}; skipping (use --one-gcp-fits to enable).",
                file=sys.stderr,
            )
            return ProcessResult(success=False)
        print(
            f"Only 1 intersection GCP: {ix}; deferring for median-scale processing.",
            file=sys.stderr,
        )
        return ProcessResult(
            success=False,
            deferred={
                "image_path": image_path,
                "output_path": output_path,
                "labels_path": labels_path,
                "centerlines_path": centerlines_path,
                "features": features,
                "gcps": gcps,
            },
        )

    A, inlier_feat_indices, seed_pair = ransac_hybrid(
        gcps, features, block_index, cos_phi, force_pair=force_intersection, debug=debug
    )
    if A is None:
        print("RANSAC failed: no valid affine found.", file=sys.stderr)
        return ProcessResult(success=False)
    print(
        f"RANSAC: {len(inlier_feat_indices)} / {len(features)} inlier labels",
        file=sys.stderr,
    )

    residuals: list[float] = []
    for i in inlier_feat_indices:
        feat = features[i]
        blocks = block_index.get(feat.text)
        if not blocks:
            continue
        lon, lat = apply_affine(A, *feat.center)
        snap = project_to_polyline(lon, lat, blocks)
        if snap is None:
            continue
        nearest_lon, nearest_lat, _ = snap
        residuals.append(
            float(np.linalg.norm(np.array([lon - nearest_lon, lat - nearest_lat])))
        )

    scale, center = _finalize_georef(
        A,
        features,
        gcps,
        inlier_feat_indices,
        residuals,
        image_path,
        output_path,
        labels_path,
        centerlines_path,
        initial_pair=seed_pair,
    )
    return ProcessResult(success=True, scale_deg_per_px=scale, center=center)


def _rotation_from_gcp_features(
    gcp: IntersectionGCP,
    block_index: dict[str, list[Block]],
) -> float | None:
    """Estimate map rotation (rad, mod π) from the two streets that form a single GCP.

    For each street, projects the GCP geo coordinate onto the street's polyline to get
    the OSM tangent angle, then combines it with the label's pixel direction angle via
    R = (dir_pix + osm_tangent) mod π. Returns the circular mean of the two estimates,
    or None if either street's polyline projection fails.
    """
    Rs: list[float] = []
    for feat, label in ((gcp.feat_a, gcp.label_a), (gcp.feat_b, gcp.label_b)):
        blocks = block_index.get(label, [])
        snap = project_to_polyline(gcp.geo[0], gcp.geo[1], blocks)
        if snap is None:
            return None
        _, _, osm_tangent = snap
        Rs.append((feat.dir_pix + osm_tangent) % math.pi)
    z = np.exp(2j * Rs[0]) + np.exp(2j * Rs[1])
    return float(np.angle(z) / 2) % math.pi


def process_deferred_image(
    deferred: dict,
    scale_deg_per_px: float,
    block_index: dict[str, list[Block]],
    cos_phi: float,
) -> ProcessResult:
    """Georeference an image that was deferred due to having only 1 intersection GCP.

    Uses the provided scale (deg/px) and estimates rotation from label-angle histogram
    cross-correlation. The rotation estimate is undirected (mod π); the 180° ambiguity
    is resolved by requiring north to point upward in the image (A[1,1] < 0, equivalently
    cos(rotation) > 0).
    """
    image_path: str = deferred["image_path"]
    output_path: str = deferred["output_path"]
    labels_path: str = deferred["labels_path"]
    centerlines_path: str = deferred["centerlines_path"]
    features: list[LabelFeature] = deferred["features"]
    gcps: list[IntersectionGCP] = deferred["gcps"]
    gcp = gcps[0]

    rotation = _rotation_from_gcp_features(gcp, block_index)
    if rotation is None:
        print("Could not estimate rotation from GCP street angles.", file=sys.stderr)
        return ProcessResult(success=False)

    # The rotation estimate is mod π; pick the directed angle where north points up.
    # In the similarity affine, A[1,1] = -scale·cos(rotation); north is up when
    # A[1,1] < 0, i.e. cos(rotation) > 0.
    if math.cos(rotation) < 0:
        rotation += math.pi

    pos_threshold = 0.001
    A = build_affine_from_scale_rotation_gcp(scale_deg_per_px, rotation, gcp, cos_phi)
    inlier_feat_indices, _ = label_inliers(
        features, block_index, A, pos_threshold, extrapolate=True
    )

    if not inlier_feat_indices:
        print("No inliers found for deferred image.", file=sys.stderr)
        return ProcessResult(success=False)
    print(
        f"Deferred: {len(inlier_feat_indices)} / {len(features)} inlier labels",
        file=sys.stderr,
    )

    residuals: list[float] = []
    for i in inlier_feat_indices:
        feat = features[i]
        blocks = block_index.get(feat.text)
        if not blocks:
            continue
        lon, lat = apply_affine(A, *feat.center)
        snap = project_to_polyline(lon, lat, blocks)
        if snap is None:
            continue
        nearest_lon, nearest_lat, _ = snap
        residuals.append(
            float(np.linalg.norm(np.array([lon - nearest_lon, lat - nearest_lat])))
        )

    scale, center = _finalize_georef(
        A,
        features,
        gcps,
        inlier_feat_indices,
        residuals,
        image_path,
        output_path,
        labels_path,
        centerlines_path,
    )
    return ProcessResult(success=True, scale_deg_per_px=scale, center=center)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a pixel→(lon,lat) affine from detected street label polygons "
            "and output GCPs in the georeference web app format."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help=(
            "Input image file(s). Labels are read from <stem>.streets.json "
            "and output is written to <stem>.georef.json."
        ),
    )
    parser.add_argument(
        "--centerlines", required=True, metavar="FILE", help="centerlines.geojson"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="Minimum OCR confidence to accept a detection (default: 0.5)",
    )
    parser.add_argument(
        "--min-long-side",
        type=float,
        default=250.0,
        metavar="PX",
        help="Minimum long side of a text polygon to accept (default: 250)",
    )
    parser.add_argument(
        "--min-short-side",
        type=float,
        default=60.0,
        metavar="PX",
        help="Minimum short side of a text polygon to accept (default: 60)",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=2.0,
        metavar="RATIO",
        help="Minimum long/short side ratio for a text polygon (default: 2.0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="PX_PER_FT",
        help=(
            "Override the scale (pixels per foot) used for deferred single-GCP images. "
            "If not set, the median scale from successfully georeferenced images is used."
        ),
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.02,
        metavar="FRAC",
        help=(
            "Ignore detections whose center is within this fraction of the image edge "
            "(default: 0.02 = 2%%). Headers, stamps, and marginal text are filtered out."
        ),
    )
    parser.add_argument(
        "--force-intersection",
        metavar="I,J",
        default=None,
        help=(
            "Force RANSAC to use GCP pair I,J (0-based indices printed during the run) "
            "regardless of score. Only valid with a single image."
        ),
    )
    parser.add_argument(
        "--scale-outlier-threshold",
        type=float,
        default=0.25,
        metavar="FRAC",
        help=(
            "Delete georef output files whose fitted scale deviates from the reference "
            "scale by more than this fraction (default: 0.25 = 25%%). "
            "Set to 0 to disable. Reference is --scale if given, else the median."
        ),
    )
    parser.add_argument(
        "--min-distance-for-outlier-km",
        type=float,
        default=1.5,
        metavar="KM",
        help=(
            "Rename georefs whose center is more than this many km from every other "
            "georeferenced page to .georef-outlier.json (default: 1.0). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--one-gcp-fits",
        action="store_true",
        default=False,
        help=(
            "Attempt to georeference pages with only 1 intersection GCP using the "
            "median scale from other pages (less reliable; disabled by default)."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print additional debug information"
    )
    args = parser.parse_args()

    force_intersection: tuple[int, int] | None = None
    if args.force_intersection is not None:
        if len(args.images) != 1:
            parser.error("--force-intersection requires exactly one image argument")
        parts = args.force_intersection.split(",")
        if len(parts) != 2:
            parser.error(
                "--force-intersection must be two comma-separated integers, e.g. '2,3'"
            )
        force_intersection = (int(parts[0]), int(parts[1]))

    geojson: dict = json.load(open(args.centerlines))
    block_index = build_block_index(geojson)
    cos_phi = compute_cos_phi(block_index)
    n_blocks = sum(len(v) for v in block_index.values())
    print(
        f"Block index: {n_blocks} segments across {len(block_index)} streets",
        file=sys.stderr,
    )

    n_success = 0
    scales: list[float] = []
    scale_records: list[
        tuple[str, float]
    ] = []  # (image_path, px_per_ft) for outlier check
    location_records: list[tuple[str, tuple[float, float]]] = []  # (image_path, center)
    deferred_list: list[dict] = []

    for image_path in args.images:
        if len(args.images) > 1:
            print(f"\n--- {image_path} ---", file=sys.stderr)
        labels_path, output_path = derive_paths(image_path)
        # Delete stale georef file before attempting a new fit so a failed run
        # doesn't leave a previous result in place.
        if os.path.exists(output_path):
            os.remove(output_path)
        for stale_suffix in (".georef-misscale.json", ".georef-outlier.json"):
            stale_path = output_path.replace(".georef.json", stale_suffix)
            if os.path.exists(stale_path):
                os.remove(stale_path)
        result = process_image(
            image_path=image_path,
            labels_path=labels_path,
            output_path=output_path,
            block_index=block_index,
            cos_phi=cos_phi,
            centerlines_path=args.centerlines,
            min_confidence=args.min_confidence,
            min_long_side=args.min_long_side,
            min_short_side=args.min_short_side,
            min_aspect_ratio=args.min_aspect_ratio,
            edge_margin=args.edge_margin,
            force_intersection=force_intersection,
            one_gcp_fits=args.one_gcp_fits,
            debug=args.debug,
        )
        if result.success:
            n_success += 1
            if result.scale_deg_per_px is not None:
                px_per_ft = deg_per_px_to_px_per_ft(result.scale_deg_per_px)
                scales.append(px_per_ft)
                scale_records.append((image_path, px_per_ft))
            if result.center is not None:
                location_records.append((image_path, result.center))
        elif result.deferred is not None:
            deferred_list.append(result.deferred)

    # Determine reference scale: explicit override takes priority, then median.
    if args.scale is not None:
        ref_scale_px_per_ft: float | None = args.scale
    elif scales:
        ref_scale_px_per_ft = float(np.median(scales))
    else:
        ref_scale_px_per_ft = None

    # Print median scale whenever multiple images were processed.
    if scales and len(args.images) > 1:
        print(
            f"\nMedian scale: {float(np.median(scales)):.4f} px/ft ({len(scales)} images)",
            file=sys.stderr,
        )

    # Process deferred (1-GCP) images using the reference scale.
    if deferred_list:
        if ref_scale_px_per_ft is None:
            print(
                f"\n{len(deferred_list)} deferred image(s) skipped: no scale available. "
                "Use --scale PX_PER_FT to set manually.",
                file=sys.stderr,
            )
        else:
            scale_deg_per_px = px_per_ft_to_deg_per_px(ref_scale_px_per_ft)
            for deferred in deferred_list:
                deferred_image_path: str = deferred["image_path"]
                if len(args.images) > 1:
                    print(
                        f"\n--- {deferred_image_path} (deferred) ---", file=sys.stderr
                    )
                deferred_result = process_deferred_image(
                    deferred=deferred,
                    scale_deg_per_px=scale_deg_per_px,
                    block_index=block_index,
                    cos_phi=cos_phi,
                )
                if deferred_result.success:
                    n_success += 1
                    if deferred_result.center is not None:
                        location_records.append(
                            (deferred_image_path, deferred_result.center)
                        )

    # Drop georef files whose fitted scale is a major outlier vs the reference.
    if ref_scale_px_per_ft is not None and args.scale_outlier_threshold > 0:
        n_dropped = 0
        for img_path, px_per_ft in scale_records:
            ratio = px_per_ft / ref_scale_px_per_ft
            if abs(ratio - 1.0) > args.scale_outlier_threshold:
                _, out_path = derive_paths(img_path)
                misscale_path = out_path.replace(
                    ".georef.json", ".georef-misscale.json"
                )
                if os.path.exists(out_path):
                    os.rename(out_path, misscale_path)
                    n_success -= 1
                    n_dropped += 1
                    print(
                        f"Dropped scale outlier {img_path}: "
                        f"{px_per_ft:.4f} px/ft vs reference {ref_scale_px_per_ft:.4f} "
                        f"({ratio:.2f}×) → {misscale_path}",
                        file=sys.stderr,
                    )
        if n_dropped:
            print(f"Dropped {n_dropped} scale outlier(s).", file=sys.stderr)

    # Drop georef files whose center is far from every other georeferenced page.
    if args.min_distance_for_outlier_km > 0 and len(location_records) >= 2:
        # Only pages whose .georef.json still exists (not already renamed by scale check).
        active = [
            (p, c) for p, c in location_records if os.path.exists(derive_paths(p)[1])
        ]
        n_dropped = 0
        for img_path, center in active:
            other_centers = [c for p, c in active if p != img_path]
            min_dist_km = min(_dist_km(center, other) for other in other_centers)
            if min_dist_km > args.min_distance_for_outlier_km:
                _, out_path = derive_paths(img_path)
                outlier_path = out_path.replace(".georef.json", ".georef-outlier.json")
                os.rename(out_path, outlier_path)
                n_success -= 1
                n_dropped += 1
                print(
                    f"Dropped location outlier {img_path}: "
                    f"{min_dist_km:.1f} km from closest map → {outlier_path}",
                    file=sys.stderr,
                )
        if n_dropped:
            print(f"Dropped {n_dropped} location outlier(s).", file=sys.stderr)

    if len(args.images) > 1:
        print(f"\n{n_success}/{len(args.images)} images georeferenced", file=sys.stderr)


if __name__ == "__main__":
    main()
