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
import contextlib
import io
import json
import math
import multiprocessing
import os
import statistics
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from mapsnap.streets import (
    DIRECTION_WORDS,
    HINT_STRINGS,
    Block,
    build_block_index,
    canonical_street_matches,
    deduplicate_detections,
    hint_type_word,
    is_bare_letter,
    is_number_only,
    normalize_street,
)
from mapsnap.utils import default_centerlines, image_stem


@dataclass
class LabelFeature:
    """Features derived from one detected street label polygon."""

    raw_text: str
    text: str  # normalize_street(raw_text)
    center: tuple[float, float]  # (px, py), mean of polygon corners
    dir_pix: float  # atan2 of long-axis edge mod π, in [0, π)
    long_side: float  # length of the longest polygon edge in pixels
    short_side: float  # length of the shortest polygon edge in pixels
    promoted: bool = False  # True for detections rescued by promote_avenue_letters


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
    rotation: float | None = None  # directed rotation in radians, set on success
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
        # For nearly-square polygons the geometry gives an unreliable direction; prefer
        # the stored dir_pix from detect_text.py (which uses the original CRAFT polygon).
        # Threshold 1.5 targets genuinely square single-character labels; more elongated
        # boxes have a reliable geometric axis and should not be overridden.
        if "dir_pix" in label and short_side > 0 and long_side / short_side < 1.5:
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
                promoted=bool(label.get("promoted")),
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


# GCPs whose pixel crossings fall within this many pixels are treated as one image
# intersection with several world candidates (a jogging street's two crossings, or a
# divided road's two carriageways), not as independent control points.
SAME_PIXEL_TOL_PX = 5.0


def _distinct_pixel_gcps(
    gcps: list[IntersectionGCP],
) -> list[list[IntersectionGCP]]:
    """Group GCPs by near-coincident pixel crossing.

    Each group is a set of GCPs sharing one image point, whose geo locations are therefore
    alternatives (e.g. the two crossings of a jogging street). The number of groups is the
    number of independent control points available to fit an affine.
    """
    groups: list[list[IntersectionGCP]] = []
    for gcp in gcps:
        for group in groups:
            if (
                math.hypot(
                    gcp.pixel[0] - group[0].pixel[0], gcp.pixel[1] - group[0].pixel[1]
                )
                <= SAME_PIXEL_TOL_PX
            ):
                group.append(gcp)
                break
        else:
            groups.append([gcp])
    return groups


def _inlier_residuals(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    A: np.ndarray,
    inlier_feat_indices: list[int],
) -> list[float]:
    """Per-inlier distance (degrees) from each mapped label to its nearest street polyline."""
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
    return residuals


# Above this many candidate GCPs, ransac_hybrid's exhaustive C(n, 2) pairing is too slow (a
# key-map index page can yield hundreds of intersections). A fast robust affine RANSAC over the
# GCP correspondences first discards gross outliers so the label-scored search runs only over
# the consensus inliers.
ROBUST_PREFILTER_MIN_GCPS = 100
# Even after the pre-filter the consensus can be large, and each pair costs a full label scoring
# (~tens of ms). The consensus is geometrically clean, so only a handful of well-separated pairs
# are needed to recover the transform; pair just the best (lowest-pixel-distance) GCPs up to this
# cap. Bounds a many-GCP page to C(cap, 2) label scorings (seconds) instead of minutes.
MAX_PREFILTER_PAIRING_GCPS = 30
_M_PER_DEG_LAT: float = 111_320.0


def _robust_affine_inlier_indices(
    gcps: list[IntersectionGCP], reproj_threshold_m: float = 30.0
) -> list[int]:
    """Indices of GCPs consistent with a robust 6-parameter affine (pixel → local metres).

    Fits a full affine to the pixel↔geo correspondences with cv2.estimateAffine2D's RANSAC
    (milliseconds even for hundreds of points) and returns its inlier set. Geo is projected to
    a local equirectangular metre frame so the reprojection threshold is in metres. Used only
    to prune gross outliers before the slower label-scored similarity search; returns all
    indices if the robust fit fails to converge.
    """
    import cv2  # lazy: heavy import, only needed for many-GCP pages

    pixels = np.array([g.pixel for g in gcps], dtype=np.float64)
    geo = np.array([g.geo for g in gcps], dtype=np.float64)
    lon0, lat0 = float(geo[:, 0].mean()), float(geo[:, 1].mean())
    cos0 = math.cos(math.radians(lat0))
    metres = np.stack(
        [
            (geo[:, 0] - lon0) * cos0 * _M_PER_DEG_LAT,
            (geo[:, 1] - lat0) * _M_PER_DEG_LAT,
        ],
        axis=1,
    )
    _, mask = cv2.estimateAffine2D(
        pixels.reshape(-1, 1, 2),
        metres.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold_m,
        maxIters=5000,
        confidence=0.999,
    )
    if mask is None:
        return list(range(len(gcps)))
    return [i for i, keep in enumerate(mask.ravel()) if keep]


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

    # For pages with very many candidate GCPs (key-map index pages), the exhaustive C(n, 2)
    # pairing is prohibitively slow. Prune gross outliers with a fast robust affine first and
    # search only over the consensus inliers; the final fit is still the label-scored similarity.
    candidate_indices = list(range(n))
    if n > ROBUST_PREFILTER_MIN_GCPS:
        consensus = _robust_affine_inlier_indices(gcps)
        if len(consensus) >= 2:
            # gcps are sorted by pixel_dist ascending, so consensus[:cap] keeps the most
            # reliable intersection estimates.
            candidate_indices = consensus[:MAX_PREFILTER_PAIRING_GCPS]
            kept = len(candidate_indices)
            print(
                f"Robust-affine pre-filter: {len(consensus)} / {n} GCP inliers; pairing the "
                f"best {kept} ({kept * (kept - 1) // 2} pairs vs {n * (n - 1) // 2})",
                file=sys.stderr,
            )

    best_A: np.ndarray | None = None
    best_inliers: list[int] = []
    best_score = -float("inf")
    best_pair: tuple[int, int] | None = None

    print("Candidate intersections:")
    for a in gcps:
        print(f"  {a.label_a} x {a.label_b}")

    for pair_idx in combinations(candidate_indices, 2):
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
    extra_fields: dict | None = None,
    parameters: dict | None = None,
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
            "parameters": parameters,
        },
        "streets": streets_out,
        "intersections": intersections_out,
    }
    if extra_fields:
        result.update(extra_fields)
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


def _are_quadrant_siblings(matches: list[str]) -> bool:
    """Return True if all matches share the same base name differing only in trailing direction suffix.

    e.g. ["NORTH STREET NORTHEAST", "NORTH STREET NORTHWEST", ...] → True
    ["NORTH PLACE", "NORTH STREET NORTHEAST"] → False
    """
    if not matches:
        return False
    bases: set[str] = set()
    for m in matches:
        parts = m.rsplit(" ", 1)
        base = parts[0] if (len(parts) == 2 and parts[1] in DIRECTION_WORDS) else m
        bases.add(base)
    return len(bases) == 1


def reading_vector(polygon: list[list[float]]) -> tuple[float, float]:
    """Directed reading vector (top-left → top-right edge) of a text polygon.

    detect_text builds polygons from EasyOCR's TL,TR,BR,BL corner order (preserved through the
    90°/270° remaps), so the first edge points in the direction the text is read. This gives an
    unambiguous before/after ordering of a letter and a type-word hint along the label.
    """
    return (polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1])


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

    Once a detection is promoted at some center position, any other candidate within
    center_dedup_px is suppressed — this prevents both 'W' and 'M' from being promoted when
    they are different OCR readings of the same physical box.

    A 'column' means the same dir_pix bucket and a perpendicular offset ≤ perp_tolerance_px.
    The parallel (along-street) gap is unconstrained.
    """
    type_hints = [
        h for h in hint_detections if h.get("text", "").upper().strip() in HINT_STRINGS
    ]
    if not type_hints:
        return []

    # Sort by confidence descending so the highest-quality reading of each physical
    # box wins the promotion slot when multiple OCR readings overlap the same position.
    candidates = sorted(
        all_detections, key=lambda d: d.get("confidence", 0.0), reverse=True
    )

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

            # dir_pix is mod π, so angle=90 and angle=270 share the same bucket.
            # Require the EasyOCR `angle` to also match so a box read in the
            # opposite direction (e.g. angle=90 vs angle=270) is not promoted.
            # This uses the orientation of the "ST" hint to distinguish "M" from "W".
            hint_angle = hint.get("angle")
            det_angle = det.get("angle")
            if (
                hint_angle is not None
                and det_angle is not None
                and hint_angle != det_angle
            ):
                continue

            det_poly = det["polygon"]
            det_cx = sum(p[0] for p in det_poly) / 4.0
            det_cy = sum(p[1] for p in det_poly) / 4.0

            # Suppress candidates too close to an already-promoted detection.
            if any(
                math.hypot(det_cx - pc[0], det_cy - pc[1]) <= center_dedup_px
                for pc in promoted_centers
            ):
                continue

            det_perp = -det_cx * sin_d + det_cy * cos_d
            if abs(det_perp - hint_perp) > perp_tolerance_px:
                continue

            # Enforce the letter/type-word order along the reading direction: "<letter> STREET"
            # (the ST hint must follow the letter) vs "AVENUE <letter>" (the AVE hint must
            # precede it). proj is the letter's offset from the hint along the reading vector.
            htype = hint_type_word(hint.get("text", ""))
            if htype is not None:
                rvx, rvy = reading_vector(hint_poly)
                proj = (det_cx - hint_cx) * rvx + (det_cy - hint_cy) * rvy
                # proj<0: letter precedes hint (… "K" before "STREET"); proj>0: letter follows
                # hint ("AVENUE" before "Q" …). Reject the wrong order for each type.
                if (htype == "STREET" and proj >= 0) or (
                    htype == "AVENUE" and proj <= 0
                ):
                    continue

            matches = canonical_street_matches(text, normalized_streets)
            if len(matches) != 1:
                # Bare letter is ambiguous. Try "N ST" style: combining the letter
                # with the hint type disambiguates lettered streets (e.g. "N STREET")
                # from direction abbreviations (e.g. "NORTH"). Allow promotion when
                # all combined matches are quadrant siblings of the same street.
                hint_type_text = hint.get("text", "").strip()
                combined_matches = canonical_street_matches(
                    f"{text} {hint_type_text}", normalized_streets
                )
                if combined_matches and _are_quadrant_siblings(combined_matches):
                    matches = combined_matches
                else:
                    continue

            promoted_centers.append((det_cx, det_cy))
            # Strip "hint" so the promoted detection is treated as a regular detection
            # downstream (deduplicate_detections and the quality filter both skip hints).
            promoted_det = {k: v for k, v in det.items() if k != "hint"}
            promoted_det["dir_pix"] = round(hint_dir % math.pi, 4)
            promoted_det["promoted"] = True
            promoted.append(promoted_det)

    return promoted


def _assembled_detection(first: dict, second: dict, text: str) -> dict:
    """One detection spanning two word-part detections, tagged ``assembled=True``.

    The polygon is the oriented box containing both parts, expressed in the reading frame of
    ``first`` (the earlier word along the reading direction) so its corner order and dir_pix
    stay consistent with ordinary detections.
    """
    rvx, rvy = reading_vector(first["polygon"])
    norm = math.hypot(rvx, rvy) or 1.0
    rvx, rvy = rvx / norm, rvy / norm
    pvx, pvy = -rvy, rvx  # perpendicular (across the text)
    corners = first["polygon"] + second["polygon"]
    along = [px * rvx + py * rvy for px, py in corners]
    across = [px * pvx + py * pvy for px, py in corners]
    r0, r1, p0, p1 = min(along), max(along), min(across), max(across)
    box = [
        (r0 * rvx + p0 * pvx, r0 * rvy + p0 * pvy),
        (r1 * rvx + p0 * pvx, r1 * rvy + p0 * pvy),
        (r1 * rvx + p1 * pvx, r1 * rvy + p1 * pvy),
        (r0 * rvx + p1 * pvx, r0 * rvy + p1 * pvy),
    ]
    return {
        "polygon": [[round(x), round(y)] for x, y in box],
        "text": text,
        "confidence": min(first.get("confidence", 0.0), second.get("confidence", 0.0)),
        "angle": first.get("angle"),
        "dir_pix": round(first.get("dir_pix", 0.0) % math.pi, 4),
        "long_side": round(r1 - r0, 1),
        "short_side": round(p1 - p0, 1),
        "assembled": True,
    }


def assemble_multiword_streets(
    detections: list[dict],
    normalized_streets: set[str],
    perp_tolerance_px: float = 20.0,
    max_word_gap_frac: float = 0.7,
    min_confidence: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Combine adjacent single-word detections into a multi-word street name.

    CRAFT often splits a label like "VAN BRUNT" into one box per word; once the parts are
    recognized (the individual name words are in the OCR vocabulary), this pairs two collinear,
    adjacent word detections whose concatenation is a known street name and emits one assembled
    detection (text "VAN BRUNT", ``assembled=True``) spanning both, so the georeferencer sees
    the whole street. This is the name-portion counterpart to promote_avenue_letters.

    Two parts pair when they share an EasyOCR ``angle`` and dir_pix bucket, lie on the same
    line (perpendicular offset ≤ perp_tolerance_px), and are adjacent along the reading
    direction (gap between boxes ≤ max_word_gap_frac of the larger box's length). Both reading
    orders are tried; the order whose concatenation matches a street wins. Returns
    (assembled detections, the source parts that were consumed) so the caller can drop the
    parts — a lone "VAN" is ambiguous on its own.
    """
    bucket_size = math.pi / 12.0
    parts = [
        d
        for d in detections
        if d.get("confidence", 0.0) >= min_confidence
        and not d.get("assembled")
        and (text := d.get("text", "").strip()).isalpha()
        and len(text) >= 3
    ]
    parts.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)

    def center(poly: list[list[float]]) -> tuple[float, float]:
        return sum(p[0] for p in poly) / 4.0, sum(p[1] for p in poly) / 4.0

    assembled: list[dict] = []
    consumed: list[dict] = []
    used: set[int] = set()
    for i, a in enumerate(parts):
        if i in used:
            continue
        a_dir = a.get("dir_pix", 0.0)
        a_bucket = int(round(a_dir / bucket_size)) % 12
        cos_d, sin_d = math.cos(a_dir), math.sin(a_dir)
        a_cx, a_cy = center(a["polygon"])
        a_perp = -a_cx * sin_d + a_cy * cos_d
        a_long = a.get("long_side", 0.0)
        for j, b in enumerate(parts):
            if j == i or j in used:
                continue
            b_dir = b.get("dir_pix", 0.0)
            if int(round(b_dir / bucket_size)) % 12 != a_bucket:
                continue
            angle_a, angle_b = a.get("angle"), b.get("angle")
            if angle_a is not None and angle_b is not None and angle_a != angle_b:
                continue
            b_cx, b_cy = center(b["polygon"])
            if abs((-b_cx * sin_d + b_cy * cos_d) - a_perp) > perp_tolerance_px:
                continue
            b_long = b.get("long_side", 0.0)
            edge_gap = math.hypot(b_cx - a_cx, b_cy - a_cy) - (a_long + b_long) / 2.0
            if edge_gap > max_word_gap_frac * max(a_long, b_long):
                continue
            ordered = None
            for first, second in ((a, b), (b, a)):
                text = f"{first['text'].strip()} {second['text'].strip()}"
                if canonical_street_matches(text, normalized_streets):
                    ordered = (text, first, second)
                    break
            if ordered is None:
                continue
            text, first, second = ordered
            assembled.append(_assembled_detection(first, second, text))
            consumed.extend([a, b])
            used.update([i, j])
            break
    return assembled, consumed


def correct_square_feature_dirs(
    features: list[LabelFeature],
    hint_detections: list[dict],
    search_radius_px: float = 200.0,
    square_threshold: float = 1.5,
) -> None:
    """Correct dir_pix for square-ish features using nearby type-word hints.

    Modifies features in-place. For each non-promoted feature whose long_side/short_side
    ratio is below square_threshold, the polygon long-axis direction is unreliable (a square
    has no dominant axis). The nearest type-word hint (AVENUE, STREET …) within
    search_radius_px is used to assign the direction. No bucket filtering is applied — for
    square features dir_pix is unreliable and cannot be used to pre-select hints.

    Promoted features are skipped: their dir_pix was already set explicitly from the correct
    column hint during promotion and must not be overwritten by a different nearby hint.
    """
    type_hints = [
        h for h in hint_detections if h.get("text", "").upper().strip() in HINT_STRINGS
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
        if feat.promoted:
            continue
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


def derive_paths(image_path: str) -> tuple[str, str, str]:
    """Derive labels, output, and per-page log paths from an image path.

    Strips everything after the first '.' in the filename to form the stem:
      p123.2048px.jpg → (p123.streets.json, p123.georef.json, p123.txt)
    """
    p = Path(image_path)
    stem = image_stem(image_path)
    base = p.parent / stem
    return (
        str(base) + ".streets.json",
        str(base) + ".georef.json",
        str(base) + ".txt",
    )


def load_detections(labels_path: str) -> list[dict]:
    """Load cached label detections from a <stem>.streets.json file."""
    labels_raw = json.load(open(labels_path))
    if isinstance(labels_raw, dict):
        return labels_raw.get(
            "streets", labels_raw.get("detections", labels_raw.get("accepted", []))
        )
    return labels_raw


# Used when a volume has no detections confident enough to derive an auto threshold.
FALLBACK_MIN_SHORT_SIDE = 20.0


def compute_auto_min_short_side(
    images: list[str],
    min_confidence: float,
    percentile: float,
    include_hints: bool = False,
) -> float | None:
    """Derive a volume-wide min_short_side floor from the volume's own detections.

    Collects short_side values from confident (>= min_confidence) street detections
    (non-ignored, with at least 2 letters in the detected text; hint detections are
    excluded unless include_hints is set) across every image's cached
    <stem>.streets.json, then returns the given percentile of that distribution.
    Returns None if no qualifying detections are found anywhere in the volume.
    """
    short_sides: list[float] = []
    for image_path in images:
        labels_path, _, _ = derive_paths(image_path)
        if not os.path.exists(labels_path):
            continue
        for det in load_detections(labels_path):
            if det.get("ignore") or (not include_hints and det.get("hint")):
                continue
            text = det.get("text", "")
            n_letters = sum(1 for c in text if c.isalpha())
            if det.get("confidence", 0.0) < min_confidence or n_letters < 2:
                continue
            short_side = det.get("short_side")
            if short_side is not None:
                short_sides.append(float(short_side))

    if not short_sides:
        return None
    if len(short_sides) == 1:
        return short_sides[0]
    quantiles = statistics.quantiles(short_sides, n=100, method="inclusive")
    index = round(percentile) - 1
    index = max(0, min(len(quantiles) - 1, index))
    return quantiles[index]


def confidence_relaxed_threshold(
    confidence: float,
    min_confidence: float,
    base_threshold: float,
    high_confidence_floor: float,
) -> float:
    """Compute the size threshold required for a detection at a given confidence.

    A detection right at min_confidence must meet base_threshold; one at confidence
    1.0 only needs to meet high_confidence_floor. Confidence values in between are
    interpolated via a power law (linear in log-confidence vs. log-threshold), so a
    smaller-than-base_threshold detection can still be admitted if it's proportionally
    more confident than min_confidence (see issue #78). Returns base_threshold
    unchanged if confidence <= min_confidence or high_confidence_floor >=
    base_threshold (i.e. relaxation is disabled).
    """
    if confidence <= min_confidence or high_confidence_floor >= base_threshold:
        return base_threshold
    confidence = min(confidence, 1.0)
    exponent = math.log(high_confidence_floor / base_threshold) / math.log(
        min_confidence
    )
    return base_threshold * (min_confidence / confidence) ** exponent


class _Tee:
    """File-like object that writes to multiple streams (used to mirror logs under --debug)."""

    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def _captured_page_log(log_path: str, debug: bool, append: bool = False):
    """Redirect stdout/stderr to log_path for the duration of the block.

    Under --debug, output is also mirrored to the real stderr in addition to being
    written to log_path. append=True is used for the second (deferred-image) pass so
    its output joins the page's first-pass log rather than overwriting it.
    """
    buf = io.StringIO()
    target = _Tee(buf, sys.stderr) if debug else buf
    with contextlib.redirect_stdout(target), contextlib.redirect_stderr(target):
        yield
    with open(log_path, "a" if append else "w") as f:
        f.write(buf.getvalue())


# Module-level state populated by _init_worker, used by _process_one_image. Each
# multiprocessing worker process gets its own copy; with --num-workers=1 this is
# populated directly in the main process instead of spawning a pool.
_worker_state: dict[str, Any] = {}


def _init_worker(
    block_index: dict[str, list[Block]],
    cos_phi: float,
    centerlines_path: str,
    min_confidence: float,
    min_long_side: float,
    min_short_side: float,
    min_aspect_ratio: float,
    edge_margin: float,
    force_intersection: tuple[int, int] | None,
    one_gcp_fits: bool,
    debug: bool,
    parameters: dict,
    high_confidence_size_fraction: float,
) -> None:
    """Populate _worker_state once per worker process (or once in the main process)."""
    _worker_state.update(
        block_index=block_index,
        cos_phi=cos_phi,
        centerlines_path=centerlines_path,
        min_confidence=min_confidence,
        min_long_side=min_long_side,
        min_short_side=min_short_side,
        min_aspect_ratio=min_aspect_ratio,
        edge_margin=edge_margin,
        force_intersection=force_intersection,
        one_gcp_fits=one_gcp_fits,
        debug=debug,
        parameters=parameters,
        high_confidence_size_fraction=high_confidence_size_fraction,
    )


def _process_one_image(image_path: str) -> tuple[str, ProcessResult]:
    """Georeference one image using _worker_state, capturing its log to a <stem>.txt sidecar.

    Runs the same way whether dispatched to a multiprocessing worker (--num-workers > 1)
    or called directly in the main process (--num-workers == 1), so a page's output and
    log file are identical either way.
    """
    labels_path, output_path, log_path = derive_paths(image_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    for stale_suffix in (
        ".georef-misscale.json",
        ".georef-outlier.json",
        ".georef-1gcp.json",
    ):
        stale_path = output_path.replace(".georef.json", stale_suffix)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    with _captured_page_log(log_path, _worker_state["debug"]):
        print(f"--- {image_path} ---")
        result = process_image(
            image_path=image_path,
            labels_path=labels_path,
            output_path=output_path,
            block_index=_worker_state["block_index"],
            cos_phi=_worker_state["cos_phi"],
            centerlines_path=_worker_state["centerlines_path"],
            min_confidence=_worker_state["min_confidence"],
            min_long_side=_worker_state["min_long_side"],
            min_short_side=_worker_state["min_short_side"],
            min_aspect_ratio=_worker_state["min_aspect_ratio"],
            edge_margin=_worker_state["edge_margin"],
            force_intersection=_worker_state["force_intersection"],
            one_gcp_fits=_worker_state["one_gcp_fits"],
            debug=_worker_state["debug"],
            parameters=_worker_state["parameters"],
            high_confidence_size_fraction=_worker_state[
                "high_confidence_size_fraction"
            ],
        )
    if result.deferred is not None:
        result.deferred["log_path"] = log_path
    return image_path, result


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
    parameters: dict | None = None,
    high_confidence_size_fraction: float = 0.7,
) -> ProcessResult:
    """Fit a georeference model for one image and write GCPs to output_path.

    Returns a ProcessResult with success=True and scale_deg_per_px set on success.
    Returns success=False otherwise. When one_gcp_fits=True and exactly 1 intersection
    GCP is found, returns success=False with deferred data set (for later median-scale
    processing). With the default one_gcp_fits=False, 1-GCP pages return success=False
    with no deferred data and are skipped entirely.
    """

    with Image.open(image_path) as pil_img:
        img_w, img_h = pil_img.size

    all_detections = load_detections(labels_path)

    print(f"All detections: {len(all_detections)}")
    all_detections = [d for d in all_detections if not d.get("ignore")]
    hint_detections = [d for d in all_detections if d.get("hint")]
    all_detections = [d for d in all_detections if not d.get("hint")]
    if hint_detections:
        confident_hints = [
            d for d in hint_detections if d.get("confidence", 0) >= min_confidence
        ]
        print(
            f"Hint detections ({len(confident_hints)}): "
            + ", ".join(d["text"] for d in confident_hints),
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
    assembled, consumed = assemble_multiword_streets(all_detections, normalized_streets)
    if assembled:
        print(
            f"Assembled multi-word detections ({len(assembled)}): "
            + ", ".join(d["text"] for d in assembled),
            file=sys.stderr,
        )
        consumed_ids = {id(d) for d in consumed}
        all_detections = assembled + [
            d for d in all_detections if id(d) not in consumed_ids
        ]
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
        else:
            required_short_side = confidence_relaxed_threshold(
                det["confidence"],
                min_confidence,
                min_short_side,
                min_short_side * high_confidence_size_fraction,
            )
            required_long_side = min_long_side * (required_short_side / min_short_side)
            if not (
                det["confidence"] >= min_confidence
                and det.get("long_side", float("inf")) >= required_long_side
                and det.get("short_side", float("inf")) >= required_short_side
                and det.get("long_side", float("inf"))
                >= min_aspect_ratio * det.get("short_side", 1.0)
                and not is_number_only(det["text"])
                # Bare single letters are accepted only via promotion (the is_promoted
                # branch above); an unpromoted "M"/"W" is almost always a rotated misread of
                # a quadrant/type box, so reject it here.
                and not is_bare_letter(det["text"])
                and (
                    normalize_street(det["text"]) not in DIRECTION_WORDS
                    or det["text"].upper().strip() in normalized_streets
                )
            ):
                continue
        if edge_margin > 0 and not is_promoted:
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

    # GCPs sharing one pixel are the same image intersection with alternative world
    # locations (a jogging street crosses its cross-street twice a few metres apart, or a
    # divided road has two carriageway crossings). They cannot pair with each other to fix
    # an affine, so when every GCP collapses onto a single image point there is just one
    # control point: defer to the median-scale 1-GCP fit, which tries each world candidate.
    if len(_distinct_pixel_gcps(gcps)) == 1:
        ix = f"{gcps[0].feat_a.raw_text} x {gcps[0].feat_b.raw_text}"
        descr = (
            ix
            if len(gcps) == 1
            else f"{ix} ({len(gcps)} world candidates at one pixel)"
        )
        if not one_gcp_fits:
            print(
                f"Only 1 distinct intersection: {descr}; skipping (use --disable-one-gcp-fits to suppress).",
                file=sys.stderr,
            )
            return ProcessResult(success=False)
        print(
            f"Only 1 distinct intersection: {descr}; deferring for median-scale processing.",
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
                "img_w": img_w,
                "img_h": img_h,
                "parameters": parameters,
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

    residuals = _inlier_residuals(features, block_index, A, inlier_feat_indices)

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
        parameters=parameters,
    )
    rotation = math.atan2(float(A[1, 0]), float(-A[1, 1]))
    return ProcessResult(
        success=True, scale_deg_per_px=scale, center=center, rotation=rotation
    )


@dataclass
class RotationDecision:
    """Outcome of the rotation-direction resolution for a 1-GCP deferred image."""

    rotation: float  # chosen directed rotation (radians)
    confirmed: bool  # True when ≥2 adjacent neighbours agreed
    method: str  # "neighbor_confirmed" | "neighbor_plurality" | "north_up_fallback"
    n_agree: int  # adjacent neighbours matching the chosen direction
    n_other: int  # adjacent neighbours matching the other direction
    adjacent_deg: list[float]  # rotation of every adjacent neighbour (degrees)


def _angle_diff_abs(a: float, b: float) -> float:
    """Smallest angular separation between two angles (radians), result in [0, π]."""
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _rotation_from_neighbors(
    candidate: float,
    approx_center: tuple[float, float],
    page_dim_deg: tuple[float, float],
    neighbor_rotations: list[tuple[tuple[float, float], float]],
    angle_threshold_rad: float = 10 * math.pi / 180,
) -> RotationDecision:
    """Select and validate a directed rotation using adjacent pages' rotation angles.

    candidate is undirected (in [0, π)); the two directed candidates are candidate and
    candidate + π. Neighbours within 1.5× page dimensions in both lon and lat are
    considered adjacent.

    If one candidate has strictly more adjacent supporters than the other, that
    candidate is chosen ('neighbor_confirmed' when the winner has ≥2, else
    'neighbor_plurality'). When both are equally supported (including 0 vs 0), the
    180° ambiguity is resolved by requiring north to point upward ('north_up_fallback').
    """
    approx_lon, approx_lat = approx_center
    page_dim_lon, page_dim_lat = page_dim_deg
    thresh_lon = 1.5 * page_dim_lon
    thresh_lat = 1.5 * page_dim_lat

    adjacent = [
        rotation
        for (n_lon, n_lat), rotation in neighbor_rotations
        if abs(n_lon - approx_lon) <= thresh_lon
        and abs(n_lat - approx_lat) <= thresh_lat
    ]
    adjacent_deg = [math.degrees(r) for r in adjacent]

    c1 = sum(
        1 for r in adjacent if _angle_diff_abs(r, candidate) <= angle_threshold_rad
    )
    c2 = sum(
        1
        for r in adjacent
        if _angle_diff_abs(r, candidate + math.pi) <= angle_threshold_rad
    )

    if c1 > c2:
        chosen = candidate
        n_agree, n_other = c1, c2
        method = "neighbor_confirmed" if c1 >= 2 else "neighbor_plurality"
    elif c2 > c1:
        chosen = candidate + math.pi
        n_agree, n_other = c2, c1
        method = "neighbor_confirmed" if c2 >= 2 else "neighbor_plurality"
    else:
        # Equal support (including 0 vs 0): north-up heuristic resolves the ambiguity.
        chosen = candidate if math.cos(candidate) >= 0 else candidate + math.pi
        n_agree, n_other = c1, c2
        method = "north_up_fallback"

    confirmed = method == "neighbor_confirmed"

    # Log decision with annotated list of every adjacent neighbour.
    adj_parts = [
        f"{r_deg:.1f}°{'✓' if _angle_diff_abs(r_rad, chosen) <= angle_threshold_rad else ''}"
        for r_deg, r_rad in zip(adjacent_deg, adjacent)
    ]
    adj_str = ", ".join(adj_parts) if adj_parts else "(no adjacent pages)"
    if confirmed:
        status_str = f"confirmed by {n_agree}/{len(adjacent)} adjacent"
    elif method == "neighbor_plurality":
        status_str = (
            f"chosen by plurality ({n_agree} vs {n_other}, {len(adjacent)} adjacent)"
        )
    else:
        status_str = f"chosen by north-up fallback ({n_agree} vs {n_other}, {len(adjacent)} adjacent)"
    print(
        f"  Rotation {math.degrees(chosen):.1f}° {status_str}: {adj_str}",
        file=sys.stderr,
    )

    return RotationDecision(
        rotation=chosen,
        confirmed=confirmed,
        method=method,
        n_agree=n_agree,
        n_other=n_other,
        adjacent_deg=adjacent_deg,
    )


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
    neighbor_rotations: list[tuple[tuple[float, float], float]],
) -> ProcessResult:
    """Georeference an image that was deferred for having a single distinct intersection.

    Uses the provided scale (deg/px) and estimates rotation from the two streets at the
    GCP. The undirected rotation is then validated against adjacent pages (already
    successfully georeferenced): if ≥2 neighbours within 1.5× page dimensions agree with
    one of the two directed candidates (±180°), that candidate is used. When both are
    equally supported, the 180° ambiguity is resolved by the north-up heuristic.

    When the intersection has several world candidates (a jogging street's two crossings),
    each is fitted and the one with the most inlier labels — then the lowest total residual —
    is kept. Confirmed fits (≥2 neighbours) are written to output_path (.georef.json);
    unconfirmed fits are written to a .georef-1gcp.json sidecar and returned success=False.
    """
    image_path: str = deferred["image_path"]
    output_path: str = deferred["output_path"]
    labels_path: str = deferred["labels_path"]
    centerlines_path: str = deferred["centerlines_path"]
    features: list[LabelFeature] = deferred["features"]
    gcps: list[IntersectionGCP] = deferred["gcps"]
    img_w: int = deferred["img_w"]
    img_h: int = deferred["img_h"]
    parameters: dict | None = deferred["parameters"]

    page_dim_lat = scale_deg_per_px * img_h
    page_dim_lon = scale_deg_per_px * img_w / cos_phi
    pos_threshold = 0.001

    # Fit each world candidate for the single image point and keep the best by inlier count,
    # then by total residual. With one candidate (the usual 1-GCP case) the loop runs once.
    best: (
        tuple[
            tuple[int, float],
            np.ndarray,
            list[int],
            list[float],
            RotationDecision,
            IntersectionGCP,
        ]
        | None
    ) = None
    for gcp in gcps:
        undirected = _rotation_from_gcp_features(gcp, block_index)
        if undirected is None:
            continue
        decision = _rotation_from_neighbors(
            candidate=undirected,
            approx_center=gcp.geo,
            page_dim_deg=(page_dim_lon, page_dim_lat),
            neighbor_rotations=neighbor_rotations,
        )
        A = build_affine_from_scale_rotation_gcp(
            scale_deg_per_px, decision.rotation, gcp, cos_phi
        )
        inliers, _ = label_inliers(
            features, block_index, A, pos_threshold, extrapolate=True
        )
        if not inliers:
            continue
        residuals = _inlier_residuals(features, block_index, A, inliers)
        # Maximise inlier count, then minimise total residual (better-aligned jog side).
        key = (len(inliers), -sum(residuals))
        if best is None or key > best[0]:
            best = (key, A, inliers, residuals, decision, gcp)

    if best is None:
        print(
            "No usable deferred fit (no rotation estimate or no inliers).",
            file=sys.stderr,
        )
        return ProcessResult(success=False)

    _, A, inlier_feat_indices, residuals, decision, gcp = best
    if len(gcps) > 1:
        print(
            f"Deferred: picked 1 of {len(gcps)} world candidates at the shared pixel.",
            file=sys.stderr,
        )
    print(
        f"Deferred: {len(inlier_feat_indices)} / {len(features)} inlier labels",
        file=sys.stderr,
    )

    actual_output_path = (
        output_path
        if decision.confirmed
        else output_path.replace(".georef.json", ".georef-1gcp.json")
    )
    one_gcp_extra = {
        "one_gcp": {
            "method": decision.method,
            "confirmed": decision.confirmed,
            "rotation_deg": round(math.degrees(decision.rotation), 2),
            "n_agree": decision.n_agree,
            "n_other": decision.n_other,
            "adjacent_rotations_deg": [round(r, 1) for r in decision.adjacent_deg],
        }
    }
    scale, center = _finalize_georef(
        A,
        features,
        gcps,
        inlier_feat_indices,
        residuals,
        image_path,
        actual_output_path,
        labels_path,
        centerlines_path,
        extra_fields=one_gcp_extra,
        parameters=parameters,
    )
    return ProcessResult(
        success=decision.confirmed, scale_deg_per_px=scale, center=center
    )


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
        "--centerlines",
        metavar="FILE",
        help=(
            "centerlines.geojson (defaults to one next to the input images or their "
            "parent directory)"
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.15,
        metavar="THRESHOLD",
        help="Minimum OCR confidence to accept a detection (default: %(default)s)",
    )
    parser.add_argument(
        "--min-long-side",
        type=float,
        default=None,
        metavar="PX",
        help=(
            "Minimum long side of a text polygon to accept. If not given, defaults to "
            "2x the (auto or explicit) min-short-side."
        ),
    )
    parser.add_argument(
        "--min-short-side",
        type=float,
        default=None,
        metavar="PX",
        help=(
            "Minimum short side of a text polygon to accept. If not given, it is "
            "derived automatically from this volume's own detections (see "
            "--auto-threshold-confidence and --auto-threshold-percentile)."
        ),
    )
    parser.add_argument(
        "--auto-threshold-confidence",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help=(
            "Minimum OCR confidence for a detection to count towards the automatic "
            "min-short-side calculation (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--auto-threshold-percentile",
        type=float,
        default=25.0,
        metavar="PCT",
        help=(
            "Percentile of confident detection short sides used as the automatic "
            "min-short-side floor (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--auto-threshold-exclude-hints",
        action="store_true",
        help=(
            "Only count street-name detections (not hint/type-word detections) "
            "towards the automatic min-short-side calculation"
        ),
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=1.75,
        metavar="RATIO",
        help="Minimum long/short side ratio for a text polygon (default: %(default)s)",
    )
    parser.add_argument(
        "--high-confidence-size-fraction",
        type=float,
        default=0.7,
        metavar="FRACTION",
        help=(
            "Smaller detections are admitted if they're proportionally more "
            "confident than --min-confidence: a detection at confidence 1.0 only "
            "needs to meet this fraction of min-long-side/min-short-side, with "
            "confidences in between interpolated on a power-law curve. Set to 1.0 "
            "to disable (default: %(default)s)"
        ),
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
        default=0.0,
        metavar="FRAC",
        help=(
            "Ignore detections whose center is within this fraction of the image edge "
            "(default: %(default)s, disabled). Headers, stamps, and marginal text are filtered out."
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
        "--disable-one-gcp-fits",
        action="store_true",
        default=False,
        help=(
            "Disable georeferencing of pages with only 1 intersection GCP. "
            "By default, such pages are attempted using the median scale from other pages."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print additional debug information"
    )
    parser.add_argument(
        "--geocode_keymaps",
        action="store_true",
        help="Geocode key maps in addition to regular pages.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of worker processes for the per-image fitting pass "
            "(default: %(default)s, sequential)"
        ),
    )
    args = parser.parse_args()

    if args.centerlines is None:
        centerlines = default_centerlines(Path(args.images[0]).parent)
        if centerlines is None:
            sys.exit(
                "No --centerlines given and no centerlines.geojson found next to the "
                "input images."
            )
        args.centerlines = str(centerlines)
        print(f"Using centerlines: {args.centerlines}", file=sys.stderr)

    # A key-map index page (one with a sibling <stem>.keymap.json) has no streets to fit, so
    # ignore it for georeferencing. Other images still require a <stem>.streets.json downstream.
    kept_images = []
    for image_path in args.images:
        stem = image_stem(str(image_path))
        keymap_path = os.path.join(os.path.dirname(image_path), f"{stem}.keymap.json")
        if os.path.exists(keymap_path) and not args.geocode_keymaps:
            print(
                f"Skipping {image_path}: key-map page (pass --geocode_keymaps to geocode)",
                file=sys.stderr,
            )
        else:
            kept_images.append(image_path)
    args.images = kept_images

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

    auto_threshold_include_hints = not args.auto_threshold_exclude_hints
    if args.min_short_side is not None:
        min_short_side = args.min_short_side
    else:
        auto_min_short_side = compute_auto_min_short_side(
            args.images,
            args.auto_threshold_confidence,
            args.auto_threshold_percentile,
            auto_threshold_include_hints,
        )
        detection_kind = (
            "detections" if auto_threshold_include_hints else "street detections"
        )
        if auto_min_short_side is None:
            min_short_side = FALLBACK_MIN_SHORT_SIDE
            print(
                f"No confident (>= {args.auto_threshold_confidence:g}) {detection_kind} "
                f"found; falling back to min-short-side={min_short_side:g}px.",
                file=sys.stderr,
            )
        else:
            min_short_side = auto_min_short_side
            print(
                f"Auto min-short-side: {min_short_side:.1f}px "
                f"(p{args.auto_threshold_percentile:g} of confidence>="
                f"{args.auto_threshold_confidence:g} {detection_kind})",
                file=sys.stderr,
            )
    min_long_side = (
        args.min_long_side if args.min_long_side is not None else 2 * min_short_side
    )
    print(
        f"Thresholds: min_confidence={args.min_confidence:g} "
        f"min_long_side={min_long_side:.1f}px min_short_side={min_short_side:.1f}px",
        file=sys.stderr,
    )
    parameters = {
        "min_confidence": args.min_confidence,
        "min_long_side": min_long_side,
        "min_short_side": min_short_side,
        "min_aspect_ratio": args.min_aspect_ratio,
        "auto_threshold_confidence": args.auto_threshold_confidence,
        "auto_threshold_percentile": args.auto_threshold_percentile,
        "auto_threshold_include_hints": auto_threshold_include_hints,
        "high_confidence_size_fraction": args.high_confidence_size_fraction,
    }

    n_success = 0
    scales: list[float] = []
    scale_records: list[
        tuple[str, float]
    ] = []  # (image_path, px_per_ft) for outlier check
    location_records: list[tuple[str, tuple[float, float]]] = []  # (image_path, center)
    neighbor_rotations: list[
        tuple[tuple[float, float], float]
    ] = []  # (center, rotation)
    deferred_list: list[dict] = []

    worker_initargs = (
        block_index,
        cos_phi,
        args.centerlines,
        args.min_confidence,
        min_long_side,
        min_short_side,
        args.min_aspect_ratio,
        args.edge_margin,
        force_intersection,
        not args.disable_one_gcp_fits,
        args.debug,
        parameters,
        args.high_confidence_size_fraction,
    )
    if args.num_workers > 1:
        with multiprocessing.Pool(
            args.num_workers, initializer=_init_worker, initargs=worker_initargs
        ) as pool:
            page_results = list(
                tqdm(
                    pool.imap_unordered(_process_one_image, args.images),
                    total=len(args.images),
                    smoothing=0,
                )
            )
    else:
        _init_worker(*worker_initargs)
        page_results = [
            _process_one_image(image_path)
            for image_path in tqdm(args.images, smoothing=0)
        ]

    for image_path, result in page_results:
        if result.success:
            n_success += 1
            if result.scale_deg_per_px is not None:
                px_per_ft = deg_per_px_to_px_per_ft(result.scale_deg_per_px)
                scales.append(px_per_ft)
                scale_records.append((image_path, px_per_ft))
            if result.center is not None:
                location_records.append((image_path, result.center))
                if result.rotation is not None:
                    neighbor_rotations.append((result.center, result.rotation))
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
                with _captured_page_log(deferred["log_path"], args.debug, append=True):
                    print(f"\n--- {deferred_image_path} (deferred) ---")
                    deferred_result = process_deferred_image(
                        deferred=deferred,
                        scale_deg_per_px=scale_deg_per_px,
                        block_index=block_index,
                        cos_phi=cos_phi,
                        neighbor_rotations=neighbor_rotations,
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
                _, out_path, _ = derive_paths(img_path)
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
                _, out_path, _ = derive_paths(img_path)
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
