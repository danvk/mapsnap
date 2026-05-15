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

from mapsnap.detect_text import (
    DIRECTION_WORDS,
    canonical_street_matches,
    deduplicate_detections,
    is_number_only,
    normalize_street,
    visualize_detections,
)


@dataclass
class Block:
    """One line segment from centerlines.geojson."""

    street_name: str
    coords: np.ndarray  # shape (N, 2), columns [lon, lat]


@dataclass
class LabelFeature:
    """Features derived from one detected street label polygon."""

    raw_text: str
    text: str  # normalize_street(raw_text)
    center: tuple[float, float]  # (px, py), mean of polygon corners
    dir_pix: float  # atan2 of long-axis edge mod π, in [0, π)


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
    deferred: dict | None = None  # set when deferred (exactly 1 GCP, needs scale)


def label_features(labels: list[dict]) -> list[LabelFeature]:
    """Extract center and long-axis direction from each detected label polygon."""
    features = []
    for label in labels:
        pts = np.array(label["polygon"], dtype=float)
        center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        sides = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
        long_vec = max(sides, key=np.linalg.norm)
        dir_pix = float(np.arctan2(long_vec[1], long_vec[0])) % np.pi
        features.append(
            LabelFeature(
                raw_text=label["text"],
                text=label.get("_canonical_text") or normalize_street(label["text"]),
                center=center,
                dir_pix=dir_pix,
            )
        )
    return features


def build_block_index(geojson: dict) -> dict[str, list[Block]]:
    """Index GeoJSON centerline features by normalized street_name.

    Also adds unambiguous base-name aliases so that bare labels (e.g. "MAGAZINE")
    match full centerline names (e.g. "MAGAZINE STREET"). An alias is only added
    when exactly one full name shares that base, to avoid false matches like
    "MAGAZINE" matching both "Magazine Street" and "Magazine Place".
    """
    index: dict[str, list[Block]] = {}
    for feature in geojson["features"]:
        raw_name = feature["properties"].get("street_name", "")
        if not raw_name:
            continue
        name = normalize_street(raw_name)
        geom = feature["geometry"]
        lines = (
            geom["coordinates"]
            if geom["type"] == "MultiLineString"
            else [geom["coordinates"]]
        )
        for line in lines:
            if len(line) < 2:
                continue
            coords = np.array([[c[0], c[1]] for c in line], dtype=float)
            index.setdefault(name, []).append(Block(street_name=name, coords=coords))

    # Build aliases from bare name → full name for unambiguous cases.
    # e.g. "MAGAZINE STREET" → also index under "MAGAZINE".
    street_type_words = set(
        normalize_street(v)
        for v in [
            "Street",
            "Avenue",
            "Boulevard",
            "Place",
            "Drive",
            "Road",
            "Court",
            "Lane",
            "Terrace",
            "Highway",
            "Parkway",
            "Circle",
            "Expressway",
        ]
    )
    from collections import defaultdict

    base_to_full: dict[str, list[str]] = defaultdict(list)
    for key in index:
        parts = key.rsplit(" ", 1)
        if len(parts) == 2 and parts[1] in street_type_words:
            base_to_full[parts[0]].append(key)
    for base, full_names in base_to_full.items():
        if len(full_names) == 1 and base not in index:
            index[base] = index[full_names[0]]

    # Build aliases with direction prefix stripped for unambiguous cases.
    # e.g. "SOUTH LIBERTY STREET" → also index under "LIBERTY STREET" and "LIBERTY"
    # (the latter via the bare-name alias already added above).
    # Iterating list(index.keys()) captures both original keys and the aliases just added.
    dir_stripped_to_full: dict[str, list[str]] = defaultdict(list)
    for key in list(index.keys()):
        parts = key.split(" ", 1)
        if len(parts) == 2 and parts[0] in DIRECTION_WORDS and parts[1] not in index:
            dir_stripped_to_full[parts[1]].append(key)
    for stripped, full_names in dir_stripped_to_full.items():
        if len(full_names) == 1:
            index[stripped] = index[full_names[0]]

    return index


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
) -> tuple[float, float, float] | None:
    """Project (lon, lat) onto the nearest segment across all blocks of a street.

    When extrapolate=True (default), terminal endpoints (block start/end that connects
    to no other block) allow the projection to extend beyond the endpoint along the
    segment direction. This handles labels that lie past the end of an OSM segment
    because the historical street extended further than current data.

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
            t_min = -float("inf") if (k == 0 and start_terminal) else 0.0
            t_max = float("inf") if (k == n_segs - 1 and end_terminal) else 1.0
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
    inliers = []
    inlier_err = 0.0
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
        inliers.append(i)
        inlier_err += pos_err
    return inliers, inlier_err


def find_intersection_gcps(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
) -> list[IntersectionGCP]:
    """Find GCPs from pairs of detected labels whose streets share a GeoJSON coordinate.

    A shared coordinate means the streets physically meet at that point. The pixel
    coordinate is the crossing of the two label direction lines (extrapolated from each
    label center). Pairs are skipped if the lines are nearly parallel or if the crossing
    falls implausibly far from both labels. The geo coordinate is the centroid of all
    shared endpoint coordinates.

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
            geo_pts = np.array(list(shared))
            geo = (float(geo_pts[:, 0].mean()), float(geo_pts[:, 1].mean()))
            # Try every combination of detected instances for the two streets.
            # RANSAC will select the geometrically consistent subset.
            for fa in feats_by_text[text_a]:
                for fb in feats_by_text[text_b]:
                    ca, cb = np.array(fa.center), np.array(fb.center)
                    pixel_dist = float(np.linalg.norm(ca - cb))
                    crossing = pixel_line_intersection(ca, fa.dir_pix, cb, fb.dir_pix)
                    if crossing is None:
                        continue
                    gcps.append(
                        IntersectionGCP(
                            label_a=text_a,
                            label_b=text_b,
                            pixel=crossing,
                            geo=geo,
                            pixel_dist=pixel_dist,
                            feat_a=fa,
                            feat_b=fb,
                        )
                    )

    return sorted(gcps, key=lambda g: g.pixel_dist)


def ransac_hybrid(
    gcps: list[IntersectionGCP],
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    cos_phi: float,
    pos_threshold: float = 0.001,
    dir_threshold: float = np.pi / 6,
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
        pairs = [(gcps[k].pixel, gcps[k].geo) for k in pair_idx]
        try:
            A = solve_similarity_2pts(pairs, cos_phi)
        except np.linalg.LinAlgError:
            continue
        inliers, err = label_inliers(
            features, block_index, A, pos_threshold, dir_threshold
        )

        # Each inlier contributes (pos_threshold - pos_err) to the score; an inlier at
        # exactly the threshold boundary contributes 0, so a marginal extra inlier cannot
        # blindly beat a tighter fit with one fewer inlier.
        score = float(len(inliers)) * pos_threshold - err
        a = gcps[pair_idx[0]]
        b = gcps[pair_idx[1]]

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_A = A
            best_pair = pair_idx
            print(
                f"  {score:.06f} {a.label_a} x {a.label_b} + {b.label_a} x {b.label_b}"
            )

    return best_A, best_inliers, best_pair


def save_debug_frame(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    affine: np.ndarray,
    inlier_feat_indices: list[int],
    cos_phi: float,
    frame_n: int,
    debug_dir: str,
    note: str = "",
) -> None:
    """Save a debug PNG showing label projections, snap positions, and street polylines.

    Green = inlier label, red = outlier. Circles = projected label center, × = snap on
    polyline, dashed line = position error, colored arrow = mapped label direction,
    blue arrow = polyline tangent at snap point.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]  # noqa: PLC0415

    inlier_set = set(inlier_feat_indices)
    A_linear = affine[:, :2]

    projected = [apply_affine(affine, *feat.center) for feat in features]
    if not projected:
        return

    lons = [p[0] for p in projected]
    lats = [p[1] for p in projected]
    pad_lon = (max(lons) - min(lons)) * 0.15 + 0.003
    pad_lat = (max(lats) - min(lats)) * 0.15 + 0.003
    lon_min = min(lons) - pad_lon
    lon_max = max(lons) + pad_lon
    lat_min = min(lats) - pad_lat
    lat_max = max(lats) + pad_lat

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw street polylines that intersect the bounding box.
    for blocks in block_index.values():
        for block in blocks:
            coords = block.coords
            if (
                coords[:, 0].max() < lon_min
                or coords[:, 0].min() > lon_max
                or coords[:, 1].max() < lat_min
                or coords[:, 1].min() > lat_max
            ):
                continue
            ax.plot(
                coords[:, 0], coords[:, 1], color="lightgray", linewidth=0.8, zorder=1
            )

    arrow_scale = (lon_max - lon_min) * 0.04

    for i, feat in enumerate(features):
        lon, lat = projected[i]
        color = "green" if i in inlier_set else "red"
        feat_blocks = block_index.get(feat.text)
        snap = project_to_polyline(lon, lat, feat_blocks) if feat_blocks else None

        if snap is not None:
            s_lon, s_lat, tangent_angle = snap
            ax.plot(
                s_lon,
                s_lat,
                "x",
                color=color,
                markersize=8,
                markeredgewidth=1.5,
                zorder=3,
            )
            ax.plot(
                [lon, s_lon],
                [lat, s_lat],
                "--",
                color=color,
                linewidth=0.8,
                alpha=0.6,
                zorder=2,
            )
            # Polyline tangent arrow in blue at the snap point.
            t_dx = np.cos(tangent_angle) * arrow_scale * 0.5
            t_dy = np.sin(tangent_angle) * arrow_scale * 0.5
            ax.annotate(
                "",
                xy=(s_lon + t_dx, s_lat + t_dy),
                xytext=(s_lon - t_dx, s_lat - t_dy),
                arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 1.0},
                zorder=4,
            )

        ax.plot(lon, lat, "o", color=color, markersize=6, zorder=4)
        # Mapped direction arrow (label direction transformed into geo space).
        d_pixel = np.array([np.cos(feat.dir_pix), np.sin(feat.dir_pix)])
        d_geo = A_linear @ d_pixel
        d_norm = d_geo / (float(np.linalg.norm(d_geo)) + 1e-10)
        ax.annotate(
            "",
            xy=(lon + d_norm[0] * arrow_scale, lat + d_norm[1] * arrow_scale),
            xytext=(lon - d_norm[0] * arrow_scale, lat - d_norm[1] * arrow_scale),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
            zorder=5,
        )
        ax.text(lon, lat, f"  {feat.raw_text}", fontsize=6, color=color, zorder=6)

    theta_deg = float(
        np.degrees(np.arctan2(float(A_linear[1, 0]), float(A_linear[0, 0])))
    )
    title = (
        f"Frame {frame_n:03d}  θ={theta_deg:.1f}°  "
        f"inliers={len(inlier_feat_indices)}/{len(features)}"
    )
    if note:
        title += f"  [{note}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # 1° lon ≠ 1° lat in metric space; correct aspect so the map looks undistorted.
    ax.set_aspect(1.0 / cos_phi)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    path = os.path.join(debug_dir, f"frame_{frame_n:04d}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
) -> float:
    """Print fit stats, write georef JSON, and return scale_deg_per_px."""
    from PIL import Image as PILImage

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

    with PILImage.open(image_path) as img:
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
    return affine_scale_deg_per_px(A)


def derive_paths(image_path: str) -> tuple[str, str]:
    """Derive labels and output paths from an image path.

    Strips everything after the first '.' in the filename to form the stem:
      p123.2048px.jpg → (p123.streets.json, p123.georef.json)
    """
    p = Path(image_path)
    stem = p.name.split(".")[0]
    base = p.parent / stem
    return str(base) + ".streets.json", str(base) + ".georef.json"


def process_image(
    image_path: str,
    labels_path: str,
    output_path: str,
    block_index: dict[str, list[Block]],
    cos_phi: float,
    centerlines_path: str,
    debug_dir: str | None,
    min_confidence: float = 0.5,
    min_long_side: float = 250.0,
    min_short_side: float = 60.0,
    min_aspect_ratio: float = 2.0,
    visualize_ocr: bool = False,
    fuzzy_threshold: float = 0.0,
) -> ProcessResult:
    """Fit a georeference model for one image and write GCPs to output_path.

    Returns a ProcessResult with success=True and scale_deg_per_px set on success.
    Returns success=False with deferred data when exactly 1 intersection GCP is found
    (needs median scale from other maps to complete). Returns success=False otherwise.
    """
    labels_raw = json.load(open(labels_path))
    if isinstance(labels_raw, dict):
        all_detections: list[dict] = labels_raw.get(
            "detections", labels_raw.get("accepted", [])
        )
    else:
        all_detections = labels_raw

    print(f"All detections: {len(all_detections)}")
    normalized_streets = set(block_index.keys())
    all_detections = deduplicate_detections(
        all_detections, normalized_streets=normalized_streets
    )
    print(f"Deduped detections: {len(all_detections)}")
    labels_data = []
    for det in all_detections:
        if not (
            det["confidence"] >= min_confidence
            and det.get("long_side", float("inf")) >= min_long_side
            and det.get("short_side", float("inf")) >= min_short_side
            and det.get("long_side", float("inf"))
            >= min_aspect_ratio * det.get("short_side", 1.0)
            and not is_number_only(det["text"])
            and normalize_street(det["text"]) not in DIRECTION_WORDS
        ):
            continue
        canonicals = canonical_street_matches(
            det["text"], normalized_streets, fuzzy_threshold
        )
        # Deduplicate by block-list identity: aliases like "HENRY" and "HENRY STREET"
        # point to the same blocks, so keep only the most specific (longest) name.
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

    if visualize_ocr:
        stem = Path(image_path).name.split(".")[0]
        detect_png = str(Path(image_path).parent / (stem + ".detect.png"))
        accepted_ids = {id(d) for d in labels_data}
        rejected = [d for d in all_detections if id(d) not in accepted_ids]
        visualize_detections(image_path, labels_data, rejected, detect_png)

    features = label_features(labels_data)
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
        print(
            "Only 1 intersection GCP; deferring for median-scale processing.",
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
        gcps,
        features,
        block_index,
        cos_phi,
    )
    if A is None:
        print("RANSAC failed: no valid affine found.", file=sys.stderr)
        return ProcessResult(success=False)
    print(
        f"RANSAC: {len(inlier_feat_indices)} / {len(features)} inlier labels",
        file=sys.stderr,
    )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        save_debug_frame(
            features,
            block_index,
            A,
            inlier_feat_indices,
            cos_phi,
            0,
            debug_dir,
            note="RANSAC",
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

    scale = _finalize_georef(
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
    return ProcessResult(success=True, scale_deg_per_px=scale)


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
    debug_dir: str | None,
) -> ProcessResult:
    """Georeference an image that was deferred due to having only 1 intersection GCP.

    Uses the provided scale (deg/px) and estimates rotation from label-angle histogram
    cross-correlation. Tries both candidate orientations (R and R+π, since the histogram
    gives an undirected angle) and picks the one that produces more inlier labels.
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

    # Try both directed orientations; the rotation estimate is undirected (mod π).
    # Two-level scoring:
    #   Primary: inlier count with extrapolation at terminal endpoints, so labels near
    #            true street ends are not unfairly excluded.
    #   Tiebreaker: negated total error without extrapolation, so when counts are equal
    #               the orientation that keeps labels on actual street segments wins.
    pos_threshold = 0.001
    best_A: np.ndarray | None = None
    best_inliers: list[int] = []
    best_score: tuple[int, float] = (-1, -float("inf"))
    for rot in (rotation, rotation + math.pi):
        A_cand = build_affine_from_scale_rotation_gcp(
            scale_deg_per_px, rot, gcp, cos_phi
        )
        inliers_cand, _ = label_inliers(
            features, block_index, A_cand, pos_threshold, extrapolate=True
        )
        # Compute no-extrapolation error for the inliers as the tiebreaker.
        err_no_ext = 0.0
        for i in inliers_cand:
            feat = features[i]
            blocks = block_index.get(feat.text, [])
            lon, lat = apply_affine(A_cand, *feat.center)
            snap = project_to_polyline(lon, lat, blocks, extrapolate=False)
            if snap is not None:
                err_no_ext += float(np.linalg.norm([lon - snap[0], lat - snap[1]]))
        score: tuple[int, float] = (len(inliers_cand), -err_no_ext)
        if score > best_score:
            best_score = score
            best_inliers = inliers_cand
            best_A = A_cand

    if best_A is None:
        print("No inliers found for deferred image.", file=sys.stderr)
        return ProcessResult(success=False)

    A = best_A
    inlier_feat_indices = best_inliers
    print(
        f"Deferred: {len(inlier_feat_indices)} / {len(features)} inlier labels",
        file=sys.stderr,
    )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        save_debug_frame(
            features,
            block_index,
            A,
            inlier_feat_indices,
            cos_phi,
            0,
            debug_dir,
            note="deferred-init",
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

    scale = _finalize_georef(
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
    return ProcessResult(success=True, scale_deg_per_px=scale)


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
        "--visualize-ocr",
        action="store_true",
        help="Save annotated detection image to <stem>.detect.png for each input",
    )
    parser.add_argument(
        "--debug-dir",
        metavar="DIR",
        help="Save debug PNGs to this directory (single-image mode only)",
    )
    parser.add_argument(
        "--fuzzy-match-threshold",
        type=float,
        default=0.0,
        metavar="T",
        help=(
            "Allow fuzzy street-name matching up to this normalized Levenshtein "
            "distance (0–1). 0 disables fuzzy matching (default). "
            "0.20 catches typical OCR errors like TCHUUPITOULAS→TCHOUPITOULAS."
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
    args = parser.parse_args()

    if args.debug_dir and len(args.images) > 1:
        parser.error("--debug-dir can only be used with a single image")

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
    deferred_list: list[dict] = []

    for image_path in args.images:
        if len(args.images) > 1:
            print(f"\n--- {image_path} ---", file=sys.stderr)
        labels_path, output_path = derive_paths(image_path)
        result = process_image(
            image_path=image_path,
            labels_path=labels_path,
            output_path=output_path,
            block_index=block_index,
            cos_phi=cos_phi,
            centerlines_path=args.centerlines,
            debug_dir=args.debug_dir,
            min_confidence=args.min_confidence,
            min_long_side=args.min_long_side,
            min_short_side=args.min_short_side,
            min_aspect_ratio=args.min_aspect_ratio,
            visualize_ocr=args.visualize_ocr,
            fuzzy_threshold=args.fuzzy_match_threshold,
        )
        if result.success:
            n_success += 1
            if result.scale_deg_per_px is not None:
                scales.append(deg_per_px_to_px_per_ft(result.scale_deg_per_px))
        elif result.deferred is not None:
            deferred_list.append(result.deferred)

    # Print median scale whenever multiple images were processed.
    if scales and len(args.images) > 1:
        print(
            f"\nMedian scale: {float(np.median(scales)):.4f} px/ft ({len(scales)} images)",
            file=sys.stderr,
        )

    # Process deferred (1-GCP) images using the median scale or --scale override.
    if deferred_list:
        if args.scale is not None:
            final_scale_px_per_ft: float | None = args.scale
        elif scales:
            final_scale_px_per_ft = float(np.median(scales))
        else:
            final_scale_px_per_ft = None

        if final_scale_px_per_ft is None:
            print(
                f"\n{len(deferred_list)} deferred image(s) skipped: no scale available. "
                "Use --scale PX_PER_FT to set manually.",
                file=sys.stderr,
            )
        else:
            scale_deg_per_px = px_per_ft_to_deg_per_px(final_scale_px_per_ft)
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
                    debug_dir=args.debug_dir,
                )
                if deferred_result.success:
                    n_success += 1

    if len(args.images) > 1:
        print(f"\n{n_success}/{len(args.images)} images georeferenced", file=sys.stderr)


if __name__ == "__main__":
    main()
