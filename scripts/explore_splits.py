#!/usr/bin/env python3
"""Exploration script: validate LSD+graph approach for Sanborn split-page detection.

For each image in data/splits/, runs the pipeline and saves per-stage outputs. The
front-end is selected by DETECTION_METHOD.

Common:
  <name>.binary.png   - binarized ink mask (dark = ink, white = paper; 50px border removed)
  <name>.filtered.png - image with filtered + merged divider segments
  <name>.panels.png   - image with detected panels overlaid in color

"skeleton" front-end (medial axis):
  <name>.closed.png   - ink mask after a light morphological close
  <name>.skeleton.png - thickness-filtered medial-axis centerlines overlaid in red
  <name>.segments.png - image with raw Hough segments (green=passes filter, blue=does not)

"downscale" front-end (erode + LSD):
  <name>.mask.png     - thick-features mask (binary after erosion)
  <name>.lsd.png      - image with raw LSD segments (green=passes filter, blue=does not)

Run from the project root:
  uv run python scripts/explore_splits.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
from skimage.morphology import medial_axis

SPLITS_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/splits_output")

# Front-end that turns the binary ink mask into candidate divider segments:
#   "skeleton"  - light close → medial-axis centerlines → thickness filter → Hough
#   "downscale" - erode → shrink → LSD (the original axis-aligned-friendly approach)
#   "union"     - both of the above, concatenated; merge_collinear then collapses the
#                 duplicate detections of each divider while filling each other's gaps
DETECTION_METHOD = "union"

# Tunable pipeline parameters.
BORDER_PX = 50  # pixels to remove from each edge before processing (dark scan border)
COLOR_SPREAD_MAX = 40  # max RGB channel spread (max−min) to count a pixel as black ink;
# colored pixels (brick, vegetation, water tints) are never part of a divider
COLOR_PROXIMITY_PX = 0  # erase ink within this many px of a colored pixel (0 = off):
# building outlines hug their colored fills. Cleans building edges nicely where buildings
# are set back from dividers (e.g. nola-1896-p101), but where dividers abut colored
# buildings (washington-p244) it erases the dividers too — a net regression, so off by
# default. Set to ~5 to re-enable.
EROSION_KERNEL_PX = (
    5  # thins map linework; the pre-LSD downscale suppresses the rest, so a light
    # kernel here preserves thinner black dividers (e.g. on color scans)
)
CLOSE_KERNEL_PX = 3  # light close to fuse a divider's broken core into one solid blob;
# a heavier close would fuse neighbouring map features and erase the thick/thin distinction
SKELETON_MIN_THICK_PX = 6.0  # keep medial-axis centerlines of features at least this
# thick; well above the ~2px map-linework median but low enough to retain thin dividers
# (9px dropped too many; 6px is the sweet spot across the test set)
HOUGH_THRESHOLD = 40  # min Hough accumulator votes for a line
HOUGH_MIN_LEN_FRAC = (
    0.10  # min Hough segment length as fraction of shorter page dimension
)
HOUGH_MAX_GAP_PX = 30  # max gap (px) Hough will bridge along a line
LSD_DOWNSCALE = 0.2  # shrink mask before LSD so each thick divider collapses to a
# single thin line — LSD then detects the divider once, down its centerline, instead
# of detecting both of its edges as two parallel lines.
MIN_LENGTH_FRAC = 0.05  # min segment length as fraction of shorter page dimension
MERGE_ANGLE_DEG = 5.0  # max angle difference (°) to merge collinear segments
MERGE_PERP_PX = (
    20.0  # max perpendicular offset (px) to merge the two edges of a thick bar
)
MERGE_GAP_FRAC = 0.15  # max endpoint gap as fraction of shorter page dimension
DIVIDER_MIN_FRAC = (
    0.15  # after merging, only segments longer than this go into the graph
)
CONNECTOR_TOL_PX = (
    40.0  # max gap for treating endpoints as joined when re-admitting and
)
# welding short "step" segments that chain long divider runs (e.g. staircase dividers)
SNAP_PX = 50.0  # snap endpoints within this distance to the page boundary
EXTEND_FRAC = (
    0.20  # segments longer than this fraction of page get extended to boundary
)
EXTEND_MAX_PX = 200.0  # maximum distance to extend a segment toward boundary
JUNCTION_OVERSHOOT_FRAC = 0.06  # overshoot interior endpoints by this fraction of the
# shorter page dimension so near-miss corners/T-junctions become real crossings
JUNCTION_MAX_PX = 250.0  # max distance to extend an endpoint to reach another segment
NODE_OVERSHOOT_PX = (
    5.0  # push just past a reached segment so polygonize nodes the crossing
)
MIN_PANEL_FRAC = 0.05  # min panel area as fraction of total page area

PANEL_COLORS: list[tuple[int, int, int, int]] = [
    (255, 80, 80, 100),
    (80, 180, 80, 100),
    (80, 80, 255, 100),
    (255, 200, 80, 100),
    (200, 80, 255, 100),
    (80, 220, 220, 100),
]


def load_rgb(image_path: Path) -> np.ndarray:
    """Load image as uint8 RGB array."""
    return np.array(Image.open(image_path).convert("RGB"))


def load_gray(image_path: Path) -> np.ndarray:
    """Load image as uint8 grayscale array."""
    return np.array(Image.open(image_path).convert("L"))


def crop_border(arr: np.ndarray, border: int = BORDER_PX) -> np.ndarray:
    """Remove border pixels from all edges to exclude dark scan artifacts."""
    if arr.ndim == 2:
        return arr[border:-border, border:-border]
    return arr[border:-border, border:-border, :]


def binarize(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Threshold to a black-ink mask: dark pixels become 255, paper and color become 0.

    Otsu on luminance selects dark pixels, then a chroma gate drops any that are colored.
    Dividers are always black, so colored map content (brick, vegetation, water tints) —
    which can be dark enough to pass an Otsu luminance threshold — is excluded. Finally,
    ink near colored pixels is erased (see suppress_near_color). For grayscale scans both
    color steps are no-ops, matching plain Otsu.
    """
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    achromatic = (spread <= COLOR_SPREAD_MAX).astype(np.uint8) * 255
    ink = cv2.bitwise_and(dark, achromatic)
    return suppress_near_color(ink, spread)


def suppress_near_color(
    ink: np.ndarray, spread: np.ndarray, proximity_px: int = COLOR_PROXIMITY_PX
) -> np.ndarray:
    """Erase ink pixels within proximity_px of a colored pixel.

    Building outlines run right alongside their colored fills (brick red, frame yellow),
    while black dividers stay clear of color. Dilating the colored-pixel mask and clearing
    ink inside it removes building edges and other content hugging color, leaving dividers.
    `spread` is the per-pixel RGB channel spread (max−min) already computed by binarize.
    """
    if proximity_px <= 0:
        return ink
    colored = (spread > COLOR_SPREAD_MAX).astype(np.uint8)
    ksize = 2 * proximity_px + 1
    dilated = cv2.dilate(colored, np.ones((ksize, ksize), np.uint8))
    out = ink.copy()
    out[dilated > 0] = 0
    return out


def compute_thick_mask(
    binary: np.ndarray, kernel_px: int = EROSION_KERNEL_PX
) -> np.ndarray:
    """Erode a binary ink mask to retain only thick features."""
    kernel = np.ones((kernel_px, kernel_px), np.uint8)
    return cv2.erode(binary, kernel, iterations=1)


def run_lsd(
    image: np.ndarray, downscale: float = LSD_DOWNSCALE
) -> tuple[np.ndarray, np.ndarray]:
    """Run LSD on an image, shrinking it first so thick bars become single thin lines.

    Downscaling with area averaging collapses each thick divider's two edges into one
    antialiased ridge, so LSD detects the divider once (down its centerline) rather than
    tracing both edges as two parallel segments. Coordinates are rescaled back to the
    original resolution.

    Returns (lines, widths) with shapes (N, 4) and (N,).
    lines columns are [x1, y1, x2, y2].
    """
    small = cv2.resize(
        image, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA
    )
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    result = lsd.detect(small)
    if result[0] is None:
        return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float)
    return result[0][:, 0, :] / downscale, result[1][:, 0]


def morphological_close(
    binary: np.ndarray, kernel_px: int = CLOSE_KERNEL_PX
) -> np.ndarray:
    """Close small gaps so a divider's broken/speckled core becomes one solid blob.

    A light close lets the medial axis measure the divider's true thickness; a heavy close
    would fuse neighbouring map features and destroy the thick/thin distinction.
    """
    kernel = np.ones((kernel_px, kernel_px), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def thick_skeleton(
    closed: np.ndarray, min_thickness_px: float = SKELETON_MIN_THICK_PX
) -> np.ndarray:
    """Return medial-axis centerlines of features at least min_thickness_px wide.

    medial_axis returns the 1px skeleton plus, at each skeleton pixel, the distance to the
    nearest background pixel — the local half-thickness. Keeping pixels whose full thickness
    (2×distance) clears the threshold isolates thick divider centerlines from thin map
    linework. This is the thick/thin discrimination that erosion provides in the downscale
    front-end, but applied after skeletonization so it is orientation-independent (a steep
    diagonal collapses to a single centerline instead of two parallel edges).

    Returns a uint8 image (255 = kept skeleton pixel, 0 elsewhere).
    """
    # Seed the RNG: medial_axis breaks ties between equally-removable pixels randomly,
    # so without a fixed seed the skeleton (and thus the panel result) varies run to run.
    skeleton, distance = medial_axis(closed > 0, return_distance=True, rng=0)
    thick = (2.0 * distance >= min_thickness_px) & skeleton
    return (thick * 255).astype(np.uint8)


def run_hough(skeleton_img: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """Vectorize a 1px skeleton into line segments via the probabilistic Hough transform.

    Hough connects collinear skeleton pixels into segments and tolerates small gaps, which
    suits a thin centerline image. (LSD is an edge detector and shatters a skeleton into
    many fragments, so it is not used here.)

    Returns lines with shape (N, 4); columns are [x1, y1, x2, y2].
    """
    min_length = int(HOUGH_MIN_LEN_FRAC * min(image_h, image_w))
    lines = cv2.HoughLinesP(
        skeleton_img,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_length,
        maxLineGap=HOUGH_MAX_GAP_PX,
    )
    if lines is None:
        return np.zeros((0, 4), dtype=float)
    return lines[:, 0, :].astype(float)


def filter_segments(lines: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """Keep segments longer than MIN_LENGTH_FRAC of the shorter page dimension."""
    if len(lines) == 0:
        return lines
    min_length = MIN_LENGTH_FRAC * min(image_h, image_w)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[lengths >= min_length]


def keep_long_segments(
    segments: list[tuple[float, float, float, float]],
    image_h: int,
    image_w: int,
) -> list[tuple[float, float, float, float]]:
    """Keep only segments longer than DIVIDER_MIN_FRAC of the shorter page dimension.

    Applied after merging so that short lot lines and building edges are dropped
    while the longer merged divider segments are retained.
    """
    min_length = DIVIDER_MIN_FRAC * min(image_h, image_w)
    return [
        seg
        for seg in segments
        if (seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2 >= min_length**2
    ]


def weld_endpoints(
    segments: list[tuple[float, float, float, float]],
    tol: float = CONNECTOR_TOL_PX,
) -> list[tuple[float, float, float, float]]:
    """Snap near-coincident segment endpoints to a shared point so the graph is connected.

    Detected divider pieces often end a few pixels apart at a junction, which leaves the
    planar graph open (polygonize only nodes geometry that actually touches). Clustering
    endpoints within tol and moving each to its cluster centroid welds those near-misses
    into true shared nodes.
    """
    centroids: list[list[float]] = []  # [x, y, count] per cluster
    cluster_of: list[int] = []
    for seg in segments:
        for px, py in [(seg[0], seg[1]), (seg[2], seg[3])]:
            match = None
            for idx, (cx, cy, _) in enumerate(centroids):
                if (px - cx) ** 2 + (py - cy) ** 2 <= tol * tol:
                    match = idx
                    break
            if match is None:
                centroids.append([px, py, 1])
                cluster_of.append(len(centroids) - 1)
            else:
                cx, cy, count = centroids[match]
                centroids[match] = [
                    (cx * count + px) / (count + 1),
                    (cy * count + py) / (count + 1),
                    count + 1,
                ]
                cluster_of.append(match)

    welded = []
    for i in range(len(segments)):
        ax, ay, _ = centroids[cluster_of[2 * i]]
        bx, by, _ = centroids[cluster_of[2 * i + 1]]
        welded.append((ax, ay, bx, by))
    return welded


def keep_connectors(
    merged: list[tuple[float, float, float, float]],
    long_segments: list[tuple[float, float, float, float]],
    h: int,
    w: int,
    tol: float = CONNECTOR_TOL_PX,
) -> list[tuple[float, float, float, float]]:
    """Re-admit short segments that chain long divider runs to each other or the boundary.

    A staircase divider is a sequence of long horizontal (or vertical) runs joined by short
    perpendicular steps; the steps fall below the long-segment length filter, so dropping
    them leaves the runs disconnected and no panel boundary closes. This re-admits a short
    segment when it lies on a path between two "anchors" — a long segment or the page edge —
    while still discarding short segments that merely dangle in open space (lot lines,
    building walls). Surviving connectors and the long segments are then welded at their
    shared junctions so the network is topologically closed.
    """
    short = [seg for seg in merged if seg not in long_segments]
    if not short:
        return list(long_segments)

    long_geoms = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in long_segments]

    # Cluster short-segment endpoints into nodes and record each segment's two node ids.
    node_points: list[tuple[float, float]] = []

    def node_id(px: float, py: float) -> int:
        for idx, (qx, qy) in enumerate(node_points):
            if (px - qx) ** 2 + (py - qy) ** 2 <= tol * tol:
                return idx
        node_points.append((px, py))
        return len(node_points) - 1

    seg_nodes = [(node_id(s[0], s[1]), node_id(s[2], s[3])) for s in short]

    # A node is anchored if it touches the page boundary or any long segment.
    anchored = []
    for px, py in node_points:
        point = Point(px, py)
        on_long = any(geom.distance(point) <= tol for geom in long_geoms)
        anchored.append(on_boundary(px, py, h, w, tol) or on_long)

    # Iteratively prune connectors with a non-anchored leaf endpoint: such a segment
    # dead-ends in open space and cannot be part of an anchor-to-anchor path.
    alive = [True] * len(short)
    changed = True
    while changed:
        changed = False
        degree = [0] * len(node_points)
        for i, (a, b) in enumerate(seg_nodes):
            if alive[i]:
                degree[a] += 1
                degree[b] += 1
        for i, (a, b) in enumerate(seg_nodes):
            if alive[i] and (
                (degree[a] <= 1 and not anchored[a])
                or (degree[b] <= 1 and not anchored[b])
            ):
                alive[i] = False
                changed = True

    connectors = [short[i] for i in range(len(short)) if alive[i]]
    if not connectors:
        return list(long_segments)
    return weld_endpoints(list(long_segments) + connectors, tol)


def seg_angle_deg(line: np.ndarray) -> float:
    """Return segment angle in degrees in [0, 180)."""
    return float(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])) % 180)


def project(x: float, y: float, cos_t: float, sin_t: float) -> tuple[float, float]:
    """Project (x, y) onto direction (cos_t, sin_t) and its perpendicular.

    Returns (t, s) where t is along the line and s is perpendicular to it.
    """
    return x * cos_t + y * sin_t, -x * sin_t + y * cos_t


def merge_collinear(
    lines: np.ndarray,
    gap_tol_px: float,
) -> list[tuple[float, float, float, float]]:
    """Merge nearly-collinear segments separated by small gaps.

    Groups segments whose angle, perpendicular offset, and endpoint gap all fall
    within tolerance. Each group becomes one merged segment spanning its full extent.
    """
    if len(lines) == 0:
        return []

    angles = [seg_angle_deg(line) for line in lines]
    used = [False] * len(lines)
    merged: list[tuple[float, float, float, float]] = []

    for i in range(len(lines)):
        if used[i]:
            continue
        used[i] = True

        theta = np.radians(angles[i])
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        x0, y0, x1, y1 = (float(v) for v in lines[i])
        t0, s0 = project(x0, y0, cos_t, sin_t)
        t1, s1 = project(x1, y1, cos_t, sin_t)
        s_ref = s0
        t_vals = [t0, t1]
        s_vals = [s0, s1]

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            angle_diff = abs(angles[j] - angles[i]) % 180
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > MERGE_ANGLE_DEG:
                continue
            jx0, jy0, jx1, jy1 = (float(v) for v in lines[j])
            tj0, sj0 = project(jx0, jy0, cos_t, sin_t)
            tj1, sj1 = project(jx1, jy1, cos_t, sin_t)
            if abs(sj0 - s_ref) > MERGE_PERP_PX:
                continue
            t_group_min, t_group_max = min(t_vals), max(t_vals)
            t_j_min, t_j_max = min(tj0, tj1), max(tj0, tj1)
            gap = max(0.0, max(t_group_min, t_j_min) - min(t_group_max, t_j_max))
            if gap > gap_tol_px:
                continue
            used[j] = True
            t_vals.extend([tj0, tj1])
            s_vals.extend([sj0, sj1])

        t_min, t_max = min(t_vals), max(t_vals)
        s_avg = float(np.mean(s_vals))
        # Inverse of (t, s) → (x, y): x = t*cos - s*sin, y = t*sin + s*cos
        merged.append(
            (
                t_min * cos_t - s_avg * sin_t,
                t_min * sin_t + s_avg * cos_t,
                t_max * cos_t - s_avg * sin_t,
                t_max * sin_t + s_avg * cos_t,
            )
        )

    return merged


def snap_to_boundary(
    segments: list[tuple[float, float, float, float]],
    h: int,
    w: int,
) -> list[tuple[float, float, float, float]]:
    """Snap segment endpoints near the page boundary onto the boundary edge."""
    snapped = []
    for x0, y0, x1, y1 in segments:
        pts = []
        for px, py in [(x0, y0), (x1, y1)]:
            cx = (
                0.0
                if abs(px) < SNAP_PX
                else (float(w) if abs(px - w) < SNAP_PX else px)
            )
            cy = (
                0.0
                if abs(py) < SNAP_PX
                else (float(h) if abs(py - h) < SNAP_PX else py)
            )
            pts.append((cx, cy))
        snapped.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
    return snapped


def on_boundary(x: float, y: float, h: int, w: int, tol: float = 1.0) -> bool:
    """Return True if (x, y) lies on the page boundary (within tol pixels)."""
    return x <= tol or x >= w - tol or y <= tol or y >= h - tol


def clip_point(x: float, y: float, h: int, w: int) -> tuple[float, float]:
    """Clamp a point to the page rectangle [0, w] × [0, h]."""
    return min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))


def ray_to_segment(
    px: float,
    py: float,
    ux: float,
    uy: float,
    targets: list[LineString],
    max_dist: float,
) -> tuple[float, float] | None:
    """Return the nearest point where the ray from (px, py) in unit dir (ux, uy) meets a target.

    Considers only intersections within max_dist of the start point. Returns None if the ray
    reaches no target within that distance.
    """
    ray = LineString([(px, py), (px + ux * max_dist, py + uy * max_dist)])
    best_dist_sq = None
    best_point = None
    for target in targets:
        inter = ray.intersection(target)
        if inter.is_empty:
            continue
        for geom in getattr(inter, "geoms", [inter]):
            for cx, cy in geom.coords:
                dist_sq = (cx - px) ** 2 + (cy - py) ** 2
                if dist_sq > 1.0 and (best_dist_sq is None or dist_sq < best_dist_sq):
                    best_dist_sq = dist_sq
                    best_point = (cx, cy)
    return best_point


def extend_endpoint(
    px: float,
    py: float,
    ux: float,
    uy: float,
    others: list[LineString],
    overshoot: float,
    h: int,
    w: int,
) -> tuple[float, float]:
    """Extend one dangling endpoint outward along unit dir (ux, uy).

    If the ray reaches another segment within JUNCTION_MAX_PX, snap just past that segment so
    polygonize nodes the crossing. Otherwise fall back to a fixed overshoot, which closes
    near-miss perpendicular corners where no segment yet lies in the ray's path.
    """
    hit = ray_to_segment(px, py, ux, uy, others, JUNCTION_MAX_PX)
    if hit is not None:
        reach = ((hit[0] - px) ** 2 + (hit[1] - py) ** 2) ** 0.5 + NODE_OVERSHOOT_PX
        return clip_point(px + ux * reach, py + uy * reach, h, w)
    return clip_point(px + ux * overshoot, py + uy * overshoot, h, w)


def bridge_junctions(
    segments: list[tuple[float, float, float, float]],
    h: int,
    w: int,
) -> list[tuple[float, float, float, float]]:
    """Extend interior segment endpoints so near-miss junctions become real crossings.

    Dividers detected as separate segments frequently fall short of meeting at corners and
    T-junctions, leaving the page partition open. Each endpoint not already on the page
    boundary is extended along the segment's own direction — preferring to land on the nearest
    segment its ray crosses, and otherwise overshooting a fixed distance. The resulting tiny
    stub faces are removed by the area filter.
    """
    overshoot = JUNCTION_OVERSHOOT_FRAC * min(h, w)
    geoms = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in segments]
    result = []
    for i, (x0, y0, x1, y1) in enumerate(segments):
        others = [g for j, g in enumerate(geoms) if j != i]
        dx, dy = x1 - x0, y1 - y0
        length = (dx**2 + dy**2) ** 0.5
        if length < 1e-6:
            result.append((x0, y0, x1, y1))
            continue
        ux, uy = dx / length, dy / length
        if not on_boundary(x0, y0, h, w):
            x0, y0 = extend_endpoint(x0, y0, -ux, -uy, others, overshoot, h, w)
        if not on_boundary(x1, y1, h, w):
            x1, y1 = extend_endpoint(x1, y1, ux, uy, others, overshoot, h, w)
        result.append((x0, y0, x1, y1))
    return result


def ray_to_boundary(
    x: float, y: float, dx: float, dy: float, h: int, w: int
) -> tuple[float, float] | None:
    """Find the nearest page boundary intersection along the ray from (x,y) in dir (dx,dy).

    Returns the intersection point, or None if the ray doesn't hit any boundary within
    EXTEND_MAX_PX distance.
    """
    length = (dx**2 + dy**2) ** 0.5
    if length < 1e-6:
        return None
    best_t = None
    for boundary_val, coord, d in [(0, x, dx), (w, x, dx), (0, y, dy), (h, y, dy)]:
        if abs(d) < 1e-6:
            continue
        t = (boundary_val - coord) / d
        if t <= 0:
            continue
        ix, iy = x + t * dx, y + t * dy
        if -1 <= ix <= w + 1 and -1 <= iy <= h + 1:
            if best_t is None or t < best_t:
                best_t = t
    if best_t is None or best_t * length > EXTEND_MAX_PX:
        return None
    return x + best_t * dx, y + best_t * dy


def extend_long_segments(
    segments: list[tuple[float, float, float, float]],
    h: int,
    w: int,
) -> list[tuple[float, float, float, float]]:
    """Extend segments longer than EXTEND_FRAC of the page toward the page boundary.

    Extends each endpoint of a qualifying segment toward the nearest boundary
    intersection along the segment's direction, up to EXTEND_MAX_PX.
    """
    min_length = EXTEND_FRAC * min(h, w)
    extended = []
    for x0, y0, x1, y1 in segments:
        dx, dy = x1 - x0, y1 - y0
        length = (dx**2 + dy**2) ** 0.5
        if length >= min_length:
            hit0 = ray_to_boundary(x0, y0, -dx, -dy, h, w)
            hit1 = ray_to_boundary(x1, y1, dx, dy, h, w)
            if hit0 is not None:
                x0, y0 = hit0
            if hit1 is not None:
                x1, y1 = hit1
        extended.append((x0, y0, x1, y1))
    return extended


def build_and_polygonize(
    segments: list[tuple[float, float, float, float]],
    h: int,
    w: int,
) -> list:
    """Polygonize merged divider segments combined with the page boundary.

    Returns Shapely Polygon objects whose area exceeds MIN_PANEL_FRAC of the page.
    """
    lines = [
        LineString([(0, 0), (w, 0)]),
        LineString([(w, 0), (w, h)]),
        LineString([(w, h), (0, h)]),
        LineString([(0, h), (0, 0)]),
    ]
    for x0, y0, x1, y1 in segments:
        lines.append(LineString([(x0, y0), (x1, y1)]))
    polygons = list(polygonize(unary_union(lines)))
    min_area = MIN_PANEL_FRAC * h * w
    return [p for p in polygons if p.area >= min_area]


def draw_lsd_segments(
    rgb: np.ndarray,
    lines: np.ndarray,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Draw LSD segments on rgb: green if long enough to pass the length filter, blue otherwise."""
    out = rgb.copy()
    min_length = MIN_LENGTH_FRAC * min(image_h, image_w)
    for seg in lines:
        length = ((seg[2] - seg[0]) ** 2 + (seg[3] - seg[1]) ** 2) ** 0.5
        color = (0, 200, 0) if length >= min_length else (100, 100, 200)
        cv2.line(out, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])), color, 2)
    return out


def draw_skeleton(rgb: np.ndarray, skeleton_img: np.ndarray) -> np.ndarray:
    """Overlay a thickness-filtered skeleton on rgb in red for inspection."""
    out = rgb.copy()
    out[skeleton_img > 0] = (220, 30, 30)
    return out


def draw_merged_segments(
    rgb: np.ndarray,
    segments: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Draw merged divider segments on rgb in red."""
    out = rgb.copy()
    for x0, y0, x1, y1 in segments:
        cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), (220, 30, 30), 3)
    return out


def draw_panels(rgb: np.ndarray, panels: list) -> np.ndarray:
    """Draw panel polygons as translucent color overlays on rgb."""
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, polygon in enumerate(panels):
        color = PANEL_COLORS[i % len(PANEL_COLORS)]
        xy = [(int(x), int(y)) for x, y in polygon.exterior.coords]
        draw.polygon(xy, fill=color)
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def image_stem(image_path: Path) -> str:
    """Return image base name without extension."""
    return image_path.stem


def detect_downscale(binary: np.ndarray, rgb: np.ndarray, stem: str) -> np.ndarray:
    """Downscale front-end: erode → shrink → LSD. Saves .mask.png and .lsd.png.

    Returns raw candidate segments, shape (N, 4) with columns [x1, y1, x2, y2].
    """
    h, w = binary.shape
    mask = compute_thick_mask(binary)
    Image.fromarray(mask).save(OUTPUT_DIR / f"{stem}.mask.png")
    lines, _ = run_lsd(mask)
    print(f"  LSD: {len(lines)} raw segments")
    Image.fromarray(draw_lsd_segments(rgb, lines, h, w)).save(
        OUTPUT_DIR / f"{stem}.lsd.png"
    )
    return lines


def detect_skeleton(binary: np.ndarray, rgb: np.ndarray, stem: str) -> np.ndarray:
    """Skeleton front-end: close → medial-axis thickness filter → Hough.

    Saves .closed.png (closed ink mask), .skeleton.png (kept centerlines overlaid in red),
    and .segments.png (raw Hough segments). Returns segments, shape (N, 4) [x1, y1, x2, y2].
    """
    h, w = binary.shape
    closed = morphological_close(binary)
    Image.fromarray(closed).save(OUTPUT_DIR / f"{stem}.closed.png")
    skeleton_img = thick_skeleton(closed)
    Image.fromarray(draw_skeleton(rgb, skeleton_img)).save(
        OUTPUT_DIR / f"{stem}.skeleton.png"
    )
    lines = run_hough(skeleton_img, h, w)
    print(f"  Hough: {len(lines)} raw segments")
    Image.fromarray(draw_lsd_segments(rgb, lines, h, w)).save(
        OUTPUT_DIR / f"{stem}.segments.png"
    )
    return lines


def process_image(image_path: Path) -> None:
    """Run the full pipeline on one image and save per-stage output images."""
    stem = image_stem(image_path)
    print(f"\n{stem}")

    rgb = crop_border(load_rgb(image_path))
    gray = crop_border(load_gray(image_path))
    h, w = gray.shape

    binary = binarize(rgb, gray)
    Image.fromarray(binary).save(OUTPUT_DIR / f"{stem}.binary.png")

    if DETECTION_METHOD == "skeleton":
        lines = detect_skeleton(binary, rgb, stem)
    elif DETECTION_METHOD == "downscale":
        lines = detect_downscale(binary, rgb, stem)
    else:  # union: concatenate both front-ends' raw segments
        lines = np.vstack(
            [detect_downscale(binary, rgb, stem), detect_skeleton(binary, rgb, stem)]
        )
        print(f"  Union: {len(lines)} raw segments")

    filtered = filter_segments(lines, h, w)
    print(f"  Filtered (length): {len(filtered)} segments")

    gap_tol = MERGE_GAP_FRAC * min(h, w)
    merged = merge_collinear(filtered, gap_tol)
    long_segs = keep_long_segments(merged, h, w)
    connected = keep_connectors(merged, long_segs, h, w)
    extended = extend_long_segments(connected, h, w)
    snapped = snap_to_boundary(extended, h, w)
    bridged = bridge_junctions(snapped, h, w)
    print(f"  Merged+filtered → {len(bridged)} segments")
    Image.fromarray(draw_merged_segments(rgb, bridged)).save(
        OUTPUT_DIR / f"{stem}.filtered.png"
    )

    panels = build_and_polygonize(bridged, h, w)
    print(f"  Panels: {len(panels)}")
    for i, panel in enumerate(panels):
        print(f"    [{i + 1}] {panel.area / (h * w):.1%} of page")
    Image.fromarray(draw_panels(rgb, panels)).save(OUTPUT_DIR / f"{stem}.panels.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(SPLITS_DIR.glob("*.jpg")) + sorted(SPLITS_DIR.glob("*.png"))
    if not image_paths:
        print(f"No images found in {SPLITS_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(image_paths)} images → {OUTPUT_DIR}/")
    for path in image_paths:
        process_image(path)


if __name__ == "__main__":
    main()
