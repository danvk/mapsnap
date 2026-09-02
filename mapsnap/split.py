"""Split Sanborn map pages into individual panels.

Each scanned page may contain 2-4 map panels separated by thick black divider lines (often
non-rectangular: L-shapes, diagonals, staircases). For each input image this detects the
panel partition and, when the page actually splits, writes each panel to <base>__N.jpg in
the same directory (the panel cropped to its bounding box, with any out-of-panel area masked
white). Pages that are a single panel are left alone.

With --debug, the per-stage images are also written next to each input (binary mask, detected
segments, panel overlay, and front-end-specific stages).

Usage:
  mapsnap split IMAGE [IMAGE ...] [--debug]
"""

import argparse
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import shapely.affinity
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union
from skimage.morphology import medial_axis

from mapsnap.corner_boxes import BOX_MIN_THICK_PX, corner_boxes, panels_with_boxes
from mapsnap.keymap.log import append_keymap_log
from mapsnap.utils import image_stem, jpeg_dimensions


class PanelsJson(TypedDict):
    """Contents of a <stem>.panels.json sidecar.

    panels is a list of polygon rings (one per panel) in the pixel frame of the named
    image, recorded via width/height, ordered so panels[i-1] corresponds to
    <base>__i.jpg. The order is OIM's for two-panel sheets (the panel holding the
    sheet's bottom-left corner first) and reading order otherwise; see order_panels.
    """

    image: str
    width: int
    height: int
    panels: list[list[list[float]]]


# Front-end that turns the binary ink mask into candidate divider segments:
#   "skeleton"  - light close → medial-axis centerlines → thickness filter → Hough
#   "downscale" - erode → shrink → LSD (the original axis-aligned-friendly approach)
#   "union"     - both of the above, concatenated; merge_collinear then collapses the
#                 duplicate detections of each divider while filling each other's gaps
DETECTION_METHOD = "union"

# Tunable pipeline parameters.
BORDER_PX = 50  # pixels to remove from each edge before processing (dark scan border)
EDGE_SNAP_PX = (
    2  # when expanding cropped panels to the full frame, vertices this close to
)
# the cropped boundary are pushed out to the full-frame edge (re-opens the cropped border)
COLOR_SPREAD_MAX = 40  # max RGB channel spread (max−min) to count a pixel as black ink;
# colored pixels (brick, vegetation, water tints) are never part of a divider
EROSION_KERNEL_PX = (
    5  # thins map linework; the pre-LSD downscale suppresses the rest, so a light
    # kernel here preserves thinner black dividers (e.g. on color scans)
)
CLOSE_KERNEL_PX = (
    3  # light close to fuse a divider's broken/stepped core into one solid
)
# blob so the medial axis traces it continuously (helps washington / nola-p528). The close
# also fattens thin block/street lines fused with adjacent text, so dividers are separately
# thickness-filtered on the raw binary (see MIN_DIVIDER_THICK_PX)
MIN_DIVIDER_THICK_PX = 5.0  # drop candidate dividers whose median thickness on the raw
# binary is below this: real dividers are heavy (6-8px); dense-grid block/street lines that
# the close fattened are thin (3-4px) on the raw binary, so this rejects them
MAX_DIVIDER_CANDIDATES = (
    12  # safety gate: a page with more long candidates than this is a
)
# dense street/block grid (15-32), not a real split (≤8), so treat it as a single panel.
# Backstop for editions whose grid lines are heavy enough to pass the thickness filter
# (e.g. Brooklyn 1904-1908), where thickness alone can't tell a street from a divider
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
MIN_PANEL_FRAC = 0.01  # min panel area as fraction of total page area
SMALL_PANEL_BAND_FRAC = (
    0.05  # panels below this are "small" and must pass the shape guard
)
SMALL_PANEL_MAX_ASPECT = (
    4.0  # small-panel bbox aspect bound: real insets measure 1.0-2.3,
)
# over-fragmentation slivers (edge strips, margin cuts) 5-20 (measured on the OIM truth
# corpus, scripts/score_splits_oim.py)
SMALL_PANEL_MIN_DIM_FRAC = (
    0.10  # small-panel min bbox dimension as a fraction of the short
)
# page side: true insets measure >= 0.14, slivers <= 0.08
COVERAGE_MIN_FRAC = 0.90  # if the kept panels tile less than this, the partition is
# over-fragmented (e.g. a dense downtown page split along many building walls) — treat the
# page as a single, unsplit panel. A gutter-emptiness test was tried as a no-split signal
# but real dividers between dense adjacent panels have content hugging them, overlapping
# the spurious cases, so coverage is the more reliable guard.

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
    h, w = arr.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(
            f"image {w}×{h} is too small to crop a {border}px border from each edge"
        )
    if arr.ndim == 2:
        return arr[border:-border, border:-border]
    return arr[border:-border, border:-border, :]


def binarize(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Threshold to a black-ink mask: dark pixels become 255, paper and color become 0.

    Otsu on luminance selects dark pixels, then a chroma gate drops any that are colored.
    Dividers are always black, so colored map content (brick, vegetation, water tints) —
    which can be dark enough to pass an Otsu luminance threshold — is excluded. For grayscale
    scans the chroma gate is a no-op, matching plain Otsu.
    """
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    achromatic = (spread <= COLOR_SPREAD_MAX).astype(np.uint8) * 255
    return cv2.bitwise_and(dark, achromatic)


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
        if (-1 <= ix <= w + 1 and -1 <= iy <= h + 1) and (best_t is None or t < best_t):
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
    """Polygonize divider segments + the page boundary into the full set of faces.

    Returns every face of the partition (no area filter), so their union tiles the page.
    """
    lines = [
        LineString([(0, 0), (w, 0)]),
        LineString([(w, 0), (w, h)]),
        LineString([(w, h), (0, h)]),
        LineString([(0, h), (0, 0)]),
    ]
    for x0, y0, x1, y1 in segments:
        lines.append(LineString([(x0, y0), (x1, y1)]))
    return list(polygonize(unary_union(lines)))


def panel_compactness(geom) -> float:
    """Polsby-Popper compactness 4πA/P²: ~1 for a circle, low for ragged/disconnected shapes."""
    return 4 * np.pi * geom.area / (geom.length**2) if geom.length else 0.0


def face_divider_backed(
    face,
    thick_mask: np.ndarray,
    h: int,
    w: int,
    backed_min_frac: float = 0.6,
    search_px: int = 6,
    step_px: float = 12.0,
) -> float:
    """Fraction of a face's interior boundary that lies on thick divider ink.

    Samples points along the face's exterior ring, skipping stretches on the
    page border (those need no divider), and checks whether thick ink exists
    within ``search_px`` of each sample in the eroded thick mask. A real inset
    is fenced by heavy dividers on every interior side, so its backed fraction
    is high; a sliver from over-fragmentation is bounded by thin block/street
    linework the erosion removed. Returns the backed fraction (callers compare
    against ``backed_min_frac``; exposed for reporting).
    """
    ring = list(face.exterior.coords)
    total = 0
    backed = 0
    mh, mw = thick_mask.shape
    for (x0, y0), (x1, y1) in pairwise(ring):
        length = float(np.hypot(x1 - x0, y1 - y0))
        n = max(1, int(length / step_px))
        for k in range(n + 1):
            t = k / max(n, 1)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if on_boundary(x, y, h, w, tol=3.0):
                continue
            total += 1
            xi, yi = round(x), round(y)
            x_lo, x_hi = max(0, xi - search_px), min(mw, xi + search_px + 1)
            y_lo, y_hi = max(0, yi - search_px), min(mh, yi + search_px + 1)
            if x_lo < x_hi and y_lo < y_hi and thick_mask[y_lo:y_hi, x_lo:x_hi].any():
                backed += 1
    if total == 0:
        # Entire ring on the page border: a full-page face, trivially fine.
        return 1.0
    return backed / total


def assemble_panels(
    faces: list,
    h: int,
    w: int,
    min_panel_frac: float | None = None,
    small_face_policy: str = "glue",
    thick_mask: np.ndarray | None = None,
    backed_min_frac: float = 0.6,
) -> list:
    """Turn polygonize faces into panels that tile the whole page.

    Faces below the area floor (``min_panel_frac``, default MIN_PANEL_FRAC) are
    leftovers and are glued onto a neighbouring panel. Faces in the small band
    (floor..SMALL_PANEL_BAND_FRAC) additionally must be panel-shaped (aspect and
    minimum-dimension bounds) — this replaced PR #70's 0.05 hard floor, which
    also glued away genuine small insets (the panel-billed disaster class, #83).
    Measured on the OIM truth corpus: floor 0.01 + shape guard recovers 13/31
    small insets (from 2/31) with ~4 extra over-splits corpus-wide.

    ``small_face_policy="verified"`` keeps a sub-floor face as a real panel when
    its interior boundary is backed by thick divider ink
    (:func:`face_divider_backed` ≥ ``backed_min_frac``; requires ``thick_mask``).
    Measured WORSE than the shape guard on negatives (94.5% vs 98.6% accuracy):
    dense pages are full of heavy linework, so divider-backing fires spuriously.
    Kept for the harness record.

    If there is at most one real panel — or the real panels are too fragmented
    to trust (they cover less than COVERAGE_MIN_FRAC of the page) — the page is
    a single panel covering everything. Otherwise each remaining leftover face
    is glued onto the real panel whose union with it is most compact, so the
    panels cover 100% of the page with the least raggedness.
    """
    total = h * w
    min_area = (MIN_PANEL_FRAC if min_panel_frac is None else min_panel_frac) * total

    def small_shape_ok(face) -> bool:
        # A face in the small band must be panel-shaped: real insets are compact
        # rectangles; slivers from a stray long rule are thin strips. Measured
        # separation on the OIM corpus: true insets aspect <= 2.3 and min
        # dimension >= 14% of the short page side; slivers >= 5.0 and <= 8%.
        if face.area >= SMALL_PANEL_BAND_FRAC * total:
            return True
        x0, y0, x1, y1 = face.bounds
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) <= 0:
            return False
        return max(bw, bh) / min(bw, bh) <= SMALL_PANEL_MAX_ASPECT and min(
            bw, bh
        ) >= SMALL_PANEL_MIN_DIM_FRAC * min(h, w)

    big = [f for f in faces if f.area >= min_area and small_shape_ok(f)]
    if small_face_policy == "verified" and thick_mask is not None:
        promoted = [
            f
            for f in faces
            if f.area < min_area
            and f.area >= 0.002 * total
            and face_divider_backed(f, thick_mask, h, w) >= backed_min_frac
        ]
        big = big + promoted
    if len(big) <= 1 or sum(f.area for f in big) / total < COVERAGE_MIN_FRAC:
        return [box(0, 0, w, h)]

    # Glue each leftover face onto an *adjacent* panel — the one whose union with it is most
    # compact — so panels stay connected and grow to absorb the gap. Iterate so a leftover
    # only reachable through another leftover gets absorbed once its neighbour is.
    panels = list(big)
    kept = {id(f) for f in big}
    # Everything not kept — sub-floor faces AND shape-rejected small-band faces —
    # gets glued, so the panels still tile the page.
    leftovers = [f for f in faces if id(f) not in kept]
    progressed = True
    while leftovers and progressed:
        progressed = False
        deferred = []
        for face in leftovers:
            adjacent = [
                i for i, p in enumerate(panels) if p.intersection(face).length > 1e-9
            ]
            if not adjacent:
                deferred.append(face)
                continue
            best = max(
                adjacent,
                key=lambda i: panel_compactness(unary_union([panels[i], face])),
            )
            panels[best] = unary_union([panels[best], face])
            progressed = True
        leftovers = deferred
    # Any face never adjacent to a panel (shouldn't happen in a connected partition) is
    # attached to the nearest panel so the panels still tile the whole page.
    for face in leftovers:
        nearest = min(range(len(panels)), key=lambda i: panels[i].distance(face))
        panels[nearest] = unary_union([panels[nearest], face])
    return panels


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


def panel_basename(image_path: Path) -> str:
    """Base name for split outputs: the stem with all extensions stripped.

    e.g. p45.jpg → p45, champaign-p20.jpg → champaign-p20. Legacy multi-extension names
    like p195.scaled.jpg also collapse to p195 via image_stem.
    """
    return image_stem(str(image_path))


def panels_json_path(image_path: Path) -> Path:
    """Path to the panels.json sidecar (<stem>.panels.json) beside image_path."""
    return image_path.parent / f"{panel_basename(image_path)}.panels.json"


def read_panels_json(path: Path) -> PanelsJson:
    """Load a panels.json sidecar; see PanelsJson for the shape."""
    return json.loads(path.read_text())


def write_panels_json(
    image_path: Path, ordered_panels: list[Polygon], width: int, height: int
) -> Path:
    """Write ordered panel polygons to <stem>.panels.json beside image_path.

    Rings are in image_path's pixel frame (recorded via width/height) and ordered to
    match the __i.jpg numbering: ordered_panels[i-1] corresponds to <base>__i.jpg.
    Returns the written path.
    """
    rings = [
        [[round(x, 1), round(y, 1)] for x, y in panel.exterior.coords]
        for panel in ordered_panels
    ]
    data: PanelsJson = {
        "image": image_path.name,
        "width": width,
        "height": height,
        "panels": rings,
    }
    out_path = panels_json_path(image_path)
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def detect_downscale(
    binary: np.ndarray, rgb: np.ndarray, out_dir: Path, base: str
) -> np.ndarray:
    """Downscale front-end: erode → shrink → LSD. Saves <base>.mask.png and <base>.lsd.png.

    Returns raw candidate segments, shape (N, 4) with columns [x1, y1, x2, y2].
    """
    h, w = binary.shape
    mask = compute_thick_mask(binary)
    Image.fromarray(mask).save(out_dir / f"{base}.mask.png")
    lines, _ = run_lsd(mask)
    Image.fromarray(draw_lsd_segments(rgb, lines, h, w)).save(
        out_dir / f"{base}.lsd.png"
    )
    return lines


def detect_skeleton(
    binary: np.ndarray, rgb: np.ndarray, out_dir: Path, base: str
) -> np.ndarray:
    """Skeleton front-end: close → medial-axis thickness filter → Hough.

    Saves <base>.closed.png, <base>.skeleton.png (kept centerlines overlaid in red), and
    <base>.segments.png (raw Hough segments). Returns segments, shape (N, 4) [x1, y1, x2, y2].
    """
    h, w = binary.shape
    closed = morphological_close(binary)
    Image.fromarray(closed).save(out_dir / f"{base}.closed.png")
    skeleton_img = thick_skeleton(closed)
    Image.fromarray(draw_skeleton(rgb, skeleton_img)).save(
        out_dir / f"{base}.skeleton.png"
    )
    lines = run_hough(skeleton_img, h, w)
    Image.fromarray(draw_lsd_segments(rgb, lines, h, w)).save(
        out_dir / f"{base}.segments.png"
    )
    return lines


def detect_lines(binary: np.ndarray) -> np.ndarray:
    """Candidate divider segments in cropped coords per DETECTION_METHOD (no image output).

    The I/O-free counterpart of detect_downscale / detect_skeleton, used by compute_panels.
    """
    h, w = binary.shape
    if DETECTION_METHOD == "skeleton":
        return run_hough(thick_skeleton(morphological_close(binary)), h, w)
    if DETECTION_METHOD == "downscale":
        return run_lsd(compute_thick_mask(binary))[0]
    downscale_lines = run_lsd(compute_thick_mask(binary))[0]
    skeleton_lines = run_hough(thick_skeleton(morphological_close(binary)), h, w)
    return np.vstack([downscale_lines, skeleton_lines])


def segment_thickness(
    dist: np.ndarray, seg: tuple[float, float, float, float]
) -> float:
    """Median ink thickness (2× distance-to-background) sampled along a segment.

    `dist` is the distance transform of the raw binary ink mask. A small perpendicular
    window picks up the local ridge so a slightly off-center segment still measures the
    feature's true width.
    """
    x0, y0, x1, y1 = seg
    length = int(np.hypot(x1 - x0, y1 - y0))
    if length < 2:
        return 0.0
    widths = []
    for t in np.linspace(0.0, 1.0, length):
        x = round(x0 + t * (x1 - x0))
        y = round(y0 + t * (y1 - y0))
        window = dist[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5]
        if window.size:
            widths.append(2.0 * float(window.max()))
    return float(np.median(widths)) if widths else 0.0


def connected_dividers(
    lines: np.ndarray, h: int, w: int, binary: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Filter, merge, length-filter, drop thin (grid-line) candidates, re-admit connectors.

    The thickness filter measures each long candidate on the raw binary, so block/street
    lines that the morphological close fattened are still rejected as too thin.
    """
    merged = merge_collinear(filter_segments(lines, h, w), MERGE_GAP_FRAC * min(h, w))
    long_candidates = keep_long_segments(merged, h, w)
    if len(long_candidates) > MAX_DIVIDER_CANDIDATES:
        # Too many long candidates → dense street/block grid, not a split → single panel.
        return []
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    long_segs = [
        seg
        for seg in long_candidates
        if segment_thickness(dist, seg) >= MIN_DIVIDER_THICK_PX
    ]
    return keep_connectors(merged, long_segs, h, w)


def expand_to_full_frame(
    panels: list, cropped_h: int, cropped_w: int, border: int
) -> list:
    """Translate cropped panels into the full frame, pushing edge vertices to the full edges.

    Detection crops a border off each edge; the panels partition that cropped rectangle.
    Shifting them back by the border and snapping their cropped-boundary vertices out to the
    full-frame edges re-opens the border so the panels tile the whole scaled image (the same
    extension needed to cut split images, and the frame the panels.json truth uses).
    """
    full_h, full_w = cropped_h + 2 * border, cropped_w + 2 * border
    expanded = []
    for panel in panels:
        coords = []
        for x, y in panel.exterior.coords:
            x += border
            y += border
            if x <= border + EDGE_SNAP_PX:
                x = 0.0
            elif x >= full_w - border - EDGE_SNAP_PX:
                x = float(full_w)
            if y <= border + EDGE_SNAP_PX:
                y = 0.0
            elif y >= full_h - border - EDGE_SNAP_PX:
                y = float(full_h)
            coords.append((x, y))
        expanded.append(Polygon(coords))
    return expanded


def finalize_panels(
    connected: list[tuple[float, float, float, float]],
    cropped_h: int,
    cropped_w: int,
    border: int = BORDER_PX,
    min_panel_frac: float | None = None,
    small_face_policy: str = "glue",
    thick_mask: np.ndarray | None = None,
) -> tuple[list, list[tuple[float, float, float, float]]]:
    """Close the divider graph in the cropped frame, then expand panels to the full frame.

    Returns (full_frame_panels, cropped_bridged_segments). The graph closure runs in cropped
    coordinates (where detection happened); only the resulting panels are expanded to the
    full scaled-image frame so they line up with the panels.json ground truth.
    """
    extended = extend_long_segments(connected, cropped_h, cropped_w)
    snapped = snap_to_boundary(extended, cropped_h, cropped_w)
    bridged = bridge_junctions(snapped, cropped_h, cropped_w)
    faces = build_and_polygonize(bridged, cropped_h, cropped_w)
    panels = assemble_panels(
        faces,
        cropped_h,
        cropped_w,
        min_panel_frac=min_panel_frac,
        small_face_policy=small_face_policy,
        thick_mask=thick_mask,
    )
    return expand_to_full_frame(panels, cropped_h, cropped_w, border), bridged


def compute_panels(
    image_path: Path,
    border: int = BORDER_PX,
    min_panel_frac: float | None = None,
    small_face_policy: str = "glue",
) -> list:
    """Detect panels for one image as polygons in the full (uncropped) scaled-image frame.

    The I/O-free pipeline used by the scoring harness; process_image mirrors it while also
    writing per-stage debug images. ``min_panel_frac`` and ``small_face_policy`` are the
    #83 small-inset experiment knobs (see assemble_panels); production defaults are
    unchanged.
    """
    rgb = crop_border(load_rgb(image_path), border)
    gray = crop_border(load_gray(image_path), border)
    h, w = gray.shape
    binary = binarize(rgb, gray)
    thick = compute_thick_mask(binary) if small_face_policy == "verified" else None
    connected = connected_dividers(detect_lines(binary), h, w, binary)
    panels, _ = finalize_panels(
        connected,
        h,
        w,
        border,
        min_panel_frac=min_panel_frac,
        small_face_policy=small_face_policy,
        thick_mask=thick,
    )
    return panels


def order_panels(panels: list, height: float | None = None) -> list:
    """Panels in the order their __N numbering will follow.

    Two panels: the one holding the sheet's BOTTOM-LEFT corner comes first
    (#379). That is OIM's numbering on every two-panel truth sheet in the corpus
    -- 200 of 200, clean cuts, L-shaped cuts and corner insets alike -- because
    OIM keeps region boundaries in a y-up frame whose origin is that corner. A
    volunteer's division_number turns out to be the order the regions are drawn
    and is unpredictable past two, so three or more panels keep reading order
    (top to bottom, then left to right), which is as good as any. ``height`` is
    the image height the panels were detected in; without it two panels fall
    back to reading order.

    Ordering happens once, at the scale the panels were detected at, so a
    raw-resolution copy written from scaled polygons keeps identical numbering
    (the row-bucketing would bucket differently at 4x coordinates).
    """
    if len(panels) == 2 and height is not None:
        bottom_left = Point(0.0, float(height))
        return sorted(panels, key=lambda p: p.distance(bottom_left))
    return sorted(panels, key=lambda p: (round(p.bounds[1] / 50), p.bounds[0]))


# Everything derived from one panel image. A re-split that changes a panel's
# ring -- or its number -- makes all of these describe a different picture,
# and `mapsnap ocr --resume` keys only on a streets.json existing (plus the
# recognizer weights), so a stale one would be silently reused.
PANEL_SIDECAR_SUFFIXES = (
    "boxes.json",
    "streets.json",
    "txt",
    "contradiction.json",
)


def remove_panel_sidecars(image_path: Path, index: int) -> list[Path]:
    """Delete every derived sidecar of <base>__index beside image_path; return them.

    Covers the fixed suffixes above plus every georef variant
    (<base>__index.georef*.json). The panel image itself is handled by
    remove_split_outputs.
    """
    base = panel_basename(image_path)
    stem = f"{base}__{index}"
    removed: list[Path] = []
    candidates = [
        image_path.parent / f"{stem}.{suffix}" for suffix in PANEL_SIDECAR_SUFFIXES
    ]
    candidates += sorted(image_path.parent.glob(f"{stem}.georef*.json"))
    for path in candidates:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def rings_match(
    a: list[list[float]], b: list[list[float]], tolerance: float = 0.5
) -> bool:
    """Whether two serialized panel rings describe the same polygon.

    Compared as polygons, not vertex lists, so a ring that merely starts at a
    different vertex or runs the other way round still matches -- a spurious
    mismatch would cost a needless re-OCR, not correctness. ``tolerance`` is
    the mean boundary displacement, in pixels, that still counts as the same
    panel (the symmetric difference divided by the perimeter).
    """
    if len(a) < 3 or len(b) < 3:
        return False
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    perimeter = max(pa.length, pb.length)
    if perimeter <= 0:
        return pa.equals(pb)
    return pa.symmetric_difference(pb).area <= tolerance * perimeter


def invalidate_changed_panels(
    image_path: Path,
    old_rings: list[list[list[float]]],
    new_rings: list[list[list[float]]],
) -> list[int]:
    """Drop the sidecars of every panel index whose ring changed; return those indices.

    Compares the previous panels.json rings with the new ones index by index:
    an index whose ring is unchanged keeps its OCR reads and fits, one whose
    ring moved (or which no longer exists) loses them, so a later
    ``mapsnap ocr --resume`` re-reads exactly the panels that need it. Renumbering
    under #379 shows up here as two swapped rings.
    """
    changed: list[int] = []
    for i in range(1, max(len(old_rings), len(new_rings)) + 1):
        old = old_rings[i - 1] if i <= len(old_rings) else None
        new = new_rings[i - 1] if i <= len(new_rings) else None
        unchanged = old is not None and new is not None and rings_match(old, new)
        if not unchanged and remove_panel_sidecars(image_path, i):
            changed.append(i)
    return changed


def write_panels(image_path: Path, panels: list, base: str) -> list[Path]:
    """Write each panel to <base>__N.jpg next to image_path; return the written paths.

    A panel is cropped to its bounding box, with any pixels outside its (possibly
    non-rectangular) polygon set to white. ``panels`` must already be in reading
    order (see order_panels); numbering follows the given order. Also writes
    <base>.panels.json recording the panel polygons in image_path's pixel frame,
    in the same order as the numbering.
    """
    rgb = load_rgb(image_path)
    h, w = rgb.shape[:2]
    ordered = panels
    out_paths = []
    for i, panel in enumerate(ordered, start=1):
        mask = np.zeros((h, w), dtype=np.uint8)
        ring = np.array(
            [[round(x), round(y)] for x, y in panel.exterior.coords],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [ring], 255)
        masked = rgb.copy()
        masked[mask == 0] = 255
        minx, miny, maxx, maxy = panel.bounds
        x0, y0 = max(0, int(minx)), max(0, int(miny))
        x1, y1 = min(w, round(maxx)), min(h, round(maxy))
        out_path = image_path.parent / f"{base}__{i}.jpg"
        Image.fromarray(masked[y0:y1, x0:x1]).save(out_path, quality=92)
        out_paths.append(out_path)
    write_panels_json(image_path, ordered, w, h)
    return out_paths


# A panel edge within this fraction of the sheet's size counts as flush with it.
FLUSH_EDGE_TOLERANCE = 0.02
# A panel covering at least this fraction of the sheet is the sheet itself, not
# something cut away from it.
CUT_AWAY_MAX_AREA = 0.5


def panel_flush_edges(
    ring: list[list[float]] | list[tuple[float, float]], width: float, height: float
) -> int:
    """How many of the sheet's four edges a panel ring touches (0-4)."""
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    tol_x, tol_y = FLUSH_EDGE_TOLERANCE * width, FLUSH_EDGE_TOLERANCE * height
    return sum(
        (
            min(xs) <= tol_x,
            min(ys) <= tol_y,
            max(xs) >= width - tol_x,
            max(ys) >= height - tol_y,
        )
    )


def is_keymap_sheet(stem: str) -> bool:
    """Whether a page stem is a key-map candidate: page 0/1 families or a letter sheet.

    Mirrors mapsnap.keymap.identify.candidate_keys (page numbers 0 and 1 with
    any letter suffix -- p0, p0b, p0L, p1, p1N, p1a -- plus letter-only ids
    like pa/pb), kept here as a plain regex so the splitter does not import
    the key-map package.
    """
    if re.fullmatch(r"p[a-z]{1,2}", stem):
        return True
    match = re.fullmatch(r"p(\d+)[A-Za-z]{0,2}", stem)
    return match is not None and int(match.group(1)) in (0, 1)


def keymap_box_sheet(image_path: Path) -> bool:
    """Whether a key-map candidate sheet is one the corner-box detector may cut.

    The page-0 family and letter sheets are the key map in every volume censused
    (mapsnap.keymap.identify.detection_plan); a page-1 sheet is only when the
    volume has no page 0, and is otherwise an ordinary map page whose corner a
    diagonal street or a fold could carve into a box-shaped panel.
    """
    stem = panel_basename(image_path)
    if not is_keymap_sheet(stem):
        return False
    if re.fullmatch(r"p1[A-Za-z]{0,2}", stem):
        return not any(image_path.parent.glob("p0*.jpg"))
    return True


def box_candidates(
    lines: np.ndarray, h: int, w: int, binary: np.ndarray
) -> list[tuple[float, float, float, float]]:
    """Long segments at least BOX_MIN_THICK_PX thick, uncapped.

    The corner-box detector's input (mapsnap.corner_boxes): the divider
    pipeline's MAX_DIVIDER_CANDIDATES gate and its 5 px thickness floor are what
    hide a key map's boxed regions, so this is the same merge and length filter
    without them.
    """
    merged = merge_collinear(filter_segments(lines, h, w), MERGE_GAP_FRAC * min(h, w))
    dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return [
        seg
        for seg in keep_long_segments(merged, h, w)
        if segment_thickness(dist, seg) >= BOX_MIN_THICK_PX
    ]


def keymap_corner_box_panels(
    lines: np.ndarray, h: int, w: int, binary: np.ndarray
) -> tuple[list, list[str]]:
    """Panels cut by the corner-box detector, in the full frame, with log lines.

    No panels when no box is found, or when the boxes do not leave one clean
    remainder; the log lines say what was found either way.
    """
    candidates = box_candidates(lines, h, w, binary)
    boxes = corner_boxes(candidates, binary, h, w)
    log_lines = [
        (
            f"corner boxes: {len(boxes)} found among {len(candidates)} "
            "box-thick candidate segment(s)"
        )
    ]
    for candidate in boxes:
        share = candidate.area / (h * w)
        log_lines.append(
            f"  {'-'.join(candidate.corner)} box, {share:.1%} of the sheet: "
            f"{candidate.detail}"
        )
    panels = panels_with_boxes([candidate.polygon for candidate in boxes], h, w)
    if boxes and not panels:
        log_lines.append("  boxes do not leave one clean remainder; sheet stands whole")
    return expand_to_full_frame(panels, h, w, BORDER_PX), log_lines


def keymap_split_rejection(panels: list, width: float, height: float) -> str | None:
    """Why a key-map sheet's split must be refused, or None when it may stand.

    What a key-map sheet legitimately loses to a split is a boxed region in a
    corner or along an edge -- a KEY legend, a "graphic map of volumes" inset
    -- and every one of those is flush with at least two sheet edges. A cut-
    away flush with fewer is the splitter latching onto the key map's own
    linework: Chicago's 4% notch hanging off the top edge carried ten page
    numbers that the key-map panel then never read. Measured over every split
    key-map sheet in the corpus (9 panels): key-map panels are flush with four
    edges, legitimate cut-aways with two, the one bad cut with one. Regular
    map pages are not judged here: their panels are insets and composites of
    every shape.
    """
    for index, panel in enumerate(panels, start=1):
        x0, y0, x1, y1 = panel.bounds
        if (x1 - x0) * (y1 - y0) >= CUT_AWAY_MAX_AREA * width * height:
            continue  # the sheet itself, whatever was cut from it
        flush = panel_flush_edges(list(panel.exterior.coords), width, height)
        if flush < 2:
            return (
                f"panel {index} is flush with {flush} sheet edge(s), "
                "not a boxed corner or edge region"
            )
    return None


def panel_summary_lines(panels: list, width: float, height: float) -> list[str]:
    """One line per panel: its share of the sheet and how many edges it is flush with."""
    lines = []
    for index, panel in enumerate(panels, start=1):
        flush = panel_flush_edges(list(panel.exterior.coords), width, height)
        share = 100 * panel.area / (width * height)
        lines.append(
            f"  panel {index}: {share:.0f}% of sheet, flush with {flush} edge(s)"
        )
    return lines


def remove_split_outputs(image_path: Path) -> None:
    """Delete any existing <base>__N.jpg panels and <base>.panels.json for image_path.

    Called before re-splitting so outputs always reflect the current code and settings.
    """
    base = panel_basename(image_path)
    for stale in image_path.parent.glob(f"{base}__*.jpg"):
        stale.unlink()
    panels_path = panels_json_path(image_path)
    panels_path.unlink(missing_ok=True)


def process_image(image_path: Path, debug: bool = False) -> None:
    """Split one image into panel files, writing per-stage debug PNGs when debug is set."""
    base = panel_basename(image_path)
    out_dir = image_path.parent

    # Remember the previous panels so sidecars of panels that do not change
    # can survive the re-split (see invalidate_changed_panels).
    previous_path = panels_json_path(image_path)
    previous_rings: list[list[list[float]]] = (
        read_panels_json(previous_path)["panels"] if previous_path.exists() else []
    )

    # Regenerate from scratch: drop any stale panels from a previous run.
    remove_split_outputs(image_path)

    rgb = crop_border(load_rgb(image_path))
    gray = crop_border(load_gray(image_path))
    h, w = gray.shape
    binary = binarize(rgb, gray)

    if debug:
        Image.fromarray(binary).save(out_dir / f"{base}.binary.png")
        if DETECTION_METHOD == "skeleton":
            lines = detect_skeleton(binary, rgb, out_dir, base)
        elif DETECTION_METHOD == "downscale":
            lines = detect_downscale(binary, rgb, out_dir, base)
        else:  # union: concatenate both front-ends' raw segments
            lines = np.vstack(
                [
                    detect_downscale(binary, rgb, out_dir, base),
                    detect_skeleton(binary, rgb, out_dir, base),
                ]
            )
    else:
        lines = detect_lines(binary)

    connected = connected_dividers(lines, h, w, binary)
    panels, bridged = finalize_panels(connected, h, w)
    full_h, full_w = h + 2 * BORDER_PX, w + 2 * BORDER_PX

    if debug:
        Image.fromarray(draw_merged_segments(rgb, bridged)).save(
            out_dir / f"{base}.filtered.png"
        )
        Image.fromarray(draw_panels(load_rgb(image_path), panels)).save(
            out_dir / f"{base}.panels.png"
        )

    raw_path = image_path.parent / "raw" / image_path.name
    if raw_path.exists():
        remove_split_outputs(raw_path)

    # A single panel covering the whole page means no split was found.
    is_single_full = len(panels) == 1 and panels[0].area >= 0.99 * full_h * full_w
    # A key-map sheet only gives up boxed edge regions; anything else is a bad
    # cut and the sheet stands whole (#276).
    rejection = (
        keymap_split_rejection(panels, full_w, full_h)
        if is_keymap_sheet(base) and not is_single_full
        else None
    )
    # Where the divider pipeline leaves a key-map sheet whole, look for the
    # boxed corners its safety gates hide: a second key map, an index inset, a
    # legend (mapsnap.corner_boxes).
    box_lines: list[str] = []
    if (is_single_full or rejection) and keymap_box_sheet(image_path):
        box_panels, box_lines = keymap_corner_box_panels(lines, h, w, binary)
        if box_panels:
            panels, is_single_full, rejection = box_panels, False, None
            print(f"{image_path.name}: key-map sheet, corner box(es) cut away")
    if is_single_full:
        print(f"{image_path.name}: single panel — not split")
        for index in range(1, len(previous_rings) + 1):
            remove_panel_sidecars(image_path, index)
        if is_keymap_sheet(base):
            append_keymap_log(
                image_path, "split", ["single panel — not split", *box_lines]
            )
        return
    if rejection:
        # The stale panels were removed above; drop their sidecars too so
        # nothing downstream keeps reading them.
        print(f"{image_path.name}: key-map sheet, split rejected — {rejection}")
        removed = []
        for index in range(1, len(previous_rings) + 1):
            removed += remove_panel_sidecars(image_path, index)
        if raw_path.exists():
            remove_split_outputs(raw_path)
        append_keymap_log(
            image_path,
            "split",
            [
                f"split rejected — {rejection}",
                *panel_summary_lines(panels, full_w, full_h),
                *box_lines,
                f"removed {len(removed)} stale panel sidecar(s) from an earlier split",
            ],
        )
        return
    ordered = order_panels(panels, full_h)  # panels are in the uncropped frame
    out_paths = write_panels(image_path, ordered, base)
    print(
        f"{image_path.name}: {len(panels)} panels → {', '.join(p.name for p in out_paths)}"
    )
    if is_keymap_sheet(base):
        append_keymap_log(
            image_path,
            "split",
            [
                f"split accepted: {len(ordered)} panels → "
                + ", ".join(path.name for path in out_paths),
                *panel_summary_lines(ordered, full_w, full_h),
                *box_lines,
            ],
        )
    changed = invalidate_changed_panels(
        image_path,
        previous_rings,
        read_panels_json(panels_json_path(image_path))["panels"],
    )
    if changed:
        print(
            f"  panels {changed} changed since the last split; their reads and "
            "fits were removed (re-run ocr --resume)."
        )

    # Mirror the split onto the full-resolution copy, if one exists: the key-map
    # pipeline needs raw/<panel>.jpg when the key map shares its sheet with
    # another panel (a multi-volume index map, a legend). The splitter itself
    # can only run on the 25% images, so the raw panels are cut from the same
    # polygons scaled up to the raw frame, keeping identical numbering.
    if raw_path.exists():
        raw_width, raw_height = jpeg_dimensions(raw_path)
        scaled = [
            shapely.affinity.scale(
                panel, xfact=raw_width / w, yfact=raw_height / h, origin=(0, 0)
            )
            for panel in ordered
        ]
        raw_paths = write_panels(raw_path, scaled, base)
        print(
            f"  raw copy: {len(raw_paths)} panels → {', '.join(p.name for p in raw_paths)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Sanborn map pages into individual panel images."
    )
    parser.add_argument(
        "images", nargs="+", type=Path, metavar="IMAGE", help="Image files to split."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also write per-stage debug PNGs (binary, segments, panel overlay) "
        "next to each input image.",
    )
    args = parser.parse_args()
    for image_path in args.images:
        # Skip panels themselves (e.g. p2__1.jpg); only split whole pages.
        if "__" in panel_basename(image_path):
            continue
        process_image(image_path, debug=args.debug)


if __name__ == "__main__":
    main()
