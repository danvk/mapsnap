"""Georeference a page by aligning its colored focus area to its key-map region.

Some pages cannot be georeferenced from street OCR: when the only readable streets are mutually
parallel, no two centerlines cross, so no intersection ground-control point can form (Brooklyn
1939 v1 p45, with only 12th St and 13th St). But such a page still has a precise schematic patch
on the volume's georeferenced key map — the outline of that page's colored "focus area".

This tool detects the focus area on the page (the city blocks filled with saturated
building-material ink — red/yellow/orange/blue — against white paper and water), aligns that shape
to the page's key-map region with a 2D similarity, and composes with the key map's own georeference
to place the page. It is a *coarse* placer whose value is a real georeference where street OCR
produces none; accuracy is bounded by the key map's own georeference error and its schematic
drawing, not by street matching.

The alignment is fit in a local metre frame: the page image (pixels, y-down) maps to world metres
(y-up) as a reflected similarity (the same orientation-reversing family the street georeferencer
uses). Scale comes from the focus-area/region area ratio, absolute rotation from the printed
adjacent-sheet numbers (a near-square focus area's own principal axis is too ambiguous), and the
fit is refined by ICP and accepted by intersection-over-union of the aligned outlines.

That region pose is then **refined against the page's own street data** as hard constraints, even
when the streets are mutually parallel and yield no intersection GCP. A 4-DOF pose (centre,
page-up bearing, log-scale) is solved by robust least squares over: street point-to-line + angle
factors (reusing ``georef_from_labels.prepare_label_features`` for the same label acceptance and
assembly the main georeferencer uses); printed neighbor-number "stamp" factors that must land in
the neighbor's key-map region; a key-map region containment + centroid anchor; and a scale prior
locked to the volume-median scale. Street *angles* are trusted tightly (they fix rotation, even
region rotation flips) while street *positions* are trusted loosely (label centres sit tens of
metres off their centerlines). On Brooklyn 1939 v1 this cut p45 from 134 ft (region only) to 69 ft.

The key map's own georeference is a **6-DOF affine**: ``georef_from_labels`` fits it (a robust
affine on the key map's own intersection GCPs, ``fit_keymap_affine``) and writes the corners into
the key-map ``georef.json`` for every key-map page, correcting the scale anisotropy and skew that
most affects pages near the sheet edge. The placer just reads those corners (no refit); bilinear of
a parallelogram's corners reproduces the affine. This most helps the anisotropic key maps
(Chicago median 182 -> ~90 ft).

    uv run python -m mapsnap.keymap.align_page_region data/brooklyn_ny_1939_vol_1 --page 45 --overlay
    uv run python -m mapsnap.keymap.align_page_region data/brooklyn_ny_1939_vol_1 --all-regions
"""

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point as ShapelyPoint, Polygon
from shapely.ops import unary_union


from mapsnap.compare_iiif_georef import (
    annotation_transform_type,
    extract_gcps,
    fit_transform,
    sample_grid,
    truth_page_number,
    truth_polygons_by_page,
)
from mapsnap.georef_from_labels import LabelFeature, prepare_label_features
from mapsnap.utils import FEET_PER_METER, haversine_m
from mapsnap.keymap.fit_keymap import (
    project,
    similarity_apply,
    similarity_fit,
    unproject,
)
from mapsnap.keymap.locate import (
    KeymapLocator,
    bilinear_pixel_to_world,
    discover_keymaps,
    page_number,
)
from mapsnap.keymap.page_regions import clean_cluster_mask
from mapsnap.streets import Block, build_block_index

Point = tuple[float, float]
Model = tuple[
    float, float, float, float
]  # reflected similarity (a, b, tx, ty), see fit_keymap


@dataclass
class FocusParams:
    """Tunables for detecting a page's colored focus area and aligning it to the key map."""

    target_long_side: int = 1600  # segmentation runs on a copy scaled to this long side
    chroma_threshold: float = 18.0  # CIELAB chroma above which a pixel is colored ink
    dark_fill_margin: float = (
        20.0  # cv2-L below the paper lightness to count a subtle dark fill
    )
    dark_fill_floor: float = (
        90.0  # cv2-L floor (excludes near-black ink) for a subtle dark fill
    )
    edge_trim_frac: float = (
        0.04  # ignore colored pixels within this fraction of each image edge
    )
    open_frac: float = (
        0.002  # morphological-open radius (fraction of long side) to drop specks
    )
    simplify_frac: float = 0.004  # Douglas-Peucker tolerance (fraction of long side)
    icp_iterations: int = 12  # ICP refinement rounds
    resample_points: int = 200  # points per outline for ICP / IoU
    min_iou: float = 0.5  # accept the fit only at or above this aligned-outline IoU


@dataclass
class PageResult:
    """Outcome of aligning one page to its key-map region."""

    page: int  # page number (splits share a number: p16n and p16w are both 16)
    status: str  # "ok", "rejected", or a skip reason
    stem: str = ""  # image stem, e.g. "p45" or "p16n"
    iou: float = 0.0
    scale_m_per_px: float = 0.0
    rotation_deg: float = 0.0
    neighbors_used: int = 0
    streets_used: int = 0  # accepted street labels that constrained the pose
    rmse_ft: float | None = None  # final (street-refined if any) grid-RMSE vs truth
    rmse_region_ft: float | None = (
        None  # region-shape-only RMSE, for the street-gain comparison
    )
    max_ft: float | None = None
    corners: list[list[float]] = field(default_factory=list)
    focus_hull: list[list[float]] = field(
        default_factory=list
    )  # detected focus-area convex hull, georeferenced to world (lon, lat)


# ---------------------------------------------------------------------------
# Page focus-area detection
# ---------------------------------------------------------------------------


def load_working_rgb(image_path: Path, target_long_side: int) -> np.ndarray:
    """Load an image as float RGB in [0, 1], downscaled so its long side is target_long_side."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"could not read image {image_path}")
    height, width = bgr.shape[:2]
    scale = min(1.0, target_long_side / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(
            bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def focus_footprint(rgb: np.ndarray, params: FocusParams) -> list[Point]:
    """Convex-hull polygon (working px) of the page's colored focus area.

    A focus pixel is either saturated building-material ink (CIELAB chroma above the threshold —
    Brooklyn/Hudson red/yellow/blue) OR a subtle *dark-brown* fill whose chroma is near the paper's
    but which sits a margin darker than it (Chicago's brown blocks): below the paper lightness (a
    high percentile of L) yet above the near-black ink floor, which separates the tinted fill from
    black grid lines and text. The focus blocks span the page but are broken apart by wide uncolored
    streets and yards, so a single connected component captures only a fraction of them; the mask is
    morphologically opened to drop scanner speckle, then the convex hull of every focus pixel is the
    smallest convex outline enclosing the whole focus area. The outermost ``edge_trim_frac`` of each
    edge is ignored first: the focus area
    almost never reaches the image edge, whereas the margins carry scan artifacts (book binding,
    the creased gutter, edge-of-page bleed, stamps) that can be colored and would inflate the hull.
    Returns a Douglas-Peucker-simplified hull, or an empty list if no colored area is found. The
    hull cuts genuine concavities (a waterfront inlet); the alignment's ICP step and its IoU
    acceptance gate absorb that, but a strongly non-convex footprint is a known limitation.
    """
    # cv2's 8-bit Lab (a, b centred at 128) rather than skimage's rgb2lab: the chroma magnitude
    # is the same, and skimage's Lab conversion segfaults when it shares process state with
    # scipy.optimize's MINPACK core (the street refine). cv2 is already this module's workhorse.
    bgr_u8 = cv2.cvtColor(
        np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2Lab).astype(np.float64)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    # Saturated building-material ink (Brooklyn red/yellow/blue) is caught by chroma. A subtle
    # *dark-brown* fill (Chicago) has near-paper chroma, so it is also caught as any pixel a margin
    # darker than the paper lightness (the L its bulk sits at, estimated as a high percentile) yet
    # above the near-black ink floor — separating the tinted block fill from black grid lines/text.
    lightness = lab[:, :, 0]
    paper_lightness = float(np.percentile(lightness, 80))
    dark_fill = (lightness < paper_lightness - params.dark_fill_margin) & (
        lightness > params.dark_fill_floor
    )
    mask = (chroma > params.chroma_threshold) | dark_fill
    height_px, width_px = mask.shape
    trim_y = round(params.edge_trim_frac * height_px)
    trim_x = round(params.edge_trim_frac * width_px)
    if trim_y > 0:
        mask[:trim_y, :] = False
        mask[height_px - trim_y :, :] = False
    if trim_x > 0:
        mask[:, :trim_x] = False
        mask[:, width_px - trim_x :] = False
    long_side = max(rgb.shape[:2])
    open_radius = max(1, round(params.open_frac * long_side))
    cleaned = clean_cluster_mask(mask, close_radius=0, open_radius=open_radius)
    rows, cols = np.nonzero(cleaned)
    if len(cols) < 3:
        return []
    hull = cv2.convexHull(np.column_stack([cols, rows]).astype(np.int32))
    tolerance = max(1.0, params.simplify_frac * long_side)
    approximated = cv2.approxPolyDP(hull, tolerance, closed=True)
    return [(float(x), float(y)) for [[x, y]] in approximated]


# ---------------------------------------------------------------------------
# Key-map region + adjacency
# ---------------------------------------------------------------------------


def load_adjacency(volume: Path) -> dict:
    """The volume's ``adjacency.json`` (printed adjacent-sheet graph), or an empty doc."""
    path = volume / "adjacency.json"
    return json.loads(path.read_text()) if path.exists() else {}


def reciprocated_neighbors(adjacency: dict, page: int) -> set[int]:
    """Page numbers sharing a mutual (reciprocated) adjacency edge with ``page``."""
    neighbors: set[int] = set()
    for edge in adjacency.get("adjacency", []):
        left, right = page_number(edge[0]), page_number(edge[1])
        if left == page and right is not None:
            neighbors.add(right)
        elif right == page and left is not None:
            neighbors.add(left)
    return neighbors


def image_neighbor_directions(
    adjacency: dict, stem: str
) -> dict[int, tuple[Point, float]]:
    """Per-claimed-neighbor unit direction (from the page-image center) and confidence.

    Reads the page's own printed adjacent-sheet detections (keyed by image ``stem``, e.g. ``p45``
    or ``p16n``); a neighbor's number is printed on the margin toward that neighbor, so its
    position gives a direction in the page-image frame (y-down). Keeps the highest-confidence claim
    per number. Includes every claim (spurious ones like a misread street name are rejected later
    as rotation outliers, not by reciprocity, which is too sparse to leave enough neighbors).
    """
    page_doc = adjacency.get("pages", {}).get(stem, {})
    best: dict[int, tuple[Point, float]] = {}
    for detection in page_doc.get("detections", []):
        number = detection.get("number")
        if number is None or not detection.get("claim"):
            continue
        dx = float(detection["x_frac"]) - 0.5
        dy = float(detection["y_frac"]) - 0.5
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        confidence = float(detection.get("confidence", 0.0))
        if number not in best or confidence > best[number][1]:
            best[number] = ((dx / norm, dy / norm), confidence)
    return best


def angle_wrap(degrees: float) -> float:
    """Fold an angle in degrees to the range (-180, 180]."""
    return (degrees + 180.0) % 360.0 - 180.0


def implied_reflected_rotation(image_direction: Point, world_direction: Point) -> float:
    """Rotation (deg) of the reflected similarity mapping a unit image dir to a unit world dir."""
    ix, iy = image_direction
    wx, wy = world_direction
    a = ix * wx - iy * wy
    b = iy * wx + ix * wy
    return math.degrees(math.atan2(b, a))


def robust_rotation(
    pairs: list[tuple[Point, Point, float]], tolerance_deg: float = 30.0
) -> tuple[Point, int]:
    """Reflected rotation from the largest rotation-consistent cluster of direction pairs.

    Each pair is (image_direction, world_direction, weight). A spurious adjacent-number claim
    (a misread that happens to be a real page elsewhere) implies a rotation inconsistent with the
    true neighbors, so it falls outside the dominant cluster. Returns the fitted unit (a, b) and
    the number of inliers; (1, 0) with 0 inliers if fewer than two pairs.
    """
    if len(pairs) < 2:
        return (1.0, 0.0), 0
    thetas = [implied_reflected_rotation(pair[0], pair[1]) for pair in pairs]
    best_inliers: list[int] = []
    best_weight = -1.0
    for center in thetas:
        inliers = [
            i
            for i, theta in enumerate(thetas)
            if abs(angle_wrap(theta - center)) <= tolerance_deg
        ]
        weight = sum(pairs[i][2] for i in inliers)
        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers) and weight > best_weight
        ):
            best_inliers, best_weight = inliers, weight
    return solve_reflected_rotation([pairs[i] for i in best_inliers]), len(best_inliers)


def ring_centroid(ring: list[Point]) -> Point:
    """Vertex-mean centroid of a ring."""
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def solve_reflected_rotation(
    pairs: list[tuple[Point, Point, float]],
) -> tuple[float, float]:
    """Unit (a, b) of the reflected similarity best mapping image directions to world directions.

    Each pair is (image_direction, world_direction, weight). Solves the same least-squares system
    as ``fit_keymap.similarity_fit`` but without translation, then normalizes to a pure rotation.
    Returns (1, 0) (image up -> world north) if the system is degenerate.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (image_x, image_y), (world_x, world_y), weight in pairs:
        scale = math.sqrt(max(weight, 0.0))
        rows.append([image_x * scale, image_y * scale])
        rhs.append(world_x * scale)
        rows.append([-image_y * scale, image_x * scale])
        rhs.append(world_y * scale)
    if len(rows) < 2:
        return (1.0, 0.0)
    solution, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    a, b = float(solution[0]), float(solution[1])
    norm = math.hypot(a, b)
    return (a / norm, b / norm) if norm > 0 else (1.0, 0.0)


def similarity_from_pose(
    rotation_unit: Point, scale: float, source_centroid: Point, target_centroid: Point
) -> Model:
    """A reflected-similarity model from a unit rotation, scale, and centroid correspondence."""
    a = scale * rotation_unit[0]
    b = scale * rotation_unit[1]
    cx, cy = source_centroid
    tx = target_centroid[0] - (a * cx + b * cy)
    ty = target_centroid[1] - (b * cx - a * cy)
    return (a, b, tx, ty)


def resample_ring(points: list[Point], count: int) -> list[Point]:
    """Evenly spaced points along a closed ring's perimeter."""
    loop = points + [points[0]]
    lengths = [math.dist(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]
    perimeter = sum(lengths)
    if perimeter == 0:
        return [points[0]] * count
    step = perimeter / count
    out: list[Point] = []
    target = 0.0
    walked = 0.0
    segment = 0
    for _ in range(count):
        while segment < len(lengths) and walked + lengths[segment] < target:
            walked += lengths[segment]
            segment += 1
        if segment >= len(lengths):
            out.append(points[0])
            continue
        remainder = (target - walked) / lengths[segment] if lengths[segment] else 0.0
        start, end = loop[segment], loop[segment + 1]
        out.append(
            (
                start[0] + (end[0] - start[0]) * remainder,
                start[1] + (end[1] - start[1]) * remainder,
            )
        )
        target += step
    return out


def icp_refine(
    source: list[Point], target: list[Point], model: Model, iterations: int
) -> Model:
    """Refine a similarity by iterated closest-point matching of source onto target."""
    target_array = np.array(target)
    for _ in range(iterations):
        matched: list[Point] = []
        for point in source:
            transformed = similarity_apply(model, point)
            distances = np.hypot(
                target_array[:, 0] - transformed[0], target_array[:, 1] - transformed[1]
            )
            nearest = target_array[int(distances.argmin())]
            matched.append((float(nearest[0]), float(nearest[1])))
        model = similarity_fit(source, matched)
    return model


def polygon_from_ring(ring: list[Point]) -> BaseGeometry:
    """A repaired shapely polygon from a ring (``buffer(0)`` fixes self-intersections)."""
    return Polygon(ring).buffer(0)


def region_polygon_metres(
    rings_world: list[list[Point]], origin: Point
) -> BaseGeometry:
    """Union of a page's key-map region rings, projected to the local metre frame at ``origin``."""
    polygons = [
        polygon_from_ring(
            [project(lon, lat, origin[0], origin[1]) for lon, lat in ring]
        )
        for ring in rings_world
        if len(ring) >= 3
    ]
    return unary_union(polygons)


def transformed_page_polygon(outline: list[Point], model: Model) -> BaseGeometry:
    """The page focus outline mapped through ``model`` into the metre frame, as a polygon."""
    return polygon_from_ring([similarity_apply(model, point) for point in outline])


def polygon_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Intersection-over-union of two shapely polygons (0 when either is empty)."""
    union = a.union(b).area
    return a.intersection(b).area / union if union > 0 else 0.0


def polygon_exterior(geometry: BaseGeometry) -> list[Point]:
    """Exterior ring (open, no repeated last point) of a polygon or a multipolygon's largest part."""
    if isinstance(geometry, Polygon):
        polygon = geometry
    else:
        parts = list(getattr(geometry, "geoms", []))
        if not parts:
            return []
        polygon = max(parts, key=lambda part: part.area)
    return [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]


def align_focus_to_region(
    outline: list[Point],
    region: BaseGeometry,
    rotation_unit: Point,
    params: FocusParams,
) -> tuple[Model, float]:
    """Fit the page outline to the key-map region; return the best model and its IoU.

    Initializes a reflected similarity from the adjacency rotation, the area-ratio scale, and the
    centroid correspondence, then refines by ICP and keeps whichever of the initial and refined
    models scores the higher outline IoU.
    """
    page_polygon = polygon_from_ring(outline)
    if page_polygon.area <= 0 or region.area <= 0:
        return (1.0, 0.0, 0.0, 0.0), 0.0
    scale = math.sqrt(region.area / page_polygon.area)
    source_centroid = (page_polygon.centroid.x, page_polygon.centroid.y)
    target_centroid = (region.centroid.x, region.centroid.y)
    initial = similarity_from_pose(
        rotation_unit, scale, source_centroid, target_centroid
    )

    source_ring = resample_ring(outline, params.resample_points)
    region_ring = resample_ring(polygon_exterior(region), params.resample_points)
    refined = icp_refine(source_ring, region_ring, initial, params.icp_iterations)

    candidates = [initial, refined]
    scored = [
        (polygon_iou(transformed_page_polygon(outline, model), region), model)
        for model in candidates
    ]
    best_iou, best_model = max(scored, key=lambda item: item[0])
    return best_model, best_iou


# ---------------------------------------------------------------------------
# Georeference output
# ---------------------------------------------------------------------------


def page_corners_world(
    size: tuple[int, int], model: Model, origin: Point
) -> list[list[float]]:
    """The page image's four corners (TL, TR, BR, BL) as [lon, lat], via ``model`` then unproject."""
    width, height = size
    corners_px = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    out: list[list[float]] = []
    for corner in corners_px:
        metre_x, metre_y = similarity_apply(model, corner)
        lon, lat = unproject(metre_x, metre_y, origin[0], origin[1])
        out.append([round(lon, 7), round(lat, 7)])
    return out


def model_scale_rotation(model: Model) -> tuple[float, float]:
    """Metric scale (m/px) and rotation (degrees) of a reflected-similarity model."""
    a, b, _, _ = model
    return math.hypot(a, b), math.degrees(math.atan2(b, a))


# ---------------------------------------------------------------------------
# Street-constrained pose refinement
#
# The region shape places the page coarsely; the page's own street labels — even a
# couple of mutually-parallel ones that yield no intersection GCP — pin rotation, the
# across-street position, and scale far more tightly. This refines a 4-DOF pose
# (east, north metres about the region centroid; page-up bearing; log pixels-per-metre)
# by robust least squares over street point-to-line + angle factors, a key-map region
# containment hinge + centroid spring, and a scale prior. The factor formulation and
# sigmas are salvaged from the earlier `factor_place` experiment; length sigmas are its
# feet values converted to metres so the normalized residuals are identical.
# ---------------------------------------------------------------------------

StreetPose = tuple[
    float, float, float, float
]  # (east_m, north_m, psi_deg, log_px_per_m)

# Street label centers sit tens of metres off their centerlines (the text box is not on the
# street line), so the point-to-line *position* is only a weak constraint — deliberately loose
# here. The label *angle*, by contrast, tracks the street bearing to ~1 degree and is what
# reliably fixes rotation (including region rotation flips), so it is kept tight.
SIGMA_LINE_M = 60.0  # street point-to-line distance (loose: label centers are offset)
SIGMA_ANGLE_DEG = (
    3.0  # street text-angle vs centerline bearing (tight: the reliable signal)
)
SIGMA_CONTAIN_M = 9.144  # 30 ft: region vertex outside the page frame
SIGMA_CENTROID_M = 152.4  # 500 ft: weak page-center vs region-centroid spring
SIGMA_LOG_SCALE = 0.05  # tight scale prior: the volume-median scale is accurate (~1%)
SIGMA_STAMP_M = (
    6.096  # 20 ft: printed neighbor-number stamp outside its neighbor's region
)
STAMP_SLACK_M = 4.572  # 15 ft of free play before a stamp is penalized

# One matchable street label: page-pixel center, long-axis angle, canonical name, and the
# street's centerline segments as (start, end) arrays in the local East/North metre frame.
StreetSegments = tuple[tuple[float, float], float, str, np.ndarray, np.ndarray]

# A stamp: the page-pixel position of a printed neighbor number, and that neighbor's key-map
# region as a polygon in the local East/North metre frame; the stamp must land inside it.
Stamp = tuple[tuple[float, float], BaseGeometry]


def page_axes(psi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(page-up, page-right) unit vectors in the East/North plane for a page-up bearing."""
    r = math.radians(psi_deg)
    up = np.array([math.sin(r), math.cos(r)])
    return up, np.array([up[1], -up[0]])


def angle_difference_mod180(a: float, b: float) -> float:
    """Signed difference of two undirected bearings, in (-90, 90] degrees."""
    return (a - b + 90) % 180 - 90


def point_to_segments(
    point: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[float, float]:
    """(min distance, bearing of the nearest segment) from a point to a segment soup."""
    delta = ends - starts
    length_sq = (delta * delta).sum(axis=1)
    t = np.clip(
        ((point - starts) * delta).sum(axis=1) / np.maximum(length_sq, 1e-9), 0, 1
    )
    projected = starts + t[:, None] * delta
    distances = np.linalg.norm(point - projected, axis=1)
    nearest = int(distances.argmin())
    bearing = math.degrees(math.atan2(delta[nearest, 0], delta[nearest, 1])) % 180
    return float(distances[nearest]), bearing


def pose_world_of(
    pose: StreetPose, pixel: tuple[float, float], size: tuple[int, int]
) -> np.ndarray:
    """A page pixel's East/North metre position under a 4-DOF pose."""
    east, north, psi, log_scale = pose
    up, right = page_axes(psi)
    scale = math.exp(log_scale)
    center_x, center_y = size[0] / 2, size[1] / 2
    return (
        np.array([east, north])
        + ((center_y - pixel[1]) / scale) * up
        + ((pixel[0] - center_x) / scale) * right
    )


def street_soup_metres(
    blocks: list[Block], origin: Point
) -> tuple[np.ndarray, np.ndarray]:
    """A street's centerline segments as (starts, ends) East/North metre arrays about ``origin``."""
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    for block in blocks:
        coordinates = [
            project(float(lon), float(lat), origin[0], origin[1])
            for lon, lat in block.coords
        ]
        for first, second in zip(coordinates, coordinates[1:]):
            starts.append([first[0], first[1]])
            ends.append([second[0], second[1]])
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(starts), np.array(ends)


def street_matches(
    volume: Path,
    stem: str,
    *,
    number: int,
    size: tuple[int, int],
    origin: Point,
    locator: KeymapLocator,
    centerlines: list[dict],
    filter_params: dict,
) -> list[StreetSegments]:
    """Accepted street labels for a page, each paired with its centerline segments (metres).

    Reuses ``georef_from_labels.prepare_label_features`` (the shared label acceptance/assembly)
    against a block index restricted to the page's key-map neighborhood, so far-away same-name
    streets can't match. Returns one entry per accepted label that has a matching nearby street.
    """
    streets_path = volume / f"{stem}.streets.json"
    if not streets_path.exists():
        return []
    near = locator.restricted_features(number, centerlines)
    if not near:
        return []
    block_index = build_block_index({"type": "FeatureCollection", "features": near})
    # streets.json coordinates live in the frame OCR ran on (the original
    # 25%-scale jpg), not the caller's working frame: acceptance gates (edge
    # margins, size thresholds) must run in the label frame, and accepted
    # centers are then rescaled into `size` for the pose residuals. Passing
    # the working size here silently discarded every label in the right and
    # bottom band of pages larger than the working long side.
    doc = json.loads(streets_path.read_text())
    label_size = (int(doc["width"]), int(doc["height"]))
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        features: list[LabelFeature] = prepare_label_features(
            str(streets_path), block_index, label_size, **filter_params
        )
    scale_x = size[0] / label_size[0]
    scale_y = size[1] / label_size[1]
    # A label read as several distinct streets (VAN BRUNT vs VAN DYKE at one box) is ambiguous:
    # one candidate is wrong and would drag the fit, so drop it from position constraints (this is
    # factor_place's unambiguous-only rule). Same-name repeats along a street are kept.
    names_per_center: dict[Point, set[str]] = {}
    for feature in features:
        names_per_center.setdefault(feature.center, set()).add(feature.text)
    matches: list[StreetSegments] = []
    for feature in features:
        blocks = block_index.get(feature.text)
        if not blocks or len(names_per_center[feature.center]) > 1:
            continue
        starts, ends = street_soup_metres(blocks, origin)
        if len(starts):
            center = (feature.center[0] * scale_x, feature.center[1] * scale_y)
            matches.append((center, feature.dir_pix, feature.text, starts, ends))
    return matches


def stamp_constraints(
    number: int,
    stem: str,
    *,
    size: tuple[int, int],
    origin: Point,
    locator: KeymapLocator,
    adjacency: dict,
    scale_px_per_m: float | None = None,
) -> list[Stamp]:
    """Printed neighbor-number stamps that must land in the neighbor's key-map region.

    A page prints each adjoining sheet's number in its margin toward that sheet. Requiring that
    number's pixel to map inside the neighbor's key-map region constrains the along-street position
    that mutually-parallel streets leave free (this is factor_place's stamp factor). Only mutual
    (reciprocated) neighbors that the key map also outlines are used, and a neighbor whose region
    sits more than two page-widths away is dropped as a misread edge that reciprocated by luck.
    """
    reciprocated = reciprocated_neighbors(adjacency, number)
    page_doc = adjacency.get("pages", {}).get(stem, {})
    width, height = size
    # This page's own region is centred near the metre-frame origin, so a neighbor's distance is
    # its region centroid's distance from the origin.
    max_neighbor_distance = (
        2.0 * width / scale_px_per_m if scale_px_per_m else float("inf")
    )
    stamps: list[Stamp] = []
    seen: set[int] = set()
    for detection in page_doc.get("detections", []):
        number = detection.get("number")
        rings = locator.regions.get(number) if number is not None else None
        if (
            number not in reciprocated
            or not detection.get("claim")
            or number in seen
            or not rings
        ):
            continue
        seen.add(number)
        pixel = (
            float(detection["x_frac"]) * width,
            float(detection["y_frac"]) * height,
        )
        region_metres = unary_union(
            [
                Polygon(
                    [project(lon, lat, origin[0], origin[1]) for lon, lat in ring]
                ).buffer(0)
                for ring in rings
                if len(ring) >= 3
            ]
        )
        if region_metres.is_empty:
            continue
        if (
            math.hypot(region_metres.centroid.x, region_metres.centroid.y)
            > max_neighbor_distance
        ):
            continue
        stamps.append((pixel, region_metres))
    return stamps


def pose_residuals(
    pose: StreetPose,
    matches: list[StreetSegments],
    stamps: list[Stamp],
    *,
    region_vertices: np.ndarray,
    size: tuple[int, int],
    prior_log_scale: float,
) -> np.ndarray:
    """Normalized residual vector for the 4-DOF street + stamp + region + scale-prior factors."""
    east, north, psi, log_scale = pose
    up, right = page_axes(psi)
    scale = math.exp(log_scale)
    half_height, half_width = size[1] / scale / 2, size[0] / scale / 2
    center = np.array([east, north])
    residuals: list[float] = []
    for center_px, dir_pix, _name, starts, ends in matches:
        world = pose_world_of(pose, center_px, size)
        distance, bearing = point_to_segments(world, starts, ends)
        residuals.append(distance / SIGMA_LINE_M)
        text_bearing = (psi + 90 + math.degrees(dir_pix)) % 180
        residuals.append(
            angle_difference_mod180(text_bearing, bearing) / SIGMA_ANGLE_DEG
        )
    for pixel, neighbor_region in stamps:
        point = ShapelyPoint(pose_world_of(pose, pixel, size))
        outside = (
            0.0
            if neighbor_region.contains(point)
            else neighbor_region.boundary.distance(point)
        )
        residuals.append(max(0.0, outside - STAMP_SLACK_M) / SIGMA_STAMP_M)
    for vertex in region_vertices:
        relative = vertex - center
        along = max(0.0, abs(float(relative @ up)) - half_height)
        across = max(0.0, abs(float(relative @ right)) - half_width)
        residuals.append(math.hypot(along, across) / SIGMA_CONTAIN_M)
    centroid = region_vertices.mean(axis=0)
    residuals.append((east - centroid[0]) / SIGMA_CENTROID_M)
    residuals.append((north - centroid[1]) / SIGMA_CENTROID_M)
    residuals.append((log_scale - prior_log_scale) / SIGMA_LOG_SCALE)
    return np.array(residuals)


def init_pose_from_model(model: Model, size: tuple[int, int]) -> StreetPose:
    """Derive the 4-DOF pose (metres/bearing/log-scale) from the region-shape similarity model."""
    center_x, center_y = size[0] / 2, size[1] / 2
    center = similarity_apply(model, (center_x, center_y))
    up_point = similarity_apply(model, (center_x, center_y - 1.0))
    up_vector = (up_point[0] - center[0], up_point[1] - center[1])
    psi = math.degrees(math.atan2(up_vector[0], up_vector[1]))
    px_per_m = 1.0 / (math.hypot(*up_vector) + 1e-12)
    return (center[0], center[1], psi, math.log(px_per_m))


def refine_pose_with_streets(
    model: Model,
    matches: list[StreetSegments],
    stamps: list[Stamp],
    *,
    region_vertices: np.ndarray,
    size: tuple[int, int],
    scale_px_per_m: float | None = None,
) -> StreetPose:
    """Robust-least-squares refine of the region pose against street + region + scale factors.

    The scale prior is the volume-median scale (``scale_px_per_m``) when available — accurate to
    ~1%, and far better than the region area-ratio scale, which is biased low because the detected
    focus area is a subset of the full page frame — otherwise the region's own scale. Both
    180-degree senses of the initial bearing are solved; the lower-cost fit wins (the region
    containment and centroid factors arbitrate). Falls back to the region pose if the solve fails.
    """
    initial = init_pose_from_model(model, size)
    prior_log_scale = (
        math.log(scale_px_per_m) if scale_px_per_m is not None else initial[3]
    )
    initial = (initial[0], initial[1], initial[2], prior_log_scale)
    # Bound the log-scale to prior +/- log(4); an unbounded scale can run to overflow, giving
    # non-finite residuals that segfault the MINPACK core (this mirrors factor_place's bounds).
    lower = np.array([-np.inf, -np.inf, -np.inf, prior_log_scale - math.log(4)])
    upper = np.array([np.inf, np.inf, np.inf, prior_log_scale + math.log(4)])
    best_pose = initial
    best_cost = float("inf")
    for delta_psi in (0.0, 180.0):
        start = np.array([initial[0], initial[1], initial[2] + delta_psi, initial[3]])
        try:
            result = least_squares(
                lambda pose: pose_residuals(
                    (pose[0], pose[1], pose[2], pose[3]),
                    matches,
                    stamps,
                    region_vertices=region_vertices,
                    size=size,
                    prior_log_scale=prior_log_scale,
                ),
                start,
                loss="soft_l1",
                f_scale=1.0,
                bounds=(lower, upper),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if result.cost < best_cost:
            best_cost = float(result.cost)
            best_pose = (
                float(result.x[0]),
                float(result.x[1]),
                float(result.x[2]),
                float(result.x[3]),
            )
    return best_pose


def pose_corners_world(
    pose: StreetPose, size: tuple[int, int], origin: Point
) -> list[list[float]]:
    """The page's four corners (TL, TR, BR, BL) as [lon, lat] under a 4-DOF pose."""
    width, height = size
    corners_px = [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]
    out: list[list[float]] = []
    for corner in corners_px:
        east, north = pose_world_of(pose, corner, size)
        lon, lat = unproject(float(east), float(north), origin[0], origin[1])
        out.append([round(lon, 7), round(lat, 7)])
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def find_truth_item(truth_path: Path, page: int) -> dict | None:
    """The truth IIIF annotation for a page number that has a fittable transform, or None.

    KNOWN BUG (evaluation-only; georef output unaffected): split pages share a
    page number and OIM truth then has several items for it, but this returns
    the FIRST >=3-GCP item — so both halves of a split sheet are scored against
    the same (possibly wrong) division, and for OIM-API-built truth the item's
    source dimensions are the full parent canvas while our corners cover only
    the split image, mixing frames. The >=3 gate also skips 2-GCP helmert
    truth items entirely, so those pages get no RMSE at all.
    """
    if not truth_path.exists():
        return None
    data = json.loads(truth_path.read_text())
    for item in data.get("items", []):
        if truth_page_number(item) == page and len(extract_gcps(item)) >= 3:
            return item
    return None


def grid_rmse_ft(
    corners: list[list[float]], page: int, truth_path: Path
) -> tuple[float, float] | None:
    """(RMSE, max) placement error in feet vs the truth georef, sampled on a 7x7 grid.

    Maps a grid of fractional page positions to world through both the truth annotation's fitted
    transform and the produced corners, and reports the haversine distance. Compares page
    *placement* on the full frame, so it is independent of how the focus-area/footprint boundary
    is defined. Returns None if the page has no fittable truth annotation.
    """
    item = find_truth_item(truth_path, page)
    if item is None:
        return None
    affine = fit_transform(extract_gcps(item), annotation_transform_type(item))
    truth_width = float(item["target"]["source"]["width"])
    truth_height = float(item["target"]["source"]["height"])
    corner_points = [(float(lon), float(lat)) for lon, lat in corners]
    errors: list[float] = []
    for px, py in sample_grid(truth_width, truth_height):
        lon_t, lat_t = affine @ np.array([px, py, 1.0])
        lon_g, lat_g = bilinear_pixel_to_world(
            corner_points, 1, 1, (px / truth_width, py / truth_height)
        )
        errors.append(haversine_m(lat_t, lon_t, lat_g, lon_g) * FEET_PER_METER)
    rmse = math.sqrt(sum(error**2 for error in errors) / len(errors))
    return rmse, max(errors)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def invert_similarity(model: Model, point: Point) -> Point:
    """Map a metre-frame point back to page pixels through a reflected similarity."""
    a, b, tx, ty = model
    determinant = -(a * a + b * b)
    dx = point[0] - tx
    dy = point[1] - ty
    x = (-a * dx - b * dy) / determinant
    y = (-b * dx + a * dy) / determinant
    return (x, y)


def write_overlay(
    rgb: np.ndarray,
    outline: list[Point],
    region: BaseGeometry,
    model: Model,
    output: Path,
) -> None:
    """Draw the detected focus outline (green) and the aligned key-map region (red) on the page."""
    image = Image.fromarray((rgb * 255).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if outline:
        draw.polygon(outline, outline=(0, 200, 0, 255), width=3)
    region_pixels = [
        invert_similarity(model, point) for point in polygon_exterior(region)
    ]
    draw.polygon(region_pixels, outline=(220, 0, 0, 255), width=3)
    Image.alpha_composite(image, overlay).convert("RGB").save(output)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def align_page(
    volume: Path, stem: str, params: FocusParams, options: "RunOptions"
) -> PageResult:
    """Align one page's focus area to its key-map region and write a ``-region`` georef sidecar.

    ``stem`` is the image stem (``p45`` or a split sheet like ``p16n``); the page *number* it maps
    to (splits share one) keys the key-map region and OIM truth, while the stem keys the image
    file, the page's adjacency detections, and its street/output sidecars.
    """
    number = page_number(stem)
    if number is None:
        return PageResult(0, "bad-stem", stem=stem)
    image_path = volume / f"{stem}.jpg"
    if not image_path.exists():
        return PageResult(number, "no-image", stem=stem)
    keymaps = discover_keymaps([str(image_path)])
    if not keymaps:
        return PageResult(number, "no-keymap", stem=stem)
    # The key map's georef.json already carries a 6-DOF affine fit (georef_from_labels writes it for
    # key-map pages), so every projection the locator makes — target region, neighbor directions,
    # stamps — inherits the anisotropy/skew correction with no refit here.
    locator = KeymapLocator.from_keymaps(keymaps)
    region_rings = locator.regions.get(number, [])
    if not region_rings:
        return PageResult(number, "no-region", stem=stem)

    origin = ring_centroid(region_rings[0])
    region = region_polygon_metres(region_rings, origin)

    adjacency = load_adjacency(volume)
    reciprocated = reciprocated_neighbors(adjacency, number)
    directions = image_neighbor_directions(adjacency, stem)
    pairs: list[tuple[Point, Point, float]] = []
    for neighbor, (image_direction, confidence) in directions.items():
        rings = locator.regions.get(neighbor)
        if not rings:
            continue
        neighbor_centroid = project(*ring_centroid(rings[0]), origin[0], origin[1])
        norm = math.hypot(*neighbor_centroid)
        if norm < 1e-6:
            continue
        weight = confidence * (2.0 if neighbor in reciprocated else 1.0)
        pairs.append(
            (
                image_direction,
                (neighbor_centroid[0] / norm, neighbor_centroid[1] / norm),
                weight,
            )
        )
    if len(pairs) < 2:
        return PageResult(
            number, "too-few-neighbors", stem=stem, neighbors_used=len(pairs)
        )
    rotation_unit, inliers = robust_rotation(pairs)
    if inliers < 2:
        return PageResult(
            number, "rotation-ambiguous", stem=stem, neighbors_used=inliers
        )

    rgb = load_working_rgb(image_path, params.target_long_side)
    outline = focus_footprint(rgb, params)
    if len(outline) < 3:
        return PageResult(number, "no-focus-area", stem=stem, neighbors_used=inliers)

    model, iou = align_focus_to_region(outline, region, rotation_unit, params)
    scale, rotation = model_scale_rotation(model)
    height, width = rgb.shape[:2]
    size = (width, height)
    region_corners = page_corners_world(size, model, origin)

    result = PageResult(
        page=number,
        stem=stem,
        status="ok" if iou >= params.min_iou else "rejected",
        iou=round(iou, 3),
        scale_m_per_px=round(scale, 4),
        rotation_deg=round(rotation, 1),
        neighbors_used=inliers,
    )
    region_rmse = grid_rmse_ft(region_corners, number, options.truth_path)
    if region_rmse is not None:
        result.rmse_region_ft = round(region_rmse[0], 1)

    # Bake in the page's own street labels (even mutually-parallel ones that yield no intersection
    # GCP) and its printed neighbor stamps as hard constraints, refining the region pose.
    corners = region_corners
    if options.centerlines is not None:
        matches = street_matches(
            volume,
            stem,
            number=number,
            size=size,
            origin=origin,
            locator=locator,
            centerlines=options.centerlines,
            filter_params=options.filter_params,
        )
        stamps = stamp_constraints(
            number,
            stem,
            size=size,
            origin=origin,
            locator=locator,
            adjacency=adjacency,
            scale_px_per_m=options.scale_px_per_m,
        )
        result.streets_used = len(matches)
        if matches or stamps:
            region_vertices = np.array(polygon_exterior(region))
            pose = refine_pose_with_streets(
                model,
                matches,
                stamps,
                region_vertices=region_vertices,
                size=size,
                scale_px_per_m=options.scale_px_per_m,
            )
            # Backstop against a bad street/stamp constraint dominating: the region init is a
            # decent anchor, so a refine that slides the page center more than half a page width
            # from its region-derived start is untrustworthy — keep the region pose instead.
            init = init_pose_from_model(model, size)
            scale_px_per_m = options.scale_px_per_m or math.exp(init[3])
            page_width_m = size[0] / scale_px_per_m
            moved = math.hypot(pose[0] - init[0], pose[1] - init[1])
            if moved <= 0.5 * page_width_m:
                corners = pose_corners_world(pose, size, origin)

    result.corners = corners
    # The detected focus-area hull, georeferenced through the page's own corners (bilinear on the
    # corner quad reproduces the pose), so the debugger can show where the focus area landed. The
    # ring is closed (first vertex repeated) so it renders as a closed polygon.
    corner_points = [(float(lon), float(lat)) for lon, lat in corners]
    hull = [
        list(bilinear_pixel_to_world(corner_points, width, height, vertex))
        for vertex in outline
    ]
    result.focus_hull = hull + [hull[0]] if hull else []
    rmse = grid_rmse_ft(corners, number, options.truth_path)
    if rmse is not None:
        result.rmse_ft = round(rmse[0], 1)
        result.max_ft = round(rmse[1], 1)

    keymap_entry = locator.page_keymap(number)
    truth = options.truth_regions.get(number)
    write_georef(volume, stem, size, result, keymap=keymap_entry, truth=truth)
    if options.overlay:
        write_overlay(rgb, outline, region, model, volume / f"{stem}.region-align.png")
    return result


def write_georef(
    volume: Path,
    stem: str,
    size: tuple[int, int],
    result: PageResult,
    *,
    keymap: dict | None = None,
    truth: list[list[list[float]]] | None = None,
) -> None:
    """Write the ``<stem>.georef-region.json`` variant sidecar.

    Mirrors the standard ``<stem>.georef.json`` shape so the debugger and comparison tools can
    read it: ``width``/``height``/``corners`` plus the placement diagnostics, the ``keymap`` entry
    (its centroid ``lat``/``lon``, ``centers``, and world-space ``regions`` polygon), the OIM
    ``truth`` footprint(s) when available, and the detected ``focus_hull`` (the page's colored
    focus-area convex hull, georeferenced to world (lon, lat)).
    """
    document: dict = {
        "width": size[0],
        "height": size[1],
        "corners": result.corners,
        "method": (
            "region-shape+streets" if result.streets_used else "region-shape-align"
        ),
        "page": result.page,
        "iou": result.iou,
        "accepted": result.status == "ok",
        "scale_m_per_px": result.scale_m_per_px,
        "rotation_deg": result.rotation_deg,
        "neighbors_used": result.neighbors_used,
        "streets_used": result.streets_used,
    }
    if keymap is not None:
        document["keymap"] = keymap
    if truth is not None:
        document["truth"] = truth
    if result.focus_hull:
        document["focus_hull"] = result.focus_hull
    (volume / f"{stem}.georef-region.json").write_text(json.dumps(document, indent=2))


@dataclass
class RunOptions:
    """Non-tuning options for a run (paths, truth, and street-constraint inputs)."""

    truth_path: Path
    overlay: bool = False
    centerlines: list[dict] | None = (
        None  # OSM features for street constraints (None disables)
    )
    filter_params: dict = field(
        default_factory=dict
    )  # label-acceptance gates for prepare_label_features
    scale_px_per_m: float | None = (
        None  # volume-median scale prior for the street refine
    )
    truth_regions: dict[int, list[list[list[float]]]] = field(
        default_factory=dict
    )  # page number -> OIM truth footprint rings, for the georef sidecar's truth field


def volume_median_scale_px_per_m(volume: Path) -> float | None:
    """Median pixels-per-metre across the volume's street-georeferenced pages, or None.

    Read from each ``p<N>.georef.json``'s corner quad (page width/height in pixels over the ground
    metres its edges span). This is the accurate per-volume scale the region area-ratio cannot
    provide, and it anchors the street-constrained scale.
    """
    scales: list[float] = []
    for georef_path in sorted(volume.glob("p*.georef.json")):
        document = json.loads(georef_path.read_text())
        corners = document.get("corners")
        width, height = document.get("width"), document.get("height")
        if not corners or not width or not height or len(corners) != 4:
            continue
        top_left, top_right, _bottom_right, bottom_left = corners
        width_m = haversine_m(top_left[1], top_left[0], top_right[1], top_right[0])
        height_m = haversine_m(top_left[1], top_left[0], bottom_left[1], bottom_left[0])
        if width_m > 0 and height_m > 0:
            scales.append((width / width_m + height / height_m) / 2)
    if not scales:
        return None
    scales.sort()
    return scales[len(scales) // 2]


def volume_filter_params(volume: Path) -> dict:
    """The volume's street-label acceptance gates, read from any page's ``p<N>.georef.json``.

    Mirrors what the volume's own ``mapsnap georef`` run used (its recorded
    ``inputs.parameters``), so street acceptance here matches the main pipeline. Empty dict
    (``prepare_label_features`` defaults) if no georef sidecar records them.
    """
    keys = (
        "min_confidence",
        "min_long_side",
        "min_short_side",
        "min_aspect_ratio",
        "high_confidence_size_fraction",
    )
    for georef_path in sorted(volume.glob("p*.georef.json")):
        parameters = (
            json.loads(georef_path.read_text()).get("inputs", {}).get("parameters", {})
        )
        selected = {key: parameters[key] for key in keys if key in parameters}
        if selected:
            return selected
    return {}


def discover_pages(volume: Path, only_unfit: bool) -> list[str]:
    """Image stems with a key-map region (optionally only those lacking a successful georef).

    Returns stems (``p45``, or split sheets like ``p16n``) rather than page numbers, since a page's
    image file, adjacency detections, and output sidecars are keyed by stem while its key-map region
    is keyed by number. With ``only_unfit`` these are the pages this method is *for* (GCP-RANSAC
    georef failed, no ``<stem>.georef.json``); without it, every page the key map outlines.
    """
    image_paths = sorted(volume.glob("p*.jpg"))
    if not image_paths:
        return []
    keymap_stems = {
        km.name.split(".")[0] for km in discover_keymaps([str(image_paths[0])])
    }
    locator = KeymapLocator.from_keymaps(discover_keymaps([str(image_paths[0])]))
    stems: list[str] = []
    for image_path in image_paths:
        stem = image_path.stem
        number = page_number(stem)
        if number is None or stem in keymap_stems or number not in locator.regions:
            continue
        if only_unfit and (volume / f"{stem}.georef.json").exists():
            continue
        stems.append(stem)
    return sorted(set(stems))


def format_row(result: PageResult) -> str:
    """One aligned table row for a page result."""
    rmse = "-" if result.rmse_ft is None else f"{result.rmse_ft:>7.1f}"
    region_rmse = (
        "-" if result.rmse_region_ft is None else f"{result.rmse_region_ft:>7.1f}"
    )
    label = result.stem or f"p{result.page}"
    return (
        f"{label:<6s} {result.status:<18s} iou={result.iou:<5.2f} "
        f"rot={result.rotation_deg:>6.1f}  nbr={result.neighbors_used:<2d} "
        f"streets={result.streets_used:<2d} rmse_ft={rmse}  (region {region_rmse})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "volume", type=Path, help="Volume directory (e.g. data/brooklyn_ny_1939_vol_1)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--page", help="Single page to align: a number (45), or a split stem (p16n)."
    )
    group.add_argument(
        "--all-unfit",
        action="store_true",
        help="Align every page that has a key-map region but no successful georef.",
    )
    group.add_argument(
        "--all-regions",
        action="store_true",
        help="Align every page that has a key-map region (to measure accuracy vs OIM truth).",
    )
    parser.add_argument(
        "--chroma-threshold", type=float, default=FocusParams.chroma_threshold
    )
    parser.add_argument("--open-frac", type=float, default=FocusParams.open_frac)
    parser.add_argument(
        "--target-long-side", type=int, default=FocusParams.target_long_side
    )
    parser.add_argument("--min-iou", type=float, default=FocusParams.min_iou)
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Truth IIIF file for RMSE (default: <volume>/main.iiif.json).",
    )
    parser.add_argument(
        "--overlay", action="store_true", help="Write a page overlay PNG."
    )
    parser.add_argument(
        "--centerlines",
        type=Path,
        default=None,
        help="OSM centerlines GeoJSON for street constraints (default: <volume>/centerlines.geojson).",
    )
    parser.add_argument(
        "--no-streets",
        action="store_true",
        help="Disable street constraints (region-shape alignment only).",
    )
    args = parser.parse_args()

    params = FocusParams(
        target_long_side=args.target_long_side,
        chroma_threshold=args.chroma_threshold,
        open_frac=args.open_frac,
        min_iou=args.min_iou,
    )
    centerlines_path = args.centerlines or (args.volume / "centerlines.geojson")
    centerlines: list[dict] | None = None
    if not args.no_streets and centerlines_path.exists():
        centerlines = json.loads(centerlines_path.read_text()).get("features", [])
    truth_path = args.truth or (args.volume / "main.iiif.json")
    options = RunOptions(
        truth_path=truth_path,
        overlay=args.overlay,
        centerlines=centerlines,
        filter_params=volume_filter_params(args.volume),
        scale_px_per_m=volume_median_scale_px_per_m(args.volume),
        truth_regions=truth_polygons_by_page(truth_path) if truth_path.exists() else {},
    )

    if args.page is not None:
        stem = args.page if args.page.startswith("p") else f"p{args.page}"
        stems = [stem]
    else:
        stems = discover_pages(args.volume, only_unfit=args.all_unfit)
    if not stems:
        sys.exit("No pages to align.")
    results = [align_page(args.volume, stem, params, options) for stem in stems]
    for result in results:
        print(format_row(result))

    accepted = sum(1 for r in results if r.status == "ok")
    with_streets = [r for r in results if r.status == "ok" and r.streets_used]
    scored = [r.rmse_ft for r in results if r.rmse_ft is not None and r.status == "ok"]
    if scored:
        scored.sort()
        print(
            f"\naligned {len(results)}  accepted {accepted}  "
            f"street-constrained {len(with_streets)}  with-truth {len(scored)}  "
            f"median RMSE {scored[len(scored) // 2]:.0f} ft  "
            f"p25 {scored[len(scored) // 4]:.0f}  p75 {scored[3 * len(scored) // 4]:.0f}  "
            f"best {scored[0]:.0f}  worst {scored[-1]:.0f}"
        )
    paired = [
        (r.rmse_region_ft, r.rmse_ft)
        for r in with_streets
        if r.rmse_region_ft is not None and r.rmse_ft is not None
    ]
    if paired:
        region_median = sorted(p[0] for p in paired)[len(paired) // 2]
        street_median = sorted(p[1] for p in paired)[len(paired) // 2]
        print(
            f"street-constrained pages: median RMSE region-only {region_median:.0f} ft "
            f"-> +streets {street_median:.0f} ft ({len(paired)} pages)"
        )


if __name__ == "__main__":
    main()
