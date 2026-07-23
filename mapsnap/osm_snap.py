"""Geometry-first georeferencing against OSM (truth-free library).

Places a page by matching its road-UNet P(road) map directly against OSM
centerlines rasterized in a local metre frame — no street-name OCR required,
so the fit is robust to renamed streets and unreadable labels. The key map
supplies the coarse location; explicit rotation/scale prior ladders seed the
search; OCR street names, when available, *boost* a candidate's score but
never gate it.

Frames follow the edge-join convention (:class:`mapsnap.edge_join.FrameSpec`):
equirectangular metres, north-up rows, so page pixels map into the raster with
rotation only (page y-down and raster row-down cancel — no reflection). All
rotation math below uses that identity: for directed vectors,
``theta = page_angle - raster_angle`` (both y-down atan2 angles, and theta is
the cv2.getRotationMatrix2D angle that match_at_rotation consumes).

The harness (osm_snap_experiment.py) loads volumes, composes PageContext
objects, and evaluates against truth; nothing in this module reads truth data.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mapsnap.edge_join import (
    CHAMFER_CLAMP_M,
    FrameSpec,
    JoinCandidate,
    MatchParams,
    match_at_rotation,
    refine_and_rank,
    rotation_candidates,
    skeleton_points,
)
from mapsnap.georef_from_labels import LabelFeature, project_to_polyline
from mapsnap.streets import Block, street_base_name
from mapsnap.utils import haversine_m

OSM_RES_M = 2.0  # raster resolution
OSM_WIDTH_M = 12.0  # stroked corridor width for the OSM "P(road)" analog
REFINE_SHIFT_MAX_M = 30.0  # chamfer refinement may not slide farther than this
CONTAINMENT_MIN = 0.35  # of the footprint inside the (buffered) keymap region
CONTAINMENT_BUFFER_M = 60.0
RADIUS_SLACK_M = 60.0  # post-refinement slack on the center-distance gate
THETA_DEDUPE_DEG = 3.0
MAX_PRIOR_THETAS = 8
MERGE_SEPARATION_M = 60.0  # candidates closer than this are the same lock
CALIBRATED_RADIUS_MARGIN_M = 100.0
CALIBRATED_RADIUS_MIN_M = 150.0

# select_score weights (hand-tuned; every term is logged per candidate so a
# re-ranking experiment can refit them on cached candidates).
W_NAME = 1.0
W_CONTAIN = 0.3
W_PRIOR = 0.1

# The recipe validated by the issue-#128 exploration: generous overlap window
# (the page may sit entirely inside the OSM frame, unlike an edge join) and a
# deeper top-K since ranking happens downstream.
OSM_MATCH_PARAMS = MatchParams(
    min_overlap_m2=30_000.0,
    max_overlap_frac=1.0,
    top_k=8,
    mask_min_area=500,
)


@dataclass
class RotationPrior:
    """One rung of the rotation-prior ladder, in cv2/match_at_rotation degrees."""

    theta_deg: float
    sigma_deg: float
    # "label-pair-exact" | "label-osm-mod180" | "ransac-neighbor"
    # | "adjacency-keymap" | "mask-mod90"
    source: str


@dataclass
class ScalePrior:
    """One candidate page scale (metres per page pixel)."""

    m_per_px: float
    sigma_log: float
    source: str  # "volume-median" | "keymap-region" | "family-rung"


@dataclass
class NameAlignment:
    """OCR street-name agreement with a candidate pose (a boost, never a gate)."""

    score: float
    n_labels: int
    n_hits: int
    hits: list[dict] = field(default_factory=list)


@dataclass
class SnapCandidate:
    """One candidate placement of a page against OSM, with all ranking features."""

    world_affine: np.ndarray  # 2x3 page px -> (lon, lat)
    center: tuple[float, float]  # (lon, lat) of the posed page center
    theta_deg: float
    theta_source: str
    scale_m_per_px: float
    scale_source: str
    scale_adjust: float
    ncc: float
    ncc_fine: float
    chamfer_mean_m: float
    inlier_frac: float
    n_points: int
    jtj_eig_ratio: float
    overlap_frac: float
    refine_shift_m: float
    center_dist_m: float
    verification: float  # edge_join verification_score (-inf if implausible)
    region_containment: float | None = None
    prior_theta_residual_sigma: float | None = None
    name: NameAlignment | None = None
    plausible: bool = True
    gate_reasons: list[str] = field(default_factory=list)

    def select_score(self) -> float:
        """The ranking score: matcher verification plus soft evidence bonuses."""
        if not math.isfinite(self.verification):
            return -math.inf
        score = self.verification
        if self.name is not None:
            score += W_NAME * self.name.score
        if self.region_containment is not None:
            score += W_CONTAIN * self.region_containment
        if self.prior_theta_residual_sigma is not None:
            score += W_PRIOR * max(0.0, 1.0 - self.prior_theta_residual_sigma)
        return score


@dataclass
class PageContext:
    """Everything snap_page needs about one target page (all truth-free)."""

    stem: str
    number: int | None
    width: int
    height: int
    prob: np.ndarray  # road-UNet P(road) at page resolution
    search_centers: list[tuple[float, float]]  # keymap centers + region centroids
    radius_m: float  # calibrated search radius
    rotation_priors: list[RotationPrior]
    scale_priors: list[ScalePrior]
    keymap_regions: list[list[list[float]]] | None = None  # world rings
    label_features: list[LabelFeature] | None = None
    block_index: dict[str, list[Block]] | None = None


def frame_around(
    center_lonlat: tuple[float, float], *, half_m: float, res_m: float = OSM_RES_M
) -> FrameSpec:
    """A square FrameSpec of ±half_m metres about a lon/lat center."""
    size = int(round(2 * half_m / res_m))
    return FrameSpec(
        origin=center_lonlat,
        x_min=-half_m,
        y_max=half_m,
        res_m=res_m,
        shape=(size, size),
    )


def frame_bounds_lonlat(frame: FrameSpec) -> tuple[float, float, float, float]:
    """(min_lon, max_lon, min_lat, max_lat) covered by the frame."""
    kx, ky = frame.metre_scales()
    rows, cols = frame.shape
    x_max = frame.x_min + cols * frame.res_m
    y_min = frame.y_max - rows * frame.res_m
    return (
        frame.origin[0] + frame.x_min / kx,
        frame.origin[0] + x_max / kx,
        frame.origin[1] + y_min / ky,
        frame.origin[1] + frame.y_max / ky,
    )


def osm_rasters(
    frame: FrameSpec, features: list[dict], *, width_m: float = OSM_WIDTH_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prob, valid, skeleton) rasters of OSM centerlines in the frame.

    prob strokes each centerline at ``width_m`` — the OSM analog of a road-UNet
    P(road) map, playing the "fixed" role in match_at_rotation. skeleton is the
    same polylines at 1 px: the exact centerline (no thinning needed), which
    the chamfer distance transform is built from. valid is all-true — OSM
    knowledge covers the whole frame, unlike an edge-join anchor page.
    """
    rows, cols = frame.shape
    prob = np.zeros((rows, cols), np.float32)
    skeleton = np.zeros((rows, cols), np.uint8)
    min_lon, max_lon, min_lat, max_lat = frame_bounds_lonlat(frame)
    kx, ky = frame.metre_scales()
    thickness = max(2, int(round(width_m / frame.res_m)))
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            lines = [geometry["coordinates"]]
        elif geometry.get("type") == "MultiLineString":
            lines = geometry["coordinates"]
        else:
            continue
        for line in lines:
            if len(line) < 2:
                continue
            pts = np.asarray(line, dtype=np.float64)[:, :2]
            if (
                pts[:, 0].max() < min_lon
                or pts[:, 0].min() > max_lon
                or pts[:, 1].max() < min_lat
                or pts[:, 1].min() > max_lat
            ):
                continue
            px = np.empty_like(pts)
            px[:, 0] = ((pts[:, 0] - frame.origin[0]) * kx - frame.x_min) / frame.res_m
            px[:, 1] = (frame.y_max - (pts[:, 1] - frame.origin[1]) * ky) / frame.res_m
            poly = px.round().astype(np.int32)
            cv2.polylines(prob, [poly], False, 1.0, thickness)
            cv2.polylines(skeleton, [poly], False, 1, 1)
    valid = np.ones((rows, cols), dtype=bool)
    return prob, valid, skeleton.astype(bool)


def osm_distance_m(skeleton: np.ndarray, res_m: float = OSM_RES_M) -> np.ndarray:
    """Clamped distance transform (metres) from the OSM centerline skeleton."""
    inverted = (~skeleton).astype(np.uint8)
    distance = cv2.distanceTransform(inverted, cv2.DIST_L2, 3) * res_m
    return np.minimum(distance, CHAMFER_CLAMP_M)


def pose_theta_deg(pose: np.ndarray) -> float:
    """The cv2 rotation angle of a page-px -> raster-px similarity pose."""
    return math.degrees(math.atan2(pose[0, 1], pose[0, 0]))


def affine_theta_deg(affine_local: np.ndarray, frame: FrameSpec) -> float:
    """The cv2 rotation angle of a page-px -> lon/lat affine, via the frame."""
    return pose_theta_deg(frame.page_to_raster_affine(affine_local))


def wrap_deg(value: float) -> float:
    """Fold an angle in degrees to (-180, 180]."""
    return (value + 180.0) % 360.0 - 180.0


def raster_angle_deg(dlon: float, dlat: float, kx: float, ky: float) -> float:
    """y-down raster-frame angle of a world direction given in lon/lat deltas."""
    return math.degrees(math.atan2(-dlat * ky, dlon * kx))


def dedupe_thetas(
    priors: list[RotationPrior],
    tolerance_deg: float = THETA_DEDUPE_DEG,
    cap: int = MAX_PRIOR_THETAS,
) -> list[RotationPrior]:
    """Drop near-duplicate thetas, keeping the first (highest-rung) of each."""
    kept: list[RotationPrior] = []
    for prior in priors:
        if any(
            abs(wrap_deg(prior.theta_deg - k.theta_deg)) <= tolerance_deg for k in kept
        ):
            continue
        kept.append(prior)
        if len(kept) >= cap:
            break
    return kept


def cluster_rotation(
    thetas: list[tuple[float, float]], tolerance_deg: float = 30.0
) -> tuple[float, int]:
    """Weighted circular mean of the largest rotation-consistent cluster.

    Each entry is (theta_deg, weight). A spurious pair (misread neighbor
    number, wrong centroid) implies a rotation inconsistent with the true
    ones and falls outside the dominant cluster. Returns (mean_deg, n_inliers);
    (0, 0) for fewer than two entries.
    """
    if len(thetas) < 2:
        return 0.0, 0
    best: list[int] = []
    best_weight = -1.0
    for center, _ in thetas:
        inliers = [
            i
            for i, (theta, _) in enumerate(thetas)
            if abs(wrap_deg(theta - center)) <= tolerance_deg
        ]
        weight = sum(thetas[i][1] for i in inliers)
        if len(inliers) > len(best) or (
            len(inliers) == len(best) and weight > best_weight
        ):
            best, best_weight = inliers, weight
    sines = sum(math.sin(math.radians(thetas[i][0])) * thetas[i][1] for i in best)
    cosines = sum(math.cos(math.radians(thetas[i][0])) * thetas[i][1] for i in best)
    if sines == 0 and cosines == 0:
        return 0.0, 0
    return math.degrees(math.atan2(sines, cosines)), len(best)


def unique_street_features(
    features: list[LabelFeature],
) -> list[tuple[LabelFeature, list[str]]]:
    """(representative feature, candidate street texts) per unambiguous label.

    prepare_label_features emits one feature per candidate street of an
    ambiguous label; a label is usable here only when all its candidates are
    directional variants of one physical street (K STREET NE/NW/...), whose
    blocks then merge into one segment soup.
    """
    texts_per_center: dict[tuple[float, float], set[str]] = {}
    for feature in features:
        texts_per_center.setdefault(feature.center, set()).add(feature.text)
    result: list[tuple[LabelFeature, list[str]]] = []
    seen: set[tuple[float, float]] = set()
    for feature in features:
        if feature.center in seen:
            continue
        texts = texts_per_center[feature.center]
        if len({street_base_name(t) for t in texts}) > 1:
            continue
        seen.add(feature.center)
        result.append((feature, sorted(texts)))
    return result


def label_blocks(texts: list[str], block_index: dict[str, list[Block]]) -> list[Block]:
    """The merged block list for a label's candidate street texts."""
    blocks: list[Block] = []
    for text in texts:
        blocks.extend(block_index.get(text, []))
    return blocks


def label_osm_rotations(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    near_lonlat: tuple[float, float],
) -> list[RotationPrior]:
    """Rung (a): rotation priors from OCR street labels vs OSM bearings.

    One matched label pins the rotation mod 180 (its pixel long-axis maps onto
    its street's tangent, but either way along it) -> two directed candidates.
    A pair of labels on two *distinct* streets resolves the flip: the pixel
    offset A->B mapped through the right rotation points the same way as the
    world offset between the streets; the wrong flip reverses it exactly ->
    one "label-pair-exact" candidate.

    The streets' tangents/positions are taken at their nearest approach to
    ``near_lonlat`` (the key-map location) — the best page-position estimate
    available before any pose exists.
    """
    kx = 111_320.0 * math.cos(math.radians(near_lonlat[1]))
    ky = 110_540.0
    usable: list[tuple[LabelFeature, tuple[float, float], float]] = []
    undirected: list[RotationPrior] = []
    for feature, texts in unique_street_features(features):
        blocks = label_blocks(texts, block_index)
        projected = project_to_polyline(
            near_lonlat[0], near_lonlat[1], blocks, extrapolate=False
        )
        if projected is None:
            continue
        nlon, nlat, tangent = projected
        # The tangent is an angle in lon/lat-degree space; convert its unit
        # vector to the metre frame before taking the raster-frame angle.
        tangent_raster = raster_angle_deg(math.cos(tangent), math.sin(tangent), kx, ky)
        theta = (math.degrees(feature.dir_pix) - tangent_raster) % 180.0
        usable.append((feature, (nlon, nlat), theta))
        undirected.append(RotationPrior(theta, 4.0, "label-osm-mod180"))
        undirected.append(RotationPrior(theta - 180.0, 4.0, "label-osm-mod180"))

    exact: list[RotationPrior] = []
    for i, (feat_a, world_a, theta_a) in enumerate(usable):
        for feat_b, world_b, _ in usable[i + 1 :]:
            if street_base_name(feat_a.text) == street_base_name(feat_b.text):
                continue
            world_dx = (world_b[0] - world_a[0]) * kx
            world_dy_north = (world_b[1] - world_a[1]) * ky
            if math.hypot(world_dx, world_dy_north) < 30.0:
                continue  # streets cross here; the offset direction is noise
            world_angle = math.degrees(math.atan2(-world_dy_north, world_dx))
            page_dx = feat_b.center[0] - feat_a.center[0]
            page_dy = feat_b.center[1] - feat_a.center[1]
            if math.hypot(page_dx, page_dy) < 1e-6:
                continue
            page_angle = math.degrees(math.atan2(page_dy, page_dx))
            # theta = page_angle - raster_angle for directed vectors; of the
            # two flips theta_a/theta_a+180, keep the one agreeing in sign.
            for theta in (theta_a, theta_a - 180.0):
                if abs(wrap_deg(page_angle - world_angle - theta)) <= 60.0:
                    exact.append(
                        RotationPrior(wrap_deg(theta), 4.0, "label-pair-exact")
                    )
                    break
    return exact + undirected


def adjacency_keymap_rotations(
    image_directions: dict[int, tuple[tuple[float, float], float]],
    centroids: dict[int, tuple[float, float]],
    own_centroid: tuple[float, float],
) -> list[RotationPrior]:
    """Rung (b): rotation from printed-neighbor directions vs keymap geometry.

    A neighbor's printed number sits on the margin toward that neighbor
    (image frame); the key map's region centroids give the same direction in
    the world. Each pair implies a directed rotation; the largest consistent
    cluster wins.
    """
    kx = 111_320.0 * math.cos(math.radians(own_centroid[1]))
    ky = 110_540.0
    implied: list[tuple[float, float]] = []
    for number, (direction, confidence) in image_directions.items():
        neighbor = centroids.get(number)
        if neighbor is None:
            continue
        dlon = neighbor[0] - own_centroid[0]
        dlat = neighbor[1] - own_centroid[1]
        if math.hypot(dlon * kx, dlat * ky) < 1e-6:
            continue
        world_angle = raster_angle_deg(dlon, dlat, kx, ky)
        page_angle = math.degrees(math.atan2(direction[1], direction[0]))
        implied.append((wrap_deg(page_angle - world_angle), max(confidence, 0.05)))
    theta, inliers = cluster_rotation(implied)
    if inliers >= 2:
        return [RotationPrior(theta, 12.0, "adjacency-keymap")]
    return []


def page_scale_priors(
    volume_m_per_px: float,
    region_rings: list[list[list[float]]] | None,
    width: int,
    height: int,
) -> list[ScalePrior]:
    """The scale-prior ladder: volume median, plus a family rung on evidence.

    The key-map region's area implies a page scale; when it disagrees with the
    volume median by roughly a power of two (half/double-scale sheets), the
    corresponding family rung is added as a second candidate rather than
    trusting the schematic region size directly.
    """
    from mapsnap.keymap.locate import region_scale_m_per_px

    priors = [ScalePrior(volume_m_per_px, 0.05, "volume-median")]
    if region_rings:
        region_scale = region_scale_m_per_px(
            [[(p[0], p[1]) for p in ring] for ring in region_rings], width, height
        )
        if region_scale and region_scale > 0:
            rung = round(math.log2(region_scale / volume_m_per_px))
            if rung != 0 and abs(math.log2(region_scale / volume_m_per_px)) >= 0.6:
                priors.append(
                    ScalePrior(volume_m_per_px * (2.0**rung), 0.05, "family-rung")
                )
    return priors


def calibrated_radius_m(
    residuals_m: list[float], locator_radius_m: float
) -> tuple[float, str]:
    """Per-volume search radius from fitted pages' keymap-vs-fit residuals.

    The locator's default radius (~2x page spacing) is far looser than the key
    map's actual placement error; tightening the NCC search window to the
    observed p90 (+margin) removes most lattice aliases before they are ever
    scored. Falls back to the locator radius when too few fits exist.
    """
    if len(residuals_m) < 5:
        return locator_radius_m, "locator"
    p90 = float(np.percentile(residuals_m, 90))
    radius = p90 + CALIBRATED_RADIUS_MARGIN_M
    radius = max(CALIBRATED_RADIUS_MIN_M, min(radius, locator_radius_m))
    return radius, "calibrated"


def name_alignment(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    world_affine: np.ndarray,
    *,
    tau_m: float = 25.0,
    max_dist_m: float = 60.0,
    max_angle_deg: float = 25.0,
) -> NameAlignment:
    """Agreement between OCR street labels and OSM at a candidate pose.

    Each unambiguous label's center is projected through the pose and snapped
    to its own street's polyline; a hit needs both proximity and a matching
    direction. The +2 in the denominator keeps one lucky label from dominating
    while a renamed street (no match anywhere) scores 0, never negative.
    """
    lat0 = world_affine[1, 2]
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    eligible = unique_street_features(features)
    hits: list[dict] = []
    total = 0.0
    for feature, texts in eligible:
        blocks = label_blocks(texts, block_index)
        if not blocks:
            continue
        px, py = feature.center
        lon = world_affine[0, 0] * px + world_affine[0, 1] * py + world_affine[0, 2]
        lat = world_affine[1, 0] * px + world_affine[1, 1] * py + world_affine[1, 2]
        projected = project_to_polyline(lon, lat, blocks, extrapolate=False)
        if projected is None:
            continue
        nlon, nlat, tangent = projected
        dist = haversine_m(lat, lon, nlat, nlon)
        if dist > max_dist_m:
            continue
        # Mapped label direction vs street tangent, both as metre-frame angles.
        dx_page, dy_page = math.cos(feature.dir_pix), math.sin(feature.dir_pix)
        dlon = world_affine[0, 0] * dx_page + world_affine[0, 1] * dy_page
        dlat = world_affine[1, 0] * dx_page + world_affine[1, 1] * dy_page
        label_angle = math.degrees(math.atan2(dlat * ky, dlon * kx))
        tangent_angle = math.degrees(
            math.atan2(math.sin(tangent) * ky, math.cos(tangent) * kx)
        )
        diff = abs(label_angle - tangent_angle) % 180.0
        diff = min(diff, 180.0 - diff)
        if diff > max_angle_deg:
            continue
        value = math.exp(-dist / tau_m)
        total += value
        hits.append(
            {
                "text": feature.text,
                "dist_m": round(dist, 1),
                "angle_deg": round(diff, 1),
            }
        )
    n_labels = len(eligible)
    return NameAlignment(
        score=total / (n_labels + 2), n_labels=n_labels, n_hits=len(hits), hits=hits
    )


def region_containment_frac(
    world_affine: np.ndarray,
    page_size: tuple[int, int],
    regions: list[list[list[float]]],
) -> float:
    """Fraction of the posed footprint inside the (buffered) keymap region."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    lon_r, lat_r = regions[0][0][0], regions[0][0][1]
    kxr = 111_320.0 * math.cos(math.radians(lat_r))
    kyr = 110_540.0

    def ring_metres(ring: list[list[float]]) -> list[tuple[float, float]]:
        return [((lon - lon_r) * kxr, (lat - lat_r) * kyr) for lon, lat in ring]

    region_poly = unary_union(
        [Polygon(ring_metres(r)).buffer(0) for r in regions]
    ).buffer(CONTAINMENT_BUFFER_M)
    width, height = page_size
    corners = []
    for x, y in [(0, 0), (width, 0), (width, height), (0, height)]:
        lon = world_affine[0, 0] * x + world_affine[0, 1] * y + world_affine[0, 2]
        lat = world_affine[1, 0] * x + world_affine[1, 1] * y + world_affine[1, 2]
        corners.append([lon, lat])
    footprint = Polygon(ring_metres(corners))
    if footprint.area <= 0:
        return 0.0
    return float(footprint.intersection(region_poly).area / footprint.area)


def snap_page(
    ctx: PageContext,
    features: list[dict],
    params: MatchParams = OSM_MATCH_PARAMS,
) -> list[SnapCandidate]:
    """Candidate placements of a page against OSM around its keymap location.

    Per search center: rasterize OSM into a local frame, run masked NCC at
    every (rotation prior x scale prior), clamp-refine with chamfer, then
    attach the truth-free ranking features (name alignment, containment,
    prior residuals, refine shift). Candidates from all centers are merged,
    near-duplicate locks deduped, and the top params.top_k returned by
    select_score.
    """
    res = params.resolution_m
    page_diag_m = math.hypot(ctx.width, ctx.height) * max(
        sp.m_per_px for sp in ctx.scale_priors
    )
    half_m = ctx.radius_m + page_diag_m / 2 + 100.0
    points = skeleton_points(ctx.prob, params.mask_threshold, params.mask_min_area)
    sigma_px = max(params.blur_sigma_m / res, 0.5)
    border_px = max(1, int(round(REFINE_SHIFT_MAX_M / res)))
    page_center = np.array([ctx.width / 2.0, ctx.height / 2.0, 1.0])

    collected: list[SnapCandidate] = []
    for center in ctx.search_centers:
        frame = frame_around(center, half_m=half_m, res_m=res)
        osm_prob, valid, skeleton = osm_rasters(frame, features)
        if not skeleton.any():
            continue
        distance = osm_distance_m(skeleton, res)
        fixed_blur = cv2.GaussianBlur(osm_prob, (0, 0), sigma_px)
        region = np.ones(frame.shape, dtype=bool)
        region[:border_px, :] = False
        region[-border_px:, :] = False
        region[:, :border_px] = False
        region[:, -border_px:] = False
        search_center = frame.lonlat_to_raster(*center)
        search_radius_px = ctx.radius_m / res

        # The prior ladder, plus the mask-mod-90 sweep — always appended so the
        # theta set is never empty and covers any 180-flip a prior missed.
        thetas = dedupe_thetas(ctx.rotation_priors)
        for theta in rotation_candidates(
            osm_prob, ctx.prob, params.jitter_deg, fixed_valid=valid
        ):
            if not any(
                abs(wrap_deg(theta - k.theta_deg)) <= THETA_DEDUPE_DEG for k in thetas
            ):
                thetas.append(RotationPrior(theta, 4.0, "mask-mod90"))

        candidates: list[JoinCandidate] = []
        provenance: dict[int, tuple[RotationPrior, ScalePrior]] = {}
        for scale_prior in ctx.scale_priors:
            raster_scale = scale_prior.m_per_px / res
            for prior in thetas:
                for candidate in match_at_rotation(
                    fixed_blur,
                    valid,
                    ctx.prob,
                    scale=raster_scale,
                    theta=prior.theta_deg,
                    params=params,
                    search_center=search_center,
                    search_radius_px=search_radius_px,
                ):
                    provenance[id(candidate)] = (prior, scale_prior)
                    candidates.append(candidate)
        if not candidates:
            continue
        initial_centers = {id(c): tuple(c.pose @ page_center) for c in candidates}
        ranked = refine_and_rank(
            candidates,
            distance,
            points,
            fixed_valid=valid,
            page_shape=ctx.prob.shape[:2],
            max_overlap_frac=params.max_overlap_frac,
            region=region,
            fixed_prob=osm_prob,
            target_prob=ctx.prob,
            fine_sigma_px=max(params.fine_sigma_m / res, 0.5),
            solve_scale=True,
        )
        for candidate in ranked:
            prior, scale_prior = provenance[id(candidate)]
            refined_center = candidate.pose @ page_center
            initial = initial_centers[id(candidate)]
            refine_shift = (
                math.hypot(
                    refined_center[0] - initial[0], refined_center[1] - initial[1]
                )
                * res
            )
            world = frame.raster_pose_to_world_affine(candidate.pose)
            lon_c = (
                world[0, 0] * page_center[0]
                + world[0, 1] * page_center[1]
                + world[0, 2]
            )
            lat_c = (
                world[1, 0] * page_center[0]
                + world[1, 1] * page_center[1]
                + world[1, 2]
            )
            center_dist = min(
                haversine_m(lat_c, lon_c, c[1], c[0]) for c in ctx.search_centers
            )
            snap = SnapCandidate(
                world_affine=world,
                center=(lon_c, lat_c),
                theta_deg=candidate.theta_deg,
                theta_source=prior.source,
                scale_m_per_px=scale_prior.m_per_px,
                scale_source=scale_prior.source,
                scale_adjust=candidate.scale_adjust,
                ncc=candidate.ncc,
                ncc_fine=candidate.ncc_fine,
                chamfer_mean_m=candidate.chamfer_mean_m,
                inlier_frac=candidate.inlier_frac,
                n_points=candidate.n_points,
                jtj_eig_ratio=candidate.jtj_eig_ratio,
                overlap_frac=candidate.overlap_frac,
                refine_shift_m=refine_shift,
                center_dist_m=center_dist,
                verification=candidate.verification_score(),
                plausible=candidate.plausible,
            )
            if refine_shift > REFINE_SHIFT_MAX_M:
                snap.plausible = False
                snap.gate_reasons.append("refine-shift")
            if center_dist > ctx.radius_m + RADIUS_SLACK_M:
                snap.plausible = False
                snap.gate_reasons.append("radius")
            if ctx.keymap_regions:
                snap.region_containment = region_containment_frac(
                    world, (ctx.width, ctx.height), ctx.keymap_regions
                )
                if snap.region_containment < CONTAINMENT_MIN:
                    snap.plausible = False
                    snap.gate_reasons.append("containment")
            if ctx.rotation_priors:
                # Both flips of a mod-180 prior are present as entries, so a
                # plain directed comparison is correct for every source.
                snap.prior_theta_residual_sigma = min(
                    abs(wrap_deg(candidate.theta_deg - p.theta_deg)) / p.sigma_deg
                    for p in ctx.rotation_priors
                )
            if ctx.label_features and ctx.block_index:
                snap.name = name_alignment(ctx.label_features, ctx.block_index, world)
            if not snap.plausible:
                snap.verification = -math.inf
            collected.append(snap)

    return merge_candidates(collected, params.top_k)


def merge_candidates(
    candidates: list[SnapCandidate], top_k: int
) -> list[SnapCandidate]:
    """Dedupe near-identical locks across frames and keep the best top_k.

    Implausible candidates rank behind all plausible ones (select_score -inf)
    but are retained up to the cap so the harness can report near-misses.
    """
    ordered = sorted(candidates, key=lambda c: -c.select_score())
    kept: list[SnapCandidate] = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            separation = haversine_m(
                candidate.center[1],
                candidate.center[0],
                existing.center[1],
                existing.center[0],
            )
            if (
                separation < MERGE_SEPARATION_M
                and abs(wrap_deg(candidate.theta_deg - existing.theta_deg)) < 10.0
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= top_k:
            break
    return kept
