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

import dataclasses
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mapsnap.edge_join import (
    CHAMFER_CLAMP_M,
    INLIER_M,
    FrameSpec,
    JoinCandidate,
    MatchParams,
    dominant_orientation_deg,
    match_at_rotation,
    pose_ncc,
    refine_and_rank,
    rotation_candidates,
    skeleton_points,
)
from mapsnap.feature_index import FeatureIndex
from mapsnap.georef_from_labels import LabelFeature, project_to_polyline
from mapsnap.streets import Block, street_base_name
from mapsnap.utils import haversine_m

OSM_RES_M = 2.0  # raster resolution
OSM_WIDTH_M = 12.0  # stroked corridor width for the OSM "P(road)" analog
REFINE_SHIFT_MAX_M = 30.0  # chamfer refinement may not slide farther than this
# Hard containment gate: only egregious slides fail it. Schematic keymap
# regions run small relative to true page footprints (Hudson true locks sit
# near 0.33), so the run_join-style 0.35 threshold kills good fits here;
# discrimination comes from the ranking bonus and the radius gate instead.
CONTAINMENT_MIN = 0.15  # of the footprint inside the (buffered) keymap region
CONTAINMENT_BUFFER_M = 60.0
RADIUS_SLACK_M = 60.0  # post-refinement slack on the center-distance gate
THETA_DEDUPE_DEG = 3.0
MAX_PRIOR_THETAS = 8
# Rotation-prior confidence, for skipping the rest of the ladder. A
# label-pair-exact rung is ONE label pair's reading of the page rotation, not
# the page's: a page usually carries several, and a mis-OCR'd or mis-matched
# label makes its pair disagree with the rest. Several that all agree is
# independent corroboration; one on its own is a hypothesis (see
# confident_theta_deg).
CONFIDENT_MIN_RUNGS = 2
CONFIDENT_AGREE_DEG = 8.0
CONFIDENT_WINDOW_DEG = 12.0
MERGE_SEPARATION_M = 60.0  # candidates closer than this are the same lock
CALIBRATED_RADIUS_MARGIN_M = 100.0
CALIBRATED_RADIUS_MIN_M = 150.0

# select_score weights (hand-tuned; every term is logged per candidate so a
# re-ranking experiment can refit them on cached candidates). The two
# name-miss knobs read the environment (MAPSNAP_NAME_MISS,
# MAPSNAP_NAME_MISS_MIN_LABELS) so an A/B or a sweep is ONE build with the arm
# chosen at run time: `mapsnap fit` spawns every stage as a subprocess, and
# the environment crosses that boundary where a CLI flag would have to be
# threaded through each stage.
W_NAME = 1.0
W_CONTAIN = 0.3
W_PRIOR = 0.1

# Cost charged to a pose whose labels match NOTHING (issue #375). The reward
# half of name_alignment is floored at zero, so such a pose scores the same as
# a page with no labels at all -- and a grid alias with strong P(road) shape
# evidence outranks the truth (richmond p311: 0 of 9 labels hit, published
# 14,713 ft off, beating the 18.8 ft pose by 0.046).
#
# The penalty is a TAIL, not a slope. What the corpus shows is that *zero*
# hits with >=3 eligible labels is damning: it happens in 0.8% of accurate
# poses and 43.3% of >=500 ft poses. It shows nothing about 1-of-5 being worse
# than 3-of-5 in proportion, and a first cut that charged per miss regressed
# exactly there -- nashville p24 (a correct pose hitting 4 of 7) lost 0.17 of
# select score and stopped clearing PRODUCTION_GATE_SCORE, and brooklyn p9's
# correct pose (1 of 5) was demoted under an alias that hits 4 of 5 because
# sliding along a long avenue keeps a label near its own street.
W_NAME_MISS = float(os.environ.get("MAPSNAP_NAME_MISS", "0.5"))
# Below this many eligible labels a page says nothing either way, so the
# penalty stays inert (a couple of unmatched labels are noise, not
# contradiction). Measured on 13,326 fresh candidates across the 20 truth
# volumes, raising the floor sharpens the signal because thin-evidence pages
# are where an accurate pose legitimately matches nothing:
#
#   floor   accurate poses w/ zero hits   wrong poses w/ zero hits   ratio
#     3                0.8%                        43.6%              51:1
#     5                0.4%                        39.4%             105:1
#     6                0.2%                        36.3%             199:1
#
# Every page the corpus A/B damaged carried exactly 3 eligible labels
# (detroit p93, brooklyn p13, miami p74 -- a correct 32.8 ft pose matching
# 0 of 3); both rescues carry 9 and 10 (richmond p311, p323).
NAME_MISS_MIN_LABELS = int(os.environ.get("MAPSNAP_NAME_MISS_MIN_LABELS", "5"))

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
    """OCR street-name agreement with a candidate pose (signed: see evidence)."""

    score: float
    n_labels: int
    n_hits: int
    hits: list[dict] = field(default_factory=list)

    @property
    def evidence(self) -> float:
        """The signed term the ranking uses (see name_evidence)."""
        return name_evidence(self.score, self.n_labels, self.n_hits)


def name_evidence_of(name: dict | None) -> float | None:
    """Signed name evidence from a serialized name block, or None when absent.

    Records written before issue #375 carry no ``evidence`` key, so the value
    is recomputed from the fields that were always logged.
    """
    if not name:
        return None
    if "evidence" in name:
        return float(name["evidence"])
    score, n_labels, n_hits = (
        name.get("score"),
        name.get("n_labels"),
        name.get("n_hits"),
    )
    if score is None or n_labels is None or n_hits is None:
        return None if score is None else float(score)
    return name_evidence(float(score), int(n_labels), int(n_hits))


def name_evidence(score: float, n_labels: int, n_hits: int) -> float:
    """Name agreement as SIGNED evidence: matching nothing costs.

    ``score`` is name_alignment's reward-only value. A pose that matches at
    least one of its eligible labels keeps that value unchanged; a pose that
    matches NONE of them, on a page carrying at least NAME_MISS_MIN_LABELS
    eligible labels, is charged W_NAME_MISS scaled by the same
    n_labels/(n_labels + 2) shape the reward uses -- so contradiction grows
    with how much the page had to say, and a page with nothing to say is
    unaffected.

    Deliberately not a per-miss slope: see W_NAME_MISS for the regressions
    that shape produced.
    """
    if n_labels < NAME_MISS_MIN_LABELS or n_hits > 0:
        return score
    return score - W_NAME_MISS * n_labels / (n_labels + 2)


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
    # Stamp separations for contradiction-demoted rescue targets
    # (adjacency_gate.StampGate); None when the page carries no hints.
    # min over partners = the permissive hard-gate statistic; median = the
    # strict corroboration statistic (junk hints scatter, so no wrong pose
    # satisfies most of them).
    stamp_separation_m: float | None = None
    stamp_median_m: float | None = None
    # What the page was matched against: "osm", or "keymap:<stem>" for the
    # key map's P(road) map (#211).
    target: str = "osm"

    def select_score(self) -> float:
        """The ranking score: matcher verification plus soft evidence bonuses."""
        return selection_score(
            self.verification,
            self.name.evidence if self.name is not None else None,
            self.region_containment,
            self.prior_theta_residual_sigma,
        )


def selection_score(
    verification: float,
    name_evidence_value: float | None,
    region_containment: float | None,
    prior_theta_residual_sigma: float | None,
) -> float:
    """The ranking score: matcher verification plus soft evidence terms.

    One formula for the ladder's own candidates (SnapCandidate.select_score)
    and for synthetic poses scored after the fact (rank_pose), so a truth pose
    lands on the same footing as the search's candidates.

    ``name_evidence_value`` is the SIGNED name term (name_evidence), not
    name_alignment's reward-only score: a pose whose labels match nothing pays
    for it rather than merely failing to earn.
    """
    if not math.isfinite(verification):
        return -math.inf
    score = verification
    if name_evidence_value is not None:
        score += W_NAME * name_evidence_value
    if region_containment is not None:
        score += W_CONTAIN * region_containment
    if prior_theta_residual_sigma is not None:
        # Signed: within 1 sigma earns the bonus, a gross disagreement
        # (e.g. a 180-flip against agreeing directed priors) costs it.
        score += W_PRIOR * max(-1.0, 1.0 - prior_theta_residual_sigma)
    return score


def directed_prior_residual_sigma(
    priors: list["RotationPrior"], theta_deg: float
) -> float | None:
    """How many sigmas a rotation sits from the nearest DIRECTED prior, or None.

    Only directed priors can flag a 180-flip: the mod-180 rung emits both flips
    as entries, so including it would let the wrong flip always match one of
    the pair and pin the residual at zero.
    """
    directed = [p for p in priors if p.source != "label-osm-mod180"]
    if not directed:
        return None
    return min(abs(wrap_deg(theta_deg - p.theta_deg)) / p.sigma_deg for p in directed)


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
    # Memos of the maps derived from `prob`. The road skeleton (a Zhang
    # thinning pass) is needed by both evaluate_pose and snap_page, and the
    # dominant road orientation is re-derived once per search center. Each is a
    # pure function of `prob`, so a reuse is identical to a recompute.
    road_points_cache: dict[tuple[float, int], np.ndarray] = field(
        default_factory=dict, repr=False
    )
    orientation_cache: float | None = field(default=None, repr=False)

    def road_points(self, params: MatchParams) -> np.ndarray:
        """The page's road-skeleton points (x, y in page pixels), thinned once per page."""
        key = (params.mask_threshold, params.mask_min_area)
        if key not in self.road_points_cache:
            self.road_points_cache[key] = skeleton_points(self.prob, *key)
        return self.road_points_cache[key]

    def road_orientation_deg(self) -> float:
        """The page's dominant road direction folded to [0, 90), measured once per page."""
        if self.orientation_cache is None:
            self.orientation_cache = dominant_orientation_deg(self.prob)
        return self.orientation_cache


def cluster_search_centers(
    centers: list[tuple[float, float]], link_m: float
) -> list[tuple[float, float]]:
    """Merge search centers whose discs largely overlap (single linkage, centroid).

    A page whose key-map number is printed many times (LA's lettered p1499
    sheets inherit ~20 detections of "1499") searches every center with a full
    NCC+refine pass; centers closer than ``link_m`` cover mostly the same
    ground, so one centroid search suffices. Corpus timing showed center
    multiplicity is the strongest cost correlate (r=+0.62) while sheet size is
    not (r=+0.08); see issue #155.
    """
    if len(centers) <= 1:
        return list(centers)
    kx = 111_320.0 * math.cos(math.radians(centers[0][1]))
    clusters: list[list[tuple[float, float]]] = []
    for lon, lat in centers:
        for cluster in clusters:
            if any(
                math.hypot((lon - a) * kx, (lat - b) * 110_540.0) <= link_m
                for a, b in cluster
            ):
                cluster.append((lon, lat))
                break
        else:
            clusters.append([(lon, lat)])
    return [
        (
            sum(c[0] for c in cluster) / len(cluster),
            sum(c[1] for c in cluster) / len(cluster),
        )
        for cluster in clusters
    ]


def frame_around(
    center_lonlat: tuple[float, float], *, half_m: float, res_m: float = OSM_RES_M
) -> FrameSpec:
    """A square FrameSpec of ±half_m metres about a lon/lat center."""
    size = round(2 * half_m / res_m)
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
    frame: FrameSpec, features: FeatureIndex, *, width_m: float = OSM_WIDTH_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prob, valid, skeleton) rasters of OSM centerlines in the frame.

    prob strokes each centerline at ``width_m`` — the OSM analog of a road-UNet
    P(road) map, playing the "fixed" role in match_at_rotation. skeleton is the
    same polylines at 1 px: the exact centerline (no thinning needed), which
    the chamfer distance transform is built from. valid is all-true — OSM
    knowledge covers the whole frame, unlike an edge-join anchor page.

    The index pre-culls to the features whose bounding box reaches the frame;
    the per-line bbox test below still runs on every one of them, so what gets
    drawn is exactly what a scan of the whole volume would draw.
    """
    rows, cols = frame.shape
    prob = np.zeros((rows, cols), np.float32)
    skeleton = np.zeros((rows, cols), np.uint8)
    bounds = frame_bounds_lonlat(frame)
    min_lon, max_lon, min_lat, max_lat = bounds
    kx, ky = frame.metre_scales()
    thickness = max(2, round(width_m / frame.res_m))
    for feature in features.near_bbox(bounds):
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


@dataclass
class KeymapTarget:
    """A key map's P(road) map as a snap target, in place of OSM (#211).

    Where the street grid has changed since the survey there is nothing on OSM
    for a page to lock onto and it lands on an alias. The key map is a
    Sanborn-era drawing of the same streets, so its road-probability map can
    stand in for OSM as the thing the page's P(road) is matched against. The
    sheet is placed by the pipeline's own key-map model (keymap_model: a
    smoothed thin-plate spline through the sheet's inlier intersections).

    `model` is built lazily from `georef`, so a target pickles into a worker.
    """

    stem: str
    georef: dict
    prob: np.ndarray  # float32 in [0, 1], key-map pixels
    image_path: Path
    model_cache: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Key-map pixel (N, 2) -> lon/lat (N, 2)."""
        if self.model_cache is None:
            from mapsnap.keymap.snap import keymap_model

            self.model_cache = keymap_model(self.georef)
        return self.model_cache

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["model_cache"] = None
        return state

    def corner_centroid(self) -> tuple[float, float]:
        corners = self.georef["corners"]
        return (
            sum(c[0] for c in corners) / len(corners),
            sum(c[1] for c in corners) / len(corners),
        )


def load_keymap_targets(volume: Path) -> list[KeymapTarget]:
    """Every key map of a volume with a georef and a P(road) map, by stem."""
    raw = volume / "raw"
    targets: list[KeymapTarget] = []
    for keymap_json in sorted(raw.glob("*.keymap.json")):
        stem = keymap_json.name[: -len(".keymap.json")]
        georef_path = raw / f"{stem}.georef.json"
        prob_path = raw / f"{stem}.roadprob.png"
        if not georef_path.exists() or not prob_path.exists():
            continue
        image = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        georef = json.loads(georef_path.read_text())
        if not georef.get("corners"):
            continue
        image_path = next(
            (
                raw / f"{stem}.{ext}"
                for ext in ("jpg", "png")
                if (raw / f"{stem}.{ext}").exists()
            ),
            prob_path,
        )
        targets.append(
            KeymapTarget(
                stem=stem,
                georef=georef,
                prob=image.astype(np.float32) / 255.0,
                image_path=image_path,
            )
        )
    return targets


def nearest_keymap_target(
    targets: list[KeymapTarget], lonlat: tuple[float, float]
) -> KeymapTarget | None:
    """The key map whose corner quadrilateral contains the point, else the nearest."""
    if not targets:
        return None
    from shapely.geometry import Point, Polygon

    point = Point(lonlat)
    for target in targets:
        if Polygon(target.georef["corners"]).contains(point):
            return target
    return min(
        targets,
        key=lambda t: math.hypot(
            (t.corner_centroid()[0] - lonlat[0]) * math.cos(math.radians(lonlat[1])),
            t.corner_centroid()[1] - lonlat[1],
        ),
    )


# Inverse-mapping grid step (raster cells) and Newton iterations for
# keymap_rasters; the spline is near-affine over a search frame.
KEYMAP_RASTER_GRID = 8
KEYMAP_RASTER_NEWTON = 4
KEYMAP_SKELETON_MIN_AREA = 100


def keymap_rasters(
    frame: FrameSpec, target: KeymapTarget
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prob, valid, skeleton) rasters of a key map's P(road) map in the frame.

    The key-map analog of osm_rasters. Each raster cell is mapped back to a
    key-map pixel by inverting the key-map model: a coarse grid of cells is
    solved by Newton iteration with the model's tangent at the frame centre
    as the Jacobian (the spline is near-affine over a frame), interpolated
    to every cell, and the P(road) map is resampled through it. valid marks
    cells that land on the sheet; skeleton is the thinned road mask, which
    the chamfer distance transform is built from, as OSM's centerlines are.
    """
    from mapsnap.keymap.snap import local_tangent
    from mapsnap.road_model import road_mask, road_skeleton

    rows, cols = frame.shape
    kx, ky = frame.metre_scales()
    step = KEYMAP_RASTER_GRID
    grid_rows = np.arange(0, rows + step, step, dtype=np.float64)
    grid_cols = np.arange(0, cols + step, step, dtype=np.float64)
    cc, rr = np.meshgrid(grid_cols, grid_rows)
    lon = frame.origin[0] + (frame.x_min + cc * frame.res_m) / kx
    lat = frame.origin[1] + (frame.y_max - rr * frame.res_m) / ky
    wanted = np.column_stack([lon.ravel(), lat.ravel()])

    # Seed every node from the tangent at the sheet point nearest the frame
    # centre, then refine with the same Jacobian: Newton on a near-affine map.
    centre_lonlat = (float(lon.mean()), float(lat.mean()))
    height, width = target.prob.shape
    seed_px = np.array([width / 2.0, height / 2.0])
    tangent = local_tangent(target.model, (float(seed_px[0]), float(seed_px[1])))
    jacobian = tangent[:, :2]
    inverse = np.linalg.inv(jacobian)
    seed_world = target.model(seed_px[None, :])[0]
    pixels = seed_px + (np.array(centre_lonlat) - seed_world) @ inverse.T
    tangent = local_tangent(target.model, (float(pixels[0]), float(pixels[1])))
    inverse = np.linalg.inv(tangent[:, :2])
    pixels = np.tile(pixels, (len(wanted), 1))
    for _ in range(KEYMAP_RASTER_NEWTON):
        pixels = pixels + (wanted - target.model(pixels)) @ inverse.T

    map_x = pixels[:, 0].reshape(cc.shape).astype(np.float32)
    map_y = pixels[:, 1].reshape(cc.shape).astype(np.float32)
    # Interpolate the node grid at each cell's fractional node coordinate.
    # (cv2.resize would do this with pixel-centre alignment, shifting every
    # cell by nearly half a step.)
    fine_r, fine_c = np.mgrid[0:rows, 0:cols]
    node_c = (fine_c / step).astype(np.float32)
    node_r = (fine_r / step).astype(np.float32)
    full_x = cv2.remap(map_x, node_c, node_r, cv2.INTER_LINEAR)
    full_y = cv2.remap(map_y, node_c, node_r, cv2.INTER_LINEAR)
    valid = (full_x >= 0) & (full_x < width - 1) & (full_y >= 0) & (full_y < height - 1)
    prob = cv2.remap(
        target.prob, full_x, full_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    prob[~valid] = 0.0
    mask = road_mask(prob, threshold=0.5, min_area=KEYMAP_SKELETON_MIN_AREA)
    skeleton = road_skeleton(mask)
    return prob.astype(np.float32), valid, skeleton.astype(bool)


def target_rasters(
    frame: FrameSpec, features: FeatureIndex, target: KeymapTarget | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fixed rasters for a frame: OSM's, or the key map's when a target is given."""
    if target is None:
        return osm_rasters(frame, features)
    return keymap_rasters(frame, target)


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


def confident_theta_deg(priors: list[RotationPrior]) -> float | None:
    """The rotation the page's label pairs unanimously imply, or None.

    Each "label-pair-exact" rung comes from one pair of OCR'd labels on
    distinct streets, and the ladder normally carries several — a page with
    four usable labels emits up to six pairs. When they all agree, each pair
    independently corroborates the others (a mis-read label, or one matched to
    the wrong street, throws its pairs off by tens of degrees), and no other
    rung has ever out-scored them in the corpus, so the search can drop the
    rest of the ladder. A lone rung is just a hypothesis and gets no such
    trust: pruning to the first rung of a page whose pairs disagree costs real
    placements (issue #155).
    """
    exact = [p.theta_deg for p in priors if p.source == "label-pair-exact"]
    if len(exact) < CONFIDENT_MIN_RUNGS:
        return None
    if max(abs(wrap_deg(theta - exact[0])) for theta in exact) > CONFIDENT_AGREE_DEG:
        return None
    return exact[0]


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


def evaluate_pose(
    ctx: PageContext,
    features: FeatureIndex,
    world_affine: np.ndarray,
    params: MatchParams = OSM_MATCH_PARAMS,
    target: KeymapTarget | None = None,
) -> dict | None:
    """Score an EXISTING pose with the matcher's own evidence (no search).

    The arbitration head-to-head: an incumbent RANSAC pose and a snap
    candidate are judged by identical features — road-skeleton chamfer against
    the OSM centerlines, fine content correlation, and name alignment — so
    "challenger beats incumbent" compares like with like. The verification
    formula matches JoinCandidate.verification_score exactly. Returns None
    when the pose cannot be evaluated (no OSM in the frame, too few skeleton
    points).
    """
    res = params.resolution_m
    center_x, center_y = ctx.width / 2.0, ctx.height / 2.0
    lon_c = (
        world_affine[0, 0] * center_x
        + world_affine[0, 1] * center_y
        + world_affine[0, 2]
    )
    lat_c = (
        world_affine[1, 0] * center_x
        + world_affine[1, 1] * center_y
        + world_affine[1, 2]
    )
    diag_m = math.hypot(ctx.width, ctx.height) * max(
        sp.m_per_px for sp in ctx.scale_priors
    )
    frame = frame_around((lon_c, lat_c), half_m=diag_m / 2 + 100.0, res_m=res)
    osm_prob, valid, skeleton = target_rasters(frame, features, target)
    if not skeleton.any():
        return None
    distance = osm_distance_m(skeleton, res)
    pose = frame.page_to_raster_affine(world_affine)
    points = ctx.road_points(params)
    if len(points) < 10:
        return None
    if len(points) > 3000:
        points = points[:: len(points) // 3000 + 1]
    placed = np.column_stack([points, np.ones(len(points))]) @ pose.T
    rows = np.clip(placed[:, 1].round().astype(int), 0, distance.shape[0] - 1)
    cols = np.clip(placed[:, 0].round().astype(int), 0, distance.shape[1] - 1)
    sampled = distance[rows, cols]
    inlier_frac = float((sampled < INLIER_M).mean())
    chamfer_mean = float(sampled.mean())
    ncc_fine = pose_ncc(
        osm_prob,
        valid,
        ctx.prob,
        pose,
        sigma_px=max(params.fine_sigma_m / res, 0.5),
    )
    evaluation: dict = {
        "verification": round(
            inlier_frac + ncc_fine - chamfer_mean / CHAMFER_CLAMP_M, 4
        ),
        "inlier_frac": round(inlier_frac, 4),
        "chamfer_mean_m": round(chamfer_mean, 2),
        "ncc_fine": round(ncc_fine, 4),
        "n_points": len(sampled),
    }
    if ctx.label_features and ctx.block_index:
        name = name_alignment(ctx.label_features, ctx.block_index, world_affine)
        evaluation["name"] = {
            "score": round(name.score, 4),
            "evidence": round(name.evidence, 4),
            "n_labels": name.n_labels,
            "n_hits": name.n_hits,
        }
    return evaluation


def rank_pose(
    ctx: PageContext,
    features: FeatureIndex,
    world_affine: np.ndarray,
    params: MatchParams = OSM_MATCH_PARAMS,
    target: KeymapTarget | None = None,
) -> dict | None:
    """Score an EXISTING pose with the ladder's FULL ranking features.

    evaluate_pose's evidence plus the soft bonuses the search's own candidates
    carry (key-map region containment, directed-prior rotation residual), so
    the result has a ``select_score`` directly comparable with theirs. This is
    what puts a synthetic candidate -- the truth pose in the debugger record
    (#325) -- on the same footing as the ladder: if it outscores every
    candidate the search never reached it; if a candidate outscores it the
    page's evidence itself prefers a wrong pose. None when the pose cannot be
    evaluated (see evaluate_pose).
    """
    evaluation = evaluate_pose(ctx, features, world_affine, params, target)
    if evaluation is None:
        return None
    affine = np.asarray(world_affine, dtype=float)
    theta = math.degrees(math.atan2(-affine[1, 0], affine[0, 0]))
    containment = (
        region_containment_frac(affine, (ctx.width, ctx.height), ctx.keymap_regions)
        if ctx.keymap_regions
        else None
    )
    residual = directed_prior_residual_sigma(ctx.rotation_priors, theta)
    name = evaluation.get("name")
    evaluation["world_affine"] = [[float(v) for v in row] for row in affine]
    evaluation["theta_deg"] = round(theta, 2)
    evaluation["select_score"] = round(
        selection_score(
            evaluation["verification"],
            name_evidence_of(name),
            containment,
            residual,
        ),
        4,
    )
    if containment is not None:
        evaluation["region_containment"] = round(containment, 3)
    if residual is not None:
        evaluation["prior_theta_residual_sigma"] = round(residual, 2)
    return evaluation


def frame_thetas(
    ctx: PageContext,
    confident_deg: float | None,
    fixed: tuple[np.ndarray, np.ndarray],
    params: MatchParams,
) -> list[RotationPrior]:
    """The rotation ladder to sweep against one OSM frame (prob, valid).

    Every theta here costs a full masked NCC pass plus params.top_k chamfer
    refinements, and the refinements are where the matcher spends most of its
    time — so when the label pairs unanimously pin the rotation, the ladder
    collapses to a window around them and the mask-mod-90 sweep (four more
    rotations) is skipped. Without that corroboration the priors are deduped
    and the sweep is appended, so the set is never empty and always covers a
    180-flip a prior missed.
    """
    if confident_deg is not None:
        return dedupe_thetas(
            [
                prior
                for prior in ctx.rotation_priors
                if abs(wrap_deg(prior.theta_deg - confident_deg))
                <= CONFIDENT_WINDOW_DEG
            ]
        )
    osm_prob, valid = fixed
    thetas = dedupe_thetas(ctx.rotation_priors)
    for theta in rotation_candidates(
        osm_prob,
        ctx.prob,
        params.jitter_deg,
        fixed_valid=valid,
        target_dir=ctx.road_orientation_deg(),
    ):
        if not any(
            abs(wrap_deg(theta - k.theta_deg)) <= THETA_DEDUPE_DEG for k in thetas
        ):
            thetas.append(RotationPrior(theta, 4.0, "mask-mod90"))
    return thetas


def snap_page(
    ctx: PageContext,
    features: FeatureIndex,
    params: MatchParams = OSM_MATCH_PARAMS,
    target: KeymapTarget | None = None,
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
    # A small page (split panel) can cover less ground than the absolute
    # minimum-overlap floor; cap the floor at half its own area or the NCC
    # masks out every shift and no candidates ever surface.
    page_area_m2 = (
        ctx.width * ctx.height * min(sp.m_per_px for sp in ctx.scale_priors) ** 2
    )
    if params.min_overlap_m2 > 0.5 * page_area_m2:
        params = dataclasses.replace(params, min_overlap_m2=0.5 * page_area_m2)
    half_m = ctx.radius_m + page_diag_m / 2 + 100.0
    points = ctx.road_points(params)
    sigma_px = max(params.blur_sigma_m / res, 0.5)
    border_px = max(1, round(REFINE_SHIFT_MAX_M / res))
    page_center = np.array([ctx.width / 2.0, ctx.height / 2.0, 1.0])
    confident = confident_theta_deg(ctx.rotation_priors)

    collected: list[SnapCandidate] = []
    for center in ctx.search_centers:
        frame = frame_around(center, half_m=half_m, res_m=res)
        osm_prob, valid, skeleton = target_rasters(frame, features, target)
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

        thetas = frame_thetas(ctx, confident, (osm_prob, valid), params)

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
                target="osm" if target is None else f"keymap:{target.stem}",
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
            snap.prior_theta_residual_sigma = directed_prior_residual_sigma(
                ctx.rotation_priors, candidate.theta_deg
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
