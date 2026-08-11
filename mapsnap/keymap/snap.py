"""Georeference pages by matching their P(road) map to the key map's (#211).

A key map draws the volume's whole street network in one georeferenced sheet.
Every page is a rectangle somewhere on it, so a page whose own street OCR failed
can still be placed: extract P(road) for the page and for the key map, and slide
the page's roads over the key map's until they line up.

The search is over one scale and one rotation -- an isotropic similarity --
which is only correct in a frame where a correctly-placed page really is one.
Key-map georeferences are not that frame: Detroit's is 10.6% anisotropic, so a
page in its right place comes out visibly stretched in key-map pixels and the
right answer sits *outside* the search space. Measured on Detroit's seven
never-otherwise-fitted pages, the best isotropic pose was still 25.5 ft from
truth (median) against 52.6 ft actually achieved -- about half the error was the
search space being the wrong shape.

So the crop is first warped into a frame where the constraint holds. Polar-
decompose the key-map model's local linear part ``L = R S`` (S symmetric
positive definite) and warp by ``W = S / sqrt(det S)``::

    p_world = L p_keymap,  p_warp = W p_keymap
      =>  p_world = L W^-1 p_warp = sqrt(det S) * R p_warp

a pure rotation and scalar, exactly the family being searched. W has unit
determinant, so the crop is neither rescaled nor flipped, and the pose is
carried back through ``W^-1`` afterwards. No search dimension is added.

The key-map model is a thin-plate spline through the sheet's own inlier
intersections, so it has no single linear part; the Jacobian is taken at each
page's own region centroid and the spatial variation is followed to first order.
That local tangent is worth a little on its own (p75 142 -> 110 ft) but the
stretch is what moves the median: 60.4 -> 44.0 ft over 93 Detroit pages, with
pages within 25 ft going 5 -> 21.
"""

import argparse
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapsnap.keymap.road_prob import MIN_KEYMAP_INLIERS
from mapsnap.road_model import page_world_affine

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0
FEET_PER_METRE = 3.28084

SEARCH_SLACK_PX = 600
"""How far outside the page's expected footprint the crop reaches. The region
centroid is a good but not exact guess at where the page sits."""

PRIOR_SIGMA_PX = 260.0
"""Width of the soft prior pulling the match toward the page's key-map region.
Wide enough not to veto a real offset, tight enough to break grid aliasing."""

SCALE_LADDER = (0.94, 1.0, 1.06)
THETA_JITTER_DEG = (-3.0, 0.0, 3.0)
"""Search rungs around the volume's median page scale and rotation. The jitter
is deliberately narrow: it is a refinement of a prior, not a search for the
rotation. Pages far off the volume median (Detroit has a few at 90 degrees)
need a better prior, not a wider ladder -- widening it admits aliases."""

PEAK_SUPPRESS_PX = 70
"""Radius blanked around the winning peak before measuring the runner-up, so
the margin compares distinct placements rather than one peak against itself."""

GCP_CONFLICT_FT = 50.0
"""How far apart two readings of one key-map pixel may be before both are
discarded. A detected intersection matched to two different OSM intersections
is a contradiction, not noise: the spline can only average it, and averaging
drags the surrounding fit. Miami's key map has 99 such pairs among 507 inliers
(median 233 ft apart, max 490), and they cost it ~21 ft of page accuracy."""

TPS_SMOOTHING = 1e6
"""Thin-plate spline regularization. The intersections carry real residuals, so
an interpolating spline would chase them; this keeps the sheet-scale warp."""

STRETCH_PAD = 1.12
"""Crop padding for the unit-determinant stretch, which lengthens one axis."""

ROTATION_CLUSTER_DEG = 12.0
MIN_ROTATION_CLUSTER = 2
DISTINCT_ROTATION_DEG = 15.0
"""How neighbours' rotations become an extra hypothesis: angles within
ROTATION_CLUSTER_DEG are one cluster, a cluster needs MIN_ROTATION_CLUSTER
members to be corroborated rather than one page's bad fit, and it must sit
DISTINCT_ROTATION_DEG from the volume median to be worth a separate search."""


@dataclass(frozen=True)
class SnapTarget:
    """Where a page is expected on the key map, at what scale, and at what angles.

    The scale is the volume's median over its already-fitted pages. `rotations`
    holds every rotation worth trying, volume median first: most volumes want
    only that, but a volume with sideways-printed pages needs more than one, and
    a single estimator cannot straddle two regimes -- see `rotation_hypotheses`.
    """

    centre: tuple[float, float]
    m_per_px: float
    rotations: tuple[float, ...]

    def span_deg(self) -> float:
        """Widest rotation to allow for when sizing crops and canvases."""
        first = self.rotations[0]
        return max(abs(wrap_deg(r - first)) for r in self.rotations) + 10.0


@dataclass(frozen=True)
class Match:
    """One page placed on the key map."""

    to_keymap: np.ndarray
    score: float
    margin: float
    anisotropy_pct: float
    rotation_deg: float


def thin_plate_spline(
    source: np.ndarray, destination: np.ndarray, smoothing: float | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Smoothed thin-plate spline mapping `source` points onto `destination`.

    `smoothing` resolves at call time rather than being bound as a default, so
    that overriding TPS_SMOOTHING actually changes behaviour -- a default
    argument would freeze the value at import and silently ignore the override.
    """
    if smoothing is None:
        smoothing = TPS_SMOOTHING
    count = len(source)
    squared = ((source[:, None, :] - source[None, :, :]) ** 2).sum(-1)
    kernel = np.where(
        squared > 0, squared * 0.5 * np.log(np.maximum(squared, 1e-12)), 0.0
    ) + smoothing * np.eye(count)
    poly = np.column_stack([np.ones(count), source])
    system = np.zeros((count + 3, count + 3))
    system[:count, :count] = kernel
    system[:count, count:] = poly
    system[count:, :count] = poly.T
    rhs = np.zeros((count + 3, 2))
    rhs[:count] = destination
    solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
    weights, affine_part = solution[:count], solution[count:]

    def evaluate(query: np.ndarray) -> np.ndarray:
        squared_q = ((query[:, None, :] - source[None, :, :]) ** 2).sum(-1)
        kernel_q = np.where(
            squared_q > 0, squared_q * 0.5 * np.log(np.maximum(squared_q, 1e-12)), 0.0
        )
        return (
            kernel_q @ weights
            + np.column_stack([np.ones(len(query)), query]) @ affine_part
        )

    return evaluate


def consistent_gcps(inliers: list[dict]) -> list[dict]:
    """Drop GCPs that put one key-map pixel in two different places.

    A pixel matched to two OSM intersections more than GCP_CONFLICT_FT apart is
    a contradiction the spline cannot resolve -- it splits the difference and
    bends the neighbourhood to do it. Dropping the whole group is deliberate:
    there is no evidence here for which reading is right, and keeping either at
    random is how a plausible-looking fit ends up in the wrong street.
    """
    groups: dict[tuple[float, float], list[dict]] = {}
    for gcp in inliers:
        groups.setdefault((round(gcp["x"], 1), round(gcp["y"], 1)), []).append(gcp)

    kept = []
    for readings in groups.values():
        if len(readings) == 1:
            kept.append(readings[0])
            continue
        lons = [r["lon"] for r in readings]
        lats = [r["lat"] for r in readings]
        lon_scale = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lats[0]))
        spread = (
            math.hypot(
                (max(lons) - min(lons)) * lon_scale,
                (max(lats) - min(lats)) * M_PER_DEG_LAT,
            )
            * FEET_PER_METRE
        )
        if spread <= GCP_CONFLICT_FT:
            kept.append(readings[0])
    return kept


def keymap_model(
    georef: dict,
    min_gcps: int = MIN_KEYMAP_INLIERS,
    smoothing: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Key-map pixel -> lon/lat, as a spline through the sheet's own GCPs.

    A key map's shipped affine is a global fit; its residuals against the
    sheet's inlier intersections run to ~100 ft on Detroit, and that error is
    the floor under every page placed through it. A spline through those same
    intersections follows the sheet. Falls back to the corner affine when the
    sheet has too few of them to constrain a warp.
    """
    inliers = consistent_gcps(
        [i for i in georef.get("intersections", []) if i.get("inlier")]
    )
    affine = page_world_affine(georef)
    if len(inliers) < min_gcps:
        return lambda query: (
            np.column_stack([query[:, 0], query[:, 1], np.ones(len(query))]) @ affine.T
        )

    pixels = np.array([[i["x"], i["y"]] for i in inliers], float)
    lonlat = np.array([[i["lon"], i["lat"]] for i in inliers], float)
    origin = lonlat[0]
    lon_scale = M_PER_DEG_LON_EQUATOR * math.cos(
        math.radians(float(lonlat[:, 1].mean()))
    )
    metres = np.column_stack(
        [
            (lonlat[:, 0] - origin[0]) * lon_scale,
            (lonlat[:, 1] - origin[1]) * M_PER_DEG_LAT,
        ]
    )
    spline = thin_plate_spline(pixels, metres, smoothing=smoothing)

    def to_world(query: np.ndarray) -> np.ndarray:
        out = spline(query)
        return np.column_stack(
            [out[:, 0] / lon_scale + origin[0], out[:, 1] / M_PER_DEG_LAT + origin[1]]
        )

    return to_world


def local_tangent(
    model: Callable[[np.ndarray], np.ndarray],
    centre: tuple[float, float],
    step: float = 200.0,
) -> np.ndarray:
    """2x3 key-map-px -> lon/lat affine tangent to `model` at `centre`."""
    probe = np.array(
        [centre, [centre[0] + step, centre[1]], [centre[0], centre[1] + step]], float
    )
    world = model(probe)
    along_x = (world[1] - world[0]) / step
    along_y = (world[2] - world[0]) / step
    return np.array(
        [[along_x[0], along_y[0], world[0, 0]], [along_x[1], along_y[1], world[0, 1]]]
    )


def affine_m_per_px(affine: np.ndarray) -> float:
    """Ground metres per pixel implied by a pixel -> lon/lat affine."""
    return math.hypot(affine[1, 0], affine[1, 1]) * M_PER_DEG_LAT


def wrap_deg(delta: float) -> float:
    """An angle difference folded into (-180, 180]."""
    return (delta + 180.0) % 360.0 - 180.0


def metric_theta(affine: np.ndarray) -> float:
    """Ground rotation in degrees of a pixel -> lon/lat affine.

    Reading the angle off the lon/lat coefficients gets two things wrong. A
    degree of longitude is cos(latitude) of a degree of latitude, so the raw
    coefficients are in mismatched units -- 0.74 at Detroit, worth ~6 degrees.
    And ``atan2`` of a single column only equals the rotation for a similarity;
    once the map carries anisotropy, which key-map georeferences do by ~10%, it
    picks up an error that depends on the rotation being measured.

    Both cancel while every page in a volume sits at one angle, and neither
    cancels when comparing pages printed at different orientations. So: convert
    to metres, undo the y-down/north-up flip, and take the rotation from the
    polar decomposition, which is the rotation whatever else the map is doing.
    """
    linear = np.diag([1.0, -1.0]) @ linear_part_metres(affine, float(affine[1, 2]))
    eigenvalues, eigenvectors = np.linalg.eigh(linear.T @ linear)
    stretch = (
        eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
    )
    rotation = linear @ np.linalg.inv(stretch)
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def rotated_extent(width: int, height: int, theta_deg: float) -> tuple[int, int]:
    """Bounding box of a width x height rectangle rotated by theta.

    A page rotated inside its own unrotated bounds loses its corners; a page
    printed sideways loses most of itself. The template canvas grows to fit.
    """
    cos = abs(math.cos(math.radians(theta_deg)))
    sin = abs(math.sin(math.radians(theta_deg)))
    return int(width * cos + height * sin), int(width * sin + height * cos)


def linear_part_metres(affine: np.ndarray, latitude: float) -> np.ndarray:
    """The 2x2 linear part of a pixel -> lon/lat affine, in metres per pixel."""
    lon_scale = M_PER_DEG_LON_EQUATOR * math.cos(math.radians(latitude))
    return np.array(
        [
            [affine[0, 0] * lon_scale, affine[0, 1] * lon_scale],
            [affine[1, 0] * M_PER_DEG_LAT, affine[1, 1] * M_PER_DEG_LAT],
        ]
    )


def unit_stretch(linear: np.ndarray) -> tuple[np.ndarray, float]:
    """The shape distortion in `linear`, as a unit-determinant stretch.

    Polar decomposition splits a linear map into a rotation and a symmetric
    stretch, ``L = R S``. Only S is the problem: the matcher searches rotations
    already, but not stretches. Normalizing S to unit determinant isolates the
    shape without touching overall scale, so warping by it neither resizes nor
    flips what it is applied to.

    Returns (stretch, anisotropy percent).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(linear.T @ linear)
    stretch = (
        eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
    )
    determinant = float(np.linalg.det(stretch))
    singular = np.linalg.svd(linear, compute_uv=False)
    anisotropy = 100.0 * (singular[0] / max(singular[1], 1e-12) - 1.0)
    return stretch / math.sqrt(max(determinant, 1e-12)), float(anisotropy)


def as_3x3(matrix: np.ndarray) -> np.ndarray:
    """Promote a 2x3 affine to a 3x3 homogeneous matrix."""
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class StretchedCrop:
    """A key-map crop warped into the frame where a correct page is isotropic."""

    image: np.ndarray
    zero_centred: np.ndarray
    centre: tuple[float, float]
    m_per_px: float
    to_keymap: np.ndarray
    anisotropy_pct: float


def stretched_crop(
    probability: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    target: SnapTarget,
    page_shape: tuple[int, int],
) -> StretchedCrop | None:
    """Cut the key map around where a page belongs and undo the sheet's stretch.

    Returns None when the region falls off the sheet or is too small to match.
    """
    height, width = page_shape
    centre_x, centre_y = target.centre
    tangent = local_tangent(model, target.centre)
    stretch, anisotropy = unit_stretch(
        linear_part_metres(tangent, latitude=float(tangent[1, 2]))
    )

    nominal = affine_m_per_px(tangent)
    if nominal <= 0:
        return None
    expected = target.m_per_px / nominal
    span_x, span_y = rotated_extent(
        int(width * expected), int(height * expected), target.span_deg()
    )
    x0 = max(0, int(centre_x - span_x / 2 - SEARCH_SLACK_PX))
    x1 = min(probability.shape[1], int(centre_x + span_x / 2 + SEARCH_SLACK_PX))
    y0 = max(0, int(centre_y - span_y / 2 - SEARCH_SLACK_PX))
    y1 = min(probability.shape[0], int(centre_y + span_y / 2 + SEARCH_SLACK_PX))
    crop = probability[y0:y1, x0:x1]
    if crop.size == 0 or min(crop.shape) < 60:
        return None

    crop_h, crop_w = crop.shape
    out_w, out_h = int(crop_w * STRETCH_PAD), int(crop_h * STRETCH_PAD)
    source_centre = np.array([crop_w / 2.0, crop_h / 2.0])
    target_centre = np.array([out_w / 2.0, out_h / 2.0])
    warp = np.column_stack([stretch, target_centre - stretch @ source_centre])
    warped = cv2.warpAffine(crop, warp, (out_w, out_h))

    crop_to_keymap = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0]])
    to_keymap = (as_3x3(crop_to_keymap) @ np.linalg.inv(as_3x3(warp)))[:2, :]
    warped_georef = (as_3x3(tangent) @ as_3x3(to_keymap))[:2, :]

    blurred = cv2.GaussianBlur(warped, (0, 0), 2.0)
    moved_centre = warp[:, :2] @ np.array([centre_x - x0, centre_y - y0]) + warp[:, 2]
    return StretchedCrop(
        image=warped,
        zero_centred=blurred - cv2.blur(blurred, (91, 91)),
        centre=(float(moved_centre[0]), float(moved_centre[1])),
        m_per_px=affine_m_per_px(warped_georef),
        to_keymap=to_keymap,
        anisotropy_pct=anisotropy,
    )


def template_placement(
    page_shape: tuple[int, int], factor: float, cv_angle: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """page px -> WARPED-CROP px for a template at `cv_angle`.

    Deliberately stops at crop coordinates. The match's translation is found in
    that frame, so it has to be applied there too -- composing it after the
    crop-to-key-map step instead puts the shift through the stretch and the crop
    offset, which is a silent, page-dependent error that leaves rotation looking
    correct while every placement drifts.
    """
    height, width = page_shape
    template_w, template_h = int(width * factor), int(height * factor)
    rotation = cv2.getRotationMatrix2D(
        (template_w / 2.0, template_h / 2.0), cv_angle, 1.0
    )
    canvas = rotated_extent(template_w, template_h, cv_angle)
    rotation[0, 2] += canvas[0] / 2.0 - template_w / 2.0
    rotation[1, 2] += canvas[1] / 2.0 - template_h / 2.0
    placed = as_3x3(rotation) @ np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1.0]])
    return placed, rotation, canvas


def angle_calibration(
    crop: StretchedCrop,
    model: Callable[[np.ndarray], np.ndarray],
    page_shape: tuple[int, int],
    factor: float,
) -> Callable[[float], float]:
    """Map a wanted ground rotation to the cv2 template angle that achieves it.

    Deriving this by hand is where the sign errors live: the template is rotated
    in a y-down image, composed through a stretch-corrected crop, and read out in
    a north-up world, and getting any of those backwards produces an angle that
    is wrong by a sign or an offset. It stays invisible as long as every page
    sits at the volume median, because the error is then absorbed by the jitter.

    So measure it instead. The relation is linear with slope +/-1, so two probes
    determine it, and the slope is a free self-check.
    """
    probes = []
    for probe in (0.0, 20.0):
        placed, _, _ = template_placement(page_shape, factor, probe)
        to_keymap = (as_3x3(crop.to_keymap) @ placed)[:2, :]
        probes.append(
            metric_theta(
                page_world_affine_from_match(
                    Match(to_keymap, 0.0, 0.0, 0.0, 0.0), model, page_shape
                )
            )
        )
    slope = wrap_deg(probes[1] - probes[0]) / 20.0
    if abs(abs(slope) - 1.0) > 0.05:
        print(
            f"warning: template angle maps to output angle with slope {slope:.3f}, "
            "expected +/-1; the stretch correction may not be holding here",
            file=sys.stderr,
        )
    return lambda wanted: wrap_deg(wanted - probes[0]) / slope


def match_page(
    crop: StretchedCrop,
    probability: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    target: SnapTarget,
) -> Match | None:
    """Slide a page's P(road) over a stretched key-map crop; best pose or None.

    Every rotation in `target.rotations` is tried and the image evidence picks
    the winner, rather than one prior being chosen up front.
    """
    height, width = probability.shape
    if crop.m_per_px <= 0:
        return None
    base = target.m_per_px / crop.m_per_px
    best: Match | None = None

    for rung in SCALE_LADDER:
        factor = base * rung
        template_w, template_h = int(width * factor), int(height * factor)
        if template_w < 8 or template_h < 8:
            continue
        resized = cv2.GaussianBlur(
            cv2.resize(
                probability, (template_w, template_h), interpolation=cv2.INTER_AREA
            ),
            (0, 0),
            2.0,
        )
        to_cv_angle = angle_calibration(crop, model, probability.shape, factor)
        for wanted in target.rotations:
            for jitter in THETA_JITTER_DEG:
                cv_angle = to_cv_angle(wanted + jitter)
                placed, rotation, canvas = template_placement(
                    probability.shape, factor, cv_angle
                )
                rotated = cv2.warpAffine(resized, rotation, canvas)
                mass = float(rotated.sum())
                if mass < 1e-3:
                    continue
                if (
                    rotated.shape[0] >= crop.image.shape[0] - 2
                    or rotated.shape[1] >= crop.image.shape[1] - 2
                ):
                    continue

                response = (
                    cv2.matchTemplate(crop.zero_centred, rotated, cv2.TM_CCORR) / mass
                )
                rows, cols = np.mgrid[0 : response.shape[0], 0 : response.shape[1]]
                offset_x = cols + canvas[0] / 2.0 - crop.centre[0]
                offset_y = rows + canvas[1] / 2.0 - crop.centre[1]
                weighted = response * np.exp(
                    -0.5 * (offset_x**2 + offset_y**2) / PRIOR_SIGMA_PX**2
                )
                _, peak, _, location = cv2.minMaxLoc(weighted)
                if not math.isfinite(peak) or (best is not None and peak <= best.score):
                    continue

                suppressed = weighted.copy()
                y0 = max(0, location[1] - PEAK_SUPPRESS_PX)
                y1 = min(suppressed.shape[0], location[1] + PEAK_SUPPRESS_PX)
                x0 = max(0, location[0] - PEAK_SUPPRESS_PX)
                x1 = min(suppressed.shape[1], location[0] + PEAK_SUPPRESS_PX)
                suppressed[y0:y1, x0:x1] = -1e9
                runner_up = float(suppressed.max())

                shift = np.array(
                    [[1.0, 0, location[0]], [0, 1.0, location[1]], [0, 0, 1.0]]
                )
                best = Match(
                    to_keymap=(as_3x3(crop.to_keymap) @ shift @ placed)[:2, :],
                    score=float(peak),
                    margin=float((peak - runner_up) / max(abs(peak), 1e-9)),
                    anisotropy_pct=crop.anisotropy_pct,
                    rotation_deg=wanted + jitter,
                )
    return best


def page_world_affine_from_match(
    match: Match,
    model: Callable[[np.ndarray], np.ndarray],
    page_shape: tuple[int, int],
) -> np.ndarray:
    """2x3 page-px -> lon/lat affine for a matched page.

    The key-map model is a spline, so the composition is not affine; over one
    page footprint it is near enough that a least-squares affine is what a
    downstream consumer effectively sees, and georef sidecars store corners.
    """
    height, width = page_shape
    grid_x, grid_y = np.meshgrid(np.linspace(0, width, 6), np.linspace(0, height, 6))
    page_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    keymap_points = np.column_stack(
        [
            match.to_keymap[0, 0] * page_points[:, 0]
            + match.to_keymap[0, 1] * page_points[:, 1]
            + match.to_keymap[0, 2],
            match.to_keymap[1, 0] * page_points[:, 0]
            + match.to_keymap[1, 1] * page_points[:, 1]
            + match.to_keymap[1, 2],
        ]
    )
    design = np.column_stack(
        [page_points[:, 0], page_points[:, 1], np.ones(len(page_points))]
    )
    coefficients, *_ = np.linalg.lstsq(design, model(keymap_points), rcond=None)
    return coefficients.T


def affine_corners(
    affine: np.ndarray, page_shape: tuple[int, int]
) -> list[list[float]]:
    """The four page corners in lon/lat, in georef sidecar order."""
    height, width = page_shape
    return [
        [
            float(affine[0, 0] * x + affine[0, 1] * y + affine[0, 2]),
            float(affine[1, 0] * x + affine[1, 1] * y + affine[1, 2]),
        ]
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]


def volume_pose_medians(volume: Path) -> tuple[float, float] | None:
    """Median scale and rotation over a volume's already-fitted pages."""
    scales, thetas = [], []
    for page in sorted(volume.glob("p*.jpg")):
        if "__" in page.stem:
            continue
        for channel in ("georef-street", "georef-snap", "georef"):
            path = volume / f"{page.stem}.{channel}.json"
            if path.exists():
                affine = page_world_affine(json.loads(path.read_text()))
                scales.append(affine_m_per_px(affine))
                thetas.append(metric_theta(affine))
                break
    if not scales:
        return None
    return float(np.median(scales)), float(np.median(thetas))


def page_number(stem: str) -> int | None:
    """The integer sheet number in a stem like 'p12', or None if it has none.

    Not every sheet has one. Detroit's are all plain integers, but Miami and DC
    carry two kinds that are not: skeleton sheets (p10s, p133s), which map the
    same ground as their full-colour sibling, and genuine lettered sub-sheets
    (DC's p1a through p1d). Only the first kind is redundant, and which of a
    pN/pNs pair counts is already decided downstream by
    ``compare_iiif_georef.redundant_skeleton_keys`` -- so placement need not
    treat them specially, it only has to stop assuming every stem parses.
    """
    digits = stem[1:]
    return int(digits) if digits.isdigit() else None


def published_rotations(volume: Path) -> dict[int, float]:
    """Ground rotation of each page's own published fit, by page number."""
    rotations = {}
    for page in sorted(volume.glob("p*.jpg")):
        number = page_number(page.stem)
        if "__" in page.stem or number is None:
            continue
        for channel in ("georef-street", "georef-snap", "georef"):
            path = volume / f"{page.stem}.{channel}.json"
            if path.exists():
                rotations[number] = metric_theta(
                    page_world_affine(json.loads(path.read_text()))
                )
                break
    return rotations


def cluster_rotations(angles: list[float]) -> list[tuple[float, int]]:
    """Group angles into (centre, size) clusters, largest first."""
    clusters: list[list[float]] = []
    for angle in sorted(angles):
        for cluster in clusters:
            if abs(wrap_deg(angle - cluster[0])) <= ROTATION_CLUSTER_DEG:
                cluster.append(angle)
                break
        else:
            clusters.append([angle])
    return sorted(
        ((float(np.median(c)), len(c)) for c in clusters), key=lambda c: -c[1]
    )


def rotation_hypotheses(
    volume: Path, volume_theta: float
) -> dict[int, tuple[float, ...]]:
    """Rotations worth trying for each page: the volume median, plus neighbours'.

    Most volumes print every page at one angle and the median is the whole
    story. Some do not -- Detroit has four sideways sheets at +56 degrees
    against a -26 median, unreachable by a +/-3 degree ladder and 600-920 ft
    wrong because of it.

    Their neighbours know better, but taking the neighbour MEDIAN as the prior
    is worse than useless: pages on the boundary of such a block have a mixed
    neighbourhood, and its median is an angle no page actually has. Measured on
    Detroit, that swap fixes two pages and breaks three.

    So neighbours only ever ADD a hypothesis. A distinct, corroborated rotation
    (at least `MIN_ROTATION_CLUSTER` neighbours agreeing, far enough from the
    median to be a different regime) gets offered alongside the median, and the
    match score decides on the image evidence.
    """
    from mapsnap.edge_join_experiment import keymap_region_adjacency

    pairs, _ = keymap_region_adjacency(volume)
    neighbours: dict[int, set[int]] = {}
    for pair in pairs:
        first, second = tuple(pair)
        neighbours.setdefault(first, set()).add(second)
        neighbours.setdefault(second, set()).add(first)

    rotations = published_rotations(volume)
    hypotheses = {}
    for number, adjacent in neighbours.items():
        known = [rotations[n] for n in adjacent if n in rotations]
        extra = [
            centre
            for centre, size in cluster_rotations(known)
            if size >= MIN_ROTATION_CLUSTER
            and abs(wrap_deg(centre - volume_theta)) > DISTINCT_ROTATION_DEG
        ]
        if extra:
            hypotheses[number] = tuple(extra)
    return hypotheses


def snap_volume(
    volume: Path,
    output_dir: Path,
    min_margin: float = 0.0,
    smoothing: float | None = None,
) -> list[dict]:
    """Place every page that has a key-map region and a P(road) map.

    Writes one georef sidecar per placed page into `output_dir` and returns a
    row per page for reporting.
    """
    from mapsnap.edge_join_experiment import load_prob

    medians = volume_pose_medians(volume)
    if medians is None:
        print(f"{volume.name}: no fitted pages to take a scale from", file=sys.stderr)
        return []
    volume_m_per_px, volume_theta = medians

    hypotheses = rotation_hypotheses(volume, volume_theta)

    def rotations_for(stem: str) -> tuple[float, ...]:
        """Extra rotation hypotheses for a page, if its neighbours suggest any.

        Lettered sheets have no number to look up, so they simply get the
        volume median -- the same answer they got before neighbours existed.
        """
        number = page_number(stem)
        return () if number is None else hypotheses.get(number, ())

    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for georef_path in sorted((volume / "raw").glob("*.georef.json")):
        if ".truth." in georef_path.name:
            continue
        stem = georef_path.name[: -len(".georef.json")]
        regions_path = volume / "raw" / f"{stem}.regions.panels.json"
        prob_path = volume / "raw" / f"{stem}.roadprob.png"
        if not regions_path.exists() or not prob_path.exists():
            continue

        keymap_image = cv2.imread(str(prob_path), cv2.IMREAD_GRAYSCALE)
        if keymap_image is None:
            print(f"{stem}: unreadable P(road) map, skipping", file=sys.stderr)
            continue

        georef = json.loads(georef_path.read_text())
        model = keymap_model(georef, smoothing=smoothing)
        keymap_prob = keymap_image.astype(np.float32) / 255.0
        regions = json.loads(regions_path.read_text())
        labels = [str(label) for label in regions["labels"]]

        for page in sorted(volume.glob("p*.jpg")):
            if "__" in page.stem or page.stem[1:] not in labels:
                continue
            probability = load_prob(volume, page.stem)
            if probability is None:
                continue
            panel = np.asarray(regions["panels"][labels.index(page.stem[1:])], float)
            target = SnapTarget(
                centre=(float(panel[:, 0].mean()), float(panel[:, 1].mean())),
                m_per_px=volume_m_per_px,
                rotations=(volume_theta, *rotations_for(page.stem)),
            )
            crop = stretched_crop(keymap_prob, model, target, probability.shape)
            if crop is None:
                continue
            match = match_page(crop, probability, model, target)
            if match is None or match.margin < min_margin:
                continue

            affine = page_world_affine_from_match(match, model, probability.shape)
            (output_dir / f"{page.stem}.georef.json").write_text(
                json.dumps(
                    {
                        "width": probability.shape[1],
                        "height": probability.shape[0],
                        "corners": affine_corners(affine, probability.shape),
                        "intersections": [],
                        "source": f"keymap-snap from {stem} (#211)",
                        "match_margin": round(match.margin, 3),
                        "keymap_anisotropy_pct": round(match.anisotropy_pct, 2),
                        "rotation_deg": round(match.rotation_deg, 2),
                    },
                    indent=2,
                )
            )
            rows.append(
                {
                    "stem": page.stem,
                    "keymap": stem,
                    "margin": match.margin,
                    "anisotropy_pct": match.anisotropy_pct,
                    "off_median_rotation": abs(
                        wrap_deg(match.rotation_deg - volume_theta)
                    )
                    > DISTINCT_ROTATION_DEG,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place pages by matching their P(road) map to the key map's."
    )
    parser.add_argument("volume", type=Path, help="Volume directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write georef sidecars (default: <volume>/kmsnap)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help=(
            "Key-map spline regularization (default "
            f"{TPS_SMOOTHING:.0e}). Lower follows the sheet's intersections more "
            "closely; on Detroit that chases their noise and doubles the disasters."
        ),
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Discard matches whose peak barely beats the runner-up",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.volume / "kmsnap")
    rows = snap_volume(
        args.volume,
        output_dir,
        min_margin=args.min_margin,
        smoothing=args.smoothing,
    )
    if not rows:
        print("No pages placed.", file=sys.stderr)
        return
    anisotropy = float(np.median([row["anisotropy_pct"] for row in rows]))
    margins = sorted(row["margin"] for row in rows)
    off_median = sum(row["off_median_rotation"] for row in rows)
    print(
        f"Placed {len(rows)} page(s) into {output_dir}. "
        f"Median match margin {margins[len(margins) // 2]:.3f}; "
        f"median key-map anisotropy {anisotropy:.1f}% (corrected); "
        f"{off_median} page(s) placed at a neighbour-suggested rotation."
    )


if __name__ == "__main__":
    main()
