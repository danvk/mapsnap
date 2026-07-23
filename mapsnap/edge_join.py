"""Road-mask edge-join matcher (truth-free).

Estimates a target page's pose from an anchored neighbor by matching road-UNet
P(road) maps across their shared boundary street. Works entirely in a pair
raster frame (metres, north-up rows): the anchor is pre-rendered there by the
caller, and the target's pose is a similarity at a FIXED, caller-supplied scale
— 3 DOF (rotation + translation). Page pixels map to the raster without a
reflection (the page's y-down and the raster's row-down cancel), so a plain
rotation suffices.

Pipeline: dominant-orientation alignment proposes ~4 rotation candidates
(mod-90 offset + k*90); masked FFT normalized cross-correlation of blurred
probability maps proposes top-K translations per rotation (global over the
search window, so lattice aliasing surfaces as competing peaks rather than a
silent wrong lock); chamfer least-squares on the road skeleton polishes each
candidate and yields truth-free diagnostics (inlier fraction, mean residual,
conditioning of JtJ for along-corridor degeneracy).
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

MIN_VALID = 0.5  # validity-mask threshold after warping
CHAMFER_CLAMP_M = 30.0
INLIER_M = 5.0
MIN_STRIP_POINTS = 500  # skeleton points of strip evidence a join must have
SCALE_PRIOR_SIGMA = 0.05  # prior on log-scale deviation from the volume median


@dataclass
class JoinCandidate:
    """One candidate target pose with matcher diagnostics (all truth-free)."""

    pose: np.ndarray  # 2x3, target page px -> pair raster px
    theta_deg: float
    ncc: float
    chamfer_mean_m: float = math.inf
    inlier_frac: float = 0.0
    n_points: int = 0
    jtj_min_eig: float = 0.0
    jtj_eig_ratio: float = 0.0
    overlap_frac: float = 0.0
    ncc_fine: float = 0.0
    scale_adjust: float = 1.0
    plausible: bool = True
    refined: bool = False
    diagnostics: dict = field(default_factory=dict)

    def verification_score(self) -> float:
        """Heuristic quality combining strip agreement signals.

        - inlier_frac and chamfer measure skeleton alignment near the anchor;
        - ncc_fine (lightly blurred correlation at the refined pose) separates
          the true join from 180-degree/lattice locks that align corridors but
          not the actual drawn content;
        - poses with near-total page overlap (physically impossible) or too
          little strip evidence (sliver contacts overfit) are disqualified.
        """
        if not self.refined or not self.plausible or self.n_points < MIN_STRIP_POINTS:
            return -math.inf
        return self.inlier_frac + self.ncc_fine - self.chamfer_mean_m / CHAMFER_CLAMP_M


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 2x3 affine applying b first, then a."""
    return a @ np.vstack([b, [0.0, 0.0, 1.0]])


@dataclass
class FrameSpec:
    """A pair-local raster frame: equirectangular metres, north-up rows.

    col = (x - x_min) / res, row = (y_max - y) / res, where (x, y) are metres
    east/north of `origin`. Page images map into the raster without reflection.
    """

    origin: tuple[float, float]  # lon, lat
    x_min: float
    y_max: float
    res_m: float
    shape: tuple[int, int]  # rows, cols

    def metre_scales(self) -> tuple[float, float]:
        kx = 111_320.0 * math.cos(math.radians(self.origin[1]))
        return kx, 110_540.0

    def lonlat_to_raster(self, lon: float, lat: float) -> tuple[float, float]:
        kx, ky = self.metre_scales()
        x = (lon - self.origin[0]) * kx
        y = (lat - self.origin[1]) * ky
        return (x - self.x_min) / self.res_m, (self.y_max - y) / self.res_m

    def page_to_raster_affine(self, affine_local: np.ndarray) -> np.ndarray:
        """2x3 mapping page px -> raster px given a page px -> lon/lat affine."""
        kx, ky = self.metre_scales()
        a = affine_local
        metres = np.array(
            [
                [a[0, 0] * kx, a[0, 1] * kx, (a[0, 2] - self.origin[0]) * kx],
                [a[1, 0] * ky, a[1, 1] * ky, (a[1, 2] - self.origin[1]) * ky],
            ]
        )
        to_frame = np.array(
            [
                [1 / self.res_m, 0, -self.x_min / self.res_m],
                [0, -1 / self.res_m, self.y_max / self.res_m],
            ]
        )
        return compose(to_frame, metres)

    def raster_pose_to_world_affine(self, pose: np.ndarray) -> np.ndarray:
        """2x3 mapping page px -> (lon, lat) given a page px -> raster px pose."""
        kx, ky = self.metre_scales()
        from_raster = np.array(
            [
                [self.res_m / kx, 0, self.origin[0] + self.x_min / kx],
                [0, -self.res_m / ky, self.origin[1] + self.y_max / ky],
            ]
        )
        return compose(from_raster, pose)


def dominant_orientation_deg(
    prob: np.ndarray, valid: np.ndarray | None = None, border_px: int = 12
) -> float:
    """Dominant road direction (array-frame atan2 angle) folded to [0, 90).

    Gradient orientations are folded mod 90 so a rectangular grid votes for one
    angle. Gradients near the image border (or near the edge of `valid`) are
    excluded — the cut edge of a warped page otherwise votes for the frame axes.
    """
    blurred = cv2.GaussianBlur(prob, (0, 0), 3)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    magnitude = gx * gx + gy * gy
    interior = (
        np.ones(prob.shape, np.uint8) if valid is None else valid.astype(np.uint8)
    )
    kernel = np.ones((2 * border_px + 1, 2 * border_px + 1), np.uint8)
    magnitude *= cv2.erode(interior, kernel).astype(np.float32)
    # Angle of the gradient, quadrupled to fold mod 90 degrees.
    angle = np.arctan2(gy, gx)
    fold = np.exp(4j * angle)
    mean = (fold * magnitude).sum() / max(magnitude.sum(), 1e-9)
    if abs(mean) < 1e-9:
        return 0.0
    corridor = math.degrees(np.angle(mean) / 4) + 90.0
    return corridor % 90.0


def rotation_candidates(
    fixed_prob: np.ndarray,
    target_prob: np.ndarray,
    jitter_deg: tuple[float, ...] = (0.0,),
    fixed_valid: np.ndarray | None = None,
) -> list[float]:
    """cv2-convention rotations aligning the target's road grid to the frame's.

    The array-frame rotation mapping target directions onto fixed directions is
    fixed_dir - target_dir (mod 90); cv2.getRotationMatrix2D uses the opposite
    sign, so candidates are -(delta) + k*90 (+ jitter) for k in 0..3.
    """
    fixed_dir = dominant_orientation_deg(fixed_prob, fixed_valid)
    target_dir = dominant_orientation_deg(target_prob)
    delta = fixed_dir - target_dir
    return [
        (-delta + 90.0 * k + j + 180.0) % 360.0 - 180.0
        for k in range(4)
        for j in jitter_deg
    ]


def masked_ncc(
    fixed: np.ndarray,
    fixed_mask: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    *,
    min_overlap_px: float,
    max_overlap_px: float = math.inf,
) -> np.ndarray:
    """Masked normalized cross-correlation (Padfield) of moving over fixed.

    Returns the full-mode NCC map; entry (i, j) scores placing moving's origin
    at fixed-frame position (i - mh + 1, j - mw + 1). Shifts whose mask overlap
    falls outside [min_overlap_px, max_overlap_px] score 0 — adjacent Sanborn
    sheets only share a margin strip, so near-total overlap is physically
    impossible and would otherwise dominate on a self-similar street grid.
    """
    f = (fixed * fixed_mask).astype(np.float64)
    g = (moving * moving_mask).astype(np.float64)
    mf = fixed_mask.astype(np.float64)
    mg = moving_mask.astype(np.float64)

    def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return fftconvolve(a, b[::-1, ::-1], mode="full")

    overlap = corr(mf, mg)
    sum_fg = corr(f, g)
    sum_f = corr(f, mg)
    sum_g = corr(mf, g)
    sum_ff = corr(f * fixed, mg)
    sum_gg = corr(mf, g * moving)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.maximum(overlap, 1e-6)
        numerator = sum_fg - sum_f * sum_g / n
        var_f = np.maximum(sum_ff - sum_f**2 / n, 0)
        var_g = np.maximum(sum_gg - sum_g**2 / n, 0)
        ncc = numerator / np.sqrt(np.maximum(var_f * var_g, 1e-12))
    ncc[(overlap < min_overlap_px) | (overlap > max_overlap_px)] = 0.0
    return np.clip(np.nan_to_num(ncc), -1.0, 1.0)


def top_peaks(
    score: np.ndarray, count: int, min_separation_px: int
) -> list[tuple[float, int, int]]:
    """Up to `count` local maxima of a score map, greedily non-overlapping."""
    peaks: list[tuple[float, int, int]] = []
    working = score.copy()
    for _ in range(count):
        idx = int(np.argmax(working))
        row, col = np.unravel_index(idx, working.shape)
        value = float(working[row, col])
        if value <= 0:
            break
        peaks.append((value, int(row), int(col)))
        r0 = max(0, row - min_separation_px)
        c0 = max(0, col - min_separation_px)
        working[r0 : row + min_separation_px + 1, c0 : col + min_separation_px + 1] = -1
    return peaks


def skeleton_points(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """(N, 2) x,y skeleton points of the thresholded road mask, in page pixels."""
    from mapsnap.road_model import road_mask, road_skeleton

    mask = road_mask(prob, threshold=threshold, min_area=min_area)
    skeleton = road_skeleton(mask)
    ys, xs = np.nonzero(skeleton)
    return np.column_stack([xs, ys]).astype(np.float64)


def chamfer_refine(
    distance_m: np.ndarray,
    points_page: np.ndarray,
    pose: np.ndarray,
    max_points: int = 3000,
    huber_m: float = 6.0,
    region: np.ndarray | None = None,
    solve_scale: bool = False,
) -> tuple[np.ndarray, dict]:
    """Polish a pose by minimizing skeleton-to-anchor chamfer distance.

    distance_m is the anchor-skeleton distance transform over the pair raster
    (metres per raster cell already applied). Optimizes (dtheta, dx, dy) around
    `pose`; returns the refined pose and diagnostics.

    When `region` (a bool raster, e.g. the anchor's dilated validity) is given,
    only skeleton points landing inside it at the initial pose are scored —
    points far from the anchor otherwise sit at the distance clamp and reward
    poses that maximize page overlap instead of aligning the shared strip.
    """
    if region is not None and len(points_page):
        initial = np.column_stack([points_page, np.ones(len(points_page))]) @ pose.T
        rows = np.clip(initial[:, 1].round().astype(int), 0, region.shape[0] - 1)
        cols = np.clip(initial[:, 0].round().astype(int), 0, region.shape[1] - 1)
        points_page = points_page[region[rows, cols]]
    if len(points_page) < 10:
        return pose.copy(), {
            "chamfer_mean_m": math.inf,
            "inlier_frac": 0.0,
            "n_points": int(len(points_page)),
            "jtj_min_eig": 0.0,
            "jtj_eig_ratio": 0.0,
            "scale_adjust": 1.0,
        }
    if len(points_page) > max_points:
        step = len(points_page) // max_points + 1
        points_page = points_page[::step]
    homogeneous = np.column_stack([points_page, np.ones(len(points_page))])

    height, width = distance_m.shape

    def sample(pts: np.ndarray) -> np.ndarray:
        x = np.clip(pts[:, 0], 0, width - 1.001)
        y = np.clip(pts[:, 1], 0, height - 1.001)
        x0 = x.astype(int)
        y0 = y.astype(int)
        fx = x - x0
        fy = y - y0
        d = (
            distance_m[y0, x0] * (1 - fx) * (1 - fy)
            + distance_m[y0, x0 + 1] * fx * (1 - fy)
            + distance_m[y0 + 1, x0] * (1 - fx) * fy
            + distance_m[y0 + 1, x0 + 1] * fx * fy
        )
        return d

    base = pose.copy()
    anchor_pts = homogeneous @ base.T
    center = anchor_pts.mean(axis=0)
    n_params = 4 if solve_scale else 3
    # The scale prior enters as extra residuals so a 1-sigma deviation costs
    # like huber_m on a tenth of the points.
    n_prior = max(len(homogeneous) // 10, 1)

    def delta_matrix(params: np.ndarray) -> np.ndarray:
        dtheta = params[0]
        scale = math.exp(params[3]) if solve_scale else 1.0
        rot = cv2.getRotationMatrix2D(
            (float(center[0]), float(center[1])), math.degrees(dtheta), scale
        )
        rot[:, 2] += [params[1], params[2]]
        return rot

    def transformed(params: np.ndarray) -> np.ndarray:
        rot = delta_matrix(params)
        return anchor_pts @ rot[:, :2].T + rot[:, 2]

    def residuals(params: np.ndarray) -> np.ndarray:
        r = sample(transformed(params))
        if solve_scale:
            prior = params[3] / SCALE_PRIOR_SIGMA * huber_m
            r = np.append(r, np.full(n_prior, prior))
        return r

    result = least_squares(
        residuals,
        x0=np.zeros(n_params),
        loss="huber",
        f_scale=huber_m,
        diff_step=[1e-4, 0.25, 0.25, 1e-4][:n_params],
        max_nfev=60,
    )
    final = sample(transformed(result.x))
    inliers = final < INLIER_M
    jtj = result.jac.T @ result.jac
    eigenvalues = np.linalg.eigvalsh(jtj)
    refined = compose(delta_matrix(result.x), base)
    diagnostics = {
        "chamfer_mean_m": float(final.mean()),
        "inlier_frac": float(inliers.mean()),
        "n_points": int(len(final)),
        "jtj_min_eig": float(eigenvalues[0]),
        "jtj_eig_ratio": float(eigenvalues[0] / max(eigenvalues[-1], 1e-12)),
        "scale_adjust": float(math.exp(result.x[3])) if solve_scale else 1.0,
    }
    return refined, diagnostics


def direction_cosine(
    pose: np.ndarray,
    page_shape: tuple[int, int],
    anchor_center: tuple[float, float],
    expected_direction_page: tuple[float, float],
) -> float:
    """Agreement between a pose and the printed-neighbor-side direction.

    expected_direction_page is the unit direction from the target's image
    center toward its printed claim of the anchor's sheet number (y-down page
    frame). Returns the cosine between that direction mapped through the pose
    and the actual direction to the anchor: ~1 for the right side, ~-1 for a
    180-degree flip, ~0 for a sideways (90-degree) lock.
    """
    height, width = page_shape
    center = pose @ np.array([width / 2, height / 2, 1.0])
    actual = np.array(anchor_center) - center
    actual_norm = np.linalg.norm(actual)
    implied = pose[:, :2] @ np.array(expected_direction_page)
    implied_norm = np.linalg.norm(implied)
    if actual_norm < 1e-9 or implied_norm < 1e-9:
        return 1.0
    return float(actual @ implied / (actual_norm * implied_norm))


def pose_overlap_frac(
    pose: np.ndarray, page_shape: tuple[int, int], fixed_valid: np.ndarray
) -> float:
    """Fraction of the posed page's in-frame area overlapping the anchor."""
    ones = np.ones(page_shape, np.float32)
    warped = cv2.warpAffine(ones, pose, (fixed_valid.shape[1], fixed_valid.shape[0]))
    placed = warped > MIN_VALID
    area = placed.sum()
    if area == 0:
        return 1.0
    return float((placed & fixed_valid).sum() / area)


def pose_ncc(
    fixed_prob: np.ndarray,
    fixed_valid: np.ndarray,
    target_prob: np.ndarray,
    pose: np.ndarray,
    *,
    sigma_px: float,
    min_overlap_px: int = 500,
) -> float:
    """Correlation of the two P(road) maps over the overlap at a single pose.

    Uses a light blur, so it is sensitive to the actual drawn content (junction
    positions, corridor widths) — the signal that separates a true join from a
    lattice or 180-degree lock that only aligns corridor directions.
    """
    shape = (fixed_valid.shape[1], fixed_valid.shape[0])
    warped = cv2.warpAffine(target_prob, pose, shape)
    warped_valid = cv2.warpAffine(np.ones_like(target_prob), pose, shape) > MIN_VALID
    both = fixed_valid & warped_valid
    if both.sum() < min_overlap_px:
        return 0.0
    a = cv2.GaussianBlur(fixed_prob, (0, 0), sigma_px)[both]
    b = cv2.GaussianBlur(warped, (0, 0), sigma_px)[both]
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denominator < 1e-9:
        return 0.0
    return float((a * b).sum() / denominator)


def refine_and_rank(
    candidates: list[JoinCandidate],
    distance_m: np.ndarray,
    points_page: np.ndarray,
    fixed_valid: np.ndarray | None = None,
    page_shape: tuple[int, int] | None = None,
    max_overlap_frac: float = 1.0,
    region: np.ndarray | None = None,
    fixed_prob: np.ndarray | None = None,
    target_prob: np.ndarray | None = None,
    fine_sigma_px: float = 1.5,
    solve_scale: bool = False,
) -> list[JoinCandidate]:
    """Chamfer-refine every candidate and sort by verification score, best first.

    This is the step that separates aliased/wrong-rotation NCC peaks from the
    true join: a wrong lattice lock leaves many skeleton points far from the
    anchor's corridors, tanking its inlier fraction. Candidates whose refined
    pose overlaps the anchor by more than max_overlap_frac are disqualified
    (adjacent sheets share only a margin strip).
    """
    for candidate in candidates:
        refined, diagnostics = chamfer_refine(
            distance_m,
            points_page,
            candidate.pose,
            region=region,
            solve_scale=solve_scale,
        )
        candidate.pose = refined
        candidate.chamfer_mean_m = diagnostics["chamfer_mean_m"]
        candidate.inlier_frac = diagnostics["inlier_frac"]
        candidate.n_points = diagnostics["n_points"]
        candidate.jtj_min_eig = diagnostics["jtj_min_eig"]
        candidate.jtj_eig_ratio = diagnostics["jtj_eig_ratio"]
        candidate.scale_adjust = diagnostics["scale_adjust"]
        candidate.refined = True
        if fixed_valid is not None and page_shape is not None:
            candidate.overlap_frac = pose_overlap_frac(refined, page_shape, fixed_valid)
            candidate.plausible = candidate.overlap_frac <= max_overlap_frac
        if (
            fixed_prob is not None
            and target_prob is not None
            and fixed_valid is not None
        ):
            candidate.ncc_fine = pose_ncc(
                fixed_prob, fixed_valid, target_prob, refined, sigma_px=fine_sigma_px
            )
    return sorted(candidates, key=lambda c: -c.verification_score())


@dataclass
class MatchParams:
    """Knobs for match_pair, in metres unless noted."""

    resolution_m: float = 2.0
    blur_sigma_m: float = 8.0
    fine_sigma_m: float = 3.0  # blur for the per-pose verification correlation
    min_overlap_m2: float = 8000.0
    max_overlap_frac: float = 0.42  # of the target's area; sheets share a strip
    top_k: int = 5
    peak_separation_m: float = 60.0
    mask_threshold: float = 0.5
    mask_min_area: int = 500
    jitter_deg: tuple[float, ...] = (0.0,)
    solve_scale: bool = False  # bounded scale DOF in chamfer refinement


def warp_page(
    prob: np.ndarray, pose: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a page raster (and validity mask) into the pair frame."""
    warped = cv2.warpAffine(prob, pose, (shape[1], shape[0]))
    valid = cv2.warpAffine(np.ones_like(prob), pose, (shape[1], shape[0]))
    return warped, valid > MIN_VALID


def rotated_bounds(shape: tuple[int, int], matrix: np.ndarray) -> np.ndarray:
    """2x3 matrix shifted so the warped image fits a tight nonnegative box."""
    h, w = shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    warped = corners @ matrix.T
    shifted = matrix.copy()
    shifted[:, 2] -= warped.min(axis=0)
    return shifted


def match_at_rotation(
    fixed_blur: np.ndarray,
    fixed_valid: np.ndarray,
    target_prob: np.ndarray,
    *,
    scale: float,
    theta: float,
    params: MatchParams,
    search_center: tuple[float, float] | None = None,
    search_radius_px: float | None = None,
) -> list[JoinCandidate]:
    """Top-K NCC translation candidates for one rotation.

    fixed_blur must already be blurred by params.blur_sigma_m; search_center /
    search_radius_px (raster px) restrict where the target's CENTER may land.
    """
    res = params.resolution_m
    sigma_px = max(params.blur_sigma_m / res, 0.5)
    min_overlap_px = params.min_overlap_m2 / (res * res)
    separation = max(int(params.peak_separation_m / res), 2)

    base = cv2.getRotationMatrix2D((0.0, 0.0), theta, scale)
    tight = rotated_bounds(target_prob.shape[:2], base)
    h, w = target_prob.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    extent = (corners @ tight.T).max(axis=0).astype(int) + 1
    moving = cv2.warpAffine(target_prob, tight, (extent[0], extent[1]))
    moving_valid = (
        cv2.warpAffine(np.ones_like(target_prob), tight, (extent[0], extent[1]))
        > MIN_VALID
    )
    moving_blur = cv2.GaussianBlur(moving * moving_valid, (0, 0), sigma_px)
    max_overlap_px = params.max_overlap_frac * float(moving_valid.sum())
    score = masked_ncc(
        fixed_blur,
        fixed_valid.astype(np.float32),
        moving_blur,
        moving_valid.astype(np.float32),
        min_overlap_px=min_overlap_px,
        max_overlap_px=max_overlap_px,
    )
    if search_center is not None and search_radius_px is not None:
        # Peak (i, j) places moving's origin at (i - mh + 1, j - mw + 1);
        # the moving image's center then sits at origin + extent/2.
        mh, mw = moving.shape
        rows = np.arange(score.shape[0])[:, None] - (mh - 1) + mh / 2
        cols = np.arange(score.shape[1])[None, :] - (mw - 1) + mw / 2
        dist2 = (rows - search_center[1]) ** 2 + (cols - search_center[0]) ** 2
        score[dist2 > search_radius_px**2] = 0.0
    candidates = []
    for value, row, col in top_peaks(score, params.top_k, separation):
        mh, mw = moving.shape
        pose = tight.copy()
        pose[:, 2] += [col - mw + 1, row - mh + 1]
        candidates.append(JoinCandidate(pose=pose, theta_deg=theta, ncc=value))
    return candidates


def match_pair(
    fixed: np.ndarray,
    fixed_valid: np.ndarray,
    target_prob: np.ndarray,
    *,
    scale: float,
    params: MatchParams,
    search_center: tuple[float, float] | None = None,
    search_radius_px: float | None = None,
) -> list[JoinCandidate]:
    """Candidate poses for a target page against a pre-rendered anchor frame.

    fixed/fixed_valid: anchor P(road) and validity in the pair raster.
    target_prob: the target's P(road) at page resolution. scale: raster cells
    per target page pixel (volume-median metres/px divided by resolution_m).
    search_center/radius (raster px) restrict NCC peaks to an init window.
    """
    sigma_px = max(params.blur_sigma_m / params.resolution_m, 0.5)
    fixed_blur = cv2.GaussianBlur(fixed * fixed_valid, (0, 0), sigma_px)
    candidates: list[JoinCandidate] = []
    thetas = rotation_candidates(
        fixed, target_prob, params.jitter_deg, fixed_valid=fixed_valid
    )
    for theta in thetas:
        candidates.extend(
            match_at_rotation(
                fixed_blur,
                fixed_valid,
                target_prob,
                scale=scale,
                theta=theta,
                params=params,
                search_center=search_center,
                search_radius_px=search_radius_px,
            )
        )
    return candidates
