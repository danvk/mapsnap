"""Pose-graph solver for edge-join page placement.

Jointly estimates every page's pose (position + rotation at a fixed
volume-median scale) from all available evidence at once, instead of the
greedy chain's sequential accept/reject:

  - relative pose measurements between adjacent pages (from the edge-join
    matcher, INCLUDING sub-gate joins — their verification score sets the
    factor weight, and a robust loss lets the graph outvote bad ones);
  - absolute priors from RANSAC fits (weighted by inlier count);
  - loose absolute priors from key-map locations (which break the global
    symmetries — flips and swings — that pairwise checks cannot).

Frame convention: metres about a volume origin with y pointing SOUTH, so a
page-pixel (y-down) to frame map is a proper similarity (rotation + fixed
scale, no reflection) — the same convention the matcher's raster uses.
A pose is (x, y, theta) with page->frame: p_frame = s * R(theta) @ p_px + t.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

EARTH_M_PER_DEG_LAT = 110_540.0
EARTH_M_PER_DEG_LON = 111_320.0


@dataclass
class VolumeFrame:
    """The volume-wide metre frame (y south-positive) about a lon/lat origin."""

    lon0: float
    lat0: float
    scale_m_per_px: float

    def metre_scales(self) -> tuple[float, float]:
        return (
            EARTH_M_PER_DEG_LON * math.cos(math.radians(self.lat0)),
            EARTH_M_PER_DEG_LAT,
        )

    def lonlat_to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        kx, ky = self.metre_scales()
        return (lon - self.lon0) * kx, -(lat - self.lat0) * ky

    def affine_to_pose(self, affine: np.ndarray) -> tuple[float, float, float]:
        """(x, y, theta) of a page px -> lon/lat affine, at the fixed scale."""
        kx, ky = self.metre_scales()
        f00 = kx * affine[0, 0]
        f10 = -ky * affine[1, 0]
        theta = math.atan2(f10, f00)
        x = kx * (affine[0, 2] - self.lon0)
        y = -ky * (affine[1, 2] - self.lat0)
        return x, y, theta

    def pose_to_affine(self, x: float, y: float, theta: float) -> np.ndarray:
        """2x3 page px -> lon/lat affine of a pose at the fixed scale."""
        kx, ky = self.metre_scales()
        s = self.scale_m_per_px
        c, sn = math.cos(theta), math.sin(theta)
        return np.array(
            [
                [s * c / kx, -s * sn / kx, x / kx + self.lon0],
                [-s * sn / ky, -s * c / ky, -y / ky + self.lat0],
            ]
        )

    def relative(
        self, anchor: tuple[float, float, float], target: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """(dx, dy, dtheta): the target pose expressed in the anchor's frame."""
        ax, ay, at = anchor
        tx, ty, tt = target
        c, sn = math.cos(at), math.sin(at)
        ex, ey = tx - ax, ty - ay
        return (
            c * ex + sn * ey,
            -sn * ex + c * ey,
            wrap_radians(tt - at),
        )


def wrap_radians(value: float) -> float:
    """Fold an angle to (-pi, pi]."""
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass
class RelativeMeasurement:
    """A matcher-measured relative pose between two pages (indices into the graph)."""

    a: int
    b: int
    dx: float
    dy: float
    dtheta: float
    sigma_pos_m: float
    sigma_theta_rad: float


@dataclass
class AbsolutePrior:
    """An absolute pose (or position-only) prior for one page."""

    index: int
    x: float
    y: float
    sigma_pos_m: float
    theta: float | None = None
    sigma_theta_rad: float = math.radians(2.0)


def solve_pose_graph(
    initial: np.ndarray,
    measurements: list[RelativeMeasurement],
    priors: list[AbsolutePrior],
    huber_scale: float = 2.5,
    max_nfev: int = 200,
    loss: str = "huber",
) -> tuple[np.ndarray, dict]:
    """Robust joint solve of page poses.

    initial: (N, 3) array of (x, y, theta). Returns (solved poses,
    diagnostics). Residuals are sigma-normalized; a robust loss (`huber` or
    the heavier-tailed `cauchy`) at `huber_scale` sigmas lets grossly-wrong
    measurements be outvoted.
    """
    n = len(initial)
    m_a = np.array([m.a for m in measurements], dtype=int)
    m_b = np.array([m.b for m in measurements], dtype=int)
    m_dx = np.array([m.dx for m in measurements])
    m_dy = np.array([m.dy for m in measurements])
    m_dt = np.array([m.dtheta for m in measurements])
    m_wp = 1.0 / np.array([m.sigma_pos_m for m in measurements])
    m_wt = 1.0 / np.array([m.sigma_theta_rad for m in measurements])

    p_i = np.array([p.index for p in priors], dtype=int)
    p_x = np.array([p.x for p in priors])
    p_y = np.array([p.y for p in priors])
    p_wp = 1.0 / np.array([p.sigma_pos_m for p in priors])
    with_theta = [p for p in priors if p.theta is not None]
    pt_i = np.array([p.index for p in with_theta], dtype=int)
    pt_t = np.array([p.theta for p in with_theta])
    pt_w = 1.0 / np.array([p.sigma_theta_rad for p in with_theta])

    def residuals(params: np.ndarray) -> np.ndarray:
        poses = params.reshape(n, 3)
        parts = []
        if len(measurements):
            ca = np.cos(poses[m_a, 2])
            sa = np.sin(poses[m_a, 2])
            ex = poses[m_b, 0] - poses[m_a, 0]
            ey = poses[m_b, 1] - poses[m_a, 1]
            pred_dx = ca * ex + sa * ey
            pred_dy = -sa * ex + ca * ey
            dtheta = poses[m_b, 2] - poses[m_a, 2] - m_dt
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            parts += [
                (pred_dx - m_dx) * m_wp,
                (pred_dy - m_dy) * m_wp,
                dtheta * m_wt,
            ]
        if len(priors):
            parts += [
                (poses[p_i, 0] - p_x) * p_wp,
                (poses[p_i, 1] - p_y) * p_wp,
            ]
        if len(with_theta):
            dt = poses[pt_i, 2] - pt_t
            parts.append(((dt + np.pi) % (2 * np.pi) - np.pi) * pt_w)
        return np.concatenate(parts)

    result = least_squares(
        residuals,
        initial.reshape(-1),
        loss=loss,
        f_scale=huber_scale,
        max_nfev=max_nfev,
    )
    solved = result.x.reshape(n, 3)
    final = residuals(result.x)
    diagnostics = {
        "cost": float(result.cost),
        "n_residuals": int(len(final)),
        "rms_normalized": float(np.sqrt((final**2).mean())),
        "nfev": int(result.nfev),
    }
    return solved, diagnostics


@dataclass
class EdgeHypotheses:
    """One adjacency edge with several candidate relative poses.

    All candidates share (a, b); candidate 0 is the matcher's ranked winner.
    """

    a: int
    b: int
    candidates: list[RelativeMeasurement]


def measurement_error_m(
    m: RelativeMeasurement, poses: np.ndarray, theta_weight_m_per_deg: float = 5.0
) -> float:
    """Raw fit error (metres) of one measurement at given poses.

    Unnormalized on purpose: assignment must not favour loose-sigma junk
    candidates the way a sigma-normalized residual would.
    """
    ax, ay, at = poses[m.a]
    bx, by, bt = poses[m.b]
    c, s = math.cos(at), math.sin(at)
    ex, ey = bx - ax, by - ay
    dx_err = c * ex + s * ey - m.dx
    dy_err = -s * ex + c * ey - m.dy
    dt_err = wrap_radians(bt - at - m.dtheta)
    return (
        math.hypot(dx_err, dy_err) + abs(math.degrees(dt_err)) * theta_weight_m_per_deg
    )


def solve_pose_graph_hypotheses(
    initial: np.ndarray,
    edges: list[EdgeHypotheses],
    priors: list[AbsolutePrior],
    max_rounds: int = 5,
    trim_sigma: float = 3.0,
) -> tuple[np.ndarray, list[int], list[bool], dict]:
    """EM-style solve over multi-hypothesis edges.

    Alternate rounds of (a) robust solve with each edge's assigned candidate
    and (b) re-assigning each edge to the candidate that best fits the poses.
    On self-similar grids the true relative pose is often ranked second behind
    an aliased one-block slide; global consistency recovers it. After
    assignments converge, edges whose best candidate still misfits by more
    than `trim_sigma` of its own sigma are dropped and the graph re-solved.

    Returns (poses, assignment per edge, active flag per edge, diagnostics).
    """
    assignment = [0] * len(edges)
    poses = initial
    switches_total = 0
    for _ in range(max_rounds):
        measurements = [e.candidates[assignment[i]] for i, e in enumerate(edges)]
        poses, diag = solve_pose_graph(poses, measurements, priors)
        new_assignment = [
            min(
                range(len(e.candidates)),
                key=lambda j: measurement_error_m(e.candidates[j], poses),
            )
            for e in edges
        ]
        switches = sum(1 for a, b in zip(assignment, new_assignment) if a != b)
        switches_total += switches
        assignment = new_assignment
        if switches == 0:
            break

    # Trial-switch pass: local argmin assignment cannot escape a basin where
    # the graph has deformed to fit a wrong candidate (e.g. a slide absorbed
    # by dragging a degree-2 neighbor). For edges that still misfit, try each
    # alternate with a warm re-solve and keep it only if the global robust
    # cost drops.
    def solve_assigned(start: np.ndarray) -> tuple[np.ndarray, dict]:
        return solve_pose_graph(
            start, [e.candidates[assignment[i]] for i, e in enumerate(edges)], priors
        )

    poses, diag = solve_assigned(poses)
    for _ in range(3):
        improved = False
        for i, e in enumerate(edges):
            if len(e.candidates) < 2:
                continue
            m = e.candidates[assignment[i]]
            if measurement_error_m(m, poses) <= 2.0 * m.sigma_pos_m:
                continue
            original = assignment[i]
            best_j, best_cost, best_poses = original, diag["cost"], poses
            for j in range(len(e.candidates)):
                if j == original:
                    continue
                assignment[i] = j
                trial_poses, trial_diag = solve_assigned(poses)
                if trial_diag["cost"] < best_cost - 1e-6:
                    best_j, best_cost = j, trial_diag["cost"]
                    best_poses, diag = trial_poses, trial_diag
            assignment[i] = best_j
            if best_j != original:
                poses = best_poses
                switches_total += 1
                improved = True
        if not improved:
            break
    active = []
    for i, e in enumerate(edges):
        m = e.candidates[assignment[i]]
        error = measurement_error_m(m, poses, theta_weight_m_per_deg=0.0)
        theta_err = abs(wrap_radians(poses[e.b, 2] - poses[e.a, 2] - m.dtheta))
        active.append(
            error <= trim_sigma * m.sigma_pos_m
            and theta_err <= trim_sigma * m.sigma_theta_rad
        )
    kept = [e.candidates[assignment[i]] for i, e in enumerate(edges) if active[i]]
    poses, diag = solve_pose_graph(poses, kept, priors)
    diag["switches"] = switches_total
    diag["trimmed"] = int(len(edges) - sum(active))
    return poses, assignment, active, diag


def spanning_tree_initialization(
    n: int,
    initial_known: dict[int, tuple[float, float, float]],
    measurements: list[RelativeMeasurement],
    fallback: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """Initial poses: known poses, then BFS through the strongest measurements.

    Pages reachable from a known pose get the measurement-propagated pose;
    anything else falls back (e.g. to its key-map location).
    """
    poses: dict[int, tuple[float, float, float]] = dict(initial_known)
    # Strongest measurements first so the BFS tree prefers confident edges.
    ordered = sorted(measurements, key=lambda m: m.sigma_pos_m)
    changed = True
    while changed:
        changed = False
        for m in ordered:
            if m.a in poses and m.b not in poses:
                ax, ay, at = poses[m.a]
                c, sn = math.cos(at), math.sin(at)
                poses[m.b] = (
                    ax + c * m.dx - sn * m.dy,
                    ay + sn * m.dx + c * m.dy,
                    wrap_radians(at + m.dtheta),
                )
                changed = True
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = poses.get(i) or fallback.get(i) or (0.0, 0.0, 0.0)
    return result
