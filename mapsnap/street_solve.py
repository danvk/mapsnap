"""Georeference a page from street labels as constraints, with no intersection GCPs.

The RANSAC georeferencer needs two detected streets to *cross*, so it can anchor a
control point. Many pages never produce one: their readable streets run mutually
parallel, or the crossing falls off the sheet. But a street label still carries two
usable facts on its own — the label sits on its street (position), and its long axis
runs along that street (angle). Two distinct parallel streets fix rotation and, from
their spacing, scale; one crossing street completes the pose. No intersections needed.

This module solves that pose (issue #168). It runs only where the page already has a
coarse location prior, restricts the street vocabulary to that neighborhood, and is
built around the fact that **a quarter to a half of the assembled constraints are
wrong** — renamed streets that match elsewhere in the radius, streets rerouted since
the survey, and near-square text boxes whose long axis is meaningless. Robustness is
therefore structural, not incidental:

  * consensus over constraint pairs proposes poses; outliers never enter the fit;
  * hard angle/position gates decide inliers, tighter than RANSAC's 30 degrees;
  * a leave-one-out check rejects a pose that rests on a single constraint.

Anti-drift guards (an early version of this idea, in a predecessor repo, drifted by
re-snapping onto streets that end sooner in OSM than they did on the Sanborn sheet,
rotating the fit until it was worthless):

  G1  constraints are assembled once, from the prior, and never re-matched at a
      refined pose — the loop that produced the original drift does not exist here.
  G2  no endpoint extrapolation: snapping is clipped to the segments that exist, so a
      street truncated in OSM produces a large residual rather than a plausible one.
  G3  that residual then fails the hard position gate and the constraint is dropped.
  G4  exactly one least-squares polish, on a frozen inlier set, and the pose is
      rejected outright if the polish travels far from the consensus proposal.
  G5  the scale prior is soft but bounded, so contaminated positions cannot trade
      themselves into a plausible-looking rescale.
  G6  rotation seeds come from street-vs-OSM evidence, never a blind sweep: at four
      to six constraints a sweep finds self-consistent poses tens of degrees from
      truth (New Orleans 1896 p181 has one at ~30 degrees).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from mapsnap.georef_from_labels import LabelFeature
from mapsnap.keymap.align_page_region import (
    PoseSigmas,
    StreetPose,
    StreetSegments,
    angle_difference_mod180,
    constraints_from_features,
    point_to_segments,
    pose_residuals,
    pose_world_of,
    solve_pose,
)
from mapsnap.streets import Block, street_name_family

Point = tuple[float, float]


@dataclass(frozen=True)
class StreetGates:
    """Every tunable of the streets-only solver, in one sweepable place."""

    angle_gate_deg: float = 8.0  # inlier: |label bearing - street bearing| at the pose
    # Inlier: label centre to street distance (G3). Below a downtown block: at 80 m
    # Nashville's numbered-avenue grid let poses slide half a block and still agree
    # with every street (five fits over 200 ft); at 50 m those vanish and no genuine
    # fit is lost -- the inliers of every correct fit measured sit under 46 m.
    position_gate_m: float = 50.0
    min_inliers: int = 3
    min_inlier_fraction: float = 0.6  # a right pose explains most of what the page says
    min_distinct_streets: int = 3  # two streets determine the pose with no redundancy
    bearing_diversity_deg: float = 30.0  # two inlier bearings must differ by this much
    min_aspect: float = 2.0  # near-square boxes carry no usable long axis
    parallel_tolerance_deg: float = 10.0  # bearings this close count as parallel
    min_parallel_spacing_m: float = 20.0  # closer than this, spacing cannot fix scale
    polish_max_move_m: float = 60.0  # G4: polish travel from the consensus proposal
    polish_max_rot_deg: float = 3.0
    loo_max_shift_m: float = 30.0  # leave-one-out stability
    loo_max_rot_deg: float = 2.0
    radius_slack_m: float = 60.0  # converged center vs the prior location
    # Positions are trustworthy once a street that stops short is no longer read as a
    # position error (see extend_terminal_segments): a correct page's labels sit 1-3 m
    # from their centerlines, so the 60 m inherited from the region placer -- where a
    # label really does sit off the line -- let the fit trade position away for angle.
    sigma_line_m: float = 15.0
    sigma_angle_deg: float = 3.0  # angles are the reliable signal
    sigma_log_scale: float = 0.05  # G5: soft scale prior (bounded in solve_pose)
    sigma_centroid_m: float = 400.0  # the location prior is a neighborhood, not a fix
    max_psi_seeds: int = 8  # bearing clusters tried per page (evidence, not a sweep)
    max_scale_seeds: int = 5  # distinct scale proposals tried per bearing
    max_proposals: int = 6  # distinct placements polished per bearing, best cost wins
    # A street whose OSM data stops short of the label still constrains it: the label
    # sits on the street's line, and the sheet simply drew more of the street than
    # today's data holds -- on Detroit's east side, hundreds of metres more, since the
    # streets themselves are gone. The allowance is therefore generous and the position
    # gate does the rejecting: a label that still does not fit is an outlier, which is
    # the honest test. Detroit p5's labels sit 1-5 m from their streets' lines and 170
    # to 400 m past where OSM stops drawing them (282 ft -> 17 ft at this distance).
    terminal_extrapolation_m: float = 500.0


DEFAULT_GATES = StreetGates()


@dataclass
class PriorLocation:
    """A page's coarse location before any fitting, and where it came from."""

    center: Point  # (lon, lat) frame origin and centroid-spring anchor
    radius_m: float
    source: str  # "keymap-exact" | "keymap-family" | "fit-center" | "truth-centroid"
    centers: list[Point] = field(default_factory=list)  # all centers, for the vocab

    def __post_init__(self) -> None:
        if not self.centers:
            self.centers = [self.center]


@dataclass
class ConstraintDiag:
    """Per-constraint diagnostics at the solved pose (or at drop time)."""

    name: str
    center_px: tuple[float, float]
    position_m: float | None = None
    angle_deg: float | None = None
    inlier: bool = False
    dropped: str = ""


@dataclass
class StreetSolveResult:
    """A streets-only fit, or an abstention with the reason it abstained."""

    pose: StreetPose | None = None
    abstain: str = ""
    psi_source: str = ""
    n_constraints: int = 0
    n_inliers: int = 0
    bearing_spread_deg: float = 0.0
    polish_move_m: float = 0.0
    polish_rot_deg: float = 0.0
    loo_max_shift_m: float = 0.0
    loo_max_rot_deg: float = 0.0
    scale_source: str = ""
    cost: float = 0.0  # the soft_l1 objective at this pose, over every constraint
    diagnostics: list[ConstraintDiag] = field(default_factory=list)


def psi_from_theta(theta_deg: float) -> float:
    """Page-up bearing (StreetPose psi) for an osm_snap rotation prior's theta.

    A pixel delta at page angle ``a`` maps, under a pose with bearing psi, to a world
    direction whose y-down raster angle is ``a + psi``; osm_snap defines
    ``theta = page_angle - raster_angle``, so psi is its negation.
    """
    return -theta_deg


def distinct_lines(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    bearing_tolerance_deg: float = 5.0,
    offset_tolerance_m: float = 15.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collapse a segment soup to the distinct straight lines it lies on.

    A street's segments are mostly collinear, so the soup usually describes one or two
    lines; the consensus search needs those, not hundreds of near-duplicates. Each line
    is returned as (point on the line, unit direction).
    """
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in zip(starts, ends):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        direction = delta / length
        bearing = math.degrees(math.atan2(direction[0], direction[1])) % 180.0
        for point, existing in lines:
            existing_bearing = (
                math.degrees(math.atan2(existing[0], existing[1])) % 180.0
            )
            if abs(angle_difference_mod180(bearing, existing_bearing)) > (
                bearing_tolerance_deg
            ):
                continue
            normal = np.array([existing[1], -existing[0]])
            if abs(float((start - point) @ normal)) <= offset_tolerance_m:
                break
        else:
            lines.append((start, direction))
    return lines


def wrap_degrees(value: float) -> float:
    """An angle folded into (-180, 180]."""
    return (value + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles: list[float]) -> float:
    """Mean of full-circle angles, in [0, 360)."""
    x = float(np.mean([math.cos(math.radians(a)) for a in angles]))
    y = float(np.mean([math.sin(math.radians(a)) for a in angles]))
    return math.degrees(math.atan2(y, x)) % 360.0


def psi_votes(
    constraints: list[StreetSegments], gates: StreetGates = DEFAULT_GATES
) -> list[tuple[float, str]]:
    """Page bearings implied by the constraints themselves, most-supported first.

    Every constraint votes: its label's long axis runs along its street, so each
    straight line in that street's soup implies psi = bearing - 90 - label angle (and
    the 180-degree flip). A correct constraint votes for the true bearing; wrong ones
    scatter, so the largest cluster is the page's rotation. This is the same evidence
    ``osm_snap.label_osm_rotations`` uses, but read off the locality-trimmed soup each
    constraint actually carries, which matters where a street curves away from the
    prior — the tangent a kilometre off is not the tangent under the label.
    """
    votes: list[float] = []
    for _center, dir_pix, _name, starts, ends in constraints:
        for _point, direction in distinct_lines(starts, ends):
            bearing = math.degrees(math.atan2(direction[0], direction[1])) % 180.0
            base = (bearing - 90.0 - math.degrees(dir_pix)) % 360.0
            # A street's bearing is undirected, so each vote admits both page-up
            # senses; the pose's own bearing is not (the page has a top), so the two
            # stay separate candidates rather than averaging into a meaningless middle.
            votes.extend((base, (base + 180.0) % 360.0))
    clusters: list[list[float]] = []
    for vote in sorted(votes):
        for cluster in clusters:
            if abs(wrap_degrees(vote - cluster[0])) <= 2.0:
                cluster.append(vote)
                break
        else:
            clusters.append([vote])
    clusters.sort(key=len, reverse=True)
    return [
        (circular_mean_deg(cluster), "constraint-vote")
        for cluster in clusters[: gates.max_psi_seeds]
    ]


def extend_terminal_segments(
    starts: np.ndarray, ends: np.ndarray, distance_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a polyline's open ends outward, leaving interior joins untouched.

    New Orleans p181 shows why: SOUTH MIRO's label sits 0.3 m from its street's line
    but 49 m past where OSM stops drawing it, and measuring to the segment reads that
    shortfall as a 49 m position error. Left alone the solver moves the page to
    "fix" it, which is how a fit ends up translated off a set of otherwise perfect
    constraints. Only ends that no other segment touches are extended, so a street
    that genuinely turns is not straightened.
    """
    if not len(starts) or distance_m <= 0:
        return starts, ends
    counts: dict[tuple[float, float], int] = {}
    for point in np.vstack([starts, ends]):
        key = (round(float(point[0]), 2), round(float(point[1]), 2))
        counts[key] = counts.get(key, 0) + 1

    def is_open(point: np.ndarray) -> bool:
        return counts.get((round(float(point[0]), 2), round(float(point[1]), 2)), 0) < 2

    new_starts, new_ends = starts.copy(), ends.copy()
    for i, (start, end) in enumerate(zip(starts, ends)):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        unit = delta / length
        if is_open(start):
            new_starts[i] = start - unit * distance_m
        if is_open(end):
            new_ends[i] = end + unit * distance_m
    return new_starts, new_ends


def constraint_bearing(pose: StreetPose, dir_pix: float) -> float:
    """The world bearing (mod 180) a label's long axis claims under ``pose``."""
    return (pose[2] + 90.0 + math.degrees(dir_pix)) % 180.0


def residuals_at(
    pose: StreetPose, constraint: StreetSegments, size: tuple[int, int]
) -> tuple[float, float]:
    """(position error in metres, signed angle error in degrees) for one constraint."""
    center_px, dir_pix, _name, starts, ends = constraint
    world = pose_world_of(pose, center_px, size)
    distance, bearing = point_to_segments(world, starts, ends)
    return distance, angle_difference_mod180(constraint_bearing(pose, dir_pix), bearing)


def is_inlier(
    pose: StreetPose,
    constraint: StreetSegments,
    size: tuple[int, int],
    gates: StreetGates,
) -> bool:
    """Whether a constraint agrees with ``pose`` within both hard gates."""
    position, angle = residuals_at(pose, constraint, size)
    return position <= gates.position_gate_m and abs(angle) <= gates.angle_gate_deg


def distinct_streets(constraints: list[StreetSegments]) -> int:
    """How many different streets a constraint set speaks for (variants count once)."""
    return len({street_name_family(c[2]) for c in constraints})


def bearing_spread(pose: StreetPose, constraints: list[StreetSegments]) -> float:
    """The largest angular gap (mod 180) between any two constraints' bearings."""
    bearings = [constraint_bearing(pose, c[1]) for c in constraints]
    return max(
        (
            abs(angle_difference_mod180(a, b))
            for i, a in enumerate(bearings)
            for b in bearings[i + 1 :]
        ),
        default=0.0,
    )


def scale_candidates(
    constraints: list[StreetSegments],
    lines: list[list[tuple[np.ndarray, np.ndarray]]],
    psi_deg: float,
    prior_log_scale: float,
    size: tuple[int, int],
    gates: StreetGates,
) -> list[tuple[float, str]]:
    """Log-scale candidates: the prior, plus one per pair of parallel streets.

    Two labels on distinct parallel streets fix the scale on their own: the pixel
    distance between them across the streets, over the metre spacing of the streets,
    is pixels per metre. That is what lets this channel place a page whose scale prior
    is missing or wrong.
    """
    candidates: list[tuple[float, str]] = [(prior_log_scale, "prior")]
    for i, first in enumerate(constraints):
        for j in range(i + 1, len(constraints)):
            second = constraints[j]
            if first[2] == second[2]:
                continue  # same street: spacing is zero by construction
            for point_a, direction_a in lines[i]:
                for point_b, direction_b in lines[j]:
                    bearing_a = math.degrees(math.atan2(*direction_a)) % 180.0
                    bearing_b = math.degrees(math.atan2(*direction_b)) % 180.0
                    if (
                        abs(angle_difference_mod180(bearing_a, bearing_b))
                        > gates.parallel_tolerance_deg
                    ):
                        continue
                    normal = np.array([direction_a[1], -direction_a[0]])
                    spacing = abs(float((point_b - point_a) @ normal))
                    if spacing < gates.min_parallel_spacing_m:
                        continue
                    # Pixel separation of the two labels across the street direction,
                    # measured in the page frame at bearing psi.
                    pose = (0.0, 0.0, psi_deg, 0.0)
                    world_a = pose_world_of(pose, first[0], size)
                    world_b = pose_world_of(pose, second[0], size)
                    pixel_spacing = abs(float((world_b - world_a) @ normal))
                    if pixel_spacing < 1e-6:
                        continue
                    candidates.append(
                        (math.log(pixel_spacing / spacing), "parallel-pair")
                    )
    return dedupe_scales(candidates, gates.max_scale_seeds)


def dedupe_scales(
    candidates: list[tuple[float, str]], cap: int
) -> list[tuple[float, str]]:
    """Collapse log-scale proposals that agree within a percent, keeping the first."""
    kept: list[tuple[float, str]] = []
    for value, source in candidates:
        if all(abs(value - existing) > 0.01 for existing, _ in kept):
            kept.append((value, source))
        if len(kept) >= cap:
            break
    return kept


def batch_point_to_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(min distance, bearing of the nearest segment) for MANY points at once.

    The vectorized twin of ``align_page_region.point_to_segments`` — the same
    clipped projection and nearest-segment bearing, evaluated for an (N, 2)
    array of points in one broadcast instead of N interpreted calls. Returns
    (distances (N,), bearings (N,) in degrees mod 180).
    """
    delta = ends - starts  # (S, 2)
    length_sq = (delta * delta).sum(axis=1)  # (S,)
    diff = points[:, None, :] - starts[None]  # (N, S, 2)
    t = np.clip((diff * delta).sum(axis=2) / np.maximum(length_sq, 1e-9), 0, 1)
    projected = starts[None] + t[:, :, None] * delta[None]  # (N, S, 2)
    distances = np.linalg.norm(points[:, None, :] - projected, axis=2)  # (N, S)
    nearest = distances.argmin(axis=1)
    segment_bearings = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 180.0
    rows = np.arange(len(points))
    return distances[rows, nearest], segment_bearings[nearest]


LineEquation = tuple[
    float, float, float, float
]  # (normal_x, normal_y, n·point, bearing)


def line_equations(
    lines: list[tuple[np.ndarray, np.ndarray]],
) -> list[LineEquation]:
    """Each line as (normal, normal·point, bearing) scalars, precomputed once.

    A label lying on a known line is one linear equation in the translation
    (normal · (t + label_world) = normal · point); two independent lines
    determine t outright. Extracting the coefficients here lets the consensus
    search solve each pair in closed form instead of building numpy matrices
    a million times per page.
    """
    equations: list[LineEquation] = []
    for point, direction in lines:
        normal_x, normal_y = float(direction[1]), float(-direction[0])
        equations.append(
            (
                normal_x,
                normal_y,
                normal_x * float(point[0]) + normal_y * float(point[1]),
                math.degrees(math.atan2(*direction)) % 180.0,
            )
        )
    return equations


def consensus_poses(
    constraints: list[StreetSegments],
    psi_deg: float,
    log_scales: list[tuple[float, str]],
    size: tuple[int, int],
    gates: StreetGates,
    lines: list[list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> list[tuple[StreetPose, list[StreetSegments], str]]:
    """The most-supported distinct proposals over all pair/line/scale combinations.

    Several are kept, not one: inlier count alone picks the wrong pose where a
    block-over placement admits one more marginal constraint than the right placement
    does (Kansas City p502 -- four against truth's three). The caller polishes each
    and decides on the fitted objective, which is the thing actually being minimized.
    """
    scored: list[tuple[tuple[int, float], StreetPose, list[StreetSegments], str]] = []
    if lines is None:
        lines = [distinct_lines(c[3], c[4]) for c in constraints]
    families = [street_name_family(c[2]) for c in constraints]
    # constraint_bearing depends only on psi, which is fixed for this call, so
    # every constraint's claimed bearing is one number for the whole search.
    claimed = [constraint_bearing((0.0, 0.0, psi_deg, 0.0), c[1]) for c in constraints]
    equations = [line_equations(choices) for choices in lines]
    for log_scale, scale_source in log_scales:
        # Propose: two labels pinned exactly onto two non-parallel street lines
        # — each line is one linear equation in the translation, solved in
        # closed form. The same placement reached from different pairs is
        # merged HERE, on a 1 m grid, rather than scored repeatedly and
        # deduplicated at 20 m afterwards.
        bases = [
            pose_world_of((0.0, 0.0, psi_deg, log_scale), c[0], size)
            for c in constraints
        ]
        proposals: dict[tuple[float, float], StreetPose] = {}
        for i in range(len(constraints)):
            base_i = (float(bases[i][0]), float(bases[i][1]))
            for j in range(i + 1, len(constraints)):
                base_j = (float(bases[j][0]), float(bases[j][1]))
                for nax, nay, ndpa, bearing_a in equations[i]:
                    rhs_a = ndpa - nax * base_i[0] - nay * base_i[1]
                    for nbx, nby, ndpb, bearing_b in equations[j]:
                        if (
                            abs(angle_difference_mod180(bearing_a, bearing_b))
                            < gates.bearing_diversity_deg
                        ):
                            continue
                        det = nax * nby - nay * nbx
                        if abs(det) < 1e-6:
                            continue
                        rhs_b = ndpb - nbx * base_j[0] - nby * base_j[1]
                        tx = (rhs_a * nby - nay * rhs_b) / det
                        ty = (nax * rhs_b - rhs_a * nbx) / det
                        proposals.setdefault(
                            (round(tx), round(ty)), (tx, ty, psi_deg, log_scale)
                        )
        if not proposals:
            continue
        # Score every proposal against every constraint in one broadcast per
        # constraint: with bearing and scale fixed, a proposal only TRANSLATES
        # each label's world point (pose_world_of is base + t), so the per-pose
        # residual loop collapses to array arithmetic. This was 97% of the
        # channel's runtime as 14M scalar residuals_at calls (43 s on a
        # constraint-heavy Kansas City page; ~2 s batched).
        poses = list(proposals.values())
        translations = np.array([(p[0], p[1]) for p in poses])
        position = np.empty((len(poses), len(constraints)))
        angle = np.empty_like(position)
        for k, c in enumerate(constraints):
            distances, bearings = batch_point_to_segments(
                translations + bases[k], c[3], c[4]
            )
            position[:, k] = distances
            angle[:, k] = (claimed[k] - bearings + 90.0) % 180.0 - 90.0
        inlier_mask = (position <= gates.position_gate_m) & (
            np.abs(angle) <= gates.angle_gate_deg
        )
        counts = inlier_mask.sum(axis=1)
        for p, pose in enumerate(poses):
            if counts[p] < gates.min_inliers:
                continue
            if counts[p] < gates.min_inlier_fraction * len(constraints):
                # Kansas City p576 slid a block on two streets while four
                # others disagreed; a correct pose explains most of them.
                continue
            indices = np.flatnonzero(inlier_mask[p])
            inliers = [constraints[q] for q in indices]
            if len({families[q] for q in indices}) < gates.min_distinct_streets:
                continue
            spread = max(
                (
                    abs(angle_difference_mod180(claimed[a], claimed[b]))
                    for x, a in enumerate(indices)
                    for b in indices[x + 1 :]
                ),
                default=0.0,
            )
            if spread < gates.bearing_diversity_deg:
                continue
            # The batch already holds every inlier's position residual; the
            # mean falls out for free instead of a second residuals_at pass.
            mean_position = float(position[p, indices].mean())
            scored.append(
                (
                    (int(counts[p]), -mean_position),
                    pose,
                    inliers,
                    scale_source,
                )
            )
    scored.sort(key=lambda entry: entry[0], reverse=True)
    kept: list[tuple[StreetPose, list[StreetSegments], str]] = []
    for _score, pose, inliers, scale_source in scored:
        if any(
            math.hypot(pose[0] - other[0][0], pose[1] - other[0][1]) < 20.0
            and abs(pose[3] - other[0][3]) < 0.01
            for other in kept
        ):
            continue  # the same placement reached from another pair
        kept.append((pose, inliers, scale_source))
        if len(kept) >= gates.max_proposals:
            break
    return kept


def soft_l1_cost(residuals: np.ndarray) -> float:
    """scipy's soft_l1 cost for a residual vector (f_scale=1), so poses compare."""
    return float(0.5 * np.sum(2.0 * (np.sqrt(1.0 + residuals**2) - 1.0)))


def synthetic_ring(center_offset: Point = (0.0, 0.0)) -> np.ndarray:
    """A one-vertex 'region' so pose_residuals' centroid spring anchors on the prior.

    The containment factor is inert for a point at the page center (it is inside the
    page frame by construction), leaving only the centroid spring — which is exactly
    the location prior this channel is given.
    """
    return np.array([[center_offset[0], center_offset[1]]])


def solve_streets_pose(
    constraints: list[StreetSegments],
    *,
    size: tuple[int, int],
    prior_log_scale: float,
    psi_priors: list[tuple[float, str]],
    gates: StreetGates = DEFAULT_GATES,
    prior_radius_m: float = 0.0,
) -> StreetSolveResult:
    """Fit a 4-DOF pose from street constraints alone, or abstain with a reason.

    ``psi_priors`` are (bearing, source) seeds in StreetPose degrees — evidence, never
    a sweep (G6). ``constraints`` must already be restricted to the page's
    neighborhood and expressed in a metre frame whose origin is the prior location.
    """
    result = StreetSolveResult(n_constraints=len(constraints))
    if not psi_priors:
        result.abstain = "no-rotation-prior"
        return result
    if len(constraints) < gates.min_inliers:
        result.abstain = "too-few-constraints"
        return result

    sigmas = PoseSigmas(
        line_m=gates.sigma_line_m,
        angle_deg=gates.sigma_angle_deg,
        log_scale=gates.sigma_log_scale,
        # Weak on purpose: the prior says which neighborhood, not where in it. The
        # hard radius gate below is what bounds the answer.
        centroid_m=max(gates.sigma_centroid_m, prior_radius_m),
    )
    ring = synthetic_ring()
    best: tuple[tuple[int, float], StreetSolveResult] | None = None
    # Segment soups collapse to a handful of straight lines; computing that once per
    # page (not per bearing seed, per pair) is what keeps the search tractable.
    lines = [distinct_lines(c[3], c[4]) for c in constraints]

    for psi, psi_source in psi_priors[: gates.max_psi_seeds]:
        proposals = consensus_poses(
            constraints,
            psi,
            scale_candidates(constraints, lines, psi, prior_log_scale, size, gates),
            size,
            gates,
            lines,
        )
        for seed, inliers, scale_source in proposals:
            polished = solve_pose(
                seed,
                inliers,
                [],
                region_vertices=ring,
                size=size,
                prior_log_scale=prior_log_scale,
                sigmas=sigmas,
            )
            if polished is None:
                continue
            pose = polished[0]
            move = math.hypot(pose[0] - seed[0], pose[1] - seed[1])
            rotation = abs(angle_difference_mod180(pose[2], seed[2]))
            attempt = StreetSolveResult(
                psi_source=psi_source,
                n_constraints=len(constraints),
                scale_source=scale_source,
                polish_move_m=move,
                polish_rot_deg=rotation,
            )
            if move > gates.polish_max_move_m or rotation > gates.polish_max_rot_deg:
                attempt.abstain = "polish-runaway"
            elif not all(is_inlier(pose, c, size, gates) for c in inliers):
                # G4: an inlier that no longer agrees means the polish moved onto a
                # different interpretation; re-solving on the survivors is the drift loop.
                attempt.abstain = "polish-lost-inliers"
            elif prior_radius_m and math.hypot(pose[0], pose[1]) > (
                prior_radius_m + gates.radius_slack_m
            ):
                attempt.abstain = "outside-prior-radius"
            else:
                spread = bearing_spread(pose, inliers)
                shift, rotate = (
                    leave_one_out_spread(
                        pose, inliers, size, prior_log_scale, sigmas, ring
                    )
                    # Needs real redundancy: dropping one of four leaves an exactly
                    # determined three, which moves for sound reasons, so the check would
                    # reject good fits (LA p1467 at 34 ft) rather than fragile ones.
                    if len(inliers) >= gates.min_inliers + 2
                    else (0.0, 0.0)
                )
                attempt.bearing_spread_deg = spread
                attempt.loo_max_shift_m = shift
                attempt.loo_max_rot_deg = rotate
                if shift > gates.loo_max_shift_m or rotate > gates.loo_max_rot_deg:
                    attempt.abstain = "unstable-leave-one-out"
                else:
                    attempt.pose = pose
                    attempt.n_inliers = len(inliers)
            attempt.diagnostics = [
                ConstraintDiag(
                    name=c[2],
                    center_px=c[0],
                    position_m=residuals_at(pose, c, size)[0],
                    angle_deg=residuals_at(pose, c, size)[1],
                    inlier=c in inliers,
                )
                for c in constraints
            ]
            # Decide on the objective being minimized, over every constraint. Inlier
            # count picks whichever placement admits one more marginal street, which
            # is how p502 landed a block from a pose this objective scores better.
            attempt.cost = soft_l1_cost(
                pose_residuals(
                    pose,
                    constraints,
                    [],
                    region_vertices=ring,
                    size=size,
                    prior_log_scale=prior_log_scale,
                    sigmas=sigmas,
                )
            )
            score = (1 if attempt.pose else 0, -attempt.cost)
            if best is None or score > best[0]:
                best = (score, attempt)

    if best is None:
        result.abstain = "no-consensus"
        return result
    return best[1]


def leave_one_out_spread(
    pose: StreetPose,
    inliers: list[StreetSegments],
    size: tuple[int, int],
    prior_log_scale: float,
    sigmas: PoseSigmas,
    ring: np.ndarray,
) -> tuple[float, float]:
    """(max centre shift, max rotation change) when each inlier is dropped in turn.

    A pose that moves when one constraint leaves is resting on that constraint; with a
    quarter of constraints wrong, that is the shape a wrong fit takes. Callers only
    apply this where the inliers are redundant: a minimum-size set determines the pose
    exactly, so every member is load-bearing and the check carries no information.
    """
    max_shift = 0.0
    max_rotation = 0.0
    for index in range(len(inliers)):
        remaining = inliers[:index] + inliers[index + 1 :]
        if len(remaining) < 2:
            continue
        refit = solve_pose(
            pose,
            remaining,
            [],
            region_vertices=ring,
            size=size,
            prior_log_scale=prior_log_scale,
            sigmas=sigmas,
        )
        if refit is None:
            continue
        candidate = refit[0]
        max_shift = max(
            max_shift, math.hypot(candidate[0] - pose[0], candidate[1] - pose[1])
        )
        max_rotation = max(
            max_rotation, abs(angle_difference_mod180(candidate[2], pose[2]))
        )
    return max_shift, max_rotation


def assemble_constraints(
    features: list[LabelFeature],
    block_index: dict[str, list[Block]],
    *,
    prior: PriorLocation,
    label_size: tuple[int, int],
    working_size: tuple[int, int],
    scale_px_per_m: float,
    gates: StreetGates = DEFAULT_GATES,
) -> list[StreetSegments]:
    """Street constraints for a page: family-merged, locality-trimmed, aspect-filtered.

    Near-square label boxes are dropped here: their long axis is set by the box, not
    the text, so their angle — this channel's most trusted signal — is noise.
    """
    usable = [
        feature
        for feature in features
        if feature.short_side <= 0
        or feature.long_side / feature.short_side >= gates.min_aspect
    ]
    constraints = constraints_from_features(
        usable,
        block_index,
        origin=prior.center,
        label_size=label_size,
        working_size=working_size,
        merge_families=True,
        # Wide enough that a street crossing the page is never cut mid-page: the trim
        # exists to keep a distant branch of a merged family from capturing a label,
        # not to truncate the page's own streets.
        locality=(
            (0.0, 0.0),
            prior.radius_m
            + math.hypot(*working_size) / 2.0 / max(scale_px_per_m, 1e-6),
        ),
    )
    return [
        (
            center,
            dir_pix,
            name,
            *extend_terminal_segments(starts, ends, gates.terminal_extrapolation_m),
        )
        for center, dir_pix, name, starts, ends in constraints
    ]
