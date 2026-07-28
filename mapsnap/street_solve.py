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
    pose_world_of,
    solve_pose,
)
from mapsnap.streets import Block, street_name_family

Point = tuple[float, float]


@dataclass(frozen=True)
class StreetGates:
    """Every tunable of the streets-only solver, in one sweepable place."""

    angle_gate_deg: float = 8.0  # inlier: |label bearing - street bearing| at the pose
    position_gate_m: float = 80.0  # inlier: label center to street distance (G3)
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
    sigma_line_m: float = 60.0  # positions stay loose: label centers sit off the line
    sigma_angle_deg: float = 3.0  # angles are the reliable signal
    sigma_log_scale: float = 0.05  # G5: soft scale prior (bounded in solve_pose)
    sigma_centroid_m: float = 400.0  # the location prior is a neighborhood, not a fix
    max_psi_seeds: int = 8  # bearing clusters tried per page (evidence, not a sweep)
    max_scale_seeds: int = 5  # distinct scale proposals tried per bearing


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


def propose_translation(
    first: StreetSegments,
    second: StreetSegments,
    line_a: tuple[np.ndarray, np.ndarray],
    line_b: tuple[np.ndarray, np.ndarray],
    psi_deg: float,
    log_scale: float,
    size: tuple[int, int],
) -> StreetPose | None:
    """The pose placing two labels exactly on two given (non-parallel) street lines.

    With bearing and scale fixed, a label lying on a known line is one linear equation
    in the translation; two independent lines determine it outright.
    """
    point_a, direction_a = line_a
    point_b, direction_b = line_b
    normal_a = np.array([direction_a[1], -direction_a[0]])
    normal_b = np.array([direction_b[1], -direction_b[0]])
    matrix = np.array([normal_a, normal_b])
    if abs(float(np.linalg.det(matrix))) < 1e-6:
        return None
    base = (0.0, 0.0, psi_deg, log_scale)
    offset_a = pose_world_of(base, first[0], size)
    offset_b = pose_world_of(base, second[0], size)
    rhs = np.array(
        [
            float(normal_a @ (point_a - offset_a)),
            float(normal_b @ (point_b - offset_b)),
        ]
    )
    try:
        translation = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    return (float(translation[0]), float(translation[1]), psi_deg, log_scale)


def consensus_pose(
    constraints: list[StreetSegments],
    psi_deg: float,
    log_scales: list[tuple[float, str]],
    size: tuple[int, int],
    gates: StreetGates,
    lines: list[list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> tuple[StreetPose, list[StreetSegments], str] | None:
    """Best (pose, inliers, scale source) over all constraint-pair/line/scale proposals."""
    best: tuple[tuple[int, float], StreetPose, list[StreetSegments], str] | None = None
    if lines is None:
        lines = [distinct_lines(c[3], c[4]) for c in constraints]
    for log_scale, scale_source in log_scales:
        for i, first in enumerate(constraints):
            for j in range(i + 1, len(constraints)):
                second = constraints[j]
                for line_a in lines[i]:
                    for line_b in lines[j]:
                        bearing_a = math.degrees(math.atan2(*line_a[1])) % 180.0
                        bearing_b = math.degrees(math.atan2(*line_b[1])) % 180.0
                        if (
                            abs(angle_difference_mod180(bearing_a, bearing_b))
                            < gates.bearing_diversity_deg
                        ):
                            continue
                        pose = propose_translation(
                            first, second, line_a, line_b, psi_deg, log_scale, size
                        )
                        if pose is None:
                            continue
                        inliers = [
                            c for c in constraints if is_inlier(pose, c, size, gates)
                        ]
                        if len(inliers) < gates.min_inliers:
                            continue
                        if len(inliers) < gates.min_inlier_fraction * len(constraints):
                            # Kansas City p576 slid a block on two streets while four
                            # others disagreed; a correct pose explains most of them.
                            continue
                        if distinct_streets(inliers) < gates.min_distinct_streets:
                            continue
                        if bearing_spread(pose, inliers) < gates.bearing_diversity_deg:
                            continue
                        mean_position = float(
                            np.mean([residuals_at(pose, c, size)[0] for c in inliers])
                        )
                        score = (len(inliers), -mean_position)
                        if best is None or score > best[0]:
                            best = (score, pose, inliers, scale_source)
    if best is None:
        return None
    return best[1], best[2], best[3]


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
        proposal = consensus_pose(
            constraints,
            psi,
            scale_candidates(constraints, lines, psi, prior_log_scale, size, gates),
            size,
            gates,
            lines,
        )
        if proposal is None:
            continue
        seed, inliers, scale_source = proposal
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
                leave_one_out_spread(pose, inliers, size, prior_log_scale, sigmas, ring)
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
        score = (len(inliers) if attempt.pose else 0, -move)
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
    return constraints_from_features(
        usable,
        block_index,
        origin=prior.center,
        label_size=label_size,
        working_size=working_size,
        merge_families=True,
        locality=((0.0, 0.0), prior.radius_m + gates.position_gate_m),
    )
