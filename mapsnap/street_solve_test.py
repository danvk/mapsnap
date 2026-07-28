import math

import numpy as np

from mapsnap.keymap.align_page_region import (
    init_pose_from_model,
    pose_world_of,
)
from mapsnap.street_solve import (
    StreetGates,
    bearing_spread,
    consensus_pose,
    distinct_lines,
    psi_from_theta,
    psi_votes,
    residuals_at,
    solve_streets_pose,
)

SIZE = (1000, 1400)
SCALE = 2.0  # pixels per metre
LOG_SCALE = math.log(SCALE)


def street_along(
    point_m: tuple[float, float], bearing_deg: float, length_m: float = 600.0
) -> tuple[np.ndarray, np.ndarray]:
    """A straight street's segment soup, centred on a point, at a world bearing."""
    direction = np.array(
        [math.sin(math.radians(bearing_deg)), math.cos(math.radians(bearing_deg))]
    )
    center = np.array(point_m)
    steps = np.linspace(-length_m / 2, length_m / 2, 13)
    points = center + steps[:, None] * direction
    return points[:-1], points[1:]


def constraint_for(
    pose, name: str, pixel: tuple[float, float], bearing_deg: float, offset_m=(0.0, 0.0)
):
    """A constraint whose street passes through the pixel's true world position.

    ``offset_m`` displaces the street from that position, so a test can plant a
    constraint that the pose cannot satisfy.
    """
    world = pose_world_of(pose, pixel, SIZE)
    starts, ends = street_along(
        (world[0] + offset_m[0], world[1] + offset_m[1]), bearing_deg
    )
    dir_pix = math.radians((bearing_deg - pose[2] - 90.0) % 180.0)
    return (pixel, dir_pix, name, starts, ends)


TRUE_POSE = (120.0, -80.0, 25.0, LOG_SCALE)


def scene(pose=TRUE_POSE):
    """Two parallel streets (bearing 25) plus one crossing street (bearing 115)."""
    return [
        constraint_for(pose, "FIRST", (250.0, 400.0), 25.0),
        constraint_for(pose, "SECOND", (750.0, 400.0), 25.0),
        constraint_for(pose, "CROSS", (500.0, 1100.0), 115.0),
    ]


def test_psi_from_theta_matches_pose_convention():
    # A pixel delta maps to a raster-frame angle of page_angle + psi, so theta negates.
    for psi in (0.0, 25.0, -70.0, 170.0):
        pose = (0.0, 0.0, psi, LOG_SCALE)
        start = pose_world_of(pose, (500.0, 700.0), SIZE)
        end = pose_world_of(pose, (600.0, 700.0), SIZE)  # page angle 0
        raster = math.degrees(math.atan2(-(end[1] - start[1]), end[0] - start[0]))
        theta = 0.0 - raster
        assert abs((psi_from_theta(theta) - psi + 180) % 360 - 180) < 1e-6


def test_psi_matches_init_pose_from_model():
    # The same bearing convention the region placer derives from its similarity model.
    for psi in (0.0, 40.0, -110.0):
        pose = (0.0, 0.0, psi, LOG_SCALE)
        center = pose_world_of(pose, (SIZE[0] / 2, SIZE[1] / 2), SIZE)
        up = pose_world_of(pose, (SIZE[0] / 2, SIZE[1] / 2 - 1.0), SIZE)
        model_psi = math.degrees(math.atan2(up[0] - center[0], up[1] - center[1]))
        assert abs((model_psi - psi + 180) % 360 - 180) < 1e-6


def test_distinct_lines_collapses_a_straight_street():
    starts, ends = street_along((0.0, 0.0), 40.0)
    assert len(starts) == 12
    assert len(distinct_lines(starts, ends)) == 1


def test_recovers_pose_from_two_parallels_and_a_crossing():
    result = solve_streets_pose(
        scene(),
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
    )
    assert result.pose is not None, result.abstain
    assert result.n_inliers == 3
    assert (
        math.hypot(result.pose[0] - TRUE_POSE[0], result.pose[1] - TRUE_POSE[1]) < 5.0
    )
    assert abs(result.pose[2] - TRUE_POSE[2]) < 0.5


def test_scale_recovered_from_parallel_spacing_without_a_good_prior():
    # The prior is off by 40% and effectively unweighted: the two parallel streets'
    # spacing has to carry the scale on its own, which is the claim in issue #168.
    result = solve_streets_pose(
        scene(),
        size=SIZE,
        prior_log_scale=LOG_SCALE + math.log(1.4),
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
        gates=StreetGates(sigma_log_scale=5.0),
    )
    assert result.pose is not None, result.abstain
    assert result.scale_source == "parallel-pair"
    assert abs(result.pose[3] - LOG_SCALE) < 0.05


def test_scale_prior_strength_trades_against_the_streets():
    # With a tight prior the same scene keeps the (wrong) prior scale: the sigma is a
    # real knob, not decoration -- the harness sweeps it.
    tight = solve_streets_pose(
        scene(),
        size=SIZE,
        prior_log_scale=LOG_SCALE + math.log(1.4),
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
        gates=StreetGates(sigma_log_scale=0.01),
    )
    assert tight.pose is not None, tight.abstain
    assert abs(tight.pose[3] - (LOG_SCALE + math.log(1.4))) < 0.05


def test_rejects_a_wrong_street_a_kilometre_away():
    # LA p1499J's BROOKLYN PLACE: right name, renamed street, matched 1 km off.
    planted = scene() + [
        constraint_for(
            TRUE_POSE, "RENAMED", (500.0, 700.0), 60.0, offset_m=(900.0, 500.0)
        )
    ]
    result = solve_streets_pose(
        planted,
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
    )
    assert result.pose is not None, result.abstain
    assert result.n_inliers == 3
    dropped = [d for d in result.diagnostics if not d.inlier]
    assert [d.name for d in dropped] == ["RENAMED"]
    assert dropped[0].position_m is not None and dropped[0].position_m > 500.0


def test_rejects_a_rerouted_street_whose_angle_disagrees():
    # Kansas City's GILLHAM: the street is there, but its drawn geometry is not OSM's.
    planted = scene() + [
        constraint_for(TRUE_POSE, "REROUTED", (300.0, 900.0), 25.0 + 40.0)
    ]
    planted[-1] = (planted[-1][0], math.radians(0.0), "REROUTED", *planted[-1][3:])
    result = solve_streets_pose(
        planted,
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
    )
    assert result.pose is not None, result.abstain
    assert "REROUTED" not in [d.name for d in result.diagnostics if d.inlier]


def test_abstains_without_a_rotation_prior():
    result = solve_streets_pose(
        scene(), size=SIZE, prior_log_scale=LOG_SCALE, psi_priors=[]
    )
    assert result.pose is None and result.abstain == "no-rotation-prior"


def test_abstains_with_too_few_constraints():
    result = solve_streets_pose(
        scene()[:2],
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
    )
    assert result.pose is None and result.abstain == "too-few-constraints"


def test_abstains_when_every_street_is_parallel():
    # Three parallels leave the along-street translation free: no diverse pair exists.
    parallel = [
        constraint_for(TRUE_POSE, "A", (200.0, 300.0), 25.0),
        constraint_for(TRUE_POSE, "B", (500.0, 300.0), 25.0),
        constraint_for(TRUE_POSE, "C", (800.0, 300.0), 25.0),
    ]
    result = solve_streets_pose(
        parallel,
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
    )
    assert result.pose is None and result.abstain == "no-consensus"


def test_abstains_outside_the_prior_radius():
    result = solve_streets_pose(
        scene(),
        size=SIZE,
        prior_log_scale=LOG_SCALE,
        psi_priors=[(TRUE_POSE[2], "label-pair-exact")],
        prior_radius_m=50.0,  # the true centre sits ~144 m from the prior
    )
    assert result.pose is None and result.abstain == "outside-prior-radius"


def test_truncated_street_produces_a_large_residual_not_a_plausible_one():
    # G2/G3: OSM keeps only a stub of a street the sheet drew in full. Snapping is
    # clipped to the stub, so the label lands far from it and the gate drops it.
    stub_pose = TRUE_POSE
    world = pose_world_of(stub_pose, (500.0, 200.0), SIZE)
    starts, ends = street_along((world[0], world[1] + 400.0), 25.0, length_m=40.0)
    truncated = ((500.0, 200.0), math.radians(0.0), "STUB", starts, ends)
    position, _angle = residuals_at(stub_pose, truncated, SIZE)
    assert position > 300.0
    gates = StreetGates()
    assert position > gates.position_gate_m


def test_bearing_spread_measures_the_widest_pair():
    poses = scene()
    spread = bearing_spread(TRUE_POSE, poses)
    assert 85.0 < spread <= 90.0


def test_consensus_returns_none_without_a_diverse_pair():
    parallel = [
        constraint_for(TRUE_POSE, "A", (200.0, 300.0), 25.0),
        constraint_for(TRUE_POSE, "B", (500.0, 300.0), 25.0),
    ]
    assert (
        consensus_pose(
            parallel, TRUE_POSE[2], [(LOG_SCALE, "prior")], SIZE, StreetGates()
        )
        is None
    )


def test_init_pose_from_model_is_importable_for_convention_checks():
    # Guards the refactor: the region placer's seed helper stays public.
    assert callable(init_pose_from_model)


def test_psi_votes_finds_the_true_bearing():
    # Every correct constraint votes for the page's bearing; the flip is a separate
    # candidate rather than being averaged in (they differ by 180, not by noise).
    votes = psi_votes(scene())
    assert votes, "no bearing votes"
    best = votes[0][0]
    assert min(abs(best - TRUE_POSE[2]), abs(best - (TRUE_POSE[2] + 180))) < 1.0
    assert any(abs(v - TRUE_POSE[2]) < 1.0 for v, _ in votes)


def test_psi_votes_outvote_a_single_wrong_street():
    # Three streets agree on the bearing, one rogue is 30 degrees off: the majority
    # cluster wins, and the rogue survives only as a lower-ranked seed.
    planted = scene() + [constraint_for(TRUE_POSE, "ROGUE", (400.0, 800.0), 55.0)]
    planted[-1] = (planted[-1][0], math.radians(0.0), "ROGUE", *planted[-1][3:])
    votes = psi_votes(planted)
    assert any(abs(v - TRUE_POSE[2]) < 1.0 for v, _ in votes[:2])
