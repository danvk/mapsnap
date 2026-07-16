"""Tests for the edge-join pose-graph solver."""

import math

import numpy as np

from mapsnap.edge_join_graph import (
    AbsolutePrior,
    EdgeHypotheses,
    RelativeMeasurement,
    VolumeFrame,
    solve_pose_graph,
    solve_pose_graph_hypotheses,
    spanning_tree_initialization,
    wrap_radians,
)

FRAME = VolumeFrame(lon0=-77.0, lat0=38.9, scale_m_per_px=0.24)


def test_pose_affine_round_trip() -> None:
    affine = FRAME.pose_to_affine(1200.0, -800.0, math.radians(12.0))
    x, y, theta = FRAME.affine_to_pose(affine)
    assert abs(x - 1200.0) < 1e-6
    assert abs(y + 800.0) < 1e-6
    assert abs(math.degrees(theta) - 12.0) < 1e-9
    # An unrotated pose maps page +x to east and +y to south (lat decreases).
    a0 = FRAME.pose_to_affine(0.0, 0.0, 0.0)
    assert a0[0, 0] > 0 and abs(a0[0, 1]) < 1e-12
    assert a0[1, 1] < 0


def test_relative_measurement_consistency() -> None:
    anchor = (100.0, 50.0, math.radians(30.0))
    target = (400.0, 250.0, math.radians(120.0))
    dx, dy, dtheta = FRAME.relative(anchor, target)
    # Rebuild the target from anchor + relative and compare.
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    rebuilt = (
        anchor[0] + c * dx - s * dy,
        anchor[1] + s * dx + c * dy,
        wrap_radians(anchor[2] + dtheta),
    )
    assert np.allclose(rebuilt, target)


def make_square_graph(noise: float = 0.0):
    """Four pages on a 400m square, page 0 anchored, ring of measurements."""
    truth = np.array(
        [
            [0.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
            [400.0, 400.0, math.radians(90.0)],
            [0.0, 400.0, 0.0],
        ]
    )
    rng = np.random.default_rng(3)
    measurements = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
        dx, dy, dt = VolumeFrame(0, 0, 1).relative(tuple(truth[a]), tuple(truth[b]))
        measurements.append(
            RelativeMeasurement(
                a,
                b,
                dx + rng.normal(0, noise),
                dy + rng.normal(0, noise),
                dt,
                sigma_pos_m=5.0,
                sigma_theta_rad=math.radians(0.5),
            )
        )
    priors = [AbsolutePrior(0, 0.0, 0.0, sigma_pos_m=1.0, theta=0.0)]
    return truth, measurements, priors


def test_solver_recovers_square() -> None:
    truth, measurements, priors = make_square_graph(noise=2.0)
    rng = np.random.default_rng(5)
    initial = truth.copy()
    initial[:, :2] += rng.normal(0, 30.0, (len(truth), 2))
    initial[:, 2] += rng.normal(0, math.radians(4.0), len(truth))
    solved, _ = solve_pose_graph(initial, measurements, priors)
    assert np.abs(solved[:, :2] - truth[:, :2]).max() < 8.0
    theta_err = [abs(wrap_radians(s - t)) for s, t in zip(solved[:, 2], truth[:, 2])]
    assert max(theta_err) < math.radians(1.5)


def test_solver_outvotes_gross_outlier() -> None:
    truth, measurements, priors = make_square_graph(noise=1.0)
    # A wildly wrong extra measurement (one-block slide) on a redundant edge.
    bad = RelativeMeasurement(
        0, 2, 700.0, 120.0, 0.0, sigma_pos_m=5.0, sigma_theta_rad=math.radians(0.5)
    )
    solved, _ = solve_pose_graph(truth.copy(), measurements + [bad], priors)
    assert np.abs(solved[:, :2] - truth[:, :2]).max() < 10.0


def test_hypotheses_solver_reassigns_aliased_slide() -> None:
    """The winner candidate on one edge is a 120m grid slide; truth is rank 2.

    With the rest of the square consistent, the EM assignment loop must
    re-pick the alternate and land the page at its true pose.
    """
    truth, measurements, priors = make_square_graph(noise=1.0)
    edges = []
    for m in measurements:
        if (m.a, m.b) == (1, 2):
            slide = RelativeMeasurement(
                m.a,
                m.b,
                m.dx + 120.0,
                m.dy,
                m.dtheta,
                sigma_pos_m=m.sigma_pos_m,
                sigma_theta_rad=m.sigma_theta_rad,
            )
            edges.append(EdgeHypotheses(m.a, m.b, [slide, m]))
        else:
            edges.append(EdgeHypotheses(m.a, m.b, [m]))
    initial = truth.copy()
    initial[2, 0] += 120.0  # start at the aliased pose
    solved, assignment, active, _ = solve_pose_graph_hypotheses(initial, edges, priors)
    slide_index = next(i for i, e in enumerate(edges) if len(e.candidates) == 2)
    assert assignment[slide_index] == 1
    assert all(active)
    assert np.abs(solved[:, :2] - truth[:, :2]).max() < 8.0


def test_hypotheses_solver_trims_unsupported_edge() -> None:
    """An edge whose only candidate is grossly wrong gets trimmed, not fit."""
    truth, measurements, priors = make_square_graph(noise=1.0)
    bad = RelativeMeasurement(
        0, 2, 700.0, 120.0, 0.0, sigma_pos_m=5.0, sigma_theta_rad=math.radians(0.5)
    )
    edges = [EdgeHypotheses(m.a, m.b, [m]) for m in measurements]
    edges.append(EdgeHypotheses(0, 2, [bad]))
    solved, _, active, diag = solve_pose_graph_hypotheses(truth.copy(), edges, priors)
    assert active[:-1] == [True] * (len(edges) - 1)
    assert active[-1] is False or diag["trimmed"] >= 1
    assert np.abs(solved[:, :2] - truth[:, :2]).max() < 8.0


def test_spanning_tree_initialization_reaches_chain() -> None:
    truth, measurements, _ = make_square_graph()
    init = spanning_tree_initialization(
        5,
        {0: tuple(truth[0])},
        measurements,
        fallback={4: (999.0, 999.0, 0.0)},
    )
    assert np.abs(init[:4, :2] - truth[:, :2]).max() < 1e-6
    assert tuple(init[4]) == (999.0, 999.0, 0.0)  # unreachable page falls back
