"""Tests for the predict-then-verify prototype's core math."""

import math

import numpy as np

from mapsnap.georef_from_labels import LabelFeature
from mapsnap.predict_fit import (
    Frame,
    Match,
    Similarity,
    angle_diff_mod_pi,
    polyline_nearest,
    solve_wls,
    vote_rotation,
)


def make_label(text: str, center: tuple[float, float], dir_pix: float) -> LabelFeature:
    return LabelFeature(text, text, center, dir_pix, 100.0, 20.0)


def test_similarity_prior_maps_center_to_origin_and_roundtrips():
    T = Similarity.from_prior(0.66, math.radians(30), (800.0, 1000.0))
    assert np.allclose(T.apply(800.0, 1000.0), (0.0, 0.0), atol=1e-9)
    assert math.isclose(T.scale_ft_per_px(), 0.66)
    assert math.isclose(T.rotation(), math.radians(30))


def test_similarity_maps_image_direction_to_theta_minus_delta():
    # An image-space direction delta must map to world angle theta - delta.
    theta, delta = math.radians(25), math.radians(70)
    T = Similarity.from_prior(1.0, theta, (0.0, 0.0))
    dx, dy = T.apply(math.cos(delta), math.sin(delta))
    assert math.isclose(math.atan2(dy, dx), theta - delta, abs_tol=1e-9)


def test_polyline_nearest_clamps_to_segment():
    line = np.array([[0.0, 0.0], [100.0, 0.0]])
    dist, point, bearing = polyline_nearest(line, (50.0, 30.0))
    assert math.isclose(dist, 30.0) and point == (50.0, 0.0)
    dist, point, _ = polyline_nearest(line, (150.0, 0.0))
    assert math.isclose(dist, 50.0) and point == (100.0, 0.0)  # clamped to the end
    assert math.isclose(bearing, 0.0)


def test_frame_roundtrip():
    frame = Frame.at(-83.0, 42.36)
    x, y = frame.to_xy(-82.99, 42.365)
    lon, lat = frame.to_lonlat(x, y)
    assert math.isclose(lon, -82.99) and math.isclose(lat, 42.365)


def test_vote_rotation_finds_the_true_rotation():
    # Two labels on perpendicular streets under a 20-degree page rotation: the vote
    # theta = bearing + label_dir must peak at ~20 degrees for both.
    theta = math.radians(20)
    labels = [
        (make_label("A", (0, 0), (theta - 0.0) % math.pi), 1.0),  # E-W street: phi=0
        (
            make_label("B", (0, 0), (theta - math.pi / 2) % math.pi),
            1.0,
        ),  # N-S: phi=90
    ]
    blocks = {
        "A": [np.array([[-1000.0, 0.0], [1000.0, 0.0]])],
        "B": [np.array([[0.0, -1000.0], [0.0, 1000.0]])],
    }
    peaks = vote_rotation(labels, blocks)
    assert peaks and angle_diff_mod_pi(peaks[0], theta) < math.radians(6)


def test_solve_wls_recovers_a_known_transform():
    # Three labels on three street lines, generated from a ground-truth transform.
    truth = Similarity.from_prior(0.66, math.radians(15), (1000.0, 1200.0))
    lines = [  # (a point on the line, bearing)
        ((0.0, 300.0), 0.0),  # horizontal line y=300
        ((0.0, -400.0), 0.0),  # horizontal line y=-400
        ((250.0, 0.0), math.pi / 2),  # vertical line x=250
    ]

    def invert(x: float, y: float) -> tuple[float, float]:
        s2 = truth.p**2 + truth.q**2
        dx, dy = x - truth.tx, y - truth.ty
        return ((truth.p * dx + truth.q * dy) / s2, (truth.q * dx - truth.p * dy) / s2)

    matches = []
    # World points on each line, spread out so the pixels are non-collinear.
    on_lines = [(-600.0, 300.0), (500.0, -400.0), (250.0, 650.0)]
    for i, ((lx, ly), bearing) in enumerate(lines):
        on_line = on_lines[i]
        u, v = invert(*on_line)  # a pixel whose truth-projection IS on the line
        # Label direction consistent with the street under the truth rotation.
        dir_pix = (truth.rotation() - bearing) % math.pi
        matches.append(
            Match(make_label(f"S{i}", (u, v), dir_pix), 1.0, on_line, bearing, 10.0)
        )
    solved = solve_wls(
        matches,
        (1000.0, 1200.0),
        position_sigma_ft=1500.0,
        prior_scale_ft_per_px=0.66,
    )
    assert solved is not None
    assert math.isclose(solved.scale_ft_per_px(), 0.66, rel_tol=0.05)
    assert angle_diff_mod_pi(solved.rotation(), math.radians(15)) < math.radians(3)
