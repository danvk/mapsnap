"""Tests for key-map snap (#211)."""

import math

import numpy as np
import pytest

from mapsnap.keymap.snap import (
    SnapTarget,
    affine_corners,
    affine_m_per_px,
    affine_theta_deg,
    as_3x3,
    keymap_model,
    linear_part_metres,
    local_tangent,
    match_page,
    page_world_affine_from_match,
    stretched_crop,
    thin_plate_spline,
    unit_stretch,
)

LATITUDE = 42.36


def rotation(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array(
        [
            [math.cos(radians), -math.sin(radians)],
            [math.sin(radians), math.cos(radians)],
        ]
    )


def test_unit_stretch_has_unit_determinant():
    linear = np.diag([1.1, 0.9]) @ rotation(17.0)
    stretch, _ = unit_stretch(linear)
    assert np.linalg.det(stretch) == pytest.approx(1.0)


def test_unit_stretch_is_identity_when_isotropic():
    """A rotation-plus-uniform-scale map has no shape to remove."""
    stretch, anisotropy = unit_stretch(3.5 * rotation(31.0))
    assert stretch == pytest.approx(np.eye(2), abs=1e-9)
    assert anisotropy == pytest.approx(0.0, abs=1e-9)


def test_unit_stretch_reports_anisotropy():
    stretch, anisotropy = unit_stretch(np.diag([1.106, 1.0]))
    assert anisotropy == pytest.approx(10.6, abs=0.05)
    assert stretch == pytest.approx(np.diag([math.sqrt(1.106), 1 / math.sqrt(1.106)]))


def test_unit_stretch_makes_the_map_a_similarity():
    """The whole point: L @ inv(W) must be a scalar times a rotation.

    That is the condition under which searching one scale and one rotation can
    reach the correct pose at all.
    """
    linear = np.array([[0.87, 0.12], [-0.31, 0.79]])
    stretch, _ = unit_stretch(linear)
    residual = linear @ np.linalg.inv(stretch)
    scale = math.sqrt(abs(np.linalg.det(residual)))
    assert residual.T @ residual == pytest.approx(scale**2 * np.eye(2), abs=1e-9)


def test_unit_stretch_never_flips():
    """A stretch that reflected the crop would break template matching."""
    stretch, _ = unit_stretch(np.array([[0.0, 1.3], [0.8, 0.0]]))
    assert np.linalg.det(stretch) > 0


def test_thin_plate_spline_reproduces_an_affine():
    """With affine correspondences the spline's warp term stays out of the way."""
    source = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0], [50.0, 50.0]]
    )
    affine = np.array([[2.0, 0.3, 5.0], [-0.2, 1.7, -8.0]])
    destination = np.column_stack(
        [
            affine[0, 0] * source[:, 0] + affine[0, 1] * source[:, 1] + affine[0, 2],
            affine[1, 0] * source[:, 0] + affine[1, 1] * source[:, 1] + affine[1, 2],
        ]
    )
    spline = thin_plate_spline(source, destination, smoothing=1e-6)
    query = np.array([[25.0, 75.0]])
    expected = np.array(
        [
            [
                affine[0, 0] * 25 + affine[0, 1] * 75 + affine[0, 2],
                affine[1, 0] * 25 + affine[1, 1] * 75 + affine[1, 2],
            ]
        ]
    )
    assert spline(query) == pytest.approx(expected, abs=1e-6)


def synthetic_georef(anisotropy: float = 1.0, gcps: int = 40) -> dict:
    """A key-map georef whose pixel->world map is stretched by `anisotropy` in x."""
    width, height = 4000, 3000
    metres_per_px = 0.8
    lon_scale = 111_320.0 * math.cos(math.radians(LATITUDE))
    lon0, lat0 = -83.0, LATITUDE

    def to_lonlat(px, py):
        east = px * metres_per_px * anisotropy
        north = -py * metres_per_px
        return lon0 + east / lon_scale, lat0 + north / 110_540.0

    corners = [
        to_lonlat(x, y) for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    rng = np.random.default_rng(0)
    intersections = []
    for _ in range(gcps):
        px, py = rng.uniform(0, width), rng.uniform(0, height)
        lon, lat = to_lonlat(px, py)
        intersections.append({"x": px, "y": py, "lon": lon, "lat": lat, "inlier": True})
    return {
        "width": width,
        "height": height,
        "corners": [list(c) for c in corners],
        "intersections": intersections,
    }


def test_keymap_model_falls_back_to_the_affine_without_enough_gcps():
    georef = synthetic_georef(gcps=3)
    model = keymap_model(georef)
    lon, lat = model(np.array([[0.0, 0.0]]))[0]
    assert (lon, lat) == pytest.approx(tuple(georef["corners"][0]), abs=1e-9)


def test_thin_plate_spline_interpolates_when_barely_smoothed():
    """Unregularized, the spline passes through every point it is given."""
    rng = np.random.default_rng(11)
    source = rng.uniform(0, 4000, size=(30, 2))
    destination = source * 0.8 + rng.normal(0, 15.0, size=(30, 2))
    spline = thin_plate_spline(source, destination, smoothing=1e-3)
    assert spline(source) == pytest.approx(destination, abs=0.5)


def test_default_smoothing_does_not_chase_its_gcps():
    """The default deliberately regularizes, and that is load-bearing.

    Key-map intersections carry real residuals; an interpolating spline would
    reproduce them as ripples in the warp. The invariant worth pinning is the
    ordering: the smoothed spline stays much closer to its GCPs than a plain
    affine does, without landing on them.

    Note that `smoothing` is in source-coordinate units, so its effective
    strength depends on sheet size. It is tuned for the 5500-8400 px sheets in
    this corpus (Detroit: 88 ft affine residual, 32 ft at this setting).
    """
    rng = np.random.default_rng(5)
    source = rng.uniform(0, 6000, size=(200, 2))
    exact = np.column_stack([source[:, 0] * 0.8, -source[:, 1] * 0.8])
    destination = exact + rng.normal(0, 20.0, size=(200, 2))

    smoothed = np.linalg.norm(
        thin_plate_spline(source, destination)(source) - destination, axis=1
    )
    design = np.column_stack([source[:, 0], source[:, 1], np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, destination, rcond=None)
    affine_residual = np.linalg.norm(design @ coefficients - destination, axis=1)

    assert np.median(smoothed) > 0.5
    assert np.median(smoothed) < np.median(affine_residual)


def test_keymap_model_stays_near_its_gcps():
    """The model lands near the GCPs it was built from, in metres."""
    georef = synthetic_georef()
    model = keymap_model(georef)
    lon_scale = 111_320.0 * math.cos(math.radians(LATITUDE))
    errors = []
    for intersection in georef["intersections"]:
        lon, lat = model(np.array([[intersection["x"], intersection["y"]]]))[0]
        errors.append(
            math.hypot(
                (lon - intersection["lon"]) * lon_scale,
                (lat - intersection["lat"]) * 110_540.0,
            )
        )
    assert float(np.median(errors)) < 5.0


def test_local_tangent_recovers_scale_and_rotation():
    georef = synthetic_georef()
    tangent = local_tangent(keymap_model(georef), (2000.0, 1500.0))
    assert affine_m_per_px(tangent) == pytest.approx(0.8, rel=0.02)
    assert affine_theta_deg(tangent) == pytest.approx(0.0, abs=0.5)


def test_stretched_crop_removes_the_sheet_anisotropy():
    """A 10% stretched sheet reports it, and its corrected frame is isotropic."""
    georef = synthetic_georef(anisotropy=1.10)
    model = keymap_model(georef)
    probability = np.zeros((3000, 4000), np.float32)
    crop = stretched_crop(
        probability, model, SnapTarget((2000.0, 1500.0), 0.2, 0.0), (600, 500)
    )
    assert crop is not None
    assert crop.anisotropy_pct == pytest.approx(10.0, abs=0.6)

    corrected = local_tangent(model, (2000.0, 1500.0))
    linear = linear_part_metres(corrected, LATITUDE)
    stretch, _ = unit_stretch(linear)
    residual = linear @ np.linalg.inv(stretch)
    singular = np.linalg.svd(residual, compute_uv=False)
    assert singular[0] / singular[1] == pytest.approx(1.0, abs=1e-6)


def test_stretched_crop_returns_none_off_sheet():
    georef = synthetic_georef()
    model = keymap_model(georef)
    probability = np.zeros((3000, 4000), np.float32)
    assert (
        stretched_crop(
            probability, model, SnapTarget((-9000.0, -9000.0), 0.2, 0.0), (600, 500)
        )
        is None
    )


def test_match_page_finds_a_planted_page():
    """A page cut out of the key map is found back where it came from."""
    rng = np.random.default_rng(3)
    keymap = np.zeros((1400, 1600), np.float32)
    for x in range(60, 1600, 130):
        keymap[:, x : x + 9] = 1.0
    for y in range(50, 1400, 145):
        keymap[y : y + 9, :] = 1.0
    keymap += rng.normal(0, 0.01, keymap.shape).astype(np.float32)

    georef = synthetic_georef()
    model = keymap_model(georef)
    # The page is the key-map neighbourhood around (760, 690), at page scale.
    target = SnapTarget(centre=(760.0, 690.0), m_per_px=0.8, theta_deg=0.0)
    page = keymap[690 - 200 : 690 + 200, 760 - 240 : 760 + 240].copy()

    crop = stretched_crop(keymap, model, target, page.shape)
    assert crop is not None
    match = match_page(crop, page, target)
    assert match is not None

    affine = page_world_affine_from_match(match, model, page.shape)
    corners = affine_corners(affine, page.shape)
    expected_lon, expected_lat = model(np.array([[760.0 - 240, 690.0 - 200]]))[0]
    assert corners[0][0] == pytest.approx(expected_lon, abs=2e-4)
    assert corners[0][1] == pytest.approx(expected_lat, abs=2e-4)


def test_as_3x3_round_trips():
    affine = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert as_3x3(affine)[:2, :] == pytest.approx(affine)
    assert as_3x3(affine)[2].tolist() == [0.0, 0.0, 1.0]
