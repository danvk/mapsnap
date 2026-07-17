"""Synthetic-fixture tests for the edge-join matcher."""

import cv2
import numpy as np

from mapsnap.edge_join import (
    MatchParams,
    chamfer_refine,
    compose,
    dominant_orientation_deg,
    masked_ncc,
    match_pair,
    refine_and_rank,
    rotated_bounds,
    top_peaks,
)

RNG = np.random.default_rng(7)


def road_grid(
    shape: tuple[int, int],
    verticals: list[int],
    horizontals: list[int],
    thickness: int = 6,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """A synthetic P(road) map: a line grid, optionally rotated about center."""
    img = np.zeros(shape, np.float32)
    for x in verticals:
        cv2.line(img, (x, -1000), (x, shape[0] + 1000), 1.0, thickness)
    for y in horizontals:
        cv2.line(img, (-1000, y), (shape[1] + 1000, y), 1.0, thickness)
    if angle_deg:
        center = (shape[1] / 2, shape[0] / 2)
        rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        img = cv2.warpAffine(img, rot, (shape[1], shape[0]))
    return cv2.GaussianBlur(img, (0, 0), 2)


# An IRREGULAR grid (unambiguous under translation, unlike a uniform lattice).
IRREGULAR_V = [40, 95, 180, 230, 340]
IRREGULAR_H = [50, 150, 190, 300, 355]


def extract_page(
    world: np.ndarray, pose: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Sample a 'page' image from the world raster given page->world pose."""
    inverse = cv2.invertAffineTransform(pose)
    return cv2.warpAffine(world, inverse, size)


def pose_corner_error(
    pose_a: np.ndarray, pose_b: np.ndarray, size: tuple[int, int]
) -> float:
    """Max distance between page-corner images under two poses (raster px)."""
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    return float(np.abs(corners @ (pose_a - pose_b).T).max())


def test_dominant_orientation() -> None:
    # cv2 angle +20 rotates content by -20 in array-frame atan2 terms, so the
    # corridors sit at -20 == 70 (mod 90).
    grid = road_grid((400, 400), IRREGULAR_V, IRREGULAR_H, angle_deg=20.0)
    angle = dominant_orientation_deg(grid)
    assert min(abs(angle - 70.0), abs(angle - 70.0 - 90), abs(angle - 70 + 90)) < 2.0


def test_masked_ncc_finds_known_shift() -> None:
    world = road_grid((300, 300), IRREGULAR_V, IRREGULAR_H)
    fixed_mask = np.ones_like(world)
    moving = world[40:200, 60:240]
    moving_mask = np.ones_like(moving)
    score = masked_ncc(world, fixed_mask, moving, moving_mask, min_overlap_px=1000)
    peaks = top_peaks(score, 1, 10)
    assert peaks
    _, row, col = peaks[0]
    mh, mw = moving.shape
    assert abs((row - mh + 1) - 40) <= 1
    assert abs((col - mw + 1) - 60) <= 1


def test_match_pair_recovers_known_pose() -> None:
    world = road_grid((500, 500), IRREGULAR_V, IRREGULAR_H)
    theta = 12.0
    scale = 0.8
    true_pose = cv2.getRotationMatrix2D((0.0, 0.0), theta, scale)
    true_pose[:, 2] = [120.0, 90.0]
    page = extract_page(world, true_pose, (300, 360))
    params = MatchParams(
        resolution_m=1.0,
        blur_sigma_m=3.0,
        min_overlap_m2=2000.0,
        max_overlap_frac=1.0,  # the synthetic page lies fully inside the world
        peak_separation_m=20.0,
    )
    candidates = match_pair(
        world, np.ones_like(world, dtype=bool), page, scale=scale, params=params
    )
    assert candidates
    best = min(
        candidates,
        key=lambda c: pose_corner_error(c.pose, true_pose, (300, 360)),
    )
    # NCC grid quantization allows a couple of cells of error before refinement.
    assert pose_corner_error(best.pose, true_pose, (300, 360)) < 8.0

    # Refinement + verification must rank the true pose first, ahead of the
    # near-self-similar 90-degree lattice locks that can win on raw NCC.
    from skimage.morphology import skeletonize

    skeleton = skeletonize(world > 0.4)
    distance = cv2.distanceTransform(
        (~skeleton).astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)
    distance = np.minimum(distance, 30.0)
    page_skel = skeletonize(page > 0.4)
    ys, xs = np.nonzero(page_skel)
    points = np.column_stack([xs, ys]).astype(np.float64)
    ranked = refine_and_rank(candidates, distance, points)
    assert pose_corner_error(ranked[0].pose, true_pose, (300, 360)) < 2.0
    assert ranked[0].inlier_frac > 0.9


def test_chamfer_refine_recovers_perturbation() -> None:
    world = road_grid((500, 500), IRREGULAR_V, IRREGULAR_H)
    theta, scale = 5.0, 1.0
    true_pose = cv2.getRotationMatrix2D((0.0, 0.0), theta, scale)
    true_pose[:, 2] = [60.0, 40.0]
    page = extract_page(world, true_pose, (320, 320))

    mask = (world > 0.4).astype(np.uint8)
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask > 0)
    distance = cv2.distanceTransform(
        (~skeleton).astype(np.uint8), cv2.DIST_L2, 5
    ).astype(np.float32)
    distance = np.minimum(distance, 30.0)

    page_mask = (page > 0.4).astype(np.uint8)
    page_skel = skeletonize(page_mask > 0)
    ys, xs = np.nonzero(page_skel)
    points = np.column_stack([xs, ys]).astype(np.float64)

    perturbed = true_pose.copy()
    nudge = cv2.getRotationMatrix2D((160.0, 160.0), 1.5, 1.0)
    perturbed = compose(nudge, perturbed)
    perturbed[:, 2] += [4.0, -3.0]

    refined, diag = chamfer_refine(distance, points, perturbed)
    assert pose_corner_error(refined, true_pose, (320, 320)) < 1.5
    assert diag["inlier_frac"] > 0.9


def test_uniform_lattice_reports_multiple_peaks() -> None:
    spacing = list(range(30, 480, 60))
    world = road_grid((500, 500), spacing, spacing)
    page = extract_page(world, np.array([[1.0, 0, 90], [0, 1.0, 90]]), (300, 300))
    params = MatchParams(
        resolution_m=1.0,
        blur_sigma_m=3.0,
        min_overlap_m2=2000.0,
        max_overlap_frac=1.0,
        peak_separation_m=20.0,
    )
    candidates = match_pair(
        world, np.ones_like(world, dtype=bool), page, scale=1.0, params=params
    )
    zero_theta = [
        c for c in candidates if abs(c.theta_deg) < 1 or abs(abs(c.theta_deg) - 90) < 1
    ]
    strong = [c for c in zero_theta if c.ncc > 0.5]
    # A uniform lattice must surface as several competing strong peaks.
    assert len(strong) >= 2


def test_rotated_bounds_nonnegative() -> None:
    base = cv2.getRotationMatrix2D((0.0, 0.0), 30.0, 1.0)
    tight = rotated_bounds((200, 100), base)
    corners = np.array(
        [[0, 0, 1], [100, 0, 1], [100, 200, 1], [0, 200, 1]], dtype=float
    )
    warped = corners @ tight.T
    assert warped.min() > -1e-6
