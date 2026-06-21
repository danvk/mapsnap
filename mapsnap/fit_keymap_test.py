import math

import numpy as np

from mapsnap.fit_keymap import (
    Detection,
    GeorefPage,
    affine_apply,
    affine_fit,
    build_correspondences,
    helmert_apply,
    helmert_fit,
    inlier_pages,
    page_number,
    polygon_centroid,
    project,
    ransac,
    unproject,
)


def test_page_number():
    assert page_number("p133") == 133
    assert page_number("p133s") == 133
    assert page_number("p1b") == 1
    assert page_number("keymap") is None


def test_project_unproject_roundtrip():
    lon, lat = -77.02, 38.92
    x, y = project(lon, lat, -77.0, 38.9)
    back = unproject(x, y, -77.0, 38.9)
    assert abs(back[0] - lon) < 1e-9 and abs(back[1] - lat) < 1e-9


def test_helmert_fit_recovers_similarity():
    s, theta = 2.0, math.radians(30)
    a, b = s * math.cos(theta), s * math.sin(theta)
    true = (a, b, 10.0, -5.0)
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 2.0)]
    dst = [helmert_apply(true, p) for p in src]
    assert all(abs(f - t) < 1e-9 for f, t in zip(helmert_fit(src, dst), true))


def test_affine_fit_recovers_affine():
    # Anisotropic + shear transform a Helmert cannot represent.
    true = (2.0, 0.3, 5.0, -0.1, 1.5, -2.0)
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0)]
    dst = [affine_apply(true, p) for p in src]
    assert all(abs(f - t) < 1e-9 for f, t in zip(affine_fit(src, dst), true))


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def test_build_correspondences_pairs_matching_numbers():
    pages = [
        GeorefPage("p10", 10, (0.0, 0.0), _square(0, 0, 1)),
        GeorefPage("p11", 11, (5.0, 0.0), _square(5, 0, 1)),
    ]
    detections = [
        Detection(10, (1.0, 1.0)),
        Detection(10, (2.0, 2.0)),
        Detection(99, (9.0, 9.0)),
    ]
    corr = build_correspondences(pages, detections)
    assert sorted(c[0] for c in corr) == [0, 0]  # two matches for page 10 only


def test_inlier_pages_uses_frame_containment():
    pages = [GeorefPage("p1", 1, (0.0, 0.0), _square(0, 0, 1))]
    corr = [(0, (0.5, 0.5)), (0, (50.0, 50.0))]
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert inlier_pages(identity, affine_apply, pages, corr) == {0}


def test_ransac_affine_recovers_model_and_rejects_outlier():
    # Pages on a 2D grid (non-collinear, so an affine is determined).
    coords = [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (20.0, 0.0),
        (0.0, 20.0),
        (20.0, 10.0),
        (20.0, 20.0),
    ]
    pages = [
        GeorefPage(f"p{k}", k, coords[k], _square(*coords[k], 1.0)) for k in range(8)
    ]
    detections = [Detection(k, coords[k]) for k in range(7)]
    detections.append(Detection(7, (999.0, 999.0)))  # outlier
    corr = build_correspondences(pages, detections)
    model, inliers = ransac(
        pages,
        corr,
        fit_fn=affine_fit,
        apply_fn=affine_apply,
        sample_size=3,
        iterations=500,
        rng=np.random.default_rng(0),
    )
    assert model is not None
    assert inliers == set(range(7))  # page 7 rejected


def test_polygon_centroid():
    assert polygon_centroid([[0, 0], [2, 0], [2, 2], [0, 2]]) == (1.0, 1.0)
