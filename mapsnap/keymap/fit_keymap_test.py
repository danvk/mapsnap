import math

import numpy as np

from mapsnap.keymap.fit_keymap import (
    Detection,
    GeorefPage,
    affine_apply,
    affine_fit,
    build_correspondences,
    georef_variant,
    helmert_apply,
    helmert_fit,
    inlier_pages,
    page_number,
    polygon_centroid,
    project,
    ransac,
    superseded_stems,
    unproject,
)


def _square(cx: float, cy: float, half: float) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _page(
    number: int, cx: float, cy: float, *frames: list[tuple[float, float]]
) -> GeorefPage:
    polys = list(frames) or [_square(cx, cy, 1.0)]
    return GeorefPage(number, (cx, cy), polys, [f"p{number}"])


def test_georef_variant():
    assert georef_variant("p126.georef.json") == ("p126", "canonical")
    assert georef_variant("p239__2.georef-1gcp.json") == ("p239__2", "1gcp")
    assert georef_variant("p156.georef-misscale.json") == ("p156", "misscale")
    assert georef_variant("p241.georef-outlier.json") == ("p241", "outlier")
    assert georef_variant("p126.streets.json") is None


def test_maps_inside_uses_any_frame():
    # A page split into two pieces; a point in the second piece still counts as inside.
    page = _page(5, 0.0, 0.0, _square(0, 0, 1), _square(100, 0, 1))
    assert inlier_pages(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), affine_apply, [page], [(0, (100.0, 0.0))]
    ) == {0}


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


def test_build_correspondences_pairs_matching_numbers():
    pages = [_page(10, 0.0, 0.0), _page(11, 5.0, 0.0)]
    detections = [
        Detection(10, (1.0, 1.0)),
        Detection(10, (2.0, 2.0)),
        Detection(99, (9.0, 9.0)),
    ]
    corr = build_correspondences(pages, detections)
    assert sorted(c[0] for c in corr) == [0, 0]  # two matches for page 10 only


def test_inlier_pages_uses_frame_containment():
    pages = [_page(1, 0.0, 0.0)]
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
    pages = [_page(k, *coords[k]) for k in range(8)]
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


def test_superseded_stems(tmp_path):
    for name in ["p239.jpg", "p239__1.jpg", "p239__2.jpg", "p133.jpg", "p133s.jpg"]:
        (tmp_path / name).touch()
    assert superseded_stems(tmp_path) == {"p239"}
