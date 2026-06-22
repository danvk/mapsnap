import numpy as np

from mapsnap.keymap.fit_keymap import (
    Detection,
    GeorefPage,
    build_correspondences,
    describe_model,
    georef_variant,
    inlier_pages,
    page_number,
    polygon_centroid,
    project,
    ransac,
    similarity_apply,
    similarity_fit,
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


def test_similarity_fit_recovers_reflected_similarity():
    # Known reflected similarity: scale 2, and the orientation-reversing form.
    true = (1.7, 0.9, 10.0, -5.0)
    src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 2.0)]
    dst = [similarity_apply(true, p) for p in src]
    assert all(abs(f - t) < 1e-9 for f, t in zip(similarity_fit(src, dst), true))


def test_similarity_is_orientation_reversing():
    # det of [[a, b], [b, -a]] is negative -> a reflection (handedness flip).
    a, b = 1.7, 0.9
    assert a * (-a) - b * b < 0


def test_describe_model_reports_scale_rotation_reflected():
    summary = describe_model((2.0, 0.0, 5.0, -3.0))
    assert "scale=2.000 m/px" in summary
    assert "rotation=+0.0°" in summary
    assert "(reflected)" in summary


def test_georef_variant():
    assert georef_variant("p126.georef.json") == ("p126", "canonical")
    assert georef_variant("p239__2.georef-1gcp.json") == ("p239__2", "1gcp")
    assert georef_variant("p156.georef-misscale.json") == ("p156", "misscale")
    assert georef_variant("p241.georef-outlier.json") == ("p241", "outlier")
    assert georef_variant("p126.streets.json") is None


def test_inlier_pages_uses_any_frame():
    # A page with two pieces; under (x, -y), a point in the second piece counts as inside.
    page = _page(5, 0.0, 0.0, _square(0, 0, 1), _square(100, 0, 1))
    identity_flip = (1.0, 0.0, 0.0, 0.0)  # (x, y) -> (x, -y)
    assert inlier_pages(identity_flip, [page], [(0, (100.0, 0.0))]) == {0}


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


def test_build_correspondences_pairs_matching_numbers():
    pages = [_page(10, 0.0, 0.0), _page(11, 5.0, 0.0)]
    detections = [
        Detection(10, (1.0, 1.0)),
        Detection(10, (2.0, 2.0)),
        Detection(99, (9.0, 9.0)),
    ]
    corr = build_correspondences(pages, detections)
    assert sorted(c[0] for c in corr) == [0, 0]  # two matches for page 10 only


def test_ransac_recovers_model_and_rejects_outlier():
    # Pages 0..5 on a 2D grid; their pixels are the centroids passed through (x, -y), so the
    # recovered reflected similarity maps each pixel back onto its page. Page 6 has only a bad
    # reading, so it should be rejected.
    coords = [
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (20.0, 0.0),
        (0.0, 20.0),
    ]
    pages = [_page(k, *coords[k]) for k in range(6)]
    pages.append(_page(6, 30.0, 30.0))
    detections = [Detection(k, (cx, -cy)) for k, (cx, cy) in enumerate(coords)]
    detections.append(Detection(6, (999.0, 999.0)))  # page 6: only an outlier reading
    corr = build_correspondences(pages, detections)
    model, inliers = ransac(pages, corr, iterations=500, rng=np.random.default_rng(0))
    assert model is not None
    assert inliers == set(range(6))  # page index 6 rejected


def test_polygon_centroid():
    assert polygon_centroid([[0, 0], [2, 0], [2, 2], [0, 2]]) == (1.0, 1.0)


def test_superseded_stems(tmp_path):
    for name in ["p239.jpg", "p239__1.jpg", "p239__2.jpg", "p133.jpg", "p133s.jpg"]:
        (tmp_path / name).touch()
    assert superseded_stems(tmp_path) == {"p239"}
