import numpy as np

from mapsnap.keymap_patches import (
    build_image_patches,
    crop_patch,
    is_far_from_all,
    sample_negative_centers,
    scale_points,
)


def test_scale_points():
    pts = [(100.0, 200.0, "5"), (40.0, 80.0, "12")]
    assert scale_points(pts, 0.25) == [(25.0, 50.0, "5"), (10.0, 20.0, "12")]


def test_crop_patch_centered_interior():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[48:52, 48:52] = (10, 20, 30)  # a 4x4 marker at the center
    patch = crop_patch(img, 50, 50, 20)
    assert patch.shape == (20, 20, 3)
    # The marker lands at the patch center (half = 10).
    assert tuple(patch[10, 10]) == (10, 20, 30)


def test_crop_patch_white_pads_at_edge():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    patch = crop_patch(img, 2, 2, 20)  # center near top-left corner
    assert patch.shape == (20, 20, 3)
    # Top-left quadrant is off-image -> white padding.
    assert (patch[0, 0] == 255).all()
    # Bottom-right quadrant is on-image (black).
    assert (patch[19, 19] == 0).all()


def test_is_far_from_all():
    pts = [(50.0, 50.0, "1")]
    assert is_far_from_all(100.0, 100.0, pts, 40.0)
    assert not is_far_from_all(60.0, 50.0, pts, 40.0)  # only 10 away


def test_sample_negative_centers_respects_min_distance():
    rng = np.random.default_rng(0)
    positives = [(50.0, 50.0, "1"), (150.0, 150.0, "2")]
    centers = sample_negative_centers(
        200, 200, positives, count=50, min_dist=40.0, rng=rng
    )
    assert len(centers) > 0
    for cx, cy in centers:
        assert is_far_from_all(cx, cy, positives, 40.0)


def test_build_image_patches_counts_and_labels():
    rng = np.random.default_rng(1)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    pts = [(60.0, 60.0, "1"), (200.0, 200.0, "2")]
    patches, labels = build_image_patches(
        img, pts, size=64, neg_per_pos=3, min_neg_dist=40.0, rng=rng
    )
    assert labels[:2] == [1, 1]  # positives first
    assert sum(labels) == 2  # exactly the two positives are labeled 1
    assert len(patches) == len(labels)
    assert all(p.shape == (64, 64, 3) for p in patches)
    assert labels.count(0) > 0  # some negatives sampled
