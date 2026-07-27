from pathlib import Path

import numpy as np

from mapsnap.keymap.keymap_patches import (
    TARGET_LONG_SIDE,
    build_image_patches,
    crop_excludes_numbers,
    crop_patch,
    is_far_from_all,
    sample_negative_centers,
    scale_points,
    working_scale,
)


def test_crop_excludes_numbers():
    labels = [(100.0, 100.0)]
    # Crop directly on the label overlaps the number -> not safe.
    assert not crop_excludes_numbers(100, 100, labels, crop_half_w=55, crop_half_h=30)
    # Far horizontally (beyond crop_half_w + number_half_w) -> safe.
    assert crop_excludes_numbers(100 + 200, 100, labels, crop_half_w=55, crop_half_h=30)
    # In a vertical gap (clears in y) even if x-aligned -> safe.
    assert crop_excludes_numbers(100, 100 + 100, labels, crop_half_w=55, crop_half_h=30)
    # Close in both axes -> not safe.
    assert not crop_excludes_numbers(
        100 + 40, 100 + 20, labels, crop_half_w=55, crop_half_h=30
    )


def test_working_scale_prefers_plain_factors_in_band():
    # Every dev-corpus key map: SCALE lands near the target, so use it as-is.
    assert working_scale(5866, 7323) == 0.25  # chicago, working 1831
    assert working_scale(6091, 8422) == 0.25  # washington dc, working 2106
    # Already downscaled to ~25%: no resampling at all.
    assert working_scale(1446, 2038) == 1.0


def test_working_scale_normalises_unexpected_resolutions():
    # Asheville: 4400x5400 is only 1350px at SCALE, far below the trained size,
    # so it is normalised to the target instead.
    assert working_scale(4400, 5400) == TARGET_LONG_SIDE / 5400
    assert round(5400 * working_scale(4400, 5400)) == TARGET_LONG_SIDE
    # Neither 1.0 nor SCALE lands in the band for these either.
    assert working_scale(3999, 3999) == TARGET_LONG_SIDE / 3999
    assert working_scale(4000, 100) == TARGET_LONG_SIDE / 4000
    # A thumbnail well under the target is scaled up to it.
    assert working_scale(1100, 1350) == TARGET_LONG_SIDE / 1350


def test_working_scale_degenerate():
    assert working_scale(0, 0) == 1.0


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


def test_labels_path_for_uses_the_truth_directory():
    from mapsnap.keymap.keymap_patches import labels_path_for

    assert labels_path_for("data/vol/raw/p0.jpg") == Path(
        "data/vol/raw/truth/p0.labels.json"
    )
    # Compound extensions collapse to the stem, as the labeler writes them.
    assert labels_path_for("data/vol/raw/p35B.jpeg") == Path(
        "data/vol/raw/truth/p35B.labels.json"
    )


def test_keymap_key_is_volume_qualified():
    from mapsnap.keymap.keymap_patches import keymap_key

    # Ten volumes have a "p0", so the stem alone cannot name a key map.
    assert keymap_key(Path("data/chicago_il_1950_vol_1/raw/p0.jpg")) == (
        "chicago_il_1950_vol_1/p0"
    )
    assert keymap_key(Path("data/los_angeles_ca_1949_vol_14/raw/pa.jpg")) == (
        "los_angeles_ca_1949_vol_14/pa"
    )


def test_labelled_keymaps_finds_truth_at_any_depth(tmp_path):
    from mapsnap.keymap.keymap_patches import labelled_keymaps

    def make(volume: str, stem: str, *, image: bool = True) -> None:
        raw = tmp_path / volume / "raw"
        (raw / "truth").mkdir(parents=True, exist_ok=True)
        (raw / "truth" / f"{stem}.labels.json").write_text("{}")
        if image:
            (raw / f"{stem}.jpg").write_bytes(b"jpeg")

    make("champaign", "p1")
    make("queens_1950/vol2", "p0")  # a nested volume
    make("detroit", "p0", image=False)  # truth whose image is gone
    pairs = labelled_keymaps(tmp_path)
    assert [str(i.relative_to(tmp_path)) for i, _ in pairs] == [
        "champaign/raw/p1.jpg",
        "queens_1950/vol2/raw/p0.jpg",
    ]
    # Each image is paired with the truth file that labels it.
    for image, labels in pairs:
        assert labels == image.parent / "truth" / f"{image.stem}.labels.json"


def test_split_train_val_partitions_and_excludes():
    from mapsnap.keymap.keymap_patches import split_train_val

    images = [
        Path("data/chicago_il_1950_vol_1/raw/p0.jpg"),
        Path("data/grand_rapids_mi_1953_vol7/raw/p0a.jpg"),
        Path("data/grand_rapids_mi_1953_vol7/raw/p0b.jpg"),
        Path("data/hudson_co_nj_1950_vol_9/raw/p0.jpg"),
    ]
    # A leave-one-volume-out fold: BOTH grand rapids sheets leave training,
    # and neither may serve as the selection-val image.
    train, val = split_train_val(
        images,
        "hudson_co_nj_1950_vol_9/p0",
        {"grand_rapids_mi_1953_vol7/p0a", "grand_rapids_mi_1953_vol7/p0b"},
    )
    assert [str(p) for p in train] == ["data/chicago_il_1950_vol_1/raw/p0.jpg"]
    assert len(val) == 1

    import pytest

    with pytest.raises(ValueError, match="unknown key map"):
        split_train_val(images, "hudson_co_nj_1950_vol_9/p0", {"typo/p0"})
    with pytest.raises(ValueError, match="excluded set"):
        split_train_val(
            images, "chicago_il_1950_vol_1/p0", {"chicago_il_1950_vol_1/p0"}
        )
