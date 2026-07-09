import numpy as np

from mapsnap.keymap.sam_regions import (
    BOX_FACTOR,
    nearest_negatives,
    prompt_box,
    resolve_contested,
    seed_component,
)


def test_prompt_box_sized_from_nearest_seed():
    centers = [(200.0, 200.0), (300.0, 200.0)]  # 100 px apart
    box = prompt_box(centers[0], centers, (1000, 1000))
    half = BOX_FACTOR * 100
    assert box == [200 - half, 200 - half, 200 + half, 200 + half]


def test_prompt_box_clips_to_image():
    centers = [(10.0, 10.0), (110.0, 10.0)]
    box = prompt_box(centers[0], centers, (50, 400))
    assert box == [0.0, 0.0, 10 + BOX_FACTOR * 100, 49.0]


def test_prompt_box_single_seed_falls_back_to_image_fraction():
    box = prompt_box((50.0, 50.0), [(50.0, 50.0)], (100, 100))
    assert box[0] < 50 < box[2]


def test_nearest_negatives_excludes_self_and_orders_by_distance():
    centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (100.0, 0.0)]
    assert nearest_negatives(centers[0], centers, count=2) == [(5.0, 0.0), (10.0, 0.0)]


def test_resolve_contested_splits_overlap_by_seed_distance():
    # Two masks both claim columns 4-5; seed 0 is at x=2, seed 1 at x=7.
    left = np.zeros((3, 10), dtype=bool)
    left[:, 0:6] = True
    right = np.zeros((3, 10), dtype=bool)
    right[:, 4:10] = True
    resolved = resolve_contested({0: left, 1: right}, [(2.0, 1.0), (7.0, 1.0)])
    assert not (resolved[0] & resolved[1]).any()
    assert resolved[0][1, 4] and not resolved[0][1, 5]  # column 4 nearer seed 0
    assert resolved[1][1, 5] and not resolved[1][1, 4]  # column 5 nearer seed 1
    assert resolved[0][1, 0] and resolved[1][1, 9]  # uncontested pixels untouched


def test_resolve_contested_empty():
    assert resolve_contested({}, []) == {}


def test_seed_component_keeps_the_blob_under_the_seed():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:8] = True  # big blob
    mask[8:10, 8:10] = True  # small blob holding the seed
    component = seed_component(mask, (9.0, 9.0))
    assert component[9, 9] and not component[0, 0]


def test_seed_component_falls_back_to_largest_blob():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, 0:8] = True
    mask[8:10, 8:10] = True
    component = seed_component(mask, (5.0, 5.0))  # seed on background
    assert component[0, 0] and not component[9, 9]
