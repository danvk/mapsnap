from mapsnap.keymap.truth_regions import project_truth_regions, world_to_pixel_affine

# Identity-ish georef: lon in [0, 1] -> x in [0, 100], lat in [0, 1] -> y in [100, 0].
GEOREF = {
    "width": 100,
    "height": 100,
    "corners": [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
}


def world_square(lon: float, lat: float, size: float = 0.1) -> list[list[float]]:
    return [
        [lon, lat],
        [lon + size, lat],
        [lon + size, lat + size],
        [lon, lat + size],
        [lon, lat],
    ]


def test_affine_maps_corners_to_pixels():
    affine = world_to_pixel_affine(GEOREF)
    import numpy as np

    top_left = np.array([0.0, 1.0, 1.0]) @ affine.T
    bottom_right = np.array([1.0, 0.0, 1.0]) @ affine.T
    assert np.allclose(top_left, [0, 0])
    assert np.allclose(bottom_right, [100, 100])


def test_footprint_inside_is_kept():
    polygons, labels = project_truth_regions(GEOREF, {7: [world_square(0.4, 0.4)]})
    assert labels == ["7"]
    assert len(polygons) == 1


def test_footprint_on_other_keymap_is_dropped():
    # Projects entirely off the image (as when a volume has two key maps).
    polygons, labels = project_truth_regions(GEOREF, {7: [world_square(3.0, 0.4)]})
    assert labels == []
    assert polygons == {}


def test_footprint_mostly_inside_is_kept_mostly_outside_dropped():
    # 80% of the area inside the left edge: kept. Only 20% inside: dropped.
    truth = {7: [world_square(-0.02, 0.4)], 8: [world_square(-0.08, 0.4)]}
    polygons, labels = project_truth_regions(GEOREF, truth)
    assert labels == ["7"]


def test_split_page_keeps_only_footprints_on_this_keymap():
    truth = {7: [world_square(0.4, 0.4), world_square(3.0, 0.4)]}
    polygons, labels = project_truth_regions(GEOREF, truth)
    assert labels == ["7"]
    assert len(polygons) == 1
