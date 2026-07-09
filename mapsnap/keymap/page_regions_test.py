import numpy as np

from mapsnap.keymap.page_regions import (
    RegionParams,
    background_clusters,
    box_center,
    box_component,
    clean_cluster_mask,
    cluster_image,
    cluster_image_seeded,
    dedup_palette,
    mask_to_polygon,
    nearest_neighbor_distance,
    polygon_bounds,
    regions_panels_doc,
    scale_boxes,
    seed_palette,
    segment_page_regions,
    split_component,
    working_scale,
)


def test_working_scale():
    assert working_scale((1000, 2000), 3000) == 1.0  # already small -> no upscale
    assert working_scale((3000, 6000), 3000) == 0.5  # long side 6000 -> 0.5
    assert working_scale((7323, 5866), 3000) == 3000 / 7323


def test_cluster_image_two_colors():
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[:, :10] = [200, 40, 40]
    rgb[:, 10:] = [40, 40, 200]
    labels, centers = cluster_image(rgb, n_clusters=2, line_smooth_size=3, seed=0)
    assert centers.shape == (2, 3)
    assert len(np.unique(labels)) == 2
    assert (labels[:, 0] != labels[:, -1]).any()  # the two halves differ


def test_background_clusters():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[6:9] = 1  # 30 pixels
    labels[9:] = 2  # 10 pixels  (cluster 0 = 60 pixels)
    assert background_clusters(labels, 3, area_frac=0.5) == {0, 1}  # >= 30
    assert background_clusters(labels, 3, area_frac=0.9) == {0}  # only >= 54


def test_clean_cluster_mask_severs_thin_sliver():
    # Two blocks joined by a 2-px-wide sliver; opening (radius 3) should sever it into two pieces.
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 5:25] = True  # block A
    mask[10:30, 35:55] = True  # block B
    mask[19:21, 25:35] = True  # thin sliver joining them
    from scipy import ndimage as ndi

    _, count_before = ndi.label(mask)  # type: ignore[misc]
    assert count_before == 1  # joined before cleaning
    cleaned = clean_cluster_mask(mask, close_radius=0, open_radius=3)
    _, count_after = ndi.label(cleaned)  # type: ignore[misc]
    assert count_after == 2  # severed after
    assert cleaned[20, 15] and cleaned[20, 45]  # both blocks survive


def test_box_component_picks_majority_and_handles_off():
    components = np.zeros((20, 20), dtype=np.int32)
    components[2:10, 2:10] = 1
    components[2:10, 12:18] = 2
    assert box_component(components, (3, 3, 6, 6)) == 1
    assert box_component(components, (13, 3, 16, 6)) == 2
    assert box_component(components, (10, 10, 11, 11)) is None  # off any component


def test_split_component_partitions_between_two_seeds():
    # One wide component with a seed near each end -> split near the middle.
    component = np.zeros((20, 80), dtype=bool)
    component[5:15, 5:75] = True
    pieces = split_component(component, [(8, 8, 12, 12), (66, 8, 70, 12)])
    assert len(pieces) == 2
    assert pieces[0][10, 10] and not pieces[0][10, 70]  # left seed keeps the left
    assert pieces[1][10, 70] and not pieces[1][10, 10]  # right seed keeps the right
    assert not (pieces[0] & pieces[1]).any()  # disjoint


def test_regions_panels_doc():
    polygons = {
        0: [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)],
        2: [(5.0, 6.0), (7.0, 6.0), (7.0, 8.0)],
    }
    doc = regions_panels_doc("p0b.jpg", (100, 200), polygons, ["55", "1", "2"])
    assert doc["image"] == "p0b.jpg"
    assert doc["width"] == 100 and doc["height"] == 200
    # Rings are explicitly closed (first vertex repeated at the end), matching the
    # panels.json convention elsewhere (e.g. split.py's Shapely-derived rings).
    assert doc["panels"] == [
        [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 2.0]],
        [[5.0, 6.0], [7.0, 6.0], [7.0, 8.0], [5.0, 6.0]],
    ]
    for ring in doc["panels"]:
        assert ring[0] == ring[-1]
    # labels are parallel to panels and indexed by seed, so polygon 2 -> texts[2] == "2".
    assert doc["labels"] == ["55", "2"]


def test_nearest_neighbor_distance_grid():
    # Nearest neighbours are 10, 10, 30 -> median 10.
    assert nearest_neighbor_distance([(0.0, 0.0), (10.0, 0.0), (40.0, 0.0)]) == 10.0


def test_nearest_neighbor_distance_too_few():
    assert nearest_neighbor_distance([(1.0, 2.0)]) == 0.0
    assert nearest_neighbor_distance([]) == 0.0


def test_polygon_bounds():
    assert polygon_bounds([[3, 5], [9, 5], [9, 11], [3, 11]]) == (3, 5, 9, 11)


def test_box_center():
    assert box_center((2.0, 4.0, 6.0, 10.0)) == (4.0, 7.0)


def test_mask_to_polygon_rectangle():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 8:25] = True
    polygon = mask_to_polygon(mask, simplify_tolerance=1.0)
    assert len(polygon) == 4  # a rectangle simplifies to four corners
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    assert min(xs) <= 8 and max(xs) >= 24
    assert min(ys) <= 5 and max(ys) >= 14


def test_mask_to_polygon_empty():
    assert mask_to_polygon(np.zeros((5, 5), dtype=bool), simplify_tolerance=1.0) == []


def test_segment_page_regions_two_blocks_and_background_discard():
    # A large light-grey paper field (the background cluster) with two small saturated blocks.
    rgb = np.full((80, 80, 3), 0.85, dtype=np.float64)
    rgb[30:50, 15:30] = [0.8, 0.1, 0.1]  # red block
    rgb[30:50, 50:65] = [0.1, 0.1, 0.8]  # blue block
    params = RegionParams(n_clusters=3)
    seeds = [
        (18.0, 35.0, 27.0, 45.0),  # on the red block
        (53.0, 35.0, 62.0, 45.0),  # on the blue block
        (5.0, 5.0, 12.0, 12.0),  # on background -> discarded
    ]
    polygons = segment_page_regions(rgb, seeds, params)
    assert set(polygons) == {0, 1}  # the background seed is dropped
    assert max(x for x, _ in polygons[0]) <= 40  # red region stays left
    assert min(x for x, _ in polygons[1]) >= 45  # blue region stays right


def test_segment_page_regions_splits_shared_block():
    # One wide red block on grey paper with two page numbers in it -> split between them, not one
    # region swallowing the other.
    rgb = np.full((60, 120, 3), 0.85, dtype=np.float64)
    rgb[20:45, 15:105] = [0.8, 0.1, 0.1]  # a single wide red block
    params = RegionParams(n_clusters=3, cluster_open_radius=0)
    seeds = [(25.0, 28.0, 33.0, 36.0), (87.0, 28.0, 95.0, 36.0)]
    polygons = segment_page_regions(rgb, seeds, params)
    assert set(polygons) == {0, 1}
    assert max(x for x, _ in polygons[0]) < min(
        x for x, _ in polygons[1]
    )  # split, disjoint


def test_scale_boxes_scales_and_clamps():
    boxes = scale_boxes(
        [(10.0, 20.0, 30.0, 40.0), (0.0, 0.0, 300.0, 300.0)], 0.5, (100, 80)
    )
    assert boxes[0] == (5, 10, 15, 20)
    assert boxes[1] == (0, 0, 79, 99)  # clamped to (width-1, height-1)


def test_dedup_palette_merges_near_colors():
    colors = np.array([[50, 10, 10], [52, 10, 10], [50, 40, 10]], dtype=np.float32)
    kept = dedup_palette(colors, min_distance=6.0)
    assert len(kept) == 2  # the 2-apart twin merges; the 30-apart colour stays


def test_seed_palette_reads_block_color():
    lab = np.zeros((40, 40, 3), dtype=np.float64)
    lab[...] = [80.0, 0.0, 0.0]  # paper
    lab[10:30, 10:30] = [60.0, 30.0, 10.0]  # block
    palette = seed_palette(lab, [(15, 15, 25, 25)])
    assert np.allclose(palette[0], [60.0, 30.0, 10.0])


def test_cluster_image_seeded_falls_back_when_all_seeds_on_paper():
    # A uniform image: every seed colour is within the paper distance -> None (use k-means).
    rgb = np.full((40, 40, 3), 220, dtype=np.uint8)
    result = cluster_image_seeded(rgb, [(5, 5, 12, 12)], 3, RegionParams(n_clusters=2))
    assert result is None


def test_segment_page_regions_pale_block_near_paper():
    # A pale block whose colour plain k-means (k=2 here) merges into the paper cluster: the
    # seeded palette must still give it its own region instead of dropping the seed.
    rgb = np.full((80, 80, 3), 0.85, dtype=np.float64)
    rgb[30:50, 20:40] = [0.80, 0.78, 0.72]  # pale tan block, close to the 0.85 paper
    rgb[30:50, 55:75] = [0.1, 0.1, 0.8]  # one saturated block (so k-means has a job)
    seeds = [(25.0, 35.0, 34.0, 45.0), (60.0, 35.0, 69.0, 45.0)]
    params = RegionParams(n_clusters=2, palette_dedup_distance=4.0)
    seeded = segment_page_regions(rgb, seeds, params)
    assert set(seeded) == {0, 1}  # the pale block is found
    globally = segment_page_regions(
        rgb, seeds, RegionParams(n_clusters=2, use_global_kmeans=True)
    )
    assert 0 not in globally  # plain k=2 merges the pale block into paper and drops it
