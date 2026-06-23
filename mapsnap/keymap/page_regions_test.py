import numpy as np

from mapsnap.keymap.page_regions import (
    RegionParams,
    background_mask,
    box_center,
    keep_seed_component,
    mask_to_polygon,
    nearest_neighbor_distance,
    open_thin_arms,
    polygon_bounds,
    quantize_colors,
    quantized_lab,
    regions_panels_doc,
    segment_page_regions,
    stamp_seed_markers,
    working_scale,
)


def test_working_scale():
    assert working_scale((1000, 2000), 3000) == 1.0  # already small -> no upscale
    assert working_scale((3000, 6000), 3000) == 0.5  # long side 6000 -> 0.5
    assert working_scale((7323, 5866), 3000) == 3000 / 7323


def test_quantized_lab_two_blocks():
    # Two solid colour halves with a thin black seam; quantization yields two flat colours.
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :19] = [200, 40, 40]
    rgb[:, 19:21] = 0  # thin black seam (erased by the median blur)
    rgb[:, 21:] = [40, 40, 200]
    quant = quantized_lab(rgb, n_clusters=2, line_smooth_size=3, seed=0)
    assert len(np.unique(quant.reshape(-1, 3), axis=0)) == 2


def test_quantize_colors_two_colors():
    lab = np.zeros((10, 10, 3), dtype=np.float64)
    lab[:, :5] = [50.0, 40.0, 30.0]  # one colour
    lab[:, 5:] = [80.0, -20.0, -10.0]  # another
    quant = quantize_colors(lab, n_clusters=2, seed=0)
    # Exactly two distinct quantized colours, and each half keeps its own.
    assert len(np.unique(quant.reshape(-1, 3), axis=0)) == 2
    assert (quant[:, 0] != quant[:, 9]).any()


def test_regions_panels_doc():
    polygons = {
        0: [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0)],
        2: [(5.0, 6.0), (7.0, 6.0), (7.0, 8.0)],
    }
    doc = regions_panels_doc("p0b.jpg", (100, 200), polygons, ["55", "1", "2"])
    assert doc["image"] == "p0b.jpg"
    assert doc["width"] == 100 and doc["height"] == 200
    assert doc["panels"] == [
        [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 6.0], [7.0, 8.0]],
    ]
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


def test_stamp_seed_markers_pads_and_keeps_core():
    # A single tiny box at full-res, scale 1.0, padded by 1x its longer side.
    markers = stamp_seed_markers(
        (40, 40), [(10.0, 10.0, 14.0, 14.0)], scale=1.0, pad_frac=1.0
    )
    assert (markers == 2).any()
    # The pad (longer side 4 -> pad 4) grows the 10..14 box to 6..18 inclusive.
    assert markers[10, 10] == 2  # core
    assert markers[6, 6] == 2 and markers[18, 18] == 2  # padded corners
    assert markers[5, 5] == 0  # just outside the pad


def test_stamp_seed_markers_dense_boxes_keep_their_core():
    # Two boxes close enough that one's pad covers the other's centre; tight-box pass must
    # still leave each centre owning its own label.
    boxes = [(10.0, 10.0, 14.0, 14.0), (20.0, 10.0, 24.0, 14.0)]
    markers = stamp_seed_markers((40, 40), boxes, scale=1.0, pad_frac=2.0)
    assert markers[12, 12] == 2  # first box core
    assert markers[12, 22] == 3  # second box core


def test_background_mask_splits_paper_from_color():
    # CIELAB: white paper is light & near-grey; a saturated colour has high chroma.
    lab = np.array([[[95.0, 0.0, 2.0], [50.0, 40.0, 30.0]]])  # paper, colour
    mask = background_mask(lab, bg_lightness=78.0, bg_chroma=12.0)
    assert mask.tolist() == [[True, False]]


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


def test_open_thin_arms_removes_arm_keeps_block():
    mask = np.zeros((60, 80, 3)[:2], dtype=bool)
    mask[10:50, 10:50] = True  # 40x40 block
    mask[28:32, 50:78] = True  # 4-px-tall arm reaching right
    opened = open_thin_arms(mask, radius=5)
    assert opened[16:44, 16:44].all()  # block interior preserved (corners get rounded)
    assert not opened[30, 70]  # arm erased


def test_open_thin_arms_keeps_thin_block_via_fallback():
    # A whole region thinner than the disk would erode to nothing; fall back to the input.
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 5:35] = True  # 4-px-tall sliver only
    opened = open_thin_arms(mask, radius=5)
    assert opened.any()  # not lost


def test_keep_seed_component_drops_speck_and_fills_hole():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 2:12] = True  # big block
    mask[6, 6] = False  # interior hole -> filled back in
    mask[18, 18] = True  # detached speck -> dropped
    kept = keep_seed_component(mask)
    assert kept[6, 6]  # hole filled
    assert not kept[18, 18]  # speck removed
    assert kept[2:12, 2:12].all()


def test_segment_page_regions_two_blocks():
    # Left half saturated red, right half saturated blue, a one-pixel paper seam between them.
    rgb = np.zeros((60, 120, 3), dtype=np.float64)
    rgb[:, :59] = [0.8, 0.1, 0.1]
    rgb[:, 59:61] = [1.0, 1.0, 1.0]  # white seam
    rgb[:, 61:] = [0.1, 0.1, 0.8]
    params = RegionParams(n_clusters=3, smooth_sigma=0.5)
    polygons = segment_page_regions(
        rgb, [(27.0, 27.0, 33.0, 33.0), (87.0, 27.0, 93.0, 33.0)], params
    )
    assert set(polygons) == {0, 1}
    left_xs = [x for x, _ in polygons[0]]
    right_xs = [x for x, _ in polygons[1]]
    assert max(left_xs) <= 61  # left region stays on the red half
    assert min(right_xs) >= 59  # right region stays on the blue half


def test_segment_page_regions_seed_not_trapped_in_digit_loop():
    # A colored block with a black ring near the centre (like the hole of a "0"): the padded box
    # seed must clear the ring so the basin fills the whole block, not just the ring interior.
    rgb = np.full((60, 60, 3), [0.1, 0.7, 0.2], dtype=np.float64)  # green block
    rgb[:, :2] = rgb[:, -2:] = rgb[:2, :] = rgb[-2:, :] = 1.0  # white paper border
    rgb[24:36, 24:36] = 0.0  # black ring
    rgb[27:33, 27:33] = [0.1, 0.7, 0.2]  # green inside the ring (the digit's hole)
    params = RegionParams(n_clusters=3, smooth_sigma=0.5, seed_pad_frac=1.0)
    polygons = segment_page_regions(rgb, [(26.0, 26.0, 34.0, 34.0)], params)
    xs = [x for x, _ in polygons[0]]
    ys = [y for _, y in polygons[0]]
    # Region spans well beyond the 24..36 ring out toward the block's paper border.
    assert max(xs) - min(xs) > 40 and max(ys) - min(ys) > 40
