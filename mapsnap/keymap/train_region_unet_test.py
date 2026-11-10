import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapsnap.keymap.region_unet import (
    BACKGROUND,
    BOUNDARY,
    IGNORE,
    INTERIOR,
    NUM_CLASSES,
    TILE_SIZE,
    DecodeParams,
)
from mapsnap.keymap.train_region_unet import (
    LabelInputs,
    PreparedMeta,
    Tile,
    TruthFootprint,
    average_checkpoints,
    build_label_image,
    colored_mask,
    dilate,
    edge_strength,
    erode,
    footprint_shifts,
    hand_label_image,
    hand_truth_path,
    labels_matched_to_truth,
    load_example,
    matching_label,
    neighbor_differs,
    open_paper,
    panels_doc_rings,
    paper_color,
    paper_mask,
    photometric_jitter,
    random_tile,
    rasterize,
    region_loss,
    regions_doc_from_probabilities,
    ring_window,
    score_doc,
    score_summary,
    selector_fraction,
    shape_score,
    sharp_cores,
    shuffled_batches,
    split_names,
    tile_origin,
    tiles_to_tensors,
    to_lab,
    window_mask,
)


def rectangle(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_selector_fraction():
    item = {
        "label": "p5",
        "target": {
            "source": {"width": 100, "height": 200},
            "selector": {
                "type": "SvgSelector",
                "value": '<svg><polygon points="0,0 100,0 100,100 0,100"/></svg>',
            },
        },
    }
    assert selector_fraction(item) == 0.5
    item["target"]["selector"]["value"] = "<svg/>"
    assert selector_fraction(item) is None


def test_panels_doc_rings_rescales_and_labels():
    doc = {
        "panels": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], [[1, 1]]],
        "labels": ["7", "x"],
    }
    rings = panels_doc_rings(doc, 2.0)
    assert rings == [("7", [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)])]


def test_rasterize_window_and_morphology():
    ring = rectangle(10.5, 20.5, 30.5, 40.5)
    mask = rasterize([ring], (60, 60))
    assert mask[30, 20] and not mask[5, 5]
    rows, cols = ring_window(ring, (60, 60), margin=3)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (17, 45, 7, 35)
    window = window_mask(ring, rows, cols)
    assert window.shape == (28, 28) and window[13, 13]
    assert erode(mask, 3).sum() < mask.sum() < dilate(mask, 3).sum()


def test_neighbor_differs_marks_both_sides():
    labels = np.zeros((4, 6), np.int32)
    labels[:, 3:] = 1
    edge = neighbor_differs(labels)
    assert edge[:, 2].all() and edge[:, 3].all()
    assert not edge[:, 0].any() and not edge[:, 5].any()


def synthetic_keymap() -> tuple[np.ndarray, list[TruthFootprint]]:
    """Two colored blocks split by a heavy black line, a third unlabeled block, on tan paper."""
    rgb = np.full((300, 400, 3), (235, 225, 200), np.uint8)
    rgb[50:250, 40:200] = (120, 200, 120)  # block A, green
    rgb[50:250, 200:360] = (240, 220, 120)  # block B, yellow, color change only
    rgb[50:250, 196:204] = (20, 20, 20)  # heavy line between them
    rgb[260:295, 40:200] = (200, 140, 200)  # an unlabeled pink block
    truth = [
        TruthFootprint("1", rectangle(48, 46, 206, 256)),  # ~6-8 px off the drawn block
        TruthFootprint("2", rectangle(194, 54, 365, 246)),
    ]
    return rgb, truth


def test_edge_strength_is_high_on_lines_and_low_in_fills():
    rgb, _ = synthetic_keymap()
    strength = edge_strength(to_lab(rgb))
    assert strength[150, 200] > 0.8  # the heavy line
    assert strength[150, 100] < 0.1  # inside a flat fill


def test_paper_color_and_colored_mask():
    rgb, _ = synthetic_keymap()
    lab = to_lab(rgb)
    background = np.zeros(rgb.shape[:2], bool)
    background[:40] = True  # the paper margin above the blocks
    paper = paper_color(lab, background)
    assert abs(paper[0] - lab[5, 5, 0]) < 1.0
    colored = colored_mask(lab, paper)
    assert colored[150, 100] and colored[280, 100]
    assert not colored[10, 10]
    assert not colored[150, 200]  # ink is not a fill


def test_matching_label_drops_division_letters_unless_truth_has_them():
    assert matching_label("63w", keep_letters=False) == "63"
    assert matching_label("63W", keep_letters=True) == "63W"
    doc = {"labels": ["63W", "9A", "12"], "panels": []}
    assert labels_matched_to_truth(doc, {"labels": ["63", "9", "12"]})["labels"] == [
        "63",
        "9",
        "12",
    ]
    assert labels_matched_to_truth(doc, {"labels": ["63", "9A"]})["labels"] == [
        "63W",
        "9A",
        "12",
    ]


def test_sharp_cores_matches_lettered_reads_to_numeric_truth():
    truth = [TruthFootprint("63", rectangle(0, 0, 100, 100))]
    assert set(sharp_cores(truth, [("63W", rectangle(1, 1, 99, 99))])) == {0}


def test_sharp_cores_accepts_a_block_inside_a_wider_footprint():
    # The footprint reaches the street centerline, the block stops at the curb: IoU is
    # only 0.64, but the polygon lies inside the footprint and covers most of it.
    truth = [TruthFootprint("1", rectangle(0, 0, 100, 100))]
    assert set(sharp_cores(truth, [("1", rectangle(10, 10, 90, 90))])) == {0}


def test_footprint_shifts_follow_the_local_color_offset():
    truth = [
        TruthFootprint(str(i), rectangle(100 * i, 0, 100 * i + 80, 80))
        for i in range(6)
    ]
    color = [
        (str(i), rectangle(100 * i + 10, -5, 100 * i + 90, 75))
        for i in range(6)
        if i != 3  # the color segmenter missed block 3
    ]
    shifts = footprint_shifts(truth, color, 100.0)
    assert len(shifts) == 6
    for dx, dy in shifts:  # block 3 moves with its neighbors
        assert abs(dx - 10) < 1e-6 and abs(dy + 5) < 1e-6
    # Too few samples to trust: nothing moves.
    assert footprint_shifts(truth[:2], color[:2], 100.0) == [(0.0, 0.0)] * 2


def test_paper_mask_and_open_paper():
    rgb, _ = synthetic_keymap()
    lab = to_lab(rgb)
    background = np.zeros(rgb.shape[:2], bool)
    background[:40] = True
    paper = paper_color(lab, background)
    like = paper_mask(lab, paper, background)
    assert like[5, 5] and not like[150, 100]  # the margin yes, the green fill no
    pocket = like.copy()
    pocket[100:120, 300:320] = True  # paper-like pixels enclosed by block B
    opened = open_paper(pocket, background)
    assert opened[5, 5] and not opened[110, 310]
    # A pocket counts once it is big enough (18x18 after the 1 px erosion = 324 px).
    assert open_paper(pocket, background, min_pocket_px=300)[110, 310]
    assert not open_paper(pocket, background, min_pocket_px=400)[110, 310]


def test_build_label_image_keeps_the_street_between_two_blocks_as_paper():
    # Two colored blocks with a 40 px paper street between them; the footprints meet at
    # the street's centerline, as OIM's do. The labels must stop at the color edges.
    rgb = np.full((300, 400, 3), (235, 225, 200), np.uint8)
    rgb[50:250, 40:180] = (120, 200, 120)
    rgb[50:250, 220:360] = (240, 220, 120)
    truth = [
        TruthFootprint("1", rectangle(40, 50, 200, 250)),
        TruthFootprint("2", rectangle(200, 50, 360, 250)),
    ]
    labels, stats = build_label_image(LabelInputs(rgb, truth, [], [], [], 160.0))
    assert stats.colored == 2
    assert labels[150, 190] == BACKGROUND and labels[150, 200] == BACKGROUND
    assert labels[150, 210] == BACKGROUND
    assert labels[150, 100] == INTERIOR and labels[150, 300] == INTERIOR
    assert BOUNDARY in labels[150, 176:184]  # block A's color edge, not x=200


def test_hand_truth_path(tmp_path: Path):
    meta = make_meta(volume="town", key="p0")
    assert hand_truth_path(tmp_path, meta) is None
    traced = tmp_path / "town" / "raw" / "truth" / "p0.regions.panels.json"
    traced.parent.mkdir(parents=True)
    traced.write_text("{}")
    assert hand_truth_path(tmp_path, meta) == traced


def test_sharp_cores_accepts_only_agreeing_polygons():
    truth = [
        TruthFootprint("1", rectangle(0, 0, 100, 100)),
        TruthFootprint("2", rectangle(100, 0, 200, 100)),
    ]
    agreeing = ("1", rectangle(2, 2, 98, 98))
    disagreeing = ("2", rectangle(100, 0, 200, 40))
    assert set(sharp_cores(truth, [agreeing, disagreeing])) == {0}
    straddling = ("1", rectangle(0, 0, 130, 100))  # covers 30% of page 2's footprint
    assert sharp_cores(truth, [straddling]) == {}


def test_build_label_image_snaps_to_the_line_and_ignores_unlabeled_blocks():
    rgb, truth = synthetic_keymap()
    seeds = [(120.0, 150.0), (280.0, 150.0), (120.0, 277.0)]
    labels, stats = build_label_image(LabelInputs(rgb, truth, [], [], seeds, 160.0))
    assert labels.shape == rgb.shape[:2]
    assert stats.footprints == 2 and stats.sharp == 0
    assert labels[150, 120] == INTERIOR and labels[150, 280] == INTERIOR
    assert labels[150, 200] == BOUNDARY  # the heavy line, not the truth edge 6 px away
    assert labels[150, 194] != BOUNDARY or labels[150, 206] != BOUNDARY
    assert labels[10, 10] == BACKGROUND
    assert labels[277, 120] == IGNORE  # the unlabeled pink block
    # Where the yellow block meets the paper (a color change), the edge is labeled.
    column = labels[150, 350:370]
    assert BOUNDARY in column and BACKGROUND in column


def test_build_label_image_uses_a_sharp_color_polygon():
    rgb, truth = synthetic_keymap()
    color = [("1", rectangle(40, 50, 196, 250))]
    labels, stats = build_label_image(LabelInputs(rgb, truth, color, [], [], 160.0))
    assert stats.sharp == 1
    assert labels[150, 100] == INTERIOR and labels[150, 42] == BOUNDARY


def test_build_label_image_ignores_insets_and_truthless_seeds():
    rgb, truth = synthetic_keymap()
    inset = [rectangle(300, 260, 390, 295)]
    labels, _ = build_label_image(
        LabelInputs(rgb, truth, [], inset, [(20.0, 280.0)], 160.0)
    )
    assert labels[280, 350] == IGNORE
    assert labels[280, 20] == IGNORE


def make_meta(**overrides) -> PreparedMeta:
    base = {
        "volume": "town",
        "key": "p0",
        "scale": 0.5,
        "spacing": 100.0,
        "seeds": [(50.0, 50.0), (150.0, 50.0)],
        "texts": ["1", "2"],
        "full_size": (800, 600),
        "truth_doc": {
            "image": "p0.jpg",
            "width": 800,
            "height": 600,
            "panels": [],
            "labels": [],
        },
        "color_regions_path": None,
    }
    base.update(overrides)
    return PreparedMeta(**base)


def test_prepared_meta_roundtrip():
    meta = make_meta()
    again = PreparedMeta.from_json(json.loads(json.dumps(meta.to_json())))
    assert again == meta


def test_random_tile_shapes_and_seed_channel():
    rng = np.random.default_rng(0)
    rgb = np.full((700, 900, 3), 230, np.uint8)
    labels = np.full((700, 900), BACKGROUND, np.uint8)
    tile = random_tile(rgb, labels, make_meta(seeds=[(400.0, 300.0)]), rng)
    assert tile.rgb.shape == (TILE_SIZE, TILE_SIZE, 3) and tile.rgb.dtype == np.uint8
    assert tile.labels.shape == (TILE_SIZE, TILE_SIZE)
    # A map smaller than the window is padded with paper and IGNORE.
    small = random_tile(rgb[:200, :200], labels[:200, :200], make_meta(), rng)
    assert (small.labels == IGNORE).any()


def test_tile_origin_stays_inside_the_image_and_near_seeds():
    rng = np.random.default_rng(0)
    seeds = [(100.0, 100.0)]
    origins = [tile_origin(seeds, (700, 900), 512, rng) for _ in range(200)]
    assert all(0 <= x <= 900 - 512 and 0 <= y <= 700 - 512 for x, y in origins)
    near = sum(1 for x, y in origins if x < 100 and y < 100)  # window holds the seed
    assert near >= 100  # most tiles are seed-centered
    assert tile_origin([], (300, 300), 512, rng) == (
        0,
        0,
    )  # image smaller than the window


def test_photometric_jitter_keeps_shape_and_range():
    rng = np.random.default_rng(2)
    rgb = np.random.default_rng(3).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    out = photometric_jitter(rgb, rng)
    assert out.shape == rgb.shape and out.dtype == np.uint8


def test_region_loss_ignores_ignore_pixels():
    logits = torch.zeros((1, NUM_CLASSES, 8, 8))
    logits[0, INTERIOR] = 3.0
    targets = torch.full((1, 8, 8), INTERIOR, dtype=torch.int64)
    low = region_loss(logits, targets)
    targets[0, :4] = BACKGROUND
    high = region_loss(logits, targets)
    targets[0, :4] = IGNORE
    ignored = region_loss(logits, targets)
    assert low < high
    # Ignored pixels contribute nothing: the loss equals that of the labeled half alone.
    half = region_loss(logits[:, :, 4:], targets[:, 4:])
    assert abs(float(ignored) - float(half)) < 1e-5
    targets[:] = IGNORE
    assert torch.isfinite(region_loss(logits, targets))


def test_tiles_to_tensors_and_shuffled_batches():
    tile = Tile(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.uint8))
    inputs, targets = tiles_to_tensors([tile, tile], torch.device("cpu"))
    assert inputs.shape == (2, 3, 8, 8) and targets.shape == (2, 8, 8)
    assert inputs.is_contiguous() and targets.is_contiguous()
    batches = list(
        shuffled_batches(
            iter([tile] * 10), batch_size=4, pool_size=6, rng=np.random.default_rng(0)
        )
    )
    assert [len(batch) for batch in batches] == [4, 4]  # leftovers are dropped


def test_split_names_by_volume(tmp_path: Path):
    for name in ("a__p0", "a__p1", "b__p0", "c_vol_4__p0__2"):
        (tmp_path / f"{name}.json").write_text("{}")
    # A split-panel stem (p0__2) must not hide its volume from the val set.
    assert split_names(tmp_path, ("b", "c_vol_4")) == (
        ["a__p0", "a__p1"],
        ["b__p0", "c_vol_4__p0__2"],
    )


def test_regions_doc_from_probabilities_and_scoring():
    probabilities = np.zeros((NUM_CLASSES, 100, 200), np.float32)
    probabilities[BACKGROUND] = 1.0
    probabilities[BACKGROUND][10:90, 10:190] = 0.0
    probabilities[INTERIOR][10:90, 10:190] = 1.0
    probabilities[INTERIOR][10:90, 98:102] = 0.0
    probabilities[BOUNDARY][10:90, 98:102] = 1.0
    meta = make_meta(
        scale=0.5,
        spacing=80.0,
        seeds=[(50.0, 50.0), (150.0, 50.0)],
        full_size=(400, 200),
    )
    doc, counts = regions_doc_from_probabilities(probabilities, meta, DecodeParams())
    assert counts.unseeded == 0 and counts.dropped_seeds == 0
    assert doc["width"] == 400 and sorted(doc["labels"]) == ["1", "2"]
    truth = {
        "image": "p0.jpg",
        "width": 400,
        "height": 200,
        "panels": [
            [[20, 20], [196, 20], [196, 180], [20, 180], [20, 20]],
            [[204, 20], [380, 20], [380, 180], [204, 180], [204, 20]],
        ],
        "labels": ["1", "2"],
    }
    score = score_doc(truth, doc)
    mean, good, count = score_summary(score)
    assert count == 2 and good == 2 and mean > 0.9
    assert shape_score(truth, doc) > 0.9
    # Label-free: the same regions labeled "?" still match their footprints.
    relabeled = dict(doc, labels=["?", "?"])
    assert score_summary(score_doc(truth, relabeled))[2] == 2  # truth count stays
    assert score_summary(score_doc(truth, relabeled))[0] == 0.0
    assert shape_score(truth, relabeled) > 0.9
    probabilities[INTERIOR][10:90, 98:102] = 1.0
    probabilities[BOUNDARY][10:90, 98:102] = 0.0
    _, counts = regions_doc_from_probabilities(
        probabilities,
        make_meta(scale=0.5, spacing=80.0, seeds=[], full_size=(400, 200)),
        DecodeParams(),
    )
    assert counts.unseeded == 1


def test_average_checkpoints_means_floats_and_keeps_counters(tmp_path: Path):
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(1)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(2)}
    torch.save(a, tmp_path / "a.pt")
    torch.save(b, tmp_path / "b.pt")
    averaged = average_checkpoints([tmp_path / "a.pt", tmp_path / "b.pt"])
    assert torch.equal(averaged["w"], torch.tensor([2.0, 4.0]))
    assert int(averaged["n"]) == 2


def prepared_meta(scale: float, spacing: float, seeds: list[tuple[float, float]]):
    return PreparedMeta(
        volume="v",
        key="p0",
        scale=scale,
        spacing=spacing,
        seeds=seeds,
        texts=[str(i) for i in range(len(seeds))],
        full_size=(200, 200),
        truth_doc={},
        color_regions_path=None,
    )


def test_hand_label_image_marks_untraced_ground_as_background():
    # One traced block at full-res 20..100 (working 10..50), a detected number at
    # (90, 90) with no trace, and an inset in the lower right.
    meta = prepared_meta(0.5, 40.0, [(20.0, 20.0), (90.0, 90.0)])
    hand_doc = {
        "width": 200,
        "height": 200,
        "panels": [[[20, 20], [100, 20], [100, 100], [20, 100], [20, 20]]],
        "labels": ["7"],
    }
    inset = [rectangle(80, 80, 99, 99)]
    labels = hand_label_image(hand_doc, meta, inset, (100, 100))
    assert labels[30, 30] == INTERIOR
    assert labels[11, 30] == BOUNDARY
    # Colored or not, untraced ground is background: the point of hand labels.
    assert labels[70, 20] == BACKGROUND
    assert labels[20, 70] == BACKGROUND
    assert labels[90, 90] == IGNORE  # the untraced detection's disk
    assert labels[82, 82] == IGNORE  # the inset
    assert labels.shape == (100, 100)


def test_load_example_prefers_a_label_dir_image(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    label_dir = tmp_path / "hand"
    label_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(data_dir / "v__p0.jpg")
    Image.fromarray(np.full((4, 4), INTERIOR, np.uint8)).save(
        data_dir / "v__p0.label.png"
    )
    Image.fromarray(np.full((4, 4), BOUNDARY, np.uint8)).save(
        label_dir / "v__p0.label.png"
    )
    (data_dir / "v__p0.json").write_text(
        json.dumps(prepared_meta(1.0, 10.0, []).to_json())
    )
    assert load_example(data_dir, "v__p0")[1].max() == INTERIOR
    assert load_example(data_dir, "v__p0", label_dir)[1].max() == BOUNDARY
    assert load_example(data_dir, "v__p0", tmp_path / "none")[1].max() == INTERIOR
