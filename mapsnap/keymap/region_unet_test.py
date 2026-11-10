import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapsnap.keymap.region_unet import (
    BACKGROUND,
    BOUNDARY,
    FALLBACK_LONG_SIDE,
    INTERIOR,
    NUM_CLASSES,
    TARGET_SPACING_PX,
    UNSEEDED_LABEL,
    DecodedRegions,
    DecodeParams,
    blend_window,
    build_model,
    class_map_image,
    decode_regions,
    discover_keymap_sources,
    fill_small_holes,
    keymap_keys,
    keymap_sources,
    load_region_unet,
    model_input,
    predict_class_maps,
    predict_maps,
    region_polygons,
    resize_image,
    scale_points,
    segment_keymap,
    segment_keymap_image,
    sidecar_dir,
    tile_origins,
    working_scale_for_seeds,
)


def grid_seeds(spacing: float, count: int = 3) -> list[tuple[float, float]]:
    return [
        (spacing * (i + 1), spacing * (j + 1))
        for i in range(count)
        for j in range(count)
    ]


def test_working_scale_for_seeds_normalizes_spacing():
    seeds = grid_seeds(256.0)
    assert working_scale_for_seeds((2000, 2000), seeds) == TARGET_SPACING_PX / 256.0
    # Never upscales a map whose numbers are already close together.
    assert working_scale_for_seeds((2000, 2000), grid_seeds(64.0)) == 1.0


def test_working_scale_for_seeds_falls_back_with_few_seeds():
    assert (
        working_scale_for_seeds((6000, 4000), [(1, 1), (2, 2)])
        == FALLBACK_LONG_SIDE / 6000
    )


def test_scale_points_and_resize_image():
    assert scale_points([(10.0, 20.0)], 0.5) == [(5.0, 10.0)]
    rgb = np.full((100, 200, 3), 200, np.uint8)
    assert resize_image(rgb, 0.5).shape == (50, 100, 3)
    assert resize_image(rgb, 1.0) is rgb


def test_model_input_scales_paper_and_ink():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[0] = 255
    inputs = model_input(rgb)
    assert inputs.shape == (3, 4, 4)
    assert inputs.dtype == np.float32
    assert inputs.flags["C_CONTIGUOUS"]  # channels-last strides are 4x slower on MPS
    assert inputs[0, 0, 0] == 2.0  # paper
    assert inputs[0, 1, 0] == -2.0  # ink


def test_tile_origins_cover_the_length():
    assert tile_origins(300, 512, 64) == [0]
    origins = tile_origins(1000, 512, 64)
    assert origins[0] == 0 and origins[-1] == 1000 - 512
    assert all(b - a <= 512 - 64 for a, b in itertools.pairwise(origins))


def test_blend_window_is_positive_and_symmetric():
    window = blend_window(16)
    assert window.shape == (16, 16)
    assert (window > 0).all()
    assert np.allclose(window, window[::-1, ::-1])


def test_predict_maps_tiles_to_a_probability_map():
    model = build_model(base=4).eval()
    inputs = np.zeros((3, 70, 90), np.float32)
    probabilities = predict_maps(
        model, inputs, torch.device("cpu"), tile=64, overlap=16
    )
    assert probabilities.shape == (NUM_CLASSES, 70, 90)
    assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-4)
    # Smaller than one tile: padded and cropped back.
    small = predict_maps(
        model,
        np.zeros((3, 20, 30), np.float32),
        torch.device("cpu"),
        tile=64,
        overlap=16,
    )
    assert small.shape == (NUM_CLASSES, 20, 30)


def test_predict_class_maps_flip_average_is_a_distribution():
    model = build_model(base=4).eval()
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(3, 70, 90)).astype(np.float32)
    plain = predict_class_maps(model, inputs, torch.device("cpu"), flips=False)
    averaged = predict_class_maps(model, inputs, torch.device("cpu"))
    assert averaged.shape == plain.shape == (NUM_CLASSES, 70, 90)
    assert np.allclose(averaged.sum(axis=0), 1.0, atol=1e-4)
    assert not np.allclose(averaged, plain)  # the flips contribute
    # Averaging is itself flip-symmetric: a flipped input gives the flipped output.
    flipped = predict_class_maps(
        model, np.ascontiguousarray(inputs[:, :, ::-1]), torch.device("cpu")
    )
    assert np.allclose(flipped[:, :, ::-1], averaged, atol=1e-5)


def test_load_region_unet_roundtrip(tmp_path: Path):
    x = torch.randn((1, 3, 32, 32))
    for norm in ("batch", "group"):
        model = build_model(base=4, norm=norm).eval()
        torch.save(model.state_dict(), tmp_path / f"{norm}.pt")
        loaded = load_region_unet(tmp_path / f"{norm}.pt", torch.device("cpu"))
        assert torch.allclose(model(x), loaded(x))


def synthetic_probabilities() -> np.ndarray:
    """Two blocks side by side, a boundary column between them, paper around."""
    probabilities = np.zeros((NUM_CLASSES, 140, 200), np.float32)
    probabilities[BACKGROUND] = 1.0
    block = (slice(30, 110), slice(30, 170))
    probabilities[BACKGROUND][block] = 0.0
    probabilities[INTERIOR][block] = 1.0
    probabilities[INTERIOR][30:110, 98:102] = 0.0
    probabilities[BOUNDARY][30:110, 98:102] = 1.0
    return probabilities


def test_decode_regions_splits_blocks_at_the_boundary_and_drops_paper_seeds():
    probabilities = synthetic_probabilities()
    seeds = [(60.0, 70.0), (140.0, 70.0), (5.0, 5.0)]  # third seed sits on paper
    decoded = decode_regions(probabilities, seeds, 60.0, DecodeParams())
    assert decoded.seed_labels == [1, 2, 0]
    assert decoded.unseeded_labels == []
    assert decoded.labels[70, 60] == 1
    assert decoded.labels[70, 140] == 2
    assert decoded.labels[5, 5] == 0
    assert decoded.labels[70, 90] == 1 and decoded.labels[70, 110] == 2


def test_fill_small_holes_fills_pockets_not_margins():
    mask = np.ones((40, 60), bool)
    mask[10:14, 10:14] = False  # a glyph-sized pocket
    mask[:, 30:33] = False  # paper reaching the border: a gap between blocks
    filled = fill_small_holes(mask, max_area=50)
    assert filled[12, 12] and not filled[20, 31]
    assert not fill_small_holes(mask, max_area=10)[
        12, 12
    ]  # 16 px pocket is above the floor


def test_decode_regions_survives_a_background_glyph_at_the_seed():
    probabilities = synthetic_probabilities()
    # The model paints a paper pocket over the page number at the seed.
    probabilities[BACKGROUND][64:76, 54:66] = 1.0
    probabilities[INTERIOR][64:76, 54:66] = 0.0
    decoded = decode_regions(
        probabilities, [(60.0, 70.0), (140.0, 70.0)], 60.0, DecodeParams()
    )
    assert decoded.seed_labels == [1, 2]
    assert decoded.labels[40, 40] == 1 and decoded.labels[70, 140] == 2


def test_decode_regions_marks_unseeded_blocks():
    probabilities = synthetic_probabilities()
    decoded = decode_regions(probabilities, [(60.0, 70.0)], 60.0, DecodeParams())
    assert decoded.seed_labels == [1]
    assert decoded.unseeded_labels == [2]
    assert decoded.labels[70, 140] == 2


def test_region_polygons_one_per_label():
    probabilities = synthetic_probabilities()
    decoded = decode_regions(
        probabilities, [(60.0, 70.0), (140.0, 70.0)], 60.0, DecodeParams()
    )
    polygons = region_polygons(decoded, 60.0, DecodeParams())
    assert set(polygons) == {1, 2}
    xs = [x for x, _ in polygons[1]]
    assert min(xs) >= 29 and max(xs) <= 102


def test_region_polygons_drops_rings_and_oversized_unseeded_regions():
    labels = np.zeros((200, 200), np.int32)
    labels[5:195, 5:195] = 1  # a ring around the sheet: label 1 ...
    labels[15:185, 15:185] = 0  # ... with a hole
    labels[50:90, 50:90] = 2  # a real block
    labels[100:190, 100:190] = (
        3  # an unseeded region of 8,100 px: over 12 x spacing^2 at spacing 20
    )
    decoded = DecodedRegions(labels, [1, 2], [3])
    polygons = region_polygons(decoded, 20.0, DecodeParams())
    assert set(polygons) == {2}


def test_segment_keymap_image_returns_full_resolution_polygons():
    model = build_model(base=4).eval()
    rgb = np.full((300, 400, 3), 240, np.uint8)
    seeds = grid_seeds(100.0, 2)
    segmented = segment_keymap_image(
        rgb, seeds, ["1", "2", "3", "4"], model, device=torch.device("cpu")
    )
    assert segmented.scale == 1.0
    assert segmented.probabilities.shape == (NUM_CLASSES, 300, 400)
    assert len(segmented.polygons) == len(segmented.texts)
    assert all(text in {"1", "2", "3", "4", UNSEEDED_LABEL} for text in segmented.texts)


def test_segment_keymap_takes_the_image_from_another_directory(tmp_path: Path):
    # A mirror run's sidecars sit under runs/<tag>/raw/ while the sheet is in raw/ (#554).
    model = build_model(base=4).eval()
    sidecars = tmp_path / "runs" / "corpus-v2" / "raw"
    sidecars.mkdir(parents=True)
    (tmp_path / "raw").mkdir()
    image = tmp_path / "raw" / "p0.jpg"
    Image.fromarray(np.full((300, 400, 3), 240, np.uint8)).save(image)
    keymap_json = sidecars / "p0.keymap.json"
    streets = [
        {
            "text": str(i + 1),
            "polygon": [[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]],
        }
        for i, (x, y) in enumerate(grid_seeds(100.0, 2))
    ]
    keymap_json.write_text(json.dumps({"streets": streets}))
    segmented, used_image = segment_keymap(
        keymap_json, model, torch.device("cpu"), image_path=image
    )
    assert used_image == image
    assert segmented.probabilities.shape == (NUM_CLASSES, 300, 400)
    assert len(segmented.polygons) == len(segmented.texts)


def test_class_map_image_colors():
    probabilities = synthetic_probabilities()
    image = class_map_image(probabilities)
    assert image.shape == (140, 200, 3)
    assert tuple(image[5, 5]) == (255, 255, 255)  # paper white
    assert image[70, 60, 1] == 255 and image[70, 60, 0] == 0  # interior green
    assert image[70, 100, 0] == 255 and image[70, 100, 1] == 0  # boundary red


def make_volume(root: Path, *, run: str | None = None, with_truth: bool = True) -> Path:
    volume = root / "town_st_1900"
    (volume / "raw").mkdir(parents=True)
    (volume / "raw" / "p0.jpg").write_bytes(b"")
    sidecars = volume / "raw" if run is None else volume / "runs" / run / "raw"
    sidecars.mkdir(parents=True, exist_ok=True)
    keymaps = (
        volume / "keymaps.json"
        if run is None
        else volume / "runs" / run / "keymaps.json"
    )
    keymaps.write_text(json.dumps({"keys": ["p0"]}))
    (sidecars / "p0.georef.json").write_text("{}")
    (sidecars / "p0.keymap.json").write_text("{}")
    (sidecars / "p0.regions.panels.json").write_text("{}")
    if with_truth:
        (volume / "main.iiif.json").write_text("{}")
    return volume


def test_keymap_sources_from_raw(tmp_path: Path):
    volume = make_volume(tmp_path)
    assert keymap_keys(volume) == ["p0"]
    assert sidecar_dir(volume, "p0") == volume / "raw"
    sources = keymap_sources(volume, "p0")
    assert sources is not None
    assert sources.name == "town_st_1900__p0"
    assert sources.regions == volume / "raw" / "p0.regions.panels.json"
    assert sources.inset is None
    assert sources.truth == volume / "main.iiif.json"
    assert keymap_sources(volume, "p9") is None


def test_keymap_sources_fall_back_to_a_corpus_run(tmp_path: Path):
    volume = make_volume(tmp_path, run="corpus-v2", with_truth=False)
    assert keymap_keys(volume) == ["p0"]
    assert sidecar_dir(volume, "p0") == volume / "runs" / "corpus-v2" / "raw"
    assert (
        discover_keymap_sources(tmp_path)
        and discover_keymap_sources(tmp_path, require_truth=True) == []
    )
