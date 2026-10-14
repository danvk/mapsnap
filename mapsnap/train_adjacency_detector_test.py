"""Tests for the adjacency-number detector trainer."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import LineString, Polygon

from mapsnap.oim_neighbours import OimPage
from mapsnap.train_adjacency_detector import (
    DEFAULT_ADOPT_HEIGHTS,
    IGNORE,
    TILE,
    adopt_boxes,
    adoption_heights,
    box_bounds,
    candidate_components,
    confirmed_reads,
    fill,
    height_filtered,
    is_validation,
    manifest_positives,
    masked_loss,
    norm,
    oim_exports,
    page_entry,
    page_mask,
    page_name,
    read_page,
    sample_tile,
    tile_centre,
    truth_series,
    validate,
    volume_entries,
)


def entry(**overrides) -> dict:
    """A 1000 x 1000 page whose content outline is inset 100 px, with a neighbour '7'
    along the right edge and '3' along the top."""
    base = {
        "item": "sanborn09999_001",
        "key": "p5",
        "width": 1000,
        "height": 1000,
        "selector": [[100, 100], [900, 100], [900, 900], [100, 900], [100, 100]],
        "neighbours": {"7": [[900, 100], [900, 900]], "3": [[100, 100], [900, 100]]},
        "detections": [],
        "adopted": [],
    }
    return base | overrides


def read(key: str, box: tuple[int, int, int, int]) -> dict:
    """A digit read over box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    return {"key": key, "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_norm_and_box_bounds():
    assert norm(" 12 a") == "12A"
    assert box_bounds([[3, 4], [10, 2], [5, 9]]) == (3, 2, 10, 9)


def test_truth_series_covers_the_sample_and_dev_volumes():
    series = truth_series()
    assert "sanborn05511" in series  # Hudson
    assert len(series) > 50


def test_oim_exports_prefers_the_api_truth_over_editions(tmp_path: Path):
    (tmp_path / "oim" / "sanborn1_001").mkdir(parents=True)
    (tmp_path / "oim" / "sanborn1_001" / "main.iiif.json").write_text("{}")
    for item in ("sanborn1_001", "sanborn2_001"):
        (tmp_path / "oim-editions" / item).mkdir(parents=True)
        (tmp_path / "oim-editions" / item / "main.export.iiif.json").write_text("{}")
    exports = oim_exports(tmp_path)
    assert exports["sanborn1_001"].name == "main.iiif.json"
    assert exports["sanborn2_001"].name == "main.export.iiif.json"


def oim_page(key: str) -> OimPage:
    """A 4000 x 4000 canvas page (the mirror is 25%), outline inset 400 px."""
    selector = Polygon([(400, 400), (3600, 400), (3600, 3600), (400, 3600)])
    return OimPage(key, np.eye(3), selector, selector, 4000, 4000)


def test_page_entry_scales_to_the_mirror_and_keeps_only_neighbour_reads():
    neighbours = {"7": LineString([(3600, 400), (3600, 3600)])}
    adjacency_page = {
        "width": 1000,
        "height": 1000,
        "detections": [read("7", (910, 480, 950, 520)), read("12", (0, 0, 40, 40))],
    }
    built = page_entry(oim_page("p5"), neighbours, adjacency_page, "s3://x/p5.jpg")
    assert built["selector"][0] == [100.0, 100.0]
    assert built["neighbours"]["7"] == [[900.0, 100.0], [900.0, 900.0]]
    assert [d["key"] for d in built["detections"]] == ["7"]


def test_volume_entries_skips_split_and_unmirrored_pages(tmp_path: Path, monkeypatch):
    from mapsnap import train_adjacency_detector as trainer

    pages = [oim_page("p5"), oim_page("p6__1"), oim_page("p6__2"), oim_page("p9")]
    monkeypatch.setattr(trainer, "load_pages", lambda export: pages)
    monkeypatch.setattr(
        trainer,
        "predicted_neighbours",
        lambda page, others: {"7": LineString([(3600, 400), (3600, 3600)])},
    )
    adjacency = {"pages": {"p5": {"width": 1000, "height": 1000}, "p6__1": {}}}
    entries = volume_entries(tmp_path / "x.json", adjacency, "s3://bucket/item")
    assert [e["key"] for e in entries] == ["p5"]
    assert entries[0]["image"] == "s3://bucket/item/p5.jpg"


def test_confirmed_reads_needs_the_right_number_height_and_place():
    page = entry(
        detections=[
            read("7", (910, 480, 950, 520)),  # right number, right edge
            read("7", (300, 480, 340, 520)),  # right number, far from its edge
            read("3", (910, 480, 950, 520)),  # a neighbour's number on the wrong edge
            read("7", (910, 600, 950, 610)),  # too short to be a printed reference
            read("12", (910, 700, 950, 740)),  # not a neighbour
        ]
    )
    assert confirmed_reads(page) == [[910, 480, 950, 520, "7"]]


def test_height_filtered_drops_reads_off_the_volume_median():
    reads = {
        "a": [[0, 0, 10, 40, "1"], [0, 0, 10, 42, "2"]],
        "b": [[0, 0, 10, 44, "3"], [0, 0, 10, 10, "4"], [0, 0, 10, 100, "5"]],
    }
    kept = height_filtered(reads)
    assert [r[4] for r in kept["a"] + kept["b"]] == ["1", "2", "3"]
    assert height_filtered({"a": []}) == {"a": []}


def test_fill_rasterizes_polygons():
    mask = np.zeros((10, 10), np.uint8)
    fill(mask, Polygon([(2, 2), (6, 2), (6, 6), (2, 6)]), 9)
    assert mask[4, 4] == 9 and mask[8, 8] == 0


def test_page_mask_marks_positives_and_ignores_unconfirmed_stretches():
    page = entry()
    mask = page_mask(page, [[910, 480, 950, 520, "7"]])
    assert mask[500, 930] == 1  # the confirmed reference
    assert mask[700, 930] == 0  # elsewhere along a confirmed stretch: negative
    assert mask[50, 500] == IGNORE  # along '3', which nothing confirmed
    assert mask[500, 50] == IGNORE  # the left edge borders no OIM neighbour
    assert mask[500, 500] == 0  # the map interior


def test_manifest_positives_filters_per_volume_and_adds_adopted_boxes():
    first = entry(key="p5", detections=[read("7", (910, 480, 950, 520))])
    second = entry(key="p6", adopted=[[910, 300, 950, 340, "3"]])
    positives = manifest_positives([first, second])
    assert positives[page_name(first)] == [[910, 480, 950, 520, "7"]]
    assert positives[page_name(second)] == []
    with_adopted = manifest_positives([first, second], adopted=True)
    assert with_adopted[page_name(second)] == [[910, 300, 950, 340, "3"]]


def test_is_validation_holds_out_whole_volumes():
    held = {is_validation(f"sanborn{i:05d}_001") for i in range(100)}
    assert held == {True, False}


def test_read_page(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.full((20, 30), 200, np.uint8))
    cv2.imwrite(str(tmp_path / "masks" / "a.png"), np.ones((20, 30), np.uint8))
    image, mask = read_page(tmp_path, tmp_path / "masks", "a")
    assert image.shape == mask.shape == (20, 30)


def test_tile_centre_stays_on_the_page():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y = tile_centre((1000, 800), [[100, 100, 140, 140, "1"]], rng, 512)
        assert -300 <= x <= 1100 and -300 <= y <= 1300


def test_sample_tile_is_tile_sized_and_keeps_mask_values():
    rng = np.random.default_rng(1)
    image = np.full((700, 600), 220, np.uint8)
    mask = np.zeros((700, 600), np.uint8)
    mask[300:340, 280:320] = 1
    mask[:, :50] = IGNORE
    for _ in range(10):
        tile, tile_mask = sample_tile(image, mask, [[280, 300, 320, 340, "4"]], rng)
        assert tile.shape == tile_mask.shape == (TILE, TILE)
        assert set(np.unique(tile_mask)) <= {0, 1, IGNORE}


def test_masked_loss_ignores_unweighted_pixels():
    logits = torch.zeros(1, 1, 4, 4)
    targets = torch.zeros(1, 1, 4, 4)
    weights = torch.ones(1, 1, 4, 4)
    baseline = masked_loss(logits, targets, weights)
    # A confidently wrong pixel only counts when it is weighted.
    logits[0, 0, 0, 0] = 20.0
    weights_without = weights.clone()
    weights_without[0, 0, 0, 0] = 0
    assert masked_loss(logits, targets, weights) > baseline
    assert torch.isclose(
        masked_loss(logits, targets, weights_without),
        masked_loss(torch.zeros(1, 1, 4, 4), targets, weights_without),
        atol=1e-3,
    )


def test_validate_counts_found_boxes_and_stray_components(monkeypatch):
    from mapsnap import train_adjacency_detector as trainer
    from mapsnap.adjacency_detector import build_detector

    probability = np.zeros((100, 100), np.float32)
    probability[10:30, 10:30] = 0.9  # covers the positive
    probability[60:80, 60:80] = 0.9  # a stray component
    monkeypatch.setattr(
        trainer, "predict_page", lambda model, image, device: probability
    )
    mask = np.zeros((100, 100), np.uint8)
    mask[15:25, 15:25] = 1
    report = validate(
        build_detector(),
        [(np.zeros((100, 100), np.uint8), mask, [[15, 15, 25, 25, "1"]])],
        "cpu",
    )
    assert "1/1" in report and "stray components/page 1.00" in report


def test_adoption_heights_falls_back_without_enough_boxes():
    assert adoption_heights([]) == DEFAULT_ADOPT_HEIGHTS
    low, high = adoption_heights([[0, 0, 10, 40, "1"]] * 3)
    assert low == 24 and high > 40


def test_candidate_components_filters_height_and_existing_positives():
    components = [(910, 480, 950, 520), (910, 600, 950, 605), (912, 700, 948, 740)]
    kept = candidate_components(components, [[910, 690, 950, 745, "7"]], (20, 100))
    assert kept == [(910, 480, 950, 520)]


def test_adopt_boxes_takes_a_lone_candidate_unless_the_reader_disagrees():
    page = entry()
    lone = [(910, 480, 950, 520)]
    adopted, outcomes = adopt_boxes(page, [], lone)
    assert adopted == [[910.0, 480.0, 950.0, 520.0, "7"]]
    assert outcomes["adopted"] == 1 and outcomes["none"] == 1  # '3' has no candidate
    # A confident read of another number vetoes; a read containing it does not.
    vetoed, outcomes = adopt_boxes(page, [], lone, read=lambda c: ("12", 0.9))
    assert vetoed == [] and outcomes["read as another number"] == 1
    kept, _ = adopt_boxes(page, [], lone, read=lambda c: ("71", 0.9))
    assert len(kept) == 1
    # Two candidates near one stretch are ambiguous; confirmed neighbours are skipped.
    two = [(910, 300, 950, 340), (910, 600, 950, 640)]
    adopted, outcomes = adopt_boxes(page, [], two)
    assert adopted == [] and outcomes["ambiguous"] == 1
    adopted, _ = adopt_boxes(page, [[910, 480, 950, 520, "7"]], lone)
    assert adopted == []


def test_manifest_round_trips_through_json():
    page = entry(adopted=[[1.0, 2.0, 3.0, 4.0, "7"]])
    assert json.loads(json.dumps(page)) == page


def test_read_component_returns_text_and_confidence():
    from mapsnap.train_adjacency_detector import read_component

    class Reader:
        def __init__(self, results):
            self.results = results

        def recognize(self, gray, horizontal, free, allowlist):
            assert horizontal == [[1, 3, 2, 4]] and allowlist == "0123456789"
            return self.results

    gray = np.zeros((5, 5), np.uint8)
    assert read_component(Reader([([], " 12 ", 0.875)]), gray, (1, 2, 3, 4)) == (
        "12",
        0.875,
    )
    assert read_component(Reader([]), gray, (1, 2, 3, 4)) == ("", 0.0)
