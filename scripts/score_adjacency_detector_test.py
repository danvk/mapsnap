"""Tests for score_adjacency_detector.py."""

import json
from pathlib import Path

from score_adjacency_detector import (
    comparison_labels,
    craft_boxes,
    era,
    located_labels,
    matched_labels,
    score_adjacency,
    verdict,
    write_volume,
)


def detection(key: str, centre: tuple[float, float], **extra) -> dict:
    """A 40 x 40 digit detection centred at centre."""
    x, y = centre
    polygon = [[x - 20, y - 20], [x + 20, y - 20], [x + 20, y + 20], [x - 20, y + 20]]
    return {"key": key, "text": key, "polygon": polygon} | extra


def test_era():
    assert era(1885) == "1867-1899"
    assert era(1912) == "1900-1929"
    assert era(1950) == "1930-1961"


def test_matched_labels_needs_the_number_nearby_and_matches_one_to_one():
    labels = [
        {"x": 100, "y": 100, "text": "12"},
        {"x": 500, "y": 500, "text": "7"},
        {"x": 900, "y": 900, "text": "0"},
        {"x": 120, "y": 100, "text": "12"},
    ]
    detections = [
        detection("12", (110, 100)),
        detection("7", (100, 100)),  # wrong number for the label it sits on
        detection("", (900, 905), via="zero", text="0"),
    ]
    assert matched_labels(labels, detections) == {0, 2}


def test_located_labels_only_counts_boxes_the_adjacency_step_would_read():
    labels = [{"x": 30, "y": 500, "text": "4"}, {"x": 500, "y": 500, "text": "5"}]
    tall_in_margin = [10, 50, 480, 520]
    short_in_margin = [10, 50, 495, 505]
    tall_in_middle = [480, 520, 480, 520]
    boxes = [tall_in_margin, short_in_margin, tall_in_middle]
    assert located_labels(labels, boxes, (1000, 1000)) == {0}


def test_craft_boxes_reads_horizontal_and_free_boxes(tmp_path: Path):
    path = tmp_path / "p1.boxes.json"
    path.write_text(
        json.dumps(
            {
                "boxes": [
                    {"angle": 90, "horizontal_list": [[0, 1, 0, 1]]},
                    {
                        "angle": 0,
                        "horizontal_list": [[10, 50, 20, 60]],
                        "free_list": [[[5, 5], [15, 6], [14, 20], [4, 19]]],
                    },
                ]
            }
        )
    )
    assert craft_boxes(path) == [[10, 50, 20, 60], [4, 15, 5, 20]]
    assert craft_boxes(tmp_path / "missing.json") == []


def test_score_adjacency_counts_claims_and_edge_ends():
    truth = {
        "p5": {
            "labels": [
                {"x": 950, "y": 500, "text": "6"},
                {"x": 500, "y": 30, "text": "0"},
            ]
        },
        "p6__1": {"labels": [{"x": 30, "y": 400, "text": "5"}]},
    }
    adjacency = {
        "pages": {
            "p5": {
                "detections": [
                    detection("6", (950, 505), claim=True),
                    detection("9", (10, 10)),
                ]
            },
            # A panel: its detections are in the parent's frame, offset by the panel bbox.
            "p6__1": {
                "panel_bbox": [1000, 100, 2000, 900],
                "detections": [detection("5", (1030, 500), claim=True)],
            },
        },
        "adjacency": [["p5", "p6__1"]],
        "one_sided": [["p5", "p9"]],
    }
    t = score_adjacency(adjacency, truth)
    assert t["labels"] == 2  # zeros are not claimable
    assert t["claims"] == 2 and t["labels claimed"] == 2
    assert t["mutual edges"] == 1
    assert (t["adjacency"], t["adjacency correct"]) == (2, 2)
    assert (t["one_sided"], t["one_sided correct"]) == (1, 0)


def test_comparison_labels_marks_every_outcome():
    labels = [
        {"x": 100, "y": 100, "text": "12"},  # read
        {"x": 400, "y": 100, "text": "7"},  # misread nearby
        {"x": 700, "y": 100, "text": "9"},  # a component, but no valid read
        {"x": 900, "y": 900, "text": "0"},  # nothing there
    ]
    reads = [
        detection("12", (105, 100)),
        detection("71", (410, 100)),
        detection("33", (500, 600)),
    ]
    unread = [(705.0, 95.0), (50.0, 900.0)]
    marked = comparison_labels(labels, reads, unread)
    assert [m["text"] for m in marked] == [
        "12 ✓",
        "7 ✗ read 71",
        "9 ✗ unread",
        "0 ✗ missed",
        "33 ✗ extra",
        "",
    ]
    assert (marked[-1]["x"], marked[-1]["y"]) == (50.0, 900.0)


def test_write_volume_links_images_and_writes_truth(tmp_path: Path):
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "p1.jpg").write_bytes(b"jpeg")
    (sample / "sources.json").write_text('{"pages": {}}')
    pages = {"p1": {"width": 10, "height": 20, "labels": []}}
    write_volume(tmp_path / "out", sample, pages)
    assert (tmp_path / "out" / "p1.jpg").read_bytes() == b"jpeg"
    assert (tmp_path / "out" / "p1.jpg").is_file()
    assert json.loads((tmp_path / "out" / "adjacency-truth.json").read_text()) == {
        "pages": pages
    }
    assert (tmp_path / "out" / "sources.json").exists()


def test_verdict():
    assert verdict("12 ✓") == "✓"
    assert verdict("7 ✗ read 71") == "✗ read"
    assert verdict("0 ✗ missed") == "✗ missed"
    assert verdict("") == "unread box"
