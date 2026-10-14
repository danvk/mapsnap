"""Tests for the adjacency-number detector's inference helpers."""

from pathlib import Path

import cv2
import numpy as np
import torch

from mapsnap import adjacency_detector
from mapsnap.adjacency_detector import (
    best_reads,
    best_unresolved,
    build_detector,
    detection_centre,
    detector_digit_detections,
    load_detector,
    number_components,
    resized_box,
)


def detection(box: tuple[int, int, int, int], key: str, confidence: float) -> dict:
    """A digit detection over box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    return {
        "key": key,
        "confidence": confidence,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def test_number_components_drops_specks():
    probability = np.zeros((100, 200), np.float32)
    probability[10:30, 20:60] = 0.9  # a number
    probability[80:82, 150:152] = 0.9  # a speck
    assert number_components(probability) == [(20, 10, 60, 30)]
    assert number_components(probability, min_area=1) == [
        (20, 10, 60, 30),
        (150, 80, 152, 82),
    ]


def test_resized_box_grows_shrinks_and_clamps():
    assert resized_box((10, 10, 50, 30), 0.0, (100, 100)) == [10, 50, 10, 30]
    assert resized_box((10, 10, 50, 30), 0.5, (100, 100)) == [0, 60, 0, 40]
    assert resized_box((10, 10, 50, 30), -0.1, (100, 100)) == [12, 48, 12, 28]


def test_detection_centre():
    assert detection_centre(detection((0, 0, 10, 20), "1", 1.0)) == (5.0, 10.0)


def test_best_reads_keeps_the_most_confident_read_per_component():
    components = [(100, 100, 140, 140), (300, 300, 340, 340)]
    padded = [detection((95, 95, 145, 145), "251", 0.6)]
    bare = [detection((100, 100, 140, 140), "25", 0.98)]
    shrunk = [
        detection((104, 104, 136, 136), "2", 0.4),
        detection((0, 0, 10, 10), "9", 1.0),
    ]
    kept = best_reads(components, [padded, bare, shrunk])
    # The second component has no read; the stray read outside both is dropped.
    assert [d["key"] for d in kept] == ["25"]


def test_load_detector_round_trips_weights(tmp_path: Path):
    path = tmp_path / "detector.pt"
    torch.save(build_detector().state_dict(), path)
    model = load_detector(path)
    assert not model.training


def test_detector_digit_detections_reads_components_at_each_size(tmp_path, monkeypatch):
    image_path = tmp_path / "p1.jpg"
    cv2.imwrite(str(image_path), np.full((200, 300), 230, np.uint8))
    probability = np.zeros((200, 300), np.float32)
    probability[50:90, 100:140] = 0.9

    def fake_predict(model, gray, device):
        return probability

    calls = []

    def fake_digit_detections(
        image_path, reader, valid_keys, craft_boxes, units=None, *, unresolved=None
    ):
        boxes = craft_boxes[0]
        calls.append(boxes)
        x1, x2, y1, y2 = boxes[0]
        # Only the bare box reads cleanly.
        if [x1, x2, y1, y2] == [100, 140, 50, 90]:
            return [detection((x1, y1, x2, y2), "12", 0.99)]
        return [detection((x1, y1, x2, y2), "128", 0.5)]

    monkeypatch.setattr(adjacency_detector, "predict_page", fake_predict)
    monkeypatch.setattr(
        adjacency_detector.page_adjacency, "digit_detections", fake_digit_detections
    )
    read = detector_digit_detections(build_detector())
    kept = read(image_path, None, {"12", "128"})
    assert len(calls) == 3
    assert [d["key"] for d in kept] == ["12"]


def unresolved_read(
    box: tuple[int, int, int, int], text: str, confidence: float
) -> dict:
    """A read that resolved to no page key: a detection without key/via/number."""
    x0, y0, x1, y1 = box
    return {
        "text": text,
        "confidence": confidence,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    }


def test_best_unresolved_keeps_one_read_per_component_without_a_valid_one():
    components = [(100, 50, 140, 90), (200, 50, 240, 90)]
    kept = [detection((100, 50, 140, 90), "12", 0.99)]
    unresolved = [
        unresolved_read((100, 50, 140, 90), "13", 0.9),  # its component resolved
        unresolved_read((200, 50, 240, 90), "2", 0.7),
        unresolved_read((198, 48, 242, 92), "2", 0.95),
        unresolved_read((10, 10, 20, 20), "7", 0.99),  # in no component
    ]
    assert best_unresolved(components, kept, unresolved) == [unresolved[2]]


def test_detector_digit_detections_records_unresolved_components(tmp_path, monkeypatch):
    # Kennewick 1909 (#555): keys p5-p9, but sheets printed 1-5.
    image_path = tmp_path / "p6.jpg"
    cv2.imwrite(str(image_path), np.full((200, 300), 230, np.uint8))
    probability = np.zeros((200, 300), np.float32)
    probability[50:90, 100:140] = 0.9

    def fake_digit_detections(
        image_path, reader, valid_keys, craft_boxes, units=None, *, unresolved=None
    ):
        x1, x2, y1, y2 = craft_boxes[0][0]
        assert unresolved is not None
        unresolved.append(unresolved_read((x1, y1, x2, y2), "2", 0.9))
        return []

    monkeypatch.setattr(adjacency_detector, "predict_page", lambda m, g, d: probability)
    monkeypatch.setattr(
        adjacency_detector.page_adjacency, "digit_detections", fake_digit_detections
    )
    read = detector_digit_detections(build_detector())
    unresolved: list[dict] = []
    assert read(image_path, None, {"5", "6"}, unresolved=unresolved) == []
    assert [r["text"] for r in unresolved] == ["2"]
    # Without a list to fill, nothing changes for the caller.
    assert read(image_path, None, {"5", "6"}) == []
