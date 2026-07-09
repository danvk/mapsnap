from mapsnap.keymap.score_regions import (
    greedy_match,
    panel_polygons,
    scale_to,
    score_regions,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def square(x: float, y: float, size: float = 100.0) -> list[list[float]]:
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]


def doc(panels: list[list[list[float]]], labels: list[str], size: int = 1000) -> dict:
    return {
        "image": "p0.jpg",
        "width": size,
        "height": size,
        "panels": panels,
        "labels": labels,
    }


def test_perfect_match_scores_one():
    truth = doc([square(0, 0)], ["7"])
    predicted = doc([square(0, 0)], ["7"])
    score = score_regions(truth, predicted)
    assert score.ious == [("7", 1.0)]
    assert score.spurious == 0
    assert score.predicted_overlap == 0.0


def test_half_shifted_square():
    truth = doc([square(0, 0)], ["7"])
    predicted = doc([square(50, 0)], ["7"])  # half-overlap -> IoU 1/3
    score = score_regions(truth, predicted)
    assert abs(score.ious[0][1] - 1 / 3) < 1e-9


def test_missing_prediction_scores_zero_and_spurious_counted():
    truth = doc([square(0, 0)], ["7"])
    predicted = doc([square(500, 500)], ["8"])  # wrong label entirely
    score = score_regions(truth, predicted)
    assert score.ious == [("7", 0.0)]
    assert score.spurious == 1


def test_split_pages_match_by_greatest_intersection():
    # Page 7 has two truth footprints; predictions are swapped in order but each
    # should pair with the overlapping one.
    truth = doc([square(0, 0), square(500, 500)], ["7", "7"])
    predicted = doc([square(510, 510), square(5, 5)], ["7", "7"])
    score = score_regions(truth, predicted)
    ious = sorted(iou for _, iou in score.ious)
    assert all(iou > 0.6 for iou in ious)  # 10 px shift on 100 px squares


def test_greedy_match_prefers_larger_intersection():
    truths: list[BaseGeometry] = [Polygon(square(0, 0)), Polygon(square(80, 0))]
    # Overlaps both truths; more with the second.
    predictions: list[BaseGeometry] = [Polygon(square(60, 0))]
    matched = greedy_match(truths, predictions)
    assert matched == [(1, 0)]


def test_predicted_overlap_fraction():
    truth = doc([square(0, 0)], ["7"])
    # Two predictions for different labels that overlap each other by half.
    predicted = doc([square(0, 0), square(50, 0)], ["7", "8"])
    score = score_regions(truth, predicted)
    # intersection 50x100 = 5000 over total 20000 predicted area
    assert abs(score.predicted_overlap - 0.25) < 1e-9


def test_scale_to_rescales_coordinates():
    small = doc([square(0, 0, 10)], ["7"], size=100)
    big = doc([], [], size=1000)
    scale_to(small, big)
    assert small["panels"][0][1] == [100.0, 0.0]


def test_panel_polygons_repairs_self_intersections():
    bowtie = [[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0], [0.0, 0.0]]
    polygons = panel_polygons(doc([bowtie], ["7"]))
    assert len(polygons) == 1
    assert polygons[0][1].area > 0
