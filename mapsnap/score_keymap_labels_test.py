from mapsnap.score_keymap_labels import (
    filter_detections,
    match_detections,
    point_in_polygon,
    score,
)

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def detection(polygon, text, confidence=0.9, short_side=100.0):
    return {
        "polygon": polygon,
        "text": text,
        "confidence": confidence,
        "short_side": short_side,
    }


def test_point_in_polygon_inside_and_outside():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((-1, -1), SQUARE)


def test_filter_detections_by_confidence_and_short_side():
    dets = [
        detection(SQUARE, "1", confidence=0.9, short_side=100),
        detection(SQUARE, "2", confidence=0.2, short_side=100),  # low confidence
        detection(SQUARE, "3", confidence=0.9, short_side=30),  # small box
    ]
    kept = filter_detections(dets, min_confidence=0.5, min_short_side=50)
    assert [d["text"] for d in kept] == ["1"]


def test_match_is_one_to_one_highest_confidence_first():
    # Two detections cover the same point; the higher-confidence one claims it.
    dets = [
        detection(SQUARE, "low", confidence=0.3),
        detection(SQUARE, "high", confidence=0.95),
    ]
    labels = [{"x": 5, "y": 5, "text": "high"}]
    matches = match_detections(dets, labels)
    assert len(matches) == 1
    di, li = matches[0]
    assert dets[di]["text"] == "high"


def test_score_precision_recall_and_disagreement():
    far = [[100, 100], [110, 100], [110, 110], [100, 110]]
    dets = [
        detection(SQUARE, "21"),  # matches the point, correct text
        detection(far, "99"),  # matches nothing -> false positive
    ]
    labels = [
        {"x": 5, "y": 5, "text": "12"},  # matched but text disagrees (21 vs 12)
        {"x": 500, "y": 500, "text": "7"},  # unmatched -> false negative
    ]
    result = score(dets, labels)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["text_disagreements"] == 1
    assert result["disagreement_examples"] == [("21", "12")]


def test_score_handles_empty_detections():
    result = score([], [{"x": 1, "y": 1, "text": "1"}])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["true_positives"] == 0
