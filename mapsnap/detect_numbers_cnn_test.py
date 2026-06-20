from mapsnap.detect_numbers_cnn import (
    boxes_from_centers,
    nms_peaks,
    window_centers,
)


def test_window_centers_grid():
    centers = window_centers(100, 50, 25)
    # xs: 12, 37, 62, 87 ; ys: 12, 37
    assert (12, 12) in centers and (87, 37) in centers
    assert all(0 <= x < 100 and 0 <= y < 50 for x, y in centers)


def test_nms_peaks_suppresses_close_lower_scores():
    centers = [(10, 10), (12, 12), (100, 100)]
    scores = [0.9, 0.8, 0.7]
    kept = nms_peaks(centers, scores, min_dist=20)
    # (12,12) is within 20 of the higher-scoring (10,10) -> suppressed.
    assert centers[kept[0]] == (10, 10)
    assert (100, 100) in [centers[k] for k in kept]
    assert (12, 12) not in [centers[k] for k in kept]


def test_nms_peaks_keeps_distant_peaks():
    centers = [(0, 0), (200, 0), (0, 200)]
    scores = [0.5, 0.6, 0.7]
    kept = nms_peaks(centers, scores, min_dist=50)
    assert len(kept) == 3


def test_boxes_from_centers_clamps_to_image():
    boxes = boxes_from_centers(
        [(10, 10), (500, 500)], box_size=100, width=520, height=520
    )
    # First box clamps at the top-left corner.
    assert boxes[0] == [[0, 0], [60, 0], [60, 60], [0, 60]]
    # Second box clamps at the bottom-right.
    assert boxes[1] == [[450, 450], [520, 450], [520, 520], [450, 520]]
