from mapsnap.keymap.detect_numbers_cnn import (
    nms_peaks,
    region_bounds,
    scores_to_grid,
    window_centers,
)


def test_scores_to_grid_is_row_major():
    # window_centers iterates y-outer, x-inner, so scores reshape to (n_rows, n_cols).
    grid = scores_to_grid([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], n_cols=3, n_rows=2)
    assert grid.shape == (2, 3)
    assert grid[0, 2] == 0.2 and grid[1, 0] == 0.3


def test_window_centers_grid():
    centers = window_centers(100, 50, 25)
    assert (12, 12) in centers and (87, 37) in centers
    assert all(0 <= x < 100 and 0 <= y < 50 for x, y in centers)


def test_nms_peaks_suppresses_close_lower_scores():
    centers = [(10, 10), (12, 12), (100, 100)]
    scores = [0.9, 0.8, 0.7]
    kept = nms_peaks(centers, scores, min_dist=20)
    assert centers[kept[0]] == (10, 10)
    assert (100, 100) in [centers[k] for k in kept]
    assert (12, 12) not in [centers[k] for k in kept]


def test_nms_peaks_keeps_distant_peaks():
    centers = [(0, 0), (200, 0), (0, 200)]
    scores = [0.5, 0.6, 0.7]
    assert len(nms_peaks(centers, scores, min_dist=50)) == 3


def test_region_bounds_clamps():
    assert region_bounds(10, 10, 50, 520, 520) == (0, 0, 60, 60)
    assert region_bounds(500, 500, 50, 520, 520) == (450, 450, 520, 520)
