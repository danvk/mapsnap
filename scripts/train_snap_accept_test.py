"""Tests for train_snap_accept.py."""

import numpy as np
from train_snap_accept import best_threshold, rescue_example, standardize


def test_best_threshold_maximizes_net_value():
    probabilities = np.array([0.9, 0.8, 0.6, 0.4, 0.2])
    values = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
    # Accepting down to 0.4 nets 2; down to 0.8 also nets 2, and comes first.
    assert best_threshold(probabilities, values) == (0.8, 2.0)


def test_best_threshold_accepts_nothing_when_nothing_pays():
    assert best_threshold(np.array([0.9, 0.5]), np.array([-1.0, 0.0])) == (1.0, 0.0)


def test_standardize_reuses_fitted_columns():
    X, columns = standardize([{"a": 1.0}, {"a": 3.0}])
    assert np.allclose(X[:, 0], [-1.0, 1.0])
    X_new, _ = standardize([{"a": 5.0}], columns)
    assert np.allclose(X_new[:, 0], [3.0])


def test_rescue_example_values_good_and_disaster():
    def page(rmse_ft: float) -> dict:
        return {
            "target": "p1",
            "fit_state": "nofit",
            "width": 100,
            "height": 100,
            "candidates": [
                {
                    "world_affine": [[1e-5, 0, -74], [0, -1e-5, 40]],
                    "select_score": 1.0,
                    "inlier_frac": 0.5,
                    "ncc_fine": 0.5,
                    "chamfer_mean_m": 5.0,
                    "rmse_ft": rmse_ft,
                }
            ],
        }

    def value(rmse_ft: float) -> float:
        example = rescue_example(page(rmse_ft))
        assert example is not None
        return example[1]

    assert value(10.0) == 1.0
    assert value(100.0) == 0.0
    assert value(500.0) == -1.0
    fitted = page(10.0)
    fitted["fit_state"] = "fitted"
    assert rescue_example(fitted) is None
