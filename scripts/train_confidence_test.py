"""Tests for train_confidence.py."""

import numpy as np
import pytest
from train_confidence import auc, design, fit_logistic, fold_of, model_document, predict

from mapsnap.confidence import feature_vector, p_good


def test_auc_by_rank_sum():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_fit_logistic_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2))
    true = np.array([-0.5, 2.0, -1.0])
    p = 1 / (1 + np.exp(-(true[0] + X @ true[1:])))
    y = (rng.random(len(p)) < p).astype(float)
    w = fit_logistic(X, y, l2=0.0)
    assert w == pytest.approx(true, abs=0.08)


def test_design_fills_missing_values_and_adds_indicators():
    raw = np.array([[1.0, 5.0], [np.nan, 7.0], [3.0, 9.0]])
    X, columns = design(raw)
    assert columns[0]["fill"] == 2.0 and columns[0]["indicator"]
    assert not columns[1]["indicator"]
    assert X.shape == (3, 3)  # two standardized columns + one indicator
    assert list(X[:, 2]) == [0.0, 1.0, 0.0]
    # Applying the fitted transform to new rows reuses the training statistics.
    X_new, _ = design(np.array([[np.nan, 7.0]]), columns)
    assert X_new[0, 0] == pytest.approx(
        (2.0 - columns[0]["mean"]) / columns[0]["scale"]
    )


def test_the_written_model_reproduces_the_trained_predictions():
    """train -> JSON -> mapsnap.confidence.p_good must give the same probabilities."""
    rng = np.random.default_rng(1)
    feature_dicts = []
    for _ in range(400):
        gcps = int(rng.integers(0, 12))
        verification = float(rng.normal(1.0, 0.6)) if rng.random() > 0.2 else None
        feature_dicts.append(
            {
                "gcps": gcps,
                "verification": verification,
                "source": "georef",
                "panel": False,
            }
        )
    raw = np.array(
        [
            [np.nan if v is None else v for v in feature_vector(f)[0]]
            for f in feature_dicts
        ]
    )
    y = np.array(
        [(f["gcps"] >= 3) and (f["verification"] or 0) > 0.8 for f in feature_dicts],
        float,
    )
    X, columns = design(raw)
    w = fit_logistic(X, y)
    names = feature_vector(feature_dicts[0])[1]
    model = model_document(columns, names, w, {"version": "test"})
    expected = predict(X, w)
    actual = np.array([p_good(f, model) for f in feature_dicts])
    assert actual == pytest.approx(expected, abs=1e-9)


def test_fold_of_is_stable_and_in_range():
    assert fold_of("sanborn04023_023", 5) == fold_of("sanborn04023_023", 5)
    assert {fold_of(f"sanborn{i:05d}_001", 5) for i in range(200)} == set(range(5))
