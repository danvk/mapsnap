import numpy as np

from mapsnap.keymap.number_model import average_precision


def test_average_precision_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_average_precision_worst_ranking():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    # Positives ranked last: AP = mean(1/3, 2/4) = 0.41666...
    assert abs(average_precision(scores, labels) - (0.5 * (1 / 3 + 2 / 4))) < 1e-9


def test_average_precision_no_positives_is_zero():
    assert average_precision(np.array([0.5, 0.4]), np.array([0, 0])) == 0.0
