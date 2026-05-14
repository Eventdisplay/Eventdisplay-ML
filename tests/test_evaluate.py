"""Tests for model evaluation helpers."""

import numpy as np
import pandas as pd

from eventdisplay_ml.evaluate import evaluation_efficiency


class _DummyClassifier:
    """Minimal classifier implementing predict_proba for testing."""

    def __init__(self, positive_class_probabilities):
        self._positive_class_probabilities = np.asarray(positive_class_probabilities, dtype=float)

    def predict_proba(self, x_test):
        if len(x_test) != len(self._positive_class_probabilities):
            raise ValueError("x_test length does not match provided probabilities")
        p1 = self._positive_class_probabilities
        return np.column_stack([1.0 - p1, p1])


def test_evaluation_efficiency_returns_consistent_dataframe_lengths():
    """Efficiency table should have one row per threshold for every output column."""
    x_test = pd.DataFrame({"f1": [0, 1, 2, 3, 4]})
    y_test = pd.Series([1, 0, 1, 0, 1], dtype=int)
    model = _DummyClassifier([0.1, 0.2, 0.8, 0.6, 0.4])

    efficiency = evaluation_efficiency("xgboost", model, x_test, y_test)

    assert len(efficiency) == 101
    assert all(len(efficiency[col]) == 101 for col in efficiency.columns)
    assert np.isclose(efficiency.loc[0, "n_signal"], 3.0)
    assert np.isclose(efficiency.loc[0, "n_background"], 2.0)
