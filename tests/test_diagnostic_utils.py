"""Unit tests for pure computation functions in diagnostic_utils.py."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.diagnostic_utils import compute_generalization_metrics

# ---------------------------------------------------------------------------
# compute_generalization_metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_predictions():
    """Train and test DataFrames where predictions match exactly."""
    targets = ["Xoff_residual", "Yoff_residual"]
    data = {t: np.array([0.0, 1.0, 2.0, 3.0]) for t in targets}
    y = pd.DataFrame(data)
    return y, y.copy(), y, y.copy(), targets


def test_compute_generalization_metrics_perfect_gives_zero_rmse(perfect_predictions):
    y_train, y_train_pred, y_test, y_test_pred, targets = perfect_predictions
    metrics = compute_generalization_metrics(y_train, y_train_pred, y_test, y_test_pred, targets)
    for t in targets:
        assert metrics[t]["rmse_train"] == pytest.approx(0.0)
        assert metrics[t]["rmse_test"] == pytest.approx(0.0)


def test_compute_generalization_metrics_keys_present():
    targets = ["E_residual"]
    y = pd.DataFrame({"E_residual": [1.0, 2.0, 3.0]})
    pred = pd.DataFrame({"E_residual": [1.1, 2.1, 2.9]})
    metrics = compute_generalization_metrics(y, pred, y, pred, targets)
    for key in ("rmse_train", "rmse_test", "gap_pct", "gen_ratio"):
        assert key in metrics["E_residual"]


def test_compute_generalization_metrics_gen_ratio_one_for_identical_splits():
    targets = ["t"]
    y = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    pred = pd.DataFrame({"t": [1.5, 2.5, 3.5, 4.5]})
    metrics = compute_generalization_metrics(y, pred, y, pred, targets)
    assert metrics["t"]["gen_ratio"] == pytest.approx(1.0)


def test_compute_generalization_metrics_gap_pct_positive_when_test_worse():
    targets = ["t"]
    y_train = pd.DataFrame({"t": [1.0, 2.0, 3.0]})
    pred_train = pd.DataFrame({"t": [1.1, 2.1, 3.1]})  # small error
    y_test = pd.DataFrame({"t": [1.0, 2.0, 3.0]})
    pred_test = pd.DataFrame({"t": [2.0, 3.0, 4.0]})  # large error
    metrics = compute_generalization_metrics(y_train, pred_train, y_test, pred_test, targets)
    assert metrics["t"]["gap_pct"] > 0


def test_compute_generalization_metrics_zero_train_rmse_handled():
    """When train RMSE is zero, gap_pct and gen_ratio should handle division gracefully."""
    targets = ["t"]
    y = pd.DataFrame({"t": [1.0, 2.0]})
    pred_perfect = pd.DataFrame({"t": [1.0, 2.0]})
    pred_bad = pd.DataFrame({"t": [2.0, 3.0]})
    metrics = compute_generalization_metrics(y, pred_perfect, y, pred_bad, targets)
    assert metrics["t"]["rmse_train"] == pytest.approx(0.0)
    assert np.isinf(metrics["t"]["gap_pct"]) or metrics["t"]["gap_pct"] > 0
