"""Unit tests for the pure-python helper functions in evaluate.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.evaluate import (
    _efficiency_dataframe,
    _log_importance_table,
    evaluation_efficiency,
    feature_importance,
    target_variance,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_labels():
    """Return a simple y_test array with balanced classes."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=200)


@pytest.fixture
def mock_classifier():
    """Mock model that returns deterministic probabilities."""
    model = MagicMock()
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, size=(200, 2))
    proba[:, 1] = proba[:, 0]  # column 1 is the "positive" probability
    proba[:, 0] = 1 - proba[:, 1]
    model.predict_proba.return_value = proba
    return model


# ---------------------------------------------------------------------------
# _efficiency_dataframe
# ---------------------------------------------------------------------------


def test_efficiency_dataframe_shape(binary_labels):
    thresholds = np.linspace(0, 1, 11)
    proba = np.random.default_rng(1).uniform(0, 1, len(binary_labels))
    df = _efficiency_dataframe("test", proba, binary_labels, thresholds)
    assert len(df) == len(thresholds)
    for col in (
        "threshold",
        "signal_efficiency",
        "background_efficiency",
        "n_signal",
        "n_background",
    ):
        assert col in df.columns


def test_efficiency_dataframe_threshold_zero_gives_full_efficiency(binary_labels):
    proba = np.ones(len(binary_labels))
    thresholds = np.array([0.0, 1.0])
    df = _efficiency_dataframe("test", proba, binary_labels, thresholds)
    assert df.iloc[0]["signal_efficiency"] == pytest.approx(1.0)
    assert df.iloc[0]["background_efficiency"] == pytest.approx(1.0)


def test_efficiency_dataframe_threshold_one_gives_zero_efficiency(binary_labels):
    proba = np.zeros(len(binary_labels))
    thresholds = np.array([0.0, 1.0])
    df = _efficiency_dataframe("test", proba, binary_labels, thresholds)
    # All predictions are 0, threshold=1.0 → none pass
    assert df.iloc[1]["signal_efficiency"] == pytest.approx(0.0)
    assert df.iloc[1]["background_efficiency"] == pytest.approx(0.0)


def test_efficiency_dataframe_no_signal_returns_zero():
    y_test = np.zeros(10, dtype=int)  # all background
    proba = np.ones(10)
    thresholds = np.array([0.5])
    df = _efficiency_dataframe("test", proba, y_test, thresholds)
    assert df.iloc[0]["signal_efficiency"] == pytest.approx(0.0)


def test_efficiency_dataframe_values_in_unit_interval(binary_labels):
    proba = np.random.default_rng(99).uniform(0, 1, len(binary_labels))
    thresholds = np.linspace(0, 1, 21)
    df = _efficiency_dataframe("test", proba, binary_labels, thresholds)
    assert (df["signal_efficiency"] >= 0).all()
    assert (df["signal_efficiency"] <= 1).all()
    assert (df["background_efficiency"] >= 0).all()
    assert (df["background_efficiency"] <= 1).all()


# ---------------------------------------------------------------------------
# target_variance
# ---------------------------------------------------------------------------


def test_target_variance_perfect_prediction_logs_zero_unexplained(caplog):
    y_test = pd.DataFrame({"target_a": [1.0, 2.0, 3.0, 4.0]})
    y_pred = y_test.to_numpy()
    import logging

    with caplog.at_level(logging.INFO):
        target_variance(y_test, y_pred, ["target_a"])
    # 0.00% unexplained variance when predictions are perfect
    assert "0.00%" in caplog.text


def test_target_variance_constant_target_warns(caplog):
    y_test = pd.DataFrame({"flat": [5.0, 5.0, 5.0]})
    y_pred = np.array([[5.0], [5.0], [5.0]])
    import logging

    with caplog.at_level(logging.WARNING):
        target_variance(y_test, y_pred, ["flat"])
    assert "zero variance" in caplog.text.lower()


def test_target_variance_accepts_numpy_y_test():
    y_test = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_pred = y_test + 0.1
    # Should not raise
    target_variance(y_test, y_pred, ["t1", "t2"])


# ---------------------------------------------------------------------------
# evaluation_efficiency (uses mock model)
# ---------------------------------------------------------------------------


def test_evaluation_efficiency_returns_dataframe(mock_classifier, binary_labels):
    x_test = pd.DataFrame({"f1": range(200), "f2": range(200)})
    y_test = pd.Series(binary_labels)
    result = evaluation_efficiency("test", mock_classifier, x_test, y_test)
    assert isinstance(result, pd.DataFrame)
    assert "signal_efficiency" in result.columns
    assert len(result) == 101  # 101 thresholds from linspace(0,1,101)


def test_evaluation_efficiency_by_zenith_returns_tuple(mock_classifier, binary_labels):
    rng = np.random.default_rng(7)
    x_test = pd.DataFrame(
        {
            "f1": range(200),
            "ze_bin": rng.integers(0, 3, size=200),
        }
    )
    y_test = pd.Series(binary_labels)
    result_all, result_by_ze = evaluation_efficiency(
        "test", mock_classifier, x_test, y_test, return_by_zenith=True
    )
    assert isinstance(result_all, pd.DataFrame)
    assert isinstance(result_by_ze, dict)


def test_evaluation_efficiency_missing_ze_bin_returns_all_df(mock_classifier, binary_labels):
    x_test = pd.DataFrame({"f1": range(200)})
    y_test = pd.Series(binary_labels)
    result_all, result_by_ze = evaluation_efficiency(
        "test", mock_classifier, x_test, y_test, return_by_zenith=True
    )
    assert isinstance(result_all, pd.DataFrame)
    assert result_by_ze == {}


# ---------------------------------------------------------------------------
# feature_importance (logging only; validate no crash)
# ---------------------------------------------------------------------------


def test_feature_importance_multioutput_no_crash():
    model = MagicMock()
    est1 = MagicMock()
    est2 = MagicMock()
    est1.feature_importances_ = np.array([0.5, 0.3, 0.2])
    est2.feature_importances_ = np.array([0.1, 0.6, 0.3])
    model.estimators_ = [est1, est2]
    # Should not raise
    feature_importance(model, ["f1", "f2", "f3"], ["target1", "target2"])


def test_feature_importance_single_output_no_crash():
    model = MagicMock(spec=["feature_importances_"])
    model.feature_importances_ = np.array([0.4, 0.3, 0.3])
    # Should not raise
    feature_importance(model, ["f1", "f2", "f3"], ["label"])


def test_log_importance_table_no_crash():
    _log_importance_table(
        "my_target", np.array([0.5, 0.3, 0.2]), ["feat_a", "feat_b", "feat_c"], "xgboost"
    )
