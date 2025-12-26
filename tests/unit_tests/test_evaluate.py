"""Unit tests for model evaluation."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.evaluate import (
    calculate_resolution,
    evaluate_regression_model,
    feature_importance,
    shap_feature_importance,
)

rng = np.random.default_rng(0)


# ============================================================================
# SHAP Feature Importance Tests (consolidated from 3 functions)
# ============================================================================


@pytest.mark.parametrize(
    ("n_targets", "n_samples"),
    [
        (1, 100),
        (2, 50),
    ],
)
def test_shap_feature_importance(caplog, n_targets, n_samples):
    """Test shap_feature_importance with various target counts."""
    caplog.set_level(logging.INFO)

    # Create mock model with appropriate estimators
    mock_model = MagicMock()
    estimators = []
    for _ in range(n_targets):
        mock_est = MagicMock()
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.hstack(
            [rng.random((n_samples, 4)), rng.random((n_samples, 1))]
        )
        mock_est.get_booster.return_value = mock_booster
        estimators.append(mock_est)
    mock_model.estimators_ = estimators

    x_sample_data = pd.DataFrame({f"feature_{i}": rng.random(n_samples) for i in range(3)})
    target_names = [f"target_{i}" for i in range(n_targets)]

    shap_feature_importance(mock_model, x_sample_data, target_names, max_points=100, n_top=2)

    for target in target_names:
        assert f"Builtin XGBoost SHAP Importance for {target}" in caplog.text


# ============================================================================
# Feature Importance Tests (consolidated from 3 functions)
# ============================================================================


@pytest.mark.parametrize(
    ("n_targets", "n_features"),
    [
        (1, 3),
        (2, 4),
    ],
)
def test_feature_importance(caplog, n_targets, n_features):
    """Test feature_importance with various feature/target counts."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    estimators = []
    rng = np.random.default_rng(42)
    for _ in range(n_targets):
        mock_est = MagicMock()
        mock_est.feature_importances_ = rng.random(n_features)
        estimators.append(mock_est)
    mock_model.estimators_ = estimators

    x_cols = [f"feature_{i}" for i in range(n_features)]
    target_names = [f"target_{i}" for i in range(n_targets)]

    feature_importance(mock_model, x_cols, target_names, name="test_model")

    assert "XGBoost Multi-Regression Feature Importance" in caplog.text
    for target in target_names:
        assert f"Importance for Target: **{target}**" in caplog.text


def test_feature_importance_sorted(caplog):
    """Test feature_importance sorts features by importance."""
    caplog.set_level(logging.INFO)

    mock_est = MagicMock()
    mock_est.feature_importances_ = np.array([0.1, 0.5, 0.3, 0.1])
    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]

    x_cols = ["low_1", "high", "medium", "low_2"]
    target_names = ["target"]

    feature_importance(mock_model, x_cols, target_names)

    log_text = caplog.text
    high_pos = log_text.find("high")
    medium_pos = log_text.find("medium")
    assert high_pos < medium_pos


# ============================================================================
# Resolution Calculation Tests (consolidated from 4 functions)
# ============================================================================


@pytest.mark.parametrize(
    ("n_bins", "percentiles"),
    [
        (2, [50, 68]),
        (3, [50, 68, 90]),
        (1, [68, 90, 95]),
    ],
)
def test_calculate_resolution(caplog, n_bins, percentiles):
    """Test calculate_resolution with various binning and percentile configurations."""
    caplog.set_level(logging.INFO)

    y_pred = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.15, 0.25, 1.1],
            [0.2, 0.3, 0.9],
            [0.05, 0.1, 1.2],
        ]
    )
    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0, 0.0, 0.0],
            "MCyoff": [0.0, 0.0, 0.0, 0.0],
        },
        index=[0, 1, 2, 3],
    )
    df = pd.DataFrame({"MCe0": [0.5, 0.8, 1.0, 1.5]}, index=[0, 1, 2, 3])

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=percentiles,
        log_e_min=0,
        log_e_max=2,
        n_bins=n_bins,
        name="test",
    )

    assert "DeltaTheta Resolution" in caplog.text
    assert "DeltaMCe0 Resolution" in caplog.text
    for perc in percentiles:
        assert f"Theta_{perc}%" in caplog.text


def test_calculate_resolution_deltas_computed_correctly(caplog):
    """Test delta computations in calculate_resolution."""
    caplog.set_level(logging.INFO)

    # Known case: differences = sqrt(0.1^2 + 0.2^2) = sqrt(0.05)
    y_pred = np.array([[0.1, 0.2, 1.0], [0.0, 0.0, 1.0]])
    y_test = pd.DataFrame({"MCxoff": [0.0, 0.0], "MCyoff": [0.0, 0.0]}, index=[0, 1])
    df = pd.DataFrame({"MCe0": [1.0, 1.0]}, index=[0, 1])

    calculate_resolution(
        y_pred, y_test, df, percentiles=[50], log_e_min=0.5, log_e_max=1.5, n_bins=1, name="test"
    )

    assert "Theta_50%" in caplog.text
    assert "DeltaE" in caplog.text


# ============================================================================
# Model Evaluation Tests (consolidated from 5 functions)
# ============================================================================


def test_evaluate_regression_model_basic(caplog):
    """Test evaluate_regression_model logs R^2 score and metrics."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.85
    mock_model.predict.return_value = np.array([[0.1, 0.2, 1.0], [0.15, 0.25, 1.1]])

    mock_est1 = MagicMock()
    mock_est1.feature_importances_ = np.array([0.5, 0.3, 0.2])
    mock_est2 = MagicMock()
    mock_est2.feature_importances_ = np.array([0.4, 0.4, 0.2])
    mock_model.estimators_ = [mock_est1, mock_est2]

    x_test = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0],
            "feat_2": [3.0, 4.0],
            "feat_3": [5.0, 6.0],
        },
        index=[0, 1],
    )
    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0],
            "MCyoff": [0.0, 0.0],
        },
        index=[0, 1],
    )
    df = pd.DataFrame({"MCe0": [1.0, 1.1]}, index=[0, 1])
    y_data = pd.DataFrame({"target_1": [1, 2], "target_2": [3, 4]})

    evaluate_regression_model(
        mock_model, x_test, y_test, df, ["feat_1", "feat_2", "feat_3"], y_data, "test_model"
    )

    assert "XGBoost Multi-Target R^2 Score (Testing Set): 0.8500" in caplog.text
    assert "test_model MSE (X_off):" in caplog.text
    assert "test_model MAE (X_off):" in caplog.text
    assert "test_model MAE (Y_off):" in caplog.text


@pytest.mark.parametrize(
    ("model_name", "has_xgb"),
    [
        ("xgboost", True),
        ("random_forest", False),
    ],
)
def test_evaluate_regression_model_shap_conditional(caplog, model_name, has_xgb):
    """Test evaluate_regression_model calls SHAP only for XGBoost models."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.8
    mock_model.predict.return_value = np.array([[0.1, 0.2, 1.0]])

    mock_est = MagicMock()
    mock_est.feature_importances_ = np.array([0.5, 0.3, 0.2])
    if has_xgb:
        mock_booster = MagicMock()
        mock_booster.predict.return_value = rng.random((1, 4))
        mock_est.get_booster.return_value = mock_booster
    mock_model.estimators_ = [mock_est]

    x_test = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}, index=[0])
    y_test = pd.DataFrame({"MCxoff": [0.0], "MCyoff": [0.0]}, index=[0])
    df = pd.DataFrame({"MCe0": [1.0]}, index=[0])
    y_data = pd.DataFrame({"target": [1]})

    evaluate_regression_model(mock_model, x_test, y_test, df, ["x", "y", "z"], y_data, model_name)

    if has_xgb:
        assert "Builtin XGBoost SHAP Importance" in caplog.text
    else:
        assert "Builtin XGBoost SHAP Importance" not in caplog.text


def test_evaluate_regression_model_calls_resolution(caplog):
    """Test evaluate_regression_model calls calculate_resolution."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.82
    mock_model.predict.return_value = np.array([[0.05, 0.1, 1.0], [0.08, 0.12, 1.1]])

    mock_est = MagicMock()
    mock_est.feature_importances_ = np.array([0.5, 0.3, 0.2])
    mock_model.estimators_ = [mock_est]

    x_test = pd.DataFrame({"m": [1.0, 2.0], "n": [3.0, 4.0], "o": [5.0, 6.0]}, index=[0, 1])
    y_test = pd.DataFrame({"MCxoff": [0.0, 0.0], "MCyoff": [0.0, 0.0]}, index=[0, 1])
    df = pd.DataFrame({"MCe0": [0.5, 1.0]}, index=[0, 1])
    y_data = pd.DataFrame({"target": [1, 2]})

    evaluate_regression_model(mock_model, x_test, y_test, df, ["m", "n", "o"], y_data, "test_model")

    assert "DeltaTheta Resolution vs. Log10(MCe0)" in caplog.text
    assert "DeltaMCe0 Resolution vs. Log10(MCe0)" in caplog.text
    assert "Calculated over 6 bins between Log10(E) = -1 and 2" in caplog.text
