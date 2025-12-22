"""Unit tests for model evaluation, feature importance, and SHAP utilities."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from eventdisplay_ml.evaluate import (
    calculate_resolution,
    evaluate_model,
    feature_importance,
    shap_feature_importance,
)

# Use modern NumPy RNG to satisfy lint rules (NPY002)
rng = np.random.default_rng(0)


def test_shap_feature_importance_basic(caplog):
    """Test shap_feature_importance logs SHAP importance values."""
    caplog.set_level(logging.INFO)

    # Mock model and estimators
    mock_booster = MagicMock()
    mock_est = MagicMock()
    mock_est.get_booster.return_value = mock_booster

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]

    # Create sample data
    x_sample_data = pd.DataFrame(
        {
            "feature_1": rng.random(100),
            "feature_2": rng.random(100),
            "feature_3": rng.random(100),
        }
    )

    # Mock SHAP values (100 samples, 4 features + 1 bias)
    shap_values = rng.random((100, 4))
    mock_booster.predict.return_value = np.hstack([shap_values, rng.random((100, 1))])

    target_names = ["target_1"]

    shap_feature_importance(mock_model, x_sample_data, target_names, max_points=100, n_top=2)

    assert "Builtin XGBoost SHAP Importance for target_1" in caplog.text
    assert "feature_" in caplog.text


def test_shap_feature_importance_multiple_targets(caplog):
    """Test shap_feature_importance with multiple targets."""
    caplog.set_level(logging.INFO)

    # Mock model with multiple estimators
    mock_booster_1 = MagicMock()
    mock_est_1 = MagicMock()
    mock_est_1.get_booster.return_value = mock_booster_1

    mock_booster_2 = MagicMock()
    mock_est_2 = MagicMock()
    mock_est_2.get_booster.return_value = mock_booster_2

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est_1, mock_est_2]

    x_sample_data = pd.DataFrame(
        {
            "feat_a": rng.random(50),
            "feat_b": rng.random(50),
        }
    )

    shap_vals = rng.random((50, 3))
    mock_booster_1.predict.return_value = np.hstack([shap_vals, rng.random((50, 1))])
    mock_booster_2.predict.return_value = np.hstack([shap_vals, rng.random((50, 1))])

    target_names = ["target_x", "target_y"]

    shap_feature_importance(mock_model, x_sample_data, target_names, max_points=100, n_top=2)

    assert "Builtin XGBoost SHAP Importance for target_x" in caplog.text
    assert "Builtin XGBoost SHAP Importance for target_y" in caplog.text


def test_shap_feature_importance_sampling(caplog):
    """Test shap_feature_importance samples data correctly."""
    caplog.set_level(logging.INFO)

    mock_booster = MagicMock()
    mock_est = MagicMock()
    mock_est.get_booster.return_value = mock_booster

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]

    # Create large dataset
    x_data = pd.DataFrame(
        {
            "f1": rng.random(10000),
            "f2": rng.random(10000),
        }
    )

    shap_vals = rng.random((5000, 3))
    mock_booster.predict.return_value = np.hstack([shap_vals, rng.random((5000, 1))])

    shap_feature_importance(mock_model, x_data, ["target"], max_points=5000, n_top=1)

    # Verify predict was called with sampled data
    assert mock_booster.predict.called
    call_args = mock_booster.predict.call_args[0][0]
    assert call_args.num_row() <= 5000


def test_feature_importance_basic(caplog):
    """Test feature_importance logs feature importances correctly."""
    caplog.set_level(logging.INFO)

    # Mock model with estimators
    mock_est = MagicMock()
    mock_est.feature_importances_ = np.array([0.5, 0.3, 0.2])

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]

    x_cols = ["feature_1", "feature_2", "feature_3"]
    target_names = ["target_1"]

    feature_importance(mock_model, x_cols, target_names, name="test_model")

    assert "XGBoost Multi-Regression Feature Importance" in caplog.text
    assert "test_model Importance for Target: **target_1**" in caplog.text
    assert "feature_1" in caplog.text
    assert "feature_2" in caplog.text
    assert "feature_3" in caplog.text


def test_feature_importance_multiple_targets(caplog):
    """Test feature_importance with multiple targets."""
    caplog.set_level(logging.INFO)

    # Mock model with multiple estimators
    mock_est_1 = MagicMock()
    mock_est_1.feature_importances_ = np.array([0.6, 0.25, 0.15])

    mock_est_2 = MagicMock()
    mock_est_2.feature_importances_ = np.array([0.4, 0.35, 0.25])

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est_1, mock_est_2]

    x_cols = ["feat_a", "feat_b", "feat_c"]
    target_names = ["target_x", "target_y"]

    feature_importance(mock_model, x_cols, target_names, name="xgboost")

    assert "Importance for Target: **target_x**" in caplog.text
    assert "Importance for Target: **target_y**" in caplog.text
    assert "feat_a" in caplog.text


def test_feature_importance_sorting(caplog):
    """Test feature_importance sorts features by importance."""
    caplog.set_level(logging.INFO)

    mock_est = MagicMock()
    # Set importances in non-sorted order
    mock_est.feature_importances_ = np.array([0.1, 0.5, 0.3, 0.1])

    mock_model = MagicMock()
    mock_model.estimators_ = [mock_est]

    x_cols = ["low_1", "high", "medium", "low_2"]
    target_names = ["target"]

    feature_importance(mock_model, x_cols, target_names)

    # Check that high importance feature appears before lower ones in logs
    log_text = caplog.text
    high_pos = log_text.find("high")
    medium_pos = log_text.find("medium")
    assert high_pos < medium_pos


def test_calculate_resolution_basic(caplog):
    """Test calculate_resolution computes and logs resolution metrics."""
    caplog.set_level(logging.INFO)

    # Create test data
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

    df = pd.DataFrame(
        {
            "MCe0": [0.5, 0.8, 1.0, 1.5],
        },
        index=[0, 1, 2, 3],
    )

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=[50, 68],
        log_e_min=0,
        log_e_max=2,
        n_bins=2,
        name="test_model",
    )

    assert "--- test_model DeltaTheta Resolution vs. Log10(MCe0) ---" in caplog.text
    assert "--- test_model DeltaMCe0 Resolution vs. Log10(MCe0) ---" in caplog.text
    assert "Calculated over 2 bins between Log10(E) = 0 and 2" in caplog.text


def test_calculate_resolution_delta_theta_computation(caplog):
    """Test calculate_resolution correctly computes DeltaTheta."""
    caplog.set_level(logging.INFO)

    # Known differences: sqrt(0.1^2 + 0.2^2) = sqrt(0.05) ≈ 0.2236
    y_pred = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0],
            "MCyoff": [0.0, 0.0],
        },
        index=[0, 1],
    )

    df = pd.DataFrame(
        {
            "MCe0": [1.0, 1.0],
        },
        index=[0, 1],
    )

    calculate_resolution(
        y_pred, y_test, df, percentiles=[50], log_e_min=0.5, log_e_max=1.5, n_bins=1, name="test"
    )

    assert "Theta_50%" in caplog.text


def test_calculate_resolution_delta_e_computation(caplog):
    """Test calculate_resolution correctly computes relative energy error."""
    caplog.set_level(logging.INFO)

    y_pred = np.array(
        [
            [0.0, 0.0, 1.0],  # 10^1.0 = 10
            [0.0, 0.0, 0.5],  # 10^0.5 ≈ 3.16
        ]
    )

    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0],
            "MCyoff": [0.0, 0.0],
        },
        index=[0, 1],
    )

    df = pd.DataFrame(
        {
            "MCe0": [1.0, 0.5],  # true energies
        },
        index=[0, 1],
    )

    calculate_resolution(
        y_pred, y_test, df, percentiles=[50], log_e_min=0.0, log_e_max=1.5, n_bins=1, name="test"
    )

    assert "DeltaE" in caplog.text


def test_calculate_resolution_binning(caplog):
    """Test calculate_resolution bins data by energy correctly."""
    caplog.set_level(logging.INFO)

    y_pred = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0, 0.0, 0.0],
            "MCyoff": [0.0, 0.0, 0.0, 0.0],
        },
        index=[0, 1, 2, 3],
    )

    df = pd.DataFrame(
        {
            "MCe0": [0.0, 0.5, 1.0, 1.5],  # Different energy bins
        },
        index=[0, 1, 2, 3],
    )

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=[50, 68, 90],
        log_e_min=0.0,
        log_e_max=1.5,
        n_bins=3,
        name="test",
    )

    assert "Mean Log10(E)" in caplog.text
    assert "Theta_50%" in caplog.text
    assert "Theta_68%" in caplog.text
    assert "Theta_90%" in caplog.text


def test_calculate_resolution_multiple_targets(caplog):
    """Test calculate_resolution with multiple target outputs."""
    caplog.set_level(logging.INFO)

    y_pred = np.array(
        [
            [0.05, 0.1, 1.0],
            [0.1, 0.15, 1.1],
            [0.02, 0.05, 0.9],
            [0.08, 0.12, 1.2],
        ]
    )

    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0, 0.0, 0.0],
            "MCyoff": [0.0, 0.0, 0.0, 0.0],
        },
        index=[0, 1, 2, 3],
    )

    df = pd.DataFrame(
        {
            "MCe0": [0.5, 0.7, 1.0, 1.3],
        },
        index=[0, 1, 2, 3],
    )

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=[68, 90, 95],
        log_e_min=0,
        log_e_max=1.5,
        n_bins=2,
        name="xgboost",
    )

    assert "DeltaTheta" in caplog.text
    assert "DeltaMCe0" in caplog.text
    assert "Theta_68%" in caplog.text
    assert "DeltaE_95%" in caplog.text


def test_evaluate_model_basic(caplog):
    """Test evaluate_model logs R^2 score and error metrics."""
    caplog.set_level(logging.INFO)

    # Mock model
    mock_model = MagicMock()
    mock_model.score.return_value = 0.85
    mock_model.predict.return_value = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.15, 0.25, 1.1],
        ]
    )
    mock_model.estimators_ = [MagicMock(), MagicMock()]

    # Setup estimators for feature_importance
    for est in mock_model.estimators_:
        est.feature_importances_ = np.array([0.5, 0.3, 0.2])
        est.get_booster.return_value.predict.return_value = rng.random((2, 4))

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

    df = pd.DataFrame(
        {
            "MCe0": [1.0, 1.1],
        },
        index=[0, 1],
    )

    x_cols = ["feat_1", "feat_2", "feat_3"]
    y_data = pd.DataFrame({"target_1": [1, 2], "target_2": [3, 4]})

    evaluate_model(mock_model, x_test, y_test, df, x_cols, y_data, "test_model")

    assert "XGBoost Multi-Target R^2 Score (Testing Set): 0.8500" in caplog.text
    assert "test_model MSE (X_off):" in caplog.text
    assert "test_model MAE (X_off):" in caplog.text
    assert "test_model MAE (Y_off):" in caplog.text


def test_evaluate_model_mse_calculation(caplog):
    """Test evaluate_model correctly computes MSE for offset predictions."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.9
    mock_model.predict.return_value = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mock_model.estimators_ = [MagicMock()]
    mock_model.estimators_[0].feature_importances_ = np.array([0.5, 0.3, 0.2])
    mock_model.estimators_[0].get_booster.return_value.predict.return_value = rng.random((2, 4))

    x_test = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "f3": [5.0, 6.0],
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

    df = pd.DataFrame({"MCe0": [1.0, 1.0]}, index=[0, 1])

    # Provide target names for feature_importance
    y_data = pd.DataFrame({"target": [1, 2]})
    evaluate_model(mock_model, x_test, y_test, df, ["f1", "f2", "f3"], y_data, "test")

    # MSE should be 0 since predictions match true values
    assert "MSE (X_off): 0.0000" in caplog.text
    assert "MSE (Y_off): 0.0000" in caplog.text


def test_evaluate_model_calls_feature_importance(caplog):
    """Test evaluate_model calls feature_importance function."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.8
    mock_model.predict.return_value = np.array([[0.1, 0.2, 1.0]])
    mock_model.estimators_ = [MagicMock()]
    mock_model.estimators_[0].feature_importances_ = np.array([0.6, 0.3, 0.1])

    x_test = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}, index=[0])
    y_test = pd.DataFrame({"MCxoff": [0.0], "MCyoff": [0.0]}, index=[0])
    df = pd.DataFrame({"MCe0": [1.0]}, index=[0])

    # Provide target names for feature_importance
    y_data = pd.DataFrame({"target": [1]})
    evaluate_model(mock_model, x_test, y_test, df, ["a", "b", "c"], y_data, "test")

    assert "XGBoost Multi-Regression Feature Importance" in caplog.text


def test_evaluate_model_calls_shap_for_xgboost(caplog):
    """Test evaluate_model calls shap_feature_importance only for xgboost model."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.8
    mock_model.predict.return_value = np.array([[0.1, 0.2, 1.0]])
    mock_model.estimators_ = [MagicMock()]
    mock_model.estimators_[0].feature_importances_ = np.array([0.5, 0.3, 0.2])
    mock_booster = MagicMock()
    mock_booster.predict.return_value = rng.random((1, 4))
    mock_model.estimators_[0].get_booster.return_value = mock_booster

    x_test = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}, index=[0])
    y_test = pd.DataFrame({"MCxoff": [0.0], "MCyoff": [0.0]}, index=[0])
    df = pd.DataFrame({"MCe0": [1.0]}, index=[0])

    # Provide target names for SHAP/feature_importance
    y_data = pd.DataFrame({"target": [1]})
    evaluate_model(mock_model, x_test, y_test, df, ["x", "y", "z"], y_data, "xgboost")

    assert "Builtin XGBoost SHAP Importance" in caplog.text


def test_evaluate_model_no_shap_for_non_xgboost(caplog):
    """Test evaluate_model skips shap_feature_importance for non-xgboost models."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.75
    mock_model.predict.return_value = np.array([[0.1, 0.2, 1.0]])
    mock_model.estimators_ = [MagicMock()]
    mock_model.estimators_[0].feature_importances_ = np.array([0.4, 0.4, 0.2])

    x_test = pd.DataFrame({"p": [1.0], "q": [2.0], "r": [3.0]}, index=[0])
    y_test = pd.DataFrame({"MCxoff": [0.0], "MCyoff": [0.0]}, index=[0])
    df = pd.DataFrame({"MCe0": [1.0]}, index=[0])

    # Provide target names for feature_importance
    y_data = pd.DataFrame({"target": [1]})
    evaluate_model(mock_model, x_test, y_test, df, ["p", "q", "r"], y_data, "random_forest")

    assert "Builtin XGBoost SHAP Importance" not in caplog.text
    assert "XGBoost Multi-Regression Feature Importance" in caplog.text


def test_evaluate_model_calls_calculate_resolution(caplog):
    """Test evaluate_model calls calculate_resolution with correct parameters."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.82
    mock_model.predict.return_value = np.array(
        [
            [0.05, 0.1, 1.0],
            [0.08, 0.12, 1.1],
        ]
    )
    mock_model.estimators_ = [MagicMock()]
    mock_model.estimators_[0].feature_importances_ = np.array([0.5, 0.3, 0.2])

    x_test = pd.DataFrame({"m": [1.0, 2.0], "n": [3.0, 4.0], "o": [5.0, 6.0]}, index=[0, 1])
    y_test = pd.DataFrame(
        {
            "MCxoff": [0.0, 0.0],
            "MCyoff": [0.0, 0.0],
        },
        index=[0, 1],
    )
    df = pd.DataFrame({"MCe0": [0.5, 1.0]}, index=[0, 1])

    # Provide target names for feature_importance
    y_data = pd.DataFrame({"target": [1, 2]})
    evaluate_model(mock_model, x_test, y_test, df, ["m", "n", "o"], y_data, "test_model")

    assert "DeltaTheta Resolution vs. Log10(MCe0)" in caplog.text
    assert "DeltaMCe0 Resolution vs. Log10(MCe0)" in caplog.text
    assert "Calculated over 6 bins between Log10(E) = -1 and 2" in caplog.text


def test_evaluate_model_with_multiple_targets(caplog):
    """Test evaluate_model with multiple estimators for multi-target regression."""
    caplog.set_level(logging.INFO)

    mock_model = MagicMock()
    mock_model.score.return_value = 0.88
    mock_model.predict.return_value = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.12, 0.22, 1.05],
        ]
    )

    mock_est1 = MagicMock()
    mock_est1.feature_importances_ = np.array([0.6, 0.3, 0.1])
    mock_est2 = MagicMock()
    mock_est2.feature_importances_ = np.array([0.4, 0.4, 0.2])
    mock_model.estimators_ = [mock_est1, mock_est2]

    x_test = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "feat_c": [5.0, 6.0],
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

    df = pd.DataFrame({"MCe0": [1.0, 1.2]}, index=[0, 1])
    y_data = pd.DataFrame({"target_x": [1, 2], "target_y": [3, 4]})

    evaluate_model(
        mock_model, x_test, y_test, df, ["feat_a", "feat_b", "feat_c"], y_data, "multi_target"
    )

    assert "XGBoost Multi-Target R^2 Score (Testing Set): 0.8800" in caplog.text
    assert "MSE (X_off):" in caplog.text
    assert "MSE (Y_off):" in caplog.text
