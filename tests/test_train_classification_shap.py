"""Tests for SHAP caching in classification training/evaluation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import evaluate, models


def test_evaluate_classification_model_returns_shap_importance_for_xgboost():
    """Classification evaluation should return SHAP importance for xgboost models."""
    x_test = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    y_test = pd.Series([1, 0], dtype=np.int32)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)

    expected_shap = {"label": np.array([0.2, 0.8], dtype=np.float64)}

    with patch("eventdisplay_ml.evaluate.feature_importance"):
        with patch("eventdisplay_ml.evaluate.shap_feature_importance") as mock_shap:
            mock_shap.return_value = expected_shap
            shap_importance = evaluate.evaluate_classification_model(
                mock_model,
                x_test,
                y_test,
                x_test,
                x_test.columns.tolist(),
                "xgboost",
            )

    assert "label" in shap_importance
    np.testing.assert_allclose(shap_importance["label"], expected_shap["label"])


def test_train_classification_caches_shap_importance_and_features():
    """Training should cache per-model SHAP summary for diagnostic reuse."""
    signal_df = pd.DataFrame(
        {
            "f1": np.linspace(0.1, 1.0, 20),
            "f2": np.linspace(1.0, 2.0, 20),
        }
    )
    background_df = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, -0.1, 20),
            "f2": np.linspace(0.0, 1.0, 20),
        }
    )

    cfg = {
        "train_test_fraction": 0.5,
        "random_state": 42,
        "models": {
            "xgboost": {
                "hyper_parameters": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "random_state": 42,
                    "eval_metric": "logloss",
                }
            }
        },
    }

    mock_model = MagicMock()
    mock_model.fit.return_value = None

    expected_shap = {"label": np.array([0.1, 0.9], dtype=np.float64)}
    expected_efficiency = pd.DataFrame(
        {
            "threshold": [0.5],
            "signal_efficiency": [0.8],
            "background_efficiency": [0.2],
        }
    )

    with patch("xgboost.XGBClassifier", return_value=mock_model):
        with patch("eventdisplay_ml.models.evaluate_classification_model") as mock_eval:
            with patch("eventdisplay_ml.models.evaluation_efficiency") as mock_eff:
                mock_eval.return_value = expected_shap
                mock_eff.return_value = (expected_efficiency, {})

                result = models.train_classification([signal_df, background_df], cfg)

    assert result["models"]["xgboost"]["features"] == ["f1", "f2"]
    assert "shap_importance" in result["models"]["xgboost"]
    assert "label" in result["models"]["xgboost"]["shap_importance"]
    np.testing.assert_allclose(
        result["models"]["xgboost"]["shap_importance"]["label"],
        expected_shap["label"],
    )


def test_class_zenith_balance_weights_equalize_class_zenith_distributions():
    """Class/zenith weights should remove ze_bin prior differences between classes."""
    x_train = pd.DataFrame(
        {
            "f1": np.arange(12, dtype=np.float32),
            "ze_bin": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    y_train = pd.Series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)

    weights = models._class_zenith_balance_weights(x_train, y_train)

    assert weights.mean() == pytest.approx(1.0)
    weighted = pd.DataFrame(
        {
            "label": y_train,
            "ze_bin": x_train["ze_bin"],
            "weight": weights,
        }
    )
    for label in [0, 1]:
        class_data = weighted[weighted["label"] == label]
        ze0 = class_data.loc[class_data["ze_bin"] == 0, "weight"].sum()
        ze1 = class_data.loc[class_data["ze_bin"] == 1, "weight"].sum()
        assert ze0 / (ze0 + ze1) == pytest.approx(0.5)
        assert ze1 / (ze0 + ze1) == pytest.approx(0.5)


def test_train_classification_applies_class_zenith_weights():
    """The optional class/zenith balance weights should be passed to XGBoost."""
    signal_df = pd.DataFrame(
        {
            "f1": np.linspace(0.1, 1.0, 20),
            "ze_bin": np.repeat([0, 1], [16, 4]),
        }
    )
    background_df = pd.DataFrame(
        {
            "f1": np.linspace(-1.0, -0.1, 20),
            "ze_bin": np.repeat([0, 1], [4, 16]),
        }
    )

    cfg = {
        "train_test_fraction": 0.5,
        "random_state": 42,
        "balance_class_zenith_weights": True,
        "models": {
            "xgboost": {
                "hyper_parameters": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "random_state": 42,
                    "eval_metric": "logloss",
                }
            }
        },
    }

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    expected_efficiency = pd.DataFrame(
        {
            "threshold": [0.5],
            "signal_efficiency": [0.8],
            "background_efficiency": [0.2],
        }
    )

    with patch("xgboost.XGBClassifier", return_value=mock_model):
        with patch("eventdisplay_ml.models.evaluate_classification_model") as mock_eval:
            with patch("eventdisplay_ml.models.evaluation_efficiency") as mock_eff:
                mock_eval.return_value = {"label": np.array([0.1, 0.9])}
                mock_eff.return_value = (expected_efficiency, {})

                models.train_classification([signal_df, background_df], cfg)

    sample_weight = mock_model.fit.call_args.kwargs.get("sample_weight")
    assert sample_weight is not None
    assert len(sample_weight) == 20
    assert sample_weight.mean() == pytest.approx(1.0)
