"""Tests for remaining diagnostic_utils functions."""

from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import diagnostic_utils


@pytest.fixture
def diagnostic_model_file(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "models": {
                "xgboost": {
                    "model": "cached-model",
                    "features": ["f1", "f2"],
                    "shap_importance": {"Xoff_residual": np.array([0.3, 0.7])},
                    "generalization_metrics": {"Xoff_residual": {"rmse_train": 0.1}},
                    "residual_normality_stats": {"Xoff_residual": {"mean": 0.0, "std": 1.0}},
                    "shap_explainer": "explainer",
                }
            },
            "features": ["f1", "f2"],
            "target_mean": {"Xoff_residual": 1.0, "Yoff_residual": -1.0},
            "target_std": {"Xoff_residual": 2.0, "Yoff_residual": 0.5},
            "targets": ["Xoff_residual", "Yoff_residual"],
            "input_file_list": "inputs.txt",
            "train_test_fraction": 0.5,
            "random_state": 3,
        },
        path,
    )
    return path


@pytest.fixture
def residual_frames():
    y_test = pd.DataFrame(
        {"Xoff_residual": [0.0, 0.5, 1.0, 1.5], "Yoff_residual": [1.0, 1.5, 2.0, 2.5]}
    )
    y_pred = pd.DataFrame(
        {"Xoff_residual": [0.1, 0.4, 0.9, 1.4], "Yoff_residual": [0.8, 1.4, 2.2, 2.4]}
    )
    return y_test, y_pred


def test_load_model_cfg_returns_first_model(diagnostic_model_file):
    model_dict, model_cfg = diagnostic_utils._load_model_cfg(diagnostic_model_file)
    assert model_cfg["features"] == ["f1", "f2"]
    assert model_dict["target_mean"]["Xoff_residual"] == pytest.approx(1.0)


def test_load_stereo_regression_split_reconstructs_split(monkeypatch, diagnostic_model_file):
    df = pd.DataFrame(
        {
            "f1": np.arange(6, dtype=float),
            "f2": np.arange(6, dtype=float) + 1.0,
            "Xoff_residual": np.linspace(0.0, 0.5, 6),
            "Yoff_residual": np.linspace(1.0, 1.5, 6),
        }
    )
    monkeypatch.setattr(diagnostic_utils, "load_training_data", lambda *_: df)

    model, x_train, y_train, x_test, y_test, features, targets, model_dict = (
        diagnostic_utils.load_stereo_regression_split(diagnostic_model_file)
    )

    assert model == "cached-model"
    assert len(x_train) == 3
    assert len(x_test) == 3
    assert features == ["f1", "f2"]
    assert targets == ["Xoff_residual", "Yoff_residual"]
    assert model_dict["random_state"] == 3
    assert list(y_train.columns) == targets
    assert list(y_test.columns) == targets


def test_predict_unscaled_residuals_inverse_transforms_predictions():
    model = MagicMock()
    model.predict.return_value = np.array([[0.5, -2.0], [1.0, 0.0]])
    x_data = pd.DataFrame({"f1": [0.0, 1.0]})
    model_dict = {"target_mean": {"a": 1.0, "b": -1.0}, "target_std": {"a": 2.0, "b": 0.5}}

    result = diagnostic_utils.predict_unscaled_residuals(model, x_data, model_dict, ["a", "b"])

    assert result.loc[0, "a"] == pytest.approx(2.0)
    assert result.loc[1, "b"] == pytest.approx(-1.0)


def test_predict_unscaled_residuals_requires_scalers():
    model = MagicMock()
    model.predict.return_value = np.array([[0.0]])
    with pytest.raises(ValueError, match="Missing target standardization"):
        diagnostic_utils.predict_unscaled_residuals(model, pd.DataFrame({"f1": [0.0]}), {}, ["a"])


def test_compute_residual_normality_stats_returns_summary(residual_frames):
    y_test, y_pred = residual_frames
    stats = diagnostic_utils.compute_residual_normality_stats(y_test, y_pred, ["Xoff_residual"])
    assert stats["Xoff_residual"]["mean"] == pytest.approx(0.05)
    assert stats["Xoff_residual"]["n_samples"] == 4
    assert 0.0 <= stats["Xoff_residual"]["qq_r2"] <= 1.0


def test_compute_residual_normality_stats_skips_all_nan_target():
    y_test = pd.DataFrame({"target": [np.nan, np.nan]})
    y_pred = pd.DataFrame({"target": [0.0, 1.0]})
    assert diagnostic_utils.compute_residual_normality_stats(y_test, y_pred, ["target"]) == {}


def test_load_cached_metrics_and_normality_stats(diagnostic_model_file):
    _, metrics = diagnostic_utils.load_cached_generalization_metrics(diagnostic_model_file)
    _, normality = diagnostic_utils.load_cached_residual_normality_stats(diagnostic_model_file)
    assert metrics["Xoff_residual"]["rmse_train"] == pytest.approx(0.1)
    assert normality["Xoff_residual"]["std"] == pytest.approx(1.0)


def test_cached_loaders_return_none_for_missing_payloads(tmp_path):
    path = tmp_path / "empty.joblib"
    joblib.dump({"models": {"xgboost": {}}}, path)
    assert diagnostic_utils.load_cached_generalization_metrics(path)[1] is None
    assert diagnostic_utils.load_cached_residual_normality_stats(path)[1] is None


def test_load_model_and_importance_supports_dict_and_legacy_formats(tmp_path):
    dict_path = tmp_path / "dict.joblib"
    legacy_path = tmp_path / "legacy.joblib"
    joblib.dump(
        {"models": {"xgboost": {"features": ["f1", "f2"], "shap_importance": {"t": [0.2, 0.8]}}}},
        dict_path,
    )
    joblib.dump(
        {"models": {"xgboost": {"features": ["f1", "f2"], "feature_importances": [0.4, 0.6]}}},
        legacy_path,
    )

    assert diagnostic_utils.load_model_and_importance(dict_path, "t")[1]["f2"] == pytest.approx(0.8)
    assert diagnostic_utils.load_model_and_importance(legacy_path)[1]["f1"] == pytest.approx(0.4)
    assert diagnostic_utils.load_model_and_importance(dict_path, "missing")[1] is None


def test_get_cached_shap_explainer_and_importance_dataframe(diagnostic_model_file):
    df = diagnostic_utils.importance_dataframe(
        diagnostic_model_file, top_n=1, target_name="Xoff_residual"
    )
    assert diagnostic_utils.get_cached_shap_explainer(diagnostic_model_file) == "explainer"
    assert list(df["Feature"]) == ["f2"]
    assert df.iloc[0]["Importance"] == pytest.approx(0.7)


def test_validate_cached_data_summarizes_contents(diagnostic_model_file):
    summary = diagnostic_utils.validate_cached_data(diagnostic_model_file)
    assert summary["has_model"] is True
    assert summary["has_shap_importance"] is True
    assert summary["n_targets_with_shap"] == 1
    assert summary["n_importances_per_target"]["Xoff_residual"] == 2
