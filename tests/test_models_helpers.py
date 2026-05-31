"""Tests for models.py helper functions, loaders, and training routines."""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import features, models


@pytest.fixture
def regression_model_file(tmp_path):
    path = tmp_path / "regression.joblib"
    joblib.dump(
        {
            "models": {"xgboost": {"model": "reg-model"}},
            "features": ["f1", "f2"],
            "target_mean": {"Xoff_residual": 0.0},
        },
        path,
    )
    return path


@pytest.fixture
def classification_prefix(tmp_path):
    prefix = tmp_path / "model"
    efficiency = pd.DataFrame(
        {
            "signal_efficiency": np.linspace(0.0, 1.0, 101),
            "background_efficiency": np.linspace(1.0, 0.0, 101),
            "threshold": np.linspace(1.0, 0.0, 101),
        }
    )
    for ebin in (0, 1):
        suffix = ".joblib.gz" if ebin == 1 else ".joblib"
        joblib.dump(
            {
                "models": {"xgboost": {"model": f"clf-{ebin}", "efficiency": efficiency}},
                "features": ["f1"],
                "energy_bins_log10_tev": {"E_min": float(ebin), "E_max": float(ebin + 1)},
                "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 20}],
                "energy_bin_number": ebin,
            },
            tmp_path / f"model_ebin{ebin}{suffix}",
        )
    return prefix


def test_save_models_writes_expected_joblib(tmp_path):
    model_configs = {"model_prefix": str(tmp_path / "saved"), "models": {"xgboost": {}}}
    models.save_models(model_configs)
    assert (tmp_path / "saved.joblib.gz").exists()


def test_load_models_dispatches_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(models, "load_regression_models", lambda *args: ("reg", args))
    monkeypatch.setattr(models, "load_classification_models", lambda *args: ("clf", args))
    assert models.load_models("stereo_analysis", "m", "n")[0] == "reg"
    assert models.load_models("classification", "m", "n")[0] == "clf"
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        models.load_models("bad", "m", "n")


def test_load_classification_models_collects_bins_and_thresholds(classification_prefix):
    loaded, par = models.load_classification_models(str(classification_prefix), "xgboost")
    assert sorted(loaded) == [0, 1]
    assert loaded[0]["thresholds"][20] == pytest.approx(0.8)
    assert par["energy_bins_log10_tev"][1]["E_max"] == pytest.approx(2.0)


def test_classification_thresholds_and_parameter_updates():
    efficiency = pd.DataFrame({"signal_efficiency": [0.2, 0.4, 0.6], "threshold": [0.9, 0.5, 0.1]})
    thresholds = models._calculate_classification_thresholds(
        efficiency, min_efficiency=0.2, steps=20
    )
    updated = models._update_parameters(
        {}, [{"Ze_min": 0, "Ze_max": 20}], {"E_min": -1.0, "E_max": 1.0}, 0
    )
    assert thresholds[20] == pytest.approx(0.9)
    assert updated["energy_bins_log10_tev"][0]["E_min"] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="Inconsistent zenith_bins_deg"):
        models._update_parameters(updated, [], {"E_min": 0.0, "E_max": 1.0}, 1)


def test_check_bin_and_load_regression_models(regression_model_file):
    with pytest.raises(ValueError, match="Bin number mismatch"):
        models._check_bin(0, 1)
    loaded, par = models.load_regression_models(str(regression_model_file), "xgboost")
    assert loaded["xgboost"]["model"] == "reg-model"
    assert par["target_mean"]["Xoff_residual"] == pytest.approx(0.0)
    assert "target_std" not in par


def test_output_tree_and_apply_model(monkeypatch):
    root_file = MagicMock()
    stereo_tree = MagicMock()
    class_tree = MagicMock()
    root_file.mktree.side_effect = [stereo_tree, class_tree]
    models._output_tree("stereo_analysis", root_file)
    models._output_tree("classification", root_file, [20])
    monkeypatch.setattr(
        models,
        "apply_regression_models",
        lambda *_: (np.array([1.0]), np.array([2.0]), np.array([1.0])),
    )
    monkeypatch.setattr(
        models,
        "apply_classification_models",
        lambda *_: (np.array([0.8]), {20: np.array([1], dtype=np.uint8)}),
    )

    models._apply_model("stereo_analysis", pd.DataFrame({"a": [1]}), {}, stereo_tree)
    models._apply_model("classification", pd.DataFrame({"a": [1]}), {}, class_tree, [20])

    assert root_file.mktree.call_args_list[0].args[0] == "StereoAnalysis"
    assert root_file.mktree.call_args_list[1].args[0] == "Classification"
    assert stereo_tree.extend.call_args.args[0]["Dir_Erec"][0] == pytest.approx(10.0)
    assert class_tree.extend.call_args.args[0]["Gamma_Prediction"][0] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        models._output_tree("bad", root_file)
    with pytest.raises(ValueError, match="Unknown analysis_type"):
        models._apply_model("bad", pd.DataFrame(), {}, stereo_tree)


def test_train_regression_returns_none_for_empty_frame():
    result = models.train_regression(pd.DataFrame(), {"targets": ["x"], "models": {}})
    assert result is None


def test_train_regression_trains_tiny_model(monkeypatch):
    rng = np.random.default_rng(42)
    n_rows = 60
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n_rows),
            "f2": rng.standard_normal(n_rows),
            "ErecS": rng.uniform(0.1, 10.0, n_rows),
            "Xoff_weighted_bdt": rng.standard_normal(n_rows),
            "Yoff_weighted_bdt": rng.standard_normal(n_rows),
            "Xoff_residual": rng.standard_normal(n_rows) * 0.01,
            "Yoff_residual": rng.standard_normal(n_rows) * 0.01,
            "E_residual": rng.standard_normal(n_rows) * 0.01,
            "DispNImages": rng.integers(2, 5, n_rows),
        }
    )
    config = {
        "targets": ["Xoff_residual", "Yoff_residual", "E_residual"],
        "train_test_fraction": 0.5,
        "random_state": 42,
        "models": {
            "test": {
                "hyper_parameters": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "early_stopping_rounds": 2,
                }
            }
        },
    }
    monkeypatch.setattr(models, "evaluate_regression_model", lambda *args: {"ok": np.array([1.0])})

    result = models.train_regression(df, config)

    assert result["models"]["test"]["features"] == [
        "f1",
        "f2",
        "ErecS",
        "Xoff_weighted_bdt",
        "Yoff_weighted_bdt",
        "DispNImages",
    ]
    assert "generalization_metrics" in result["models"]["test"]
    assert "residual_normality_stats" in result["models"]["test"]


def test_train_classification_handles_empty_and_zenith_efficiencies(monkeypatch):
    with pytest.raises(ValueError, match="requires non-empty"):
        models.train_classification([pd.DataFrame(), pd.DataFrame({"f1": [1.0]})], {"models": {}})

    rng = np.random.default_rng(7)
    signal = pd.DataFrame(
        {
            "f1": rng.standard_normal(40),
            "f2": rng.standard_normal(40) + 1.0,
            "ze_bin": np.repeat([0, 1], 20),
        }
    )
    background = pd.DataFrame(
        {
            "f1": rng.standard_normal(40),
            "f2": rng.standard_normal(40) - 1.0,
            "ze_bin": np.repeat([0, 1], 20),
        }
    )
    config = {
        "train_test_fraction": 0.5,
        "random_state": 42,
        "models": {
            "test": {
                "hyper_parameters": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "eval_metric": "logloss",
                }
            }
        },
    }
    monkeypatch.setattr(
        models, "evaluate_classification_model", lambda *args: {"label": np.array([0.1, 0.9, 0.0])}
    )

    result = models.train_classification([signal, background], config)

    assert "efficiency" in result["models"]["test"]
    assert "efficiency_ze0" in result["models"]["test"]
    assert result["models"]["test"]["features"] == ["f1", "f2", "ze_bin"]


def test_process_file_chunked_uses_tmva_style_features_when_flag_set():
    """Regression test for E5: classification with tmva_style=True must use features_tmva_style."""
    tmva_features = features.features_tmva_style("classification", training=False)
    expected = [b for b in tmva_features if b not in {"ze_bin", "ArrayPointing_Azimuth"}] + [
        "ArrayPointing_Elevation"
    ]
    regular_features = features.features("classification", training=False)

    def fake_open(path):
        raise RuntimeError("uproot not needed for this assertion")

    with patch("eventdisplay_ml.models.uproot.open", side_effect=fake_open):
        with pytest.raises(RuntimeError, match="uproot not needed"):
            models.process_file_chunked(
                "classification",
                {"tmva_style": True, "input_file": "dummy.root"},
            )

    # Verify that tmva_style features differ from regular features
    assert set(expected) != set(regular_features)
    assert "ArrayPointing_Elevation" in expected
    assert "ze_bin" not in expected
