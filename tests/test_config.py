"""Tests for configure_training and configure_apply in config.py."""

import json
import sys
from unittest.mock import MagicMock

import pytest

from eventdisplay_ml import config


@pytest.fixture
def model_parameters_file(tmp_path):
    path = tmp_path / "model_parameters.json"
    path.write_text(
        json.dumps(
            {
                "energy_bins_log10_tev": [{"E_min": -1.0, "E_max": 1.0}],
                "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 20}],
                "tmva_style": " yes ",
            }
        )
    )
    return path


def test_configure_training_stereo_updates_models(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model_prefix",
            "model",
            "--input_file_list",
            "inputs.txt",
            "--max_cores",
            "3",
            "--random_state",
            "7",
            "--min_images",
            "4",
        ],
    )
    monkeypatch.setattr(
        config,
        "hyper_parameters",
        lambda *_: {"xgboost": {"hyper_parameters": {}}, "warn": {"hyper_parameters": None}},
    )
    monkeypatch.setattr(config, "target_features", lambda *_: ["target_a"])
    monkeypatch.setattr(config, "pre_cuts_regression", lambda min_images: f"cut_{min_images}")

    result = config.configure_training("stereo_analysis")

    assert result["models"]["xgboost"]["hyper_parameters"] == {"n_jobs": 3, "random_state": 7}
    assert result["pre_cuts"] == "cut_4"
    assert result["targets"] == ["target_a"]


def test_configure_training_classification_parses_tmva_style(monkeypatch, model_parameters_file):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model_prefix",
            "model",
            "--input_signal_file_list",
            "signal.txt",
            "--input_background_file_list",
            "background.txt",
            "--model_parameters",
            str(model_parameters_file),
            "--energy_bin_number",
            "0",
            "--max_cores",
            "2",
        ],
    )
    monkeypatch.setattr(
        config, "hyper_parameters", lambda *_: {"xgboost": {"hyper_parameters": {}}}
    )
    monkeypatch.setattr(config, "target_features", lambda *_: ["label"])
    monkeypatch.setattr(config, "pre_cuts_classification", lambda e_min, e_max: (e_min, e_max))

    result = config.configure_training("classification")

    assert result["tmva_style"] is True
    assert result["pre_cuts"][0] == pytest.approx(0.1)
    assert result["pre_cuts"][1] == pytest.approx(10.0)
    assert result["energy_bins_log10_tev"]["E_min"] == pytest.approx(-1.0)
    assert result["models"]["xgboost"]["hyper_parameters"]["n_jobs"] == 2


def test_configure_training_classification_handles_boolean_tmva_style(monkeypatch, tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "energy_bins_log10_tev": [{"E_min": 0.0, "E_max": 0.5}],
                "zenith_bins_deg": [],
                "tmva_style": False,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model_prefix",
            "model",
            "--input_signal_file_list",
            "signal.txt",
            "--input_background_file_list",
            "background.txt",
            "--model_parameters",
            str(params_path),
        ],
    )
    monkeypatch.setattr(
        config, "hyper_parameters", lambda *_: {"xgboost": {"hyper_parameters": {}}}
    )
    monkeypatch.setattr(config, "target_features", lambda *_: ["label"])
    monkeypatch.setattr(config, "pre_cuts_classification", lambda **_: "cuts")

    result = config.configure_training("classification")

    assert result["tmva_style"] is False
    assert result["pre_cuts"] == "cuts"


def test_configure_apply_stereo_loads_scalers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input_file",
            "input.root",
            "--model_prefix",
            "model",
            "--output_file",
            "output.root",
            "--model_name",
            "custom",
        ],
    )
    load_models = MagicMock(
        return_value=(
            {"custom": {"model": object()}},
            {
                "energy_bins_log10_tev": [{"E_min": -1.0, "E_max": 1.0}],
                "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 20}],
                "target_mean": {"Xoff_residual": 0.0},
                "target_std": {"Xoff_residual": 1.0},
            },
        )
    )
    monkeypatch.setattr(config, "load_models", load_models)

    result = config.configure_apply("stereo_analysis")

    load_models.assert_called_once_with("stereo_analysis", "model", "custom")
    assert result["target_mean"]["Xoff_residual"] == pytest.approx(0.0)
    assert result["target_std"]["Xoff_residual"] == pytest.approx(1.0)


def test_configure_apply_classification_omits_regression_scalers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--input_file",
            "input.root",
            "--model_prefix",
            "model",
            "--output_file",
            "out.root",
        ],
    )
    monkeypatch.setattr(
        config, "load_models", lambda *_: ({0: {"model": object()}}, {"zenith_bins_deg": []})
    )

    result = config.configure_apply("classification")

    assert "target_mean" not in result
    assert result["models"][0]["model"] is not None


def test_configure_training_classification_missing_energy_bins_no_attribute_error(
    monkeypatch, tmp_path
):
    """Regression test for E6: missing energy_bins_log10_tev must not raise AttributeError."""
    params_path = tmp_path / "params_no_bins.json"
    params_path.write_text(json.dumps({}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model_prefix",
            "model",
            "--input_signal_file_list",
            "signal.txt",
            "--input_background_file_list",
            "background.txt",
            "--model_parameters",
            str(params_path),
        ],
    )
    monkeypatch.setattr(
        config, "hyper_parameters", lambda *_: {"xgboost": {"hyper_parameters": {}}}
    )
    monkeypatch.setattr(config, "target_features", lambda *_: ["label"])
    monkeypatch.setattr(config, "pre_cuts_classification", lambda e_min, e_max: (e_min, e_max))
    # Return params without energy_bins_log10_tev to trigger the fallback path
    monkeypatch.setattr(
        config.utils,
        "load_model_parameters",
        lambda *_: {"tmva_style": False, "zenith_bins_deg": []},
    )
    # Patch np.power so None exponents don't raise TypeError (we only care about no AttributeError)
    monkeypatch.setattr(config.np, "power", lambda base, exp: exp)

    result = config.configure_training("classification")

    assert result["energy_bins_log10_tev"] == []
