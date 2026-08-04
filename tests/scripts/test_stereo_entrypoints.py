"""Unit tests for production stereo-regression console-script wiring."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from eventdisplay_ml.scripts import apply_xgb_stereo, train_xgb_stereo


def test_train_stereo_entrypoint_runs_the_complete_regression_pipeline(monkeypatch, caplog):
    """The training CLI must connect configuration, loading, training, and saving unchanged."""
    caplog.set_level("INFO")
    configured = {"input_file_list": "gamma_inputs.txt", "model_prefix": "stereo_model"}
    loaded_data = pd.DataFrame({"feature": [1.0]})
    trained = {"model_prefix": "stereo_model", "models": {"xgboost": {"model": object()}}}
    configure = MagicMock(return_value=configured)
    load_data = MagicMock(return_value=loaded_data)
    train = MagicMock(return_value=trained)
    save = MagicMock()
    monkeypatch.setattr(train_xgb_stereo, "configure_training", configure)
    monkeypatch.setattr(train_xgb_stereo, "load_training_data", load_data)
    monkeypatch.setattr(train_xgb_stereo, "train_regression", train)
    monkeypatch.setattr(train_xgb_stereo, "save_models", save)

    train_xgb_stereo.main()

    configure.assert_called_once_with("stereo_analysis")
    load_data.assert_called_once_with(configured, "gamma_inputs.txt", "stereo_analysis")
    train.assert_called_once_with(loaded_data, configured)
    save.assert_called_once_with(trained)
    assert "stereo_analysis model trained successfully" in caplog.text


def test_train_stereo_entrypoint_does_not_save_when_regression_training_fails(monkeypatch):
    """A failed regression fit must propagate and never create a partial artifact."""
    configured = {"input_file_list": "gamma_inputs.txt"}
    save = MagicMock()
    monkeypatch.setattr(train_xgb_stereo, "configure_training", lambda *_args: configured)
    monkeypatch.setattr(
        train_xgb_stereo, "load_training_data", lambda *_args: pd.DataFrame({"feature": [1.0]})
    )
    monkeypatch.setattr(
        train_xgb_stereo,
        "train_regression",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    monkeypatch.setattr(train_xgb_stereo, "save_models", save)

    with pytest.raises(RuntimeError, match="fit failed"):
        train_xgb_stereo.main()

    save.assert_not_called()


def test_apply_stereo_entrypoint_passes_the_loaded_configuration_to_streaming(monkeypatch):
    """The apply CLI must select stereo analysis and preserve loaded model metadata."""
    configured = {
        "models": {"xgboost": {"model": object()}},
        "target_mean": {"Xoff_residual": 0.0},
        "target_std": {"Xoff_residual": 1.0},
    }
    configure = MagicMock(return_value=configured)
    process = MagicMock()
    monkeypatch.setattr(apply_xgb_stereo, "configure_apply", configure)
    monkeypatch.setattr(apply_xgb_stereo, "process_file_chunked", process)

    apply_xgb_stereo.main()

    configure.assert_called_once_with("stereo_analysis")
    process.assert_called_once_with("stereo_analysis", configured)


def test_apply_stereo_entrypoint_does_not_stream_when_configuration_fails(monkeypatch):
    """Invalid model configuration must stop before input ROOT data are processed."""
    process = MagicMock()
    monkeypatch.setattr(
        apply_xgb_stereo,
        "configure_apply",
        lambda *_args: (_ for _ in ()).throw(ValueError("missing target_std")),
    )
    monkeypatch.setattr(apply_xgb_stereo, "process_file_chunked", process)

    with pytest.raises(ValueError, match="missing target_std"):
        apply_xgb_stereo.main()

    process.assert_not_called()
