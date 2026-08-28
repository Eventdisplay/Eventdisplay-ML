"""Regression production contracts for training, persisted models, and output.

These tests deliberately exercise only the stereo-regression path.  They pin the
contracts that must remain stable while the classification pipeline evolves:
target exclusion and order during training, scaler provenance, serialized-model
loading, feature reordering during inference, residual reconstruction, and the
one-output-row-per-input-event guarantee.
"""

from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from eventdisplay_ml import models


class CapturingRegressor:
    """Minimal regressor that captures the exact arrays passed to ``fit``."""

    best_iteration = 0
    best_score = 0.0

    def fit(self, x_values, y_values, **kwargs):
        """Record training inputs without fitting a model."""
        self.x_train = np.array(x_values, copy=True)
        self.y_train = np.array(y_values, copy=True)
        self.fit_kwargs = kwargs
        return self

    def predict(self, x_values):
        """Return a zero residual for every requested event."""
        return np.zeros((len(x_values), 3), dtype=np.float32)


class OrderedPredictionRegressor:
    """Serializable predictor whose output exposes the received feature order."""

    def predict(self, x_values):
        """Generate residuals from the first two persisted feature columns."""
        # ``feature_beta`` must be first.  If inference stops reindexing to the
        # persisted order, these residuals (and thus final physics quantities)
        # change immediately.
        beta = x_values.iloc[:, 0].to_numpy(dtype=float)
        alpha = x_values.iloc[:, 1].to_numpy(dtype=float)
        return np.column_stack((beta, alpha, beta - alpha))


@pytest.fixture
def regression_frame():
    """Return indexed data with values that make ordering errors observable."""
    n_events = 240
    row_number = np.arange(n_events, dtype=float)
    return pd.DataFrame(
        {
            "feature_beta": 1000.0 + row_number,
            "feature_alpha": -500.0 - row_number,
            "ErecS": np.full(n_events, 3.0),
            "DispNImages": np.full(n_events, 2),
            "Xoff_weighted_bdt": 0.1 * row_number,
            "Yoff_weighted_bdt": -0.2 * row_number,
            "Xoff_residual": 2.0 + 0.01 * row_number,
            "Yoff_residual": -3.0 - 0.02 * row_number,
            "E_residual": 0.5 + 0.001 * row_number,
        },
        index=10_000 + 7 * np.arange(n_events),
    )


def test_regression_training_contract_excludes_targets_and_uses_train_only_scalers(
    regression_frame,
):
    """Pin arrays, target order, split provenance, and weights sent to XGBoost."""
    targets = ["Xoff_residual", "Yoff_residual", "E_residual"]
    config = {
        "targets": targets,
        "train_test_fraction": 0.5,
        "random_state": 19,
        "eval_max_events": 0,
        "models": {"xgboost": {"hyper_parameters": {}}},
    }
    captured_model = CapturingRegressor()

    with (
        patch("xgboost.XGBRegressor", return_value=captured_model),
        patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
    ):
        result = models.train_regression(regression_frame, config)

    feature_columns = [column for column in regression_frame if column not in targets]
    train_positions, test_positions = train_test_split(
        np.arange(len(regression_frame)), train_size=0.5, random_state=19
    )
    expected_train_features = regression_frame.iloc[train_positions][feature_columns].to_numpy(
        dtype=np.float32
    )
    expected_train_targets = regression_frame.iloc[train_positions][targets]
    expected_mean = expected_train_targets.mean()
    expected_std = expected_train_targets.std()
    expected_scaled_targets = ((expected_train_targets - expected_mean) / expected_std).to_numpy(
        dtype=np.float32
    )

    assert result["features"] == feature_columns
    assert result["models"]["xgboost"]["features"] == feature_columns
    np.testing.assert_array_equal(captured_model.x_train, expected_train_features)
    np.testing.assert_allclose(captured_model.y_train, expected_scaled_targets, atol=1e-7)
    assert captured_model.fit_kwargs["sample_weight"].shape == (len(train_positions),)
    assert captured_model.fit_kwargs["sample_weight_eval_set"][0].shape == (len(test_positions),)
    assert result["target_mean"] == pytest.approx(expected_mean.to_dict(), abs=0.0)
    assert result["target_std"] == pytest.approx(expected_std.to_dict(), abs=0.0)


def test_regression_training_reduced_profile_selects_requested_columns(monkeypatch):
    n_events = 240
    row_number = np.arange(n_events, dtype=float)
    reduced_columns = [
        "Xoff_weighted_bdt",
        "Yoff_weighted_bdt",
        "Xoff_intersect",
        "Yoff_intersect",
        "Diff_Xoff",
        "Diff_Yoff",
        "DispNImages",
        "Erec",
        "ErecS",
        "EmissionHeight",
        "Geomagnetic_Angle",
        "array_footprint",
        *[f"width_length_{i}" for i in range(4)],
        *[f"R_core_{i}" for i in range(4)],
        *[f"loss_{i}" for i in range(4)],
    ]
    data = {column: row_number + offset for offset, column in enumerate(reduced_columns)}
    data.update(
        {
            "Xoff_residual": 1.0 + row_number,
            "Yoff_residual": 2.0 + row_number,
            "E_residual": 0.01 + 0.001 * row_number,
        }
    )
    data["ErecS"] = np.full(n_events, 3.0)
    data["DispNImages"] = np.full(n_events, 2)
    frame = pd.DataFrame(data)
    captured_model = CapturingRegressor()
    monkeypatch.setattr("xgboost.XGBRegressor", lambda **_: captured_model)
    monkeypatch.setattr("eventdisplay_ml.models.evaluate_regression_model", lambda *_args: {})

    result = models.train_regression(
        frame,
        {
            "targets": ["Xoff_residual", "Yoff_residual", "E_residual"],
            "feature_profile": "reduced",
            "train_test_fraction": 0.5,
            "random_state": 19,
            "eval_max_events": 0,
            "diagnostic_max_events": 0,
            "models": {"xgboost": {"hyper_parameters": {}}},
        },
    )

    assert result["features"] == reduced_columns
    assert result["models"]["xgboost"]["features"] == reduced_columns


def test_persisted_regression_model_preserves_feature_order_and_reconstructs_truth(
    tmp_path, monkeypatch
):
    """Load a model artifact and apply it with shuffled input columns.

    This is the production boundary: model features and target scalers come from
    disk, while the flattened event data can be in a different column order.
    """
    model_prefix = tmp_path / "stereo_model"
    feature_order = [
        "feature_beta",
        "feature_alpha",
        "Xoff_weighted_bdt",
        "Yoff_weighted_bdt",
        "ErecS",
    ]
    target_mean = {"Xoff_residual": 1.0, "Yoff_residual": -2.0, "E_residual": 0.25}
    target_std = {"Xoff_residual": 0.5, "Yoff_residual": 2.0, "E_residual": 0.1}
    joblib.dump(
        {
            "models": {"xgboost": {"model": OrderedPredictionRegressor()}},
            "features": feature_order,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        tmp_path / "stereo_model.joblib.gz",
    )
    loaded_models, loaded_parameters = models.load_regression_models(str(model_prefix), "xgboost")

    # Deliberately not in persisted feature order, and includes a column that
    # must not reach the model.
    flattened = pd.DataFrame(
        {
            "ignored_new_column": [99.0, 98.0],
            "feature_alpha": [4.0, 7.0],
            "ErecS": [10.0, 100.0],
            "Xoff_weighted_bdt": [0.2, -0.5],
            "feature_beta": [3.0, 11.0],
            "Yoff_weighted_bdt": [-1.0, 2.0],
        }
    )
    monkeypatch.setattr(models, "flatten_feature_data", lambda *_args, **_kwargs: flattened)
    monkeypatch.setattr(models.data_processing, "print_variable_statistics", lambda *_args: None)

    pred_xoff, pred_yoff, pred_log_energy = models.apply_regression_models(
        pd.DataFrame({"event": [1, 2]}),
        {"models": loaded_models, **loaded_parameters},
    )

    # Raw predicted residuals are [beta, alpha, beta-alpha], then each target
    # is inverse-standardized and added to its DispBDT baseline.
    np.testing.assert_allclose(pred_xoff, [2.7, 6.0])
    np.testing.assert_allclose(pred_yoff, [5.0, 14.0])
    np.testing.assert_allclose(pred_log_energy, [1.15, 2.65])


def test_stereo_output_writer_keeps_nan_energy_rows_and_converts_only_log_energy(monkeypatch):
    """Pin ROOT payload names, float32 conversion, and row-preserving NaNs."""
    tree = MagicMock()
    monkeypatch.setattr(
        models,
        "apply_regression_models",
        lambda *_args: (
            np.array([1.25, np.nan]),
            np.array([-2.5, 3.5]),
            np.array([2.0, np.nan]),
        ),
    )

    models._apply_model(
        "stereo_analysis",
        pd.DataFrame({"event": [1, 2], "runNumber": [12345, 12345], "eventNumber": [1, 2]}),
        {},
        tree,
    )

    payload = tree.extend.call_args.args[0]
    assert list(payload) == ["Dir_Xoff", "Dir_Yoff", "Dir_Erec", "runNumber", "eventNumber"]
    assert all(payload[key].dtype == np.float32 for key in ("Dir_Xoff", "Dir_Yoff", "Dir_Erec"))
    assert all(payload[key].dtype == np.int32 for key in ("runNumber", "eventNumber"))
    assert len(payload["Dir_Xoff"]) == 2
    assert payload["Dir_Erec"][0] == pytest.approx(100.0)
    assert np.isnan(payload["Dir_Xoff"][1])
    assert np.isnan(payload["Dir_Erec"][1])
