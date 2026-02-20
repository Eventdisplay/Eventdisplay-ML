"""Tests for standardized target inversion in regression inference."""

import numpy as np
import pandas as pd

from eventdisplay_ml import models


class DummyModel:
    """Simple model that returns fixed scaled residuals."""

    def __init__(self, preds_scaled):
        self._preds_scaled = np.asarray(preds_scaled, dtype=np.float64)

    def predict(self, _x):
        """Return fixed predictions regardless of input."""
        return self._preds_scaled


def test_apply_regression_models_inverts_standardization(monkeypatch):
    df_flat = pd.DataFrame(
        {
            "Xoff_weighted_bdt": [10.0, 20.0],
            "Yoff_weighted_bdt": [30.0, 40.0],
            "ErecS": [100.0, 1000.0],
        }
    )

    preds_scaled = np.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 2.0],
        ]
    )

    model_configs = {
        "models": {"xgboost": {"model": DummyModel(preds_scaled), "features": df_flat.columns}},
        "target_mean": {
            "Xoff_residual": 1.0,
            "Yoff_residual": 2.0,
            "E_residual": 0.5,
        },
        "target_std": {
            "Xoff_residual": 2.0,
            "Yoff_residual": 3.0,
            "E_residual": 0.1,
        },
    }

    def _mock_flatten(*_args, **_kwargs):
        return df_flat

    monkeypatch.setattr(models, "flatten_feature_data", _mock_flatten)
    monkeypatch.setattr(models.data_processing, "print_variable_statistics", lambda *_: None)

    pred_xoff, pred_yoff, pred_erec_log = models.apply_regression_models(
        pd.DataFrame({"dummy": [0, 1]}), model_configs
    )

    expected_xoff = np.array([11.0, 23.0])
    expected_yoff = np.array([35.0, 42.0])
    expected_erec_log = np.array([2.4, 3.7])

    np.testing.assert_allclose(pred_xoff, expected_xoff, rtol=0, atol=1e-8)
    np.testing.assert_allclose(pred_yoff, expected_yoff, rtol=0, atol=1e-8)
    np.testing.assert_allclose(pred_erec_log, expected_erec_log, rtol=0, atol=1e-8)
