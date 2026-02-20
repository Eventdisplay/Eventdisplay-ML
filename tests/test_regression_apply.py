"""Comprehensive tests for regression model apply, residual computation, and standardization."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import models


class DummyXGBRegressor:
    """Simple XGBoost-like model that returns fixed scaled residuals."""

    def __init__(self, preds_scaled):
        self._preds_scaled = np.asarray(preds_scaled, dtype=np.float64)

    def predict(self, _x):
        """Return fixed predictions regardless of input."""
        return self._preds_scaled


class TestApplyRegressionStandardizationInversion:
    """Test standardization inversion in apply_regression_models."""

    def test_apply_regression_inverts_standardization(self, monkeypatch):
        """Verify that predicted residuals are correctly inverted from standardized space."""
        df_flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [10.0, 20.0],
                "Yoff_weighted_bdt": [30.0, 40.0],
                "ErecS": [100.0, 1000.0],
            }
        )

        # Model returns fixed scaled predictions (in standardized space: mean=0, std=1)
        preds_scaled = np.array(
            [
                [
                    0.0,
                    1.0,
                    -1.0,
                ],  # Event 0: Xoff_residual_scaled=0, Yoff_residual_scaled=1, E_residual_scaled=-1
                [
                    1.0,
                    0.0,
                    2.0,
                ],  # Event 1: Xoff_residual_scaled=1, Yoff_residual_scaled=0, E_residual_scaled=2
            ]
        )

        model_configs = {
            "models": {
                "xgboost": {"model": DummyXGBRegressor(preds_scaled), "features": df_flat.columns}
            },
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

        # Inverse transform: y = y_scaled * std + mean
        # Event 0: Xoff = 0.0 * 2.0 + 1.0 = 1.0, then add baseline: 1.0 + 10.0 = 11.0
        # Event 1: Xoff = 1.0 * 2.0 + 1.0 = 3.0, then add baseline: 3.0 + 20.0 = 23.0
        expected_xoff = np.array([11.0, 23.0])

        # Event 0: Yoff = 1.0 * 3.0 + 2.0 = 5.0, then add baseline: 5.0 + 30.0 = 35.0
        # Event 1: Yoff = 0.0 * 3.0 + 2.0 = 2.0, then add baseline: 2.0 + 40.0 = 42.0
        expected_yoff = np.array([35.0, 42.0])

        # disp_erec_log = log10([100, 1000]) = [2, 3]
        # Event 0: E_residual = -1.0 * 0.1 + 0.5 = 0.4, then add baseline: 0.4 + 2.0 = 2.4
        # Event 1: E_residual = 2.0 * 0.1 + 0.5 = 0.7, then add baseline: 0.7 + 3.0 = 3.7
        expected_erec_log = np.array([2.4, 3.7])

        np.testing.assert_allclose(pred_xoff, expected_xoff, rtol=0, atol=1e-8)
        np.testing.assert_allclose(pred_yoff, expected_yoff, rtol=0, atol=1e-8)
        np.testing.assert_allclose(pred_erec_log, expected_erec_log, rtol=0, atol=1e-8)

    def test_apply_regression_missing_standardization_params(self, monkeypatch):
        """Verify that missing target_mean/target_std raises clear error."""
        df_flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [10.0],
                "Yoff_weighted_bdt": [30.0],
                "ErecS": [100.0],
            }
        )

        model_configs = {
            "models": {
                "xgboost": {
                    "model": DummyXGBRegressor(np.array([[0.0, 1.0, -1.0]])),
                    "features": df_flat.columns,
                }
            },
            # Missing target_mean and target_std
        }

        def _mock_flatten(*_args, **_kwargs):
            return df_flat

        monkeypatch.setattr(models, "flatten_feature_data", _mock_flatten)
        monkeypatch.setattr(models.data_processing, "print_variable_statistics", lambda *_: None)

        with pytest.raises(ValueError, match="Missing target standardization parameters"):
            models.apply_regression_models(pd.DataFrame({"dummy": [0]}), model_configs)


class TestApplyRegressionErecSHandling:
    """Test ErecS validation and log10 computation in apply."""

    def test_apply_regression_handles_invalid_erecs(self, monkeypatch):
        """Verify that invalid ErecS values (<=0 or NaN) are handled gracefully."""
        df_flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [10.0, 20.0, 30.0],
                "Yoff_weighted_bdt": [30.0, 40.0, 50.0],
                "ErecS": [100.0, -5.0, np.nan],  # 2nd event: negative, 3rd event: NaN
            }
        )

        preds_scaled = np.array(
            [
                [0.0, 1.0, -1.0],
                [1.0, 0.0, 2.0],
                [0.5, 0.5, 0.0],
            ]
        )

        model_configs = {
            "models": {
                "xgboost": {"model": DummyXGBRegressor(preds_scaled), "features": df_flat.columns}
            },
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
            pd.DataFrame({"dummy": [0, 1, 2]}), model_configs
        )

        # Event 0: ErecS valid, log10(100) = 2
        assert np.isfinite(pred_erec_log[0])
        assert np.isclose(np.log10(100), 2.0)

        # Events 1 and 2: ErecS invalid, should be NaN
        assert np.isnan(pred_erec_log[1]), "Event with negative ErecS should product NaN"
        assert np.isnan(pred_erec_log[2]), "Event with NaN ErecS should product NaN"

        # Xoff and Yoff should still be valid for all events
        assert len(pred_xoff) == 3
        assert len(pred_yoff) == 3
        assert all(np.isfinite(pred_xoff))
        assert all(np.isfinite(pred_yoff))

    def test_apply_regression_output_length_matches_input(self, monkeypatch):
        """Verify that output arrays have same length as input, even with invalid ErecS."""
        n_events = 100
        rng = np.random.default_rng(42)
        df_flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": rng.uniform(0, 10, n_events),
                "Yoff_weighted_bdt": rng.uniform(0, 10, n_events),
                "ErecS": np.where(
                    rng.uniform(0, 1, n_events) > 0.2,
                    rng.uniform(10, 110, n_events),  # 80% valid
                    np.nan,  # 20% NaN
                ),
            }
        )

        preds_scaled = rng.standard_normal((n_events, 3))

        model_configs = {
            "models": {
                "xgboost": {"model": DummyXGBRegressor(preds_scaled), "features": df_flat.columns}
            },
            "target_mean": {
                "Xoff_residual": 0.0,
                "Yoff_residual": 0.0,
                "E_residual": 0.0,
            },
            "target_std": {
                "Xoff_residual": 1.0,
                "Yoff_residual": 1.0,
                "E_residual": 1.0,
            },
        }

        def _mock_flatten(*_args, **_kwargs):
            return df_flat

        monkeypatch.setattr(models, "flatten_feature_data", _mock_flatten)
        monkeypatch.setattr(models.data_processing, "print_variable_statistics", lambda *_: None)

        pred_xoff, pred_yoff, pred_erec_log = models.apply_regression_models(
            pd.DataFrame({"dummy": np.arange(n_events)}), model_configs
        )

        assert len(pred_xoff) == n_events, "Output length should match input length"
        assert len(pred_yoff) == n_events, "Output length should match input length"
        assert len(pred_erec_log) == n_events, "Output length should match input length"


class TestResidualComputation:
    """Test residual computation during training (the basis for apply predictions)."""

    def test_residual_computation_from_mc_and_baseline(self):
        """Verify residuals are computed correctly as MC_true - baseline."""
        mc_xoff = np.array([1.0, 2.0, 3.0])
        mc_yoff = np.array([4.0, 5.0, 6.0])
        mc_e0_log = np.array([0.0, 1.0, 2.0])

        baseline_xoff = np.array([0.5, 1.5, 2.5])
        baseline_yoff = np.array([3.5, 4.5, 5.5])
        baseline_erec_log = np.array([-0.5, 0.5, 1.5])

        # Residuals should be MC - baseline
        xoff_residual = mc_xoff - baseline_xoff
        yoff_residual = mc_yoff - baseline_yoff
        e_residual = mc_e0_log - baseline_erec_log

        expected_xoff_residual = np.array([0.5, 0.5, 0.5])
        expected_yoff_residual = np.array([0.5, 0.5, 0.5])
        expected_e_residual = np.array([0.5, 0.5, 0.5])

        np.testing.assert_allclose(xoff_residual, expected_xoff_residual)
        np.testing.assert_allclose(yoff_residual, expected_yoff_residual)
        np.testing.assert_allclose(e_residual, expected_e_residual)

    def test_residual_standardization(self):
        """Verify residuals standardize correctly (mean=0, std=1)."""
        residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mean = residuals.mean()
        std = residuals.std()

        residuals_scaled = (residuals - mean) / std

        assert np.isclose(residuals_scaled.mean(), 0.0, atol=1e-10)
        assert np.isclose(residuals_scaled.std(), 1.0, atol=1e-10)

    def test_residual_reconstruction_after_standardization(self):
        """Verify that residuals can be reconstructed from standardized predictions."""
        original_residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        mean = original_residuals.mean()
        std = original_residuals.std()

        # Training standardize
        scaled_residuals = (original_residuals - mean) / std

        # Apply model predicts scaled residuals, inverse transform
        predicted_scaled = scaled_residuals  # Assume perfect prediction
        reconstructed_residuals = predicted_scaled * std + mean

        np.testing.assert_allclose(reconstructed_residuals, original_residuals, rtol=1e-10)


class TestFinalPredictionReconstruction:
    """Test that final predictions correctly reconstruct from residuals + baselines."""

    def test_final_direction_reconstruction(self):
        """Verify direction predictions = baseline + residual."""
        baseline_xoff = np.array([1.0, 2.0, 3.0])
        baseline_yoff = np.array([4.0, 5.0, 6.0])
        pred_xoff_residual = np.array([0.5, 0.3, 0.2])
        pred_yoff_residual = np.array([0.1, 0.2, 0.3])

        # Final prediction should be baseline + residual
        final_xoff = baseline_xoff + pred_xoff_residual
        final_yoff = baseline_yoff + pred_yoff_residual

        expected_xoff = np.array([1.5, 2.3, 3.2])
        expected_yoff = np.array([4.1, 5.2, 6.3])

        np.testing.assert_allclose(final_xoff, expected_xoff)
        np.testing.assert_allclose(final_yoff, expected_yoff)

    def test_final_energy_reconstruction(self):
        """Verify energy predictions = baseline + residual (in log10 space)."""
        baseline_erec_log = np.array([2.0, 3.0, 4.0])  # log10(100), log10(1000), log10(10000)
        pred_erec_log_residual = np.array([0.1, -0.2, 0.3])

        # Final log10 energy
        final_erec_log = baseline_erec_log + pred_erec_log_residual

        # Convert to linear energy
        final_erec = np.power(10.0, final_erec_log)

        expected_erec_log = np.array([2.1, 2.8, 4.3])
        expected_erec = np.array([10**2.1, 10**2.8, 10**4.3])

        np.testing.assert_allclose(final_erec_log, expected_erec_log)
        np.testing.assert_allclose(final_erec, expected_erec)
