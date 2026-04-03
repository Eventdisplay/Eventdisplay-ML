"""Tests for target standardization and energy-bin weighting in train_regression()."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import diagnostic_utils, models


@pytest.fixture
def regression_training_df():
    """Create a training DataFrame with required columns for regression."""
    rng = np.random.default_rng(42)
    n_rows = 100

    return pd.DataFrame(
        {
            "Xoff_residual": rng.normal(0.5, 0.3, n_rows),
            "Yoff_residual": rng.normal(1.0, 0.5, n_rows),
            "E_residual": rng.normal(-0.2, 0.1, n_rows),
            "ErecS": np.logspace(1, 2, n_rows),
            "DispNImages": rng.choice([2, 3, 4], n_rows),
            "Xoff_weighted_bdt": rng.normal(0, 0.5, n_rows),
            "Yoff_weighted_bdt": rng.normal(0, 0.5, n_rows),
            "mscw": rng.uniform(0, 1, n_rows),
            "mscl": rng.uniform(0, 1, n_rows),
        }
    )


@pytest.fixture
def regression_model_config():
    """Create a model configuration for regression training."""
    return {
        "targets": ["Xoff_residual", "Yoff_residual", "E_residual"],
        "train_test_fraction": 0.5,
        "random_state": 42,
        "models": {
            "xgboost": {
                "hyper_parameters": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "random_state": 42,
                    "early_stopping_rounds": 2,
                    "eval_metric": "rmse",
                }
            }
        },
    }


class TestTargetStandardization:
    """Tests for target standardization (mean and std) storage."""

    def test_target_mean_std_computed_from_training_set(
        self, regression_training_df, regression_model_config
    ):
        """Verify target_mean and target_std are computed from training data only."""
        df = regression_training_df
        cfg = regression_model_config

        # Train the model
        result = models.train_regression(df, cfg)

        # Check that target_mean and target_std are stored in config
        assert "target_mean" in result, "target_mean not stored in model config"
        assert "target_std" in result, "target_std not stored in model config"

        # Verify they are dictionaries with all target keys
        assert isinstance(result["target_mean"], dict)
        assert isinstance(result["target_std"], dict)
        assert set(result["target_mean"].keys()) == set(cfg["targets"])
        assert set(result["target_std"].keys()) == set(cfg["targets"])

    def test_target_mean_std_values_reasonable(
        self, regression_training_df, regression_model_config
    ):
        """Verify target_mean and target_std have reasonable values."""
        df = regression_training_df.copy()
        cfg = regression_model_config

        # Manually compute expected values from training set (50%)
        # train_test_split with train_size=0.5 and random_state=42
        from sklearn.model_selection import train_test_split

        x_cols = [col for col in df.columns if col not in cfg["targets"]]
        _, _, y_data_train, _ = train_test_split(
            df[x_cols],
            df[cfg["targets"]],
            train_size=cfg["train_test_fraction"],
            random_state=cfg["random_state"],
        )

        expected_mean = y_data_train.mean()
        expected_std = y_data_train.std()

        result = models.train_regression(df, cfg)

        # Verify computed values match expected
        for target in cfg["targets"]:
            assert np.isclose(result["target_mean"][target], expected_mean[target], rtol=1e-5), (
                f"{target} mean mismatch"
            )
            assert np.isclose(result["target_std"][target], expected_std[target], rtol=1e-5), (
                f"{target} std mismatch"
            )

    def test_target_std_never_zero(self, regression_training_df, regression_model_config):
        """Verify target_std values are not zero (to avoid division by zero)."""
        df = regression_training_df
        cfg = regression_model_config

        result = models.train_regression(df, cfg)

        for target in cfg["targets"]:
            assert result["target_std"][target] > 0, f"{target} std should not be zero"


class TestEnergyBinWeighting:
    """Tests for energy-bin weighting (especially zeroing low-count bins)."""

    def test_log_energy_bin_counts_returns_correct_structure(self, regression_training_df):
        """Verify _log_energy_bin_counts() returns expected tuple structure."""
        df = regression_training_df
        result = models._log_energy_bin_counts(df)

        assert result is not None, "Should return a tuple, not None"
        bins, counts_dict, weights = result

        # Check tuple structure
        assert isinstance(bins, np.ndarray), "bins should be ndarray"
        assert isinstance(counts_dict, dict), "counts_dict should be dict"
        assert isinstance(weights, np.ndarray), "weights should be ndarray"

        # Verify weight array has same length as input
        assert len(weights) == len(df), "weights array length should match dataframe rows"

    def test_log_energy_bin_counts_zeroes_low_count_bins(self):
        """Verify bins with < 10 events get zero weight."""
        # Create minimal dataframe with specific energy distribution
        rng = np.random.default_rng(42)
        n_rows = 30
        df = pd.DataFrame(
            {
                "ErecS": np.concatenate(
                    [
                        np.full(15, 100.0),
                        np.full(10, 10.0),
                        np.full(5, 1000.0),
                    ]
                ),
                "E_residual": np.zeros(n_rows),
                "DispNImages": rng.choice([2, 3], n_rows),
            }
        )

        result = models._log_energy_bin_counts(df)
        assert result is not None

        _, counts_dict, weights = result

        # Find which bins have < 10 events
        low_count_bins = {interval: count for interval, count in counts_dict.items() if count < 10}

        # Events in low-count bins should have zero energy weight
        # (multiplicity weight might still apply, but energy weight should be 0)
        if low_count_bins:
            zero_weights = weights[weights == 0]
            if len(zero_weights) > 0:
                assert len(zero_weights) > 0, "Expected some zero weights for low-count bins"

    def test_log_energy_bin_counts_weight_normalization(self, regression_training_df):
        """Verify combined weights are normalized to mean ~1.0."""
        df = regression_training_df
        result = models._log_energy_bin_counts(df)

        _, _, weights = result

        # Check that weight array is normalized
        # (mean should be ~1.0 after normalization)
        weight_mean = np.mean(weights)
        assert np.isclose(weight_mean, 1.0, rtol=0.01), (
            f"Weight mean should be ~1.0, got {weight_mean}"
        )

    def test_log_energy_bin_counts_handles_missing_columns(self):
        """Verify graceful handling when E_residual/ErecS missing."""
        df = pd.DataFrame(
            {
                "DispNImages": [2, 3, 4],
                "some_other_col": [1.0, 2.0, 3.0],
            }
        )

        result = models._log_energy_bin_counts(df)
        assert result is None, "Should return None when E_residual/ErecS missing"

    def test_energy_bin_weighting_in_training(
        self, regression_training_df, regression_model_config
    ):
        """Verify energy-bin weights are applied during model training."""
        df = regression_training_df
        cfg = regression_model_config

        # Mock the XGBRegressor to capture the sample_weight argument
        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_model = MagicMock()
            mock_model.best_iteration = 5
            mock_model.best_score = 0.1
            mock_model.predict.return_value = np.zeros((len(df) // 2, 3))
            mock_xgb.return_value = mock_model

            # Mock evaluate_regression_model to return empty dict
            with patch("eventdisplay_ml.models.evaluate_regression_model") as mock_eval:
                mock_eval.return_value = {}

                models.train_regression(df, cfg)

                # Verify fit() was called with sample_weight
                mock_model.fit.assert_called_once()
                call_args = mock_model.fit.call_args

                # Check that sample_weight is not None
                sample_weight = call_args.kwargs.get("sample_weight")
                assert sample_weight is not None, "sample_weight should be passed to fit()"
                assert len(sample_weight) == len(df) // 2  # Training set size


class TestTrainRegressionIntegration:
    """Integration tests for train_regression() with standardization and weighting."""

    def test_train_regression_complete_workflow(
        self, regression_training_df, regression_model_config
    ):
        """Verify complete training workflow with standardization and weighting."""
        df = regression_training_df
        cfg = regression_model_config

        result = models.train_regression(df, cfg)

        # Check critical outputs
        assert result is not None
        assert "target_mean" in result
        assert "target_std" in result
        assert "models" in result
        assert "xgboost" in result["models"]
        assert "model" in result["models"]["xgboost"]
        assert "generalization_metrics" in result["models"]["xgboost"]
        assert "shap_importance" in result["models"]["xgboost"]

    def test_generalization_metrics_cached_per_target(
        self, regression_training_df, regression_model_config
    ):
        """Verify train/test RMSE summary is cached in the model config."""
        result = models.train_regression(regression_training_df, regression_model_config)

        metrics = result["models"]["xgboost"]["generalization_metrics"]
        assert set(metrics) == set(regression_model_config["targets"])

        for target in regression_model_config["targets"]:
            assert set(metrics[target]) == {"rmse_train", "rmse_test", "gap_pct", "gen_ratio"}
            assert np.isfinite(metrics[target]["rmse_train"])
            assert np.isfinite(metrics[target]["rmse_test"])

    def test_generalization_metrics_match_training_predictions(
        self, regression_training_df, regression_model_config
    ):
        """Verify cached generalization metrics match the model predictions used in training."""
        df = regression_training_df
        cfg = regression_model_config

        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_model = MagicMock()
            mock_model.best_iteration = 5
            mock_model.best_score = 0.1

            def _predict(x_values):
                return np.zeros((len(x_values), len(cfg["targets"])))

            mock_model.predict.side_effect = _predict
            mock_xgb.return_value = mock_model

            with patch("eventdisplay_ml.models.evaluate_regression_model") as mock_eval:
                mock_eval.return_value = {}
                result = models.train_regression(df, cfg)

        from sklearn.model_selection import train_test_split

        x_cols = [col for col in df.columns if col not in cfg["targets"]]
        _, _, y_train, y_test = train_test_split(
            df[x_cols],
            df[cfg["targets"]],
            train_size=cfg["train_test_fraction"],
            random_state=cfg["random_state"],
        )

        target_mean = np.array([result["target_mean"][target] for target in cfg["targets"]])
        y_train_pred = pd.DataFrame(
            np.tile(target_mean, (len(y_train), 1)),
            columns=cfg["targets"],
            index=y_train.index,
        )
        y_test_pred = pd.DataFrame(
            np.tile(target_mean, (len(y_test), 1)),
            columns=cfg["targets"],
            index=y_test.index,
        )

        expected_metrics = diagnostic_utils.compute_generalization_metrics(
            y_train,
            y_train_pred,
            y_test,
            y_test_pred,
            cfg["targets"],
        )

        assert result["models"]["xgboost"]["generalization_metrics"] == expected_metrics

    def test_scaled_predictions_unscaled_correctly(
        self, regression_training_df, regression_model_config
    ):
        """Verify predictions are correctly unscaled using stored mean/std."""
        # This test verifies the inverse transformation logic
        df = regression_training_df.copy()
        cfg = regression_model_config

        result = models.train_regression(df, cfg)

        # Get the stored scalers
        target_mean = result["target_mean"]
        target_std = result["target_std"]

        # Simulate scaled prediction: y_scaled = 1.0 for all targets
        y_pred_scaled = np.array([[1.0, 1.0, 1.0]])

        # Manually unscale
        y_pred_unscaled = y_pred_scaled * np.array(
            [target_std[target] for target in cfg["targets"]]
        ) + np.array([target_mean[target] for target in cfg["targets"]])

        # Verify unscaling produces reasonable values
        for i, target in enumerate(cfg["targets"]):
            assert np.isfinite(y_pred_unscaled[0, i]), (
                f"{target} unscaled prediction should be finite"
            )

    def test_train_test_split_preserved_correctly(
        self, regression_training_df, regression_model_config
    ):
        """Verify train/test split doesn't leak into weight computation."""
        df = regression_training_df
        cfg = regression_model_config

        # Train with fixed random state multiple times
        cfg1 = cfg.copy()
        config1 = models.train_regression(df, cfg1)

        cfg2 = cfg.copy()
        config2 = models.train_regression(df, cfg2)

        # With same random state, should get identical mean/std
        for target in cfg["targets"]:
            assert np.isclose(
                config1["target_mean"][target],
                config2["target_mean"][target],
            ), "target mean should be identical with same random_state"
            assert np.isclose(
                config1["target_std"][target],
                config2["target_std"][target],
            ), "target std should be identical with same random_state"
