"""Tests for target standardization and energy-bin weighting in train_regression()."""

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml import diagnostic_utils, models, utils


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

    @pytest.mark.parametrize("seed_mode", ["missing", "none"])
    def test_training_record_contains_all_regression_seeds_and_parameters(
        self, regression_training_df, regression_model_config, seed_mode
    ):
        """Persist the effective reproducibility and XGBoost training record."""
        cfg = copy.deepcopy(regression_model_config)
        if seed_mode == "missing":
            cfg.pop("random_state")
        else:
            cfg["random_state"] = None
        cfg["models"]["xgboost"]["hyper_parameters"].pop("random_state")
        cfg["eval_max_events"] = 10
        cfg["diagnostic_max_events"] = 10

        class _CapturingRegressor:
            best_iteration = 0
            best_score = 0.0

            def __init__(self, **kwargs):
                self.params = kwargs

            def fit(self, *_args, **_kwargs):
                return self

            def predict(self, x_values):
                return np.zeros((len(x_values), len(cfg["targets"])))

            def get_params(self, deep=False):  # noqa: ARG002
                return self.params

        with (
            patch("eventdisplay_ml.models.train_test_split", wraps=train_test_split) as split_mock,
            patch(
                "eventdisplay_ml.models._sample_eval_indices",
                wraps=models._sample_eval_indices,
            ) as sample_mock,
            patch("xgboost.XGBRegressor", side_effect=_CapturingRegressor) as xgb_mock,
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(regression_training_df, cfg)

        record = result["training_parameters"]
        assert record["random_state"] == utils.DEFAULT_REGRESSION_RANDOM_STATE
        assert record["random_seeds"] == {
            "data_sampling": utils.DEFAULT_REGRESSION_RANDOM_STATE,
            "train_test_split": utils.DEFAULT_REGRESSION_RANDOM_STATE,
            "validation_sampling": utils.DEFAULT_REGRESSION_RANDOM_STATE,
            "diagnostic_sampling": utils.DEFAULT_REGRESSION_RANDOM_STATE,
            "shap_sampling": utils.DEFAULT_REGRESSION_RANDOM_STATE,
            "xgboost": {
                "xgboost": {"random_state": utils.DEFAULT_REGRESSION_RANDOM_STATE},
            },
        }
        assert cfg["random_state"] == utils.DEFAULT_REGRESSION_RANDOM_STATE
        assert split_mock.call_args.kwargs["random_state"] == utils.DEFAULT_REGRESSION_RANDOM_STATE
        assert sample_mock.call_count >= 3
        assert all(
            call.args[2] == utils.DEFAULT_REGRESSION_RANDOM_STATE for call in sample_mock.call_args_list
        )
        assert (
            xgb_mock.call_args.kwargs["random_state"] == utils.DEFAULT_REGRESSION_RANDOM_STATE
        )
        model_record = record["models"]["xgboost"]
        assert (
            model_record["hyper_parameters"] == cfg["models"]["xgboost"]["hyper_parameters"]
        )
        assert model_record["effective_hyper_parameters"]["early_stopping_rounds"] == 2
        assert (
            model_record["effective_hyper_parameters"]["random_state"]
            == utils.DEFAULT_REGRESSION_RANDOM_STATE
        )
        assert (
            model_record["xgboost_parameters"]["random_state"]
            == utils.DEFAULT_REGRESSION_RANDOM_STATE
        )
        assert record["features"] == result["features"]
        assert record["targets"] == cfg["targets"]


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
        """Verify bins below the minimum population get zero weight."""
        n_rows = 140
        df = pd.DataFrame(
            {
                "ErecS": np.concatenate(
                    [
                        np.full(120, 0.1),
                        np.full(20, 10.0),
                    ]
                ),
                "E_residual": np.zeros(n_rows),
                "DispNImages": np.full(n_rows, 3),
            }
        )

        result = models._log_energy_bin_counts(df)
        assert result is not None

        _, counts_dict, weights = result

        assert any(
            0 < count < models._MIN_WEIGHTED_ENERGY_BIN_EVENTS for count in counts_dict.values()
        )
        assert np.all(weights[-20:] == 0)

    def test_energy_bin_weights_use_inverse_square_root(self):
        """A four-times smaller energy bin should get twice the per-event weight."""
        df = pd.DataFrame(
            {
                "ErecS": np.concatenate([np.full(400, 0.1), np.full(100, 10.0)]),
                "E_residual": np.zeros(500),
                "DispNImages": np.full(500, 3),
            }
        )

        weights = models._log_energy_bin_counts(df)[2]

        assert weights[400] / weights[0] == pytest.approx(2.0)

    def test_regression_weights_are_capped_and_normalized(self):
        """Combined weights must retain mean one without exceeding the configured cap."""
        weights, _ = models._regression_sample_weights(
            np.concatenate([np.full(10, 0.1), np.full(90, 10.0)]),
            np.zeros(100),
            np.full(100, 3),
            bins=np.array([-2.0, 0.0, 2.0]),
            energy_bin_weights=np.array([1000.0, 1.0]),
            multiplicity_mean_square=9.0,
            max_weight=50.0,
        )

        assert weights.max() <= 50.0
        assert weights.mean() == pytest.approx(1.0)

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

    def test_eval_max_events_limits_xgboost_eval_set(
        self, regression_training_df, regression_model_config
    ):
        """Verify only a bounded test subset is passed to XGBoost eval_set."""
        df = regression_training_df
        cfg = regression_model_config.copy()
        cfg["eval_max_events"] = 10

        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_model = MagicMock()
            mock_model.best_iteration = 5
            mock_model.best_score = 0.1
            mock_model.predict.side_effect = lambda x_values: np.zeros(
                (len(x_values), len(cfg["targets"]))
            )
            mock_xgb.return_value = mock_model

            with patch("eventdisplay_ml.models.evaluate_regression_model") as mock_eval:
                mock_eval.return_value = {}
                models.train_regression(df, cfg)

        eval_set = mock_model.fit.call_args.kwargs["eval_set"]
        eval_weights = mock_model.fit.call_args.kwargs["sample_weight_eval_set"]
        assert len(eval_set) == 1
        assert len(eval_set[0][0]) == 10
        assert len(eval_weights) == 1
        assert len(eval_weights[0]) == 10

    def test_early_stopping_callback_keeps_only_best_model(
        self, regression_training_df, regression_model_config
    ):
        """Verify configured early stopping uses a save-best callback."""
        df = regression_training_df
        cfg = regression_model_config

        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_model = MagicMock()
            mock_model.best_iteration = 5
            mock_model.best_score = 0.1
            mock_model.predict.side_effect = lambda x_values: np.zeros(
                (len(x_values), len(cfg["targets"]))
            )
            mock_xgb.return_value = mock_model

            with patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}):
                models.train_regression(df, cfg)

        constructor_kwargs = mock_xgb.call_args.kwargs
        assert "early_stopping_rounds" not in constructor_kwargs
        assert len(constructor_kwargs["callbacks"]) == 1
        callback = constructor_kwargs["callbacks"][0]
        assert callback.rounds == 2
        assert callback.save_best is True

    def test_diagnostic_max_events_limits_training_predictions(
        self, regression_training_df, regression_model_config
    ):
        """Verify generalization diagnostics predict only a bounded training sample."""
        df = regression_training_df
        cfg = regression_model_config.copy()
        cfg["diagnostic_max_events"] = 10

        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_model = MagicMock()
            mock_model.best_iteration = 5
            mock_model.best_score = 0.1
            prediction_lengths = []

            def _predict(x_values):
                prediction_lengths.append(len(x_values))
                return np.zeros((len(x_values), len(cfg["targets"])))

            mock_model.predict.side_effect = _predict
            mock_xgb.return_value = mock_model

            with patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}):
                models.train_regression(df, cfg)

        assert prediction_lengths[0] == 10
        assert sum(prediction_lengths[1:]) == len(df) // 2


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

    def test_memory_optimized_training_matches_reference_without_eval_cap(
        self, regression_training_df, regression_model_config
    ):
        """The array/chunking path must preserve the previous XGBoost result."""
        df = regression_training_df
        cfg = regression_model_config
        targets = cfg["targets"]
        x_cols = [column for column in df.columns if column not in targets]
        hyper_parameters = cfg["models"]["xgboost"]["hyper_parameters"]

        x_train, x_test, y_train, y_test = train_test_split(
            df[x_cols],
            df[targets],
            train_size=cfg["train_test_fraction"],
            random_state=cfg["random_state"],
        )
        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train_scaled = (y_train - y_mean) / y_std
        y_test_scaled = (y_test - y_mean) / y_std
        train_weight_result = models._log_energy_bin_counts_from_arrays(
            df.loc[y_train.index, "ErecS"].to_numpy(),
            df.loc[y_train.index, "E_residual"].to_numpy(),
            df.loc[y_train.index, "DispNImages"].to_numpy(),
            return_weight_config=True,
        )
        reference_weights = train_weight_result[2]
        reference_eval_weights, _ = models._regression_sample_weights(
            df.loc[y_test.index, "ErecS"].to_numpy(),
            df.loc[y_test.index, "E_residual"].to_numpy(),
            df.loc[y_test.index, "DispNImages"].to_numpy(),
            **train_weight_result[3],
        )

        reference_model = xgb.XGBRegressor(**hyper_parameters)
        reference_model.fit(
            x_train,
            y_train_scaled,
            sample_weight=reference_weights,
            sample_weight_eval_set=[reference_eval_weights],
            eval_set=[(x_test, y_test_scaled)],
            verbose=False,
        )

        cfg["eval_max_events"] = 0
        cfg["prediction_chunk_size"] = 13
        with patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}):
            result = models.train_regression(df, cfg)
        optimized_model = result["models"]["xgboost"]["model"]

        np.testing.assert_allclose(
            optimized_model.predict(df[x_cols].to_numpy(dtype=np.float32)),
            reference_model.predict(df[x_cols]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert optimized_model.best_iteration == reference_model.best_iteration
        assert optimized_model.best_score == pytest.approx(reference_model.best_score, abs=1e-12)
        assert result["target_mean"] == pytest.approx(y_mean.to_dict(), abs=0.0)
        assert result["target_std"] == pytest.approx(y_std.to_dict(), abs=0.0)
