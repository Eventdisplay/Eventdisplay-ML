"""Production-grade guard tests for the stereo-regression path.

These tests exist specifically to catch regressions introduced by changes to
the classification pipeline that shares infrastructure with the regression path.
Every test targets a concrete, observable contract.  If any of these fail after
a classification refactoring, the regression path has been broken.

Coverage areas
--------------
1. ``train_regression`` – feature/target separation, standardisation, sample
   weights, train/test split, determinism, empty-input guard.
2. ``apply_regression_models`` – residual inversion, feature reordering, ErecS
   handling, high-multiplicity dispatch, output shape, NaN propagation.
3. ``load_regression_models`` – artifact structure, missing-key handling,
   mutual independence from classification loader.
4. ``_apply_model`` dispatch – stereo tree names, float32 dtypes, 10^x energy
   conversion, row-count preservation.
5. Internal helpers – ``_feature_array``, ``_predict_unscaled_chunked``,
   ``_sample_eval_indices``, ``_regression_sample_weights``,
   ``_log_energy_bin_counts_from_arrays``.
6. ``process_file_chunked`` – stereo path only, chunk/index invariants.
7. Feature schema – target list, analysis-type tag, pointing-offset exclusion.
8. ``_output_tree`` – stereo branch set, no classification branches present.
"""

import math
from unittest.mock import MagicMock, patch

import awkward as ak
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from eventdisplay_ml import data_processing, models
from eventdisplay_ml.models import (
    _feature_array,
    _log_energy_bin_counts_from_arrays,
    _output_tree,
    _predict_unscaled_chunked,
    _regression_sample_weights,
    _sample_eval_indices,
)

# ---------------------------------------------------------------------------
# Minimal helpers shared across tests
# ---------------------------------------------------------------------------

TARGETS = ["Xoff_residual", "Yoff_residual", "E_residual"]


def _make_regression_df(n=300, seed=0):
    """Return a deterministic regression-ready DataFrame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Xoff_residual": rng.normal(0.1, 0.4, n),
            "Yoff_residual": rng.normal(-0.2, 0.5, n),
            "E_residual": rng.normal(0.05, 0.2, n),
            "ErecS": np.logspace(0, 2, n),
            "DispNImages": rng.choice([2, 3, 4], n),
            "Xoff_weighted_bdt": rng.normal(0, 0.5, n),
            "Yoff_weighted_bdt": rng.normal(0, 0.5, n),
            "feature_A": rng.uniform(-1, 1, n),
            "feature_B": rng.uniform(0, 10, n),
        }
    )
    return df


def _make_base_config(seed=7, n_estimators=5):
    return {
        "targets": TARGETS,
        "train_test_fraction": 0.5,
        "random_state": seed,
        "eval_max_events": 0,
        "models": {
            "xgboost": {
                "hyper_parameters": {
                    "n_estimators": n_estimators,
                    "max_depth": 2,
                    "random_state": seed,
                }
            }
        },
    }


class _ZeroResidualModel:
    """Predict zero residuals for all events – simplest possible stand-in."""

    best_iteration = 0
    best_score = 0.0

    def fit(self, *args, **kwargs):
        return self

    def predict(self, x):
        return np.zeros((len(x), 3), dtype=np.float32)


class _RecordingModel:
    """Record every ``fit`` call for later inspection."""

    best_iteration = 0
    best_score = 0.0

    def __init__(self):
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, x, y, **kw):
        self.fit_calls.append(
            {"x": np.array(x, copy=True), "y": np.array(y, copy=True), "kwargs": kw}
        )
        return self

    def predict(self, x):
        self.predict_calls.append(np.array(x, copy=True))
        return np.zeros((len(x), 3), dtype=np.float32)


class _ColumnEchoModel:
    """Return the first three feature columns as residuals so order is detectable."""

    def predict(self, x_df):
        arr = x_df.to_numpy(dtype=float) if hasattr(x_df, "to_numpy") else np.asarray(x_df)
        return arr[:, :3]


# ===========================================================================
# 1. train_regression
# ===========================================================================


class TestTrainRegressionFeatureTargetSeparation:
    """Targets must be excluded from feature columns, never passed to XGBoost."""

    def test_targets_absent_from_feature_list(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert not any(t in result["features"] for t in TARGETS), (
            "One or more targets leaked into the feature list"
        )

    def test_feature_list_matches_columns_minus_targets(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        expected = [c for c in df.columns if c not in set(TARGETS)]
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert result["features"] == expected

    def test_xgboost_x_train_has_no_target_column(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        x_cols_seen = rec.fit_calls[0]["x"].shape[1]
        expected_n_features = len(df.columns) - len(TARGETS)
        assert x_cols_seen == expected_n_features


class TestTrainRegressionTargetStandardisation:
    """Scalers must be computed from training data only and stored correctly."""

    def test_target_mean_and_std_stored_in_result(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert "target_mean" in result
        assert "target_std" in result
        assert set(result["target_mean"]) == set(TARGETS)
        assert set(result["target_std"]) == set(TARGETS)

    def test_scalers_come_from_train_split_not_full_dataset(self):
        df = _make_regression_df(n=200, seed=1)
        cfg = _make_base_config(seed=1)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        train_idx, _ = train_test_split(np.arange(len(df)), train_size=0.5, random_state=1)
        for t in TARGETS:
            expected_mean = df.iloc[train_idx][t].mean()
            expected_std = df.iloc[train_idx][t].std()
            assert result["target_mean"][t] == pytest.approx(expected_mean, abs=1e-9)
            assert result["target_std"][t] == pytest.approx(expected_std, abs=1e-9)

    def test_y_train_passed_to_xgboost_is_standardised(self):
        df = _make_regression_df(n=200, seed=2)
        cfg = _make_base_config(seed=2)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        y_passed = rec.fit_calls[0]["y"]  # shape (n_train, 3)
        for col_idx, t in enumerate(TARGETS):
            col = y_passed[:, col_idx]
            assert abs(col.mean()) < 0.15, f"Target {t}: scaled mean too far from 0"
            assert abs(col.std() - 1.0) < 0.15, f"Target {t}: scaled std too far from 1"

    def test_target_std_is_never_zero(self):
        # All three targets must have non-zero std in training.
        df = _make_regression_df(n=300, seed=3)
        cfg = _make_base_config(seed=3)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        for t in TARGETS:
            assert result["target_std"][t] > 0.0, f"Target {t} has zero std"

    def test_target_order_in_scaler_dicts_matches_targets_list(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert list(result["target_mean"].keys()) == TARGETS
        assert list(result["target_std"].keys()) == TARGETS


class TestTrainRegressionSampleWeights:
    """Sample weights must cover the training set, be finite, and be positive."""

    def test_sample_weights_shape_matches_train_size(self):
        n = 400
        df = _make_regression_df(n=n, seed=4)
        cfg = _make_base_config(seed=4)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        weights = rec.fit_calls[0]["kwargs"]["sample_weight"]
        assert weights is None or len(weights) == n // 2

    def test_sample_weights_are_finite_and_positive(self):
        df = _make_regression_df(n=400, seed=5)
        cfg = _make_base_config(seed=5)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        w = rec.fit_calls[0]["kwargs"]["sample_weight"]
        if w is not None:
            assert np.all(np.isfinite(w)), "Sample weights contain non-finite values"
            assert np.all(w >= 0), "Sample weights contain negative values"

    def test_eval_weights_shape_matches_eval_set(self):
        df = _make_regression_df(n=400, seed=6)
        cfg = _make_base_config(seed=6)
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        kw = rec.fit_calls[0]["kwargs"]
        if kw.get("sample_weight_eval_set") is not None:
            eval_w = kw["sample_weight_eval_set"]
            assert isinstance(eval_w, list)
            assert len(eval_w) == 1
            assert eval_w[0] is None or len(eval_w[0]) == len(kw.get("eval_set", [[]])[0][0])


class TestTrainRegressionSplitDeterminism:
    """The same random_state must always yield the same feature array passed to fit."""

    def test_identical_seed_gives_identical_x_train(self):
        df = _make_regression_df(n=200, seed=10)
        cfg_a = _make_base_config(seed=42)
        cfg_b = _make_base_config(seed=42)
        rec_a, rec_b = _RecordingModel(), _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec_a),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df.copy(), cfg_a)
        with (
            patch("xgboost.XGBRegressor", return_value=rec_b),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df.copy(), cfg_b)

        np.testing.assert_array_equal(rec_a.fit_calls[0]["x"], rec_b.fit_calls[0]["x"])

    def test_different_seed_gives_different_x_train(self):
        df = _make_regression_df(n=200, seed=10)
        rec_a, rec_b = _RecordingModel(), _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec_a),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df.copy(), _make_base_config(seed=1))
        with (
            patch("xgboost.XGBRegressor", return_value=rec_b),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df.copy(), _make_base_config(seed=2))

        # Different seeds should produce different splits (very likely with 200 rows)
        assert not np.array_equal(rec_a.fit_calls[0]["x"], rec_b.fit_calls[0]["x"])


class TestTrainRegressionEdgeCases:
    def test_empty_dataframe_returns_none(self):
        result = models.train_regression(pd.DataFrame(), _make_base_config())
        assert result is None

    def test_result_contains_features_key_at_top_level(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert "features" in result
        # Must also be stored inside the per-model dict
        assert "features" in result["models"]["xgboost"]

    def test_model_object_stored_in_models_dict(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            result = models.train_regression(df, cfg)

        assert result["models"]["xgboost"]["model"] is rec

    def test_x_train_dtype_is_float32(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        assert rec.fit_calls[0]["x"].dtype == np.float32

    def test_y_train_dtype_is_float32(self):
        df = _make_regression_df()
        cfg = _make_base_config()
        rec = _RecordingModel()
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            models.train_regression(df, cfg)

        assert rec.fit_calls[0]["y"].dtype == np.float32


# ===========================================================================
# 2. apply_regression_models
# ===========================================================================


class TestApplyRegressionResidualInversion:
    """Residuals must be unscaled and added to the DispBDT baseline."""

    def _build_apply_config(self, model, features, target_mean, target_std):
        return {
            "models": {"xgboost": {"model": model, "features": features}},
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def _flat_df(self, xoff_bdt, yoff_bdt, erec_s):
        return pd.DataFrame(
            {
                "Xoff_weighted_bdt": xoff_bdt,
                "Yoff_weighted_bdt": yoff_bdt,
                "ErecS": erec_s,
            }
        )

    def test_zero_residual_model_returns_mean_plus_baseline(self, monkeypatch):
        """When scaled residual = 0, final = mean + baseline."""
        flat = self._flat_df([10.0], [20.0], [100.0])
        target_mean = {"Xoff_residual": 3.0, "Yoff_residual": -5.0, "E_residual": 0.4}
        target_std = {"Xoff_residual": 2.0, "Yoff_residual": 1.5, "E_residual": 0.1}

        class _ZeroModel:
            def predict(self, x):
                return np.zeros((len(x), 3))

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = self._build_apply_config(_ZeroModel(), flat.columns.tolist(), target_mean, target_std)
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [2]}), cfg
        )

        # xoff: 0*2 + 3 + 10 = 13
        np.testing.assert_allclose(pred_xoff, [13.0], atol=1e-7)
        # yoff: 0*1.5 + (-5) + 20 = 15
        np.testing.assert_allclose(pred_yoff, [15.0], atol=1e-7)
        # log10(100)=2; E: 0*0.1 + 0.4 + 2 = 2.4
        np.testing.assert_allclose(pred_e, [2.4], atol=1e-7)

    def test_unit_scaled_residual_inverts_correctly(self, monkeypatch):
        """Scaled residual=1 => physical residual = std + mean."""
        flat = self._flat_df([0.0], [0.0], [10.0])
        target_mean = {"Xoff_residual": 1.0, "Yoff_residual": 2.0, "E_residual": 0.5}
        target_std = {"Xoff_residual": 2.0, "Yoff_residual": 3.0, "E_residual": 0.1}

        class _OnesModel:
            def predict(self, x):
                return np.ones((len(x), 3))

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = self._build_apply_config(_OnesModel(), flat.columns.tolist(), target_mean, target_std)
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [2]}), cfg
        )

        # xoff: 1*2 + 1 + 0 = 3
        np.testing.assert_allclose(pred_xoff, [3.0], atol=1e-7)
        # yoff: 1*3 + 2 + 0 = 5
        np.testing.assert_allclose(pred_yoff, [5.0], atol=1e-7)
        # log10(10)=1; E: 1*0.1 + 0.5 + 1 = 1.6
        np.testing.assert_allclose(pred_e, [1.6], atol=1e-7)

    def test_negative_scaled_residual(self, monkeypatch):
        flat = self._flat_df([5.0], [-5.0], [1000.0])
        target_mean = {"Xoff_residual": 0.0, "Yoff_residual": 0.0, "E_residual": 0.0}
        target_std = {"Xoff_residual": 1.0, "Yoff_residual": 1.0, "E_residual": 1.0}

        class _MinusOneModel:
            def predict(self, x):
                return -1.0 * np.ones((len(x), 3))

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = self._build_apply_config(
            _MinusOneModel(), flat.columns.tolist(), target_mean, target_std
        )
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [2]}), cfg
        )

        np.testing.assert_allclose(pred_xoff, [4.0], atol=1e-7)
        np.testing.assert_allclose(pred_yoff, [-6.0], atol=1e-7)
        np.testing.assert_allclose(pred_e, [2.0], atol=1e-7)  # log10(1000)-1 = 2

    def test_multiple_events_independent(self, monkeypatch):
        n = 5
        xoff_bdt = np.arange(n, dtype=float)
        yoff_bdt = np.arange(n, dtype=float) * -2
        erec_s = 10.0 ** np.arange(n, dtype=float)
        flat = self._flat_df(xoff_bdt, yoff_bdt, erec_s)

        class _ConstantModel:
            def predict(self, x):
                # Always predicts scaled residual [0.5, -0.5, 1.0]
                return np.tile([0.5, -0.5, 1.0], (len(x), 1))

        target_mean = {"Xoff_residual": 1.0, "Yoff_residual": -1.0, "E_residual": 0.2}
        target_std = {"Xoff_residual": 2.0, "Yoff_residual": 4.0, "E_residual": 0.5}

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = self._build_apply_config(
            _ConstantModel(), flat.columns.tolist(), target_mean, target_std
        )
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(
            pd.DataFrame({"DispNImages": np.full(n, 2)}), cfg
        )

        # physical residual_xoff = 0.5*2 + 1 = 2.0; final = baseline + 2.0
        np.testing.assert_allclose(pred_xoff, xoff_bdt + 2.0, atol=1e-7)
        # physical residual_yoff = -0.5*4 + (-1) = -3.0
        np.testing.assert_allclose(pred_yoff, yoff_bdt - 3.0, atol=1e-7)
        # physical residual_e = 1.0*0.5 + 0.2 = 0.7
        expected_e = np.arange(n, dtype=float) + 0.7  # log10(10^i) = i, then +0.7
        np.testing.assert_allclose(pred_e, expected_e, atol=1e-7)


class TestApplyRegressionFeatureReordering:
    """The model must receive features in the persisted order, not the input order."""

    def test_shuffled_input_columns_produce_correct_physics(self, monkeypatch, tmp_path):
        feature_order = [
            "feat_alpha",
            "feat_beta",
            "ErecS",
            "Xoff_weighted_bdt",
            "Yoff_weighted_bdt",
        ]
        target_mean = {"Xoff_residual": 0.0, "Yoff_residual": 0.0, "E_residual": 0.0}
        target_std = {"Xoff_residual": 1.0, "Yoff_residual": 1.0, "E_residual": 1.0}

        class _FirstColModel:
            """Returns first column value as all three residuals, so order is detectable."""

            def predict(self, x):
                col0 = x.iloc[:, 0].to_numpy(dtype=float) if hasattr(x, "iloc") else x[:, 0]
                return np.column_stack([col0, col0, col0])

        # Build flattened frame in non-persisted order
        flat_shuffled = pd.DataFrame(
            {
                "Yoff_weighted_bdt": [0.0, 0.0],
                "ErecS": [100.0, 100.0],
                "feat_beta": [99.0, 99.0],  # must NOT be first after reindex
                "feat_alpha": [7.0, 7.0],  # must be first after reindex → residual = 7.0
                "Xoff_weighted_bdt": [5.0, 5.0],
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat_shuffled)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = {
            "models": {"xgboost": {"model": _FirstColModel(), "features": feature_order}},
            "target_mean": target_mean,
            "target_std": target_std,
        }
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [2, 2]}), cfg
        )

        # After reindex to persisted order, column 0 is feat_alpha = 7.0.
        # xoff = 7.0 + Xoff_weighted_bdt(5.0) = 12.0
        np.testing.assert_allclose(pred_xoff, [12.0, 12.0], atol=1e-7)

    def test_extra_input_columns_not_forwarded_to_model(self, monkeypatch):
        """Columns present in input but not in persisted features must be dropped."""
        feature_order = ["Xoff_weighted_bdt", "Yoff_weighted_bdt", "ErecS"]
        received_shapes = []

        class _ShapeRecordModel:
            def predict(self, x):
                received_shapes.append(x.shape if hasattr(x, "shape") else (len(x),))
                return np.zeros((len(x), 3))

        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [1.0],
                "Yoff_weighted_bdt": [2.0],
                "ErecS": [10.0],
                "extra_col_should_be_dropped": [999.0],
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        cfg = {
            "models": {"xgboost": {"model": _ShapeRecordModel(), "features": feature_order}},
            "target_mean": dict.fromkeys(TARGETS, 0.0),
            "target_std": dict.fromkeys(TARGETS, 1.0),
        }
        models.apply_regression_models(pd.DataFrame({"DispNImages": [2]}), cfg)

        n_cols = received_shapes[0][1]
        assert n_cols == len(feature_order), (
            f"Model received {n_cols} columns but should have received {len(feature_order)}"
        )


class TestApplyRegressionErecSHandling:
    """Invalid ErecS must propagate to NaN energy output without corrupting direction."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

    def _run(self, monkeypatch, erec_s_values, scaled_preds=None):
        n = len(erec_s_values)
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": np.zeros(n),
                "Yoff_weighted_bdt": np.zeros(n),
                "ErecS": erec_s_values,
            }
        )
        if scaled_preds is None:
            scaled_preds = np.zeros((n, 3))

        class _FixedModel:
            def predict(self, x):
                return scaled_preds

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = {
            "models": {"xgboost": {"model": _FixedModel(), "features": flat.columns.tolist()}},
            "target_mean": dict.fromkeys(TARGETS, 0.0),
            "target_std": dict.fromkeys(TARGETS, 1.0),
        }
        return models.apply_regression_models(pd.DataFrame({"DispNImages": np.full(n, 2)}), cfg)

    def test_negative_erecs_produce_nan_energy(self, monkeypatch):
        _, _, pred_e = self._run(monkeypatch, [-5.0, 100.0])
        assert np.isnan(pred_e[0])
        assert not np.isnan(pred_e[1])

    def test_zero_erecs_produces_nan_energy(self, monkeypatch):
        _, _, pred_e = self._run(monkeypatch, [0.0])
        assert np.isnan(pred_e[0])

    def test_nan_erecs_produces_nan_energy(self, monkeypatch):
        _, _, pred_e = self._run(monkeypatch, [np.nan])
        assert np.isnan(pred_e[0])

    def test_invalid_erecs_do_not_corrupt_direction(self, monkeypatch):
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [10.0, 20.0],
                "Yoff_weighted_bdt": [30.0, 40.0],
                "ErecS": [np.nan, 100.0],
            }
        )

        class _ConstModel:
            def predict(self, x):
                return np.tile([1.0, 2.0, 3.0], (len(x), 1))

        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = {
            "models": {"xgboost": {"model": _ConstModel(), "features": flat.columns.tolist()}},
            "target_mean": dict.fromkeys(TARGETS, 0.0),
            "target_std": dict.fromkeys(TARGETS, 1.0),
        }
        pred_xoff, pred_yoff, _ = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [2, 2]}), cfg
        )

        # Direction must still be valid for the invalid-energy event
        assert np.isfinite(pred_xoff[0])
        assert np.isfinite(pred_yoff[0])

    def test_all_erecs_valid_no_nan_energy(self, monkeypatch):
        _, _, pred_e = self._run(monkeypatch, [1.0, 10.0, 100.0, 1000.0])
        assert all(np.isfinite(pred_e))

    def test_output_length_equals_input_length(self, monkeypatch):
        n = 97
        rng = np.random.default_rng(0)
        erec_s = np.where(rng.random(n) > 0.3, rng.uniform(1, 1000, n), np.nan)
        pred_xoff, pred_yoff, pred_e = self._run(monkeypatch, erec_s)
        assert len(pred_xoff) == n
        assert len(pred_yoff) == n
        assert len(pred_e) == n


class TestApplyRegressionHighMultiplicity:
    """High-multiplicity events must route to the dedicated model when configured."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

    def _make_two_model_cfg(self, flat_df, low_pred, high_pred):
        class _SelectableModel:
            def __init__(self, val):
                self._val = val

            def predict(self, x):
                return np.tile(self._val, (len(x), 1))

        zero_mean = dict.fromkeys(TARGETS, 0.0)
        unit_std = dict.fromkeys(TARGETS, 1.0)
        return {
            "models": {
                "xgboost": {"model": _SelectableModel(low_pred), "features": flat_df.columns}
            },
            "models_high_multiplicity": {
                "xgboost": {
                    "model": _SelectableModel(high_pred),
                    "features": flat_df.columns,
                }
            },
            "target_mean": zero_mean,
            "target_std": unit_std,
            "target_mean_high_multiplicity": zero_mean,
            "target_std_high_multiplicity": unit_std,
        }

    def test_multiplicity_2_uses_primary_model(self, monkeypatch):
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [0.0],
                "Yoff_weighted_bdt": [0.0],
                "ErecS": [10.0],
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = self._make_two_model_cfg(flat, [5.0, 5.0, 5.0], [9.0, 9.0, 9.0])

        pred_xoff, _, _ = models.apply_regression_models(pd.DataFrame({"DispNImages": [2]}), cfg)
        np.testing.assert_allclose(pred_xoff, [5.0])  # low model

    def test_multiplicity_3_uses_high_model(self, monkeypatch):
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [0.0],
                "Yoff_weighted_bdt": [0.0],
                "ErecS": [10.0],
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = self._make_two_model_cfg(flat, [5.0, 5.0, 5.0], [9.0, 9.0, 9.0])

        pred_xoff, _, _ = models.apply_regression_models(pd.DataFrame({"DispNImages": [3]}), cfg)
        np.testing.assert_allclose(pred_xoff, [9.0])  # high model

    def test_multiplicity_1_returns_nan(self, monkeypatch):
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": [0.0],
                "Yoff_weighted_bdt": [0.0],
                "ErecS": [10.0],
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = self._make_two_model_cfg(flat, [5.0, 5.0, 5.0], [9.0, 9.0, 9.0])

        pred_xoff, _, _ = models.apply_regression_models(pd.DataFrame({"DispNImages": [1]}), cfg)
        assert np.isnan(pred_xoff[0])

    def test_mixed_multiplicity_routes_correctly(self, monkeypatch):
        n = 4
        flat = pd.DataFrame(
            {
                "Xoff_weighted_bdt": np.zeros(n),
                "Yoff_weighted_bdt": np.zeros(n),
                "ErecS": np.ones(n) * 10.0,
            }
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        cfg = self._make_two_model_cfg(flat, [2.0, 2.0, 2.0], [7.0, 7.0, 7.0])

        # mult: [1, 2, 3, 4]
        pred_xoff, _, _ = models.apply_regression_models(
            pd.DataFrame({"DispNImages": [1, 2, 3, 4]}), cfg
        )
        assert np.isnan(pred_xoff[0])  # mult=1: unrouted → NaN
        assert pred_xoff[1] == pytest.approx(2.0)  # low model
        assert pred_xoff[2] == pytest.approx(7.0)  # high model
        assert pred_xoff[3] == pytest.approx(7.0)  # high model


class TestApplyRegressionMissingParams:
    def test_missing_target_mean_raises_value_error(self, monkeypatch):
        flat = pd.DataFrame(
            {"Xoff_weighted_bdt": [1.0], "Yoff_weighted_bdt": [2.0], "ErecS": [10.0]}
        )
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        class _M:
            def predict(self, x):
                return np.zeros((len(x), 3))

        cfg = {
            "models": {"xgboost": {"model": _M(), "features": flat.columns.tolist()}},
            # no target_mean or target_std
        }
        with pytest.raises(ValueError, match="target standardization"):
            models.apply_regression_models(pd.DataFrame({"DispNImages": [2]}), cfg)


# ===========================================================================
# 3. load_regression_models
# ===========================================================================


class TestLoadRegressionModels:
    """Artifact loading must be independent of the classification loader."""

    def _write_artifact(self, tmp_path, feature_list=None, include_scalers=True):
        if feature_list is None:
            feature_list = ["f1", "f2", "ErecS", "Xoff_weighted_bdt", "Yoff_weighted_bdt"]
        payload = {
            "models": {"xgboost": {"model": _ZeroResidualModel()}},
            "features": feature_list,
        }
        if include_scalers:
            payload["target_mean"] = dict.fromkeys(TARGETS, 0.1)
            payload["target_std"] = dict.fromkeys(TARGETS, 1.0)
        path = tmp_path / "stereo_test.joblib.gz"
        joblib.dump(payload, path)
        return path

    def test_load_returns_model_and_parameters(self, tmp_path):
        self._write_artifact(tmp_path)
        loaded_models, par = models.load_regression_models(str(tmp_path / "stereo_test"), "xgboost")
        assert "xgboost" in loaded_models
        assert "model" in loaded_models["xgboost"]
        assert "features" in loaded_models["xgboost"]

    def test_load_preserves_feature_order(self, tmp_path):
        feat = ["z_col", "a_col", "ErecS", "Xoff_weighted_bdt", "Yoff_weighted_bdt"]
        self._write_artifact(tmp_path, feature_list=feat)
        loaded_models, _ = models.load_regression_models(str(tmp_path / "stereo_test"), "xgboost")
        assert loaded_models["xgboost"]["features"] == feat

    def test_load_restores_target_scalers(self, tmp_path):
        self._write_artifact(tmp_path)
        _, par = models.load_regression_models(str(tmp_path / "stereo_test"), "xgboost")
        assert "target_mean" in par
        assert "target_std" in par
        for t in TARGETS:
            assert t in par["target_mean"]
            assert t in par["target_std"]

    def test_load_missing_file_raises_error(self, tmp_path):
        with pytest.raises((FileNotFoundError, Exception)):
            models.load_regression_models(str(tmp_path / "nonexistent"), "xgboost")

    def test_load_regression_does_not_touch_classification_loader(self, tmp_path, monkeypatch):
        """load_regression_models must never call load_classification_models."""
        self._write_artifact(tmp_path)
        called = []
        monkeypatch.setattr(
            models, "load_classification_models", lambda *a, **k: called.append(True)
        )
        models.load_regression_models(str(tmp_path / "stereo_test"), "xgboost")
        assert called == [], "load_classification_models was called during regression load"

    def test_load_models_dispatcher_routes_stereo_correctly(self, tmp_path, monkeypatch):
        """The top-level load_models dispatcher must route 'stereo_analysis' to regression."""
        expected = ({"xgboost": {"model": None, "features": []}}, {})
        monkeypatch.setattr(models, "load_regression_models", lambda *a: expected)
        result = models.load_models("stereo_analysis", str(tmp_path / "model"), "xgboost")
        assert result == expected

    def test_load_models_dispatcher_raises_for_unknown_type(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            models.load_models("unknown_type", str(tmp_path / "model"), "xgboost")


# ===========================================================================
# 4. _apply_model dispatch
# ===========================================================================


class TestApplyModelStereoDispatch:
    """_apply_model must write Dir_Xoff, Dir_Yoff, Dir_Erec with correct types."""

    def _tree_and_apply(self, monkeypatch, pred_xoff, pred_yoff, pred_log_e):
        tree = MagicMock()
        monkeypatch.setattr(
            models,
            "apply_regression_models",
            lambda *a: (
                np.asarray(pred_xoff, dtype=np.float64),
                np.asarray(pred_yoff, dtype=np.float64),
                np.asarray(pred_log_e, dtype=np.float64),
            ),
        )
        models._apply_model(
            "stereo_analysis",
            pd.DataFrame({"event": range(len(pred_xoff))}),
            {},
            tree,
        )
        return tree.extend.call_args.args[0]

    def test_stereo_payload_keys(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [1.0], [2.0], [3.0])
        assert set(payload) == {"Dir_Xoff", "Dir_Yoff", "Dir_Erec"}

    def test_stereo_payload_dtype_float32(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        for key, arr in payload.items():
            assert arr.dtype == np.float32, f"Branch {key} has dtype {arr.dtype}, expected float32"

    def test_energy_converted_from_log10(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [0.0], [0.0], [2.0])
        assert payload["Dir_Erec"][0] == pytest.approx(100.0, rel=1e-5)

    def test_energy_log10_minus1_gives_01(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [0.0], [0.0], [-1.0])
        assert payload["Dir_Erec"][0] == pytest.approx(0.1, rel=1e-5)

    def test_nan_energy_propagates_to_output(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [1.0, 2.0], [3.0, 4.0], [np.nan, 1.0])
        assert np.isnan(payload["Dir_Erec"][0])
        assert np.isfinite(payload["Dir_Erec"][1])

    def test_nan_direction_propagates(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [np.nan], [1.0], [1.0])
        assert np.isnan(payload["Dir_Xoff"][0])

    def test_row_count_preserved(self, monkeypatch):
        n = 50
        payload = self._tree_and_apply(
            monkeypatch,
            np.ones(n),
            np.zeros(n),
            np.full(n, 2.0),
        )
        assert len(payload["Dir_Xoff"]) == n
        assert len(payload["Dir_Yoff"]) == n
        assert len(payload["Dir_Erec"]) == n

    def test_no_classification_branches_written(self, monkeypatch):
        payload = self._tree_and_apply(monkeypatch, [1.0], [1.0], [1.0])
        classification_keys = {"Gamma_Prediction", "Is_Gamma_80", "Is_Gamma_70"}
        assert not (set(payload) & classification_keys)

    def test_unknown_analysis_type_raises(self):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            models._apply_model("bad_type", pd.DataFrame({"x": [1]}), {}, MagicMock())


# ===========================================================================
# 5. Internal helpers
# ===========================================================================


class TestFeatureArrayHelper:
    def test_returns_float32_array(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        arr = _feature_array(df, np.array([0, 2]), ["a", "b"])
        assert arr.dtype == np.float32

    def test_selects_correct_rows(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0], "y": [1.0, 2.0, 3.0, 4.0]})
        arr = _feature_array(df, np.array([1, 3]), ["x", "y"])
        np.testing.assert_array_equal(arr, [[20.0, 2.0], [40.0, 4.0]])

    def test_selects_correct_columns(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        arr = _feature_array(df, np.array([0]), ["b", "a"])
        assert arr[0, 0] == pytest.approx(2.0)
        assert arr[0, 1] == pytest.approx(1.0)

    def test_empty_row_index(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        arr = _feature_array(df, np.array([], dtype=int), ["a", "b"])
        assert arr.shape == (0, 2)


class TestPredictUnscaledChunked:
    def _run(self, n=10, chunk_size=None):
        df = pd.DataFrame(
            {
                "f1": np.arange(n, dtype=float),
                "f2": np.arange(n, dtype=float) * 2,
                "Xoff_residual": np.zeros(n),
                "Yoff_residual": np.zeros(n),
                "E_residual": np.zeros(n),
            }
        )
        targets = TARGETS
        x_cols = ["f1", "f2"]
        y_mean = pd.Series({"Xoff_residual": 1.0, "Yoff_residual": 2.0, "E_residual": 3.0})
        y_std = pd.Series({"Xoff_residual": 2.0, "Yoff_residual": 0.5, "E_residual": 1.0})

        class _IdentModel:
            """Returns scaled input columns as residuals."""

            def predict(self, x):
                arr = np.asarray(x, dtype=np.float32)
                return np.column_stack([arr[:, 0], arr[:, 1], arr[:, 0] - arr[:, 1]])

        row_indices = np.arange(n)
        return _predict_unscaled_chunked(
            _IdentModel(), df, row_indices, x_cols, y_mean, y_std, targets, chunk_size
        )

    def test_output_shape(self):
        result = self._run(n=15)
        assert result.shape == (15, 3)

    def test_output_columns_are_targets(self):
        result = self._run(n=5)
        assert list(result.columns) == TARGETS

    def test_chunked_vs_single_pass_identical(self):
        full = self._run(n=20, chunk_size=None)
        chunked = self._run(n=20, chunk_size=4)
        pd.testing.assert_frame_equal(full, chunked)

    def test_unscaling_is_applied(self):
        # For event 0: f1=0, f2=0 → scaled_pred=[0,0,0] → unscaled = 0*std+mean
        result = self._run(n=1)
        np.testing.assert_allclose(result.iloc[0]["Xoff_residual"], 0.0 * 2.0 + 1.0)
        np.testing.assert_allclose(result.iloc[0]["Yoff_residual"], 0.0 * 0.5 + 2.0)

    def test_index_matches_df_index(self):
        n = 8
        df = pd.DataFrame(
            {"f1": np.arange(n, dtype=float), "f2": np.zeros(n)}, index=np.arange(100, 100 + n)
        )
        y_mean = pd.Series(dict.fromkeys(TARGETS, 0.0))
        y_std = pd.Series(dict.fromkeys(TARGETS, 1.0))

        class _Zero:
            def predict(self, x):
                return np.zeros((len(x), 3))

        result = _predict_unscaled_chunked(
            _Zero(), df, np.arange(n), ["f1", "f2"], y_mean, y_std, TARGETS, None
        )
        assert list(result.index) == list(df.index)


class TestSampleEvalIndices:
    def test_no_cap_returns_all_indices(self):
        idx = np.arange(100)
        result = _sample_eval_indices(idx, max_events=None, random_state=0)
        np.testing.assert_array_equal(result, idx)

    def test_zero_cap_returns_all_indices(self):
        idx = np.arange(50)
        result = _sample_eval_indices(idx, max_events=0, random_state=0)
        np.testing.assert_array_equal(result, idx)

    def test_cap_larger_than_pool_returns_all(self):
        idx = np.arange(30)
        result = _sample_eval_indices(idx, max_events=100, random_state=0)
        np.testing.assert_array_equal(result, idx)

    def test_cap_limits_output_size(self):
        idx = np.arange(200)
        result = _sample_eval_indices(idx, max_events=50, random_state=0)
        assert len(result) == 50

    def test_sampled_indices_are_subset_of_input(self):
        idx = np.arange(500)
        result = _sample_eval_indices(idx, max_events=100, random_state=42)
        assert set(result).issubset(set(idx))

    def test_same_seed_gives_same_subset(self):
        idx = np.arange(200)
        r1 = _sample_eval_indices(idx, max_events=60, random_state=7)
        r2 = _sample_eval_indices(idx, max_events=60, random_state=7)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_give_different_subsets(self):
        idx = np.arange(200)
        r1 = _sample_eval_indices(idx, max_events=60, random_state=1)
        r2 = _sample_eval_indices(idx, max_events=60, random_state=2)
        assert not np.array_equal(r1, r2)


class TestRegressionSampleWeights:
    def _build_inputs(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        erec_s = np.logspace(0, 2.5, n)
        e_residual = rng.normal(0, 0.1, n)
        disp_nimages = rng.choice([2, 3, 4], n)
        return erec_s, e_residual, disp_nimages

    def test_weights_are_finite_and_positive(self):
        erec_s, e_residual, disp_nimages = self._build_inputs()
        result = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)
        weights = result[2]
        assert np.all(np.isfinite(weights)), "Non-finite weights found"
        assert np.all(weights >= 0), "Negative weights found"

    def test_weights_are_capped(self):
        from eventdisplay_ml.models import _MAX_REGRESSION_SAMPLE_WEIGHT

        erec_s, e_residual, disp_nimages = self._build_inputs()
        weights = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)[2]
        assert np.all(weights <= _MAX_REGRESSION_SAMPLE_WEIGHT + 1e-6), (
            "Weights exceed the hard cap"
        )

    def test_weights_length_matches_input(self):
        n = 150
        erec_s, e_residual, disp_nimages = self._build_inputs(n=n)
        weights = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)[2]
        assert len(weights) == n

    def test_higher_multiplicity_gets_higher_weight(self):
        """Events with more images must receive proportionally higher weights."""
        # Two events with same energy, different multiplicity
        erec_s = np.array([100.0, 100.0])
        e_residual = np.array([0.0, 0.0])
        disp_nimages = np.array([2, 4])
        # _regression_sample_weights directly to probe the multiplicity component
        bins = np.linspace(-2, 2.5, 10)
        energy_bin_weights = np.ones(9)
        mult_mean_sq = float(np.mean(np.square(disp_nimages, dtype=np.float64)))
        weights, _ = _regression_sample_weights(
            erec_s,
            e_residual,
            disp_nimages,
            bins=bins,
            energy_bin_weights=energy_bin_weights,
            multiplicity_mean_square=mult_mean_sq,
            max_weight=50.0,
        )
        # w_mult = n_tel^2 / mean_sq, so mult=4 must exceed mult=2
        assert weights[1] > weights[0], (
            "Multiplicity-4 event should have higher weight than multiplicity-2"
        )

    def test_invalid_erecs_get_zero_weight(self):
        erec_s = np.array([-1.0, 0.0, np.nan, 100.0])
        e_residual = np.array([0.0, 0.0, 0.0, 0.0])
        disp_nimages = np.array([2, 2, 2, 2])
        bins = np.linspace(-2, 2.5, 10)
        energy_bin_weights = np.ones(9)
        mult_mean_sq = 4.0
        weights, _ = _regression_sample_weights(
            erec_s,
            e_residual,
            disp_nimages,
            bins=bins,
            energy_bin_weights=energy_bin_weights,
            multiplicity_mean_square=mult_mean_sq,
            max_weight=50.0,
        )
        assert weights[0] == 0.0, "Negative ErecS should yield weight=0"
        assert weights[1] == 0.0, "Zero ErecS should yield weight=0"
        assert weights[2] == 0.0, "NaN ErecS should yield weight=0"
        assert weights[3] > 0.0, "Valid ErecS must yield positive weight"

    def test_normalization_scale_reuse_gives_same_weights(self):
        erec_s, e_residual, disp_nimages = self._build_inputs(n=100)
        bins = np.linspace(-2, 2.5, 10)
        energy_bin_weights = np.ones(9)
        mult_mean_sq = float(np.mean(np.square(disp_nimages, dtype=np.float64)))

        w1, norm_scale = _regression_sample_weights(
            erec_s,
            e_residual,
            disp_nimages,
            bins=bins,
            energy_bin_weights=energy_bin_weights,
            multiplicity_mean_square=mult_mean_sq,
            max_weight=50.0,
        )
        w2, _ = _regression_sample_weights(
            erec_s,
            e_residual,
            disp_nimages,
            bins=bins,
            energy_bin_weights=energy_bin_weights,
            multiplicity_mean_square=mult_mean_sq,
            max_weight=50.0,
            normalization_scale=norm_scale,
        )
        np.testing.assert_array_equal(w1, w2)

    def test_all_invalid_erecs_raises(self):
        # When every event has an invalid ErecS, no weights can be computed and
        # the function must raise ValueError rather than silently returning zeros.
        n = 20
        erec_s = np.full(n, np.nan)
        e_residual = np.zeros(n)
        disp_nimages = np.full(n, 2)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(ValueError):
                _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)


class TestLogEnergyBinCountsFromArrays:
    def test_returns_three_tuple(self):
        erec_s = np.logspace(0, 2, 100)
        e_residual = np.zeros(100)
        disp_nimages = np.full(100, 2)
        result = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)
        assert len(result) == 3

    def test_weight_config_returned_when_requested(self):
        erec_s = np.logspace(0, 2, 200)
        e_residual = np.zeros(200)
        disp_nimages = np.full(200, 2)
        result = _log_energy_bin_counts_from_arrays(
            erec_s, e_residual, disp_nimages, return_weight_config=True
        )
        assert len(result) == 4
        weight_config = result[3]
        assert "bins" in weight_config
        assert "energy_bin_weights" in weight_config
        assert "multiplicity_mean_square" in weight_config
        assert "max_weight" in weight_config
        assert "normalization_scale" in weight_config

    def test_counts_dict_has_pd_interval_keys(self):
        erec_s = np.logspace(0, 2, 100)
        e_residual = np.zeros(100)
        disp_nimages = np.full(100, 2)
        _, counts, _ = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)
        # All keys should be pd.Interval
        assert all(isinstance(k, pd.Interval) for k in counts.keys())

    def test_bins_span_expected_energy_range(self):
        erec_s = np.logspace(0, 2, 100)
        e_residual = np.zeros(100)
        disp_nimages = np.full(100, 2)
        bins, _, _ = _log_energy_bin_counts_from_arrays(erec_s, e_residual, disp_nimages)
        from eventdisplay_ml.models import _EVAL_LOG_E_MAX, _EVAL_LOG_E_MIN

        assert bins[0] == _EVAL_LOG_E_MIN
        assert bins[-1] == _EVAL_LOG_E_MAX


# ===========================================================================
# 6. _output_tree stereo structure
# ===========================================================================


class TestOutputTreeStereo:
    """Stereo output tree must have exactly three float32 branches."""

    def test_stereo_creates_correct_tree(self):
        root_file = MagicMock()
        mock_tree = MagicMock()
        root_file.mktree.return_value = mock_tree

        result = _output_tree("stereo_analysis", root_file)

        root_file.mktree.assert_called_once()
        call_args = root_file.mktree.call_args
        tree_name = call_args.args[0]
        branches = call_args.args[1]

        assert tree_name == "StereoAnalysis"
        assert set(branches.keys()) == {"Dir_Xoff", "Dir_Yoff", "Dir_Erec"}
        assert all(v is np.float32 for v in branches.values())
        assert result is mock_tree

    def test_stereo_tree_has_no_classification_branches(self):
        root_file = MagicMock()
        root_file.mktree.return_value = MagicMock()
        _output_tree("stereo_analysis", root_file)
        branches = root_file.mktree.call_args.args[1]
        assert "Gamma_Prediction" not in branches
        assert not any(k.startswith("Is_Gamma") for k in branches)

    def test_unknown_analysis_type_raises(self):
        with pytest.raises(ValueError, match="Unknown analysis_type"):
            _output_tree("bad_type", MagicMock())


# ===========================================================================
# 7. Feature schema – regression-specific invariants
# ===========================================================================


class TestRegressionFeatureSchema:
    """These invariants must survive classification refactoring."""

    def test_target_list_exact_order(self):
        from eventdisplay_ml import features

        assert features.target_features("stereo_analysis") == [
            "Xoff_residual",
            "Yoff_residual",
            "E_residual",
        ]

    def test_training_features_include_mc_truth(self):
        from eventdisplay_ml import features

        train = features.features("stereo_analysis", training=True)
        assert "MCxoff" in train
        assert "MCyoff" in train
        assert "MCe0" in train

    def test_inference_features_exclude_mc_truth(self):
        from eventdisplay_ml import features

        infer = features.features("stereo_analysis", training=False)
        assert "MCxoff" not in infer
        assert "MCyoff" not in infer
        assert "MCe0" not in infer

    def test_pointing_offsets_excluded_from_stereo(self):
        from eventdisplay_ml import features

        excluded = features.excluded_features("stereo_analysis", ntel=2)
        assert "fpointing_dx_0" in excluded
        assert "fpointing_dy_0" in excluded
        assert "fpointing_dx_1" in excluded
        assert "fpointing_dy_1" in excluded

    def test_analysis_type_tag_is_stereo_not_classification(self):
        """Passing 'stereo_analysis' to features() must never yield classification-only columns."""
        from eventdisplay_ml import features

        infer = features.features("stereo_analysis", training=False)
        classification_only = {"ze_bin", "Gamma_Prediction", "Is_Gamma"}
        assert not any(c in infer for c in classification_only)


# ===========================================================================
# 8. process_file_chunked – stereo path invariants
# ===========================================================================


class TestProcessFileChunkedStereo:
    """Streaming must reset chunk indices and obey max_events for the stereo path."""

    def _setup_mocks(self, monkeypatch, chunks, max_events=None):
        input_root = MagicMock()
        input_root.__enter__.return_value = {"data": MagicMock()}
        input_root.__exit__.return_value = False
        output_root = MagicMock()
        output_root.__enter__.return_value = output_root
        output_root.__exit__.return_value = False
        tree = MagicMock()
        applied = []

        monkeypatch.setattr(models.uproot, "open", lambda *a: input_root)
        monkeypatch.setattr(models.uproot, "recreate", lambda *a: output_root)
        monkeypatch.setattr(models.uproot, "iterate", lambda *a, **k: chunks)
        monkeypatch.setattr(
            models.data_processing, "read_telescope_config", lambda *a: {"max_tel_id": 3}
        )
        monkeypatch.setattr(
            models.data_processing, "_resolve_branch_aliases", lambda *a: (["ErecS"], {})
        )
        monkeypatch.setattr(models.data_processing, "_ensure_fpointing_fields", lambda c: c)
        monkeypatch.setattr(models.features, "features", lambda *a, **k: ["ErecS"])
        monkeypatch.setattr(models, "_output_tree", lambda *a: tree)
        monkeypatch.setattr(
            models,
            "_apply_model",
            lambda at, chunk, *a: applied.append((at, chunk.copy())),
        )
        return applied

    def test_chunk_indices_reset_to_zero_based(self, monkeypatch):
        chunks = [ak.Array([{"ErecS": 1.0}, {"ErecS": 2.0}])]
        applied = self._setup_mocks(monkeypatch, chunks)
        models.process_file_chunked(
            "stereo_analysis",
            {"input_file": "in.root", "output_file": "out.root", "chunk_size": 10},
        )
        assert applied[0][1].index.tolist() == [0, 1]

    def test_max_events_limits_total_processed(self, monkeypatch):
        chunks = [
            ak.Array([{"ErecS": 1.0}, {"ErecS": 2.0}, {"ErecS": 3.0}]),
            ak.Array([{"ErecS": 4.0}, {"ErecS": 5.0}]),
        ]
        applied = self._setup_mocks(monkeypatch, chunks, max_events=4)
        models.process_file_chunked(
            "stereo_analysis",
            {"input_file": "in.root", "output_file": "out.root", "max_events": 4, "chunk_size": 3},
        )
        total = sum(len(chunk) for _, chunk in applied)
        assert total == 4

    def test_analysis_type_passed_as_stereo(self, monkeypatch):
        chunks = [ak.Array([{"ErecS": 1.0}])]
        applied = self._setup_mocks(monkeypatch, chunks)
        models.process_file_chunked(
            "stereo_analysis",
            {"input_file": "in.root", "output_file": "out.root"},
        )
        assert all(at == "stereo_analysis" for at, _ in applied)

    def test_empty_chunk_is_skipped(self, monkeypatch):
        chunks = [ak.Array([]), ak.Array([{"ErecS": 1.0}])]
        applied = self._setup_mocks(monkeypatch, chunks)
        models.process_file_chunked(
            "stereo_analysis",
            {"input_file": "in.root", "output_file": "out.root"},
        )
        assert len(applied) == 1  # empty chunk was skipped


# ===========================================================================
# 9. End-to-end regression: train → persist → load → apply round-trip
# ===========================================================================


class TestRegressionEndToEnd:
    """Train a tiny stand-in regressor (via a patched ``xgboost.XGBRegressor``), save it,
    load it, and verify predictions round-trip correctly through the production stack.
    """

    def test_full_round_trip_prediction_shape_and_dtype(self, tmp_path, monkeypatch):
        """Train → joblib dump → load_regression_models → apply → correct output shape."""
        rng = np.random.default_rng(99)
        n = 200
        df = pd.DataFrame(
            {
                "Xoff_residual": rng.normal(0, 0.3, n),
                "Yoff_residual": rng.normal(0, 0.4, n),
                "E_residual": rng.normal(0, 0.15, n),
                "ErecS": np.logspace(0, 2, n),
                "DispNImages": rng.choice([2, 3, 4], n),
                "Xoff_weighted_bdt": rng.normal(0, 0.5, n),
                "Yoff_weighted_bdt": rng.normal(0, 0.5, n),
                "feat_A": rng.uniform(-1, 1, n),
            }
        )
        rec = _RecordingModel()
        cfg = {
            "targets": TARGETS,
            "train_test_fraction": 0.5,
            "random_state": 99,
            "eval_max_events": 0,
            "models": {"xgboost": {"hyper_parameters": {}}},
        }
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            trained = models.train_regression(df, cfg)

        # Manually store the recording model (train_regression puts it in cfg)
        model_prefix = tmp_path / "rtt_model"
        artifact_path = model_prefix.with_suffix(".joblib.gz")
        joblib.dump(trained, artifact_path)

        loaded_models_dict, loaded_params = models.load_regression_models(
            str(model_prefix), "xgboost"
        )
        assert "target_mean" in loaded_params
        assert "target_std" in loaded_params

        apply_df = df.head(10).copy()
        flat = apply_df[[c for c in df.columns if c not in set(TARGETS)]]
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        apply_cfg = {"models": loaded_models_dict, **loaded_params}
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(apply_df, apply_cfg)

        assert len(pred_xoff) == 10
        assert len(pred_yoff) == 10
        assert len(pred_e) == 10
        assert pred_xoff.dtype in (np.float32, np.float64)

    def test_scalers_from_trained_model_invert_apply_predictions(self, tmp_path, monkeypatch):
        """Scalers stored during training must round-trip: apply recovers finite predictions."""
        rng = np.random.default_rng(77)
        n = 200
        df = pd.DataFrame(
            {
                "Xoff_residual": rng.normal(0, 0.3, n),
                "Yoff_residual": rng.normal(0, 0.4, n),
                "E_residual": rng.normal(0, 0.15, n),
                "ErecS": np.ones(n),  # log10(1)=0 as baseline
                "DispNImages": np.full(n, 2),
                "Xoff_weighted_bdt": np.zeros(n),
                "Yoff_weighted_bdt": np.zeros(n),
                "feat": rng.uniform(0, 1, n),
            }
        )
        rec = _RecordingModel()
        cfg = {
            "targets": TARGETS,
            "train_test_fraction": 0.5,
            "random_state": 77,
            "eval_max_events": 0,
            "models": {"xgboost": {"hyper_parameters": {}}},
        }
        with (
            patch("xgboost.XGBRegressor", return_value=rec),
            patch("eventdisplay_ml.models.evaluate_regression_model", return_value={}),
        ):
            trained = models.train_regression(df, cfg)

        # Scalers must be well-defined (finite mean, positive std)
        for t in TARGETS:
            assert math.isfinite(trained["target_mean"][t])
            assert trained["target_std"][t] > 0

        # Predictions on 10-event apply frame should all be finite
        apply_df = df.head(10).copy()
        flat = apply_df[[c for c in df.columns if c not in set(TARGETS)]]
        monkeypatch.setattr(models, "flatten_feature_data", lambda *a, **k: flat)
        monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *a: None)

        apply_cfg = {
            "models": trained["models"],
            **{k: trained[k] for k in ("target_mean", "target_std")},
        }
        pred_xoff, pred_yoff, pred_e = models.apply_regression_models(apply_df, apply_cfg)

        assert all(np.isfinite(pred_xoff))
        assert all(np.isfinite(pred_yoff))
        assert all(np.isfinite(pred_e))
