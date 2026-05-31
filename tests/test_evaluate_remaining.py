"""Tests for remaining evaluate.py branches (ze_bin, SHAP by energy)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from eventdisplay_ml import evaluate


@pytest.fixture
def proba_model():
    model = MagicMock()
    model.predict_proba.return_value = np.column_stack([np.full(20, 0.3), np.full(20, 0.7)])
    return model


def test_evaluation_efficiency_returns_empty_dict_without_ze_bin(proba_model):
    x_test = pd.DataFrame({"f1": np.ones(20)})
    y_test = pd.Series([1, 0] * 10)
    efficiency_all, efficiencies_by_zenith = evaluate.evaluation_efficiency(
        "test", proba_model, x_test, y_test, return_by_zenith=True
    )
    assert len(efficiency_all) == 101
    assert efficiencies_by_zenith == {}


def test_evaluation_efficiency_groups_by_zenith(proba_model):
    x_test = pd.DataFrame({"f1": np.ones(20), "ze_bin": [0] * 10 + [1] * 10})
    y_test = pd.Series([1, 0] * 10)
    _, efficiencies_by_zenith = evaluate.evaluation_efficiency(
        "test", proba_model, x_test, y_test, return_by_zenith=True
    )
    assert sorted(efficiencies_by_zenith) == [0, 1]
    assert len(efficiencies_by_zenith[0]) == 101


def test_feature_importance_logs_joint_message(caplog):
    model = SimpleNamespace(feature_importances_=np.array([0.2, 0.8]))
    with caplog.at_level("INFO"):
        evaluate.feature_importance(model, ["f1", "f2"], ["a", "b"], name="xgboost")
    assert "JOINT importance" in caplog.text


def test_evaluate_regression_model_runs_shap_per_energy(monkeypatch):
    rng = np.random.default_rng(5)
    x_train = pd.DataFrame({"f1": rng.standard_normal(40), "f2": rng.standard_normal(40)})
    y_train = pd.DataFrame({"target": 0.5 * x_train["f1"] - 0.2 * x_train["f2"]})
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(x_train, y_train)
    x_test = x_train.iloc[:20].reset_index(drop=True)
    y_test = y_train.iloc[:20].reset_index(drop=True)
    y_pred = y_test.copy()
    df = pd.DataFrame(
        {
            "MCe0": np.linspace(-1.0, 1.0, 20),
            "Xoff_weighted_bdt": np.zeros(20),
            "Yoff_weighted_bdt": np.zeros(20),
            "ErecS": np.ones(20),
        }
    )
    called = {"value": False}

    def fake_resolution(*_args, **_kwargs):
        called["value"] = True

    class FakeCutResult:
        def __init__(self, codes):
            self.cat = type("CatAccessor", (), {"codes": np.asarray(codes, dtype=int)})()

    monkeypatch.setattr(evaluate, "calculate_resolution", fake_resolution)
    monkeypatch.setattr(
        evaluate, "shap_feature_importance", lambda *_: {"target": np.array([0.2, 0.8])}
    )
    monkeypatch.setattr(evaluate.pd, "cut", lambda values, **_: FakeCutResult([0] * 10 + [1] * 10))

    result = evaluate.evaluate_regression_model(
        model,
        x_test,
        y_pred,
        y_test,
        df,
        ["f1", "f2"],
        y_test,
        "xgboost",
        shap_per_energy=True,
    )

    assert "target" in result
    assert called["value"] is True


def test_shap_feature_importance_by_energy_handles_real_xgboost(monkeypatch):
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame({"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)})
    y_train = pd.DataFrame({"target": x_train["f1"] + 0.1 * x_train["f2"]})
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(x_train, y_train)
    x_test = x_train.iloc[:25].reset_index(drop=True)
    y_test = y_train.iloc[:25].reset_index(drop=True)
    df = pd.DataFrame({"MCe0": np.linspace(-1.0, 1.0, 25)})

    class FakeCutResult:
        def __init__(self, codes):
            self.cat = type("CatAccessor", (), {"codes": np.asarray(codes, dtype=int)})()

    monkeypatch.setattr(
        evaluate.pd, "cut", lambda values, **_: FakeCutResult(([0] * 12) + ([1] * 13))
    )
    evaluate.shap_feature_importance_by_energy(model, x_test, df, y_test, ["target"])
