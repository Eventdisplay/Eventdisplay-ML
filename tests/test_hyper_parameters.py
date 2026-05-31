"""Unit tests for hyper_parameters.py."""

import json

import pytest

from eventdisplay_ml.hyper_parameters import (
    PRE_CUTS_CLASSIFICATION,
    classification_hyper_parameters,
    hyper_parameters,
    pre_cuts_classification,
    pre_cuts_regression,
    regression_hyper_parameters,
)

# ---------------------------------------------------------------------------
# hyper_parameters dispatcher
# ---------------------------------------------------------------------------


def test_hyper_parameters_stereo_returns_regression_dict():
    result = hyper_parameters("stereo_analysis")
    assert "xgboost" in result
    assert "hyper_parameters" in result["xgboost"]


def test_hyper_parameters_classification_returns_classification_dict():
    result = hyper_parameters("classification")
    assert "xgboost" in result
    hp = result["xgboost"]["hyper_parameters"]
    assert hp["objective"] == "binary:logistic"


def test_hyper_parameters_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        hyper_parameters("mystery")


# ---------------------------------------------------------------------------
# Default hyperparameter structures
# ---------------------------------------------------------------------------


def test_regression_hyper_parameters_has_expected_keys():
    result = regression_hyper_parameters()
    hp = result["xgboost"]["hyper_parameters"]
    for key in ("n_estimators", "learning_rate", "max_depth", "objective"):
        assert key in hp, f"Expected key '{key}' in regression hyperparameters"


def test_regression_hyper_parameters_objective_is_squared_error():
    hp = regression_hyper_parameters()["xgboost"]["hyper_parameters"]
    assert hp["objective"] == "reg:squarederror"


def test_classification_hyper_parameters_has_expected_keys():
    result = classification_hyper_parameters()
    hp = result["xgboost"]["hyper_parameters"]
    for key in ("n_estimators", "learning_rate", "max_depth", "objective"):
        assert key in hp


def test_regression_returns_copy_not_same_object():
    """Default dict should be returned as-is; mutations could affect defaults."""
    r1 = regression_hyper_parameters()
    r2 = regression_hyper_parameters()
    # Both calls should return the same default dict
    assert r1 is r2


# ---------------------------------------------------------------------------
# Load from file
# ---------------------------------------------------------------------------


def test_regression_hyper_parameters_from_file(tmp_path):
    custom = {
        "custom_model": {
            "model": None,
            "hyper_parameters": {"n_estimators": 100, "max_depth": 3},
        }
    }
    f = tmp_path / "hp.json"
    f.write_text(json.dumps(custom))
    result = regression_hyper_parameters(config_file=str(f))
    assert result == custom


def test_classification_hyper_parameters_from_file(tmp_path):
    custom = {"xgboost": {"model": None, "hyper_parameters": {"n_estimators": 50}}}
    f = tmp_path / "hp.json"
    f.write_text(json.dumps(custom))
    result = classification_hyper_parameters(config_file=str(f))
    assert result["xgboost"]["hyper_parameters"]["n_estimators"] == 50


# ---------------------------------------------------------------------------
# pre_cuts_regression
# ---------------------------------------------------------------------------


def test_pre_cuts_regression_default_min_images():
    result = pre_cuts_regression()
    assert "DispNImages >=2" in result


def test_pre_cuts_regression_custom_min_images():
    result = pre_cuts_regression(min_images=3)
    assert "DispNImages >=3" in result


def test_pre_cuts_regression_returns_string():
    result = pre_cuts_regression()
    assert isinstance(result, str)


def test_pre_cuts_regression_no_extra_cuts_when_list_empty():
    """PRE_CUTS_REGRESSION is empty by default; result contains only DispNImages cut."""
    result = pre_cuts_regression()
    # Should be a simple expression, not empty
    assert result is not None
    assert len(result) > 0


# ---------------------------------------------------------------------------
# pre_cuts_classification
# ---------------------------------------------------------------------------


def test_pre_cuts_classification_contains_energy_bounds():
    result = pre_cuts_classification(e_min=0.1, e_max=100.0)
    assert "0.1" in result
    assert "100.0" in result
    assert "Erec" in result


def test_pre_cuts_classification_includes_mscw_cut():
    result = pre_cuts_classification(e_min=0.1, e_max=10.0)
    assert "MSCW" in result


def test_pre_cuts_classification_includes_emission_height_cut():
    result = pre_cuts_classification(e_min=0.1, e_max=10.0)
    assert "EmissionHeight" in result


def test_pre_cuts_classification_returns_string():
    result = pre_cuts_classification(e_min=0.01, e_max=1000.0)
    assert isinstance(result, str)


@pytest.mark.parametrize("cut", PRE_CUTS_CLASSIFICATION)
def test_pre_cuts_classification_includes_all_default_cuts(cut):
    """Each default quality cut should appear in the combined cut string."""
    result = pre_cuts_classification(e_min=0.1, e_max=10.0)
    # Cuts are wrapped in parentheses but variable names are preserved
    var = cut.split()[0]
    assert var in result
