"""Unit tests for training variables selection utilities."""

import eventdisplay_ml.training_variables


def test_xgb_per_telescope_training_variables():
    """Ensure per-telescope training variables are provided as a list and include expected keys."""
    variables = eventdisplay_ml.training_variables.xgb_per_telescope_training_variables()
    assert isinstance(variables, list)
    assert "Disp_T" in variables
    assert "R_core" in variables


def test_xgb_regression_training_variables():
    """Ensure array-level training variables include array metadata fields."""
    variables = eventdisplay_ml.training_variables.xgb_regression_training_variables()
    assert isinstance(variables, list)
    assert "DispNImages" in variables
    assert "EmissionHeight" in variables


def test_xgb_all_regression_training_variables():
    """Ensure combined training variables include per-telescope and array-level fields."""
    variables = eventdisplay_ml.training_variables.xgb_all_regression_training_variables()
    assert isinstance(variables, list)
    assert "Disp_T" in variables
    assert "R_core" in variables
    assert "DispNImages" in variables
    assert "EmissionHeight" in variables


def test_xgb_all_classification_training_variables():
    """Ensure combined classification variables exclude energy fields and include expected keys."""
    variables = eventdisplay_ml.training_variables.xgb_all_classification_training_variables()
    assert isinstance(variables, list)
    # Energy fields should be excluded
    assert "E" not in variables
    assert "ES" not in variables
    # Per-telescope variables
    assert "Disp_T" in variables
    assert "R_core" in variables
    # Classification variables
    assert "MSCW" in variables
    assert "MSCL" in variables
    assert "EmissionHeight" in variables


def test_xgb_all_regression_training_variables_content():
    """Test that xgb_all_regression_training_variables returns correct combined variables."""
    variables = eventdisplay_ml.training_variables.xgb_all_regression_training_variables()
    # Should include all per-telescope and regression variables
    per_telescope = eventdisplay_ml.training_variables.xgb_per_telescope_training_variables()
    regression = eventdisplay_ml.training_variables.xgb_regression_training_variables()
    for var in per_telescope:
        assert var in variables
    for var in regression:
        assert var in variables
    # Length should be sum of both lists
    assert len(variables) == len(per_telescope) + len(regression)
