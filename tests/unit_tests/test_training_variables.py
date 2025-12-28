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
