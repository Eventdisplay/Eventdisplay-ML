"""Unit tests for training variables selection utilities."""

import eventdisplay_ml.features


def test_telescope_features():
    """Ensure per-telescope training variables are provided as a list and include expected keys."""
    variables = eventdisplay_ml.features.telescope_features()
    assert isinstance(variables, list)
    assert "Disp_T" in variables
    assert "R_core" in variables


def test__regression_features():
    """Ensure array-level training variables include array metadata fields."""
    variables = eventdisplay_ml.features._regression_features()
    assert isinstance(variables, list)
    assert "DispNImages" in variables
    assert "EmissionHeight" in variables


def test__regression_features():
    """Ensure combined training variables include per-telescope and array-level fields."""
    variables = eventdisplay_ml.features._regression_features()
    assert isinstance(variables, list)
    assert "Disp_T" in variables
    assert "R_core" in variables
    assert "DispNImages" in variables
    assert "EmissionHeight" in variables


def test__classification_features():
    """Ensure combined classification variables exclude energy fields and include expected keys."""
    variables = eventdisplay_ml.features._classification_features()
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
