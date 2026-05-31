"""Hyperparameter for classification and regression models."""

import json
import logging
from importlib.resources import files
from pathlib import Path

_logger = logging.getLogger(__name__)

_CONFIGS_DIR = files("eventdisplay_ml") / "configs"

PRE_CUTS_REGRESSION = []

PRE_CUTS_CLASSIFICATION = [
    "Erec > 0",
    "MSCW > -2",
    "MSCW < 2",
    "MSCL > -2",
    "MSCL < 5",
    "EmissionHeight > 0",
    "EmissionHeight < 50",
]


def hyper_parameters(analysis_type, config_file=None):
    """Get hyperparameters for XGBoost model based on analysis type."""
    if analysis_type == "stereo_analysis":
        return regression_hyper_parameters(config_file)
    if analysis_type == "classification":
        return classification_hyper_parameters(config_file)
    raise ValueError(f"Unknown analysis type: {analysis_type}")


def regression_hyper_parameters(config_file=None):
    """Get hyperparameters for XGBoost regression model."""
    path = (
        Path(config_file) if config_file else _CONFIGS_DIR / "default_hyperparameters_stereo.json"
    )
    return _load_hyper_parameters_from_file(path)


def classification_hyper_parameters(config_file=None):
    """Get hyperparameters for XGBoost classification model."""
    path = (
        Path(config_file)
        if config_file
        else _CONFIGS_DIR / "default_hyperparameters_classification.json"
    )
    return _load_hyper_parameters_from_file(path)


def _load_hyper_parameters_from_file(config_file):
    """Load hyperparameters from a JSON file."""
    with config_file.open() as f:
        hyperparameters = json.load(f)
    _logger.info(f"Loaded hyperparameters from {config_file}: {hyperparameters}")
    return hyperparameters


def pre_cuts_regression(min_images=2):
    """
    Get pre-cuts for regression analysis.

    Parameters
    ----------
    min_images : int
        Minimum number of images (DispNImages) for quality cut (default: 2).

    Returns
    -------
    str or None
        Pre-cut string for filtering events.
    """
    cuts = [f"DispNImages >={min_images}"]
    if PRE_CUTS_REGRESSION:
        cuts.extend(PRE_CUTS_REGRESSION)
    event_cut = " & ".join(f"({c})" for c in cuts)
    _logger.info(f"Pre-cuts (regression): {event_cut if event_cut else 'None'}")
    return event_cut if event_cut else None


def pre_cuts_classification(e_min, e_max):
    """Get pre-cuts for classification analysis (no multiplicity filter)."""
    event_cut = f"(Erec >= {e_min}) & (Erec < {e_max})"
    event_cut += " & " + " & ".join(f"({c})" for c in PRE_CUTS_CLASSIFICATION)
    _logger.info(f"Pre-cuts (classification): {event_cut}")
    return event_cut
