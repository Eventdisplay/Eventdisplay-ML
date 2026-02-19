"""Hyperparameter for classification and regression models."""

import json
import logging

_logger = logging.getLogger(__name__)


XGB_REGRESSION_HYPERPARAMETERS = {
    "xgboost": {
        "model": None,
        "hyper_parameters": {
            "n_estimators": 1000,
            "learning_rate": 0.1,  # Shrinkage
            "max_depth": 10,
            "min_child_weight": 5.0,  # Equivalent to MinNodeSize=1.0% for XGBoost
            "objective": "reg:squarederror",
            "n_jobs": 8,
            "random_state": None,
            "tree_method": "hist",
            "subsample": 0.7,  # Default sensible value
            "colsample_bytree": 0.7,  # Default sensible value
        },
    }
}

XGB_CLASSIFICATION_HYPERPARAMETERS = {
    "xgboost": {
        "model": None,
        "hyper_parameters": {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "auc"],
            "n_estimators": 5000,
            "early_stopping_rounds": 50,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": None,
            "n_jobs": 48,
        },
    }
}

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
    if config_file:
        return _load_hyper_parameters_from_file(config_file)
    _logger.info(f"Default hyperparameters: {XGB_REGRESSION_HYPERPARAMETERS}")
    return XGB_REGRESSION_HYPERPARAMETERS


def classification_hyper_parameters(config_file=None):
    """Get hyperparameters for XGBoost classification model."""
    if config_file:
        return _load_hyper_parameters_from_file(config_file)
    _logger.info(f"Default hyperparameters: {XGB_CLASSIFICATION_HYPERPARAMETERS}")
    return XGB_CLASSIFICATION_HYPERPARAMETERS


def _load_hyper_parameters_from_file(config_file):
    """Load hyperparameters from a JSON file."""
    with open(config_file) as f:
        hyperparameters = json.load(f)
    _logger.info(f"Loaded hyperparameters from {config_file}: {hyperparameters}")
    return hyperparameters


def pre_cuts_regression(n_tel, min_images=2):
    """
    Get pre-cuts for regression analysis.

    Parameters
    ----------
    n_tel : int or None
        Number of telescopes (not currently used).
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


def pre_cuts_classification(n_tel, e_min, e_max):
    """Get pre-cuts for classification analysis (no multiplicity filter)."""
    event_cut = f"(Erec >= {e_min}) & (Erec < {e_max})"
    event_cut += " & " + " & ".join(f"({c})" for c in PRE_CUTS_CLASSIFICATION)
    _logger.info(f"Pre-cuts (classification): {event_cut}")
    return event_cut
