"""Hyperparameter for classification and regression models."""

import json
import logging

_logger = logging.getLogger(__name__)


XGB_REGRESSION_HYPERPARAMETERS = {
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.1,  # Shrinkage
        "max_depth": 5,
        "min_child_weight": 1.0,  # Equivalent to MinNodeSize=1.0% for XGBoost
        "objective": "reg:squarederror",
        "n_jobs": 4,
        "random_state": None,
        "tree_method": "hist",
        "subsample": 0.7,  # Default sensible value
        "colsample_bytree": 0.7,  # Default sensible value
    }
}

XGB_CLASSIFICATION_HYPERPARAMETERS = {
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",  # TODO AUC ?
        "n_estimators": 100,  # TODO probably too low
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": None,
    }
}


def regression_hyperparameters(config_file=None):
    """Get hyperparameters for XGBoost regression model."""
    if config_file:
        return _load_hyper_parameters_from_file(config_file)
    _logger.info(f"Default hyperparameters: {XGB_REGRESSION_HYPERPARAMETERS}")
    return XGB_REGRESSION_HYPERPARAMETERS


def classification_hyperparameters(config_file=None):
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
