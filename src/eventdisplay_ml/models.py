"""Apply models for regression and classification tasks."""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from eventdisplay_ml.data_processing import flatten_data_vectorized
from eventdisplay_ml.training_variables import (
    xgb_per_telescope_training_variables,
)

_logger = logging.getLogger(__name__)


def load_classification_models(model_dir, model_parameters):
    """
    Load XGBoost classification models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the trained model files
    model_parameters : str or None
        Path to a JSON file defining which models to load.

    Returns
    -------
    dict
        A dictionary mapping the number of telescopes (n_tel) and energy bin
        to the corresponding loaded model objects.
    """
    par = _load_model_parameters(model_parameters)

    file_name_template = par.get("model_file_name", "gamma_hadron_bdt")

    models = {}
    model_dir_path = Path(model_dir)

    for n_tel in range(2, 5):
        models[n_tel] = {}
        for e_bin in range(len(par["energy_bins_log10_tev"])):
            file = f"{file_name_template}_ntel{n_tel}_bin{e_bin}.joblib"
            model_filename = model_dir_path / file

            if model_filename.exists():
                _logger.info(f"Loading model: {model_filename}")
                models[n_tel][e_bin] = joblib.load(model_filename)
            else:
                _logger.warning(f"Model not found: {model_filename}")

    return models, par


def _load_model_parameters(model_parameters):
    """Load model parameters from a JSON file."""
    try:
        with open(model_parameters) as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model parameters file not found: {model_parameters}") from exc


def load_regression_models(model_dir):
    """
    Load XGBoost models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the trained model files
        named ``dispdir_bdt_ntel{n_tel}_xgboost.joblib``.

    Returns
    -------
    dict[int, Any]
        A dictionary mapping the number of telescopes (n_tel) to the
        corresponding loaded model objects. Only models whose files
        exist in ``model_dir`` are included.
    """
    models = {}
    model_dir_path = Path(model_dir)
    for n_tel in range(2, 5):
        model_filename = model_dir_path / f"dispdir_bdt_ntel{n_tel}_xgboost.joblib"
        if model_filename.exists():
            _logger.info(f"Loading model: {model_filename}")
            models[n_tel] = joblib.load(model_filename)
        else:
            _logger.warning(f"Model not found: {model_filename}")
    return models


def apply_regression_models(df, models):
    """
    Apply trained XGBoost models for stereo analysis to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    models : dict
        Preloaded models dictionary (as returned by :func:`load_models`).

    Returns
    -------
    pred_xoff : numpy.ndarray
        Array of predicted Xoff values for each event in the chunk, aligned
        with the index of ``df``.
    pred_yoff : numpy.ndarray
        Array of predicted Yoff values for each event in the chunk, aligned
        with the index of ``df``.
    pred_erec : numpy.ndarray
        Array of predicted Erec values for each event in the chunk, aligned
        with the index of ``df``.
    """
    n_events = len(df)
    preds = np.full((n_events, 3), np.nan, dtype=np.float32)

    grouped = df.groupby("DispNImages")

    for n_tel, group_df in grouped:
        n_tel = int(n_tel)
        if n_tel < 2 or n_tel not in models:
            _logger.warning(f"No model for n_tel={n_tel}")
            continue

        _logger.info(f"Processing {len(group_df)} events with n_tel={n_tel}")

        x_features = features(group_df, n_tel, analysis_type="stereo_analysis")
        preds[group_df.index] = models[n_tel].predict(x_features)

    return preds[:, 0], preds[:, 1], preds[:, 2]


def apply_classification_models(df, models):
    """
    Apply trained XGBoost classification models to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    models: dict
        Preloaded models dictionary

    Returns
    -------
    class_probability : numpy.ndarray
        Array of predicted class probabilities for each event in the chunk, aligned
        with the index of ``df``.
    """
    class_probability = np.full(len(df), np.nan, dtype=np.float32)

    # 1. Group by Number of Images (n_tel)
    for n_tel, group_ntel_df in df.groupby("DispNImages"):
        n_tel = int(n_tel)
        if n_tel < 2 or n_tel not in models:
            _logger.warning(f"No model for n_tel={n_tel}")
            continue

        # 2. Group by Energy Bin (e_bin)
        for e_bin, group_df in group_ntel_df.groupby("e_bin"):
            e_bin = int(e_bin)
            if e_bin == -1:
                continue
            if e_bin not in models[n_tel]:
                _logger.warning(f"No model for n_tel={n_tel}, e_bin={e_bin}")
                continue

            _logger.info(f"Processing {len(group_df)} events: n_tel={n_tel}, bin={e_bin}")

            x_features = features(group_df, n_tel, analysis_type="classification")
            class_probability[group_df.index] = models[n_tel][e_bin].predict_proba(x_features)[:, 1]

    return class_probability


def features(group_df, ntel, analysis_type):
    """Get flattened features for a group of events with given telescope multiplicity."""
    if analysis_type == "stereo_analysis":
        training_vars = [*xgb_per_telescope_training_variables(), "fpointing_dx", "fpointing_dy"]
    else:
        training_vars = xgb_per_telescope_training_variables()

    df_flat = flatten_data_vectorized(
        group_df,
        ntel,
        training_vars,
        analysis_type=analysis_type,
        apply_pointing_corrections=(analysis_type == "stereo_analysis"),
    )

    excluded_columns = {"MCxoff", "MCyoff", "MCe0", "label", "class"}
    if analysis_type == "stereo_analysis":
        excluded_columns.update(
            {
                *[f"fpointing_dx_{i}" for i in range(ntel)],
                *[f"fpointing_dy_{i}" for i in range(ntel)],
            }
        )
    else:
        excluded_columns.update(
            {
                "Erec",
                *[f"E_{i}" for i in range(ntel)],
                *[f"ES_{i}" for i in range(ntel)],
            }
        )

    return df_flat.drop(columns=excluded_columns, errors="ignore")
