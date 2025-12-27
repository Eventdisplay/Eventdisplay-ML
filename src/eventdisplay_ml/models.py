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


def apply_regression_models(df, models_or_dir, selection_mask=None):
    """
    Apply trained XGBoost models for stereo analysis to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    models_or_dir : dict[int, Any] or str
        Either a preloaded models dictionary (as returned by :func:`load_models`)
        or a path to a model directory. If a string is provided, models are
        loaded on the fly to satisfy test expectations.
    selection_mask : pandas.Series or None
        Optional mask; False entries are marked with -999 in outputs.

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
    pred_xoff = np.full(n_events, np.nan, dtype=np.float32)
    pred_yoff = np.full(n_events, np.nan, dtype=np.float32)
    pred_erec = np.full(n_events, np.nan, dtype=np.float32)
    if isinstance(models_or_dir, str):
        models = load_regression_models(models_or_dir)
    else:
        models = models_or_dir

    grouped = df.groupby("DispNImages")

    for n_tel, group_df in grouped:
        n_tel = int(n_tel)
        if int(n_tel) < 2:
            continue
        if n_tel not in models:
            _logger.warning(
                f"No model available for n_tel={n_tel}, skipping {len(group_df)} events"
            )
            continue

        _logger.info(f"Processing {len(group_df)} events with n_tel={n_tel}")

        training_vars_with_pointing = [
            *xgb_per_telescope_training_variables(),
            "fpointing_dx",
            "fpointing_dy",
        ]
        df_flat = flatten_data_vectorized(
            group_df,
            n_tel,
            training_vars_with_pointing,
            analysis_type="stereo_analysis",
            apply_pointing_corrections=True,
            dtype=np.float32,
        )

        excluded_columns = ["MCxoff", "MCyoff", "MCe0"]
        for n in range(n_tel):
            excluded_columns.append(f"fpointing_dx_{n}")
            excluded_columns.append(f"fpointing_dy_{n}")

        feature_cols = [col for col in df_flat.columns if col not in excluded_columns]
        x_features = df_flat[feature_cols]

        model = models[n_tel]
        predictions = model.predict(x_features)

        for i, idx in enumerate(group_df.index):
            pred_xoff[idx] = predictions[i, 0]
            pred_yoff[idx] = predictions[i, 1]
            pred_erec[idx] = predictions[i, 2]

    if selection_mask is not None:
        pred_xoff = np.where(selection_mask, pred_xoff, -999.0)
        pred_yoff = np.where(selection_mask, pred_yoff, -999.0)
        pred_erec = np.where(selection_mask, pred_erec, -999.0)

    return pred_xoff, pred_yoff, pred_erec


def apply_classification_models(df, models):
    """
    Apply trained XGBoost classification models to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    models: dict
        Preloaded models dictionary
    model_parameters : dict
        Model parameters defining energy and zenith angle bins.

    Returns
    -------
    class_probability : numpy.ndarray
        Array of predicted class probabilities for each event in the chunk, aligned
        with the index of ``df``.
    """
    class_probability = np.full(len(df), np.nan, dtype=np.float32)

    # 1. Group by Number of Images (n_tel)
    grouped_ntel = df.groupby("DispNImages")

    for n_tel, group_ntel_df in grouped_ntel:
        n_tel = int(n_tel)
        if n_tel < 2 or n_tel not in models:
            continue

        # 2. Group by Energy Bin (e_bin)
        grouped_ebin = group_ntel_df.groupby("e_bin")

        for e_bin, group_df in grouped_ebin:
            e_bin = int(e_bin)

            if e_bin == -1:
                continue

            if e_bin not in models[n_tel]:
                _logger.warning(f"No model for n_tel={n_tel}, e_bin={e_bin}")
                continue

            _logger.info(f"Processing {len(group_df)} events: n_tel={n_tel}, bin={e_bin}")

            # Prepare features (same logic as your regression)
            training_vars = xgb_per_telescope_training_variables()
            df_flat = flatten_data_vectorized(
                group_df,
                n_tel,
                training_vars,
                analysis_type="classification",
                apply_pointing_corrections=False,
                dtype=np.float32,
            )

            excluded = ["label", "class", "Erec", "MCe0"]
            for n in range(n_tel):
                excluded.append(f"E_{n}")
                excluded.append(f"ES_{n}")
            feature_cols = [col for col in df_flat.columns if col not in excluded]
            x_features = df_flat[feature_cols]

            model = models[n_tel][e_bin]
            probs = model.predict_proba(x_features)[:, 1]

            for i, idx in enumerate(group_df.index):
                class_probability[idx] = probs[i]

    return class_probability
