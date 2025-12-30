"""Apply models for regression and classification tasks."""

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import uproot

from eventdisplay_ml import features
from eventdisplay_ml.data_processing import (
    apply_image_selection,
    energy_in_bins,
    flatten_feature_data,
    zenith_in_bins,
)
from eventdisplay_ml.utils import parse_image_selection

_logger = logging.getLogger(__name__)


def load_models(analysis_type, model_prefix):
    """
    Load XGBoost models based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis ("stereo_analysis" or "classification").
    model_prefix : str
        Prefix path to the trained model files.

    Returns
    -------
    dict
        A dictionary of loaded models.
    dict, optional
        A dictionary of model parameters (only for classification).
    """
    if analysis_type == "stereo_analysis":
        return load_regression_models(model_prefix)
    if analysis_type == "classification":
        return load_classification_models(model_prefix)
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def load_classification_models(model_prefix):
    """
    Load XGBoost classification models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files. Models are expected to be named
        ``{model_prefix}_ntel{n_tel}_bin{e_bin}.joblib``.

    Returns
    -------
    dict, dict
        A dictionary mapping the number of telescopes (n_tel) and energy bin
        to the corresponding loaded model objects. Also returns a dictionary
        of model parameters.
    """
    model_prefix = Path(model_prefix)
    model_dir_path = Path(model_prefix.parent)

    models = {}
    par = {}
    for n_tel in range(2, 5):
        pattern = f"{model_prefix.name}_ntel{n_tel}_bin*.joblib"
        for file in sorted(model_dir_path.glob(pattern)):
            match = re.search(r"_bin(\d+)\.joblib$", file.name)
            if not match:
                _logger.warning(f"Could not extract energy bin from filename: {file.name}")
                continue
            e_bin = int(match.group(1))
            _logger.info(f"Loading model: {file}")
            model_data = joblib.load(file)
            models.setdefault(n_tel, {})[e_bin] = model_data["model"]
            par = _update_parameters(par, model_data.get("parameters", {}), e_bin)

    _logger.info(f"Loaded classification model parameters: {par}")
    return models, par


def _update_parameters(full_params, single_bin_params, e_bin_number):
    """Merge a single-bin model parameters into the full parameters dict."""
    energy_bin = single_bin_params["energy_bins_log10_tev"]
    zenith_bins = single_bin_params["zenith_bins_deg"]

    if "energy_bins_log10_tev" not in full_params:
        full_params["energy_bins_log10_tev"] = []
        full_params["zenith_bins_deg"] = zenith_bins

    while len(full_params["energy_bins_log10_tev"]) <= e_bin_number:
        full_params["energy_bins_log10_tev"].append(None)

    full_params["energy_bins_log10_tev"][e_bin_number] = energy_bin
    if full_params.get("zenith_bins_deg") != zenith_bins:
        raise ValueError(f"Inconsistent zenith_bins_deg for energy bin {e_bin_number}")

    return full_params


def load_regression_models(model_prefix):
    """
    Load XGBoost models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files. Models are expected to be named
        ``{model_prefix}_ntel{n_tel}_xgboost.joblib``.

    Returns
    -------
    dict[int, Any]
        A dictionary mapping the number of telescopes (n_tel) to the
        corresponding loaded model objects. Only models whose files
        exist in ``model_dir`` are included.
    """
    model_prefix = Path(model_prefix)
    model_dir_path = Path(model_prefix.parent)

    models = {}
    for n_tel in range(2, 5):
        model_filename = model_dir_path / f"{model_prefix.name}_ntel{n_tel}.joblib"
        if model_filename.exists():
            _logger.info(f"Loading model: {model_filename}")
            model_data = joblib.load(model_filename)
            models[n_tel] = model_data["model"]
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
        Preloaded models dictionary.

    Returns
    -------
    pred_xoff : numpy.ndarray
        Array of predicted Xoff values for each event in the chunk.
    pred_yoff : numpy.ndarray
        Array of predicted Yoff values for each event in the chunk.
    pred_erec : numpy.ndarray
        Array of predicted Erec values for each event in the chunk.
    """
    preds = np.full((len(df), 3), np.nan, dtype=np.float32)

    grouped = df.groupby("DispNImages")

    for n_tel, group_df in grouped:
        n_tel = int(n_tel)
        if n_tel < 2 or n_tel not in models:
            _logger.warning(f"No model for n_tel={n_tel}")
            continue

        _logger.info(f"Processing {len(group_df)} events with n_tel={n_tel}")

        x_features = flatten_feature_data(
            group_df, n_tel, analysis_type="stereo_analysis", training=False
        )
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
                _logger.warning("Skipping events with e_bin = -1")
                continue
            if e_bin not in models[n_tel]:
                _logger.warning(f"No model for n_tel={n_tel}, e_bin={e_bin}")
                continue

            _logger.info(f"Processing {len(group_df)} events: n_tel={n_tel}, bin={e_bin}")

            x_features = flatten_feature_data(
                group_df, n_tel, analysis_type="classification", training=False
            )
            class_probability[group_df.index] = models[n_tel][e_bin].predict_proba(x_features)[:, 1]

    return class_probability


def process_file_chunked(
    analysis_type,
    input_file,
    output_file,
    models,
    image_selection,
    model_parameters=None,
    max_events=None,
    chunk_size=500000,
):
    """
    Stream events from an input file in chunks, apply XGBoost models, write events.

    Parameters
    ----------
    input_file : str
        Path to the input file containing a "data" TTree.
    output_file : str
        Path to the output file to create.
    models : dict
        Dictionary of loaded XGBoost models for regression.
    image_selection : str
        String specifying which telescope indices to select.
    model_parameters : dict, optional
        Dictionary of model parameters.
    max_events : int, optional
        Maximum number of events to process.
    chunk_size : int, optional
        Number of events to read and process per chunk.
    """
    branch_list = features.features(analysis_type, training=False)
    selected_indices = parse_image_selection(image_selection)

    _logger.info(f"Chunk size: {chunk_size}")
    if max_events:
        _logger.info(f"Maximum events to process: {max_events}")

    with uproot.recreate(output_file) as root_file:
        tree = _output_tree(analysis_type, root_file)
        total_processed = 0

        for df_chunk in uproot.iterate(
            f"{input_file}:data",
            branch_list,
            library="pd",
            step_size=chunk_size,
        ):
            if df_chunk.empty:
                continue

            df_chunk = apply_image_selection(df_chunk, selected_indices, analysis_type)
            if df_chunk.empty:
                continue
            if max_events is not None and total_processed >= max_events:
                break

            # Reset index to local chunk indices (0, 1, 2, ...) to avoid
            # index out-of-bounds when indexing chunk-sized output arrays
            df_chunk = df_chunk.reset_index(drop=True)
            if analysis_type == "classification":
                df_chunk["e_bin"] = energy_in_bins(
                    df_chunk, model_parameters["energy_bins_log10_tev"]
                )
                df_chunk["ze_bin"] = zenith_in_bins(
                    90.0 - df_chunk["ArrayPointing_Elevation"].values,
                    model_parameters["zenith_bins_deg"],
                )

            _apply_model(analysis_type, df_chunk, models, tree)

            total_processed += len(df_chunk)
            _logger.info(f"Processed {total_processed} events so far")

    _logger.info(f"Total processed events written: {total_processed}")


def _output_tree(analysis_type, root_file):
    """
    Generate output tree structure for the given analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis")
    root_file : uproot.writing.WritingFile
        Uproot file object to create the tree in.

    Returns
    -------
    uproot.writing.WritingTTree
        Output tree.
    """
    if analysis_type == "stereo_analysis":
        return root_file.mktree(
            "StereoAnalysis",
            {"Dir_Xoff": np.float32, "Dir_Yoff": np.float32, "Dir_Erec": np.float32},
        )
    if analysis_type == "classification":
        return root_file.mktree("Classification", {"IsGamma": np.float32})
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _apply_model(analysis_type, df_chunk, models, tree):
    """
    Apply models to the data chunk.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis")
    df_chunk : pandas.DataFrame
        Data chunk to process.
    models : dict
        Dictionary of loaded XGBoost models.
    tree : uproot.writing.WritingTTree
        Output tree to write results to.
    """
    if analysis_type == "stereo_analysis":
        pred_xoff, pred_yoff, pred_erec = apply_regression_models(df_chunk, models)
        tree.extend(
            {
                "Dir_Xoff": np.asarray(pred_xoff, dtype=np.float32),
                "Dir_Yoff": np.asarray(pred_yoff, dtype=np.float32),
                "Dir_Erec": np.power(10.0, pred_erec, dtype=np.float32),
            }
        )
    elif analysis_type == "classification":
        pred_proba = apply_classification_models(df_chunk, models)
        tree.extend(
            {
                "IsGamma": np.asarray(pred_proba, dtype=np.float32),
            }
        )
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
