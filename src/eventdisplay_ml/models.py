"""Apply models for regression and classification tasks."""

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml import features, utils
from eventdisplay_ml.data_processing import (
    apply_image_selection,
    energy_in_bins,
    flatten_feature_data,
    zenith_in_bins,
)
from eventdisplay_ml.evaluate import (
    evaluate_classification_model,
    evaluate_regression_model,
    evaluation_efficiency,
)

_logger = logging.getLogger(__name__)


def save_models(model_configs):
    """Save trained models to files."""
    joblib.dump(
        model_configs,
        utils.output_file_name(
            model_configs.get("model_prefix"),
            model_configs.get("n_tel"),
            model_configs.get("energy_bin_number"),
        ),
    )


def load_models(analysis_type, model_prefix, model_name):
    """
    Load models based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis ("stereo_analysis" or "classification").
    model_prefix : str
        Prefix path to the trained model files.
    model_name : str
        Name of the model to load.

    Returns
    -------
    dict
        A dictionary of loaded models.
    dict, optional
        A dictionary of model parameters
    """
    if analysis_type == "stereo_analysis":
        return load_regression_models(model_prefix, model_name)
    if analysis_type == "classification":
        return load_classification_models(model_prefix, model_name)
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def load_classification_models(model_prefix, model_name):
    """
    Load XGBoost classification models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files.
    model_name : str
        Name of the model to load.

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
        pattern = f"{model_prefix.name}_ntel{n_tel}_ebin*.joblib"
        models.setdefault(n_tel, {})
        for file in sorted(model_dir_path.glob(pattern)):
            match = re.search(r"_ebin(\d+)\.joblib$", file.name)
            if not match:
                _logger.warning(f"Could not extract energy bin from filename: {file.name}")
                continue
            e_bin = int(match.group(1))
            _logger.info(f"Loading model for n_tel={n_tel}, e_bin={e_bin}: {file}")
            model_data = joblib.load(file)
            _check_bin(e_bin, model_data.get("energy_bin_number"))
            _check_bin(n_tel, model_data.get("n_tel"))
            models[n_tel].setdefault(e_bin, {})
            try:
                models[n_tel][e_bin]["model"] = model_data["models"][model_name]["model"]
            except KeyError:
                raise KeyError(f"Model name '{model_name}' not found in file: {file}")
            models[n_tel][e_bin]["features"] = model_data.get("features", [])
            models[n_tel][e_bin]["efficiency"] = model_data["models"][model_name].get("efficiency")
            models[n_tel][e_bin]["thresholds"] = _calculate_classification_thresholds(
                models[n_tel][e_bin]["efficiency"]
            )
            par = _update_parameters(
                par,
                model_data.get("zenith_bins_deg"),
                model_data.get("energy_bins_log10_tev", {}),
                e_bin,
            )

    _logger.info(f"Loaded classification model parameters: {par}")
    return models, par


def _calculate_classification_thresholds(efficiency, min_efficiency=0.2, steps=5):
    """
    Calculate classification thresholds for given signal efficiencies.

    Returns thresholds for signal efficiencies indexed by integer percentage values.

    Parameters
    ----------
    efficiency : pd.DataFrame
        DataFrame with 'signal_efficiency' and 'threshold' columns.
    min_efficiency : float
        Minimum signal efficiency to consider.
    steps : int
        Step size in percent for efficiency thresholds.

    Returns
    -------
    dict[int, float]
        Mapping from efficiency (percent) to classification threshold.
    """
    df = efficiency.copy()
    df = df.sort_values("signal_efficiency")
    eff_targets = np.arange(min_efficiency * 100, 100, steps) / 100.0
    thresholds = np.interp(
        eff_targets,
        df["signal_efficiency"].values,
        df["threshold"].values,
    )

    thresholds = dict(zip((eff_targets * 100).astype(int), thresholds))
    lines = [f"  {k:>3d}% : {float(v):.4f}" for k, v in sorted(thresholds.items())]
    _logger.info(
        "Calculated classification thresholds:\n%s",
        "\n".join(lines),
    )
    return thresholds


def _check_bin(expected, actual):
    """Check if expected and actual bin numbers match."""
    if expected != actual:
        raise ValueError(f"Bin number mismatch: expected {expected}, got {actual}")


def _update_parameters(full_params, zenith_bins, energy_bin, e_bin_number):
    """Merge a single-bin model parameters into the full parameters dict."""
    if "energy_bins_log10_tev" not in full_params:
        full_params["energy_bins_log10_tev"] = []
        full_params["zenith_bins_deg"] = zenith_bins

    if e_bin_number is not None:
        while len(full_params["energy_bins_log10_tev"]) <= e_bin_number:
            full_params["energy_bins_log10_tev"].append(None)
        full_params["energy_bins_log10_tev"][e_bin_number] = energy_bin

    if full_params.get("zenith_bins_deg") != zenith_bins:
        raise ValueError(f"Inconsistent zenith_bins_deg for energy bin {e_bin_number}")
    return full_params


def load_regression_models(model_prefix, model_name):
    """
    Load XGBoost models for different telescope multiplicities from a directory.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files.
    model_name : str
        Name of the model to load.

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
            _logger.info(f"Loading model for n_tel={n_tel}: {model_filename}")
            model_data = joblib.load(model_filename)
            _check_bin(n_tel, model_data.get("n_tel"))
            models.setdefault(n_tel, {})["model"] = model_data["models"][model_name]["model"]
            models[n_tel]["features"] = model_data.get("features", [])
        else:
            _logger.warning(f"Model not found: {model_filename}")

    _logger.info("Loaded regression models.")
    return models, {}


def apply_regression_models(df, model_configs):
    """
    Apply trained XGBoost models for stereo analysis to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    model_configs : dict
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
    models = model_configs["models"]

    for n_tel, group_df in grouped:
        n_tel = int(n_tel)
        if n_tel < 2 or n_tel not in models:
            _logger.warning(f"No model for n_tel={n_tel}")
            continue

        _logger.info(f"Processing {len(group_df)} events with n_tel={n_tel}")

        flatten_data = flatten_feature_data(
            group_df, n_tel, analysis_type="stereo_analysis", training=False
        )
        flatten_data = flatten_data.reindex(columns=models[n_tel]["features"])
        model = models[n_tel]["model"]
        preds[group_df.index] = model.predict(flatten_data)

    return preds[:, 0], preds[:, 1], preds[:, 2]


def apply_classification_models(df, model_configs, threshold_keys):
    """
    Apply trained XGBoost classification models to a DataFrame chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    model_configs : dict
        Preloaded models dictionary
    threshold_keys : list[int]
        Efficiency thresholds (percent) for which to compute binary gamma flags.

    Returns
    -------
    class_probability : numpy.ndarray
        Array of predicted class probabilities for each event in the chunk, aligned
        with the index of ``df``.
    is_gamma : dict[int, numpy.ndarray]
        Mapping from efficiency threshold (percent) to binary arrays (0/1) indicating
        whether each event passes the corresponding classification threshold using
        that bin's stored thresholds.
    """
    class_probability = np.full(len(df), np.nan, dtype=np.float32)
    is_gamma = {eff: np.zeros(len(df), dtype=np.uint8) for eff in threshold_keys}
    models = model_configs["models"]

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

            flatten_data = flatten_feature_data(
                group_df, n_tel, analysis_type="classification", training=False
            )
            model = models[n_tel][e_bin]["model"]
            flatten_data = flatten_data.reindex(columns=models[n_tel][e_bin]["features"])
            class_probs = model.predict_proba(flatten_data)[:, 1]
            class_probability[group_df.index] = class_probs

            thresholds = models[n_tel][e_bin].get("thresholds", {})
            for eff, threshold in thresholds.items():
                if eff in is_gamma:
                    is_gamma[eff][group_df.index] = (class_probs >= threshold).astype(np.uint8)

    return class_probability, is_gamma


def process_file_chunked(analysis_type, model_configs):
    """
    Stream events from an input file in chunks, apply XGBoost models, write events.

    Parameters
    ----------
    analysis_type : str
        Type of analysis ("stereo_analysis" or "classification").
    model_configs : dict
        Dictionary of model configurations.
    """
    branch_list = features.features(analysis_type, training=False)
    _logger.info(f"Using branches: {branch_list}")

    selected_indices = utils.parse_image_selection(model_configs.get("image_selection"))

    max_events = model_configs.get("max_events", None)
    chunk_size = model_configs.get("chunk_size", 500000)
    _logger.info(f"Chunk size: {chunk_size}")
    if max_events:
        _logger.info(f"Maximum events to process: {max_events}")
    threshold_keys = None
    if analysis_type == "classification":
        threshold_keys = sorted(
            {
                eff
                for n_tel_models in model_configs["models"].values()
                for e_bin_models in n_tel_models.values()
                for eff in (e_bin_models.get("thresholds") or {}).keys()
            }
        )

    with uproot.recreate(model_configs.get("output_file")) as root_file:
        tree = _output_tree(analysis_type, root_file, threshold_keys)
        total_processed = 0

        for df_chunk in uproot.iterate(
            f"{model_configs.get('input_file')}:data",
            branch_list,
            library="pd",
            step_size=model_configs.get("chunk_size"),
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
                df_chunk["e_bin"] = energy_in_bins(df_chunk, model_configs["energy_bins_log10_tev"])
                df_chunk["ze_bin"] = zenith_in_bins(
                    90.0 - df_chunk["ArrayPointing_Elevation"].values,
                    model_configs["zenith_bins_deg"],
                )

            _apply_model(analysis_type, df_chunk, model_configs, tree, threshold_keys)

            total_processed += len(df_chunk)
            _logger.info(f"Processed {total_processed} events so far")

    _logger.info(f"Total processed events written: {total_processed}")


def _output_tree(analysis_type, root_file, threshold_keys=None):
    """
    Generate output tree structure for the given analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis")
    root_file : uproot.writing.WritingFile
        Uproot file object to create the tree in.
    threshold_keys : list[int], optional
        Efficiency thresholds (percent) for which to create binary gamma flag branches.

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
        branches = {"Gamma_Prediction": np.float32}
        for eff in threshold_keys or []:
            branches[f"Is_Gamma_{eff}"] = np.uint8
        return root_file.mktree("Classification", branches)
    raise ValueError(f"Unknown analysis_type: {analysis_type}")


def _apply_model(analysis_type, df_chunk, model_config, tree, threshold_keys=None):
    """
    Apply models to the data chunk.

    Parameters
    ----------
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis")
    df_chunk : pandas.DataFrame
        Data chunk to process.
    model_config : dict
        Dictionary of loaded XGBoost models.
    tree : uproot.writing.WritingTTree
        Output tree to write results to.
    threshold_keys : list[int], optional
        Efficiency thresholds (percent) for which to compute binary gamma flags.
    """
    if analysis_type == "stereo_analysis":
        pred_xoff, pred_yoff, pred_erec = apply_regression_models(df_chunk, model_config)
        tree.extend(
            {
                "Dir_Xoff": np.asarray(pred_xoff, dtype=np.float32),
                "Dir_Yoff": np.asarray(pred_yoff, dtype=np.float32),
                "Dir_Erec": np.power(10.0, pred_erec, dtype=np.float32),
            }
        )
    elif analysis_type == "classification":
        pred_proba, pred_is_gamma = apply_classification_models(
            df_chunk, model_config, threshold_keys or []
        )

        tree_payload = {"Gamma_Prediction": np.asarray(pred_proba, dtype=np.float32)}
        for eff, flags in pred_is_gamma.items():
            tree_payload[f"Is_Gamma_{eff}"] = np.asarray(flags, dtype=np.uint8)

        tree.extend(tree_payload)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")


def train_regression(df, model_configs):
    """
    Train a single XGBoost model for multi-target regression.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    model_configs : dict
        Dictionary of model configurations.
    """
    n_tel = model_configs["n_tel"]
    if df.empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return None

    x_cols = df.columns.difference(model_configs["targets"])
    _logger.info(f"Features ({len(x_cols)}): {', '.join(list(x_cols))}")
    model_configs["features"] = list(x_cols)
    x_data, y_data = df[x_cols], df[model_configs["targets"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name} for n_tel={n_tel}...")
        model = xgb.XGBRegressor(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
        evaluate_regression_model(model, x_test, y_test, df, x_cols, y_data, name)
        cfg["model"] = model

    return model_configs


def train_classification(df, model_configs):
    """
    Train a single XGBoost model for gamma/hadron classification.

    Parameters
    ----------
    df : list of pd.DataFrame
        Training data.
    model_configs : dict
        Dictionary of model configurations.
    """
    n_tel = model_configs["n_tel"]
    if df[0].empty or df[1].empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return None

    df[0]["label"] = 1
    df[1]["label"] = 0
    full_df = pd.concat([df[0], df[1]], ignore_index=True)
    x_data = full_df.drop(columns=["label"])
    _logger.info(f"Features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    model_configs["features"] = list(x_data.columns)
    y_data = full_df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
        stratify=y_data,
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name} for n_tel={n_tel}...")
        model = xgb.XGBClassifier(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train)
        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)
        cfg["model"] = model
        cfg["efficiency"] = evaluation_efficiency(name, model, x_test, y_test)

    return model_configs
