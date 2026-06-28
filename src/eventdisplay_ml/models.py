"""Apply models for regression and classification tasks."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml import data_processing, diagnostic_utils, features, utils
from eventdisplay_ml.data_processing import (
    energy_interpolation_bins,
    flatten_feature_data,
    zenith_in_bins,
)
from eventdisplay_ml.evaluate import (
    evaluate_classification_model,
    evaluate_regression_model,
    evaluation_efficiency,
)

# Energy ranges for evaluation bins (log10(E/TeV))
_EVAL_LOG_E_MIN = -2
_EVAL_LOG_E_MAX = 2.5
_EVAL_LOG_E_BINS = 9
_MIN_WEIGHTED_ENERGY_BIN_EVENTS = 100
_MAX_REGRESSION_SAMPLE_WEIGHT = 50.0

_logger = logging.getLogger(__name__)


def save_models(model_configs):
    """Save trained models to files.

    Models already have per-target SHAP importance values cached during evaluation.
    """
    memory_profile = model_configs.get("memory_profile", False)
    utils.log_memory_checkpoint("save_models:start", enabled=memory_profile)
    output_file = utils.output_file_name(
        model_configs.get("model_prefix"),
        energy_bin_number=model_configs.get("energy_bin_number"),
    )
    joblib.dump(model_configs, output_file)
    utils.log_memory_checkpoint("save_models:end", enabled=memory_profile)


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
    Load XGBoost classification models.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files.
    model_name : str
        Name of the model to load.

    Returns
    -------
    dict, dict
        A dictionary mapping energy bins to the corresponding loaded model objects.
        Also returns a dictionary of model parameters.
    """
    model_prefix = Path(model_prefix)
    model_dir_path = Path(model_prefix.parent)

    models = {}
    par = {}

    if not model_dir_path.is_dir():
        raise FileNotFoundError(f"Classification model directory not found: '{model_dir_path}'")

    pattern = re.compile(rf"^{re.escape(model_prefix.name)}_ebin(\d+)\.joblib(?:\.gz)?$")
    matched_files = [
        file for file in model_dir_path.iterdir() if file.is_file() and pattern.match(file.name)
    ]
    files_by_bin = {}
    for file in matched_files:
        match = pattern.match(file.name)
        if not match:
            continue
        e_bin = int(match.group(1))
        existing = files_by_bin.get(e_bin)
        if existing is None or file.name.endswith(".joblib.gz"):
            files_by_bin[e_bin] = file
    files = [files_by_bin[e_bin] for e_bin in sorted(files_by_bin)]

    _logger.info(f"Loading classification models from {files}")
    if not files:
        raise FileNotFoundError(
            "No classification model files found for prefix "
            f"'{model_prefix}'. Expected files named "
            f"'{model_prefix.name}_ebin<N>.joblib' or "
            f"'{model_prefix.name}_ebin<N>.joblib.gz' in '{model_dir_path}'."
        )
    for file in files:
        match = pattern.match(file.name)
        if not match:
            _logger.warning(f"Could not extract energy bin from filename: {file.name}")
            continue
        e_bin = int(match.group(1))
        _logger.info(f"Loading model for e_bin={e_bin}: {file}")
        model_data = utils.load_joblib(file)
        _check_bin(e_bin, model_data.get("energy_bin_number"))
        models.setdefault(e_bin, {})
        try:
            models[e_bin]["model"] = model_data["models"][model_name]["model"]
        except KeyError:
            raise KeyError(f"Model name '{model_name}' not found in file: {file}")
        models[e_bin]["features"] = model_data.get("features", [])
        models[e_bin]["efficiency"] = model_data["models"][model_name].get("efficiency")
        models[e_bin]["thresholds"] = _calculate_classification_thresholds(
            models[e_bin]["efficiency"]
        )
        energy_bin_metadata = _validate_energy_bin_metadata(
            model_data.get("energy_bins_log10_tev"),
            file,
        )
        par = _update_parameters(
            par,
            model_data.get("zenith_bins_deg"),
            energy_bin_metadata,
            e_bin,
        )
    if not par.get("energy_bins_log10_tev"):
        raise ValueError(
            f"Classification models for prefix '{model_prefix}' do not define "
            "'energy_bins_log10_tev'. Re-train or provide model files with bin metadata."
        )
    if not par.get("zenith_bins_deg"):
        raise ValueError(
            f"Classification models for prefix '{model_prefix}' do not define "
            "'zenith_bins_deg'. Re-train or provide model files with bin metadata."
        )
    _logger.info(f"Loaded classification models. Parameters: {par}")
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


def _validate_energy_bin_metadata(energy_bin, model_file):
    """Validate per-file classification energy bin metadata.

    Parameters
    ----------
    energy_bin : Any
        Raw ``energy_bins_log10_tev`` value loaded from one model file.
    model_file : pathlib.Path
        Source model file path used for error reporting.

    Returns
    -------
    dict
        Validated metadata containing ``E_min`` and ``E_max`` keys.
    """
    if not isinstance(energy_bin, dict):
        raise ValueError(
            "Classification model file "
            f"'{model_file}' has invalid 'energy_bins_log10_tev' metadata: "
            "expected a dict with keys 'E_min' and 'E_max'."
        )

    missing_keys = [key for key in ("E_min", "E_max") if key not in energy_bin]
    if missing_keys:
        missing = ", ".join(f"'{key}'" for key in missing_keys)
        raise ValueError(
            "Classification model file "
            f"'{model_file}' has incomplete 'energy_bins_log10_tev' metadata: "
            f"missing required key(s): {missing}."
        )

    return energy_bin


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
    Load XGBoost models.

    Parameters
    ----------
    model_prefix : str
        Prefix path to the trained model files.
    model_name : str
        Name of the model to load.

    Returns
    -------
    dict
        Model dictionary.
    """
    model_path = utils.resolve_joblib_path(model_prefix)
    _logger.info(f"Loading regression model: {model_path}")

    model_data = utils.load_joblib(model_path)
    models = {
        model_name: {
            "model": model_data["models"][model_name]["model"],
            "features": model_data.get("features", []),
        }
    }
    par = {}
    for key in ("target_mean", "target_std"):
        if key in model_data:
            par[key] = model_data[key]
        else:
            _logger.warning("Missing '%s' in regression model file: %s", key, model_path)

    _logger.info("Loaded regression model.")
    return models, par


def apply_regression_models(df, model_configs):
    """
    Apply trained XGBoost model for stereo analysis to all events.

    By default, all events are processed with a single model. If a high-multiplicity
    model is configured, it is used for events with more than two images.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    model_configs : dict
        Preloaded models dictionary with 'tel_config' key.

    Returns
    -------
    pred_xoff : numpy.ndarray
        Array of predicted Xoff values for each event in the chunk.
    pred_yoff : numpy.ndarray
        Array of predicted Yoff values for each event in the chunk.
    pred_erec : numpy.ndarray
        Array of predicted Erec values for each event in the chunk.
    """
    _logger.info(f"Processing {len(df)} events")

    tel_config = model_configs.get("tel_config")
    n_tel = tel_config["max_tel_id"] + 1 if tel_config else 4

    flatten_data = flatten_feature_data(
        df,
        n_tel,
        analysis_type="stereo_analysis",
        training=False,
        tel_config=tel_config,
        observatory=model_configs.get("observatory", "veritas"),
        preview_rows=model_configs.get("preview_rows", 20),
    )

    def predict(models, target_mean_cfg, target_std_cfg, mask=None):
        model_data = next(iter(models.values()))
        model_input = flatten_data.reindex(columns=model_data["features"])
        if mask is not None:
            model_input = model_input.loc[mask]

        if not target_mean_cfg or not target_std_cfg:
            raise ValueError(
                "Missing target standardization parameters (target_mean/target_std). "
                "Regenerate the regression model or load a model file that includes them."
            )
        target_mean = np.array(
            [target_mean_cfg[key] for key in ("Xoff_residual", "Yoff_residual", "E_residual")]
        )
        target_std = np.array(
            [target_std_cfg[key] for key in ("Xoff_residual", "Yoff_residual", "E_residual")]
        )
        return model_data["model"].predict(model_input) * target_std + target_mean

    model_data = next(iter(model_configs["models"].values()))
    primary_model_input = flatten_data.reindex(columns=model_data["features"])
    data_processing.print_variable_statistics(primary_model_input)

    high_models = model_configs.get("models_high_multiplicity")
    if high_models is None:
        preds = predict(
            model_configs["models"],
            model_configs.get("target_mean"),
            model_configs.get("target_std"),
        )
    else:
        multiplicity = df["DispNImages"].to_numpy()
        low_mask = multiplicity == 2
        high_mask = multiplicity > 2
        preds = np.full((len(df), 3), np.nan, dtype=np.float64)
        if np.any(low_mask):
            preds[low_mask] = predict(
                model_configs["models"],
                model_configs.get("target_mean"),
                model_configs.get("target_std"),
                low_mask,
            )
        if np.any(high_mask):
            preds[high_mask] = predict(
                high_models,
                model_configs.get("target_mean_high_multiplicity"),
                model_configs.get("target_std_high_multiplicity"),
                high_mask,
            )

    flatten_data = primary_model_input

    # Model predicts residuals, so add them to DispBDT baseline
    # Extract DispBDT predictions from the flattened data
    disp_xoff = flatten_data["Xoff_weighted_bdt"].values
    disp_yoff = flatten_data["Yoff_weighted_bdt"].values
    erec_s = flatten_data["ErecS"].values
    valid_erec_mask = (erec_s > 0) & np.isfinite(erec_s)
    if not np.all(valid_erec_mask):
        n_invalid = np.count_nonzero(~valid_erec_mask)
        _logger.warning(
            "Found %d events with ErecS <= 0 or non-finite during apply; "
            "keeping entries but setting log10(ErecS) to NaN.",
            n_invalid,
        )
    # Compute log10 only for valid values to avoid RuntimeWarning
    disp_erec_log = np.full_like(erec_s, np.nan, dtype=np.float64)
    disp_erec_log[valid_erec_mask] = np.log10(erec_s[valid_erec_mask])

    # Add residual predictions to baseline
    pred_xoff = preds[:, 0] + disp_xoff
    pred_yoff = preds[:, 1] + disp_yoff
    pred_erec_log = preds[:, 2] + disp_erec_log

    return pred_xoff, pred_yoff, pred_erec_log


def apply_classification_models(df, model_configs, threshold_keys):
    """
    Apply trained XGBoost classification model to all events.

    All events are processed with models trained on all multiplicities.
    Features are created for all telescopes with DEFAULT_FILL_VALUE defaults for missing telescopes.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunk of events to process.
    model_configs : dict
        Preloaded models dictionary with structure {e_bin: {model, features, thresholds}}
        and 'tel_config' key.
    threshold_keys : list[int]
        Efficiency thresholds (percent) for which to compute binary gamma flags.

    Returns
    -------
    class_probability : numpy.ndarray
        Array of predicted class probabilities for each event in the chunk, aligned
        with the index of ``df``.
    is_gamma : dict[int, numpy.ndarray]
        Mapping from efficiency threshold (percent) to binary arrays (0/1) indicating
        whether each event passes the corresponding classification threshold.
    """
    class_probability = np.full(len(df), np.nan, dtype=np.float32)
    is_gamma = {eff: np.zeros(len(df), dtype=np.uint8) for eff in threshold_keys}
    models = model_configs["models"]

    tel_config = model_configs.get("tel_config")
    n_tel = tel_config["max_tel_id"] + 1 if tel_config else 4

    for bin_pair, group_df in df.groupby(["e_bin_lo", "e_bin_hi"], dropna=False):
        e_bin_lo, e_bin_hi = int(bin_pair[0]), int(bin_pair[1])
        if e_bin_lo == -1 or e_bin_hi == -1:
            _logger.warning("Skipping events with invalid energy interpolation bins")
            continue

        _logger.info(
            "Processing %d events with interpolation bins (%d, %d)",
            len(group_df),
            e_bin_lo,
            e_bin_hi,
        )

        flatten_data = flatten_feature_data(
            group_df,
            n_tel,
            analysis_type="classification",
            training=False,
            tel_config=tel_config,
            observatory=model_configs.get("observatory", "veritas"),
            preview_rows=model_configs.get("preview_rows", 20),
        )
        model_lo = models[e_bin_lo]["model"]
        model_hi = models[e_bin_hi]["model"]
        flatten_lo = flatten_data.reindex(columns=models[e_bin_lo]["features"])
        flatten_hi = flatten_data.reindex(columns=models[e_bin_hi]["features"])

        class_probs_lo = model_lo.predict_proba(flatten_lo)[:, 1]
        if e_bin_lo == e_bin_hi:
            class_probs = class_probs_lo
        else:
            class_probs_hi = model_hi.predict_proba(flatten_hi)[:, 1]
            alpha = group_df["e_alpha"].to_numpy(dtype=np.float32)
            class_probs = (1.0 - alpha) * class_probs_lo + alpha * class_probs_hi
        class_probability[group_df.index] = class_probs

        thresholds_lo = models[e_bin_lo].get("thresholds", {})
        thresholds_hi = models[e_bin_hi].get("thresholds", {})
        for eff in threshold_keys:
            if eff in is_gamma:
                thr_lo = thresholds_lo.get(eff)
                if thr_lo is None:
                    continue
                if e_bin_lo == e_bin_hi:
                    threshold = thr_lo
                else:
                    thr_hi = thresholds_hi.get(eff)
                    if thr_hi is None:
                        continue
                    alpha = group_df["e_alpha"].to_numpy(dtype=np.float32)
                    threshold = (1.0 - alpha) * thr_lo + alpha * thr_hi
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
    tmva_style = model_configs.get("tmva_style", False)
    if tmva_style and analysis_type == "classification":
        branch_list = features.features_tmva_style(analysis_type, training=False)
        # Auxiliary inputs needed for derived features and energy-bin selection.
        branch_list = [b for b in branch_list if b not in {"ze_bin", "ArrayPointing_Azimuth"}]
        for required in ("ArrayPointing_Elevation", "Erec"):
            if required not in branch_list:
                branch_list.append(required)
    else:
        branch_list = features.features(analysis_type, training=False)
    _logger.info(f"Using branches: {branch_list}")
    rename_map = {}

    # Read telescope configuration from input file and resolve branch aliases
    with uproot.open(model_configs.get("input_file")) as root_file:
        tel_config = data_processing.read_telescope_config(root_file)
        model_configs["tel_config"] = tel_config

        tree = root_file["data"]
        branch_list, rename_map = data_processing._resolve_branch_aliases(tree, branch_list)

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
                for e_bin_models in model_configs["models"].values()
                for eff in (e_bin_models.get("thresholds") or {}).keys()
            }
        )

    executor = ThreadPoolExecutor(max_workers=model_configs.get("max_cores", 8))
    with uproot.recreate(model_configs.get("output_file")) as root_file:
        tree = _output_tree(analysis_type, root_file, threshold_keys)
        total_processed = 0

        for chunk_ak in uproot.iterate(
            f"{model_configs.get('input_file')}:data",
            branch_list,
            library="ak",
            step_size=model_configs.get("chunk_size"),
            decompression_executor=executor,
        ):
            if len(chunk_ak) == 0:
                continue

            if rename_map:
                rename_present = {k: v for k, v in rename_map.items() if k in chunk_ak.fields}
                if rename_present:
                    chunk_ak = data_processing._rename_fields(chunk_ak, rename_present)
            chunk_ak = data_processing._ensure_fpointing_fields(chunk_ak)

            if max_events is not None:
                remaining = max_events - total_processed
                if remaining <= 0:
                    break
                if len(chunk_ak) > remaining:
                    chunk_ak = chunk_ak[:remaining]

            chunk_dict = {}
            for field in chunk_ak.fields:
                field_data = chunk_ak[field]
                try:
                    ak.num(field_data)
                    chunk_dict[field] = ak.to_list(field_data)
                except (TypeError, ValueError):
                    chunk_dict[field] = data_processing._to_numpy_1d(field_data)

            df_chunk = pd.DataFrame(chunk_dict)
            # Reset index to local chunk indices (0, 1, 2, ...) to avoid
            # index out-of-bounds when indexing chunk-sized output arrays
            df_chunk = df_chunk.reset_index(drop=True)
            if analysis_type == "classification":
                e_bin_lo, e_bin_hi, e_alpha = energy_interpolation_bins(
                    df_chunk, model_configs["energy_bins_log10_tev"]
                )
                df_chunk["e_bin_lo"] = e_bin_lo
                df_chunk["e_bin_hi"] = e_bin_hi
                df_chunk["e_alpha"] = e_alpha
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


def _feature_array(df, row_indices, x_cols):
    """Build a float32 feature array for selected row positions."""
    column_indices = df.columns.get_indexer(x_cols)
    return df.iloc[row_indices, column_indices].to_numpy(dtype=np.float32, copy=True)


def _predict_unscaled_chunked(model, df, row_indices, x_cols, y_mean, y_std, targets, chunk_size):
    """Predict residuals in chunks and return unscaled predictions as a DataFrame."""
    if chunk_size is None or chunk_size <= 0:
        chunk_size = max(len(row_indices), 1)

    prediction_dtype = np.result_type(np.float32, y_mean.dtype, y_std.dtype)
    predictions = np.empty((len(row_indices), len(targets)), dtype=prediction_dtype)
    for start in range(0, len(row_indices), chunk_size):
        chunk_indices = row_indices[start : start + chunk_size]
        x_chunk = _feature_array(df, chunk_indices, x_cols)
        stop = start + len(chunk_indices)
        pred_scaled = np.asarray(model.predict(x_chunk)).reshape(len(chunk_indices), -1)
        pred_unscaled = pred_scaled * y_std.values + y_mean.values
        predictions[start:stop] = pred_unscaled
        del x_chunk, pred_scaled, pred_unscaled
    return pd.DataFrame(
        predictions,
        columns=targets,
        index=df.index[row_indices],
    )


def _sample_eval_indices(test_idx, max_events, random_state):
    """Select a bounded subset of test indices for XGBoost eval_set."""
    if max_events is None or max_events <= 0 or len(test_idx) <= max_events:
        return test_idx
    rng = np.random.default_rng(random_state)
    return rng.choice(test_idx, max_events, replace=False)


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
    if df.empty:
        _logger.warning("Skipping training due to empty data.")
        return None

    memory_profile = model_configs.get("memory_profile", False)
    utils.log_memory_checkpoint("train_regression:start", df, enabled=memory_profile)

    # Exclude target residuals from features
    excluded_cols = set(model_configs["targets"])
    x_cols = [col for col in df.columns if col not in excluded_cols]
    _logger.info(f"Features ({len(x_cols)}): {', '.join(list(x_cols))}")
    model_configs["features"] = list(x_cols)
    targets = model_configs["targets"]
    utils.log_memory_checkpoint("after feature list creation", df, enabled=memory_profile)

    row_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        row_indices,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
    )
    utils.log_memory_checkpoint("after index train_test_split", enabled=memory_profile)

    # Verify indices are preserved correctly
    _logger.info(
        f"Train indices: min={df.index[train_idx].min()}, "
        f"max={df.index[train_idx].max()}, len={len(train_idx)}"
    )
    _logger.info(
        f"Test indices: min={df.index[test_idx].min()}, "
        f"max={df.index[test_idx].max()}, len={len(test_idx)}"
    )

    # Calculate energy bin weights for balancing ONLY on training data
    # This avoids data leakage from test set distribution
    train_erec_s = df["ErecS"].to_numpy()[train_idx]
    train_e_residual = df["E_residual"].to_numpy()[train_idx]
    train_disp_nimages = df["DispNImages"].to_numpy()[train_idx]
    bin_result = _log_energy_bin_counts_from_arrays(
        train_erec_s,
        train_e_residual,
        train_disp_nimages,
        return_weight_config=True,
    )
    weights_train = bin_result[2] if bin_result else None
    weight_config = bin_result[3] if bin_result else None
    del train_erec_s
    del train_e_residual
    del train_disp_nimages
    utils.log_memory_checkpoint("after sample-weight calculation", enabled=memory_profile)

    # Standardize targets to prevent energy from dominating direction in multi-target learning
    # Compute mean and std from training data only
    target_indices = df.columns.get_indexer(targets)
    y_train = df.iloc[train_idx, target_indices]
    y_test = df.iloc[test_idx, target_indices]
    y_mean = y_train.mean()
    y_std = y_train.std()

    _logger.info("Target standardization (training set):")
    for target in model_configs["targets"]:
        _logger.info(f"  {target}: mean={y_mean[target]:.6f}, std={y_std[target]:.6f}")

    y_train_scaled = (y_train - y_mean) / y_std

    # Store scalers for later use during inference
    model_configs["target_mean"] = y_mean.to_dict()
    model_configs["target_std"] = y_std.to_dict()

    _logger.info(f"Training events: {len(train_idx)}, Testing events: {len(test_idx)}")
    if weights_train is not None:
        _logger.info(
            f"Using energy-bin-based sample weights (mean={weights_train.mean():.3f}, "
            f"std={weights_train.std():.3f})"
        )

    eval_idx = _sample_eval_indices(
        test_idx,
        model_configs.get("eval_max_events", 200000),
        model_configs.get("random_state", None),
    )
    weights_eval = None
    if weight_config is not None:
        weights_eval, _ = _regression_sample_weights(
            df["ErecS"].to_numpy()[eval_idx],
            df["E_residual"].to_numpy()[eval_idx],
            df["DispNImages"].to_numpy()[eval_idx],
            **weight_config,
        )
        _logger.info(
            "Validation sample weights: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            weights_eval.mean(),
            weights_eval.std(),
            weights_eval.min(),
            weights_eval.max(),
        )
    _logger.info(f"XGBoost eval_set test events: {len(eval_idx)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name}")
        x_train = _feature_array(df, train_idx, x_cols)
        y_train_scaled_array = y_train_scaled.to_numpy(dtype=np.float32, copy=True)
        x_eval = _feature_array(df, eval_idx, x_cols)
        y_eval_scaled_array = ((df.iloc[eval_idx, target_indices] - y_mean) / y_std).to_numpy(
            dtype=np.float32, copy=True
        )
        eval_set = [(x_train, y_train_scaled_array), (x_eval, y_eval_scaled_array)]
        utils.log_memory_checkpoint("after building XGBoost fit arrays", enabled=memory_profile)

        utils.log_memory_checkpoint(f"{name}: before XGBRegressor init", enabled=memory_profile)
        model = xgb.XGBRegressor(**cfg.get("hyper_parameters", {}))
        utils.log_memory_checkpoint(f"{name}: before model.fit", enabled=memory_profile)
        model.fit(
            x_train,
            y_train_scaled_array,
            sample_weight=weights_train,
            sample_weight_eval_set=[weights_train, weights_eval],
            eval_set=eval_set,
            verbose=False,
        )
        utils.log_memory_checkpoint(f"{name}: after model.fit", enabled=memory_profile)
        _logger.info(
            f"Training stopped at iteration {model.best_iteration} "
            f"(best score: {model.best_score:.4f})"
        )

        del x_train
        del x_eval
        del y_train_scaled_array
        del y_eval_scaled_array
        del eval_set
        utils.log_memory_checkpoint(f"{name}: after releasing fit arrays", enabled=memory_profile)

        prediction_chunk_size = model_configs.get("prediction_chunk_size", 200000)
        y_train_pred = _predict_unscaled_chunked(
            model,
            df,
            train_idx,
            x_cols,
            y_mean,
            y_std,
            targets,
            prediction_chunk_size,
        )
        utils.log_memory_checkpoint(
            f"{name}: after training-set prediction",
            enabled=memory_profile,
        )

        # Predict on scaled targets and inverse transform back to original scale
        y_pred = _predict_unscaled_chunked(
            model,
            df,
            test_idx,
            x_cols,
            y_mean,
            y_std,
            targets,
            prediction_chunk_size,
        )
        utils.log_memory_checkpoint(
            f"{name}: after test-set prediction",
            enabled=memory_profile,
        )

        generalization_metrics = diagnostic_utils.compute_generalization_metrics(
            y_train,
            y_train_pred,
            y_test,
            y_pred,
            targets,
        )

        residual_normality_stats = diagnostic_utils.compute_residual_normality_stats(
            y_test,
            y_pred,
            targets,
        )

        shap_idx = _sample_eval_indices(
            test_idx,
            1000,
            model_configs.get("random_state", None),
        )
        x_test_shap = df.iloc[shap_idx, df.columns.get_indexer(x_cols)]
        utils.log_memory_checkpoint(f"{name}: before regression evaluation", enabled=memory_profile)
        shap_importance = evaluate_regression_model(
            model, x_test_shap, y_pred, y_test, df, x_cols, y_test, name
        )
        del x_test_shap
        utils.log_memory_checkpoint(f"{name}: after regression evaluation", enabled=memory_profile)
        cfg["model"] = model
        cfg["features"] = x_cols  # Store feature names for later use
        cfg["generalization_metrics"] = generalization_metrics
        cfg["residual_normality_stats"] = residual_normality_stats
        cfg["shap_importance"] = shap_importance  # Store per-target SHAP importance from evaluation

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
    if df[0].empty or df[1].empty:
        raise ValueError(
            "Classification training requires non-empty signal and background data. "
            f"signal_events={len(df[0])}, background_events={len(df[1])}."
        )

    df[0]["label"] = 1
    df[1]["label"] = 0
    full_df = pd.concat([df[0], df[1]], ignore_index=True)
    ze_data = full_df["ze_bin"] if "ze_bin" in full_df.columns else None
    x_data = full_df.drop(columns=["label"])
    if model_configs.get("ignore_ze_bin", False):
        if model_configs.get("balance_class_zenith_weights", False):
            raise ValueError("Cannot use ignore_ze_bin with balance_class_zenith_weights.")
        if "ze_bin" in x_data.columns:
            _logger.info("Removing ze_bin from classification training features.")
            x_data = x_data.drop(columns=["ze_bin"])
    _logger.info(f"Features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    model_configs["features"] = list(x_data.columns)
    y_data = full_df["label"]

    split_inputs = [x_data, y_data]
    if ze_data is not None:
        split_inputs.append(ze_data)

    split_result = train_test_split(
        *split_inputs,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
        stratify=y_data,
    )
    if ze_data is None:
        x_train, x_test, y_train, y_test = split_result
        ze_test = None
    else:
        x_train, x_test, y_train, y_test, _, ze_test = split_result

    _logger.info(f"Training events: {len(x_train)}, Testing events: {len(x_test)}")
    weights_train = None
    if model_configs.get("balance_class_zenith_weights", False):
        weights_train = _class_zenith_balance_weights(x_train, y_train)
        _logger.info(
            "Using class/zenith sample weights "
            f"(mean={weights_train.mean():.3f}, std={weights_train.std():.3f}, "
            f"min={weights_train.min():.3f}, max={weights_train.max():.3f})"
        )
    eval_set = [(x_train, y_train), (x_test, y_test)]

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name}")
        model = xgb.XGBClassifier(**cfg.get("hyper_parameters", {}))
        fit_kwargs = {"eval_set": eval_set, "verbose": True}
        if weights_train is not None:
            fit_kwargs["sample_weight"] = weights_train
        model.fit(x_train, y_train, **fit_kwargs)

        shap_importance = evaluate_classification_model(
            model,
            x_test,
            y_test,
            full_df,
            x_data.columns.tolist(),
            name,
        )
        cfg["model"] = model
        cfg["features"] = x_data.columns.tolist()  # Store feature names for diagnostics
        efficiency_all, efficiencies_by_zenith = evaluation_efficiency(
            name, model, x_test, y_test, return_by_zenith=True, ze_bins=ze_test
        )
        cfg["efficiency"] = efficiency_all
        for ze_bin, ze_efficiency in efficiencies_by_zenith.items():
            cfg[f"efficiency_ze{ze_bin}"] = ze_efficiency
        cfg["shap_importance"] = shap_importance

    return model_configs


def _class_zenith_balance_weights(x_train, y_train):
    """Compute sample weights that equalize class distributions over ze_bin."""
    if "ze_bin" not in x_train.columns:
        raise ValueError(
            "Cannot apply class/zenith balancing because training features do not include ze_bin."
        )

    labels = pd.Series(y_train, index=x_train.index, name="label")
    ze_bins = pd.Series(x_train["ze_bin"], index=x_train.index, name="ze_bin")
    valid = labels.notna() & ze_bins.notna()
    n_invalid = int((~valid).sum())
    if n_invalid:
        _logger.warning(
            "Found %d training events with missing label or ze_bin; using default weight=1.0 before normalization.",
            n_invalid,
        )

    labels_valid = labels[valid]
    ze_valid = ze_bins[valid]
    total_valid = len(labels_valid)
    if total_valid == 0:
        raise ValueError("Cannot apply class/zenith balancing with no valid training events.")

    target_fraction = ze_valid.value_counts(normalize=True).sort_index()
    weights = pd.Series(1.0, index=x_train.index, dtype=np.float64)

    _logger.info("Class/zenith balancing target distribution:")
    for ze_bin, frac in target_fraction.items():
        _logger.info(f"  ze_bin={ze_bin}: target_fraction={frac:.6f}")

    for label in sorted(labels_valid.unique()):
        class_mask = labels_valid == label
        class_ze = ze_valid[class_mask]
        observed_fraction = class_ze.value_counts(normalize=True).sort_index()
        _logger.info(f"Class/zenith balancing weights for label={label}:")

        for ze_bin, target_frac in target_fraction.items():
            obs_frac = observed_fraction.get(ze_bin, 0.0)
            if obs_frac <= 0:
                _logger.info(f"  ze_bin={ze_bin}: no events for this class; no weight assigned")
                continue

            weight = target_frac / obs_frac
            mask = valid & (labels == label) & (ze_bins == ze_bin)
            weights.loc[mask] = weight
            _logger.info(
                f"  ze_bin={ze_bin}: observed_fraction={obs_frac:.6f}, "
                f"weight={weight:.6f}, events={int(mask.sum())}"
            )

    weight_values = weights.to_numpy(dtype=np.float32)
    mean_weight = weight_values.mean()
    if mean_weight > 0:
        weight_values /= mean_weight

    return weight_values


def _log_energy_bin_counts(df):
    """Log counts of training events per evaluation energy bin using true log10 energy.

    Returns
    -------
    tuple or None
        (bin_edges, counts_dict, weights_array) where:
        - bin_edges: np.ndarray of bin boundaries
        - counts_dict: dict mapping intervals to event counts
        - weights_array: np.ndarray of capped inverse-square-root energy weights combined
                         with multiplicity weights
        Returns None if E_residual not found.
    """
    if "E_residual" not in df or "ErecS" not in df:
        _logger.warning("E_residual or ErecS not found; skipping energy-bin availability printout.")
        return None

    return _log_energy_bin_counts_from_arrays(
        df["ErecS"].to_numpy(),
        df["E_residual"].to_numpy(),
        df["DispNImages"].to_numpy(),
    )


def _regression_sample_weights(
    erec_s,
    e_residual,
    disp_nimages,
    *,
    bins,
    energy_bin_weights,
    multiplicity_mean_square,
    max_weight,
    normalization_scale=None,
):
    """Calculate capped regression weights using parameters derived from training data."""
    valid_erec = (erec_s > 0) & np.isfinite(erec_s)
    mc_e0 = np.full(len(erec_s), np.nan, dtype=np.float64)
    mc_e0[valid_erec] = e_residual[valid_erec] + np.log10(erec_s[valid_erec])
    bin_indices = pd.cut(mc_e0, bins=bins, include_lowest=True, labels=False)

    w_energy = np.zeros(len(erec_s), dtype=np.float64)
    for i, weight in enumerate(energy_bin_weights):
        w_energy[bin_indices == i] = weight
    w_multiplicity = np.square(disp_nimages, dtype=np.float64) / multiplicity_mean_square
    raw_weights = w_energy * w_multiplicity

    if normalization_scale is None:
        if not np.any(raw_weights > 0):
            raise ValueError("No regression events remain after energy-bin weighting.")
        if np.count_nonzero(raw_weights) * max_weight < len(raw_weights):
            raise ValueError("Too few weighted regression events to normalize within max_weight.")

        low, high = 0.0, 1.0 / raw_weights.mean()
        while np.minimum(raw_weights * high, max_weight).mean() < 1.0:
            high *= 2.0
        for _ in range(60):
            middle = 0.5 * (low + high)
            if np.minimum(raw_weights * middle, max_weight).mean() < 1.0:
                low = middle
            else:
                high = middle
        normalization_scale = high

    weights = np.minimum(raw_weights * normalization_scale, max_weight).astype(np.float32)
    return weights, normalization_scale


def _log_energy_bin_counts_from_arrays(
    erec_s,
    e_residual,
    disp_nimages,
    return_weight_config=False,
):
    """Log counts and compute energy/multiplicity training weights from arrays."""
    # Handle ErecS with proper checks for valid values (> 0)
    disp_erec_log = np.where(erec_s > 0, np.log10(erec_s), np.nan)
    mc_e0 = e_residual + disp_erec_log

    bins = np.linspace(_EVAL_LOG_E_MIN, _EVAL_LOG_E_MAX, _EVAL_LOG_E_BINS + 1)
    categories = pd.cut(mc_e0, bins=bins, include_lowest=True)
    counts = pd.Series(categories).value_counts(sort=False).sort_index()
    _logger.info("Training events per energy bin (log10 E true):")
    for interval, count in counts.items():
        _logger.info(f"  {interval.left:.2f} to {interval.right:.2f} : {int(count)}")

    # Use inverse-square-root balancing. Sparse bins are excluded and the final
    # energy-times-multiplicity event weights are capped.
    count_per_bin = counts.values
    eligible = count_per_bin >= _MIN_WEIGHTED_ENERGY_BIN_EVENTS
    if not np.any(eligible):
        _logger.warning(
            "No energy bin has at least %d events; weighting all populated bins.",
            _MIN_WEIGHTED_ENERGY_BIN_EVENTS,
        )
        eligible = count_per_bin > 0
    reference_count = np.median(count_per_bin[eligible])
    energy_bin_weights = np.zeros_like(count_per_bin, dtype=np.float64)
    energy_bin_weights[eligible] = np.sqrt(reference_count / count_per_bin[eligible])

    _logger.info(
        "Energy bin weights (inverse-sqrt count; bins with <%d events excluded): %s",
        _MIN_WEIGHTED_ENERGY_BIN_EVENTS,
        energy_bin_weights,
    )

    # Calculate multiplicity weights (prioritize higher-multiplicity events)
    mult_counts = pd.Series(disp_nimages).value_counts()
    _logger.info("Training events per multiplicity:")
    for mult, count in mult_counts.items():
        _logger.info(f"  {int(mult)} telescopes: {int(count)}")

    multiplicity_mean_square = np.mean(np.square(disp_nimages, dtype=np.float64))
    w_multiplicity = np.square(disp_nimages, dtype=np.float64) / multiplicity_mean_square

    _logger.info(
        "Multiplicity weights (quadratic, normalized): "
        f"mean={w_multiplicity.mean():.3f}, "
        f"std={w_multiplicity.std():.3f}, "
        f"min={w_multiplicity.min():.3f}, "
        f"max={w_multiplicity.max():.3f}"
    )

    weight_config = {
        "bins": bins,
        "energy_bin_weights": energy_bin_weights,
        "multiplicity_mean_square": multiplicity_mean_square,
        "max_weight": _MAX_REGRESSION_SAMPLE_WEIGHT,
    }
    combined_weights, normalization_scale = _regression_sample_weights(
        erec_s,
        e_residual,
        disp_nimages,
        **weight_config,
    )
    weight_config["normalization_scale"] = normalization_scale

    _logger.info(
        f"Combined weights (energy x multiplicity): "
        f"mean={combined_weights.mean():.3f}, "
        f"std={combined_weights.std():.3f}, "
        f"min={combined_weights.min():.3f}, "
        f"max={combined_weights.max():.3f} "
        f"(cap={_MAX_REGRESSION_SAMPLE_WEIGHT:.1f})"
    )

    result = (bins, dict(counts.items()), combined_weights)
    if return_weight_config:
        return (*result, weight_config)
    return result
