"""Apply models for regression and classification tasks."""

import logging
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eventdisplay_ml import data_processing, diagnostic_utils, features, utils
from eventdisplay_ml.data_processing import (
    energy_interpolation_bins,
    flatten_feature_data,
    zenith_in_bins,
)
from eventdisplay_ml.evaluate import (
    classification_thresholds_from_signal,
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
_MODEL_VALIDATION_MEMORY_BYTES = 4 * 1024**3
_MODEL_VALIDATION_TIMEOUT_SECONDS = 120

_logger = logging.getLogger(__name__)


def _validate_saved_model(model_path):
    """Validate a saved model in a memory-limited subprocess."""
    command = [
        sys.executable,
        "-m",
        "eventdisplay_ml._model_validation",
        str(model_path),
        str(_MODEL_VALIDATION_MEMORY_BYTES),
    ]
    validation_env = os.environ.copy()
    # The validator applies a memory limit; unrestricted BLAS thread pools can
    # allocate one workspace per host CPU and fail before the model is read.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        validation_env[variable] = "1"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_MODEL_VALIDATION_TIMEOUT_SECONDS,
            env=validation_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Saved model validation timed out after "
            f"{_MODEL_VALIDATION_TIMEOUT_SECONDS} seconds: {model_path}"
        ) from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"Saved model validation failed with exit code {result.returncode}: "
            f"{model_path}\n{detail}"
        )


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
    output_path = Path(output_file)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".joblib.gz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        joblib.dump(model_configs, temporary_path)
        _logger.info("Validating saved model: %s", temporary_path)
        _validate_saved_model(temporary_path)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    _logger.info("Saved and validated model: %s", output_path)
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
        calibration = model_data["models"][model_name].get("signal_threshold_calibration")
        models[e_bin]["thresholds"] = _calculate_classification_thresholds(
            models[e_bin]["efficiency"], calibration=calibration
        )
        models[e_bin]["support"] = model_data["models"][model_name].get("support", {})
        energy_bin_metadata = _validate_energy_bin_metadata(
            model_data.get("energy_bins_log10_tev"),
            file,
        )
        models[e_bin]["energy_center"] = 0.5 * (
            float(energy_bin_metadata["E_min"]) + float(energy_bin_metadata["E_max"])
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


def _calculate_classification_thresholds(efficiency, min_efficiency=0.2, steps=5, calibration=None):
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
    if efficiency is None or len(efficiency) == 0:
        raise ValueError("Classification efficiency diagnostics are missing from the model file.")
    df = efficiency.copy()
    if calibration is not None:
        calibrated = pd.DataFrame(calibration)
        if {"signal_efficiency_target", "threshold"}.issubset(calibrated.columns):
            df = pd.concat(
                [
                    df[["signal_efficiency", "threshold"]],
                    calibrated[["signal_efficiency_target", "threshold"]].rename(
                        columns={"signal_efficiency_target": "signal_efficiency"}
                    ),
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["signal_efficiency"], keep="last")
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

    try:
        e_min = float(energy_bin["E_min"])
        e_max = float(energy_bin["E_max"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Classification model file "
            f"'{model_file}' has non-numeric energy-bin metadata for 'E_min'/'E_max'."
        ) from exc
    if not np.isfinite(e_min) or not np.isfinite(e_max) or e_min >= e_max:
        raise ValueError(
            "Classification model file "
            f"'{model_file}' has invalid energy-bin metadata: require finite E_min < E_max."
        )
    return {"E_min": e_min, "E_max": e_max}


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
        if "ze_bin" in group_df:
            zenith_values = pd.to_numeric(group_df["ze_bin"], errors="coerce")
            valid_zenith = zenith_values.notna() & (zenith_values >= 0)
            if not valid_zenith.all():
                _logger.warning(
                    "Skipping %d events with invalid/out-of-range zenith bins during "
                    "classification apply",
                    int((~valid_zenith).sum()),
                )
                group_df = group_df.loc[valid_zenith]
                if group_df.empty:
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
        resolved_lo = _resolve_classification_bin(models, e_bin_lo)
        resolved_hi = _resolve_classification_bin(models, e_bin_hi)
        model_lo = models[resolved_lo]["model"]
        model_hi = models[resolved_hi]["model"]
        missing_lo = sorted(set(models[resolved_lo]["features"]) - set(flatten_data.columns))
        missing_hi = sorted(set(models[resolved_hi]["features"]) - set(flatten_data.columns))
        if missing_lo or missing_hi:
            raise ValueError(
                "Classification model/input feature schema mismatch: "
                f"low-bin missing={missing_lo}, high-bin missing={missing_hi}."
            )
        flatten_lo = flatten_data.loc[:, models[resolved_lo]["features"]]
        flatten_hi = flatten_data.loc[:, models[resolved_hi]["features"]]

        class_probs_lo = model_lo.predict_proba(flatten_lo)[:, 1]
        interpolate_models = resolved_lo != resolved_hi
        alpha = _classification_interpolation_alpha(
            group_df,
            e_bin_lo,
            e_bin_hi,
            resolved_lo,
            resolved_hi,
            models,
        )
        if not interpolate_models:
            class_probs = class_probs_lo
        else:
            class_probs_hi = model_hi.predict_proba(flatten_hi)[:, 1]
            class_probs = (1.0 - alpha) * class_probs_lo + alpha * class_probs_hi
        class_probability[group_df.index] = class_probs

        thresholds_lo = models[resolved_lo].get("thresholds", {})
        thresholds_hi = models[resolved_hi].get("thresholds", {})
        for eff in threshold_keys:
            if eff in is_gamma:
                thr_lo = thresholds_lo.get(eff)
                if thr_lo is None:
                    continue
                if not interpolate_models:
                    threshold = thr_lo
                else:
                    thr_hi = thresholds_hi.get(eff)
                    if thr_hi is None:
                        continue
                    threshold = (1.0 - alpha) * thr_lo + alpha * thr_hi
                is_gamma[eff][group_df.index] = (class_probs >= threshold).astype(np.uint8)

    return class_probability, is_gamma


def _classification_interpolation_alpha(
    group_df,
    requested_lo,
    requested_hi,
    resolved_lo,
    resolved_hi,
    models,
):
    """Return interpolation weights in the energy coordinate of resolved models.

    For complete model grids, the precomputed ``e_alpha`` is already correct.
    When a requested bin is borrowed, recompute the coordinate from the event
    energy and the actual model centers so scores and calibrated thresholds use
    the same models and energy geometry.
    """
    if resolved_lo == resolved_hi:
        return np.zeros(len(group_df), dtype=np.float32)

    fallback_alpha = group_df["e_alpha"].to_numpy(dtype=np.float32)
    borrowed = (requested_lo, requested_hi) != (resolved_lo, resolved_hi)
    if not borrowed:
        return fallback_alpha

    center_lo = models[resolved_lo].get("energy_center")
    center_hi = models[resolved_hi].get("energy_center")
    if (
        center_lo is None
        or center_hi is None
        or not np.isfinite(center_lo)
        or not np.isfinite(center_hi)
        or center_hi <= center_lo
    ):
        raise ValueError(
            "Cannot interpolate borrowed classification energy-bin models without "
            "finite energy-center metadata."
        )
    if "Erec" not in group_df:
        raise ValueError(
            "Cannot interpolate borrowed classification energy-bin models without Erec."
        )

    erec = pd.to_numeric(group_df["Erec"], errors="coerce").to_numpy(dtype=np.float64)
    valid_energy = np.isfinite(erec) & (erec > 0.0)
    if not np.all(valid_energy):
        raise ValueError(
            "Cannot interpolate borrowed classification energy-bin models for events "
            "with non-positive or non-finite Erec."
        )
    alpha = (np.log10(erec) - float(center_lo)) / float(center_hi - center_lo)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _resolve_classification_bin(models, requested_bin):
    """Resolve a missing energy-bin model to the nearest available model."""
    if requested_bin in models:
        return requested_bin
    available = sorted(models)
    if not available:
        raise ValueError("No classification models are available for application.")
    nearest = min(available, key=lambda candidate: abs(candidate - requested_bin))
    _logger.warning(
        "No classification model for energy bin %d; borrowing nearest bin %d.",
        requested_bin,
        nearest,
    )
    return nearest


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
        eval_set = [(x_eval, y_eval_scaled_array)]
        utils.log_memory_checkpoint("after building XGBoost fit arrays", enabled=memory_profile)

        utils.log_memory_checkpoint(f"{name}: before XGBRegressor init", enabled=memory_profile)
        hyper_parameters = dict(cfg.get("hyper_parameters", {}))
        early_stopping_rounds = hyper_parameters.pop("early_stopping_rounds", None)
        if early_stopping_rounds is not None:
            hyper_parameters["callbacks"] = [
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True,
                )
            ]
        model = xgb.XGBRegressor(**hyper_parameters)
        utils.log_memory_checkpoint(f"{name}: before model.fit", enabled=memory_profile)
        model.fit(
            x_train,
            y_train_scaled_array,
            sample_weight=weights_train,
            sample_weight_eval_set=[weights_eval],
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
        diagnostic_train_idx = _sample_eval_indices(
            train_idx,
            model_configs.get("diagnostic_max_events", 100000),
            model_configs.get("random_state", None),
        )
        y_train_diagnostic = (
            y_train
            if diagnostic_train_idx is train_idx
            else df.iloc[diagnostic_train_idx, target_indices]
        )
        _logger.info(
            "Post-training diagnostic training events: %d",
            len(diagnostic_train_idx),
        )
        y_train_pred = _predict_unscaled_chunked(
            model,
            df,
            diagnostic_train_idx,
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
            y_train_diagnostic,
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

    left_columns = set(df[0].columns)
    right_columns = set(df[1].columns)
    if left_columns != right_columns:
        raise ValueError(
            "Signal/background classification schemas differ. "
            f"Only signal: {sorted(left_columns - right_columns)}; "
            f"only background: {sorted(right_columns - left_columns)}"
        )

    signal = df[0].copy()
    background = df[1].copy()
    signal["label"] = 1
    background["label"] = 0
    full_df = pd.concat([signal, background], ignore_index=True)
    ze_data = full_df["ze_bin"] if "ze_bin" in full_df.columns else None
    if model_configs.get("balance_class_zenith_weights", False) and ze_data is None:
        raise ValueError("Class/zenith balancing requires the derived ze_bin column.")
    if ze_data is not None:
        zenith_values = pd.to_numeric(ze_data, errors="coerce")
        invalid_zenith = zenith_values.isna() | (zenith_values < 0)
        if invalid_zenith.any():
            raise ValueError(
                "Classification training contains out-of-range or invalid zenith bins: "
                f"{int(invalid_zenith.sum())} events."
            )

    profile = model_configs.get("feature_profile", "robust")
    feature_columns = features.classification_feature_columns(
        full_df.columns,
        profile=profile,
        ignore_ze_bin=model_configs.get("ignore_ze_bin", False),
    )
    for column in feature_columns:
        signal_all_nan = bool(signal[column].isna().all())
        background_all_nan = bool(background[column].isna().all())
        if signal_all_nan != background_all_nan:
            raise ValueError(f"Classification feature '{column}' is all-NaN in only one class.")
        if signal_all_nan:
            raise ValueError(f"Classification feature '{column}' is all-NaN in both classes.")
    x_data = full_df.loc[:, feature_columns]
    _logger.info(f"Features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    model_configs["features"] = list(x_data.columns)
    y_data = full_df["label"]

    train_idx, validation_idx, test_idx, split_metadata = _classification_split_indices(
        y_data,
        full_df.get("__source_file"),
        train_fraction=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state"),
        grouped=model_configs.get("grouped_split", True),
    )
    # Keep a small, explicitly reserved gamma subset for score-threshold
    # calibration.  It is never used for fitting or assessment metrics.
    test_signal_idx = test_idx[y_data.iloc[test_idx].to_numpy() == 1]
    calibration_idx = np.asarray([], dtype=int)
    test_signal_groups = (
        full_df.iloc[test_signal_idx]["__source_file"]
        if "__source_file" in full_df.columns
        else None
    )
    if test_signal_groups is not None and test_signal_groups.nunique() >= 2:
        calibration_groups, _assessment_groups = train_test_split(
            test_signal_groups.unique(),
            test_size=0.5,
            random_state=model_configs.get("random_state"),
        )
        calibration_idx = test_signal_idx[test_signal_groups.isin(calibration_groups).to_numpy()]
    elif len(test_signal_idx) >= 2:
        calibration_idx, _assessment_signal_idx = train_test_split(
            test_signal_idx, test_size=0.5, random_state=model_configs.get("random_state")
        )
    if len(calibration_idx):
        test_idx = np.asarray(
            sorted(set(test_idx) - set(calibration_idx)),
            dtype=int,
        )
    x_train, x_validation, x_test = (
        x_data.iloc[idx] for idx in (train_idx, validation_idx, test_idx)
    )
    y_train, y_validation, y_test = (
        y_data.iloc[idx] for idx in (train_idx, validation_idx, test_idx)
    )
    ze_test = ze_data.iloc[test_idx] if ze_data is not None else None
    _logger.info(
        "Classification split: train=%d validation=%d test=%d (%s)",
        len(x_train),
        len(x_validation),
        len(x_test),
        split_metadata["method"],
    )
    model_configs["classification_split"] = split_metadata
    model_configs["classification_split"]["n_signal_calibration"] = len(calibration_idx)
    model_configs["classification_split"]["calibration_grouped"] = bool(
        test_signal_groups is not None and test_signal_groups.nunique() >= 2
    )
    model_configs["classification_feature_profile"] = profile
    model_configs["nuisance_diagnostics"] = _classification_nuisance_diagnostics(full_df, y_data)
    weights_train = None
    weights_validation = None
    if model_configs.get("balance_class_zenith_weights", False):
        target_ze_fraction = _class_zenith_target_fraction(full_df.iloc[train_idx])
        weights_train = _class_zenith_balance_weights(
            full_df.iloc[train_idx],
            y_train,
            target_ze_fraction=target_ze_fraction,
        )
        weights_validation = _class_zenith_balance_weights(
            full_df.iloc[validation_idx],
            y_validation,
            target_ze_fraction=target_ze_fraction,
        )
        _logger.info(
            "Using class/zenith sample weights "
            f"(mean={weights_train.mean():.3f}, std={weights_train.std():.3f}, "
            f"min={weights_train.min():.3f}, max={weights_train.max():.3f})"
        )
    eval_x, eval_y = x_validation, y_validation
    eval_weights = weights_validation
    eval_max_events = model_configs.get("eval_max_events", 0)
    if eval_max_events and eval_max_events > 0 and len(eval_x) > eval_max_events:
        eval_indices = eval_x.sample(
            n=eval_max_events,
            random_state=model_configs.get("random_state"),
        ).index
        eval_x = eval_x.loc[eval_indices]
        eval_y = eval_y.loc[eval_indices]
        if eval_weights is not None:
            eval_weights = (
                pd.Series(weights_validation, index=x_validation.index).loc[eval_indices].to_numpy()
            )
        _logger.info("Limited XGBoost validation set to %d events", eval_max_events)
    eval_set = [(x_train, y_train), (eval_x, eval_y)]

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name}")
        model = xgb.XGBClassifier(**cfg.get("hyper_parameters", {}))
        fit_kwargs = {"eval_set": eval_set, "verbose": True}
        if weights_train is not None:
            fit_kwargs["sample_weight"] = weights_train
            fit_kwargs["sample_weight_eval_set"] = [weights_train, eval_weights]
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
        try:
            calibration_frame = x_data.iloc[calibration_idx]
            if calibration_frame.empty:
                raise ValueError("no reserved gamma calibration events")
            test_signal_scores = model.predict_proba(calibration_frame)[:, 1]
            cfg["signal_threshold_calibration"] = classification_thresholds_from_signal(
                test_signal_scores
            )
        except (TypeError, ValueError, IndexError) as exc:
            # Lightweight mocks/legacy estimators may not expose probabilities;
            # keep the model usable but make the missing calibration explicit.
            _logger.warning("Could not compute held-out signal thresholds for %s: %s", name, exc)
            cfg["signal_threshold_calibration"] = None
        cfg["support"] = {
            "n_train": len(x_train),
            "n_validation": len(x_validation),
            "n_test": len(x_test),
            "n_signal_test": int((y_test == 1).sum()),
            "n_background_test": int((y_test == 0).sum()),
            "n_signal_calibration": len(calibration_idx),
            "fallback_policy": "held_out_model_only; inspect support before applying",
        }

    return model_configs


def _classification_split_indices(y_data, groups, train_fraction, random_state, grouped=True):
    """Create class-stratified train/validation/test indices.

    Grouping is attempted only when every class has at least six source
    groups.  Sparse VERITAS lists commonly contain one file per class, so the
    deterministic event-level fallback is intentional and recorded in model
    metadata rather than pretending that grouping was achieved.
    """
    if not isinstance(y_data, pd.Series):
        y_data = pd.Series(y_data)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_test_fraction must be between zero and one.")
    rng = random_state
    if groups is not None and not isinstance(groups, pd.Series):
        groups = pd.Series(groups, index=y_data.index)
    use_groups = grouped and groups is not None and groups.notna().all()
    if use_groups:
        use_groups = all(groups[y_data == label].nunique() >= 6 for label in y_data.unique())

    train, validation, test = [], [], []
    if use_groups:
        for label in sorted(y_data.unique()):
            label_mask = y_data.to_numpy() == label
            label_groups = np.asarray(groups[label_mask].unique())
            n_train_groups = int(np.ceil(len(label_groups) * train_fraction))
            n_hold_groups = len(label_groups) - n_train_groups
            if n_train_groups < 1 or n_hold_groups < 2:
                raise ValueError(
                    "Grouped classification split cannot create separate validation "
                    "and test groups for label="
                    f"{label}: train_test_fraction={train_fraction} leaves "
                    f"{n_hold_groups} holdout groups. Reduce train_test_fraction or "
                    "provide more source groups."
                )
            g_train, g_hold = train_test_split(
                label_groups,
                train_size=train_fraction,
                random_state=rng,
            )
            if len(g_hold) < 2:
                raise ValueError(
                    "Grouped classification split cannot create separate validation "
                    f"and test groups for label={label}: only {len(g_hold)} holdout "
                    "groups remain after the training split."
                )
            g_validation, g_test = train_test_split(
                g_hold,
                train_size=0.5,
                random_state=rng,
            )
            train.extend(np.flatnonzero(label_mask & groups.isin(g_train).to_numpy()))
            validation.extend(np.flatnonzero(label_mask & groups.isin(g_validation).to_numpy()))
            test.extend(np.flatnonzero(label_mask & groups.isin(g_test).to_numpy()))
        method = "grouped_source_file"
    else:
        for label in sorted(y_data.unique()):
            label_idx = np.flatnonzero(y_data.to_numpy() == label)
            n_train_events = int(np.ceil(len(label_idx) * train_fraction))
            n_hold_events = len(label_idx) - n_train_events
            if n_train_events < 1 or n_hold_events < 2:
                raise ValueError(
                    "Classification split cannot create separate validation and test "
                    "events for label="
                    f"{label}: train_test_fraction={train_fraction} leaves "
                    f"{n_hold_events} holdout events. Reduce train_test_fraction or "
                    "provide more events."
                )
            label_train, label_hold = train_test_split(
                label_idx,
                train_size=train_fraction,
                random_state=rng,
            )
            if len(label_hold) < 2:
                raise ValueError(
                    "Classification split cannot create separate validation and test "
                    f"events for label={label}: only {len(label_hold)} holdout events "
                    "remain after the training split."
                )
            label_validation, label_test = train_test_split(
                label_hold,
                train_size=0.5,
                random_state=rng,
            )
            train.extend(label_train)
            validation.extend(label_validation)
            test.extend(label_test)
        method = "stratified_event_fallback"

    return (
        np.asarray(sorted(train), dtype=int),
        np.asarray(sorted(validation), dtype=int),
        np.asarray(sorted(test), dtype=int),
        {
            "method": method,
            "grouped_requested": bool(grouped),
            "source_groups_available": bool(use_groups),
        },
    )


def _classification_nuisance_diagnostics(df, labels):
    """Measure separability of routing/activity proxies without serializing a model."""
    candidates = {}
    if "ze_bin" in df:
        candidates["ze_bin"] = df["ze_bin"]
    activity = [column for column in df.columns if column.startswith("tel_active_")]
    if activity:
        candidates["tel_active_count"] = df[activity].sum(axis=1, skipna=True)
    telescope_columns = [column for column in df.columns if re.search(r"_\d+$", str(column))]
    if telescope_columns:
        candidates["feature_missing_fraction"] = df[telescope_columns].isna().mean(axis=1)
    diagnostics = {}
    for name, values in candidates.items():
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna() & labels.notna()
        if valid.sum() < 4 or labels[valid].nunique() < 2 or numeric[valid].nunique() < 2:
            diagnostics[name] = {"auc": np.nan, "n": int(valid.sum())}
            continue
        auc = float(roc_auc_score(labels[valid], numeric[valid]))
        diagnostics[name] = {
            "auc": auc,
            "n": int(valid.sum()),
            "shortcut_strength": max(auc, 1.0 - auc),
        }
    return diagnostics


def _class_zenith_target_fraction(x_train):
    """Return the fixed zenith target distribution derived from training data."""
    if "ze_bin" not in x_train.columns:
        raise ValueError("Cannot derive a zenith target distribution without ze_bin.")
    ze_bins = pd.to_numeric(x_train["ze_bin"], errors="coerce")
    valid = ze_bins.notna() & (ze_bins >= 0)
    if not valid.any():
        raise ValueError("Cannot derive a zenith target distribution with no valid ze_bin.")
    counts = ze_bins[valid].value_counts().sort_index().astype(float)
    return counts / counts.sum()


def _normalize_capped_weights(weights, weight_cap):
    """Normalize positive weights to mean one while enforcing a hard upper bound."""
    if not np.isfinite(weight_cap) or weight_cap <= 0:
        raise ValueError("weight_cap must be a finite positive number.")
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=float(weight_cap), neginf=0.0)
    values = np.clip(values, 0.0, float(weight_cap))
    if not np.any(values):
        return values.astype(np.float32)
    if weight_cap < 1.0:
        _logger.warning(
            "weight_cap=%s is below one; returning capped weights without mean-one normalization.",
            weight_cap,
        )
        return values.astype(np.float32)
    if np.count_nonzero(values) * float(weight_cap) < values.size:
        raise ValueError(
            "Cannot normalize weights to mean one while enforcing weight_cap: "
            "too many zero-weight events."
        )

    target_sum = float(values.size)
    lower, upper = 0.0, 1.0
    while np.minimum(values * upper, weight_cap).sum() < target_sum:
        upper *= 2.0
    for _ in range(64):
        scale = 0.5 * (lower + upper)
        if np.minimum(values * scale, weight_cap).sum() < target_sum:
            lower = scale
        else:
            upper = scale
    return np.minimum(values * upper, float(weight_cap)).astype(np.float32)


def _class_zenith_balance_weights(
    x_train,
    y_train,
    weight_cap=10.0,
    smoothing=1.0,
    target_ze_fraction=None,
):
    """Compute capped, smoothed weights equalizing class distributions over ``ze_bin``.

    ``smoothing`` prevents a single sparse background bin from receiving an
    arbitrarily large weight.  The cap is an explicit robustness guard for the
    sparse-background regime common in VERITAS training lists.

    ``target_ze_fraction`` optionally supplies the target distribution derived
    from the training split.  Passing it when weighting validation data keeps
    evaluation on the same target population rather than recalculating a
    distribution from validation composition.
    """
    if "ze_bin" not in x_train.columns:
        raise ValueError(
            "Cannot apply class/zenith balancing because training features do not include ze_bin."
        )

    labels = pd.Series(y_train, index=x_train.index, name="label")
    ze_bins = pd.Series(x_train["ze_bin"], index=x_train.index, name="ze_bin")
    valid = labels.notna() & ze_bins.notna() & (ze_bins >= 0)
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

    if target_ze_fraction is None:
        target_fraction = _class_zenith_target_fraction(x_train)
    else:
        target_fraction = pd.Series(target_ze_fraction, dtype=float)
        target_fraction = target_fraction.replace([np.inf, -np.inf], np.nan).dropna()
        target_fraction = target_fraction[target_fraction > 0]
        if target_fraction.empty:
            raise ValueError("The zenith target distribution must contain positive mass.")
        target_fraction = target_fraction / target_fraction.sum()
    all_ze = np.asarray(target_fraction.index)
    weights = pd.Series(1.0, index=x_train.index, dtype=np.float64)

    _logger.info("Class/zenith balancing target distribution:")
    for ze_bin, frac in target_fraction.items():
        _logger.info(f"  ze_bin={ze_bin}: target_fraction={frac:.6f}")

    for label in sorted(labels_valid.unique()):
        class_mask = labels_valid == label
        class_ze = ze_valid[class_mask]
        _logger.info(f"Class/zenith balancing weights for label={label}:")

        for ze_bin, target_frac in target_fraction.items():
            obs_count = float((class_ze == ze_bin).sum())
            class_total = float(len(class_ze))
            if obs_count == 0:
                # There is no unbiased within-class estimate for an absent
                # bin.  Give it a finite pseudo-count, then cap the resulting
                # weight; events in absent bins remain at weight one.
                obs_frac = smoothing / (class_total + smoothing * len(all_ze))
            else:
                obs_frac = obs_count / class_total
            weight = min(float(target_frac / obs_frac), float(weight_cap))
            mask = valid & (labels == label) & (ze_bins == ze_bin)
            weights.loc[mask] = weight
            _logger.info(
                f"  ze_bin={ze_bin}: observed_fraction={obs_frac:.6f}, "
                f"weight={weight:.6f}, events={int(mask.sum())}"
            )

    # Equalize total influence of the two classes as well as their zenith
    # shapes.  This prevents a large simulated signal sample from dominating
    # a sparse background sample even when raw event counts differ.
    class_labels = sorted(labels_valid.unique())
    target_class_total = total_valid / len(class_labels)
    for label in class_labels:
        class_mask = valid & (labels == label)
        class_sum = float(weights.loc[class_mask].sum())
        if class_sum > 0:
            weights.loc[class_mask] *= target_class_total / class_sum

    return _normalize_capped_weights(weights.to_numpy(dtype=np.float64), weight_cap)


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

    w_energy = np.ones(len(erec_s), dtype=np.float64)
    w_energy[~valid_erec] = 0.0
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
            "No energy bin has at least %d events; disabling min-bin exclusion and weighting all populated bins.",
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
