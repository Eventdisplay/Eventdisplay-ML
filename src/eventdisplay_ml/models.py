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

from eventdisplay_ml import data_processing, features, utils
from eventdisplay_ml.data_processing import (
    energy_in_bins,
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

_logger = logging.getLogger(__name__)


def save_models(model_configs):
    """Save trained models to files."""
    joblib.dump(
        model_configs,
        utils.output_file_name(
            model_configs.get("model_prefix"),
            energy_bin_number=model_configs.get("energy_bin_number"),
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

    pattern = f"{model_prefix.name}_ebin*.joblib"
    files = sorted(model_dir_path.glob(pattern))

    _logger.info("Loading classification models")
    for file in files:
        match = re.search(r"_ebin(\d+)\.joblib$", file.name)
        if not match:
            _logger.warning(f"Could not extract energy bin from filename: {file.name}")
            continue
        e_bin = int(match.group(1))
        _logger.info(f"Loading model for e_bin={e_bin}: {file}")
        model_data = joblib.load(file)
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
        par = _update_parameters(
            par,
            model_data.get("zenith_bins_deg"),
            model_data.get("energy_bins_log10_tev", {}),
            e_bin,
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
    model_path = Path(model_prefix).with_suffix(".joblib")
    _logger.info(f"Loading regression model: {model_path}")

    model_data = joblib.load(model_path)
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

    All events are processed with a single model trained on all multiplicities.
    Features are created for all telescopes with DEFAULT_FILL_VALUE defaults for missing telescopes.

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

    models = model_configs["models"]
    model_data = next(iter(models.values()))
    flatten_data = flatten_data.reindex(columns=model_data["features"])
    data_processing.print_variable_statistics(flatten_data)

    model = model_data["model"]
    preds_scaled = model.predict(flatten_data)

    # Inverse transform predictions from standardized space back to original scale
    # Model was trained on standardized targets (mean=0, std=1)
    target_mean_cfg = model_configs.get("target_mean")
    target_std_cfg = model_configs.get("target_std")
    if not target_mean_cfg or not target_std_cfg:
        raise ValueError(
            "Missing target standardization parameters (target_mean/target_std). "
            "Regenerate the regression model or load a model file that includes them."
        )

    target_mean = np.array(
        [
            target_mean_cfg["Xoff_residual"],
            target_mean_cfg["Yoff_residual"],
            target_mean_cfg["E_residual"],
        ]
    )
    target_std = np.array(
        [
            target_std_cfg["Xoff_residual"],
            target_std_cfg["Yoff_residual"],
            target_std_cfg["E_residual"],
        ]
    )

    # Inverse standardization: y = y_scaled * std + mean
    preds = preds_scaled * target_std + target_mean

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

    for e_bin, group_df in df.groupby("e_bin"):
        e_bin = int(e_bin)
        if e_bin == -1:
            _logger.warning("Skipping events with e_bin = -1")
            continue

        _logger.info(f"Processing {len(group_df)} events with bin={e_bin}")

        flatten_data = flatten_feature_data(
            group_df,
            n_tel,
            analysis_type="classification",
            training=False,
            tel_config=tel_config,
            observatory=model_configs.get("observatory", "veritas"),
            preview_rows=model_configs.get("preview_rows", 20),
        )
        model = models[e_bin]["model"]
        flatten_data = flatten_data.reindex(columns=models[e_bin]["features"])
        class_probs = model.predict_proba(flatten_data)[:, 1]
        class_probability[group_df.index] = class_probs

        thresholds = models[e_bin].get("thresholds", {})
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
    if df.empty:
        _logger.warning("Skipping training due to empty data.")
        return None

    # Exclude target residuals AND MC truth columns from features
    # MC truth columns must not be used as features (would be data leakage)
    # Note: MC truth columns are not added to features (only residuals are added)
    excluded_cols = set(model_configs["targets"])
    x_cols = [col for col in df.columns if col not in excluded_cols]
    _logger.info(f"Features ({len(x_cols)}): {', '.join(list(x_cols))}")
    model_configs["features"] = list(x_cols)
    x_data, y_data = df[x_cols], df[model_configs["targets"]]

    # Split data first to avoid data leakage in weight computation
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
    )

    # Verify indices are preserved correctly
    _logger.info(
        f"Train indices: min={y_train.index.min()}, max={y_train.index.max()}, len={len(y_train)}"
    )
    _logger.info(
        f"Test indices: min={y_test.index.min()}, max={y_test.index.max()}, len={len(y_test)}"
    )

    # Calculate energy bin weights for balancing ONLY on training data
    # This avoids data leakage from test set distribution
    df_train = df.loc[y_train.index]
    bin_result = _log_energy_bin_counts(df_train)
    weights_train = bin_result[2] if bin_result else None

    # Standardize targets to prevent energy from dominating direction in multi-target learning
    # Compute mean and std from training data only
    y_mean = y_train.mean()
    y_std = y_train.std()

    _logger.info("Target standardization (training set):")
    for target in model_configs["targets"]:
        _logger.info(f"  {target}: mean={y_mean[target]:.6f}, std={y_std[target]:.6f}")

    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    # Store scalers for later use during inference
    model_configs["target_mean"] = y_mean.to_dict()
    model_configs["target_std"] = y_std.to_dict()

    _logger.info(f"Training events: {len(x_train)}, Testing events: {len(x_test)}")
    if weights_train is not None:
        _logger.info(
            f"Using energy-bin-based sample weights (mean={weights_train.mean():.3f}, "
            f"std={weights_train.std():.3f})"
        )

    eval_set = [(x_train, y_train_scaled), (x_test, y_test_scaled)]

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name}")
        model = xgb.XGBRegressor(**cfg.get("hyper_parameters", {}))
        model.fit(
            x_train,
            y_train_scaled,
            sample_weight=weights_train,
            eval_set=eval_set,
            verbose=True,
        )
        _logger.info(
            f"Training stopped at iteration {model.best_iteration} "
            f"(best score: {model.best_score:.4f})"
        )

        # Predict on scaled targets and inverse transform back to original scale
        y_pred_scaled = model.predict(x_test)
        y_pred = pd.DataFrame(
            y_pred_scaled * y_std.values + y_mean.values,
            columns=model_configs["targets"],
            index=y_test.index,
        )

        evaluate_regression_model(model, x_test, y_pred, y_test, df, x_cols, y_data, name)
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
    if df[0].empty or df[1].empty:
        _logger.warning("Skipping training due to empty data.")
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

    _logger.info(f"Training events: {len(x_train)}, Testing events: {len(x_test)}")
    eval_set = [(x_train, y_train), (x_test, y_test)]

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name}")
        model = xgb.XGBClassifier(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train, eval_set=eval_set, verbose=True)
        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)
        cfg["model"] = model
        cfg["efficiency"] = evaluation_efficiency(name, model, x_test, y_test)

    return model_configs


def _log_energy_bin_counts(df):
    """Log counts of training events per evaluation energy bin using true log10 energy.

    Returns
    -------
    tuple or None
        (bin_edges, counts_dict, weights_array) where:
        - bin_edges: np.ndarray of bin boundaries
        - counts_dict: dict mapping intervals to event counts
        - weights_array: np.ndarray of inverse-count weights for each event (normalized
                         for both energy and multiplicity)
        Returns None if E_residual not found.
    """
    # Reconstruct MC truth energy from residual + DispBDT baseline
    if "E_residual" not in df or "ErecS" not in df:
        _logger.warning("E_residual or ErecS not found; skipping energy-bin availability printout.")
        return None

    # Handle ErecS with proper checks for valid values (> 0)
    erec_s = df["ErecS"].values
    disp_erec_log = np.where(erec_s > 0, np.log10(erec_s), np.nan)
    mc_e0 = df["E_residual"].values + disp_erec_log

    bins = np.linspace(_EVAL_LOG_E_MIN, _EVAL_LOG_E_MAX, _EVAL_LOG_E_BINS + 1)
    categories = pd.cut(mc_e0, bins=bins, include_lowest=True)
    counts = pd.Series(categories).value_counts(sort=False).sort_index()
    _logger.info("Training events per energy bin (log10 E true):")
    for interval, count in counts.items():
        _logger.info(f"  {interval.left:.2f} to {interval.right:.2f} : {int(count)}")

    # Calculate inverse-count weights for balancing (events in low-count bins get higher weight)
    # Bins with fewer than 10 events get zero weight (excluded from training)
    bin_indices = pd.cut(mc_e0, bins=bins, include_lowest=True, labels=False)
    count_per_bin = counts.values
    # Only invert counts >= 10 to avoid divide-by-zero warning
    inverse_counts = np.zeros_like(count_per_bin, dtype=np.float64)
    mask = count_per_bin >= 10
    inverse_counts[mask] = 1.0 / count_per_bin[mask]
    # Normalize by mean of non-zero weights only
    valid_weights = inverse_counts[inverse_counts > 0]
    if len(valid_weights) > 0:
        inverse_counts = inverse_counts / valid_weights.mean()

    # Assign weight to each event based on its energy bin
    w_energy = np.ones(len(df), dtype=np.float32)
    for i, inv_count in enumerate(inverse_counts):
        mask = bin_indices == i
        w_energy[mask] = inv_count

    _logger.info(f"Energy bin weights (inverse-count, normalized): {inverse_counts}")

    # Calculate multiplicity weights (prioritize higher-multiplicity events)
    mult_counts = df["DispNImages"].value_counts()
    _logger.info("Training events per multiplicity:")
    for mult, count in mult_counts.items():
        _logger.info(f"  {int(mult)} telescopes: {int(count)}")

    w_multiplicity = (df["DispNImages"] ** 2).to_numpy().astype(np.float32)
    w_multiplicity /= np.mean(w_multiplicity)

    _logger.info(
        "Multiplicity weights (inverse-frequency, normalized): "
        f"mean={w_multiplicity.mean():.3f}, "
        f"std={w_multiplicity.std():.3f}, "
        f"min={w_multiplicity.min():.3f}, "
        f"max={w_multiplicity.max():.3f}"
    )

    # Combine energy and multiplicity weights
    combined_weights = w_energy * w_multiplicity

    # Normalize combined weights so mean is 1.0 to keep learning rate effective
    combined_weights = combined_weights / np.mean(combined_weights)

    _logger.info(
        f"Combined weights (energy x multiplicity): "
        f"mean={combined_weights.mean():.3f}, "
        f"std={combined_weights.std():.3f}, "
        f"min={combined_weights.min():.3f}, "
        f"max={combined_weights.max():.3f}"
    )

    return bins, dict(counts.items()), combined_weights
