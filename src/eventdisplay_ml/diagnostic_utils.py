"""Utilities for inspecting cached diagnostic data in model joblib files.

The current stereo regression cache stores per-target SHAP importances under
``models[<name>]["shap_importance"]``. This module supports that layout while
remaining compatible with older cache keys where possible.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from eventdisplay_ml.data_processing import load_training_data

_logger = logging.getLogger(__name__)


def _load_model_cfg(model_file):
    """Load full model dictionary and the first model configuration entry."""
    model_dict = joblib.load(model_file)
    models = model_dict.get("models", {})
    model_cfg = next(iter(models.values())) if models else None
    return model_dict, model_cfg


def load_stereo_regression_split(model_file, input_file_list=None):
    """Load a stereo regression model and reconstruct its train/test split.

    Parameters
    ----------
    model_file : str
        Path to trained model joblib file.
    input_file_list : str or None, optional
        Optional override for the input file list stored in the model metadata.

    Returns
    -------
    tuple
        Trained model, reconstructed x_train, y_train, x_test, y_test,
        feature names, target names, and full model metadata.
    """
    _logger.info(f"Loading model from {model_file}")
    model_dict, model_cfg = _load_model_cfg(model_file)

    if model_cfg is None:
        raise ValueError(f"No models found in model file: {model_file}")

    model = model_cfg.get("model")
    if model is None:
        raise ValueError(f"No trained model object found in model file: {model_file}")

    file_list = input_file_list or model_dict.get("input_file_list")
    if not file_list:
        raise ValueError(
            "No input file list available. Provide --input_file_list or retrain with "
            "input_file_list stored in the model metadata."
        )

    _logger.info(f"Rebuilding training data from input file list: {file_list}")
    df = load_training_data(model_dict, file_list, "stereo_analysis")

    features = model_cfg.get("features") or model_dict.get("features", [])
    targets = model_dict.get("targets", ["Xoff_residual", "Yoff_residual", "E_residual"])
    if not features:
        raise ValueError(f"No feature list found in model file: {model_file}")

    x_data = df[features]
    y_data = df[targets]

    _logger.info("Reconstructing train/test split from model metadata")
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_dict.get("train_test_fraction", 0.5),
        random_state=model_dict.get("random_state", None),
    )

    _logger.info(
        "Reconstructed split with %d training and %d test events",
        len(x_train),
        len(x_test),
    )
    return model, x_train, y_train, x_test, y_test, features, targets, model_dict


def predict_unscaled_residuals(model, x_data, model_dict, target_names):
    """Predict residual targets and inverse-standardize them to original scale."""
    preds_scaled = model.predict(x_data)

    target_mean_cfg = model_dict.get("target_mean")
    target_std_cfg = model_dict.get("target_std")
    if not target_mean_cfg or not target_std_cfg:
        raise ValueError(
            "Missing target standardization parameters (target_mean/target_std) in model "
            "file. This diagnostic requires a residual-trained stereo model."
        )

    target_mean = np.array([target_mean_cfg[target] for target in target_names], dtype=np.float64)
    target_std = np.array([target_std_cfg[target] for target in target_names], dtype=np.float64)

    preds = preds_scaled * target_std + target_mean
    return pd.DataFrame(preds, columns=target_names, index=x_data.index)


def compute_generalization_metrics(y_train, y_train_pred, y_test, y_test_pred, target_names):
    """Compute train/test RMSE and relative generalization gap per target."""
    metrics = {}

    for target_name in target_names:
        rmse_train = np.sqrt(mean_squared_error(y_train[target_name], y_train_pred[target_name]))
        rmse_test = np.sqrt(mean_squared_error(y_test[target_name], y_test_pred[target_name]))

        if rmse_train == 0:
            gap_pct = 0.0 if rmse_test == 0 else np.inf
        else:
            gap_pct = (rmse_test - rmse_train) / rmse_train * 100

        if np.isfinite(gap_pct) and gap_pct > 0:
            gen_ratio = rmse_train / gap_pct
        elif gap_pct <= 0:
            gen_ratio = 999.0
        else:
            gen_ratio = 0.0

        metrics[target_name] = {
            "rmse_train": float(rmse_train),
            "rmse_test": float(rmse_test),
            "gap_pct": float(gap_pct),
            "gen_ratio": float(gen_ratio),
        }

    return metrics


def load_cached_generalization_metrics(model_file):
    """Load cached train/test RMSE summary from a model file if available."""
    _logger.info(f"Loading cached generalization metrics from {model_file}")
    model_dict, model_cfg = _load_model_cfg(model_file)

    if model_cfg is None:
        _logger.warning("No models found in model file")
        return model_dict, None

    metrics = model_cfg.get("generalization_metrics")
    if not isinstance(metrics, dict) or not metrics:
        _logger.warning("No cached generalization metrics found in model file")
        return model_dict, None

    _logger.info("Loaded cached generalization metrics for %d targets", len(metrics))
    return model_dict, metrics


def compute_residual_normality_stats(y_test, y_test_pred, target_names):
    """Compute Gaussian fit parameters and normality tests for residuals."""
    stats_dict = {}

    for target_name in target_names:
        residuals = y_test[target_name].values - y_test_pred[target_name].values
        residuals_clean = residuals[~np.isnan(residuals)]

        if len(residuals_clean) == 0:
            _logger.warning(f"Skipping {target_name}: no finite residuals")
            continue

        # Gaussian parameters
        mean = float(np.mean(residuals_clean))
        std = float(np.std(residuals_clean))

        # Normality tests
        _, p_ks = stats.kstest(residuals_clean, "norm", args=(mean, std))
        ad_result = stats.anderson(residuals_clean, dist="norm")
        ad_stat = float(ad_result.statistic)
        ad_crit_5 = float(
            ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else np.nan
        )

        # Skewness and kurtosis
        skewness = float(stats.skew(residuals_clean))
        kurtosis = float(stats.kurtosis(residuals_clean))

        # Quantile-Quantile test (visual)
        _, (_, _, qq_r) = stats.probplot(residuals_clean, dist="norm")
        qq_r2 = float(qq_r**2)

        # Outlier count
        n_outliers = int(np.sum(np.abs(residuals_clean) > 3 * std))

        stats_dict[target_name] = {
            "mean": mean,
            "std": std,
            "p_ks": float(p_ks),
            "ad_stat": ad_stat,
            "ad_crit_5": ad_crit_5,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "qq_r2": qq_r2,
            "n_outliers": n_outliers,
            "n_samples": len(residuals_clean),
        }

    return stats_dict


def load_cached_residual_normality_stats(model_file):
    """Load cached residual normality statistics from a model file if available."""
    _logger.info(f"Loading cached residual normality stats from {model_file}")
    model_dict, model_cfg = _load_model_cfg(model_file)

    if model_cfg is None:
        _logger.warning("No models found in model file")
        return model_dict, None

    normality_stats = model_cfg.get("residual_normality_stats")
    if not isinstance(normality_stats, dict) or not normality_stats:
        _logger.warning("No cached residual normality statistics found in model file")
        return model_dict, None

    _logger.info("Loaded cached residual normality stats for %d targets", len(normality_stats))
    return model_dict, normality_stats


def load_model_and_importance(model_file, target_name=None):
    """
    Load model dict and precomputed feature importance.

    Parameters
    ----------
    model_file : str
        Path to joblib model file.

    Returns
    -------
    dict
        Full model dictionary with model metadata.
    dict or None
        Precomputed feature importances {feature_name: importance_value} for
        the selected target.
    """
    _logger.info(f"Loading model from {model_file}")

    model_dict, model_cfg = _load_model_cfg(model_file)

    if model_cfg is None:
        _logger.warning("No models found in model file")
        return model_dict, None

    shap_importance = model_cfg.get("shap_importance")
    features = model_cfg.get("features") or model_dict.get("features", [])

    if shap_importance is None:
        # Backward compatibility for legacy key if present.
        shap_importance = model_cfg.get("feature_importances")

    if shap_importance is None or not features:
        _logger.warning("No cached feature importances found in model file")
        return model_dict, None

    if isinstance(shap_importance, dict):
        if not shap_importance:
            _logger.warning("Cached SHAP importance dictionary is empty")
            return model_dict, None

        selected_target = target_name or next(iter(shap_importance))
        importances = shap_importance.get(selected_target)
        if importances is None:
            _logger.warning(
                "Target %r not found in cached SHAP importance; available targets: %s",
                selected_target,
                list(shap_importance.keys()),
            )
            return model_dict, None

        importance_dict = dict(zip(features, importances, strict=False))
        _logger.info(
            "Loaded cached SHAP importances for target %s (%d features)",
            selected_target,
            len(importance_dict),
        )
        return model_dict, importance_dict

    importance_dict = dict(zip(features, shap_importance, strict=False))
    _logger.info("Loaded cached feature importances (%d features)", len(importance_dict))
    return model_dict, importance_dict


def get_cached_shap_explainer(model_file):
    """
    Retrieve cached SHAP explainer if available.

    Parameters
    ----------
    model_file : str
        Path to joblib model file.

    Returns
    -------
    shap.TreeExplainer or None
        Cached SHAP explainer, or None if not available.
    """
    _logger.info(f"Loading SHAP explainer from {model_file}")
    _, model_cfg = _load_model_cfg(model_file)

    if model_cfg is None:
        return None

    explainer = model_cfg.get("shap_explainer")
    if explainer is not None:
        _logger.info("Successfully loaded cached SHAP explainer")
        return explainer

    _logger.warning("No cached SHAP explainer found. Will compute on-the-fly.")
    return None


def importance_dataframe(model_file, top_n=25, target_name=None):
    """
    Get feature importance as a sorted pandas DataFrame.

    Parameters
    ----------
    model_file : str
        Path to joblib model file.
    top_n : int, optional
        Return only top N features by importance (default 25).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["Feature", "Importance"] sorted by importance.
    """
    _, importance_dict = load_model_and_importance(model_file, target_name=target_name)

    if importance_dict is None:
        _logger.error("Cannot create importance dataframe: no importances found")
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Feature": list(importance_dict.keys()),
            "Importance": list(importance_dict.values()),
        }
    ).sort_values("Importance", ascending=False)

    if top_n:
        df = df.head(top_n)

    return df


def validate_cached_data(model_file):
    """
    Check what data is cached in the model file.

    Parameters
    ----------
    model_file : str
        Path to joblib model file.

    Returns
    -------
    dict
        Summary of cached data availability.
    """
    model_dict, model_cfg = _load_model_cfg(model_file)
    model_cfg = model_cfg or {}

    shap_importance = model_cfg.get("shap_importance")
    has_shap_importance = shap_importance is not None
    shap_targets = list(shap_importance.keys()) if isinstance(shap_importance, dict) else []

    summary = {
        "has_model": "model" in model_cfg,
        "has_features": "features" in model_cfg,
        "has_shap_importance": has_shap_importance,
        "has_generalization_metrics": "generalization_metrics" in model_cfg,
        "has_residual_normality_stats": "residual_normality_stats" in model_cfg,
        "has_feature_importances": "feature_importances" in model_cfg,  # legacy key
        "has_shap_explainer": "shap_explainer" in model_cfg,
        "has_target_mean": "target_mean" in model_dict,
        "has_target_std": "target_std" in model_dict,
        "n_features": len(model_cfg.get("features", [])),
        "n_targets_with_shap": len(shap_targets),
        "shap_targets": shap_targets,
        "generalization_targets": list(model_cfg.get("generalization_metrics", {}).keys()),
        "residual_normality_targets": list(model_cfg.get("residual_normality_stats", {}).keys()),
        "n_importances": len(model_cfg.get("feature_importances", []))
        if "feature_importances" in model_cfg
        else 0,
    }

    if isinstance(shap_importance, dict):
        summary["n_importances_per_target"] = {
            target: len(values) for target, values in shap_importance.items()
        }
    else:
        summary["n_importances_per_target"] = {}

    return summary
