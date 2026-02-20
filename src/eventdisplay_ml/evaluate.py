"""Evaluation of machine learning models for event display."""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from eventdisplay_ml.features import target_features

_logger = logging.getLogger(__name__)


def evaluation_efficiency(name, model, x_test, y_test):
    """Calculate signal and background efficiency as a function of threshold."""
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    thresholds = np.linspace(0, 1, 101)

    n_signal = (y_test == 1).sum()
    n_background = (y_test == 0).sum()

    eff_signal = []
    eff_background = []

    for t in thresholds:
        pred = y_pred_proba >= t
        eff_signal.append(((pred) & (y_test == 1)).sum() / n_signal if n_signal else 0)
        eff_background.append(((pred) & (y_test == 0)).sum() / n_background if n_background else 0)
        _logger.info(
            f"{name} Threshold: {t:.2f} | "
            f"Signal Efficiency: {eff_signal[-1]:.4f} | "
            f"Background Efficiency: {eff_background[-1]:.4f}"
        )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "signal_efficiency": eff_signal,
            "background_efficiency": eff_background,
        }
    )


def evaluate_classification_model(model, x_test, y_test, df, x_cols, name):
    """Evaluate the trained model on the test set and log performance metrics."""
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = (y_pred == y_test).mean()
    _logger.info(f"XGBoost Classification Accuracy (Testing Set): {accuracy:.4f}")

    _logger.info(f"--- Confusion Matrix for {name} ---")
    cm = confusion_matrix(y_test, y_pred)
    _logger.info(f"\n{cm}")

    _logger.info(f"--- Classification Report for {name} ---")
    report = classification_report(y_test, y_pred, digits=4)
    _logger.info(f"\n{report}")

    feature_importance(model, x_cols, ["label"], name)
    if name == "xgboost":
        shap_feature_importance(model, x_test, ["label"])


def evaluate_regression_model(
    model, x_test, y_test, df, x_cols, y_data, name, shap_per_energy=False
):
    """Evaluate the trained model on the test set and log performance metrics."""
    score = model.score(x_test, y_test)
    _logger.info(f"XGBoost Multi-Target R^2 Score (Testing Set): {score:.4f}")
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    _logger.info(f"{name} Mean Squared Error (All targets): {mse:.4f}")
    mae = mean_absolute_error(y_test, y_pred)
    _logger.info(f"{name} Mean Absolute Error (All targets): {mae:.4f}")

    target_variance(y_test, y_pred, y_data.columns)
    feature_importance(model, x_cols, y_data.columns, name)
    if name == "xgboost":
        shap_feature_importance(model, x_test, y_data.columns)
        if shap_per_energy:
            shap_feature_importance_by_energy(model, x_test, df, y_test, y_data.columns)

    df_pred = pd.DataFrame(y_pred, columns=target_features("stereo_analysis"))
    calculate_resolution(
        df_pred,
        y_test,
        df,
        percentiles=[68, 90, 95],
        log_e_min=-2,
        log_e_max=2.5,
        n_bins=9,
        name=name,
    )


def target_variance(y_test, y_pred, targets):
    """Calculate and log variance explained per target."""
    y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

    mse_values = np.mean((y_test_np - y_pred) ** 2, axis=0)
    variance_values = np.var(y_test_np, axis=0)

    _logger.info("--- Performance Per Target ---")
    for i, name in enumerate(targets):
        # Fraction of variance unexplained (lower is better, 0.0 is perfect)
        if variance_values[i] != 0:
            unexplained = mse_values[i] / variance_values[i]
        else:
            unexplained = np.nan
            _logger.warning(
                "Target '%s' has zero variance in the test set; unexplained variance is undefined.",
                name,
            )

        _logger.info(
            f"Target: {name:12s} | MSE: {mse_values[i]:.6f} | "
            f"Unexplained Variance: {unexplained:.2%}"
        )


def calculate_resolution(y_pred, y_test, df, percentiles, log_e_min, log_e_max, n_bins, name):
    """Compute angular and energy resolution based on predictions."""
    # Model predicts residuals, so reconstruct full predictions and MC truth
    # from residuals and DispBDT baseline
    _logger.debug(
        f"Evaluation: y_test indices min={y_test.index.min()}, max={y_test.index.max()}, len={len(y_test)}"
    )
    _logger.debug(
        f"Evaluation: df shape={df.shape}, index min={df.index.min()}, max={df.index.max()}"
    )

    disp_xoff = df.loc[y_test.index, "Xoff_weighted_bdt"].values
    disp_yoff = df.loc[y_test.index, "Yoff_weighted_bdt"].values

    # Handle ErecS with proper checks for valid values
    erec_s = df.loc[y_test.index, "ErecS"].values
    disp_erec_log = np.where(erec_s > 0, np.log10(erec_s), np.nan)

    # Reconstruct MC truth from residuals in y_test (residual = MC_true - DispBDT)
    mc_xoff_true = y_test["Xoff_residual"].values + disp_xoff
    mc_yoff_true = y_test["Yoff_residual"].values + disp_yoff
    mc_e0_true = y_test["E_residual"].values + disp_erec_log

    # Reconstruct predictions from residual predictions
    mc_xoff_pred = y_pred["Xoff_residual"].values + disp_xoff
    mc_yoff_pred = y_pred["Yoff_residual"].values + disp_yoff
    mc_e0_pred = y_pred["E_residual"].values + disp_erec_log

    results_df = pd.DataFrame(
        {
            "MCxoff_true": mc_xoff_true,
            "MCyoff_true": mc_yoff_true,
            "MCxoff_pred": mc_xoff_pred,
            "MCyoff_pred": mc_yoff_pred,
            "MCe0_pred": mc_e0_pred,
            "MCe0": mc_e0_true,
        }
    )

    # Optional previous method columns
    for col in ["Xoff_weighted_bdt", "Yoff_weighted_bdt", "ErecS"]:
        if col in df.columns:
            results_df[col] = df.loc[y_test.index, col].values

    # Calculate angular resolution for BDT prediction
    results_df["DeltaTheta"] = np.hypot(
        results_df["MCxoff_true"] - results_df["MCxoff_pred"],
        results_df["MCyoff_true"] - results_df["MCyoff_pred"],
    )

    # Calculate angular resolution for previous method (weighted_bdt)
    if "Xoff_weighted_bdt" in results_df.columns:
        results_df["DeltaTheta_weighted"] = np.hypot(
            results_df["MCxoff_true"] - results_df["Xoff_weighted_bdt"],
            results_df["MCyoff_true"] - results_df["Yoff_weighted_bdt"],
        )

    # Energy resolutions
    def rel_error(pred_col):
        return (
            np.abs(10 ** results_df[pred_col] - 10 ** results_df["MCe0"]) / 10 ** results_df["MCe0"]
        )

    results_df["DeltaMCe0"] = rel_error("MCe0_pred")
    if "ErecS" in results_df.columns:
        results_df["DeltaMCe0_ErecS"] = rel_error("ErecS")

    # Bin by LogE
    results_df["LogE"] = results_df["MCe0"]
    bins = np.linspace(log_e_min, log_e_max, n_bins + 1)
    results_df["E_bin"] = pd.cut(results_df["LogE"], bins=bins, include_lowest=True)
    results_df.dropna(subset=["E_bin"], inplace=True)
    g = results_df.groupby("E_bin", observed=False)
    mean_loge_by_bin = g["LogE"].mean().round(3)

    def log_percentiles(col, label, method):
        data = {f"{label}_{p}%": g[col].quantile(p / 100).values for p in percentiles}
        df_out = pd.DataFrame(data, index=mean_loge_by_bin.index)
        df_out.insert(0, "Mean Log10(E)", mean_loge_by_bin.values)
        df_out.index.name = "Log10(E) Bin Range"
        df_out = df_out.dropna()
        _logger.info(f"--- {method} vs Log10(MCe0) ---")
        _logger.info(f"Calculated over {n_bins} bins [{log_e_min}, {log_e_max}]")
        _logger.info(f"\n{df_out.to_markdown(floatfmt='.4f')}")

    # Compute and log percentiles for angular and energy resolutions
    for col, label, method in [
        ("DeltaTheta", "Theta", f"{name} (BDT)"),
        ("DeltaTheta_weighted", "Theta", "Previous (weighted_bdt)"),
    ]:
        if col in results_df.columns:
            log_percentiles(col, label, method)

    for col, label, method in [
        ("DeltaMCe0", "DeltaE", f"{name} (BDT)"),
        ("DeltaMCe0_ErecS", "DeltaE", "Previous (ErecS)"),
    ]:
        if col in results_df.columns:
            log_percentiles(col, label, method)


def feature_importance(model, x_cols, target_names, name=None):
    """Feature importance handling both MultiOutputRegressor and native Multi-target."""
    _logger.info("--- XGBoost Feature Importance ---")

    # Case 1: Scikit-Learn MultiOutputRegressor
    if hasattr(model, "estimators_"):
        for i, est in enumerate(model.estimators_):
            target = target_names[i] if (target_names and i < len(target_names)) else f"target_{i}"
            _log_importance_table(target, est.feature_importances_, x_cols, name)

    # Case 2: Native Multi-target OR Single-target Classifier
    else:
        importances = getattr(model, "feature_importances_", None)

        if importances is not None:
            if target_names is not None and len(target_names) > 0:
                # Convert to list to ensure .join works regardless of input type
                target_str = ", ".join(map(str, target_names))
            else:
                target_str = "Target"

            # Check if it's actually multi-target to set the log message
            if target_names is not None and len(target_names) > 1:
                _logger.info("Note: Native XGBoost multi-target provides JOINT importance.")

            _log_importance_table(target_str, importances, x_cols, name)


def _log_importance_table(target_label, values, x_cols, name):
    """Format and log the importance dataframe for printing."""
    df = pd.DataFrame({"Feature": x_cols, "Importance": values}).sort_values(
        "Importance", ascending=False
    )
    _logger.info(f"### {name} Importance for: **{target_label}**")
    _logger.info(f"\n{df.head(25).to_markdown(index=False)}")


def shap_feature_importance(model, x_data, target_names, max_points=1000, n_top=25):
    """Feature importance using SHAP values for native multi-target XGBoost."""
    x_sample = x_data.sample(n=min(len(x_data), max_points), random_state=None)
    n_features = len(x_data.columns)
    n_targets = len(target_names)

    dmatrix = xgb.DMatrix(x_sample)
    shap_vals = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = shap_vals.reshape(len(x_sample), n_targets, n_features + 1)

    for i, target in enumerate(target_names):
        target_shap = shap_vals[:, i, :-1]

        imp = np.abs(target_shap).mean(axis=0)
        idx = np.argsort(imp)[::-1]

        _logger.info(f"=== SHAP Importance for {target} ===")
        for j in idx[:n_top]:
            if j < n_features:
                _logger.info(f"{x_data.columns[j]:25s}  {imp[j]:.6e}")


def shap_feature_importance_by_energy(
    model,
    x_test,
    df,
    y_test,
    target_names,
    log_e_min=-2.0,
    log_e_max=2.5,
    n_bins=9,
    max_points=1000,
    n_top=5,
):
    """Calculate SHAP feature importance for each energy bin.

    Computes SHAP values separately for events in different energy ranges,
    allowing analysis of feature importance as a function of energy.
    Uses the same energy binning as calculate_resolution for consistency.
    Outputs results in tabular format for easy comparison across energy bins.
    """
    # Extract energy values and create bins
    mce0_values = df.loc[y_test.index, "MCe0"].values
    bins = np.linspace(log_e_min, log_e_max, n_bins + 1)
    # Use pd.cut with include_lowest=True to match calculate_resolution binning
    bin_categories = pd.cut(mce0_values, bins=bins, include_lowest=True, right=True)
    # Convert categorical bins to 1-based integer indices (NaN -> code -1, becomes 0)
    bin_indices = bin_categories.cat.codes + 1

    n_features = len(x_test.columns)
    n_targets = len(target_names)

    # Store importance values for each target across all bins
    target_importance_data = {target: {} for target in target_names}
    bin_info = []

    # Collect stratified samples for all bins, then compute SHAP once
    sampled_frames = []
    sampled_bin_labels = []

    for bin_idx in range(1, n_bins + 1):
        mask = bin_indices == bin_idx
        n_events = mask.sum()

        if n_events == 0:
            continue

        bin_lower = bins[bin_idx - 1]
        bin_upper = bins[bin_idx]
        mean_log_e = mce0_values[mask].mean()

        # Use a stable, unique bin label based on the explicit energy range
        bin_label = f"[{bin_lower:.2f}, {bin_upper:.2f}]"
        bin_info.append(
            {
                "label": bin_label,
                "mean_log_e": mean_log_e,
                "n_events": n_events,
                "range": bin_label,
            }
        )

        x_bin = x_test.iloc[mask]
        n_sample = min(len(x_bin), max_points)
        x_sample = x_bin.sample(n=n_sample, random_state=None)

        sampled_frames.append(x_sample)
        sampled_bin_labels.extend([bin_label] * len(x_sample))

    if not sampled_frames:
        _logger.info("No events found in any energy bin for SHAP calculation.")
        return

    x_sampled_all = pd.concat(sampled_frames, axis=0)
    dmatrix = xgb.DMatrix(x_sampled_all)
    shap_vals = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = shap_vals.reshape(len(x_sampled_all), n_targets, n_features + 1)

    # Aggregate SHAP importance per bin from the single SHAP run
    sampled_bin_labels = np.array(sampled_bin_labels)
    for i, target in enumerate(target_names):
        target_shap = shap_vals[:, i, :-1]
        for info in bin_info:
            bin_label = info["label"]
            bin_mask = sampled_bin_labels == bin_label
            if not np.any(bin_mask):
                continue

            imp = np.abs(target_shap[bin_mask]).mean(axis=0)
            for j, feature_name in enumerate(x_test.columns):
                if feature_name not in target_importance_data[target]:
                    target_importance_data[target][feature_name] = {}
                target_importance_data[target][feature_name][bin_label] = imp[j]

    # Create and display tables for each target
    _logger.info(f"\n{'=' * 100}")
    _logger.info("SHAP Feature Importance by Energy Bin (Tabular Format)")
    _logger.info(f"Calculated over {n_bins} bins [{log_e_min}, {log_e_max}]")
    _logger.info(f"{'=' * 100}")

    # Display bin information
    _logger.info("\nEnergy Bin Information:")
    for info in bin_info:
        _logger.info(f"  {info['label']:12s}: Range {info['range']:15s}, N = {info['n_events']:6d}")

    for target in target_names:
        _logger.info(f"\n\n=== SHAP Importance for {target} ===")

        # Find top N features in each bin, then take union of all top features
        all_top_features = set()
        for info in bin_info:
            bin_label = info["label"]
            # Get importance values for this bin
            bin_importance = {
                feature: values.get(bin_label, 0)
                for feature, values in target_importance_data[target].items()
            }
            # Get top N features for this bin
            top_in_bin = sorted(bin_importance.items(), key=lambda x: x[1], reverse=True)[:n_top]
            all_top_features.update([f[0] for f in top_in_bin])

        # Sort features by their average importance across all bins
        feature_avg_importance = {}
        for feature_name in all_top_features:
            values = [
                target_importance_data[target][feature_name].get(info["label"], 0)
                for info in bin_info
            ]
            feature_avg_importance[feature_name] = np.mean(values)

        sorted_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)

        # Build DataFrame with all features that were top N in at least one bin
        data_rows = []
        for feature_name, _ in sorted_features:
            row = {"Feature": feature_name}
            for info in bin_info:
                bin_label = info["label"]
                value = target_importance_data[target][feature_name].get(bin_label, np.nan)
                row[bin_label] = value
            data_rows.append(row)

        df_table = pd.DataFrame(data_rows)
        _logger.info(f"\n{df_table.to_markdown(index=False, floatfmt='.4e')}")
