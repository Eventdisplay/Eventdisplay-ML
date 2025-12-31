"""Evaluation of machine learning models for event display."""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

    from sklearn.metrics import classification_report, confusion_matrix

    _logger.info(f"--- Confusion Matrix for {name} ---")
    cm = confusion_matrix(y_test, y_pred)
    _logger.info(f"\n{cm}")

    _logger.info(f"--- Classification Report for {name} ---")
    report = classification_report(y_test, y_pred, digits=4)
    _logger.info(f"\n{report}")

    feature_importance(model, x_cols, ["label"], name)
    if name == "xgboost":
        shap_feature_importance(model, x_test, ["label"])


def evaluate_regression_model(model, x_test, y_test, df, x_cols, y_data, name):
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

    df_pred = pd.DataFrame(y_pred, columns=target_features("stereo_analysis"))
    calculate_resolution(
        df_pred,
        y_test,
        df,
        percentiles=[68, 90, 95],
        log_e_min=-1,
        log_e_max=2,
        n_bins=6,
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
        unexplained = mse_values[i] / variance_values[i]

        _logger.info(
            f"Target: {name:12s} | MSE: {mse_values[i]:.6f} | "
            f"Unexplained Variance: {unexplained:.2%}"
        )


def calculate_resolution(y_pred, y_test, df, percentiles, log_e_min, log_e_max, n_bins, name):
    """Compute angular and energy resolution based on predictions."""
    results_df = pd.DataFrame(
        {
            "MCxoff_true": y_test["MCxoff"].values,
            "MCyoff_true": y_test["MCyoff"].values,
            "MCxoff_pred": y_pred["MCxoff"].values,
            "MCyoff_pred": y_pred["MCyoff"].values,
            "MCe0_pred": y_pred["MCe0"].values,
            "MCe0": df.loc[y_test.index, "MCe0"].values,
        }
    )

    results_df["DeltaTheta"] = np.sqrt(
        (results_df["MCxoff_true"] - results_df["MCxoff_pred"]) ** 2
        + (results_df["MCyoff_true"] - results_df["MCyoff_pred"]) ** 2
    )
    results_df["DeltaMCe0"] = np.abs(
        np.power(10, results_df["MCe0_pred"]) - np.power(10, results_df["MCe0"])
    ) / np.power(10, results_df["MCe0"])

    results_df["LogE"] = results_df["MCe0"]
    bins = np.linspace(log_e_min, log_e_max, n_bins + 1)
    results_df["E_bin"] = pd.cut(results_df["LogE"], bins=bins, include_lowest=True)
    results_df.dropna(subset=["E_bin"], inplace=True)

    g = results_df.groupby("E_bin", observed=False)
    mean_loge_by_bin = g["LogE"].mean().round(3)

    def percentile_series(col, p):
        return g[col].quantile(p / 100)

    for col, label in [("DeltaTheta", "Theta"), ("DeltaMCe0", "DeltaE")]:
        data = {f"{label}_{p}%": percentile_series(col, p).values for p in percentiles}

        output_df = pd.DataFrame(data, index=mean_loge_by_bin.index)
        output_df.insert(0, "Mean Log10(E)", mean_loge_by_bin.values)
        output_df.index.name = "Log10(E) Bin Range"
        output_df = output_df.dropna()

        _logger.info(f"--- {name} {col} Resolution vs. Log10(MCe0) ---")
        _logger.info(
            f"Calculated over {n_bins} bins between Log10(E) = {log_e_min} and {log_e_max}"
        )
        _logger.info(f"\n{output_df.to_markdown(floatfmt='.4f')}")


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


def shap_feature_importance(model, x_data, target_names, max_points=20000, n_top=25):
    """Feature importance using SHAP values for native multi-target XGBoost."""
    x_sample = x_data.sample(n=min(len(x_data), max_points), random_state=42)
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
