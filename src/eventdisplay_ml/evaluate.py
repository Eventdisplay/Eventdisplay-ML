"""Evaluation of machine learning models for event display."""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

_logger = logging.getLogger(__name__)


def write_efficiency_csv(model, x_test, y_test, output_file):
    """Write signal and background efficiency as a function of threshold to CSV."""
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

    pd.DataFrame(
        {
            "threshold": thresholds,
            "signal_efficiency": eff_signal,
            "background_efficiency": eff_background,
        }
    ).to_csv(output_file, index=False)

    _logger.info(f"Wrote signal and background efficiency CSV files to {output_file}")


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
    mse_x = mean_squared_error(y_test["MCxoff"], y_pred[:, 0])
    mse_y = mean_squared_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MSE (X_off): {mse_x:.4f}, MSE (Y_off): {mse_y:.4f}")
    mae_x = mean_absolute_error(y_test["MCxoff"], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MAE (X_off): {mae_x:.4f}")
    _logger.info(f"{name} MAE (Y_off): {mae_y:.4f}")

    feature_importance(model, x_cols, y_data.columns, name)
    if name == "xgboost":
        shap_feature_importance(model, x_test, y_data.columns)

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=[68, 90, 95],
        log_e_min=-1,
        log_e_max=2,
        n_bins=6,
        name=name,
    )


def calculate_resolution(y_pred, y_test, df, percentiles, log_e_min, log_e_max, n_bins, name):
    """Compute angular and energy resolution based on predictions."""
    results_df = pd.DataFrame(
        {
            "MCxoff_true": y_test["MCxoff"].values,
            "MCyoff_true": y_test["MCyoff"].values,
            "MCxoff_pred": y_pred[:, 0],
            "MCyoff_pred": y_pred[:, 1],
            "MCe0_pred": y_pred[:, 2],
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


def _iter_targets(model, target_names):
    """Iterate over targets in multi-/single-output models."""
    if hasattr(model, "estimators_"):  # MultiOutputRegressor
        for i, est in enumerate(model.estimators_):
            target = target_names[i] if i < len(target_names) else f"target_{i}"
            yield target, est
    else:
        target = target_names[0] if target_names else "target"
        yield target, model


def feature_importance(model, x_cols, target_names, name=None):
    """Feature importance using built-in XGBoost method."""
    _logger.info("--- XGBoost Feature Importance ---")

    for target, est in _iter_targets(model, target_names):
        importances = getattr(est, "feature_importances_", None)
        if importances is None:
            _logger.info("No feature_importances_ found.")
            continue

        df = pd.DataFrame({"Feature": x_cols, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )
        _logger.info(f"\n### {name} Importance for Target: **{target}**")
        _logger.info(f"\n{df.head(15).to_markdown(index=False)}")


def shap_feature_importance(model, x_data, target_names, max_points=20000, n_top=25):
    """Feature importance using SHAP values from XGBoost."""
    x_sample = x_data.sample(n=min(len(x_data), max_points), random_state=0)
    n_features = len(x_data.columns)

    for target, est in _iter_targets(model, target_names):
        if not hasattr(est, "get_booster"):
            _logger.info("Model does not support SHAP feature importance.")
            continue

        shap_vals = est.get_booster().predict(xgb.DMatrix(x_sample), pred_contribs=True)[:, :-1]

        imp = np.abs(shap_vals).mean(axis=0)
        idx = np.argsort(imp)[::-1]

        _logger.info(f"=== Builtin XGBoost SHAP Importance for {target} ===")
        for j in idx[:n_top]:
            if j < n_features:
                _logger.info(f"{x_data.columns[j]:25s}  {imp[j]:.6e}")
