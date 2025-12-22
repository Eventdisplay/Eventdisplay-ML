"""Evaluation of machine learning models for event display."""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

_logger = logging.getLogger(__name__)


def evaluate_model(model, x_test, y_test, df, x_cols, y_data, name):
    """Evaluate the trained model on the test set and log performance metrics."""
    score = model.score(x_test, y_test)
    _logger.info(f"XGBoost Multi-Target R^2 Score (Testing Set): {score:.4f}")
    y_pred = model.predict(x_test)
    mse_x = mean_squared_error(y_test["MCxoff"], y_pred[:, 0])
    mse_y = mean_squared_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MSE (X_off): {mse_x:.4f}, MSE (Y_off): {mse_y:.4f}")
    mae_x = mean_absolute_error(y_test["MCxoff"], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MAE (X_off): {mae_x:.4f}, MAE (Y_off): {mae_y:.4f}")

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


def feature_importance(model, x_cols, target_names, name=None):
    """Log feature importance from the trained XGBoost model."""
    _logger.info("--- XGBoost Multi-Regression Feature Importance ---")
    for i, estimator in enumerate(model.estimators_):
        target = target_names[i]
        _logger.info(f"\n### {name} Importance for Target: **{target}**")

        importances = estimator.feature_importances_
        importance_df = pd.DataFrame({"Feature": x_cols, "Importance": importances})

        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        _logger.info(f"\n{importance_df.head(15).to_markdown(index=False)}")


def shap_feature_importance(model, x_data, target_names, max_points=20000, n_top=25):
    """Use XGBoost's builtin SHAP."""
    x_sample = x_data.sample(n=min(len(x_data), max_points), random_state=0)
    for i, est in enumerate(model.estimators_):
        target = target_names[i]

        # Builtin XGBoost SHAP values (n_samples, n_features+1)
        # Last column is the bias term: drop it
        shap_vals = est.get_booster().predict(xgb.DMatrix(x_sample), pred_contribs=True)
        shap_vals = shap_vals[:, :-1]  # drop bias column

        # Global importance: mean(|SHAP|)
        imp = np.abs(shap_vals).mean(axis=0)
        idx = np.argsort(imp)[::-1]
        n_features = len(x_data.columns)

        _logger.info(f"\n=== Builtin XGBoost SHAP Importance for {target} ===")
        for j in idx[:n_top]:
            # Guard against mismatches between SHAP array length and feature columns
            if j >= n_features:
                continue
            _logger.info(f"{x_data.columns[j]:25s}  {imp[j]:.6e}")
