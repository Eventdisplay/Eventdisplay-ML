"""Permutation importance for stereo regression models.

This diagnostic rebuilds the held-out test split from the model metadata and the
original training input files, then shuffles features one-by-one and measures
the degradation in residual RMSE.

Usage:
    python diagnostic_permutation_importance.py \
        --model_file trained_stereo.joblib \
        --output_dir diagnostics/ \
        --top_n 20
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from eventdisplay_ml.data_processing import load_training_data

_logger = logging.getLogger(__name__)


def load_data_and_model(model_file, input_file_list=None):
    """Load trained model and rebuild the held-out test split.

    Parameters
    ----------
    model_file : str
        Path to trained model joblib file.
    input_file_list : str or None, optional
        Optional override for the input file list stored in the model metadata.

    Returns
    -------
    tuple
        Trained model, reconstructed x_test, y_test, feature names, target names,
        and full model metadata.
    """
    _logger.info(f"Loading model from {model_file}")
    model_dict = joblib.load(model_file)
    models = model_dict.get("models", {})
    if not models:
        raise ValueError(f"No models found in model file: {model_file}")

    model_cfg = next(iter(models.values()))
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
    _, x_test, _, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_dict.get("train_test_fraction", 0.5),
        random_state=model_dict.get("random_state", None),
    )

    _logger.info(f"Reconstructed test set with {len(x_test)} events")
    return model, x_test, y_test, features, targets, model_dict


def predict_unscaled_residuals(model, x_test, model_dict, target_names):
    """Predict residual targets and inverse-standardize them to original scale."""
    preds_scaled = model.predict(x_test)

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
    return pd.DataFrame(preds, columns=target_names, index=x_test.index)


def compute_baseline_rmse(model, x_test, y_test, model_dict, target_names):
    """Compute baseline RMSE on unshuffled test set."""
    y_pred = predict_unscaled_residuals(model, x_test, model_dict, target_names)

    baseline_rmse = {}
    for target in target_names:
        mse = mean_squared_error(y_test[target], y_pred[target])
        baseline_rmse[target] = np.sqrt(mse)
        _logger.info(f"  Baseline RMSE ({target}): {baseline_rmse[target]:.6f}")

    return baseline_rmse, y_pred


def permutation_importance(model, x_test, y_test, baseline_rmse, target_names, model_dict):
    """Compute permutation importance for each feature."""
    _logger.info("Computing permutation importance...")

    importance = {target: {} for target in target_names}
    rng = np.random.default_rng(model_dict.get("random_state", None))

    for feat_idx, feat_name in enumerate(x_test.columns):
        x_shuffled = x_test.copy()
        x_shuffled.iloc[:, feat_idx] = rng.permutation(x_shuffled.iloc[:, feat_idx].to_numpy())

        y_pred_shuffled = predict_unscaled_residuals(model, x_shuffled, model_dict, target_names)

        for target_name in target_names:
            mse_shuffled = mean_squared_error(y_test[target_name], y_pred_shuffled[target_name])
            rmse_shuffled = np.sqrt(mse_shuffled)

            # Importance = relative RMSE increase when feature is shuffled
            relative_importance = (rmse_shuffled - baseline_rmse[target_name]) / baseline_rmse[
                target_name
            ]
            importance[target_name][feat_name] = relative_importance

    return importance


def plot_permutation_importance(importance, output_path, top_n=20):
    """Plot permutation importance for each target."""
    _logger.info(f"Creating permutation importance plots (top {top_n})...")

    targets = list(importance.keys())
    _, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 8))
    if len(targets) == 1:
        axes = [axes]

    for ax, target_name in zip(axes, targets):
        imp_df = (
            pd.DataFrame(
                list(importance[target_name].items()),
                columns=["feature", "importance"],
            )
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )
        colors = ["red" if x < 0 else "green" for x in imp_df["importance"]]

        ax.barh(imp_df["feature"], imp_df["importance"] * 100.0, color=colors, alpha=0.7)
        ax.set_xlabel("Relative RMSE Increase (%)")
        ax.set_title(f"Permutation Importance\n{target_name}")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    _logger.info(f"Saved permutation importance plot to {output_path}")
    plt.close()


def diagnose_baseline_anchoring(importance):
    """Check if model is anchored in conventional baselines."""
    _logger.info("=== Physics Check: Baseline Anchoring ===")

    baseline_features = ["Xoff_weighted_bdt", "Yoff_weighted_bdt", "ErecS"]

    for target, imp_dict in importance.items():
        _logger.info(f"\n{target}:")

        baseline_contrib = sum(
            imp_dict.get(feat, 0) for feat in baseline_features if feat in imp_dict
        )
        total_contrib = sum(imp for imp in imp_dict.values() if imp > 0)

        anchor_pct = (baseline_contrib / total_contrib * 100) if total_contrib > 0 else 0

        _logger.info(f"  Baseline features contribution: {anchor_pct:.1f}%")
        _logger.info("  (Expect >70% for well-anchored model)")

        for feat in baseline_features:
            if feat in imp_dict:
                _logger.info(f"    {feat}: {imp_dict[feat] * 100:.2f}%")


def main():
    """Rebuild the held-out test split and compute permutation importance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True, help="Path to trained model joblib file")
    parser.add_argument(
        "--input_file_list",
        default=None,
        help=(
            "Optional override for the training input file list. If omitted, the path stored "
            "in the model file is used."
        ),
    )
    parser.add_argument("--output_dir", default="diagnostics", help="Output directory")
    parser.add_argument("--top_n", type=int, default=20, help="Top N features to plot")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, x_test, y_test, _features, target_names, model_dict = load_data_and_model(
        args.model_file,
        args.input_file_list,
    )

    _logger.info("Computing baseline RMSE...")
    baseline_rmse, _ = compute_baseline_rmse(model, x_test, y_test, model_dict, target_names)

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        baseline_rmse,
        target_names,
        model_dict,
    )

    output_path = Path(args.output_dir) / "permutation_importance.png"
    plot_permutation_importance(importance, output_path, args.top_n)

    diagnose_baseline_anchoring(importance)


if __name__ == "__main__":
    main()
