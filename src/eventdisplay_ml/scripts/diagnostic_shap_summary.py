#!/usr/bin/env python3
r"""SHAP Feature Importance: Show cached feature importances from training.

Displays the top 20 features for each reconstruction target (Xoff, Yoff, Energy)
using XGBoost native feature importances cached during training.

This script requires no test data - it reads directly from the cached importance
values stored in the model file during training.

Usage:
    python diagnostic_shap_summary.py \\
        --model_file stereo_model_Xoff_residual.joblib \\
        --output_dir diagnostics/
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def load_model_config(model_file):
    """Load model configuration with cached feature importances."""
    _logger.info(f"Loading model from {model_file}")
    model_dict = joblib.load(model_file)
    models = model_dict.get("models", {})
    model_cfg = next(iter(models.values()))

    return model_cfg, model_dict


def plot_feature_importance(features, importances, target_name, output_dir):
    """Create feature importance bar plot for a single target.

    Parameters
    ----------
    features : list
        Feature names.
    importances : array
        Importance values.
    target_name : str
        Name of the target (e.g., "Xoff_residual").
    output_dir : str
        Output directory for plot.
    """
    # Create DataFrame and sort by importance
    importance_df = (
        pd.DataFrame({"Feature": features, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(20)
    )

    # Create plot
    _, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
    ax.set_xlabel("XGBoost Importance Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Top 20 Feature Importances: {target_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / f"shap_importance_{target_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    _logger.info(f"Saved {target_name} importance plot to {output_path}")
    plt.close()

    # Log top features
    _logger.info(f"\n=== Top 10 Features for {target_name} ===")
    for feat, imp in zip(
        importance_df["Feature"].tail(10)[::-1], importance_df["Importance"].tail(10)[::-1]
    ):
        _logger.info(f"  {feat:35s}  {imp:.6f}")


def main():
    """Load cached SHAP importance from model file and create plots for each target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True, help="Path to trained model joblib file")
    parser.add_argument("--output_dir", default="diagnostics", help="Output directory for plots")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model configuration with cached data
    model_cfg, _model_dict = load_model_config(args.model_file)

    shap_importance = model_cfg.get("shap_importance")
    features = model_cfg.get("features")

    if shap_importance is None:
        _logger.error("ERROR: No cached SHAP importance found in model file!")
        _logger.error("Make sure the model was trained with the updated code.")
        return

    if features is None:
        _logger.error("ERROR: No feature list found in model file!")
        return

    _logger.info(f"Loaded {len(features)} features from cache")
    _logger.info(f"Found per-target SHAP importance for: {list(shap_importance.keys())}")

    # Create plots for each target using cached SHAP importance
    for target_name, importances in shap_importance.items():
        _logger.info(f"\nProcessing {target_name}...")
        plot_feature_importance(features, importances, target_name, args.output_dir)

    _logger.info(f"\n✓ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
