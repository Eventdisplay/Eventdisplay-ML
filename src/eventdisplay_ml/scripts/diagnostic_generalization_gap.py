r"""Generalization Ratio: Quantify overfitting gap between train and test performance.

Uses cached train/test RMSE values written during training when available.
For older model files without cached metrics, it falls back to rebuilding the
original train/test split from the stored input metadata.

Usage:
    python diagnostic_generalization_gap.py \\
        --model_file trained_stereo.joblib \\
        --output_dir diagnostics/
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eventdisplay_ml import diagnostic_utils

_logger = logging.getLogger(__name__)


def compute_rmse_and_gaps(model, x_train, y_train, x_test, y_test, model_dict, target_names):
    """Compute RMSE for train and test, derive generalization metrics."""
    _logger.info("Computing train and test RMSE...")

    y_train_pred = diagnostic_utils.predict_unscaled_residuals(
        model,
        x_train,
        model_dict,
        target_names,
    )
    y_test_pred = diagnostic_utils.predict_unscaled_residuals(
        model,
        x_test,
        model_dict,
        target_names,
    )

    metrics = diagnostic_utils.compute_generalization_metrics(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        target_names,
    )

    for target_name in target_names:
        gap_pct = metrics[target_name]["gap_pct"]
        _logger.info(f"\n{target_name}:")
        _logger.info(f"  Train RMSE:       {metrics[target_name]['rmse_train']:.6f}")
        _logger.info(f"  Test RMSE:        {metrics[target_name]['rmse_test']:.6f}")
        _logger.info(f"  Gap:              {gap_pct:.2f}%")
        _logger.info(f"  Generalization:   {'PASS' if gap_pct < 10 else 'WARN'} (threshold <10%)")

    return metrics


def plot_generalization_metrics(metrics, output_dir):
    """Create visualization of train/test RMSE and generalization gap."""
    _logger.info("Creating generalization plots...")

    targets = list(metrics.keys())

    # Plot 1: Train vs Test RMSE
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: RMSE comparison
    rmse_train = [metrics[target]["rmse_train"] for target in targets]
    rmse_test = [metrics[target]["rmse_test"] for target in targets]

    x_pos = np.arange(len(targets))
    width = 0.35

    axes[0].bar(x_pos - width / 2, rmse_train, width, label="Train RMSE", alpha=0.8)
    axes[0].bar(x_pos + width / 2, rmse_test, width, label="Test RMSE", alpha=0.8)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Training vs Test Performance")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(targets, rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Subplot 2: Generalization gap
    gaps = [metrics[target]["gap_pct"] for target in targets]
    colors = ["green" if gap < 10 else "orange" if gap < 15 else "red" for gap in gaps]

    axes[1].bar(targets, gaps, color=colors, alpha=0.7)
    axes[1].axhline(
        y=10,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Safe threshold (10%)",
    )
    axes[1].axhline(
        y=15,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Warning (15%)",
    )
    axes[1].set_ylabel("Gap (%)")
    axes[1].set_title("Generalization Gap: (Test-Train)/Train x 100%")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(targets, rotation=15)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "generalization_gap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    _logger.info(f"Saved generalization plot to {output_path}")
    plt.close()


def diagnose_overfitting(metrics):
    """Summary diagnosis of overfitting status."""
    _logger.info("\n%s", "=" * 60)
    _logger.info("OVERFITTING DIAGNOSIS")
    _logger.info("%s", "=" * 60)

    all_gaps = [metrics[target]["gap_pct"] for target in metrics]
    mean_gap = np.mean(all_gaps)

    _logger.info(f"\nMean Generalization Gap: {mean_gap:.2f}%")

    if mean_gap < 5:
        status = "EXCELLENT - Model shows minimal overfitting"
    elif mean_gap < 10:
        status = "GOOD - Model generalization is safe"
    elif mean_gap < 15:
        status = "ACCEPTABLE - Minor overfitting, monitor carefully"
    else:
        status = "WARNING - Significant overfitting detected"

    _logger.info(f"Status: {status}")
    _logger.info("\nPer-target breakdown:")
    for target, data in metrics.items():
        _logger.info(f"  {target}: {data['gap_pct']:.2f}% gap")


def main():
    """Load cached generalization metrics or fall back to reconstructing the split."""
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _, metrics = diagnostic_utils.load_cached_generalization_metrics(args.model_file)

    if metrics is None:
        _logger.info(
            "Cached generalization metrics are unavailable; rebuilding the train/test split"
        )
        model, x_train, y_train, x_test, y_test, _, target_names, model_dict = (
            diagnostic_utils.load_stereo_regression_split(
                args.model_file,
                args.input_file_list,
            )
        )

        metrics = compute_rmse_and_gaps(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            model_dict,
            target_names,
        )
    else:
        _logger.info("Using cached generalization metrics from the model file")

    plot_generalization_metrics(metrics, args.output_dir)
    diagnose_overfitting(metrics)


if __name__ == "__main__":
    main()
