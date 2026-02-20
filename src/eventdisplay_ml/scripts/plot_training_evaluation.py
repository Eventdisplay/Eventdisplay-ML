"""
Plot XGBoost training evaluation metrics for stereo and classification models.

This script loads a trained model (stereo or classification) from a joblib file
and plots the evaluation results stored during training. It visualizes the
training vs validation metrics curves to assess model convergence and potential
overfitting.

Example usage:
    # Stereo model
    python plot_training_evaluation.py \
        --model_file tmp_cta_testing/stereo_south/dispdir_bdt.joblib \
        --output_file training_curves.png

    # Classification model
    python plot_training_evaluation.py \
        --model_file tmp_testing/cl/classify_bdt_ebin2.joblib \
        --output_file classification_curves.png
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def plot_training_curves(evals_result, output_file=None):
    """
    Plot training and validation curves from XGBoost evaluation results.

    Parameters
    ----------
    evals_result : dict
        Dictionary containing evaluation results from XGBoost model.
        Expected format: {'validation_0': {'rmse': [...]}, 'validation_1': {'rmse': [...]}}
    output_file : str or Path, optional
        Path to save the output figure. If None, display interactively.
    """
    if not evals_result:
        _logger.warning("No evaluation results found in model.")
        return

    # Determine how many datasets were tracked (typically training and test)
    n_datasets = len(evals_result)
    dataset_names = list(evals_result.keys())

    _logger.info(f"Found {n_datasets} evaluation datasets: {dataset_names}")

    # Get all metrics tracked for the first dataset
    metrics = list(evals_result[dataset_names[0]].keys())
    n_metrics = len(metrics)

    _logger.info(f"Metrics tracked: {metrics}")

    # Create subplots for each metric
    _, axes = plt.subplots(n_metrics, 1, figsize=(10, 6 * n_metrics), squeeze=False)
    axes = axes.flatten()

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    labels = {
        "validation_0": "Training",
        "validation_1": "Test",
        "train": "Training",
        "test": "Test",
    }

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for dataset_idx, dataset_name in enumerate(dataset_names):
            if metric not in evals_result[dataset_name]:
                continue

            values = evals_result[dataset_name][metric]
            epochs = np.arange(1, len(values) + 1)

            label = labels.get(dataset_name, dataset_name)
            color = colors[dataset_idx % len(colors)]

            ax.plot(epochs, values, label=label, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel("Boosting Round (Iteration)", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"Training Progress: {metric.upper()}", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Log scale for y-axis if values span multiple orders of magnitude
        if len(values) > 0:
            value_range = np.max(values) / (np.min(values) + 1e-10)
            if value_range > 100:
                ax.set_yscale("log")
                _logger.info(f"Using log scale for {metric} (value range: {value_range:.1f})")

    plt.tight_layout()

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        _logger.info(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Plot XGBoost training evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot XGBoost training evaluation metrics from trained model "
            "(stereo or classification)."
        )
    )
    parser.add_argument(
        "--model_file",
        required=True,
        type=str,
        help="Path to the trained model joblib file (e.g., dispdir_bdt.joblib).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the output plot (PNG/PDF). If not provided, display interactively.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _logger.info(f"Loading model from: {model_path}")
    model_configs = joblib.load(model_path)

    # Extract the XGBoost model and its evaluation results
    if "models" not in model_configs:
        raise ValueError("Model file does not contain 'models' key.")

    if "xgboost" not in model_configs["models"]:
        raise ValueError("Model file does not contain 'xgboost' model.")

    xgb_model = model_configs["models"]["xgboost"]["model"]

    if not hasattr(xgb_model, "evals_result"):
        raise AttributeError(
            "XGBoost model does not have 'evals_result' method. "
            "Model may not have been trained with eval_set parameter."
        )

    evals_result = xgb_model.evals_result()

    _logger.info(f"Model type: {type(xgb_model).__name__}")
    _logger.info(f"Number of boosting rounds: {xgb_model.get_booster().num_boosted_rounds()}")

    # Additional model info
    if "features" in model_configs:
        _logger.info(f"Number of features: {len(model_configs['features'])}")
    if "targets" in model_configs:
        _logger.info(f"Target variables: {model_configs['targets']}")

    plot_training_curves(evals_result, args.output_file)

    _logger.info("Plotting completed successfully.")


if __name__ == "__main__":
    main()
