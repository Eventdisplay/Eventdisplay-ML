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

from eventdisplay_ml import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _joblib_basename(model_path):
    """Return basename without .joblib/.joblib.gz suffixes."""
    name = Path(model_path).name
    return name.removesuffix(".joblib.gz").removesuffix(".joblib")


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_file",
        type=str,
        help="Path to a single trained model joblib file (e.g., dispdir_bdt.joblib).",
    )
    group.add_argument(
        "--model_dir",
        type=str,
        help=(
            "Directory containing multiple joblib model files. "
            "All *.joblib files will be processed.",
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Path to save the output plot (PNG/PDF). If not provided, display interactively. "
            "Only for single file mode."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save all output plots (required if --model_dir is used).",
    )

    args = parser.parse_args()
    if args.model_file:
        model_path = utils.resolve_joblib_path(args.model_file)

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

        output_file = args.output_file
        if output_file is None:
            output_file = f"training_evaluation_{_joblib_basename(model_path)}.png"
            _logger.info(f"No --output_file given. Saving to {output_file}")

        plot_training_curves(evals_result, output_file)
        _logger.info("Plotting completed successfully.")

    elif args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if not args.output_dir:
            raise ValueError("--output_dir must be specified when using --model_dir.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        discovered_files = sorted(
            set(model_dir.glob("*.joblib")).union(model_dir.glob("*.joblib.gz"))
        )
        files_by_name = {}
        for model_path in discovered_files:
            key = _joblib_basename(model_path)
            existing = files_by_name.get(key)
            if existing is None or model_path.name.endswith(".joblib.gz"):
                files_by_name[key] = model_path
        joblib_files = [files_by_name[name] for name in sorted(files_by_name)]
        if not joblib_files:
            raise FileNotFoundError(f"No joblib files found in directory: {model_dir}")

        for model_path in joblib_files:
            _logger.info(f"Loading model from: {model_path}")
            try:
                model_configs = joblib.load(model_path)
                if "models" not in model_configs or "xgboost" not in model_configs["models"]:
                    _logger.error(f"Skipping {model_path}: missing 'models/xgboost' key.")
                    continue

                xgb_model = model_configs["models"]["xgboost"].get("model")
                if not hasattr(xgb_model, "evals_result"):
                    _logger.error(f"Skipping {model_path}: model missing 'evals_result'.")
                    continue

                evals_result = xgb_model.evals_result()
                output_file = output_dir / f"training_evaluation_{_joblib_basename(model_path)}.png"
                plot_training_curves(evals_result, output_file)
                _logger.info(f"Saved plot for {model_path.name} to {output_file}")
            except Exception as e:
                _logger.exception(f"Skipping {model_path}: failed to process model ({e})")
                continue

        _logger.info("Batch plotting completed.")


if __name__ == "__main__":
    main()
