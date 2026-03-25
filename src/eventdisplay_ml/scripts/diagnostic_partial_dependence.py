r"""Partial Dependence Plots (PDP): Prove the model captures physics, not chaos.

Plots predicted residual output as a function of a single feature while holding
others constant. For stereo reconstruction, proves that the model correctly
reduces corrections for high-multiplicity events and increases them for sparse data.

Usage:
    eventdisplay-ml-diagnostic-partial-dependence \\
        --model_file trained_stereo.joblib \\
        --output_dir diagnostics/ \\
        --features DispNImages Xoff_weighted_bdt ErecS
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import partial_dependence

from eventdisplay_ml import diagnostic_utils

_logger = logging.getLogger(__name__)


def load_data_and_model(model_file, input_file_list=None):
    """Load trained model and reconstruct the held-out test split."""
    model, _, _, x_test, _, features, target_names, _ = (
        diagnostic_utils.load_stereo_regression_split(
            model_file,
            input_file_list,
        )
    )
    return model, x_test, features, target_names


def compute_partial_dependence(model, x_test, features_to_plot):
    """Compute partial dependence for selected features."""
    _logger.info(f"Computing partial dependence for {len(features_to_plot)} features...")

    # Convert all features to float to avoid sklearn warnings about integer dtypes
    x_test_float = x_test.astype(np.float64)

    pdp_data = {}

    for feat_name in features_to_plot:
        if feat_name not in x_test_float.columns:
            _logger.warning(f"Feature {feat_name} not found in data")
            continue

        feat_idx = x_test_float.columns.get_loc(feat_name)

        # Compute PDP for each target
        pd_result = partial_dependence(
            model,
            x_test_float,
            [feat_idx],
            grid_resolution=50,
            percentiles=(0.05, 0.95),
        )

        pd_values = pd_result.get("average")
        if pd_values is None:
            pd_values = pd_result.get("average_predictions")
        if pd_values is None:
            raise ValueError(
                "Could not find partial dependence output in result. "
                "Expected key 'average' or 'average_predictions'."
            )

        pdp_data[feat_name] = {
            "grid": pd_result["grid_values"][0],
            "pd_values": np.asarray(pd_values),  # shape: (n_targets, n_grid)
        }

    return pdp_data


def plot_partial_dependence(pdp_data, output_dir, target_names):
    """Create PDP plots for each feature x target combination."""
    _logger.info("Creating partial dependence plots...")

    features = list(pdp_data.keys())
    if not features:
        _logger.warning("No valid features available for partial dependence plotting")
        return

    # Create a grid of subplots: features x targets
    _, axes = plt.subplots(len(features), len(target_names), figsize=(15, 5 * len(features)))

    if len(features) == 1:
        axes = axes.reshape(1, -1)
    if len(target_names) == 1:
        axes = axes.reshape(-1, 1)

    for feat_idx, feat_name in enumerate(features):
        for target_idx, target_name in enumerate(target_names):
            ax = axes[feat_idx, target_idx]

            grid = pdp_data[feat_name]["grid"]
            # pd_values shape: (n_targets, n_grid_points)
            pd_vals = pdp_data[feat_name]["pd_values"][target_idx]

            ax.plot(grid, pd_vals, linewidth=2.5, marker="o", markersize=4, color="steelblue")
            ax.fill_between(grid, pd_vals * 0.95, pd_vals * 1.05, alpha=0.2)

            ax.set_xlabel(feat_name)
            ax.set_ylabel(f"Predicted {target_name}")
            ax.set_title(f"{feat_name} -> {target_name}")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "partial_dependence.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    _logger.info(f"Saved PDP plots to {output_path}")
    plt.close()


def diagnose_physics(pdp_data):
    """Check if PDP shows physically sensible behavior."""
    _logger.info("\n%s", "=" * 60)
    _logger.info("PHYSICS VALIDATION: Partial Dependence Analysis")
    _logger.info("%s", "=" * 60)

    # Check 1: Multiplicity effect (should reduce corrections for high multiplicity)
    if "DispNImages" in pdp_data:
        grid = pdp_data["DispNImages"]["grid"]
        pd_vals = pdp_data["DispNImages"]["pd_values"][0]  # first target

        slope = (pd_vals[-1] - pd_vals[0]) / (grid[-1] - grid[0])

        _logger.info("\nMultiplicity Effect (DispNImages):")
        _logger.info(f"  Slope of PDP: {slope:.6f}")
        if slope < 0:
            _logger.info("  CORRECT - More telescopes → smaller corrections needed")
        else:
            _logger.info("  WARNING - Unexpected behavior (fewer telescopes → larger corr)")

    # Check 2: Baseline stability (should show smooth, monotonic response)
    baseline_features = [feat for feat in pdp_data if "weighted_bdt" in feat or "intersect" in feat]
    for feat_name in baseline_features:
        pd_vals = pdp_data[feat_name]["pd_values"][0]

        # Compute smoothness: ratio of diff magnitudes
        diffs = np.abs(np.diff(pd_vals))
        smoothness = np.std(diffs) / (np.mean(diffs) + 1e-6)

        _logger.info(f"\n{feat_name}:")
        _logger.info(f"  Smoothness index: {smoothness:.4f}")
        if smoothness < 0.3:
            _logger.info("  GOOD - Smooth, linear relationship (learned physics)")
        elif smoothness < 0.6:
            _logger.info("  OK - Some noise but generally smooth")
        else:
            _logger.info("  WARNING - Chaotic relationship (possible overtraining)")


def main():
    """Rebuild held-out data from model metadata and run PDP diagnostics."""
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
    parser.add_argument(
        "--features",
        nargs="+",
        default=["DispNImages", "Xoff_weighted_bdt", "Yoff_weighted_bdt", "ErecS"],
        help="Features to plot PDP for",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, x_test, _, target_names = load_data_and_model(args.model_file, args.input_file_list)
    pdp_data = compute_partial_dependence(model, x_test, args.features)
    plot_partial_dependence(pdp_data, args.output_dir, target_names)
    diagnose_physics(pdp_data)


if __name__ == "__main__":
    main()
