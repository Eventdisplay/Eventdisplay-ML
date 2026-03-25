r"""Residual Normality & Outlier Check: Validate statistical quality of predictions.

Tests if residuals are Gaussian and centered at zero. Non-normal residuals indicate
the model is failing on specific event types (e.g., edge-of-camera or low-multiplicity).
Heavy tails indicate outliers; skewness indicates systematic bias.

Usage:
    eventdisplay-ml-diagnostic-residual-normality \\
    --model_file trained_stereo.joblib \\
        --output_dir diagnostics/
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from eventdisplay_ml import diagnostic_utils

_logger = logging.getLogger(__name__)


def load_predictions_and_targets(model_file, input_file_list=None):
    """Rebuild test split and compute predicted residuals."""
    model, _, _, x_test, y_test, _, target_names, model_dict = (
        diagnostic_utils.load_stereo_regression_split(
            model_file,
            input_file_list,
        )
    )
    y_pred = diagnostic_utils.predict_unscaled_residuals(model, x_test, model_dict, target_names)
    return y_pred, y_test, target_names


def compute_residuals(y_pred, y_true, target_names):
    """Compute residuals (predicted - true)."""
    residuals = {}

    for target_name in target_names:
        if target_name in y_true.columns and target_name in y_pred.columns:
            residuals[target_name] = y_pred[target_name].values - y_true[target_name].values

    return residuals


def compute_normality_stats(residuals):
    """Compute Gaussian fit parameters and normality tests."""
    stats_dict = {}

    for target_name, resid in residuals.items():
        # Remove NaN values
        resid_clean = resid[~np.isnan(resid)]
        if len(resid_clean) == 0:
            _logger.warning(f"Skipping {target_name}: no finite residuals")
            continue

        # Gaussian parameters
        mean = np.mean(resid_clean)
        std = np.std(resid_clean)

        # Normality tests
        _, p_ks = stats.kstest(resid_clean, "norm", args=(mean, std))
        ad_result = stats.anderson(resid_clean, dist="norm")
        ad_stat = ad_result.statistic
        ad_crit_5 = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else np.nan

        # Skewness and kurtosis
        skewness = stats.skew(resid_clean)
        kurtosis = stats.kurtosis(resid_clean)

        # Quantile-Quantile test (visual)
        _, (_, _, qq_r) = stats.probplot(resid_clean, dist="norm")
        qq_r2 = qq_r**2

        stats_dict[target_name] = {
            "mean": mean,
            "std": std,
            "p_ks": p_ks,
            "ad_stat": ad_stat,
            "ad_crit_5": ad_crit_5,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "qq_r2": qq_r2,
            "n_outliers": np.sum(np.abs(resid_clean) > 3 * std),
            "n_samples": len(resid_clean),
        }

    return stats_dict


def plot_residual_diagnostics(residuals, stats_dict, output_dir):
    """Create comprehensive residual diagnostic plots."""
    _logger.info("Creating residual diagnostic plots...")

    target_names = list(residuals.keys())
    if not target_names:
        _logger.warning("No residual targets to plot")
        return

    # Create a 2xN grid: histogram + Q-Q plot for each target
    _, axes = plt.subplots(2, len(target_names), figsize=(5 * len(target_names), 10))

    if len(target_names) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, target_name in enumerate(target_names):
        resid = residuals[target_name][~np.isnan(residuals[target_name])]
        stat = stats_dict[target_name]

        # Row 0: Histogram with Gaussian overlay
        ax_hist = axes[0, col_idx]
        ax_hist.hist(resid, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black")

        # Overlay Gaussian
        x_range = np.linspace(resid.min(), resid.max(), 100)
        gaussian = stats.norm.pdf(x_range, stat["mean"], stat["std"])
        ax_hist.plot(x_range, gaussian, "r-", linewidth=2, label="Normal fit")

        ax_hist.axvline(stat["mean"], color="green", linestyle="--", linewidth=2, label="Mean")
        ax_hist.set_xlabel("Residual value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"{target_name}\nmu={stat['mean']:.4f}, sigma={stat['std']:.4f}")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.3)

        # Row 1: Q-Q plot
        ax_qq = axes[1, col_idx]
        stats.probplot(resid, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"Q-Q Plot (R²={stat['qq_r2']:.4f})")
        ax_qq.grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "residual_diagnostics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    _logger.info(f"Saved residual diagnostics to {output_path}")
    plt.close()


def diagnose_residual_quality(residuals, stats_dict):
    """Provide detailed diagnosis of residual quality."""
    _logger.info("\n%s", "=" * 60)
    _logger.info("RESIDUAL NORMALITY & OUTLIER ANALYSIS")
    _logger.info("%s", "=" * 60)

    for target_name, stat in stats_dict.items():
        _logger.info(f"\n{target_name}:")
        _logger.info(f"  Mean:  {stat['mean']:.6f} (expect ~0)")
        _logger.info(f"  Std:   {stat['std']:.6f}")

        if np.abs(stat["mean"]) < stat["std"] * 0.1:
            _logger.info("  ✓ GOOD - Residuals centered at zero")
        elif np.abs(stat["mean"]) < stat["std"] * 0.2:
            _logger.info("  ~ OK - Small systematic offset")
        else:
            _logger.info("  ✗ WARNING - Significant bias detected")

        _logger.info(f"\n  Skewness: {stat['skewness']:.4f}")
        if np.abs(stat["skewness"]) < 0.2:
            _logger.info("    ✓ GOOD - Symmetric distribution")
        elif np.abs(stat["skewness"]) < 0.5:
            _logger.info("    ~ OK - Mild asymmetry")
        else:
            _logger.info("    ✗ WARNING - Strong skew (model failing on certain events)")

        _logger.info(f"\n  Kurtosis: {stat['kurtosis']:.4f}")
        if np.abs(stat["kurtosis"]) < 0.5:
            _logger.info("    ✓ GOOD - Gaussian-like tails")
        elif np.abs(stat["kurtosis"]) < 1.0:
            _logger.info("    ~ OK - Slightly heavy/light tails")
        else:
            _logger.info("    ✗ WARNING - Heavy tails (outliers present)")

        _logger.info(f"\n  Outliers (>3sigma): {stat['n_outliers']} events")
        outlier_pct = stat["n_outliers"] / stat["n_samples"] * 100
        if outlier_pct < 0.3:
            _logger.info("    ✓ GOOD - Minimal outliers")
        elif outlier_pct < 1.0:
            _logger.info("    ~ OK - Few outliers")
        else:
            _logger.info("    ✗ WARNING - Excessive outliers")

        _logger.info(f"\n  Kolmogorov-Smirnov test: p={stat['p_ks']:.4f}")
        if stat["p_ks"] > 0.05:
            _logger.info("    ✓ Gaussian hypothesis NOT rejected (p > 0.05)")
        else:
            _logger.info("    ✗ Distribution deviates from Gaussian (p < 0.05)")

        _logger.info(
            f"\n  Anderson-Darling normality: stat={stat['ad_stat']:.4f}, "
            f"5% crit={stat['ad_crit_5']:.4f}"
        )
        if stat["ad_stat"] < stat["ad_crit_5"]:
            _logger.info("    ✓ Anderson-Darling does not reject normality at 5%")
        else:
            _logger.info("    ✗ Anderson-Darling rejects normality at 5%")

        _logger.info(f"\n  Q-Q R²: {stat['qq_r2']:.4f}")
        if stat["qq_r2"] > 0.98:
            _logger.info("    ✓ Excellent Gaussian fit")
        elif stat["qq_r2"] > 0.95:
            _logger.info("    ✓ Good Gaussian fit")
        else:
            _logger.info("    ~ Fair fit, consider investigating tails")


def main():
    """Load cached residual normality stats or fall back to reconstructing the split."""
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

    _, stats_dict = diagnostic_utils.load_cached_residual_normality_stats(args.model_file)

    residuals = None
    if stats_dict is None:
        _logger.info("Cached residual normality statistics unavailable; rebuilding from test split")
        y_pred, y_true, target_names = load_predictions_and_targets(
            args.model_file,
            args.input_file_list,
        )
        residuals = compute_residuals(y_pred, y_true, target_names)
        stats_dict = compute_normality_stats(residuals)
        plot_residual_diagnostics(residuals, stats_dict, args.output_dir)
    else:
        _logger.info("Using cached residual normality statistics from the model file")
        _logger.info(
            "Note: Diagnostic plots skipped when using cached statistics; "
            "rerun without cache to regenerate plots"
        )

    diagnose_residual_quality(residuals or {}, stats_dict)


if __name__ == "__main__":
    main()
