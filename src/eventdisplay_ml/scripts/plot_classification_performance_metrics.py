"""
Compare performance of TMVA and XGB gamma/hadron separator (efficiency based metrics).

./plot_classification_performance_metrics.py \
        AP/BDTtraining/GammaHadronBDTs_V6_DISP/V6_2016_2017_ATM61/NTel2-Soft/ \
        AP/CARE_202404/V6_2016_2017_ATM61_gamma/TrainXGBGammaHadron/

Notes the differences between TMVA and XGB implementations:

- TMVA uses always the first zenith bin (XGB uses all zenith angles)
- XGB uses the 4-telescope configuration (TMVA uses all telescopes)

"""

import argparse
import csv
import logging
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import uproot

from eventdisplay_ml import utils

mpl.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def safe_label(label):
    """Return a filesystem-safe label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "xgb"


def xgb_run_label(path):
    """Return a default label for one XGB model directory."""
    return Path(path).resolve().name


def xgb_run_inputs(xgb_dirs, xgb_labels=None):
    """Build XGB run descriptors from CLI inputs."""
    if xgb_labels is not None and len(xgb_labels) != len(xgb_dirs):
        raise ValueError("Number of --xgb-label values must match number of --xgb_dir entries.")
    labels = xgb_labels if xgb_labels is not None else [xgb_run_label(path) for path in xgb_dirs]
    if len(set(labels)) != len(labels):
        raise ValueError("XGB run labels must be unique.")
    return [
        {"path": Path(path), "label": label} for path, label in zip(xgb_dirs, labels, strict=True)
    ]


def plot_efficiencies(ax, xgb_curves, x_root=None, y_effs=None, y_effb=None):
    """Plot Signal and Background efficiencies vs. cut value (threshold)."""
    if x_root is not None and y_effs is not None and y_effb is not None:
        ax.plot(x_root, y_effs, label="TMVA BDT Eff S", color="blue", linestyle="-", linewidth=2)
        ax.plot(x_root, y_effb, label="TMVA BDT Eff B", color="red", linestyle="-", linewidth=2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, curve in enumerate(xgb_curves):
        color = colors[index % len(colors)]
        ax.plot(
            curve["threshold"],
            curve["signal_efficiency"],
            label=f"{curve['label']} Eff S",
            color=color,
            linestyle=":",
            linewidth=2,
        )
        ax.plot(
            curve["threshold"],
            curve["background_efficiency"],
            label=f"{curve['label']} Eff B",
            color=color,
            linestyle="--",
            linewidth=3,
        )

    ax.set_xlabel("Cut value (Threshold)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Signal / Background Efficiency")
    ax.set_ylim(0, 1.05)


def plot_qfactor(ax, xgb_curves, y_effs=None, y_effb=None):
    """Plot Q-factor: Signal efficiency / sqrt(Background efficiency)."""
    if y_effs is not None and y_effb is not None:
        q_tmva = np.divide(y_effs, np.sqrt(y_effb), out=np.zeros_like(y_effs), where=y_effb != 0)
        ax.plot(y_effs, q_tmva, label=f"TMVA (Max Q: {np.max(q_tmva):.2f})", color="blue")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, curve in enumerate(xgb_curves):
        color = colors[index % len(colors)]
        y_effs_xgb = np.asarray(curve["signal_efficiency"], dtype=float)
        y_effb_xgb = np.asarray(curve["background_efficiency"], dtype=float)
        q_xgb = np.divide(
            y_effs_xgb, np.sqrt(y_effb_xgb), out=np.zeros_like(y_effs_xgb), where=y_effb_xgb != 0
        )
        ax.plot(
            y_effs_xgb,
            q_xgb,
            label=f"{curve['label']} (Max Q: {np.max(q_xgb):.2f})",
            color=color,
            linestyle="--",
            linewidth=3,
        )

    ax.set_xlabel(r"Gamma Efficiency ($\epsilon_{\gamma}$)")
    ax.set_ylabel(r"Q-factor ($\epsilon_{\gamma} / \sqrt{\epsilon_{h}}$)")
    ax.set_title("Q-Factor")


def plot_roc(ax, xgb_curves, y_effs=None, y_effb=None):
    """Plot ROC curve: Signal efficiency vs. 1 - Background efficiency."""
    if y_effs is not None and y_effb is not None:
        auc_tmva = -np.trapezoid(1 - y_effb, y_effs)
        ax.plot(y_effs, 1 - y_effb, label=f"TMVA (AUC: {auc_tmva:.2f})", color="blue")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, curve in enumerate(xgb_curves):
        color = colors[index % len(colors)]
        y_effs_xgb = np.asarray(curve["signal_efficiency"], dtype=float)
        y_effb_xgb = np.asarray(curve["background_efficiency"], dtype=float)
        auc_xgb = -np.trapezoid(1 - y_effb_xgb, y_effs_xgb)
        ax.plot(
            y_effs_xgb,
            1 - y_effb_xgb,
            label=f"{curve['label']} (AUC: {auc_xgb:.2f})",
            color=color,
            linestyle="--",
            linewidth=3,
        )

    ax.margins(x=0.02)
    ax.set_xlabel("Gamma Efficiency (Signal)")
    ax.set_ylabel("Hadron Rejection (1 - Background Eff)")
    ax.set_title("ROC")


def plot_score_distributions(ax, xgb_curves, x_root=None, y_effs=None, y_effb=None):
    """Reconstructs and plots the probability density of the MVA scores."""
    if x_root is not None and y_effs is not None and y_effb is not None:
        pdf_s_tmva = -np.gradient(y_effs, x_root)
        pdf_b_tmva = -np.gradient(y_effb, x_root)
        ax.fill_between(x_root, pdf_s_tmva, alpha=0.2, color="blue", label="TMVA Signal")
        ax.fill_between(x_root, pdf_b_tmva, alpha=0.2, color="red", label="TMVA Background")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, curve in enumerate(xgb_curves):
        color = colors[index % len(colors)]
        threshold = np.asarray(curve["threshold"], dtype=float)
        pdf_s_xgb = -np.gradient(curve["signal_efficiency"], threshold)
        pdf_b_xgb = -np.gradient(curve["background_efficiency"], threshold)
        ax.plot(
            threshold,
            pdf_s_xgb,
            color=color,
            linestyle=":",
            label=f"{curve['label']} Signal",
            linewidth=2,
        )
        ax.plot(
            threshold,
            pdf_b_xgb,
            color=color,
            linestyle="--",
            label=f"{curve['label']} Background",
            linewidth=3,
        )

    ax.set_xlabel("MVA Score (Normalized)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Score Distributions")


def load_efficiency_tmva(path, ebin, zebin=0):
    """Load efficiencies from TMVA root files."""
    root_dir = Path(path)
    file_path = tmva_root_file(root_dir, ebin, zebin)
    if file_path is None:
        expected_file_name = (
            f"BDT_{ebin}_{zebin}.root"
            if tmva_plain_bdt_files_available(root_dir)
            else f"TMVA.BDT_{ebin}_{zebin}.root"
        )
        _logger.warning(
            "TMVA ROOT file unavailable in %s (ebin=%s, zebin=%s). "
            "Expected %s. Plotting XGB only for this bin.",
            path,
            ebin,
            zebin,
            expected_file_name,
        )
        return None

    try:
        with uproot.open(file_path) as rf:
            base_path = "Method_BDT/BDT_0"
            effs_rt = rf[f"{base_path}/MVA_BDT_0_effS"]
            effb_rt = rf[f"{base_path}/MVA_BDT_0_effB"]
            x_root_raw = (
                effs_rt.axis().centers() if hasattr(effs_rt, "axis") else effs_rt.values(axis=0)
            )
            x_min = np.min(x_root_raw)
            x_max = np.max(x_root_raw)
            if x_max == x_min:
                _logger.warning(
                    "TMVA efficiency axis is degenerate in %s (ebin=%s, zebin=%s); skipping TMVA overlay.",
                    file_path,
                    ebin,
                    zebin,
                )
                return None
            # map [x_min, x_max] -> [0, 1]
            x_root = (x_root_raw - x_min) / (x_max - x_min)
            y_effs = effs_rt.values()
            y_effb = effb_rt.values()
    except (OSError, KeyError) as exc:
        _logger.warning(
            "TMVA efficiency histograms unavailable in %s (ebin=%s, zebin=%s): %s. "
            "Plotting XGB only for this bin.",
            file_path,
            ebin,
            zebin,
            exc,
        )
        return None

    return x_root, y_effs, y_effb


def tmva_root_file(path, ebin, zebin):
    """Return a TMVA ROOT file path for supported filename conventions."""
    root_dir = Path(path)
    file_name = (
        f"BDT_{ebin}_{zebin}.root"
        if tmva_plain_bdt_files_available(root_dir)
        else f"TMVA.BDT_{ebin}_{zebin}.root"
    )
    file_path = root_dir / file_name
    return file_path if file_path.exists() else None


def tmva_plain_bdt_files_available(path):
    """Return True when the TMVA directory contains plain BDT ROOT outputs."""
    return any(Path(path).glob("BDT_*_*.root"))


def load_xgb_model_data(path, ebin):
    """Load XGB joblib model payload for one energy bin."""
    model_file = utils.resolve_joblib_path(Path(path) / f"gammahadron_bdt_ebin{ebin}")
    return utils.load_joblib(model_file)


def load_efficiency_xgb(data_joblib, ebin, zebin=-1):
    """Load efficiencies from XGB model payload."""
    model_data = data_joblib["models"]["xgboost"]

    if zebin < 0:
        efficiency_key = "efficiency"
    else:
        efficiency_key = f"efficiency_ze{zebin}"
        if efficiency_key not in model_data:
            available_ze_bins = []
            for key in model_data:
                match = re.fullmatch(r"efficiency_ze(\d+)", key)
                if match:
                    available_ze_bins.append(int(match.group(1)))
            available_ze_bins = sorted(set(available_ze_bins))
            raise KeyError(
                f"Efficiency key '{efficiency_key}' not found for ebin {ebin}. "
                f"Available zenith bins: {available_ze_bins or 'none'}."
            )

    if efficiency_key not in model_data:
        raise KeyError(f"Efficiency key '{efficiency_key}' not found for ebin {ebin}.")

    df_xgboost = model_data[efficiency_key]

    x_joblib = df_xgboost["threshold"]
    y_effs_xgb = df_xgboost["signal_efficiency"]
    y_effb_xgb = df_xgboost["background_efficiency"]

    return x_joblib, y_effs_xgb, y_effb_xgb


def xgb_zenith_bins(data_joblib):
    """Return available XGB zenith bins from loaded joblib model payload."""
    model_data = data_joblib["models"]["xgboost"]
    ze_bins = []
    for key in model_data:
        match = re.fullmatch(r"efficiency_ze(\d+)", key)
        if match:
            ze_bins.append(int(match.group(1)))
    return sorted(set(ze_bins))


def efficiency_at_signal_efficiency(
    signal_efficiency, background_efficiency, target_signal_efficiency
):
    """Return background efficiency at the closest available signal efficiency."""
    signal_efficiency = np.asarray(signal_efficiency, dtype=float)
    background_efficiency = np.asarray(background_efficiency, dtype=float)
    valid = np.isfinite(signal_efficiency) & np.isfinite(background_efficiency)
    if not np.any(valid):
        return np.nan, np.nan

    index = np.argmin(np.abs(signal_efficiency[valid] - target_signal_efficiency))
    return signal_efficiency[valid][index], background_efficiency[valid][index]


def zenith_uniformity_summary(data_joblib, ebin, target_signal_efficiency=0.8, model_label="xgb"):
    """Summarize zenith stability using background efficiency at fixed signal efficiency."""
    available_ze_bins = xgb_zenith_bins(data_joblib)
    if not available_ze_bins:
        return None

    _, y_effs_overall, y_effb_overall = load_efficiency_xgb(data_joblib, ebin, -1)
    overall_signal_efficiency, overall_background_efficiency = efficiency_at_signal_efficiency(
        y_effs_overall, y_effb_overall, target_signal_efficiency
    )

    ze_background_efficiencies = []
    ze_signal_efficiencies = []
    for ze_bin in available_ze_bins:
        _, y_effs_ze, y_effb_ze = load_efficiency_xgb(data_joblib, ebin, ze_bin)
        ze_signal_efficiency, ze_background_efficiency = efficiency_at_signal_efficiency(
            y_effs_ze, y_effb_ze, target_signal_efficiency
        )
        ze_signal_efficiencies.append(ze_signal_efficiency)
        ze_background_efficiencies.append(ze_background_efficiency)

    ze_background_efficiencies = np.asarray(ze_background_efficiencies, dtype=float)
    finite_background = ze_background_efficiencies[np.isfinite(ze_background_efficiencies)]
    if finite_background.size == 0:
        return None

    best_background_efficiency = np.min(finite_background)
    worst_background_efficiency = np.max(finite_background)
    mean_background_efficiency = np.mean(finite_background)
    std_background_efficiency = np.std(finite_background)

    worst_bin_index = int(np.nanargmax(ze_background_efficiencies))
    worst_ze_bin = available_ze_bins[worst_bin_index]
    best_bin_index = int(np.nanargmin(ze_background_efficiencies))
    best_ze_bin = available_ze_bins[best_bin_index]

    if best_background_efficiency > 0:
        worst_to_best_ratio = worst_background_efficiency / best_background_efficiency
    else:
        worst_to_best_ratio = np.inf
    if overall_background_efficiency > 0:
        worst_to_overall_ratio = worst_background_efficiency / overall_background_efficiency
    else:
        worst_to_overall_ratio = np.inf
    if mean_background_efficiency > 0:
        relative_std = std_background_efficiency / mean_background_efficiency
    else:
        relative_std = np.inf

    return {
        "model_label": model_label,
        "energy_bin": ebin,
        "target_signal_efficiency": target_signal_efficiency,
        "overall_signal_efficiency": overall_signal_efficiency,
        "overall_background_efficiency": overall_background_efficiency,
        "n_zenith_bins": len(available_ze_bins),
        "best_zenith_bin": best_ze_bin,
        "best_zenith_background_efficiency": best_background_efficiency,
        "worst_zenith_bin": worst_ze_bin,
        "worst_zenith_background_efficiency": worst_background_efficiency,
        "mean_zenith_background_efficiency": mean_background_efficiency,
        "std_zenith_background_efficiency": std_background_efficiency,
        "relative_std_zenith_background_efficiency": relative_std,
        "worst_to_best_background_efficiency_ratio": worst_to_best_ratio,
        "worst_to_overall_background_efficiency_ratio": worst_to_overall_ratio,
    }


def zenith_background_efficiency_rows(
    data_joblib, ebin, target_signal_efficiency=0.8, model_label="xgb"
):
    """Return per-zenith background efficiency rows at fixed signal efficiency."""
    rows = []
    for ze_bin in xgb_zenith_bins(data_joblib):
        _, y_effs_ze, y_effb_ze = load_efficiency_xgb(data_joblib, ebin, ze_bin)
        ze_signal_efficiency, ze_background_efficiency = efficiency_at_signal_efficiency(
            y_effs_ze, y_effb_ze, target_signal_efficiency
        )
        rows.append(
            {
                "model_label": model_label,
                "energy_bin": ebin,
                "zenith_bin": ze_bin,
                "target_signal_efficiency": target_signal_efficiency,
                "signal_efficiency": ze_signal_efficiency,
                "background_efficiency": ze_background_efficiency,
            }
        )
    return rows


def write_zenith_uniformity_summary(summary_rows, output_path):
    """Write zenith-uniformity metric rows to CSV."""
    if not summary_rows:
        _logger.warning("No zenith-uniformity summary rows available.")
        return

    fieldnames = list(summary_rows[0])
    with Path(output_path).open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    _logger.info("Wrote zenith-uniformity summary to %s", output_path)


def tmva_zenith_bins(path, ebin):
    """Return available TMVA zenith bins from supported ROOT filenames."""
    ze_bins = []
    root_dir = Path(path)
    if tmva_plain_bdt_files_available(root_dir):
        pattern = re.compile(rf"^BDT_{ebin}_(\d+)\.root$")
        file_paths = root_dir.glob(f"BDT_{ebin}_*.root")
    else:
        pattern = re.compile(rf"^TMVA\.BDT_{ebin}_(\d+)\.root$")
        file_paths = root_dir.glob(f"TMVA.BDT_{ebin}_*.root")

    for file_path in file_paths:
        match = pattern.match(file_path.name)
        if match:
            ze_bins.append(int(match.group(1)))
    return sorted(set(ze_bins))


def resolve_tmva_zebin(xgb_zebin, available_tmva_bins, fallback_tmva_bin=None):
    """Resolve TMVA zenith bin aligned to XGB zenith bin where possible."""
    if not available_tmva_bins:
        return None
    fallback_bin = fallback_tmva_bin
    if fallback_bin is None:
        fallback_bin = available_tmva_bins[0]
    if xgb_zebin < 0:
        return fallback_bin if fallback_bin in available_tmva_bins else None
    if xgb_zebin in available_tmva_bins:
        return xgb_zebin
    if fallback_bin in available_tmva_bins:
        return fallback_bin
    return None


def zenith_plot_label(zebin):
    """Return human-readable zenith label for plot/file naming."""
    return "overall" if zebin < 0 else f"ze{zebin}"


def style_axis(ax):
    """Apply common style settings to a matplotlib axis."""
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)


def xgb_efficiency_curve(data_joblib, ebin, zebin, label):
    """Return one labeled XGB efficiency curve."""
    x_joblib, y_effs_xgb, y_effb_xgb = load_efficiency_xgb(data_joblib, ebin, zebin)
    return {
        "label": label,
        "threshold": x_joblib,
        "signal_efficiency": y_effs_xgb,
        "background_efficiency": y_effb_xgb,
    }


def make_figure(xgb_curves, x_root=None, y_effs=None, y_effb=None):
    """Build 2x2 diagnostics figure for XGB with optional TMVA overlays."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharex=False, layout="constrained")

    for ax in axs.flatten():
        style_axis(ax)

    plot_efficiencies(axs[0, 0], xgb_curves, x_root, y_effs, y_effb)
    plot_qfactor(axs[0, 1], xgb_curves, y_effs, y_effb)
    plot_roc(axs[1, 0], xgb_curves, y_effs, y_effb)
    plot_score_distributions(axs[1, 1], xgb_curves, x_root, y_effs, y_effb)

    for ax in axs.flatten():
        ax.legend(fontsize=9, frameon=False, loc="best")

    return fig


def plot_zenith_uniformity_vs_energy(summary_rows, output_path):
    """Plot zenith stability metrics versus energy bin for all XGB runs."""
    if not summary_rows:
        _logger.warning("No zenith-uniformity summary rows available for plotting.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for ax in axes:
        style_axis(ax)

    labels = sorted({row["model_label"] for row in summary_rows})
    for label in labels:
        rows = sorted(
            [row for row in summary_rows if row["model_label"] == label],
            key=lambda row: row["energy_bin"],
        )
        energy = np.asarray([row["energy_bin"] for row in rows], dtype=float)
        overall = np.asarray([row["overall_background_efficiency"] for row in rows], dtype=float)
        best = np.asarray([row["best_zenith_background_efficiency"] for row in rows], dtype=float)
        worst = np.asarray([row["worst_zenith_background_efficiency"] for row in rows], dtype=float)
        ratio = np.asarray(
            [row["worst_to_overall_background_efficiency_ratio"] for row in rows], dtype=float
        )

        axes[0].fill_between(energy, best, worst, alpha=0.15)
        axes[0].plot(energy, overall, marker="o", linestyle="-", label=f"{label} overall")
        axes[0].plot(energy, worst, marker="s", linestyle="--", label=f"{label} worst ze")
        axes[1].plot(energy, ratio, marker="o", linestyle="-", label=label)

    target_efficiency = summary_rows[0]["target_signal_efficiency"]
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Background efficiency")
    axes[0].set_title(f"Zenith Stability at Gamma Efficiency {target_efficiency:g}")
    axes[0].legend(fontsize=9, frameon=False, loc="best")
    axes[1].axhline(1.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Energy bin")
    axes[1].set_ylabel("Worst ze / overall")
    axes[1].set_title("Inclusive-Curve Optimism")
    axes[1].legend(fontsize=9, frameon=False, loc="best")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Wrote zenith-uniformity plot to %s", output_path)


def plot_zenith_background_efficiency_heatmap(heatmap_rows, model_label, output_path):
    """Plot background efficiency by energy and zenith bin for one XGB run."""
    rows = [row for row in heatmap_rows if row["model_label"] == model_label]
    if not rows:
        _logger.warning("No heatmap rows available for model '%s'.", model_label)
        return

    energy_bins = sorted({row["energy_bin"] for row in rows})
    zenith_bins = sorted({row["zenith_bin"] for row in rows})
    matrix = np.full((len(zenith_bins), len(energy_bins)), np.nan)
    energy_index = {energy_bin: index for index, energy_bin in enumerate(energy_bins)}
    zenith_index = {zenith_bin: index for index, zenith_bin in enumerate(zenith_bins)}
    for row in rows:
        matrix[zenith_index[row["zenith_bin"]], energy_index[row["energy_bin"]]] = row[
            "background_efficiency"
        ]

    positive_values = matrix[np.isfinite(matrix) & (matrix > 0)]
    if positive_values.size == 0:
        _logger.warning(
            "No positive background efficiencies available for heatmap '%s'.", model_label
        )
        return

    plot_matrix = np.where(matrix > 0, np.log10(matrix), np.nan)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(plot_matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(energy_bins)), labels=energy_bins)
    ax.set_yticks(np.arange(len(zenith_bins)), labels=zenith_bins)
    ax.set_xlabel("Energy bin")
    ax.set_ylabel("Zenith bin")
    ax.set_title(f"{model_label}: log10 Background Efficiency")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10 background efficiency")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Wrote zenith background-efficiency heatmap to %s", output_path)


def selected_xgb_bins(zenith_bin_xgb, available_xgb_bins):
    """Resolve which XGB zenith bins to plot."""
    return [-1, *available_xgb_bins] if zenith_bin_xgb is None else [zenith_bin_xgb]


def tmva_overlay_data(root_dir, ebin, xgb_zebin, tmva_zebin):
    """Return TMVA overlay data tuple or None when TMVA is unavailable for this zenith bin."""
    if root_dir is None:
        return None
    if tmva_zebin is None:
        _logger.warning(
            "No TMVA zenith-bin match for XGB %s in ebin %s; plotting XGB only.",
            zenith_plot_label(xgb_zebin),
            ebin,
        )
        return None
    return load_efficiency_tmva(root_dir, ebin, tmva_zebin)


def main():
    """Plot TMVA and XGBoost performance metrics."""
    parser = argparse.ArgumentParser(description="Plot TMVA and XGBoost metrics.")
    parser.add_argument(
        "--tmva_dir",
        type=str,
        default=None,
        help="Path to TMVA BDT ROOT files (optional).",
    )
    parser.add_argument(
        "--xgb_dir",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Path(s) to XGB BDT joblib files (required). Pass several directories to overlay "
            "their curves in the same plots."
        ),
    )
    parser.add_argument(
        "--xgb-label",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels for --xgb_dir entries. Number of labels must match directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for PNG files (default: current directory).",
    )
    parser.add_argument(
        "--energy-bin",
        type=int,
        choices=range(9),
        default=None,
        help="Plot only a single energy bin (0-8). If omitted, all bins are processed.",
    )
    parser.add_argument(
        "--zenith-bin-tmva",
        type=int,
        default=None,
        help=(
            "Zenith bin index for TMVA ROOT files (second digit in BDT_<ebin>_<zebin>.root). "
            "If omitted, uses the first available TMVA zenith bin for overall or unmatched XGB bins."
        ),
    )
    parser.add_argument(
        "--zenith-bin-xgb",
        type=int,
        default=None,
        help=(
            "XGB zenith bin to plot. If omitted, plots overall (-1) and all available ze bins. "
            "Use -1 for overall or >=0 for efficiency_zeN."
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help=(
            "CSV file for zenith-uniformity metrics. If omitted, writes "
            "zenith_uniformity_summary.csv in the output directory."
        ),
    )
    parser.add_argument(
        "--summary-signal-efficiency",
        type=float,
        default=0.8,
        help="Signal efficiency used for zenith-uniformity metrics (default: 0.8).",
    )
    args = parser.parse_args()
    root_dir = args.tmva_dir
    xgb_runs = xgb_run_inputs(args.xgb_dir, args.xgb_label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    heatmap_rows = []

    # assume energy binning is identical in XGB and TMVA files.
    energy_bins = [args.energy_bin] if args.energy_bin is not None else range(9)
    for ebin in energy_bins:
        xgb_model_data_by_label = {}
        available_xgb_bins = set()
        for run in xgb_runs:
            xgb_model_data = load_xgb_model_data(run["path"], ebin)
            xgb_model_data_by_label[run["label"]] = xgb_model_data
            summary_row = zenith_uniformity_summary(
                xgb_model_data, ebin, args.summary_signal_efficiency, run["label"]
            )
            if summary_row is not None:
                summary_row["xgb_dir"] = str(run["path"])
                summary_rows.append(summary_row)
            heatmap_rows.extend(
                zenith_background_efficiency_rows(
                    xgb_model_data, ebin, args.summary_signal_efficiency, run["label"]
                )
            )
            available_xgb_bins.update(xgb_zenith_bins(xgb_model_data))

        available_xgb_bins = sorted(available_xgb_bins)
        available_tmva_bins = tmva_zenith_bins(root_dir, ebin) if root_dir else []
        xgb_bins_to_plot = selected_xgb_bins(args.zenith_bin_xgb, available_xgb_bins)

        for xgb_zebin in xgb_bins_to_plot:
            xgb_curves = []
            for run in xgb_runs:
                try:
                    xgb_curves.append(
                        xgb_efficiency_curve(
                            xgb_model_data_by_label[run["label"]],
                            ebin,
                            xgb_zebin,
                            run["label"],
                        )
                    )
                except KeyError as exc:
                    _logger.warning(
                        "Skipping %s for ebin %s, %s: %s",
                        run["label"],
                        ebin,
                        zenith_plot_label(xgb_zebin),
                        exc,
                    )
            if not xgb_curves:
                continue

            tmva_zebin = resolve_tmva_zebin(xgb_zebin, available_tmva_bins, args.zenith_bin_tmva)

            tmva_data = tmva_overlay_data(root_dir, ebin, xgb_zebin, tmva_zebin)
            if tmva_data is None:
                fig = make_figure(xgb_curves)
            else:
                x_root, y_effs, y_effb = tmva_data
                fig = make_figure(xgb_curves, x_root, y_effs, y_effb)

            ze_label = zenith_plot_label(xgb_zebin)
            _logger.info(f"Plotting plot_performance_metrics for ebin {ebin}, {ze_label}")
            output_path = output_dir / f"plot_performance_metrics_ebin{ebin}_{ze_label}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    summary_file = (
        Path(args.summary_file)
        if args.summary_file is not None
        else output_dir / "zenith_uniformity_summary.csv"
    )
    write_zenith_uniformity_summary(summary_rows, summary_file)
    plot_zenith_uniformity_vs_energy(summary_rows, output_dir / "zenith_uniformity_vs_energy.png")
    for run in xgb_runs:
        plot_zenith_background_efficiency_heatmap(
            heatmap_rows,
            run["label"],
            output_dir / f"zenith_background_efficiency_heatmap_{safe_label(run['label'])}.png",
        )


if __name__ == "__main__":
    main()
