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
import logging
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import uproot

from eventdisplay_ml import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def plot_efficiencies(ax, x_joblib, y_effs_xgb, y_effb_xgb, x_root=None, y_effs=None, y_effb=None):
    """Plot Signal and Background efficiencies vs. cut value (threshold)."""
    if x_root is not None and y_effs is not None and y_effb is not None:
        ax.plot(x_root, y_effs, label="TMVA BDT Eff S", color="blue", linestyle="-", linewidth=2)
        ax.plot(x_root, y_effb, label="TMVA BDT Eff B", color="red", linestyle="-", linewidth=2)
    ax.plot(x_joblib, y_effs_xgb, label="XGB Eff S", color="cyan", linestyle="--", linewidth=2)
    ax.plot(
        x_joblib, y_effb_xgb, label="XGB Eff B", color="darkorange", linestyle="--", linewidth=4
    )

    ax.set_xlabel("Cut value (Threshold)")
    ax.set_ylabel("Efficiency")
    ax.set_title("Signal / Background Efficiency")
    ax.set_ylim(0, 1.05)


def plot_qfactor(ax, y_effs_xgb, y_effb_xgb, y_effs=None, y_effb=None):
    """Plot Q-factor: Signal efficiency / sqrt(Background efficiency)."""
    q_xgb = np.divide(
        y_effs_xgb, np.sqrt(y_effb_xgb), out=np.zeros_like(y_effs_xgb), where=y_effb_xgb != 0
    )

    if y_effs is not None and y_effb is not None:
        q_tmva = np.divide(y_effs, np.sqrt(y_effb), out=np.zeros_like(y_effs), where=y_effb != 0)
        ax.plot(y_effs, q_tmva, label=f"TMVA (Max Q: {np.max(q_tmva):.2f})", color="blue")
    ax.plot(
        y_effs_xgb,
        q_xgb,
        label=f"XGBoost (Max Q: {np.max(q_xgb):.2f})",
        color="cyan",
        linestyle="--",
        linewidth=4,
    )

    ax.set_xlabel(r"Gamma Efficiency ($\epsilon_{\gamma}$)")
    ax.set_ylabel(r"Q-factor ($\epsilon_{\gamma} / \sqrt{\epsilon_{h}}$)")
    ax.set_title("Q-Factor")


def plot_roc(ax, y_effs_xgb, y_effb_xgb, y_effs=None, y_effb=None):
    """Plot ROC curve: Signal efficiency vs. 1 - Background efficiency."""
    auc_xgb = -np.trapezoid(1 - y_effb_xgb, y_effs_xgb)
    if y_effs is not None and y_effb is not None:
        auc_tmva = -np.trapezoid(1 - y_effb, y_effs)
        ax.plot(y_effs, 1 - y_effb, label=f"TMVA (AUC: {auc_tmva:.2f})", color="blue")
    ax.plot(
        y_effs_xgb,
        1 - y_effb_xgb,
        label=f"XGBoost (AUC: {auc_xgb:.2f})",
        color="cyan",
        linestyle="--",
        linewidth=4,
    )

    ax.margins(x=0.02)
    ax.set_xlabel("Gamma Efficiency (Signal)")
    ax.set_ylabel("Hadron Rejection (1 - Background Eff)")
    ax.set_title("ROC")


def plot_score_distributions(
    ax, x_joblib, y_effs_xgb, y_effb_xgb, x_root=None, y_effs=None, y_effb=None
):
    """Reconstructs and plots the probability density of the MVA scores."""
    # The derivative of the efficiency curve is the probability density function (PDF)
    # We use negative gradient because efficiency decreases as threshold increases
    pdf_s_xgb = -np.gradient(y_effs_xgb, x_joblib)
    pdf_b_xgb = -np.gradient(y_effb_xgb, x_joblib)

    if x_root is not None and y_effs is not None and y_effb is not None:
        pdf_s_tmva = -np.gradient(y_effs, x_root)
        pdf_b_tmva = -np.gradient(y_effb, x_root)
        ax.fill_between(x_root, pdf_s_tmva, alpha=0.2, color="blue", label="TMVA Signal")
        ax.fill_between(x_root, pdf_b_tmva, alpha=0.2, color="red", label="TMVA Background")

    ax.plot(x_joblib, pdf_s_xgb, color="cyan", linestyle="--", label="XGB Signal", linewidth=4)
    ax.plot(
        x_joblib, pdf_b_xgb, color="darkorange", linestyle="--", label="XGB Background", linewidth=4
    )

    ax.set_xlabel("MVA Score (Normalized)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Score Distributions")


def load_efficiency_tmva(path, ebin, zebin=0):
    """Load efficiencies from TMVA root files."""
    file_path = Path(path) / f"BDT_{ebin}_{zebin}.root"
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
            # map [-x_min, x_max] -> [0, 1]
            x_root = (x_root_raw - x_min) / (x_max - x_min)
            y_effs = effs_rt.values()
            y_effb = effb_rt.values()
    except OSError as exc:
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


def load_efficiency_xgb(path, ebin, zebin=-1):
    """Load efficiencies from XGB files."""
    model_file = utils.resolve_joblib_path(Path(path) / f"gammahadron_bdt_ebin{ebin}")
    data_joblib = joblib.load(model_file)
    efficiency_key = "efficiency" if zebin < 0 else f"efficiency_ze{zebin}"
    model_data = data_joblib["models"]["xgboost"]
    if efficiency_key not in model_data:
        _logger.warning(
            "Efficiency key '%s' not found for ebin %s. Falling back to 'efficiency'.",
            efficiency_key,
            ebin,
        )
        efficiency_key = "efficiency"
    df_xgboost = model_data[efficiency_key]

    x_joblib = df_xgboost["threshold"]
    y_effs_xgb = df_xgboost["signal_efficiency"]
    y_effb_xgb = df_xgboost["background_efficiency"]

    return x_joblib, y_effs_xgb, y_effb_xgb


def xgb_zenith_bins(path, ebin):
    """Return available XGB zenith bins from the joblib model file."""
    model_file = utils.resolve_joblib_path(Path(path) / f"gammahadron_bdt_ebin{ebin}")
    data_joblib = joblib.load(model_file)
    model_data = data_joblib["models"]["xgboost"]
    ze_bins = []
    for key in model_data:
        match = re.fullmatch(r"efficiency_ze(\d+)", key)
        if match:
            ze_bins.append(int(match.group(1)))
    return sorted(set(ze_bins))


def tmva_zenith_bins(path, ebin):
    """Return available TMVA zenith bins from BDT_<ebin>_<zebin>.root filenames."""
    ze_bins = []
    pattern = re.compile(rf"^BDT_{ebin}_(\d+)\.root$")
    for file_path in Path(path).glob(f"BDT_{ebin}_*.root"):
        match = pattern.match(file_path.name)
        if match:
            ze_bins.append(int(match.group(1)))
    return sorted(set(ze_bins))


def resolve_tmva_zebin(xgb_zebin, available_tmva_bins, fallback_tmva_bin):
    """Resolve TMVA zenith bin aligned to XGB zenith bin where possible."""
    if xgb_zebin < 0:
        return fallback_tmva_bin if fallback_tmva_bin in available_tmva_bins else None
    if xgb_zebin in available_tmva_bins:
        return xgb_zebin
    if fallback_tmva_bin in available_tmva_bins:
        return fallback_tmva_bin
    return None


def zenith_plot_label(zebin):
    """Return human-readable zenith label for plot/file naming."""
    return "overall" if zebin < 0 else f"ze{zebin}"


def style_axis(ax):
    """Apply common style settings to a matplotlib axis."""
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)


def make_figure(x_joblib, y_effs_xgb, y_effb_xgb, x_root=None, y_effs=None, y_effb=None):
    """Build 2x2 diagnostics figure for XGB with optional TMVA overlays."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharex=False)
    fig.set_constrained_layout(True)

    for ax in axs.flatten():
        style_axis(ax)

    plot_efficiencies(axs[0, 0], x_joblib, y_effs_xgb, y_effb_xgb, x_root, y_effs, y_effb)
    plot_qfactor(axs[0, 1], y_effs_xgb, y_effb_xgb, y_effs, y_effb)
    plot_roc(axs[1, 0], y_effs_xgb, y_effb_xgb, y_effs, y_effb)
    plot_score_distributions(axs[1, 1], x_joblib, y_effs_xgb, y_effb_xgb, x_root, y_effs, y_effb)

    for ax in axs.flatten():
        ax.legend(fontsize=9, frameon=False, loc="best")

    plt.tight_layout()
    return fig


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
        required=True,
        help="Path to XGB BDT joblib files (required).",
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
        default=0,
        help="Zenith bin index for TMVA ROOT files (second digit in BDT_<ebin>_<zebin>.root). Default: 0.",
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
    args = parser.parse_args()
    root_dir = args.tmva_dir
    joblib_dir = args.xgb_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # assume energy binning is identical in XGB and TMVA files.
    energy_bins = [args.energy_bin] if args.energy_bin is not None else range(9)
    for ebin in energy_bins:
        available_xgb_bins = xgb_zenith_bins(joblib_dir, ebin)
        available_tmva_bins = tmva_zenith_bins(root_dir, ebin) if root_dir else []
        xgb_bins_to_plot = selected_xgb_bins(args.zenith_bin_xgb, available_xgb_bins)

        for xgb_zebin in xgb_bins_to_plot:
            x_joblib, y_effs_xgb, y_effb_xgb = load_efficiency_xgb(joblib_dir, ebin, xgb_zebin)
            tmva_zebin = resolve_tmva_zebin(xgb_zebin, available_tmva_bins, args.zenith_bin_tmva)

            tmva_data = tmva_overlay_data(root_dir, ebin, xgb_zebin, tmva_zebin)
            if tmva_data is None:
                fig = make_figure(x_joblib, y_effs_xgb, y_effb_xgb)
            else:
                x_root, y_effs, y_effb = tmva_data
                fig = make_figure(x_joblib, y_effs_xgb, y_effb_xgb, x_root, y_effs, y_effb)

            ze_label = zenith_plot_label(xgb_zebin)
            _logger.info(f"Plotting plot_performance_metrics for ebin {ebin}, {ze_label}")
            output_path = output_dir / f"plot_performance_metrics_ebin{ebin}_{ze_label}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
