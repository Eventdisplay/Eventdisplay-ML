"""
Compare performance of TMVA and XGB gamma/hadron separator (efficiency based metrics).

./plot_performance_metrics.py \
        AP/BDTtraining/GammaHadronBDTs_V6_DISP/V6_2016_2017_ATM61/NTel2-Soft/BDT_3_0.root \
        AP/CARE_202404/V6_2016_2017_ATM61_gamma/TrainXGBGammaHadron/dispdir_bdt_ntel4_ebin3.joblib

"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import uproot

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def plot_efficiencies(ax, x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb):
    """Plot Signal and Background efficiencies vs. cut value (threshold)."""
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


def plot_qfactor(ax, y_effs, y_effb, y_effs_xgb, y_effb_xgb):
    """Plot Q-factor: Signal efficiency / sqrt(Background efficiency)."""
    q_tmva = np.divide(y_effs, np.sqrt(y_effb), out=np.zeros_like(y_effs), where=y_effb != 0)
    q_xgb = np.divide(
        y_effs_xgb, np.sqrt(y_effb_xgb), out=np.zeros_like(y_effs_xgb), where=y_effb_xgb != 0
    )

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


def plot_roc(ax, y_effs, y_effb, y_effs_xgb, y_effb_xgb):
    """Plot ROC curve: Signal efficiency vs. 1 - Background efficiency."""
    auc_tmva = -np.trapezoid(1 - y_effb, y_effs)
    auc_xgb = -np.trapezoid(1 - y_effb_xgb, y_effs_xgb)
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


def plot_score_distributions(ax, x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb):
    """Reconstructs and plots the probability density of the MVA scores."""
    # The derivative of the efficiency curve is the probability density function (PDF)
    # We use negative gradient because efficiency decreases as threshold increases
    pdf_s_tmva = -np.gradient(y_effs, x_root)
    pdf_b_tmva = -np.gradient(y_effb, x_root)

    pdf_s_xgb = -np.gradient(y_effs_xgb, x_joblib)
    pdf_b_xgb = -np.gradient(y_effb_xgb, x_joblib)

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
    with uproot.open(file_path) as rf:
        base_path = "Method_BDT/BDT_0"
        effs_rt = rf[f"{base_path}/MVA_BDT_0_effS"]
        effb_rt = rf[f"{base_path}/MVA_BDT_0_effB"]
        x_root_raw = (
            effs_rt.axis().centers() if hasattr(effs_rt, "axis") else effs_rt.values(axis=0)
        )
        x_min = np.min(x_root_raw)
        x_max = np.max(x_root_raw)
        # map [-x_min, x_max] -> [0, 1]
        x_root = (x_root_raw - x_min) / (x_max - x_min)
        y_effs = effs_rt.values()
        y_effb = effb_rt.values()

    return x_root, y_effs, y_effb


def load_efficiency_xgb(path, ebin, ntel=4):
    """Load efficiencies from XGB files."""
    # 2. XGBoost
    data_joblib = joblib.load(Path(path) / f"gammahadron_bdt_ntel{ntel}_ebin{ebin}.joblib")
    df_xgboost = data_joblib["models"]["xgboost"]["efficiency"]

    x_joblib = df_xgboost["threshold"]
    y_effs_xgb = df_xgboost["signal_efficiency"]
    y_effb_xgb = df_xgboost["background_efficiency"]

    return x_joblib, y_effs_xgb, y_effb_xgb


def main():
    """Plot TMVA and XGBoost performance metrics."""
    parser = argparse.ArgumentParser(description="Plot TMVA and XGBoost metrics.")
    parser.add_argument("root_dir", help="Path to the  TMVA BDT .root file")
    parser.add_argument("joblib_dir", help="Path to the XGB BDT .joblib file")
    args = parser.parse_args()

    # assume energy binning is identical in XGB and TMVA files.
    for ebin in range(9):
        x_root, y_effs, y_effb = load_efficiency_tmva(args.root_dir, ebin)
        x_joblib, y_effs_xgb, y_effb_xgb = load_efficiency_xgb(args.joblib_dir, ebin)

        fig, axs = plt.subplots(2, 2, figsize=(16, 16), sharex=False)
        fig.set_constrained_layout(True)

        for ax in axs.flatten():
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.2)

        plot_efficiencies(axs[0, 0], x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb)
        plot_qfactor(axs[0, 1], y_effs, y_effb, y_effs_xgb, y_effb_xgb)
        plot_roc(axs[1, 0], y_effs, y_effb, y_effs_xgb, y_effb_xgb)
        plot_score_distributions(
            axs[1, 1], x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb
        )

        for ax in axs.flatten():
            ax.legend(fontsize=9, frameon=False, loc="best")

        plt.tight_layout()
        _logger.info(f"Plotting plot_performance_metrics for ebin {ebin}")
        plt.savefig(f"plot_performance_metrics_ebin{ebin}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
