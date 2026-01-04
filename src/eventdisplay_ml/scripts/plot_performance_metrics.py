"""
Compare performance of TMVA and XGB gamma/hadron separator (efficiency based metrics).

./plot_performance_metrics.py \
        AP/BDTtraining/GammaHadronBDTs_V6_DISP/V6_2016_2017_ATM61/NTel2-Soft/BDT_3_0.root \
        AP/CARE_202404/V6_2016_2017_ATM61_gamma/TrainXGBGammaHadron/dispdir_bdt_ntel4_ebin3.joblib

"""

import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import uproot


def plot_efficiencies(ax, x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb):
    """Plot Signal and Background efficiencies vs. cut value (threshold)."""
    ax.plot(x_root, y_effs, label="TMVA BDT Eff S", color="blue", linestyle="-", linewidth=2)
    ax.plot(x_root, y_effb, label="TMVA BDT Eff B", color="red", linestyle="-", linewidth=2)
    ax.plot(x_joblib, y_effs_xgb, label="XGB Eff S", color="cyan", linestyle="--", linewidth=2)
    ax.plot(
        x_joblib, y_effb_xgb, label="XGB Eff B", color="darkorange", linestyle="--", linewidth=2
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
    )

    ax.set_xlabel(r"Gamma Efficiency ($\epsilon_{\gamma}$)")
    ax.set_ylabel(r"Q-factor ($\epsilon_{\gamma} / \sqrt{\epsilon_{h}}$)")
    ax.set_title("Q-Factor")


def plot_roc(ax, y_effs, y_effb, y_effs_xgb, y_effb_xgb):
    """Plot ROC curve: Signal efficiency vs. 1 - Background efficiency."""
    ax.plot(y_effs, 1 - y_effb, label="TMVA", color="blue")
    ax.plot(y_effs_xgb, 1 - y_effb_xgb, label="XGBoost", color="cyan", linestyle="--")

    ax.margins(x=0.02)
    ax.set_xlabel("Gamma Efficiency (Signal)")
    ax.set_ylabel("Hadron Rejection (1 - Background Eff)")
    ax.set_title("ROC")


def main():
    """Plot TMVA and XGBoost performance metrics."""
    parser = argparse.ArgumentParser(description="Plot TMVA and XGBoost metrics.")
    parser.add_argument("root_file", help="Path to the .root file")
    parser.add_argument("joblib_file", help="Path to the .joblib file")
    args = parser.parse_args()

    # 1. TMVA
    with uproot.open(args.root_file) as rf:
        base_path = "Method_BDT/BDT_0"
        effs_rt = rf[f"{base_path}/MVA_BDT_0_effS"]
        effb_rt = rf[f"{base_path}/MVA_BDT_0_effB"]
        x_root_raw = (
            effs_rt.axis().centers() if hasattr(effs_rt, "axis") else effs_rt.values(axis=0)
        )
        max_val = np.max(np.abs(x_root_raw))
        # map [-max_val, max_val] -> [0, 1]
        x_root = (x_root_raw + max_val) / (2 * max_val)
        y_effs = effs_rt.values()
        y_effb = effb_rt.values()

    # 2. XGBoost
    data_joblib = joblib.load(args.joblib_file)
    df_xgboost = data_joblib["models"]["xgboost"]["efficiency"]

    x_joblib = df_xgboost["threshold"]
    y_effs_xgb = df_xgboost["signal_efficiency"]
    y_effb_xgb = df_xgboost["background_efficiency"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharex=False)
    fig.set_constrained_layout(True)

    for ax in axs:
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2)

    plot_efficiencies(axs[0], x_root, y_effs, y_effb, x_joblib, y_effs_xgb, y_effb_xgb)
    plot_qfactor(axs[1], y_effs, y_effb, y_effs_xgb, y_effb_xgb)
    plot_roc(axs[2], y_effs, y_effb, y_effs_xgb, y_effb_xgb)

    for ax in axs:
        ax.legend(fontsize=9, frameon=False, loc="best")

    plt.tight_layout()
    print("Plotting plot_performance_metrics")
    plt.savefig("plot_performance_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
