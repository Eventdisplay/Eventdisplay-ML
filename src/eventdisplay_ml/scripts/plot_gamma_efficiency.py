import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot


def get_containment_data(
    directory="/lustre/fs23/group/veritas/IRFPRODUCTION/v492/AP/CARE_202404/V6_2016_2017_ATM61_gamma/MSCW_RECID0_DISP/",
):
    """Parses ROOT files and calculates containment levels."""
    results = []
    # Regex to extract parameters from filename: 50deg_1.25wob_NOISE600.mscw.xgb_gh.root
    pattern = re.compile(r"(\d+)deg_([\d.]+)wob_NOISE(\d+)\.mscw\.xgb_gh\.root")

    files = [f for f in os.listdir(directory) if f.endswith(".xgb_gh.root")]

    for filename in files:
        match = pattern.match(filename)
        if match:
            ze = int(match.group(1))
            wob = float(match.group(2))
            nsb = int(match.group(3))

            with uproot.open(os.path.join(directory, filename)) as f:
                # Access the 'Classification' tree
                if "Classification" in f:
                    tree = f["Classification"]
                    # Load Gamma_Prediction as a numpy array
                    gamma_pred = tree["Gamma_Prediction"].array(library="np")

                    # Calculate 70% and 95% containment (percentiles)
                    # Note: Using 30th and 5th percentiles if 'containment' refers
                    # to the top fraction of a [0, 1] score.
                    # Here we use standard 70th and 95th percentiles.
                    p70 = np.percentile(gamma_pred, 70)
                    p95 = np.percentile(gamma_pred, 95)

                    results.append({"ze": ze, "wob": wob, "nsb": nsb, "p70": p70, "p95": p95})

    return pd.DataFrame(results)


def plot_grid(df, x_axis_var, panel_vars, title, output_name):
    """
    Plots containment levels vs x_axis_var.
    Creates a grid of panels based on panel_vars (row, col).
    """
    rows = sorted(df[panel_vars[0]].unique())
    cols = sorted(df[panel_vars[1]].unique())

    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(5 * len(cols), 4 * len(rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, r_val in enumerate(rows):
        for j, c_val in enumerate(cols):
            ax = axes[i, j]
            # Filter data for this specific panel
            subset = df[(df[panel_vars[0]] == r_val) & (df[panel_vars[1]] == c_val)].sort_values(
                by=x_axis_var
            )

            if not subset.empty:
                ax.plot(subset[x_axis_var], subset["p70"], "o-", label="70% Containment")
                ax.plot(subset[x_axis_var], subset["p95"], "s--", label="95% Containment")

            ax.set_title(f"{panel_vars[0]}={r_val}, {panel_vars[1]}={c_val}")
            if i == len(rows) - 1:
                ax.set_xlabel(x_axis_var.upper())
            if j == 0:
                ax.set_ylabel("Gamma_Prediction Level")
            if i == 0 and j == 0:
                ax.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name)
    plt.show()


# Execution
df = get_containment_data()

if not df.empty:
    # 1. vs ZE for each (wob, NSB)
    plot_grid(df, "ze", ["wob", "nsb"], "Containment vs Zenith Angle", "containment_vs_ze.png")

    # 2. vs Wobble for each (ZE, NSB)
    plot_grid(df, "wob", ["ze", "nsb"], "Containment vs Wobble Offset", "containment_vs_wob.png")

    # 3. vs NSB for each (ZE, wob)
    plot_grid(df, "nsb", ["ze", "wob"], "Containment vs NSB Level", "containment_vs_nsb.png")
else:
    print("No matching ROOT files found.")
