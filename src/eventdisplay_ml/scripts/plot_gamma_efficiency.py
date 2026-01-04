"""
Plot gamma efficiency containment levels from mscw XGB gh MC files.

Allows to corr check the gamma efficiency at different containment levels
as function of zenith angle, wobble offset, and NSB level.
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def get_containment_data(directory):
    """Parse files and calculates containment levels."""
    directory = Path(directory)
    results = []
    # Regex to extract parameters from filename: 50deg_1.25wob_NOISE600.mscw.xgb_gh.root
    pattern = re.compile(r"(\d+)deg_([\d.]+)wob_NOISE(\d+)\.mscw\.xgb_gh\.root")

    files = [f.name for f in directory.iterdir() if f.name.endswith(".xgb_gh.root")]

    for filename in files:
        match = pattern.match(filename)
        if match:
            ze = int(match.group(1))
            wob = float(match.group(2))
            nsb = int(match.group(3))
            _logger.info(f"Processing file: {filename} for ze={ze}, wob={wob}, nsb={nsb}")

            with uproot.open(directory / filename) as f:
                if "Classification" in f:
                    tree = f["Classification"]
                    gamma_pred = tree["Gamma_Prediction"].array(library="np")
                    p70 = np.percentile(gamma_pred, 70)
                    p95 = np.percentile(gamma_pred, 95)
                    results.append({"ze": ze, "wob": wob, "nsb": nsb, "p70": p70, "p95": p95})

    return pd.DataFrame(results)


def plot_grid(df, x_axis_var, panel_vars, title, output_name):
    """
    Plot containment levels vs x_axis_var.

    Creates a grid of panels based on panel_vars (row, col).
    """
    rows = sorted(df[panel_vars[0]].unique())
    cols = sorted(df[panel_vars[1]].unique())

    _, axes = plt.subplots(
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


def main():
    """Plot gamma efficiency containment levels."""
    parser = argparse.ArgumentParser(description="Plot gamma efficiency containment levels.")
    parser.add_argument("directory", help="Directory containing .xgb_gh.root files")
    args = parser.parse_args()

    df = get_containment_data(args.directory)

    if not df.empty:
        plot_grid(df, "ze", ["wob", "nsb"], "Containment vs Zenith Angle", "containment_vs_ze.png")
        plot_grid(
            df, "wob", ["ze", "nsb"], "Containment vs Wobble Offset", "containment_vs_wob.png"
        )
        plot_grid(df, "nsb", ["ze", "wob"], "Containment vs NSB Level", "containment_vs_nsb.png")
    else:
        _logger.warning("No valid data found to plot.")


if __name__ == "__main__":
    main()
