"""
Plot gamma efficiency containment levels from mscw XGB gh MC files.

Allows to check the gamma efficiency at different containment levels
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
    """
    Parse files in mscw MC directory and calculates containment levels.

    Fixed 70 and 95% containment levels are calculated for each file.

    Parameters
    ----------
    directory : str or Path
        Directory containing .xgb_gh.root files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ze, wob, nsb, p70, p95
    """
    directory = Path(directory)
    results = []
    # Regex to extract parameters from filename: 50deg_1.25wob_NOISE600.mscw.xgb_gh.root
    pattern = re.compile(r"(\d+)deg_([\d.]+)wob_NOISE(\d+)\.mscw\.xgb_gh\.root")
    files = sorted(f.name for f in directory.iterdir() if f.name.endswith(".xgb_gh.root"))

    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue

        ze, wob, nsb = int(match.group(1)), float(match.group(2)), int(match.group(3))
        if nsb not in [160, 300, 600]:
            continue

        _logger.info(f"Processing file: {filename} for ze={ze}, wob={wob}, nsb={nsb}")

        path = f"{directory / filename}:Classification"

        try:
            data = uproot.concatenate(f"{path}/Gamma_Prediction", library="np")
            gamma_pred = data["Gamma_Prediction"].astype(np.float32)
            if len(gamma_pred) > 0 and not np.all(np.isnan(gamma_pred)):
                p70 = np.nanpercentile(gamma_pred, 100 - 70)
                p95 = np.nanpercentile(gamma_pred, 100 - 95)

            if not np.isnan(p70) and not np.isnan(p95):
                cos_ze = np.cos(np.deg2rad(ze))
                air_mass = 1.0 / cos_ze
                results.append(
                    {"ze": ze, "air_mass": air_mass, "wob": wob, "nsb": nsb, "p70": p70, "p95": p95}
                )
                _logger.info(f"Percentiles p70: {p70}, p95: {p95}")
            else:
                _logger.warning("Nan percentiles")
        except Exception as e:
            _logger.error(f"Failed reading {filename}: {e}")

    return pd.DataFrame(results)


def plot_grid(df, x_axis_var, panel_vars, title, output_name):
    """Plot containment levels vs x_axis_var."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(df[panel_vars[0]].unique())))
    line_vars = sorted(df[panel_vars[0]].unique())
    cols = sorted(df[panel_vars[1]].unique())

    _, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5), sharey=True, squeeze=False)

    for j, c_val in enumerate(cols):
        ax = axes[0, j]
        for k, l_val in enumerate(line_vars):
            subset = df[(df[panel_vars[1]] == c_val) & (df[panel_vars[0]] == l_val)].sort_values(
                by=x_axis_var
            )

            if not subset.empty:
                label_p70 = f"{l_val:.2f} {panel_vars[0]}" if j == 0 else None
                ax.plot(subset[x_axis_var], subset["p70"], "o-", color=colors[k], label=label_p70)
                # Use alpha or dashed for 95% to keep it clean
                ax.plot(subset[x_axis_var], subset["p95"], "s--", color=colors[k], alpha=0.5)

        ax.set_title(f"{panel_vars[1]} = {c_val}")
        ax.set_xlabel(x_axis_var.upper())
        if j == 0:
            ax.set_ylabel("Gamma_Prediction Level")
            ax.legend(title=panel_vars[0], bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name)
    _logger.info(f"Plotted {output_name}")


def main():
    """Plot gamma efficiency containment levels."""
    parser = argparse.ArgumentParser(description="Plot gamma efficiency containment levels.")
    parser.add_argument("directory", help="Directory containing .xgb_gh.root files")
    args = parser.parse_args()

    df = get_containment_data(args.directory)

    if not df.empty:
        plot_grid(
            df,
            "air_mass",
            ["wob", "nsb"],
            "Containment (70/95%) vs Air Mass",
            "containment_vs_airmass.png",
        )
        plot_grid(
            df,
            "wob",
            ["air_mass", "nsb"],
            "Containment (70/95%) vs Wobble Offset",
            "containment_vs_wob.png",
        )
    else:
        _logger.warning("No valid data found to plot.")


if __name__ == "__main__":
    main()
