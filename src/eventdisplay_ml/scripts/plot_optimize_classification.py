"""
Plot the cut optimization results from optimize_classification.py.

Plots:

- signal efficiency vs energy with optimized cut for each zenith bin
- significance vs energy with optimized cut for each zenith bin
- background efficiency vs energy with optimized cut for each zenith bin

Input is the ECSV file produced by optimize_classification.py, output
are png files in the current directory.

"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _plot_by_zenith(
    table,
    zenith_angles,
    colors,
    output_dir,
    y_values_fn,
    y_label,
    title,
    filename,
):
    """Create one energy-dependent plot with one curve per zenith bin."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = table["zenith_deg"] == zen
        energy = np.asarray(table["log10_energy_tev"][mask])
        y_values = np.asarray(y_values_fn(table, mask))

        # Keep curves monotonic in energy for readability.
        order = np.argsort(energy)
        ax.plot(
            energy[order],
            y_values[order],
            marker="o",
            label=f"Zenith {zen:.1f}°",
            color=colors[i],
        )

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)


def plot_results(ecsv_file, output_dir=None):
    """Read and plot results from optimize_classification.py ECSV output file."""
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load the ECSV table
    t = Table.read(ecsv_file, format="ascii.ecsv")
    _logger.info(f"Loaded {len(t)} rows from {ecsv_file}")

    # Get unique zenith angles and sort them
    zenith_angles = np.unique(t["zenith_deg"])
    zenith_angles = np.sort(zenith_angles)
    _logger.info(f"Found {len(zenith_angles)} zenith angle bins: {zenith_angles}")

    # Create color map for zenith angles
    colors = plt.cm.viridis(np.linspace(0, 1, len(zenith_angles)))

    plot_specs = [
        {
            "y_values_fn": lambda table, mask: table["gamma_efficiency"][mask],
            "y_label": "Signal Efficiency (gamma-ray efficiency)",
            "title": "Signal Efficiency vs Energy",
            "filename": "signal_efficiency_vs_energy.png",
            "required_columns": ["gamma_efficiency"],
        },
        {
            "y_values_fn": lambda table, mask: table["significance"][mask],
            "y_label": "Li & Ma Significance",
            "title": "Significance vs Energy",
            "filename": "significance_vs_energy.png",
            "required_columns": ["significance"],
        },
        {
            "y_values_fn": lambda table, mask: table["background_efficiency"][mask],
            "y_label": "Background Efficiency",
            "title": "Background Efficiency vs Energy",
            "filename": "background_efficiency_vs_energy.png",
            "required_columns": ["background_efficiency"],
        },
        {
            "y_values_fn": lambda table, mask: (
                table["gamma_ray_rate"][mask] * table["gamma_efficiency"][mask]
            ),
            "y_label": "Gamma-ray Rate * Signal Efficiency [1/s]",
            "title": "Gamma-ray Rate * Signal Efficiency vs Energy",
            "filename": "gamma_rate_times_signal_efficiency_vs_energy.png",
            "required_columns": ["gamma_ray_rate", "gamma_efficiency"],
        },
        {
            "y_values_fn": lambda table, mask: (
                table["background_rate"][mask] * table["background_efficiency"][mask]
            ),
            "y_label": "Background Rate * Background Efficiency [1/s]",
            "title": "Background Rate * Background Efficiency vs Energy",
            "filename": "background_rate_times_background_efficiency_vs_energy.png",
            "required_columns": ["background_rate", "background_efficiency"],
        },
    ]

    missing_core_columns = [
        col for col in ["log10_energy_tev", "zenith_deg"] if col not in t.colnames
    ]
    if missing_core_columns:
        raise ValueError(f"Input ECSV is missing required columns: {missing_core_columns}")

    for spec in plot_specs:
        missing_for_spec = [col for col in spec["required_columns"] if col not in t.colnames]
        if missing_for_spec:
            _logger.warning(
                "Skipping plot '%s' due to missing columns: %s",
                spec["filename"],
                missing_for_spec,
            )
            continue
        _plot_by_zenith(
            table=t,
            zenith_angles=zenith_angles,
            colors=colors,
            output_dir=output_dir,
            y_values_fn=spec["y_values_fn"],
            y_label=spec["y_label"],
            title=spec["title"],
            filename=spec["filename"],
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot cut optimization results from optimize_classification.py"
    )
    parser.add_argument("ecsv_file", help="ECSV file produced by optimize_classification.py")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNG files (default: current directory)",
    )

    args = parser.parse_args()
    plot_results(args.ecsv_file, args.output_dir)


if __name__ == "__main__":
    main()
