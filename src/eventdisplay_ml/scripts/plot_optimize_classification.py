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

    # Plot 1: Signal efficiency vs energy
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = t["zenith_deg"] == zen
        energy = t["log10_energy_tev"][mask]
        gamma_eff = t["gamma_efficiency"][mask]
        ax.plot(energy, gamma_eff, marker="o", label=f"Zenith {zen:.1f}°", color=colors[i])

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Signal Efficiency (gamma-ray efficiency)")
    ax.set_title("Signal Efficiency vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "signal_efficiency_vs_energy.png"
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)

    # Plot 2: Significance vs energy
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = t["zenith_deg"] == zen
        energy = t["log10_energy_tev"][mask]
        significance = t["significance"][mask]
        ax.plot(energy, significance, marker="o", label=f"Zenith {zen:.1f}°", color=colors[i])

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Li & Ma Significance")
    ax.set_title("Significance vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "significance_vs_energy.png"
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)

    # Plot 3: Background efficiency vs energy
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = t["zenith_deg"] == zen
        energy = t["log10_energy_tev"][mask]
        bg_eff = t["background_efficiency"][mask]
        ax.plot(energy, bg_eff, marker="o", label=f"Zenith {zen:.1f}°", color=colors[i])

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Background Efficiency")
    ax.set_title("Background Efficiency vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "background_efficiency_vs_energy.png"
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)

    # Plot 4: Gamma-ray rate * signal efficiency vs energy
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = t["zenith_deg"] == zen
        energy = t["log10_energy_tev"][mask]
        gamma_rate = t["gamma_ray_rate"][mask] * t["gamma_efficiency"][mask]
        ax.plot(energy, gamma_rate, marker="o", label=f"Zenith {zen:.1f}°", color=colors[i])

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Gamma-ray Rate * Signal Efficiency [1/s]")
    ax.set_title("Gamma-ray Rate * Signal Efficiency vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "gamma_rate_times_signal_efficiency_vs_energy.png"
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)

    # Plot 5: Background rate * background efficiency vs energy
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = t["zenith_deg"] == zen
        energy = t["log10_energy_tev"][mask]
        bkg_rate = t["background_rate"][mask] * t["background_efficiency"][mask]
        ax.plot(energy, bkg_rate, marker="o", label=f"Zenith {zen:.1f}°", color=colors[i])

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Background Rate * Background Efficiency [1/s]")
    ax.set_title("Background Rate * Background Efficiency vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "background_rate_times_background_efficiency_vs_energy.png"
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)


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
