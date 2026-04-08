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


def _select_profile_zenith_angles(zenith_angles, step_deg=10.0):
    """Select representative zenith slices close to a fixed angular spacing."""
    zenith_angles = np.sort(np.asarray(zenith_angles, dtype=float))
    if zenith_angles.size == 0:
        return zenith_angles

    target_zenith_angles = np.arange(
        np.ceil(zenith_angles.min() / step_deg) * step_deg,
        zenith_angles.max() + 0.5 * step_deg,
        step_deg,
    )

    selected_zenith_angles = []
    for target_zenith in target_zenith_angles:
        closest_index = np.argmin(np.abs(zenith_angles - target_zenith))
        closest_zenith = zenith_angles[closest_index]
        if not selected_zenith_angles or not np.isclose(selected_zenith_angles[-1], closest_zenith):
            selected_zenith_angles.append(closest_zenith)

    if not np.isclose(selected_zenith_angles[0], zenith_angles[0]):
        selected_zenith_angles.insert(0, zenith_angles[0])
    if not np.isclose(selected_zenith_angles[-1], zenith_angles[-1]):
        selected_zenith_angles.append(zenith_angles[-1])

    return np.array(selected_zenith_angles, dtype=float)


def _reshape_surface(table, value_column):
    """Reshape a table column onto the regular energy/zenith grid."""
    energy_axis = np.sort(np.unique(np.asarray(table["log10_energy_tev"], dtype=float)))
    zenith_axis = np.sort(np.unique(np.asarray(table["zenith_deg"], dtype=float)))
    surface = np.full((len(zenith_axis), len(energy_axis)), np.nan, dtype=float)

    energy_index = {float(value): idx for idx, value in enumerate(energy_axis)}
    zenith_index = {float(value): idx for idx, value in enumerate(zenith_axis)}

    for energy, zenith, value in zip(
        table["log10_energy_tev"], table["zenith_deg"], table[value_column], strict=True
    ):
        surface[zenith_index[float(zenith)], energy_index[float(energy)]] = value

    return energy_axis, zenith_axis, surface


def _plot_colz(table, output_dir, value_column, colorbar_label, title, filename, log_z=False):
    """Create a 2D colormap plot in log10 energy and zenith."""
    energy_axis, zenith_axis, surface = _reshape_surface(table, value_column)

    if np.all(~np.isfinite(surface)):
        _logger.warning("Skipping colz plot '%s' because all values are non-finite.", filename)
        return

    plot_surface = surface.copy()
    if log_z:
        positive_mask = plot_surface > 0
        if not np.any(positive_mask):
            _logger.warning(
                "Skipping colz plot '%s' because no positive values are available.",
                filename,
            )
            return
        plot_surface[positive_mask] = np.log10(plot_surface[positive_mask])
        plot_surface[~positive_mask] = np.nan
        colorbar_label = f"log10({colorbar_label})"

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.pcolormesh(energy_axis, zenith_axis, plot_surface, shading="nearest", cmap="viridis")
    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Zenith Angle [deg]")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150)
    _logger.info(f"Saved {output_path}")
    plt.close(fig)


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
    profile_zenith_angles = _select_profile_zenith_angles(zenith_angles)
    _logger.info(
        "Using %d representative zenith slices for 1D energy plots: %s",
        len(profile_zenith_angles),
        profile_zenith_angles,
    )

    # Create color map for zenith angles
    colors = plt.cm.viridis(np.linspace(0, 1, len(profile_zenith_angles)))

    # Signal and background rates (no change of binning)
    _plot_rates_by_zenith(
        table=t,
        zenith_angles=profile_zenith_angles,
        colors=colors,
        output_dir=output_dir,
        filename="signal_background_rates_vs_energy.png",
    )

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
            zenith_angles=profile_zenith_angles,
            colors=colors,
            output_dir=output_dir,
            y_values_fn=spec["y_values_fn"],
            y_label=spec["y_label"],
            title=spec["title"],
            filename=spec["filename"],
        )

    colz_specs = [
        {
            "value_column": "gamma_efficiency",
            "colorbar_label": "Signal Efficiency",
            "title": "Signal Efficiency in Energy and Zenith",
            "filename": "signal_efficiency_colz.png",
            "log_z": False,
        },
        {
            "value_column": "significance",
            "colorbar_label": "Li & Ma Significance",
            "title": "Significance in Energy and Zenith",
            "filename": "significance_colz.png",
            "log_z": True,
        },
        {
            "value_column": "background_efficiency",
            "colorbar_label": "Background Efficiency",
            "title": "Background Efficiency in Energy and Zenith",
            "filename": "background_efficiency_colz.png",
            "log_z": False,
        },
        {
            "value_column": "gamma_ray_rate",
            "colorbar_label": "Gamma-ray Rate [1/s]",
            "title": "Gamma-ray Rate in Energy and Zenith",
            "filename": "gamma_rate_colz.png",
            "log_z": True,
        },
        {
            "value_column": "background_rate",
            "colorbar_label": "Background Rate [1/s]",
            "title": "Background Rate in Energy and Zenith",
            "filename": "background_rate_colz.png",
            "log_z": True,
        },
    ]

    for spec in colz_specs:
        if spec["value_column"] not in t.colnames:
            _logger.warning(
                "Skipping colz plot '%s' due to missing column: %s",
                spec["filename"],
                spec["value_column"],
            )
            continue
        _plot_colz(
            table=t,
            output_dir=output_dir,
            value_column=spec["value_column"],
            colorbar_label=spec["colorbar_label"],
            title=spec["title"],
            filename=spec["filename"],
            log_z=spec["log_z"],
        )


def _plot_rates_by_zenith(table, zenith_angles, colors, output_dir, filename):
    """Plot signal and background rates vs energy for each zenith angle."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, zen in enumerate(zenith_angles):
        mask = table["zenith_deg"] == zen
        energy = np.asarray(table["log10_energy_tev"][mask])
        signal_rate = np.asarray(table["gamma_ray_rate"][mask])
        background_rate = np.asarray(table["background_rate"][mask])

        order = np.argsort(energy)
        ax.plot(
            energy[order],
            signal_rate[order],
            marker="o",
            label=f"Signal, Zenith {zen:.1f}°",
            color=colors[i],
            linestyle="-",
        )
        ax.plot(
            energy[order],
            background_rate[order],
            marker="s",
            label=f"Background, Zenith {zen:.1f}°",
            color=colors[i],
            linestyle="--",
        )

    ax.set_xlabel("log10(Energy [TeV])")
    ax.set_ylabel("Rate [1/s]")
    ax.set_title("Signal and Background Rates vs Energy (per zenith bin)")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / filename
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
