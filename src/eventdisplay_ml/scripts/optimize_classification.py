"""
Optimize classification cuts for a target source strength.

This script derives a smooth, energy/zenith-dependent cut (gamma efficiency) that
maximizes the Li & Ma significance for a given fraction of the Crab flux, using
rate surfaces stored in a ROOT file.

Input ROOT file must contain:
- TGraph2DErrors gONRate : signal + background rate (1/s)
- TGraph2DErrors gBGRate : background rate (1/s)

The x-axis is log10(TeV), y-axis is zenith angle in degrees.
"""

# Review notes:
# - Programmer: prioritize clarity, vectorization, and safe fallbacks for ROOT I/O.
# - Statistician: Li & Ma Eq. 17 used; ensure alpha and live_time reflect analysis.
# - Gamma-ray astronomer: signal is derived from ON-BG; validate rate inputs and
#   the background-efficiency model (eps_b = eps_s**power) for your instrument.

import argparse
import logging
from pathlib import Path

import numpy as np
import uproot
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# signal-to-background exposure ratio (can be modified via command-line argument)
_ALPHA = 1.0 / 6.0


def _extract_tgraph2d(graph):
    """Extract x, y, z arrays from a TGraph2D/TGraph2DErrors."""
    try:
        x, y, z = graph.values()
        return np.asarray(x), np.asarray(y), np.asarray(z)
    except Exception:  # pragma: no cover - fallback for ROOT variants
        x = np.asarray(graph.member("fX"))
        y = np.asarray(graph.member("fY"))
        z = np.asarray(graph.member("fZ"))
        return x, y, z


def _load_rates(root_path):
    """Load ON and background rates from a ROOT file."""
    with uproot.open(root_path) as rf:
        g_on = rf["gONRate"]
        g_bg = rf["gBGRate"]
        x_on, y_on, r_on = _extract_tgraph2d(g_on)
        x_bg, y_bg, r_bg = _extract_tgraph2d(g_bg)

    if not (np.allclose(x_on, x_bg) and np.allclose(y_on, y_bg)):
        raise ValueError("gONRate and gBGRate do not share the same (x,y) points.")

    return x_on, y_on, r_on, r_bg


def _li_ma_significance(n_on, n_off, alpha):
    """Compute Li & Ma significance (Eq. 17), vectorized."""
    n_on = np.asarray(n_on, dtype=float)
    n_off = np.asarray(n_off, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_tot = n_on + n_off

        term1 = np.where(
            n_on > 0,
            n_on * np.log((1.0 + alpha) / alpha * (n_on / n_tot)),
            0.0,
        )

        term2 = np.where(
            n_off > 0,
            n_off * np.log((1.0 + alpha) * (n_off / n_tot)),
            0.0,
        )

        value = 2.0 * (term1 + term2)

    value = np.where(np.isfinite(value) & (value > 0), value, 0.0)

    sign = np.sign(n_on - alpha * n_off)
    return sign * np.sqrt(value)


def _optimize_cut(
    signal_rate,
    bg_rate,
    alpha,
    live_time_s,
    gamma_eff_grid,
    bg_eff_power,
):
    """Find gamma-efficiency cut that maximizes Li & Ma significance."""
    signal_rate = np.asarray(signal_rate, dtype=float)
    bg_rate = np.asarray(bg_rate, dtype=float)

    eff_s = gamma_eff_grid[None, :]
    eff_b = eff_s**bg_eff_power

    sig_rates = signal_rate[:, None] * eff_s
    bg_rates = bg_rate[:, None] * eff_b

    n_on = (sig_rates + bg_rates) * live_time_s
    n_off = bg_rates * live_time_s / alpha

    sig = _li_ma_significance(n_on, n_off, alpha)
    best_idx = np.nanargmax(sig, axis=1)

    best_eff = gamma_eff_grid[best_idx]
    best_sig = sig[np.arange(sig.shape[0]), best_idx]

    return best_eff, best_sig


def _significance_for_eff(
    signal_rate,
    bg_rate,
    gamma_eff,
    bg_eff_power,
    alpha,
    live_time_s,
):
    """Compute Li & Ma significance for a fixed gamma-efficiency cut."""
    eff_s = np.asarray(gamma_eff, dtype=float)
    eff_b = eff_s**bg_eff_power

    sig_rates = signal_rate * eff_s
    bg_rates = bg_rate * eff_b

    n_on = (sig_rates + bg_rates) * live_time_s
    n_off = bg_rates * live_time_s / alpha

    return _li_ma_significance(n_on, n_off, alpha)


def _smooth_field(
    x,
    y,
    values,
    grid_size,
    sigma,
):
    """Interpolate to a grid and apply Gaussian smoothing."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_grid = np.linspace(x_min, x_max, grid_size[0])
    y_grid = np.linspace(y_min, y_max, grid_size[1])
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")

    grid = griddata((x, y), values, (xx, yy), method="linear")
    if np.any(~np.isfinite(grid)):
        grid_nearest = griddata((x, y), values, (xx, yy), method="nearest")
        grid = np.where(np.isfinite(grid), grid, grid_nearest)

    grid = gaussian_filter(grid, sigma=sigma, mode="nearest")
    return x_grid, y_grid, grid


def _evaluate_on_points(
    x,
    y,
    x_grid,
    y_grid,
    grid,
):
    """Evaluate a gridded surface back on the original points."""
    interpolator = RegularGridInterpolator((y_grid, x_grid), grid, bounds_error=False)
    points = np.column_stack([y, x])
    values = interpolator(points)
    return np.asarray(values)


def _write_csv(
    output_path,
    rows,
):
    header = (
        "log10_energy_TeV,zenith_deg,cut_gamma_eff,cut_gamma_eff_smooth,"
        "significance,significance_smooth"
    )
    np.savetxt(
        output_path,
        np.asarray(list(rows)),
        fmt="%.6f",
        delimiter=",",
        header=header,
        comments="",
    )


def main():
    """Optimize classification cuts for a given source strength."""
    parser = argparse.ArgumentParser(
        description="Optimize gamma/hadron cuts using Li & Ma significance."
    )
    parser.add_argument("input_file", help="Input ROOT file with gONRate and gBGRate")
    parser.add_argument(
        "source_strength",
        type=float,
        help="Source strength as fraction of Crab (e.g. 0.1 for 10%)",
    )
    parser.add_argument(
        "--live_time",
        type=float,
        default=3600.0,
        help="Observation time in seconds (default: 3600)",
    )
    parser.add_argument(
        "--bg-eff-power",
        type=float,
        default=2.0,
        help="Background efficiency model: eps_b = eps_s**power (default: 2.0)",
    )
    parser.add_argument(
        "--gamma-eff-min",
        type=float,
        default=0.05,
        help="Minimum gamma efficiency to scan (default: 0.05)",
    )
    parser.add_argument(
        "--gamma-eff-steps",
        type=int,
        default=40,
        help="Number of gamma-efficiency steps to scan (default: 40)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=_ALPHA,
        help=f"ON/OFF exposure ratio alpha (default: {_ALPHA:.3f})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=(30, 30),
        metavar=("NX", "NY"),
        help="Interpolation grid size in (x,y) (default: 30 30)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma in grid units (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_cuts.csv",
        help="Output CSV file (default: optimized_cuts.csv)",
    )

    args = parser.parse_args()

    if args.source_strength <= 0:
        raise ValueError("source_strength must be positive (e.g. 0.1 for 10% Crab)")

    input_path = Path(args.input_file)
    x, y, on_rate, bg_rate = _load_rates(input_path)

    signal_rate_crab = np.clip(on_rate - bg_rate, 0.0, None)
    signal_rate = signal_rate_crab * float(args.source_strength)

    gamma_eff_grid = np.linspace(args.gamma_eff_min, 1.0, args.gamma_eff_steps)

    best_eff, best_sig = _optimize_cut(
        signal_rate,
        bg_rate,
        alpha=args.alpha,
        live_time_s=args.live_time,
        gamma_eff_grid=gamma_eff_grid,
        bg_eff_power=args.bg_eff_power,
    )

    x_grid, y_grid, eff_grid = _smooth_field(
        x, y, best_eff, grid_size=tuple(args.grid_size), sigma=args.smooth_sigma
    )
    eff_smooth = _evaluate_on_points(x, y, x_grid, y_grid, eff_grid)
    eff_smooth = np.clip(eff_smooth, args.gamma_eff_min, 1.0)

    sig_smooth = _significance_for_eff(
        signal_rate,
        bg_rate,
        gamma_eff=eff_smooth,
        bg_eff_power=args.bg_eff_power,
        alpha=args.alpha,
        live_time_s=args.live_time,
    )

    rows = zip(x, y, best_eff, eff_smooth, best_sig, sig_smooth)
    _write_csv(Path(args.output), rows)

    _logger.info(
        "Optimized cuts written to %s (alpha=%.3f, strength=%.3f)",
        args.output,
        args.alpha,
        args.source_strength,
    )


if __name__ == "__main__":
    main()
