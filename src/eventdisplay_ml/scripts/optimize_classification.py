"""
Optimize classification cuts for a target source strength.

This script derives a smooth, energy/zenith-dependent cut (gamma efficiency) that
maximizes the Li & Ma significance for a given fraction of the Crab flux, using
rate surfaces stored in a ROOT file.

Input ROOT file must contain:
- TGraph2DErrors gONRate : signal + background rate (1/s)
- TGraph2DErrors gBGRate : background rate (1/s)

"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import uproot
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_ALPHA = 1.0 / 6.0


def _load_multi_bin_roc(joblib_paths):
    """
    Load multiple joblib files and creates 2D interpolators.

    Returns
    -------
    bg_interp : LinearNDInterpolator
        Interpolator for background efficiency as a function of (log10_energy, signal_efficiency).
    thresh_interp : LinearNDInterpolator
        Interpolator for BDT threshold as a function of (log10_energy, signal_efficiency).
    energy_bins_map : dict
        Mapping from energy bin center to (E_min, E_max) for each energy bin.
    """
    all_coords = []
    all_bg = []
    all_thresh = []
    energy_bins_map = {}

    _logger.info(f"Loading {len(joblib_paths)} energy-dependent ROC files...")

    for path in joblib_paths:
        try:
            data = joblib.load(path)
            ebins = data["energy_bins_log10_tev"]
            e_min = ebins["E_min"]
            e_max = ebins["E_max"]
            e_center = (e_min + e_max) / 2.0
            energy_bins_map[e_center] = (e_min, e_max)

            df = data["models"]["xgboost"]["efficiency"]

            for _, row in df.iterrows():
                all_coords.append([e_center, row["signal_efficiency"]])
                all_bg.append(row["background_efficiency"])
                all_thresh.append(row["threshold"])
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Error loading {path}: {e}")

    coords = np.array(all_coords)
    bg_interp = LinearNDInterpolator(coords, np.array(all_bg))
    thresh_interp = LinearNDInterpolator(coords, np.array(all_thresh))

    return bg_interp, thresh_interp, energy_bins_map


def _extract_tgraph2d(graph):
    """Extract x, y, z arrays from a TGraph2D/TGraph2DErrors."""
    try:
        x, y, z = graph.values()
        return np.asarray(x), np.asarray(y), np.asarray(z)
    except Exception:
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
        term1 = np.where(n_on > 0, n_on * np.log((1.0 + alpha) / alpha * (n_on / n_tot)), 0.0)
        term2 = np.where(n_off > 0, n_off * np.log((1.0 + alpha) * (n_off / n_tot)), 0.0)
        value = 2.0 * (term1 + term2)
    value = np.where(np.isfinite(value) & (value > 0), value, 0.0)
    return np.sign(n_on - alpha * n_off) * np.sqrt(value)


def _optimize_cut_2d(log10_e, signal_rate, bg_rate, alpha, livetime_s, gamma_eff_grid, bg_interp):
    """Find gamma-efficiency cut that maximizes Li & Ma using energy-aware ROC."""
    best_effs = []
    for i, e_val in enumerate(log10_e):
        eval_points = np.column_stack([np.full(len(gamma_eff_grid), e_val), gamma_eff_grid])
        eff_s = gamma_eff_grid
        eff_b = np.nan_to_num(bg_interp(eval_points), nan=1.0)

        n_on = (signal_rate[i] * eff_s + bg_rate[i] * eff_b) * livetime_s
        n_off = (bg_rate[i] * eff_b) * livetime_s / alpha
        sigs = _li_ma_significance(n_on, n_off, alpha)

        best_effs.append(eff_s[np.nanargmax(sigs)])
    return np.array(best_effs)


def _map_energy_to_bins(energy_values, energy_bins_map):
    """
    Map energy values to their bin min/max.

    For each energy value, find the closest bin center and return its min/max.
    """
    bin_centers = np.array(sorted(energy_bins_map.keys()))
    e_min_list = []
    e_max_list = []

    for e_val in energy_values:
        # Find closest bin center
        closest_idx = np.argmin(np.abs(bin_centers - e_val))
        closest_center = bin_centers[closest_idx]
        e_min, e_max = energy_bins_map[closest_center]
        e_min_list.append(e_min)
        e_max_list.append(e_max)

    return np.array(e_min_list), np.array(e_max_list)


def _smooth_field(x, y, values, grid_size, sigma):
    """Interpolate to a grid and apply Gaussian smoothing."""
    x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
    x_grid = np.linspace(x_min, x_max, grid_size[0])
    y_grid = np.linspace(y_min, y_max, grid_size[1])
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    grid = griddata((x, y), values, (xx, yy), method="linear")
    if np.any(~np.isfinite(grid)):
        grid = np.where(
            np.isfinite(grid), grid, griddata((x, y), values, (xx, yy), method="nearest")
        )
    grid = gaussian_filter(grid, sigma=sigma, mode="nearest")
    return x_grid, y_grid, grid


def _evaluate_on_points(x, y, x_grid, y_grid, grid):
    """Evaluate gridded surface back on points."""
    interpolator = RegularGridInterpolator((y_grid, x_grid), grid, bounds_error=False)
    return interpolator(np.column_stack([y, x]))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optimize classification cuts.")
    parser.add_argument("input_root", help="ROOT file with rate surfaces")
    parser.add_argument("roc_files", nargs="+", help="List of ebin*.joblib files")
    parser.add_argument("source_strength", type=float, help="Fraction of Crab (e.g. 0.1 for 10%%)")
    parser.add_argument("--livetime", type=float, default=3600.0)
    parser.add_argument("--gamma-eff-min", type=float, default=0.01)
    parser.add_argument("--gamma-eff-steps", type=int, default=200)
    parser.add_argument("--grid-size", type=int, nargs=2, default=(50, 50))
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--output", type=str, default="optimized_cuts.ecsv")

    args = parser.parse_args()

    bg_interp, thresh_interp, energy_bins_map = _load_multi_bin_roc(args.roc_files)
    x, y, on_rate, bg_rate = _load_rates(Path(args.input_root))
    signal_rate = np.clip(on_rate - bg_rate, 0.0, None) * args.source_strength

    energy_min, energy_max = _map_energy_to_bins(x, energy_bins_map)

    gamma_eff_grid = np.linspace(args.gamma_eff_min, 1.0, args.gamma_eff_steps)
    best_eff = _optimize_cut_2d(
        x, signal_rate, bg_rate, _ALPHA, args.livetime, gamma_eff_grid, bg_interp
    )

    x_grid, y_grid, eff_grid = _smooth_field(
        x, y, best_eff, tuple(args.grid_size), args.smooth_sigma
    )
    eff_smooth = np.clip(
        _evaluate_on_points(x, y, x_grid, y_grid, eff_grid), args.gamma_eff_min, 1.0
    )

    eval_coords = np.column_stack([x, eff_smooth])
    bg_eff_smooth = np.nan_to_num(bg_interp(eval_coords), nan=1.0)
    thresholds_smooth = np.nan_to_num(thresh_interp(eval_coords), nan=1.0)

    n_on = (signal_rate * eff_smooth + bg_rate * bg_eff_smooth) * args.livetime
    n_off = (bg_rate * bg_eff_smooth) * args.livetime / _ALPHA
    sig_smooth = _li_ma_significance(n_on, n_off, _ALPHA)

    t = Table()
    t.meta["alpha"] = _ALPHA
    t.meta["livetime_s"] = args.livetime
    t.meta["source_strength"] = args.source_strength

    t["log10_energy_tev"] = x
    t["log10_energy_tev_min"] = energy_min
    t["log10_energy_tev_max"] = energy_max
    t["zenith_deg"] = y
    t["gamma_ray_rate"] = signal_rate
    t["background_rate"] = bg_rate
    t["gamma_efficiency"] = eff_smooth
    t["background_efficiency"] = bg_eff_smooth
    t["bdt_threshold"] = thresholds_smooth
    t["significance"] = sig_smooth

    t.write(args.output, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
