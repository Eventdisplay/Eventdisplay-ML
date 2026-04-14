"""
Optimize classification cuts for a target source strength.

This script derives a smooth, energy/zenith-dependent cut (gamma efficiency) that
maximizes the Li & Ma significance for a given fraction of the Crab flux, using
rate surfaces stored in a ROOT file.

Input ROOT file must contain:

- TGraph2DErrors gONRate : signal + background rate (1/s)
- TGraph2DErrors gBGRate : background rate (1/s)

Signal rates are expected to be derived from Crab observations, and can be re-weighted to a
different source strength and spectral index using the `source_strength` positional argument
and the `--source-index` parameter.

Usage:

 python src/eventdisplay_ml/scripts/optimize_classification.py \
    tmp_vts/rates_V6_2016_2017_ATM61.root \
    tmp_vts/gammahadron_bdt_ebin*.joblib 1.

"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import uproot
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

_ALPHA = 1.0 / 6.0
# expect Crab spectrum for input signal rate
_CRAB_INDEX = 2.63


def _validate_source_index(source_index):
    """Validate that the source spectral index is in the supported range."""
    if not 2.0 <= float(source_index) <= 5.0:
        raise ValueError(f"Source spectral index must be within [2, 5], got {source_index}.")


def _spectral_reweight_factor(log10_energy_tev, source_index, reference_index=_CRAB_INDEX):
    """
    Compute spectral weights to re-normalize rates from reference to source index.

    Assumes power-law spectra dN/dE ~ E^-index and log10_energy_tev = log10(E / TeV).
    """
    _validate_source_index(source_index)
    delta_index = float(source_index) - float(reference_index)
    return np.power(10.0, -delta_index * np.asarray(log10_energy_tev, dtype=float))


@dataclass
class RateGrid:
    """Rate-grid quantities used throughout the cut optimization."""

    # Fine interpolation grid (flattened energy * zenith mesh)
    log10_energy_tev: np.ndarray
    zenith_deg: np.ndarray
    on_rate: np.ndarray
    background_rate: np.ndarray
    signal_rate: np.ndarray
    energy_axis: np.ndarray
    zenith_axis: np.ndarray
    # Coarse model grid (aligned to ML model energy/zenith bins)
    model_energy_axis: np.ndarray
    model_zenith_axis: np.ndarray
    model_log10_energy: np.ndarray
    model_zenith_deg: np.ndarray
    model_signal_rate: np.ndarray
    model_background_rate: np.ndarray


def _load_multi_bin_roc(joblib_paths):
    """
    Load multiple joblib files and creates 2D interpolators.

    Returns
    -------
    bg_interp : LinearNDInterpolator
        Interpolator for background efficiency as a function of
        (log10_energy, signal_efficiency).
    thresh_interp : LinearNDInterpolator
        Interpolator for BDT threshold as a function of
        (log10_energy, signal_efficiency).
    energy_bins_map : dict
        Mapping from energy bin center to (E_min, E_max) for each energy bin.
    zenith_bins_deg : list[dict]
        Zenith bin definitions from the classification models.
    """
    all_coords = []
    all_bg = []
    all_thresh = []
    energy_bins_map = {}
    zenith_bins_deg = None

    _logger.info(f"Loading {len(joblib_paths)} energy-dependent ROC files...")

    for path in joblib_paths:
        try:
            data = joblib.load(path)
            ebins = data["energy_bins_log10_tev"]
            e_min = ebins["E_min"]
            e_max = ebins["E_max"]
            e_center = (e_min + e_max) / 2.0
            energy_bins_map[e_center] = (e_min, e_max)

            model_zenith_bins = data.get("zenith_bins_deg", [])
            if zenith_bins_deg is None:
                zenith_bins_deg = model_zenith_bins
            elif zenith_bins_deg != model_zenith_bins:
                raise ValueError("Inconsistent zenith binning across ROC files.")

            df = data["models"]["xgboost"]["efficiency"]

            for _, row in df.iterrows():
                all_coords.append([e_center, row["signal_efficiency"]])
                all_bg.append(row["background_efficiency"])
                all_thresh.append(row["threshold"])
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Error loading {path}: {e}") from e

    coords = np.array(all_coords)
    bg_interp = LinearNDInterpolator(coords, np.array(all_bg))
    thresh_interp = LinearNDInterpolator(coords, np.array(all_thresh))

    return bg_interp, thresh_interp, energy_bins_map, zenith_bins_deg or []


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


def _zenith_to_inverse_cosine(zenith_deg):
    """Convert zenith angles in degrees to 1 / cos(ze)."""
    return 1.0 / np.cos(np.deg2rad(zenith_deg))


def _inverse_cosine_to_zenith(inverse_cosine_zenith):
    """Convert 1 / cos(ze) values back to zenith angles in degrees."""
    cosine_zenith = np.clip(1.0 / np.asarray(inverse_cosine_zenith, dtype=float), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine_zenith))


def _build_uniform_axis(min_value, max_value, step):
    """Build an inclusive axis with a requested spacing."""
    if step <= 0:
        raise ValueError(f"Grid spacing must be positive, got {step}.")

    axis = np.arange(min_value, max_value + 0.5 * step, step, dtype=float)
    if axis.size == 0:
        return np.array([min_value, max_value], dtype=float)
    if axis[-1] < max_value and not np.isclose(axis[-1], max_value):
        axis = np.append(axis, max_value)
    else:
        axis[-1] = max_value
    axis[0] = min_value
    return np.unique(axis)


def _build_bin_edges_from_centers(centers):
    """Construct bin edges from monotonically increasing bin centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("At least one bin center is required.")

    if centers.size == 1:
        half_width = 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width], dtype=float)

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first_edge = centers[0] - 0.5 * (centers[1] - centers[0])
    last_edge = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def _reshape_surface(log10_energy, zenith_deg, values):
    """Reshape scattered regular-grid points into a 2D surface."""
    energy_axis = np.sort(np.unique(np.asarray(log10_energy, dtype=float)))
    zenith_axis = np.sort(np.unique(np.asarray(zenith_deg, dtype=float)))

    if energy_axis.size * zenith_axis.size != len(values):
        raise ValueError("Rate surface does not form a complete rectangular grid.")

    surface = np.full((len(zenith_axis), len(energy_axis)), np.nan, dtype=float)
    energy_index = {float(value): idx for idx, value in enumerate(energy_axis)}
    zenith_index = {float(value): idx for idx, value in enumerate(zenith_axis)}

    for energy_value, zenith_value, rate_value in zip(
        log10_energy, zenith_deg, values, strict=True
    ):
        surface[zenith_index[float(zenith_value)], energy_index[float(energy_value)]] = rate_value

    if np.any(~np.isfinite(surface)):
        raise ValueError("Rate surface contains missing grid points.")

    return energy_axis, zenith_axis, surface


def _build_rate_interpolator(log10_energy, zenith_deg, values):
    """Build a regular-grid interpolator for a rate surface."""
    energy_axis, zenith_axis, surface = _reshape_surface(log10_energy, zenith_deg, values)
    inverse_cosine_zenith_axis = _zenith_to_inverse_cosine(zenith_axis)
    interpolator = RegularGridInterpolator(
        (inverse_cosine_zenith_axis, energy_axis),
        surface,
        bounds_error=False,
        fill_value=None,
    )
    return interpolator, energy_axis, zenith_axis


def _mesh_energy_zenith(log10_energy_axis, zenith_deg_axis):
    """Create flattened energy and zenith arrays from 1D axes."""
    energy_mesh, zenith_mesh = np.meshgrid(log10_energy_axis, zenith_deg_axis, indexing="xy")
    return energy_mesh.ravel(), zenith_mesh.ravel()


def _sample_rate_interpolator(interpolator, log10_energy, zenith_deg):
    """Evaluate a rate interpolator on arrays of energy and zenith values."""
    sample_points = np.column_stack([_zenith_to_inverse_cosine(zenith_deg), log10_energy])
    return np.asarray(interpolator(sample_points), dtype=float)


def _build_fine_rate_grid(
    log10_energy,
    zenith_deg,
    on_rate,
    background_rate,
    energy_bin_width,
    inverse_cosine_zenith_bin_width,
):
    """Interpolate ON and background rates onto a finer regular grid."""
    on_interpolator, energy_axis, zenith_axis = _build_rate_interpolator(
        log10_energy, zenith_deg, on_rate
    )
    background_interpolator, _, _ = _build_rate_interpolator(
        log10_energy, zenith_deg, background_rate
    )

    fine_energy_axis = _build_uniform_axis(energy_axis.min(), energy_axis.max(), energy_bin_width)
    fine_inverse_cosine_zenith_axis = _build_uniform_axis(
        _zenith_to_inverse_cosine(zenith_axis).min(),
        _zenith_to_inverse_cosine(zenith_axis).max(),
        inverse_cosine_zenith_bin_width,
    )
    fine_zenith_axis = _inverse_cosine_to_zenith(fine_inverse_cosine_zenith_axis)
    fine_log10_energy, fine_zenith_deg = _mesh_energy_zenith(fine_energy_axis, fine_zenith_axis)

    fine_on_rate = _sample_rate_interpolator(on_interpolator, fine_log10_energy, fine_zenith_deg)
    fine_background_rate = _sample_rate_interpolator(
        background_interpolator, fine_log10_energy, fine_zenith_deg
    )

    return {
        "log10_energy_tev": fine_log10_energy,
        "zenith_deg": fine_zenith_deg,
        "on_rate": fine_on_rate,
        "background_rate": fine_background_rate,
        "energy_axis": fine_energy_axis,
        "zenith_axis": fine_zenith_axis,
        "on_interpolator": on_interpolator,
        "background_interpolator": background_interpolator,
    }


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


def _evaluate_efficiency_interpolator(interpolator, log10_energy, gamma_efficiency, energy_limits):
    """Evaluate the ROC interpolator while clipping to the supported energy range."""
    clipped_energy = np.clip(
        np.asarray(log10_energy, dtype=float), energy_limits[0], energy_limits[1]
    )
    eval_points = np.column_stack([clipped_energy, np.asarray(gamma_efficiency, dtype=float)])
    return np.asarray(interpolator(eval_points), dtype=float)


def _model_zenith_bin_centers(zenith_bins_deg):
    """Return model zenith bin centers in degrees."""
    if not zenith_bins_deg:
        raise ValueError("No zenith binning found in ROC files.")
    return np.array(
        [(zenith_bin["Ze_min"] + zenith_bin["Ze_max"]) / 2.0 for zenith_bin in zenith_bins_deg],
        dtype=float,
    )


def _interpolate_efficiency_surface(
    model_energy_axis,
    model_zenith_axis,
    efficiency_surface,
    target_log10_energy,
    target_zenith_deg,
):
    """Interpolate an efficiency surface in energy and cos(ze)."""
    model_cos_zenith_axis = np.cos(np.deg2rad(np.asarray(model_zenith_axis, dtype=float)))
    order = np.argsort(model_cos_zenith_axis)
    interpolator = RegularGridInterpolator(
        (model_cos_zenith_axis[order], np.asarray(model_energy_axis, dtype=float)),
        np.asarray(efficiency_surface, dtype=float)[order, :],
        bounds_error=False,
        fill_value=None,
    )

    target_cos_zenith = np.cos(np.deg2rad(np.asarray(target_zenith_deg, dtype=float)))
    target_cos_zenith = np.clip(
        target_cos_zenith, model_cos_zenith_axis.min(), model_cos_zenith_axis.max()
    )
    target_log10_energy = np.clip(
        np.asarray(target_log10_energy, dtype=float),
        np.min(model_energy_axis),
        np.max(model_energy_axis),
    )

    return np.asarray(
        interpolator(np.column_stack([target_cos_zenith, target_log10_energy])),
        dtype=float,
    )


def _fill_rate(
    root_path: Path,
    energy_bins_map: dict,
    zenith_bins_deg: list,
    source_strength: float,
    source_index: float,
    energy_bin_width: float,
    inverse_cosine_zenith_bin_width: float,
) -> RateGrid:
    """
    Build a RateGrid from raw ROOT rates and ML model bin definitions.

    Parameters
    ----------
    root_path :
        Path to a ROOT file containing ``gONRate`` and ``gBGRate`` TGraph2D objects.
    energy_bins_map :
        Mapping from energy bin centre to (E_min, E_max) as returned by
        `_load_multi_bin_roc`.
    zenith_bins_deg :
        Zenith bin definitions from the classification models.
    source_strength :
        Fraction of the Crab Nebula flux used to compute the signal rate.
    source_index :
        Power-law source spectral index for dN/dE ~ E^-index.
    energy_bin_width :
        Output bin width in log10(E/TeV) for the fine rate grid.
    inverse_cosine_zenith_bin_width :
        Output bin width in 1/cos(ze) for the fine rate grid.

    Returns
    -------
    RateGrid
        Fully populated rate-grid dataclass.
    """
    x, y, on_rate, bg_rate = _load_rates(root_path)
    raw_grid = _build_fine_rate_grid(
        x, y, on_rate, bg_rate, energy_bin_width, inverse_cosine_zenith_bin_width
    )

    model_energy_axis = np.array(sorted(energy_bins_map.keys()), dtype=float)
    model_zenith_axis = _model_zenith_bin_centers(zenith_bins_deg)
    model_log10_energy, model_zenith_deg = _mesh_energy_zenith(model_energy_axis, model_zenith_axis)

    model_on_rate = _sample_rate_interpolator(
        raw_grid["on_interpolator"], model_log10_energy, model_zenith_deg
    )
    model_background_rate = _sample_rate_interpolator(
        raw_grid["background_interpolator"], model_log10_energy, model_zenith_deg
    )
    fine_reweight = _spectral_reweight_factor(raw_grid["log10_energy_tev"], source_index)
    model_reweight = _spectral_reweight_factor(model_log10_energy, source_index)

    return RateGrid(
        log10_energy_tev=raw_grid["log10_energy_tev"],
        zenith_deg=raw_grid["zenith_deg"],
        on_rate=raw_grid["on_rate"],
        background_rate=raw_grid["background_rate"],
        signal_rate=(
            np.clip(raw_grid["on_rate"] - raw_grid["background_rate"], 0.0, None)
            * source_strength
            * fine_reweight
        ),
        energy_axis=raw_grid["energy_axis"],
        zenith_axis=raw_grid["zenith_axis"],
        model_energy_axis=model_energy_axis,
        model_zenith_axis=model_zenith_axis,
        model_log10_energy=model_log10_energy,
        model_zenith_deg=model_zenith_deg,
        model_signal_rate=(
            np.clip(model_on_rate - model_background_rate, 0.0, None)
            * source_strength
            * model_reweight
        ),
        model_background_rate=model_background_rate,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Optimize classification cuts.")
    parser.add_argument("input_root", help="ROOT file with rate surfaces")
    parser.add_argument("roc_files", nargs="+", help="List of ebin*.joblib files")
    parser.add_argument("source_strength", type=float, help="Fraction of Crab (e.g. 0.1 for 10%%)")
    parser.add_argument(
        "--source-index",
        "--source_index",
        type=float,
        default=_CRAB_INDEX,
        help=(
            "Power-law spectral index for source reweighting with dN/dE ~ E^-index "
            "(allowed range: 2 to 5; default: Crab index)."
        ),
    )
    parser.add_argument("--livetime", type=float, default=3600.0)
    parser.add_argument("--gamma-eff-min", type=float, default=0.01)
    parser.add_argument("--gamma-eff-steps", type=int, default=200)
    parser.add_argument(
        "--energy-bin-width",
        type=float,
        default=0.125,
        help="Output bin width in log10(E/TeV) for the interpolated rate grid.",
    )
    parser.add_argument(
        "--inverse-cosine-zenith-bin-width",
        type=float,
        default=0.05,
        help="Output bin width in 1/cos(ze) for the interpolated rate grid.",
    )
    parser.add_argument("--output", type=str, default="optimized_cuts.ecsv")

    args = parser.parse_args()

    bg_interp, thresh_interp, energy_bins_map, zenith_bins_deg = _load_multi_bin_roc(args.roc_files)
    rate_grid = _fill_rate(
        Path(args.input_root),
        energy_bins_map,
        zenith_bins_deg,
        args.source_strength,
        args.source_index,
        args.energy_bin_width,
        args.inverse_cosine_zenith_bin_width,
    )

    best_eff_model = _optimize_cut_2d(
        rate_grid.model_log10_energy,
        rate_grid.model_signal_rate,
        rate_grid.model_background_rate,
        _ALPHA,
        args.livetime,
        np.linspace(args.gamma_eff_min, 1.0, args.gamma_eff_steps),
        bg_interp,
    )
    best_eff_surface = best_eff_model.reshape(
        len(rate_grid.model_zenith_axis), len(rate_grid.model_energy_axis)
    )

    eff_smooth = np.clip(
        _interpolate_efficiency_surface(
            rate_grid.model_energy_axis,
            rate_grid.model_zenith_axis,
            best_eff_surface,
            rate_grid.log10_energy_tev,
            rate_grid.zenith_deg,
        ),
        args.gamma_eff_min,
        1.0,
    )
    model_energy_limits = (
        np.min(rate_grid.model_energy_axis),
        np.max(rate_grid.model_energy_axis),
    )
    bg_eff_smooth = np.nan_to_num(
        _evaluate_efficiency_interpolator(
            bg_interp,
            rate_grid.log10_energy_tev,
            eff_smooth,
            model_energy_limits,
        ),
        nan=1.0,
    )
    thresholds_smooth = np.nan_to_num(
        _evaluate_efficiency_interpolator(
            thresh_interp,
            rate_grid.log10_energy_tev,
            eff_smooth,
            model_energy_limits,
        ),
        nan=1.0,
    )

    n_on = (
        rate_grid.signal_rate * eff_smooth + rate_grid.background_rate * bg_eff_smooth
    ) * args.livetime
    n_off = (rate_grid.background_rate * bg_eff_smooth) * args.livetime / _ALPHA
    sig_smooth = _li_ma_significance(n_on, n_off, _ALPHA)

    energy_edges = _build_bin_edges_from_centers(rate_grid.energy_axis)
    energy_min = np.repeat(energy_edges[:-1], len(rate_grid.zenith_axis))
    energy_max = np.repeat(energy_edges[1:], len(rate_grid.zenith_axis))

    t = Table()
    t.meta["alpha"] = _ALPHA
    t.meta["livetime_s"] = args.livetime
    t.meta["source_strength"] = args.source_strength
    t.meta["source_index"] = args.source_index
    t.meta["reference_index"] = _CRAB_INDEX
    t.meta["energy_bin_width_log10_tev"] = args.energy_bin_width
    t.meta["inverse_cosine_zenith_bin_width"] = args.inverse_cosine_zenith_bin_width

    t["log10_energy_tev"] = rate_grid.log10_energy_tev
    t["log10_energy_tev_min"] = energy_min
    t["log10_energy_tev_max"] = energy_max
    t["zenith_deg"] = rate_grid.zenith_deg
    t["gamma_ray_rate"] = rate_grid.signal_rate
    t["background_rate"] = rate_grid.background_rate
    t["gamma_efficiency"] = eff_smooth
    t["background_efficiency"] = bg_eff_smooth
    t["bdt_threshold"] = thresholds_smooth
    t["significance"] = sig_smooth

    t.write(args.output, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
