"""Tests for optimize_classification grid and interpolation helpers."""

import numpy as np
import pytest

from eventdisplay_ml.scripts.optimize_classification import (
    _CRAB_INDEX,
    _build_fine_rate_grid,
    _interpolate_efficiency_surface,
    _inverse_cosine_to_zenith,
    _spectral_reweight_factor,
)


def test_build_fine_rate_grid_interpolates_rates_on_requested_axes():
    """Interpolate rate surfaces onto a finer energy and 1/cos(ze) grid."""
    energy = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    zenith = np.array([0.0, 0.0, 60.0, 60.0], dtype=float)
    inverse_cosine_zenith = 1.0 / np.cos(np.deg2rad(zenith))

    on_rate = 10.0 + 2.0 * energy + 3.0 * inverse_cosine_zenith
    background_rate = 4.0 + energy + 0.5 * inverse_cosine_zenith

    fine_grid = _build_fine_rate_grid(
        energy,
        zenith,
        on_rate,
        background_rate,
        energy_bin_width=0.5,
        inverse_cosine_zenith_bin_width=0.5,
    )

    expected_energy_axis = np.array([0.0, 0.5, 1.0], dtype=float)
    expected_inverse_cosine_zenith_axis = np.array([1.0, 1.5, 2.0], dtype=float)
    expected_zenith_axis = _inverse_cosine_to_zenith(expected_inverse_cosine_zenith_axis)

    assert np.allclose(fine_grid["energy_axis"], expected_energy_axis)
    assert np.allclose(fine_grid["zenith_axis"], expected_zenith_axis)

    energy_mesh, inverse_cosine_zenith_mesh = np.meshgrid(
        expected_energy_axis,
        expected_inverse_cosine_zenith_axis,
        indexing="xy",
    )
    expected_on_rate = 10.0 + 2.0 * energy_mesh.ravel() + 3.0 * inverse_cosine_zenith_mesh.ravel()
    expected_background_rate = 4.0 + energy_mesh.ravel() + 0.5 * inverse_cosine_zenith_mesh.ravel()

    assert np.allclose(fine_grid["on_rate"], expected_on_rate)
    assert np.allclose(fine_grid["background_rate"], expected_background_rate)


def test_interpolate_efficiency_surface_uses_energy_and_cos_zenith():
    """Interpolate efficiency on energy and cos(ze), clipping at model edges."""
    model_energy_axis = np.array([0.0, 1.0], dtype=float)
    model_zenith_axis = np.array([0.0, 60.0], dtype=float)
    model_cos_zenith_axis = np.cos(np.deg2rad(model_zenith_axis))
    efficiency_surface = np.array(
        [
            0.2 + 0.1 * model_energy_axis + 0.3 * model_cos_zenith_axis[0],
            0.2 + 0.1 * model_energy_axis + 0.3 * model_cos_zenith_axis[1],
        ],
        dtype=float,
    )

    target_energy = np.array([0.5, -1.0], dtype=float)
    target_zenith = np.array([np.rad2deg(np.arccos(0.75)), 80.0], dtype=float)
    interpolated = _interpolate_efficiency_surface(
        model_energy_axis,
        model_zenith_axis,
        efficiency_surface,
        target_energy,
        target_zenith,
    )

    expected = np.array(
        [
            0.2 + 0.1 * 0.5 + 0.3 * 0.75,
            0.2 + 0.1 * 0.0 + 0.3 * model_cos_zenith_axis.min(),
        ],
        dtype=float,
    )

    assert np.allclose(interpolated, expected)


def test_spectral_reweight_factor_is_unity_for_crab_index():
    """Crab-to-Crab reweighting should keep rates unchanged."""
    log10_energy = np.array([-1.0, 0.0, 1.0], dtype=float)
    weights = _spectral_reweight_factor(log10_energy, _CRAB_INDEX)
    assert np.allclose(weights, np.ones_like(log10_energy))


def test_spectral_reweight_factor_reweights_power_law_relative_to_crab():
    """Reweight factor follows E^-(index - crab_index), normalized at 1 TeV."""
    log10_energy = np.array([-1.0, 0.0, 1.0], dtype=float)
    source_index = 3.63
    expected = np.array([10.0, 1.0, 0.1], dtype=float)
    weights = _spectral_reweight_factor(log10_energy, source_index)
    assert np.allclose(weights, expected)


def test_spectral_reweight_factor_rejects_out_of_range_indices():
    """Only source indices in [2, 5] are accepted."""
    with pytest.raises(ValueError, match=r"within \[2, 5\]"):
        _spectral_reweight_factor(np.array([0.0]), 1.9)
    with pytest.raises(ValueError, match=r"within \[2, 5\]"):
        _spectral_reweight_factor(np.array([0.0]), 5.1)
