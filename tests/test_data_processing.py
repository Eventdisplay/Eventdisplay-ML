"""Tests for data_processing."""

import numpy as np
import pandas as pd

from eventdisplay_ml.data_processing import energy_interpolation_bins, zenith_in_bins


def test_zenith_in_bins_numeric_edges_clips_and_handles_boundaries():
    """Numeric bin edges should clip out-of-range values and place edge values consistently."""
    zenith_angles = np.array([-5.0, 0.0, 9.9, 10.0, 19.9, 20.0, 42.0], dtype=float)
    bins = [0.0, 10.0, 20.0, 30.0]

    result = zenith_in_bins(zenith_angles, bins)

    np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int32))
    assert result.dtype == np.int32


def test_zenith_in_bins_dict_bins_matches_numeric_definition():
    """Dict-style zenith bins should map to the same indices as explicit numeric edges."""
    zenith_angles = np.array([-5.0, 0.0, 9.9, 10.0, 19.9, 20.0, 42.0], dtype=float)
    dict_bins = [
        {"Ze_min": 0.0, "Ze_max": 10.0},
        {"Ze_min": 10.0, "Ze_max": 20.0},
        {"Ze_min": 20.0, "Ze_max": 30.0},
    ]

    result = zenith_in_bins(zenith_angles, dict_bins)

    np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int32))
    assert result.dtype == np.int32


def test_energy_interpolation_bins_interpolates_and_clamps_with_invalid_events():
    """Interpolation bins should handle interior, edge, and invalid energies robustly."""
    df_chunk = pd.DataFrame({"Erec": [0.0, 0.1, 1.0, 10.0, 100.0]})
    bins = [
        {"E_min": -1.5, "E_max": -0.5},
        None,
        {"E_min": 0.5, "E_max": 1.5},
    ]

    e_bin_lo, e_bin_hi, e_alpha = energy_interpolation_bins(df_chunk, bins)

    np.testing.assert_array_equal(e_bin_lo, np.array([-1, 0, 0, 0, 2], dtype=np.int32))
    np.testing.assert_array_equal(e_bin_hi, np.array([-1, 0, 2, 2, 2], dtype=np.int32))
    np.testing.assert_allclose(e_alpha, np.array([0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32))
