"""
Unit tests for telescope variable indexing and alignment.

Tests verify that Disp_* variables are always indexed by DispTelList_T,
while other variables use mode-dependent indexing:
- VERITAS (R_core present): fixed telescope-ID indexing
- CTAO (ImgSel_list present): variable-length indexing
"""

import numpy as np
import pandas as pd

from eventdisplay_ml.data_processing import (
    _normalize_telescope_variable_to_tel_id_space,
    _to_dense_array,
)


class TestVERITASFixedIndexing:
    """Test VERITAS mode with fixed telescope indexing (R_core present)."""

    def test_disp_alignment_with_disptellist(self):
        """Disp_T indexed by DispTelList_T maps correctly to telescope positions."""
        # Event with 4 telescopes, all active
        disp_tel_list = np.array([0, 1, 2, 3])
        disp_t_raw = np.array([-0.9926, 0.9848, 0.4954, -0.7832])

        # Convert to dense arrays
        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        # Normalize using DispTelList_T
        max_tel_id = 3
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        # Verify each Disp_T value maps to its telescope position
        for i, tel_id in enumerate(disp_tel_list):
            assert np.isclose(disp_normalized[0, tel_id], disp_t_raw[i])

    def test_size_fixed_indexing(self):
        """Size uses fixed telescope-ID indexing in VERITAS mode."""
        size_raw = np.array([2264.55, 2919.08, 3199.95, 4169.71])

        # Convert to dense array
        size_dense = _to_dense_array(pd.Series([size_raw]))

        # Normalize with None (fixed indexing)
        max_tel_id = 3
        size_normalized = _normalize_telescope_variable_to_tel_id_space(
            size_dense, None, max_tel_id, 1
        )

        # Values should be at same positions (identity mapping)
        # Using decimal=4 to account for float32 precision
        np.testing.assert_array_almost_equal(size_normalized[0], size_raw, decimal=4)

    def test_partial_telescope_participation(self):
        """Only subset of telescopes active in event."""
        # Only telescopes 0 and 1 participated
        disp_tel_list = np.array([0, 1])
        disp_t_raw = np.array([-1.8474, -1.3305])

        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        # Need to consider max_tel_id from actual data
        max_tel_id = 3  # Even though only 2 telescopes, array has 4
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        # Check active telescopes
        assert np.isclose(disp_normalized[0, 0], -1.8474)
        assert np.isclose(disp_normalized[0, 1], -1.3305)

        # Inactive telescopes should be NaN
        assert np.isnan(disp_normalized[0, 2])
        assert np.isnan(disp_normalized[0, 3])


class TestCTAOVariableIndexing:
    """Test CTAO mode with variable-length indexing (ImgSel_list present, no R_core)."""

    def test_disp_alignment_contiguous_telescopes(self):
        """Disp_T maps correctly for contiguous telescope IDs."""
        # Telescopes 1, 2, 3 participated
        disp_tel_list = np.array([1, 2, 3])
        disp_t_raw = np.array([0.4193, 0.4511, 0.3853])

        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        max_tel_id = 3
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        # Telescope 0 should be NaN (not in list)
        assert np.isnan(disp_normalized[0, 0])

        # Telescopes 1, 2, 3 should have correct values
        assert np.isclose(disp_normalized[0, 1], 0.4193)
        assert np.isclose(disp_normalized[0, 2], 0.4511)
        assert np.isclose(disp_normalized[0, 3], 0.3853)

    def test_disp_alignment_sparse_telescopes(self):
        """Disp_T maps correctly for non-contiguous telescope IDs."""
        # Telescopes at positions 0, 3, 8
        disp_tel_list = np.array([0, 3, 8])
        disp_t_raw = np.array([0.9182, -0.5703, -0.4690])

        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        max_tel_id = 8
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        # Check active telescopes
        assert np.isclose(disp_normalized[0, 0], 0.9182)
        assert np.isclose(disp_normalized[0, 3], -0.5703)
        assert np.isclose(disp_normalized[0, 8], -0.4690)

        # Check inactive positions are NaN
        assert np.isnan(disp_normalized[0, 1])
        assert np.isnan(disp_normalized[0, 2])
        assert np.isnan(disp_normalized[0, 4])
        assert np.isnan(disp_normalized[0, 5])
        assert np.isnan(disp_normalized[0, 6])
        assert np.isnan(disp_normalized[0, 7])

    def test_disp_alignment_high_telescope_ids(self):
        """Disp_T maps correctly for high telescope IDs."""
        # Telescopes 4, 5, 6, 7, 18
        disp_tel_list = np.array([4, 5, 6, 7, 18])
        disp_t_raw = np.array([1.9557, 1.2722, 2.2075, 3.1824, 2.6806])

        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        max_tel_id = 18
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        # Verify each mapping
        assert np.isclose(disp_normalized[0, 4], 1.9557)
        assert np.isclose(disp_normalized[0, 5], 1.2722)
        assert np.isclose(disp_normalized[0, 6], 2.2075)
        assert np.isclose(disp_normalized[0, 7], 3.1824)
        assert np.isclose(disp_normalized[0, 18], 2.6806)

        # Sample check for NaN positions
        assert np.isnan(disp_normalized[0, 0])
        assert np.isnan(disp_normalized[0, 10])
        assert np.isnan(disp_normalized[0, 15])

    def test_size_imgsel_indexing(self):
        """Size uses ImgSel_list-based variable indexing in CTAO mode."""
        # Size indexed by ImgSel_list positions
        imgsel_list = np.array([0, 4, 5, 6, 18])
        size_raw = np.array([1272.09, 2702.85, 14277.14, 16766.03, 1632.80])

        # Convert to dense arrays
        size_dense = _to_dense_array(pd.Series([size_raw]))
        imgsel_dense = _to_dense_array(pd.Series([imgsel_list]))

        max_tel_id = 18
        size_normalized = _normalize_telescope_variable_to_tel_id_space(
            size_dense, imgsel_dense, max_tel_id, 1
        )

        # Size values should map to their telescope IDs via ImgSel_list
        assert np.isclose(size_normalized[0, 0], 1272.09)
        assert np.isclose(size_normalized[0, 4], 2702.85)
        assert np.isclose(size_normalized[0, 5], 14277.14)
        assert np.isclose(size_normalized[0, 6], 16766.03)
        assert np.isclose(size_normalized[0, 18], 1632.80)


class TestCrossModeConsistency:
    """Test that indexing is consistent across modes."""

    def test_disp_always_uses_disptellist_veritas(self):
        """In VERITAS mode, Disp_T uses DispTelList_T not R_core indexing."""
        # This is the critical test: even when R_core present,
        # Disp_T must use DispTelList_T

        # Simulate VERITAS R_core mode (fixed indexing)
        size_raw = np.array([728.16, 846.64, 0.0, 318.92])  # All 4 positions

        # But Disp_T is sparse (only 3 telescopes)
        disp_tel_list = np.array([0, 1, 3])  # Note: no telescope 2
        disp_t_raw = np.array([1.435, 1.3659, 1.8052])

        # Size uses fixed indexing (None)
        size_dense = _to_dense_array(pd.Series([size_raw]))
        size_normalized = _normalize_telescope_variable_to_tel_id_space(size_dense, None, 3, 1)

        # Disp uses DispTelList_T indexing
        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, 3, 1
        )

        # Size has all 4 values at fixed positions
        assert np.isclose(size_normalized[0, 0], 728.16)
        assert np.isclose(size_normalized[0, 1], 846.64)
        assert np.isclose(size_normalized[0, 2], 0.0)
        assert np.isclose(size_normalized[0, 3], 318.92)

        # Disp has values at positions 0, 1, 3 (tel 2 is NaN)
        assert np.isclose(disp_normalized[0, 0], 1.435)
        assert np.isclose(disp_normalized[0, 1], 1.3659)
        assert np.isnan(disp_normalized[0, 2])  # Not in DispTelList_T
        assert np.isclose(disp_normalized[0, 3], 1.8052)

    def test_ctao_disp_and_size_both_variable_indexed(self):
        """In CTAO mode, both use variable indexing (should match)."""
        # When ImgSel_list == DispTelList_T, both should produce same structure
        telescope_list = np.array([9, 10])

        size_raw = np.array([1114.23, 64983.75])
        disp_raw = np.array([1.6124, 0.8031])

        size_dense = _to_dense_array(pd.Series([size_raw]))
        disp_dense = _to_dense_array(pd.Series([disp_raw]))
        index_dense = _to_dense_array(pd.Series([telescope_list]))

        max_tel_id = 10
        size_normalized = _normalize_telescope_variable_to_tel_id_space(
            size_dense, index_dense, max_tel_id, 1
        )
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, index_dense, max_tel_id, 1
        )

        # Both should have NaN at same positions
        size_valid = ~np.isnan(size_normalized[0])
        disp_valid = ~np.isnan(disp_normalized[0])
        np.testing.assert_array_equal(size_valid, disp_valid)

        # Both should have values at positions 9 and 10
        assert np.isclose(size_normalized[0, 9], 1114.23)
        assert np.isclose(size_normalized[0, 10], 64983.75)
        assert np.isclose(disp_normalized[0, 9], 1.6124)
        assert np.isclose(disp_normalized[0, 10], 0.8031)


class TestMultipleEvents:
    """Test processing multiple events in batch."""

    def test_batch_normalization(self):
        """Multiple events normalized correctly in single call."""
        # 3 events with varying telescope participation
        disp_tel_list_1 = np.array([0, 1, 2, 3])
        disp_tel_list_2 = np.array([0, 1])
        disp_tel_list_3 = np.array([1, 2, 3])

        disp_raw_1 = np.array([-0.99, 0.98, 0.50, -0.78])
        disp_raw_2 = np.array([-1.85, -1.33])
        disp_raw_3 = np.array([0.42, 0.45, 0.39])

        # Create batch arrays
        disp_batch = _to_dense_array(pd.Series([disp_raw_1, disp_raw_2, disp_raw_3]))
        disptel_batch = _to_dense_array(
            pd.Series([disp_tel_list_1, disp_tel_list_2, disp_tel_list_3])
        )

        max_tel_id = 3
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_batch, disptel_batch, max_tel_id, 3
        )

        # Event 0: all 4 telescopes
        assert np.isclose(disp_normalized[0, 0], -0.99)
        assert np.isclose(disp_normalized[0, 1], 0.98)
        assert np.isclose(disp_normalized[0, 2], 0.50)
        assert np.isclose(disp_normalized[0, 3], -0.78)

        # Event 1: only telescopes 0, 1
        assert np.isclose(disp_normalized[1, 0], -1.85)
        assert np.isclose(disp_normalized[1, 1], -1.33)
        assert np.isnan(disp_normalized[1, 2])
        assert np.isnan(disp_normalized[1, 3])

        # Event 2: only telescopes 1, 2, 3
        assert np.isnan(disp_normalized[2, 0])
        assert np.isclose(disp_normalized[2, 1], 0.42)
        assert np.isclose(disp_normalized[2, 2], 0.45)
        assert np.isclose(disp_normalized[2, 3], 0.39)
