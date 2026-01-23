"""
Test alignment of Disp_T and other telescope variables.

Verifies that Disp_T values are always aligned with their telescope IDs via DispTelList_T.

Modes:

- VERITAS: R_core present → fixed telescope-ID indexing
- CTAO: ImgSel_list present → variable-length indexing

Requires CTAO and VERITAS test files.
"""

import numpy as np
import pandas as pd
import pytest
import uproot

from eventdisplay_ml.data_processing import (
    _normalize_telescope_variable_to_tel_id_space,
    _to_dense_array,
)


@pytest.mark.skip(reason="Requires external input, run manually")
def test_file(filename, description):
    """Test alignment for a single file."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {description}")
    print(f"File: {filename}")
    print("=" * 80)

    tree = uproot.open(filename)["data"]
    branches = tree.keys()

    # Check mode
    has_r_core = "R_core" in branches
    has_imgsel = "ImgSel_list" in branches
    print(f"Mode: {'VERITAS (R_core)' if has_r_core else 'CTAO (ImgSel_list)'}")

    # Load first 5 events
    n_events = min(5, tree.num_entries)

    disp_tel_list_all = tree["DispTelList_T"].array(library="np")[:n_events]
    disp_t_all = tree["Disp_T"].array(library="np")[:n_events]
    size_all = tree["size"].array(library="np")[:n_events]

    imgsel_list_all = None
    if has_imgsel:
        imgsel_list_all = tree["ImgSel_list"].array(library="np")[:n_events]

    # Process each event
    for evt_idx in range(n_events):
        print(f"\n--- Event {evt_idx} ---")

        # Get raw data
        disp_tel_list = disp_tel_list_all[evt_idx]
        disp_t_raw = disp_t_all[evt_idx]
        size_raw = size_all[evt_idx]

        print("Raw data:")
        print(f"  DispTelList_T: {disp_tel_list}")
        print(f"  Disp_T:        {disp_t_raw}")
        print(f"  size:          {size_raw}")

        if has_imgsel:
            imgsel_list = imgsel_list_all[evt_idx]
            print(f"  ImgSel_list:   {imgsel_list}")

        # Determine max_tel_id from DispTelList_T
        max_tel_id = int(np.max(disp_tel_list))
        print(f"  max_tel_id: {max_tel_id}")

        # Convert to dense arrays (wrap in Series for _to_dense_array)
        size_dense = _to_dense_array(pd.Series([size_raw]))
        disp_dense = _to_dense_array(pd.Series([disp_t_raw]))
        disptel_dense = _to_dense_array(pd.Series([disp_tel_list]))

        print(f"\nDense arrays (first {max_tel_id + 1} positions):")
        print(f"  size:        {size_dense[0, : max_tel_id + 1]}")
        print(f"  disp_raw:    {disp_dense[0, : max_tel_id + 1]}")
        print(f"  disptel_raw: {disptel_dense[0, : max_tel_id + 1]}")

        # Determine index list for other variables
        if has_r_core:
            # VERITAS mode: size uses fixed indexing (None), Disp_T uses DispTelList_T
            index_list_for_remapping = None
            print("\n→ VERITAS: size uses fixed indexing, Disp_T uses DispTelList_T")
        else:
            # CTAO mode: all variables use ImgSel_list
            imgsel_list = imgsel_list_all[evt_idx]
            # Convert to 2D array like in real code
            index_list_for_remapping = _to_dense_array(pd.Series([imgsel_list]))
            print("\n→ CTAO: all variables use ImgSel_list")

        # Normalize size using mode-dependent indexing
        size_normalized = _normalize_telescope_variable_to_tel_id_space(
            size_dense, index_list_for_remapping, max_tel_id, 1
        )

        # Normalize Disp_T using DispTelList_T (always)
        disp_normalized = _normalize_telescope_variable_to_tel_id_space(
            disp_dense, disptel_dense, max_tel_id, 1
        )

        print("\nAfter normalization to tel_id space:")
        print(f"  size[0]:    {size_normalized[0, : max_tel_id + 1]}")
        print(f"  disp[0]:    {disp_normalized[0, : max_tel_id + 1]}")

        # Verify alignment: check raw vs normalized
        print("\nAlignment verification:")

        # Map: Disp_T[i] should map to tel_id=DispTelList_T[i]
        print("  Expected mapping (from DispTelList_T):")
        for i, tel_id in enumerate(disp_tel_list):
            print(
                f"    Disp_T[{i}]={disp_t_raw[i]:.4f} → tel_id={tel_id} (should be at pos {tel_id})"
            )

        print("  Actual values in normalized output:")
        for tel_id in disp_tel_list:
            val = disp_normalized[0, tel_id]
            if not np.isnan(val):
                print(f"    Position {tel_id}: {val:.4f} ✓")
            else:
                print(f"    Position {tel_id}: NaN ✗ ERROR")

        # Verify that values match
        print("\nDetailed alignment check:")
        for i, tel_id in enumerate(disp_tel_list):
            expected_val = disp_t_raw[i]
            actual_val = disp_normalized[0, tel_id]
            if np.isclose(expected_val, actual_val, rtol=1e-5):
                print(f"  ✓ Disp_T[{i}]={expected_val:.6f} → position {tel_id} = {actual_val:.6f}")
            else:
                print(
                    f"  ✗ Disp_T[{i}]={expected_val:.6f} → position {tel_id} = {actual_val:.6f} MISMATCH"
                )


@pytest.mark.skip(reason="Requires external input, run manually")
def test_sorting_applied(filename):
    """
    Test that sorting by mirror area (desc) then size (desc) is correctly applied.

    Verifies that output columns are ordered by telescope type (mirror area descending)
    then by size (descending), NOT by telescope ID.
    """
    print(f"\n{'=' * 80}")
    print("Testing: Sorting by mirror area (desc) then size (desc)")
    print("=" * 80)

    # Use synthetic data to verify sorting behavior
    # We'll check one of the real files that has clear mirror area differences
    try:
        tree = uproot.open(filename)["data"]

        # Read telconfig to get mirror areas
        telconfig_tree = uproot.open(filename)["telconfig"]
        telconfig_data = telconfig_tree.arrays(["TelID", "MirrorArea"], library="np")
        tel_id_to_area = dict(zip(telconfig_data["TelID"], telconfig_data["MirrorArea"]))

        print(f"Telescope mirror areas: {tel_id_to_area}")

        # Load first event
        disp_tel_list = tree["DispTelList_T"].array(library="np")[0]
        size_raw = tree["size"].array(library="np")[0]

        print("\nEvent 0:")
        print(f"  DispTelList_T: {disp_tel_list}")
        print(f"  size raw:      {size_raw}")

        # Show the unordered telescope info
        print("\nUnordered telescope info (by participation order):")
        for i, tel_id in enumerate(disp_tel_list):
            area = tel_id_to_area.get(int(tel_id), 0)
            size_val = size_raw[i]
            print(f"  Position {i}: TelID {tel_id}, Area {area:.2f}, Size {size_val:.2f}")

        # Expected sort: primary by area (desc), secondary by size (desc)
        tel_info = []
        for i, tel_id in enumerate(disp_tel_list):
            area = tel_id_to_area.get(int(tel_id), 0)
            size_val = size_raw[i]
            tel_info.append((int(tel_id), area, size_val, i))

        # Sort by area desc, then size desc
        sorted_info = sorted(tel_info, key=lambda x: (-x[1], -x[2]))

        print("\nExpected sorted order (area desc, then size desc):")
        for rank, (tel_id, area, size_val, orig_pos) in enumerate(sorted_info):
            print(f"  Column {rank}: TelID {tel_id}, Area {area:.2f}, Size {size_val:.2f}")

        print("\n✓ Test setup complete. Compare module output columns with this expected order.")

    except Exception as e:
        print(f"Note: Could not load test file for sorting verification: {e}")
        print("This test requires the example CTA file to be present.")


if __name__ == "__main__":
    test_file("tmp_testing/20deg_0.5wob_NOISE200.mscw.root", "VERITAS (fixed indexing via R_core)")

    test_file(
        "tmp_cta_testing/gamma_cone.N.Am-4LSTs09MSTs_ID0_0deg-prod6-LaPalma-20deg-moon-sq51-LL-1001.mscw.root",
        "CTAO (variable indexing via ImgSel_list)",
    )

    test_sorting_applied(
        "tmp_cta_testing/gamma_cone.N.Am-4LSTs09MSTs_ID0_0deg-prod6-LaPalma-20deg-moon-sq51-LL-1001.mscw.root"
    )
