"""Tests for ImgSel_list-mode telescope sorting and alignment."""

import numpy as np
import pandas as pd

from eventdisplay_ml import features
from eventdisplay_ml.data_processing import flatten_telescope_data_vectorized


def test_imgsel_sorting_and_alignment():
    """
    Verify ImgSel_list-mode event is normalized to telescope-ID space.

    Test sorting by mirror area (desc) then size (desc) is correctly applied.

    Scenario: 4 telescopes with equal mirror area; 2 active telescopes via ImgSel_list.
    Expect sorting purely by size (desc) and Disp_T aligned accordingly.
    """
    tel_config = {
        "n_tel": 4,
        "tel_ids": np.array([0, 1, 2, 3], dtype=int),
        "mirror_areas": np.array([100.0, 100.0, 100.0, 100.0], dtype=float),
        "tel_x": np.array([0.0, 1000.0, 2000.0, 3000.0], dtype=float),
        "tel_y": np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        "max_tel_id": 3,
        "tel_types": {100.0: [0, 1, 2, 3]},
    }

    # ImgSel_list-mode (no R_core): active telescopes are 1 and 3
    df = pd.DataFrame(
        {
            "ImgSel_list": [np.array([1, 3], dtype=np.uint32)],
            "DispTelList_T": [np.array([1, 3], dtype=np.uint32)],
            # size indexed by ImgSel_list positions (tel 1, tel 3)
            "size": [np.array([3725.51, 2640.01], dtype=float)],
            # Disp_T indexed by DispTelList_T positions (tel 1, tel 3)
            "Disp_T": [np.array([1.56872, 1.50883], dtype=float)],
            "Xcore": [0.0],
            "Ycore": [0.0],
            "ArrayPointing_Elevation": [70.0],
            "ArrayPointing_Azimuth": [180.0],
            "Xoff": [0.0],
            "Yoff": [0.0],
            "Xoff_intersect": [0.0],
            "Yoff_intersect": [0.0],
            "DispNImages": [2],
            "Erec": [1.0],
            "ErecS": [1.0],
            "EmissionHeight": [10.0],
            "img2_ang": [45.0],
            "cosphi": [np.array([0.0, 0.0], dtype=float)],
            "sinphi": [np.array([1.0, 1.0], dtype=float)],
            "loss": [np.array([0.0, 0.0], dtype=float)],
            "dist": [np.array([0.0, 0.0], dtype=float)],
            "width": [np.array([1.0, 1.0], dtype=float)],
            "length": [np.array([1.0, 1.0], dtype=float)],
            "asym": [np.array([0.0, 0.0], dtype=float)],
            "tgrad_x": [np.array([0.0, 0.0], dtype=float)],
        }
    )

    df_flat = flatten_telescope_data_vectorized(
        df,
        n_tel=tel_config["max_tel_id"] + 1,
        features=features.telescope_features("stereo_analysis"),
        analysis_type="stereo_analysis",
        training=True,
        tel_config=tel_config,
        observatory="ctao-north",
    )

    # Expected order by size desc (mirror areas equal): tel 1 (3725.51) then tel 3 (2640.01)
    expected_size_log = [np.log10(3725.51), np.log10(2640.01)]  # Descending order
    # Disp_T aligned to sorted order (tel 1 first, then tel 3)
    expected_disp = [1.56872, 1.50883]

    # Assert mirror_area columns exist and equal 100 for sorted positions
    assert df_flat["mirror_area_0"].iloc[0] == 100.0
    assert df_flat["mirror_area_1"].iloc[0] == 100.0

    # Assert size columns (log10 after clipping/log10 in flatten) match expected
    np.testing.assert_allclose(
        [df_flat["size_0"].iloc[0], df_flat["size_1"].iloc[0]], expected_size_log, rtol=1e-5
    )
    # Remaining positions should be NaN (no active telescopes beyond the two)
    assert np.isnan(df_flat["size_2"].iloc[0])
    assert np.isnan(df_flat["size_3"].iloc[0])

    # Assert Disp_T aligned to sorted order
    np.testing.assert_allclose(
        [df_flat["Disp_T_0"].iloc[0], df_flat["Disp_T_1"].iloc[0]], expected_disp, rtol=1e-5
    )
    assert np.isnan(df_flat["Disp_T_2"].iloc[0])
    assert np.isnan(df_flat["Disp_T_3"].iloc[0])
