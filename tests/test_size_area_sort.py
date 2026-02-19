"""Tests for telescope sorting by mirror area then size."""

import numpy as np
import pandas as pd

from eventdisplay_ml import features
from eventdisplay_ml.data_processing import flatten_telescope_data_vectorized


def test_mirror_area_then_size_sorting():
    """Telescopes are ordered by mirror area (desc) then size (desc)."""
    tel_config = {
        "n_tel": 3,
        "tel_ids": np.array([0, 1, 2], dtype=int),
        "mirror_areas": np.array([50.0, 100.0, 50.0], dtype=float),
        "tel_x": np.array([0.0, 1000.0, 2000.0], dtype=float),
        "tel_y": np.array([0.0, 0.0, 0.0], dtype=float),
        "max_tel_id": 2,
        "tel_types": {50.0: [0, 2], 100.0: [1]},
    }

    df = pd.DataFrame(
        {
            "DispTelList_T": [np.array([0, 1, 2], dtype=float)],
            "size": [np.array([10.0, 8.0, 20.0], dtype=float)],
            "R_core": [np.array([100.0, 200.0, 150.0], dtype=float)],
            "Xcore": [0.0],
            "Ycore": [0.0],
            "ArrayPointing_Elevation": [70.0],
            "ArrayPointing_Azimuth": [180.0],
            "Xoff": [0.0],
            "Yoff": [0.0],
            "Xoff_intersect": [0.0],
            "Yoff_intersect": [0.0],
            "DispNImages": [3],
            "Erec": [1.0],
            "ErecS": [1.0],
            "EmissionHeight": [10.0],
            "img2_ang": [45.0],
            # Supply base telescope arrays referenced in feature list; others will default to NaN
            "Disp_T": [np.array([0.0, 0.0, 0.0], dtype=float)],
            "cosphi": [np.array([1.0, 1.0, 1.0], dtype=float)],
            "sinphi": [np.array([0.0, 0.0, 0.0], dtype=float)],
            "loss": [np.array([0.0, 0.0, 0.0], dtype=float)],
            "dist": [np.array([0.0, 0.0, 0.0], dtype=float)],
            "width": [np.array([1.0, 1.0, 1.0], dtype=float)],
            "length": [np.array([1.0, 1.0, 1.0], dtype=float)],
            "asym": [np.array([0.0, 0.0, 0.0], dtype=float)],
            "tgrad_x": [np.array([0.0, 0.0, 0.0], dtype=float)],
        }
    )

    df_flat = flatten_telescope_data_vectorized(
        df,
        n_tel=tel_config["max_tel_id"] + 1,
        features=features.telescope_features("stereo_analysis"),
        analysis_type="stereo_analysis",
        training=True,
        tel_config=tel_config,
    )

    # Expected telescope order: TelID 1 (area 100), TelID 2 (area 50, size 20), TelID 0 (area 50, size 10)
    # Note: sizes are not clipped before sorting.
    expected_areas = [100.0, 50.0, 50.0]
    expected_sizes = [np.log10(8.0), np.log10(20.0), np.log10(10.0)]
    expected_rel_x = [1000.0, 2000.0, 0.0]

    np.testing.assert_allclose(
        df_flat[["mirror_area_0", "mirror_area_1", "mirror_area_2"]].iloc[0], expected_areas
    )
    np.testing.assert_allclose(
        df_flat[["size_0", "size_1", "size_2"]].iloc[0], expected_sizes, rtol=1e-5
    )
    np.testing.assert_allclose(
        df_flat[["tel_rel_x_0", "tel_rel_x_1", "tel_rel_x_2"]].iloc[0], expected_rel_x
    )
