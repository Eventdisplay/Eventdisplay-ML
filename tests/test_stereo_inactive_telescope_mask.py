"""Optional masking of images that were not used by the DispBDT in stereo features."""

import numpy as np
import pandas as pd

from eventdisplay_ml.data_processing import flatten_telescope_data_vectorized

TEL_CONFIG = {
    "max_tel_id": 3,
    "tel_ids": np.arange(4),
    "mirror_area": np.full(4, 100.0),
    "tel_x": np.array([0.0, 100.0, 0.0, -100.0]),
    "tel_y": np.array([100.0, 0.0, -100.0, 0.0]),
}


def _events():
    # Event 0: telescopes 1 and 3 used. Telescopes 0 and 2 carry sentinel/real-looking
    # image values that the DispBDT did not use.
    return pd.DataFrame(
        {
            "DispTelList_T": [np.array([1, 3]), np.array([0, 1, 2, 3])],
            "size": [np.array([0.0, 500.0, 900.0, 300.0]), np.array([400.0, 500.0, 600.0, 700.0])],
            "width": [np.array([0.0, 0.05, 0.09, 0.04]), np.full(4, 0.05)],
            "R_core": [np.array([-99.0, 50.0, 70.0, 60.0]), np.full(4, 50.0)],
            "cosphi": [np.full(4, 0.6), np.full(4, 0.6)],
            "sinphi": [np.full(4, 0.8), np.full(4, 0.8)],
            "Disp_T": [np.array([0.3, 0.2]), np.array([0.1, 0.2, 0.3, 0.4])],
            "Xoff": [0.1, 0.2],
            "Yoff": [0.1, 0.2],
            "Xoff_intersect": [0.1, 0.2],
            "Yoff_intersect": [0.1, 0.2],
            "DispNImages": [2, 4],
            "img2_ang": [30.0, 40.0],
            "EmissionHeight": [10.0, 11.0],
            "Xcore": [10.0, 10.0],
            "Ycore": [10.0, 10.0],
            "ArrayPointing_Elevation": [70.0, 70.0],
            "ArrayPointing_Azimuth": [10.0, 10.0],
        }
    )


def _flatten(analysis_type, mask_unused_images=False):
    return flatten_telescope_data_vectorized(
        _events(),
        4,
        ["size", "width", "R_core", "cosphi", "sinphi", "Disp_T", "tel_active"],
        analysis_type,
        training=False,
        tel_config=TEL_CONFIG,
        preview_rows=0,
        mask_unused_images=mask_unused_images,
    )


def test_stereo_masks_images_not_in_disptellist():
    """With the switch on, unused images become NaN and do not take a sorted slot."""
    df = _flatten("stereo_analysis", mask_unused_images=True)
    # Event 0 uses two telescopes: two valid slots, then NaN, whatever the unused images hold.
    for var in ("size", "width", "R_core"):
        assert df.loc[0, [f"{var}_0", f"{var}_1"]].notna().all()
        assert df.loc[0, [f"{var}_2", f"{var}_3"]].isna().all()
    # Valid slots are sorted by size (log10 applied), so the unused 900-size image does not take a slot.
    assert np.isclose(df.loc[0, "size_0"], np.log10(500.0))
    assert np.isclose(df.loc[0, "size_1"], np.log10(300.0))
    # Event with four used images is unaffected.
    assert np.isclose(df.loc[1, "size_0"], np.log10(700.0))
    assert df.loc[1, [f"size_{i}" for i in range(4)]].notna().all()


def test_stereo_default_keeps_unused_images():
    """Without the switch the historical behavior is kept: unused images stay visible."""
    df = _flatten("stereo_analysis")
    assert df.loc[0, ["size_2", "size_3"]].notna().all()
    assert df.loc[0, ["width_2", "width_3"]].notna().all()
    # The unused 900-size image (telescope 2) sorts first, as before.
    assert np.isclose(df.loc[0, "size_0"], np.log10(900.0))
