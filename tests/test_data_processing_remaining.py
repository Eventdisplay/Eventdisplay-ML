"""Tests for remaining data_processing helper branches."""

from unittest.mock import MagicMock

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from conftest import create_base_df

from eventdisplay_ml import data_processing


@pytest.fixture
def tel_config():
    return {
        "n_tel": 2,
        "tel_ids": np.array([0, 1]),
        "mirror_area": np.array([100.0, 50.0]),
        "mirror_areas": np.array([100.0, 50.0]),
        "tel_x": np.array([0.0, 100.0]),
        "tel_y": np.array([0.0, 0.0]),
        "max_tel_id": 1,
        "tel_types": {100.0: [0], 50.0: [1]},
    }


@pytest.fixture
def awkward_records():
    return ak.Array(
        [
            {"DispTelList_T": [0, 1], "old": [1.0, 2.0]},
            {"DispTelList_T": [0], "old": [3.0]},
        ]
    )


def test_read_telescope_config_groups_telescopes():
    telconfig_tree = MagicMock()
    telconfig_tree.arrays.return_value = {
        "NTel": np.array([2]),
        "TelID": np.array([0, 1]),
        "MirrorArea": np.array([100.0, 50.0]),
        "TelX": np.array([0.0, 100.0]),
        "TelY": np.array([0.0, 0.0]),
    }
    root_file = {"telconfig": telconfig_tree}

    result = data_processing.read_telescope_config(root_file)

    assert result["n_tel"] == 2
    assert result["max_tel_id"] == 1
    assert result["tel_types"][100.0] == [0]


def test_resolve_branch_aliases_handles_fallback_and_missing_optional(caplog):
    tree = MagicMock()
    tree.keys.return_value = {"R", "size", "Erec"}
    resolved, rename = data_processing._resolve_branch_aliases(
        tree,
        ["R_core", "mirror_area", "size", "fpointing_dx", "Erec", "SizeSecondMax"],
    )
    assert resolved == ["R", "size", "Erec"]
    assert rename == {"R": "R_core"}

    tree.keys.return_value = {"size"}
    missing, _ = data_processing._resolve_branch_aliases(tree, ["R_core", "size"])
    assert missing == ["size"]
    assert "fallback 'R' not found" in caplog.text


def test_ensure_fpointing_fields_and_rename_fields(awkward_records):
    ensured = data_processing._ensure_fpointing_fields(awkward_records)
    renamed = data_processing._rename_fields(ensured, {"old": "new"})
    assert "fpointing_dx" in renamed.fields
    assert "fpointing_dy" in renamed.fields
    assert ak.to_list(renamed["new"])[0] == [1.0, 2.0]
    assert "old" not in renamed.fields


def test_to_dense_array_and_numpy_helpers_cover_remaining_branches():
    dense = data_processing._to_dense_array(ak.Array([[1.0, 2.0], [3.0]]))
    assert dense.shape == (2, 2)
    assert np.isnan(dense[1, 1])

    with pytest.raises(ValueError, match="convertible"):
        data_processing._to_dense_array(42)

    fallback = data_processing._to_numpy_1d([1, 2], np.float32)
    awkward_array = data_processing._to_numpy_1d(ak.Array([1, 2]), np.float32)
    assert fallback.dtype == np.float32
    assert awkward_array.tolist() == [1.0, 2.0]


def test_has_field_supports_awkward_and_fallback():
    assert data_processing._has_field(ak.Array([{"field": 1}]), "field") is True

    class BrokenContains:
        def __contains__(self, _):
            raise TypeError("bad")

    assert data_processing._has_field(BrokenContains(), "field") is False


def test_flatten_feature_data_drops_size_columns_for_classification(tel_config):
    df = create_base_df(n_rows=2, n_tel=2)
    df["ImgSel_list"] = [np.array([0, 1]), np.array([0, 1])]
    df["MSCW"] = [1.0, 2.0]
    df["MSCL"] = [0.1, 0.2]
    df["EChi2S"] = [4.0, 5.0]
    df["EmissionHeight"] = [100.0, 120.0]
    df["EmissionHeightChi2"] = [6.0, 7.0]
    df["Xcore"] = [10.0, 20.0]
    df["Ycore"] = [0.0, 5.0]
    df["DispAbsSumWeigth"] = [1.0, 2.0]
    df["ArrayPointing_Azimuth"] = [0.0, 10.0]
    df["ArrayPointing_Elevation"] = [70.0, 65.0]
    df["ze_bin"] = [0, 1]

    result = data_processing.flatten_feature_data(
        df,
        2,
        "classification",
        training=False,
        tel_config=tel_config,
        observatory="ctao",
        preview_rows=0,
    )

    assert "size_0" not in result.columns
    assert "cosphi_0" in result.columns
    assert "ze_bin" in result.columns


def test_calculate_array_footprint_returns_negative_one_on_value_error(tel_config, monkeypatch):
    monkeypatch.setattr(
        data_processing, "ConvexHull", lambda *_: (_ for _ in ()).throw(ValueError("bad"))
    )
    footprints = data_processing._calculate_array_footprint(tel_config, np.array([[0.0, 1.0, 1.0]]))
    assert footprints[0] == pytest.approx(-1.0)


def test_extra_columns_classification_handles_optional_columns():
    df = pd.DataFrame(
        {
            "MSCW": [1.0],
            "MSCL": [2.0],
            "EChi2S": [10.0],
            "EmissionHeight": [100.0],
            "EmissionHeightChi2": [5.0],
            "SizeSecondMax": [1000.0],
            "Xcore": [3.0],
            "Ycore": [4.0],
            "DispAbsSumWeigth": [7.0],
            "ze_bin": [2],
        }
    )
    result = data_processing.extra_columns(df, "classification", False, df.index)
    assert result.loc[0, "Core_Distance"] == pytest.approx(5.0)
    assert result.loc[0, "DispAbsSumWeigth"] == pytest.approx(5.0)
    assert result.loc[0, "ze_bin"] == pytest.approx(2.0)
    assert result.loc[0, "SizeSecondMax"] == pytest.approx(3.0)


def test_energy_interpolation_bins_handles_edge_cases():
    empty = pd.DataFrame({"Erec": []})
    invalid = pd.DataFrame({"Erec": [0.0, -1.0]})
    single = pd.DataFrame({"Erec": [1.0, 10.0]})
    bins = [{"E_min": -1.0, "E_max": 1.0}]
    nan_bins = [None]

    assert np.all(data_processing.energy_interpolation_bins(empty, bins)[0] == -1)
    assert np.all(data_processing.energy_interpolation_bins(single, nan_bins)[0] == -1)
    assert np.all(data_processing.energy_interpolation_bins(invalid, bins)[0] == -1)
    lo, hi, alpha = data_processing.energy_interpolation_bins(single, bins)
    assert lo.tolist() == [0, 0]
    assert hi.tolist() == [0, 0]
    assert alpha.tolist() == pytest.approx([0.0, 0.0])


def test_print_variable_statistics_handles_empty_and_non_empty_columns(caplog, capsys):
    df = pd.DataFrame({"filled": [1.0, 3.0], "empty": [np.nan, np.nan]})
    with caplog.at_level("INFO"):
        data_processing.print_variable_statistics(df)
    assert "filled" in caplog.text
    assert "empty: No data" in capsys.readouterr().out


def test_load_training_data_tmva_style_classification(monkeypatch, tel_config):
    arrays = ak.Array(
        [
            {"ArrayPointing_Elevation": 70.0, "ArrayPointing_Azimuth": 30.0, "MSCW": 1.0},
            {"ArrayPointing_Elevation": 40.0, "ArrayPointing_Azimuth": 40.0, "MSCW": 2.0},
        ]
    )
    tree = MagicMock()
    tree.num_entries = 2
    tree.arrays.return_value = arrays
    root_file = MagicMock()
    root_file.__enter__.return_value = {"data": tree, "telconfig": MagicMock()}
    root_file.__exit__.return_value = False
    monkeypatch.setattr(data_processing.utils, "read_input_file_list", lambda _: ["file.root"])
    monkeypatch.setattr(data_processing.uproot, "open", lambda _: root_file)
    monkeypatch.setattr(data_processing, "read_telescope_config", lambda _: tel_config)
    monkeypatch.setattr(
        data_processing, "_resolve_branch_aliases", lambda tree, branches: (branches, {})
    )
    monkeypatch.setattr(data_processing, "_ensure_fpointing_fields", lambda arr: arr)
    monkeypatch.setattr(
        data_processing.features_module,
        "features_tmva_style",
        lambda *_args, **_kwargs: ["MSCW", "ze_bin", "ArrayPointing_Azimuth"],
    )
    monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *_: None)

    result = data_processing.load_training_data(
        {
            "tmva_style": True,
            "zenith_bins_deg": [{"Ze_min": 0, "Ze_max": 30}, {"Ze_min": 30, "Ze_max": 60}],
            "max_cores": 1,
        },
        "inputs.txt",
        "classification",
    )

    assert "ArrayPointing_Azimuth" not in result.columns
    assert "ArrayPointing_Elevation" not in result.columns
    assert result["ze_bin"].tolist() == [0, 1]


def test_load_training_data_stereo_adds_residuals(monkeypatch, tel_config):
    arrays = ak.Array(
        [
            {"MCxoff": 1.2, "MCyoff": 2.1, "MCe0": 10.0},
            {"MCxoff": 1.4, "MCyoff": 2.4, "MCe0": 100.0},
        ]
    )
    tree = MagicMock()
    tree.num_entries = 2
    tree.arrays.return_value = arrays
    root_data = {"data": tree, "telconfig": MagicMock()}
    root_file = MagicMock()
    root_file.__enter__.return_value = root_data
    root_file.__exit__.return_value = False
    monkeypatch.setattr(data_processing.utils, "read_input_file_list", lambda _: ["file.root"])
    monkeypatch.setattr(data_processing.uproot, "open", lambda _: root_file)
    monkeypatch.setattr(data_processing, "read_telescope_config", lambda _: tel_config)
    monkeypatch.setattr(
        data_processing, "_resolve_branch_aliases", lambda tree, branches: (branches, {})
    )
    monkeypatch.setattr(data_processing, "_ensure_fpointing_fields", lambda arr: arr)
    monkeypatch.setattr(
        data_processing,
        "flatten_telescope_data_vectorized",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Xoff_weighted_bdt": [1.0, 1.0],
                "Yoff_weighted_bdt": [2.0, 2.0],
                "ErecS": [5.0, 10.0],
            }
        ),
    )
    monkeypatch.setattr(data_processing, "print_variable_statistics", lambda *_: None)

    result = data_processing.load_training_data({"max_cores": 1}, "inputs.txt", "stereo_analysis")

    assert result["Xoff_residual"].tolist() == pytest.approx([0.2, 0.4])
    assert result["Yoff_residual"].tolist() == pytest.approx([0.1, 0.4])
    assert result["E_residual"].tolist() == pytest.approx([np.log10(10.0) - np.log10(5.0), 1.0])
