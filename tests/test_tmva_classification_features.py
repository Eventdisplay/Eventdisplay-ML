"""Tests for TMVA-style classification feature selection and training data shaping."""

import awkward as ak
import numpy as np

from eventdisplay_ml import data_processing


def test_load_training_data_tmva_drops_pointing_keeps_ze_bin(monkeypatch):
    """TMVA-style loading should derive ze_bin from elevation and exclude pointing features."""

    class FakeTree:
        def __init__(self, arrays):
            self._arrays = arrays
            self.num_entries = len(arrays)

        def keys(self):
            return list(self._arrays.fields)

        def arrays(self, branches, cut=None, library="ak", decompression_executor=None):
            return ak.Array({name: self._arrays[name] for name in branches})

    class FakeRootFile:
        def __init__(self, tree):
            self._tree = tree

        def __contains__(self, key):
            return key == "data"

        def __getitem__(self, key):
            if key == "data":
                return self._tree
            raise KeyError(key)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    arrays = ak.Array(
        {
            "DispNImages": np.array([2, 3], dtype=np.int32),
            "EChi2S": np.array([1.0, 2.0], dtype=np.float32),
            "EmissionHeight": np.array([10.0, 12.0], dtype=np.float32),
            "EmissionHeightChi2": np.array([0.2, 0.3], dtype=np.float32),
            "MSCW": np.array([0.1, -0.1], dtype=np.float32),
            "MSCL": np.array([0.05, -0.05], dtype=np.float32),
            "SizeSecondMax": np.array([50.0, 60.0], dtype=np.float32),
            "ArrayPointing_Elevation": np.array([70.0, 55.0], dtype=np.float32),
        }
    )

    monkeypatch.setattr(data_processing.utils, "read_input_file_list", lambda _: ["dummy.root"])
    monkeypatch.setattr(
        data_processing,
        "read_telescope_config",
        lambda _: {"max_tel_id": 3},
    )
    monkeypatch.setattr(
        data_processing.uproot,
        "open",
        lambda _: FakeRootFile(FakeTree(arrays)),
    )

    model_configs = {
        "tmva_style": True,
        "zenith_bins_deg": [0.0, 20.0, 40.0, 60.0, 90.0],
        "max_cores": 1,
    }

    df = data_processing.load_training_data(model_configs, "ignored.txt", "classification")

    assert "ze_bin" in df.columns
    assert "ArrayPointing_Elevation" not in df.columns
    assert "ArrayPointing_Azimuth" not in df.columns
