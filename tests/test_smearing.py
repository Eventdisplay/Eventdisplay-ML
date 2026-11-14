"""Tests for image perturbation and configuration."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml.smearing import load_smearing, smear_images


def test_smearing_is_stable_across_chunks_and_preserves_missing_and_inactive():
    """Identical event IDs give identical perturbations in any chunk."""
    events = pd.DataFrame({"runNumber": [1, 1, 2], "eventNumber": [4, 5, 6]})
    values = np.array([[10.0, np.nan], [20.0, 30.0], [40.0, 50.0]])
    active = np.array([[True, True], [True, False], [True, True]])
    config = {"seed": 42, "fraction": 1.0, "size": 0.1}
    result = smear_images(values, "size", events, active, config)
    pieces = np.concatenate(
        [
            smear_images(values[:1], "size", events[:1], active[:1], config),
            smear_images(values[1:], "size", events[1:], active[1:], config),
        ]
    )
    np.testing.assert_equal(result, pieces)
    assert np.isnan(result[0, 1])
    assert result[1, 1] == 30.0
    assert result[0, 0] != 10.0


def test_yaml_validation_and_zero_fraction(tmp_path):
    """Invalid strengths fail and zero fraction leaves values untouched."""
    path = tmp_path / "smear.yaml"
    path.write_text("seed: 3\nfraction: 0\nwidth: 0.01\n")
    config = load_smearing(path)
    result = smear_images(
        np.array([[0.1]]),
        "width",
        pd.DataFrame({"runNumber": [1], "eventNumber": [2]}),
        np.ones((1, 1), bool),
        config,
    )
    assert result[0, 0] == 0.1
    path.write_text("width: -1\n")
    with pytest.raises(ValueError, match="width"):
        load_smearing(path)
