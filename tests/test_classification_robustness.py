"""Focused tests for the code-only classification hardening contract."""

import numpy as np
import pandas as pd
import pytest

from eventdisplay_ml import features, models
from eventdisplay_ml.evaluate import (
    _efficiency_dataframe,
    classification_thresholds_from_signal,
)


def test_robust_profile_excludes_routing_and_activity_columns():
    columns = [
        "MSCW",
        "MSCL",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "Core_Distance",
        "size_0",
        "width_0",
        "length_0",
        "tel_active_0",
        "ze_bin",
        "Erec",
        "__source_file",
    ]
    assert features.classification_feature_columns(columns) == [
        "MSCW",
        "MSCL",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "Core_Distance",
        "size_0",
        "width_0",
        "length_0",
        "ze_bin",
    ]


def test_extended_profile_retains_zenith_but_not_provenance():
    columns = [
        "MSCW",
        "ze_bin",
        "Erec",
        "tel_active_0",
        "mirror_area_0",
        "__source_file",
    ]
    assert features.classification_feature_columns(columns, profile="extended") == [
        "MSCW",
        "ze_bin",
    ]


def test_grouped_split_keeps_source_groups_disjoint_when_supported():
    y = pd.Series(np.repeat([0, 1], 60))
    groups = pd.Series(np.tile(np.arange(6), 20))
    train, validation, test, metadata = models._classification_split_indices(
        y, groups, train_fraction=0.5, random_state=7, grouped=True
    )
    assert metadata["method"] == "grouped_source_file"
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[validation])
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    assert set(groups.iloc[validation]).isdisjoint(groups.iloc[test])


def test_grouped_split_uses_global_assignment_for_overlapping_group_ids():
    y = pd.Series(np.repeat([0, 1], 60))
    groups = pd.Series(np.concatenate([np.tile(np.arange(6), 10), np.tile(np.arange(2, 8), 10)]))
    train, validation, test, metadata = models._classification_split_indices(
        y, groups, train_fraction=0.5, random_state=7, grouped=True
    )
    assert metadata["groups_overlap_labels"] is True
    train_groups = set(groups.iloc[train])
    validation_groups = set(groups.iloc[validation])
    test_groups = set(groups.iloc[test])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_event_split_reports_insufficient_holdout_events():
    y = pd.Series([0, 0, 1, 1])
    with pytest.raises(ValueError, match="holdout events"):
        models._classification_split_indices(
            y, None, train_fraction=0.5, random_state=7, grouped=False
        )


def test_grouped_split_reports_insufficient_holdout_groups():
    y = pd.Series(np.repeat([0, 1], 60))
    groups = pd.Series(np.tile(np.arange(6), 20))
    with pytest.raises(ValueError, match="holdout groups"):
        models._classification_split_indices(
            y, groups, train_fraction=0.9, random_state=7, grouped=True
        )


def test_nuisance_diagnostics_handles_telescope_ids_above_63():
    frame = pd.DataFrame(
        {
            "size_64": [1.0, np.nan, 1.0, np.nan],
            "width_128": [0.1, 0.2, np.nan, np.nan],
        }
    )
    diagnostics = models._classification_nuisance_diagnostics(frame, pd.Series([0, 0, 1, 1]))
    assert diagnostics["feature_missing_fraction"]["n"] == 4


def test_threshold_calibration_is_quantile_based_and_background_limit_nonzero():
    calibration = classification_thresholds_from_signal(np.linspace(0.1, 0.9, 9))
    assert calibration["threshold"].is_monotonic_decreasing
    efficiency = _efficiency_dataframe(
        "test", np.array([0.9, 0.1]), np.array([1, 0]), np.array([0.5])
    )
    assert efficiency.loc[0, "background_efficiency_upper95"] > 0
