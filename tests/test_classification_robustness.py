"""Focused tests for classification hardening."""

import numpy as np
import pandas as pd

from eventdisplay_ml import data_processing, features, models


def test_telescope_config_comparison_tolerates_float_noise_and_nan():
    first = {
        "tel_ids": np.array([1, 2]),
        "mirror_area": np.array([100.0, np.nan]),
        "tel_x": np.array([0.0, 10.0]),
        "tel_y": np.array([1.0, 2.0]),
    }
    second = {
        "tel_ids": np.array([1, 2]),
        "mirror_area": np.array([100.0 + 1e-8, np.nan]),
        "tel_x": np.array([0.0, 10.0 + 1e-8]),
        "tel_y": np.array([1.0, 2.0]),
    }

    assert data_processing._telescope_configs_match(first, second)


def test_robust_profile_excludes_routing_and_activity_columns():
    columns = [
        "MSCW",
        "MSCL",
        "width_0",
        "length_0",
        "tel_active_0",
        "mirror_area_0",
        "ze_bin",
        "Erec",
        "__source_file",
    ]

    assert features.classification_feature_columns(columns, profile="robust") == [
        "MSCW",
        "MSCL",
        "width_0",
        "length_0",
        "ze_bin",
    ]


def test_extended_profile_retains_historical_features_but_excludes_routing():
    columns = [
        "MSCW",
        "DispNImages",
        "ze_bin",
        "Erec",
        "tel_active_0",
        "__source_file",
    ]

    assert features.classification_feature_columns(columns) == [
        "MSCW",
        "DispNImages",
        "ze_bin",
        "tel_active_0",
    ]


def test_grouped_split_keeps_source_files_disjoint():
    labels = pd.Series(np.repeat([0, 1], 80))
    groups = pd.Series(
        np.concatenate([np.repeat(np.arange(8), 10), np.repeat(np.arange(8, 16), 10)])
    )

    train, validation, test, method = models._classification_split_indices(
        labels, groups, train_fraction=0.5, random_state=7
    )

    assert method == "grouped_source_file"
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[validation])
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    assert set(groups.iloc[validation]).isdisjoint(groups.iloc[test])


def test_classification_source_groups_are_unique_across_classes():
    df = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "__source_file_id": [0, 1, 0, 1],
            "__source_file": ["signal_a.root", "signal_b.root", "background_a.root", "background_b.root"],
        }
    )

    assert models._classification_source_groups(df).tolist() == [
        "1:signal_a.root",
        "1:signal_b.root",
        "0:background_a.root",
        "0:background_b.root",
    ]
    assert models._classification_source_groups(df.drop(columns="__source_file")).tolist() == [
        "1:0",
        "1:1",
        "0:0",
        "0:1",
    ]
