"""Focused tests for classification hardening."""

import numpy as np
import pandas as pd

from eventdisplay_ml import features, models


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
