"""Training variables for XGBoost direction reconstruction."""


def xgb_per_telescope_training_variables():
    """
    Telescope-type training variables for XGB.

    Disp variables with different indexing logic in data preparation.
    """
    return [
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
        "E",
        "ES",
        "cen_x",
        "cen_y",
        "cosphi",
        "sinphi",
        "loss",
        "size",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
    ]


def xgb_regression_training_variables():
    """Array-level training variables for XGB regression."""
    return [
        "DispNImages",
        "DispTelList_T",
        "Xoff",
        "Yoff",
        "Xoff_intersect",
        "Yoff_intersect",
        "Erec",
        "ErecS",
        "EmissionHeight",
    ]


def xgb_classification_training_variables():
    """Array-level training variables for XGB classification."""
    return [
        "DispNImages",
        "DispTelList_T",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "MSCW",
        "MSCL",
    ]


def xgb_all_regression_training_variables():
    """All training variables for XGB regression."""
    return xgb_per_telescope_training_variables() + xgb_regression_training_variables()


def xgb_all_classification_training_variables():
    """All training variables for XGB classification."""
    var_per_telescope = xgb_per_telescope_training_variables()
    # no energies for classification
    var_per_telescope.remove("E")
    var_per_telescope.remove("ES")

    return var_per_telescope + xgb_classification_training_variables()
