"""Features used for XGB training and prediction."""


def target_features(analysis_type):
    """
    Get target features based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.

    Returns
    -------
    list
        List of target feature names.
    """
    if analysis_type == "stereo_analysis":
        return ["Xoff_residual", "Yoff_residual", "E_residual"]  # sequence matters
    if "classification" in analysis_type:
        return []
    raise ValueError(f"Unknown analysis type: {analysis_type}")


def excluded_features(analysis_type, ntel):
    """
    Features not to be used for training/prediction.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.
    ntel : int
        Number of telescopes.

    Returns
    -------
    set
        Set of excluded feature names.
    """
    if analysis_type == "stereo_analysis":
        return {
            # Pointing corrections applied during preprocessing
            *[f"fpointing_dx_{i}" for i in range(ntel)],
            *[f"fpointing_dy_{i}" for i in range(ntel)],
        }
    if "classification" in analysis_type:
        return {
            "Erec",
            *[f"cen_x_{i}" for i in range(ntel)],
            *[f"cen_y_{i}" for i in range(ntel)],
            *[f"size_{i}" for i in range(ntel)],
            *[f"E_{i}" for i in range(ntel)],
            *[f"ES_{i}" for i in range(ntel)],
            *[f"fpointing_dx_{i}" for i in range(ntel)],
            *[f"fpointing_dy_{i}" for i in range(ntel)],
        }
    raise ValueError(f"Unknown analysis type: {analysis_type}")


def telescope_features(analysis_type):
    """
    Telescope-type features.

    Disp variables with different indexing logic in data preparation.

    Parameters
    ----------
    analysis_type : str
        Type of analysis, e.g. ``"classification"`` or ``"stereo_analysis"``.

    Returns
    -------
    list
        List of telescope-level feature names.
    """
    var = [
        "size",
        "cosphi",
        "sinphi",
        "loss",
        "dist",
        "width",
        "length",
        "asym",
        "tgrad_x",
        "R_core",
        "fpointing_dx",
        "fpointing_dy",
        "mirror_area",
        "tel_rel_x",
        "tel_rel_y",
        "tel_active",
    ]
    if analysis_type == "classification":
        return var

    return [
        *var,
        "cen_x",
        "cen_y",
        "E",
        "ES",
        "Disp_T",
        "DispXoff_T",
        "DispYoff_T",
        "DispWoff_T",
        "ntubes",
        "nlowgain",
    ]


def _regression_features(training):
    """Regression features."""
    var = [
        *telescope_features("stereo_analysis"),
        "DispNImages",
        "DispTelList_T",
        "ImgSel_list",
        "img2_ang",
        "Xoff",
        "Yoff",
        "Xoff_intersect",
        "Yoff_intersect",
        "Erec",
        "ErecS",
        "EmissionHeight",
        "ArrayPointing_Elevation",
        "ArrayPointing_Azimuth",
        "Xcore",
        "Ycore",
    ]
    if training:
        # Load MC truth values (residuals will be computed from these)
        return ["MCxoff", "MCyoff", "MCe0", *var]
    return var


def _classification_features():
    """Classification features."""
    var_tel = telescope_features("classification")
    var_array = [
        "DispNImages",
        "DispTelList_T",
        "ImgSel_list",
        "EChi2S",
        "EmissionHeight",
        "EmissionHeightChi2",
        "MSCW",
        "MSCL",
        "ArrayPointing_Elevation",
        "ArrayPointing_Azimuth",
        "Xcore",
        "Ycore",
    ]
    # energy used to bin the models, but not as feature
    # size used for sorting events during flattening, but not as feature
    return var_tel + var_array + ["Erec", "size"]


def clip_intervals():
    """
    Get clip intervals for variables.

    Returns
    -------
    dict
        Dictionary mapping variable names to clip intervals (min, max).
        Use None for no lower/upper bound.
    """
    fov_max = 7.0
    energy_min = 1e-3
    return {
        # Intersection results - avoid badly reconstructed events
        "Xoff_weighted_bdt": (-fov_max, fov_max),
        "Yoff_weighted_bdt": (-fov_max, fov_max),
        "Xoff_intersect": (-fov_max, fov_max),
        "Yoff_intersect": (-fov_max, fov_max),
        "Diff_Xoff": (-fov_max, fov_max),
        "Diff_Yoff": (-fov_max, fov_max),
        "DispXoff_T": (-fov_max, fov_max),
        "DispYoff_T": (-fov_max, fov_max),
        # Energy-related variables - log10 transformation with lower bound
        "Erec": (energy_min, None),
        "ErecS": (energy_min, None),
        "EChi2S": (energy_min, None),
        "EmissionHeightChi2": (1e-6, None),
        "img2_ang": (0.0, 360.0),
        # Per-telescope energy and size variables - log10 transformation with lower bound
        "size": (1, None),
        "E": (energy_min, None),
        "ES": (energy_min, None),
        "ntubes": (1, None),
        "nlowgain": (1, None),
        # Derived variables - avoid numerical issues
        "size_dist2": (1.0, None),
        "tgrad_x": (-50.0, 50.0),
        # Physical bounds
        "EmissionHeight": (0, 120),  # top of atmosphere
        "R_core": (-10, None),  # badly reconstructed events
    }


def features(analysis_type, training=True):
    """
    Get features based on analysis type.

    Parameters
    ----------
    analysis_type : str
        Type of analysis.
    training : bool, optional
        If True (default), return features including target features.
        If False, return only non-target features (i.e. features used
        for prediction).

    Returns
    -------
    list
        List of feature names.
    """
    if analysis_type == "stereo_analysis":
        return _regression_features(training)
    if "classification" in analysis_type:
        return _classification_features()
    raise ValueError(f"Unknown analysis type: {analysis_type}")
