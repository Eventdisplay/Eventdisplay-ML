"""
Shared data processing utilities for XGBoost analysis.

Provides common functions for flattening and preprocessing telescope array data.
"""

import logging

import numpy as np
import pandas as pd
import uproot

from eventdisplay_ml.training_variables import (
    xgb_all_classification_training_variables,
    xgb_all_regression_training_variables,
    xgb_per_telescope_training_variables,
)

_logger = logging.getLogger(__name__)


def flatten_data_vectorized(
    df,
    n_tel,
    training_variables,
    analysis_type,
    apply_pointing_corrections=False,
    dtype=None,
):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns, handles
    telescope indexing via DispTelList_T, and creates derived features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data. If apply_pointing_corrections
        is True, must also contain "fpointing_dx" and "fpointing_dy".
    n_tel : int
        Number of telescopes to flatten for.
    training_variables : list[str]
        List of training variable names to flatten.
    apply_pointing_corrections : bool, optional
        If True, apply pointing offset corrections to cen_x and cen_y.
        Set to True for inference, False for training. Default is False.
    dtype : numpy.dtype, optional
        Data type to cast flattened features to. If None, the main flattened
        features are not explicitly cast, but extra derived columns are created
        with dtype ``np.float32``. Use np.float32 for memory efficiency in inference.

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame with per-telescope columns suffixed by ``_{i}``
        for telescope index ``i``, plus derived features (disp_x, disp_y,
        loss_loss, loss_dist, width_length, size, E, ES, and optionally
        pointing-corrected cen_x/cen_y), and extra columns (Xoff,
        Xoff_intersect, Erec, EmissionHeight, etc.).
    """
    flat_features = {}
    tel_list_col = "DispTelList_T"

    tel_list_matrix = _to_dense_array(df[tel_list_col])

    for var_name in training_variables:
        if var_name in df:
            data_matrix = _to_dense_array(df[var_name])
        else:
            _logger.debug(
                "Training variable %s missing in input data; filling with NaN values.",
                var_name,
            )
            data_matrix = np.full((len(df), n_tel), np.nan)

        for i in range(n_tel):
            col_name = f"{var_name}_{i}"

            if var_name.startswith("Disp"):
                # Case 1: Simple index i
                if i < data_matrix.shape[1]:
                    flat_features[col_name] = data_matrix[:, i]
                else:
                    flat_features[col_name] = np.full(len(df), np.nan)
            else:
                # Case 2: Index lookup via DispTelList_T
                target_tel_indices = tel_list_matrix[:, i].astype(int)

                row_indices = np.arange(len(df))
                valid_mask = (target_tel_indices >= 0) & (target_tel_indices < data_matrix.shape[1])
                result = np.full(len(df), np.nan)
                result[valid_mask] = data_matrix[
                    row_indices[valid_mask], target_tel_indices[valid_mask]
                ]

                flat_features[col_name] = result

    df_flat = pd.DataFrame(flat_features, index=df.index)

    if dtype is not None:
        df_flat = df_flat.astype(dtype)

    new_cols = {}
    for i in range(n_tel):
        new_cols[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
        new_cols[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        new_cols[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
        new_cols[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        new_cols[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)

        df_flat[f"size_{i}"] = np.log10(np.clip(df_flat[f"size_{i}"], 1e-6, None))
        if "E_{i}" in df_flat:
            df_flat[f"E_{i}"] = np.log10(np.clip(df_flat[f"E_{i}"], 1e-6, None))
        if "ES_{i}" in df_flat:
            df_flat[f"ES_{i}"] = np.log10(np.clip(df_flat[f"ES_{i}"], 1e-6, None))

        if apply_pointing_corrections:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]

    df_flat = pd.concat([df_flat, pd.DataFrame(new_cols, index=df.index)], axis=1)

    cast_type = dtype if dtype is not None else np.float32
    if analysis_type == "stereo_analysis":
        extra_cols = pd.DataFrame(
            {
                "Xoff_weighted_bdt": df["Xoff"].astype(cast_type),
                "Yoff_weighted_bdt": df["Yoff"].astype(cast_type),
                "Xoff_intersect": df["Xoff_intersect"].astype(cast_type),
                "Yoff_intersect": df["Yoff_intersect"].astype(cast_type),
                "Diff_Xoff": (df["Xoff"] - df["Xoff_intersect"]).astype(cast_type),
                "Diff_Yoff": (df["Yoff"] - df["Yoff_intersect"]).astype(cast_type),
                "Erec": np.log10(np.clip(df["Erec"], 1e-6, None)).astype(cast_type),
                "ErecS": np.log10(np.clip(df["ErecS"], 1e-6, None)).astype(cast_type),
                "EmissionHeight": df["EmissionHeight"].astype(cast_type),
            },
            index=df.index,
        )
    else:  # classification
        extra_cols = pd.DataFrame(
            {
                "MSCW": df["MSCW"].astype(cast_type),
                "MSCL": df["MSCL"].astype(cast_type),
                "EChi2S": np.log10(np.clip(df["EChi2S"], 1e-6, None)).astype(cast_type),
                "EmissionHeight": df["EmissionHeight"].astype(cast_type),
                "EmissionHeightChi2": np.log10(
                    np.clip(df["EmissionHeightChi2"], 1e-6, None)
                ).astype(cast_type),
            },
            index=df.index,
        )

    return pd.concat([df_flat, extra_cols], axis=1)


def _to_padded_array(arrays):
    """Convert list of variable-length arrays to fixed-size numpy array, padding with NaN."""
    max_len = max(len(arr) if hasattr(arr, "__len__") else 1 for arr in arrays)
    result = np.full((len(arrays), max_len), np.nan)
    for i, arr in enumerate(arrays):
        if hasattr(arr, "__len__"):
            result[i, : len(arr)] = arr
        else:
            result[i, 0] = arr
    return result


def _to_dense_array(col):
    """
    Convert a column of variable-length telescope data to a dense 2D numpy array.

    Handles uproot's awkward-style variable-length arrays from ROOT files
    by converting to plain Python lists first to avoid per-element iteration overhead.

    Parameters
    ----------
    col : pandas.Series
        Column containing variable-length arrays.

    Returns
    -------
    numpy.ndarray
        2D numpy array with shape (n_events, max_telescopes), padded with NaN.
    """
    arrays = col.tolist() if hasattr(col, "tolist") else list(col)
    try:
        return np.vstack(arrays)
    except (ValueError, TypeError):
        return _to_padded_array(arrays)


def load_training_data(
    input_files, n_tel, max_events, analysis_type="stereo_analysis", erec_range=None
):
    """
    Load and flatten training data from the mscw file for the requested telescope multiplicity.

    Parameters
    ----------
    input_files : list[str]
        List of input mscw ROOT files.
    n_tel : int
        Telescope multiplicity to filter on.
    max_events : int
        Maximum number of events to load. If <= 0, load all available events.
    analysis_type : str, optional
        Type of analysis: "stereo_analysis", "signal_classification", or "background_classification".
    erec_range : tuple(float, float), optional
        Range of log10(Erec/TeV) for event selection: (min, max)
    """
    _logger.info(f"\n--- Loading and Flattening Data for {analysis_type} for n_tel = {n_tel} ---")
    _logger.info(
        "Max events to process: "
        f"{max_events if max_events is not None and max_events > 0 else 'All available'}"
    )

    event_cut = f"(DispNImages == {n_tel})"
    if analysis_type == "stereo_analysis":
        branch_list = ["MCxoff", "MCyoff", "MCe0", *xgb_all_regression_training_variables()]
    elif analysis_type in ("signal_classification", "background_classification"):
        branch_list = [*xgb_all_classification_training_variables()]
        event_cut += "& (Erec > 0) & (MSCW > -2) & (MSCW < 2) & (MSCL > -2) & (MSCL < 5)"
        event_cut += "& (EmissionHeight>0) & (EmissionHeight<50)"
        if erec_range is not None:
            e_min, e_max = (10**v for v in erec_range)
            event_cut += f"& (Erec >= {e_min}) & (Erec <= {e_max})"
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    dfs = []

    if not input_files:
        _logger.error("No input files provided.")
        return pd.DataFrame()
    if max_events is not None and max_events > 0:
        max_events_per_file = max_events // len(input_files)
    else:
        max_events_per_file = None

    for f in input_files:
        try:
            with uproot.open(f) as root_file:
                if "data" in root_file:
                    _logger.info(f"Processing file: {f}")
                    tree = root_file["data"]
                    df = tree.arrays(branch_list, cut=event_cut, library="pd")
                    _logger.info(f"Number of events after filter {event_cut}: {len(df)}")
                    if max_events_per_file and len(df) > max_events_per_file:
                        df = df.sample(n=max_events_per_file, random_state=42)
                    if not df.empty:
                        dfs.append(df)
                else:
                    _logger.warning(f"File: {f} does not contain a 'data' tree.")
        except Exception as e:
            _logger.error(f"Error opening or reading file {f}: {e}")

    if len(dfs) == 0:
        _logger.error("No data loaded from input files.")
        return pd.DataFrame()

    data_tree = pd.concat(dfs, ignore_index=True)
    _logger.info(f"Total events for n_tel={n_tel}: {len(data_tree)}")

    df_flat = flatten_data_vectorized(
        data_tree,
        n_tel,
        xgb_per_telescope_training_variables(),
        analysis_type,
        apply_pointing_corrections=False,
    )

    if analysis_type == "stereo_analysis":
        df_flat["MCxoff"] = data_tree["MCxoff"]
        df_flat["MCyoff"] = data_tree["MCyoff"]
        df_flat["MCe0"] = np.log10(data_tree["MCe0"])

    # Keep events even if some optional training variables are missing; only drop
    # columns that are entirely NaN (e.g., missing branches like DispXoff_T).
    df_flat.dropna(axis=1, how="all", inplace=True)
    _logger.info(f"Final events for n_tel={n_tel} after cleanup: {len(df_flat)}")

    return df_flat


def apply_image_selection(df, selected_indices, analysis_type):
    """
    Filter and pad telescope lists for selected indices.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    selected_indices : list[int] or None
        List of selected telescope indices. If None or all 4 telescopes
        are selected, the DataFrame is returned unchanged.
    analysis_type : str, optional
        Type of analysis (e.g., "stereo_analysis")

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated "DispTelList_T" and "DispNImages" columns,
        and per-telescope variables padded to length 4 with NaN.
    """
    if selected_indices is None or len(selected_indices) == 4:
        return df

    selected_set = set(selected_indices)

    def calculate_intersection(tel_list):
        return [tel_idx for tel_idx in tel_list if tel_idx in selected_set]

    df = df.copy()
    df["DispTelList_T_new"] = df["DispTelList_T"].apply(calculate_intersection)
    df["DispNImages_new"] = df["DispTelList_T_new"].apply(len)

    _logger.info(
        f"\n{df[['DispNImages', 'DispTelList_T', 'DispNImages_new', 'DispTelList_T_new']].head(20).to_string()}"
    )

    df["DispTelList_T"] = df["DispTelList_T_new"]
    df["DispNImages"] = df["DispNImages_new"]
    df = df.drop(columns=["DispTelList_T_new", "DispNImages_new"])

    if analysis_type == "stereo_analysis":
        pad_vars = [
            *xgb_per_telescope_training_variables(),
            "fpointing_dx",
            "fpointing_dy",
        ]
    else:
        pad_vars = xgb_per_telescope_training_variables()
    for var_name in pad_vars:
        if var_name in df.columns:
            df[var_name] = df[var_name].apply(_pad_to_four)

    return df


def _pad_to_four(arr_like):
    """Pad a per-telescope array-like to length 4 with NaN values."""
    if isinstance(arr_like, (list, np.ndarray)):
        arr = np.asarray(arr_like, dtype=np.float32)
        pad = max(0, 4 - arr.shape[0])
        if pad:
            arr = np.pad(arr, (0, pad), mode="constant", constant_values=np.nan)
        return arr
    return arr_like
