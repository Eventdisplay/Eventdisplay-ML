"""
Shared data processing utilities for XGBoost stereo analysis.

Provides common functions for flattening and preprocessing telescope array data.
"""

import numpy as np
import pandas as pd


def flatten_data_vectorized(
    df,
    n_tel,
    training_variables,
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
        Input DataFrame containing telescope data. Must include columns for
        all variables in ``training_variables``, plus "DispTelList_T".
        If ``apply_pointing_corrections`` is True, must also contain
        "fpointing_dx" and "fpointing_dy".
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

    def to_padded_array(arrays):
        """Convert list of variable-length arrays to fixed-size numpy array, padding with NaN."""
        max_len = max(len(arr) if hasattr(arr, "__len__") else 1 for arr in arrays)
        result = np.full((len(arrays), max_len), np.nan)
        for i, arr in enumerate(arrays):
            if hasattr(arr, "__len__"):
                result[i, : len(arr)] = arr
            else:
                result[i, 0] = arr
        return result

    def to_dense_array(col):
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
            return to_padded_array(arrays)

    tel_list_matrix = to_dense_array(df[tel_list_col])

    for var_name in training_variables:
        data_matrix = to_dense_array(df[var_name])

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
        df_flat[f"E_{i}"] = np.log10(np.clip(df_flat[f"E_{i}"], 1e-6, None))
        df_flat[f"ES_{i}"] = np.log10(np.clip(df_flat[f"ES_{i}"], 1e-6, None))

        if apply_pointing_corrections:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]

    df_flat = pd.concat([df_flat, pd.DataFrame(new_cols, index=df.index)], axis=1)

    cast_type = dtype if dtype is not None else np.float32
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

    return pd.concat([df_flat, extra_cols], axis=1)
