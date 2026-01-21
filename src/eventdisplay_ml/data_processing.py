"""
Data processing for XGBoost analysis.

Provides common functions for flattening and preprocessing telescope array data.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from scipy.spatial import ConvexHull, QhullError

from eventdisplay_ml import features, utils
from eventdisplay_ml.geomag import calculate_geomagnetic_angles

_logger = logging.getLogger(__name__)

# Default fill value for missing telescope-dependent data
DEFAULT_FILL_VALUE = np.nan


def read_telescope_config(root_file):
    """
    Read telescope configuration from ROOT file.

    Parameters
    ----------
    root_file : uproot file handle
        Open ROOT file containing the telconfig tree.

    Returns
    -------
    dict
        Dictionary with telescope configuration:
        - 'n_tel': Total number of telescopes
        - 'tel_ids': Array of telescope IDs
        - 'mirror_areas': Array of mirror areas for each telescope
        - 'tel_x': Array of telescope X positions
        - 'tel_y': Array of telescope Y positions
        - 'max_tel_id': Maximum telescope ID
        - 'tel_types': Dictionary mapping mirror area to list of telescope IDs
    """
    if "telconfig" not in root_file:
        _logger.warning("No telconfig tree found in ROOT file, using default 4 telescopes")
        return {
            "n_tel": 4,
            "tel_ids": np.array([0, 1, 2, 3]),
            "mirror_areas": np.array([1.0, 1.0, 1.0, 1.0]),
            "tel_x": np.array([0.0, 0.0, 0.0, 0.0]),
            "tel_y": np.array([0.0, 0.0, 0.0, 0.0]),
            "max_tel_id": 3,
            "tel_types": {1.0: [0, 1, 2, 3]},
        }

    telconfig_tree = root_file["telconfig"]
    telconfig_data = telconfig_tree.arrays(
        ["NTel", "TelID", "MirrorArea", "TelX", "TelY"], library="np"
    )

    n_tel = int(telconfig_data["NTel"][0])
    tel_ids = telconfig_data["TelID"]
    mirror_areas = telconfig_data["MirrorArea"]
    tel_x = telconfig_data["TelX"]
    tel_y = telconfig_data["TelY"]
    max_tel_id = int(np.max(tel_ids))

    # Group telescopes by mirror area (telescope type)
    tel_types = {}
    for tel_id, mirror_area in zip(tel_ids, mirror_areas):
        if mirror_area not in tel_types:
            tel_types[mirror_area] = []
        tel_types[mirror_area].append(int(tel_id))

    _logger.info(f"Telescope configuration: {n_tel} telescopes, max TelID: {max_tel_id}")
    _logger.info(f"Telescope types by mirror area: {tel_types}")

    return {
        "n_tel": n_tel,
        "tel_ids": tel_ids,
        "mirror_areas": mirror_areas,
        "tel_x": tel_x,
        "tel_y": tel_y,
        "max_tel_id": max_tel_id,
        "tel_types": tel_types,
    }


def _resolve_branch_aliases(tree, branch_list):
    """
    Resolve branch name aliases (e.g. R_core vs R) and drop missing optional branches.

    Resolves differences between CTAO and VERITAS branch naming conventions.
    """
    keys = set(tree.keys())
    resolved = []
    rename = {}
    missing = set()

    # R_core vs R
    for b in branch_list:
        if b == "R_core" and b not in keys:
            if "R" in keys:
                resolved.append("R")
                rename["R"] = "R_core"
                _logger.info("Branch 'R_core' not found; using 'R'")
            else:
                _logger.warning("Branches 'R_core' and fallback 'R' not found")
        else:
            resolved.append(b)

    # Drop synthesized branches
    synthesized = {
        "mirror_areas",
        "tel_rel_x",
        "tel_rel_y",
        "tel_shower_x",
        "tel_shower_y",
        "tel_active",
    }
    resolved = [b for b in resolved if b not in synthesized]

    # Drop missing optional branches
    optional = {"fpointing_dx", "fpointing_dy", "E", "Erec", "ErecS"}
    final = []
    for b in resolved:
        if b in optional and b not in keys:
            missing.add(b)
            _logger.info(f"Branch '{b}' not found; will fill with defaults")
        else:
            final.append(b)

    return final, rename, missing


def _ensure_fpointing_fields(arr):
    """Ensure fpointing_dx and fpointing_dy exist; fill zeros if missing."""
    fields = set(getattr(arr, "fields", []) or [])
    if "DispTelList_T" in fields:
        zeros_like_tel = ak.values_astype(ak.zeros_like(arr["DispTelList_T"]), np.float32)
        if "fpointing_dx" not in fields:
            arr = ak.with_field(arr, zeros_like_tel, "fpointing_dx")
        if "fpointing_dy" not in fields:
            arr = ak.with_field(arr, zeros_like_tel, "fpointing_dy")
    return arr


def _ensure_optional_scalar_fields(arr, missing_optional):
    """Fill optional scalar fields like Erec/ErecS with defaults when missing."""
    fields = set(getattr(arr, "fields", []) or [])
    n = len(arr)
    if "Erec" in missing_optional and "Erec" not in fields:
        arr = ak.with_field(arr, np.full(n, DEFAULT_FILL_VALUE, dtype=np.float32), "Erec")
    if "ErecS" in missing_optional and "ErecS" not in fields:
        arr = ak.with_field(arr, np.full(n, DEFAULT_FILL_VALUE, dtype=np.float32), "ErecS")
    return arr


def _rename_fields_ak(arr, rename_map):
    """Rename fields in an Awkward Array for keys present in the array.

    Parameters
    ----------
    arr : ak.Array
        Input Awkward Array with record fields.
    rename_map : dict[str, str]
        Mapping from old field names to new field names.

    Returns
    -------
    ak.Array
        Array with fields renamed where applicable.
    """
    fields = set(getattr(arr, "fields", []) or [])
    for old, new in (rename_map or {}).items():
        if old in fields and old != new:
            # Add new field with the same data
            arr = ak.with_field(arr, arr[old], where=new)
            # Remove the old field if the function exists; otherwise leave both
            try:
                arr = ak.without_field(arr, old)
            except Exception:
                pass
            fields = set(getattr(arr, "fields", []) or [])
    return arr


def _make_mirror_area_columns(tel_config, max_tel_id, n_evt, default_value):
    """Build constant mirror area columns from tel_config."""
    columns = {}
    tel_id_to_mirror = dict(zip(tel_config["tel_ids"], tel_config["mirror_areas"]))
    for tel_idx in range(max_tel_id + 1):
        col_name = f"mirror_areas_{tel_idx}"
        if tel_idx in tel_id_to_mirror:
            mirror_val = float(tel_id_to_mirror[tel_idx])
            if mirror_val == 0.0:
                mirror_val = 100.0
            columns[col_name] = np.full(n_evt, mirror_val, dtype=np.float32)
        else:
            columns[col_name] = np.full(n_evt, default_value, dtype=np.float32)
    return columns


def _make_tel_active_columns(tel_list_matrix, max_tel_id, n_evt):
    """Build binary telescope active columns indicating which telescopes participated in events."""
    columns = {}
    active_matrix = np.zeros((n_evt, max_tel_id + 1), dtype=np.float32)
    row_indices, col_indices = np.where(~np.isnan(tel_list_matrix))
    tel_ids = tel_list_matrix[row_indices, col_indices].astype(int)
    valid_mask = tel_ids <= max_tel_id
    active_matrix[row_indices[valid_mask], tel_ids[valid_mask]] = 1.0
    for tel_idx in range(max_tel_id + 1):
        columns[f"tel_active_{tel_idx}"] = active_matrix[:, tel_idx]
    return columns


def _make_relative_coord_columns(
    var,
    tel_config,
    max_tel_id,
    n_evt,
    core_x,
    core_y,
    elev_rad,
    azim_rad,
    valid_mask,
    default_value,
):
    """Build relative/shower coordinate columns for a single synthetic variable."""
    columns = {}
    cos_elev = np.cos(elev_rad)
    cos_azim = np.cos(azim_rad)
    sin_azim = np.sin(azim_rad)

    all_tel_x = np.full(max_tel_id + 1, np.nan)
    all_tel_y = np.full(max_tel_id + 1, np.nan)

    for tid, tx, ty in zip(tel_config["tel_ids"], tel_config["tel_x"], tel_config["tel_y"]):
        if tid <= max_tel_id:
            all_tel_x[tid] = tx
            all_tel_y[tid] = ty

    rel_x = all_tel_x[:, np.newaxis] - core_x
    rel_y = all_tel_y[:, np.newaxis] - core_y

    if var == "tel_rel_x":
        results = rel_x
    elif var == "tel_rel_y":
        results = rel_y
    elif var == "tel_shower_x":
        results = -sin_azim * rel_x + cos_azim * rel_y
    elif var == "tel_shower_y":
        results = cos_azim * cos_elev * rel_x + sin_azim * cos_elev * rel_y
    else:
        results = np.full((max_tel_id + 1, n_evt), default_value)

    for tel_idx in range(max_tel_id + 1):
        col_name = f"{var}_{tel_idx}"

        # If the telescope doesn't exist in config, it remains NaN from step 2
        res = results[tel_idx]

        # Apply validity mask (e.g. handle events where reconstruction failed)
        final_values = np.where(valid_mask & ~np.isnan(res), res, default_value).astype(np.float32)
        columns[col_name] = final_values

    return columns


def _flatten_variable_columns(
    var, data, tel_list_matrix, max_tel_id, n_evt, default_value, r_core_data=None
):
    """Flatten one variable into per-telescope columns, optionally sorted by distance."""
    columns = {}

    if var.startswith("Disp"):
        for tel_idx in range(max_tel_id + 1):
            col_name = f"{var}_{tel_idx}"
            if tel_idx < data.shape[1]:
                columns[col_name] = data[:, tel_idx]
            else:
                columns[col_name] = np.full(n_evt, default_value, dtype=np.float32)
        return columns

    full_matrix = np.full((n_evt, max_tel_id + 1), default_value, dtype=np.float32)
    row_indices, col_indices = np.where(~np.isnan(tel_list_matrix))
    tel_ids = tel_list_matrix[row_indices, col_indices].astype(int)

    valid_mask = tel_ids <= max_tel_id
    full_matrix[row_indices[valid_mask], tel_ids[valid_mask]] = data[
        row_indices[valid_mask], col_indices[valid_mask]
    ]

    # Apply distance-based sorting if R_core data is available
    if r_core_data is not None:
        # Vectorized sorting: use argsort to reorder columns by distance
        # Pad R_core to match max_tel_id
        if r_core_data.shape[1] <= max_tel_id:
            padded = np.full((n_evt, max_tel_id + 1), np.nan)
            padded[:, : r_core_data.shape[1]] = r_core_data
            r_core_data = padded

        # For each event, create sort indices that put NaNs last
        sort_indices = np.zeros((n_evt, max_tel_id + 1), dtype=int)
        for evt_idx in range(n_evt):
            distances = r_core_data[evt_idx]
            # Create sort order: NaN-safe (NaN values go last)
            valid_mask = ~np.isnan(distances)
            valid_idx = np.where(valid_mask)[0]
            nan_idx = np.where(~valid_mask)[0]
            if len(valid_idx) > 0:
                sorted_valid = valid_idx[np.argsort(distances[valid_idx])]
                sort_indices[evt_idx, : len(sorted_valid)] = sorted_valid
                if len(nan_idx) > 0:
                    sort_indices[evt_idx, len(sorted_valid) :] = nan_idx
            else:
                sort_indices[evt_idx] = np.arange(max_tel_id + 1)

        full_matrix = full_matrix[np.arange(n_evt)[:, np.newaxis], sort_indices]

    for tel_idx in range(max_tel_id + 1):
        columns[f"{var}_{tel_idx}"] = full_matrix[:, tel_idx]

    return columns


def flatten_telescope_data_vectorized(
    df, n_tel, features, analysis_type, training=True, tel_config=None
):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns sorted by distance
    to the shower core. The column index represents proximity to the core rather
    than telescope ID (column 0 = closest telescope, column 1 = second closest, etc).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing telescope data.
    n_tel : int
        Number of telescopes to flatten for (maximum telescope index).
    features : list[str]
        List of training variable names to flatten.
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis").
    training : bool, optional
        If True, indicates training mode. Default is True.
    tel_config : dict, optional
        Telescope configuration dictionary with 'max_tel_id' and 'tel_types'.

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame with per-telescope columns suffixed by ``_{i}``
        where ``i`` represents the i-th closest telescope to the shower core,
        plus derived features, and array features. Missing telescopes are
        filled with NaN.
    """
    flat_features = {}
    tel_list_matrix = _to_dense_array(df["DispTelList_T"])
    n_evt = len(df)
    default_value = DEFAULT_FILL_VALUE

    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    core_x = _to_numpy_1d(df["Xcore"], np.float32)
    core_y = _to_numpy_1d(df["Ycore"], np.float32)
    core_x[core_x <= -90000] = np.nan
    core_y[core_y <= -90000] = np.nan
    elev_rad = np.radians(_to_numpy_1d(df["ArrayPointing_Elevation"], np.float32))
    azim_rad = np.radians(_to_numpy_1d(df["ArrayPointing_Azimuth"], np.float32))
    valid_mask = np.isfinite(core_x) & np.isfinite(core_y)

    # Pre-load R_core data for distance-based sorting
    r_core_data = _to_dense_array(df["R_core"]) if _has_field(df, "R_core") else None

    for var in features:
        if var == "mirror_areas" and tel_config:
            flat_features.update(
                _make_mirror_area_columns(tel_config, max_tel_id, n_evt, default_value)
            )
            continue

        if var == "tel_active":
            _logger.info(f"Computing synthetic feature: {var}")
            flat_features.update(_make_tel_active_columns(tel_list_matrix, max_tel_id, n_evt))
            continue

        if var in ("tel_rel_x", "tel_rel_y", "tel_shower_x", "tel_shower_y") and tel_config:
            _logger.info(f"Computing synthetic feature: {var}")
            flat_features.update(
                _make_relative_coord_columns(
                    var,
                    tel_config,
                    max_tel_id,
                    n_evt,
                    core_x,
                    core_y,
                    elev_rad,
                    azim_rad,
                    valid_mask,
                    default_value,
                )
            )
            continue

        data = _to_dense_array(df[var]) if _has_field(df, var) else np.full((n_evt, n_tel), np.nan)
        flat_features.update(
            _flatten_variable_columns(
                var, data, tel_list_matrix, max_tel_id, n_evt, default_value, r_core_data
            )
        )

    index = _get_index(df, n_evt)
    df_flat = flatten_telescope_variables(n_tel, flat_features, index, tel_config)
    return pd.concat(
        [df_flat, extra_columns(df, analysis_type, training, index, tel_config)], axis=1
    )


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
    if isinstance(col, ak.Array):
        padded = ak.pad_none(col, target=int(ak.max(ak.num(col))), axis=1)
        return ak.to_numpy(ak.fill_none(padded, np.nan))

    if isinstance(col, pd.Series):
        col = col.values

    try:
        ak_arr = ak.from_iter(col)
        padded = ak.pad_none(ak_arr, target=ak.max(ak.num(ak_arr)), axis=1)
        return ak.to_numpy(ak.fill_none(padded, np.nan))
    except (ValueError, TypeError):
        return _to_padded_array(col)


def _to_numpy_1d(x, dtype=np.float32):
    """Convert Series/array/ak.Array to a 1D numpy array with dtype."""
    if hasattr(x, "to_numpy"):
        try:
            return x.to_numpy(dtype=dtype)
        except TypeError:
            return np.asarray(x).astype(dtype)
    if isinstance(x, ak.Array):
        return ak.to_numpy(x).astype(dtype)
    return np.asarray(x, dtype=dtype)


def _has_field(df_like, name):
    """Check presence of a field/column in pandas DataFrame or Awkward Array."""
    if isinstance(df_like, pd.DataFrame):
        return name in df_like.columns
    if isinstance(df_like, (ak.Array, ak.Record)):
        return name in (getattr(df_like, "fields", []) or [])
    try:
        return name in df_like
    except (TypeError, AttributeError):
        return False


def _get_index(df_like, n):
    """Get an index for DataFrame construction from pandas or fallback to RangeIndex."""
    if isinstance(df_like, pd.DataFrame):
        return df_like.index
    return pd.RangeIndex(n)


def flatten_feature_data(group_df, ntel, analysis_type, training, tel_config=None):
    """
    Get flattened features for events.

    All events are flattened with features indexed by actual telescope ID.

    Parameters
    ----------
    group_df : pandas.DataFrame
        DataFrame with events to flatten.
    ntel : int
        Maximum telescope index.
    analysis_type : str
        Type of analysis.
    training : bool
        Whether in training mode.
    tel_config : dict, optional
        Telescope configuration dictionary.
    """
    df_flat = flatten_telescope_data_vectorized(
        group_df,
        ntel,
        features.telescope_features(analysis_type),
        analysis_type=analysis_type,
        training=training,
        tel_config=tel_config,
    )
    max_tel_id = tel_config["max_tel_id"] if tel_config else ntel - 1
    excluded_columns = set(features.target_features(analysis_type)) | set(
        features.excluded_features(analysis_type, max_tel_id + 1)
    )
    return df_flat.drop(columns=excluded_columns, errors="ignore")


def load_training_data(model_configs, file_list, analysis_type):
    """
    Load and flatten training data from the mscw file.

    Processes all events regardless of telescope multiplicity. Features are created
    for all telescopes with default value NaN for missing telescopes.
    Reads telescope configuration from the ROOT file to determine the number
    and types of telescopes.

    Parameters
    ----------
    model_configs : dict
        Dictionary containing model configuration parameters.
    file_list : str
        Path to text file containing list of input mscw files.
    analysis_type : str
        Type of analysis (e.g., "stereo_analysis").

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame ready for training.
    """
    max_events = model_configs.get("max_events", None)
    random_state = model_configs.get("random_state", None)

    _logger.info(f"--- Loading and Flattening Data for {analysis_type} ---")
    _logger.info("Processing all events regardless of multiplicity")
    _logger.info(
        "Max events to process: "
        f"{max_events if max_events is not None and max_events > 0 else 'All available'}"
    )
    if analysis_type == "classification":
        _logger.info(f"Adding zenith binning: {model_configs.get('zenith_bins_deg', [])}")

    input_files = utils.read_input_file_list(file_list)

    branch_list = features.features(analysis_type, training=True)
    _logger.info(f"Branch list: {branch_list}")
    if max_events is not None and max_events > 0:
        max_events_per_file = max_events // len(input_files)
    else:
        max_events_per_file = None
    _logger.info(f"Max events per file: {max_events_per_file}")

    tel_config = None  # Will be read from first file
    dfs = []
    executor = ThreadPoolExecutor(max_workers=model_configs.get("max_cores", 1))
    total_files = len(input_files)
    for file_idx, f in enumerate(input_files, start=1):
        try:
            with uproot.open(f) as root_file:
                if "data" not in root_file:
                    _logger.warning(f"File: {f} does not contain a 'data' tree.")
                    continue

                # Read telescope configuration from first file
                if tel_config is None:
                    tel_config = read_telescope_config(root_file)
                    model_configs["tel_config"] = tel_config

                _logger.info(f"Processing file: {f} (file {file_idx}/{total_files})")
                tree = root_file["data"]
                resolved_branch_list, rename_map, missing_optional = _resolve_branch_aliases(
                    tree, branch_list
                )
                df_ak = tree.arrays(
                    resolved_branch_list,
                    cut=model_configs.get("pre_cuts", None),
                    library="ak",
                    decompression_executor=executor,
                )
                if rename_map:
                    rename_present = {k: v for k, v in rename_map.items() if k in df_ak.fields}
                    if rename_present:
                        df_ak = _rename_fields_ak(df_ak, rename_present)
                # Ensure optional scalar fields and fpointing fields exist
                df_ak = _ensure_optional_scalar_fields(df_ak, missing_optional)
                df_ak = _ensure_fpointing_fields(df_ak)
                if len(df_ak) == 0:
                    continue

                if max_events_per_file and len(df_ak) > max_events_per_file:
                    rng = np.random.default_rng(random_state)
                    indices = rng.choice(len(df_ak), max_events_per_file, replace=False)
                    df_ak = df_ak[indices]

                n_before = tree.num_entries
                _logger.info(
                    f"Number of events before / after event cut: {n_before} / {len(df_ak)}"
                    f" (fraction retained: {len(df_ak) / n_before:.4f})"
                )

                # Flatten using telescope configuration (conversion to pandas happens inside)
                df_flat = flatten_telescope_data_vectorized(
                    df_ak,
                    tel_config["max_tel_id"] + 1,
                    features.telescope_features(analysis_type),
                    analysis_type,
                    training=True,
                    tel_config=tel_config,
                )
                if analysis_type == "stereo_analysis":
                    df_flat["MCxoff"] = _to_numpy_1d(df_ak["MCxoff"], np.float32)
                    df_flat["MCyoff"] = _to_numpy_1d(df_ak["MCyoff"], np.float32)
                    df_flat["MCe0"] = np.log10(_to_numpy_1d(df_ak["MCe0"], np.float32))
                elif analysis_type == "classification":
                    df_flat["ze_bin"] = zenith_in_bins(
                        90.0 - _to_numpy_1d(df_ak["ArrayPointing_Elevation"], np.float32),
                        model_configs.get("zenith_bins_deg", []),
                    )

                dfs.append(df_flat)

                del df_ak
        except Exception as e:
            raise FileNotFoundError(f"Error opening or reading file {f}: {e}") from e

    df_final = pd.concat(dfs, ignore_index=True)
    df_final.dropna(axis=1, how="all", inplace=True)
    _logger.info(f"Total events loaded: {len(df_final)}")

    # Log multiplicity distribution
    if "DispNImages" in df_final.columns:
        mult_counts = df_final["DispNImages"].value_counts().sort_index()
        for mult, count in mult_counts.items():
            _logger.info(f"\tDispNImages={int(mult)}: {count} events")

    if analysis_type == "classification":
        counts = df_final["ze_bin"].value_counts().sort_index()
        for zb, n in counts.items():
            _logger.info(f"\tze_bin={zb}: {n} events")

    if len(df_final) == 0:
        raise ValueError("No data loaded from input files.")

    print_variable_statistics(df_final)

    return df_final


def apply_clip_intervals(df, n_tel=None, apply_log10=None):
    """
    Apply clip intervals to matching columns.

    Modifies the dataframe in place. Handles NaN default values for missing telescopes
    by preserving them throughout clipping and log10 transformation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to apply clipping to (modified in place).
    n_tel : int, optional
        Number of telescopes. If provided, applies to per-telescope columns (var_0, var_1, etc.).
    apply_log10 : list, optional
        List of variable base names to apply log10 transformation after clipping.
    """
    if apply_log10 is None:
        apply_log10 = []

    clip_intervals = features.clip_intervals()

    for var_base, (vmin, vmax) in clip_intervals.items():
        if n_tel is not None:
            for i in range(n_tel):
                col_name = f"{var_base}_{i}"
                if col_name in df.columns:
                    mask_valid = df[col_name].notna()
                    df.loc[mask_valid, col_name] = df.loc[mask_valid, col_name].clip(vmin, vmax)
                    if var_base in apply_log10:
                        mask_to_log = mask_valid & (df[col_name] > 0)
                        df.loc[mask_to_log, col_name] = np.log10(df.loc[mask_to_log, col_name])
        else:
            if var_base in df.columns:
                mask_valid = df[var_base].notna()
                df.loc[mask_valid, var_base] = df.loc[mask_valid, var_base].clip(vmin, vmax)
                if var_base in apply_log10:
                    mask_to_log = mask_valid & (df[var_base] > 0)
                    df.loc[mask_to_log, var_base] = np.log10(df.loc[mask_to_log, var_base])


def flatten_telescope_variables(n_tel, flat_features, index, tel_config=None):
    """Generate dataframe for telescope variables flattened for all telescopes.

    Creates features for all telescope IDs, using NaN as default value for missing data.

    Parameters
    ----------
    n_tel : int
        Maximum telescope index (for backward compatibility).
    flat_features : dict
        Dictionary of flattened feature arrays.
    index : pandas.Index
        DataFrame index.
    tel_config : dict, optional
        Telescope configuration with 'max_tel_id' key.
    """
    df_flat = pd.DataFrame(flat_features, index=index)
    df_flat = df_flat.astype(np.float32)

    # Determine max telescope ID from config or use n_tel
    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    new_cols = {}
    for i in range(max_tel_id + 1):  # Iterate over all possible telescopes
        if f"Disp_T_{i}" in df_flat:
            new_cols[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
            new_cols[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        if f"loss_{i}" in df_flat and f"dist_{i}" in df_flat:
            new_cols[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
            new_cols[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        if f"width_{i}" in df_flat and f"length_{i}" in df_flat:
            new_cols[f"size_dist2_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)
            new_cols[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)

    df_flat = pd.concat([df_flat, pd.DataFrame(new_cols, index=index)], axis=1)

    apply_clip_intervals(
        df_flat, n_tel=max_tel_id + 1, apply_log10=["size", "E", "ES", "size_dist2"]
    )

    for i in range(max_tel_id + 1):  # Iterate over all possible telescope indices
        if f"cen_x_{i}" in df_flat and f"fpointing_dx_{i}" in df_flat:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
        if f"cen_y_{i}" in df_flat and f"fpointing_dy_{i}" in df_flat:
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]
        df_flat = df_flat.drop(columns=[f"fpointing_dx_{i}", f"fpointing_dy_{i}"], errors="ignore")

    return df_flat


def _calculate_array_footprint(tel_config, elev_rad, azim_rad, tel_list_matrix):
    """
    Calculate array footprint area using convex hull of active telescope positions per event.

    Telescope positions are transformed to shower-centered coordinates in the shower plane
    before computing the footprint convex hull.

    Parameters
    ----------
    tel_config : dict
        Telescope configuration with 'tel_x', 'tel_y', and 'tel_ids' arrays.
    elev_rad : numpy.ndarray
        Elevation angles in radians, shape (n_evt,).
    azim_rad : numpy.ndarray
        Azimuth angles in radians, shape (n_evt,).
    tel_list_matrix : numpy.ndarray
        Matrix of telescope IDs participating in each event, shape (n_evt, max_tel).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_evt,) with footprint area for each event based on active telescopes.
    """
    n_evt = len(elev_rad)
    footprints = np.zeros(n_evt, dtype=np.float32)

    # Pre-map all telescope positions to a dense array aligned with tel_list_matrix IDs
    max_id = int(np.nanmax(tel_list_matrix)) if np.any(~np.isnan(tel_list_matrix)) else 0
    lookup_x = np.zeros(max_id + 1)
    lookup_y = np.zeros(max_id + 1)
    for tid, tx, ty in zip(tel_config["tel_ids"], tel_config["tel_x"], tel_config["tel_y"]):
        lookup_x[int(tid)] = tx
        lookup_y[int(tid)] = ty

    # 1. Vectorized Transformation (Do this for ALL events at once)
    sin_elev = np.sin(elev_rad)
    cos_azim = np.cos(azim_rad)
    sin_azim = np.sin(azim_rad)

    # 2. Iterate only for the ConvexHull
    for i in range(n_evt):
        tids = tel_list_matrix[i]
        tids = tids[~np.isnan(tids)].astype(int)

        if len(tids) < 3:
            continue

        # Fast indexing
        tx = lookup_x[tids]
        ty = lookup_y[tids]

        # Apply the rotation/projection for this specific event's geometry
        # Standard Shower-Plane (Tilted) transformation:
        xs = -sin_azim[i] * tx + cos_azim[i] * ty
        ys = (cos_azim[i] * tx + sin_azim[i] * ty) * sin_elev[i]

        try:
            points = np.column_stack([xs, ys])
            footprints[i] = ConvexHull(points).volume
        except QhullError:
            # This happens if telescopes are collinear (all in a line)
            # or if the points are too close together for the algorithm
            _logger.debug(f"Degenerate geometry for event {i}: telescopes are collinear.")
            footprints[i] = 0.0

        except ValueError as e:
            # This catches shape mismatches or empty arrays that slipped through
            _logger.error(f"Value error in ConvexHull for event {i}: {e}")
            footprints[i] = 0.0

    return footprints


def extra_columns(df, analysis_type, training, index, tel_config=None):
    """Add extra columns required for analysis type."""
    if analysis_type == "stereo_analysis":
        data = {
            "Xoff_weighted_bdt": _to_numpy_1d(df["Xoff"], np.float32),
            "Yoff_weighted_bdt": _to_numpy_1d(df["Yoff"], np.float32),
            "Xoff_intersect": _to_numpy_1d(df["Xoff_intersect"], np.float32),
            "Yoff_intersect": _to_numpy_1d(df["Yoff_intersect"], np.float32),
            "Diff_Xoff": (
                _to_numpy_1d(df["Xoff"], np.float32)
                - _to_numpy_1d(df["Xoff_intersect"], np.float32)
            ).astype(np.float32),
            "Diff_Yoff": (
                _to_numpy_1d(df["Yoff"], np.float32)
                - _to_numpy_1d(df["Yoff_intersect"], np.float32)
            ).astype(np.float32),
            "DispNImages": _to_numpy_1d(df["DispNImages"], np.int32),
            "Erec": _to_numpy_1d(df["Erec"], np.float32),
            "ErecS": _to_numpy_1d(df["ErecS"], np.float32),
            "EmissionHeight": _to_numpy_1d(df["EmissionHeight"], np.float32),
            "Geomagnetic_Angle": calculate_geomagnetic_angles(
                _to_numpy_1d(df["ArrayPointing_Azimuth"], np.float32),
                _to_numpy_1d(df["ArrayPointing_Elevation"], np.float32),
            ),
        }
        # Add array footprint if telescope configuration is available
        if tel_config is not None:
            elev_rad = np.radians(_to_numpy_1d(df["ArrayPointing_Elevation"], np.float32))
            azim_rad = np.radians(_to_numpy_1d(df["ArrayPointing_Azimuth"], np.float32))
            tel_list_matrix = _to_dense_array(df["DispTelList_T"])
            data["array_footprint"] = _calculate_array_footprint(
                tel_config, elev_rad, azim_rad, tel_list_matrix
            )
    elif "classification" in analysis_type:
        data = {
            "MSCW": _to_numpy_1d(df["MSCW"], np.float32),
            "MSCL": _to_numpy_1d(df["MSCL"], np.float32),
            "EChi2S": _to_numpy_1d(df["EChi2S"], np.float32),
            "EmissionHeight": _to_numpy_1d(df["EmissionHeight"], np.float32),
            "EmissionHeightChi2": _to_numpy_1d(df["EmissionHeightChi2"], np.float32),
        }
        if not training:
            data["ze_bin"] = _to_numpy_1d(df["ze_bin"], np.float32)

    df_extra = pd.DataFrame(data, index=index)
    apply_clip_intervals(
        df_extra,
        apply_log10=[
            "EChi2S",
            "EmissionHeightChi2",
            "Erec",
            "ErecS",
        ],
    )
    return df_extra


def zenith_in_bins(zenith_angles, bins):
    """Apply zenith binning based on zenith angles and given bin edges."""
    if isinstance(bins[0], dict):
        bins = [b["Ze_min"] for b in bins] + [bins[-1]["Ze_max"]]
    bins = np.asarray(bins, dtype=float)
    idx = np.clip(np.digitize(zenith_angles, bins) - 1, 0, len(bins) - 2)
    return idx.astype(np.int32)


def energy_in_bins(df_chunk, bins):
    """Apply energy binning based on reconstructed energy and given limits."""
    centers = np.array([(b["E_min"] + b["E_max"]) / 2 if b is not None else np.nan for b in bins])
    valid = (df_chunk["Erec"].to_numpy() > 0) & ~np.isnan(centers).all()
    e_bin = np.full(len(df_chunk), -1, dtype=np.int32)
    log_e = np.log10(df_chunk.loc[valid, "Erec"].to_numpy())
    distances = np.abs(log_e[:, None] - centers)
    distances[:, np.isnan(centers)] = np.inf

    e_bin[valid] = np.argmin(distances, axis=1)
    df_chunk["e_bin"] = e_bin
    return df_chunk["e_bin"]


def print_variable_statistics(df):
    """
    Print min, max, mean, and RMS for each variable in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing variables loaded using branch_list.
    """
    for col in df.columns:
        data = df[col].dropna().to_numpy()
        if data.size == 0:
            print(f"{col}: No data")
            continue
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        rms_val = np.sqrt(np.mean(np.square(data)))
        _logger.info(
            f"{col:25s} min: {min_val:10.4g}  max: {max_val:10.4g}  "
            f"mean: {mean_val:10.4g}  rms: {rms_val:10.4g}"
        )
