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

from eventdisplay_ml import features as features_module
from eventdisplay_ml import utils
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
        - 'mirror_area': Array of mirror areas for each telescope
        - 'tel_x': Array of telescope X positions
        - 'tel_y': Array of telescope Y positions
        - 'max_tel_id': Maximum telescope ID
        - 'tel_types': Dictionary mapping mirror area to list of telescope IDs
    """
    telconfig_tree = root_file["telconfig"]
    telconfig_data = telconfig_tree.arrays(
        ["NTel", "TelID", "MirrorArea", "TelX", "TelY"], library="np"
    )

    n_tel = int(telconfig_data["NTel"][0])
    tel_ids = telconfig_data["TelID"]
    # Keep array of mirror areas; avoid shadowing this name later
    mirror_area_arr = telconfig_data["MirrorArea"]
    tel_x = telconfig_data["TelX"]
    tel_y = telconfig_data["TelY"]
    max_tel_id = int(np.max(tel_ids))

    # Group telescopes by mirror area (telescope type)
    tel_types = {}
    for tel_id, area in zip(tel_ids, mirror_area_arr):
        key = round(area, 2)
        if key not in tel_types:
            tel_types[key] = []
        tel_types[key].append(int(tel_id))
    _logger.info(f"Telescope configuration: {n_tel} telescopes, max TelID: {max_tel_id}")
    _logger.info(f"Telescope types by mirror area: {tel_types}")

    return {
        "n_tel": n_tel,
        "tel_ids": tel_ids,
        # Provide both singular and plural keys for compatibility
        "mirror_area": mirror_area_arr,
        "mirror_areas": mirror_area_arr,
        "tel_x": tel_x,
        "tel_y": tel_y,
        "max_tel_id": max_tel_id,
        "tel_types": tel_types,
    }


def _resolve_branch_aliases(tree, branch_list):
    """
    Resolve branch name aliases (e.g. R_core vs R) and drop missing optional branches.

    The 'data' tree in the mscw files differ slightly between CTAO and VERITAS
    Eventdisplay versions:

    - 'R_core': VERITAS-style fixed telescope ID indexing
    - 'R': CTAO-style variable-length ImgSel_list indexing

    Parameters
    ----------
    tree : uproot tree
        Uproot tree containing the data branches.
    branch_list : list of str
        List of desired branch names.

    Returns
    -------
    tuple
        (resolved_branch_list, rename_map)
        - resolved_branch_list: List of actual branch names to read.
        - rename_map: Dict mapping old names to new names for renaming after reading.
    """
    keys = set(tree.keys())
    resolved = []
    rename = {}

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
        "mirror_area",
        "tel_rel_x",
        "tel_rel_y",
        "tel_active",
    }
    resolved = [b for b in resolved if b not in synthesized]

    # Drop missing optional branches
    optional = {"fpointing_dx", "fpointing_dy", "E", "Erec", "ErecS", "nlowgain"}
    final = [b for b in resolved if b not in optional or b in keys]

    return final, rename


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


def _rename_fields(arr, rename_map):
    """Rename record fields present in an Awkward Array."""
    for old, new in (rename_map or {}).items():
        fields = ak.fields(arr)
        if old in fields and old != new:
            arr = ak.with_field(arr, arr[old], new)
            arr = ak.without_field(arr, old)
    return arr


def _make_mirror_area_columns(tel_config, max_tel_id, n_evt, sort_indices=None):
    """Build constant mirror area columns from tel_config with optional sorting."""
    base = np.full((n_evt, max_tel_id + 1), DEFAULT_FILL_VALUE, dtype=np.float32)
    # Accept both 'mirror_area' (singular) and 'mirror_areas' (plural) for compatibility
    mirror_areas = tel_config.get("mirror_area")
    if mirror_areas is None:
        mirror_areas = tel_config.get("mirror_areas")
    if mirror_areas is None:
        raise KeyError("tel_config must provide 'mirror_area' or 'mirror_areas' array")
    tel_id_to_mirror = dict(zip(tel_config["tel_ids"], mirror_areas))

    for tel_idx, mirror_val in tel_id_to_mirror.items():
        if tel_idx <= max_tel_id:
            mirror_val = float(mirror_val) if mirror_val != 0.0 else 100.0
            base[:, tel_idx] = mirror_val

    if sort_indices is not None:
        base = base[np.arange(n_evt)[:, np.newaxis], sort_indices]

    return {f"mirror_area_{i}": base[:, i] for i in range(max_tel_id + 1)}


def _make_tel_active_columns(tel_list_matrix, max_tel_id, n_evt, sort_indices=None):
    """Build binary telescope active columns, optionally sorting."""
    columns = {}
    active_matrix = np.zeros((n_evt, max_tel_id + 1), dtype=np.float32)
    row_indices, col_indices = np.where(~np.isnan(tel_list_matrix))
    tel_ids = tel_list_matrix[row_indices, col_indices].astype(int)
    valid_mask = tel_ids <= max_tel_id
    active_matrix[row_indices[valid_mask], tel_ids[valid_mask]] = 1.0

    if sort_indices is not None:
        active_matrix = active_matrix[np.arange(n_evt)[:, np.newaxis], sort_indices]

    for tel_idx in range(max_tel_id + 1):
        columns[f"tel_active_{tel_idx}"] = active_matrix[:, tel_idx]
    return columns


def _ground_to_shower_coords(x, y, sin_azim, cos_azim, sin_elev):
    """Transform ground coordinates to shower-plane coordinates.

    Rotates around vertical axis by azimuth, then projects onto shower plane.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinate(s), shape (...,).
    y : numpy.ndarray
        Y coordinate(s), shape (...,).
    sin_azim : numpy.ndarray
        Sine of azimuth angle, shape (...,).
    cos_azim : numpy.ndarray
        Cosine of azimuth angle, shape (...,).
    sin_elev : numpy.ndarray
        Sine of elevation angle, shape (...,).

    Returns
    -------
    shower_x : numpy.ndarray
        Shower-plane X coordinate.
    shower_y : numpy.ndarray
        Shower-plane Y coordinate scaled by sin(elevation).

    Notes
    -----
    Original formula (azimuth definition of Eventdisplay is relevant)

        shower_x = -sin_azim * x + cos_azim * y
        shower_y = sin_elev * (cos_azim * x + sin_azim * y)
        return shower_x, shower_y
    """
    s_az = -sin_azim
    c_az = -cos_azim

    shower_x = c_az * x + s_az * y
    shower_y = sin_elev * (-s_az * x + c_az * y)

    return shower_x, shower_y


def _make_relative_coord_columns(
    var,
    tel_config,
    max_tel_id,
    n_evt,
    core_x,
    core_y,
    elev_rad,
    azim_rad,
    sort_indices=None,
):
    """Build relative/shower coordinate columns for a single synthetic variable."""
    columns = {}

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
    else:
        results = np.full((max_tel_id + 1, n_evt), DEFAULT_FILL_VALUE)

    # results shape: (max_tel_id + 1, n_evt) -> transpose to (n_evt, max_tel_id + 1)
    event_matrix = results.T

    # Apply validity mask and default fill
    valid_mask = np.isfinite(core_x) & np.isfinite(core_y)
    event_matrix = np.where(
        valid_mask[:, np.newaxis] & ~np.isnan(event_matrix), event_matrix, DEFAULT_FILL_VALUE
    )

    # Reorder if needed
    if sort_indices is not None:
        event_matrix = event_matrix[np.arange(n_evt)[:, np.newaxis], sort_indices]

    for tel_idx in range(max_tel_id + 1):
        columns[f"{var}_{tel_idx}"] = event_matrix[:, tel_idx].astype(np.float32)

    return columns


def _normalize_telescope_variable_to_tel_id_space(data, index_list, max_tel_id, n_evt):
    """Remap telescope variable from any index list to telescope-ID space.

    Takes data indexed by an arbitrary index list (DispTelList_T, ImgSel_list, or fixed positions)
    and remaps it to telescope-ID-indexed space (column i = telescope ID i).

    Modes:

    - VERITAS (R_core): Fixed indexing, index_list=None
    - CTAO (ImgSel_list): Variable indexing, index_list=ImgSel_list or DispTelList_T

    Parameters
    ----------
    data : numpy.ndarray
        Data array, shape (n_evt, n_active) from variable-length or (n_evt, n_tel) from fixed.
    index_list : numpy.ndarray or None
        Index remapping list, shape (n_evt, n_active). Each entry is a telescope ID or position.
        If None, data is already in telescope-ID space (VERITAS R_core fixed indexing).
    max_tel_id : int
        Maximum telescope ID.
    n_evt : int
        Number of events.

    Returns
    -------
    numpy.ndarray
        Data remapped to telescope-ID space, shape (n_evt, max_tel_id + 1).
    """
    if index_list is None:
        # Already in telescope-ID space (VERITAS R_core fixed indexing)
        full_matrix = np.full((n_evt, max_tel_id + 1), DEFAULT_FILL_VALUE, dtype=np.float32)
        col_cap = min(data.shape[1], max_tel_id + 1)
        full_matrix[:, :col_cap] = data[:, :col_cap]
        return full_matrix

    # Variable-length indexed (CTAO ImgSel_list or DispTelList_T mode)
    full_matrix = np.full((n_evt, max_tel_id + 1), DEFAULT_FILL_VALUE, dtype=np.float32)
    row_indices, col_indices = np.where(~np.isnan(index_list))
    tel_ids = index_list[row_indices, col_indices].astype(int)
    valid_mask = tel_ids <= max_tel_id
    full_matrix[row_indices[valid_mask], tel_ids[valid_mask]] = data[
        row_indices[valid_mask], col_indices[valid_mask]
    ]
    return full_matrix


def _clip_size_array(size_array):
    """Clip size array to specified min/max values."""
    vmin, vmax = features_module.clip_intervals().get("size", (None, None))
    clipped = size_array.copy()
    mask = ~np.isnan(clipped)
    if vmin is not None:
        clipped[mask] = np.maximum(clipped[mask], vmin)
        mask = ~np.isnan(clipped)
    if vmax is not None:
        clipped[mask] = np.minimum(clipped[mask], vmax)
    return clipped


def flatten_telescope_data_vectorized(
    df, n_tel, features, analysis_type, training=True, tel_config=None, observatory="veritas"
):
    """
    Vectorized flattening of telescope array columns.

    Converts per-telescope arrays into individual feature columns sorted by mirror area and
    size.

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
    """
    flat_features = {}
    tel_list_matrix = _to_dense_array(df["DispTelList_T"])
    n_evt = len(df)
    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    # Index remapping mode for CTAO-style variable-length indexing
    index_list_for_remapping = None  # None = telescope-ID-indexed (VERITAS R_core mode)
    if observatory.lower() == "veritas":
        _logger.info("Detected VERITAS R_core fixed telescope-ID indexing mode.")
    else:
        _logger.info("Detected CTAO ImgSel_list variable-length indexing mode.")
        index_list_for_remapping = _to_dense_array(df["ImgSel_list"])

    active_mask = np.zeros((n_evt, max_tel_id + 1), dtype=bool)
    row_indices, col_indices = np.where(~np.isnan(tel_list_matrix))
    tel_ids = tel_list_matrix[row_indices, col_indices].astype(int)
    valid_tel_mask = tel_ids <= max_tel_id
    active_mask[row_indices[valid_tel_mask], tel_ids[valid_tel_mask]] = True

    # Pre-load and normalize size to telescope-ID space for sorting
    size_data = _normalize_telescope_variable_to_tel_id_space(
        _to_dense_array(df["size"]), index_list_for_remapping, max_tel_id, n_evt
    )
    size_data = _clip_size_array(size_data)

    core_x, core_y = _get_core_arrays(df)

    # Sorting by mirror area (desc; proxy for telescope type), then size (desc)
    sort_indices = _compute_size_area_sort_indices(size_data, active_mask, tel_config, max_tel_id)

    for var in features:
        if var == "mirror_area" and tel_config:
            flat_features.update(
                _make_mirror_area_columns(tel_config, max_tel_id, n_evt, sort_indices)
            )
            continue

        if var == "tel_active":
            _logger.info(f"Computing synthetic feature: {var}")
            flat_features.update(
                _make_tel_active_columns(tel_list_matrix, max_tel_id, n_evt, sort_indices)
            )
            continue

        if var in ("tel_rel_x", "tel_rel_y") and tel_config:
            _logger.info(f"Computing synthetic feature: {var}")
            flat_features.update(
                _make_relative_coord_columns(
                    var,
                    tel_config,
                    max_tel_id,
                    n_evt,
                    core_x,
                    core_y,
                    np.radians(_to_numpy_1d(df["ArrayPointing_Elevation"], np.float32)),
                    np.radians(_to_numpy_1d(df["ArrayPointing_Azimuth"], np.float32)),
                    sort_indices,
                )
            )
            continue

        data = _to_dense_array(df[var]) if _has_field(df, var) else np.full((n_evt, n_tel), np.nan)

        # Disp_* variables always use DispTelList_T, regardless of mode
        if var.startswith("Disp") and "_T" in var:
            data_normalized = _normalize_telescope_variable_to_tel_id_space(
                data, tel_list_matrix, max_tel_id, n_evt
            )
        else:
            # All other variables use the mode-dependent index
            # (None for VERITAS R_core fixed, ImgSel_list for CTAO variable)
            data_normalized = _normalize_telescope_variable_to_tel_id_space(
                data, index_list_for_remapping, max_tel_id, n_evt
            )

        # All variables are now in telescope-ID space; apply sorting and flatten uniformly
        data_normalized = data_normalized[np.arange(n_evt)[:, np.newaxis], sort_indices]

        for tel_idx in range(max_tel_id + 1):
            flat_features[f"{var}_{tel_idx}"] = data_normalized[:, tel_idx]

    index = _get_index(df, n_evt)
    df_flat = flatten_telescope_variables(n_tel, flat_features, index, tel_config, analysis_type)
    return pd.concat(
        [df_flat, extra_columns(df, analysis_type, training, index, tel_config, observatory)],
        axis=1,
    )


def _to_dense_array(col):
    """
    Convert a column of variable-length telescope data to a dense 2D numpy array.

    Handles uproot's awkward-style variable-length arrays from ROOT files
    by converting to plain Python lists first to avoid per-element iteration overhead.

    - right-pad each event to equal length with NaN
    - return as 2D numpy array of shape (n_events, max_telescopes)

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
    except (ValueError, TypeError) as exc:
        raise ValueError("Input column must be convertible to an Awkward Array.") from exc


def _get_core_arrays(df):
    """Extract core position arrays from DataFrame."""
    # Make copies to ensure arrays are writable (some sources return read-only views)
    core_x = _to_numpy_1d(df["Xcore"], np.float32).copy()
    core_y = _to_numpy_1d(df["Ycore"], np.float32).copy()
    # Filter out sentinel values and apply physical bounds
    # shower cores beyond +-10 km are cut
    core_x[(core_x <= -90000) | (np.abs(core_x) > 10000)] = np.nan
    core_y[(core_y <= -90000) | (np.abs(core_y) > 10000)] = np.nan
    return core_x, core_y


def _compute_size_area_sort_indices(size_data, active_mask, tel_config, max_tel_id):
    """Compute sorting indices: mirror area (desc) then size (desc).

    Missing telescopes (NaN size or no mirror area) are sorted to the end.
    """
    n_evt = active_mask.shape[0]

    # Map mirror areas to a dense lookup by telescope ID
    mirror_lookup = np.full(max_tel_id + 1, np.nan, dtype=np.float32)
    # Accept both legacy ('mirror_area') and plural ('mirror_areas') keys
    mirror_areas = tel_config.get("mirror_area")
    if mirror_areas is None:
        mirror_areas = tel_config.get("mirror_areas")
    if mirror_areas is None:
        raise KeyError("tel_config must provide 'mirror_area' (array) or 'mirror_areas'")
    for tid, area in zip(tel_config["tel_ids"], mirror_areas):
        if tid <= max_tel_id:
            mirror_lookup[int(tid)] = float(area)

    # size_data is already in tel_id space (shape: n_evt x (max_tel_id + 1))
    sizes = size_data

    # Sort per-event: primary = mirror area (desc), secondary = size (desc)
    # Area has highest priority ALWAYS, even if size is NaN.
    # NaN mirror areas go to the very end; within equal area groups,
    # valid sizes come before NaN sizes, and then by size descending.
    sort_indices = np.zeros((n_evt, max_tel_id + 1), dtype=int)

    for evt_idx in range(n_evt):
        tel_entries = []
        for tel_idx in range(max_tel_id + 1):
            area = mirror_lookup[tel_idx]
            size_val = sizes[evt_idx, tel_idx]
            # Build sort key:
            #   1) valid area first (0), NaN area last (1)
            #   2) area descending via negative value
            #   3) within same area: valid size first (0), NaN last (1)
            #   4) size descending via negative value
            area_valid = 0 if not np.isnan(area) else 1
            size_valid = 0 if not np.isnan(size_val) else 1
            area_key = -area if area_valid == 0 else 0.0
            size_key = -size_val if size_valid == 0 else 0.0
            tel_entries.append((tel_idx, area_valid, area_key, size_valid, size_key))

        tel_entries.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
        sort_indices[evt_idx] = np.array([t[0] for t in tel_entries])

    return sort_indices


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


def flatten_feature_data(
    group_df, ntel, analysis_type, training, tel_config=None, observatory="veritas"
):
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
        features_module.telescope_features(analysis_type),
        analysis_type=analysis_type,
        training=training,
        tel_config=tel_config,
        observatory=observatory,
    )
    max_tel_id = tel_config["max_tel_id"] if tel_config else ntel - 1
    excluded_columns = set(features_module.target_features(analysis_type)) | set(
        features_module.excluded_features(analysis_type, max_tel_id + 1)
    )
    return df_flat.drop(columns=excluded_columns, errors="ignore")


def load_training_data(model_configs, file_list, analysis_type):
    """
    Load and flatten training data from the mscw file.

    Processes all events regardless of telescope multiplicity. Features are created
    for all telescopes with default value DEFAULT_FILL_VALUE for missing telescopes.
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

    branch_list = features_module.features(analysis_type, training=True)
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

                if tel_config is None:
                    tel_config = read_telescope_config(root_file)
                    model_configs["tel_config"] = tel_config

                _logger.info(f"Processing file: {f} (file {file_idx}/{total_files})")
                tree = root_file["data"]
                resolved_branch_list, rename_map = _resolve_branch_aliases(tree, branch_list)
                df = tree.arrays(
                    resolved_branch_list,
                    cut=model_configs.get("pre_cuts", None),
                    library="ak",
                    decompression_executor=executor,
                )
                if rename_map:
                    rename_present = {k: v for k, v in rename_map.items() if k in df.fields}
                    if rename_present:
                        df = _rename_fields(df, rename_present)
                df = _ensure_fpointing_fields(df)
                if len(df) == 0:
                    continue

                if max_events_per_file and len(df) > max_events_per_file:
                    rng = np.random.default_rng(random_state)
                    indices = rng.choice(len(df), max_events_per_file, replace=False)
                    df = df[indices]

                n_before = tree.num_entries
                _logger.info(
                    f"Number of events before / after event cut: {n_before} / {len(df)}"
                    f" (fraction retained: {len(df) / n_before:.4f})"
                )

                df_flat = flatten_telescope_data_vectorized(
                    df,
                    tel_config["max_tel_id"] + 1,
                    features_module.telescope_features(analysis_type),
                    analysis_type,
                    training=True,
                    tel_config=tel_config,
                    observatory=model_configs.get("observatory", "veritas"),
                )
                if analysis_type == "stereo_analysis":
                    df_flat["MCxoff"] = _to_numpy_1d(df["MCxoff"], np.float32)
                    df_flat["MCyoff"] = _to_numpy_1d(df["MCyoff"], np.float32)
                    df_flat["MCe0"] = np.log10(_to_numpy_1d(df["MCe0"], np.float32))
                elif analysis_type == "classification":
                    df_flat["ze_bin"] = zenith_in_bins(
                        90.0 - _to_numpy_1d(df["ArrayPointing_Elevation"], np.float32),
                        model_configs.get("zenith_bins_deg", []),
                    )

                dfs.append(df_flat)

                del df
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

    clip_intervals = features_module.clip_intervals()

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


def flatten_telescope_variables(n_tel, flat_features, index, tel_config=None, analysis_type=None):
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
    analysis_type : str, optional
        Type of analysis, e.g. "classification" or "stereo_analysis".
    """
    df_flat = pd.DataFrame(flat_features, index=index)
    df_flat = df_flat.astype(np.float32)

    # Determine max telescope ID from config or use n_tel
    max_tel_id = tel_config["max_tel_id"] if tel_config else (n_tel - 1)

    keep_size_vars = analysis_type == "stereo_analysis"
    if not keep_size_vars:
        _logger.info(f"Dropping 'size'-related variables for {analysis_type} analysis.")

    new_cols = {}
    for i in range(max_tel_id + 1):  # Iterate over all possible telescopes
        if f"Disp_T_{i}" in df_flat:
            new_cols[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
            new_cols[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        if f"loss_{i}" in df_flat and f"dist_{i}" in df_flat:
            new_cols[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
            new_cols[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        if f"size_{i}" in df_flat and f"dist_{i}" in df_flat and keep_size_vars:
            new_cols[f"size_dist2_{i}"] = df_flat[f"size_{i}"] / (df_flat[f"dist_{i}"] ** 2 + 1e-6)
        if f"width_{i}" in df_flat and f"length_{i}" in df_flat:
            new_cols[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)

    df_flat = pd.concat([df_flat, pd.DataFrame(new_cols, index=index)], axis=1)

    # inspect ordering and magnitudes before clipping/log10
    size_cols = [c for c in df_flat.columns if c.startswith("size_")][: max_tel_id + 1]
    area_cols = [c for c in df_flat.columns if c.startswith("mirror_area_")][: max_tel_id + 1]
    disp_cols = [c for c in df_flat.columns if c.startswith("Disp_T_")][: max_tel_id + 1]
    preview = df_flat[size_cols + area_cols + disp_cols].head(20)
    _logger.info(
        "Sorted telescope sizes (pre-clip/log10), first 20 events: \n"
        f"{preview.to_string(index=False)}"
    )

    apply_clip_intervals(
        df_flat,
        n_tel=max_tel_id + 1,
        apply_log10=["size", "ntubes", "nlowgain", "E", "ES", "size_dist2"],
    )

    for i in range(max_tel_id + 1):  # Iterate over all possible telescope indices
        if f"cen_x_{i}" in df_flat and f"fpointing_dx_{i}" in df_flat:
            df_flat[f"cen_x_{i}"] = df_flat[f"cen_x_{i}"] + df_flat[f"fpointing_dx_{i}"]
        if f"cen_y_{i}" in df_flat and f"fpointing_dy_{i}" in df_flat:
            df_flat[f"cen_y_{i}"] = df_flat[f"cen_y_{i}"] + df_flat[f"fpointing_dy_{i}"]
        df_flat = df_flat.drop(columns=[f"fpointing_dx_{i}", f"fpointing_dy_{i}"], errors="ignore")
        if not keep_size_vars:
            df_flat = df_flat.drop(columns=[f"size_{i}"], errors="ignore")

    return df_flat


def _calculate_array_footprint(tel_config, tel_list_matrix):
    """
    Calculate array footprint area using convex hull of active telescope positions per event.

    Calculation in ground coordinates, not in shower coordinates.

    For 2-telescope events, footprint is the distance between the two telescopes.

    Parameters
    ----------
    tel_config : dict
        Telescope configuration with 'tel_x', 'tel_y', and 'tel_ids' arrays.
    tel_list_matrix : numpy.ndarray
        Matrix of telescope IDs participating in each event, shape (n_evt, max_tel).

    Returns
    -------
    numpy.ndarray
        Array of shape (n_evt,) with footprint area for each event based on active telescopes.
    """
    n_evt = len(tel_list_matrix)
    footprints = np.full(n_evt, -1.0, dtype=np.float32)

    # Pre-map all telescope positions to a dense array aligned with tel_list_matrix IDs
    max_id = int(np.nanmax(tel_list_matrix)) if np.any(~np.isnan(tel_list_matrix)) else 0
    lookup_x = np.zeros(max_id + 1)
    lookup_y = np.zeros(max_id + 1)
    for tid, tx, ty in zip(tel_config["tel_ids"], tel_config["tel_x"], tel_config["tel_y"]):
        lookup_x[int(tid)] = tx
        lookup_y[int(tid)] = ty

    #  Iterate only for the ConvexHull
    for i in range(n_evt):
        tids = tel_list_matrix[i]
        tids = tids[~np.isnan(tids)].astype(int)

        if len(tids) == 2:
            tx1 = lookup_x[tids[0]]
            ty1 = lookup_y[tids[0]]
            tx2 = lookup_x[tids[1]]
            ty2 = lookup_y[tids[1]]
            footprints[i] = np.hypot(tx2 - tx1, ty2 - ty1)
            continue

        if len(tids) < 2:
            continue

        # Fast indexing
        xs = lookup_x[tids]
        ys = lookup_y[tids]

        try:
            points = np.column_stack([xs, ys])
            footprints[i] = ConvexHull(points).volume
        except QhullError:
            # This happens if telescopes are collinear (all in a line)
            # or if the points are too close together for the algorithm
            _logger.debug(f"Degenerate geometry for event {i}: telescopes are collinear.")
            footprints[i] = -1.0

        except ValueError as e:
            # This catches shape mismatches or empty arrays that slipped through
            _logger.error(f"Value error in ConvexHull for event {i}: {e}")
            footprints[i] = -1.0

    return footprints


def extra_columns(df, analysis_type, training, index, tel_config=None, observatory="veritas"):
    """Add extra columns required for analysis type."""
    if analysis_type == "stereo_analysis":
        n = len(index)
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
            # These may be absent in some datasets; if missing, fill with NaN
            "Erec": (
                _to_numpy_1d(df["Erec"], np.float32)
                if _has_field(df, "Erec")
                else np.full(n, DEFAULT_FILL_VALUE, dtype=np.float32)
            ),
            "ErecS": (
                _to_numpy_1d(df["ErecS"], np.float32)
                if _has_field(df, "ErecS")
                else np.full(n, DEFAULT_FILL_VALUE, dtype=np.float32)
            ),
            "EmissionHeight": _to_numpy_1d(df["EmissionHeight"], np.float32),
            "Geomagnetic_Angle": calculate_geomagnetic_angles(
                _to_numpy_1d(df["ArrayPointing_Azimuth"], np.float32),
                _to_numpy_1d(df["ArrayPointing_Elevation"], np.float32),
                observatory=observatory,
            ),
        }
        # Add array footprint if telescope configuration is available
        if tel_config is not None:
            tel_list_matrix = _to_dense_array(df["DispTelList_T"])
            data["array_footprint"] = _calculate_array_footprint(tel_config, tel_list_matrix)
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
