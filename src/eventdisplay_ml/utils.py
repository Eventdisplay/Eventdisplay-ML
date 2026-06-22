"""Utility functions for Eventdisplay-ML."""

import json
import logging
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import joblib

_logger = logging.getLogger(__name__)
_profile_start_time = None
_profile_last_time = None


def _max_rss_gb():
    """Return the process peak resident set size in GB."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return max_rss / 1024**3
    return max_rss / 1024**2


def _current_rss_gb():
    """Return current resident set size in GB when available."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    return _max_rss_gb()


def log_memory_checkpoint(label, df=None, enabled=False):
    """Log process RSS, peak RSS, timing, and optional dataframe memory for profiling large jobs."""
    if not enabled:
        return

    global _profile_last_time, _profile_start_time
    now = time.perf_counter()
    if _profile_start_time is None:
        _profile_start_time = now
    elapsed_s = now - _profile_start_time
    delta_s = 0.0 if _profile_last_time is None else now - _profile_last_time
    _profile_last_time = now

    msg = (
        f"Memory checkpoint [{label}]: "
        f"rss={_current_rss_gb():.2f} GB, max_rss={_max_rss_gb():.2f} GB, "
        f"elapsed={elapsed_s:.2f} s, delta={delta_s:.2f} s"
    )
    if df is not None:
        df_memory_gb = df.memory_usage(deep=True).sum() / 1024**3
        msg += f", dataframe={df_memory_gb:.2f} GB, shape={df.shape}"
    _logger.info(msg)


def resolve_joblib_path(path_or_prefix):
    """Resolve model path supporting .joblib.gz (preferred) and .joblib."""
    path = Path(path_or_prefix)
    path_str = str(path)

    if path_str.endswith(".joblib.gz"):
        candidates = [path, Path(path_str.removesuffix(".gz"))]
    elif path_str.endswith(".joblib"):
        candidates = [Path(f"{path_str}.gz"), path]
    else:
        candidates = [Path(f"{path_str}.joblib.gz"), Path(f"{path_str}.joblib"), path]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not resolve model file from '{path_or_prefix}'.")


def load_joblib(path):
    """Load a joblib payload while tolerating NumPy 2.5's legacy-pickle warning.

    Joblib restores persisted arrays by assigning their shape. NumPy 2.5
    deprecates that implementation detail, but the serialized data and restored
    array are unaffected. Keep the suppression local so other deprecations remain
    visible (and fatal in the test suite).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        return joblib.load(path)


def read_input_file_list(input_file_list):
    """
    Read a list of input files from a text file.

    Parameters
    ----------
    input_file_list : str
        Path to the text file containing the list of input files.

    Returns
    -------
    list of str
        List of input file paths.
    """
    try:
        with open(input_file_list) as f:
            input_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Error: Input file list not found: {input_file_list}") from exc

    if not input_files:
        raise ValueError(f"Error: No input files found in the list: {input_file_list}")

    _logger.info(f"Read {len(input_files)} input files from {input_file_list}")
    return input_files


def parse_image_selection(image_selection_str):
    """
    Parse the image_selection parameter.

    Parameters
    ----------
    image_selection_str : str
        Image selection parameter as a string. Can be either a
        bit-coded value (e.g., 14 = 0b1110 = telescopes 1,2,3) or a
        comma-separated indices (e.g., "1,2,3")

    Returns
    -------
    list[int] or None
        List of telescope indices.
    """
    if not image_selection_str:
        return None

    # Parse as comma-separated indices
    if "," in image_selection_str:
        try:
            indices = [int(x.strip()) for x in image_selection_str.split(",")]
            _logger.info(f"Image selection indices: {indices}")
            return indices
        except ValueError:
            pass

    # Parse as bit-coded value
    try:
        bit_value = int(image_selection_str)
        indices = [i for i in range(4) if (bit_value >> i) & 1]
        _logger.info(f"Image selection from bit-coded value {bit_value}: {indices}")
        return indices
    except ValueError:
        raise ValueError(
            f"Invalid image_selection format: {image_selection_str}. "
            "Use bit-coded value (e.g., 14) or comma-separated indices (e.g., '1,2,3')"
        )


def load_model_parameters(model_parameters, energy_bin_number=None):
    """
    Load model parameters from a JSON file.

    Reduce the energy bins to only the specified energy bin number if provided.
    """
    try:
        with open(model_parameters) as f:
            para = json.load(f)
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(f"Model parameters file not found: {model_parameters}") from exc

    if energy_bin_number is not None:
        try:
            para["energy_bins_log10_tev"] = para["energy_bins_log10_tev"][energy_bin_number]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Invalid energy bin number {energy_bin_number} for model parameters."
            ) from exc
    return para


def load_energy_range(model_parameters):
    """Load the log10(Erec/TeV) energy range from model parameters."""
    try:
        e = model_parameters["energy_bins_log10_tev"]
        return 10 ** e["E_min"], 10 ** e["E_max"]
    except (KeyError, IndexError) as exc:
        raise ValueError("Invalid or missing energy range in model parameters.") from exc


def output_file_name(model_prefix, n_tel=None, energy_bin_number=None):
    """Generate output filename for the trained model.

    Parameters
    ----------
    model_prefix : str or Path
        Base path for the model file.
    n_tel : int or None
        Number of telescopes. If None, uses 'all' to indicate model handles all multiplicities.
    energy_bin_number : int, optional
        Energy bin number for classification models.
    """
    model_prefix = Path(model_prefix)

    output_dir = model_prefix.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    filename = f"{model_prefix!s}"
    if n_tel is not None:
        filename = f"{model_prefix!s}_ntel{n_tel}"
    if energy_bin_number is not None:
        filename += f"_ebin{energy_bin_number}"
    filename += ".joblib.gz"
    _logger.info(f"Output filename: {filename}")
    return filename
