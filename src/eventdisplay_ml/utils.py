"""Utility functions for Eventdisplay-ML."""

import json
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)


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
    filename += ".joblib"
    _logger.info(f"Output filename: {filename}")
    return filename
