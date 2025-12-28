"""Utility functions for Eventdisplay-ML."""

import json
import logging

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


def load_model_parameters(model_parameters):
    """Load model parameters from a JSON file."""
    try:
        with open(model_parameters) as f:
            return json.load(f)
    except (FileNotFoundError, TypeError) as exc:
        raise FileNotFoundError(f"Model parameters file not found: {model_parameters}") from exc


def load_energy_range(model_parameters, energy_bin_number):
    """Load the log10(Erec/TeV) range for a given energy bin from model parameters."""
    par = load_model_parameters(model_parameters)
    try:
        e = par["energy_bins_log10_tev"][energy_bin_number]
        return 10 ** e["E_min"], 10 ** e["E_max"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Invalid energy bin number {energy_bin_number} for model parameters."
        ) from exc
