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


def load_energy_range(model_parameters, energy_bin_number=0):
    """Load the log10(Erec/TeV) range for a given energy bin from model parameters."""
    try:
        e = model_parameters["energy_bins_log10_tev"]
        return 10 ** e["E_min"], 10 ** e["E_max"]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"Invalid energy bin number {energy_bin_number} for model parameters."
        ) from exc


def output_file_name(model_prefix, n_tel, energy_bin_number=None):
    """Generate output filename for the trained model."""
    model_prefix = Path(model_prefix)

    output_dir = model_prefix.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    filename = f"{model_prefix}_ntel{n_tel}"
    if energy_bin_number is not None:
        filename += f"_ebin{energy_bin_number}"
    filename += ".joblib"
    _logger.info(f"Output filename: {filename}")
    return filename
