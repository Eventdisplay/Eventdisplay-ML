"""Configuration for XGBoost model training."""

import argparse
import logging

import numpy as np

from eventdisplay_ml import utils
from eventdisplay_ml.features import target_features
from eventdisplay_ml.hyper_parameters import (
    hyper_parameters,
    pre_cuts_classification,
    pre_cuts_regression,
)
from eventdisplay_ml.models import load_models

_logger = logging.getLogger(__name__)


def configure_training(analysis_type):
    """Configure model training based on command-line arguments."""
    parser = argparse.ArgumentParser(description=(f"Train XGBoost models for {analysis_type}."))

    if analysis_type == "stereo_analysis":
        parser.add_argument(
            "--input_file_list", help=f"List of input mscw files for {analysis_type}."
        )
    if analysis_type == "classification":
        parser.add_argument("--input_signal_file_list", help="List of input signal mscw files.")
        parser.add_argument(
            "--input_background_file_list", help="List of input background mscw files."
        )

    parser.add_argument(
        "--model_prefix",
        required=True,
        help=("Path to directory for writing XGBoost models (without n_tel / energy bin suffix)."),
    )
    parser.add_argument(
        "--hyperparameter_config",
        help="Path to JSON file with hyperparameter configuration.",
        default=None,
        type=str,
    )
    parser.add_argument("--n_tel", type=int, help="Telescope multiplicity (2, 3, or 4).")
    parser.add_argument(
        "--train_test_fraction",
        type=float,
        help="Fraction of data for training (e.g., 0.5).",
        default=0.5,
    )
    parser.add_argument(
        "--max_events",
        type=int,
        help="Maximum number of events to process across all files.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Random state for train/test split.",
        default=None,
    )

    if analysis_type == "classification":
        parser.add_argument(
            "--model_parameters",
            type=str,
            help=("Path to model parameter file (JSON) defining energy and zenith bins."),
        )
        parser.add_argument(
            "--energy_bin_number",
            type=int,
            help="Energy bin number for selection (optional).",
            default=0,
        )
    parser.add_argument(
        "--max_cores",
        type=int,
        help="Maximum number of CPU cores to use for training.",
        default=1,
    )

    model_configs = vars(parser.parse_args())

    _logger.info(f"--- XGBoost {analysis_type} training ---")
    _logger.info(f"Telescope multiplicity: {model_configs.get('n_tel')}")
    _logger.info(f"Model output prefix: {model_configs.get('model_prefix')}")
    _logger.info(f"Train vs test fraction: {model_configs['train_test_fraction']}")
    _logger.info(f"Max events: {model_configs['max_events']}")
    _logger.info(f"Max CPU cores: {model_configs['max_cores']}")

    model_configs["models"] = hyper_parameters(
        analysis_type, model_configs.get("hyperparameter_config")
    )
    model_configs["models"]["xgboost"]["hyper_parameters"]["n_jobs"] = model_configs["max_cores"]
    model_configs["targets"] = target_features(analysis_type)

    if analysis_type == "stereo_analysis":
        model_configs["pre_cuts"] = pre_cuts_regression(model_configs.get("n_tel"))
    elif analysis_type == "classification":
        _logger.info(f"Energy bin {model_configs['energy_bin_number']}")
        model_parameters = utils.load_model_parameters(
            model_configs["model_parameters"], model_configs["energy_bin_number"]
        )
        model_configs["pre_cuts"] = pre_cuts_classification(
            model_configs.get("n_tel"),
            e_min=np.power(10.0, model_parameters.get("energy_bins_log10_tev", []).get("E_min")),
            e_max=np.power(10.0, model_parameters.get("energy_bins_log10_tev", []).get("E_max")),
        )
        model_configs["energy_bins_log10_tev"] = model_parameters.get("energy_bins_log10_tev", [])
        model_configs["zenith_bins_deg"] = model_parameters.get("zenith_bins_deg", [])

    return model_configs


def configure_apply(analysis_type):
    """Configure model application based on command-line arguments."""
    parser = argparse.ArgumentParser(description=(f"Apply XGBoost models for {analysis_type}."))

    parser.add_argument(
        "--input_file",
        required=True,
        metavar="INPUT.root",
        help="Path to input mscw file",
    )
    parser.add_argument(
        "--model_prefix",
        required=True,
        metavar="MODEL_PREFIX",
        help=("Path to directory containing XGBoost models (without n_tel / energy bin suffix)."),
    )
    parser.add_argument(
        "--model_name",
        default="xgboost",
        help="Model name to load (default: xgboost)",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        metavar="OUTPUT.root",
        help="Output file path for predictions",
    )
    parser.add_argument(
        "--image_selection",
        type=str,
        default="15",
        help=(
            "Optional telescope selection. Can be bit-coded (e.g., 14 for telescopes 1,2,3) "
            "or comma-separated indices (e.g., '1,2,3'). "
            "Keeps events with all selected telescopes or 4-telescope events. "
            "Default is 15, which selects all 4 telescopes."
        ),
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=None,
        help="Maximum number of events to process (default: all events)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500000,
        help="Number of events to process per chunk (default: 500000)",
    )
    parser.add_argument(
        "--max_cores",
        type=int,
        help="Maximum number of CPU cores to use for training.",
        default=1,
    )

    model_configs = vars(parser.parse_args())

    _logger.info(f"--- XGBoost {analysis_type} evaluation ---")
    _logger.info(f"Input file: {model_configs.get('input_file')}")
    _logger.info(f"Model prefix: {model_configs.get('model_prefix')}")
    _logger.info(f"Output file: {model_configs.get('output_file')}")
    _logger.info(f"Image selection: {model_configs.get('image_selection')}")
    _logger.info(f"Max events: {model_configs.get('max_events')}")
    _logger.info(f"Max cores: {model_configs.get('max_cores')}")

    model_configs["models"], par = load_models(
        analysis_type, model_configs["model_prefix"], model_configs["model_name"]
    )
    model_configs["energy_bins_log10_tev"] = par.get("energy_bins_log10_tev", [])
    model_configs["zenith_bins_deg"] = par.get("zenith_bins_deg", [])

    return model_configs
