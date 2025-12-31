"""
Train XGBBoost models for gamma/hadron classification.

Uses image and stereo parameters to train classification BDTs to separate
gamma-ray events from hadronic background events.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import argparse
import logging

import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split

from eventdisplay_ml import hyper_parameters, utils
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import (
    evaluate_classification_model,
    evaluation_efficiency,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(
    df,
    n_tel,
    model_prefix,
    train_test_fraction,
    model_parameters,
    energy_bin_number,
    hyperparameter_config,
):
    """
    Train a single XGBoost model for gamma/hadron classification.

    Parameters
    ----------
    df : list of pd.DataFrame
        List containing signal and background DataFrames.
    n_tel : int
        Telescope multiplicity.
    model_prefix : str
        Directory to save the trained model.
    train_test_fraction : float
        Fraction of data to use for training.
    model_parameters : dict,
        Dictionary of model parameters.
    energy_bin_number : int
        Energy bin number (for naming the output model).
    hyperparameter_config : str, optional
        Path to JSON file with hyperparameter configuration, by default None.
    """
    if df[0].empty or df[1].empty:
        _logger.warning(f"Skip training for n_tel={n_tel} due to empty signal / background data.")
        return

    df[0]["label"] = 1
    df[1]["label"] = 0
    full_df = pd.concat([df[0], df[1]], ignore_index=True)
    x_data = full_df.drop(columns=["label"])
    _logger.info(f"Training features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    y_data = full_df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_test_fraction, random_state=None, stratify=y_data
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    configs = hyper_parameters.classification_hyperparameters(hyperparameter_config)

    for name, para in configs.items():
        _logger.info(f"Training with {name} for n_tel={n_tel}...")
        model = xgb.XGBClassifier(**para)
        model.fit(x_train, y_train)

        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)

        dump(
            {
                "model": model,
                "features": x_data.columns.tolist(),
                "hyperparameters": para,
                "efficiency": evaluation_efficiency(name, model, x_test, y_test),
                "parameters": model_parameters,
                "n_tel": n_tel,
                "energy_bin_number": energy_bin_number,
            },
            utils.output_file_name(model_prefix, name, n_tel, energy_bin_number),
        )


def main():
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description=("Train XGBoost models for gamma/hadron classification.")
    )
    parser.add_argument("--input_signal_file_list", help="List of input signal mscw ROOT files.")
    parser.add_argument(
        "--input_background_file_list", help="List of input background mscw ROOT files."
    )
    parser.add_argument(
        "--model_prefix",
        required=True,
        help=(
            "Path to directory for writing XGBoost classification models "
            "(without n_tel and energy bin suffix)."
        ),
    )
    parser.add_argument(
        "--hyperparameter_config",
        help="Path to JSON file with hyperparameter configuration.",
        default=None,
        type=str,
    )
    parser.add_argument("--ntel", type=int, help="Telescope multiplicity (2, 3, or 4).")
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

    args = parser.parse_args()

    _logger.info("--- XGBoost Classification Training ---")
    _logger.info(f"Telescope multiplicity: {args.ntel}")
    _logger.info(f"Model output prefix: {args.model_prefix}")
    _logger.info(f"Train vs test fraction: {args.train_test_fraction}")
    _logger.info(f"Max events: {args.max_events}")
    _logger.info(f"Energy bin {args.energy_bin_number}")

    model_parameters = utils.load_model_parameters(args.model_parameters, args.energy_bin_number)

    event_lists = [
        load_training_data(
            utils.read_input_file_list(file_list),
            args.ntel,
            args.max_events,
            analysis_type="classification",
            model_parameters=model_parameters,
        )
        for file_list in (args.input_signal_file_list, args.input_background_file_list)
    ]

    train(
        event_lists,
        args.ntel,
        args.model_prefix,
        args.train_test_fraction,
        model_parameters,
        args.energy_bin_number,
        args.hyperparameter_config,
    )
    _logger.info("XGBoost classification model trained successfully.")


if __name__ == "__main__":
    main()
