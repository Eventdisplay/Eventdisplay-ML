"""
Train XGBBoost models for gamma/hadron classification.

Uses image and stereo parameters to train classification BDTs to separate
gamma-ray events from hadronic background events.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split

from eventdisplay_ml import utils
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import evaluate_classification_model, write_efficiency_csv

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(
    signal_df,
    background_df,
    n_tel,
    model_prefix,
    train_test_fraction,
    model_parameters,
    energy_bin_number,
):
    """
    Train a single XGBoost model for gamma/hadron classification.

    Parameters
    ----------
    signal_df : Pandas DataFrame
        Pandas DataFrame with signal training data.
    background_df : Pandas DataFrame
        Pandas DataFrame with background training data.
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
    """
    if signal_df.empty or background_df.empty:
        _logger.warning(f"Skip training for n_tel={n_tel} due to empty signal / background data.")
        return

    model_prefix = Path(model_prefix)
    output_dir = model_prefix.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    signal_df["label"] = 1
    background_df["label"] = 0
    full_df = pd.concat([signal_df, background_df], ignore_index=True)
    x_data = full_df.drop(columns=["label"])
    _logger.info(f"Training features ({len(x_data.columns)}): {', '.join(x_data.columns)}")
    y_data = full_df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_test_fraction, random_state=42, stratify=y_data
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",  # TMP AUC ?
        "n_estimators": 100,  # TMP probably too low
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    configs = {"xgboost": xgb.XGBClassifier(**xgb_params)}
    for name, model in configs.items():
        _logger.info(f"Training with {name} for n_tel={n_tel}...")
        _logger.info(f"parameters: {xgb_params}")
        model.fit(x_train, y_train)

        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)

        output_filename = (
            Path(output_dir) / f"{model_prefix.name}_{name}_ntel{n_tel}_bin{energy_bin_number}"
        )
        efficiency = write_efficiency_csv(
            name,
            model,
            x_test,
            y_test,
            output_filename.with_suffix(".efficiency.csv"),
        )
        dump(
            {
                "model": model,
                "features": x_data.columns.tolist(),
                "hyperparameters": xgb_params,
                "efficiency": efficiency,
                "parameters": model_parameters,
                "n_tel": n_tel,
                "energy_bin_number": energy_bin_number,
            },
            output_filename.with_suffix(".joblib"),
        )
        _logger.info(f"{name} model saved to: {output_filename.with_suffix('.joblib')}")


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
        "--model-prefix",
        required=True,
        help=(
            "Path to directory for writing XGBoost classification models "
            "(without n_tel and energy bin suffix)."
        ),
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
        "--model-parameters",
        type=str,
        help=("Path to model parameter file (JSON) defining which models to load. "),
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
        event_lists[0],
        event_lists[1],
        args.ntel,
        args.model_prefix,
        args.train_test_fraction,
        model_parameters,
        args.energy_bin_number,
    )
    _logger.info("XGBoost classification model trained successfully.")


if __name__ == "__main__":
    main()
