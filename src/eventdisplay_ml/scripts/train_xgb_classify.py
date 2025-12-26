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
from eventdisplay_ml.evaluate import evaluate_classification_model

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(signal_df, background_df, n_tel, output_dir, train_test_fraction):
    """
    Train a single XGBoost model for gamma/hadron classification.

    Parameters
    ----------
    - signal_df: Pandas DataFrame with signal training data.
    - background_df: Pandas DataFrame with background training data.
    - n_tel: Telescope multiplicity.
    - output_dir: Directory to save the trained model.
    - train_test_fraction: Fraction of data to use for training.
    """
    if signal_df.empty or background_df.empty:
        _logger.warning(
            f"Skipping training for n_tel={n_tel} due to empty signal or background data."
        )
        return

    signal_df["label"] = 1
    background_df["label"] = 0
    full_df = pd.concat([signal_df, background_df], ignore_index=True)
    x_data = full_df.drop(columns=["label"])
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

        output_filename = Path(output_dir) / f"classify_bdt_ntel{n_tel}_{name}.joblib"
        dump(model, output_filename)
        _logger.info(f"{name} model saved to: {output_filename}")

        evaluate_classification_model(model, x_test, y_test, full_df, x_data.columns.tolist(), name)


def main():
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description=("Train XGBoost models for gamma/hadron classification.")
    )
    parser.add_argument("--input_signal_file_list", help="List of input signal mscw ROOT files.")
    parser.add_argument(
        "--input_background_file_list", help="List of input background mscw ROOT files."
    )
    parser.add_argument("--ntel", type=int, help="Telescope multiplicity (2, 3, or 4).")
    parser.add_argument("--output_dir", help="Output directory for XGBoost models and weights.")
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

    args = parser.parse_args()

    input_signal_files = utils.read_input_file_list(args.input_signal_file_list)
    input_background_files = utils.read_input_file_list(args.input_background_file_list)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    _logger.info("--- XGBoost Classification Training ---")
    _logger.info(f"Signal input files: {len(input_signal_files)}")
    _logger.info(f"Background input files: {len(input_background_files)}")
    _logger.info(f"Telescope multiplicity: {args.ntel}")
    _logger.info(f"Output directory: {output_dir}")
    _logger.info(
        f"Train vs test fraction: {args.train_test_fraction}, Max events: {args.max_events}"
    )

    signal_events = load_training_data(
        input_signal_files, args.ntel, args.max_events, analysis_type="signal_classification"
    )

    background_events = load_training_data(
        input_background_files,
        args.ntel,
        args.max_events,
        analysis_type="background_classification",
    )

    train(signal_events, background_events, args.ntel, output_dir, args.train_test_fraction)

    _logger.info("XGBoost model trained successfully.")


if __name__ == "__main__":
    main()
