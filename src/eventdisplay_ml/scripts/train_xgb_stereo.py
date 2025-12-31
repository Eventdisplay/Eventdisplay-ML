"""
Train XGBoost BDTs stereo reconstruction (direction, energy).

Uses x,y offsets calculated from intersection and dispBDT methods plus
image parameters to train multi-target regression BDTs to predict x,y offsets.

Uses energy related values to estimate event energy.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import argparse
import logging

import xgboost as xgb
from sklearn.model_selection import train_test_split

from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import evaluate_regression_model
from eventdisplay_ml.features import target_features
from eventdisplay_ml.hyper_parameters import (
    pre_cuts_regression,
    regression_hyperparameters,
)
from eventdisplay_ml.models import save_models

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(df, model_configs):
    """
    Train a single XGBoost model for multi-target regression.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with training data.
    model_configs : dict
        Dictionary of model configurations.
    """
    n_tel = model_configs["n_tel"]
    if df.empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return None

    x_cols = df.columns.difference(model_configs["targets"])
    _logger.info(f"Training variables ({len(x_cols)}): {x_cols}")
    model_configs["features"] = list(x_cols)
    x_data, y_data = df[x_cols], df[model_configs["targets"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=model_configs.get("train_test_fraction", 0.5),
        random_state=model_configs.get("random_state", None),
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    for name, cfg in model_configs.get("models", {}).items():
        _logger.info(f"Training {name} for n_tel={n_tel}...")
        model = xgb.XGBRegressor(**cfg.get("hyper_parameters", {}))
        model.fit(x_train, y_train)
        evaluate_regression_model(model, x_test, y_test, df, x_cols, y_data, name)
        cfg["model"] = model

    return model_configs


def configure():
    """Configure training."""
    parser = argparse.ArgumentParser(
        description=("Train XGBoost Multi-Target BDTs for Stereo Analysis (Direction, Energy).")
    )
    parser.add_argument("--input_file_list", help="List of input mscw files.")
    parser.add_argument(
        "--model_prefix",
        required=True,
        help=("Path to directory for writing XGBoost regression models (without n_tel suffix)."),
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

    model_configs = vars(parser.parse_args())

    _logger.info("--- XGBoost Regression Training ---")
    _logger.info(f"Telescope multiplicity: {model_configs.get('n_tel')}")
    _logger.info(f"Model output prefix: {model_configs.get('model_prefix')}")
    _logger.info(f"Train vs test fraction: {model_configs['train_test_fraction']}")
    _logger.info(f"Max events: {model_configs['max_events']}")

    model_configs["models"] = regression_hyperparameters(model_configs.get("hyperparameter_config"))
    model_configs["targets"] = target_features("stereo_analysis")
    model_configs["pre_cuts"] = pre_cuts_regression(model_configs.get("n_tel"))

    return model_configs


def main():
    """Run the training pipeline."""
    model_configs = configure()
    df_flat = load_training_data(model_configs, "stereo_analysis")
    model_configs = train(df_flat, model_configs)
    save_models(model_configs)

    _logger.info("XGBoost regression model trained successfully.")


if __name__ == "__main__":
    main()
