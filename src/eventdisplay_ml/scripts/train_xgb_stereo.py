"""
Train XGBoost BDTs stereo reconstruction (direction, energy).

Uses x,y offsets calculated from intersection and dispBDT methods plus
image parameters to train multi-target regression BDTs to predict x,y offsets.

Uses energy related values to estimate event energy.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import argparse
import logging
from pathlib import Path

import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from eventdisplay_ml import utils
from eventdisplay_ml.data_processing import load_training_data
from eventdisplay_ml.evaluate import evaluate_regression_model
from eventdisplay_ml.features import target_features

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def train(df, n_tel, model_prefix, train_test_fraction):
    """
    Train a single XGBoost model for multi-target regression (Xoff, Yoff, MCe0).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with training data.
    n_tel : int
        Telescope multiplicity.
    model_prefix : str
        Directory to save the trained model.
    train_test_fraction : float
        Fraction of data to use for training.
    """
    if df.empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return

    model_prefix = Path(model_prefix)
    output_dir = model_prefix.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    targets = target_features("stereo_analysis")
    x_cols = [col for col in df.columns if col not in targets]
    x_data = df[x_cols]
    y_data = df[targets]

    _logger.info(f"Training variables ({len(x_cols)}): {x_cols}")

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=1.0 - train_test_fraction,
        random_state=None,
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    xgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.1,  # Shrinkage
        "max_depth": 5,
        "min_child_weight": 1.0,  # Equivalent to MinNodeSize=1.0% for XGBoost
        "objective": "reg:squarederror",
        "n_jobs": 4,
        "random_state": None,
        "tree_method": "hist",
        "subsample": 0.7,  # Default sensible value
        "colsample_bytree": 0.7,  # Default sensible value
    }
    configs = {
        "xgboost": xgb.XGBRegressor(**xgb_params),
    }

    for name, estimator in configs.items():
        _logger.info(f"Training with {name} for n_tel={n_tel}...")
        _logger.info(f"parameters: {xgb_params}")
        model = MultiOutputRegressor(estimator)
        model.fit(x_train, y_train)

        output_filename = Path(output_dir) / f"{model_prefix.name}_{name}_ntel{n_tel}.joblib"
        dump(
            {
                "model": model,
                "features": x_cols,
                "target": targets,
                "hyperparameters": xgb_params,
                "n_tel": n_tel,
            },
            output_filename,
        )
        _logger.info(f"{name} model saved to: {output_filename}")

        evaluate_regression_model(model, x_test, y_test, df, x_cols, y_data, name)


def main():
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description=("Train XGBoost Multi-Target BDTs for Stereo Analysis (Direction, Energy).")
    )
    parser.add_argument("--input_file_list", help="List of input mscw files.")
    parser.add_argument(
        "--model-prefix",
        required=True,
        help=("Path to directory for writing XGBoost regression models (without n_tel suffix)."),
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
    args = parser.parse_args()

    _logger.info("--- XGBoost Regression Training ---")
    _logger.info(f"Telescope multiplicity: {args.ntel}")
    _logger.info(f"Model output prefix: {args.model_prefix}")
    _logger.info(f"Train vs test fraction: {args.train_test_fraction}")
    _logger.info(f"Max events: {args.max_events}")

    df_flat = load_training_data(
        utils.read_input_file_list(args.input_file_list),
        args.ntel,
        args.max_events,
        analysis_type="stereo_analysis",
    )
    train(df_flat, args.ntel, args.model_prefix, args.train_test_fraction)
    _logger.info("XGBoost regression model trained successfully.")


if __name__ == "__main__":
    main()
