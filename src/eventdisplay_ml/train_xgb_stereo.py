"""
Train XGBoost Multi-Target BDTs for direction and energy reconstruction.

Uses x,y offsets calculated from intersection and dispBDT methods plus
image parameters to train multi-target regression BDTs to predict x,y offsets.

Uses energy related values to estimate event energy.

Separate BDTs are trained for 2, 3, and 4 telescope multiplicity events.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from eventdisplay_ml.training_variables import (
    xgb_all_training_variables,
    xgb_per_telescope_training_variables,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("trainXGBoostforStereoAnalysis")


def load_and_flatten_data(input_files, n_tel, max_events):
    """Load and flatten ROOT data for the requested telescope multiplicity."""
    _logger.info(f"\n--- Loading and Flattening Data for n_tel = {n_tel} ---")
    _logger.info(f"Max events to process: {max_events if max_events else 'All available'}")

    branch_list = ["MCxoff", "MCyoff", "MCe0", *xgb_all_training_variables()]

    dfs = []
    if max_events:
        max_events_per_file = max_events // len(input_files)
    else:
        max_events_per_file = None

    for f in input_files:
        try:
            with uproot.open(f) as root_file:
                if "data" in root_file:
                    _logger.info(f"Processing file: {f}")
                    tree = root_file["data"]
                    df = tree.arrays(branch_list, library="pd")
                    df = df[df["DispNImages"] == n_tel]
                    _logger.info(f"Number of events after n_tel filter: {len(df)}")
                    if max_events_per_file and len(df) > max_events_per_file:
                        df = df.sample(n=max_events_per_file, random_state=42)
                    if not df.empty:
                        dfs.append(df)
                else:
                    _logger.warning(f"File: {f} does not contain a 'data' tree.")
        except Exception as e:
            _logger.error(f"Error opening or reading file {f}: {e}")

    if len(dfs) == 0:
        _logger.error("No data loaded from input files.")
        return pd.DataFrame()

    data_tree = pd.concat(dfs, ignore_index=True)
    _logger.info(f"Total events for n_tel={n_tel}: {len(data_tree)}")

    if data_tree.empty:
        return pd.DataFrame()

    # Compute weights (not used in training, but for monitoring)
    # - R (to reflect physical sky area)
    sample_weights = np.hypot(data_tree["MCxoff"], data_tree["MCyoff"])

    df_flat = flatten_data_vectorized(data_tree, n_tel, xgb_per_telescope_training_variables())

    df_flat["MCxoff"] = data_tree["MCxoff"]
    df_flat["MCyoff"] = data_tree["MCyoff"]
    df_flat["MCe0"] = np.log10(data_tree["MCe0"])
    df_flat["sample_weight"] = sample_weights

    df_flat.dropna(inplace=True)
    _logger.info(f"Final events for n_tel={n_tel} after cleanup: {len(df_flat)}")

    return df_flat


def flatten_data_vectorized(df, n_tel, training_variables):
    """Vectorized flattening of telescope array columns."""
    flat_features = {}

    try:
        tel_list_matrix = np.vstack(df["DispTelList_T"].values)
    except ValueError:
        tel_list_matrix = np.array(df["DispTelList_T"].tolist())

    for var_name in training_variables:
        # Data matrix has shape (n_events, max_n_tel)
        try:
            data_matrix = np.vstack(df[var_name].values)
        except ValueError:
            data_matrix = np.array(df[var_name].tolist())

        for i in range(n_tel):
            col_name = f"{var_name}_{i}"

            if var_name.startswith("Disp"):
                # Case 1: Simple index i
                if i < data_matrix.shape[1]:
                    flat_features[col_name] = data_matrix[:, i]
                else:
                    flat_features[col_name] = np.full(len(df), np.nan)
            else:
                # Case 2: Index lookup via DispTelList_T
                target_tel_indices = tel_list_matrix[:, i].astype(int)

                row_indices = np.arange(len(df))
                valid_mask = (target_tel_indices >= 0) & (target_tel_indices < data_matrix.shape[1])
                result = np.full(len(df), np.nan)
                result[valid_mask] = data_matrix[
                    row_indices[valid_mask], target_tel_indices[valid_mask]
                ]

                flat_features[col_name] = result

    # Convert dictionary to DataFrame once at the end
    df_flat = pd.DataFrame(flat_features, index=df.index)

    # Additional derived features to support finding certain event classes
    for i in range(n_tel):
        df_flat[f"disp_x_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"cosphi_{i}"]
        df_flat[f"disp_y_{i}"] = df_flat[f"Disp_T_{i}"] * df_flat[f"sinphi_{i}"]
        df_flat[f"loss_loss_{i}"] = df_flat[f"loss_{i}"] ** 2
        df_flat[f"loss_dist_{i}"] = df_flat[f"loss_{i}"] * df_flat[f"dist_{i}"]
        df_flat[f"width_length_{i}"] = df_flat[f"width_{i}"] / (df_flat[f"length_{i}"] + 1e-6)
        df_flat[f"size_{i}"] = np.log10(df_flat[f"size_{i}"] + 1e-6)
        df_flat[f"E_{i}"] = np.log10(np.clip(df_flat[f"E_{i}"], 1e-6, None))
        df_flat[f"ES_{i}"] = np.log10(np.clip(df_flat[f"ES_{i}"], 1e-6, None))

    df_flat["Xoff_weighted_bdt"] = df["Xoff"]
    df_flat["Yoff_weighted_bdt"] = df["Yoff"]
    df_flat["Xoff_intersect"] = df["Xoff_intersect"]
    df_flat["Yoff_intersect"] = df["Yoff_intersect"]
    df_flat["Diff_Xoff"] = df["Xoff"] - df["Xoff_intersect"]
    df_flat["Diff_Yoff"] = df["Yoff"] - df["Yoff_intersect"]
    df_flat["Erec"] = np.log10(np.clip(df["Erec"], 1e-6, None))
    df_flat["ErecS"] = np.log10(np.clip(df["ErecS"], 1e-6, None))
    df_flat["EmissionHeight"] = df["EmissionHeight"]

    return df_flat


def train(df, n_tel, output_dir, train_test_fraction):
    """
    Train a single XGBoost model for multi-target regression (Xoff, Yoff, MCe0).

    Parameters
    ----------
    - df: Pandas DataFrame with training data.
    - n_tel: Telescope multiplicity.
    - output_dir: Directory to save the trained model.
    - train_test_fraction: Fraction of data to use for training.
    """
    if df.empty:
        _logger.warning(f"Skipping training for n_tel={n_tel} due to empty data.")
        return

    # Separate feature and target columns
    x_cols = [col for col in df.columns if col not in ["MCxoff", "MCyoff", "MCe0", "sample_weight"]]
    x_data = df[x_cols]
    y_data = df[["MCxoff", "MCyoff", "MCe0"]]

    _logger.info(f"Training variables ({len(x_cols)}): {x_cols}")

    x_train, x_test, y_train, y_test, w_train, _ = train_test_split(
        x_data,
        y_data,
        df["sample_weight"],
        test_size=1.0 - train_test_fraction,
        random_state=None,
    )

    _logger.info(f"n_tel={n_tel}: Training events: {len(x_train)}, Testing events: {len(x_test)}")

    # Parse TMVA options (simplified mapping to XGBoost parameters)
    # The default TMVA string is:
    # !V:NTrees=800:BoostType=Grad:Shrinkage=0.1:MaxDepth=4:MinNodeSize=1.0%

    # Note: XGBoost defaults to gbtree (Gradient Boosting).
    # MultiOutputRegressor requires a base estimator (e.g., plain XGBRegressor)
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
    _logger.info(
        f"Sample weights (not(!) used in training) - min: {w_train.min():.6f}, "
        f"max: {w_train.max():.6f}, mean: {w_train.mean():.6f}"
    )

    for name, estimator in configs.items():
        _logger.info(f"Training with {name} for n_tel={n_tel}...")
        _logger.info(f"parameters: {xgb_params}")
        model = MultiOutputRegressor(estimator)
        model.fit(x_train, y_train)

        output_filename = Path(output_dir) / f"dispdir_bdt_ntel{n_tel}_{name}.joblib"
        dump(model, output_filename)
        _logger.info(f"{name} model saved to: {output_filename}")

        evaluate_model(model, x_test, y_test, df, x_cols, y_data, name)


def evaluate_model(model, x_test, y_test, df, x_cols, y_data, name):
    """Evaluate the trained model on the test set and log performance metrics."""
    score = model.score(x_test, y_test)
    _logger.info(f"XGBoost Multi-Target R^2 Score (Testing Set): {score:.4f}")
    y_pred = model.predict(x_test)
    mse_x = mean_squared_error(y_test["MCxoff"], y_pred[:, 0])
    mse_y = mean_squared_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MSE (X_off): {mse_x:.4f}, MSE (Y_off): {mse_y:.4f}")
    mae_x = mean_absolute_error(y_test["MCxoff"], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test["MCyoff"], y_pred[:, 1])
    _logger.info(f"{name} MAE (X_off): {mae_x:.4f}, MAE (Y_off): {mae_y:.4f}")

    feature_importance(model, x_cols, y_data.columns, name)
    if name == "xgboost":
        shap_feature_importance(model, x_test, y_data.columns)

    calculate_resolution(
        y_pred,
        y_test,
        df,
        percentiles=[68, 90, 95],
        log_e_min=-1,
        log_e_max=2,
        n_bins=6,
        name=name,
    )


def calculate_resolution(y_pred, y_test, df, percentiles, log_e_min, log_e_max, n_bins, name):
    """Compute angular and energy resolution based on predictions."""
    results_df = pd.DataFrame(
        {
            "MCxoff_true": y_test["MCxoff"].values,
            "MCyoff_true": y_test["MCyoff"].values,
            "MCxoff_pred": y_pred[:, 0],
            "MCyoff_pred": y_pred[:, 1],
            "MCe0_pred": y_pred[:, 2],
            "MCe0": df.loc[y_test.index, "MCe0"].values,
        }
    )

    results_df["DeltaTheta"] = np.sqrt(
        (results_df["MCxoff_true"] - results_df["MCxoff_pred"]) ** 2
        + (results_df["MCyoff_true"] - results_df["MCyoff_pred"]) ** 2
    )
    results_df["DeltaMCe0"] = np.abs(
        np.power(10, results_df["MCe0_pred"]) - np.power(10, results_df["MCe0"])
    ) / np.power(10, results_df["MCe0"])

    results_df["LogE"] = results_df["MCe0"]
    bins = np.linspace(log_e_min, log_e_max, n_bins + 1)
    results_df["E_bin"] = pd.cut(results_df["LogE"], bins=bins, include_lowest=True)
    results_df.dropna(subset=["E_bin"], inplace=True)

    g = results_df.groupby("E_bin", observed=False)
    mean_loge_by_bin = g["LogE"].mean().round(3)

    def percentile_series(col, p):
        return g[col].quantile(p / 100)

    for col, label in [("DeltaTheta", "Theta"), ("DeltaMCe0", "DeltaE")]:
        data = {f"{label}_{p}%": percentile_series(col, p).values for p in percentiles}

        output_df = pd.DataFrame(data, index=mean_loge_by_bin.index)
        output_df.insert(0, "Mean Log10(E)", mean_loge_by_bin.values)
        output_df.index.name = "Log10(E) Bin Range"
        output_df = output_df.dropna()

        _logger.info(f"--- {name} {col} Resolution vs. Log10(MCe0) ---")
        _logger.info(
            f"Calculated over {n_bins} bins between Log10(E) = {log_e_min} and {log_e_max}"
        )
        _logger.info(f"\n{output_df.to_markdown(floatfmt='.4f')}")


def feature_importance(model, x_cols, target_names, name=None):
    """Log feature importance from the trained XGBoost model."""
    _logger.info("--- XGBoost Multi-Regression Feature Importance ---")
    for i, estimator in enumerate(model.estimators_):
        target = target_names[i]
        _logger.info(f"\n### {name} Importance for Target: **{target}**")

        importances = estimator.feature_importances_
        importance_df = pd.DataFrame({"Feature": x_cols, "Importance": importances})

        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        _logger.info(f"\n{importance_df.head(15).to_markdown(index=False)}")


def shap_feature_importance(model, x_data, target_names, max_points=20000, n_top=25):
    """Use XGBoost's builtin SHAP."""
    x_sample = x_data.sample(n=min(len(x_data), max_points), random_state=0)
    for i, est in enumerate(model.estimators_):
        target = target_names[i]

        # Builtin XGBoost SHAP values (n_samples, n_features+1)
        # Last column is the bias term: drop it
        shap_vals = est.get_booster().predict(xgb.DMatrix(x_sample), pred_contribs=True)
        shap_vals = shap_vals[:, :-1]  # drop bias column

        # Global importance: mean(|SHAP|)
        imp = np.abs(shap_vals).mean(axis=0)
        idx = np.argsort(imp)[::-1]

        _logger.info(f"\n=== Builtin XGBoost SHAP Importance for {target} ===")
        for j in idx[:n_top]:
            _logger.info(f"{x_data.columns[j]:25s}  {imp[j]:.6e}")


def main():
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description=("Train XGBoost Multi-Target BDTs for Stereo Analysis (Direction, Energy).")
    )
    parser.add_argument("--input_file_list", help="List of input mscw ROOT files.")
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

    try:
        with open(args.input_file_list) as f:
            input_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Error: Input file list not found: {args.input_file_list}"
        ) from exc

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    _logger.info("--- XGBoost Multi-Target Training ---")
    _logger.info(f"Input files: {len(input_files)}")
    _logger.info(f"Telescope multiplicity: {args.ntel}")
    _logger.info(f"Output directory: {output_dir}")
    _logger.info(
        f"Train vs test fraction: {args.train_test_fraction}, Max events: {args.max_events}"
    )

    df_flat = load_and_flatten_data(input_files, args.ntel, args.max_events)
    train(df_flat, args.ntel, output_dir, args.train_test_fraction)
    _logger.info("\nXGBoost model trained successfully.")


if __name__ == "__main__":
    main()
