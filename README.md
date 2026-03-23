# Machine learning for Eventdisplay

[![LICENSE](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Eventdisplay/Eventdisplay-ML/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/eventdisplay/eventdisplay-ml)](https://github.com/Eventdisplay/Eventdisplay-ML/releases)
[![pypi](https://badge.fury.io/py/eventdisplay-ml.svg)](https://badge.fury.io/py/eventdisplay-ml)
[![DOI](https://zenodo.org/badge/1120034687.svg)](https://doi.org/10.5281/zenodo.18117884)

Toolkit to interface and run machine learning methods together with the Eventdisplay software package for gamma-ray astronomy data analysis.

Provides examples on how to use e.g., scikit-learn or XGBoost regression trees to estimate event direction, energies, and gamma/hadron separators.

Introduces a Python environment and a scripts directory to support training and inference.

Input is provided through the `mscw` output (`data` trees).

## Direction and energy reconstruction using XGBoost

Stereo analysis methods implemented in Eventdisplay provide direction / energies per event resp telescope image. The machine learner implemented Eventdisplay-ML uses XGB Boost regression trees. Features are all estimators (e.g. DispBDT or intersection method results) plus additional features (mostly image parameters) to get a better estimator for directions and energies.

Output is a single ROOT tree called `StereoAnalysis` with the same number of events as the input tree.

### Training Stereo Reconstruction Models

The stereo regression training pipeline uses multi-target XGBoost to predict residuals (deviations from baseline reconstructions):

**Targets:** `[Xoff_residual, Yoff_residual, E_residual]` (direction and energy improvements)

**Key techniques:**

- **Target standardization:** Targets are mean-centered and scaled to unit variance during training, allowing multi-output learning with balanced feature importance across direction and energy
- **Energy-bin weighting:** Events are weighted inversely by energy bin density; bins with fewer than 10 events are excluded from training to prevent overfitting on low-statistics regions
- **Multiplicity weighting:** Higher-multiplicity events (more telescopes) receive higher sample weights to prioritize high-confidence reconstructions
- **Per-target SHAP importance:** Feature importance values computed during training for each target and cached for later analysis

**Training command:**

```bash
eventdisplay-ml-train-xgb-stereo \
    --input_file_list train_files.txt \
    --model_prefix models/stereo_model \
    --max_events 100000 \
    --train_test_fraction 0.5 \
    --max_cores 8
```

**Output:** Joblib model file containing:

- XGBoost booster
- Target standardization scalers (mean/std)
- Feature list and SHAP importance rankings
- Training metadata (random state, hyperparameters)

### Applying Stereo Reconstruction Models

The apply pipeline loads trained models and makes predictions on new events:

**Process:**

1. Load training features from input ROOT files
2. Make predictions in (mean=0, std=1) standardized space
3. Inverse-standardize predictions using stored target_mean and target_std
4. Combine with baseline reconstructions: `final = baseline + residual`
5. Validate energy values and convert to log10 space

**Key safeguards:**

- Invalid energy values (≤0 or NaN) produce NaN outputs but preserve all input event rows
- Missing standardization parameters raise clear ValueError (prevents silent data corruption)
- Output row count always equals input row count

**Apply command:**

```bash
eventdisplay-ml-apply-xgb-stereo \
    --input_file_list apply_files.txt \
    --output_file_list output_files.txt \
    --model_prefix models/stereo_model
```

**Output:** ROOT files with `StereoAnalysis` tree containing reconstructed Xoff, Yoff, and log10(E).

## Gamma/hadron separation using XGBoost

Gamma/hadron separation is performed using XGB Boost classification trees. Features are image parameters and stereo reconstruction parameters provided by Eventdisplay.
Training is performed in overlapping energy bins to account for energy dependence of the classification.
The zenith angle dependence is accounted for by including the zenith angle as a binned feature in the training.

Output is a single ROOT tree called `Classification` with the same number of events as the input tree. It contains the classification prediction (`Gamma_Prediction`) and boolean flags (e.g. `Is_Gamma_75` for 75% signal efficiency cut).

## Testing

### Test Coverage

The package includes comprehensive unit tests focused on the stereo regression pipeline:

**Core test modules:**

- `tests/test_train_regression_standardization.py`: Training pipeline tests (11 tests)
  - Target standardization computation from training set only
  - Energy-bin weighting logic including low-count bin exclusion
  - Train/test split isolation to prevent data leakage
  - Integration tests for complete training workflows

- `tests/test_regression_apply.py`: Apply pipeline tests (9 tests)
  - Standardization inversion (loading and applying target scalers)
  - Safe energy validation and log10 computation
  - Residual reconstruction and final prediction assembly
  - Output row count preservation with invalid energy handling
  - Error handling for missing standardization parameters

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_train_regression_standardization.py -v

# Run with coverage report
pytest tests/ --cov=src/eventdisplay_ml --cov-report=html
```

**Test results:** 34 tests passing, 2 skipped (require external data files)

**Coverage:** ~42% overall (models: 43%, data_processing: 53%, evaluate: 51%)

### Diagnostic Tools

The committed regression diagnostics in this branch are:

**1) Cached SHAP feature-importance summary**

Script: `src/eventdisplay_ml/scripts/diagnostic_shap_summary.py`

Purpose:

- Load per-target SHAP importances cached in the trained model file
- Create one top-20 feature plot per residual target (`Xoff_residual`, `Yoff_residual`, `E_residual`)

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated PNGs

Run:

```bash
python -m eventdisplay_ml.scripts.diagnostic_shap_summary \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/
```

Outputs:

- `diagnostics/shap_importance_Xoff_residual.png`
- `diagnostics/shap_importance_Yoff_residual.png`
- `diagnostics/shap_importance_E_residual.png`

Notes:

- This tool reads cached values from the model file (no test data required).
- It expects `shap_importance` and `features` to be present in model metadata.

**2) Training-evaluation curves**

Script/entry point: `src/eventdisplay_ml/scripts/plot_training_evaluation.py` via
`eventdisplay-ml-plot-training-evaluation`

Purpose:

- Plot XGBoost training vs validation metric curves from `evals_result()`
- Useful for checking convergence and overfitting behavior

Required inputs:

- `--model_file`: trained model `.joblib` containing an XGBoost model
- `--output_file`: output image path (optional; if omitted, plot is shown interactively)

Run:

```bash
eventdisplay-ml-plot-training-evaluation \
  --model_file models/stereo_model.joblib \
  --output_file diagnostics/training_curves.png
```

Output:

- Figure with one panel per tracked metric (for example `rmse`), showing training and test curves.

## Generative AI disclosure

Generative AI tools (including Claude, ChatGPT, and Gemini) were used to assist with code development, debugging, and documentation drafting. All AI-assisted outputs were reviewed, validated, and, where necessary, modified by the authors to ensure accuracy and reliability.

## Citing this Software

Please cite this software if it is used for a publication, see the [Zenodo record](https://doi.org/10.5281/zenodo.18117884) and [CITATION.cff](CITATION.cff) for details.
