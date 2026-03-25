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

**Targets:** `[Xoff_residual, Yoff_residual, E_residual]` (residulas on direction and energy as reconstruction by the BDT stereo reconstruction method)

**Key techniques:**

- **Target standardization:** Targets are mean-centered and scaled to unit variance during training
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

- XGBoost trained model object
- Target standardization scalers (mean/std)
- Feature list and SHAP importance rankings
- Training metadata (random state, hyperparameters)

### Applying Stereo Reconstruction Models

The apply pipeline loads trained models and makes predictions:

**Key safeguards:**

- Invalid energy values (≤0 or NaN) produce NaN outputs but preserve all input event rows
- Missing standardization parameters raise ValueError (prevents silent data corruption)
- Output row count always equals input row count

**Apply command:**

```bash
eventdisplay-ml-apply-xgb-stereo \
    --input_file_list apply_files.txt \
    --output_file_list output_files.txt \
    --model_prefix models/stereo_model
**Output:** ROOT files with `StereoAnalysis` tree containing reconstructed Xoff, Yoff, and log10(E).

## Gamma/hadron separation using XGBoost

Gamma/hadron separation is performed using XGB Boost classification trees. Features are image parameters and stereo reconstruction parameters provided by Eventdisplay.
Training is performed in overlapping energy bins to account for energy dependence of the classification.
The zenith angle dependence is accounted for by including the zenith angle as a binned feature in the training.

Output is a single ROOT tree called `Classification` with the same number of events as the input tree. It contains the classification prediction (`Gamma_Prediction`) and boolean flags (e.g. `Is_Gamma_75` for 75% signal efficiency cut).

## Diagnostic Tools

The committed regression diagnostics in this branch are:

### SHAP feature-importance summary

 Tests: Feature importance

- Load per-target SHAP importances cached in the trained model file
- Create one top-20 feature plot per residual target (`Xoff_residual`, `Yoff_residual`, `E_residual`)

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated PNGs

Run:

```bash
  eventdisplay-ml-diagnostic-shap-summary \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/
```

Outputs:

- `diagnostics/shap_importance_Xoff_residual.png`
- `diagnostics/shap_importance_Yoff_residual.png`
- `diagnostics/shap_importance_E_residual.png`

### Permutation importance

- Rebuild the held-out test split from the model metadata and original input files
- Shuffle one feature at a time and measure the relative RMSE increase per residual target
- Validate predictive dependence on features rather than cached model attribution

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated plots
- `--top_n`: number of top features to include in the plot (optional)
- `--input_file_list`: optional override if the path stored in the model metadata is no longer valid

Run:

```bash
eventdisplay-ml-diagnostic-permutation-importance \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/ \
  --top_n 20
```

Optional override:

```bash
eventdisplay-ml-diagnostic-permutation-importance \
  --model_file models/stereo_model.joblib \
  --input_file_list files.txt \
  --output_dir diagnostics/
```

Output:

- `diagnostics/permutation_importance.png`

Notes:

- This diagnostic is slower than the SHAP summary because it rebuilds the processed test split.
- It is the better choice when you want to measure actual performance sensitivity to each feature.

### Generalization gap

- Read the cached train/test RMSE summary written during training
- Compare final train and test RMSE for each residual target
- Quantify the overfitting gap after training is complete

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated plots
- `--input_file_list`: optional override if the path stored in the model metadata is no longer valid

Run:

```bash
eventdisplay-ml-diagnostic-generalization-gap \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/
```

Optional override:

```bash
eventdisplay-ml-diagnostic-generalization-gap \
  --model_file models/stereo_model.joblib \
  --input_file_list files.txt \
  --output_dir diagnostics/
```

Output:

- `diagnostics/generalization_gap.png`

Notes:

- This diagnostic measures final overfitting by comparing train and test residual RMSE.
- Older model files without cached metrics fall back to rebuilding the original train/test split.
- Unlike `plot_training_evaluation.py`, it summarizes final RMSE, not the per-iteration XGBoost training history.

### Partial Dependence Plots

- Visualize how each feature influences model predictions
- Prove the model captures physics by checking that multiplicity reduces corrections and baselines show smooth relationships

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated plots (optional; default: `diagnostics`)
- `--features`: space-separated list of features to plot (optional; default: `DispNImages Xoff_weighted_bdt Yoff_weighted_bdt ErecS`)
- `--input_file_list`: optional override if the path stored in the model metadata is no longer valid

Run:

```bash
eventdisplay-ml-diagnostic-partial-dependence \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/ \
  --features DispNImages Xoff_weighted_bdt ErecS
```

Optional override:

```bash
eventdisplay-ml-diagnostic-partial-dependence \
  --model_file models/stereo_model.joblib \
  --input_file_list files.txt \
  --features Xoff_weighted_bdt Yoff_weighted_bdt
```

Output:

- `diagnostics/partial_dependence.png` (grid of feature × target subplots)

Notes:

- PDP displays predicted residual output as a function of a single feature while holding others constant
- Multiplicity effect: high-multiplicity events should show smaller corrections (negative slope)
- Baseline stability: baseline features (e.g., `weighted_bdt`) should show smooth, linear relationships
- This diagnostic rebuilds the held-out test split and is slower than SHAP summary

### Residual Normality Diagnostics

- Validate that model residuals follow a normal distribution
- Detect outlier events and check for systematic biases in reconstruction errors

Required inputs:

- `--model_file`: trained stereo model `.joblib`
- `--output_dir`: directory for generated plots (optional; default: `diagnostics`)
- `--input_file_list`: optional override if the path stored in the model metadata is no longer valid

Run:

```bash
eventdisplay-ml-diagnostic-residual-normality \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/
```

Optional override:

```bash
eventdisplay-ml-diagnostic-residual-normality \
  --model_file models/stereo_model.joblib \
  --input_file_list files.txt
```

Output:

- Residual normality statistics printed to console:
  - Mean and standard deviation per target
  - Kolmogorov-Smirnov test p-value (normality test)
  - Anderson-Darling test statistic and critical value
  - Skewness and kurtosis
  - Q-Q plot R² value
  - Number of outliers (>3σ) per target
- `diagnostics/residual_diagnostics_Xoff.png`, `diagnostics/residual_diagnostics_Yoff.png`, `diagnostics/residual_diagnostics_E.png` (if cache miss forces reconstruction)

Notes:

- Residual normality stats are cached during training and loaded from the model file for fast retrieval
- Diagnostic plots (histograms, Q-Q plots) are only generated when the split must be reconstructed
- Invalid KS test or Anderson-Darling results (NaN/inf) are reported as special values
- Outlier counts help identify events with unusually large reconstruction errors

### Training-evaluation curves

- Plot XGBoost training vs validation metric curves
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
