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

### Training-evaluation curves**

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
