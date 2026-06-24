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

**Targets:** `[Xoff_residual, Yoff_residual, E_residual]` (residuals on direction and energy as reconstruction by the BDT stereo reconstruction method)

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
```


**Output:** ROOT files with `StereoAnalysis` tree containing reconstructed Xoff, Yoff, and log10(E).

## Gamma/hadron separation using XGBoost

Gamma/hadron separation is performed using XGB Boost classification trees. Features are image parameters and stereo reconstruction parameters provided by Eventdisplay.
Training is performed in overlapping energy bins to account for energy dependence of the classification.
The zenith angle dependence is accounted for by including the zenith angle as a binned feature in the training.

Output is a single ROOT tree called `Classification` with the same number of events as the input tree. It contains the classification prediction (`Gamma_Prediction`) and boolean flags (e.g. `Is_Gamma_75` for 75% signal efficiency cut).

### Training Gamma/Hadron Separation Models

Gamma/hadron training uses gamma-ray simulations as signal and observed background data as background. This is a powerful setup, but it requires additional validation because the classifier can learn simulation-vs-data differences that are not physical gamma/hadron separation.

**Training command:**

```bash
eventdisplay-ml-train-xgb-classify \
    --input_signal_file_list signal_files.txt \
    --input_background_file_list background_files.txt \
    --model_parameters model_parameters.json \
    --energy_bin_number 0 \
    --model_prefix models/gammahadron_bdt \
    --max_events 5000000 \
    --train_test_fraction 0.5 \
    --max_cores 8
```

Repeat the command for all configured energy bins. The output files are named automatically with an energy-bin suffix, for example `gammahadron_bdt_ebin0.joblib.gz`.

**Optional zenith-distribution balancing:**

Gamma-ray simulations are often produced at fixed zenith angles, while observed background data have continuous zenith distributions. The model uses `ze_bin` as a coarse zenith-conditioning feature. To reduce the chance that `ze_bin` becomes a signal/background prior instead of a conditioning variable, enable class/zenith weights:

```bash
eventdisplay-ml-train-xgb-classify \
    --input_signal_file_list signal_files.txt \
    --input_background_file_list background_files.txt \
    --model_parameters model_parameters.json \
    --energy_bin_number 0 \
    --model_prefix models/gammahadron_bdt \
    --balance_class_zenith_weights
```

With this option, the training split is weighted so gamma and background have the same `ze_bin` distribution in the selected energy bin. The feature `ze_bin` is still available to the classifier, so zenith-dependent shower morphology can still be learned. The weights only remove the simplest class-prior shortcut.

**Optional training without `ze_bin`:**

For a direct stress test of zenith-bin dependence, remove `ze_bin` from the classifier input:

```bash
eventdisplay-ml-train-xgb-classify \
    --input_signal_file_list signal_files.txt \
    --input_background_file_list background_files.txt \
    --model_parameters model_parameters.json \
    --energy_bin_number 0 \
    --model_prefix models/gammahadron_bdt_nozebin \
    --ignore_ze_bin
```

This keeps the same event selection but excludes `ze_bin` from the training features. The held-out `ze_bin` values are still used as evaluation metadata, so per-zenith efficiency tables and plots remain available. It is intended for comparison against the default and `--balance_class_zenith_weights` trainings. Do not combine `--ignore_ze_bin` with `--balance_class_zenith_weights`.

### Gamma/Hadron Validation Checklist

Use these diagnostics before trusting unusually good gamma/hadron performance:

1. Check training curves with `eventdisplay-ml-plot-training-evaluation`. Training and validation AUC/logloss should be close. A very large train/test gap indicates overfitting, but a small gap does not rule out simulation-vs-data leakage if train and test are split from the same mixed samples.
2. Check SHAP importances with `eventdisplay-ml-diagnostic-shap-summary`. Expected high-ranking features are `MSCW`, `MSCL`, `EmissionHeight`, image widths/lengths, timing gradients, and core-distance terms. Treat dominant `ze_bin`, run-condition, pointing, missing-value, or file-production features as audit triggers.
3. Check per-energy and per-zenith ROC/Q-factor plots with `eventdisplay-ml-plot-classification-performance-metrics`. Compare `overall` plots to each `zeM` plot. Suspicious signs are one zenith bin carrying most of the inclusive performance, very sharp score separation in only one bin, or high performance only where gamma/background statistics differ strongly.
4. Apply the model to independent gamma MC and plot containment levels with `eventdisplay-ml-plot-classification-gamma-efficiency`. Gamma score percentiles should vary smoothly with zenith angle, wobble offset, NSB, and energy. Discontinuous behavior at training bin boundaries is a warning sign.
5. Compare standard training to `--balance_class_zenith_weights`. If inclusive performance drops modestly but per-zenith behavior becomes smoother, the balanced model is usually the safer production choice. A large performance drop means the original model was using zenith composition strongly.
6. Validate final cuts with `optimize_classification.py` and `plot_optimize_classification.py`. The optimized gamma efficiency, background efficiency, and significance surfaces should be smooth in energy and zenith unless there is a known rate feature.

Recommended high-value stress tests that are not fully automated yet:

- Train with `--ignore_ze_bin`, compare to the default and balanced trainings, and inspect per-zenith performance. The no-`ze_bin` model should be worse but not catastrophically different.
- Reweight gamma and background to the same distributions in zenith, wobble offset, NSB/noise, multiplicity, and reconstructed energy before training or evaluation.
- Hold out entire observing-condition groups instead of random events, for example a zenith band, wobble offset, NSB range, run period, or background file group. This detects shortcuts that random train/test splits hide.
- Evaluate on independent real-data control regions. The background score distribution should be stable across OFF regions and across run conditions after accounting for energy and zenith.
- Compare feature distributions for gamma MC and background data inside each energy and zenith bin before training. Variables that separate classes before shower morphology is considered are possible domain-shift handles.

## Diagnostic Tools

### SHAP feature-importance summary

Analysis type: stereo reconstruction and gamma/hadron classification.

Tests: Feature importance

- Load per-target SHAP importances cached in the trained model file
- Create one top-20 feature plot per target
  - Stereo: `Xoff_residual`, `Yoff_residual`, `E_residual`
  - Classification: `label` (gamma vs hadron)

Required inputs:

- `--model_file`: trained stereo or classification model `.joblib`
- `--model_dir`: directory with trained model `.joblib` or `.joblib.gz` files
- `--output_dir`: directory for generated PNGs

Use either `--model_file` or `--model_dir`.

Run:

```bash
  eventdisplay-ml-diagnostic-shap-summary \
  --model_file models/stereo_model.joblib \
  --output_dir diagnostics/

  eventdisplay-ml-diagnostic-shap-summary \
  --model_file models/classification_model_ebin0.joblib \
  --output_dir diagnostics/

  eventdisplay-ml-diagnostic-shap-summary \
  --model_dir models/ \
  --output_dir diagnostics/
```

Outputs (stereo):

- `diagnostics/shap_importance_Xoff_residual.png`
- `diagnostics/shap_importance_Yoff_residual.png`
- `diagnostics/shap_importance_E_residual.png`

Outputs (classification):

- `diagnostics/shap_importance_label.png`

With `--model_dir`, output plot names use the model filename as their base, for example
`classification_model_ebin0_label.png`.

Note: SHAP importances are cached during training. Existing model files trained before this feature was added will report a missing-cache error. Inference (`apply_xgb_classify`) does not require retraining, but running this diagnostic on a classification model does.

### Permutation importance

Analysis type: stereo reconstruction.

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

Analysis type: stereo reconstruction.

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

Analysis type: stereo reconstruction.

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

Analysis type: stereo reconstruction.

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
- `diagnostics/residual_diagnostics.png` (single 2xN grid; generated on cache miss when reconstruction is required)

Notes:

- Residual normality stats are cached during training and loaded from the model file for fast retrieval
- Diagnostic plots (histograms, Q-Q plots) are only generated when the split must be reconstructed
- Invalid KS test or Anderson-Darling results (NaN/inf) are reported as special values
- Outlier counts help identify events with unusually large reconstruction errors

### Training-evaluation curves

Analysis type: stereo reconstruction or gamma/hadron separation, depending on the model file.

- Plot XGBoost training vs validation metric curves
- Useful for checking convergence and overfitting behavior

Run:

```bash
eventdisplay-ml-plot-training-evaluation \
  --model_file models/stereo_model.joblib \
  --output_file diagnostics/training_curves.png
```

or for all joblib files in a directory:

```bash
eventdisplay-ml-plot-training-evaluation \
  --model_dir models/ \
  --output_dir diagnostics/
```

Output:

- Figure with one panel per tracked metric (for example `rmse`), showing training and test curves.

### Classification performance comparison

Analysis type: gamma/hadron separation.

- Compare XGBoost and optional TMVA BDT efficiency curves
- Plot signal/background efficiency, Q-factor, ROC, and reconstructed score distributions
- Produce one plot per energy bin and XGB zenith bin

Required inputs:

- `--xgb_dir`: directory with trained classification model files named `gammahadron_bdt_ebinN.joblib` or `.joblib.gz`
- `--output_dir`: directory for generated plots

Optional inputs:

- `--tmva_dir`: directory with TMVA ROOT files named `BDT_<energy_bin>_<zenith_bin>.root` or, if no plain BDT files exist, `TMVA.BDT_<energy_bin>_<zenith_bin>.root`
- `--energy-bin`: plot only one energy bin (`0`-`8`)
- `--zenith-bin-xgb`: plot only one XGB zenith bin; omit to plot overall and all available XGB zenith bins
- `--zenith-bin-tmva`: TMVA zenith bin used for overall or unmatched XGB bins; omit to use the first available TMVA zenith bin for each energy bin
- `--summary-file`: CSV file for zenith-uniformity metrics; omit to write `zenith_uniformity_summary.csv` in the output directory
- `--summary-signal-efficiency`: gamma efficiency used for zenith-uniformity metrics; default is `0.8`

Run:

```bash
eventdisplay-ml-plot-classification-performance-metrics \
  --tmva_dir tmva_models/ \
  --xgb_dir xgb_models/ \
  --output_dir diagnostics/
```

Equivalent source-tree invocation:

```bash
python src/eventdisplay_ml/scripts/plot_classification_performance_metrics.py \
  --tmva_dir tmva_models/ \
  --xgb_dir xgb_models/ \
  --output_dir diagnostics/
```

Output:

- `diagnostics/plot_performance_metrics_ebinN_overall.png`
- `diagnostics/plot_performance_metrics_ebinN_zeM.png`
- `diagnostics/zenith_uniformity_summary.csv`

Notes:

- If `--tmva_dir` is omitted or no matching TMVA ROOT file exists for a bin, the corresponding plot is generated with XGB curves only.
- When plain `BDT_*_*.root` files are present, they define the TMVA file set; `TMVA.BDT_*_*.root` files are only used as a fallback convention when no plain BDT files exist.
- For leakage checks, inspect both `overall` and every `zeM` plot. Good inclusive performance is not sufficient if one zenith bin or one score edge dominates the result.
- The zenith-uniformity CSV reports background efficiency at fixed gamma efficiency for each energy bin. Prefer models with low `worst_to_best_background_efficiency_ratio`, low `worst_to_overall_background_efficiency_ratio`, and acceptable `worst_zenith_background_efficiency`; these metrics are more relevant for zenith stability than the inclusive AUC alone.

### Gamma-efficiency containment on applied MC

Analysis type: gamma/hadron separation applied to gamma-ray simulations.

- Read applied `.xgb_gh.root` files containing the `Classification/Gamma_Prediction` branch
- Extract 70% and 95% gamma-score containment levels
- Plot containment versus air mass and wobble offset, split by NSB/noise level

Required input:

- Directory containing files named like `50deg_1.25wob_NOISE600.mscw.xgb_gh.root`

Run:

```bash
eventdisplay-ml-plot-classification-gamma-efficiency applied_gamma_mc/
```

Outputs:

- `containment_vs_airmass.png`
- `containment_vs_wob.png`

Notes:

- Use this on gamma MC samples not used for training whenever possible.
- The containment curves should be smooth in air mass, wobble offset, and NSB. Sharp discontinuities can indicate that the classifier learned production conditions or binning artifacts.
- This plot checks gamma acceptance stability; it does not measure background leakage by itself.

### Classification cut optimization

Analysis type: final gamma/hadron cut selection.

- `optimize_classification.py` combines trained ROC curves with ON/background rate surfaces
- It finds gamma-efficiency cuts that maximize Li & Ma significance for a requested source strength
- `plot_optimize_classification.py` visualizes the resulting energy/zenith surfaces

Required inputs:

- ROOT file containing `gONRate` and `gBGRate` `TGraph2D` or `TGraph2DErrors` objects
- Trained classification model files for all energy bins
- Source strength as a fraction of the Crab flux

Run:

```bash
python src/eventdisplay_ml/scripts/optimize_classification.py \
  rates.root \
  models/gammahadron_bdt_ebin*.joblib.gz \
  0.1 \
  --output diagnostics/optimized_cuts.ecsv

python src/eventdisplay_ml/scripts/plot_optimize_classification.py \
  diagnostics/optimized_cuts.ecsv \
  --output-dir diagnostics/
```

Outputs include:

- `signal_efficiency_vs_energy.png`
- `background_efficiency_vs_energy.png`
- `significance_vs_energy.png`
- `signal_efficiency_colz.png`
- `background_efficiency_colz.png`
- `significance_colz.png`

Notes:

- Smooth optimized-cut surfaces are expected. Isolated spikes, checkerboard structures, or abrupt zenith-bin jumps usually mean the ROC curves, rate surfaces, or interpolation inputs need inspection.
- This optimization inherits any bias in the trained classifier. Run the leakage checks above before interpreting high significances as physics performance.

## Generative AI disclosure

Generative AI tools (including Claude, ChatGPT, and Gemini) were used to assist with code development, debugging, and documentation drafting. All AI-assisted outputs were reviewed, validated, and, where necessary, modified by the authors to ensure accuracy and reliability.

## Citing this Software

Please cite this software if it is used for a publication, see the [Zenodo record](https://doi.org/10.5281/zenodo.18117884) and [CITATION.cff](CITATION.cff) for details.
