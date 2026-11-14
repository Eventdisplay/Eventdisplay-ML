# Eventdisplay-ML

Machine learning tools for Eventdisplay gamma-ray astronomy analysis. The package
trains and applies XGBoost models for stereo direction/energy reconstruction and
gamma/hadron separation using Eventdisplay `mscw` ROOT files.

## Install

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install eventdisplay-ml
```

To install the current source checkout instead (for development or unreleased
changes), run these commands from the repository root:

```bash
python -m pip install -e ".[dev,tests]"
```

The package includes the command-line tools for training, applying models, and
creating diagnostics. For Eventdisplay data workflows, start with the examples
in [Usage](docs/usage.md). The scripts consume Eventdisplay `mscw` ROOT files;
they do not produce those files.

## Workflows

- **Stereo reconstruction:** train residual regressors and apply them to produce
  a `StereoAnalysis` tree. See [stereo instructions](docs/usage.md#direction-and-energy-reconstruction-using-xgboost).
- **Gamma/hadron separation:** train energy-binned classifiers and apply them to
  produce a `Classification` tree. See [classification instructions](docs/usage.md#gammahadron-separation-using-xgboost).
- **Diagnostics:** inspect model performance and feature behavior. See the
  [diagnostic commands](docs/usage.md#diagnostic-tools).

## Development

```bash
python -m pip install -e ".[dev,tests]"
ruff check src tests
pytest tests
```

See [Usage](docs/usage.md) for training, application, and validation examples.
For citations, see the [Zenodo record](https://doi.org/10.5281/zenodo.18117884)
and [CITATION.cff](CITATION.cff).

### Optional stereo image smearing

`eventdisplay-ml-train-xgb-stereo` and `eventdisplay-ml-apply-xgb-stereo` accept
`--smearing_config configs/image_smearing.example.yaml`. Use this only on MC.
Smearing is disabled when the option is omitted. YAML values are Gaussian
standard deviations; `fraction` chooses the fraction of events perturbed, so 0.5
mixes nominal and perturbed events. Use `fraction: 1` to perturb all events.
The sample values are exploratory, not calibrated. Training settings are stored
in model training metadata; application requires an explicit configuration and
logs its resolved values. Run/event identifiers must be present and unique per
physical event. Repeated identifiers receive identical perturbations, independent
of chunk size. Use a different seed to vary the realization.

The perturbations act on active, valid telescope images before size sorting and
feature transformations. The centroid shift updates camera distance by the
change in centroid radius; size-dependent ML features are recalculated. Existing
DispBDT, event-level reconstruction and event selections are not rerun. Thus
this is an approximation, not a replacement for image cleaning or a full
Eventdisplay reconstruction. Keep inference on observational data unsmeared.
