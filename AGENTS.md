# AGENTS.md

Guidance for Codex and other AI agents working in this repository.

## Project context

`eventdisplay-ml` is a Python package for applying machine learning to Eventdisplay
IACT analysis products. It reads Eventdisplay `mscw` ROOT files, flattens telescope
and stereo-reconstruction quantities, trains XGBoost models, and writes model or ROOT
outputs for downstream Eventdisplay workflows.

The two main workflows are:

- Stereo reconstruction: multi-target XGBoost residual regression for `Xoff`, `Yoff`,
  and `log10(E)` corrections relative to the DispBDT baseline.
- Gamma/hadron separation: energy-binned XGBoost classifiers with zenith-bin metadata
  and signal-efficiency thresholds.

Keep the package as a thin, robust layer over existing Eventdisplay products. Do not
replace physics conventions, ROOT tree names, feature definitions, or model file formats
without explicit user direction.

## Repository layout

- `src/eventdisplay_ml/`: package source.
- `src/eventdisplay_ml/scripts/`: console entry points declared in `pyproject.toml`.
- `src/eventdisplay_ml/configs/`: default XGBoost hyperparameter JSON files.
- `tests/`: unit tests with small synthetic data and mocks.
- `docs/changes/`: Towncrier release fragments.
- `tmp_testing*`, `diagnostics/`, `.venv/`, `.ruff_cache/`, `.pytest_cache/`: ignored
  local/generated artifacts. Treat them as examples or scratch output only.

## Development environment

Python support is `>=3.12`; the conda environment pins Python 3.13 and
`xgboost=3.1.3` because later XGBoost versions changed output formats.

Common commands:

```bash
conda env create -f environment.yml
conda activate eventdisplay_ml
python -m pip install -e ".[tests,dev]"
ruff check src tests
ruff format src tests
pytest tests
```

The `pyproject.toml` pytest `testpaths` entry may not match the current flat `tests/`
layout, so prefer `pytest tests` for full local test runs.

## Coding rules

- Follow existing NumPy-style docstrings and `logging.getLogger(__name__)` usage.
- Preserve CLI argument names and console script behavior unless a migration is requested.
- Keep feature order stable where code comments say sequence matters.
- Use `pathlib.Path` for new filesystem logic where practical.
- Do not add broad exception swallowing around physics/data validation; fail loudly on
  missing model metadata, invalid bins, or row-count mismatches.
- Avoid loading large ROOT/model files in tests. Prefer small synthetic arrays,
  DataFrames, mocks, and temporary files.

## Data and model conventions

- Input ROOT files normally contain `data` and `telconfig` trees.
- Preserve output row counts when applying models; invalid events should produce `NaN`
  predictions rather than dropped rows where that is the established behavior.
- Missing telescope data are represented with `NaN`; XGBoost histogram trees handle
  these natively.
- Telescope-level features are sorted by image size, and for heterogeneous arrays by
  mirror area before size.
- VERITAS and CTAO branch variants both matter. Keep compatibility paths such as
  `R_core` vs `R` and `DispTelList_T` vs `ImgSel_list` indexing behavior.
- Classification models are stored as `PREFIX_ebin<N>.joblib` or `.joblib.gz`; loaders
  prefer compressed files when both exist.

## Working with ignored artifacts

Ignored `tmp_testing_vts/` files include representative HTCondor wrappers, logs, ROOT
outputs, trained `.joblib(.gz)` models, and diagnostic PNGs. They document real batch
usage, for example VERITAS gamma/hadron training with `eventdisplay-ml-train-xgb-classify`,
but they are not source files.

Do not commit generated ROOT files, trained models, logs, Condor output, cache
directories, or diagnostic plots unless the user explicitly asks for a curated fixture.

## Before finishing changes

Run the narrowest useful tests for the touched code, then broaden to `pytest tests` when
shared data processing, model loading, or CLI behavior changes. For formatting/lint-only
changes, run `ruff check` and `ruff format --check` on the touched paths.
