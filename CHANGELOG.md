# Changelog

All notable changes to the Eventdisplay-ML project will be documented in this file.
Changes for upcoming releases can be found in the [docs/changes](docs/changes) directory.

This changelog is generated using [Towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

## [v0.5.0](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.5.0) - 2026-01-25

### New Features

- Introduces telescope type handling for CTAO simulations by updating the stereo reconstruction pipeline to work with telescope-dependent variables across different telescope configurations. The key architectural change is moving from training separate models per telescope multiplicity (2, 3, 4 telescopes) to a single unified model that handles all multiplicities together. This is a major change applicable for both stereo and classification tasks. ([#29](https://github.com/Eventdisplay/Eventdisplay-ML/pull/29))
- Add a telescope presence flag (tel_active) and implement combined weighting for both energy and telescope multiplicity in the training process. ([#34](https://github.com/Eventdisplay/Eventdisplay-ML/pull/34))
- Introduced sorting of telescope-dependent variables by mirror area (as proxy to telescope type) and size. ([#38](https://github.com/Eventdisplay/Eventdisplay-ML/pull/38))
- Add CTAO-specific support for telescope indexing/sorting and geomagnetic angle calculation by introducing an observatory configuration, new geomagnetic field presets, and updated sorting behavior (mirror area first, then size). ([#39](https://github.com/Eventdisplay/Eventdisplay-ML/pull/39))
- Reduces reliance on elevation/azimuth-derived coordinates and expands per-telescope feature set by adding channel-count features. ([#41](https://github.com/Eventdisplay/Eventdisplay-ML/pull/41))

### Maintenance

- Migrate the data loading pipeline from pandas to Awkward Array for improved performance when processing the ROOT files. Enable parallel decompression through ThreadPoolExecutor (use `--max_cores` argument). ([#31](https://github.com/Eventdisplay/Eventdisplay-ML/pull/31))


## [v0.4.0](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.4.0) - 2026-01-20

### New Features

- Apply unified clipping settings to feature variables. ([#28](https://github.com/Eventdisplay/Eventdisplay-ML/pull/28))
- Add angle between pointing direction and geomagnetic field vector as feature. ([#28](https://github.com/Eventdisplay/Eventdisplay-ML/pull/28))


## [v0.3.0](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.3.0) - 2026-01-14

### New Features

- Calculation classification thresholds for signal efficiencies and fill as boolean to classification trees. ([#18](https://github.com/Eventdisplay/Eventdisplay-ML/pull/18))
- Add plotting scripts for classification efficiency.
  Add plotting scripts to compare TMVA and XGB performance for classification ([#21](https://github.com/Eventdisplay/Eventdisplay-ML/pull/21))

### Maintenance

- Add Zenodo entry to: https://doi.org/10.5281/zenodo.18117884 . ([#17](https://github.com/Eventdisplay/Eventdisplay-ML/pull/17))
- Improve memory efficiency of training: loading and flattening data frames per file. ([#24](https://github.com/Eventdisplay/Eventdisplay-ML/pull/24))


## [v0.2.0](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.2.0) - 2026-01-01

### New Features

- add classification routines for gamma/hadron separation.
- add pre-training quality cuts.

([#13](https://github.com/Eventdisplay/Eventdisplay-ML/pull/13))

### Maintenance

- refactoring code to minimize duplication and improve maintainability.
- unified command line interface for all scripts.
- unit tests are disabled for now due to rapid changes in the codebase.

([#13](https://github.com/Eventdisplay/Eventdisplay-ML/pull/13))


## [v0.1.1](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.1.1) - 2025-12-22

### Maintenance

- Add PyPI project. ([#12](https://github.com/Eventdisplay/Eventdisplay-ML/pull/12))


## [v0.1.0](https://github.com/Eventdisplay/Eventdisplay-ML/releases/tag/v0.1.0) - 2025-12-22

First release of Eventdisplay-ML. Provides basic functionality for direction and energy reconstruction applied to VERITAS data and simulations.

### New Features

- Train and apply scripts for direction and energy reconstruction. ([#4](https://github.com/Eventdisplay/Eventdisplay-ML/pull/4))

### Maintenance

- Initial commit of CI workflows. ([#2](https://github.com/Eventdisplay/Eventdisplay-ML/pull/2))
- Initial commit and mv of python scripts from https://github.com/VERITAS-Observatory/EventDisplay_v4/pull/331. ([#3](https://github.com/Eventdisplay/Eventdisplay-ML/pull/3))
- Introduce data processing module to avoid code duplication. ([#8](https://github.com/Eventdisplay/Eventdisplay-ML/pull/8))
- Add unit tests. ([#10](https://github.com/Eventdisplay/Eventdisplay-ML/pull/10))
