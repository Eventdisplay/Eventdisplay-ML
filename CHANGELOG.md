# Changelog

All notable changes to the Eventdisplay-ML project will be documented in this file.
Changes for upcoming releases can be found in the [docs/changes](docs/changes) directory.

This changelog is generated using [Towncrier](https://towncrier.readthedocs.io/).

<!-- towncrier release notes start -->

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
