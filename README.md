# Machine learning for Eventdisplay

[![LICENSE](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Eventdisplay/Eventdisplay-ML/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/eventdisplay/eventdisplay-ml)](https://github.com/Eventdisplay/Eventdisplay-ML/releases)
[![pypi](https://badge.fury.io/py/eventdisplay-ml.svg)](https://badge.fury.io/py/eventdisplay-ml)

Toolkit to interface and run machine learning methods together with the Eventdisplay software package for gamma-ray astronomy data analysis.

Provides examples on how to use e.g., scikit-learn or XGBoost regression trees to estimate event direction, energies, and gamma/hadron separators.

Introduces a Python environment and a scripts directory to support training and inference.

## Direction and energy reconstruction using XGBoost

Stereo analysis methods implemented in Eventdisplay provide direction / energies per event resp telescope image. The machine learner implemented Eventdisplay-ML uses XGB Boost regression trees. Features are all estimators (e.g. DispBDT or intersection method results) plus additional features (mostly image parameters) to get a better estimator for directions and energies.

Input is provided through the `mscw` output (`data` trees).

Output is a single ROOT tree called `StereAnalysis` with the same number of events as the input tree.


## Citing this Software

Please cite this software if it is us
ed for a publication, see the Zenodo record and [CITATION.cff](CITATION.cff) for details.
