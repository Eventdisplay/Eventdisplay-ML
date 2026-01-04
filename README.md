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

## Gamma/hadron separation using XGBoost

Gamma/hadron separation is performed using XGB Boost classification trees. Features are image parameters and stereo reconstruction parameters provided by Eventdisplay.
Training is performed in overlapping energy bins to account for energy dependence of the classification.
The zenith angle dependence is accounted for by including the zenith angle as a binned feature in the training.

Output is a single ROOT tree called `Classification` with the same number of events as the input tree. It contains the classification prediction (`Gamma_Prediction`) and boolean flags (e.g. `Is_Gamma_75` for 75% signal efficiency cut).

## Citing this Software

Please cite this software if it is used for a publication, see the [Zenodo record](https://doi.org/10.5281/zenodo.18117884) and [CITATION.cff](CITATION.cff) for details.
