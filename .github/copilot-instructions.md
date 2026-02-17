# Copilot Instructions for Eventdisplay-ML

## Project Overview

**Eventdisplay-ML** is a machine learning toolkit for gamma-ray astronomy that interfaces with the [Eventdisplay](https://github.com/Eventdisplay/Eventdisplay) software package. It provides XGBoost-based models for:

1. **Stereo reconstruction** (direction and energy estimation) from telescope array data
2. **Gamma/hadron separation** (classification) for cosmic ray event rejection

The project processes data from **Imaging Atmospheric Cherenkov Telescopes (IACTs)**, specifically VERITAS and CTAO (Cherenkov Telescope Array Observatory).

### Key Scientific Context

- **Input**: ROOT files from Eventdisplay analysis (`mscw` files containing `data` trees)
- **Output**: ROOT files with reconstructed event parameters (direction, energy, classification)
- **Domain**: Gamma-ray astronomy, air shower physics, telescope array data analysis
- **Instruments**: Multi-telescope arrays (VERITAS, CTAO) with varying telescope configurations

## Repository Structure

```
Eventdisplay-ML/
├── .github/
│   └── workflows/          # CI/CD workflows (linting, tests, releases)
├── docs/
│   └── changes/            # Towncrier changelog fragments
├── src/
│   └── eventdisplay_ml/
│       ├── scripts/        # Command-line entry points
│       ├── config.py       # Training/inference configuration
│       ├── data_processing.py  # Data loading, flattening, preprocessing
│       ├── evaluate.py     # Model evaluation metrics
│       ├── features.py     # Feature definitions and exclusions
│       ├── geomag.py       # Geomagnetic field calculations
│       ├── hyper_parameters.py  # XGBoost hyperparameters
│       ├── models.py       # Model training and inference
│       └── utils.py        # Utility functions
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_*.py           # Unit tests (focus on telescope indexing)
│   └── resources/          # Test data files
├── pyproject.toml          # Package configuration
├── environment.yml         # Conda environment specification
└── .pre-commit-config.yaml # Pre-commit hooks configuration
```

## Technology Stack

### Core Dependencies
- **Python**: >= 3.12 (strictly typed, modern Python)
- **XGBoost**: Machine learning library for regression and classification
- **uproot**: Read ROOT files without ROOT installation
- **awkward**: Handle variable-length array data from ROOT
- **pandas/numpy**: Data manipulation
- **scikit-learn**: ML utilities and metrics
- **matplotlib**: Visualization

### Development Tools
- **pytest**: Testing framework with coverage, mocking, profiling
- **ruff**: Fast Python linter and formatter (replaces black, isort, flake8)
- **pre-commit**: Git hooks for code quality
- **towncrier**: Changelog management
- **setuptools-scm**: Version management from git tags

## Build, Test, and Lint Commands

### Installation

```bash
# Install in editable mode with test dependencies
pip install -e .[tests]

# Or use conda/mamba
conda env create -f environment.yml
conda activate eventdisplay_ml
```

### Testing

```bash
# Run all tests with coverage
pytest tests/

# Run specific test file
pytest tests/test_telescope_indexing.py -v

# Run with specific markers or patterns
pytest tests/ -k "test_disp"
```

**Important**: Some tests are marked with `@pytest.mark.skip` because they require external data files (VERITAS/CTAO ROOT files). These are for manual validation only.

### Linting and Formatting

```bash
# Run all pre-commit hooks
SKIP=no-commit-to-branch pre-commit run --all-files

# Run ruff (linter + formatter)
ruff check src/ tests/
ruff format src/ tests/

# Run isort (import sorting)
isort src/ tests/ --profile black

# Run codespell (spelling)
codespell --write-changes
```

### Building and Running

The package provides CLI entry points (see `pyproject.toml` scripts section):

```bash
# Training
eventdisplay-ml-train-xgb-stereo --help
eventdisplay-ml-train-xgb-classify --help

# Inference
eventdisplay-ml-apply-xgb-stereo --help
eventdisplay-ml-apply-xgb-classify --help

# Plotting
eventdisplay-ml-plot-classification-performance-metrics --help
eventdisplay-ml-plot-classification-gamma-efficiency --help
```

## Code Style and Conventions

### General Python Style

- **Line length**: 100 characters (configured in ruff)
- **Docstring style**: NumPy format (see `tool.ruff.lint.pydocstyle.convention`)
- **Import style**: Sorted by isort with black compatibility
- **Quote style**: Double quotes for strings
- **Indentation**: 4 spaces (no tabs)

### Key Conventions

1. **Logging**: Use `logging` module with `_logger = logging.getLogger(__name__)` pattern
   - Example: `_logger.info(f"Processing {n_events} events")`

2. **Type hints**: Modern Python typing preferred but not strictly enforced everywhere
   - Use for new code when practical

3. **Array handling**: Prefer awkward arrays for variable-length data
   - Use `ak.to_numpy()` or `ak.to_pandas()` for conversion
   - Be aware of jagged array structures

4. **Feature naming**: Telescope-dependent features use suffix `_{i}` where `i` is telescope index
   - Example: `size_0`, `size_1`, `size_2` for 3 telescopes
   - Special variables: `DispTelList_T`, `ImgSel_list` control indexing

5. **File I/O**: Always use `pathlib.Path` for file operations
   - Configured via `PTH` rules in ruff

6. **Error handling**: Use explicit exception types
   - Avoid bare `except:` clauses

7. **No commented-out code**: Enforced by `ERA` rules in ruff

### Scientific Computing Conventions

- **Energy units**: log10(E/TeV) for energy binning
- **Angles**: Degrees for user-facing, radians for internal calculations
- **Coordinates**: Camera coordinates (x, y) in degrees
- **Fill values**: Use `np.nan` for missing telescope data (see `DEFAULT_FILL_VALUE`)

## Domain-Specific Knowledge

### For Gamma-Ray Astronomy Experts

**Air Shower Physics Context**:
- The code processes **stereoscopic** observations from multiple telescopes
- Each gamma-ray or cosmic ray creates an air shower with Cherenkov light
- Multiple telescopes observe the same shower from different positions
- Key reconstruction tasks:
  - **Direction**: Where in the sky did the particle come from?
  - **Energy**: What was the primary particle's energy?
  - **Particle type**: Gamma ray or hadron (cosmic ray background)?

**Telescope Array Indexing** (Critical!):
- Two data formats exist:
  1. **VERITAS mode**: Fixed telescope ID indexing with `R_core` present
  2. **CTAO mode**: Variable-length indexing with `ImgSel_list` present
- The code automatically detects and handles both modes
- `DispTelList_T` maps telescope IDs to array indices for Disp values
- Missing telescopes (no trigger) are filled with `NaN`

**Key Physics Features**:
- **Disp method**: Direction reconstruction from image parameters
- **Intersection method**: Direction from stereoscopic geometry
- **Image parameters**: Hillas parameters (length, width, size, etc.)
- **Emission height**: Altitude of maximum shower development
- **Geomagnetic angle**: Angle between pointing and Earth's magnetic field

**Observatory-Specific**:
- VERITAS: 4 telescopes in Utah, ~12m diameter mirrors
- CTAO: Multiple telescope types (LST, MST, SST) with different mirror sizes
- Telescope sorting: By mirror area first (proxy for type), then by image size

### For Machine Learning Experts

**Model Architecture**:
- **Stereo reconstruction**: Multi-output regression (XGBoost)
  - Targets: `[MCxoff, MCyoff, MCe0]` (x offset, y offset, log energy)
  - Single model handles all telescope multiplicities (2-4+ telescopes)
  - Features: Telescope-level arrays + event-level parameters

- **Classification**: Binary classification (XGBoost)
  - Target: Gamma vs hadron (implicit in training data split)
  - Training in energy bins to handle energy-dependent features
  - Zenith angle included as binned feature

**Feature Engineering**:
- **Telescope-level features**: Variable-length arrays (up to max_tel_id)
  - Sorted by mirror area, then size
  - Missing telescopes padded with `NaN`
  - Features: image parameters, geometry, telescope positions

- **Event-level features**: Scalars
  - Reconstructed values from other methods (Disp, intersection)
  - Array geometry, pointing direction
  - Geomagnetic field angle

- **Excluded features**: See `features.excluded_features()`
  - Pointing corrections excluded in stereo mode
  - Energy and position features excluded in classification

**Training Details**:
- Train/test split: Configurable (default 0.5)
- Sample weighting: By energy and telescope multiplicity
- Hyperparameters: Defined in `hyper_parameters.py` or JSON config
- Parallel training: Use `--max_cores` flag
- Early stopping: Configured via XGBoost parameters

**Key Functions**:
- `train_regression()` in `models.py`: Main training loop for stereo
- `train_classification()` in `models.py`: Main training loop for classification
- `load_training_data()` in `data_processing.py`: Data loading and preprocessing
- `flatten_feature_data()` in `data_processing.py`: Convert telescope arrays to flat features

**Performance Metrics**:
- Regression: Angular resolution, energy resolution, bias
- Classification: ROC curves, efficiency at fixed background rates
- Evaluation in energy bins (see `_EVAL_LOG_E_*` constants)

### For Python/Computing Experts

**Data Pipeline**:
1. **Loading**: ROOT → uproot → awkward arrays → pandas DataFrame
2. **Preprocessing**: Flatten telescope arrays, sort, pad, add derived features
3. **Training**: XGBoost with custom sample weights
4. **Inference**: Load models, apply to new data, write ROOT output
5. **Evaluation**: Calculate metrics, generate plots

**Performance Considerations**:
- **Parallel decompression**: Use `ThreadPoolExecutor` with `max_cores` argument
- **Memory**: Large ROOT files can be memory-intensive
  - Use `max_events` parameter to limit processing
- **Awkward arrays**: More efficient than nested lists/numpy for variable-length data
- **Model persistence**: Use `joblib.dump()` for XGBoost models

**Critical Implementation Details**:

1. **Telescope indexing resolution** (`data_processing.py`):
   ```python
   # Function: _resolve_branch_aliases()
   # Handles R_core (VERITAS) vs R (CTAO) branch name differences
   # Function: _normalize_telescope_variable_to_tel_id_space()
   # Maps ImgSel_list indices to telescope ID space
   ```

2. **Array flattening** (`data_processing.py`):
   ```python
   # Function: flatten_feature_data()
   # Converts [event, telescope] jagged arrays to [event, max_tel_id] dense arrays
   # Sorting by mirror area and size happens here
   ```

3. **Feature preparation**:
   ```python
   # Function: prepare_feature_set()
   # Combines telescope-level and event-level features
   # Applies exclusions and adds derived features
   ```

**Error Handling Patterns**:
- Missing branches in ROOT files → Use `_resolve_branch_aliases()`
- Variable telescope participation → Pad with `NaN`
- File not found → Clear error messages with file paths
- Model loading → Check energy bin patterns, handle missing files

**Testing Patterns**:
- Fixtures in `conftest.py` create synthetic DataFrames
- Focus on telescope indexing correctness (most complex logic)
- Mock ROOT file I/O for unit tests
- Integration tests marked as skipped (require real data)

**Version Management**:
- Uses `setuptools-scm` for automatic versioning from git tags
- Version written to `src/eventdisplay_ml/_version.py` (auto-generated, gitignored)
- Update `CITATION.cff` manually for releases
- Changelog via towncrier: add fragments to `docs/changes/`, run `towncrier build`

## Common Patterns and Gotchas

### Critical Gotchas

1. **Telescope Indexing is Complex**:
   - VERITAS uses fixed indices (0-3 for 4 telescopes)
   - CTAO uses variable indices (telescope IDs can be 0, 5, 12, 42, etc.)
   - `DispTelList_T` is the ground truth for mapping
   - Never assume contiguous telescope IDs!

2. **Array Lengths are Variable**:
   - Not all telescopes trigger on every event
   - Arrays must be padded to `max_tel_id + 1`
   - Use `_to_dense_array()` helper for conversion

3. **Branch Name Aliases**:
   - Some branches have different names: `R_core` vs `R`
   - Use `_resolve_branch_aliases()` for compatibility

4. **Feature Sorting**:
   - Telescope features must be sorted consistently
   - Order: mirror area (descending), then size (descending)
   - This is critical for model consistency!

5. **Energy Binning**:
   - Classification uses overlapping energy bins
   - Energy is in log10(E/TeV)
   - Different models for different energy ranges

6. **NaN Handling**:
   - XGBoost handles NaN natively (missing value feature)
   - Don't convert NaN to 0 or other values!

### Useful Patterns

**Loading ROOT Data**:
```python
import uproot
tree = uproot.open("file.root")["data"]
arrays = tree.arrays(["branch1", "branch2"], library="np")
```

**Creating Test Fixtures**:
```python
# Use helpers from conftest.py
df = create_base_df(n_rows=10, n_tel=4)
```

**Adding New Features**:
1. Add to `telescope_features()` or as event-level feature
2. Update `excluded_features()` if needed for specific analyses
3. Add preprocessing in `flatten_feature_data()` or `prepare_feature_set()`
4. Add test coverage for the new feature

**Model Training Workflow**:
```bash
# 1. Prepare input file list (text file with paths)
# 2. Train model
eventdisplay-ml-train-xgb-stereo \
    --input_file_list inputs.txt \
    --model_prefix models/stereo_model \
    --max_events 100000 \
    --train_test_fraction 0.5 \
    --max_cores 8

# 3. Apply model
eventdisplay-ml-apply-xgb-stereo \
    --input_file_list apply_inputs.txt \
    --output_file_list outputs.txt \
    --model_prefix models/stereo_model
```

## Development Workflow

### Making Changes

1. **Create branch**: Work on feature branches, not `main`
2. **Install dev tools**: `pip install -e .[dev,tests]`
3. **Make changes**: Follow code conventions
4. **Add tests**: Unit tests in `tests/` directory
5. **Run tests**: `pytest tests/`
6. **Run linters**: `pre-commit run --all-files`
7. **Add changelog fragment**: Create `docs/changes/PR_NUMBER.TYPE.md`
   - Types: `feature`, `bugfix`, `api`, `maintenance`, `doc`
8. **Commit**: Pre-commit hooks run automatically
9. **Push**: CI runs linting and tests

### Changelog Management

Create a changelog fragment in `docs/changes/`:
```bash
# Example: docs/changes/123.feature.md
echo "Add support for new telescope type." > docs/changes/123.feature.md
```

When releasing, run:
```bash
towncrier build --version v1.2.3
```

### CI/CD

- **Linting job**: Runs ruff, isort, codespell, CITATION.cff validation
- **Tests job**: Runs pytest with coverage
- **Release workflow**: Builds and publishes to PyPI (manual trigger)

## Security and Quality Considerations

### Security

1. **Input validation**: Validate file paths, ensure files exist
2. **No secrets in code**: Never commit API keys, tokens, etc.
3. **Dependency security**: Keep dependencies updated
4. **Data privacy**: No personal data in test files or logs

### Quality

1. **Test coverage**: Aim for >80% coverage (currently ~27%, needs improvement)
2. **Documentation**: All public functions should have docstrings
3. **Type hints**: Use where practical for better IDE support
4. **Code review**: All changes go through PR review
5. **Reproducibility**: Set `random_state` for deterministic results

### Known Issues and Workarounds

**Issue**: Some tests require external ROOT files
- **Workaround**: Tests marked with `@pytest.mark.skip`, run manually with data

**Issue**: Awkward array version compatibility
- **Workaround**: Pin to tested versions in `pyproject.toml`

**Issue**: Large model files
- **Workaround**: Don't commit model files, only code to generate them

**Issue**: Memory usage with large datasets
- **Workaround**: Use `--max_events` to limit processing, or process in batches

## Quick Reference

### Important Files

- `src/eventdisplay_ml/data_processing.py`: Core data loading/preprocessing logic
- `src/eventdisplay_ml/models.py`: Training and inference
- `src/eventdisplay_ml/features.py`: Feature definitions
- `pyproject.toml`: Package metadata, dependencies, tool configuration
- `.pre-commit-config.yaml`: Code quality checks

### Key Commands

```bash
# Development setup
pip install -e .[dev,tests]

# Run tests
pytest tests/ -v

# Run linters
pre-commit run --all-files

# Train stereo model
eventdisplay-ml-train-xgb-stereo --input_file_list files.txt --model_prefix model

# Apply classification
eventdisplay-ml-apply-xgb-classify --input_file_list files.txt --output_file_list out.txt --model_prefix model
```

### Debugging Tips

1. **Enable debug logging**: Add `logging.basicConfig(level=logging.DEBUG)` to scripts
2. **Inspect telescope indexing**: Use `_logger.info()` in `data_processing.py` functions
3. **Check array shapes**: Print DataFrame columns and shapes before/after flattening
4. **Visualize features**: Use matplotlib to plot feature distributions
5. **Profile code**: Use `pytest-profiling` for performance analysis

## Getting Help

- **Documentation**: README.md and inline docstrings
- **Issues**: https://github.com/Eventdisplay/Eventdisplay-ML/issues
- **Eventdisplay docs**: https://github.com/Eventdisplay/Eventdisplay

## Attribution

This project uses generative AI tools (Claude, ChatGPT, Gemini) for code development and documentation. All AI-generated content is reviewed and validated by the authors.

## License

BSD 3-Clause License - See LICENSE file for details.
