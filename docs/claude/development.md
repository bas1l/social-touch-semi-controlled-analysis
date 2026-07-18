# Development Commands

```bash
# Create environment (first time)
conda env create -f environment.yml

# Activate and editable-install
conda activate social-touch-analysis
pip install -e .            # from repo root

# Run all tests
pytest

# Run a single test file
pytest tests/test_rf_inflection_boundary.py

# Run the analysis processing pipeline
python scripts/analysis_workflow_processing.py

# Run the analysis viewers pipeline
python scripts/analysis_workflow_viewers.py

# Launch the GUI pipeline runner
python scripts/launch_analysis_runner_gui.py
```

There is no linter or formatter configured. Python ≥3.10 is required.

The root `pyproject.toml` maps packages via `package-dir = {"" = "src"}`
and sets `pythonpath = ["src"]` for pytest.
