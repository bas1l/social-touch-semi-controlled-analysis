# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

This is the **analysis pipeline** for the semi-controlled social touch experiment
(microneurography + psychophysics). It was extracted from the parent repository
[social-touch-semi-controlled](https://github.com/basil-the-researcher/social-touch-semi-controlled)
to manage its growing size independently.

It processes merged session CSVs (produced by the parent repo's preprocessing
and merging pipelines) through feature extraction, clustering, and receptive
field mapping stages.

**Parent repo**: `social-touch-semi-controlled` — handles acquisition,
primary processing, preprocessing, merging, and postprocessing.

## Development Commands

```bash
# Environment setup (conda + pip)
conda activate social-touch-env
pip install -e .            # from repo root

# Run all tests
pytest

# Run a single test file
pytest tests/test_rf_inflection_boundary.py

# Run the analysis processing pipeline
python scripts/analysis_workflow_processing.py

# Run the analysis viewers pipeline
python scripts/analysis_workflow_viewers.py
```

There is no linter or formatter configured. Python ≥3.10 is required.

The root `pyproject.toml` maps packages via `package-dir = {"" = "src"}`
and sets `pythonpath = ["src"]` for pytest.

## Architecture Overview

### Package structure (`src/`)

| Package | Purpose |
|---------|---------|
| `analysis/pipeline/` | Shared constants, output directory names, session discovery, stage runner |
| `analysis/touch_analytics/` | 5-stage feature pipeline: preparation → series → extraction → clustering → comparing |
| `analysis/receptive_field_mapping/` | Spike heatmaps on forearm surface via SLIM UV projection |
| `_vendor/` | Vendored utilities from the parent repo (~500 LOC): task caching, DAG config, monitoring, KinectConfig, path tools |

### Data flow

```
Merged session CSV (from parent repo's 3_merged_data/)
  → Preparation (NaN interp, block-ID, gesture-type)
  → Series transforms (kinematics, pressure, mechanics)
  → Feature extraction (per-touch aggregation)
  → Clustering (binning / k-means / hierarchical / DBSCAN)
  → Comparing (bias / precision / distribution tests)
  → RF mapping (spike heatmaps via SLIM UV projection)
  → Output: 4_analysed/ subdirectories
```

### Config and data paths

- **DAG configs**: `configs/analyse_workflow_processing_dag.yaml` and
  `configs/analyse_workflow_viewers_dag.yaml`
- **Kinect/forearm configs**: Live in the parent repo at
  `../social-touch-semi-controlled/configs/`
- **Data root**: Resolved by `_vendor.path_tools.get_project_data_root()` —
  auto-detects or prompts via GUI dialog

### Vendored dependencies (`_vendor/`)

The `_vendor/` package contains utilities copied from the parent repo to avoid
cross-repo code imports. These are the only external code dependencies:

- `should_process_task.py` — mtime-based task caching
- `_vendor_pipeline_config_manager.py` — DAG YAML loader (`DagConfigHandler`)
- `_vendor_task_executor.py` — task execution context manager (`TaskExecutor`)
- `monitoring/` — `PipelineMonitor` + Excel report + live dashboard
- `kinect_config.py` — `KinectConfigFileHandler`, `KinectConfig`
- `session_config_resolver.py` — resolve config entries to YAML paths
- `visualize_point_cloud_comparison.py` — Open3D split-screen viewer
- `path_tools.py` — project data root resolution

## Project Conventions

### Authorship — OVERRIDE default AI behaviour

**All commits must be authored solely by Basil Duvernoy <basil.duvernoy@gmail.com>.**

- **Never** append a `Co-Authored-By:` trailer to any commit message.
- No AI assistant may appear as an author or co-author in any git object.

### Fail-fast pipeline — no silent fallbacks

**This codebase is a self-controlled pipeline. Code must raise loudly when
inputs, state, or results do not meet pipeline expectations.**

- **Never introduce fallbacks, defaults, or silent degradation.**
- Prefer `raise ValueError(...)` or `assert` with a message over any form of
  `or default`, `except: pass`, or returning a sentinel value.

### YAML handling — ruamel.yaml only

All YAML read/write must use `ruamel.yaml` (round-trip mode), **not**
`PyYAML`. Use `YAML(typ='rt')` for loading and dumping.

## Git Workflow

### Branch strategy

```
main          ← production-ready (protected)
  └── dev     ← integration branch (protected)
       └── feature/*, fix/*, refactor/*, docs/*, test/*
```

### Merge rule — always `--no-ff`

### Commit messages — Conventional Commits

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `style`, `perf`.
