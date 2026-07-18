# Architecture

## Package structure (`src/`)

| Package | Purpose |
|---------|---------|
| `analysis/pipeline/` | Shared constants, output directory names, session discovery, stage runner |
| `analysis/touch_analytics/` | Feature pipeline sub-packages: `preparation/`, `representation/`, `feature_extraction/`, `reduction/`, `clustering/`, `comparing/`, `evaluation/`; top-level `*_pipeline.py` entry points drive each stage |
| `analysis/receptive_field_mapping/` | Spike heatmaps on forearm surface via SLIM UV projection; split into `surface/` (SLIM UV, projection), `metrics/` (boundary, grid, PCA), `rendering/` (all figure renderers), `pipelines/` (Prefect flow entry points), `data/` (loaders/IO), `gui/` (PyVista/PyQt5 viewers) |
| `utils/pipeline/` | `DagConfigModel` — ruamel.yaml round-trip model for GUI-driven DAG YAML editing |
| `utils/gui/analysis_runner_gui/` | PyQt5 GUI pipeline runner: three-column layout (workflow selector, task panel, session dirs), Prefect server manager, console, DAG graph view, config dialogs |
| `_vendor/` | Vendored utilities from the parent repo (~500 LOC): task caching, DAG config, monitoring, KinectConfig, path tools |

## Data flow

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

## Config and data paths

- **DAG configs**: `configs/analyse_workflow_processing_dag.yaml` and
  `configs/analyse_workflow_viewers_dag.yaml`
- **GUI runner config**: `configs/analysis_runner_gui.yaml` — lists workflow entries
  shown in the `AnalysisRunnerGUI` workflow selector
- **Kinect/forearm configs**: Live in the parent repo at
  `../social-touch-semi-controlled/configs/`
- **Data root**: Resolved by `_vendor.path_tools.get_project_data_root()` —
  auto-detects or prompts via GUI dialog

## Vendored dependencies (`_vendor/`)

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
