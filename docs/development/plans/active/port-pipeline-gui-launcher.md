# Plan: Port Pipeline GUI Launcher (Analysis-only)

**Date:** 2026-06-24
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/port-pipeline-gui-launcher`

---

## Overview

Port and rework the parent repo's `dag_launcher` GUI into a new `analysis_runner_gui` module scoped exclusively to the analysis workflows in this repo. The parent repo's launcher covered acquisition and postprocessing pipelines; this version targets only the two analysis pipelines (processing + viewers).

## Problem Statement

As this repo was extracted from the parent repo, it retained a dependency on the parent repo's GUI code (`dag_launcher`). That module was tightly coupled to parent-repo paths and workflows that don't exist here. A clean, scoped port is needed so the GUI can be run standalone in this repo.

## Goals

### In Scope
1. Import `dag_launcher` source from parent repo as a starting baseline
2. Rework it into `analysis_runner_gui` — rename, scope to analysis-only, clean internal references
3. Replace `launch_pipeline_gui.py` + `launcher.yaml` with `launch_analysis_runner_gui.py` + `analysis_runner_gui.yaml`
4. Delete the old `dag_launcher` module and its associated entry points

### Out of Scope
- Acquisition or postprocessing workflow support (parent repo only)
- New GUI features beyond the functional port (see pending plans for follow-on work)
- Automated tests for the GUI

## Success Criteria

- [ ] `python scripts/launch_analysis_runner_gui.py` opens the GUI
- [ ] Both Analysis [Processing] and Analysis [Viewers] workflows appear in the workflow selector
- [ ] Task panel, DAG graph view, and console behave as in the parent repo launcher
- [ ] `dag_launcher` module and old entry points are fully removed
- [ ] `CLAUDE.md` updated to reflect the new module path

---

## Technical Design

### Approach

Copy the `dag_launcher` source verbatim as a committed baseline, then rework in-place: rename the package, rename key classes (`LauncherWindow` → `RunnerWindow`, `LauncherConfig` → `RunnerConfig`), strip parent-repo-specific workflows from the config, and update import paths throughout.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Copy verbatim, rework in-place | Clear git history; review diff is meaningful | Initial commit is large | Chosen |
| Rework in parent repo, copy result | Cleaner history in parent | Cross-repo coordination overhead | Rejected |

### Architecture Changes

```
Before:
  scripts/launch_pipeline_gui.py
  configs/launcher.yaml
  src/utils/gui/dag_launcher/  (15 modules)

After:
  scripts/launch_analysis_runner_gui.py
  configs/analysis_runner_gui.yaml
  src/utils/gui/analysis_runner_gui/  (19 modules)
```

---

## Implementation Plan

### Phase 1: Baseline import (completed)

- [x] Copy `dag_launcher` module verbatim from parent repo
- [x] Add `scripts/launch_pipeline_gui.py` entry point (from parent)
- [x] Add `configs/launcher.yaml` (from parent)
- [x] Add `.gitignore` for Python artifacts
- [x] Pin `libigl==2.6.1`; add `grandalf` and `pytest` deps

**Files Modified:**
- `src/utils/gui/dag_launcher/` — 15 modules added
- `scripts/launch_pipeline_gui.py` — initial entry point
- `configs/launcher.yaml` — initial config

**Dependencies:** None

### Phase 2: Rework into analysis_runner_gui (in progress)

- [ ] Create `src/utils/gui/analysis_runner_gui/` module (rename, scope, clean imports)
- [ ] Rename `LauncherWindow` → `RunnerWindow` in `runner_window.py`
- [ ] Rename `LauncherConfig` → `RunnerConfig` in `runner_config.py`
- [ ] Create `scripts/launch_analysis_runner_gui.py` scoped to analysis workflows
- [ ] Create `configs/analysis_runner_gui.yaml` (analysis-only workflow entries)
- [ ] Delete `src/utils/gui/dag_launcher/`, `scripts/launch_pipeline_gui.py`, `configs/launcher.yaml`
- [ ] Update `configs/analyse_workflow_processing_dag.yaml` as needed
- [ ] Update `CLAUDE.md` to reference new module/script paths
- [ ] Commit all Phase 2 changes

**Files Modified:**
- `src/utils/gui/analysis_runner_gui/` — 19 modules (new)
- `src/utils/gui/dag_launcher/` — deleted
- `scripts/launch_analysis_runner_gui.py` — new entry point
- `scripts/launch_pipeline_gui.py` — deleted
- `configs/analysis_runner_gui.yaml` — new config
- `configs/launcher.yaml` — deleted
- `configs/analyse_workflow_processing_dag.yaml` — updated
- `CLAUDE.md` — updated

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] `python scripts/launch_analysis_runner_gui.py` opens without errors
- [ ] Workflow selector shows Analysis [Processing] and Analysis [Viewers]
- [ ] Clicking a workflow populates the task panel
- [ ] DAG graph renders correctly
- [ ] Console output appears when a workflow is run

---

## Rollback Plan

1. `git revert` the Phase 2 commit(s) to restore `dag_launcher` and old entry points
2. No data migrations; pure code/config change

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Import paths broken after rename | Medium | Medium | Run launch script and verify no ImportError |
| DAG config keys changed | Low | Low | Compare old `launcher.yaml` against new `analysis_runner_gui.yaml` |
