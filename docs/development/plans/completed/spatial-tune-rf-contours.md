# Plan: Manual per-(session, gesture) RF-contour parameter tuning task

**Date:** 2026-07-09
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-07-18 09:50
**Base Branch:** `dev`
**Branch:** `feature/spatial-tune-rf-contours`

---

## Overview

Add a manual pipeline task — `spatial_tune_rf_contours` — that launches an
interactive PyQt5 GUI for tuning the receptive-field contour-generation parameters
of the `spatial_extract_boundaries` stage, independently for each (session, gesture)
combination, and persists the tuned values as one JSON per combination. Then wire
`spatial_extract_boundaries` to consume those per-combination JSON overrides so each
session/gesture gets its own optimised contour instead of one global parameter set.

## Problem Statement

The radial "foot-of-mountain" contour detector is driven by five parameters
(`min_overlap_pct`, `median_filter_size`, `radial_gauss_sigma`,
`radial_hess_sigma`, `radial_envelope_smooth_sigma`). Today these live as **single
global values** in `configs/analyse_workflow_processing_dag.yaml`, applied uniformly
to every session and every gesture subset (`all`, `stroke`, `tap`,
`stroke_proximal`, `stroke_distal`). In practice the ideal parameters differ by
session (RF shape, coverage, noise) and by gesture. There is no tool to inspect the
intermediate stages (raw → smoothed → Hessian λmax → contour) against real data and
dial in per-combination values. The existing
`docs/spatial_extract_boundaries/hessian_explainer.html` illustrates the effect of
σ_gauss / σ_hess but only on a synthetic dome; it neither loads real sessions nor
persists anything.

## Goals

### In Scope
1. A per-combination JSON schema + I/O (single source of truth shared by the GUI
   writer and the boundary-stage reader), stored in a dedicated output subfolder.
2. An interactive PyQt5 GUI: session + gesture dropdowns, five parameter sliders,
   three side-by-side 2D matplotlib heatmaps (raw / smoothed / λmax) with contour
   overlay, a user-drawn arbitrarily-oriented cross-section that plots each stage's
   values along the cut, a PyVista 3D view with a stage selector, a **Validate**
   button that saves the JSON, auto-load of an existing JSON on selection, and a
   toggle to a "Defined" table (session × gesture, green when validated).
3. A `viewer_required` processing-DAG task that launches the GUI, with skip logic
   (skip when not forced and every combination already has a JSON) mirroring
   `spatial_configure_slim_uv` / `spatial_set_camera`.
4. Downstream consumption: `spatial_extract_boundaries` optionally loads the
   per-(session, gesture) JSONs and applies each combination's parameters when
   regenerating contours.

### Out of Scope
- Tuning any boundary parameter other than the five listed (e.g. `boundary_method`,
  `inflection_sigma`, `flip_u`, `cmap` remain global).
- Non-radial detectors (`gradient`, `inflection`) — the tuner targets the active
  `radial` method only.
- Automated GUI/UI tests (consistent with the repo's existing GUI plans).
- Batch/headless re-optimisation or any automatic parameter search — this is a
  manual tool.

## Success Criteria

- [ ] `spatial_tune_rf_contours` appears in the processing DAG and launches the GUI
      via `scripts/launch_analysis_runner_gui.py`.
- [ ] The GUI shows, for a selected (session, gesture), live raw/smoothed/λmax 2D
      heatmaps + arbitrary cross-section + 3D stage view, all with the current
      contour overlaid, updating on Recompute.
- [ ] **Validate** writes `…/<session>_<gesture>_contour_params.json`; the combo and
      the "Defined" table cell turn green; re-selecting the combo auto-loads it.
- [ ] With every combination already saved and `force_processing: false`, the task
      prints a skip message and opens no window; `force_processing: true` opens it.
- [ ] With `use_tuned_params: true`, `spatial_extract_boundaries` regenerates each
      contour using its per-combination JSON, and raises `ValueError` (naming the
      missing combo) if a required JSON is absent.
- [ ] With `use_tuned_params: false`, boundary output is unchanged vs. the current
      pipeline (no regression); `pytest -k boundary` passes.

---

## Technical Design

### Approach

Fuse two existing, proven templates rather than build from scratch:
- **`scripts/sandbox/single_peak_contour/app.py`** (`SinglePeakContourExplorer`) —
  already loads a real session's population-response NPZ, rebuilds the grid, runs the
  Gaussian→Hessian→radial-foot chain live, and overlays contours on linked PyVista 3D
  surfaces + 2D cross-sections. Its `data_loader.build_session_grid` composes only
  **production** functions.
- **`src/analysis/receptive_field_mapping/gui/slim_uv_config_viewer.py`**
  (`SlimUvConfigViewer`) — the production pattern for a per-item, config-saving GUI:
  combo navigation, load-on-select, **Accept/Validate** persistence, green
  highlighting of completed items, Skip-to-next, deferred VTK init.

The tuner is a new production GUI that takes the interaction model of the first and
the persistence/lifecycle model of the second.

**Single-source-of-truth for the contour math.** To guarantee the GUI preview and
the pipeline produce identical contours, expose a public
`compute_radial_foot_stages(...)` in `rf_radial_foot_boundary.py` that returns the
intermediate `smoothed` and `lmax` fields alongside the contour; both
`compute_radial_foot_boundary` (pipeline) and the GUI call it. Production must not
import from `scripts/sandbox`, so the sandbox's grid-build composition is promoted
into a small production helper.

**No dependency cycle.** The GUI consumes only *parameter-independent* raw arrays
from the boundary NPZ (`heatmap_pre_threshold_<gesture>`, `unique_count_<gesture>`,
`forearm_uv/faces/V`), so `spatial_extract_boundaries` can produce the NPZ first
(global params), the GUI writes JSONs, then a forced re-run consumes them. The tuner
`depends_on: [spatial_extract_boundaries]`; the consumer does **not** depend on the
tuner (JSONs are soft inputs) — the DAG stays acyclic.

**Fail-fast, no silent fallbacks** (repo rule + user's standing feedback). Missing or
malformed JSONs are not silently defaulted. Consumption is gated by an explicit
`use_tuned_params` flag: off = pure global params (bootstrap / current behavior); on =
strict, raising if any required combination JSON is absent. This resolves the
bootstrap-vs-consume tension without a silent fallback.

### Alternatives Considered

| Decision point | Option | Pros | Cons | Decision |
|----------------|--------|------|------|----------|
| Missing JSON on consume | **`use_tuned_params` flag; strict-raise when on** | Honors no-fallbacks rule; explicit bootstrap path | One extra flag | **Chosen** |
| | Logged fallback to global defaults | Always runnable | Silent-ish degradation; violates repo rule | Rejected |
| Param scope | **Per-gesture map `dict[gtype → GestureContourParams]`** | Both prepare & extract already loop per gesture; drops in cleanly | — | **Chosen** |
| | Flat `(session,gtype) → params` | Uniform key | Forces per-session funcs to know session_id | Rejected |
| 2D views | **matplotlib heatmap panels + arbitrary cut** | Matches user's spec & the HTML demo | New canvas widget | **Chosen** |
| | Reuse sandbox `ContourCrossSectionView` only | Less code | 1D profiles, not full field images | Rejected |
| Contour math reuse | **Shared `compute_radial_foot_stages`** | GUI == pipeline byte-for-byte | Small refactor of existing fn | **Chosen** |
| | Reimplement in the GUI | Isolated | Drift risk; duplicated math | Rejected |
| View widget source | **Port sandbox views into `src/`** | Production may not import `scripts/sandbox` | Copy cost | **Chosen** |

### Architecture Changes

Respects the existing boundary stage's pipe-and-filter layering
(`verify → load → prepare → process → render/persist`) and its DTO hand-offs; the new
per-gesture parameters travel as a plain serializable map on the existing
`PreparedBoundaryData` DTO (uniform filter boundaries). New GUI is isolated in the
`gui/` layer; new config I/O in the `data/` layer.

```
New:
  src/analysis/receptive_field_mapping/data/rf_contour_params_io.py   # schema + JSON I/O + grid helper
  src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py # the GUI
  4_analysed/spatial_tune_rf_contours/iff_<metric>/<session>/<session>_<gesture>_contour_params.json

Modified (per-gesture threading + wiring):
  data/rf_boundary_types.py, data/rf_boundary_preparation.py,
  metrics/rf_boundary_extraction.py, metrics/rf_radial_foot_boundary.py,
  pipelines/rf_population_response_field_pipeline.py,
  pipelines/rf_boundary_verification.py, pipelines/rf_cluster_gui_launchers.py,
  pipeline/output_dirs.py, scripts/analysis_workflow_processing.py,
  configs/analyse_workflow_processing_dag.yaml (+ .layout.json)
```

---

## Implementation Plan

### Phase 1: Shared config schema + JSON I/O
**Goal:** One authoritative schema + path convention for the tuned parameters.

**Completed:** 2026-07-09

- [x] Add `GestureContourParams` (frozen dataclass, the 5 params) to `rf_boundary_types.py`.
- [x] Add `SPATIAL_TUNE_RF_CONTOURS = "spatial_tune_rf_contours"` to `output_dirs.py`.
- [x] Create `data/rf_contour_params_io.py`: `contour_params_root(database_path, iff_metric)`,
      `contour_params_path(root, session_id, gtype)`, `save_contour_params(...)` (5 params +
      `session_id`/`gesture`/`created_at`/`modified_at`), `load_contour_params(path)`
      (fail-fast on missing/unknown key, wrong type, non-odd/-positive `median_filter_size`),
      `defaults_from_boundary_params(params)`, and `load_gesture_contour_params(root, session_id,
      gesture_keys, params, strict)` → `dict[gtype → GestureContourParams]` (incl. synthetic `'stroke'`).
- [x] Promote the sandbox grid-build composition (`load_session_arrays` / `build_session_grid`)
      into a production helper (in `rf_contour_params_io.py` or a new `data/rf_contour_grid.py`),
      composing only existing production functions.

**Files Modified:** `data/rf_boundary_types.py`, `pipeline/output_dirs.py`,
`data/rf_contour_params_io.py` (new).
**Dependencies:** None.

### Phase 2: Shared contour "stages" helper
**Goal:** Single source of truth for smoothed/λmax/contour so GUI == pipeline.
**Completed:** 2026-07-09

- [x] Add public `compute_radial_foot_stages(grid_u, grid_v, grid_z, gauss_sigma, hess_sigma,
      envelope_smooth_sigma) -> {smoothed, lmax, contour_uv, contour_rc, peak_rc}` to
      `metrics/rf_radial_foot_boundary.py`, exposing what `_compute_hessian_lmax` already computes.
- [x] Refactor `compute_radial_foot_boundary` to call it internally (no behavior change).

**Files Modified:** `metrics/rf_radial_foot_boundary.py`.
**Dependencies:** None (parallel to Phase 1).

### Phase 3: The interactive GUI
**Goal:** The tuning window and its launcher.

**Completed:** 2026-07-09

- [x] Create `gui/rf_contour_tuning_viewer.py` (`RFContourTuningViewer(QMainWindow)`):
  - [x] Top bar: session + gesture combos (gestures from NPZ `gesture_types`), green-highlight
        combos with an existing JSON.
  - [x] Param panel: 5 spin sliders + **Recompute** + **Validate** (`save_contour_params`); auto-load
        JSON (or `defaults_from_boundary_params`) on selection.
  - [x] 2D matplotlib `FigureCanvas`: raw/smoothed/λmax `imshow` + contour overlay; user-drawn
        arbitrary-orientation cut → 4th axes plotting each stage along the segment with contour
        crossings marked.
  - [x] 3D PyVista `QtInteractor` (deferred VTK init as in `SlimUvConfigViewer`) with a
        Raw/Smoothed/λmax radio selector + contour overlay.
  - [x] Toggle button → "Defined" `QTableWidget` (sessions × gestures, green when validated).
  - [x] Live compute reuses Phase-1 grid helper + Phase-2 `compute_radial_foot_stages`.
- [x] Add `launch_rf_contour_tuning_viewer(missing_items, dag_defaults)` to
      `pipelines/rf_cluster_gui_launchers.py`, mirroring `launch_slim_uv_config_viewer`
      (resolve each session's boundary NPZ under `spatial_extract_boundaries/iff_<metric>/`).

**Files Modified:** `gui/rf_contour_tuning_viewer.py` (new),
`pipelines/rf_cluster_gui_launchers.py`, `gui/__init__.py` (export the new viewer).
**Dependencies:** Phases 1, 2.

### Phase 4: Task wiring (processing DAG)
**Goal:** Register the manual task with skip/force logic.

**Started:** 2026-07-09
**Completed:** 2026-07-09

- [x] Add `spatial_tune_rf_contours_flow(input_items, force_processing, interactive, iff_metric,
      <5 defaults>)` to `scripts/analysis_workflow_processing.py`, modeled on
      `spatial_configure_slim_uv_flow`: `interactive=false` → no-op; skip when not forced and every
      (session × gesture) JSON exists; else launch the GUI on sessions with any missing combo.
- [x] Register it in `_build_pipeline_stages` (after the `spatial_extract_boundaries` entry).
- [x] Add the `spatial_tune_rf_contours` block to `configs/analyse_workflow_processing_dag.yaml`
      (`category: viewer_required`, `enabled: true`, `depends_on: [spatial_extract_boundaries]`,
      `options`: `force_processing`, `interactive`, `iff_metric`, 5 defaults) + node coords in
      `configs/analyse_workflow_processing_dag.layout.json`.

**Files Modified:** `scripts/analysis_workflow_processing.py`,
`configs/analyse_workflow_processing_dag.yaml`, `configs/analyse_workflow_processing_dag.layout.json`.
**Dependencies:** Phase 3.

### Phase 5: Downstream consumption
**Goal:** `spatial_extract_boundaries` applies per-combination params.

**Started:** 2026-07-09
**Completed:** 2026-07-09

- [x] Add `use_tuned_params: bool` option (default `false`) to the `spatial_extract_boundaries`
      DAG entry; thread as `contour_params_dir: Path | None` into
      `run_population_response_field_extraction` (`None` when off).
- [x] When on: per session build `per_gesture_params` via `load_gesture_contour_params(..., strict=True)`;
      pass it into `prepare_session_boundary_data(inputs, params, per_gesture_params)`; store it on
      `PreparedBoundaryData`.
- [x] `prepare_session_boundary_data` / `_build_gesture_results` / `_add_subset`: use
      `per_gesture_params[gtype].min_overlap_pct` and `.median_filter_size` (replace the global
      scalars); add a comment near the PCA-alignment block noting `'all'`'s `min_overlap_pct`
      governs the shared UV frame.
- [x] `extract_session_boundaries`: read `prepared.per_gesture_params[gtype]` for the 3 radial sigmas
      (no signature change).
- [x] Staleness: pass `input_paths + existing override JSON paths` to `boundary_stage_is_up_to_date`
      (add `extra_input_paths: list = []`); do **not** add JSONs to the 4-tuple consumed by
      `load_boundary_inputs`.

> **Byte-identical no-regression note:** `use_tuned_params: false` passes
> `per_gesture_params=None`, so `_build_gesture_results` / the grid loop /
> `extract_session_boundaries` read the global `BoundaryParams` scalars via the
> *unchanged* code path. This is deliberate: the production config sets
> `median_filter_size: null` (filter disabled → the `generic_filter` block is
> skipped entirely), a state `GestureContourParams` cannot represent (it requires
> a positive odd int) and `defaults_from_boundary_params` raises on. Routing the
> off-path through `GestureContourParams` would therefore either crash or take a
> different (size-1 `generic_filter`) code path — not byte-for-byte identical. The
> `None`-fallback branch guarantees byte-identical output.

**Files Modified:** `pipelines/rf_population_response_field_pipeline.py`,
`data/rf_boundary_preparation.py`, `data/rf_boundary_types.py`,
`metrics/rf_boundary_extraction.py`, `pipelines/rf_boundary_verification.py`,
`scripts/analysis_workflow_processing.py`, `configs/analyse_workflow_processing_dag.yaml`.
**Dependencies:** Phase 1.

---

## Addendum (2026-07-09): Shared param-free field-generation task

**Motivation.** In Phases 1–5 the tuner reads the *parameter-independent* raw
arrays (`heatmap_pre_threshold_<g>`, `unique_count_<g>`, `n_touches_<g>`,
`forearm_uv/faces/V`, `gesture_types`) out of the `spatial_extract_boundaries`
NPZ, forcing `spatial_tune_rf_contours` to `depends_on:
[spatial_extract_boundaries]` and pushing tuned params back in as *soft* inputs
(`use_tuned_params` + mtime staleness). But those arrays are **computed by** the
boundary stage's `prepare` step purely from **upstream** artifacts
(`spatial_map_single_touch`'s RF NPZ + `spatial_precompute_slim_uv`'s SLIM cache
+ merged CSV/PLY), using only production functions (`compute_rf_heatmap`,
`compute_unique_touch_count`) — nothing param-dependent. Hoisting that
computation into a dedicated upstream task makes the fields a single source of
truth that **both** the tuner and the extractor consume, and inverts the awkward
dependency so tuned params flow *forward* into the extractor as real inputs.

**New acyclic wiring.**
```
spatial_map_single_touch ─┐
spatial_precompute_slim_uv ┴─► spatial_build_response_fields ─► <session>_response_fields.npz
                                          │                                │
                                          ▼                                ▼
                                spatial_tune_rf_contours ──►  spatial_extract_boundaries
```
- `spatial_build_response_fields` `depends_on: [spatial_map_single_touch, spatial_precompute_slim_uv]`.
- `spatial_tune_rf_contours` `depends_on: [spatial_build_response_fields]` (was `[spatial_extract_boundaries]`).
- `spatial_extract_boundaries` `depends_on: [spatial_build_response_fields, spatial_tune_rf_contours]`.

`use_tuned_params` is retained: it still gates strict-consume (on) vs. pure-global
bootstrap (off), which is required because the tuner is a manual/interactive task
and no JSONs exist on a fresh run. The new task only relocates the param-free
generation; it does not change any numerics (identical production functions on
identical upstream inputs ⇒ byte-identical fields).

### Phase 6: Shared param-free field-generation task
**Goal:** One upstream task materialises the param-free per-(session, gesture) fields.

**Started:** 2026-07-09
**Completed:** 2026-07-09

- [x] Add `SPATIAL_BUILD_RESPONSE_FIELDS = "spatial_build_response_fields"` to `output_dirs.py`.
- [x] Extract the param-free portion of `_build_gesture_results` / `_add_subset`
      (the `compute_rf_heatmap` + `compute_unique_touch_count` + SLIM-map + `n_touches`
      steps, *excluding* thresholding/interpolation) into a reusable production helper
      in `data/rf_boundary_preparation.py` (e.g. `build_param_free_fields(inputs) ->
      dict[gtype -> (slim_raw, slim_unique_count, n_touches)]`), so both the new task
      and the extractor call it (no duplicated math).
- [x] Create `pipelines/rf_response_fields_pipeline.py`: `build_session_response_fields(inputs)`
      (compose `build_param_free_fields` + mesh) and `run_response_field_generation(input_items,
      force_processing, ...)`; save `<session>_response_fields.npz` under
      `SPATIAL_BUILD_RESPONSE_FIELDS/` with keys `heatmap_pre_threshold_<g>`,
      `unique_count_<g>`, `n_touches_<g>`, `forearm_uv`, `forearm_faces`, `forearm_V`,
      `gesture_types`, `session_id`. Reuse `resolve_boundary_input_paths` /
      `load_boundary_inputs`. Fail-fast on missing upstream inputs; add an up-to-date
      (mtime) skip mirroring the boundary stage.
- [x] Add `spatial_build_response_fields_flow(...)` to `scripts/analysis_workflow_processing.py`
      and register it in `_build_pipeline_stages` **before** `spatial_extract_boundaries`.
- [x] Add the `spatial_build_response_fields` block to
      `configs/analyse_workflow_processing_dag.yaml` (`category: automated`, `enabled: true`,
      `depends_on: [spatial_map_single_touch, spatial_precompute_slim_uv]`) + node coords in
      `configs/analyse_workflow_processing_dag.layout.json`.

**Files Modified:** `pipeline/output_dirs.py`, `data/rf_boundary_preparation.py`,
`pipelines/rf_response_fields_pipeline.py` (new), `scripts/analysis_workflow_processing.py`,
`configs/analyse_workflow_processing_dag.yaml` (+ `.layout.json`).
**Dependencies:** Phases 1, 5.

### Phase 7: Rewire consumers onto the shared fields
**Goal:** Tuner and extractor both read the new NPZ; dependency edges inverted.

**Started:** 2026-07-09
**Completed:** 2026-07-09

- [x] Tuner launcher (`pipelines/rf_cluster_gui_launchers.py`) + flow: resolve each session's
      NPZ from `SPATIAL_BUILD_RESPONSE_FIELDS/<session>/<session>_response_fields.npz` (was
      `spatial_extract_boundaries/iff_<metric>/`; iff-metric subfolder dropped — the param-free
      arrays are metric-independent); the JSON-output convention `contour_params_root(db,
      iff_metric)` is unchanged. Changed the DAG `spatial_tune_rf_contours` `depends_on` to
      `[spatial_build_response_fields]`. Verified the tuner grid helper
      (`load_session_arrays`/`build_session_grid`) reads only keys present in the new NPZ.
- [x] Extractor: added `load_param_free_fields_npz` (+ `resolve_response_fields_npz`) to
      `rf_boundary_io.py`; the pipeline reads the param-free arrays from the shared NPZ and threads
      them into `prepare_session_boundary_data(..., param_free=...)`, which now delegates
      thresholding via the slimmed `_build_gesture_results(param_free, params, per_gesture_params)`.
      SLIM vertex colours + the PCA/interpolation mesh are still derived from the raw PLY/slim-cache
      inputs (not in the shared NPZ), so no NPZ extension was needed. **Byte-identical** parity
      asserted by `tests/test_rf_response_fields_parity.py` (recompute vs. NPZ round-trip).
- [x] DAG: `spatial_extract_boundaries` `depends_on: [spatial_build_response_fields,
      spatial_tune_rf_contours]`; the response-fields NPZ is added to the boundary stage's staleness
      inputs (`extra_input_paths`). `use_tuned_params` semantics unchanged; gesture-key enumeration
      for strict-consume now reads the shared NPZ (no longer requires a prior boundary NPZ).
- [x] Updated `.layout.json`: repositioned `spatial_tune_rf_contours` to `[2528.0, 776.0]` so it
      sits between `spatial_build_response_fields` and `spatial_extract_boundaries` (edges are
      auto-derived from `depends_on`).

**Files Modified:** `pipelines/rf_cluster_gui_launchers.py`,
`data/rf_boundary_preparation.py`, `pipelines/rf_population_response_field_pipeline.py`,
`pipelines/rf_boundary_verification.py`, `scripts/analysis_workflow_processing.py`,
`configs/analyse_workflow_processing_dag.yaml` (+ `.layout.json`).
**Dependencies:** Phase 6.

---

## Testing Plan

### Unit Tests
- [ ] `save_contour_params` → `load_contour_params` round-trips the 5 params.
- [ ] `load_contour_params` raises on missing key, unknown key, wrong type, and even/non-positive
      `median_filter_size`.
- [ ] `load_gesture_contour_params(strict=True)` raises (naming the combo) when a JSON is absent;
      includes the synthetic `'stroke'` key.
- [ ] `compute_radial_foot_stages` returns fields whose contour equals the existing
      `compute_radial_foot_boundary` output on a fixture grid (no drift).

### Integration Tests
- [x] `use_tuned_params: false` → boundary NPZ/PNGs unchanged vs. baseline on a fixture session.
      (Covered by `tests/test_rf_response_fields_parity.py`: the param-free fields read from the
      shared NPZ are byte-identical to recomputing them, and the thresholded per-gesture results
      match — the only thing that changed between paths — so all downstream output is unchanged.)
- [ ] `use_tuned_params: true` with per-combo JSONs → each gesture's contour reflects its params.
- [ ] Editing one JSON marks the boundary stage stale (mtime gate) and regenerates only affected
      sessions.

### Manual Verification
- [ ] Launch via `scripts/launch_analysis_runner_gui.py`: 3 heatmaps + arbitrary cut + 3D stage
      switch redraw on Recompute with contour overlays.
- [ ] Validate writes the JSON; combo + Defined-table cell turn green; re-selecting auto-loads it.
- [ ] Skip logic: all JSONs present + not forced → no window; forced → window opens.

### Edge Cases
- [ ] Gesture subset with zero touches (already skipped upstream) does not require a JSON.
- [ ] Deleting a JSON to "revert" is not mtime-detectable — documented limitation; `force_processing:true`
      always regenerates.
- [ ] All-NaN / empty heatmap raises cleanly in the GUI (no silent blank view).

---

## Documentation Plan

- [ ] Update `docs/spatial_extract_boundaries/README.md`: note per-(session,gesture) tuning +
      `use_tuned_params` consumption.
- [ ] Update `CLAUDE.md` / `docs/claude/architecture.md` if the new task/output dir warrants a mention.
- [ ] Add a short usage note for the tuner GUI (how to launch, Validate, Defined table).

## Rollback Plan

1. Pure additive code + config. To revert, `git revert` the feature commits; the new task is
   `enabled`-gated and `use_tuned_params` defaults to `false`, so disabling either restores prior
   behavior without a revert.
2. No data migrations. The `spatial_tune_rf_contours/` output folder and any JSONs can be deleted
   independently; with `use_tuned_params: false` they are ignored.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GUI preview diverges from pipeline contour | Medium | High | Shared `compute_radial_foot_stages`; unit test asserting parity |
| PyVista/VTK init crash on Windows | Medium | Medium | Deferred `_deferred_start` init copied verbatim from `SlimUvConfigViewer` |
| `'all'` overlap shifts every gesture's UV frame unexpectedly | Medium | Medium | Document the coupling; surface in GUI (it recomputes from the same `'all'` frame) |
| Silent fallback creeping in on missing JSON | Low | High | Strict `use_tuned_params` gate; raise with combo name; covered by test |
| Importing from `scripts/sandbox` in production | Low | Medium | Port views + promote grid helper into `src/`; no sandbox import |

## References

- Related plans (completed): `foot-of-mountain-boundary-methods.md`,
  `rf-radial-foot-boundary-island-cleaning.md`, `dag-launcher-boundary-method-dropdown.md`,
  `port-pipeline-gui-launcher.md`.
- `docs/spatial_extract_boundaries/README.md`, `docs/spatial_extract_boundaries/hessian_explainer.html`.
- Templates: `scripts/sandbox/single_peak_contour/app.py`,
  `src/analysis/receptive_field_mapping/gui/slim_uv_config_viewer.py`.

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- configs/analyse_workflow_processing_dag.layout.json
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/spatial-tune-rf-contours.md
- scripts/analysis_workflow_processing.py
- src/analysis/pipeline/output_dirs.py
- src/analysis/receptive_field_mapping/__init__.py
- src/analysis/receptive_field_mapping/data/rf_boundary_io.py
- src/analysis/receptive_field_mapping/data/rf_boundary_preparation.py
- src/analysis/receptive_field_mapping/data/rf_boundary_types.py
- src/analysis/receptive_field_mapping/data/rf_contour_params_io.py
- src/analysis/receptive_field_mapping/gui/__init__.py
- src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py
- src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py
- src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py
- src/analysis/receptive_field_mapping/pipelines/rf_boundary_verification.py
- src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py
- src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- src/analysis/receptive_field_mapping/pipelines/rf_response_fields_pipeline.py
- tests/test_rf_radial_foot_boundary.py
- tests/test_rf_response_fields_parity.py
