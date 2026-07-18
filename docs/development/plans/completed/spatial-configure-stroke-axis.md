# Plan: Manual Stroke-Axis GUI (`spatial_configure_stroke_axis`)

**Date:** 2026-07-09
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-07-18 09:50
**Base Branch:** `feature/spatial-tune-rf-contours`
**Branch:** `feature/spatial-configure-stroke-axis`

---

## Overview

Add a per-session manual PyQt5 GUI node, `spatial_configure_stroke_axis`, that
displays each single-touch stroke's motion direction in SLIM-UV space as arrows
coloured by their current proximal/distal label, lets the researcher **swap** the
labels and draw an editable **stroke axis** (auto-initialised from the average of
the per-stroke UV motion vectors), and persists one JSON per session. The node
sits upstream of `spatial_build_response_fields`, which consumes it via a
`use_manual_stroke_axis` toggle so the corrected labels propagate to contour
tuning, boundary extraction, and the proximal/distal comparison.

## Problem Statement

Proximal/distal stroke direction is currently decided **in 3D**, far upstream in
`touch_prepare_sessions`: `classify_gesture_type` fits a degree-1 polynomial to
`contact_location_x` over frame index and assigns `slope > 0 → stroke_proximal`,
else `stroke_distal`
(`src/analysis/touch_analytics/preparation/gesture_type.py`). That `gesture_type`
label then propagates through single-touch RF maps → response fields →
per-gesture boundaries → the proximal/distal comparison.

After SLIM-UV flattening the stroke direction is **not** aligned with any axis,
and the PCA alignment used downstream (`compute_rf_pca_alignment`, which
force-signs the major axis to +U) is data-driven, so proximal/distal routinely
**flip per session**. The only existing lever is a static `flip_u` boundary
param — there is no interactive, per-session control. This is exactly the
imbalance documented in `scripts/diagnose_stroke_direction.py`. Researchers
currently cannot inspect or correct the UV-space direction assignment, which
undermines every downstream proximal-vs-distal comparison.

## Goals

### In Scope
1. A per-session 2D-UV manual GUI showing single-touch stroke motion arrows
   coloured by current proximal/distal label, with a live **Swap labels** control.
2. An editable **stroke axis** in UV space, auto-initialised from the sign-aligned
   average of the per-stroke UV motion vectors and redrawable by rubber-band drag;
   the axis defines the direction along which proximal/distal displacement is measured.
3. Fail-fast per-session JSON persistence (axis endpoints + swap flag + timestamps),
   mirroring the `rf_contour_params_io` conventions.
4. Consumption at `spatial_build_response_fields` (via `use_manual_stroke_axis`) so
   the relabelling propagates to **contour tuning, boundary extraction, and the
   proximal/distal comparison** with no further per-stage change.
5. Full DAG wiring (output-dir constant, launcher, `@flow`, stage entry, YAML,
   layout) plus staleness coupling so editing the JSON re-triggers downstream stages.

### Out of Scope
- 3D / PyVista visualisation (2D UV matplotlib only).
- Changing the upstream 3D `classify_gesture_type` rule or the
  `touch_prepare_sessions` stage.
- Overriding the boundary-stage PCA alignment angle / `flip_u` (rejected — see
  Alternatives).
- Persisting a per-frame UV trajectory artifact (trajectories are derived at
  GUI/consumer time from the prepared CSV + SLIM cache).
- Automatic (non-interactive) axis inference as a batch stage — the node is a
  manual GUI; `interactive=false` is a no-op.

## Success Criteria

- [ ] Running `spatial_configure_stroke_axis` (interactive) opens a GUI that renders
      per-stroke UV arrows coloured by current label with the auto-init axis drawn.
- [ ] **Swap labels** flips arrow colours live; redrawing the axis re-colours strokes.
- [ ] **Validate** writes
      `4_analysed/spatial_configure_stroke_axis/<session>/<session>_stroke_axis.json`
      (schema below), greens the session combo, and preserves `created_at` on re-save.
- [ ] With `spatial_build_response_fields: {use_manual_stroke_axis: true}`, a pure
      Swap makes `heatmap_pre_threshold_stroke_proximal`/`_stroke_distal` and
      `n_touches_stroke_*` swap, while `all` and synthetic `stroke` stay byte-identical.
- [ ] `spatial_tune_rf_contours`, `spatial_extract_boundaries`, and
      `spatial_compare_proximal_distal` reflect the manual choice with no further config.
- [ ] Editing/re-validating the JSON marks `spatial_build_response_fields` stale
      (re-processes rather than skips) via `extra_input_paths`.
- [ ] `use_manual_stroke_axis: true` with a missing JSON raises `FileNotFoundError`
      (no silent fallback).
- [ ] Unit tests for the IO round-trip/fail-fast and for
      `compute_stroke_uv_motion`/`initialize_stroke_axis`/`relabel_strokes` pass.

---

## Technical Design

### Approach

Introduce the node **upstream** of `spatial_build_response_fields` and inject the
correction at the single point where per-touch proximal/distal membership is
formed — `pop_data.gesture_types`, consumed by
`build_gesture_touch_indices(gesture_types, gtype)` inside
`_compute_param_free_fields` (`data/rf_boundary_preparation.py`). Overwriting that
array at the response-fields build makes the `stroke_proximal_*` / `stroke_distal_*`
arrays in `<session>_response_fields.npz` themselves carry the correction, so every
later stage that reads the NPZ inherits it automatically.

A **single shared compute module** (`metrics/rf_stroke_axis.py`) drives both the
GUI preview and the pipeline consumer, so the two can never drift — the same
pattern as `compute_radial_foot_stages` being shared by the contour tuner and its
pipeline. Per-stroke UV motion is derived by projecting per-frame
`contact_location_{x,y,z}` into UV via `barycentric_uv_lookup` on the SLIM cache,
then a degree-1 polyfit per UV component — the 2D analogue of the existing 3D rule,
preserving its `stroke_unknown` short/NaN handling.

Applicable pipeline-engineering principles (from
`~/.claude/knowledge/data-pipeline-engineering/`): this is a **DAG-orchestrated
pipe-and-filter** stage; it is **idempotent** (skip-if-JSON-exists gate + mtime
staleness), uses **frozen DTOs** for config, keeps a **single source of truth** for
the math, and enforces **fail-fast error boundaries** (no silent fallback, per the
project rule in `docs/claude/fail-fast-pipeline.md`).

**Per-session JSON schema:**
```json
{ "session_id": "...", "axis_start_uv": [u0,v0], "axis_end_uv": [u1,v1],
  "swap_proximal_distal": false, "created_at": "...Z", "modified_at": "...Z" }
```
Endpoints (not `angle_deg`) are stored because the GUI artifact *is* a two-point
rubber-band line; direction is derived at projection time, avoiding drift.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Inject relabel at response-fields build (overwrite `pop_data.gesture_types`) | Corrects the per-gesture NPZ heatmaps → contour tuning + boundaries + comparison all inherit it with zero extra edits | Requires the node upstream of response-fields build; re-derives UV motion in the consumer | **Chosen** |
| Override boundary-stage PCA angle + `flip_u` | Localised to boundary prep | Only re-orients the shared UV frame; cannot re-assign which *touches* are proximal/distal, so heatmaps stay wrong and contour tuning (reads NPZ heatmaps) does not inherit it | Rejected |
| Correct labels at `touch_prepare_sessions` (3D) | Single upstream source | Pre-SLIM; cannot use UV motion / the SLIM cache the researcher inspects; contradicts the UV-based requirement | Rejected |
| Store axis as `angle_deg` + origin | Compact | Redundant with drawn endpoints; risks drift from the persisted line geometry | Rejected |
| Add 3D PyVista view | Richer | Out of scope; adds the deferred-VTK init complexity for no stated need | Rejected |

### Architecture Changes

New modules:
- `data/rf_stroke_axis_io.py` — path helpers, fail-fast save/load, auto-init helper
  (mirrors `rf_contour_params_io.py`).
- `metrics/rf_stroke_axis.py` — shared compute (`compute_stroke_uv_motion`,
  `initialize_stroke_axis`, `relabel_strokes`).
- `gui/rf_stroke_axis_viewer.py` — 2D-only `QMainWindow` (mirrors
  `rf_contour_tuning_viewer.py` conventions; **no** deferred-VTK pattern).

New DTOs in `data/rf_boundary_types.py`: `StrokeAxisConfig` (frozen, `__post_init__`
fail-fast; `direction()`/`signed_direction()`), `StrokeUvMotion` (mutable — arrays
`touch_keys`, `current_labels`, `start_uv`, `end_uv`, `vectors`, `valid_mask`,
`centroid_uv`).

Modified: `pipeline/output_dirs.py` (new constant), `pipelines/rf_response_fields_pipeline.py`
(consumer injection + toggle + staleness), `pipelines/rf_cluster_gui_launchers.py`
(launcher), `receptive_field_mapping/__init__.py` (re-export),
`scripts/analysis_workflow_processing.py` (`@flow`, stage entry, import),
`configs/analyse_workflow_processing_dag.yaml` (+ `.layout.json`).

**Dependency ordering:**
```
touch_prepare_sessions ──┐                 (prepared CSV: contact_location_*, gesture_type)
spatial_precompute_slim_uv ┴─► spatial_configure_stroke_axis  (NEW manual GUI → per-session JSON)
                                          │
spatial_map_single_touch ─────────────────┤
                                          ▼
                               spatial_build_response_fields  (consumer: use_manual_stroke_axis)
                                          │  regroups proximal/distal in the NPZ heatmaps
                                          ▼
              spatial_tune_rf_contours → spatial_extract_boundaries → spatial_compare_proximal_distal
```

---

## Implementation Plan

### Phase 1: Data model, IO, and shared compute
**Goal:** The persistence + math foundation, usable headless and unit-testable.

- [x] 1.1 — Add `StrokeAxisConfig` (frozen, fail-fast `__post_init__`, `direction()`,
      `signed_direction()`) and `StrokeUvMotion` DTOs to `rf_boundary_types.py`.
- [x] 1.2 — Create `data/rf_stroke_axis_io.py`: `stroke_axis_root`, `stroke_axis_path`,
      `save_stroke_axis`, `load_stroke_axis` (fail-fast), `load_stroke_axis_created_at`,
      `default_stroke_axis_config`. UTC `%Y-%m-%dT%H:%M:%SZ` stamps; preserve `created_at`.
- [x] 1.3 — Create `metrics/rf_stroke_axis.py`: `compute_stroke_uv_motion(slim_cache,
      prepared_df)`, `initialize_stroke_axis(motion)` (sign-align by current label then
      mean+normalize; raise if no valid vector), `relabel_strokes(gesture_types,
      touch_triple_keys, motion, config)`. Reuse `barycentric_uv_lookup`,
      `load_slim_uv_cache`, `SlimUvCache` from `surface/forearm_slim_uv.py`.
- [x] 1.4 — Add `SPATIAL_CONFIGURE_STROKE_AXIS = "spatial_configure_stroke_axis"` to
      `pipeline/output_dirs.py`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_boundary_types.py` — new DTOs
- `src/analysis/receptive_field_mapping/data/rf_stroke_axis_io.py` — new IO module
- `src/analysis/receptive_field_mapping/metrics/rf_stroke_axis.py` — new compute module
- `src/analysis/pipeline/output_dirs.py` — new constant

**Dependencies:** None

### Phase 2: The GUI
**Goal:** Interactive per-session viewer for inspecting/swapping/redrawing the axis.

- [x] 2.1 — Create `gui/rf_stroke_axis_viewer.py` `RFStrokeAxisViewer(QMainWindow)`:
      constructor `(sessions: list[dict], stroke_axis_root: Path, parent=None)` with
      fail-fast validation (`session_id`, `prepared_csv`, `slim_cache_path`).
- [x] 2.2 — Toolbar session combo + "Show Defined Table" toggle; left panel with info
      labels + **Swap labels** / **Recompute axis** / **Validate** buttons; central
      `QStackedWidget` (arrows canvas / Defined `QTableWidget`); status bar.
- [x] 2.3 — `_ArrowsUVCanvas`: `quiver`/`FancyArrow` per valid stroke coloured by label
      (proximal/distal fixed colours, unknown grey) + editable axis `Line2D`; rubber-band
      press/motion/release handlers + `draw_idle()` (copy `_Stages2DCanvas` pattern, store
      endpoints in UV data coords, guard on `self._toolbar.mode`).
- [x] 2.4 — `_load_current` (load SLIM cache + prepared CSV → `compute_stroke_uv_motion`
      → load JSON or `default_stroke_axis_config`), `_render` (reclassify + recolour),
      `_on_swap_clicked`, `_on_reset_axis_clicked`, `_on_validate_clicked`
      (`save_stroke_axis`, green combo via `_GREEN_BG`+`setItemData`, `QMessageBox`).

**Files Modified:**
- `src/analysis/receptive_field_mapping/gui/rf_stroke_axis_viewer.py` — new GUI

**Dependencies:** Phase 1

### Phase 3: Consumer integration + DAG wiring
**Goal:** Wire the node into the pipeline and make the relabelling propagate.

- [x] 3.1 — In `pipelines/rf_response_fields_pipeline.py`: add
      `apply_manual_stroke_axis(inputs, prepared_csv, slim_cache_path, stroke_axis_json)`
      (→ `relabel_strokes` overwriting `inputs.pop_data.gesture_types`); call it in
      `run_response_field_generation` after `load_boundary_inputs` and before
      `build_session_response_fields`, guarded by `use_manual_stroke_axis`; raise
      `FileNotFoundError` if the toggle is on and JSON absent; extend signatures with
      `use_manual_stroke_axis: bool = False`; pass `extra_input_paths=[stroke_axis_json]`
      to `boundary_stage_is_up_to_date`.
- [x] 3.2 — Add `launch_stroke_axis_viewer(missing_items, dag_defaults)` to
      `pipelines/rf_cluster_gui_launchers.py` (resolve per-session `prepared_csv` +
      `slim_cache_path` fail-fast; standard Qt bootstrap). Re-export via
      `receptive_field_mapping/__init__.py`.
- [x] 3.3 — Add `@flow spatial_configure_stroke_axis_flow(input_items,
      force_processing=False, interactive=True)` to `scripts/analysis_workflow_processing.py`
      (interactive no-op + skip-if-JSON-exists gate; then `launch_stroke_axis_viewer`);
      import the launcher; add a stage entry in `_build_pipeline_stages()` *before*
      `spatial_build_response_fields`; add `use_manual_stroke_axis` to that stage's params
      lambda.
- [x] 3.4 — `configs/analyse_workflow_processing_dag.yaml`: new
      `spatial_configure_stroke_axis` block (`category: viewer_required`,
      `depends_on: [touch_prepare_sessions, spatial_precompute_slim_uv]`, options
      `interactive`/`force_processing`); amend `spatial_build_response_fields` with
      `use_manual_stroke_axis: false` and add `spatial_configure_stroke_axis` to its
      `depends_on`. Add a node coordinate to `.layout.json`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/pipelines/rf_response_fields_pipeline.py` — consumer
- `src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — launcher
- `src/analysis/receptive_field_mapping/__init__.py` — re-export
- `scripts/analysis_workflow_processing.py` — flow, stage entry, import
- `configs/analyse_workflow_processing_dag.yaml` + `configs/analyse_workflow_processing_dag.layout.json` — node + toggle

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `tests/test_rf_stroke_axis_io.py` — save/load round-trip; fail-fast on missing
      file, missing/extra key, wrong-typed endpoints, non-bool swap, degenerate
      (start==end) axis; `created_at` preserved on re-save, `modified_at` bumped.
- [ ] `tests/test_rf_stroke_axis_relabel.py` (mirror `test_rf_response_fields_parity.py`)
      — synthetic `SlimUvCache` + prepared DataFrame: `compute_stroke_uv_motion` vectors +
      `stroke_unknown` for <2-frame/all-NaN strokes (parity with `classify_gesture_type`);
      `initialize_stroke_axis` returns a proximal-pointing unit vector; `relabel_strokes`
      is a no-op when the axis reproduces current labels, and exactly swaps
      proximal↔distal when `swap_proximal_distal` flips (taps/unknown untouched).

### Integration Tests
- [ ] Optional parity test: `run_response_field_generation(..., use_manual_stroke_axis=True)`
      with an auto-init axis (no swap) yields per-gesture heatmaps identical to the `False`
      path when the manual UV direction agrees with the 3D classifier.
- [ ] Pure-Swap propagation: after a Swap, `heatmap_pre_threshold_stroke_proximal`/
      `_stroke_distal` and `n_touches_stroke_*` swap; `all`/synthetic `stroke` unchanged.

### Manual Verification
- [ ] Run `spatial_configure_stroke_axis` (interactive) with upstream stages present;
      confirm arrows/axis render, Swap flips colours, axis redraw re-colours, Validate
      writes the JSON and greens the combo.
- [ ] Set `spatial_build_response_fields: {use_manual_stroke_axis: true,
      force_processing: true}`; run it, then `spatial_tune_rf_contours`,
      `spatial_extract_boundaries`, `spatial_compare_proximal_distal`; confirm each
      reflects the manual choice.
- [ ] Re-Validate (mtime bump); re-run `spatial_build_response_fields` with
      `force_processing:false` → re-processes rather than skips.

### Edge Cases
- [ ] Session with zero valid stroke vectors → `initialize_stroke_axis` raises; GUI
      surfaces a modal and requires manual draw (no crash, no silent default).
- [ ] `use_manual_stroke_axis: true` with missing JSON → `FileNotFoundError`.
- [ ] Strokes with all-NaN / single-frame contact_location → `stroke_unknown`, excluded
      from proximal/distal heatmaps.

---

## Documentation Plan

- [ ] Update `CLAUDE.md` / `docs/claude/architecture.md` with the new stage + its place
      in the DAG (upstream of `spatial_build_response_fields`).
- [ ] Note the `use_manual_stroke_axis` toggle in the response-fields stage docs.
- [ ] Inline docstrings on the shared compute functions describing the UV motion
      derivation and the parity with the 3D `classify_gesture_type` rule.

---

## Rollback Plan

1. **Before deployment:** the feature is inert unless
   `spatial_configure_stroke_axis.enabled: true` and
   `spatial_build_response_fields.use_manual_stroke_axis: true`; leaving the toggle
   `false` keeps the existing 3D-labelled behaviour.
2. **Data considerations:** no migrations. The only new artifact is the per-session
   JSON under `4_analysed/spatial_configure_stroke_axis/`; deleting it plus setting the
   toggle `false` fully reverts. Re-run `spatial_build_response_fields` with
   `force_processing:true` to regenerate NPZs from the 3D labels.
3. **Rollback procedure:** revert the feature commits; the new files are additive
   (delete `rf_stroke_axis_*` modules + GUI + tests) and the edits to existing files
   are localised (toggle, stage entry, YAML block).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sign-aligned mean axis initialised wrong when current labels are already flipped | Med | Med | User inspects arrows and redraws/swaps; axis is a starting point, not authoritative |
| `barycentric_uv_lookup` misses points outside the mesh (returns NaN) | Med | Med | Drop NaN UV frames; `<2` valid → `stroke_unknown` (fail-fast sentinel, warned) |
| Circular dependency if node placed after response-fields build | Low | High | Node depends only on `touch_prepare_sessions` + `spatial_precompute_slim_uv` (verified) |
| Consumer re-derivation drifts from GUI preview | Low | High | Single shared `metrics/rf_stroke_axis.py` used by both |
| DAG task-name vs output-dir-constant mismatch (`spatial_precompute_slim_uv` task vs `spatial_slim_uv` dir) | Med | Low | Use the task name in `depends_on`, the constant for paths (documented in plan) |
| Prepared CSV lacks per-frame `contact_location_*` (aggregated away downstream) | Low | High | Read the `_prepared.csv` (not `_series_augmented.csv`); fail-fast on missing columns |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — data/IO/compute | ~1 day | None |
| Phase 2 — GUI | ~1.5 days | Phase 1 |
| Phase 3 — consumer + DAG wiring | ~1 day | Phase 2 |

---

## References

- Adjacent plan: `docs/development/plans/active/spatial-tune-rf-contours.md`
  (GUI + IO + DAG-node templates this plan mirrors)
- GUI template: `src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py`
- IO template: `src/analysis/receptive_field_mapping/data/rf_contour_params_io.py`
- Consumer: `src/analysis/receptive_field_mapping/pipelines/rf_response_fields_pipeline.py`
- SLIM UV: `src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py`
- Current 3D rule: `src/analysis/touch_analytics/preparation/gesture_type.py`
- Motivating diagnostic: `scripts/diagnose_stroke_direction.py`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- configs/analyse_workflow_processing_dag.layout.json
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/spatial-configure-stroke-axis.md
- scripts/analysis_workflow_processing.py
- src/analysis/pipeline/output_dirs.py
- src/analysis/receptive_field_mapping/__init__.py
- src/analysis/receptive_field_mapping/data/rf_boundary_types.py
- src/analysis/receptive_field_mapping/data/rf_stroke_axis_io.py
- src/analysis/receptive_field_mapping/gui/rf_stroke_axis_viewer.py
- src/analysis/receptive_field_mapping/metrics/rf_stroke_axis.py
- src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py
- src/analysis/receptive_field_mapping/pipelines/rf_response_fields_pipeline.py
