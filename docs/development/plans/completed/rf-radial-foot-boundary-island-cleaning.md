# Plan: Island-clean the RF heatmap + add a "radial foot (snapped)" boundary method

**Date:** 2026-06-26
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-26 23:09
**Base Branch:** `feature/single-peak-delineation-sandbox`
**Branch:** `feature/rf-radial-foot-boundary`

---

## Overview

Add two entities to the production `spatial_extract_boundaries` task: (1) a
heatmap-cleaning step that keeps only the connected component containing the
global peak (sparse "islands" → NaN) before any contour is drawn, and (2) a new
`boundary_method: radial` that promotes the sandbox "radial snapped to data edge"
foot contour (computed on the Hessian λmax field) into production, with `gauss σ`
and `hess σ` exposed as DAG/GUI options.

## Problem Statement

`spatial_extract_boundaries` (Prefect flow `spatial_extract_boundaries_flow` →
`run_population_response_field_extraction`) extracts a closed boundary from each
session's population RF heatmap, currently via `gradient` or `inflection`. Two
gaps:

1. **Fragmented heatmap.** The interpolated 150×150 `grid_z` is frequently not one
   unified blob — it has sparse disconnected groups ("islands") alongside the main
   peak. Every radial/region contour mechanism then produces artifacts.
2. **Missing delineation.** The sandbox `scripts/sandbox/single_peak_contour/`
   prototyped a "radial snapped to data edge" foot contour on the Hessian λmax
   field that is not yet available as a production `boundary_method`.

## Goals

### In Scope
1. `clean_heatmap_islands(grid_z)` helper; applied to `grid_z` before contouring so
   the cleaned (peak-component-only) field is used **everywhere** — rendered PNGs,
   saved NPZ `grid_z`, and all boundary methods.
2. New `metrics/rf_radial_foot_boundary.py` (dataclass + orchestrator + serializer)
   promoting the sandbox snapped radial-foot extractor.
3. Wire `boundary_method: radial` through validation, dispatch (per-gesture +
   composite), NPZ save, DAG config, GUI dropdown, and the Prefect flow.
4. Expose `radial_gauss_sigma` (8.0) and `radial_hess_sigma` (5.0) as DAG/GUI
   options; all other radial parameters keep sandbox defaults.
5. Unit tests for the new boundary method and the cleaning helper.

### Out of Scope
- Changing the `gradient`/`inflection` methods or their defaults.
- Promoting the other sandbox extractors (spill-point, LoG-blob, iso-fit,
  region-growth).
- Adding `radial` to the separate `rf_spatial_tuning_pipeline.py` task (different
  task; can be a follow-up).
- Detrending / persistence layers from `docs/single_peak_delineation_plan.md`.

## Success Criteria

- [ ] `clean_heatmap_islands` keeps only the peak component, NaN-fills islands, and
      raises `ValueError` on an all-NaN grid.
- [ ] `compute_radial_foot_boundary` returns a `RadialFootBoundary` with a closed
      contour enclosing the peak on a clean Gaussian; returns `None` on
      all-NaN/flat/None/peak-at-border grids.
- [ ] `boundary_method: radial` runs end-to-end on a real session; the saved NPZ
      contains `radial_*` and `boundary_*` keys; the rendered heatmap is island-free.
- [ ] The GUI shows `radial_gauss_sigma`/`radial_hess_sigma` only when
      `boundary_method = radial`.
- [x] `pytest tests/test_rf_radial_foot_boundary.py tests/test_rf_inflection_boundary.py`
      passes.

---

## Technical Design

### Approach

Mirror the existing boundary-method architecture exactly. `metrics/rf_radial_foot_boundary.py`
follows `metrics/rf_gradient_boundary.py` (dataclass / `compute_*` orchestrator /
`*_to_dict`), reusing the polygon + sampling helpers from `rf_inflection_boundary`
and porting only the λmax + radial-snap math from the sandbox (production must not
import from `scripts/sandbox`, which is off the package path). Island cleaning is a
pipeline-policy step inserted right after grid interpolation, kept out of the
renderer-layer `compute_interpolated_grid`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New metrics module mirroring gradient/inflection | Matches existing seam; reuses all helpers; isolated | One more module | **Chosen** |
| Add radial logic inside `rf_gradient_boundary.py` | Fewer files | Conflates two methods; muddies the file | Rejected |
| Import the sandbox extractor directly | No port needed | Sandbox is off the package path & imports sibling sandbox pkgs; couples prod to scratch | Rejected |
| Island-clean inside `compute_interpolated_grid` | One call site | Pushes pipeline policy into the renderer layer | Rejected |
| Keep islands by size threshold | Keeps multiple large blobs | Adds a tunable threshold; not "one unified blob" | Rejected (keep peak component) |

### Architecture Changes

- **New:** `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py`
  (`RadialFootBoundary`, `compute_radial_foot_boundary`, `_compute_hessian_lmax`,
  ported `_extrapolate`/radial-snap/`_smooth_radii_circular`,
  `radial_foot_boundary_to_dict`, optional `_save_radial_snapshots`).
- **New helper:** `clean_heatmap_islands` in
  `src/analysis/receptive_field_mapping/data/rf_population_heatmap.py`.
- **Modified:** `pipelines/rf_population_response_field_pipeline.py` (imports,
  signature, validation, dual dispatch, island clean, NPZ radial block,
  `_SessionCompositeData` field, crop-guard message),
  `scripts/analysis_workflow_processing.py` (flow params + options dict),
  `configs/analyse_workflow_processing_dag.yaml` (two options + comment),
  `src/utils/gui/analysis_runner_gui/task_detail_panel.py` (dropdown + visibility +
  group).
- **Tests:** new `tests/test_rf_radial_foot_boundary.py`.

`RadialFootBoundary` carries the same 11 fields as `GradientBoundary`/
`InflectionBoundary` (`contour_uv, area_uv, perimeter_uv, circularity, centroid_uv,
peak_uv, pca_major_uv, pca_minor_uv, pca_orientation_deg, mean_iff_on_contour,
iff_at_centroid`) **plus** `lmax: np.ndarray` — the shared 11 are exactly what
`_save_boundary_fields` and `inflection_boundary_to_dict` duck-type on.

---

## Implementation Plan

### Phase 1: Heatmap island cleaning
**Goal:** One unified, island-free `grid_z` everywhere it is used.

**Started:** 2026-06-26
**Completed:** 2026-06-26

- [x] 1.1 Add `clean_heatmap_islands(grid_z) -> np.ndarray` to `rf_population_heatmap.py`:
      raise `ValueError` if all-NaN; `scipy.ndimage.label` (4-connectivity) on
      `~np.isnan(grid_z)`; peak via `np.nanargmax`; keep the peak's component on a
      copy, NaN elsewhere.
- [x] 1.2 Import it in `rf_population_response_field_pipeline.py` (block lines 25–31).
- [x] 1.3 Per-gesture pass: `grid_z = clean_heatmap_islands(grid_z)` after lines
      412–415, before `per_gesture_grids[gtype] = …` (line 416).
- [x] 1.4 Composite pass: `grid_z_g = clean_heatmap_islands(grid_z_g)` after lines
      575–578, before `precomputed_grids[gtype] = …` (line 579).

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_population_heatmap.py` — new helper.
- `src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
  — import + two insertion points.

**Dependencies:** None.

### Phase 2: New radial-foot metric module
**Goal:** Production `compute_radial_foot_boundary`.

**Started:** 2026-06-26
**Completed:** 2026-06-26

- [x] 2.1 `RadialFootBoundary` dataclass (11 shared fields + `lmax`).
- [x] 2.2 Reuse from `rf_inflection_boundary`: `find_peak_location`,
      `contour_pixels_to_uv`, `compute_polygon_area/perimeter/centroid`,
      `compute_contour_pca`, `sample_grid_along_contour`, `sample_grid_at_uv_point`,
      `compute_laplacian_arrays`.
- [x] 2.3 Port `_extrapolate` (from sandbox `contour_explorer/nan_aware.py`), the
      Hessian λmax math (`contour_explorer/operators.py` `_hessian_eigvals` +
      `op_hessian_eigval`, keeping the `try/except TypeError` guard), and
      `extract_radial_plateau_foot` snapped path + `_smooth_radii_circular`
      (`single_peak_contour/delineation.py`). Keep only the snapped `contour_rc`.
- [x] 2.4 `_compute_hessian_lmax(grid_z, gauss_sigma=8.0, hess_sigma=5.0)`:
      `smoothed = compute_laplacian_arrays(grid_z, gauss_sigma)[0]`; `ex =
      _extrapolate(smoothed)`; `hessian_matrix(ex, sigma=hess_sigma, order="rc",
      use_gaussian_derivatives=True)`; `lmax = hessian_matrix_eigvals(h)[0]`;
      re-mask `lmax[np.isnan(grid_z)] = np.nan`.
- [x] 2.5 `compute_radial_foot_boundary(grid_u, grid_v, grid_z, gauss_sigma=8.0,
      hess_sigma=5.0, n_angles=360, savgol_window=31, plateau_size=1,
      prominence=None, require_positive=True, snapshot_dir=None, snapshot_label="",
      contour_color="green")` — mirror `compute_gradient_ridge` for guards (return
      `None` on None/all-NaN/peak-None/peak-at-border), UV conversion, the 11
      metrics, and return; wrap the ported extractor in `try … except ValueError:
      return None`.
- [x] 2.6 `radial_foot_boundary_to_dict` (copy `gradient_boundary_to_dict`; same 11
      keys; `lmax` excluded); optional `_save_radial_snapshots`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py` — new.

**Dependencies:** None (independent of Phase 1).

### Phase 3: Pipeline + flow + config + GUI wiring
**Goal:** `boundary_method: radial` selectable and persisted.

**Started:** 2026-06-26
**Completed:** 2026-06-26

- [x] 3.1 `rf_population_response_field_pipeline.py`: import
      `compute_radial_foot_boundary`, `radial_foot_boundary_to_dict` (after lines
      37–41).
- [x] 3.2 Add `radial_gauss_sigma: float = 8.0, radial_hess_sigma: float = 5.0` to
      `run_population_response_field_extraction` (lines 93–108).
- [x] 3.3 Validation tuple → `("gradient", "inflection", "radial")` (line 150) +
      message (line 153).
- [x] 3.4 `_SessionCompositeData`: add
      `gesture_radial_boundaries: dict = field(default_factory=dict)` (lines 81–84);
      populate at `composite_queue.append(...)` (after line 506).
- [x] 3.5 Per-gesture dispatch (lines 401–402 declare; 436–439 select): compute
      `radial_boundary` (independent of `inflection_sigma`) when method == "radial";
      extend the selector with an `elif boundary_method == "radial"` branch.
- [x] 3.6 Composite dispatch (lines 580–596): add a `radial` branch independent of
      `inflection_sigma`.
- [x] 3.7 `_save_response_fields_npz` (lines 813–902): add param
      `gesture_radial_boundaries: dict | None = None`; add a `radial`-prefixed block
      mirroring the gradient block (882–891), storing `radial_lmax_{gtype}`; pass the
      dict from the call site (460–482).
- [x] 3.8 Generalize crop-guard message (lines 670–673) to "requires a boundary".
- [x] 3.9 `scripts/analysis_workflow_processing.py`: add `radial_gauss_sigma`/
      `radial_hess_sigma` to `spatial_extract_boundaries_flow` (448–491) and forward
      from the options dict (1639–1659) with defaults 8.0 / 5.0.
- [x] 3.10 `configs/analyse_workflow_processing_dag.yaml` (186–198, ruamel.yaml):
      add `radial_gauss_sigma: 8.0`, `radial_hess_sigma: 5.0`, and a comment on
      `boundary_method` listing `gradient | inflection | radial`. Keep default
      `gradient`.
- [x] 3.11 `task_detail_panel.py`: add `("Radial foot (snapped)", "radial")` to
      `_ENUM_OPTIONS["boundary_method"]` (129–132); add
      `"radial_gauss_sigma": {"radial"}`, `"radial_hess_sigma": {"radial"}` to the
      `boundary_method` entry of `_OPTION_VISIBILITY` (135–138); classify both as
      `"method"` in `_OPTION_GROUP_OF` (148–166). Float options auto-render via
      `_make_scalar_edit_handler` — no widget code.

**Files Modified:**
- `src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
- `scripts/analysis_workflow_processing.py`
- `configs/analyse_workflow_processing_dag.yaml`
- `src/utils/gui/analysis_runner_gui/task_detail_panel.py`

**Dependencies:** Phases 1 & 2.

### Phase 4: Tests
**Goal:** Cover the new method and the cleaning helper.

**Started:** 2026-06-26
**Completed:** 2026-06-26

- [x] 4.1 New `tests/test_rf_radial_foot_boundary.py` mirroring
      `tests/test_rf_inflection_boundary.py` (reuse `_stub` preamble +
      `_make_gaussian_grid`).

**Files Modified:**
- `tests/test_rf_radial_foot_boundary.py` — new.

**Dependencies:** Phases 1–3.

---

## Testing Plan

### Unit Tests
- [x] Circular Gaussian → not None; `contour_uv` (N,2); circularity ≥ 0.7; centroid
      & peak near grid center; area/perimeter > 0; `mean_iff_on_contour <
      iff_at_centroid`, both finite (no inflection exp(-1) assertion — the foot lands
      farther out). Also asserts the contour encloses the peak.
- [x] Elliptical Gaussian → `pca_major > pca_minor`, ratio > 1.5. (A ~3:1 elongated
      footprint is used: vertex-PCA of a contour understates the aspect ratio, the
      same caveat the inflection test notes, so a 2:1 σ on a circular footprint
      lands at ~1.27.)
- [x] Serialization round-trips `json.dumps`; key set == the 11 shared keys (no
      `lmax`); no NaN, no numpy types.
- [x] `clean_heatmap_islands`: main blob + NaN-walled secondary blob → secondary
      NaN'd, peak component kept, peak cell preserved; all-NaN → `ValueError`.
      (Also asserts cleaning returns a copy / does not mutate the input.)

### Integration Tests
- [x] `pytest tests/test_rf_radial_foot_boundary.py tests/test_rf_inflection_boundary.py`
      passes (27 + 35 = 62 passed).

### Manual Verification
- [ ] GUI runner (`python scripts/launch_analysis_runner_gui.py`): set
      `spatial_extract_boundaries.boundary_method = radial`, confirm
      `radial_gauss_sigma`/`radial_hess_sigma` appear only then; `force_processing:
      true`; run one session (e.g. `ST13-03`); inspect
      `4_analysed/spatial_extract_boundaries/iff_mean/<session>/` — island-free
      heatmap with the red radial-foot contour; NPZ has `radial_*` + `boundary_*`.
- [ ] Cross-check against `python scripts/sandbox_single_peak_contour.py` (radial,
      gauss σ=8) — production contour ≈ sandbox snapped foot.

### Edge Cases
- [ ] all-NaN / flat / None grid_z / peak-at-border → `compute_radial_foot_boundary`
      returns None.
- [ ] Grid whose only valid region is a single tiny island at the peak → cleaning
      keeps it; contour either encloses peak or method returns None (no crash).

---

## Documentation Plan

- [ ] No README change. Note the new `radial` method + the two sigma options in the
      `spatial_extract_boundaries` config comment (Phase 3.10).
- [ ] Inline docstrings on the new module + helper (algorithm + fail-fast contract),
      matching the gradient/inflection modules.
- [ ] Move this plan to `completed/` on ship (via `/plan-finish`).

---

## Rollback Plan

1. **Before merge:** changes are confined to one feature branch; revert the branch.
2. **Data considerations:** no migration. Reruns with `force_processing` rewrite
   `grid_z`/PNGs/NPZ for processed sessions — to restore prior outputs, revert the
   code and re-run with `force_processing`. Default `boundary_method` stays
   `gradient`, so behavior is unchanged unless explicitly set to `radial`.
3. **Procedure:** revert the feature commits; the new module/test files are deleted;
   no public interface of `gradient`/`inflection` changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `hessian_matrix(use_gaussian_derivatives=...)` kwarg absent in installed skimage | Low | Med | Keep ported `try/except TypeError` fallback |
| λmax sign convention wrong | Low | Med | `hessian_matrix_eigvals` returns descending → `eigs[0]` is λmax; `require_positive` keys on the positive foot ring; verify in unit test |
| Inner extractor `ValueError` escapes the orchestrator | Med | Med | Wrap call in `try … except ValueError: return None`; covered by edge-case tests |
| Island cleaning changes existing saved outputs | High (by design) | Low | Documented; only on `force_processing` reruns; default method unchanged |
| Cleaning isolates the peak into a too-small component | Low | Med | Method returns None on degenerate; crop-guard already requires a non-None 'all' boundary |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (island cleaning) | ~0.25 day | None |
| Phase 2 (radial module) | ~0.5 day | None |
| Phase 3 (wiring) | ~0.5 day | Phases 1 & 2 |
| Phase 4 (tests) | ~0.25 day | Phases 1–3 |

---

## References

- Sandbox: `scripts/sandbox/single_peak_contour/` (`delineation.py`, `app.py`),
  `scripts/sandbox/contour_explorer/` (`operators.py`, `nan_aware.py`)
- Pattern source: `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`,
  `.../rf_inflection_boundary.py`
- Pipeline: `src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
- Brief: `docs/single_peak_delineation_plan.md`
- Tests to mirror: `tests/test_rf_inflection_boundary.py`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rf-radial-foot-boundary-island-cleaning.md
- scripts/analysis_workflow_processing.py
- src/analysis/receptive_field_mapping/data/rf_population_heatmap.py
- src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py
- src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- src/utils/gui/analysis_runner_gui/task_detail_panel.py
- tests/test_rf_radial_foot_boundary.py
