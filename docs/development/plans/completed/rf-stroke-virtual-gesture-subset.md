# Plan: Add "stroke" Virtual Gesture Subset to Population RF Boundary Extraction

**Date:** 2026-05-25
**Author:** Basil Duvernoy
**Status:** Completed
**Started:** 2026-05-25
**Completed:** 2026-08-06
**Base Branch:** `feature/rf-center-proximal-distal-comparison`
**Branch:** `feature/rf-stroke-virtual-gesture-subset`

> **Provenance:** this plan was authored in the `social-touch-semi-controlled`
> repository on branch `feature/rf-stroke-virtual-gesture-subset`, which was never
> merged. The producer half (the synthetic `'stroke'` subset in
> `rf_boundary_preparation.py`) already existed here; the consumer half — the
> `centroid_stroke` baseline in `rf_proximal_distal_comparison_pipeline.py` — was
> ported by hand on 2026-08-06. The document is rescued here because the originating
> branch is being deleted along with that repository's `analysis/` package.

---

## Overview

Add a `"stroke"` virtual gesture subset to the `extract_population_rf_response_field_boundaries`
pipeline task. This subset combines `stroke_proximal` and `stroke_distal` touches (excluding `tap`
and `stroke_unknown`), producing a stroke-only aggregate heatmap with boundary detection. The
downstream `compare_rf_center_proximal_distal` task switches from `"all"` to `"stroke"` as its
reference baseline, removing tap contamination from centroid offset calculations.

## Problem Statement

The center comparison pipeline computes proximal/distal centroid offsets relative to the `"all"`
gesture aggregate, which includes tap touches. Since tap RF fields have different spatial
characteristics than strokes, using `"all"` as the reference contaminates the baseline when the
goal is specifically to compare stroke direction effects.

## Goals

### In Scope
1. New `"stroke"` virtual subset in the boundary extraction pipeline (heatmap, boundary, PNGs, NPZ)
2. Include `"stroke"` in composite panel rendering (5-panel layout)
3. Switch center comparison reference from `"all"` to `"stroke"`

### Out of Scope
- GUI changes (Touch Population Explorer checkboxes) — the GUI iterates `gesture_types` from the
  NPZ and will pick up `"stroke"` automatically
- Adding `"stroke"` to `GESTURE_TYPES` in `shared_constants.py` — it is a virtual aggregate, not
  a preparation-stage classification
- Configurable reference gesture type in the DAG config — premature; can be added later if needed

## Success Criteria

- [ ] Running `extract_population_rf_response_field_boundaries` produces `*_stroke*.png` files and
  the NPZ contains `heatmap_stroke`, `boundary_centroid_uv_stroke`, `grid_u_stroke` etc.
- [ ] The `gesture_types` array in the NPZ includes `'stroke'`
- [ ] Composite PNGs show 5 panels: all, stroke, tap, stroke_proximal, stroke_distal
- [ ] Running `compare_rf_center_proximal_distal` uses `boundary_centroid_uv_stroke` as reference
- [ ] The aggregate plot axis labels read "stroke center" (not "all-gesture center")
- [ ] The summary CSV columns are `centroid_*_stroke` (not `centroid_*_all`)

---

## Technical Design

### Approach

Add `'stroke'` to the existing `subsets` list in the pipeline loop, with touch index selection via
`np.isin(pop_data.gesture_types, ('stroke_proximal', 'stroke_distal'))`. This mirrors how `'all'`
uses `np.arange()` for all touches. The rest of the pipeline (heatmap computation, boundary
detection, NPZ saving, PNG rendering) is already generic over `results.keys()` — no changes needed
beyond the loop.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `np.isin` inline in pipeline loop | Minimal change, self-contained, explicit | Slightly longer than extending `build_gesture_touch_indices` | **Chosen** |
| Extend `build_gesture_touch_indices` to accept multi-type | Reusable helper | Over-engineering for one call site; changes function signature | Rejected |
| Add `STROKE_TYPES` to `shared_constants.py` | Centralized | Only one consumer; shared_constants docstring forbids single-subsystem constants | Rejected |

### Architecture Changes

No new modules or classes. A private module-level constant `_STROKE_TYPES` is added to the pipeline
file. The `_PANEL_ORDER` list in the renderer gains one entry.

**Knowledge-base constraints respected:**
- Neuron-wide projection centroid is unchanged — gesture filtering operates on touch indices, not
  the projection origin (`note-3d-to-2d-surface-projection-algorithms.md`)
- `stroke_unknown` is explicitly excluded from the `"stroke"` subset by only including
  `stroke_proximal` and `stroke_distal` in the filter

---

## Implementation Plan

### Phase 1: Upstream Pipeline — Add "stroke" Subset (~15 min)
**Goal:** Produce stroke-only heatmaps, boundaries, PNGs, and NPZ data

- [x] Add `_STROKE_TYPES = ('stroke_proximal', 'stroke_distal')` constant at module level
- [x] Update `subsets` list from `['all'] + list(GESTURE_TYPES)` to `['all', 'stroke'] + list(GESTURE_TYPES)`
- [x] Add `elif gtype == 'stroke'` branch using `np.where(np.isin(pop_data.gesture_types, _STROKE_TYPES))[0]`
- [x] Update `_PANEL_ORDER` in renderer to `['all', 'stroke', 'tap', 'stroke_proximal', 'stroke_distal']`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — constant + subset list + elif branch
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py` — `_PANEL_ORDER`

**Dependencies:** None

### Phase 2: Downstream — Switch Center Comparison Reference (~15 min)
**Goal:** Use `"stroke"` instead of `"all"` as the reference baseline for centroid offsets

- [x] Update `_REQUIRED_GTYPES` from `('all', 'stroke_proximal', 'stroke_distal')` to `('stroke', 'stroke_proximal', 'stroke_distal')`
- [x] Rename `centroid_all` → `centroid_stroke`, load from `boundary_centroid_uv_stroke`
- [x] Rename CSV columns and dict keys from `*_all` to `*_stroke`
- [x] Update aggregate plot axis labels from `"ΔU from all-gesture center"` to `"ΔU from stroke center"`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — required gtypes, centroid loading, CSV columns
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py` — axis labels

**Dependencies:** Phase 1

### Phase 3: Documentation (~10 min)
**Goal:** Keep architecture docs in sync

- [x] Update `code/src/analysis/CLAUDE.md` — population response fields bullet (mention `"stroke"` subset) and center comparison bullet (reference changed to `"stroke"`)

**Files Modified:**
- `code/src/analysis/CLAUDE.md`

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `extract_population_rf_response_field_boundaries` with `force_processing: true` — confirm `*_stroke*.png` files appear and the NPZ contains stroke keys
- [ ] Visually inspect the stroke heatmap — it should look like `all` minus the tap contribution
- [ ] Run `compare_rf_center_proximal_distal` with `force_processing: true` — confirm the aggregate plot and CSV use stroke-based columns

### Edge Cases
- [ ] Session with no stroke touches (only taps) — the `"stroke"` subset should be skipped with a warning (existing 0-touch guard at line 208)
- [ ] Old NPZ files without `"stroke"` key — the center comparison should skip those sessions with a warning (existing missing-key guard at line 69)

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` with `"stroke"` subset and reference change

---

## Rollback Plan

All changes are additive to the NPZ format. To revert:
1. `git revert` the commit
2. Re-run `extract_population_rf_response_field_boundaries` with `force_processing: true` to regenerate NPZs without the `"stroke"` key

No data migration or schema versioning needed.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 5-panel composite too wide for display | Low | Low | Renderer scales dynamically; can split to 2 rows later if needed |
| Old NPZ files cause center comparison to skip all sessions | Med | Low | User re-runs upstream with `force_processing: true`; guard logs clear warnings |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | 15 min | None |
| Phase 2 | 15 min | Phase 1 |
| Phase 3 | 10 min | Phase 2 |
| **Total** | **~40 min** | |
