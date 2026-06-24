# Plan: Fix gradient-ridge boundary tracing the data-mask edge

**Date:** 2026-06-24
**Author:** Basil Duvernoy
**Status:** Approved
**Base Branch:** `feature/port-pipeline-gui-launcher`
**Branch:** `fix/gradient-boundary-nan-mask-spike`

---

## Overview

The gradient-ridge boundary detector (`compute_gradient_ridge`) is supposed to find the ring of
steepest slope around the RF peak.  Instead it traces the outer edge of the data/NaN mask —
the boundary of where `grid_z` has valid values — producing a contour that sits at ~77 % of
the peak response value instead of the bell-curve flank.  A one-line change in
`compute_gradient_magnitude` removes the artificial gradient spike that causes this misbehaviour,
and the sandbox diagnostic is updated to make the fix visually verifiable.

## Problem Statement

`compute_gradient_magnitude(smoothed, nan_mask)` fills NaN cells with zero before calling
`np.gradient`:

```python
filled = np.where(nan_mask, 0.0, smoothed)   # current — creates step-function at boundary
```

`nan_mask = np.isnan(grid_z)`, but `smoothed` (produced by the NaN-aware Gaussian normalisation
in `compute_laplacian_arrays`) extends valid values ~3σ pixels *beyond* the `grid_z` NaN
boundary (the "Gaussian halo").  Zeroing that halo creates a step-function discontinuity exactly
at the data edge, which is the sharpest gradient feature in the entire field.  Every radial
ray's `argmax(|∇z|)` lands on it, so the contour traces the data mask rather than the
biological RF boundary.

Evidence (session ST13-03, `sandbox_gradient_ridge_boundary.py`):
- Fig 1 panel 2: smoothed field extends well beyond the data blob in panel 1.
- Fig 1 panel 4: bright gradient ring at the data perimeter, not inside.
- Fig 2 right: all ● (chosen-radius) dots spike within 5–22 px of centre — at the mask edge.
- Fig 1 panel 5 text: "gradient ridge height = 77 % of peak" — far too high for an edge contour.

## Goals

### In Scope
1. Fix `compute_gradient_magnitude` so the gradient is computed on the smooth Gaussian halo
   rather than a 0-padded step function.
2. Confirm the fix in the sandbox by re-running all figures on ST13-03 and verifying the contour
   moves inward from the mask boundary.
3. Add a NaN-boundary overlay to `fig2_radial_profiling` so the fix (and any future regression)
   is immediately visible.

### Out of Scope
- Changing the radial-profiling algorithm or the Savitzky-Golay smoothing.
- Changing `compute_laplacian_arrays` or the inflection-boundary path.
- Reprocessing any pipeline output files (NPZ/PNG).
- Evaluating whether the gradient-ridge boundary is biologically meaningful (separate concern).

## Success Criteria

- [ ] `compute_gradient_magnitude` no longer fills the Gaussian halo with 0; it fills only cells
  that are NaN in the `smoothed` field itself.
- [ ] Re-running the sandbox on ST13-03 produces a gradient-ridge contour that lies visibly
  *inside* the white dashed NaN-boundary outline on fig2 left panel.
- [ ] Fig 2 right panel: ● dots sit at a gradient peak that follows the bell-curve profile
  (i.e., the ● radius corresponds to a region where the dashed `grid_z` line is still
  noticeably above 0).
- [ ] Fig 1 panel 5 text: gradient-ridge height % of peak is substantially lower than 77 %,
  consistent with the mid-slope of a bell curve.
- [ ] No changes to `compute_laplacian_arrays`, `compute_inflection_boundary`, or any pipeline
  orchestration code.

---

## Technical Design

### Approach

Replace the fill value in `compute_gradient_magnitude` so that the Gaussian halo is preserved
during gradient computation:

```python
# Before
filled = np.where(nan_mask, 0.0, smoothed)

# After
filled = np.where(np.isnan(smoothed), 0.0, smoothed)
```

`nan_mask` (from `grid_z`) is still passed to the function and is still used for the output
re-masking (`grad_mag[nan_mask] = np.nan`), which correctly excludes the halo from the returned
array.  The gradient computation itself now sees a smooth Gaussian continuation at the boundary
rather than a step to 0, so no artificial spike is generated.  After re-masking, the boundary-
adjacent valid cells have correct (non-spiked) gradients; halo cells are NaN in the output.

The downstream radial profiling in `_extract_ridge_via_radial_profiling` fills `grad_mag` NaN→0
before sampling.  After the fix, `grad_mag` is NaN at the halo (because `grad_mag[nan_mask]`
covers the halo), so those cells sample as 0, and `argmax` now finds the genuine inner gradient
peak.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Fill `smoothed` halo with 0 (status quo) | Simple | Creates spike → traces mask boundary | Rejected (the bug) |
| Fill with `np.isnan(smoothed)` (proposed) | One-line fix; preserves halo continuity | None identified | **Chosen** |
| Erode `nan_mask` by 1–2 px before gradient | Removes boundary-adjacent spike | Requires `scipy.ndimage.binary_erosion`; arbitrarily shrinks valid region | Rejected — over-engineered |
| Clip each ray at `grid_z` NaN boundary in radial profiling | Targeted; doesn't touch gradient computation | Needs new parameter; doesn't fix root cause | Rejected — treats symptom |
| Replace NaN→0 with NaN→`nanmean(smoothed)` | Reduces (but doesn't eliminate) the step | Step from local value to global mean still creates spurious spike | Rejected |

### Architecture Changes

No new modules.  Two files change:

```
src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py   ← one-line fix
scripts/sandbox_gradient_ridge_boundary.py                             ← fig2 overlay added
```

---

## Implementation Plan

### Phase 1: Fix `compute_gradient_magnitude`
**Goal:** Remove the boundary spike at its source.

- [ ] In `rf_gradient_boundary.py` line 99, change
  `filled = np.where(nan_mask, 0.0, smoothed)` →
  `filled = np.where(np.isnan(smoothed), 0.0, smoothed)`.
- [ ] Verify `nan_mask` is still used on the output re-masking line (line 103); no change needed.

**Files Modified:**
- `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py` — L99: one-line change

**Dependencies:** None

### Phase 2: Sandbox diagnostic enhancement
**Goal:** Make the NaN-boundary vs. contour separation visually verifiable in fig2.

- [ ] In `fig2_radial_profiling` (sandbox), after `ax_map.imshow(...)`, overlay the `grid_z`
  NaN boundary using `skimage.measure.find_contours` on `(~np.isnan(grid_z)).astype(float)` at
  level 0.5.  Draw as a white dashed line (`'--'`, lw=0.8, alpha=0.6) labelled `'data boundary'`.
- [ ] Add `'data boundary'` to the map panel's legend entry.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — `fig2_radial_profiling`: NaN-boundary overlay

**Dependencies:** Phase 1 (run sandbox with both changes together)

---

## Testing Plan

### Unit Tests
- [ ] No new unit test files required.  Existing tests in `tests/` should continue to pass
  (gradient boundary is not currently tested at unit level; regression risk is minimal for a
  one-line change that only affects cells outside `grid_z`'s valid region).

### Manual Verification (primary gate)
- [ ] Re-run `python scripts/sandbox_gradient_ridge_boundary.py` on ST13-03.
- [ ] **fig2 left panel**: lime-green contour sits *inside* the white dashed NaN-boundary line.
- [ ] **fig2 right panel**: ● dots are at a radius where the dashed `grid_z` curve is still
  clearly non-zero (indicating the chosen radius is on the bell-curve flank, not at the mask
  edge).
- [ ] **fig1 panel 5 text**: gradient-ridge height % of peak is noticeably below 77 %.
- [ ] **fig3 (sigma sweep)**: contours for all σ values remain well-formed and enclosed within
  the data region.
- [ ] **fig4 (3D surface)**: gradient-ridge contour (green) is below the peak and above the
  base of the bell, not at the base edge.

### Edge Cases
- [ ] All-NaN `smoothed` input: `np.where(np.isnan(smoothed), 0.0, smoothed)` returns all-zero
  → `grad_mag` is 0 everywhere → `compute_gradient_ridge` returns `None` via the flatness check
  (existing guard, unchanged).
- [ ] `smoothed` with no NaN (e.g., full-grid data): `np.isnan(smoothed)` is all-False →
  `filled = smoothed` unchanged → behaviour identical to status quo for fully-valid grids.

---

## Documentation Plan

- [ ] No README or CLAUDE.md changes needed (this is a bug fix, not an architectural change).
- [ ] Update the docstring of `compute_gradient_magnitude` to explain why `np.isnan(smoothed)`
  is used instead of `nan_mask` for the fill.

---

## Rollback Plan

The change is a single-line edit.  To revert:

1. In `rf_gradient_boundary.py` L99, restore:
   `filled = np.where(nan_mask, 0.0, smoothed)`
2. Remove the NaN-boundary overlay lines added to `fig2_radial_profiling` in the sandbox.

No data files, NPZ outputs, or pipeline state are touched by this fix.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fix reveals that the true gradient ridge is too far inward to be useful | Low | Medium | Verify with fig3 sigma sweep; if area is wrong, address separately as a calibration issue |
| Gradient now flat inside halo → `np.gradient` produces numerical artefacts near halo edge | Low | Low | Halo values decrease smoothly; gradient is smooth; halo cells are masked NaN in output regardless |
| `skimage.measure.find_contours` import missing in sandbox | Low | Low | `skimage` is already imported transitively via `rf_inflection_boundary`; add explicit import if needed |

---

## References

- Internal scratch analysis: `C:\Users\basil\.claude\plans\analyse-in-the-background-majestic-hollerith.md`
- Diagnostic outputs: `scripts/_sandbox_gradient_ridge_out/` (fig1–fig4)
- Implementation files:
  - `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`
  - `src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py` (`compute_laplacian_arrays`)
  - `scripts/sandbox_gradient_ridge_boundary.py`
