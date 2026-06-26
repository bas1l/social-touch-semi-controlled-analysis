# Plan: "Foot of the mountain" RF boundary — sandbox investigation

**Date:** 2026-06-24
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-26 23:09
**Base Branch:** `feature/dag-launcher-boundary-method-dropdown`
**Branch:** `feature/foot-of-mountain-boundary-sandbox`

---

## Overview

Prototype several "foot of the mountain" receptive-field boundary detectors in the existing
diagnostic sandbox and overlay them side-by-side with the current gradient-ridge boundary, so the
researcher can visually compare which best traces the *base* of the RF response (where the flat
background first rises into the peak) before any one is promoted to a pipeline module.

## Problem Statement

The current gradient-ridge boundary (`compute_gradient_ridge`) catches the **inflection point** —
the steepest slope on the flank of the bell-shaped response (~61% of peak height; radius ≈ σ for a
Gaussian). The researcher instead wants the **foot/toe** of the response: where the background
bends upward into the base of the mountain. For a Gaussian `A·exp(−r²/2σ²)` this is the point of
maximum *upward* curvature in the radial profile, at radius **√3·σ ≈ 1.73σ (~22% of peak)** — much
lower and wider than the inflection ring. No method in the codebase currently targets this, and the
right operationalisation ("foot") is not obvious, so it needs visual comparison on real session
data before committing to a pipeline implementation.

## Goals

### In Scope
1. Add three candidate foot-detection methods to the sandbox: profile-curvature maximum, a
   dependency-free Kneedle knee-point, and a low-fraction threshold reference.
2. Add one comparison figure overlaying all candidates plus the gradient ridge, with a per-method
   quantitative summary (mean radius px, mean height % of peak).
3. Keep everything confined to the sandbox script — investigation only.

### Out of Scope
- Any new pipeline metric module (`rf_foot_boundary.py`) or pipeline wiring — deferred to a
  follow-up plan once a method is chosen.
- Reprocessing NPZ/PNG pipeline outputs.
- Adding `kneed` (or any other) runtime dependency.
- Changing the gradient-ridge or Laplacian-inflection methods.

## Success Criteria

- [ ] Running the sandbox produces a new `fig6_foot_methods.png` with all three foot contours plus
  the gradient ridge overlaid on the `grid_z` heatmap and the white-dashed data boundary.
- [ ] All three foot contours sit **outside** (lower/wider than) the gradient-ridge ring and
  **inside** the data boundary.
- [ ] The right panel shows, for ~6 sample rays, the response profile `z(r)` with each method's
  chosen foot marked, and the curvature trace `z''(r)` for the curvature method.
- [ ] The text panel reports foot heights ≈ 20–25% of peak vs gradient ridge ≈ 60%.
- [ ] No changes outside `scripts/sandbox_gradient_ridge_boundary.py`; no new dependency.

---

## Technical Design

### Approach

Reuse the sandbox's existing radial-profiling machinery (rays cast from the peak) rather than
writing new ray code. Each foot extractor mirrors the loop in
`_extract_ridge_via_radial_profiling` (`rf_gradient_boundary.py` L112–194): cast `n_angles` rays,
pick a per-ray radius, smooth the radii circularly with Savitzky–Golay, convert to UV with
`contour_pixels_to_uv`. The only change per method is the **per-ray selection rule**:

- **Profile-curvature max** ("toe of slope", geomorphology): sample the `smoothed` profile `z(r)`
  along the ray; compute `z''(r)` via `scipy.signal.savgol_filter(deriv=2)`; restrict to radii
  beyond the per-ray inflection (max |z'|) and pick the radius of maximum positive (concave-up)
  curvature — the shoulder where the curve flattens into background. Analytically √3σ for a
  Gaussian.
- **Kneedle knee-point**: take the monotone-decreasing portion of `z(r)` from the peak; normalise
  r and z to [0,1]; subtract the straight chord from first→last sample; the knee is the `argmax` of
  that deviation (convex, decreasing). Implemented inline (≈10 lines) — no `kneed` dependency.
- **Threshold reference**: `skimage.measure.find_contours(grid_z_filled, level=frac*peak)` with
  `frac≈0.2`; select the contour whose polygon encloses the peak. A simple reference ring, not
  radial.

The existing `sample_ray()` (sandbox L248) already returns `(radii, values)` and is reused for
profile sampling and the right-panel plots.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Profile-curvature max | Principled; exact √3σ for Gaussian; reuses ray code; no new dep | Sensitive to noise in `z''` → needs smoothing | **Chosen (candidate 1)** |
| Kneedle knee-point | Robust to noise; tunable; well-established | Inline reimpl needed to avoid `kneed` dep | **Chosen (candidate 2)** |
| Low-fraction threshold | Trivial; good visual reference | Arbitrary threshold; sensitive to background level | **Chosen (candidate 3, reference only)** |
| Watershed from background | Standard segmentation | Data has hard NaN edge → basin collapses onto data-mask boundary (the artifact just fixed) | Rejected |
| `kneed` library | Canonical Kneedle | Adds a runtime dependency for an investigation | Rejected (inline instead) |

### Architecture Changes

No new modules, no pipeline changes. All additions live in
`scripts/sandbox_gradient_ridge_boundary.py`:
- 3 new extractor functions after `sample_ray`.
- 1 new figure function `fig6_foot_methods`.
- A `DRAW_FOOT_METHODS = True` toggle in the tunable block and a call in `main()`.

Read-only references (must NOT be modified):
- `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`
  (`_extract_ridge_via_radial_profiling` pattern).
- `src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
  (`contour_pixels_to_uv`, `compute_laplacian_arrays`).

---

## Implementation Plan

### Phase 1: Foot-extractor helpers
**Goal:** Add the three candidate per-ray/threshold foot detectors.

- [x] Add `extract_foot_curvature(grid_z, smoothed, peak_rc, n_angles, savgol_window)` — radial
  loop, `savgol_filter(deriv=2)` on each ray profile, max positive curvature beyond the inflection,
  circular savgol smoothing of radii, `contour_pixels_to_uv` conversion.
- [x] Add `extract_foot_kneedle(grid_z, peak_rc, n_angles, savgol_window)` — inline chord-deviation
  knee per ray, same smoothing + UV conversion.
- [x] Add `extract_foot_threshold(grid_z, peak_rc, frac=0.2)` — `find_contours` at `frac*peak`,
  select peak-enclosing contour.
- [x] Fail-fast: mirror the existing per-ray degeneracy handling; raise loudly if the whole field
  yields no foot (consistent with `_extract_ridge_via_radial_profiling` returning `None`). No
  silent default contours.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — three new functions after `sample_ray`.

**Dependencies:** None

### Phase 2: Comparison figure + wiring
**Goal:** Visualise all candidates against the gradient ridge.

- [x] Add `fig6_foot_methods(...)`: left panel = `grid_z` heatmap + white-dashed data boundary
  (reuse the fig2 overlay) + four contours (gradient ridge lime/reference, curvature, kneedle,
  threshold) + peak marker + legend; right panel = ~6 sample-ray response profiles `z(r)` with each
  method's foot marked and the `z''(r)` curvature trace; text annotation = per-method mean radius
  (px) and mean height (% of peak) via `contour_height`.
- [x] Add `DRAW_FOOT_METHODS = True` to the tunable block (~L535) and call `fig6_foot_methods`
  after the existing figure calls in `main()` (~L600).

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new figure function + `main()` wiring + toggle.

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] None — this is a diagnostic sandbox script with no unit-test surface (consistent with the
  existing sandbox, which is not unit-tested).

### Manual Verification (primary gate)
- [ ] Run `python scripts/sandbox_gradient_ridge_boundary.py` on the hardcoded ST13-03 session.
- [ ] `fig6_foot_methods.png`: all three foot contours sit outside the lime gradient ridge and
  inside the white-dashed data boundary.
- [ ] Right panel: foot dots land at ~20–25% of peak on each ray's shoulder, not at the steep
  midslope.
- [ ] Text panel: foot mean heights ≈ 20–25% vs gradient ridge ≈ 60% of peak.

### Edge Cases
- [ ] Near-Gaussian session: curvature foot radius ≈ √3 × gradient-ridge radius (theory check).
- [ ] Ray reaching the NaN/data boundary before flattening: extractor must not place the foot on
  the mask edge (clip at last valid sample; rely on the nan-mask-spike fix already in place).
- [ ] Flat/degenerate field: extractors raise loudly rather than returning a default ring.

---

## Documentation Plan

- [x] Update the sandbox module docstring (`scripts/sandbox_gradient_ridge_boundary.py` header) to
  list the new Fig 6 and the foot-of-mountain methods.
- [ ] No README/CLAUDE.md change (investigation only; no pipeline surface).

---

## Rollback Plan

All changes are additive and confined to one script. To revert:

1. Remove the three extractor functions, `fig6_foot_methods`, the `DRAW_FOOT_METHODS` toggle, and
   its call in `main()` from `scripts/sandbox_gradient_ridge_boundary.py`.

No data, NPZ, pipeline, or dependency state is touched.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `z''` curvature too noisy → jittery curvature foot | Med | Low | Savitzky–Golay smoothing on the profile and circular smoothing on the radii (reuse `SAVGOL_WINDOW`) |
| Inline Kneedle disagrees with canonical `kneed` | Low | Low | It's a reference candidate; if it wins, the promotion plan can swap in `kneed` and validate |
| Foot lands on data-mask edge for short rays | Med | Med | Clip rays at last valid sample; nan-mask-spike fix already prevents the boundary gradient artifact |
| Threshold `frac` arbitrary | Med | Low | Shown only as a reference ring; expose `frac` as a tunable constant |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~½ day | None |
| Phase 2 | ~½ day | Phase 1 |

---

## References

- Harness scratch plan: `C:\Users\basil\.claude\plans\analyse-in-the-background-majestic-hollerith.md`
- Prior fix (prerequisite): `docs/development/plans/active/fix-gradient-boundary-nan-mask-spike.md`
- Literature: Kneedle ([paper](https://raghavan.usc.edu/papers/kneedle-simplex11.pdf),
  [kneed lib](https://kneed.readthedocs.io/)); profile curvature / toe-of-slope
  ([spatial analysis](https://www.spatialanalysisonline.com/HTML/profiles_and_curvature.htm),
  [DEM landform classification](https://www.intechopen.com/chapters/55617));
  watershed ([skimage](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)).
- Implementation references:
  - `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`
  - `src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
  - `scripts/sandbox_gradient_ridge_boundary.py`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- docs/development/plans/active/foot-of-mountain-boundary-sandbox.md
- scripts/sandbox_gradient_ridge_boundary.py
