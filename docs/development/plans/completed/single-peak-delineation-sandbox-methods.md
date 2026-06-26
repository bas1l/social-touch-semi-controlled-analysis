# Plan: Port the four single-peak delineation methods into the RF-boundary sandbox

**Date:** 2026-06-25
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-26 23:09
**Base Branch:** `feature/foot-of-mountain-boundary-methods`
**Branch:** `feature/single-peak-delineation-sandbox`

---

## Overview

`docs/single_peak_delineation_plan.md` is a self-contained brief that defines **four**
defensible boundary definitions for a single soft peak in a 2-D scalar field and recommends
computing all four for comparison rather than committing to one. This plan ports those four
methods into the existing RF-boundary sandbox (`scripts/sandbox_gradient_ridge_boundary.py`)
so they can be overlaid and compared on real RF heatmap sessions. Method 4.1 already exists in
the sandbox and is reused; the work is to add three new extractors (4.2, 4.3, 4.4) into the
proven per-method integration seam.

## Problem Statement

The sandbox currently prototypes several boundary candidates, but most are **radial** ray-casts
that assume a star-convex single mountain. The single-peak brief offers shape-faithful,
non-radial definitions (spill point, LoG blob) plus a reproducible parametric reference
(iso-fit) that the sandbox lacks. Without these in the comparison figure, the task owner cannot
visually judge which boundary definition to eventually promote into the production `metrics/`
package. The RF population firing-rate heatmap `grid_z` (150×150 NaN-masked scalar field in
forearm UV space) is exactly the "single soft peak" the brief assumes, so the methods apply
directly.

## Goals

### In Scope
1. Add `extract_foot_spill_point` (method 4.2 — superlevel-set area knee).
2. Add `extract_foot_log_blob` (method 4.3 — scale-space LoG zero-crossing ring).
3. Add `extract_foot_iso_fit` (method 4.4 — 2-D Gaussian fit + half-max/FWHM contour).
4. Wire all three (plus the already-present 4.1 `extract_foot_curvature`) into the
   `fig6_foot_methods` comparison panel with tunable `FOOT_PARAMS` and optional diagnostics.

### Out of Scope
- Re-implementing method 4.1 — `extract_foot_curvature` already is the radial curvature
  inflection; it is reused, not rebuilt.
- The monolithic `delineate_peak()` dict architecture from the brief's Section 6 — the sandbox
  uses one `extract_foot_*` per method; that is the right altitude for a comparison sandbox.
- Full topological persistence pruning (brief Section 5) — overlaps the already-present
  `extract_foot_prominence`; not needed for a single clean RF peak.
- Any change to `metrics/`, the Prefect pipeline, configs, or automated tests — promotion of a
  winning method is a separate follow-up plan (mirrors the `foot-of-mountain-boundary-methods`
  convention: prototype in sandbox, promote later).

## Success Criteria

- [x] Three new extractor functions exist in `scripts/sandbox_gradient_ridge_boundary.py`, each
      returning an `(N,2)` `(row,col)` contour and raising `ValueError` on degeneracy.
- [x] `fig6_foot_methods` overlays the three new contours (distinct colours) alongside the
      existing rings, with per-method mean radius and height-%-of-peak in the text panel.
- [ ] On the hardcoded sessions the new contours are **closed, enclose the peak, and sit inside
      the white data boundary**; spill point / iso-fit land in a plausible fraction of peak and
      LoG blob yields a smooth roughly-circular ring with a sensible σ.
- [ ] On a flat/degenerate or all-NaN field every new extractor raises `ValueError` (no default
      ring).
- [x] No new third-party dependency (`scipy` / `scikit-image` already supply everything).

---

## Technical Design

### Approach

Each method is implemented as an `extract_foot_<name>(grid_z[, smoothed], peak_rc, **params)
-> (N,2) (row,col) contour` function — the exact contract the existing extractors
(`extract_foot_curvature/kneedle/threshold/watershed/prominence`) already follow, each raising
a descriptive `ValueError` on degeneracy (fail-fast, no fallback ring). They are then added as
rows to the `methods` list in `fig6_foot_methods`. The figure's overlay loop and
`contour_height` annotation already handle arbitrary closed contours, and non-radial contours
auto-skip the per-ray marker via the existing `len(contour) != n_angles` guard (as
`threshold_foot` does today), so the integration is isolated and reuses every helper.

Reused helpers (all already present): `find_peak_location`, `compute_laplacian_arrays`
(NaN-aware Gaussian smoothing + Laplacian), `find_contours` (imported), `contour_height`, and
the point-in-polygon "select the contour enclosing the peak" pattern from
`extract_foot_threshold` (lines ~524–547, via `matplotlib.path.Path`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-method `extract_foot_*` functions feeding `fig6` | Matches existing seam; isolated; reuses all helpers | Diagnostics live in figure text, not a return dict | **Chosen** |
| Monolithic `delineate_peak()` returning a contours+diagnostics dict (brief §6) | One entry point; structured diagnostics | Conflicts with sandbox pattern; larger surface; over-engineered for a comparison sandbox | Rejected |
| Re-implement method 4.1 fresh | Literal port of the brief | Duplicates `extract_foot_curvature`; violates reuse convention | Rejected (reuse existing) |
| Add a persistence library (`gudhi`) for §5 pruning | Battle-tested TDA | New heavy dependency; against minimal-dep convention; not needed for one clean peak | Rejected |

### Architecture Constraints

- **Sandbox-only.** No edits to `metrics/` or the pipeline (repo convention: prototype in
  sandbox, promote later).
- **Fail-fast — no fallbacks.** Per `CLAUDE.md`, every extractor raises a descriptive
  `ValueError` on degeneracy rather than returning a default ring or sentinel. The NaN-fill in
  the LoG method is a documented model assumption (asserted against an all-NaN field), not a
  silent fallback.
- **Authorship:** Basil only; no `Co-Authored-By` trailer.

### Architecture Changes

New functions added to `scripts/sandbox_gradient_ridge_boundary.py`:
`extract_foot_spill_point`, `extract_foot_log_blob`, `extract_foot_iso_fit`, plus three new
`FOOT_PARAMS` sub-dicts and three new rows in the `fig6_foot_methods` `methods` list. New
imports: `scipy.ndimage.label`, `scipy.ndimage.gaussian_laplace`, `scipy.optimize.curve_fit`.
No new modules, no interface changes elsewhere.

---

## Implementation Plan

### Phase 1: Spill-point extractor (method 4.2)
**Goal:** Non-radial boundary at the level where the peak's component floods into the background.

- [x] 1.1 Add `extract_foot_spill_point(grid_z, smoothed, peak_rc, n_levels=200, detrend=False)`.
- [x] 1.2 Work on `smoothed` with `valid = ~np.isnan(grid_z)`. Optionally detrend a planar
      background first (exposed knob, default off); fail-fast if the plane fit is degenerate.
- [x] 1.3 Sweep threshold `t` over `n_levels` from just below peak toward background; at each
      `t`, label components (`ndi_label`), take the one containing `peak_rc`, record its area.
- [x] 1.4 Spill level = `t` at `argmax(dArea/d(-t))`; boundary = `find_contours(field,
      spill_level)` selecting the polygon enclosing the peak. Raise `ValueError` if no component
      grows, the knee is undefined, or no contour encloses the peak.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor + `scipy.ndimage.label` import.

**Dependencies:** None.

### Phase 2: LoG-blob extractor (method 4.3)
**Goal:** Scale-space characteristic-size ring via normalized Laplacian-of-Gaussian.

- [x] 2.1 Add `extract_foot_log_blob(grid_z, peak_rc, sigma_min=2.0, sigma_max=30.0, n_sigma=20)`.
- [x] 2.2 Build a NaN-filled copy (NaNs → valid-region minimum / background) so
      `gaussian_laplace` does not propagate NaN; assert against an all-NaN field.
- [x] 2.3 For each σ, `resp = sigma**2 * gaussian_laplace(filled, sigma)`; pick σ* maximizing
      `-resp[peak_rc]`. Record `log_characteristic_sigma = σ*`.
- [x] 2.4 Boundary = LoG zero-crossing ring at σ*: `find_contours(resp_sigma_star, 0.0)`
      selecting the ring enclosing the peak. Raise `ValueError` if the response never peaks in
      range or no zero-crossing ring encloses the peak.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor + `gaussian_laplace` import.

**Dependencies:** None (independent of Phase 1).

### Phase 3: Iso-fit extractor + figure wiring (method 4.4)
**Goal:** Reproducible parametric reference contour, plus comparison-panel integration.

- [x] 3.1 Add `extract_foot_iso_fit(grid_z, peak_rc, iso_level=0.5, model="gaussian")`: fit a
      2-D elliptical Gaussian (amplitude, x0, y0, σx, σy, θ, offset) to valid pixels via
      `curve_fit`, seeded from `peak_rc`/`nanmax`/rough width. Leave a `super_gaussian` option
      stubbed.
- [x] 3.2 Evaluate the fitted surface; contour at `offset + iso_level*amplitude` via
      `find_contours`, select the ring enclosing the peak. Compute fit R² for diagnostics. Raise
      `ValueError` if `curve_fit` fails to converge or no contour encloses the peak.
- [x] 3.3 Call the three new extractors in `fig6_foot_methods` and add three `methods` rows
      (e.g. spill point = yellow, LoG blob = white, iso-fit = springgreen). Optionally extend the
      text panel with diagnostics (`log_characteristic_sigma`, `spill_level`, iso-fit R²).
- [x] 3.4 Add `FOOT_PARAMS` sub-dicts (`spill_point`, `log_blob`, `iso_fit`) and thread them
      through the `fig6_foot_methods(...)` call in `main`, mirroring the existing
      `curvature`/`threshold` knob passing; extend the `fig6_foot_methods` signature.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor, `methods` rows, params,
  `scipy.optimize.curve_fit` import.

**Dependencies:** Phases 1 & 2 (for the figure to overlay all three).

---

## Testing Plan

### Unit Tests
- None added in this sandbox phase (out of scope; sandbox is exploratory). Tests are added when a
  method is promoted to `metrics/`, mirroring `tests/test_rf_inflection_boundary.py`.

### Integration Tests
- [ ] Run `python scripts/sandbox_gradient_ridge_boundary.py` with
      `FIGURES["fig6_foot_methods"]["show"] = True`; confirm it writes
      `scripts/_sandbox_gradient_ridge_out/fig6_foot_methods.png` without error.

### Manual Verification
- [ ] Inspect the figure: the three new contours are closed, enclose the peak, and sit inside
      the white data boundary; spill point / iso-fit in a plausible fraction of peak; LoG blob a
      smooth ring with sensible σ.
- [ ] Text panel reports each method's mean radius and height-%-of-peak (plus σ* / spill level /
      R² if diagnostics extension added).
- [ ] Re-run on ≥2 sessions (`ST13_03`, `ST14_01/02/04`) — one near-circular, one irregular RF —
      to confirm the methods diverge where the peak is asymmetric.

### Edge Cases
- [ ] Flat / degenerate or all-NaN field → each new extractor raises `ValueError` (no ring).
- [ ] Peak adjacent to the NaN data mask → contour stays inside the data boundary or raises.

---

## Documentation Plan

- [x] Inline docstrings on each new extractor (algorithm + fail-fast contract), matching the
      detail of the existing `extract_foot_*` docstrings.
- [ ] No README/CLAUDE.md change (sandbox-only; no architecture change). Revisit on promotion.

---

## Rollback Plan

1. Changes are confined to one sandbox script on a feature branch; revert the branch or the
   single commit to fully undo.
2. No data migrations, no pipeline/state changes, no public interface changes — nothing to
   reverse beyond the file edit.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NaN mask makes `gaussian_laplace` propagate NaN (4.3) | High | Med | Documented background-fill before convolution; assert against all-NaN; mask result back |
| `curve_fit` fails to converge on irregular RFs (4.4) | Med | Low | Good seeds from peak/nanmax; fail-fast `ValueError`; report R² so poor fits are visible |
| Spill-point knee blurs on tilted background (4.2) | Med | Low | Exposed `detrend` knob; sandbox-only visual review |
| New contour not closed / doesn't enclose peak | Med | Low | Reuse the enclose-the-peak selection from `extract_foot_threshold`; fail-fast if none |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (spill point) | ~0.5 day | None |
| Phase 2 (LoG blob) | ~0.5 day | None |
| Phase 3 (iso-fit + wiring) | ~0.5 day | Phases 1 & 2 |

---

## References

- Brief: `docs/single_peak_delineation_plan.md` (the four methods 4.1–4.4)
- Sibling plan: `docs/development/plans/active/foot-of-mountain-boundary-methods.md`
  (watershed/prominence/curvature-field; same sandbox, same integration seam)
- Sandbox: `scripts/sandbox_gradient_ridge_boundary.py` (existing candidates, `fig6_foot_methods`)
- Future promotion target: `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`,
  `.../rf_inflection_boundary.py`
- Tests to mirror on promotion: `tests/test_rf_inflection_boundary.py`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- docs/development/plans/active/single-peak-delineation-sandbox-methods.md
- scripts/sandbox_gradient_ridge_boundary.py
