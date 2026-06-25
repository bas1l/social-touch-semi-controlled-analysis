# Plan: Non-radial "foot of the mountain" boundary methods for RF heatmaps

**Date:** 2026-06-25
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/foot-of-mountain-boundary-sandbox`
**Branch:** `feature/foot-of-mountain-boundary-methods`

---

## Overview

Add three **non-radial** "foot of the mountain" boundary extractors to the RF-boundary
sandbox so they can be compared against the existing radial candidates on real sessions.
The RF population firing-rate heatmap `grid_z` (a 150×150 NaN-masked scalar field in UV
space) is a topographic-style map whose response peak is a "mountain"; we want to delineate
where that mountain ends. Online research shows the canonical cues for a peak's extent —
**topological prominence** (the saddle/col) and **curvature slope-break** (the
geomorphological *footslope*) — and the **watershed** catchment method, none of which the
current non-threshold candidates capture in a shape-faithful, non-radial way.

## Problem Statement

`scripts/sandbox_gradient_ridge_boundary.py` already prototypes **five** boundary
candidates, but **four of the five are radial ray-casting** from the peak — one radius per
angle (gradient ridge ~60% of peak, Laplacian inflection ~37%, curvature foot ~20–25%,
Kneedle foot ~20–25%). Radial casting assumes a **star-convex single mountain** and cannot
represent irregular, multi-lobed, or peak-with-neighbour fields. Only `threshold_foot` is
non-radial, and it is a naïve fixed-fraction iso-ring with no notion of *where the mountain
actually ends* (slope break or saddle). We need non-radial candidates grounded in the
standard topographic cues before any method is promoted into the production `metrics/`
package.

### Research summary (cues & methods — the "analyse online" deliverable)

Cues used to bound a peak in a scalar/topographic field:
- **Slope / gradient magnitude** — flank is steep; the foot is where slope decays to flat.
- **Curvature / slope break** — the foot is a convex-up break (profile-curvature
  zero-crossing → positive); the *footslope* class in the geomorphons scheme and Dikau's
  slope+curvature classification.
- **Level-set / fraction-of-peak** — iso-contour at x% of peak; in RF neuroscience the
  half-maximum (FWHM) or an SD contour of a fitted 2D Gaussian.
- **Topological prominence / persistence** — flood the surface; a peak's basin grows until
  it merges with a neighbour at a saddle (col); the contour at that level is the
  parameter-light, non-radial extent of one mountain.
- **Watershed lines** — boundaries along valleys/saddles separating competing peaks.

Sources: [Detecting Mountain Peaks & Delineating Shapes (MDPI Remote Sensing 4(3):784)](https://www.mdpi.com/2072-4292/4/3/784) ·
[Terrain feature points from DEMs via contour context](https://www.tandfonline.com/doi/full/10.1080/10106049.2024.2351904) ·
[Topological peak detection in 2D — persistent homology (sthu.org)](https://www.sthu.org/code/codesnippets/imagepers.html) ·
[skimage watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) ·
[Object representations at multiple scales from DEMs (PMC3087114)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3087114/)

## Goals

### In Scope
1. Add `extract_foot_watershed` — catchment boundary via `skimage.segmentation.watershed`
   on the inverted field, cut at saddles against neighbour peaks.
2. Add `extract_foot_prominence` — inline 0-D persistence (union-find); boundary = the
   iso-contour at the saddle/col level where the target peak merges.
3. Add `extract_foot_curvature_field` — non-radial footslope: the outer convex-up break
   enclosing the inflection basin, via `find_contours` on the curvature field.
4. Wire all three into the `fig6_foot_methods` comparison panel with tunable `FOOT_PARAMS`.

### Out of Scope
- Any change to the production metrics modules (`metrics/rf_gradient_boundary.py`,
  `metrics/rf_inflection_boundary.py`) or the Prefect pipeline — promotion of a winning
  method is a **separate follow-up plan**.
- 2D Gaussian-fit + SD-contour method (considered but de-prioritised; assumes Gaussian
  shape — the non-radial methods above are more general).
- Adding `gudhi` or any new third-party dependency.
- Automated tests for the sandbox extractors (added only on promotion to `metrics/`).

## Success Criteria

- [ ] Three new extractor functions exist in `scripts/sandbox_gradient_ridge_boundary.py`,
      each returning an `(N,2)` `(row, col)` contour and raising `ValueError` on degeneracy.
- [ ] `fig6_foot_methods` overlays all three new contours (distinct colours) alongside the
      existing gradient-ridge/curvature/kneedle/threshold rings, with per-method mean-radius
      and height-%-of-peak in the text panel.
- [ ] On the hardcoded sessions, the three new contours are **closed, enclose the peak, and
      sit inside the white data boundary**; prominence/footslope land in ~15–30% of peak.
- [ ] On a flat/degenerate field every new extractor raises `ValueError` (no default ring).
- [ ] No new package added; `scikit-image` (already a dependency) supplies watershed.

---

## Technical Design

### Approach

Implement three extractors that each consume the already-computed `grid_z` and NaN-aware
`smoothed` field and return a closed `(N,2)` pixel contour, then add one row per method to
the `methods` list in `fig6_foot_methods`. The integration seam already supports non-radial
contours: entries are `(label, contour_rc, color)`, and contours whose point count differs
from `n_angles` are handled (the per-ray marker loop skips them, exactly as for
`threshold_foot`). The height/area annotation via `contour_height` works on any closed
contour. This keeps the change isolated to the sandbox and reuses every existing helper.

Reused helpers (all in `scripts/sandbox_gradient_ridge_boundary.py` /
`metrics/rf_inflection_boundary.py`): `find_peak_location`, `sample_ray`, `contour_height`,
`compute_laplacian_arrays` (NaN-aware smoothing + Laplacian), `find_contours` (already
imported), and the point-in-polygon "select the contour enclosing the peak" pattern from
`extract_foot_threshold` (lines ~524–547).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Topological prominence (inline union-find) | Principled "mountain extent"; non-radial; parameter-light (one prominence floor) | Slightly more code; needs saddle bookkeeping | **Chosen** |
| Watershed (skimage) | Standard; handles competing peaks; no new dep | Sensitive to marker choice / over-segmentation | **Chosen** |
| Curvature-inflection field (non-radial) | Direct footslope cue; non-radial analog of existing radial curvature foot | Noisy curvature needs smoothing | **Chosen** |
| 2D Gaussian fit + SD contour | Robust; RF-standard | Assumes Gaussian shape; fails on irregular RFs | Rejected (out of scope) |
| `gudhi` persistence library | Battle-tested TDA | New heavy dependency; against minimal-dep convention | Rejected (inline union-find instead) |

### Architecture Constraints

- **Sandbox-only.** No edits to `metrics/` or the pipeline; this is a selection/comparison
  step. (Repo convention: prototype in sandbox, promote later.)
- **Fail-fast — no fallbacks.** Per `CLAUDE.md` and the standing no-fallbacks rule, every
  extractor raises a descriptive `ValueError` on degeneracy rather than returning a default
  ring or sentinel. This applies in the sandbox exactly as in pipeline modules.
- **Authorship:** Basil only; no `Co-Authored-By` trailer.

### Architecture Changes

New functions added to `scripts/sandbox_gradient_ridge_boundary.py`:
`extract_foot_watershed`, `extract_foot_prominence`, `extract_foot_curvature_field`, plus
new `FOOT_PARAMS` sub-dicts and three new rows in the `fig6_foot_methods` `methods` list.
No new modules, no interface changes elsewhere.

---

## Implementation Plan

### Phase 1: Watershed extractor (easiest first)
**Started:** 2026-06-25
**Completed:** 2026-06-25
**Goal:** Catchment-line boundary cut at saddles against neighbour peaks.

- [x] 1.1 Add `extract_foot_watershed(grid_z, smoothed, peak_rc, min_peak_distance, ...)`.
- [x] 1.2 Build markers: target peak = label 1; competing maxima from
      `skimage.feature.peak_local_max(smoothed, ...)` = distinct labels; background on the
      NaN/low region. Run `watershed(-smoothed, markers, mask=~np.isnan(grid_z))`.
- [x] 1.3 Boundary = `find_contours((labels == 1).astype(float), 0.5)`, then the
      enclosing-the-peak selection. Raise `ValueError` if empty / none enclose the peak.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor.

**Dependencies:** None.

### Phase 2: Prominence extractor (the principled one)
**Started:** 2026-06-25
**Completed:** 2026-06-25
**Goal:** Iso-contour at the saddle/col where the peak merges.

- [x] 2.1 Add `extract_foot_prominence(grid_z, smoothed, peak_rc, min_prominence_frac)`.
- [x] 2.2 Inline 0-D persistence via union-find: sort valid pixels descending; add one at a
      time; union with already-added 8-neighbours; track each pixel's component; record the
      **saddle level** at which the target peak's component merges into a higher one.
- [x] 2.3 Boundary = `find_contours(smoothed, saddle_level)` selecting the contour enclosing
      the peak. `min_prominence_frac` guards against noise merges. Raise `ValueError` if the
      peak never merges and no prominence floor resolves a level.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor.

**Dependencies:** None (independent of Phase 1).

### Phase 3: Non-radial curvature-field extractor + figure wiring
**Started:** 2026-06-25
**Completed:** 2026-06-25
**Goal:** Footslope as the outer convex-up break, plus comparison-panel integration.

- [x] 3.1 Add `extract_foot_curvature_field(grid_z, smoothed, peak_rc, ...)`: peak basin =
      connected concave region (Laplacian<0) around the peak; footslope = first convex-up
      break **outside** it, extracted via `find_contours` on the curvature/Laplacian field
      and selected to enclose the peak yet lie outside the inflection basin. Reuse
      `compute_laplacian_arrays`. Raise `ValueError` if no break encloses the peak.
- [x] 3.2 Add three rows to the `methods` list in `fig6_foot_methods` (e.g. `watershed`
      = yellow, `prominence` = white, `curvature field` = springgreen).
- [x] 3.3 Add `FOOT_PARAMS` sub-dicts (`watershed`, `prominence`, `curvature_field`) and
      thread them through the `fig6_foot_methods(...)` call in `main`, mirroring the existing
      curvature/threshold knob passing. Keep behind the existing `FIGURES` gate.

**Files Modified:**
- `scripts/sandbox_gradient_ridge_boundary.py` — new extractor, `methods` rows, params.

**Dependencies:** Phases 1 & 2 (for the figure to overlay all three).

---

## Testing Plan

### Unit Tests
- None added in this sandbox phase (out of scope; sandbox is exploratory). Tests are added
  when/if a method is promoted to `metrics/`, mirroring `tests/test_rf_inflection_boundary.py`.

### Integration Tests
- [ ] Run `python scripts/sandbox_gradient_ridge_boundary.py` end-to-end with
      `FIGURES["fig6_foot_methods"]["show"] = True`; confirm it produces
      `scripts/_sandbox_gradient_ridge_out/fig6_foot_methods.png` without error.

### Manual Verification
- [ ] Inspect the output figure: the three new contours are closed, enclose the peak, sit
      inside the white data boundary, and at/outside the lime gradient ridge.
- [ ] Text panel reports each method's mean radius and height-%-of-peak (expect
      prominence/footslope ~15–30%; watershed cutting at the saddle when a neighbour exists).
- [ ] Re-run on ≥2 sessions (one near-circular, one irregular RF) to confirm the non-radial
      methods track non-star-convex extent where the radial methods cannot.

### Edge Cases
- [ ] Flat / degenerate field → each new extractor raises `ValueError` (fail-fast), no ring.
- [ ] Peak adjacent to the NaN data mask → contour stays inside the data boundary or raises.

---

## Documentation Plan

- [ ] Inline docstrings on each new extractor (algorithm + fail-fast contract), matching the
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
| Watershed over-segments (spurious neighbour peaks) | Med | Low | Tune `min_peak_distance`; sandbox-only, visual review |
| Inline union-find persistence subtly wrong | Med | Med | Validate the saddle contour visually against iso-contours; cross-check prominence height on a known single-peak session |
| Curvature field noisy → jagged footslope | Med | Low | Reuse the NaN-aware Gaussian smoothing already in `compute_laplacian_arrays`; expose sigma |
| NaN mask interacts badly with watershed/contours | Low | Med | Mask with `~np.isnan(grid_z)`; clip contours to the data boundary; fail-fast if none enclose the peak |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (watershed) | ~0.5 day | None |
| Phase 2 (prominence) | ~1 day | None |
| Phase 3 (curvature field + wiring) | ~0.5 day | Phases 1 & 2 |

---

## References

- Sandbox: `scripts/sandbox_gradient_ridge_boundary.py` (existing 5 candidates, `fig6_foot_methods`)
- Production metrics (future promotion target): `src/analysis/receptive_field_mapping/metrics/rf_gradient_boundary.py`, `.../rf_inflection_boundary.py`
- Tests to mirror on promotion: `tests/test_rf_inflection_boundary.py`
- Prior commit: `d9d026b feat(sandbox): add foot-of-mountain RF boundary candidates and comparison figure`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- docs/development/plans/active/foot-of-mountain-boundary-methods.md
- scripts/sandbox_gradient_ridge_boundary.py
