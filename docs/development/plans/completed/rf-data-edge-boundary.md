# Plan: Data-Edge RF Boundary (peak at the mapped-region border)

**Date:** 2026-07-18
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-07-18 12:32
**Base Branch:** `feature/rf-contour-failure-diagnostics`
**Branch:** `feature/rf-contour-failure-diagnostics` (implement-here — stacks on the uncommitted failure-diagnostic work; NO new branch)

---

## Overview

When a receptive-field peak sits at the edge of the mapped forearm region, the radial
foot-of-mountain extractor cannot trace a closed contour (it assumes an interior peak
with field on all sides) and raises `ContourExtractionError(SEED_OUTSIDE_FOOTPRINT)`.
This feature adds an explicit **boundary-closed** path: build the contour from the
foot-of-mountain arc over the interior directions, closed by walking the mapped-region
(valid-data) edge between the two transition points, then reuse the existing footprint
envelope to produce a valid closed polygon comprising the hotspot.

## Problem Statement

The mapped forearm region is a **real physical limit** (not an arbitrary crop). RFs whose
hotspot runs into that limit are, per the user, "not uncommon." Today they fail extraction
outright: rays pointing off the region collapse (radius 1.0), the star polygon does not
enclose the peak, and `envelope_contour_to_footprint` raises `SEED_OUTSIDE_FOOTPRINT`
(Site 5). The user cannot get *any* boundary for these RFs, in the GUI or the batch pipeline.
Because the edge is a real limit, the honest boundary follows the foot where data exists and
the mapped-region edge where it doesn't — and its enclosed area is a true area (no censoring).

## Goals

### In Scope
1. Detect an **edge peak** explicitly (a contiguous angular sector of rays that ran off the
   mapped-data edge / found no foot, i.e. peak near the valid-data boundary).
2. Construct a **boundary-closed enclosing polygon** = interior foot-arc + a walk along the
   valid-data (`~isnan(grid_z)`) boundary between the two transition points, and feed it to
   the existing `envelope_contour_to_footprint` so the final closed contour is produced by the
   same downstream machinery (footprint clip + marching-squares + smoothing).
3. Make the edge path work **identically in the GUI preview and the batch pipeline** (single
   source of truth `compute_radial_foot_stages`), returning a normal `contour_uv`/`contour_rc`
   through the **unchanged return contract** (no new flag).
4. Validate with **synthetic known-answer fixtures** (edge peak; interior control) and a
   **prevalence scan** over the database to size the problem and harvest real test cases.

### Out of Scope
- **Censored / edge-limited flag** on the output — deliberately omitted. The mapped edge is a
  real limit, so the area is honest; downstream treats an edge RF like any other. (This
  diverges from the generic "tag edge-limited provenance" guidance by explicit user decision;
  recorded in Definitions + ADR note.)
- **Corner RFs** (peak against *two* edges → two separate off-data sectors). Phase-2 detects
  this and **raises explicitly with context** (visible deferred debt); handling is future work,
  revisited after the prevalence scan.
- Any change to the interior-peak path's geometry (must stay byte-identical → parity).
- New tunable parameters / GUI widgets beyond what already exists.
- Changing the peak locator (`find_peak_location`).

## Success Criteria

- [ ] A synthetic **edge-peak** grid that currently raises `SEED_OUTSIDE_FOOTPRINT` now returns
      a **closed, simple (non-self-intersecting) `contour_rc`** enclosing the peak, via
      `compute_radial_foot_stages`.
- [ ] Its enclosed area matches a hand-computed value within a stated grid-discretization
      tolerance (`assert_allclose`, not exact).
- [ ] An **interior-peak** control fixture returns a contour **byte-identical** to the
      pre-change output (`tests/test_rf_response_fields_parity.py` + a new characterization test).
- [ ] Edge detection is **total**: every input classifies as interior / single-edge /
      corner-or-unresolved; the last raises with context (no silent default).
- [ ] The GUI tuner draws a closed contour on a real edge-RF case that previously showed the
      `SEED_OUTSIDE_FOOTPRINT` diagnostic.
- [x] Prevalence scan reports the count (and lists the (session, gesture) cases) of edge RFs
      across the database. *(Phase 1: 22/55 edge RFs, 8 SEED_OUTSIDE_FOOTPRINT — see Phase 1 results.)*
- [ ] No silent fallback: the edge path is entered by explicit detection, not by catching the
      existing raise.

## Definitions

- **Edge peak:** a located peak for which a contiguous angular sector of cast rays fails to
  find a positive-λmax foot because those rays leave the valid-data region (`~isnan(grid_z)`)
  — equivalently, the peak lies within the region such that ≥ one contiguous run of rays has
  `found[i] == False` due to running off the mapped data. Detection also cross-checks peak
  distance to the valid-data boundary.
- **Valid-data / mapped-region boundary:** the edge of `~isnan(grid_z)` — the real physical
  limit of the mapped forearm blob (already island-cleaned upstream to one connected component).
  This is the line the contour is closed along. NOT the footprint (`grid_z > 0`) edge.
- **Transition points:** the two rays bounding the off-data sector — the last `found` ray
  before the gap and the first `found` ray after it. Their foot endpoints (at `snapped_radii`,
  which sit on the last valid-data cell) are where the foot-arc meets the border walk.
- **Boundary-closed polygon:** the closed polygon formed by the interior foot-arc (over the
  `found` sector) spliced with the valid-data boundary sub-path between the two transition
  points, chosen as the sub-path that (with the arc) encloses the peak.
- **Clean (per the user):** a valid closed polygon on screen comprising the hotspot. Concretely:
  closed, simple, encloses the peak cell, and its border side lies on the mapped-region edge.
- **Corner RF (out of scope, must raise):** ≥ 2 disjoint off-data sectors.

---

## Technical Design

### Approach

The final contour is always the marching-squares boundary of `(enclosing_polygon ∩ footprint)`'s
seed-connected component. Edge peaks fail for one reason only: the star polygon (built from
per-ray radii, with off-data rays collapsed to radius 1.0) does **not enclose the peak**, so
`envelope_contour_to_footprint` raises `SEED_OUTSIDE_FOOTPRINT`. Therefore the minimal, honest
fix is to **hand the envelope an enclosing polygon** for the edge case, then reuse every
downstream step unchanged.

That enclosing polygon is the boundary-closed polygon: interior foot-arc (the `found` rays'
`snapped_radii` endpoints) + a walk along the valid-data boundary between the two transition
points. The valid-data boundary is traced with `find_contours(~isnan(grid_z), 0.5)` — the same
primitive already used on the footprint mask in `_select_enclosing_contour`. The sub-path
between the two transition points that encloses the peak is chosen deterministically (stable
tie-break) and spliced onto the foot-arc.

Because the edge path only activates on explicit detection and produces a normal contour through
the existing keys, the interior path is untouched (parity), the contract is unchanged (no flag),
and a genuinely unresolvable geometry (corner, or a splice that doesn't yield a peak-enclosing
simple polygon) still **raises** with context.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Boundary-closed splice** (foot-arc + walk valid-data boundary) → existing envelope | Honest; reuses envelope clip + marching-squares; controllable; handles peak-on-edge via the real boundary; contract unchanged | New splice code (transition detection, boundary sub-path selection) | **Chosen** |
| **Reach-to-edge star** (set off-data rays' radius to `raw_limit` instead of 1.0 so the star encloses the peak) | Tiny change; reuses everything | Degenerate at peak-exactly-on-edge (peak lands on polygon boundary → still not enclosed); piecewise-linear border at ray resolution | Rejected (degeneracy) — reconsider only if splice proves intractable |
| **Swap to footprint/level-set boundary for edge RFs** | Simple; footprint already closed | Different boundary definition than interior RFs → areas not comparable | Rejected (comparability) |
| **Mirror-pad the field, run interior method, clip back** | Smooth closed contour | Fabricates unobserved half — violates fail-fast / scientific honesty | Rejected |
| **Flag + censor** | Downstream can annotate | User decided the edge is real → area honest → no censoring | Rejected (per decision) |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `_extract_radial_plateau_foot` (metrics/rf_radial_foot_boundary.py) | Cast rays, per-ray foot; **surface `found`** in its return dict (currently discarded) | grid + λmax + peak → radii/rc arrays + `found`/`clipped` | GUI, footprint clipping, UV |
| **NEW** `_boundary_close_edge_peak(...)` (same module) | Given `found`, `snapped_radii`, `angles`, `peak_rc`, `grid_z`: detect the off-data sector, walk the valid-data boundary between transition points, splice → enclosing polygon `contour_rc` that contains the peak; RAISE on corner/degenerate | per-ray arrays + grid_z → enclosing `contour_rc` | UV mapping, GUI, min_overlap threshold value |
| `_extract_contour_stage` (same module) | Orchestrate foot → (edge? boundary-close : star) → envelope → UV. Chooses the path by explicit edge detection | grid + peak → contour_uv/rc | GUI |
| `envelope_contour_to_footprint` (unchanged) | Rasterise enclosing polygon, ∩ footprint, seed-component, marching-squares, smooth | polygon + grid_z → closed contour_rc | how the polygon was built (interior vs edge) |
| `compute_radial_foot_stages` (unchanged signature/return) | Single source of truth for GUI + pipeline | grids + params → stages dict | which caller invoked it |
| **NEW** `scripts/scan_edge_rfs.py` (or tools/) | Prevalence scan: count/list RFs with peak within k cells of valid-data edge across DB | DB path → count + (session,gesture) list + saved fixtures | contour algorithm internals |

```
_extract_contour_stage
  ├─ _extract_radial_plateau_foot  → {contour_rc, contour_unsnapped_rc, clipped_mask, found(NEW)}
  ├─ if edge_peak(found, peak_rc, grid_z):
  │      enclosing = _boundary_close_edge_peak(found, snapped_radii, angles, peak_rc, grid_z)   # NEW
  │  else:
  │      enclosing = contour_unsnapped_rc   # existing star
  └─ envelope_contour_to_footprint(enclosing, grid_z, ...)  → contour_rc → UV   # unchanged
```

---

## Implementation Plan

### Phase 1: Prevalence scan + real fixtures
**Goal:** size the problem and harvest real edge-RF cases; no algorithm change.

**Phase 1 completed 2026-07-18**

- [x] Add a read-only scan (`scripts/scan_edge_rfs.py` or a tools module) that, over the boundary
      NPZs in the database, locates each peak (`find_peak_location`), measures its distance to the
      valid-data boundary (`~isnan(grid_z)` via distance transform), and reports the count + the
      list of (session, gesture) within k cells, and which currently raise `SEED_OUTSIDE_FOOTPRINT`.
- [x] Save 1–3 real edge-RF grids as compact `.npz` test fixtures (or record how to load them).
- [x] Run it if the DB is reachable; otherwise emit the exact command for the user to run and
      capture the count.

**Files:** `scripts/scan_edge_rfs.py` (new); `tests/fixtures/` (new fixtures, optional)
**Dependencies:** None

**Phase 1 results (2026-07-18)** — scan run over `iff_mean`, 11 sessions × 5 gestures, k=2.
Canonical `spatial_extract_boundaries/` was renamed for a parameter sweep, so the run targeted
`spatial_extract_boundaries_bkp/iff_mean` (the last full run) via `--boundary-subdir`.

| Metric | Count |
|--------|-------|
| Total RFs (session × gesture) | 55 |
| Within k=2 cells of the valid-data edge | 21 |
| Currently raising `SEED_OUTSIDE_FOOTPRINT` | 8 |
| Edge peaks (within-k OR seed-outside union) | 22 |
| Other compute failures | 0 |

`SEED_OUTSIDE_FOOTPRINT` cases (8): `ST13-02/stroke_distal`, `ST16-05/{all, stroke, stroke_distal,
stroke_proximal}`, `ST18-01/stroke`, `ST18-04/{stroke, tap}`. All but one (`ST18-01/stroke`, dist 4.0)
also fall within k=2 — confirming edge-proximity and the current failure are tightly coupled.
Edge RFs are **not rare** (~40% within k=2), validating the feature. Fixtures for the seed-outside
cases saved under `tests/fixtures/rf_edge/`; full per-RF report at `reports/edge_rf_scan.json`.

Exact reproduce command (user): once the canonical dir is restored, drop `--boundary-subdir`:

    conda run -n social-touch-analysis python scripts/scan_edge_rfs.py --k 2 \
        --boundary-subdir spatial_extract_boundaries_bkp --iff-metric mean \
        --save-fixtures 3 --out reports/edge_rf_scan.json

### Phase 2: Core boundary-closed algorithm
**Goal:** produce an enclosing polygon for single-edge peaks; corner raises.

- [ ] Surface `found` from `_extract_radial_plateau_foot` (add to its return dict; keep everything
      else identical).
- [ ] Add `_boundary_close_edge_peak(...)`: detect the single contiguous off-data sector from
      `found`; identify the two transition rays; compute their foot endpoints; trace the valid-data
      boundary with `find_contours(~isnan(grid_z), 0.5)`; select the boundary sub-path between the
      two transition points that encloses the peak (deterministic tie-break); splice foot-arc +
      sub-path into a closed, simple polygon; assert it encloses the peak and is non-self-intersecting.
- [ ] **Fail-fast:** ≥ 2 off-data sectors (corner), a splice that doesn't enclose the peak, or a
      self-intersecting result → raise `ContourExtractionError` with context (new branch or a
      descriptive message reusing the diagnostics payload).
- [ ] Explicit, total edge detection predicate (interior / single-edge / corner-or-unresolved).

**Files:** `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py`
**Dependencies:** None (parallelizable with Phase 1)

### Phase 3: Wire through the shared stage + diagnostic tie-in
**Goal:** both callers get the edge path; the SEED_OUTSIDE_FOOTPRINT edge case now resolves.

- [ ] In `_extract_contour_stage`, choose star vs boundary-closed by the explicit edge detection,
      then call the unchanged `envelope_contour_to_footprint`.
- [ ] Confirm `compute_radial_foot_stages` returns the normal contour keys for an edge peak
      (contract unchanged); interior path byte-identical.
- [ ] Optional tie-in: for cases that STILL fail (corner/unresolved), emit a border-aware
      `ContourFailureDiagnostics` message ("peak at data corner / boundary — unhandled") so the
      GUI explains it instead of the misleading "grow the footprint".

**Files:** `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py`;
possibly `gui/rf_contour_tuning_viewer.py` (advice text only)
**Dependencies:** Phase 2

### Phase 4: Tests + docs
**Goal:** lock correctness and parity.

- [ ] Synthetic known-answer fixtures: (a) hotspot centred on one edge (half-figure, hand-computed
      clipped area), (b) interior control, (c) corner (asserts RAISE). Areas via `assert_allclose`.
- [ ] Characterization/parity test: interior path output unchanged pre/post; add GUI-path vs
      pipeline-path identical-geometry assertion for the same edge fixture.
- [ ] Property checks: returned polygon closed, simple, area ≤ mapped-region area, encloses peak.
- [ ] Run `tests/test_rf_response_fields_parity.py` + `tests/test_rf_radial_foot_boundary.py`.
- [ ] Changelog `docs/changelogs/rf-data-edge-boundary.md`; brainstorm → Handed off; ADR note on
      "close along real edge, no censor flag".

**Files:** `tests/test_rf_radial_foot_boundary.py`; `docs/changelogs/`;
`docs/development/brainstorms/rf-data-edge-boundary.md`
**Dependencies:** Phase 2–3

---

## Testing Plan

### Unit Tests
- [ ] Edge-peak fixture → closed, simple, peak-enclosing `contour_rc`; area ≈ hand value.
- [ ] Interior control → byte-identical to pre-change contour.
- [ ] Corner fixture (two off-data sectors) → raises `ContourExtractionError` with context.
- [ ] Edge detection predicate is total (interior / edge / corner classified deterministically).
- [ ] `_boundary_close_edge_peak` sub-path selection picks the peak-enclosing side.

### Integration Tests
- [ ] `compute_radial_foot_stages` on the edge fixture returns normal contour keys (no error).
- [ ] GUI-path and pipeline-path return identical geometry for the same edge fixture (parity).

### Manual Verification
- [ ] Load a real edge-RF (session, gesture) from the prevalence scan in the tuner → closed
      contour drawn on screen comprising the hotspot.

### Edge Cases
- [ ] Peak exactly on a grid line / one cell from the edge (near-interior).
- [ ] Two rays exiting the edge but re-entering (non-contiguous) → deterministic classify/raise.
- [ ] Self-intersecting splice candidate → raises, not emitted.

---

## Documentation Plan

- [ ] `docs/changelogs/rf-data-edge-boundary.md` — edge-peak boundary-closing; no censor flag; corner deferred.
- [ ] Brainstorm `rf-data-edge-boundary.md` → Handed off.
- [ ] Inline docstrings on `_boundary_close_edge_peak` + the edge-detection predicate, incl. the
      ADR note (why close along the valid-data edge; why no censor flag; corner raises).
- [ ] Update the module docstring's border-handling note.

---

## Rollback Plan

1. Additive + gated behind explicit edge detection; interior path untouched. Revert = drop the
   Phase-2/3 edits (the branch is not yet merged).
2. No data migration; no output-schema change (contract unchanged).
3. If the edge path misbehaves, the detection predicate can be tightened to fall back to the
   existing raise (which the failure-diagnostic feature already explains) — an explicit revert,
   not a silent fallback.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Boundary sub-path selection picks the wrong side → polygon excludes peak | Med | High | Assert peak-enclosure + simplicity post-splice; raise on failure; deterministic tie-break; unit test both sides |
| Edge path alters interior output → parity break | Low | High | Path gated by explicit detection; characterization + parity tests; interior fixtures byte-identical |
| Corner RFs more common than assumed | Unknown | Med | Prevalence scan (Phase 1) measures it; corners raise visibly, not silently — revisit if count is high |
| Prevalence scan can't reach the DB in the impl environment | Med | Low | Correctness rides synthetic fixtures; scan emits a user-runnable command if DB absent |
| `find_contours` on valid-data mask returns multiple loops (holes) | Low | Med | Use the outer loop of the peak's component (island-cleaned upstream to one blob); assert single outer loop else raise |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (prevalence scan) | ~0.5 day | None |
| Phase 2 (core algorithm) | ~1–1.5 days | None |
| Phase 3 (wiring + diagnostic) | ~0.5 day | Phase 2 |
| Phase 4 (tests/docs) | ~0.5–1 day | Phase 2–3 |

---

## References

- Brainstorm: `docs/development/brainstorms/rf-data-edge-boundary.md`
- Precursor (uncommitted, this branch): `docs/development/plans/active/rf-contour-failure-diagnostics.md`
- Constraint: `docs/claude/fail-fast-pipeline.md`
- Related: `docs/development/plans/completed/rf-radial-foot-boundary-island-cleaning.md`
