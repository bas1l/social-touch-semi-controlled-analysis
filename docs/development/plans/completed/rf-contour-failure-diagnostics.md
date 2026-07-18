# Plan: RF Contour Live Failure Diagnostics + Exposed Plateau Gate

**Date:** 2026-07-18
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-07-18 12:32
**Base Branch:** `dev`
**Branch:** `feature/rf-contour-failure-diagnostics`

---

## Overview

When the RF contour tuner (`RFContourTuningViewer`) fails to trace a contour even
though the λmax peak and its curvature ring are visibly present, the user gets no
usable explanation of *why* or *which knob to change*. This feature (1) surfaces a
**live, quantitative, per-branch failure diagnostic** on every Recompute, and (2)
**exposes the previously-hardcoded plateau-detection gate** (`prominence`,
`plateau_size`) as tunable parameters so the most common such failure (Site 3) is
actually user-fixable rather than a dead end.

## Problem Statement

The user judges contour quality purely by eye (3D interactive view + raw/processed
heatmaps). The failure they cannot resolve is the "green-but-can't-fit" case: a
visibly good λmax field whose contour extraction still fails. Three concrete gaps:

1. **The actionable hint is suppressed exactly when tuning.** `_classify_contour_error`
   maps a failure to a "try knob X" hint, but the full hint only fires as a modal
   on session/gesture *switch* (`notify_no_contour=True`); on Recompute/toggle it
   is suppressed and the status line shows only the truncated first sentence of the
   raw error.
2. **The Site-3 message actively misleads.** It reads *"field too flat or peak runs
   off the data footprint"* — which contradicts what the user sees (a clear ring),
   destroying trust and pointing at the wrong mental model.
3. **The real Site-3 lever is unreachable.** `prominence` / `plateau_size` /
   `require_positive` are hardcoded function defaults (`None` / `1` / `True`), never
   exposed in YAML, the DTOs, or the 5 sliders. So even a correct diagnosis of
   "the plateau gate rejected your ring" left the user with no knob to turn.

## Goals

### In Scope
1. A `ContourFailureDiagnostics` structured payload emitted by the compute layer
   (numbers + a branch identifier), carried out of the three failure sites via a
   loud raise, and surfaced on the existing `compute_radial_foot_stages(allow_partial=True)`
   partial-return path.
2. A single authoritative **branch → plain-language cause + which-knob** map in the
   GUI, replacing the fragile substring-matching in `_classify_contour_error`, and
   a **live** per-branch quantitative message shown on the quiet Recompute/toggle
   path (not only on gesture switch).
3. Expose `prominence` and `plateau_size` as tunable per-(session, gesture) params
   (new DTO fields, config defaults, params-IO schema, GUI widgets), threaded
   through **both** the GUI preview and the batch pipeline so a Validated value is
   actually used downstream.
4. Fix the misleading Site-3 message text.

### Out of Scope
- Any parameter *sweep*, thumbnail gallery, sensitivity filmstrip, or
  auto-optimizer/ranker — explicitly rejected in the brainstorm (contour *quality*
  is not machine-computable; only *fittability* is). See
  `docs/development/brainstorms/rf-contour-param-sweep.md`.
- Exposing `require_positive` as a knob (the visible-ring case is positive by
  construction, so this gate is not the bottleneck; leave hardcoded `True`).
- Changing any contour math / output. The diagnostic change must be strictly
  additive so GUI↔pipeline parity is preserved.
- A generic telemetry/diagnostics framework for hypothetical future sites (YAGNI).

## Success Criteria

- [ ] On a Site-3 / Site-4 / Site-5 failure, clicking **Recompute** shows a
      status message naming the branch in plain language, the correct knob(s), and
      the actual numbers that gated it (e.g. `0/360 rays formed a plateau; peak
      λmax=0.82 > 0 → the detector found no radial ridge: lower prominence / reduce
      plateau size, or adjust gauss/hess σ`).
- [ ] The old "field too flat or peak runs off the data footprint" text no longer
      appears for a ring-present Site-3 failure.
- [ ] `prominence` and `plateau_size` appear as tunable widgets in the tuner, are
      saved to / loaded from `*_contour_params.json`, and a Validated value changes
      the contour produced by the **batch pipeline** (not just the preview).
- [ ] Existing (pre-feature) `*_contour_params.json` files still load without error
      (new keys resolve to documented defaults at the loader boundary).
- [x] `tests/test_rf_response_fields_parity.py` and
      `tests/test_rf_radial_foot_boundary.py` still pass (no math/parity drift).
- [x] New known-answer unit tests force each of the three branches and assert the
      diagnostic's numeric fields.
- [ ] No silent fallback introduced anywhere; every malformed input still raises
      with context.

## Definitions

- **Failure branch:** exactly one of the three ring-present failure modes in
  `rf_radial_foot_boundary.py`:
  - **Site 3 — `NO_RADIAL_PLATEAU`:** no ray produced a qualifying λmax plateau.
    Discriminator: `contour_unsnapped_rc` was never produced; `found.sum()==0`.
  - **Site 5 — `SEED_OUTSIDE_FOOTPRINT`:** a contour was traced but the seed cell
    is not painted in the footprint. Discriminator: `contour_unsnapped_rc` present,
    `footprint[peak]==False`.
  - **Site 4 — `NO_ENCLOSING_CONTOUR`:** contour(s) traced but none enclose the
    peak. Discriminator: `contour_unsnapped_rc` present, `len(contours)>0`, none
    enclose.
- **Master discriminator:** *was `contour_unsnapped_rc` produced?* No → detection
  failure (Site 3). Yes → geometry failure (Site 4/5), split by `footprint[peak]`.
- **Sanctioned optional JSON key:** a key that may be *absent* (resolved to a
  documented default at the single loader boundary in `rf_contour_params_io.load_contour_params`)
  but, when *present*, must be well-formed or it raises. This is the existing
  `param_enabled` precedent and is explicitly NOT a silent fallback.
- **Live diagnostic:** rendered on the quiet path (`notify_no_contour=False`), i.e.
  during Recompute and checkbox toggles — the moments the user is actively tuning.

---

## Technical Design

### Approach

Two additive pillars, both riding existing seams so no math or parity changes:

**Pillar A — diagnostics.** The three inner functions already raise `ValueError` on
failure. Replace each raise with `ContourExtractionError(ValueError)` carrying a
frozen `ContourFailureDiagnostics` payload (branch enum + the in-scope numbers).
Because it subclasses `ValueError`, the existing `except ValueError` in
`compute_radial_foot_stages` catches it unchanged; in the `allow_partial=True`
branch, attach `exc.diagnostics` to the returned dict under a new `"diagnostics"`
key (the `"error"` string stays for backward-compat). The compute layer never
imports PyQt and never formats a human sentence — it emits data. The GUI owns a
single `branch → (cause text, knob advice)` map and renders the live message.

**Pillar B — expose the gate.** Add `prominence` and `plateau_size` to
`GestureContourParams` (and defaults to `BoundaryParams` + YAML), thread them
through `rf_contour_params_io` (schema/save/load, as sanctioned-optional keys),
add `effective_*` resolvers, and pass them into `compute_radial_foot_stages` from
**both** the GUI `_recompute` and the pipeline `compute_radial_foot_boundary`
call in `rf_boundary_extraction.py`. `prominence` gets a toggle (unchecked→`None`);
`plateau_size` is a plain int spinbox (`1` = natural off).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Custom exception carrying a diagnostics DTO | Additive; keeps loud raises (fail-fast); subclass of ValueError so existing catch works; no math change | One new exception type + payload dataclass | **Chosen** |
| Refactor inner fns to *return* a success/failure union | "Cleaner" than exceptions | Large refactor of the return-None convention; risks parity tests; violates YAGNI | Rejected |
| Silent `None` + reconstruct reason in GUI | No compute change | Impossible (numbers are gone) and violates no-silent-fallback | Rejected |
| Pre-format the "which knob" sentence in the compute layer | Fewer GUI changes | Couples presentation into compute; violates layering | Rejected |
| Keep `prominence`/`plateau_size` hardcoded, diagnose-only | Smallest scope | Leaves Site-3 unfixable from GUI — user explicitly chose to expose the gate | Rejected (per user decision) |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `rf_boundary_types.py` | Own the contracts: add `ContourFailureBranch` enum, `ContourFailureDiagnostics` frozen dataclass, and the two new fields on `GestureContourParams` (+ `effective_*` resolvers, + optional `prominence` toggle in `ContourParamToggles`) | dataclass defs | PyQt, file paths, GUI |
| `rf_radial_foot_boundary.py` | Detect boundary; on failure raise `ContourExtractionError` carrying the branch + numbers; surface `diagnostics` on the `allow_partial` partial dict | grids + params → stages dict (now optionally with `diagnostics`) | PyQt, human-readable wording |
| `rf_contour_params_io.py` | Persist/loads the params incl. the two new sanctioned-optional keys; thread `prominence`/`plateau_size` into `build_session_grid`? (No — they belong to the radial-foot stage, not the grid) | JSON ↔ `GestureContourParams` | GUI, matplotlib |
| `rf_boundary_extraction.py` (pipeline) | Pass tuned `prominence`/`plateau_size` into `compute_radial_foot_boundary` | per-gesture params → boundaries | GUI |
| `rf_contour_tuning_viewer.py` (GUI) | Own the single `branch → (cause, knob)` map; render live diagnostic on the quiet path; add the two widgets (+ toggle), tooltips, help | stages dict → rendered status | the *numeric* detection internals (only reads the payload) |

```
ContourFailureBranch = Enum(NO_RADIAL_PLATEAU, SEED_OUTSIDE_FOOTPRINT, NO_ENCLOSING_CONTOUR)

@dataclass(frozen=True)
class ContourFailureDiagnostics:
    branch: ContourFailureBranch
    n_angles: int | None = None            # Site 3
    found_count: int | None = None         # Site 3 (rays with a plateau)
    peak_lmax: float | None = None         # Site 3 (nanmax λmax; >0 proves curvature)
    footprint_at_seed: bool | None = None  # Site 5
    footprint_cells: int | None = None     # Site 5 (grid_z>0 count)
    n_candidate_contours: int | None = None# Site 4 (len(contours))
    # Optional fields stay None when not-applicable — "zero" vs "absent" kept distinct.

class ContourExtractionError(ValueError):
    def __init__(self, message: str, diagnostics: ContourFailureDiagnostics): ...
```

---

## Implementation Plan

### Phase 1: Diagnostic infrastructure (compute layer)
**Goal:** the three failure sites emit structured numbers via a loud raise, surfaced on the partial path. Independent of the new knobs.

_Phase 1 completed 2026-07-18._

- [x] Add `ContourFailureBranch`, `ContourFailureDiagnostics`, `ContourExtractionError` to `rf_boundary_types.py`.
- [x] Site 3 (`_extract_radial_plateau_foot`, ~L322): raise `ContourExtractionError` with `branch=NO_RADIAL_PLATEAU`, `n_angles`, `found_count=int(found.sum())`, `peak_lmax=float(np.nanmax(lmax))`; **fix the misleading message**.
- [x] Site 5 (`envelope_contour_to_footprint`, ~L446): `branch=SEED_OUTSIDE_FOOTPRINT`, `footprint_at_seed=bool(footprint[peak_r, peak_c])`, `footprint_cells=int(footprint.sum())`.
- [x] Site 4 (`_select_enclosing_contour`, ~L398): `branch=NO_ENCLOSING_CONTOUR`, `n_candidate_contours=len(contours)`.
- [x] In `compute_radial_foot_stages` `allow_partial=True` except block: if `isinstance(exc, ContourExtractionError)`, add `diagnostics=exc.diagnostics` to the partial dict; keep `error=str(exc)`. Full-success and `allow_partial=False` paths unchanged.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_boundary_types.py`
- `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py`

**Dependencies:** None

### Phase 2: Expose `prominence` + `plateau_size` as tunable params
**Goal:** the plateau gate becomes user-tunable end-to-end (GUI preview **and** batch pipeline).

_Phase 2 completed 2026-07-18._

- [x] `GestureContourParams`: add `prominence: float | None` and `plateau_size: int` fields + `effective_prominence()` / `effective_plateau_size()` resolvers; add `prominence` to `ContourParamToggles` (unchecked→None).
- [x] `BoundaryParams`: add defaults (`radial_prominence`, `radial_plateau_size`) so `defaults_from_boundary_params` can seed them.
- [x] `configs/analyse_workflow_processing_dag.yaml`: add `radial_prominence` / `radial_plateau_size` in **both** boundary-param blocks (the agent noted ~L246-248 and ~L281-283). Wired YAML→BoundaryParams through `scripts/analysis_workflow_processing.py` (DAG task-options dict → `spatial_extract_boundaries_flow` → `run_population_response_field_extraction` → `BoundaryParams(...)`).
- [x] `rf_contour_params_io.py`: extend `PARAM_KEYS`? No — keep them as **sanctioned-optional** keys (`prominence`→None, `plateau_size`→1) resolved at the loader boundary like `param_enabled`, so existing JSONs still load; malformed still raises. Update `save_contour_params` to write them; `load_contour_params` to read/default/validate.
- [x] Thread `prominence`/`plateau_size` into `compute_radial_foot_boundary` from `rf_boundary_extraction.py` (pipeline path) so Validated values take effect downstream.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_boundary_types.py`
- `src/analysis/receptive_field_mapping/data/rf_contour_params_io.py`
- `src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py`
- `configs/analyse_workflow_processing_dag.yaml`

**Dependencies:** None (parallelizable with Phase 1, but Phase 3 needs both)

### Phase 3: GUI integration
**Goal:** live per-branch diagnostic + the two new widgets, wired to both pillars.

_Phase 3 completed 2026-07-18._

- [x] Single `_BRANCH_ADVICE: dict[ContourFailureBranch, str]` map (cause + knob), replacing the substring-matching in `_classify_contour_error`. Site-3 advice now names `prominence`/`plateau_size` (reachable) alongside the sigmas.
- [x] `_recompute`: when partial, read `stages["diagnostics"]` and render a live quantitative message into `_info_label` on the quiet path (`notify_no_contour=False`); keep the modal on gesture-switch using the same text.
- [x] Add `prominence` (QDoubleSpinBox + toggle checkbox) and `plateau_size` (QSpinBox, min 1) to `_PARAM_LABELS`, `_build_param_panel`, `_build_params_from_widgets`, `_set_widgets_from_params`; tooltips; extend `_HELP_TEXT`.
- [x] Pass `effective_prominence()` / `effective_plateau_size()` into `compute_radial_foot_stages` from `_recompute`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py`

**Dependencies:** Phase 1 + Phase 2

### Phase 4: Tests + docs
**Goal:** lock behavior; prove no parity drift.

_Phase 4 completed 2026-07-18._

- [x] Known-answer unit tests (in `tests/test_rf_radial_foot_boundary.py`): craft fixtures that force each branch; assert `diagnostics` fields against hand-computed values.
- [x] Params-IO round-trip test (extend `tests/test_rf_contour_params.py`): new keys save/load; absent keys default; malformed raises.
- [x] Confirm `tests/test_rf_response_fields_parity.py` passes unchanged.
- [x] Update `_HELP_TEXT` (Phase 3), the brainstorm status (Handed off, prior), and a changelog entry (`docs/changelogs/rf-contour-failure-diagnostics.md`).

**Files Modified:**
- `tests/test_rf_radial_foot_boundary.py`, `tests/test_rf_contour_params.py`
- `docs/development/brainstorms/rf-contour-param-sweep.md`, `docs/changelogs/`

**Dependencies:** Phase 1–3

---

## Testing Plan

### Unit Tests
- [x] Site-3 fixture (full-grid concave paraboloid, λmax never positive) → `NO_RADIAL_PLATEAU`, `found_count==0`, `n_angles` echoes arg, `peak_lmax == nanmax(lmax)`.
- [x] Site-5 fixture (all-negative dome, empty `grid_z>0` footprint) → `SEED_OUTSIDE_FOOTPRINT`, `footprint_at_seed is False`, `footprint_cells==0`.
- [x] Site-4 fixture (monotone plane ramp) → `NO_ENCLOSING_CONTOUR`, `n_candidate_contours>=1` (also covered by a direct `_select_enclosing_contour` unit test).
- [x] `GestureContourParams.effective_prominence()` returns `None` when toggle off, the value when on; `effective_plateau_size()` returns the stored int.
- [x] Params-IO: absent `prominence`/`plateau_size` → documented defaults; present-but-malformed → raises with file context.

### Integration Tests
- [ ] `compute_radial_foot_stages(allow_partial=True)` on a Site-3 grid returns a dict containing a `diagnostics` payload with the correct branch.
- [ ] A tuned `plateau_size`/`prominence` in a `*_contour_params.json` changes the boundary produced by the pipeline path (`rf_boundary_extraction`).

### Manual Verification
- [ ] Launch the tuner on a known "ring present but can't fit" (session, gesture); Recompute; confirm the live message names the branch, the knobs, and the numbers.
- [ ] Lower `prominence` (toggle on) until the Site-3 case fits; Validate; confirm JSON has the new keys.

### Edge Cases
- [ ] Existing pre-feature JSON (no new keys) loads and previews without error.
- [ ] A partial result whose exception is a plain `ValueError` (not `ContourExtractionError`) still surfaces `error` and does not crash (no `diagnostics` key).

---

## Documentation Plan

- [ ] Update `_HELP_TEXT` in the tuner (new params + toggle + what the live diagnostic means).
- [ ] Update the brainstorm file status → Handed off.
- [x] Add changelog entry: `docs/changelogs/rf-contour-failure-diagnostics.md`.
- [ ] Inline docstrings on the new dataclass/enum/exception and the two new params.

---

## Rollback Plan

1. Feature is additive and isolated to the radial-foot detector + its tuner. To
   revert: `git revert` the feature commits (or delete the feature branch before
   merge).
2. **Data considerations:** the two new params are sanctioned-optional JSON keys,
   so reverting the code leaves any already-written keys as harmless extras — but
   the loader's `extra`-key check would then reject them. Mitigation: if reverting
   after any Validate, note that JSONs written with the new keys must have those
   keys stripped (or re-Validated pre-revert). Document in the changelog.
3. No pipeline output schema change; no migration.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Enriching raise sites accidentally alters contour output → parity break | Low | High | Change is raise-path only; run parity tests each phase; assert no change on success path |
| New optional JSON keys break the loader's strict `extra`-key rejection on revert | Med | Low | Documented in Rollback; keys are sanctioned-optional while feature is present |
| `prominence`/`plateau_size` not threaded into the pipeline path → Validated value ignored downstream | Med | High | Explicit Phase-2 task + integration test asserting pipeline boundary changes |
| Branch discriminator misclassifies (e.g. Site-4 vs Site-5) | Low | Med | Discriminator is `contour_unsnapped_rc` presence + `footprint[peak]`; each branch raised at its own site, so branch id is set at the source, not re-inferred |
| Scope creep back toward a sweep/optimizer | Low | Med | Explicitly out of scope; brainstorm rejection recorded |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (diagnostics) | ~0.5 day | None |
| Phase 2 (expose gate) | ~0.5–1 day | None |
| Phase 3 (GUI) | ~0.5–1 day | Phase 1 + 2 |
| Phase 4 (tests/docs) | ~0.5 day | Phase 1–3 |

---

## References

- Brainstorm: `docs/development/brainstorms/rf-contour-param-sweep.md`
- Related completed plans: `docs/development/plans/completed/rf-contour-tuning-param-toggles-and-help.md`,
  `spatial-tune-rf-contours.md`, `rf-radial-foot-boundary-island-cleaning.md`,
  `fix-gradient-boundary-nan-mask-spike.md`
- Constraint: `docs/claude/fail-fast-pipeline.md`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- configs/analyse_workflow_processing_dag.yaml
- docs/changelogs/rf-contour-failure-diagnostics.md
- docs/development/brainstorms/rf-contour-param-sweep.md
- docs/development/plans/active/rf-contour-failure-diagnostics.md
- scripts/analysis_workflow_processing.py
- src/analysis/receptive_field_mapping/data/rf_boundary_types.py
- src/analysis/receptive_field_mapping/data/rf_contour_params_io.py
- src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py
- src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py
- src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py
- src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- tests/test_rf_contour_params.py
- tests/test_rf_radial_foot_boundary.py
