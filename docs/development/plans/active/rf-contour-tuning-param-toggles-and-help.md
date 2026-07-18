# Plan: RF Contour Tuning — Per-Parameter Toggles, Help, and Inline Status

**Date:** 2026-07-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/spatial-configure-stroke-axis`
**Branch:** `feature/rf-contour-param-toggles`

---

## Overview

Extend the per-(session, gesture) RF contour tuning GUI (`RFContourTuningViewer`) and
its persisted config so a researcher can **enable/disable individual
contour-detection steps**, understand what each control does, and see at a glance
whether the last recompute produced a contour. The on/off state is saved in the
per-combination JSON and honoured identically by the interactive preview and the
batch `spatial_extract_boundaries` pipeline (single source of truth).

## Problem Statement

The contour tuner exposes five parameters (`min_overlap_pct`,
`median_filter_size`, `radial_gauss_sigma`, `radial_hess_sigma`,
`radial_envelope_smooth_sigma`) as always-on spinboxes. Three limitations:

1. **No way to skip a step.** A researcher cannot ask "what does the contour look
   like with no median filter / no pre-smoothing?" without hunting for a sentinel
   value — and for most parameters the frozen `GestureContourParams` dataclass
   forbids the natural "off" value (`median_filter_size` can't be `None`/`0`; the
   sigmas can't be `0`). The pipeline's existing "median filter disabled" state is
   already routed *around* the dataclass via a `None` fallback
   (see `spatial-tune-rf-contours.md`, "Byte-identical no-regression note"),
   confirming the gap.
2. **No in-panel explanation.** New users don't know what each parameter or the
   Default / Recompute / Validate buttons do.
3. **Disruptive failure pop-up.** When a contour can't be traced, `_recompute`
   raises a modal `QMessageBox` **every** time — including on the Recompute
   button, where the researcher is actively dialling parameters and expects only a
   quiet on-screen status, not an interrupting dialog.

## Goals

### In Scope
1. A checkbox per **toggleable** parameter (`min_overlap_pct`,
   `median_filter_size`, `radial_gauss_sigma`, `radial_envelope_smooth_sigma`)
   that, when unticked, **skips that processing step** in both the GUI preview and
   the production pipeline. `radial_hess_sigma` (the Hessian scale — the core
   detector algorithm) has **no** checkbox.
2. Persist the on/off state in the per-(session, gesture) contour-params JSON and
   thread it through `GestureContourParams` → `PreparedBoundaryData.per_gesture_params`
   into the two production consumer sites.
3. Default state for a combination with no saved flags: `min_overlap_pct` **ON**,
   `median_filter_size` **OFF**, `radial_gauss_sigma` **ON**,
   `radial_envelope_smooth_sigma` **ON**.
4. A help affordance ("?" button opening a dialog) explaining the five parameters,
   the four toggles, and the three action buttons; plus per-widget tooltips.
5. An inline status area below the action buttons: a coloured LED
   (green = contour drawn, red = no contour) + a contour-info text line.
6. Suppress the failure **modal** on the Recompute button; keep it only for a
   session/gesture switch. The LED/text reflect the state in both cases.
7. A parity test proving the GUI preview and the pipeline produce identical
   contours for the same toggle state (both ON and OFF paths).

### Out of Scope
- A checkbox for `radial_hess_sigma` (explicitly excluded — core algorithm).
- Changing the actual contour math / detector behaviour.
- Migrating any already-written contour-params JSON on disk (the format is
  unreleased — `rf_contour_params_io.py` is untracked — so there is no shipped
  tuned-config data to preserve).
- Reworking the global (non-tuned) `use_tuned_params=False` path — it stays
  byte-identical (global `BoundaryParams` scalars, all steps as configured).
- A generic "step registry" for future steps (YAGNI — exactly four flags today).

## Success Criteria

- [ ] Each of the four toggleable parameters has a working checkbox; the spinbox
      is visually disabled (greyed) when its checkbox is unticked.
- [ ] Unticking a checkbox skips the corresponding step in the GUI preview:
      `min_overlap` → keep all contacted vertices; `median_filter` → no median
      filtering; `radial_gauss` → no Gaussian pre-smoothing; `radial_envelope` →
      no envelope smoothing.
- [ ] The same toggle state, saved via Validate, changes the batch pipeline output
      identically (verified by the parity test).
- [ ] `radial_hess_sigma` has no checkbox and is always applied.
- [ ] A combination with no saved flags shows: min_overlap ON, median OFF,
      gauss ON, envelope ON.
- [ ] An older contour-params JSON lacking the `param_enabled` key still loads
      (no `missing`/`extra` key error) and resolves to the default flag state.
- [ ] A "?" button opens a dialog explaining all five parameters, the toggles, and
      the three buttons; each parameter widget has a tooltip.
- [ ] Below the action buttons, a green LED + "contour points=N" shows on success;
      a red LED + "contour: none — <reason>" shows on failure.
- [ ] Clicking Recompute never opens a modal dialog; switching session/gesture
      still opens the detailed "Contour not drawn" modal on failure.
- [ ] All new/changed code is fail-fast: malformed JSON (wrong types, non-bool
      flags) still raises with file context.

## Definitions

- **Toggleable parameter:** one of the four params that gets an enable/disable
  checkbox (`min_overlap_pct`, `median_filter_size`, `radial_gauss_sigma`,
  `radial_envelope_smooth_sigma`). `radial_hess_sigma` is **not** toggleable.
- **Disabled / "skipped" step — concrete, testable meaning per parameter:**
  - `min_overlap_pct` disabled ⇒ effective value `0.0` ⇒
    `compute_threshold_from_ratio(0, n)` floors to threshold `1` ⇒ every contacted
    vertex is kept (weakest possible threshold; no vertices greyed out).
  - `median_filter_size` disabled ⇒ effective value `None` ⇒
    `compute_interpolated_grid(..., median_filter_size=None)` skips the median
    filter entirely.
  - `radial_gauss_sigma` disabled ⇒ effective value `0.0` ⇒
    `scipy.ndimage.gaussian_filter(sigma=0)` is the identity ⇒ no pre-smoothing
    (the Hessian runs on the raw normalised field).
  - `radial_envelope_smooth_sigma` disabled ⇒ effective value `0.0` ⇒
    `_smooth_closed_contour` returns the contour unchanged (documented no-op).
- **Contour drawn (green LED):** `compute_radial_foot_stages` returned a non-`None`
  `contour_rc`. **No contour (red LED):** partial result (`contour_uv is None`).
- **Switch (modal shown):** a recompute triggered by `_load_current` after a
  session or gesture combo change (or the initial deferred start).
  **Recompute (no modal):** a recompute triggered by the Recompute button.

---

## Technical Design

### Approach

Introduce a small **frozen `ContourParamToggles` dataclass** — a map of four
named booleans — mirroring the established `SlimUvCleanSteps` /
`slim_uv_config_viewer.py` pattern (the repo's canonical "persisted map of enable
flags" precedent). Carry it as one field on `GestureContourParams`, so it rides
the *existing* `per_gesture_params` plumbing into the pipeline with no new
threading. Keep the five value fields and their validation exactly as they are.

Resolve a disabled step to its concrete "skip" value at **one place** — a set of
`effective_*()` helper methods on `GestureContourParams` — and have *both*
consumers (GUI `_recompute` and the two pipeline sites) call those helpers. This
keeps the skip-decision as *connascence of name* (both read the same method)
rather than *connascence of algorithm* (each re-deriving "should I skip?"), per
guide `05`. The numerical cores stay pure transforms; the decision to invoke them
lives at the composition boundary (guide `02` — avoid control coupling).

Persist the flags as a **nested `param_enabled` JSON object**. Restructure
`load_contour_params` to accept it as an *optional* key with a documented default
resolved at the single loader boundary — the guide-sanctioned reconciliation of
"no silent fallbacks" with "flag-less JSONs must load": a missing optional field
with a documented, behaviour-defining default is not a silent fallback; a
scattered inline `.get(default)` would be. Any *malformed* flag (non-bool, unknown
sub-key) still raises with file context.

The help and status changes are GUI-only (`rf_contour_tuning_viewer.py`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Nested `ContourParamToggles` dataclass on `GestureContourParams`, `effective_*()` resolvers | Matches repo `SlimUvCleanSteps` precedent; rides existing plumbing; single skip-decision point; value fields untouched | One new small dataclass + nested JSON key | **Chosen** |
| Relax `GestureContourParams` validation to allow sentinel "off" values (`median=None`, `sigma=0`) | No new type | Loses the positive/odd/`>0` guarantees the value fields exist to enforce; "off" and "0.0 is a real value" become indistinguishable; violates fail-fast intent | Rejected |
| Flat `*_enabled` bool sibling keys in JSON + flat fields on `GestureContourParams` | Slightly less nesting | Diverges from the `SlimUvCleanSteps` nested-map precedent; pollutes flat `PARAM_KEYS`; mixes value + flag concerns on one flat contract | Rejected |
| GUI-preview-only toggles (no persistence) | Smallest change | Preview would diverge from the pipeline output — breaks the single-source-of-truth the tuner exists to guarantee | Rejected (per user decision) |
| Keep the failure modal on Recompute | No change | Disruptive during active parameter dialling (user report) | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `ContourParamToggles` (new, `rf_boundary_types.py`) | Hold the four enable flags; validate each is a genuine `bool` | `4 bools` → frozen value object; `to_dict()`/`from_dict()` | JSON paths, GUI widgets, skip *values* (only ON/OFF) |
| `GestureContourParams` (`rf_boundary_types.py`) | The five tuned values **+** a `toggles` field; expose `effective_*()` skip-resolution | values + toggles → `effective_min_overlap_pct/ median_filter_size/ gauss_sigma/ envelope_smooth_sigma` | which consumer calls it (GUI vs pipeline) |
| `rf_contour_params_io.py` (save/load) | Serialize/parse the nested `param_enabled` object; resolve absent key to default at the one loader boundary | JSON ⇄ `GestureContourParams` (incl. toggles) | GUI, pipeline, the meaning of "skip" |
| `rf_boundary_preparation.py` (2 sites) | Use `effective_min_overlap_pct()` / `effective_median_filter_size()` when `per_gesture_params` is set | prepared arrays + params → thresholded/interpolated grid | the toggle *storage* shape |
| `rf_boundary_extraction.py` (radial branch) | Use `effective_gauss_sigma()` / `radial_hess_sigma` / `effective_envelope_smooth_sigma()` | grid + params → radial boundary | the toggle *storage* shape |
| `rf_contour_tuning_viewer.py` (GUI) | Render checkboxes + tooltips + "?" help + LED/status; call `effective_*()` for the preview; gate the modal on switch-vs-recompute | user input ⇄ `GestureContourParams`; preview | pipeline internals |

```
GestureContourParams
  ├─ min_overlap_pct: float          (validated 0..100)
  ├─ median_filter_size: int         (validated positive odd)
  ├─ radial_gauss_sigma: float       (validated > 0)
  ├─ radial_hess_sigma: float        (validated > 0)          # no toggle
  ├─ radial_envelope_smooth_sigma: float  (validated >= 0)
  ├─ toggles: ContourParamToggles = field(default_factory=ContourParamToggles)
  ├─ effective_min_overlap_pct() -> float          # 0.0 when disabled
  ├─ effective_median_filter_size() -> int | None  # None when disabled
  ├─ effective_gauss_sigma() -> float              # 0.0 when disabled
  └─ effective_envelope_smooth_sigma() -> float     # 0.0 when disabled

ContourParamToggles (frozen)
  ├─ min_overlap_pct: bool = True
  ├─ median_filter_size: bool = False   # default OFF (user decision)
  ├─ radial_gauss_sigma: bool = True
  └─ radial_envelope_smooth_sigma: bool = True

JSON:
  { ...five params..., "param_enabled": { "min_overlap_pct": true,
    "median_filter_size": false, "radial_gauss_sigma": true,
    "radial_envelope_smooth_sigma": true }, ...meta... }
```

---

## Implementation Plan

### Phase 1: Config contract + persistence (foundation)
**Started:** 2026-07-18
**Completed:** 2026-07-18

**Goal:** The toggle state exists as a validated, round-trippable part of the
per-(session, gesture) config, backward-compatible with flag-less JSONs.

- [x] 1.1 — Add frozen `ContourParamToggles` to `rf_boundary_types.py`: four bool
      fields with the default state (median `False`, others `True`); fail-fast
      `__post_init__` rejecting non-`bool`; `to_dict()` (`asdict`) and a
      `from_dict(cls, data)` classmethod that reads each field via
      `data.get(name, <field default>)` (backward-compat) and rejects unknown
      sub-keys.
- [x] 1.2 — Add `toggles: ContourParamToggles = field(default_factory=ContourParamToggles)`
      to `GestureContourParams` and the four `effective_*()` methods returning the
      concrete skip values defined in **Definitions**.
- [x] 1.3 — In `rf_contour_params_io.py`: define `TOGGLEABLE_PARAM_KEYS`; add an
      `_OPTIONAL_JSON_KEYS = ("param_enabled",)`; change the `extra` check to
      exclude optional keys; parse `ContourParamToggles.from_dict(data.get("param_enabled", {}))`;
      validate the value is a mapping when present (else raise with file path).
- [x] 1.4 — In `save_contour_params`: serialize `params.toggles.to_dict()` under
      `"param_enabled"` (canonical key order preserved).
- [x] 1.5 — In `defaults_from_boundary_params`: construct `GestureContourParams`
      with default `ContourParamToggles()` (median OFF, others ON).

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_boundary_types.py`
- `src/analysis/receptive_field_mapping/data/rf_contour_params_io.py`

**Dependencies:** None

### Phase 2: Pipeline consumers honour the toggles
**Started:** 2026-07-18
**Completed:** 2026-07-18

**Goal:** The batch `spatial_extract_boundaries` path skips disabled steps
identically to the preview — only on the `use_tuned_params` (per_gesture_params
set) path; the global path stays byte-identical.

- [x] 2.1 — `rf_boundary_preparation.py` (~L210): when `per_gesture_params` is set,
      `min_overlap_pct = per_gesture_params[gtype].effective_min_overlap_pct()`.
- [x] 2.2 — `rf_boundary_preparation.py` (~L334): when set,
      `median_filter_size = per_gesture_params[gtype].effective_median_filter_size()`.
- [x] 2.3 — `rf_boundary_extraction.py` (~L87–95): use
      `gp.effective_gauss_sigma()`, `gp.radial_hess_sigma`,
      `gp.effective_envelope_smooth_sigma()`.
- [x] 2.4 — Verify `compute_radial_foot_boundary` and `compute_radial_foot_stages`
      accept `gauss_sigma=0.0` without a `> 0` assertion (both should, but confirm;
      if either validates `> 0`, relax to `>= 0` or gate at the call site).

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/rf_boundary_preparation.py`
- `src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py`
- (verify only) `src/analysis/receptive_field_mapping/metrics/rf_radial_foot_boundary.py`

**Dependencies:** Phase 1

### Phase 3: GUI — checkboxes
**Started:** 2026-07-18
**Completed:** 2026-07-18

**Goal:** Per-parameter checkboxes wired to the config, driving the preview via
`effective_*()`.

- [x] 3.1 — `_build_param_panel`: add `self._param_enabled: dict[str, QCheckBox]`
      for the four toggleable keys; lay each checkbox beside its spinbox row (no
      checkbox for `radial_hess_sigma`). Connect `toggled` to enable/disable the
      paired spinbox and to a light preview refresh (no modal).
- [x] 3.2 — `_set_widgets_from_params`: set each checkbox from
      `params.toggles.*` (wrapped in `blockSignals`), and grey the spinbox to match.
- [x] 3.3 — `_build_params_from_widgets`: build `ContourParamToggles` from
      `{key: cb.isChecked()}` and pass it as `toggles=` into `GestureContourParams`.
- [x] 3.4 — `_recompute`: build the grid with
      `params.effective_min_overlap_pct()` / `effective_median_filter_size()` and
      call `compute_radial_foot_stages` with `effective_gauss_sigma()` /
      `radial_hess_sigma` / `effective_envelope_smooth_sigma()`.
- [x] 3.5 — `_on_default_clicked`: also reset the checkboxes to the default state.

**Files Modified:**
- `src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py`

**Dependencies:** Phase 1

### Phase 4: GUI — help + inline status
**Started:** 2026-07-18
**Completed:** 2026-07-18

**Goal:** Explanatory help and a non-disruptive status indicator.

- [x] 4.1 — Add a "?" `QPushButton` (in/near the action row) opening a
      `QMessageBox`/`QDialog` with concise, accurate descriptions of the five
      parameters, the four toggles, and the Default / Recompute / Validate buttons.
- [x] 4.2 — Add `setToolTip(...)` to each parameter label/spinbox and checkbox.
- [x] 4.3 — Add a status row below the action buttons: a small LED `QLabel`
      (coloured circle via stylesheet) + the existing `_info_label` text. Green =
      contour drawn; red = no contour.
- [x] 4.4 — Change `_recompute(self, notify_no_contour: bool = True)`: on a partial
      result set the LED red and the text to "contour: none — <short reason>";
      show the detailed `QMessageBox.warning` **only** when `notify_no_contour`.
      On success set the LED green.
- [x] 4.5 — Add `_on_recompute_clicked` slot calling
      `self._recompute(notify_no_contour=False)`; connect the Recompute button to
      it. Ensure `_load_current` calls `self._recompute()` (default `True`).

**Files Modified:**
- `src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py`

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [x] `ContourParamToggles`: defaults are median `False` / others `True`;
      non-bool field raises; `to_dict`/`from_dict` round-trip; unknown sub-key
      raises; absent sub-key → field default.
- [x] `GestureContourParams.effective_*()`: enabled returns the stored value;
      disabled returns the skip value (`0.0` / `None` / `0.0` / `0.0`).
- [x] `save_contour_params` → `load_contour_params` round-trips the toggles.
- [x] `load_contour_params` on a JSON **without** `param_enabled` loads and yields
      default toggles (no `missing`/`extra` error).
- [x] `load_contour_params` on a JSON with a non-bool flag raises with file path.

### Integration Tests
- [x] **Parity (mirror `tests/test_rf_response_fields_parity.py`):** for a synthetic
      grid, `compute_radial_foot_stages` invoked with the GUI's `effective_*()`
      values equals `compute_radial_foot_boundary` invoked with the same
      `effective_*()` values — for toggles ALL ON and for each toggle OFF —
      `np.testing.assert_array_equal` on the contour. Add to
      `tests/test_rf_radial_foot_boundary.py` (already imports both entry points).
- [ ] Pipeline honour: `_build_gesture_results` / extraction with a
      `per_gesture_params` carrying `median_filter_size` OFF produces the same grid
      as passing `median_filter_size=None` directly.

### Manual Verification
- [ ] Launch the tuner (via the AnalysisRunnerGUI / launcher); untick median →
      Recompute → preview changes, LED reflects result, no modal.
- [ ] Untick radial_gauss → preview shows un-pre-smoothed field; Validate; confirm
      the JSON has `param_enabled.radial_gauss_sigma = false`.
- [ ] Switch to a gesture whose contour fails → modal appears; click Recompute →
      no modal, red LED stays.
- [ ] Click "?" → dialog lists all params/toggles/buttons.

### Edge Cases
- [ ] All four toggles OFF: no threshold beyond floor, no median, no pre-smoothing,
      no envelope smoothing — still fails-fast on genuine no-data (all-NaN grid).
- [ ] `radial_hess_sigma` has no checkbox and cannot be disabled from the GUI.
- [ ] Re-Validate preserves `created_at` (existing behaviour unaffected by the new key).

---

## Documentation Plan

- [ ] Update the tuner module docstring in `rf_contour_tuning_viewer.py` (the
      five-parameter description) to note the four toggles + `radial_hess` always-on.
- [ ] Update `rf_contour_params_io.py` module docstring / `PARAM_KEYS` comment to
      document the optional `param_enabled` object and its defaults.
- [ ] Note the toggle semantics in the active `spatial-tune-rf-contours.md` design
      record (or link this plan from it).
- [ ] No README/CLAUDE.md architecture change required (no new stage, no new path).

---

## Rollback Plan

1. The change is additive and gated: the `use_tuned_params=False` path is
   untouched, so reverting is low-risk.
2. **Data considerations:** the only on-disk change is a new optional
   `param_enabled` key in contour-params JSONs. Old code cannot read a JSON that
   *has* the key (its `extra`-key check rejects it), so if rolling back after any
   JSON was written with the key, delete/re-Validate those JSONs (the format is
   unreleased, so this is acceptable). No migration script needed.
3. **Rollback procedure:** revert the feature branch merge commit; restore the
   four files to their pre-branch state.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `compute_radial_foot_boundary`/`_stages` rejects `gauss_sigma=0` | Low | Med | Phase 2.4 verifies; if so, gate skip at the call site (pass raw field / skip smoothing branch) instead of `sigma=0` |
| Median-OFF default silently changes a locally-written test JSON's result on reload | Med | Low | Format unreleased (no shipped configs); default is documented + surfaced in the checkbox; user explicitly chose median-OFF default |
| Preview vs. pipeline drift on the skip decision | Low | High | Single `effective_*()` resolvers used by both; parity test covers ON and OFF |
| Adding `toggles` field breaks existing `GestureContourParams(...)` positional constructions in tests | Med | Low | `toggles` has a `default_factory`, so it is keyword-optional; audit `tests/` call sites |
| Nested `from_dict` too strict (requires all sub-keys) → old JSON fails | Med | Med | `from_dict` uses `.get(name, default)` per field, only rejecting *unknown* sub-keys |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (config + persistence) | ~120 lines across 2 files | None |
| Phase 2 (pipeline consumers) | ~15 lines across 2–3 files | Phase 1 |
| Phase 3 (GUI checkboxes) | ~80 lines, 1 file | Phase 1 |
| Phase 4 (GUI help + status) | ~90 lines, 1 file | Phase 3 |
| Tests | parity + unit, ~150 lines | Phases 1–2 |

---

## References

- Related plan: `docs/development/plans/active/spatial-tune-rf-contours.md`
  (built the tuner; "Byte-identical no-regression note" on `median_filter_size=null`)
- Related plan: `docs/development/plans/active/spatial-configure-stroke-axis.md`
- Checkbox↔config precedent: `slim_uv_config_viewer.py` + `SlimUvCleanSteps`
  (`surface/slim_uv_config_io.py`)
- Parity-test template: `tests/test_rf_response_fields_parity.py`
- Pipeline-engineering guides: `~/.claude/knowledge/data-pipeline-engineering/`
  (`01` two-level SSOT, `02` config-resolved-once / avoid control coupling,
  `03` provenance & reproducibility, `05` DRY / naming / connascence)

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- docs/development/plans/active/rf-contour-tuning-param-toggles-and-help.md
- src/analysis/receptive_field_mapping/data/rf_boundary_preparation.py
- src/analysis/receptive_field_mapping/data/rf_boundary_types.py
- src/analysis/receptive_field_mapping/data/rf_contour_params_io.py
- src/analysis/receptive_field_mapping/gui/rf_contour_tuning_viewer.py
- src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py
- tests/test_rf_contour_params.py
- tests/test_rf_radial_foot_boundary.py

> **Commit note:** `rf_boundary_types.py`, `rf_boundary_preparation.py`,
> `rf_boundary_extraction.py`, and `tests/test_rf_radial_foot_boundary.py` were
> already modified in the working tree (parent `feature/spatial-configure-stroke-axis`
> work) before this feature branched. Their diffs therefore contain pre-existing,
> unrelated edits alongside this feature's additions. Committing them as whole files
> bundles that pre-existing work. See the commit decision at implementation time.

---
