# Plan: DAG Launcher — boundary_method Dropdown and Conditional Visibility

**Date:** 2026-06-24
**Author:** Basil Duvernoy
**Status:** Approved
**Base Branch:** `feature/port-pipeline-gui-launcher`
**Branch:** `feature/dag-launcher-boundary-method-dropdown`

---

## Overview

The `spatial_extract_boundaries` task in the DAG launcher GUI exposes `boundary_method` as a free-text `QLineEdit`, but it accepts only two valid values (`"gradient"` / `"inflection"`). Additionally, `inflection_sigma` and `iff_metric` are always rendered regardless of whether they are relevant to the currently selected method/mode, creating a noisy panel that obscures option dependencies. This plan converts `boundary_method` to a `QComboBox` dropdown and introduces conditional section visibility so dependent options appear only when they apply.

---

## Problem Statement

When a user opens `spatial_extract_boundaries` in the DAG launcher:

1. `boundary_method` appears as a plain text field with no hint of valid values, making it easy to enter an invalid string that will fail at runtime.
2. `inflection_sigma` is always visible even when `boundary_method == "gradient"`, where it has no effect. This hides the causal link between the two options.
3. `iff_metric` is always visible even when `neuron_mode == "spike"`, where IFF metrics are irrelevant.

The result is a panel that neither validates input nor communicates option relationships, increasing the risk of misconfiguration.

---

## Goals

### In Scope
1. Convert `boundary_method` from a free-text field to a `QComboBox` with choices `"gradient"` and `"inflection"`.
2. Hide `inflection_sigma` when `boundary_method == "gradient"`; show it only when `"inflection"` is selected.
3. Hide `iff_metric` when `neuron_mode == "spike"`; show it only when `"iff"` is selected.

### Out of Scope
- Conditional visibility for options in other tasks (deferred; extend `_OPTION_VISIBILITY` later if needed).
- Visual grouping / section reordering of `spatial_extract_boundaries` options.
- Any changes to the underlying pipeline code or YAML structure.

---

## Success Criteria

- [ ] `boundary_method` renders as a dropdown showing "Gradient (default)" and "Inflection".
- [ ] `inflection_sigma` section is hidden on load when `boundary_method == "gradient"` and visible when `== "inflection"`.
- [ ] `inflection_sigma` toggles visibility live as the dropdown changes.
- [ ] `iff_metric` section is hidden on load when `neuron_mode == "spike"` and visible when `== "iff"`.
- [ ] `iff_metric` toggles visibility live as the `neuron_mode` dropdown changes.
- [ ] Saving the config and reopening the launcher restores correct initial visibility from the saved values.
- [ ] No regressions in other tasks (all other options render and save correctly).

---

## Technical Design

### Approach

All changes are confined to `src/utils/gui/dag_launcher/task_detail_panel.py`. The existing `_OPTION_ENUMS` dict already drives `QComboBox` rendering for options like `neuron_mode`, `cmap`, and `contour_color`; `boundary_method` simply needs to be added to it.

Conditional visibility is implemented by:
1. A new module-level `_OPTION_VISIBILITY` table that maps controller keys → dependent keys → the set of controller values that make each dependent visible.
2. A `self._sections: dict[str, QWidget]` registry populated in `show_task()` as each section widget is created.
3. An initial-visibility pass at the end of `show_task()` that reads the current option values and calls `setVisible()`.
4. An update in `_make_enum_handler()` that calls `setVisible()` on affected dependents each time an enum value changes.

This matches the existing pattern in `cluster_group_dialog.py` (`_on_outlier_method_changed` / `_sync_outlier_param_row`) and `clustering_config_dialog.py` (checkbox `.toggled.connect(widget.setVisible)`).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `_OPTION_VISIBILITY` table + `self._sections` registry | Generalises to future controller/dependent pairs with zero per-task code; consistent with existing `_OPTION_ENUMS` pattern | None significant | **Chosen** |
| Per-task custom section builder for `spatial_extract_boundaries` | Very explicit | Requires duplicating all scalar/enum rendering logic; doesn't scale | Rejected |
| Nested `QGroupBox` with a dedicated method for each dependency | Clear hierarchy | One method per controller/dependent pair; verbose and not DRY | Rejected |

### Architecture Changes

No new files or classes. Changes to one existing file:

- `src/utils/gui/dag_launcher/task_detail_panel.py` — add `boundary_method` to `_OPTION_ENUMS`; add `_OPTION_VISIBILITY` constant; add `self._sections` instance dict; update `show_task()` and `_make_enum_handler()`.

---

## Implementation Plan

### Phase 1: enum registration and visibility infrastructure

**Goal:** Add `boundary_method` to the dropdown registry and introduce the visibility system.

- [ ] 1.1 — Add `"boundary_method"` entry to `_OPTION_ENUMS` (after `"contour_color"`, before closing `}`)
  ```python
  "boundary_method": [
      ("Gradient (default)", "gradient"),
      ("Inflection", "inflection"),
  ],
  ```
- [ ] 1.2 — Add `_OPTION_VISIBILITY` constant after `_OPTION_ENUMS`:
  ```python
  _OPTION_VISIBILITY: dict[str, dict[str, set]] = {
      "boundary_method": {"inflection_sigma": {"inflection"}},
      "neuron_mode":     {"iff_metric": {"iff"}},
  }
  ```
- [ ] 1.3 — Add `self._sections: dict[str, QWidget] = {}` to `__init__`, after `self._task_name`.

**Files Modified:**
- `src/utils/gui/dag_launcher/task_detail_panel.py` — three localised additions (~10 lines total)

**Dependencies:** None

### Phase 2: wire up visibility in show_task() and _make_enum_handler()

**Goal:** Populate `_sections` during the build loop and apply / maintain visibility.

- [ ] 2.1 — In `show_task()`, add `self._sections.clear()` immediately after the `deleteLater()` while-loop.
- [ ] 2.2 — In `show_task()`, add `self._sections[key] = widget` on the line before every `self._layout.insertWidget(i, widget)` call.
- [ ] 2.3 — In `show_task()`, add the initial visibility pass after the loop:
  ```python
  for controller, dependents in _OPTION_VISIBILITY.items():
      if controller not in options:
          continue
      current_val = options[controller]
      for dep_key, visible_when in dependents.items():
          if dep_key in self._sections:
              self._sections[dep_key].setVisible(current_val in visible_when)
  ```
- [ ] 2.4 — In `_make_enum_handler()`, add the live-update block after `self.task_changed.emit()`:
  ```python
  if key in _OPTION_VISIBILITY:
      for dep_key, visible_when in _OPTION_VISIBILITY[key].items():
          if dep_key in self._sections:
              self._sections[dep_key].setVisible(saved_val in visible_when)
  ```

**Files Modified:**
- `src/utils/gui/dag_launcher/task_detail_panel.py` — ~15 additional lines across two methods

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] No unit tests exist for the GUI panel; no new test file required for this change.

### Manual Verification

Launch the GUI and select `spatial_extract_boundaries`:

- [ ] `boundary_method` renders as a `QComboBox` (not a text field); default shows "Gradient (default)".
- [ ] `inflection_sigma` group box is **not visible** on initial load (gradient is the default).
- [ ] Change `boundary_method` to "Inflection" → `inflection_sigma` group box **appears immediately**.
- [ ] Change back to "Gradient (default)" → `inflection_sigma` group box **disappears immediately**.
- [ ] `iff_metric` group box is **visible** on initial load (neuron_mode defaults to "iff").
- [ ] Change `neuron_mode` to "Spike" → `iff_metric` group box **disappears immediately**.
- [ ] Change `neuron_mode` back to "IFF (default)" → `iff_metric` group box **reappears**.
- [ ] Save the config with `boundary_method = "inflection"` and `neuron_mode = "spike"`, reopen the launcher → `inflection_sigma` visible, `iff_metric` hidden on load.
- [ ] Switch to a different task and back to `spatial_extract_boundaries` → visibility is re-applied correctly.

### Edge Cases
- [ ] Tasks without `boundary_method` or `inflection_sigma` in their options dict render without error (guards in `_OPTION_VISIBILITY` loop).
- [ ] Tasks that call `show_task()` recursively (e.g., after cluster group delete) rebuild `_sections` cleanly and re-apply visibility.

---

## Documentation Plan

- [ ] No user-facing docs needed — this is an internal GUI improvement.
- [ ] No CLAUDE.md update needed — no architectural change.

---

## Rollback Plan

This change is limited to one file with no data migration:

1. `git revert` the single commit or `git checkout HEAD~1 -- src/utils/gui/dag_launcher/task_detail_panel.py` to restore the original file.
2. No YAML data is affected; reverting does not invalidate any saved config.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `show_task()` re-entrancy: a recursive call rebuilds `_sections` before the outer call reads it | Low | Low | `self._sections.clear()` at the top of `show_task()` ensures the registry is always rebuilt from scratch; the initial-visibility pass runs at the end of each call, so it always sees the freshly built registry |
| `None` sentinel in `_OPTION_ENUMS` hides a dependent that should be visible | Low | Low | `None in {"inflection"}` is `False`, so `None` values correctly hide dependents — no special-casing needed |
| Qt layout collapse leaves an awkward gap | Low | Low | `setVisible(False)` on a `QGroupBox` in a `QVBoxLayout` collapses both the widget and its spacing |

---

## References

- Existing pattern: `src/utils/gui/dag_launcher/cluster_group_dialog.py` — `_on_outlier_method_changed` / `_sync_outlier_param_row`
- Existing pattern: `src/utils/gui/dag_launcher/clustering_config_dialog.py` — `_vf_check.toggled.connect(self._vf_nested.setVisible)`
- Pipeline implementation: `src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — `boundary_method` validation at lines 150–154
