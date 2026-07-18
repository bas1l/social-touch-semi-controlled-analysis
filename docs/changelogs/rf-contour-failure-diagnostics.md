# RF Contour Live Failure Diagnostics + Exposed Plateau Gate

**Date:** 2026-07-18
**Author:** Basil Duvernoy
**Plan:** `docs/development/plans/active/rf-contour-failure-diagnostics.md`

## Summary

The RF contour tuner (`RFContourTuningViewer`) now reports **why** contour
extraction failed, in plain language with the actual numbers that gated it, and
the plateau-detection gate that most often causes the "ring visible but can't
fit" case is now user-tunable end-to-end.

## Changes

- **Live per-branch failure diagnostics.** The three ring-present failure sites
  in the radial-foot detector now raise a `ContourExtractionError` carrying a
  frozen `ContourFailureDiagnostics` payload (a `ContourFailureBranch` enum plus
  the structured numbers each branch measured). `compute_radial_foot_stages(
  allow_partial=True)` surfaces that payload under a new `diagnostics` key. The
  tuner owns a single `branch -> (cause, which-knob)` map and renders the live
  quantitative message on the quiet Recompute/toggle path — not only on a
  gesture switch.
  - `NO_RADIAL_PLATEAU` — no ray formed a qualifying λmax plateau
    (`n_angles`, `found_count`, `peak_lmax`).
  - `SEED_OUTSIDE_FOOTPRINT` — a contour traced but the seed cell is not painted
    (`footprint_at_seed`, `footprint_cells`).
  - `NO_ENCLOSING_CONTOUR` — contour(s) traced but none enclose the peak
    (`n_candidate_contours`).
- **Fixed the misleading Site-3 message.** The old "field too flat or peak runs
  off the data footprint" text no longer appears for a ring-present failure; the
  detector now states that the peak-detection gate found no radial ridge.
- **Exposed `prominence` / `plateau_size` as tunable params.** Added as fields on
  `GestureContourParams` (with `effective_prominence()` / `effective_plateau_size()`
  resolvers and a `prominence` toggle on `ContourParamToggles`), as defaults on
  `BoundaryParams`, in the DAG YAML, in the params-IO schema, and as GUI widgets.
  A Validated value is threaded through **both** the GUI preview and the batch
  pipeline (`rf_boundary_extraction`), so it actually changes the contour produced
  downstream.
- **Backward-compatible JSON.** `prominence` / `plateau_size` are
  *sanctioned-optional* top-level keys: absent, they resolve to documented
  defaults (`None`, `1`) at the single loader boundary, so pre-feature
  `*_contour_params.json` files still load. Present-but-malformed values still
  raise with file context (no silent fallback).

## Behavioral note

Minor, intended: the peak-not-found / peak-at-border partials carry no
diagnostics payload (their failure is a plain `ValueError`, not a
`ContourExtractionError`). The tuner now shows the raw error text for those cases
instead of the old canned "try knob X" hint. Only the three enumerated
ring-present branches get the structured live diagnostic.

## Rollback note

Because the two gate keys are sanctioned-optional while the feature is present,
reverting the code re-enables the loader's strict unknown-key rejection. Any
`*_contour_params.json` written **with** `prominence` / `plateau_size` after a
Validate must have those keys stripped (or be re-Validated pre-revert) before the
reverted loader will accept them. No pipeline output schema change; no migration.
