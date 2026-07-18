# Brainstorm: RF contour parameter sweep / auto-tuning

**Started:** 2026-07-18   **Last matured:** 2026-07-18   **Status:** Handed off

## Real goal (north star) — PIVOTED 2026-07-18
The true pain is NOT searching the landscape. It is: *the IFF heatmap looks
visually fine, the user believes the params are okay, yet the contour-fit fails
("can't fit the contour") with no usable indication of why or which knob to
change.* This is a **diagnosis / feedback** problem, not a search problem. The
brute-force framing was a proposed solution to the wrong problem.

Quality of the *contour shape* is judged purely by the user's eye (3D interactive
map + raw & processed heatmaps). But "does a contour fit AT ALL" is
machine-checkable — and that is exactly the thing the user is stuck on.

## The crux (decides everything)
Is "a good contour" **computable** or **purely visual (the researcher's eye)**?
- Purely visual  -> build a *viewer* (gallery / filmstrip of variants to scan).
- Computable (even partial score) -> build a *ranker/optimizer* (auto-search,
  present ranked winner).

## Current system (grounding)
- 5 params, 4 toggleable: `min_overlap_pct`, `median_filter_size`,
  `radial_gauss_sigma`, `radial_hess_sigma` (always on),
  `radial_envelope_smooth_sigma`.
- Per-(session, gesture): dial spinners+toggles -> Recompute (preview) ->
  Validate (save JSON). One combo, one value at a time.
- Recompute -> `build_session_grid` -> `compute_radial_foot_stages` -> contour
  drawn (green LED) OR partial/no-contour (red LED) with a reason string.
- `_classify_contour_error` already maps failure reasons -> "try this slider"
  hints. Seed for guidance.

## Alternatives on the table
- **A. Full grid sweep -> thumbnail gallery** (literal brute force). Only useful
  with a ranker; otherwise just moves the eyeballing downstream.
- **B. One-at-a-time sensitivity filmstrip** — hold 4 fixed, sweep 1. Directly
  answers "which param matters for THIS dataset." Cheap.
- **C. Auto-optimize** — define contour-quality score, search param space,
  present ranked winner + let user nudge. Requires computable quality.
- **D. Guided next-step** — extend `_classify_contour_error` so the tool
  recommends the next param to change. Requires little/no new compute.

## Key findings (2026-07-18)
- **Quality = pure eye** (3D + raw/processed heatmaps). Rules out an
  auto-*selector*; triage (does-a-contour-fit) is still machine-checkable.
- **Failure mode = "green-but-can't-fit":** heatmap looks fine, contour stage
  fails, no idea which knob.
- **The hint already exists but is hidden.** `_classify_contour_error` maps each
  failure -> cause + which slider. But the full hint modal only fires on
  session/gesture SWITCH (`notify_no_contour=True`); on Recompute / toggle it is
  suppressed and the status line shows only the first sentence of the error
  (actionable part stripped). Surfacing it during active tuning is a cheap, high-
  value fix.
- **Mental-model trap:** heatmap is governed by `min_overlap_pct` +
  `median_filter_size`; contour fit is governed downstream by the 3 sigmas
  (`radial_gauss_sigma`/`radial_hess_sigma`/`radial_envelope_smooth_sigma`) which
  barely move the heatmap. "Heatmap looks good => params good" is a false
  inference. The failure is visible in the λmax panel, which the user may not be
  looking at.

## Reframed direction — SETTLED 2026-07-18
Want confirmed = **(a) live "why did it fail + which knob", while tuning.** NOT a
sweep, NOT an optimizer. The rescue sweep (b) is parked, the landscape explorer
(c) is dropped.

Decisive new fact: the user reports the **λmax peak AND its ring are visibly
present**, yet extraction still fails. So this is a *parameter / algorithmic-
threshold* problem, not a data (footless-peak) problem. The failure contradicts
what the eye sees.

Implication for design: a *canned* which-knob hint may be insufficient precisely
here — "try raising hess_sigma" is a guess when a catchable ring is visibly
present. The valuable version is a **quantitative diagnostic**: report the actual
computed reason the extractor walked past a visible ring (e.g. ring λmax below the
sought contour level; ring doesn't close; ring clipped by footprint mask). This
requires surfacing intermediate quantities from `compute_radial_foot_stages` /
`rf_boundary_extraction`, not just the canned error string.
(Subagent dispatched to inventory the failure points + available diagnostics.)

## Failure taxonomy (from code inventory, 2026-07-18)
Ring-present failures are ALWAYS one of 3 branches in `rf_radial_foot_boundary.py`:
- **Site 3** (line ~322) "no λmax foot plateau on any ray": detection-gate
  rejection. Discriminator: `contour_unsnapped_rc` absent, `found.sum()==0`,
  `nanmax(lmax)>0`. Knob: gauss/hess σ — OR non-exposed `prominence` (caveat).
  NOTE: its message "field too flat or peak runs off footprint" is MISLEADING
  when a ring is visible — a likely root of the user's distrust.
- **Site 5** (line ~446) "seed not inside contour ∩ footprint": footprint clipped
  the seed. Discriminator: `contour_unsnapped_rc` present, `footprint[peak]==False`.
  Knob: lower `min_overlap_pct`.
- **Site 4** (line ~398) "no contour encloses the peak": traced but open/fragmented.
  Discriminator: `contour_unsnapped_rc` present, `len(contours)>0`, none enclose.
  Knob: `radial_envelope_smooth_sigma` / `median_filter_size`.
- **Master discriminator:** did `contour_unsnapped_rc` get produced? No -> detection
  (Site 3); Yes -> geometry (Site 4/5), split by `footprint[peak]`.
- Pipeline layer (`rf_boundary_extraction.py`) DISCARDS the reason (logger.warning
  + returns None); the `error`/diagnostics only survive on the GUI
  `compute_radial_foot_stages(allow_partial=True)` path.

## Buildable shape — two tiers (user's call)
- **Tier 1:** un-suppress existing `_classify_contour_error` hint on Recompute +
  show full (not truncated) reason. Cheap; canned hints; misleading text remains.
- **Tier 2 (solves the stated problem):** enrich the 3 raise sites with in-scope
  numbers (`found.sum()`, `nanmax(lmax)`, `footprint[peak]`, `len(contours)`) and
  surface a live per-branch quantitative diagnostic + correct knob. Requires
  threading diagnostics out of the extractor (or enriching raise strings) + panel
  UI. Tier 1's UI work is a subset, so not wasted if started first.
- Plan MUST check whether `prominence`/`plateau_size` are exposed knobs or
  hardcoded (decides if Site-3 failures are user-fixable at all).

## Open questions
- Are `prominence`/`plateau_size` exposed or hardcoded? (blocks Site-3 fixability)
- Tier 1 vs Tier 2 ambition (recommendation: Tier 2).

## Status: MATURED — ready for /plan-create (Tier 2) pending user go-ahead.

## Session log
- 2026-07-18: Framed intent; grounded in current tuner; posed the compute-vs-eye
  crux and 4 candidate forms (A gallery / B sensitivity / C optimizer / D guide).
- 2026-07-18: Matured to Tier-2 diagnostic; mapped 3-branch failure taxonomy;
  confirmed prominence/plateau_size hardcoded; user chose to expose the gate.
  Handed off to plan `docs/development/plans/pending/rf-contour-failure-diagnostics.md`.
