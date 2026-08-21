# Plan: Depth-Weighted IFF Attribution in Receptive-Field Mapping

**Date:** 2026-08-21
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/port-stroke-centroid-baseline`
**Branch:** `feature/depth-weighted-iff-attribution`

---

## Overview

Every vertex inside a contact patch is currently credited with the neuron's full instantaneous
firing frequency (IFF), regardless of how deeply the skin was actually indented there. This plan
uses the per-vertex penetration depth field — already produced upstream as parquet sidecars — to
weight that credit, so a vertex pressed to 47% of a frame's maximum depth receives 47% of the
weight rather than 100%. The weight is `(depth / frame max depth) ** alpha`, and `alpha = 0`
reproduces today's behaviour exactly, which is the proof that the machinery is wired correctly.

## Problem Statement

`_compute_touch_rf` credits every contacted vertex identically within a frame. This inflates the
flanks of a contact patch with their neighbour's best moment, and it can put the receptive-field
peak on the wrong vertex.

Worked example — five vertices in a row, a finger sweeping left to right across three frames, true
receptive field at v3, IFF of 50 / 100 / 40 Hz:

| vertex | today | depth-weighted |
|---|---|---|
| v1 (one frame) | 50.0 | 50.0 |
| v2 (flank) | **75.0 <- peak** | 65.9 |
| v3 (**true RF centre**) | 63.3 | **73.5 <- peak** |
| v4 (flank) | 70.0 | 57.1 |
| v5 (one frame) | 40.0 | 40.0 |

In the frame where the finger sat on v3 and the neuron fired at 100 Hz, v2 and v4 were inside the
patch and received that 100 Hz at full credit — identical to v3. Today's map therefore peaks on v2.
Weighting corrects it because v2 was pressed to only 47% of that frame's maximum depth.

Secondary problem: the same three-line averaging block exists in **nine** places (the pipeline,
`rf_population_heatmap.py`, and seven copies across GUI viewer windows), none of which share a
helper and none of which have any test coverage. Adding weighting to one of them guarantees that
the viewer windows draw a different map from the one written to disk.

## Goals

### In Scope

1. Commit the already-built depth-field loader (`contact_depth_field_io.py` + 24 tests) and its
   `pyarrow>=15` declaration.
2. Merge the nine duplicate accumulator blocks into one shared, tested function — with **no
   behaviour change** — pinned by characterization tests.
3. Feed per-contact-point penetration depth through the playback loader and its on-disk cache.
4. Apply depth weighting inside the single shared accumulator, with explicit guards on every
   degenerate case.
5. Emit a confidence channel (`weight_sum`, `n_eff`) alongside every weighted estimate.
6. Prove `alpha = 0` reproduces today's output exactly.
7. Thread `depth_weight_alpha` from YAML to the estimator as a **required** argument, recorded in
   run provenance.
8. Ship one diagnostic that can falsify the whole premise (per-vertex weight variation).

### Out of Scope

- **Automatic selection of alpha.** No held-out prediction harness, no cross-validation, no sweep
  framework. `alpha` is fixed at `1.0` in config; the dial exists only so the parity test can run.
- **Damping thin-evidence vertices inside the estimate.** Rejected — see Alternatives Considered.
  Confidence is a separate channel consumed at display time.
- **Weighting the `max` map.** `np.maximum.at` has no meaningful weighted analogue.
- **Cross-touch weighting** in `rf_population_heatmap.py` (deeper touches outranking shallower
  ones). The shared helper is introduced there; only per-frame weighting is applied.
- **Fixing the two KDTree vertex-snapping sites** (`rf_explorer_data.py:345`,
  `touch_population_data.py:686`). Known-bad, tracked separately.
- **`rf_cluster_pipeline.py:181-190`** float-coordinate-tuple aggregation. Needs a `vertex_id`
  before weighting can reach it.
- **The `touch_analytics` clustering finding** (brainstorm section 2). Independent, unresolved,
  tracked separately.
- **Visual verification of the depth field past Kinect Space 1.** Explicitly ruled out as a gate in
  brainstorm section 4 — weighting only redistributes credit *within* a patch via `vertex_id`, and
  the RF chain is translation-invariant.

## Success Criteria

- [ ] With `depth_weight_alpha = 0.0`, `single_touch_rf_maps_mean.npz` is **byte-identical** to the
      pre-change output on the same inputs.
- [ ] Hand-computed weighted mean over the 7-vertex / 3-frame fixture matches the implementation to
      `rtol=1e-12`, including v6 (always shallow) and v7 (always deep) both returning approximately
      their unweighted values.
- [ ] Exactly **one** function in the codebase computes a per-vertex weighted mean; the other eight
      call sites delegate to it.
- [ ] A characterization test pins each of the nine former call sites' output across the merge.
- [ ] `d_max == 0`, `sum(w) == 0`, and NaN depth each raise a typed error with file/frame/vertex
      context. No NaN reaches an output array.
- [ ] `weight_sum` and `n_eff` are emitted per vertex and present in the saved artifacts.
- [ ] `depth_weight_alpha` appears in `single_touch_rf_summary.json`, so a config change invalidates
      the stage and two runs are distinguishable on disk.
- [ ] Calling the estimator without `alpha` raises `TypeError`. There is no default value anywhere.
- [ ] The diagnostic reports per-vertex weight variation for one session and states whether the
      feature can have any effect at all on that data.

## Definitions

- **weight (`w`)**: how fully a vertex was pressed relative to the deepest point of that same
  frame. `w = (max(depth_i, 0) / max_j max(depth_j, 0)) ** alpha`. Range `[0, 1]`; exactly `1.0`
  for the deepest vertex in the frame.
- **max-normalised**: divided by the frame's own **maximum** depth, not the sum of depths. Chosen
  so `alpha=0` yields all-ones and large alpha yields the old `contact_depth = max()`.
- **normalise-before-exponent**: `(d/d_max)**alpha`, not `d**alpha / sum(d**alpha)`. At `alpha=0`
  the first gives 1 for every vertex; the second gives `1/K`. Only the first preserves the baseline.
- **reproduces today exactly**: `np.array_equal` on the output arrays — not `assert_allclose`.
  Achieved by running the *same* code path with `weights = ones`, so the reduction call and
  summation order are literally identical. If this proves unachievable it is downgraded to a
  **stated tolerance** and the plan is amended; it is never silently tested with `approx`.
- **confidence channel**: `weight_sum = sum(w)` and Kish effective sample size
  `n_eff = (sum w)^2 / sum(w^2)`, emitted per vertex beside the estimate. Consumed by display code
  to dim or mask thin-evidence vertices. It never modifies the estimate itself.
- **no behaviour change (Phase 2)**: for every one of the nine call sites, output arrays are
  `np.array_equal` to their pre-refactor values on the pinned fixtures.

---

## Technical Design

### Approach

One weight function, one accumulator, one estimator — three separate functions because they are
three separate reasons to change.

```
w = (max(d, 0) / max_frame(max(d, 0))) ** alpha        # weight function, knows alpha
value[v] = sum_frames(w * IFF) / sum_frames(w)         # accumulator, knows nothing about alpha
n_eff[v] = (sum w)^2 / sum(w^2)                        # confidence, separate channel
```

The denominator is the **weight sum**, never the frame count. Dividing by the count would leave the
weight in the result as a scale factor and the number would stop being a firing rate: a vertex
touched three times at 100 Hz must report 100 Hz whatever its weights were.

`contact_count` in `_compute_touch_rf` is already `float64` and already the divisor, so it becomes
the weight-sum accumulator with no structural change — but it is **renamed** (`weight_sum`), because
silently keeping a name while changing its definition is the most damaging possible outcome for
anyone re-reading an old figure.

**Critically, there is no `if alpha == 0` branch.** `alpha = 0` runs the same code with
`weights = ones`. A branch would be control coupling and would make the baseline test vacuous — it
would test the old code rather than proving the new code reduces to it.

#### What the weighting can and cannot do

Adding two archetypes to the worked example — v6 always shallow (stroke rim), v7 always deep
(stroke midline) — **neither moves**:

| vertex | weights across frames | today | weighted |
|---|---|---|---|
| v6 always shallow | `0.13, 0.13, 0.10` — flat | 63.3 | 65.8 |
| v7 always deep | `0.92, 0.93, 0.95` — flat | 63.3 | 63.3 |
| v3 RF centre | `0.42, 1.00, 0.50` — **varying** | 63.3 | **73.5** |

A roughly-constant weight cancels between numerator and denominator. **Absolute depth level is
irrelevant; only frame-to-frame variation in a vertex's own depth does anything.** `n_eff` confirms
it — v6 retains 2.96 of 3 effective observations despite weights of 0.1.

This yields a real falsification path, which Phase 6 tests: **if depth is roughly uniform across
each contact patch in the real data, this feature produces today's map at every alpha.**

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **A: `sum(w*IFF) / sum(w)`** | Stays in Hz; single-frame vertices provably invariant; peak lands on v3; no tuning knob | A chronically-shallow vertex reports a normal rate, which reads as wrong until the mechanism is understood | **Chosen** |
| B: `sum(w*IFF) / N` | Damps always-shallow vertices strongly (v6: 63.3 to 7.9); `alpha=0` parity needs no extra condition | **Peak moves to v7, not v3.** Damping v6 and inflating v7 are the same operation. v7 is deep because of stroke geometry and arm curvature, not the neuron — so this imports stimulus geometry into the RF map. Output is no longer in Hz | Rejected |
| C: `sum(w*IFF) / (sum(w) + k)` | Damps v6 (to 17.4) *and* keeps the peak on v3 | `k` is a tuning knob that can move the peak — v3 leads v7 only while `k < 1.5` on the worked example. A parameter that decides where the receptive field is must be chosen on principle. Weakens parity: needs both `alpha=0` and `k=0` | Rejected for now |
| Sum-normalise: `d**a / sum(d**a)` | Each frame contributes equal total evidence at every alpha | At `alpha=0` gives `1/K`, not 1 — **destroys the baseline anchor**. Collapses a frame's contribution from `K` to `1`, a far more violent change than requested | Rejected |
| Softmax `exp(d/T)` | Spans the same limits | `T` carries units of mm and needs retuning per session; normalised powers are scale-invariant | Rejected (brainstorm section 4) |
| Weight the `max` map too | Consistency | A weighted maximum has no meaning; weighting changes its units | Rejected |
| Weight the pipeline only, leave 9 copies | Smallest diff | Viewer windows would draw the v2-peaked map while saved files hold the v3-peaked one — the one failure mode that actively misleads, and this feature is judged by eye | Rejected |
| Choose alpha by visual sharpness | Cheap | Sharpness improves under *any* concentration of credit, correct or not | Rejected (brainstorm section 4) |
| Recover `vertex_id` by KDTree | Avoids threading depth through the cache | Contract forbids it; measured 29 mm off-surface at touch boundaries | Rejected (data contract) |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs -> Outputs | Must NOT know about |
|---|---|---|---|
| `contact_depth_field_io.py` (exists) | Read + validate parquet sidecars, join reference PLY | `csv_path, expect_space, session_id` -> `DepthField` | IFF, receptive fields, alpha, weighting |
| `vertex_weights.py` (**new**) | Turn per-frame depths into per-frame weights | `depth_mm: (K,), alpha: float` -> `w: (K,) in [0,1]` | IFF, vertices, frames, accumulation, files |
| `vertex_accumulator.py` (**new**) | The one weighted per-vertex reduction | `(vertex_idx, values, weights, n_vertices)` -> `(value_sum, value_max, weight_sum, weight_sq_sum)` | alpha, depth, mm, parquet, GUI, files |
| `touch_playback_data.py` (changed) | Carry per-contact-point depth onto `TouchEvent` and its cache | CSV + sidecar -> `TouchEvent.frame_depths` | alpha, weighting, IFF semantics |
| `rf_single_touch_pipeline.py` (changed) | Orchestrate per-touch RF computation | `TouchEvent, alpha` -> mean/max pairs + confidence | parquet layout, PLY, YAML |

```
weight function      depth (mm)  ->  w in [0,1]      knows alpha, nothing else
accumulator          w + IFF     ->  sums            knows nothing about alpha
estimator            sums        ->  Hz + n_eff      knows nothing about depth
```

The thing that varies is the **weight function**, not the aggregation. Hiding `depth -> w` behind
one small function keeps the accumulator ignorant of alpha entirely, and makes an alternative
weighting (saturating, thresholded) an addition rather than an edit.

### Known hazards this design must survive

1. **The cache dedup is depth-unsafe.** `_save_playback_cache` deduplicates contact-point groups by
   `id()` of the vertex array (`touch_playback_data.py:131`), exploiting that the CSV parser reuses
   one ndarray for consecutive identical `contact_points` strings. **Depth is not implied by the
   contact-point string** — two frames with identical contact coordinates can carry different
   penetration depths. Depth arrays must therefore be stored **per contact-frame**, not per unique
   group, or one frame's depth is silently broadcast across every frame sharing that group.
2. **There is no frame index to join on.** The playback loader never reads one; CSV rows are already
   at nerve rate (~1 kHz) and the 30 Hz Kinect structure survives only as *runs of identical
   `contact_points` text*. Attaching 30 Hz depth to 1 kHz rows requires either adding a frame column
   to `_REQUIRED_COLUMNS` (lines 22-30) or reconstructing runs by the same string-identity
   mechanism. **The join must be on frame_index *value*, never row position.**
3. **A pre-existing implicit weighting already exists.** Contact geometry is forward-filled *up* to
   the nerve rate (`:481-483`) while IFF is never averaged down, so each Kinect frame is credited
   about 33 times, weighted by how long it was held. Depth weights multiply on top of this. Not
   introduced here, but it must be known before interpreting any change in the result.
4. **Duplicate `(frame_index, vertex_id)` pairs.** The contract says possible; the design says not.
   Duplicates are currently harmless because they double numerator and denominator equally — a
   cancellation that only holds while the weight is 1. **Assert, do not reduce**: check uniqueness
   per frame where weights are built and raise on violation.
5. **`_CACHE_SCHEMA_VERSION` mismatch currently returns `None` and silently recomputes**
   (`:235`). Do not extend that pattern to the depth path. Bump the version **and** extend
   `required_keys` in the same change, or a v3 cache passes the version check while lacking depth.
6. **`_save_playback_cache` swallows write failures with `logger.warning`.** Pre-existing silent
   fallback; the depth path must not inherit it.
7. **Zero is not absent.** `depth = 0.0` (grazing) and `depth = missing` (no sidecar row) are
   different facts that both become `w = 0` at `alpha > 0`. The fail-fast check must happen at the
   loader boundary, where they are still separable.

---

## Implementation Plan

### Phase 1: Land the loader
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** Get the already-built, already-tested depth-field reader into version control.

- [x] 1.1 — Commit `src/analysis/receptive_field_mapping/data/contact_depth_field_io.py` and
      `tests/test_contact_depth_field_io.py` (24 tests) unchanged.
- [x] 1.2 — Commit the `data/__init__.py` re-exports.
- [x] 1.3 — Commit `pyarrow>=15` in `environment.yml` and `pyproject.toml`; run a `conda install`
      pass so the env matches a fresh solve (pyarrow is currently only `pip install`ed).
      **Deviation:** the `conda install` pass was **not** run. `pyarrow 25.0.1` is already
      pip-installed into `social-touch-analysis` and working; layering a conda install over a
      working pip install risks breaking the env during an unattended run. Both declarations are
      committed. To re-solve the env deliberately, run:
      `conda env update -n social-touch-analysis -f environment.yml --prune`
- [x] 1.4 — Commit `docs/data-contracts/contact-depth-field.md` and the brainstorm.

**Files Modified:** the four paths above.

**Dependencies:** None.

### Phase 2: Merge the nine accumulators — no behaviour change
**Goal:** Exactly one function computes a per-vertex mean, pinned by tests, before any weighting
exists.

- [ ] 2.1 — Write characterization tests pinning current output for all nine sites on synthetic
      fixtures. **This pinned output is the `alpha=0` baseline; the two tasks are the same task.**
- [ ] 2.2 — Create `data/vertex_accumulator.py` with
      `accumulate_vertex_values(vertex_idx, values, weights, n_vertices) -> AccumResult`
      (`value_sum`, `value_max`, `weight_sum`, `weight_sq_sum`). `weights` is **required**; callers
      pass explicit ones at this stage.
- [ ] 2.3 — Redirect all nine call sites to it. Note four are in `touch_playback_explorer.py`
      (`:308` replay-to-slider, `:549` replay-inclusive, `:602` incremental single-frame, `:959`
      export loop) and two are bincount-based rather than `np.add.at` — the helper must cover both
      shapes or the bincount sites keep a thin adapter.
- [ ] 2.4 — Confirm every characterization test still passes, `np.array_equal`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/vertex_accumulator.py` — new
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — `:83-85`
- `src/analysis/receptive_field_mapping/data/rf_population_heatmap.py` — `:23-27`
- `src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — `:635`
- `src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` — `:545`, `:628`
- `src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — `:308`, `:549`, `:602`, `:959`
- `tests/test_vertex_accumulator.py` — new

**Dependencies:** None (can run parallel to Phase 1).

### Phase 3: Carry depth to the estimator
**Goal:** Per-contact-point penetration depth reaches `_compute_touch_rf`, correctly aligned.

- [ ] 3.1 — Add `frame_depths: list  # (K_i,) float64` to `TouchEvent` (`:49-59`), aligned with
      `frame_vertex_indices`.
- [ ] 3.2 — Load the sidecar via `load_depth_field(...)` and join by `vertex_id` and `frame_index`
      **value**. Raise on a contacted vertex with no depth row — never `.get(vertex_id, 0.0)`.
- [ ] 3.3 — Assert `(frame_index, vertex_id)` uniqueness per frame; raise on violation.
- [ ] 3.4 — Expand 30 Hz depth to nerve rate by the same run mechanism as the vertices, and apply
      the identical `ffill` as `:481-483` or depth desynchronises from the points.
- [ ] 3.5 — Store depth **per contact-frame**, not per dedup group (hazard 1). Bump
      `_CACHE_SCHEMA_VERSION` 3 to 4 (`:20`), add keys to `required_keys` (`:243-250`), unpack them,
      add shape checks in the `:288-352` block, pass to the constructor at `:367`.
- [ ] 3.6 — Reject NaN depth at the loader boundary with file/frame/vertex context.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/touch_playback_data.py` — `:20`, `:22-30`, `:49-59`,
  `:120-137`, `:243-250`, `:288-352`, `:367`, `:481-483`, `:540-591`
- `tests/test_touch_playback_depth.py` — new

**Dependencies:** Phase 1.

### Phase 4: Apply the weighting
**Goal:** The estimator becomes a weighted mean, with every degenerate case raising.

- [ ] 4.1 — Create `data/vertex_weights.py`:
      `vertex_weights(depth_mm: np.ndarray, alpha: float) -> np.ndarray`. Clamp negatives
      (grazing) to 0, divide by the frame max, then raise to `alpha`. **Normalise before the
      exponent.** `alpha` is positional and required.
- [ ] 4.2 — Guard `d_max == 0` with an explicit raise. Do **not** rely on `NaN ** 0 == 1.0`.
- [ ] 4.3 — Rename `contact_count` to `weight_sum` in `_compute_touch_rf` (`:66`, `:87-92`).
- [ ] 4.4 — Pass real weights into the shared accumulator. `neuron_values[fi]` is currently a
      **scalar** broadcast over the frame's vertices; with weights it becomes a `(K_i,)` array.
- [ ] 4.5 — Guard `sum(w) == 0` per vertex with an explicit policy and raise. Never `0/0 -> NaN`
      into the map, and never fall back to the uniform mean.
- [ ] 4.6 — Leave `val_max` unweighted; document why in the docstring.
- [ ] 4.7 — Emit `weight_sum` and `n_eff = (sum w)^2 / sum(w^2)` per vertex into the returned
      structure and the saved `.npz`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/vertex_weights.py` — new
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — `:38-42`, `:64-66`,
  `:79-109`, `:224`, `:239-244`
- `tests/test_vertex_weights.py` — new

**Dependencies:** Phases 2, 3.

### Phase 5: Config, provenance, GUI option
**Goal:** `alpha` is configurable, required, and recorded.

- [ ] 5.1 — Add `depth_weight_alpha: 1.0` under `options:` for `spatial_map_single_touch` in
      `configs/analyse_workflow_processing_dag.yaml` (`:48-55`). **ruamel.yaml round-trip only.**
- [ ] 5.2 — Add to the stage registry `params` lambda in `scripts/analysis_workflow_processing.py`
      (`:1965-1972`), beside `neuron_mode`.
- [ ] 5.3 — Thread through `spatial_map_single_touch_flow` (`:429-455`) into
      `run_single_touch_rf_mapping` (`:112-118`) into `_compute_touch_rf` (`:38-42`). **Required
      argument at every level — no default value anywhere.**
- [ ] 5.4 — Add `depth_weight_alpha` to `_OPTION_GROUP_OF` in
      `src/utils/gui/analysis_runner_gui/task_detail_panel.py` (`:157-173`) —
      `spatial_map_single_touch` is in `_GROUPED_TASKS` (`:182`), so an ungrouped option **fails the
      render**.
- [ ] 5.5 — Record `depth_weight_alpha`, the sidecar path, and the cache schema version in
      `single_touch_rf_summary.json` (`:251-258`) so a config change invalidates the stage and two
      runs are distinguishable on disk.
- [ ] 5.6 — Mirror into `configs/analyse_workflow_dag.yaml` if that DAG exposes the same stage.

**Files Modified:** the five paths above.

**Dependencies:** Phase 4.

### Phase 6: Prove it, then try to break it
**Goal:** The baseline is proven and the premise is tested against real data.

- [ ] 6.1 — Parity test: `depth_weight_alpha = 0.0` produces output **byte-identical** to the
      Phase 2 pinned baseline. Model on `tests/test_rf_response_fields_parity.py`
      (`_assert_exactly_equal`, `:220`), with a docstring stating why exact equality is the right
      invariant.
- [ ] 6.2 — Explicitly cover clamped grazing vertices at `alpha=0`: `max(d,0)` is `0.0`, and
      `0.0 ** 0 == 1.0`, so they must still receive weight 1.
- [ ] 6.3 — Diagnostic script: for one session, report the distribution of **per-vertex weight
      variation** (max/min of each vertex's weights across its frames). Near-1 everywhere means the
      feature cannot change anything at any alpha.
- [ ] 6.4 — Run `alpha=0` vs `alpha=1` on one session; plot per-vertex `|delta|` against distance
      from the RF hotspot. **Expected:** near-zero at the centre, largest at the periphery. **If
      instead the whole map shifts uniformly**, the weights are picking up the ~33x frame
      re-emission (hazard 3), not spatial structure — stop and investigate before trusting any
      output.

**Files Modified:**
- `tests/test_rf_depth_weighting_parity.py` — new
- `scripts/diagnose_depth_weight_variation.py` — new

**Dependencies:** Phase 5.

---

## Testing Plan

### Unit Tests
- [ ] `vertex_weights`: `alpha=0` gives all ones, including for clamped grazing vertices.
- [ ] `vertex_weights`: `alpha=1` on depths `0.2, 0.8, 1.5, 0.7, 0.1` gives
      `0.13, 0.53, 1.0, 0.47, 0.07`.
- [ ] `vertex_weights`: deepest vertex always receives exactly `1.0`, every alpha.
- [ ] `vertex_weights`: scale invariance — doubling every depth leaves weights unchanged.
- [ ] `vertex_weights`: `d_max == 0` raises; NaN depth raises. Neither returns a value.
- [ ] `vertex_weights`: negative (grazing) depths clamp to 0 and never subtract from `sum(w)`.
- [ ] Accumulator: hand-computed weighted mean, not a self-comparison — a wrong weighting still
      returns a plausible Hz value.
- [ ] Accumulator: `sum(w) == 0` for a vertex raises.
- [ ] `n_eff`: flat weights give `n_eff` approximately `N`; one dominant frame gives `n_eff`
      approximately 1.

### Integration Tests
- [ ] The full 7-vertex / 3-frame fixture end to end: peak moves from v2 to v3; v1 and v5 unchanged;
      **v6 (always shallow) and v7 (always deep) both within 1e-9 of their unweighted values.**
- [ ] `alpha=0` byte-identical to the Phase 2 baseline through the whole pipeline.
- [ ] Cache round-trip: write v4, read back, depth arrays match per **frame** — specifically a case
      where two frames share a `contact_points` string but carry different depths (hazard 1).
- [ ] A v3 cache on disk is rejected loudly, not silently recomputed into a depth-free result.

### Manual Verification
- [ ] Launch the GUI (`scripts/launch_pipeline_gui.py`); confirm `depth_weight_alpha` renders in the
      task detail panel and the stage runs.
- [ ] Open a viewer window and the saved map for the same touch; confirm they now agree.
- [ ] Confirm `single_touch_rf_summary.json` carries alpha, and that changing alpha re-runs the
      stage.

### Edge Cases
- [ ] A frame where every vertex is grazing (`d_max = 0`) raises with frame context.
- [ ] A vertex contacted in exactly one frame gives a value identical to today, any alpha (the
      weight cancels).
- [ ] All-identical depths across a patch gives weights all 1.0 and output identical to today.
- [ ] Duplicate `(frame_index, vertex_id)` in a frame raises (assert, do not reduce).
- [ ] Contacted vertex with no depth row in the sidecar raises with file/frame/vertex context.
- [ ] A session whose sidecar is absent entirely gives a distinct error from "vertex missing from
      sidecar".

---

## Documentation Plan

- [ ] Changelog: `docs/changelogs/depth-weighted-iff-attribution.md`.
- [ ] ADR recording *why* depth-weighted, why `(d/d_max)**alpha`, what `alpha=0` guarantees, why
      estimator A over B and C, and the `sum(w) = 0` policy. The rejection of B and C is the part
      that will otherwise be re-litigated.
- [ ] Docstring on `_compute_touch_rf` stating the mean is weighted and the max is not.
- [ ] Docstring on `vertex_weights` noting that normalise-before-exponent is only true
      normalisation at `alpha=1`.
- [ ] Fix the false docstring at `rf_data_loader.py:99` ("always contains contact points in
      RF-centered space" — false for 7 of 11 sessions) and the contradiction between
      `gui/preparation_viewer_data.py:62` and `shared_constants.py:61`, both touched by this work.
- [ ] Update `docs/data-contracts/contact-depth-field.md` with the consumer contract now that one
      exists.

---

## Rollback Plan

1. **Before merge:** the branch is `feature/depth-weighted-iff-attribution` off
   `feature/port-stroke-centroid-baseline`; each phase is its own commit, so any phase can be
   reverted individually. Phase 2 (the merge) is behaviour-neutral and can stay even if the
   weighting is abandoned.
2. **Data considerations:** the playback cache goes v3 to v4. Rolling back requires **deleting** v4
   caches — v3 code must not read a v4 file. No source data is modified; sidecars are read-only.
   Output `.npz` files gain keys, so downstream readers (`touch_population_data.py:905`,
   `single_touch_rf_explorer.py:99`, `rf_population_grid_pipeline.py`) must tolerate the additions
   or be updated in the same commit.
3. **Rollback procedure:** revert Phases 6 through 3 in reverse order, delete generated caches and
   `single_touch_rf_maps_*.npz`, re-run the stage. Setting `depth_weight_alpha: 0.0` is **not** a
   rollback — it is a baseline, and the parity test is what makes that trustworthy.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cache dedup silently broadcasts one frame's depth across a group (hazard 1) | **High** | **High** | Store depth per contact-frame, not per group. Dedicated test with two frames sharing a `contact_points` string but differing depths |
| Depth misaligns from contact points through the ffill / run expansion | Med | **High** | Join on `frame_index` **value**, never row position; apply the identical ffill; assert per-frame length equality against `frame_vertex_indices` |
| `alpha=0` is not byte-identical due to float summation order | **High** | Med | Same code path with `weights=ones`, same reduction call. If still unequal, downgrade to a **stated** tolerance and amend the plan — never silently switch to `approx` |
| Depth is near-uniform across patches, so the feature does nothing | Med | Med | Phase 6.3 diagnostic measures this directly and can falsify the premise cheaply |
| Change is dominated by the ~33x frame re-emission rather than spatial structure (hazard 3) | Med | **High** | Phase 6.4 predicts centre-vs-periphery; a uniform shift is the failure signature and halts the work |
| Merging nine untested call sites regresses a GUI view | Med | Med | Characterization tests written **before** the merge (2.1); merge is behaviour-neutral by construction |
| `sum(w)` near zero amplifies noise on a barely-grazed vertex | Med | Med | Explicit raise on `sum(w) == 0`; `n_eff` emitted so thin evidence is visible rather than rendered as an ordinary value |
| Upstream nearest-vertex assignment is approximate, and weighting amplifies mis-assignment | Med | Med | Document that per-vertex attribution remains approximate at the nearest-neighbour tolerance. Not fixed here — the two KDTree sites are out of scope |
| New GUI option breaks the task detail panel render | Low | Low | 5.4 adds the key to `_OPTION_GROUP_OF`; manual verification covers it |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| 1 — Land the loader | Small (already built) | None |
| 2 — Merge nine accumulators | **Large** (9 sites, 0 existing tests) | None |
| 3 — Carry depth to the estimator | **Large** (cache schema + alignment) | Phase 1 |
| 4 — Apply the weighting | Small (the maths is three lines) | Phases 2, 3 |
| 5 — Config, provenance, GUI | Small | Phase 4 |
| 6 — Prove it, then break it | Medium | Phase 5 |

The weighting itself is the smallest phase in the plan. The cost is the plumbing and the fact that
nothing here was ever tested.

---

## References

- Brainstorm: `docs/development/brainstorms/contact-depth-field-adoption.md` — section 8 records
  these decisions in plain language
- Data contract: `docs/data-contracts/contact-depth-field.md`
- Prior art (parity-test pattern): `tests/test_rf_response_fields_parity.py`,
  `tests/test_rf_stroke_axis_parity.py`
- Prior art (accumulator tests): `tests/test_rf_population_grid_pipeline.py`
- Prior art (synthetic parquet fixtures): `tests/test_contact_depth_field_io.py`
- Rejected sweep precedent: `docs/development/brainstorms/rf-contour-param-sweep.md` — a sweep
  without a computable ranker does not get built here
