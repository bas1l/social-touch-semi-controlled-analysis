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
9. Source `vertex_id` from the depth sidecar instead of re-deriving it by KDTree, so the depth join
   is exact by construction rather than approximate.

### Out of Scope

- **Automatic selection of alpha.** No held-out prediction harness, no cross-validation, no sweep
  framework. `alpha` is fixed at `1.0` in config; the dial exists only so the parity test can run.
- **Damping thin-evidence vertices inside the estimate.** Rejected — see Alternatives Considered.
  Confidence is a separate channel consumed at display time.
- **Weighting the `max` map.** `np.maximum.at` has no meaningful weighted analogue.
- **Cross-touch weighting** in `rf_population_heatmap.py` (deeper touches outranking shallower
  ones). The shared helper is introduced there; only per-frame weighting is applied.
- **Fixing the two *remaining* KDTree vertex-snapping sites** (`rf_explorer_data.py:345`,
  `touch_population_data.py:686`). Known-bad, tracked separately. The **third** site,
  `touch_playback_data.py:521,568`, is *not* out of scope — Phase 2.5 deletes it, because the whole
  depth join depended on it.
- **`rf_cluster_pipeline.py:181-190`** float-coordinate-tuple aggregation. Needs a `vertex_id`
  before weighting can reach it.
- **The `touch_analytics` clustering finding** (brainstorm section 2). Independent, unresolved,
  tracked separately.
- **Visual verification of the depth field past Kinect Space 1.** Explicitly ruled out as a gate in
  brainstorm section 4 — weighting only redistributes credit *within* a patch via `vertex_id`, and
  the RF chain is translation-invariant.

## Success Criteria

- [ ] With `depth_weight_alpha = 0.0`, `single_touch_rf_maps_mean.npz` is **byte-identical** to the
      baseline **re-pinned in Phase 2.5** on the same inputs. This proves one thing and only one
      thing: the weighting machinery is a genuine no-op at `alpha = 0`, because the same code path
      runs with `weights = ones`. It **no longer claims** the maps match what the pipeline produced
      before this branch — Phase 2.5 replaces KDTree vertex snapping with the sidecar's own
      `vertex_id`, which moves credit between vertices deliberately.
- [ ] Hand-computed weighted mean over the 7-vertex / 3-frame fixture matches the implementation to
      `rtol=1e-12`, including v6 (always shallow) and v7 (always deep) both returning approximately
      their unweighted values.
- [ ] Exactly **one** function in the codebase computes a per-vertex weighted mean; the other eight
      call sites delegate to it.
- [x] A characterization test pins each of the nine former call sites' output across the merge.
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
- **ordered correspondence**: within a single `frame_index`, the k-th point listed in that frame's
  CSV `contact_points` cell is the k-th parquet row carrying that `frame_index`. The parent repo
  states `frame_index` is the only exact join key the two artifacts share, and enforces this
  correspondence after every stage with `assert_row_counts_agree_with_csv`. Both `vertex_id` and
  `signed_depth_mm` are read off that one row; there is no vertex matching step.
- **ordered correspondence is not row-position joining**: pairing parquet row *i* with CSV row *i*
  across a whole file is unsafe and the contract warns against it — the CSV is at nerve rate with a
  variable number of points per frame, the parquet is at frame rate. The documented, enforced
  contract is narrower: locate the frame by `frame_index` **value**, *then* index by position
  **inside** that frame. These are different operations. Anyone who "generalises" the second into
  the first breaks the join silently, which is why the distinction is written down here.

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
| Recover `vertex_id` by KDTree | Avoids threading depth through the cache | Contract forbids it; measured 29 mm off-surface at touch boundaries. **A third KDTree site existed at `touch_playback_data.py:521,568`** — not either of the two listed as out of scope, and the one the entire depth design depended on. It is **deleted**, not reconciled, in Phase 2.5 | Rejected (data contract) |
| **Join on `frame_index` value + ordered position within that frame** | Exact by construction: row k of a frame *is* point k of that frame's `contact_points` cell, so `vertex_id` and `signed_depth_mm` come off a single row with no matching step and no tolerance. The parent repo declares `frame_index` the only exact join key the two artifacts share and enforces the per-frame row-count agreement after every stage (`assert_row_counts_agree_with_csv`) | Requires `frame_index` to survive to the CSV — it is dropped at `preparation_pipeline.py:32` — and requires this repo to re-assert the per-frame count itself | **Chosen** |
| Match `vertex_id` between the KDTree assignment and the sidecar, as a cross-check | Would look like a safety net | The KDTree assigns a *different* vertex from the one the sidecar recorded — that discrepancy **is** the 29 mm error, not a symptom of some other bug. A check that cannot pass is not a check: it would either be disabled or would block every session | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs -> Outputs | Must NOT know about |
|---|---|---|---|
| `contact_depth_field_io.py` (exists) | Read + validate parquet sidecars, join reference PLY | `csv_path, expect_space, session_id` -> `DepthField` | IFF, receptive fields, alpha, weighting |
| `vertex_weights.py` (**new**) | Turn per-frame depths into per-frame weights | `depth_mm: (K,), alpha: float` -> `w: (K,) in [0,1]` | IFF, vertices, frames, accumulation, files |
| `vertex_accumulator.py` (**new**) | The one weighted per-vertex reduction | `(vertex_idx, values, weights, n_vertices)` -> `(value_sum, value_max, weight_sum, weight_sq_sum)` | alpha, depth, mm, parquet, GUI, files |
| `touch_playback_data.py` (changed) | Resolve the block sidecar from `source_block_file`, read `vertex_id` **and** depth off the frame's rows, carry both onto `TouchEvent` and its cache | CSV (with `frame_index`, `source_block_file`) + sidecar -> `TouchEvent.frame_vertex_indices` + `.frame_depths` | alpha, weighting, IFF semantics, and *how* a stage directory is named or produced — see the boundary note below |
| `rf_single_touch_pipeline.py` (changed) | Orchestrate per-touch RF computation | `TouchEvent, alpha` -> mean/max pairs + confidence | parquet layout, PLY, YAML |

```
weight function      depth (mm)  ->  w in [0,1]      knows alpha, nothing else
accumulator          w + IFF     ->  sums            knows nothing about alpha
estimator            sums        ->  Hz + n_eff      knows nothing about depth
```

The thing that varies is the **weight function**, not the aggregation. Hiding `depth -> w` behind
one small function keeps the accumulator ignorant of alpha entirely, and makes an alternative
weighting (saturating, thresholded) an addition rather than an edit.

#### Boundary cost of Phase 2.5, stated plainly

Phase 2.5 makes `touch_playback_data.py` — a data-*loading* module — aware that there is a merged
data root and a blocks stage subdirectory. That is a real new coupling between a loader and upstream
stage layout, and the table above must not pretend the boundary is intact. The smallest honest
contract, rather than a fictional one:

- `touch_playback_data.py` receives the merged root and the stage subdirectory name **from config**,
  and the block filename **from the `source_block_file` column**. It composes no path fragment from
  repo knowledge and hardcodes neither string.
- `depth_field_path_for_csv()` (already built, Phase 1) owns the CSV-name -> parquet-name rule. No
  second implementation of that rule is written anywhere.
- `coordinate_space` is **read from parquet metadata**, never inferred from the directory name. The
  config says where to look; the file says what it is. That split is what holds the coupling to
  *location* and stops it becoming a coupling to *meaning*.

The alternative — a dedicated resolver module mapping `(session, block) -> parquet path` — is **not**
built. It would have exactly one caller and would move the same two config keys one file further
away without removing the coupling.

### Known hazards this design must survive

1. **The cache dedup is depth-unsafe.** `_save_playback_cache` deduplicates contact-point groups by
   `id()` of the vertex array (`touch_playback_data.py:131`), exploiting that the CSV parser reuses
   one ndarray for consecutive identical `contact_points` strings. **Depth is not implied by the
   contact-point string** — two frames with identical contact coordinates can carry different
   penetration depths. Depth arrays must therefore be stored **per contact-frame**, not per unique
   group, or one frame's depth is silently broadcast across every frame sharing that group.
2. **`frame_index` exists, but it is deleted upstream.** It is listed in `_DROP_COLUMNS`
   (`src/analysis/touch_analytics/preparation_pipeline.py:32`), applied at `:164` immediately before
   `to_csv` at `:165` — so the playback loader never sees it, and the 30 Hz Kinect structure
   survives in the CSV only as *runs of identical `contact_points` text*. Phase 2.5 restores the
   column, which invalidates `_prepared.csv` and `_series_augmented.csv`; a full pipeline re-run is
   planned and that cost is accepted. **The join is on `frame_index` *value*, then ordered position
   inside that frame — never whole-file row position** (see *ordered correspondence*, Definitions).
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
   `required_keys` in the same change, or a stale cache passes the version check while lacking
   depth. Done twice: Phase 2.5 took the schema 3 -> 4 (retired KDTree indices) and Phase 3 took
   it 4 -> 5 (depth), each bumping the version and extending `required_keys` in one change.
6. **`_save_playback_cache` swallows write failures with `logger.warning`.** Pre-existing silent
   fallback; the depth path must not inherit it.
7. **Zero is not absent.** `depth = 0.0` (grazing) and `depth = missing` (no sidecar row) are
   different facts that both become `w = 0` at `alpha > 0`. The fail-fast check must happen at the
   loader boundary, where they are still separable.
8. **`parse_contact_points` must not silently drop a malformed triplet.** The parent repo's parser
   keeps a point only `if len(parts) == 3` and supplies **no `else`** — a malformed triplet vanishes
   without a word. That behaviour is exactly why `assert_row_counts_agree_with_csv` exists upstream.
   On this side, one dropped point shifts every subsequent point of that frame onto the **wrong**
   sidecar row, producing a plausible-looking map that is wrong. This repo's parser must raise on
   any triplet it cannot parse; task 2.5.6's per-frame count assertion is the second line of
   defence, not the first.
9. **Two block-id spellings exist and are never converted.** `block-order02` comes from the video
   stem and is used through preprocessing; `block-order-02` is used from merging onward. The parent
   repo uses each verbatim in its own domain and converts between them nowhere. `block_order_id` in
   the aggregated CSV is parsed with `block-order-(\d+)` and is a **zero-padded string** (`"02"`),
   or `None` when the regex misses. Never `int()` it, never reformat it, and never synthesise a
   parquet filename from it — pass `source_block_file` to `depth_field_path_for_csv()`.
10. **`vertex_id` exists only from the projection stage onward.** `blocks_projected/`,
    `blocks_pca_calibrated/` and `blocks_rf_centered/` carry it (schema version 2);
    `blocks_filtered/`, `blocks_registered/` and `blocks_deduped/` do **not** (schema version 1).
    Pointing the stage-subdirectory config at a pre-projection stage yields a parquet with no
    `vertex_id` column at all. The loader must raise on the missing column and name the stage in the
    message, not surface a downstream `KeyError`.

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
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** Exactly one function computes a per-vertex mean, pinned by tests, before any weighting
exists.

- [x] 2.1 — Write characterization tests pinning current output for all nine sites on synthetic
      fixtures. **This pinned output is the `alpha=0` baseline; the two tasks are the same task.**
      Fixtures live in `tests/rf_accumulator_fixtures.py` (a plain module, not a conftest fixture)
      so Phase 6's parity test imports the *same* arrays. `worked_example_touch()` is the plan's
      7-vertex / 3-frame example and reproduces its table exactly: unweighted peak on v2 at 75.0 Hz,
      v3 at 190/3. `WORKED_EXAMPLE_DEPTHS_MM` carries depths whose alpha=1 weights are the plan's
      (v3 `0.42, 1.00, 0.50`; v6 `0.13, 0.13, 0.10`; v7 `0.92, 0.93, 0.95`), with a different
      maximum depth per frame so a missing max-normalisation cannot hide.
- [x] 2.2 — Create `data/vertex_accumulator.py` with
      `accumulate_vertex_values(vertex_idx, values, weights, n_vertices) -> AccumResult`
      (`value_sum`, `value_max`, `weight_sum`, `weight_sq_sum`). `weights` is **required**; callers
      pass explicit ones at this stage.
      **Addition:** two further entry points onto the *same* reduction —
      `empty_accumulator(n_vertices)` and `accumulate_vertex_values_into(result, ...)`. Sites 8 and
      9 grow a running accumulator one frame at a time; reducing into a fresh array and adding the
      result would stop being bit-identical the moment a frame repeats a vertex index
      (`state + x + x` vs `state + (x + x)`), so in-place `np.add.at` is what preserves them.
      `accumulate_vertex_values` is `empty_accumulator` + `accumulate_vertex_values_into`.
- [x] 2.3 — Redirect all nine call sites to it. Note four are in `touch_playback_explorer.py`
      (`:308` replay-to-slider, `:549` replay-inclusive, `:602` incremental single-frame, `:959`
      export loop) and two are bincount-based rather than `np.add.at` — the helper must cover both
      shapes or the bincount sites keep a thin adapter.
      **Resolution:** no adapter was needed. `np.bincount(weights=...)` and `np.add.at` are both
      sequential unbuffered float64 additions in array order; `TestReductionEquivalence` asserts
      they agree with `np.array_equal` over a 200 000-point chain spanning eight orders of
      magnitude, so all three bincount sites moved onto `np.add.at` with no tolerance introduced.
      **Deviation:** three of the nine sites were Qt methods that drew while they computed and could
      not be tested without a QApplication. Their numeric cores were lifted to module-level
      functions *in the same module* — `compute_contact_point_heatmap`
      (`touch_population_explorer`), `vertex_value_mean` (`rf_feature_space_explorer`, shared by
      both of its sites), and `accumulate_touch_frame` / `replay_touch_frames` /
      `mean_heatmap_scalars` (`touch_playback_explorer`, shared by all four of its sites). The Qt
      methods are now delegations with no arithmetic left in them. The numerics were kept in their
      own modules rather than moved to `data/` to hold the diff to the file set this phase declared.
- [x] 2.4 — Confirm every characterization test still passes, `np.array_equal`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/vertex_accumulator.py` — new
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-exports
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — `:83-85`
- `src/analysis/receptive_field_mapping/data/rf_population_heatmap.py` — `:23-27`
- `src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — `:635`
- `src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` — `:545`, `:628`
- `src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — `:308`, `:549`, `:602`, `:959`
- `tests/test_vertex_accumulator.py` — new
- `tests/rf_accumulator_fixtures.py` — new

**Known consequences of the merge (behaviour-neutral on outputs, but visible):**
- `value_max` is now accumulated at all nine sites, not just the pipeline. Where values contain NaN
  this emits numpy's `invalid value encountered in maximum` RuntimeWarning from call sites that were
  previously silent. No output array changes; the warning was already emitted by
  `_compute_touch_rf`.
- `empty_accumulator` raises on `n_vertices <= 0`, where the previous code silently allocated
  zero-length arrays. Consistent with the fail-fast rule; only reachable on a mesh with no vertices.
- `TouchPopulationExplorer`'s unknown-heatmap-mode message is now raised by
  `compute_contact_point_heatmap` and names that function instead of the class.

**Dependencies:** None (can run parallel to Phase 1).

### Phase 2.5: Source vertex identity from the sidecar
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** Retire the third KDTree vertex re-derivation and make the sidecar reachable, so that the
depth join is exact by construction. Phase 3 as originally written rested on a wrong premise — that
a `vertex_id` computed on this side could be matched against the sidecar's. It cannot: the sidecar's
`vertex_id` **is** the answer, and it arrives on the same row as the depth. The join therefore moves
here, and Phase 3 keeps only the transport.

The join is `frame_index` **value**, then **ordered position within that frame** — the k-th parsed
contact point of a frame is the k-th parquet row carrying that `frame_index`. This is the parent
repo's documented contract (`frame_index` is the only exact join key the two artifacts share) and it
is enforced there after every stage by `assert_row_counts_agree_with_csv`. It is **not** a licence to
join rows to rows by position across a whole file; see *ordered correspondence* in Definitions.

- [x] 2.5.1 — Restore `frame_index`: remove it from `_DROP_COLUMNS`
      (`src/analysis/touch_analytics/preparation_pipeline.py:32`; the drop is applied at `:164`,
      immediately before `to_csv` at `:165`). Verify the column survives into `_prepared.csv` and
      `_series_augmented.csv`. Both artifacts are invalidated by this change; a full pipeline re-run
      is planned and that cost is accepted.
- [x] 2.5.2 — Add `frame_index` to `_REQUIRED_COLUMNS` (`touch_playback_data.py:22-30`) and
      forward-fill it **inside the same `groupby([...]).ffill()` statement** that already fills
      `contact_points` (`:481-483`) — not in a second statement alongside it. One statement makes
      desynchronisation of the frame index from the contact points structurally impossible rather
      than merely tested for.
- [x] 2.5.3 — Add sidecar-location config: the **merged-data root** and the **blocks stage
      subdirectory** name (`blocks_rf_centered`). These are the only genuinely new config keys.
      Resolve the block parquet by feeding the aggregated CSV's `source_block_file` column — a
      basename, e.g. `ST13-03_semicontrolled_block-order-02_merged_data.csv` — to the already-built
      `depth_field_path_for_csv()`. **No new artifact is produced and there is no session-level
      sidecar**: the parent repo emits these per block only, deliberately. Do not synthesise the
      filename from `block_order_id` (hazard 9).
- [x] 2.5.4 — Read `coordinate_space` from the parquet metadata and **assert** it is one of the
      three documented terminal values a `blocks_rf_centered/*.parquet` can legitimately carry:
      `rf_centered` (the normal case), `pca_calibrated` (the RF shift was identity, so the stage was
      a byte-identical passthrough), or `kinect_space_1` (the block had no ICP snapshot). Record
      which value was found in run provenance. **This is not a fallback.** Depth weighting consumes
      exactly two things — depth magnitude and vertex identity — and neither is spatial, so the
      operation is genuinely space-agnostic. The trap `contact_depth_field_io.py` closes is
      *inferring* the space from a folder name; reading it from metadata and asserting it against a
      closed set is the opposite of that, and recording it means a surprising value shows up in the
      run record instead of being absorbed.
- [x] 2.5.5 — Delete the KDTree vertex assignment at `touch_playback_data.py:521,568` and take
      `vertex_id` off the sidecar row instead: locate the frame by `frame_index` **value**, then
      take the k-th row of that frame for the k-th parsed contact point. The KDTree is **removed,
      not reconciled** — matching its output against the sidecar's `vertex_id` is a check that
      cannot pass (Alternatives Considered).
- [x] 2.5.6 — Hard per-frame count assertion: the number of parsed contact points for a
      `frame_index` **must equal** the number of sidecar rows carrying that `frame_index`. Raise
      with the parquet path, the `frame_index`, and both counts on any mismatch. This assertion is
      the entire reason ordered correspondence is safe; without it a single dropped triplet
      (hazard 8) shifts every later point in the frame onto the wrong row. It mirrors the parent
      repo's `assert_row_counts_agree_with_csv`.
- [x] 2.5.7 — Re-pin the Phase 2 characterization baseline against the new vertex assignment, and
      **record the magnitude of the resulting map change**: the per-vertex `|delta|` distribution
      and how many contact points changed vertex identity at all. The old baseline is not preserved
      and must not be — the point of this phase is that the previous assignment was wrong. Success
      Criterion 1 and task 6.1 are both restated against this re-pinned baseline.

**Implementation notes (2026-08-21)**

- **Sidecar location config — one key fewer and one key more than planned.** The
  *merged root* did **not** become a config key: the DAG already resolves the session
  merged root as the parent of the aggregated session CSV in `input_items`
  (`session_merged_output_dir: 3_merged/<session_id>`), and `resolve_forearm_ply` already
  reads it that way. Adding an absolute path to YAML would have duplicated a value the DAG
  owns and hardcoded a machine path. What *was* added, under
  `spatial_map_single_touch.options.contact_depth_field`, is `blocks_stage_dir`
  (`blocks_rf_centered`) and `block_csv_stem_suffix` (`_pca-xyz`). The suffix is
  necessary and was not anticipated by the plan: `source_block_file` is the
  *pre-PCA* basename (`..._merged_data.csv`) while a terminal stage CSV is
  `..._merged_data_pca-xyz.csv`, so the marker substitution alone yields a name that
  does not exist in `blocks_rf_centered/`. The same two keys were mirrored onto
  `explore_touch_playback` in `configs/analyse_workflow_viewers_dag.yaml`, because the
  playback GUI calls the same loader and must draw the same assignment.
  `configs/analyse_workflow_dag.yaml` does **not** define this stage, so nothing was
  mirrored there.
- **File-order accessor added to the loader module.** `DepthField.frame()` sorts by
  `vertex_id` defensively, which is correct for aggregating callers and **fatal** for
  ordered correspondence. `DepthField.frame_row_positions_in_file_order()` was added
  beside it, documented as the accessor the join must use and why the two are kept
  separate rather than parameterised. `declared_coordinate_space()` was added to read
  `coordinate_space` from the parquet schema without loading a row group, together with
  `TERMINAL_RF_CENTERED_SPACES` — the closed set of three the assertion checks.
- **Cache schema bumped 3 to 4 here, not in Phase 3.** A v3 cache holds KDTree-derived
  vertex indices; reusing one would silently serve the wrong answer while the code looked
  correct. The bump, the three `depth_provenance_*` arrays and their entries in
  `required_keys` landed in the same change (hazard 5). **Phase 3 must therefore bump 4 to
  5**, not 3 to 4.
- **Reuse key is `(contact_points text, frame_index)`, not the text alone.** Two Kinect
  frames can carry byte-identical contact-point text while addressing different sidecar
  rows. Keying the parse-reuse optimisation on the text alone would broadcast one frame's
  vertex identities across the other. The array object is still shared *within* a frame,
  so `_save_playback_cache`'s `id()`-based dedup keeps working — and now dedups only rows
  that genuinely share a frame.
- **`parse_contact_points` hardened in two places.** The canonical copy in
  `rf_data_loader.py` and the tight inline parser in `touch_playback_data.py` both raise on
  any bracketed group that is not exactly three floats (hazard 8). They were not merged:
  their regexes differ (`[^\[\]]+` vs `[^\]]+`) and unifying them would have changed
  behaviour on a path this phase does not otherwise touch.
- **2.5.7, honestly.** The Phase 2 characterization baseline is **numerically unchanged**,
  and that is correct rather than a miss: `tests/rf_accumulator_fixtures.py` hands
  `frame_vertex_indices` to the reduction directly, so it pins the *reduction*, never the
  vertex source — all 41 tests in `tests/test_vertex_accumulator.py` still pass by
  `np.array_equal`. What was re-pinned is the *meaning*: the module docstring now records
  that those indices are sidecar `vertex_id` values and that the `alpha = 0` parity test
  therefore claims only that the weighting is an exact no-op, never agreement with
  pre-2.5 maps. **The magnitude of the real map change was not measured**, because this
  work is forbidden to read the experimental database. `scripts/diagnose_vertex_reassignment.py`
  performs the measurement — reassigned contact-point count and the per-vertex `|delta|`
  distribution — and must be run on one session before any Phase 6 result is trusted.

**Files Modified:**
- `src/analysis/touch_analytics/preparation_pipeline.py` — `frame_index` removed from `_DROP_COLUMNS`
- `src/analysis/receptive_field_mapping/data/touch_playback_data.py` — KDTree deleted;
  `_BlockVertexSource`, `_load_block_vertex_source`, `_parse_contact_points_strict`,
  `DepthFieldProvenance`; cache v3 to v4
- `src/analysis/receptive_field_mapping/data/contact_depth_field_io.py` —
  `frame_row_positions_in_file_order()`, `declared_coordinate_space()`,
  `TERMINAL_RF_CENTERED_SPACES`
- `src/analysis/receptive_field_mapping/data/rf_data_loader.py` — `parse_contact_points` raises
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-exports
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` —
  `contact_depth_field` option, blocks-dir resolution, provenance in the summary JSON
- `src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — playback GUI launcher
- `scripts/analysis_workflow_processing.py`, `scripts/analysis_workflow_viewers.py` — flow + params
- `scripts/generate_rf_single_touch_detail.py` — figure script call site
- `configs/analyse_workflow_processing_dag.yaml`, `configs/analyse_workflow_viewers_dag.yaml` —
  `contact_depth_field` block. **ruamel.yaml round-trip only.**
- `tests/rf_accumulator_fixtures.py` — baseline re-pinned (documentation; arrays unchanged)
- `tests/test_touch_playback_vertex_source.py` — new, 26 tests
- `scripts/diagnose_vertex_reassignment.py` — new, measures the real-data magnitude

**Dependencies:** Phases 1, 2.

### Phase 3: Carry depth to the estimator
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** Transport per-contact-point penetration depth from the sidecar row that Phase 2.5 has
already located, through the cache, to `_compute_touch_rf`. **The join moved to Phase 2.5.** Depth
comes off the *same row* as `vertex_id`, so this phase contains no sidecar lookup, no `frame_index`
matching and no alignment step — those tasks were deleted from here, not shrunk. What remains is
storage and validation.

- [x] 3.1 — Add `frame_depths: list  # (K_i,) float64` to `TouchEvent` (`:49-59`), aligned with
      `frame_vertex_indices`. Both are built from the same sidecar rows, in the same order, in the
      same loop.
- [x] 3.2 — Read `signed_depth_mm` off the rows already located in 2.5.5. There is no second join
      and no `.get(vertex_id, ...)` anywhere.
- [x] 3.3 — Assert `(frame_index, vertex_id)` uniqueness per frame and raise on violation
      (hazard 4). **Assert, do not reduce** — the cancellation that makes duplicates harmless today
      only holds while every weight is 1.
- [x] 3.4 — Store depth **per contact-frame**, not per dedup group. Hazard 1 is unchanged by the
      corrected join and is still the most dangerous item in this plan: two frames with identical
      `contact_points` text can carry different depths, and `_save_playback_cache` dedups groups by
      `id()` of the vertex array (`:131`). Bump `_CACHE_SCHEMA_VERSION` **4 to 5** (Phase 2.5 already took it 3 to 4), add the keys
      to `required_keys` (`:243-250`) **in the same change** (hazard 5), unpack them, add shape
      checks in the `:288-352` block, pass to the constructor at `:367`.
- [x] 3.5 — Reject NaN depth at the loader boundary with file/frame/vertex context. Zero is not
      absent (hazard 7); a *missing* row is now impossible by construction because 2.5.6 asserts the
      per-frame counts agree, so NaN is the only remaining form of absence and it raises.

**Implementation notes (2026-08-21)**

- **Hazard 1 was already half-closed by Phase 2.5, and the test that proves the depth
  channel is safe had to be built accordingly.** Phase 2.5's parse-reuse key is
  `(contact_points text, frame_index)`, so two *different* Kinect frames never share a
  vertex-array object and therefore never land in the same `id()`-keyed dedup group. A
  cold-load test with two frames sharing contact-point text but differing depths
  consequently **cannot distinguish** per-group storage from per-frame storage: it passes
  either way. That test is kept as the behavioural claim, but the one with teeth
  (`test_a_shared_dedup_group_still_carries_per_frame_depths`) drives
  `_save_playback_cache` / `_load_playback_cache` directly with a `TouchEvent` whose two
  frames hold the **same ndarray object** and different depth arrays — the exact shape
  `id()` dedup collapses. Per-group depth storage returns one frame's depths for both and
  fails it. The mitigation and the dedup live in different functions, so relying on the
  reuse key alone would have made the depth channel's correctness a property of code that
  has no idea depth exists.
- **Depth cache layout.** Two arrays, both indexed by contact-frame position `i` — the
  same `i` that indexes `cp_frame_group` / `cp_frame_touch` / `cp_frame_fi`, and
  explicitly *not* the group index: `cp_frame_depth_data` (float64, concatenated) and
  `cp_frame_depth_offsets` (int64, `n_contact_frames + 1`). On read, each contact frame's
  depth count is checked against its dedup group's point count, which is the check that
  would catch a broadcast, plus offset monotonicity, endpoint agreement and a NaN sweep.
- **`_save_playback_cache`'s warn-on-write-failure was not inherited.** Every depth
  consistency check runs *before* the `try`, so a misaligned depth channel raises while a
  failed write still only costs a cache. The docstring now says so, because the two look
  identical from the call site.
- **`TouchEvent.frame_depths` is a required field with no default**, which forced four
  existing construction sites to supply it: `tests/rf_accumulator_fixtures.py` (both
  fixtures), `scripts/diagnose_vertex_reassignment.py` and one test in
  `tests/test_touch_playback_vertex_source.py`. A default would have let a depth-free
  touch reach the weighting stage looking structurally valid. The Phase 2 characterization
  numbers are unchanged — all 41 accumulator tests still pass by `np.array_equal`.
- **Sign, stated once.** `WORKED_EXAMPLE_DEPTHS_MM` is *penetration* (positive, what the
  weight function will consume); `TouchEvent.frame_depths` is `signed_depth_mm` (negative
  = penetrating, what the file stores). `WORKED_EXAMPLE_SIGNED_DEPTHS_MM` was added beside
  it as the negated form so both are visible in one place, and Phase 4 recovers the first
  from the second through `penetration_mm` — the single documented negation point.
- **`SIGNED_DEPTH_COLUMN` added to `contact_depth_field_io.py`** so the column name is not
  spelled a second time in a second module. Not an accessor and not new behaviour.
- **`vertex_ids_for_frame` became `rows_for_frame`**, returning a `_FrameRows` NamedTuple
  of `(vertex_ids, signed_depth_mm)`. One join answers both questions, which is the point:
  a separate depth accessor would have invited a `vertex_id`-keyed lookup and quietly
  reintroduced the matching step ordered correspondence exists to remove.
- **The Phase 2.5 test's `_CACHE_SCHEMA_VERSION == 4` pin was relaxed to `> 3`.** Its claim
  is that the schema has moved past 3 and stays past it, which no later bump invalidates;
  the exact current version is pinned once, in `tests/test_touch_playback_depth.py`.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/touch_playback_data.py` — `_FrameRows`,
  `_BlockVertexSource.rows_for_frame` (uniqueness + NaN assertions), `TouchEvent.frame_depths`,
  cache v4 to v5 with per-contact-frame depth storage and its shape checks
- `src/analysis/receptive_field_mapping/data/contact_depth_field_io.py` — `SIGNED_DEPTH_COLUMN`
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-export
- `tests/test_touch_playback_depth.py` — new, 23 tests
- `tests/rf_accumulator_fixtures.py` — `frame_depths` on both fixtures;
  `WORKED_EXAMPLE_SIGNED_DEPTHS_MM`, `NAN_AND_DUPLICATE_SIGNED_DEPTHS_MM`
- `tests/test_touch_playback_vertex_source.py` — `frame_depths` at one construction site;
  version pin relaxed to `> 3`
- `scripts/diagnose_vertex_reassignment.py` — `frame_depths` passed through unchanged

**Dependencies:** Phases 1, 2.5.

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
      baseline **re-pinned in Phase 2.5** — not to the pre-branch output. Model on
      `tests/test_rf_response_fields_parity.py` (`_assert_exactly_equal`, `:220`), with a docstring
      stating both halves of the claim: it **proves** the weighting machinery is an exact no-op at
      `alpha = 0`, because the same code path runs with `weights = ones`; it says **nothing** about
      agreement with maps produced before the Phase 2.5 vertex fix, which changed them on purpose.
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
- [ ] `alpha=0` byte-identical to the **Phase 2.5 re-pinned** baseline through the whole pipeline.
- [ ] Vertex identity comes from the sidecar: a synthetic frame whose sidecar `vertex_id` differs
      from the nearest-vertex answer resolves to the **sidecar's** value.
- [ ] `frame_index` and `contact_points` are filled by one `ffill` statement and cannot drift: a
      fixture with gaps in both columns yields matched, non-null pairs on every row.
- [x] Cache round-trip: write v5, read back, depth arrays match per **frame** — specifically a case
      where two frames share a `contact_points` string but carry different depths (hazard 1), and
      a second case where two frames genuinely share one dedup group, which is the only form of
      the hazard the Phase 2.5 reuse key does not already prevent.
- [x] A v4 cache on disk is rejected loudly, not silently recomputed into a depth-free result.

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
- [x] Duplicate `(frame_index, vertex_id)` in a frame raises (assert, do not reduce).
- [ ] A frame whose parsed contact-point count differs from its sidecar row count raises with the
      parquet path, the `frame_index`, and both counts (2.5.6).
- [ ] A malformed contact-point triplet raises in `parse_contact_points` rather than being dropped
      (hazard 8).
- [ ] A block sidecar whose `coordinate_space` is `pca_calibrated` or `kinect_space_1` loads and is
      recorded in provenance; a fourth, undocumented value raises (2.5.4).
- [ ] A parquet from a pre-projection stage (no `vertex_id` column) raises naming the stage
      (hazard 10).
- [ ] A session whose block sidecar is absent entirely gives a distinct error from a per-frame count
      mismatch.

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
      exists: the join is `frame_index` **value** plus ordered position *within* that frame;
      whole-file row-position joining is forbidden, and why; the per-frame count assertion is
      mandatory; and the three `coordinate_space` values a terminal `blocks_rf_centered/` file may
      legitimately carry.
- [ ] Record in the ADR why the KDTree vertex re-derivation at `touch_playback_data.py:521,568` was
      **deleted** rather than reconciled against the sidecar, so the "cross-check" is not
      reintroduced later as an improvement.

---

## Rollback Plan

1. **Before merge:** the branch is `feature/depth-weighted-iff-attribution` off
   `feature/port-stroke-centroid-baseline`; each phase is its own commit, so any phase can be
   reverted individually. Phase 2 (the merge) is behaviour-neutral and can stay even if the
   weighting is abandoned.
2. **Data considerations:** the playback cache goes v3 to v4 (Phase 2.5) and v4 to v5 (Phase 3).
   Rolling back requires **deleting** the newer caches — older code must not read a newer file. No source data is modified; sidecars are read-only.
   Separately, Phase 2.5 restores `frame_index` upstream, which invalidates every `_prepared.csv`
   and `_series_augmented.csv`; those must be regenerated, and reverting Phase 2.5 re-drops the
   column and invalidates them a second time. Do not revert 2.5 to "get the old maps back" — the
   old maps were built on the wrong vertex assignment.
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
| Depth misaligns from contact points through the ffill / run expansion | Med | **High** | Join on `frame_index` **value**, then ordered position *within* the frame — never whole-file row position; fill `frame_index` in the **same** `ffill` statement as `contact_points` (2.5.2); assert per-frame count equality against the sidecar (2.5.6) |
| `parse_contact_points` drops a malformed triplet and shifts the rest of the frame onto the wrong sidecar rows (hazard 8) | Low | **High** | Parser raises rather than skipping; 2.5.6's per-frame count assertion catches any survivor before it reaches a map |
| A terminal `blocks_rf_centered/` parquet carries `pca_calibrated` or `kinect_space_1` and a strict `expect_space` check blocks the run | **High** | Med | 2.5.4 asserts membership in the three documented terminal values and records which was found; weighting reads only depth magnitude and vertex identity, neither of which is spatial |
| Restoring `frame_index` invalidates every `_prepared.csv` / `_series_augmented.csv` | **Certain** | Low | Accepted: a full pipeline re-run is already planned. Recorded here so the cost is not rediscovered mid-implementation |
| `alpha=0` is not byte-identical due to float summation order | **High** | Med | Same code path with `weights=ones`, same reduction call. If still unequal, downgrade to a **stated** tolerance and amend the plan — never silently switch to `approx` |
| Depth is near-uniform across patches, so the feature does nothing | Med | Med | Phase 6.3 diagnostic measures this directly and can falsify the premise cheaply |
| Change is dominated by the ~33x frame re-emission rather than spatial structure (hazard 3) | Med | **High** | Phase 6.4 predicts centre-vs-periphery; a uniform shift is the failure signature and halts the work |
| Merging nine untested call sites regresses a GUI view | Med | Med | Characterization tests written **before** the merge (2.1); merge is behaviour-neutral by construction |
| `sum(w)` near zero amplifies noise on a barely-grazed vertex | Med | Med | Explicit raise on `sum(w) == 0`; `n_eff` emitted so thin evidence is visible rather than rendered as an ordinary value |
| Upstream nearest-vertex assignment is approximate, and weighting amplifies mis-assignment | Med | Med | **Fixed for the playback path** in 2.5.5: `vertex_id` comes off the sidecar row, so attribution there is exact rather than nearest-neighbour. The two *other* KDTree sites (`rf_explorer_data.py:345`, `touch_population_data.py:686`) remain out of scope and still approximate; 2.5.7 records how far the corrected map moved |
| New GUI option breaks the task detail panel render | Low | Low | 5.4 adds the key to `_OPTION_GROUP_OF`; manual verification covers it |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|---|---|---|
| 1 — Land the loader | Small (already built) | None |
| 2 — Merge nine accumulators | **Large** (9 sites, 0 existing tests) | None |
| 2.5 — Source vertex identity from the sidecar | **Large** (upstream column restore, join rewrite, baseline re-pin) | Phases 1, 2 |
| 3 — Carry depth to the estimator | Medium (cache schema only — the join moved to 2.5) | Phases 1, 2.5 |
| 4 — Apply the weighting | Small (the maths is three lines) | Phases 2, 3 |
| 5 — Config, provenance, GUI | Small | Phase 4 |
| 6 — Prove it, then break it | Medium | Phase 5 |

The weighting itself is the smallest phase in the plan. The cost is the plumbing and the fact that
nothing here was ever tested. Phase 2.5 is new cost, not moved cost: it fixes a wrong vertex
assignment that the original Phase 3 assumed away, and it changes the output maps on its own, before
any weighting exists.

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
