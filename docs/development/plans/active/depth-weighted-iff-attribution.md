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
| v1 (one frame) | 50.0 | 50.00 |
| v2 (flank) | **75.0 <- peak** | 65.99 |
| v3 (**true RF centre**) | 63.3 | **73.44 <- peak** |
| v4 (flank) | 70.0 | 57.14 |
| v5 (one frame) | 40.0 | 40.00 |

Every weighted figure in this document is the value produced by the committed fixture
(`tests/rf_accumulator_fixtures.py`), whose depths were chosen so that the 2-dp weight tables
below are the **exact** weights rather than rounded ones — `0.39 / 3.00` is exactly `0.13`. The
plan and `docs/development/brainstorms/contact-depth-field-adoption.md` therefore quote the same
numbers, and the tests pin them at `rtol=1e-12`.

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

- [x] With `depth_weight_alpha = 0.0`, `single_touch_rf_maps_mean.npz` is **byte-identical** to the
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
- [x] `depth_weight_alpha` appears in `single_touch_rf_summary.json`, so a config change invalidates
      the stage and two runs are distinguishable on disk. The sentinel also carries
      `playback_cache_schema_version` and, from Phase 2.5, each block's sidecar path and the
      `coordinate_space` that file declared.
- [x] Calling the estimator without `alpha` raises `TypeError`. There is no default value anywhere.
- [ ] The diagnostic reports per-vertex weight variation for one session and states whether the
      feature can have any effect at all on that data. **Still open on purpose:** the diagnostic
      exists (`scripts/diagnose_depth_weight_variation.py`) and is validated on synthetic fixtures,
      but it has never been run on a session, because this work was forbidden to read the
      experimental database. A human must run it.

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
| v6 always shallow | `0.13, 0.13, 0.10` — flat | 63.3 | 65.28 |
| v7 always deep | `0.92, 0.93, 0.95` — flat | 63.3 | 63.21 |
| v3 RF centre | `0.42, 1.00, 0.50` — **varying** | 63.3 | **73.44** |

A roughly-constant weight cancels between numerator and denominator. **Absolute depth level is
irrelevant; only frame-to-frame variation in a vertex's own depth does anything.** `n_eff` confirms
it — v6 retains 2.96 of 3 effective observations despite weights of 0.1.

This yields a real falsification path, which Phase 6 tests: **if depth is roughly uniform across
each contact patch in the real data, this feature produces today's map at every alpha.**

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **A: `sum(w*IFF) / sum(w)`** | Stays in Hz; single-frame vertices provably invariant; peak lands on v3; no tuning knob | A chronically-shallow vertex reports a normal rate, which reads as wrong until the mechanism is understood | **Chosen** |
| B: `sum(w*IFF) / N` | Damps always-shallow vertices strongly (v6: 63.3 to 7.83); `alpha=0` parity needs no extra condition | **Peak moves to v7, not v3.** Damping v6 and inflating v7 are the same operation. v7 is deep because of stroke geometry and arm curvature, not the neuron — so this imports stimulus geometry into the RF map. Output is no longer in Hz | Rejected |
| C: `sum(w*IFF) / (sum(w) + k)` | Damps v6 (to 17.28 at `k = 1`) *and* keeps the peak on v3 | `k` is a tuning knob that can move the peak — v3 leads v7 only while `k < 1.53` on the worked example. A parameter that decides where the receptive field is must be chosen on principle. Weakens parity: needs both `alpha=0` and `k=0` | Rejected for now |
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
4. **Duplicate `(frame_index, vertex_id)` pairs — struck; this was never a hazard.** As originally
   written: "The contract says possible; the design says not. Duplicates are currently harmless
   because they double numerator and denominator equally — a cancellation that only holds while the
   weight is 1. **Assert, do not reduce**: check uniqueness per frame where weights are built and
   raise on violation." That was implemented (task 3.3), it fired on real data
   (2022-06-15_ST14-02, block-order-01, `frame_index` 163, `vertex_id` 6254 twice among 156 rows),
   and its premise is **false**. IFF is a per-frame *scalar* shared by every contact point of the
   frame, so a vertex hit at weights `w1`, `w2` gains `IFF_f*w1 + IFF_f*w2 == IFF_f*(w1+w2)` in the
   numerator and `w1+w2` in the denominator: `IFF_f` factors out at **any** weights. The check was
   removed, and nothing replaced it — the rows are summed by `np.add.at`, which is transport rather
   than reduction. The duplicates arise benignly: XY dedup runs in `blocks_deduped`, before
   `vertex_id` is assigned in `blocks_projected`, so it guarantees distinct positions, not distinct
   vertices. Consequence, and intended: such a vertex carries weight `w1 + w2 > 1` for that frame,
   consistent with the deliberate absence of per-frame normalisation. The weighted **mean** equals
   that of one row of weight `w1 + w2`; `n_eff` does not, and is entitled to differ.
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
- [~] 3.3 — ~~Assert `(frame_index, vertex_id)` uniqueness per frame and raise on violation
      (hazard 4). **Assert, do not reduce** — the cancellation that makes duplicates harmless today
      only holds while every weight is 1.~~ **Implemented, then reverted.** The assertion blocked
      real runs, and its premise was wrong: IFF is a per-frame scalar, so it factors out of both
      numerator and denominator at *any* weights, not only at 1 (hazard 4, struck). The check was
      deleted; the duplicate rows are summed by the accumulator, and nothing lenient replaced it.
      The other loader guards — per-frame count agreement, `vertex_id` bounds, NaN depth — are
      untouched. The arithmetic claim that replaced the raise is pinned with hand-computed numbers
      in `tests/test_vertex_weights.py::TestADuplicatedVertexIsCreditedAtTheSummedWeight`, and the
      transport claim in
      `tests/test_touch_playback_depth.py::TestDuplicateFrameVertexPairsAreCarried`.
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
  `_BlockVertexSource.rows_for_frame` (bounds + NaN assertions; the uniqueness assertion was
  later removed — hazard 4, struck), `TouchEvent.frame_depths`,
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
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** The estimator becomes a weighted mean, with every degenerate case raising.

- [x] 4.1 — Create `data/vertex_weights.py`:
      `vertex_weights(penetration_mm: np.ndarray, alpha: float) -> np.ndarray`. Clamp negatives
      (grazing) to 0, divide by the frame max, then raise to `alpha`. **Normalise before the
      exponent.** `alpha` is positional and required.
- [x] 4.2 — Guard `d_max == 0` with an explicit raise. Do **not** rely on `NaN ** 0 == 1.0`.
- [x] 4.3 — Rename `contact_count` to `weight_sum` in `_compute_touch_rf` (`:66`, `:87-92`).
- [x] 4.4 — Pass real weights into the shared accumulator. `neuron_values[fi]` is currently a
      **scalar** broadcast over the frame's vertices; with weights it becomes a `(K_i,)` array.
- [x] 4.5 — Guard `sum(w) == 0` per vertex with an explicit policy and raise. Never `0/0 -> NaN`
      into the map, and never fall back to the uniform mean.
- [x] 4.6 — Leave `val_max` unweighted; document why in the docstring.
- [x] 4.7 — Emit `weight_sum` and `n_eff = (sum w)^2 / sum(w^2)` per vertex into the returned
      structure and the saved `.npz`.

**Implementation notes (2026-08-21)**

- **The argument is named `penetration_mm`, not `depth_mm`.** The plan wrote `depth_mm`,
  which does not say which sign convention it wants — and the sign is the single thing a
  depth-weighting bug is most likely to get wrong. The parameter name now states it:
  positive means pressed *into* the skin. Feeding the stored `signed_depth_mm` straight
  in makes every point clamp to zero and trips the `d_max == 0` guard, so the mistake is
  an exception rather than a map with the peak on the shallowest vertex
  (`TestSignConvention::test_forgetting_the_negation_would_be_caught_not_absorbed`).
- **`penetration_from_signed_mm()` added to `contact_depth_field_io.py`** — outside the
  file list this phase declared, and deliberately so. `penetration_mm(frames)` takes a
  `DataFrame`; `TouchEvent.frame_depths` is a plain array walked one frame at a time in a
  hot loop. The two alternatives were wrapping every frame in a throwaway `DataFrame`
  (wasteful) or spelling the negation a second time in the estimator (which is how a sign
  convention drifts). Instead `penetration_mm` now delegates to the new array-level
  function, so there is still **exactly one negation in the codebase** and the two entry
  points cannot disagree. Output is bit-identical: the column is already float64.
- **`_compute_touch_rf` returns a `TouchRFMaps` NamedTuple**, not a 2-tuple —
  `(mean_pairs, max_pairs, weight_sum_pairs, n_eff_pairs)`, all four covering the same
  vertices in the same order, including through the NaN-mean filter. Seven existing call
  sites were updated to unpack it and to pass an explicit `alpha`
  (`tests/test_vertex_accumulator.py` x3, `tests/test_touch_playback_vertex_source.py` x2,
  `scripts/diagnose_vertex_reassignment.py` x2 — all `0.0`, because those tests pin the
  *reduction* and the *vertex assignment*, neither of which may move under weighting).
  That the Phase 2 characterization tests still pass by `np.array_equal` while running the
  weighted code path is Success Criterion 1's claim, arriving early for the pipeline site.
- **The contacted set is taken from the vertex lists, not from `weight_sum > 0`.** Those
  two agree at `alpha = 0` and part company above it, and 4.5 depends on the difference:
  *never contacted* and *contacted but weightless* both leave `weight_sum == 0`, and only
  the second is an error. Reading the contacted set off `np.unique(concatenate(...))`
  keeps them separable and is identical to `np.where(weight_sum > 0)[0]` at `alpha = 0`,
  so parity is unaffected.
- **`alpha < 0` raises.** Not asked for by the plan, but a negative exponent inverts the
  meaning of the weight (it would credit the *shallowest* contact most) and sends a
  clamped grazing point to infinity. Alpha must also be finite and a real number; `bool`
  is rejected.
- **No `if alpha == 0` branch, asserted on the source text.**
  `TestNoAlphaZeroBranch::test_the_source_contains_no_alpha_equality_branch` greps
  `inspect.getsource`. That is a structural claim no numeric test can make — a branch
  returning ones is numerically indistinguishable from the arithmetic that produces them —
  and it is what keeps the Phase 6 parity test from becoming vacuous.
- **The fixture and the plan's original table are both internally correct; they differ
  because one rounds the weights first.** *(Diagnosis corrected in Phase 5 — the Phase 4
  note recorded here previously called the plan's v6 figure "stale" and claimed it did not
  follow from the plan's own weight table. That was wrong, and it was wrong in a way worth
  writing down: it compared a number computed from exact weights against a weight table
  printed to 2 dp.)*

  The plan's per-vertex means were computed from the **exact** underlying weights. For v6
  those are `0.125, 0.13333, 0.100`, giving `23.583 / 0.35833 = 65.81` — the figure the
  plan quoted. `tests/rf_accumulator_fixtures.py` then chose depths that make the plan's
  *displayed* 2-dp weight table exact instead: frame 0 is `0.90, 3.00, 1.26, 0.39, 2.76`
  over `d_max = 3.00`, which is precisely `0.30, 1.00, 0.42, 0.13, 0.92`, and v6's three
  weights are exactly `0.13, 0.13, 0.10`. On those weights the same estimator gives
  `23.5 / 0.36 = 65.28`. Neither figure is an error; they are two different weight sets
  agreeing to 2 dp. Phase 5 rewrites both documents onto the fixture's values so that the
  numbers a reader sees are the numbers the test suite pins.

  Reproduced on the committed fixture depths at `alpha = 1`:

  | vertex | today | weighted | n_eff |
  |---|---|---|---|
  | v1 | 50.000 | 50.000 | 1.000 |
  | v2 | **75.000 <- old peak** | 65.986 | 1.770 |
  | v3 | 63.333 | **73.438 <- new peak** | 2.584 |
  | v4 | 70.000 | 57.143 | 1.690 |
  | v5 | 40.000 | 40.000 | 1.000 |
  | v6 always shallow | 63.333 | 65.278 | 2.959 |
  | v7 always deep | 63.333 | 63.214 | 2.999 |

  The peak moves from v2 to v3 as claimed, and v6 / v7 still barely move. `n_eff` for v6
  comes out at 2.959 — the 2.96 the plan quotes — which is the independent confirmation
  that the weights, on either weight set, are the ones the design intends.
- **The Testing Plan's "v6 and v7 both within 1e-9 of their unweighted values" is not
  achievable and was not implemented as written.** Both vertices are unweighted at
  63.333; weighted, v6 is 65.278 (1.94 Hz away) and v7 is 63.214 (0.12 Hz away). 1e-9
  would require the weights to be *exactly* flat, which the fixture's depths are not — and
  if they were, the test would prove nothing. The claim that is actually true, and is what
  the test asserts, is the *contrast*: v6 and v7 move by 1.94 and 0.12 Hz while v3 moves by
  10.10 Hz, factors of 5 and 80. Every value is additionally pinned to its hand-computed
  closed form at `rtol=1e-12`. Success Criterion 2's wording ("approximately their
  unweighted values") is the accurate one; the Testing Plan line should be read against it.
- **`depth_weight_alpha` is required at every level and has no default anywhere**, so the
  DAG stage cannot run until 5.2 registers the parameter. `spatial_map_single_touch_flow`
  takes it as **keyword-only and required** — a caller-side change made here rather than in
  Phase 5, because leaving the flow calling a now-required argument without it would have
  been dead code the moment it was written. Nothing in `configs/` was touched.
- **`.npz` gains three keys**, in both the mean and the max file: `rf_weight_sum`,
  `rf_n_eff` (vertex-aligned with that file's own `rf_data`) and `depth_weight_alpha`
  (a weight sum is uninterpretable without the exponent that produced it). Recording alpha
  in `single_touch_rf_summary.json`, which is what makes a config change *invalidate* the
  stage, is 5.5 and was left there.
- **`scripts/generate_rf_single_touch_detail.py` was not updated and is now stale.** It
  re-implements the mean branch of `_compute_touch_rf` inline (`:128`, sum / count / mean)
  rather than calling it, so its figures show the **unweighted** map. It was not one of the
  nine call sites Phase 2 merged, and wiring it needs an `alpha` source it does not have
  until Phase 5. It must either take `--depth-weight-alpha` in Phase 5 or delegate to
  `_compute_touch_rf`; until then its output disagrees with the saved `.npz`, which is
  exactly the failure mode the nine-site merge existed to prevent.

**Files Modified:**
- `src/analysis/receptive_field_mapping/data/vertex_weights.py` — new
- `src/analysis/receptive_field_mapping/data/contact_depth_field_io.py` —
  `penetration_from_signed_mm()`; `penetration_mm()` delegates to it
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-exports
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` —
  `TouchRFMaps`, `_touch_label`, weighted `_compute_touch_rf`, `depth_weight_alpha`
  threaded through `run_single_touch_rf_mapping`, three new `.npz` keys
- `scripts/analysis_workflow_processing.py` — `depth_weight_alpha` keyword-only on
  `spatial_map_single_touch_flow` (caller pass-through only; no config)
- `scripts/diagnose_vertex_reassignment.py` — explicit `alpha=0.0`, new return type
- `tests/test_vertex_weights.py` — new, 122 tests
- `tests/test_vertex_accumulator.py`, `tests/test_touch_playback_vertex_source.py` —
  explicit `alpha=0.0`, new return type

**Dependencies:** Phases 2, 3.

### Phase 5: Config, provenance, GUI option
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** `alpha` is configurable, required, and recorded.

- [x] 5.1 — Add `depth_weight_alpha: 1.0` under `options:` for `spatial_map_single_touch` in
      `configs/analyse_workflow_processing_dag.yaml` (`:48-55`). **ruamel.yaml round-trip only.**
- [x] 5.2 — Add to the stage registry `params` lambda in `scripts/analysis_workflow_processing.py`
      (`:1965-1972`), beside `neuron_mode`.
- [x] 5.3 — Thread through `spatial_map_single_touch_flow` (`:429-455`) into
      `run_single_touch_rf_mapping` (`:112-118`) into `_compute_touch_rf` (`:38-42`). **Required
      argument at every level — no default value anywhere.**
- [x] 5.4 — Add `depth_weight_alpha` to `_OPTION_GROUP_OF` in
      `src/utils/gui/analysis_runner_gui/task_detail_panel.py` (`:157-173`) —
      `spatial_map_single_touch` is in `_GROUPED_TASKS` (`:182`), so an ungrouped option **fails the
      render**.
- [x] 5.5 — Record `depth_weight_alpha`, the sidecar path, and the cache schema version in
      `single_touch_rf_summary.json` (`:251-258`) so a config change invalidates the stage and two
      runs are distinguishable on disk.
- [x] 5.6 — Mirror into `configs/analyse_workflow_dag.yaml` if that DAG exposes the same stage.
      **Not applicable, and now asserted rather than remembered:** that file defines no
      `spatial_map_single_touch` task, so there is nothing to mirror into.
      `TestProcessingDagConfig::test_the_combined_reference_dag_does_not_define_this_stage`
      fails the day that changes.

**Implementation notes (2026-08-21)**

- **Phase 4 had already threaded alpha to the flow; only the registry was missing.**
  `spatial_map_single_touch_flow` -> `run_single_touch_rf_mapping` -> `_compute_touch_rf`
  was complete and required at every hop before this phase started. 5.3 was therefore a
  verification, not an edit, and it is now pinned two ways: a call test per level
  (`TestMissingAlphaIsATypeError`) and a **signature** test per level
  (`TestNoDefaultAnywhere`). The second exists because a default added later would make
  the first quietly stop failing while still passing.
- **The registry does not use `.get(key, fallback)`.** `neuron_mode` beside it does, which
  is why `_required_option()` was added rather than following the neighbouring line:
  `DagConfigHandler.get_task_options` returns `{}` for an unknown task, so `.get` would
  turn a deleted config key into `None` and the run would die much later with a `TypeError`
  that never names the key. `_required_option` raises naming
  `tasks.spatial_map_single_touch.options.depth_weight_alpha` and listing what was present.
- **`contact_depth_field` had to be grouped too, or the GUI still failed.** Phase 2.5 added
  that nested block to the stage's options without adding it to `_OPTION_GROUP_OF`, and
  `_insert_grouped` requires **every** non-`force_processing` option of a grouped task to
  resolve a group — so the task detail panel raised on `spatial_map_single_touch` before
  this phase, for a reason that had nothing to do with alpha. Both keys are now in the
  `method` group, and the test iterates the *config's* option list rather than a hardcoded
  one, so the next option added without a group fails here instead of in front of the user.
  As a dict, `contact_depth_field` renders through the existing complex-value branch (a
  clickable label opening the YAML editor); `depth_weight_alpha`, a float, renders as a
  free-text `QLineEdit` cast by `_native_type`. The config value is written `1.0` and a test
  asserts it parses as `float`, because an int there would make the panel reject `0.5`.
- **`_CACHE_SCHEMA_VERSION` became `PLAYBACK_CACHE_SCHEMA_VERSION`.** 5.5 wants the cache
  layout in the summary, and reaching across a module boundary for an underscore-prefixed
  name to get it is how a private constant becomes public by accident. It is renamed once,
  re-exported from `data/__init__.py`, and its comment says why it is public.
  `touch_population_data.py` keeps its own private copy — different cache, not shared.
- **The sidecar paths were already in the summary (Phase 2.5) and were extended, not
  duplicated.** The `contact_depth_field.blocks` list keeps the per-block
  `source_block_file` / `sidecar_path` / observed `coordinate_space` rows; `alpha` and the
  cache version are new top-level keys beside them.
- **`configs/analyse_workflow_dag.yaml` *does* define `explore_touch_playback`** (`:398`),
  contradicting the Phase 2.5 note above, which said it did not. It is a reference config
  that no code loads — nothing in the repo opens that filename — so the stale
  `explore_touch_playback` block there is documentation drift rather than a broken run, and
  it was left alone. Recorded so it is not rediscovered as a bug.

**Additional work folded into this phase**

- **`scripts/generate_rf_single_touch_detail.py` no longer disagrees with the `.npz`.**
  Phase 4 flagged it: it kept an inline copy of the mean branch, so its `03b` figure drew
  the **unweighted** map while the saved file held the weighted one — the exact failure the
  nine-site merge existed to prevent. It now takes a **required** `--depth-weight-alpha` and
  *calls* `_compute_touch_rf`, plotting what that returns. `Σ w·IFF` is recovered as
  `mean × weight_sum` rather than accumulated again, so no arithmetic is duplicated: the
  panels are `Σ w·IFF`, `Σ w`, and their ratio, which reduce to the old
  `Σ IFF` / `count` / `mean` at `alpha = 0`. It imports `_compute_touch_rf` under its private
  name, matching the precedent `scripts/diagnose_vertex_reassignment.py` set in Phase 2.5;
  renaming the function to public would have touched ~50 lines across four test modules for
  no behavioural gain, and was deliberately not done.
- **`scripts/generate_rf_workflow_pptx.py` captions followed.** That deck embeds
  `03b_grouping_averaging.png` and described the middle panel as "how many of the 200 frames
  touched each vertex". It is `Σ w` now, so the caption says so — a slide describing the
  figure incorrectly is the same class of error as the figure itself.
- **The worked-example numbers in this plan and in the brainstorm were reconciled** onto the
  fixture's exact values; see the corrected note under Phase 4.

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — `depth_weight_alpha: 1.0` + comment
  (ruamel round-trip; the diff is a clean 10-line insertion, no reflow)
- `scripts/analysis_workflow_processing.py` — `_required_option()`, registry param
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` —
  `depth_weight_alpha` and `playback_cache_schema_version` in the sentinel
- `src/analysis/receptive_field_mapping/data/touch_playback_data.py` — constant renamed public
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-export
- `src/utils/gui/analysis_runner_gui/task_detail_panel.py` — `depth_weight_alpha` **and**
  `contact_depth_field` in `_OPTION_GROUP_OF`
- `scripts/generate_rf_single_touch_detail.py` — `--depth-weight-alpha`, delegates to
  `_compute_touch_rf`
- `scripts/generate_rf_workflow_pptx.py` — captions
- `tests/test_touch_playback_depth.py`, `tests/test_touch_playback_vertex_source.py` —
  constant rename
- `tests/test_depth_weight_alpha_config.py` — new, 25 tests
- `docs/development/brainstorms/contact-depth-field-adoption.md` — section 8 reconciled

**Dependencies:** Phase 4.

### Phase 6: Prove it, then try to break it
**Started:** 2026-08-21
**Completed:** 2026-08-21

**Goal:** The baseline is proven and the premise is tested against real data.

- [x] 6.1 — Parity test: `depth_weight_alpha = 0.0` produces output **byte-identical** to the
      baseline **re-pinned in Phase 2.5** — not to the pre-branch output. Model on
      `tests/test_rf_response_fields_parity.py` (`_assert_exactly_equal`, `:220`), with a docstring
      stating both halves of the claim: it **proves** the weighting machinery is an exact no-op at
      `alpha = 0`, because the same code path runs with `weights = ones`; it says **nothing** about
      agreement with maps produced before the Phase 2.5 vertex fix, which changed them on purpose.
- [x] 6.2 — Explicitly cover clamped grazing vertices at `alpha=0`: `max(d,0)` is `0.0`, and
      `0.0 ** 0 == 1.0`, so they must still receive weight 1.
- [x] 6.3 — Diagnostic script: for one session, report the distribution of **per-vertex weight
      variation** (max/min of each vertex's weights across its frames). Near-1 everywhere means the
      feature cannot change anything at any alpha.
- [x] 6.4 — Run `alpha=0` vs `alpha=1` on one session; plot per-vertex `|delta|` against distance
      from the RF hotspot. **Expected:** near-zero at the centre, largest at the periphery. **If
      instead the whole map shifts uniformly**, the weights are picking up the ~33x frame
      re-emission (hazard 3), not spatial structure — stop and investigate before trusting any
      output. **Delivered as a script, not as a run** — see the note below.

**Implementation notes (2026-08-21)**

- **The parity is genuinely byte-identical. Nothing was downgraded.** The risk table gave
  "`alpha=0` is not byte-identical due to float summation order" a **High** likelihood with a
  documented escape hatch (state a tolerance and amend the plan). The escape hatch was not needed
  and was not used: every comparison in `tests/test_rf_depth_weighting_parity.py` is
  `np.array_equal(..., equal_nan=True)`, and all 25 tests pass. Nothing in this branch is asserted
  with `assert_allclose` or `approx`. The claim is made at two levels, because the success
  criterion is about the *file*:
  - **function level** — `_compute_touch_rf(touch, n, mode, 0.0)` against the frozen
    `test_vertex_accumulator._legacy_compute_touch_rf`, over both fixtures and both neuron modes.
    The frozen function is **imported, not copied**: a second frozen copy is a second thing that
    can be "fixed" later, and the value of a characterization baseline is that exactly one of it
    exists.
  - **artifact level** — `run_single_touch_rf_mapping` is run twice over a synthetic session
    (loader and PLY resolver monkeypatched, so no experimental data is read), once normally at
    `alpha = 0` and once with the frozen estimator substituted, and the `rf_data` and
    `touch_id_map` arrays are compared out of the two `single_touch_rf_maps_*.npz` files.

  The artifact-level test carries an explicit **non-vacuity assertion**: the frozen estimator
  predates the confidence channel and emits none, so an empty `rf_weight_sum` in the baseline run
  is the fingerprint that the substitution actually took effect. Without it, a monkeypatch that
  silently failed to apply would compare the weighted run against itself and pass. There is also a
  **negative control** (`test_alpha_one_actually_moves_the_map`): every other assertion in that
  class would hold equally for an implementation that ignored `alpha` entirely.
- **6.1 also pins the weight vector, not only the output.** Equality of the *results* can survive
  a wrong weight vector, because any constant weight cancels between numerator and denominator.
  `test_every_weight_is_exactly_one` asserts the vector itself, so `alpha = 0` is pinned as "all
  weights are exactly 1" rather than "the answer came out the same".
- **6.2 is tested for the contrast, not just the case.** `alpha = 0` is the *only* exponent at
  which a permanently-grazing vertex is harmless: at `alpha > 0` it accumulates zero weight and
  `_compute_touch_rf` raises. The fixture (`_grazing_touch`) asserts both halves — the grazing
  vertex reports its unweighted 20 Hz at `alpha = 0`, and the same touch raises at `alpha = 1`.
  Asserting only the first would leave `0.0 ** 0.0 == 1.0` looking like an accident.
- **6.3 and 6.4 are shipped as scripts and are UNRUN on real data.** This work was forbidden to
  read the experimental database, so both were validated on synthetic fixtures only — including a
  deliberately flat-depth fixture that drives `diagnose_depth_weight_variation` to its FALSIFIED
  verdict, so the failure path is exercised rather than assumed. **Neither has ever seen a real
  session.** They must be run, in that order, before any depth-weighted output from this branch is
  trusted, alongside `scripts/diagnose_vertex_reassignment.py` from Phase 2.5.
- **6.4 became a second script rather than sharing the first.** The plan listed one new script;
  6.4 as written ("run ... and plot") cannot be a test and cannot be executed here, so it is
  `scripts/diagnose_depth_weight_delta_vs_distance.py`. Its `--out-png` is **optional**: the three
  printed statistics are the verdict and the picture is illustration. That is not a fallback — it
  is an explicit caller choice — and it exists because `matplotlib.savefig` crashes natively
  (`0xC06D007F`) in this repo's conda environment, a pre-existing fault that also aborts several
  unrelated test modules. The plotting branch is the one piece of Phase 6 that could not be
  executed at all.
- **The plan's 6.3 wording is loose and the script does not repeat it.** "Near-1 everywhere means
  the feature cannot change anything at any alpha" is true only for a ratio of *exactly* 1: since
  `ratio(alpha) = ratio(1) ** alpha`, a ratio of 1.05 reaches 2 at `alpha ~= 14`. The script
  separates the two — it counts exactly-frozen vertices apart from effectively-flat ones and
  prints the alpha that would be needed — and says in its own docstring that raising alpha until
  something moves is choosing the exponent to manufacture an effect rather than to express a
  mechanism.
- **Two vertex classes are excluded from 6.3's headline distribution, and counted instead.**
  Single-frame vertices (ratio 1 by construction — they would pad the "flat" share with vertices
  that were never eligible to vary) and vertices with a zero weight (ratio infinite — counted,
  never divided by). A coefficient of variation is reported beside the ratio because max/min is a
  two-sample statistic that one outlying frame can dominate.
- **6.4 reads the hotspot off the *baseline* map.** Taking it from the weighted map would make the
  plot self-fulfilling: the weighting would be measured against the peak it had just moved. Its
  three reported statistics are the centre/periphery contrast, the uniform fraction
  (`|median signed delta| / mean |delta|`, the decisive one — 1 is a pure common offset, which is
  the hazard-3 failure signature) and a rank correlation, explicitly labelled the weakest.

**Additional work folded into this phase — the viewer gap**

- **`touch_playback_explorer` was still accumulating with `weights = np.ones(...)`**, so the
  playback window drew the **unweighted** map while the `.npz` held the weighted one. The plan's
  manual-verification item ("open a viewer window and the saved map for the same touch; confirm
  they now agree") would have failed. Closing it is in scope precisely because the nine-site merge
  was accepted on the grounds that the screen and the pipeline could not diverge; leaving it open
  would have spent that cost and kept the failure.
- **`data/touch_frame_weights.py` (new) is now the single depth -> weight conversion.** The
  conversion (align check, `penetration_from_signed_mm`, `vertex_weights`, error context) was
  inline in `_compute_touch_rf`. Re-spelling it in the viewer would have reintroduced the
  divergence one layer up — shared *reduction*, drifting *weights*. Both callers now use this
  module. It is a new file the plan did not list, and it is the honest home for the one piece of
  knowledge that spans both sides: `vertex_weights` must not learn about `TouchEvent`, and
  `touch_playback_data` (a loader) must not learn about alpha.
- **`alpha` is threaded viewers-DAG -> registry -> flow -> launcher -> window**, required and
  keyword-only at every hop, exactly as on the processing side. `TouchPlaybackExplorer` now takes
  `depth_weight_alpha` as a required keyword-only argument, pinned by a signature test.
- **`mean_heatmap_scalars`'s second parameter was renamed `contact_count` -> `weight_sum`**, for
  the same reason `_compute_touch_rf`'s was in Phase 4: above `alpha = 0` it is no longer a count,
  and keeping the name would make an old screenshot and a new one silently incomparable.
- **Empty frames are now skipped in `accumulate_touch_frame`.** `vertex_weights` raises on an
  empty array by design (a frame with no contact points has no maximum depth), where the old
  `np.ones(0)` was silently fine. Behaviour on the fixtures is unchanged — the accumulation was
  already a no-op — and the Phase 2 characterization tests still pass by `np.array_equal`.
- **Four Phase 2 characterization call sites were updated to pass `alpha = 0.0` explicitly**
  (`tests/test_vertex_accumulator.py`, sites 6–9), following the precedent Phase 4 set: those
  tests pin the *reduction*, which may not move under weighting.
- **A `_required_option` helper was duplicated into `scripts/analysis_workflow_viewers.py`.** That
  script declares "no dependency on analysis_workflow_processing.py" in its own header and
  `DagConfigHandler` is vendored, so the alternatives were cross-importing two entry-point scripts
  or editing vendored code. Six duplicated lines is the cheapest of the three, and the duplication
  is documented in the docstring rather than left to be discovered.
- **The two configs are pinned to agree.** `TestViewersDagCarriesTheSameAlpha` asserts the viewers
  DAG's `explore_touch_playback.options.depth_weight_alpha` equals the processing DAG's value for
  `spatial_map_single_touch`. A viewer at a different alpha from the pipeline is the same failure
  the nine-site merge existed to prevent, reintroduced one layer up as config drift instead of
  code duplication.
- **The other three ones-passing sites were deliberately left alone**: `rf_population_heatmap.py`,
  `touch_population_explorer.py` and `rf_feature_space_explorer.py`. They aggregate
  *already-reduced per-touch values*, so weighting them would be **cross-touch weighting** — a
  deeper touch outranking a shallower one — which this plan lists explicitly as out of scope.
  Their `np.ones` is correct rather than stale, and `compute_rf_heatmap`'s docstring already says
  so.
- **The manual-verification item is now automated.** `TestPlaybackViewerAgreesWithTheSavedMap`
  replays the viewer's accumulation and compares it against `_compute_touch_rf` at four alphas,
  with no `QApplication`. One documented difference remains, and it is a presentation choice
  rather than a numeric one: the estimator *drops* vertices whose mean is NaN (so a "no neural
  data" touch reports zero vertices instead of an invisible heatmap), while the viewer keeps them
  NaN and paints them grey. The comparison is over the vertices the saved map contains, and the
  NaN fixture is included so the difference is exercised rather than avoided.

**Files Modified:**
- `tests/test_rf_depth_weighting_parity.py` — new, 25 tests
- `scripts/diagnose_depth_weight_variation.py` — new (6.3), **unrun on real data**
- `scripts/diagnose_depth_weight_delta_vs_distance.py` — new (6.4), **unrun on real data**
- `src/analysis/receptive_field_mapping/data/touch_frame_weights.py` — new; the single
  depth -> weight conversion, shared by the pipeline and the viewer
- `src/analysis/receptive_field_mapping/data/__init__.py` — re-exports
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — delegates to
  `touch_frame_weights`; `_touch_label` moved there
- `src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — weighted accumulation,
  required `depth_weight_alpha`, `contact_count` -> `weight_sum`
- `src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — alpha through the
  playback launcher
- `scripts/analysis_workflow_viewers.py` — `_required_option`, registry param, flow signature
- `configs/analyse_workflow_viewers_dag.yaml` — `depth_weight_alpha: 1.0` + comment (clean 10-line
  insertion, no reflow; round-trip verified with ruamel)
- `tests/test_vertex_accumulator.py` — explicit `alpha=0.0` at sites 6–9
- `tests/test_depth_weight_alpha_config.py` — `TestViewersDagCarriesTheSameAlpha`

**Dependencies:** Phase 5.

---

## Testing Plan

### Unit Tests
- [x] `vertex_weights`: `alpha=0` gives all ones, including for clamped grazing vertices.
- [x] `vertex_weights`: `alpha=1` on depths `0.2, 0.8, 1.5, 0.7, 0.1` gives
      `0.13, 0.53, 1.0, 0.47, 0.07`.
- [x] `vertex_weights`: deepest vertex always receives exactly `1.0`, every alpha.
- [x] `vertex_weights`: scale invariance — doubling every depth leaves weights unchanged.
- [x] `vertex_weights`: `d_max == 0` raises; NaN depth raises. Neither returns a value.
- [x] `vertex_weights`: negative (grazing) depths clamp to 0 and never subtract from `sum(w)`.
- [x] Accumulator: hand-computed weighted mean, not a self-comparison — a wrong weighting still
      returns a plausible Hz value.
- [x] Accumulator: `sum(w) == 0` for a vertex raises.
- [x] A vertex hit **twice in one frame** is credited with that frame's IFF at the summed weight
      `w1 + w2` (above 1, and that is intended), and its weighted mean is *identical* to that of a
      single row of weight `w1 + w2` — hand-computed at `alpha = 1`, with the collapsed reference
      built by handing the accumulator explicit weights rather than by re-running the estimator.
      The Kish `n_eff` deliberately does **not** match the collapsed form; the equivalence is a
      statement about the mean only.
- [x] `n_eff`: flat weights give `n_eff` approximately `N`; one dominant frame gives `n_eff`
      approximately 1.

### Integration Tests
- [x] The full 7-vertex / 3-frame fixture end to end: peak moves from v2 to v3; v1 and v5 unchanged;
      v6 (always shallow) and v7 (always deep) both close to their unweighted values.
      **Amended (Phase 4):** "within 1e-9" is not achievable and was not implemented as written —
      v6 moves 1.94 Hz and v7 0.12 Hz, against v3's 10.10 Hz. The tested claim is the *contrast*
      (factors of 5 and 80) plus every value pinned to its hand-computed closed form at
      `rtol=1e-12`. Success Criterion 2's "approximately their unweighted values" is the
      accurate wording.
- [x] `alpha=0` byte-identical to the **Phase 2.5 re-pinned** baseline through the whole pipeline.
      Asserted with `np.array_equal` at the estimator *and* at the saved `.npz`, over a synthetic
      session. No tolerance was introduced anywhere.
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
      task detail panel and the stage runs. **Partially covered automatically (Phase 5):**
      `TestGuiOptionGrouping` resolves a group for every option key the config actually declares
      for the stage, headlessly — the same call `_insert_grouped` makes, with no `QApplication`
      constructed. That is the half that used to raise. Opening a real window and typing a value
      into the field is still manual.
- [x] Open a viewer window and the saved map for the same touch; confirm they now agree.
      **Automated in Phase 6** (`TestPlaybackViewerAgreesWithTheSavedMap`). The viewer accumulated
      with `weights = ones` until then, so this item would have *failed*; it now runs the same
      shared conversion and the same shared reduction at the same alpha, checked headlessly at four
      alphas. Opening a real window once, by eye, is still worth doing.
- [ ] Confirm `single_touch_rf_summary.json` carries alpha, and that changing alpha re-runs the
      stage.

### Edge Cases
- [x] A frame where every vertex is grazing (`d_max = 0`) raises with frame context.
- [x] A vertex contacted in exactly one frame gives a value identical to today, any alpha (the
      weight cancels).
- [x] All-identical depths across a patch gives weights all 1.0 and output identical to today.
- [x] Duplicate `(frame_index, vertex_id)` in a frame is **carried through, not rejected and not
      reduced**: both rows keep their own depth, the vertex is credited with the frame's IFF at the
      summed weight `w1 + w2` (which may exceed 1), and its weighted mean is identical to that of a
      single row of weight `w1 + w2` — hand-computed, at `alpha = 1`, in
      `TestADuplicatedVertexIsCreditedAtTheSummedWeight`. Supersedes the original item, "raises
      (assert, do not reduce)", whose premise was false (hazard 4, struck).
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

---

## Modified Files

<!-- AUTO-GENERATED at the end of Phase 6. Every file this branch
     (`feature/depth-weighted-iff-attribution`) changed relative to its base
     (`feature/port-stroke-centroid-baseline`), across Phases 1-6, sorted
     alphabetically. Per-phase attribution stays in each phase's own
     **Files Modified** list; this is the flat union, for review and for the
     rollback procedure. Regenerate with:

       git log --name-only --pretty=format: feature/port-stroke-centroid-baseline..HEAD

     unioned with any still-uncommitted paths from `git status --porcelain`. -->

- `configs/analyse_workflow_processing_dag.yaml`
- `configs/analyse_workflow_viewers_dag.yaml`
- `docs/data-contracts/contact-depth-field.md`
- `docs/development/brainstorms/contact-depth-field-adoption.md`
- `docs/development/plans/active/depth-weighted-iff-attribution.md`
- `docs/receptive_field_workflow/README.md`
- `environment.yml`
- `pyproject.toml`
- `scripts/analysis_workflow_processing.py`
- `scripts/analysis_workflow_viewers.py`
- `scripts/diagnose_depth_weight_delta_vs_distance.py`
- `scripts/diagnose_depth_weight_variation.py`
- `scripts/diagnose_vertex_reassignment.py`
- `scripts/generate_rf_single_touch_detail.py`
- `scripts/generate_rf_workflow_pptx.py`
- `src/analysis/receptive_field_mapping/data/__init__.py`
- `src/analysis/receptive_field_mapping/data/contact_depth_field_io.py`
- `src/analysis/receptive_field_mapping/data/rf_data_loader.py`
- `src/analysis/receptive_field_mapping/data/rf_population_heatmap.py`
- `src/analysis/receptive_field_mapping/data/touch_frame_weights.py`
- `src/analysis/receptive_field_mapping/data/touch_playback_data.py`
- `src/analysis/receptive_field_mapping/data/vertex_accumulator.py`
- `src/analysis/receptive_field_mapping/data/vertex_weights.py`
- `src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
- `src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
- `src/analysis/receptive_field_mapping/gui/touch_population_explorer.py`
- `src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py`
- `src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py`
- `src/analysis/touch_analytics/preparation_pipeline.py`
- `src/utils/gui/analysis_runner_gui/task_detail_panel.py`
- `tests/rf_accumulator_fixtures.py`
- `tests/test_contact_depth_field_io.py`
- `tests/test_depth_weight_alpha_config.py`
- `tests/test_rf_depth_weighting_parity.py`
- `tests/test_touch_playback_depth.py`
- `tests/test_touch_playback_vertex_source.py`
- `tests/test_vertex_accumulator.py`
- `tests/test_vertex_weights.py`

38 files.
