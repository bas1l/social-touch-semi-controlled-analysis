# Brainstorm: Adopting the per-vertex contact depth field

**Started:** 2026-08-20   **Last matured:** 2026-08-20   **Status:** In progress — loader built, weighting design open

## Real goal (north star)
Replace the single per-frame `contact_depth` scalar with the **per-vertex penetration depth field**
now produced upstream, and use it to weight how each contacted vertex is credited with IFF when
building receptive-field maps. Today every vertex in a contact patch receives identical credit;
`max()` discards both the size of the contact and the shape of the indentation across it.

Authoritative spec: [`docs/data-contracts/contact-depth-field.md`](../../data-contracts/contact-depth-field.md).

---

## 1. What was built (uncommitted, on `feature/port-stroke-centroid-baseline`)

`src/analysis/receptive_field_mapping/data/contact_depth_field_io.py` — reader for the parquet
sidecars, plus `tests/test_contact_depth_field_io.py` (24 tests, all passing).

```python
load_depth_field(csv_path, *, expect_space, session_id, forearm_ply_path=None) -> DepthField
depth_field_path_for_csv(csv_path) -> Path
forearm_ply_path_for_sidecar(sidecar_path, session_id) -> Path
validate_depth_field_schema(table) -> None
penetration_mm(frames) -> np.ndarray
CoordinateSpaceError / DepthFieldSchemaError / ReferencePlyMismatchError
```

Design decisions taken:
- **`expect_space` is required.** Caller declares the space it assumes; the loader reads
  `coordinate_space` from parquet metadata and raises on mismatch. Closes the folder-name trap by
  construction — see section 2.
- **PLY join happens inside the loader**, with `reference_ply_vertex_count` validated before any
  `vertex_id` lookup.
- **PLY load is eager, not lazy** (deviation from the original sketch). A wrong-forearm join must
  fail at load time, not deep inside an analysis. The `.npy` cache in `load_forearm_vertices()`
  makes it cheap.
- **`signed_depth_mm` exposed verbatim** (negative = penetrating), with `penetration_mm()` as the
  single documented negation point.

Also changed: `data/__init__.py` re-exports; `pyarrow>=15` declared in `environment.yml` and
`pyproject.toml`. **Note:** pyarrow was not installed in `social-touch-analysis` at all (not even
transitively via pandas) — it was `pip install`ed into the conda env. Worth a `conda install` pass
so the env matches a fresh solve.

Nothing consumes the loader yet. `parse_contact_points()` at `rf_data_loader.py:200` and the
`%.1f` string blob path are untouched.

---

## 2. Audit: does existing analysis assume `blocks_rf_centered/` is RF-centred?

Only 4 of 11 sessions are genuinely RF-centred (ST13-03, ST14-02, ST16-05, ST18-04). The other 7
take the RF passthrough, where the directory is a byte-identical copy of `blocks_pca_calibrated/`
and the RF centre was never subtracted.

### The RF-mapping chain is structurally immune — this is the load-bearing negative
Every spatial statistic in `receptive_field_mapping/` is computed in a SLIM UV frame whose origin is
**data-derived**, and every reported offset is a **within-session difference**:
- `surface/forearm_slim_uv.py:211` — UV origin is the mesh vertex nearest the IFF-weighted centroid
  of contacted vertices, not `(0,0,0)`.
- `data/rf_boundary_preparation.py:295` — UV then rigidly re-aligned by PCA on the heatmap.

A rigid translation of the input cloud cancels out of the whole boundary / centroid / hotspot chain.
Commit `e2d65f3` (stroke-centroid baseline) is clean: it swapped one data-derived within-session
reference for another.

### The real exposure is in `touch_analytics/`
`clustering_pipeline.py:126` feeds `mean_contact_x/y/z` — raw absolute coordinates — as clustering
features; `:287` pools all 11 sessions with `pd.concat`; `CartesianBinningClusterer` then bins
equal-width over the **pooled** min/max at `n_bins: 20`. Each session has its **own** PCA fit, so all
eleven frames differ regardless of passthroughs. A location bin is not a consistent anatomical region.

Live, not dormant — `location: [mean]` enabled at `configs/analyse_workflow_processing_dag.yaml:670,
751, 768` and `configs/analyse_workflow_dag.yaml:126, 207, 224`.

**Not yet verified:** the provenance of `contact_location_*` back through `extraction_pipeline.py:443`.
Two comments in the repo contradict each other about it — `gui/preparation_viewer_data.py:62` says
"RF-centered", `shared_constants.py:61` says "raw 30 Hz Kinect samples". Resolve this before treating
the clustering results as invalid; it determines how bad the finding is.

### Docstrings that are simply false
- `rf_data_loader.py:99` — "always contains contact points in RF-centered space". False for 7/11.
  No wrong number falls out today, but 12 call sites are entitled to trust it.
- `gui/preparation_viewer_data.py:62` — same claim, and contradicts `shared_constants.py:61`.
- Published figures: `scripts/generate_rf_workflow_doc_figures.py:336` hardcodes `forearm_rf_centered`
  for `SESSION = "2022-06-17_ST16-02"` — a passthrough session — and labels it "RF-centred".
  Repeated in `docs/receptive_field_workflow/README.md:31,47,62`.

### Nothing reads `coordinate_space`
Zero hits for `parquet` / `read_table` / `coordinate_space` outside the contract doc. The new loader
is the first consumer.

---

## 3. ST18-01: resolved, and worse than the contract implies

**ICP registers forearm snapshots to each other within a session.** A session with one snapshot has
nothing to register — `register_session_forearms.py:106` returns early, no transforms file is
written, and `apply_icp_registration.py:281` copies everything through unchanged. The passthrough
condition is literally a missing `{session_id}_registration_transforms.json`.

It corrects for the **arm** shifting on the armrest between snapshots, not the camera —
`kinect_space_1` is one fixed sensor frame for the whole session.

Checked against the data (one-off, authorised):

| Session | Blocks | Forearm snapshots | Coverage |
|---|---|---|---|
| ST16-05 | 16 | **16** | one per block, all 16, each averaged over 14-53 frames |
| ST14-02 | 6 | **6** | one per block, each averaged over 12-21 frames |
| **ST18-01** | 16 | **1** | block-order01 only, **single unaveraged frame** |

Mechanically this is the benign branch. But every other session takes a fresh snapshot **per block**,
which only makes sense if the arm is expected to move between blocks. ST18-01's single snapshot almost
certainly records that annotation stopped after block 1 — not a judgement that the arm stayed still.
The pipeline cannot distinguish those cases; both give `len(params_list) == 1`.

Likely also explains the vertex-count anomaly: ST18-01's reference is 1,094 vertices against
7,374-17,929 elsewhere, and it is the only one built from a single depth frame rather than an average
of 12-106. Averaging fills depth dropouts. **ST13-01 (1,807 vertices) is worth checking the same way.**

Also: two `_with_normals.ply` files exist for ST18-01 block-order01 but only frame 90 is annotated —
frame_0011 is a stale orphan. PLY count on disk is not a reliable proxy for snapshot count.

Net: ST18-01's 16 blocks carry a forearm reference captured once, from one unaveraged frame in
block 1, with no correction for arm movement across the rest of the session, and nothing checks it.

PCA calibration is **per-session, pooled across all blocks**, fitted on sticker trajectories
(`set_xyz_reference_from_gestures.py:499`). It has no passthrough branch, so ST18-01's stage-4/5 output
is genuinely `pca_calibrated` — the `kinect_space_1` label applies only to stages 1-3.

Diagnostic script (scratchpad, not in repo): `count_forearm_references.py` — reports snapshot count,
`*_with_normals.ply` present, and whether transforms exist, for any session. Distinguishes the three
causes of a missing transforms file (single snapshot / clouds missing / GUI cancelled). Worth moving
into `scripts/` if kept.

---

## 4. The weighting design

### Parameterise so today's method is one setting of a dial
```
weight of vertex i  =  max(depth_i, 0) ** alpha        (normalised per frame — see open question)
```
- `alpha = 0` -> all weights equal -> **exactly today's uniform attribution**
- `alpha = 1` -> credit proportional to penetration
- `alpha -> infinity` -> all credit to the deepest vertex -> **exactly what the old `contact_depth` max() did**

Both existing analyses are the two ends of the dial. This is a sweep, not a new model, and it can
fail informatively: if the best alpha is 0, the idea is refuted cleanly.

**Powers, not softmax.** `exp(d/T)` spans the same limits but T carries units of mm and needs
retuning per session. Normalised powers are scale-invariant — rescaling every depth in a frame leaves
the weights unchanged, since the factor cancels in the normaliser. Useful given depth magnitudes may
differ across sessions.

### Choosing alpha
Held-out prediction, not visual sharpness. Build the map from a training subset of touches, predict
IFF on held-out touches, pick the alpha that predicts best. Sharpness improves under any concentration
of credit whether or not it is correct. The `alpha = 0` point gives the current method as a baseline
in identical units.

### The unverified-depth blocker does NOT gate this
The depth field has never been visually confirmed after Kinect Space 1. But weighting only
redistributes credit *within* a patch via `vertex_id`, and the RF chain is translation-invariant —
a displaced patch displaces the map identically with or without weights. The failure that would
matter (depths attached to the wrong vertices) is what upstream verified numerically: bitwise depth
preservation, 0.0999 mm vertex-to-PLY residual. Still want the visual check before trusting absolute
RF positions.

---

## 5. Where it plugs in

### The seam is one function
`pipelines/rf_single_touch_pipeline.py:79-85` (`_compute_touch_rf`) — the only place raw per-frame
IFF meets contact geometry in the persisted pipeline. Everything else consumes its output.

```python
np.add.at(val_sum,       verts, neuron_values[fi])
np.maximum.at(val_max,   verts, neuron_values[fi])
np.add.at(contact_count, verts, 1.0)
...
mean_values = val_sum[contacted_indices] / contact_count[contacted_indices]
```

`contact_count` is already `float64` — a weight-sum accumulator that always adds 1. The change is
`1.0 -> w` and `neuron_values[fi] -> neuron_values[fi] * w`; the existing division then yields a
weighted mean with no structural change.

### The work is feeding weights in, not the maths
- **Feeder seam:** `data/touch_playback_data.py:540-591`, where the vertex block is built and cached
  as `prev_vtx`. Depth is constant across a run, same as the vertex block, so it caches identically.
  Needs `frame_vertex_weights` on the `TouchEvent` dataclass (`:57`).
- **Cache schema bump required** — `_load_playback_cache` validates a fixed key list (`:246`, `:274`,
  `:311-320`).
- **Join on `frame_index` value**, never row position, then broadcast along the same run-length
  expansion the vertices already take.

### Optional / later
- `data/rf_population_heatmap.py:16-28` — only if you also want *cross-touch* weighting (deeper
  touches outranking shallower ones). If per-frame weighting is enough, leave it: `forearm_slim_uv.py`
  and the GUI stay consistent for free, since they re-average this output.
- **Six near-identical copies** of the same three-line accumulator exist in GUI display paths
  (`touch_population_explorer.py:635`, `rf_feature_space_explorer.py:545,628`,
  `touch_playback_explorer.py:308,549,602,959`). They will show the old unweighted picture until
  updated. Not a blocker; decide explicitly.
- **Do not start at** `rf_grid_interpolation.py` — purely downstream, never sees IFF or vertices.
- `rf_cluster_pipeline.py:181-190` aggregates on **float coordinate tuples**, the exact failure the
  contract warns about (one physical vertex split across several keys). Needs a `vertex_id` before
  weighting can reach it.

---

## 6. Rate handling — a pre-existing implicit weighting

**Contact geometry is forward-filled UP to the nerve rate; IFF is never averaged down to 30 Hz.**
`touch_playback_data.py:480` ffills `contact_points`; each nerve-rate row then re-emits the whole
30 Hz patch, so one Kinect frame is credited ~33 times.

Consequence: `n_frames` in `_compute_touch_rf` is the **nerve-rate row count**, and the per-vertex
mean is therefore already weighted by **how long each camera frame was held** — a frame stretched
over 40 rows outweighs one over 25.

Not introduced by this work and possibly intentional, but depth weights will multiply on top of it.
Know this before interpreting any change in the result.

---

## 7. Open questions for next session

1. **Normalise per frame, or not?** — the one blocking decision.
   - *Normalise:* each frame is one observation of the neuron; you distribute fixed evidence across
     candidate vertices ("probability map"). Frames with large total penetration get downweighted.
   - *Don't:* a bigger, deeper contact deposits more total credit ("weight map").
   - It does **not** cancel out of the weighted mean: the per-frame normaliser varies across frames,
     so it reweights frames, not just vertices.
2. **What happens to the `max` map?** A weighted mean is meaningful; a weighted max is not. Leave max
   unweighted, or emit only a mean in the weighted variant.
3. **Can a vertex appear twice in one frame?**
   - Contract says **yes** — "projection carries no uniqueness constraint... the pipeline's own viewer
     takes the deeper."
   - User says **no, by design**.
   - Mechanism that would break the guarantee: XY dedup runs in `blocks_deduped/`, *before*
     `vertex_id` is assigned in `blocks_projected/`. Dedup guarantees distinct positions, not distinct
     vertices — two points further apart than `dedup_epsilon` can still snap to the same vertex when
     the mesh is coarser than epsilon. More likely on low-vertex sessions (ST18-01: 1,094).
   - **Resolution: assert, don't reduce.** Check uniqueness per frame where weights are built and
     raise on violation. Costs nothing if the design guarantee holds; loud failure if it doesn't.
   - Settleable empirically: count `(frame_index, vertex_id)` pairs vs unique ones over one session's
     sidecars. Reads parquet only.
   - **Note this is currently invisible:** duplicates cancel today because they double both numerator
     and denominator. That cancellation only works while the weight is 1. Depth weighting makes a
     currently-harmless case load-bearing.
4. **Grazing contacts** — `penetration_mm` returns small negatives for them (legal data, not errors,
   per the contract). They would subtract from `contact_count` and can flip its sign. Clamp at the
   seam. Wrinkle: at `alpha = 0` grazing vertices must keep weight 1, or `alpha = 0` no longer
   reproduces today's behaviour exactly.
5. **Whether to fix the two KDTree snapping sites** — `rf_explorer_data.py:345` and
   `touch_population_data.py:686` re-derive vertex identity by KDTree at a 15 mm threshold, which the
   contract explicitly says not to do. The depth field carries `vertex_id` directly. Small change,
   removes a known error source from every population heatmap.
6. **Raise the `touch_analytics` clustering finding** (section 2) — independent of this work, but it
   can invalidate results already produced.

---

## 8. Decisions taken 2026-08-21 — in plain words

Section 7's open questions are resolved. This section is deliberately written without
notation so it can be re-read cold.

### The weight, in one sentence
**How fully was this vertex pressed, compared with the deepest point of the press at that
same instant?** 1.0 means it was the deepest point. 0.1 means it was barely touched while
somewhere else took the force.

```
w = ( depth of this vertex / deepest depth anywhere in this frame ) ** alpha
```

Divide by the frame's own **maximum**, not by the sum of depths. Reasons:

- Weights land in [0, 1] with 1 at the deepest point, which is what the quantity means.
- It degrades gently from today. Today every vertex has weight 1. Max-normalising only ever
  lowers weights *from* 1 — the deepest vertex stays at 1. Sum-normalising instead crushes
  every weight to about 1/K, changing a frame's total contribution from K to 1.
- Both ends of the dial land where they should: alpha=0 gives all-ones (today exactly),
  alpha to infinity gives all credit to the deepest vertex (the old `contact_depth = max()`
  exactly).
- Scale-invariant: double every depth in a frame and nothing changes, because the factor
  cancels. That was the real reason to normalise at all — depth magnitudes may differ
  between sessions.

### Order of operations matters
Normalise **before** raising to the power, not after.

- `(d_i / d_max) ** alpha` at alpha=0 gives 1 for every vertex. Today reproduced exactly.
- `d_i**alpha / sum(d_j**alpha)` at alpha=0 gives 1/K for every vertex. Today NOT reproduced.

At alpha=1 the two are identical, so this costs nothing and preserves the baseline test.
Caveat for later: "normalise before" is only true normalisation at alpha=1; at alpha=2 the
weights no longer sum to a fixed total. Document it if alpha is ever swept.

### alpha is fixed at 1 for now
alpha stays configurable ONLY so the alpha=0 parity test can run. No automatic alpha
chooser in this work. Per the fail-fast rule it is a **required** argument with the value
written in YAML — an un-passed alpha is a TypeError, never a silent 1.0.

### The estimator: divide by the sum of weights
```
value of vertex v  =  sum over frames ( w * IFF )  /  sum over frames ( w )
```
Denominator is the weight sum, NOT the frame count. Dividing by the count would leave the
weight in the answer as a scale factor, so the number would stop being a firing rate. A
vertex touched 3 times at 100 Hz must report 100 Hz whatever its weights were.

### Worked example that settled it
Five vertices in a row, finger sweeping left to right over three frames, true receptive
field at v3. IFF was 50, 100, 40 Hz.

| vertex        | today          | weighted        |
|---------------|----------------|-----------------|
| v1 one frame  | 50.0           | 50.0            |
| v2 flank      | **75.0 <- peak** | 65.9          |
| v3 RF centre  | 63.3           | **73.5 <- peak** |
| v4 flank      | 70.0           | 57.1            |
| v5 one frame  | 40.0           | 40.0            |

**Today's map puts the peak on the wrong vertex.** In the frame where the finger was on v3
and the neuron fired 100 Hz, v2 and v4 were inside the contact patch and received that
100 Hz at full credit, identical to v3. The flanks get inflated by their neighbour's best
moment. Weighting fixes it because v2 was pressed to only 47% of that frame's maximum.

### What the weighting can and cannot do
Two more vertices were added to the example: v6 always shallow (rim of the stroke),
v7 always deep (midline of the stroke). **Neither one moved.**

- v6 weights across frames: 0.13, 0.13, 0.10 — flat. Cancels out. 63.3 -> 65.8
- v7 weights across frames: 0.92, 0.93, 0.95 — flat. Cancels out. 63.3 -> 63.3
- v3 weights across frames: 0.42, 1.00, 0.50 — **varying.** 63.3 -> 73.5

**Absolute depth level is irrelevant. Only frame-to-frame variation in a vertex's own depth
does anything.** Effective sample size confirms it: v6 keeps n_eff = 2.96 out of 3 frames
despite weights of 0.1. Being shallow costs nothing; being *inconsistently* shallow is what
carries information.

Consequence, and a real falsification path: **if depth is roughly uniform across each contact
patch in the real data, this feature produces today's map at every alpha.** Measure the
per-vertex weight variation before assuming the weighting will do anything.

### Rejected: damping shallow vertices in the number itself
A chronically-shallow vertex reporting a normal firing rate looks wrong, and the instinct is
to damp it. Three estimators were compared on the 7-vertex example:

| vertex          | today | A: /sum(w) | B: /count | C: /(sum(w)+1) |
|-----------------|-------|------------|-----------|----------------|
| v3 true RF      | 63.3  | **73.5**   | 46.9      | **48.3**       |
| v6 always shallow | 63.3 | 65.8      | **7.9**   | 17.4           |
| v7 always deep  | 63.3  | 63.3       | **59.1**  | 46.6           |
| peak lands on   | v2 X  | **v3 OK**  | **v7 X**  | **v3 OK**      |

**B rejected.** Damping v6 and inflating v7 are the same operation — you cannot have one
without the other. Under B the peak moves to v7, and v7 is deep because of where the
experimenter chose to stroke and how the arm curves, **not because of the neuron.** B imports
stimulus geometry into the receptive field map, and its output is no longer in Hz.

**C rejected for now.** Shrinkage works and keeps the peak on v3, but `k` is a tuning knob
that can move the peak (v3 leads v7 only while k < 1.5 in this example). A parameter that
decides where the receptive field is must be chosen on principle, not fitted by eye. It also
weakens the parity test, which would then need both alpha=0 and k=0.

**A chosen.** It separates two claims that B and C conflate:
1. *Attribution* — within one frame, v3 deserves more of that 100 Hz than v6 does. A does
   this, and it is what moves the peak from v2 to v3.
2. *Confidence* — v6 was never firmly pressed, so we do not really know what v6 does. That
   is a statement about evidence, not about firing rate.

A keeps attribution in the number and moves confidence to a **separate channel**: carry the
weight sum and Kish effective sample size `n_eff = (sum w)^2 / sum(w^2)` alongside every
estimate, and dim or mask thin-evidence vertices **at display time**. Same visual outcome,
no tuning knob deciding the peak, and the numbers stay honest Hz.

### Guards that are not optional
- **d_max = 0** (every vertex in a frame grazing) is a 0/0. NumPy would quietly rescue
  alpha=0 via `NaN ** 0 == 1.0`. Explicit check, explicit raise — no relying on IEEE trivia.
- **sum(w) = 0** for a vertex is reachable at alpha > 0. Today `count >= 1` always. Explicit
  policy, never a NaN flowing into the map.
- **NaN depth** rejected at the loader boundary, where "zero" and "absent" are still separable.
  At alpha > 0 both collapse to w = 0.

### Scope decision
The averaging maths exists in **9 places** (pipeline, `rf_population_heatmap.py`, and 7 copies
across GUI viewer windows). They are merged into one shared function **first**, as a
no-behaviour-change step pinned by tests, and weighting lands in that single place afterwards.
Reason: with alpha fixed at 1 this gets judged by looking at maps, and a viewer window drawing
the old v2-peaked picture while the saved file holds the v3-peaked one is the one failure mode
that would actively mislead.

### `max` map stays unweighted
`np.maximum.at` has no meaningful weighted analogue — weighting a maximum changes its units.
The mean map is weighted; the max map is not. Documented, not silently inconsistent.
