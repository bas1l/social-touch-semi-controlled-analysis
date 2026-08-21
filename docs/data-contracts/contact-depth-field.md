# Data Contract: Contact Depth Field

**Producer:** `social-touch-semi-controlled` (the pipeline repo)
**Consumer:** this repo
**Schema version:** 2
**Contract written:** 2026-08-20
**Status of the data:** produced for all 11 sessions / 99 blocks

---

## What is newly available

Every contact frame now carries a **per-vertex penetration depth**, not just the single
`contact_depth` scalar the merged CSV has always held.

Until now the pipeline computed a signed distance at *every* forearm vertex inside the contact
patch, took `max()`, wrote that one number to `contact_depth`, and threw the rest away. That field is
now persisted as a **parquet sidecar** beside every stage CSV, and — as of 2026-08-19 — it is carried
through all five postprocessing spatial stages, so it arrives in the same frame as the coordinates
you already analyse.

One row per **(frame, contact vertex)**.

This is what makes depth-weighted analysis possible, e.g. the IFF weighting

```
w(depth_i) x RF_sensitivity(position_i)
```

which needs depth and receptive-field sensitivity expressed over the *same* vertices, in the *same*
space.

---

## The three things to know before you write any code

**1. `vertex_id` is the join key, not the coordinates.**
It is an integer index into the session's reference forearm PLY, and it is **invariant across the
post-projection spaces**. Join it against whichever forearm you want and read coordinates from there.
The `x/y/z` in the file are float32 and have been through several transforms with rounding at three
separate points; the vertex index has not.

**2. `coordinate_space` is metadata, never the directory name.**
Because of documented passthrough branches, the folder lies for **8 of 11 sessions**. Read
`coordinate_space` from the file's metadata and branch on that. Details in
[Coordinate space](#coordinate-space-the-one-real-trap) — this is the single most likely way to get
silently wrong answers.

**3. Join to a frame by `frame_index`, never by row position.**
The merged CSV is upsampled ~33x to the nerve rate and *not* uniformly. Row *i* of the sidecar has no
relationship to row *i* of the CSV.

---

## Where the files are

Each sidecar sits **in the same directory as the stage CSV it describes**, and differs from it only
in one name segment and the extension:

```
<block>_merged_data.csv            ->  <block>_contact_depth_field.parquet
<block>_merged_data_pca-xyz.csv    ->  <block>_contact_depth_field_pca-xyz.parquet
```

The rule is **substitution of the `_merged_data` marker with `_contact_depth_field`**, not stripping
a trailing suffix — that is what keeps it valid after the PCA stage renames the stem.

Under a session's merged output root:

| Directory | Stage | Sidecar? |
|---|---|---|
| `blocks_merged/` | merged, pre neural-quality filter | **no** |
| `blocks_filtered/` | after `filter_contact_depth_field_by_neural_quality` | yes |
| `blocks_registered/` | after ICP registration | yes |
| `blocks_deduped/` | after XY deduplication | yes |
| `blocks_projected/` | after contact projection | yes — **first with `vertex_id`** |
| `blocks_pca_calibrated/` | after PCA calibration | yes |
| `blocks_rf_centered/` | after RF centring | yes |

`blocks_merged/` has no sidecar by construction: it is written *before* the neural-quality filter,
whose output lands in `blocks_filtered/`. That is expected, not a missing file.

---

## Schema

Exactly two legal column layouts: the six required columns, or those six followed by `vertex_id`.
Names, order and dtypes are all exact — nothing is coerced on read.

| # | Column | dtype | Notes |
|---|---|---|---|
| 0 | `frame_index` | `int32` | Kinect frame. **The join key.** |
| 1 | `time_s` | `float64` | per-frame timestamp, repeated on every row of that frame |
| 2 | `x` | `float32` | |
| 3 | `y` | `float32` | |
| 4 | `z` | `float32` | in the space named by the metadata |
| 5 | `signed_depth_mm` | `float64` | **negative = penetrating** |
| 6 | `vertex_id` | `int32` | **optional** — present only from `blocks_projected/` onward |

`signed_depth_mm` is float64 on purpose: it must stay bit-identical to the value the CSV's
`contact_depth` was derived from. `x/y/z` are float32 because their CSV counterpart is already
quantised to 0.1 mm.

### Metadata keys

Parquet file-level metadata, all string-valued.

| Key | Always present | Meaning |
|---|---|---|
| `schema_version` | yes | `"1"` or `"2"`. v2 adds the optional `vertex_id` + reference-PLY provenance. A v2 file may legally carry no `vertex_id`. |
| `coordinate_space` | yes in practice | one of `kinect_space_1`, `icp_registered`, `pca_calibrated`, `rf_centered` |
| `units` | yes | always `mm` — there is no unit conversion anywhere in the pipeline |
| `sign_convention` | yes | always `negative_is_penetrating` |
| `source_recording` | yes | |
| `produced_by` | yes | |
| `pipeline_stage` | no | `merging` or `postprocessing`; absent on a raw Space-1 sidecar |
| `reference_ply` | from projection on | which forearm the `vertex_id` indexes |
| `reference_ply_vertex_count` | from projection on | **validate against your PLY before joining** |
| `dedup_epsilon` | from projection on | the epsilon that defined the vertex set |

---

## Coordinate space: the one real trap

The pipeline has two **passthrough** branches that copy their input unchanged rather than
transforming it. When that happens, the honest thing to declare is the space the data is *actually*
in — so the sidecar in `blocks_rf_centered/` may legitimately say `pca_calibrated`.

- **RF centring passthrough** — when no receptive-field cluster is found, `blocks_rf_centered/` is a
  byte-identical copy of `blocks_pca_calibrated/`. **The RF centre was not subtracted.** If your
  analysis assumes the origin is the RF centre, it is wrong for these sessions.
- **ICP passthrough** — when a session has no registration transforms, `blocks_registered/` (and
  therefore `blocks_deduped/` and `blocks_projected/`, which move no points) stays in
  `kinect_space_1`.

### What that means for the current dataset

Measured across all 11 sessions x 99 blocks (495 resolutions). **P** = passthrough.

| Session | Blocks | Ref. vertices | stages 1-3 | stage 4 | stage 5 (`blocks_rf_centered/`) |
|---|---|---|---|---|---|
| 2022-06-14_ST13-01 | 4 | 1 807 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-14_ST13-02 | 8 | 13 271 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-14_ST13-03 | 10 | 11 062 | `icp_registered` | `pca_calibrated` | `rf_centered` |
| 2022-06-15_ST14-01 | 9 | 10 483 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-15_ST14-02 | 6 | 10 712 | `icp_registered` | `pca_calibrated` | `rf_centered` |
| 2022-06-15_ST14-04 | 3 | 11 840 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-17_ST16-02 | 15 | 17 578 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-17_ST16-03 | 5 | 17 852 | `icp_registered` | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-17_ST16-05 | 16 | 17 929 | `icp_registered` | `pca_calibrated` | `rf_centered` |
| **2022-06-22_ST18-01** | 16 | 1 094 | **`kinect_space_1` P** | `pca_calibrated` | `pca_calibrated` **P** |
| 2022-06-22_ST18-04 | 7 | 7 374 | `icp_registered` | `pca_calibrated` | `rf_centered` |

**Only 4 of 11 sessions are genuinely RF-centred.** Seven take the RF passthrough; ST18-01 takes both.

This is not only a depth-field concern — it applies equally to the CSV and the forearm PLY in those
directories, which this repo already loads. `forearm_rf_centered/*.ply` is likewise a copy of the PCA
one in a passthrough session.

---

## `vertex_id`: the space-independent join

`vertex_id` is the row index, in file order, into the session's reference forearm PLY. The three
post-projection forearm PLYs (`forearm_deduped/`, `forearm_pca_calibrated/`, `forearm_rf_centered/`)
are transformed **in place, in file order, with no reordering, filtering or count change** — so
vertex *i* is the same physical vertex in all of them.

Practical consequences:

- Join `vertex_id` against whichever forearm PLY matches the space you want to work in.
- You can aggregate across frames or blocks by `vertex_id` without any coordinate matching. (This
  also avoids a known failure mode: aggregating on float coordinate tuples parsed from the `%.1f`
  `contact_points` blob splits one physical vertex across several keys.)
- **Validate before joining.** Check `reference_ply_vertex_count` against `len(ply.points)`. A
  re-deduplication at a different epsilon renumbers every vertex, and nothing upstream would notice.
- `vertex_id` is **absent** before `blocks_projected/`. There is no fallback, and you should not
  invent one — nearest-vertex snapping on this data was measured overshooting 29 mm off-surface at
  touch boundaries.
- Two rows in one frame may address the same vertex; projection carries no uniqueness constraint. If
  you need one value per vertex, reduce explicitly (the pipeline's own viewer takes the deeper).

---

## Sign and units

- Storage is **signed**, millimetres, `negative = penetrating`. The sign comes from a raycasting scene
  built on the hand mesh and queried at forearm vertices: inside the hand is negative.
- For display or weighting use **`penetration_depth_mm = -signed_depth_mm`**, a positive magnitude.
  Negate **exactly once**, at the boundary of your code, over the whole column. Because negation
  reverses order, a colour range becomes `(-signed_max, -signed_min)` — not `(-min, -max)`.
- Values above `epsilon = 1e-5` mm positive exist and are grazing contacts, not errors.
- The pipeline rejects anything beyond 200 mm as implausible. A value of `0.03` where you expected
  `30` is the metres-vs-millimetres tell.

---

## Rules

**Do**

- Read `coordinate_space` from metadata and assert it is what you expect.
- Group by `frame_index`. Sort defensively — order is not promised by the schema.
- Treat a frame that is **absent from the file** as *no contact in that frame*.
- Validate `reference_ply_vertex_count` before any `vertex_id` join.

**Do not**

- Infer the space from the directory name.
- Join by row position, or assume the sidecar and the CSV are row-aligned.
- Forward-fill or interpolate across frames. The contact patch changes membership frame to frame, so
  row *i* of frame *n* is **not** the same vertex as row *i* of frame *n+1*.
- Re-derive `vertex_id` yourself from coordinates. The CSV holds float64 parsed from `%.1f` text
  while the parquet holds full float32, so a re-derivation picks *different* vertices.
- Confuse **zero rows** (no contact) with **a missing file** (the artifact was never produced).

---

## Invariants you can rely on

- Per frame, `max(abs(signed_depth_mm))` equals the CSV's `contact_depth` for that frame, to about
  one ULP of the CSV's decimal round trip (`rtol = atol = 1e-9`).
- `signed_depth_mm` is **bitwise unchanged** by ICP registration, projection, PCA calibration and RF
  centring. Depth is measured once, in preprocessing, and never recomputed.
- **Deduplication is the only stage that changes it**: when rows collapse, the survivor inherits the
  **deepest** (most negative) value of its group, taken verbatim. Row counts drop by roughly 35-45%;
  the per-frame maximum is unchanged.
- Sidecar and CSV agree on the contact-point count for every frame, at every stage.

A subtlety if you compute a range over a whole recording: the *shallow* end can differ slightly
between `blocks_registered/` and `blocks_deduped/`. That is dedup dropping the single row that
carried the shallow extremum — `min`/`max` are taken over a row set, and the set shrank. No depth
value changed.

---

## Worked example

```python
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import open3d as o3d

SESSION = "2022-06-15_ST14-02"
BLOCK   = "block-order-01"
root    = MERGED_ROOT / SESSION            # your data root

sidecar = (root / "blocks_rf_centered"
           / f"{SESSION}_semicontrolled_{BLOCK}_contact_depth_field_pca-xyz.parquet")

table = pq.read_table(sidecar)
meta  = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

# 1. Never infer the space from the folder.
space = meta["coordinate_space"]
if space != "rf_centered":
    # Passthrough session: RF centring copied its input, the origin is NOT the RF centre.
    ...  # handle or skip, but do not silently treat it as RF-centred

df = table.to_pandas()

# 2. Join to a frame by frame_index, never by row position.
frame = df.loc[df["frame_index"] == 1260]
penetration_mm = -frame["signed_depth_mm"].to_numpy()      # negate once

# 3. Resolve geometry through vertex_id, after validating provenance.
ply   = o3d.io.read_point_cloud(str(root / "forearm_rf_centered" / f"{SESSION}_forearm.ply"))
verts = np.asarray(ply.points)
assert len(verts) == int(meta["reference_ply_vertex_count"]), "wrong reference forearm"

xyz = verts[frame["vertex_id"].to_numpy()]                  # (N, 3) in `space`

# 4. Aggregate across frames by vertex, not by coordinate.
deepest_per_vertex = (
    df.assign(penetration_mm=lambda d: -d["signed_depth_mm"])
      .groupby("vertex_id")["penetration_mm"]
      .max()
)
```

**Size note.** There is no streaming reader anywhere in the producer — every read is a whole-file
`pq.read_table(...).to_pandas()`. Sidecars run roughly 5-90 MB per block. If you iterate over many
blocks, load lazily and bound residency rather than accumulating.

---

## What has *not* been verified

Stated plainly so nobody builds on a false assumption:

- **The field has never been visually confirmed in any space after Kinect Space 1.** Its propagation
  is verified numerically — row-count agreement at every stage, bitwise depth preservation, and a
  `vertex_id`-to-PLY residual of 0.0999 mm on a translating session and 0.0499 mm on a passthrough
  one (exactly the ratio the two-vs-one rounding predicts). But no one has yet looked at the
  RF-centred patch sitting on the RF-centred forearm and confirmed it is anatomically where it
  should be.
- Verification covered 2 sessions end to end in depth; the space/coverage sweep above covers all 11.

If something looks spatially wrong downstream, treat it as an open question about the propagation,
not as a settled artifact.

---

## Provenance

In the producer repo, `social-touch-semi-controlled`:

| What | Where |
|---|---|
| Format, schema, validation | `code/src/preprocessing/motion_analysis/tactile_quantification/io/contact_depth_field_io.py` |
| Depth computation, sign convention | `.../tactile_quantification/model/contact_depth_field.py` |
| Stage transforms, sidecar path rule | `code/src/postprocessing/depth_field_stage_io.py` |
| A worked reader with a bounded cache | `code/src/merging/contact_depth_field_series.py` |
| Design record | `docs/development/plans/completed/propagate-contact-depth-field-through-postprocessing.md` |
| Motivation | `docs/development/brainstorms/per-vertex-contact-depth.md` |

`depth_field_path_for_csv()` in `depth_field_stage_io.py` implements the sidecar naming rule if you
would rather port it than reimplement it.
