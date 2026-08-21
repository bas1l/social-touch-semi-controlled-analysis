"""Can depth weighting change anything on this session at all?  The cheap falsification.

What this measures
------------------
For every forearm vertex, the **variation of its own weight across the frames it
appears in**: ``max_f w[v, f] / min_f w[v, f]``, where
``w = (penetration / frame max penetration) ** alpha``.

Why that is the right quantity, and not the depth level
-------------------------------------------------------
The estimator is ``value[v] = sum_f(w * IFF) / sum_f(w)``.  A weight that is the
same in every frame **cancels between numerator and denominator** and leaves the
vertex's value exactly where it was.  So the absolute depth of a vertex is
irrelevant: a vertex that is always shallow and a vertex that is always deep both
report their unweighted value.  Only *frame-to-frame variation in a vertex's own
weight* moves anything.

That gives the whole feature a falsification path, which is what this script runs:

    **If each contact patch is roughly uniform in depth, or each vertex sees
    roughly the same relative depth every time it is touched, then depth
    weighting reproduces today's map and the premise of the feature is wrong on
    this data.**

Reading the output
------------------
``ratio = max(w) / min(w)`` per vertex, over the frames where that vertex was
contacted, pooled over every touch in the session.

* **ratio == 1.0 exactly** -> that vertex cannot move at *any* alpha.  Its weight
  is constant, so the cancellation is exact.
* **ratio near 1 (say < 1.05)** -> that vertex barely moves at ``alpha = 1``.  It
  is not frozen: raising alpha raises the ratio, because
  ``(max w / min w) ** alpha = ratio ** alpha``.  A ratio of 1.05 needs
  ``alpha ~= 14`` to reach a ratio of 2.  This script therefore reports the alpha
  that would be needed, rather than repeating the looser claim that a near-1
  ratio kills the feature at every alpha — that is true only for ratio exactly 1.
* **ratio large (>= 2) for a substantial share of vertices** -> the weighting has
  real material to work with and the premise survives this test.

**The failure signature** — the result that says stop — is: the ratio
distribution is concentrated at 1, the ``effectively flat`` share is close to
100%, and ``alpha needed for a 2x spread`` at the median is large (tens).  In that
case depth weighting is a no-op on this data at any alpha you would defend, and no
amount of tuning changes it.  Report the numbers and stop; do not raise alpha
until something moves, because at that point the exponent is being chosen to
manufacture an effect rather than to express a mechanism.

Two vertex classes are excluded from the headline distribution and reported
separately, because including them would bias the answer in opposite directions:

* **single-frame vertices** — contacted in exactly one frame.  Their ratio is
  ``1.0`` by construction and carries no information at all; they would pad the
  "flat" share with vertices that were never eligible to vary.
* **zero-weight vertices** — contacted at least once with a clamped (grazing)
  penetration, so ``min(w) == 0`` and the ratio is infinite.  These are counted
  and named rather than divided by: at ``alpha > 0`` a vertex whose weights are
  *all* zero makes ``_compute_touch_rf`` raise, and one whose weights are only
  *sometimes* zero is the most extreme variation there is.

A coefficient of variation (std / mean of each vertex's weights) is reported
beside the ratio because the ratio is a two-sample statistic — one deep frame and
one grazing frame produce a huge ratio even if the other 200 frames are identical.
The two disagreeing is itself informative and is worth seeing.

Scope
-----
This is a *diagnostic*, not a stage: it writes no artifact and feeds nothing.
It reads one session and prints.  It has never been run on real data by the
author of this branch (that work was forbidden to read the experimental
database), so its numbers are unknown and **it must be run before any depth-
weighted result from this branch is trusted**.

Usage
-----
    python scripts/diagnose_depth_weight_variation.py \\
        --session 2022-06-15_ST14-02 \\
        --merged-root  "<db>/3_merged" \\
        --prepared-csv "<db>/4_analysed/preparation/<session>_prepared.csv" \\
        [--depth-weight-alpha 1.0] \\
        [--blocks-stage-dir blocks_rf_centered] \\
        [--flat-ratio-threshold 1.05]

Every path is an argument.  Nothing is discovered, defaulted or guessed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_data_loader import (  # noqa: E402
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.touch_frame_weights import (  # noqa: E402
    touch_frame_weights,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    load_playback_data,
)


def collect_vertex_weights(
    playback, depth_weight_alpha: float
) -> Dict[int, List[float]]:
    """Every weight every vertex received, keyed by vertex index.

    Pooled across all touches of the session.  Pooling is the right default for
    the *premise* question — "does this vertex's weight vary at all, ever?" — and
    it is deliberately more generous to the feature than a per-touch view would
    be, since a vertex touched in two different touches at two different depths
    counts as varying here.  A flat result under the generous view is therefore a
    strong negative.

    The weights come from ``touch_frame_weights``, the same function the pipeline
    and the playback viewer use.  Re-deriving them here would let this diagnostic
    measure something the estimator does not do.
    """
    per_vertex: Dict[int, List[float]] = {}
    for block_id in playback.block_order_ids:
        for trial_id in playback.trial_ids_by_block[block_id]:
            for touch in playback.touches_by_block_trial[(block_id, trial_id)]:
                for fi, verts in enumerate(touch.frame_vertex_indices):
                    if len(verts) == 0:
                        continue
                    weights = touch_frame_weights(touch, fi, depth_weight_alpha)
                    for vertex_idx, weight in zip(np.asarray(verts), weights):
                        per_vertex.setdefault(int(vertex_idx), []).append(float(weight))
    return per_vertex


def summarise(per_vertex: Dict[int, List[float]], flat_ratio_threshold: float) -> dict:
    """Split the vertices into the three classes and describe the eligible ones."""
    ratios: List[float] = []
    cvs: List[float] = []
    n_single_frame = 0
    n_zero_weight = 0
    n_frames_total = 0

    for weights in per_vertex.values():
        arr = np.asarray(weights, dtype=np.float64)
        n_frames_total += arr.size
        if arr.size == 1:
            n_single_frame += 1
            continue
        w_min = float(arr.min())
        if w_min <= 0.0:
            # min == 0 makes the ratio infinite. Counted, never divided by.
            n_zero_weight += 1
            continue
        ratios.append(float(arr.max()) / w_min)
        mean = float(arr.mean())
        cvs.append(float(arr.std()) / mean)

    return {
        "n_vertices": len(per_vertex),
        "n_frames_total": n_frames_total,
        "n_single_frame": n_single_frame,
        "n_zero_weight": n_zero_weight,
        "ratios": np.asarray(ratios, dtype=np.float64),
        "cvs": np.asarray(cvs, dtype=np.float64),
        "flat_ratio_threshold": flat_ratio_threshold,
    }


def _alpha_for_two_fold(ratio_at_alpha: float, alpha: float) -> float:
    """Alpha at which this vertex's weight spread would reach 2x.

    ``ratio(alpha) = ratio(1) ** alpha``, so ``ratio(1) = ratio(alpha) ** (1/alpha)``
    and the alpha reaching 2 is ``log 2 / log ratio(1)``.  ``inf`` when the ratio
    is exactly 1: no exponent rescues an exactly-constant weight.
    """
    if alpha <= 0.0:
        raise ValueError(
            "_alpha_for_two_fold: this extrapolation is undefined at alpha = 0, "
            "where every weight is 1 by construction. Run the diagnostic at the "
            "alpha the pipeline is configured with."
        )
    ratio_at_one = ratio_at_alpha ** (1.0 / alpha)
    if ratio_at_one <= 1.0:
        return math.inf
    return math.log(2.0) / math.log(ratio_at_one)


def report(summary: dict, alpha: float) -> None:
    ratios = summary["ratios"]
    cvs = summary["cvs"]
    threshold = summary["flat_ratio_threshold"]

    print()
    print(f"depth_weight_alpha used        : {alpha}")
    print(f"vertices contacted             : {summary['n_vertices']}")
    print(f"contact points (vertex-frames) : {summary['n_frames_total']}")
    print(f"  contacted in one frame only  : {summary['n_single_frame']} "
          f"(ratio is 1 by construction — excluded)")
    print(f"  with a zero (grazing) weight : {summary['n_zero_weight']} "
          f"(ratio infinite — excluded, counted here)")
    print(f"  eligible for the ratio       : {ratios.size}")

    if ratios.size == 0:
        print()
        print("VERDICT: no vertex was contacted in more than one frame with a "
              "non-zero weight. This session cannot answer the question; check "
              "that the touches and the sidecar depths loaded as expected.")
        return

    print()
    print("per-vertex weight variation  max(w)/min(w):")
    for q in (5, 25, 50, 75, 90, 99, 100):
        print(f"  p{q:<3d}                        : {np.percentile(ratios, q):.4f}")
    print(f"  mean                         : {ratios.mean():.4f}")

    flat = float(np.count_nonzero(ratios < threshold))
    exactly_flat = float(np.count_nonzero(ratios == 1.0))
    two_fold = float(np.count_nonzero(ratios >= 2.0))
    print(f"  exactly 1.0 (frozen)         : {exactly_flat:.0f} "
          f"({100.0 * exactly_flat / ratios.size:.2f}%)")
    print(f"  < {threshold} (effectively flat)     : {flat:.0f} "
          f"({100.0 * flat / ratios.size:.2f}%)")
    print(f"  >= 2.0 (materially varying)  : {two_fold:.0f} "
          f"({100.0 * two_fold / ratios.size:.2f}%)")

    print()
    print("per-vertex weight coefficient of variation  std(w)/mean(w):")
    for q in (50, 90, 100):
        print(f"  p{q:<3d}                        : {np.percentile(cvs, q):.4f}")

    median_ratio = float(np.median(ratios))
    needed = _alpha_for_two_fold(median_ratio, alpha)
    print()
    if not math.isfinite(needed):
        print("alpha needed for a 2x weight spread at the median vertex: infinite "
              "(that vertex's weight is exactly constant; no exponent moves it)")
    elif needed >= 1000.0:
        # Printed in scientific notation rather than rounded away: a ratio of
        # 1 + a few ULP is not exactly 1, so this comes out astronomically large
        # instead of infinite. That is the honest answer -- "no reachable alpha"
        # -- and hiding the difference between it and a genuine infinity would
        # hide the difference between a frozen weight and a nearly-frozen one.
        print(f"alpha needed for a 2x weight spread at the median vertex: "
              f"{needed:.3g}  (i.e. no reachable alpha)")
    else:
        print(f"alpha needed for a 2x weight spread at the median vertex: {needed:.2f}")

    print()
    flat_share = 100.0 * flat / ratios.size
    if flat_share >= 90.0:
        print("VERDICT: FALSIFIED ON THIS SESSION. Almost every vertex sees a")
        print("near-constant weight across the frames that touch it, so the")
        print("weight cancels between numerator and denominator and the map is")
        print("today's map. Depth weighting cannot help here. Do NOT raise alpha")
        print("to force a difference: that chooses the exponent to manufacture an")
        print("effect rather than to express a mechanism.")
    elif two_fold >= 0.10 * ratios.size:
        print("VERDICT: the premise survives. A substantial share of vertices are")
        print("pressed to materially different relative depths on different")
        print("frames, which is exactly the material the weighting redistributes.")
        print("This says the feature CAN act; whether it acts correctly is what")
        print("scripts/diagnose_depth_weight_delta_vs_distance.py tests.")
    else:
        print("VERDICT: marginal. The weighting has some material but not much.")
        print("Expect a small change to the maps. Judge it on the centre-vs-")
        print("periphery structure (diagnose_depth_weight_delta_vs_distance.py),")
        print("not on whether the numbers moved.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--merged-root", required=True, type=Path)
    parser.add_argument("--prepared-csv", required=True, type=Path)
    parser.add_argument("--depth-weight-alpha", type=float, default=1.0)
    parser.add_argument("--blocks-stage-dir", default="blocks_rf_centered")
    parser.add_argument(
        "--flat-ratio-threshold",
        type=float,
        default=1.05,
        help="max(w)/min(w) below which a vertex counts as effectively flat.",
    )
    args = parser.parse_args()

    if args.depth_weight_alpha <= 0.0:
        raise ValueError(
            f"diagnose_depth_weight_variation: --depth-weight-alpha must be > 0, got "
            f"{args.depth_weight_alpha}. At alpha = 0 every weight is exactly 1 by "
            f"construction, so every ratio is 1 and the diagnostic would report a "
            f"guaranteed answer rather than a measured one."
        )

    session_root = args.merged_root / args.session
    forearm_ply = resolve_forearm_ply(session_root, args.session)
    if forearm_ply is None:
        raise FileNotFoundError(
            f"diagnose_depth_weight_variation: forearm PLY not found in {session_root}"
        )

    playback = load_playback_data(
        args.prepared_csv,
        forearm_ply,
        depth_blocks_dir=session_root / args.blocks_stage_dir,
        session_id=args.session,
    )

    print(f"session                        : {args.session}")
    print("Depth-field provenance:")
    for pv in playback.depth_field_provenance:
        print(f"  {pv.source_block_file}  space={pv.coordinate_space}")

    per_vertex = collect_vertex_weights(playback, args.depth_weight_alpha)
    report(summarise(per_vertex, args.flat_ratio_threshold), args.depth_weight_alpha)


if __name__ == "__main__":
    main()
