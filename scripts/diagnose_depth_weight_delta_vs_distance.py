"""Is the depth-weighted change *spatial*, or is it the 33x frame re-emission?

What this measures
------------------
Runs one session's single-touch RF mapping twice — at ``--baseline-alpha`` (0.0,
the unweighted baseline) and at ``--depth-weight-alpha`` (1.0, the shipped
setting) — aggregates each into a session-level mean map, and plots the
**per-vertex ``|delta|`` against that vertex's distance from the RF hotspot**.

The prediction, and what would falsify it
-----------------------------------------
Depth weighting redistributes credit *within* a contact patch.  A vertex at the
receptive-field centre is the deepest point of the frames that matter, so its
weight is near 1 whenever it is touched and the redistribution barely reaches it.
A vertex on the flank of a patch is the one that was being credited with its
neighbour's best moment, and it is where the correction lands.

    **Expected:** ``|delta|`` near zero at the centre, largest at the periphery.

    **Failure signature:** the whole map shifts by roughly the same amount at
    every distance.  A uniform shift is not a within-patch redistribution — it is
    the weights picking up the pre-existing implicit weighting (hazard 3 of the
    plan: contact geometry is forward-filled up to the nerve rate while IFF is
    never averaged down, so each Kinect frame is already credited about 33 times,
    weighted by how long it was held).  In that case the number that moved is a
    dwell-time artefact wearing a depth-weighting costume, and the work stops
    until that is understood.  Do not ship a map on the strength of "it changed".

Two numbers decide it, and both are printed:

* ``centre/periphery contrast`` — mean ``|delta|`` in the farthest distance bin
  divided by the mean in the nearest.  Much greater than 1 is the expected
  result.  Close to 1 is the failure signature.
* ``uniform fraction`` — ``|median(signed delta)| / mean(|signed delta|)``.  A
  pure common offset gives 1; a pure redistribution (as much up as down) gives
  about 0.  Near 1 is the failure signature, and it is the more decisive of the
  two because a uniform shift can hide behind a mild distance trend.

A rank correlation of ``|delta|`` against distance is printed beside them.  It is
the weakest of the three — it is significant for almost any monotone trend,
including a trivial one — so it is reported for completeness and should not be
the number anyone quotes.

Why the aggregation is unweighted across touches
------------------------------------------------
The session map is the plain mean of the per-touch maps (``compute_rf_heatmap``),
with every touch weighted 1.0.  Cross-touch weighting — a deeper touch outranking
a shallower one — is explicitly out of scope for this branch: the depth weighting
is a *within-frame* redistribution, and each touch's value has already been
reduced over its own frames before it gets here.  Using a different aggregation
in the diagnostic than in the product would measure something the pipeline does
not compute.

Scope
-----
A *diagnostic*, not a stage: it prints, and writes one PNG when ``--out-png`` is
given; it feeds nothing and no stage reads it.  The **verdict is decided by the
printed numbers**, not by the picture: the plot is there so the shape can be
seen, and the three statistics are there so the shape does not have to be
trusted to an eye.  ``--out-png`` is therefore optional rather than required --
an environment whose matplotlib cannot render (this repo has met one) still gets
the whole diagnosis.  It has never been run on real data by the author of this
branch (that work was forbidden to read the experimental database), so its
numbers are unknown and **it must be run, together with
``scripts/diagnose_depth_weight_variation.py``, before any depth-weighted result
from this branch is trusted.**  Run the variation script first: if the weights
are flat there, this plot will be blank and the reason will already be known.

Usage
-----
    python scripts/diagnose_depth_weight_delta_vs_distance.py \\
        --session 2022-06-15_ST14-02 \\
        --merged-root  "<db>/3_merged" \\
        --prepared-csv "<db>/4_analysed/preparation/<session>_prepared.csv" \\
        [--out-png     "reports/depth_weight_delta_vs_distance.png"] \\
        [--depth-weight-alpha 1.0] [--baseline-alpha 0.0] \\
        [--blocks-stage-dir blocks_rf_centered] \\
        [--neuron-mode iff] [--n-bins 10]

Every path is an argument.  Nothing is discovered, defaulted or guessed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_data_loader import (  # noqa: E402
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (  # noqa: E402
    compute_rf_heatmap,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    load_playback_data,
)
from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (  # noqa: E402
    _compute_touch_rf,
)


def session_mean_map(playback, n_vertices: int, neuron_mode: str, alpha: float):
    """Session-level mean RF map at *alpha*, via the shared cross-touch helper."""
    rf_vertex_indices: List[np.ndarray] = []
    rf_values: List[np.ndarray] = []
    for block_id in playback.block_order_ids:
        for trial_id in playback.trial_ids_by_block[block_id]:
            for touch in playback.touches_by_block_trial[(block_id, trial_id)]:
                pairs = _compute_touch_rf(
                    touch, n_vertices, neuron_mode, alpha
                ).mean_pairs
                if pairs:
                    rf_vertex_indices.append(
                        np.array([p[0] for p in pairs], dtype=np.int64)
                    )
                    rf_values.append(
                        np.array([p[1] for p in pairs], dtype=np.float64)
                    )
                else:
                    rf_vertex_indices.append(np.empty(0, dtype=np.int64))
                    rf_values.append(np.empty(0, dtype=np.float64))
    return compute_rf_heatmap(
        list(range(len(rf_values))), rf_vertex_indices, rf_values, n_vertices
    )


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, implemented on numpy alone.

    Ties get their average rank.  Written out rather than imported so this
    diagnostic depends on nothing beyond numpy for its statistics — the numbers
    are meant to be checkable by hand from the printed bin table.
    """
    if x.size < 2:
        raise ValueError(
            f"rank_correlation: need at least 2 points, got {x.size}."
        )

    def _ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
        # Average the ranks of tied runs.
        sorted_vals = values[order]
        start = 0
        for i in range(1, values.size + 1):
            if i == values.size or sorted_vals[i] != sorted_vals[start]:
                if i - start > 1:
                    ranks[order[start:i]] = ranks[order[start:i]].mean()
                start = i
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        raise ValueError(
            "rank_correlation: one of the inputs is constant, so the correlation "
            "is undefined. That is itself a finding — report it rather than "
            "substituting a zero."
        )
    return float((rx * ry).sum() / denom)


def bin_by_distance(
    distance: np.ndarray, delta: np.ndarray, n_bins: int
) -> List[Tuple[float, float, float, int]]:
    """Equal-count distance bins -> ``(mean distance, mean |delta|, median signed, n)``.

    Equal-count rather than equal-width: forearm vertices are not uniformly
    distributed in distance from any one point, and equal-width bins would put
    almost every vertex in the middle bins and leave the two bins the verdict
    depends on nearly empty.
    """
    if distance.size < n_bins:
        raise ValueError(
            f"bin_by_distance: {distance.size} vertices cannot fill {n_bins} bins."
        )
    order = np.argsort(distance, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = []
    for chunk in chunks:
        rows.append(
            (
                float(distance[chunk].mean()),
                float(np.abs(delta[chunk]).mean()),
                float(np.median(delta[chunk])),
                int(chunk.size),
            )
        )
    return rows


def plot(rows, distance, delta, hotspot_vertex, alpha, out_png: Path) -> None:
    """Scatter of ``|delta|`` vs distance with the binned means over it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.scatter(distance, np.abs(delta), s=4, alpha=0.25, color="0.6",
               label="per-vertex |delta|")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color="crimson",
            label="bin mean |delta|")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel(f"distance from RF hotspot (vertex {hotspot_vertex}), mesh units")
    ax.set_ylabel("|mean IFF at alpha=%g  -  mean IFF at baseline| (Hz)" % alpha)
    ax.set_title(
        "Expected: near zero at the centre, largest at the periphery.\n"
        "A flat line is the failure signature (uniform shift, not redistribution).",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--merged-root", required=True, type=Path)
    parser.add_argument("--prepared-csv", required=True, type=Path)
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Optional PNG of |delta| vs distance. The printed statistics are the "
             "verdict; this is the picture of it.",
    )
    parser.add_argument("--depth-weight-alpha", type=float, default=1.0)
    parser.add_argument("--baseline-alpha", type=float, default=0.0)
    parser.add_argument("--blocks-stage-dir", default="blocks_rf_centered")
    parser.add_argument("--neuron-mode", default="iff", choices=("iff", "spike"))
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    if args.depth_weight_alpha == args.baseline_alpha:
        raise ValueError(
            f"diagnose_depth_weight_delta_vs_distance: --depth-weight-alpha and "
            f"--baseline-alpha are both {args.baseline_alpha}. The two maps would be "
            f"identical and every delta would be exactly 0, which is a tautology "
            f"rather than a measurement."
        )

    session_root = args.merged_root / args.session
    forearm_ply = resolve_forearm_ply(session_root, args.session)
    if forearm_ply is None:
        raise FileNotFoundError(
            f"diagnose_depth_weight_delta_vs_distance: forearm PLY not found in "
            f"{session_root}"
        )

    playback = load_playback_data(
        args.prepared_csv,
        forearm_ply,
        depth_blocks_dir=session_root / args.blocks_stage_dir,
        session_id=args.session,
    )
    forearm_vertices = playback.session_data.forearm_vertices
    n_vertices = len(forearm_vertices)

    print(f"session                        : {args.session}")
    print("Depth-field provenance:")
    for pv in playback.depth_field_provenance:
        print(f"  {pv.source_block_file}  space={pv.coordinate_space}")

    baseline = session_mean_map(
        playback, n_vertices, args.neuron_mode, args.baseline_alpha
    )
    weighted = session_mean_map(
        playback, n_vertices, args.neuron_mode, args.depth_weight_alpha
    )

    both = ~np.isnan(baseline) & ~np.isnan(weighted)
    if not both.any():
        raise ValueError(
            "diagnose_depth_weight_delta_vs_distance: no vertex has a value in both "
            "maps. Nothing can be compared; check that the session loaded touches."
        )

    # The hotspot is read off the BASELINE map. Reading it off the weighted map
    # would make the plot self-fulfilling: the weighting would be measured
    # against the peak it had just moved.
    hotspot = int(np.nanargmax(baseline))
    distance_all = np.linalg.norm(
        forearm_vertices - forearm_vertices[hotspot], axis=1
    )

    idx = np.flatnonzero(both)
    distance = distance_all[idx]
    delta = weighted[idx] - baseline[idx]

    rows = bin_by_distance(distance, delta, args.n_bins)

    print()
    print(f"neuron_mode                    : {args.neuron_mode}")
    print(f"baseline alpha / test alpha    : {args.baseline_alpha} / "
          f"{args.depth_weight_alpha}")
    print(f"vertices in both maps          : {idx.size}")
    print(f"RF hotspot vertex (baseline)   : {hotspot} "
          f"(mean IFF {baseline[hotspot]:.4f} Hz)")
    print(f"mean |delta|                   : {np.abs(delta).mean():.6f} Hz")
    print(f"max  |delta|                   : {np.abs(delta).max():.6f} Hz")
    print()
    print("distance bin (equal count):")
    print(f"  {'mean dist':>12}  {'mean |delta|':>13}  {'median delta':>13}  {'n':>6}")
    for mean_d, mean_abs, median_signed, n in rows:
        print(f"  {mean_d:12.4f}  {mean_abs:13.6f}  {median_signed:13.6f}  {n:6d}")

    inner, outer = rows[0][1], rows[-1][1]
    contrast = outer / inner if inner > 0 else np.inf
    mean_abs_delta = float(np.abs(delta).mean())
    uniform_fraction = (
        abs(float(np.median(delta))) / mean_abs_delta if mean_abs_delta > 0 else 0.0
    )
    rho = rank_correlation(distance, np.abs(delta))

    print()
    print(f"centre/periphery contrast      : {contrast:.3f}  "
          f"(outer bin mean |delta| / inner bin mean |delta|)")
    print(f"uniform fraction               : {uniform_fraction:.3f}  "
          f"(|median signed delta| / mean |delta|; 1 = pure common offset)")
    print(f"rank corr(|delta|, distance)   : {rho:.3f}  (weakest of the three)")

    if args.out_png is None:
        print("plot                           : not requested (--out-png omitted)")
    else:
        plot(rows, distance, delta, hotspot, args.depth_weight_alpha, args.out_png)
        print(f"plot written                   : {args.out_png}")

    print()
    if mean_abs_delta == 0.0:
        print("VERDICT: the two maps are identical. Depth weighting did nothing on")
        print("this session. Run scripts/diagnose_depth_weight_variation.py — the")
        print("weights are almost certainly flat, and that is the real finding.")
    elif uniform_fraction >= 0.5:
        print("VERDICT: FAILURE SIGNATURE. Most of the change is a common offset")
        print("applied to the whole map, not a redistribution of credit inside")
        print("contact patches. That is what hazard 3 predicts: the weights are")
        print("multiplying on top of the ~33x forward-filled frame re-emission and")
        print("picking up dwell time, not spatial structure. STOP and investigate")
        print("before trusting any weighted output.")
    elif contrast >= 2.0:
        print("VERDICT: as predicted. The change is concentrated on the periphery")
        print("of the receptive field and near zero at its centre, which is what a")
        print("within-patch redistribution of credit looks like.")
    else:
        print("VERDICT: inconclusive. The change is not a uniform offset, but it is")
        print("not clearly centre-vs-periphery either. Look at the plot before")
        print("drawing any conclusion, and treat the rank correlation as the")
        print("weakest evidence in this report, not the strongest.")


if __name__ == "__main__":
    main()
