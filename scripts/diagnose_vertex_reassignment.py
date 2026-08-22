"""Measure how far retiring the nearest-vertex KDTree moved the single-touch RF maps.

Phase 2.5 of ``docs/development/plans/active/depth-weighted-iff-attribution.md``
replaced nearest-vertex snapping with the ``vertex_id`` recorded by the
contact-depth-field parquet sidecars. That changes the maps deliberately, and the plan
requires the size of the change to be recorded rather than quietly absorbed.

The magnitude is a property of the real data, so it cannot be measured from synthetic
fixtures. This script measures it on one session and reports two things:

* how many contact points changed vertex identity at all, and
* the per-vertex ``|delta|`` distribution of the resulting mean RF map.

The retired nearest-vertex assignment is reconstructed **here, for measurement only**.
It is not a fallback and must never be reintroduced into the loader: the KDTree assigns
a different vertex from the one the sidecar recorded, and that discrepancy *is* the
measured 29 mm off-surface error at touch boundaries.

Usage
-----
    python scripts/diagnose_vertex_reassignment.py \\
        --session 2022-06-15_ST14-02 \\
        --merged-root  "<db>/3_merged" \\
        --prepared-csv "<db>/4_analysed/preparation/<session>_prepared.csv" \\
        [--blocks-stage-dir blocks_rf_centered] \\
        [--neuron-mode iff]

Every path is an argument. Nothing is discovered, defaulted or guessed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_data_loader import (  # noqa: E402
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    TouchEvent,
    load_playback_data,
)
from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (  # noqa: E402
    _compute_touch_rf,
)


def _kdtree_reassign(touch: TouchEvent, forearm_vertices: np.ndarray) -> TouchEvent:
    """Return *touch* with the RETIRED nearest-vertex assignment, for comparison only."""
    from scipy.spatial import cKDTree

    tree = cKDTree(forearm_vertices)
    frame_vertex_indices = []
    for pts in touch.frame_contact_pts:
        if len(pts) == 0:
            frame_vertex_indices.append(np.empty(0, dtype=np.int64))
            continue
        _, idx = tree.query(np.asarray(pts, dtype=np.float64))
        frame_vertex_indices.append(np.asarray(idx, dtype=np.int64))
    return TouchEvent(
        block_order_id=touch.block_order_id,
        trial_id=touch.trial_id,
        single_touch_id=touch.single_touch_id,
        gesture_type=touch.gesture_type,
        frame_contact_pts=touch.frame_contact_pts,
        frame_vertex_indices=frame_vertex_indices,
        # Depth is untouched: only vertex identity is being reassigned here.
        frame_depths=touch.frame_depths,
        frame_spikes=touch.frame_spikes,
        frame_iff=touch.frame_iff,
    )


def _dense(pairs, n_vertices: int) -> np.ndarray:
    dense = np.full(n_vertices, np.nan, dtype=np.float64)
    for vertex_idx, value in pairs:
        dense[vertex_idx] = value
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--merged-root", required=True, type=Path)
    parser.add_argument("--prepared-csv", required=True, type=Path)
    parser.add_argument("--blocks-stage-dir", default="blocks_rf_centered")
    parser.add_argument("--neuron-mode", default="iff", choices=("iff", "spike"))
    args = parser.parse_args()

    session_root = args.merged_root / args.session
    forearm_ply = resolve_forearm_ply(session_root, args.session)
    if forearm_ply is None:
        raise FileNotFoundError(
            f"diagnose_vertex_reassignment: forearm PLY not found in {session_root}"
        )

    playback = load_playback_data(
        args.prepared_csv,
        forearm_ply,
        depth_blocks_dir=session_root / args.blocks_stage_dir,
        session_id=args.session,
    )
    forearm_vertices = playback.session_data.forearm_vertices
    n_vertices = len(forearm_vertices)

    print("Depth-field provenance:")
    for pv in playback.depth_field_provenance:
        print(f"  {pv.source_block_file}  space={pv.coordinate_space}")

    n_points = 0
    n_points_changed = 0
    deltas: list[np.ndarray] = []
    n_touches = 0
    n_touches_changed = 0

    for block_id in playback.block_order_ids:
        for trial_id in playback.trial_ids_by_block[block_id]:
            for touch in playback.touches_by_block_trial[(block_id, trial_id)]:
                old_touch = _kdtree_reassign(touch, forearm_vertices)

                changed_here = 0
                for new_idx, old_idx in zip(
                    touch.frame_vertex_indices, old_touch.frame_vertex_indices
                ):
                    n_points += len(new_idx)
                    changed_here += int(np.count_nonzero(new_idx != old_idx))
                n_points_changed += changed_here

                # alpha = 0: this script measures the *vertex reassignment* and
                # nothing else, so the estimator must stay the unweighted one.
                # Any depth weighting here would mix two effects in one number.
                new_mean = _compute_touch_rf(
                    touch, n_vertices, args.neuron_mode, 0.0
                ).mean_pairs
                old_mean = _compute_touch_rf(
                    old_touch, n_vertices, args.neuron_mode, 0.0
                ).mean_pairs
                new_dense = _dense(new_mean, n_vertices)
                old_dense = _dense(old_mean, n_vertices)

                both = ~np.isnan(new_dense) & ~np.isnan(old_dense)
                only_new = ~np.isnan(new_dense) & np.isnan(old_dense)
                only_old = np.isnan(new_dense) & ~np.isnan(old_dense)
                delta = np.abs(new_dense[both] - old_dense[both])
                deltas.append(delta)

                n_touches += 1
                if changed_here or only_new.any() or only_old.any():
                    n_touches_changed += 1

    all_deltas = np.concatenate(deltas) if deltas else np.empty(0)
    print()
    print(f"session                        : {args.session}")
    print(f"touches                        : {n_touches}")
    print(f"touches whose map changed      : {n_touches_changed}")
    print(f"contact points                 : {n_points}")
    pct = 100.0 * n_points_changed / n_points if n_points else 0.0
    print(f"contact points reassigned      : {n_points_changed} ({pct:.2f}%)")
    if all_deltas.size:
        print(f"per-vertex |delta| (Hz) n      : {all_deltas.size}")
        print(f"  mean                         : {all_deltas.mean():.4f}")
        for q in (50, 90, 99, 100):
            print(f"  p{q:<3d}                        : {np.percentile(all_deltas, q):.4f}")
        nonzero = float(np.count_nonzero(all_deltas))
        print(f"  vertices with |delta| > 0    : {nonzero:.0f} "
              f"({100.0 * nonzero / all_deltas.size:.2f}%)")


if __name__ == "__main__":
    main()
