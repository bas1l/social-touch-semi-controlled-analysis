"""Synthetic fixtures for the shared per-vertex accumulator.

These fixtures are deliberately in a module of their own rather than inside a
single test file, because they serve **two** purposes that must not drift
apart:

1. They pin the pre-refactor behaviour of the nine duplicated accumulator
   blocks (``tests/test_vertex_accumulator.py``).
2. The outputs pinned on them *are* the ``depth_weight_alpha = 0`` baseline
   that the depth-weighting parity test compares against. A weighted run at
   ``alpha = 0`` must reproduce these arrays byte for byte.

Baseline status after Phase 2.5 (re-pin)
----------------------------------------
Phase 2.5 replaced nearest-vertex snapping with the ``vertex_id`` recorded by
the contact-depth-field sidecar, which moves credit between vertices on purpose.
These fixtures are **numerically unchanged** by that, and that is not an
oversight — they hand ``frame_vertex_indices`` to the reduction directly, so
they pin the *reduction*, never the vertex source. What changed is what those
indices now mean: every one of them is a sidecar ``vertex_id``, the value the
producing pipeline measured, not an answer re-derived here from coordinates.

Consequently the ``alpha = 0`` parity test (Phase 6.1) compares against these
same arrays, and its claim is exactly one thing: the weighting machinery is an
exact no-op at ``alpha = 0``. It says nothing about agreement with maps produced
before Phase 2.5 — those were built on a different, wrong vertex assignment, and
the size of that difference is a property of the real data, measured by
``scripts/diagnose_vertex_reassignment.py``.

Nothing here reads the experimental database. Every array is constructed in
code from literals or a seeded generator.
"""

import numpy as np

from analysis.receptive_field_mapping.data.touch_playback_data import TouchEvent

__all__ = [
    "WORKED_EXAMPLE_N_VERTICES",
    "WORKED_EXAMPLE_DEPTHS_MM",
    "WORKED_EXAMPLE_SIGNED_DEPTHS_MM",
    "NAN_AND_DUPLICATE_SIGNED_DEPTHS_MM",
    "worked_example_touch",
    "nan_and_duplicate_touch",
    "random_contact_points",
    "random_rf_touch_maps",
]


# ----------------------------------------------------------------------
# Fixture A — the plan's worked example
# ----------------------------------------------------------------------
#
# Seven vertices in a row (v1..v7 == indices 0..6), a finger sweeping left to
# right across three frames, the neuron firing at 50 / 100 / 40 Hz.
#
#   frame 0 (50 Hz) touches v1 v2 v3 v6 v7
#   frame 1 (100 Hz) touches    v2 v3 v4 v6 v7
#   frame 2 (40 Hz) touches        v3 v4 v5 v6 v7
#
# Unweighted per-vertex means (today's behaviour, and the alpha=0 baseline):
#
#   v1 50.0 | v2 75.0 (peak) | v3 63.33 | v4 70.0 | v5 40.0 | v6 63.33 | v7 63.33
#
# v6 is a permanently shallow vertex (stroke rim) and v7 a permanently deep one
# (stroke midline); both are included so the weighting phase can show that a
# roughly-constant weight cancels between numerator and denominator.
WORKED_EXAMPLE_N_VERTICES = 7

_WORKED_EXAMPLE_FRAME_VERTICES = [
    np.array([0, 1, 2, 5, 6], dtype=np.int64),
    np.array([1, 2, 3, 5, 6], dtype=np.int64),
    np.array([2, 3, 4, 5, 6], dtype=np.int64),
]

_WORKED_EXAMPLE_IFF = np.array([50.0, 100.0, 40.0], dtype=np.float64)
_WORKED_EXAMPLE_SPIKES = np.array([True, False, True], dtype=bool)

# Penetration depth in millimetres, aligned element-for-element with
# ``_WORKED_EXAMPLE_FRAME_VERTICES``. Each frame has a different maximum depth
# (3.0, 2.0, 4.0 mm) so that a max-normalised weight is not accidentally equal
# to the raw depth, which would hide a missing normalisation.
#
# The resulting alpha=1 weights (depth / frame max depth) are:
#   frame 0: v1 0.30  v2 1.00  v3 0.42  v6 0.13  v7 0.92
#   frame 1: v2 0.47  v3 1.00  v4 0.40  v6 0.13  v7 0.93
#   frame 2: v3 0.50  v4 1.00  v5 0.30  v6 0.10  v7 0.95
WORKED_EXAMPLE_DEPTHS_MM = [
    np.array([0.90, 3.00, 1.26, 0.39, 2.76], dtype=np.float64),
    np.array([0.94, 2.00, 0.80, 0.26, 1.86], dtype=np.float64),
    np.array([2.00, 4.00, 1.20, 0.40, 3.80], dtype=np.float64),
]

# ``WORKED_EXAMPLE_DEPTHS_MM`` above is **penetration** magnitude — positive,
# the quantity the weight function consumes. ``TouchEvent.frame_depths`` carries
# the sidecar column as stored, which is *signed* with negative = penetrating,
# so the touch fixture negates once. Keeping both forms visible here is
# deliberate: it is the sign flip that a depth-weighting bug is most likely to
# get wrong, and ``penetration_mm`` is the single place production code flips it.
WORKED_EXAMPLE_SIGNED_DEPTHS_MM = [-d for d in WORKED_EXAMPLE_DEPTHS_MM]


def _contact_pts_for(frame_vertices):
    """Plausible (K_i, 3) contact coordinates; only the GUI's left view reads them."""
    return [
        np.stack(
            [verts.astype(np.float64), np.zeros(len(verts)), np.ones(len(verts))],
            axis=1,
        )
        for verts in frame_vertices
    ]


def worked_example_touch() -> TouchEvent:
    """The 7-vertex / 3-frame touch described at the top of this module."""
    frame_vertices = [v.copy() for v in _WORKED_EXAMPLE_FRAME_VERTICES]
    return TouchEvent(
        block_order_id="1",
        trial_id=1,
        single_touch_id=1,
        gesture_type="stroke_proximal",
        frame_contact_pts=_contact_pts_for(frame_vertices),
        frame_vertex_indices=frame_vertices,
        frame_depths=[d.copy() for d in WORKED_EXAMPLE_SIGNED_DEPTHS_MM],
        frame_spikes=_WORKED_EXAMPLE_SPIKES.copy(),
        frame_iff=_WORKED_EXAMPLE_IFF.copy(),
    )


# ----------------------------------------------------------------------
# Fixture B — the degenerate shapes the real data actually contains
# ----------------------------------------------------------------------

def nan_and_duplicate_touch() -> TouchEvent:
    """A touch exercising the awkward cases the nine blocks must keep handling.

    * frame 0 — an **empty** frame (no contact points at all)
    * frame 1 — a frame whose vertex list contains a **duplicate** index, so the
      reduction adds the same value to the same vertex twice in sequence; this
      is where a naive "reduce then add" rewrite stops being bit-identical
    * frame 2 — a **NaN** IFF (unit not held during the touch window), which
      must poison the mean of every vertex it touches
    * frame 3 — an ordinary frame, so at least one vertex has a NaN-free mean
    * frame 4 — values with a huge magnitude spread against a tiny one, so that
      any change in summation order shows up in the last mantissa bits
    """
    frame_vertices = [
        np.array([], dtype=np.int64),
        np.array([1, 3, 3, 4], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([3, 4, 5], dtype=np.int64),
        np.array([5, 5, 5, 6, 6], dtype=np.int64),
    ]
    return TouchEvent(
        block_order_id="2",
        trial_id=3,
        single_touch_id=4,
        gesture_type="tap",
        frame_contact_pts=_contact_pts_for(frame_vertices),
        frame_vertex_indices=frame_vertices,
        frame_depths=[d.copy() for d in NAN_AND_DUPLICATE_SIGNED_DEPTHS_MM],
        frame_spikes=np.array([False, True, True, False, True], dtype=bool),
        frame_iff=np.array([12.5, 1e8, np.nan, 37.25, 1e-8], dtype=np.float64),
    )


NAN_AND_DUPLICATE_N_VERTICES = 8

# Signed depths (negative = penetrating) aligned element-for-element with each
# frame's vertex list above, including the empty frame and the duplicated
# vertex. The duplicate is kept: the reduction must go on handling it
# bit-identically. The *loader* refuses to produce one (a duplicate
# ``(frame_index, vertex_id)`` raises there), which is a statement about what
# may enter the pipeline, not about what this reduction must survive.
NAN_AND_DUPLICATE_SIGNED_DEPTHS_MM = [
    np.array([], dtype=np.float64),
    np.array([-1.5, -2.25, -0.75, -3.0], dtype=np.float64),
    np.array([-0.5, -1.25, -2.0], dtype=np.float64),
    np.array([-4.0, -0.125, -1.0], dtype=np.float64),
    np.array([-2.5, -2.5, -0.25, -3.75, -1.0], dtype=np.float64),
]


# ----------------------------------------------------------------------
# Fixture C — flat contact-point arrays, as the population / feature-space
# viewers hold them
# ----------------------------------------------------------------------

def random_contact_points(seed: int = 20260821, n_points: int = 20_000,
                          n_verts: int = 733) -> dict:
    """Flat contact-point arrays with heavy vertex repetition.

    ``n_points >> n_verts`` on purpose: every vertex is hit tens of times, so
    each per-vertex total is a long chain of float additions and a reduction
    that blocked its summation differently would disagree in the last bits.
    IFF values span eight orders of magnitude for the same reason.
    """
    rng = np.random.default_rng(seed)
    cp_vertex_idx = rng.integers(0, n_verts, size=n_points).astype(np.int64)
    exponents = rng.integers(-4, 5, size=n_points)
    cp_iff = rng.random(n_points) * (10.0 ** exponents)
    cp_spike = rng.random(n_points) < 0.15
    cp_frame_idx = rng.integers(0, 512, size=n_points).astype(np.int64)
    cp_mask = rng.random(n_points) < 0.6
    return {
        "n_verts": n_verts,
        "cp_vertex_idx": cp_vertex_idx,
        "cp_iff": cp_iff,
        "cp_spike": cp_spike,
        "cp_frame_idx": cp_frame_idx,
        "cp_mask": cp_mask,
    }


def random_rf_touch_maps(seed: int = 4711, n_touches: int = 40,
                         n_verts: int = 733) -> dict:
    """Per-touch (vertex_indices, values) RF maps, as the population heatmap reads them.

    One touch is deliberately empty and one carries NaN values, because both
    occur in the saved ``.npz`` maps and both change what the mean must be.
    """
    rng = np.random.default_rng(seed)
    rf_vertex_indices = []
    rf_values = []
    for t in range(n_touches):
        if t == 3:
            rf_vertex_indices.append(np.array([], dtype=np.int64))
            rf_values.append(np.array([], dtype=np.float64))
            continue
        k = int(rng.integers(1, 60))
        verts = rng.integers(0, n_verts, size=k).astype(np.int64)
        vals = rng.random(k) * (10.0 ** rng.integers(-4, 5, size=k))
        if t == 7:
            vals[0] = np.nan
        rf_vertex_indices.append(verts)
        rf_values.append(vals)
    return {
        "n_verts": n_verts,
        "touch_indices": list(range(n_touches)),
        "rf_vertex_indices": rf_vertex_indices,
        "rf_values": rf_values,
    }
