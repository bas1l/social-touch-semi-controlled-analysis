"""Shared UV stroke-axis math for the ``spatial_configure_stroke_axis`` node.

Single source of truth used by **both** the manual GUI preview and the
``spatial_build_response_fields`` consumer, so the two can never drift (the same
pattern as the shared param-free field math in ``rf_boundary_preparation``).

The core idea is the 2D-UV analogue of the upstream 3D rule
``classify_gesture_type`` (``touch_analytics/preparation/gesture_type.py``):

* The 3D rule fits a degree-1 polynomial to ``contact_location_x`` over frame
  index and calls ``slope > 0`` -> ``'stroke_proximal'`` else ``'stroke_distal'``;
  it returns ``'stroke_unknown'`` when fewer than two valid (non-NaN) frames
  exist.
* Here we first project each frame's ``contact_location_{x,y,z}`` into SLIM-UV
  space via :func:`barycentric_uv_lookup`, then fit a degree-1 polynomial to the
  ``u`` and ``v`` components independently over frame index. The per-stroke UV
  motion vector is the net fitted displacement ``end_uv - start_uv``. A stroke
  is projected onto a researcher-drawn axis (:class:`StrokeAxisConfig`) instead
  of the fixed +x axis, and ``proj > 0`` -> ``'stroke_proximal'`` else
  ``'stroke_distal'`` — preserving the 3D rule's sign convention and its
  ``'stroke_unknown'`` short (< 2 frame) / all-NaN handling.

The module is deliberately headless (no Qt / matplotlib imports) so it is
unit-testable and importable from the pipeline consumer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import TOUCH_ID_COLS
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    StrokeAxisConfig,
    StrokeUvMotion,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    SlimUvCache,
    barycentric_uv_lookup,
    load_slim_uv_cache,
)

logger = logging.getLogger(__name__)

# Per-frame 3D contact position columns — the same triple the 3D rule requires.
_CONTACT_LOCATION_COLS = (
    "contact_location_x",
    "contact_location_y",
    "contact_location_z",
)
_STROKE_LABELS = ("stroke_proximal", "stroke_distal")

# Re-exported so callers can grab the cache loader from one module.
__all__ = [
    "SlimUvCache",
    "load_slim_uv_cache",
    "compute_stroke_uv_motion",
    "initialize_stroke_axis",
    "relabel_strokes",
]


def compute_stroke_uv_motion(
    slim_cache: SlimUvCache,
    prepared_df: pd.DataFrame,
) -> StrokeUvMotion:
    """Derive per-stroke SLIM-UV motion from a session's prepared CSV.

    For every single-touch ``type_metadata == 'stroke'`` group (``single_touch_id
    != 0``), the per-frame ``contact_location_{x,y,z}`` positions are projected
    into UV via :func:`barycentric_uv_lookup`, then a degree-1 polynomial is fit
    to the ``u`` and ``v`` components independently over frame index (``0 .. n-1``,
    matching the 3D rule's use of ``np.arange(len(valid))``). The stroke's motion
    vector is the net fitted displacement ``end_uv - start_uv``.

    Parity with :func:`classify_gesture_type` (3D): a stroke with fewer than two
    valid (all of x/y/z non-NaN) contact frames is flagged
    ``valid_mask == False`` — the UV analogue of ``'stroke_unknown'`` — and its
    UV entries are NaN. Frame order is the CSV group order (not re-sorted), again
    mirroring the 3D rule.

    Parameters
    ----------
    slim_cache:
        Loaded per-session :class:`SlimUvCache` (from :func:`load_slim_uv_cache`).
    prepared_df:
        The session ``_prepared.csv`` as a DataFrame. Must carry
        ``block_order_id`` / ``trial_id`` / ``single_touch_id`` /
        ``contact_location_{x,y,z}`` / ``gesture_type`` / ``type_metadata``.

    Returns
    -------
    StrokeUvMotion
        One entry per stroke, in CSV group order.

    Raises
    ------
    ValueError
        If any required column is absent (fail-fast — the aggregated
        ``_series_augmented.csv`` drops the per-frame contact columns; read the
        ``_prepared.csv``).
    """
    required = (
        list(TOUCH_ID_COLS)
        + list(_CONTACT_LOCATION_COLS)
        + ["gesture_type", "type_metadata"]
    )
    missing = [c for c in required if c not in prepared_df.columns]
    if missing:
        raise ValueError(
            f"compute_stroke_uv_motion: prepared DataFrame is missing required "
            f"column(s) {missing}. Expected the per-frame '_prepared.csv' "
            f"(not the aggregated '_series_augmented.csv')."
        )

    touch_df = prepared_df[prepared_df["single_touch_id"] != 0]
    stroke_df = touch_df[touch_df["type_metadata"] == "stroke"]

    keys: list[tuple[int, int, int]] = []
    labels: list[str] = []
    # Per-stroke validity (>= 2 non-NaN contact frames), aligned to CSV group order.
    valids: list[bool] = []
    # For the valid strokes only, the per-stroke frame count and their contact xyz,
    # collected so ALL valid frames can be UV-projected in ONE barycentric lookup
    # (which rebuilds a KDTree over every face centroid per call): one tree build
    # per session instead of one per stroke.
    valid_frame_counts: list[int] = []
    valid_xyz_blocks: list[np.ndarray] = []

    for key, group in stroke_df.groupby(list(TOUCH_ID_COLS), sort=False):
        keys.append(tuple(int(k) for k in key))
        labels.append(str(group["gesture_type"].iloc[0]))

        valid_group = group.dropna(subset=list(_CONTACT_LOCATION_COLS))
        n = len(valid_group)
        if n < 2:
            # < 2 valid frames: the UV analogue of 'stroke_unknown' (the 3D rule
            # needs >= 2 points for a degree-1 fit). Warn and mark invalid.
            logger.warning(
                "compute_stroke_uv_motion: stroke (block=%s, trial=%s, touch=%s) has "
                "only %d valid contact frame(s) — flagged invalid (UV 'stroke_unknown').",
                key[0], key[1], key[2], n,
            )
            valids.append(False)
            continue

        valid_xyz_blocks.append(
            valid_group[list(_CONTACT_LOCATION_COLS)].to_numpy(dtype=np.float64)
        )
        valid_frame_counts.append(n)
        valids.append(True)

    # Single batched UV projection: concatenate every valid stroke's (n_i, 3) xyz
    # into one array, project once, then split the (sum n_i, 2) result back per
    # stroke. barycentric_uv_lookup is row-independent (per-point nearest-face
    # query + vectorized barycentric interp), so the split rows are byte-for-byte
    # identical to projecting each stroke separately — this is a pure speedup.
    if valid_xyz_blocks:
        all_xyz = np.concatenate(valid_xyz_blocks, axis=0)
        all_uv = barycentric_uv_lookup(slim_cache, all_xyz)  # (sum n_i, 2)
    else:
        all_uv = np.empty((0, 2), dtype=np.float64)
    offsets = np.cumsum([0, *valid_frame_counts])

    starts: list[list[float]] = []
    ends: list[list[float]] = []
    vectors: list[list[float]] = []
    centroids: list[list[float]] = []

    _nan_uv = [float("nan"), float("nan")]

    vi = 0  # running index into the valid-stroke UV blocks
    for is_valid in valids:
        if not is_valid:
            starts.append(list(_nan_uv))
            ends.append(list(_nan_uv))
            vectors.append(list(_nan_uv))
            centroids.append(list(_nan_uv))
            continue

        n = valid_frame_counts[vi]
        uv = all_uv[offsets[vi]:offsets[vi + 1]]  # (n, 2)
        frames = np.arange(n, dtype=np.float64)

        slope_u, intercept_u = np.polyfit(frames, uv[:, 0], 1)
        slope_v, intercept_v = np.polyfit(frames, uv[:, 1], 1)

        last = float(n - 1)
        start = [float(intercept_u), float(intercept_v)]
        end = [
            float(intercept_u + slope_u * last),
            float(intercept_v + slope_v * last),
        ]
        starts.append(start)
        ends.append(end)
        vectors.append([end[0] - start[0], end[1] - start[1]])
        centroids.append([float(uv[:, 0].mean()), float(uv[:, 1].mean())])
        vi += 1

    n_strokes = len(keys)
    return StrokeUvMotion(
        touch_keys=np.array(keys, dtype=np.int64).reshape(n_strokes, 3),
        current_labels=np.array(labels, dtype=object),
        start_uv=np.array(starts, dtype=np.float64).reshape(n_strokes, 2),
        end_uv=np.array(ends, dtype=np.float64).reshape(n_strokes, 2),
        vectors=np.array(vectors, dtype=np.float64).reshape(n_strokes, 2),
        valid_mask=np.array(valids, dtype=bool),
        centroid_uv=np.array(centroids, dtype=np.float64).reshape(n_strokes, 2),
    )


def initialize_stroke_axis(motion: StrokeUvMotion) -> np.ndarray:
    """Auto-initialise the stroke-axis direction from the current labels.

    Sign-aligns every valid stroke's UV motion vector by its *current* label —
    ``'stroke_distal'`` vectors are negated, ``'stroke_proximal'`` kept — so all
    aligned vectors point roughly toward proximal, then returns the normalised
    mean. The result is therefore a proximal-pointing unit vector for which
    ``relabel_strokes`` reproduces the current labels (a no-op axis), which the
    GUI offers as the editable starting point.

    Raises
    ------
    ValueError
        If no valid stroke carries a proximal/distal label (nothing to average),
        or if the sign-aligned mean is degenerate (zero length) — no silent
        default axis.
    """
    align_mask = motion.valid_mask & np.isin(
        motion.current_labels, np.array(_STROKE_LABELS, dtype=object)
    )
    if not np.any(align_mask):
        raise ValueError(
            "initialize_stroke_axis: no valid stroke with a proximal/distal label "
            "to average — cannot auto-initialise the axis. Draw it manually."
        )

    vecs = motion.vectors[align_mask].astype(np.float64)
    labels = motion.current_labels[align_mask]
    signs = np.where(labels == "stroke_distal", -1.0, 1.0)
    aligned = vecs * signs[:, np.newaxis]

    mean_vec = aligned.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm == 0.0:
        raise ValueError(
            "initialize_stroke_axis: the sign-aligned mean UV motion vector is "
            "zero-length (proximal and distal strokes cancel exactly) — cannot "
            "auto-initialise the axis. Draw it manually."
        )
    return mean_vec / norm


def relabel_strokes(
    gesture_types: np.ndarray,
    touch_triple_keys: np.ndarray,
    motion: StrokeUvMotion,
    config: StrokeAxisConfig,
) -> np.ndarray:
    """Reassign proximal/distal labels by projecting UV motion onto the axis.

    Returns a **copy** of ``gesture_types`` in which every entry currently
    labelled ``'stroke_proximal'`` or ``'stroke_distal'`` is re-decided from its
    UV motion vector: ``proj = vector . config.signed_direction()``, then
    ``proj > 0`` -> ``'stroke_proximal'`` else ``'stroke_distal'`` (the 2D-UV
    analogue of the 3D ``slope > 0`` rule). Taps, ``'stroke_unknown'``, ``None``
    and every non-stroke entry are left untouched, so an auto-initialised axis
    (no swap) is a no-op and toggling ``swap_proximal_distal`` exactly exchanges
    proximal <-> distal.

    Population strokes are matched to their UV motion by the
    ``(block_order_id, trial_id, single_touch_id)`` triple.

    Parameters
    ----------
    gesture_types:
        (T,) object array of per-touch labels (``pop_data.gesture_types``).
    touch_triple_keys:
        (T, 3) int64 array aligned to ``gesture_types``
        (``pop_data.touch_triple_keys``).
    motion:
        Per-stroke UV motion from :func:`compute_stroke_uv_motion`.
    config:
        The researcher's :class:`StrokeAxisConfig`.

    Raises
    ------
    ValueError
        If ``gesture_types`` and ``touch_triple_keys`` disagree in length, or a
        population proximal/distal stroke has no valid matching UV motion (a
        genuine inconsistency between the population NPZ and the prepared CSV) —
        no silent skip.
    """
    gesture_types = np.asarray(gesture_types, dtype=object)
    touch_triple_keys = np.asarray(touch_triple_keys)
    if len(gesture_types) != len(touch_triple_keys):
        raise ValueError(
            f"relabel_strokes: gesture_types ({len(gesture_types)}) and "
            f"touch_triple_keys ({len(touch_triple_keys)}) length mismatch."
        )

    signed = config.signed_direction()

    # Map stroke triple -> its motion row index (only valid strokes carry a vector).
    motion_index: dict[tuple[int, int, int], int] = {
        tuple(int(x) for x in motion.touch_keys[s]): s
        for s in range(len(motion.touch_keys))
    }

    out = gesture_types.copy()
    for i, label in enumerate(out):
        if label not in _STROKE_LABELS:
            continue  # taps, stroke_unknown, None — untouched
        key = tuple(int(x) for x in touch_triple_keys[i])
        s = motion_index.get(key)
        if s is None or not bool(motion.valid_mask[s]):
            raise ValueError(
                f"relabel_strokes: population stroke {key} labelled {label!r} has "
                f"no valid UV motion vector — the population NPZ and the prepared "
                f"CSV are inconsistent (re-run the upstream stages)."
            )
        proj = float(np.dot(motion.vectors[s], signed))
        out[i] = "stroke_proximal" if proj > 0 else "stroke_distal"
    return out
