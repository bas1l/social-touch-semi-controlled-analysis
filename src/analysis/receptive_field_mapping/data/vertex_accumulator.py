"""The one per-vertex weighted reduction used by receptive-field mapping.

Every place in this codebase that accumulates a per-vertex sum/mean of a
neuron value over contact points routes through this module.  Before it
existed the same three-line ``np.add.at`` block was copy-pasted in nine
places (the single-touch pipeline, the population heatmap, and seven blocks
across three GUI viewer windows), so a change to one of them silently made
the viewers disagree with the maps written to disk.

Contract
--------
This module knows about **vertices, values and weights** and nothing else.
It must never learn about penetration depth, ``alpha``, millimetres, parquet
sidecars, Qt, or files.  Turning a physical quantity into a weight is the job
of the weight function; turning these sums into a firing rate is the job of
the estimator that divides ``value_sum`` by ``weight_sum``.

Semantics
---------
For a batch of contact points ``i`` with vertex ``v_i``, value ``x_i`` and
weight ``w_i``::

    value_sum[v]     = sum_{i: v_i = v} w_i * x_i
    value_max[v]     = max_{i: v_i = v} x_i        # UNWEIGHTED, see below
    weight_sum[v]    = sum_{i: v_i = v} w_i
    weight_sq_sum[v] = sum_{i: v_i = v} w_i**2

``value_max`` is deliberately **unweighted**.  A weighted maximum has no
meaning: scaling a sample by ``w`` does not make it "less of a maximum", it
makes it a different number in different units.  The max map answers "what is
the strongest response ever seen at this vertex", which is a statement about
the observed values themselves, so weights play no part in it.

``weight_sq_sum`` is accumulated even though nothing reads it yet.  It is the
denominator of the Kish effective sample size ``n_eff = (sum w)^2 / sum(w^2)``
that the confidence channel will report; accumulating it here keeps the
reduction single-pass and keeps callers from having to re-walk the points.

Weight validity — non-negative, non-NaN, non-degenerate — is **not** checked
here.  That is a property of how the weights were derived, so it is enforced
where they are built and where they are divided by, not in a reduction that
has no way to describe what went wrong.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AccumResult",
    "empty_accumulator",
    "accumulate_vertex_values",
    "accumulate_vertex_values_into",
]


@dataclass(frozen=True)
class AccumResult:
    """Per-vertex reduction totals.  All four arrays have shape ``(n_vertices,)``.

    The dataclass is frozen so the *set* of arrays cannot be swapped out, but
    the arrays themselves are mutable on purpose: incremental callers (GUI
    playback, which adds one frame per rendered frame) hold an ``AccumResult``
    as running state and grow it in place via
    :func:`accumulate_vertex_values_into`.
    """

    value_sum: np.ndarray
    value_max: np.ndarray
    weight_sum: np.ndarray
    weight_sq_sum: np.ndarray


def empty_accumulator(n_vertices: int) -> AccumResult:
    """Allocate a zeroed accumulator for a mesh of ``n_vertices`` vertices.

    ``value_max`` starts at ``-inf`` so that the first observed value at a
    vertex wins; untouched vertices stay at ``-inf`` and are identified by
    ``weight_sum == 0``, never by inspecting ``value_max``.
    """
    if not isinstance(n_vertices, (int, np.integer)) or isinstance(n_vertices, bool):
        raise TypeError(
            f"empty_accumulator: n_vertices must be an integer, got "
            f"{type(n_vertices).__name__}."
        )
    if n_vertices <= 0:
        raise ValueError(
            f"empty_accumulator: n_vertices must be positive, got {n_vertices}. "
            f"A mesh with no vertices cannot carry a receptive field."
        )
    n = int(n_vertices)
    return AccumResult(
        value_sum=np.zeros(n, dtype=np.float64),
        value_max=np.full(n, -np.inf, dtype=np.float64),
        weight_sum=np.zeros(n, dtype=np.float64),
        weight_sq_sum=np.zeros(n, dtype=np.float64),
    )


def accumulate_vertex_values_into(
    result: AccumResult,
    vertex_idx: np.ndarray,
    values,
    weights: np.ndarray,
) -> None:
    """Add one batch of contact points into an existing accumulator, in place.

    Parameters
    ----------
    result:
        Accumulator to grow.  Its four arrays are modified in place.
    vertex_idx:
        ``(K,)`` integer array; the vertex each contact point belongs to.
        Repeats are allowed and are accumulated one at a time, in array order.
    values:
        Either a scalar or a ``(K,)`` array.  A scalar is the normal case for
        a neuron value, because a frame has exactly one instantaneous firing
        frequency however many vertices it touched; it is broadcast across the
        batch.
    weights:
        ``(K,)`` array.  **Required, and never scalar** — a per-point weight is
        a per-point fact, and allowing a scalar here would make ``1.0`` an
        idiomatic shorthand for "unweighted", which is precisely the implicit
        default this module exists to eliminate.  Pass ``np.ones(K)`` when the
        points genuinely carry equal weight.

    Notes
    -----
    The reduction is ``np.add.at``/``np.maximum.at``: unbuffered, sequential,
    in array order.  That is what makes the accumulation associative *in the
    same way* as the nine hand-written blocks this replaces, so a run with
    ``weights = ones`` is bit-identical to them rather than merely close.
    """
    if not isinstance(result, AccumResult):
        raise TypeError(
            f"accumulate_vertex_values_into: result must be an AccumResult, got "
            f"{type(result).__name__}."
        )

    idx = np.asarray(vertex_idx)
    if idx.ndim != 1:
        raise ValueError(
            f"accumulate_vertex_values_into: vertex_idx must be 1-D, got shape "
            f"{idx.shape}."
        )
    if idx.dtype.kind not in ("i", "u"):
        raise TypeError(
            f"accumulate_vertex_values_into: vertex_idx must have an integer "
            f"dtype, got {idx.dtype}. Float indices would be silently truncated."
        )

    n_vertices = result.value_sum.shape[0]
    k = idx.shape[0]
    if k > 0:
        lo = int(idx.min())
        hi = int(idx.max())
        if lo < 0 or hi >= n_vertices:
            raise IndexError(
                f"accumulate_vertex_values_into: vertex_idx range [{lo}, {hi}] "
                f"falls outside [0, {n_vertices}). Negative indices would wrap "
                f"around and corrupt the far end of the mesh silently."
            )

    val = np.asarray(values, dtype=np.float64)
    if val.ndim == 0:
        val = np.broadcast_to(val, (k,))
    elif val.shape != (k,):
        raise ValueError(
            f"accumulate_vertex_values_into: values must be a scalar or have "
            f"shape ({k},) to match vertex_idx, got shape {val.shape}."
        )

    wgt = np.asarray(weights, dtype=np.float64)
    if wgt.shape != (k,):
        raise ValueError(
            f"accumulate_vertex_values_into: weights must have shape ({k},) to "
            f"match vertex_idx, got shape {wgt.shape}. Weights are required and "
            f"are never broadcast from a scalar — pass np.ones({k}) for "
            f"equally-weighted points."
        )

    np.add.at(result.value_sum, idx, val * wgt)
    np.maximum.at(result.value_max, idx, val)
    np.add.at(result.weight_sum, idx, wgt)
    np.add.at(result.weight_sq_sum, idx, wgt * wgt)


def accumulate_vertex_values(
    vertex_idx: np.ndarray,
    values,
    weights: np.ndarray,
    n_vertices: int,
) -> AccumResult:
    """Reduce one batch of contact points into a fresh accumulator.

    Convenience over :func:`empty_accumulator` +
    :func:`accumulate_vertex_values_into` for callers that reduce everything in
    a single call.  See :func:`accumulate_vertex_values_into` for the argument
    contract and :class:`AccumResult` for the outputs.
    """
    result = empty_accumulator(n_vertices)
    accumulate_vertex_values_into(result, vertex_idx, values, weights)
    return result
