"""Accumulator sums -> the estimate, and the single definition of "no estimate".

``vertex_accumulator`` produces ``value_sum`` and ``weight_sum``; this module is
the one place that turns that pair into a firing rate, and the one place that
decides where **no rate exists**.  It completes the trio that keeps the saved
maps and the on-screen heatmap the same numbers:

* :mod:`analysis.receptive_field_mapping.data.touch_frame_weights` — one depth
  -> weight conversion,
* :mod:`analysis.receptive_field_mapping.data.vertex_accumulator` — one
  reduction,
* this module — one estimator, and one answer to "which vertices have no
  estimate at all".

Before it existed the third step was spelled twice: ``_compute_touch_rf``
*raised* on a contacted vertex whose weights all came out zero, while
``gui.touch_playback_explorer.mean_heatmap_scalars`` quietly painted the same
vertex grey.  The pipeline therefore refused sessions the viewer had already
drawn without complaint.  A viewer disagreeing with the file is the failure mode
the shared-reduction work existed to prevent, so the predicate lives here and
both callers import it.

What "no estimate" means
------------------------
``NaN`` in the returned array means **no estimate**.  It never means "zero
response", and it must never be read as one.  Two distinct facts land on it:

``weight_sum == 0``
    Either the vertex was never contacted, or it was contacted and every frame
    that touched it was **grazing** — it sat on or above the skin surface while
    the press landed elsewhere.  At ``depth_weight_alpha > 0`` such a vertex
    accumulates exactly zero total weight, so the weighted mean is ``0/0``.
    There is **no fallback to the unweighted mean**: that would mix two
    different estimators inside one map, and the resulting number would not be
    comparable with its neighbours.  The vertex has zero penetration evidence,
    so the honest output is "no estimate", not a value and not a stop.

    The two sub-cases are distinguished by the *caller*, which knows the
    contacted set: ``weight_sum == 0`` alone cannot separate "never touched"
    from "touched, weightless".  See :func:`count_no_estimate`.

``value_sum`` is NaN
    Every frame contributing to that vertex carried a NaN neuron value — e.g.
    the unit was not held during the touch window.  There is a press but no
    neural measurement.

At ``depth_weight_alpha = 0`` the first case is **unreachable for a contacted
vertex**: every weight is then exactly ``1.0`` (``0.0 ** 0.0 == 1.0``), so a
contacted vertex always has ``weight_sum >= 1``.  Nothing in this module can
therefore move an ``alpha = 0`` output.

Contract
--------
This module knows about **per-vertex sums and weights** and nothing else.  It
must never learn about penetration depth, ``alpha``, millimetres, ``TouchEvent``,
Qt or files.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NoEstimateCounts",
    "weighted_mean_or_nan",
    "has_estimate",
    "count_no_estimate",
]


@dataclass(frozen=True)
class NoEstimateCounts:
    """How many entries have no estimate, split by cause.

    The two causes are kept apart because they are different statements about
    the data: ``zero_weight`` says the press never pressed in at that vertex,
    ``nan_value`` says there was no neural measurement while it did.  Both are
    excluded from the emitted map; collapsing them into one number would make
    "the electrode dropped out" indistinguishable from "the finger only grazed".
    """

    zero_weight: int
    nan_value: int

    @property
    def total(self) -> int:
        """Entries excluded from the map for either reason."""
        return self.zero_weight + self.nan_value


def _as_pair(value_sum, weight_sum):
    """Validate and return the two 1-D float arrays this module operates on."""
    values = np.asarray(value_sum, dtype=np.float64)
    weights = np.asarray(weight_sum, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1:
        raise ValueError(
            f"vertex_estimate: value_sum and weight_sum must be 1-D, one entry "
            f"per vertex; got shapes {values.shape} and {weights.shape}."
        )
    if values.shape != weights.shape:
        raise ValueError(
            f"vertex_estimate: value_sum and weight_sum must cover the same "
            f"vertices in the same order; got {values.shape} and "
            f"{weights.shape}. A mismatch means the numerator and the "
            f"denominator have been sliced differently, which would divide one "
            f"vertex's total by another vertex's weight."
        )
    return values, weights


def weighted_mean_or_nan(value_sum, weight_sum) -> np.ndarray:
    """Return ``value_sum / weight_sum``, or ``NaN`` where there is no estimate.

    The divisor is the **weight sum**, never a frame or contact count: dividing
    by the count would leave the weight in the answer as a scale factor and the
    number would stop being a firing rate.

    ``NaN`` in the result means *no estimate* — see the module docstring.  It is
    never a measured zero.

    Notes
    -----
    Where ``weight_sum > 0`` the returned value is exactly
    ``value_sum / weight_sum`` — the same IEEE division, on the same operands,
    that a bare ``value_sum / weight_sum`` would produce.  The guard only
    substitutes the divisor where the quotient would be ``0/0``, so it cannot
    perturb any vertex that has an estimate.
    """
    values, weights = _as_pair(value_sum, weight_sum)
    has_weight = weights > 0.0
    return np.where(has_weight, values / np.where(has_weight, weights, 1.0), np.nan)


def has_estimate(value_sum, weight_sum) -> np.ndarray:
    """Boolean mask: ``True`` where a weighted mean exists for that entry.

    This is *the* predicate.  The pipeline uses it to decide which vertices
    reach the saved ``.npz``; the playback viewer's ``NaN`` mask is its
    complement, so the vertices painted grey on screen and the vertices absent
    from the file are the same set by construction rather than by agreement.
    """
    return ~np.isnan(weighted_mean_or_nan(value_sum, weight_sum))


def count_no_estimate(value_sum, weight_sum) -> NoEstimateCounts:
    """Count entries with no estimate, split by cause.

    Parameters
    ----------
    value_sum, weight_sum:
        The accumulator totals **already restricted to the contacted entries**.
        Restriction is the caller's job because only the caller holds the
        contacted set: an untouched vertex and an all-grazing vertex both leave
        ``weight_sum == 0``, and counting the first as an exclusion would report
        the whole un-pressed mesh as missing data.
    """
    values, weights = _as_pair(value_sum, weight_sum)
    zero_weight = weights <= 0.0
    no_estimate = np.isnan(weighted_mean_or_nan(values, weights))
    return NoEstimateCounts(
        zero_weight=int(np.count_nonzero(zero_weight)),
        nan_value=int(np.count_nonzero(no_estimate & ~zero_weight)),
    )
