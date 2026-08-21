"""Turn per-frame penetration depths into per-frame contact weights.

One question, answered once: **how fully was this vertex pressed, compared with
the deepest point of the press at that same instant?**  ``1.0`` means it *was*
the deepest point of the frame; ``0.1`` means it was barely touched while the
force landed somewhere else.

Contract
--------
This module knows about **penetration depth in millimetres and ``alpha``**, and
nothing else.  It must never learn about IFF, vertices, frames, accumulation,
parquet sidecars, Qt or files.  It is handed one frame's depths and hands back
that frame's weights; who those depths belong to and what is done with the
weights afterwards are somebody else's concern.  That separation is what keeps
:mod:`analysis.receptive_field_mapping.data.vertex_accumulator` ignorant of
``alpha`` entirely, and makes an alternative weighting (saturating,
thresholded) an addition here rather than an edit there.

Sign convention
---------------
The argument is **penetration**: positive means pressed *into* the skin.  The
sidecar stores the opposite sign (``signed_depth_mm``, negative = penetrating),
so callers must convert with
:func:`analysis.receptive_field_mapping.data.contact_depth_field_io.penetration_from_signed_mm`
first.  That function is the single negation point in the codebase and this one
deliberately does not become a second: a module that accepted either sign, or
guessed from the data which it had been given, would turn a sign error into a
plausible-looking map instead of an exception.

The maths
---------
::

    p_i   = max(penetration_i, 0)        # grazing clamped to zero
    d_max = max_j p_j                    # this frame's own maximum
    w_i   = (p_i / d_max) ** alpha       # NORMALISE BEFORE THE EXPONENT

Why max-normalised, not sum-normalised:

* Weights land in ``[0, 1]`` with exactly ``1.0`` at the deepest point, which is
  what the quantity means.
* It degrades gently from the unweighted behaviour.  Today every vertex carries
  weight ``1``; max-normalising only ever lowers weights *from* ``1``.
  Sum-normalising instead crushes every weight to about ``1/K`` and changes a
  frame's total contribution from ``K`` to ``1``.
* Both ends of the dial land where they should: ``alpha = 0`` gives all ones
  (today, exactly), and ``alpha -> inf`` gives all the credit to the deepest
  vertex, which is the old ``contact_depth = max()``.
* It is scale-invariant.  Double every depth in a frame and nothing changes,
  because the factor cancels — depth magnitudes may differ between sessions and
  the weight must not.

Why the exponent comes **last**:

* ``(p_i / d_max) ** alpha`` at ``alpha = 0`` gives ``1`` for every vertex, so
  the unweighted baseline is reproduced exactly and the parity test means
  something.
* ``p_i**alpha / sum_j p_j**alpha`` at ``alpha = 0`` gives ``1/K`` and destroys
  that baseline.

At ``alpha = 1`` the two forms are identical, so ordering them this way costs
nothing.  Note that "normalise" is only literally true at ``alpha = 1``: at
``alpha = 2`` the weights no longer sum to a fixed total.  Only the *maximum* is
pinned at ``1.0``, at every alpha.  That matters if ``alpha`` is ever swept.

There is deliberately **no** ``if alpha == 0`` fast path.  ``alpha = 0`` runs
this same arithmetic and gets ones out of it because ``x ** 0.0 == 1.0``
exactly, for every finite ``x`` including ``0.0``.  A branch would be control
coupling, and it would make the baseline parity test vacuous: it would prove the
old code still works, not that the new code reduces to it.
"""

import numpy as np

__all__ = ["vertex_weights"]


def vertex_weights(penetration_mm: np.ndarray, alpha: float) -> np.ndarray:
    """Return one frame's contact weights from that frame's penetration depths.

    Parameters
    ----------
    penetration_mm:
        ``(K,)`` array of **penetration** magnitudes in millimetres for the K
        contact points of a single frame — positive means pressed into the
        skin.  Small negative values are grazing contacts, not errors; they are
        clamped to ``0``.  Convert from the stored ``signed_depth_mm`` with
        ``contact_depth_field_io.penetration_from_signed_mm``.
    alpha:
        Weighting exponent.  **Positional and required** — there is no default
        anywhere in this codebase, so an un-passed ``alpha`` is a ``TypeError``
        rather than a silent ``1.0``.  ``0`` reproduces the unweighted
        behaviour exactly; ``1`` credits a vertex in proportion to how deeply it
        was pressed; larger values concentrate credit on the deepest point.

    Returns
    -------
    ``(K,)`` float64 array in ``[0, 1]``, with exactly ``1.0`` at the deepest
    contact point of the frame.

    Raises
    ------
    TypeError
        ``alpha`` is not a real number, or ``penetration_mm`` cannot be read as
        floats.
    ValueError
        ``penetration_mm`` is not 1-D, is empty, or holds a non-finite value;
        ``alpha`` is negative or non-finite; or every contact point in the frame
        is grazing, so ``d_max == 0`` and the ratio is ``0/0``.

    Notes
    -----
    The ``d_max == 0`` case is checked and raised explicitly rather than left to
    NumPy.  NumPy would quietly rescue ``alpha = 0`` through ``nan ** 0 == 1.0``
    and return garbage for every other alpha; relying on that is IEEE trivia,
    not a design.  A frame in which nothing was actually pressed is a statement
    about the data that must reach a human.
    """
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"vertex_weights: alpha must be a real number, got "
            f"{type(alpha).__name__}."
        )
    alpha_f = float(alpha)
    if not np.isfinite(alpha_f):
        raise ValueError(f"vertex_weights: alpha must be finite, got {alpha!r}.")
    if alpha_f < 0.0:
        raise ValueError(
            f"vertex_weights: alpha must be >= 0, got {alpha_f}. A negative "
            f"exponent inverts the meaning of the weight — it would credit the "
            f"shallowest contact most — and sends a grazing (zero) point to "
            f"infinity."
        )

    depths = np.asarray(penetration_mm, dtype=np.float64)
    if depths.ndim != 1:
        raise ValueError(
            f"vertex_weights: penetration_mm must be 1-D, one entry per contact "
            f"point of a single frame; got shape {depths.shape}."
        )
    if depths.size == 0:
        raise ValueError(
            "vertex_weights: penetration_mm is empty. A frame with no contact "
            "points has no maximum depth and therefore no weights; callers skip "
            "empty frames rather than asking for a zero-length answer."
        )
    if not np.isfinite(depths).all():
        bad = np.flatnonzero(~np.isfinite(depths))
        raise ValueError(
            f"vertex_weights: penetration_mm holds non-finite values at "
            f"position(s) {bad.tolist()[:10]} of {depths.size} "
            f"(values {depths[bad].tolist()[:10]}). Missing depth is rejected at "
            f"the loader boundary, where 'zero' and 'absent' are still "
            f"separable; by here they would both collapse to weight 0."
        )

    # Grazing contacts sit just above the surface and carry a small *negative*
    # penetration. They are real contacts, not errors, so they are clamped to 0
    # (no credit) rather than dropped or allowed to go negative — a negative
    # weight would subtract a frame's firing rate from the vertex's total, and
    # an odd alpha would preserve the sign all the way into the map.
    clamped = np.maximum(depths, 0.0)

    d_max = float(clamped.max())
    if d_max <= 0.0:
        raise ValueError(
            f"vertex_weights: every one of the {depths.size} contact points in "
            f"this frame is grazing (max penetration {depths.max():.6g} mm <= 0), "
            f"so the frame's maximum depth is 0 and every weight would be 0/0. "
            f"There is no defensible weight to return: the frame records a touch "
            f"that never pressed in."
        )

    return (clamped / d_max) ** alpha_f
