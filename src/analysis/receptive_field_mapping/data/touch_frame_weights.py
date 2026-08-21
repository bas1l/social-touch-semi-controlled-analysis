"""One frame of one touch -> that frame's contact weights.

This is the adapter between a :class:`TouchEvent` (which stores the sidecar
column *as stored* — ``signed_depth_mm``, negative = penetrating) and
:func:`analysis.receptive_field_mapping.data.vertex_weights.vertex_weights`
(which consumes *penetration*, positive = pressed in).

Why it exists at all
--------------------
Two consumers must produce **the same weights for the same frame**:

* ``rf_single_touch_pipeline._compute_touch_rf`` — writes the saved
  ``single_touch_rf_maps_mean.npz``;
* ``gui.touch_playback_explorer`` — draws the heatmap a human looks at.

The whole reason the nine duplicated accumulator blocks were merged into one
shared reduction was that a viewer drawing a different map from the one on disk
is the failure mode that actively misleads. Re-spelling the depth -> weight
conversion in the viewer would reintroduce exactly that risk one layer up: the
reduction would be shared while the *weights* fed into it drifted. So the
conversion is written once, here, and both call it.

What it deliberately does not do
--------------------------------
It does not accumulate, it does not know about IFF, and it does not choose
``alpha`` — the caller supplies it, required, with no default at any level.
:mod:`analysis.receptive_field_mapping.data.vertex_weights` stays ignorant of
``TouchEvent``, frames and sidecars; this module is the only place that knows
both sides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from analysis.receptive_field_mapping.data.contact_depth_field_io import (
    penetration_from_signed_mm,
)
from analysis.receptive_field_mapping.data.vertex_weights import vertex_weights

if TYPE_CHECKING:  # pragma: no cover - typing only
    from analysis.receptive_field_mapping.data.touch_playback_data import TouchEvent

__all__ = ["touch_label", "touch_frame_weights"]


def touch_label(touch: "TouchEvent") -> str:
    """Identify a touch in an error message: block / trial / single-touch id."""
    return (
        f"block={touch.block_order_id!r} trial={touch.trial_id!r} "
        f"single_touch={touch.single_touch_id!r}"
    )


def touch_frame_weights(
    touch: "TouchEvent", frame_idx: int, depth_weight_alpha: float
) -> np.ndarray:
    """Return the ``(K_i,)`` contact weights of frame *frame_idx* of *touch*.

    Parameters
    ----------
    touch:
        A ``TouchEvent`` from ``load_playback_data()``.  ``frame_depths[i]`` holds
        the sidecar's ``signed_depth_mm`` for the same contact points, in the same
        order, as ``frame_vertex_indices[i]`` — both are read off the same sidecar
        rows in the same loop.
    frame_idx:
        Index into the touch's per-frame lists.
    depth_weight_alpha:
        Weighting exponent, **required**.  ``0.0`` returns all ones — including for
        grazing contacts clamped to zero penetration, because ``0.0 ** 0.0`` is
        exactly ``1.0`` — so the unweighted behaviour is reproduced by the same
        arithmetic rather than by a branch.

    Raises
    ------
    ValueError
        The frame's depth array does not align with its vertex array, or
        ``vertex_weights`` rejects the frame (non-finite depth, or every contact
        point grazing so the frame's maximum penetration is 0).  The touch and
        frame are named in the message; a bare complaint about an array would not
        say which touch to go and look at.
    """
    verts = touch.frame_vertex_indices[frame_idx]
    signed_depths = touch.frame_depths[frame_idx]
    if len(signed_depths) != len(verts):
        raise ValueError(
            f"touch_frame_weights: touch {touch_label(touch)} frame {frame_idx} has "
            f"{len(verts)} contact points but {len(signed_depths)} depths. "
            f"Both are read off the same sidecar rows in the same loop, so a "
            f"mismatch means the two channels have desynchronised upstream."
        )
    try:
        return vertex_weights(
            penetration_from_signed_mm(signed_depths), depth_weight_alpha
        )
    except ValueError as exc:
        raise ValueError(
            f"touch_frame_weights: touch {touch_label(touch)} frame {frame_idx} "
            f"(vertices {np.asarray(verts).tolist()}): {exc}"
        ) from exc
