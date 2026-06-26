"""NaN-aware helpers shared by the preprocessing operators.

None of scipy/skimage's derivative operators are NaN-aware: NaNs propagate and
Gaussian convolution smears them inward.  We therefore fill the masked region by
nearest-neighbour extrapolation (the same approach ``compute_laplacian_arrays``
uses internally for its Laplacian), run the operator on the filled field, then
re-mask the originally-NaN cells.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def _extrapolate(fieldarr: np.ndarray) -> np.ndarray:
    """Fill NaN cells by nearest-neighbour extrapolation. Raises if all-NaN."""
    mask = np.isnan(fieldarr)
    if not mask.any():
        return fieldarr.astype(float, copy=True)
    if mask.all():
        raise ValueError("operator received an all-NaN field")
    _distances, indices = distance_transform_edt(mask, return_indices=True)
    out = fieldarr.astype(float, copy=True)
    out[mask] = fieldarr[indices[0][mask], indices[1][mask]]
    return out


def _nan_aware(fieldarr: np.ndarray, fn) -> np.ndarray:
    """Apply *fn* to the nearest-neighbour-extrapolated field, then re-mask NaNs."""
    mask = np.isnan(fieldarr)
    result = np.asarray(fn(_extrapolate(fieldarr)), dtype=float)
    result[mask] = np.nan
    return result
