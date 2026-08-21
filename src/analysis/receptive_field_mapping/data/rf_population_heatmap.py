"""Shared computation functions for population RF heatmaps."""

import numpy as np
from scipy.ndimage import label

from analysis.pipeline.shared_constants import GESTURE_TYPES  # noqa: F401  re-exported for consumers
from analysis.receptive_field_mapping.data.vertex_accumulator import (
    accumulate_vertex_values_into,
    empty_accumulator,
)


def compute_rf_heatmap(
    touch_indices: list[int],
    rf_vertex_indices: list[np.ndarray],
    rf_values: list[np.ndarray],
    n_verts: int,
) -> np.ndarray:
    """Mean RF value per vertex across selected touches. NaN for uncontacted.

    Every touch contributes with weight ``1.0``. Cross-touch weighting (a
    deeper touch outranking a shallower one) is deliberately not applied here:
    the depth weighting is a *within-frame* redistribution of credit, and each
    touch's RF value has already been reduced over its own frames.
    """
    accum = empty_accumulator(n_verts)
    for idx in touch_indices:
        verts = rf_vertex_indices[idx]
        vals = rf_values[idx]
        if len(verts) == 0:
            continue
        accumulate_vertex_values_into(
            accum, verts, vals, np.ones(len(verts), dtype=np.float64)
        )
    result = accum.value_sum
    count = accum.weight_sum
    nonzero = count > 0
    result[nonzero] /= count[nonzero]
    result[~nonzero] = np.nan
    return result


def compute_unique_touch_count(
    cp_vertex_idx: np.ndarray,
    cp_touch_idx: np.ndarray,
    cp_mask: np.ndarray,
    n_verts: int,
) -> np.ndarray:
    """Per-vertex count of distinct touches among masked contact points."""
    vertex_idx = cp_vertex_idx[cp_mask]
    touch_idx = cp_touch_idx[cp_mask]

    if len(vertex_idx) == 0:
        return np.zeros(n_verts, dtype=np.int64)

    # Encode (vertex, touch) pairs as a single integer for fast deduplication.
    max_touch = int(touch_idx.max()) + 1
    key = vertex_idx * max_touch + touch_idx
    unique_keys = np.unique(key)

    # Decode vertex indices from deduplicated keys.
    unique_verts = unique_keys // max_touch

    return np.bincount(unique_verts, minlength=n_verts).astype(np.int64)


def compute_threshold_from_ratio(ratio_pct: float, n_filtered: int) -> int:
    """Convert a percentage threshold to an absolute integer count."""
    return max(1, round(ratio_pct / 100 * n_filtered))


def apply_vertex_threshold(
    heatmap_val: np.ndarray,
    unique_touch_count: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Set below-threshold contacted vertices to -1.0 (grey marker)."""
    result = heatmap_val.copy()
    # A vertex is "contacted" when it has a positive unique-touch count.
    contacted = unique_touch_count > 0
    below_threshold = contacted & (unique_touch_count < threshold)
    result[below_threshold] = -1.0
    return result


def build_gesture_touch_indices(gesture_types: np.ndarray, gtype: str) -> np.ndarray:
    """Return touch indices where gesture_types == gtype."""
    return np.where(gesture_types == gtype)[0]


def clean_heatmap_islands(grid_z: np.ndarray) -> np.ndarray:
    """Keep only the connected component of ``grid_z`` containing the global peak.

    The interpolated heatmap is frequently not one unified blob: sparse
    disconnected groups ("islands") surround the main peak and produce artifacts
    in every downstream contour mechanism. This collapses the field to a single
    unified region by NaN-filling every component that does not contain the
    global maximum.

    Algorithm:
      1. ``valid = ~np.isnan(grid_z)``.
      2. Label connected components of ``valid`` with ``scipy.ndimage.label``
         using the default 4-connectivity structure (matching
         ``_select_peak_basin_contour``'s convention in rf_inflection_boundary).
      3. Locate the peak cell via ``np.nanargmax``.
      4. On a copy, set every cell outside the peak's component to NaN; return
         the copy.

    Fail-fast contract: raises ``ValueError`` if ``grid_z`` is all-NaN, since
    there is then no valid component (and no peak) to retain.
    """
    if np.all(np.isnan(grid_z)):
        raise ValueError(
            "clean_heatmap_islands: grid_z is all-NaN; no valid component to retain."
        )

    valid = ~np.isnan(grid_z)
    labeled, _ = label(valid)
    peak_rc = np.unravel_index(np.nanargmax(grid_z), grid_z.shape)
    peak_label = labeled[peak_rc]

    cleaned = grid_z.copy()
    cleaned[labeled != peak_label] = np.nan
    return cleaned
