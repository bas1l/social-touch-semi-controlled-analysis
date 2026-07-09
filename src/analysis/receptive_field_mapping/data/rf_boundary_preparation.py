"""Preparation layer for the ``spatial_extract_boundaries`` stage.

Turns raw :class:`BoundaryInputs` into a :class:`PreparedBoundaryData` — i.e.
everything derived *before* boundary detection: the SLIM<->raw vertex mapping,
per-gesture population heatmaps (incl. the synthetic ``stroke`` subset), PCA
alignment of the UV frame, the interpolated + island-cleaned grids, and the
session-wide colour scale. All pure transforms; no disk I/O.
"""

import logging

import numpy as np
from scipy.spatial import KDTree

from analysis.pipeline.shared_constants import GESTURE_TYPES
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryInputs,
    BoundaryParams,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.data.rf_grid_interpolation import (
    compute_interpolated_grid,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    build_gesture_touch_indices,
    clean_heatmap_islands,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)
from analysis.receptive_field_mapping.metrics.rf_pca_alignment import (
    apply_uv_alignment,
    compute_rf_pca_alignment,
)

logger = logging.getLogger(__name__)

_WARN_THRESHOLD_MM = 1.0
_ERROR_THRESHOLD_MM = 5.0
_CANONICAL_ORDER = ['all', 'stroke', 'tap', 'stroke_proximal', 'stroke_distal']


def _map_slim_to_raw(pop_data, slim_V: np.ndarray, session_id: str) -> np.ndarray:
    """KDTree nearest-neighbour map from SLIM vertices to original forearm vertices."""
    # The SLIM cache stores cleaned-mesh vertices (after BPA + clean_mesh +
    # flatten_slim), which can include synthetic centroid vertices from
    # interior hole filling.  These vertices are not in the raw PLY, so a
    # handful will be >1 mm from their nearest raw neighbour.  Genuine
    # misalignment (wrong PLY) would show distances of 10+ mm.
    orig_tree = KDTree(pop_data.forearm_vertices)
    distances, nearest_orig_for_slim = orig_tree.query(slim_V)
    max_dist_mm = float(distances.max())
    if max_dist_mm > _ERROR_THRESHOLD_MM:
        raise ValueError(
            f"[Population Response Fields] {session_id}: KDTree nearest-neighbour "
            f"mapping from SLIM vertices to original forearm vertices has a "
            f"maximum distance of {max_dist_mm:.3f} mm (threshold: "
            f"{_ERROR_THRESHOLD_MM} mm). The SLIM mesh and forearm PLY are "
            "misaligned — re-run 'spatial_precompute_slim_uv' after "
            "verifying the forearm PLY."
        )
    if max_dist_mm > _WARN_THRESHOLD_MM:
        n_over = int((distances > _WARN_THRESHOLD_MM).sum())
        logger.info(
            "[Population Response Fields] %s: %d / %d SLIM vertices are >%.0f mm "
            "from the nearest raw PLY vertex (max=%.3f mm) — expected for "
            "hole-fill centroid vertices.",
            session_id, n_over, len(slim_V),
            _WARN_THRESHOLD_MM, max_dist_mm,
        )
    return nearest_orig_for_slim


def _build_gesture_results(
    pop_data,
    rf_data,
    n_verts: int,
    nearest_orig_for_slim: np.ndarray,
    min_overlap_pct: float,
    session_id: str,
):
    """Compute per-gesture SLIM heatmaps + the synthetic ``stroke`` subset.

    Returns ``(results, per_gesture_slim_raw, per_gesture_slim_unique_count)``
    with ``results`` in canonical gesture order.
    """
    subsets = ['all'] + list(GESTURE_TYPES)
    results: dict = {}
    per_gesture_slim_raw: dict = {}
    per_gesture_slim_unique_count: dict = {}

    def _add_subset(gtype: str, gesture_touch_indices: np.ndarray) -> None:
        n_gesture_touches = len(gesture_touch_indices)
        cp_mask = np.isin(pop_data.cp_touch_idx, gesture_touch_indices)
        heatmap = compute_rf_heatmap(
            gesture_touch_indices,
            rf_data.rf_vertex_indices,
            rf_data.rf_values,
            n_verts,
        )
        unique_count = compute_unique_touch_count(
            pop_data.cp_vertex_idx,
            pop_data.cp_touch_idx,
            cp_mask,
            n_verts,
        )
        per_gesture_slim_raw[gtype] = heatmap[nearest_orig_for_slim]
        per_gesture_slim_unique_count[gtype] = unique_count[nearest_orig_for_slim]
        threshold = compute_threshold_from_ratio(min_overlap_pct, n_gesture_touches)
        thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)
        slim_heatmap = thresholded[nearest_orig_for_slim]
        results[gtype] = (slim_heatmap, n_gesture_touches, threshold)

    for gtype in subsets:
        if gtype == 'all':
            gesture_touch_indices = np.arange(len(pop_data.touch_triple_keys))
        else:
            gesture_touch_indices = build_gesture_touch_indices(
                pop_data.gesture_types, gtype
            )

        if len(gesture_touch_indices) == 0:
            logger.warning(
                "[Population Response Fields] %s: no touches for gesture type '%s' — skipping.",
                session_id, gtype,
            )
            continue

        _add_subset(gtype, gesture_touch_indices)

    # --- Synthesize 'stroke' = stroke_proximal + stroke_distal ---
    _sp_in = 'stroke_proximal' in results
    _sd_in = 'stroke_distal' in results
    if _sp_in or _sd_in:
        parts = []
        if _sp_in:
            parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_proximal'))
        if _sd_in:
            parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_distal'))
        stroke_touch_indices = np.concatenate(parts)

        if len(stroke_touch_indices) > 0:
            _add_subset('stroke', stroke_touch_indices)

    results = {k: results[k] for k in _CANONICAL_ORDER if k in results}
    return results, per_gesture_slim_raw, per_gesture_slim_unique_count


def prepare_session_boundary_data(
    inputs: BoundaryInputs,
    params: BoundaryParams,
) -> PreparedBoundaryData | None:
    """Derive everything needed before boundary detection for one session.

    Returns ``None`` when no gesture subset had any touches (the caller then
    writes an empty sentinel and skips). Raises ``ValueError`` if the mandatory
    ``'all'`` subset is missing when other subsets are present.
    """
    session_id = inputs.session_id
    pop_data = inputs.pop_data
    rf_data = inputs.rf_data
    slim_V = inputs.slim_V
    slim_faces = inputs.slim_faces
    forearm_uv = inputs.forearm_uv
    n_verts = len(pop_data.forearm_vertices)

    # --- SLIM -> raw vertex mapping ---
    nearest_orig_for_slim = _map_slim_to_raw(pop_data, slim_V, session_id)

    # --- Map PLY vertex colors to SLIM vertices via the same mapping ---
    if inputs.raw_vertex_colors is not None:
        slim_colors_rgb = inputs.raw_vertex_colors[nearest_orig_for_slim].astype(np.float64) / 255.0
        alpha_col = np.ones((len(slim_colors_rgb), 1), dtype=np.float64)
        slim_vertex_colors = np.hstack([slim_colors_rgb, alpha_col])
    else:
        slim_vertex_colors = None

    # --- Per-gesture heatmaps (+ synthetic stroke), canonical order ---
    results, per_gesture_slim_raw, per_gesture_slim_unique_count = _build_gesture_results(
        pop_data, rf_data, n_verts, nearest_orig_for_slim,
        params.min_overlap_pct, session_id,
    )

    if not results:
        logger.warning(
            "[Population Response Fields] %s: no gesture subsets had touches — no PNGs produced.",
            session_id,
        )
        return None

    if 'all' not in results:
        raise ValueError(
            f"[Population Response Fields] {session_id}: 'all' gesture type missing from "
            f"results — cannot compute PCA alignment. This should not happen."
        )

    # --- PCA alignment of the UV frame (from the 'all' heatmap) ---
    all_heatmap, _, _ = results['all']
    alignment_center, alignment_rotation_matrix, alignment_angle_deg = compute_rf_pca_alignment(
        forearm_uv, all_heatmap
    )
    logger.info(
        "[Population Response Fields] %s: PCA alignment — center=(%.3f, %.3f) angle=%.1f°",
        session_id, alignment_center[0], alignment_center[1], alignment_angle_deg,
    )
    forearm_uv = apply_uv_alignment(forearm_uv, alignment_center, alignment_rotation_matrix)

    if params.flip_u:
        forearm_uv[:, 0] *= -1
        logger.info(
            "[Population Response Fields] %s: U-axis flipped (flip_u=True)",
            session_id,
        )

    # --- Session-wide colour scale ---
    finite_maxima = [
        float(np.nanmax(h))
        for (h, _, _) in results.values()
        if np.any(np.isfinite(h) & (h >= 0))
    ]
    if not finite_maxima:
        raise ValueError(
            f"[Population Response Fields] {session_id}: all heatmaps are empty or "
            f"below-threshold — no valid colour scale can be determined. "
            f"Check upstream RF data."
        )
    session_vmax = max(finite_maxima)
    finite_minima = [
        float(np.nanmin(h[np.isfinite(h) & (h > 0)]))
        for (h, _, _) in results.values()
        if np.any(np.isfinite(h) & (h > 0))
    ]
    session_vmin = min(finite_minima) if finite_minima else session_vmax * 1e-3

    # --- Interpolated + island-cleaned grids per gesture ---
    per_gesture_grids: dict = {}
    for gtype, (slim_heatmap, _n_touches, _threshold) in results.items():
        grid_u, grid_v, grid_z = compute_interpolated_grid(
            forearm_uv, slim_faces, slim_V, slim_heatmap,
            median_filter_size=params.median_filter_size,
        )
        grid_z = clean_heatmap_islands(grid_z)
        per_gesture_grids[gtype] = (grid_u, grid_v, grid_z)

    return PreparedBoundaryData(
        session_id=session_id,
        output_dir=inputs.session_output_dir,
        sentinel=inputs.sentinel,
        forearm_uv=forearm_uv,
        slim_faces=slim_faces,
        slim_V=slim_V,
        results=results,
        per_gesture_grids=per_gesture_grids,
        per_gesture_slim_raw=per_gesture_slim_raw,
        per_gesture_slim_unique_count=per_gesture_slim_unique_count,
        session_vmax=session_vmax,
        session_vmin=session_vmin,
        min_overlap_pct=params.min_overlap_pct,
        alignment_center=alignment_center,
        alignment_rotation_matrix=alignment_rotation_matrix,
        alignment_angle_deg=alignment_angle_deg,
        flip_u=params.flip_u,
        slim_vertex_colors=slim_vertex_colors,
        forearm_ply_path=inputs.forearm_ply_path,
    )
