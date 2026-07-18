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


def _compute_param_free_fields(
    pop_data,
    rf_data,
    n_verts: int,
    nearest_orig_for_slim: np.ndarray,
    session_id: str,
) -> dict:
    """Single source of the parameter-free per-gesture field math.

    For every gesture subset (``'all'`` + :data:`GESTURE_TYPES` plus the
    synthetic ``'stroke'`` = stroke_proximal + stroke_distal) computes, using
    only production functions and *no* contour parameter, the SLIM-mapped raw
    heatmap, the SLIM-mapped unique-touch count, and the touch count:

        {gtype: (slim_raw, slim_unique_count, n_touches)}

    in canonical gesture order. Zero-touch subsets are skipped with a warning.
    This is the exact param-free computation the boundary extractor used to inline
    inside ``_add_subset`` — factored out so the ``spatial_build_response_fields``
    task and the extractor share it with no duplicated math.
    """
    subsets = ['all'] + list(GESTURE_TYPES)
    fields: dict = {}

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
        fields[gtype] = (
            heatmap[nearest_orig_for_slim],
            unique_count[nearest_orig_for_slim],
            n_gesture_touches,
        )

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
    _sp_in = 'stroke_proximal' in fields
    _sd_in = 'stroke_distal' in fields
    if _sp_in or _sd_in:
        parts = []
        if _sp_in:
            parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_proximal'))
        if _sd_in:
            parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_distal'))
        stroke_touch_indices = np.concatenate(parts)

        if len(stroke_touch_indices) > 0:
            _add_subset('stroke', stroke_touch_indices)

    fields = {k: fields[k] for k in _CANONICAL_ORDER if k in fields}
    return fields


def build_param_free_fields(inputs: BoundaryInputs) -> dict[str, tuple]:
    """Compute the parameter-free per-(session, gesture) response fields.

    Public production helper shared by the ``spatial_build_response_fields`` task
    and the boundary extractor. Given raw :class:`BoundaryInputs`, returns

        {gtype: (slim_raw, slim_unique_count, n_touches)}

    in canonical gesture order (incl. the synthetic ``'stroke'`` subset). Uses
    only production functions (``compute_rf_heatmap`` /
    ``compute_unique_touch_count`` + the SLIM<->raw nearest-neighbour map) and no
    contour parameter, so identical upstream inputs yield byte-identical fields.
    ``slim_raw`` is float64, ``slim_unique_count`` is int64, ``n_touches`` is a
    Python ``int``.
    """
    pop_data = inputs.pop_data
    rf_data = inputs.rf_data
    n_verts = len(pop_data.forearm_vertices)
    nearest_orig_for_slim = _map_slim_to_raw(pop_data, inputs.slim_V, inputs.session_id)
    return _compute_param_free_fields(
        pop_data, rf_data, n_verts, nearest_orig_for_slim, inputs.session_id,
    )


def _build_gesture_results(
    param_free: dict,
    params: BoundaryParams,
    per_gesture_params,
):
    """Threshold the param-free per-gesture fields into SLIM heatmaps.

    Returns ``(results, per_gesture_slim_raw, per_gesture_slim_unique_count)``
    with ``results`` in the (canonical) gesture order of ``param_free``.

    ``param_free`` (``{gtype -> (slim_raw, slim_unique_count, n_touches)}``) is
    the parameter-free field math — either recomputed via
    :func:`build_param_free_fields` or read verbatim from the shared
    ``spatial_build_response_fields`` NPZ (byte-identical, since float64/int64
    round-trip exactly). Thresholding is applied on top here. Because
    ``apply_vertex_threshold`` is elementwise, the SLIM-map-then-threshold order
    used here is byte-identical to the previous threshold-then-SLIM-map order.

    ``per_gesture_params`` (``dict[gtype -> GestureContourParams]`` or ``None``)
    supplies the per-(session, gesture) ``min_overlap_pct`` when
    ``use_tuned_params`` is on; when ``None`` the global
    ``params.min_overlap_pct`` scalar is used for every subset (the unchanged,
    byte-identical default path).
    """
    results: dict = {}
    per_gesture_slim_raw: dict = {}
    per_gesture_slim_unique_count: dict = {}

    for gtype, (slim_raw, slim_unique_count, n_gesture_touches) in param_free.items():
        per_gesture_slim_raw[gtype] = slim_raw
        per_gesture_slim_unique_count[gtype] = slim_unique_count
        min_overlap_pct = (
            per_gesture_params[gtype].effective_min_overlap_pct()
            if per_gesture_params is not None
            else params.min_overlap_pct
        )
        threshold = compute_threshold_from_ratio(min_overlap_pct, n_gesture_touches)
        slim_heatmap = apply_vertex_threshold(slim_raw, slim_unique_count, threshold)
        results[gtype] = (slim_heatmap, n_gesture_touches, threshold)

    return results, per_gesture_slim_raw, per_gesture_slim_unique_count


def prepare_session_boundary_data(
    inputs: BoundaryInputs,
    params: BoundaryParams,
    per_gesture_params=None,
    param_free: dict | None = None,
) -> PreparedBoundaryData | None:
    """Derive everything needed before boundary detection for one session.

    Returns ``None`` when no gesture subset had any touches (the caller then
    writes an empty sentinel and skips). Raises ``ValueError`` if the mandatory
    ``'all'`` subset is missing when other subsets are present.

    ``param_free`` (``{gtype -> (slim_raw, slim_unique_count, n_touches)}`` or
    ``None``) is the parameter-free per-gesture field math. The pipeline passes
    the fields read verbatim from the shared ``spatial_build_response_fields``
    NPZ (single source of truth). When ``None`` they are recomputed from
    ``inputs`` via :func:`build_param_free_fields` — byte-identical, since
    float64/int64 round-trip through ``np.savez`` exactly (the recompute path is
    used by the parity test). Either way the SLIM vertex colours and the PCA/
    interpolation mesh are still derived from the raw ``inputs`` (they are not in
    the shared NPZ).

    ``per_gesture_params`` (``dict[gtype -> GestureContourParams]`` or ``None``)
    carries the per-(session, gesture) tuned ``min_overlap_pct`` /
    ``median_filter_size`` when ``use_tuned_params`` is on. When ``None`` the
    global ``params`` scalars are used unchanged (byte-identical default path).
    It is stored verbatim on the returned :class:`PreparedBoundaryData` so the
    extract layer can read the per-gesture radial sigmas.
    """
    session_id = inputs.session_id
    pop_data = inputs.pop_data
    slim_V = inputs.slim_V
    slim_faces = inputs.slim_faces
    forearm_uv = inputs.forearm_uv

    # --- SLIM -> raw vertex mapping (needed for the per-vertex colours) ---
    nearest_orig_for_slim = _map_slim_to_raw(pop_data, slim_V, session_id)

    # --- Map PLY vertex colors to SLIM vertices via the same mapping ---
    if inputs.raw_vertex_colors is not None:
        slim_colors_rgb = inputs.raw_vertex_colors[nearest_orig_for_slim].astype(np.float64) / 255.0
        alpha_col = np.ones((len(slim_colors_rgb), 1), dtype=np.float64)
        slim_vertex_colors = np.hstack([slim_colors_rgb, alpha_col])
    else:
        slim_vertex_colors = None

    # --- Param-free per-gesture fields: from the shared NPZ or recomputed ---
    if param_free is None:
        param_free = build_param_free_fields(inputs)

    # --- Per-gesture heatmaps (+ synthetic stroke), canonical order ---
    results, per_gesture_slim_raw, per_gesture_slim_unique_count = _build_gesture_results(
        param_free, params, per_gesture_params,
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
    # NB: the 'all' subset's min_overlap_pct governs this shared UV frame — the
    # alignment is derived from the 'all' heatmap and then applied to every
    # gesture's grid, so tuning 'all''s overlap shifts the frame for all subsets.
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
        median_filter_size = (
            per_gesture_params[gtype].effective_median_filter_size()
            if per_gesture_params is not None
            else params.median_filter_size
        )
        grid_u, grid_v, grid_z = compute_interpolated_grid(
            forearm_uv, slim_faces, slim_V, slim_heatmap,
            median_filter_size=median_filter_size,
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
        per_gesture_params=per_gesture_params,
    )
