"""I/O layer for the ``spatial_extract_boundaries`` stage.

Owns everything that touches disk for the boundary-extraction stage:

* path resolution + fail-fast existence guards for the upstream artifacts,
* loading the raw per-session inputs into a :class:`BoundaryInputs`,
* persisting the response-fields NPZ and the JSON sentinel.

The persistence functions read the derived arrays (``smoothed`` / ``laplacian``
/ ``gradient_mag``) straight from :class:`BoundaryResults` — they are computed
once in the processing layer and never recomputed here.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from analysis.pipeline.output_dirs import (
    SPATIAL_MAP_SINGLE_TOUCH,
    SPATIAL_SLIM_UV,
    TOUCH_COMPUTE_SERIES,
)
from analysis.pipeline.shared_constants import single_touch_npz_filename
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryInputs,
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_colors,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    inflection_boundary_to_dict,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    load_slim_uv_cache,
    uv_points_to_xyz,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def boundary_session_output_dir(output_dir: Path, session_id: str) -> Path:
    return output_dir / session_id


def boundary_sentinel_path(session_output_dir: Path, session_id: str) -> Path:
    return session_output_dir / f'{session_id}_population_response_fields_done.json'


# ---------------------------------------------------------------------------
# Input path resolution + existence guards (fail-fast)
# ---------------------------------------------------------------------------

def resolve_boundary_input_paths(
    csv_path: Path,
    database_path: Path,
    session_id: str,
    iff_metric: str,
) -> List[Path]:
    """Resolve the four upstream input artifacts, raising loudly if any is missing.

    Returns ``[series_csv, forearm_ply, single_touch_npz, slim_cache]`` — the
    exact set fed to the staleness gate.
    """
    npz_filename = single_touch_npz_filename(iff_metric)

    series_csv_path = (
        database_path / '4_analysed' / TOUCH_COMPUTE_SERIES
        / f'{session_id}_series_augmented.csv'
    )
    if not series_csv_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: series-augmented CSV not found — "
            f"run 'touch_compute_series' first: {series_csv_path}"
        )

    forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
    if forearm_ply_path is None:
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: forearm PLY not found in "
            f"{csv_path.parent} — RF-centred PLY must exist."
        )

    npz_path = (
        database_path / '4_analysed' / SPATIAL_MAP_SINGLE_TOUCH
        / session_id / npz_filename
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: {npz_filename} not found: "
            f"{npz_path}. Enable 'spatial_map_single_touch' in the DAG config and re-run."
        )

    slim_cache_path = (
        database_path / '4_analysed' / SPATIAL_SLIM_UV
        / session_id / f'{session_id}_slim_uv.npz'
    )
    if not slim_cache_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: SLIM UV cache not found: "
            f"{slim_cache_path}. Enable 'spatial_precompute_slim_uv' in the DAG "
            f"config and re-run to generate the cache."
        )

    return [series_csv_path, forearm_ply_path, npz_path, slim_cache_path]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_boundary_inputs(
    session_id: str,
    session_output_dir: Path,
    sentinel: Path,
    input_paths: List[Path],
) -> BoundaryInputs:
    """Load the raw per-session data from resolved ``input_paths``.

    ``input_paths`` must be ``[series_csv, forearm_ply, single_touch_npz,
    slim_cache]`` as returned by :func:`resolve_boundary_input_paths`.
    """
    series_csv_path, forearm_ply_path, npz_path, slim_cache_path = input_paths

    pop_data = load_population_data(series_csv_path, forearm_ply_path)
    n_verts = len(pop_data.forearm_vertices)
    rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_verts)

    cache = load_slim_uv_cache(slim_cache_path)
    raw_colors = load_forearm_vertex_colors(forearm_ply_path)

    return BoundaryInputs(
        session_id=session_id,
        session_output_dir=session_output_dir,
        sentinel=sentinel,
        pop_data=pop_data,
        rf_data=rf_data,
        forearm_uv=cache.uv,
        slim_V=cache.V,
        slim_faces=cache.F,
        raw_vertex_colors=raw_colors,
        forearm_ply_path=forearm_ply_path,
        input_paths=input_paths,
    )


# ---------------------------------------------------------------------------
# Persistence: response-fields NPZ
# ---------------------------------------------------------------------------

def _save_boundary_fields(
    data_dict: dict,
    prefix: str,
    gtype: str,
    boundary,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
) -> None:
    data_dict[f'{prefix}_contour_uv_{gtype}'] = boundary.contour_uv.astype(np.float64)
    data_dict[f'{prefix}_centroid_uv_{gtype}'] = np.array(boundary.centroid_uv, dtype=np.float64)
    data_dict[f'{prefix}_peak_uv_{gtype}'] = np.array(boundary.peak_uv, dtype=np.float64)
    data_dict[f'{prefix}_perimeter_uv_{gtype}'] = np.float64(boundary.perimeter_uv)
    data_dict[f'{prefix}_area_uv_{gtype}'] = np.float64(boundary.area_uv)
    data_dict[f'{prefix}_circularity_{gtype}'] = np.float64(boundary.circularity)
    data_dict[f'{prefix}_pca_major_uv_{gtype}'] = np.float64(boundary.pca_major_uv)
    data_dict[f'{prefix}_pca_minor_uv_{gtype}'] = np.float64(boundary.pca_minor_uv)
    data_dict[f'{prefix}_pca_orientation_deg_{gtype}'] = np.float64(boundary.pca_orientation_deg)
    data_dict[f'{prefix}_mean_iff_on_contour_{gtype}'] = np.float64(boundary.mean_iff_on_contour)
    data_dict[f'{prefix}_iff_at_centroid_{gtype}'] = np.float64(boundary.iff_at_centroid)

    contour_xyz = uv_points_to_xyz(
        boundary.contour_uv, forearm_uv, forearm_faces, forearm_V,
    )
    centroid_uv_arr = np.array(boundary.centroid_uv, dtype=np.float64).reshape(1, 2)
    centroid_xyz = uv_points_to_xyz(
        centroid_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    contour_xyz_closed = np.vstack([contour_xyz, contour_xyz[:1]])
    perimeter_xyz_mm = float(
        np.sum(np.linalg.norm(np.diff(contour_xyz_closed, axis=0), axis=1))
    )

    c = centroid_xyz
    edges_i = contour_xyz[:-1] - c
    edges_j = contour_xyz[1:] - c
    closing_i = contour_xyz[-1] - c
    closing_j = contour_xyz[0] - c
    cross_vecs = np.vstack([
        np.cross(edges_i, edges_j),
        np.cross(closing_i, closing_j).reshape(1, 3),
    ])
    area_xyz_mm2 = 0.5 * float(np.sum(np.linalg.norm(cross_vecs, axis=1)))

    peak_uv_arr = np.array(boundary.peak_uv, dtype=np.float64).reshape(1, 2)
    peak_xyz = uv_points_to_xyz(
        peak_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    data_dict[f'{prefix}_contour_xyz_{gtype}'] = contour_xyz.astype(np.float64)
    data_dict[f'{prefix}_centroid_xyz_{gtype}'] = centroid_xyz.astype(np.float64)
    data_dict[f'{prefix}_peak_xyz_{gtype}'] = peak_xyz.astype(np.float64)
    data_dict[f'{prefix}_perimeter_xyz_mm_{gtype}'] = np.float64(perimeter_xyz_mm)
    data_dict[f'{prefix}_area_xyz_mm2_{gtype}'] = np.float64(area_xyz_mm2)


def save_boundary_outputs_npz(
    prepared: PreparedBoundaryData,
    results: BoundaryResults,
    params: BoundaryParams,
) -> Path:
    """Persist ``{session_id}_population_response_fields.npz``.

    Derived arrays (``smoothed`` / ``laplacian`` / ``gradient_mag`` / radial
    ``lmax``) are taken from *results*; nothing is recomputed here.
    """
    session_id = prepared.session_id
    forearm_uv = prepared.forearm_uv
    forearm_faces = prepared.slim_faces
    forearm_V = prepared.slim_V
    inflection_sigma = params.inflection_sigma

    npz_path = prepared.output_dir / f'{session_id}_population_response_fields.npz'

    data_dict: dict = {
        'forearm_uv': forearm_uv.astype(np.float64),
        'forearm_faces': forearm_faces.astype(np.int32),
        'forearm_V': forearm_V.astype(np.float64),
        'session_id': np.array(session_id, dtype=object),
        'neuron_mode': np.array(params.neuron_mode, dtype=object),
        'min_overlap_pct': np.float64(prepared.min_overlap_pct),
        'gesture_types': np.array(list(prepared.results.keys()), dtype=object),
        'inflection_sigma': np.float64(inflection_sigma if inflection_sigma is not None else float('nan')),
        'flip_u': np.bool_(prepared.flip_u),
        'boundary_method': np.array(results.boundary_method, dtype=object),
    }

    for gtype in prepared.results.keys():
        slim_heatmap, n_touches, threshold = prepared.results[gtype]
        grid_u, grid_v, grid_z = prepared.per_gesture_grids[gtype]
        data_dict[f'heatmap_{gtype}'] = slim_heatmap.astype(np.float64)
        data_dict[f'heatmap_pre_threshold_{gtype}'] = prepared.per_gesture_slim_raw[gtype].astype(np.float64)
        data_dict[f'unique_count_{gtype}'] = prepared.per_gesture_slim_unique_count[gtype].astype(np.int64)
        data_dict[f'n_touches_{gtype}'] = np.int64(n_touches)
        data_dict[f'threshold_{gtype}'] = np.int64(threshold)
        data_dict[f'grid_u_{gtype}'] = grid_u.astype(np.float64)
        data_dict[f'grid_v_{gtype}'] = grid_v.astype(np.float64)
        data_dict[f'grid_z_{gtype}'] = grid_z.astype(np.float64)

        if inflection_sigma is not None:
            data_dict[f'smoothed_{gtype}'] = results.per_gesture_smoothed[gtype].astype(np.float64)
            data_dict[f'laplacian_{gtype}'] = results.per_gesture_laplacian[gtype].astype(np.float64)

        boundary = results.gesture_boundaries.get(gtype)
        if boundary is not None:
            _save_boundary_fields(data_dict, 'boundary', gtype, boundary,
                                  forearm_uv, forearm_faces, forearm_V)

    # ---- Inflection boundary (explicit prefix) ----
    if results.gesture_inflection_boundaries:
        for gtype in prepared.results.keys():
            infl_boundary = results.gesture_inflection_boundaries.get(gtype)
            if infl_boundary is not None:
                _save_boundary_fields(data_dict, 'inflection', gtype, infl_boundary,
                                      forearm_uv, forearm_faces, forearm_V)

    # ---- Gradient ridge boundary ----
    if results.gesture_gradient_boundaries:
        for gtype in prepared.results.keys():
            grad_boundary = results.gesture_gradient_boundaries.get(gtype)
            if inflection_sigma is not None and f'smoothed_{gtype}' in data_dict:
                data_dict[f'gradient_mag_{gtype}'] = results.per_gesture_gradient_mag[gtype].astype(np.float64)
            if grad_boundary is not None:
                _save_boundary_fields(data_dict, 'gradient', gtype, grad_boundary,
                                      forearm_uv, forearm_faces, forearm_V)

    # ---- Radial foot (snapped) boundary ----
    if results.gesture_radial_boundaries:
        for gtype in prepared.results.keys():
            rb = results.gesture_radial_boundaries.get(gtype)
            if rb is not None:
                data_dict[f'radial_lmax_{gtype}'] = rb.lmax.astype(np.float64)
                _save_boundary_fields(data_dict, 'radial', gtype, rb,
                                      forearm_uv, forearm_faces, forearm_V)

    data_dict['alignment_center_uv'] = prepared.alignment_center.astype(np.float64)
    data_dict['alignment_rotation_matrix'] = prepared.alignment_rotation_matrix.astype(np.float64)
    data_dict['alignment_rotation_deg'] = np.float64(prepared.alignment_angle_deg)

    if prepared.slim_vertex_colors is not None:
        data_dict['slim_vertex_colors'] = prepared.slim_vertex_colors.astype(np.float64)

    np.savez(npz_path, **data_dict)
    logger.info("[Population Response Fields] %s: saved response fields NPZ → %s", session_id, npz_path.name)
    return npz_path


# ---------------------------------------------------------------------------
# Persistence: JSON sentinel
# ---------------------------------------------------------------------------

def write_boundary_sentinel(
    sentinel: Path,
    session_id: str,
    produced: List[Path],
    inflection_boundaries: dict | None = None,
    vertex_data_npz: Path | None = None,
) -> None:
    serialized_boundaries = {}
    if inflection_boundaries:
        for gtype, b in inflection_boundaries.items():
            serialized_boundaries[gtype] = inflection_boundary_to_dict(b) if b is not None else None

    data = {
        'session_id': session_id,
        'n_pngs': len(produced),
        'pngs': [str(p) for p in produced],
    }
    if serialized_boundaries:
        data['inflection_boundaries'] = serialized_boundaries
    if vertex_data_npz is not None:
        data['vertex_data_npz'] = str(vertex_data_npz)

    with open(sentinel, 'w') as f:
        json.dump(data, f, indent=2)
