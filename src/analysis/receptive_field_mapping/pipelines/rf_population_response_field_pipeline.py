"""Per-session pipeline that renders population RF heatmap PNGs via SLIM UV.

Thin orchestrator for the ``spatial_extract_boundaries`` stage. It wires the
five concern-separated layers and owns the cross-session barrier:

    verify   -> pipelines.rf_boundary_verification
    load     -> data.rf_boundary_io          (BoundaryInputs)
    prepare  -> data.rf_boundary_preparation (PreparedBoundaryData)
    process  -> metrics.rf_boundary_extraction (BoundaryResults)
    render   -> rendering.rf_boundary_render
    persist  -> data.rf_boundary_io          (NPZ + sentinel)

Phase A runs verify -> load -> prepare -> process -> persist NPZ + per-gesture
render for each session, accumulating the results. Phase B renders the
cross-session composites on a shared global colour scale + UV extent (the only
step that genuinely needs every session at once) and writes the final sentinels.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from analysis.pipeline.shared_constants import session_id_from_path
from analysis.receptive_field_mapping.data.rf_boundary_io import (
    boundary_sentinel_path,
    boundary_session_output_dir,
    load_boundary_inputs,
    resolve_boundary_input_paths,
    save_boundary_outputs_npz,
    write_boundary_sentinel,
)
from analysis.receptive_field_mapping.data.rf_boundary_preparation import (
    prepare_session_boundary_data,
)
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.metrics.rf_boundary_extraction import (
    extract_session_boundaries,
)
from analysis.receptive_field_mapping.pipelines.rf_boundary_verification import (
    boundary_stage_is_up_to_date,
    validate_boundary_params,
)
from analysis.receptive_field_mapping.rendering.rf_boundary_render import (
    render_global_composites,
    render_session_figures,
)

logger = logging.getLogger(__name__)

# (prepared, results, produced-paths, response-fields NPZ path) accumulated in
# Phase A and consumed by the Phase-B composite pass + final sentinel write.
_SessionBundle = Tuple[PreparedBoundaryData, BoundaryResults, List[Path], Path]


def run_population_response_field_extraction(
    session_configs: list,
    neuron_mode: str,
    output_dir: Path,
    min_overlap_pct: float = 25.0,
    force_processing: bool = False,
    median_filter_size: int | None = None,
    inflection_sigma: float | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    iff_metric: str = "mean",
    flip_u: bool = False,
    contour_color: str = "red",
    circular_crop_margin: float = 0.0,
    boundary_method: str = "gradient",
    radial_gauss_sigma: float = 8.0,
    radial_hess_sigma: float = 5.0,
    radial_envelope_smooth_sigma: float = 1.5,
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session config, produces one PNG per gesture subset (all, stroke,
    tap, stroke_proximal, stroke_distal) plus two composite PNGs (scatter and
    interpolated) under ``4_analysed/spatial_extract_boundaries/{session_id}/``.
    ``stroke`` is a virtual subset combining stroke_proximal + stroke_distal.

    Composites use a global colour scale and UV axis range across all sessions
    so they are directly comparable.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used by
        ``run_single_touch_rf_mapping``.
    output_dir:
        Root output directory for this task
        (e.g. ``database_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES``).
    min_overlap_pct:
        Minimum percentage of touches that must contact a vertex for it to be
        included in the heatmap (default 25 %).
    force_processing:
        If True, reprocess sessions even when the sentinel file exists.
    inflection_sigma:
        Gaussian smoothing sigma for Laplacian inflection boundary detection.
        Pass ``None`` to disable boundary computation entirely.
    iff_metric:
        Which IFF aggregation NPZ to consume — ``"mean"`` (default) or
        ``"max"``.  Must be one of ``IFF_METRICS``.
    flip_u:
        If True, negate the U-axis (column 0) of the aligned forearm UV
        coordinates after PCA alignment. Mirrors the heatmap and all
        boundary metrics along the vertical axis of the output plots.
    """
    validate_boundary_params(iff_metric, boundary_method)

    params = BoundaryParams(
        neuron_mode=neuron_mode,
        min_overlap_pct=min_overlap_pct,
        median_filter_size=median_filter_size,
        inflection_sigma=inflection_sigma,
        heatmap_space=heatmap_space,
        cmap=cmap,
        iff_metric=iff_metric,
        flip_u=flip_u,
        contour_color=contour_color,
        circular_crop_margin=circular_crop_margin,
        boundary_method=boundary_method,
        radial_gauss_sigma=radial_gauss_sigma,
        radial_hess_sigma=radial_hess_sigma,
        radial_envelope_smooth_sigma=radial_envelope_smooth_sigma,
    )

    # ---- Phase A: per-session verify -> load -> prepare -> process -> persist ----
    bundles: List[_SessionBundle] = []

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        session_output_dir = boundary_session_output_dir(output_dir, session_id)
        sentinel = boundary_sentinel_path(session_output_dir, session_id)

        input_paths = resolve_boundary_input_paths(
            csv_path, database_path, session_id, iff_metric,
        )
        if boundary_stage_is_up_to_date(input_paths, sentinel, force_processing):
            print(f"[Population Response Fields] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Population Response Fields] {session_id}: processing...")

        inputs = load_boundary_inputs(
            session_id, session_output_dir, sentinel, input_paths,
        )
        prepared = prepare_session_boundary_data(inputs, params)

        if prepared is None:
            session_output_dir.mkdir(parents=True, exist_ok=True)
            write_boundary_sentinel(sentinel, session_id, produced=[])
            continue

        session_output_dir.mkdir(parents=True, exist_ok=True)
        (session_output_dir / "aggregated").mkdir(parents=True, exist_ok=True)
        inspection_dir = session_output_dir / "inspection"
        inspection_dir.mkdir(parents=True, exist_ok=True)

        results = extract_session_boundaries(prepared, params, inspection_dir)
        vertex_data_npz = save_boundary_outputs_npz(prepared, results, params)
        produced = render_session_figures(prepared, results, params)
        print(
            f"[Population Response Fields] {session_id}: done — {len(produced)} PNG(s) written."
        )

        bundles.append((prepared, results, produced, vertex_data_npz))

    # ---- Phase B: cross-session composites (global colour scale + UV limits) ----
    if not bundles:
        return

    render_global_composites(
        [(prepared, results, produced) for prepared, results, produced, _ in bundles],
        params,
    )

    for prepared, results, produced, vertex_data_npz in bundles:
        write_boundary_sentinel(
            prepared.sentinel, prepared.session_id, produced=produced,
            inflection_boundaries=results.gesture_boundaries,
            vertex_data_npz=vertex_data_npz,
        )
