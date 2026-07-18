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

import numpy as np

from analysis.pipeline.shared_constants import session_id_from_path
from analysis.receptive_field_mapping.data.rf_boundary_io import (
    boundary_method_output_dir,
    boundary_sentinel_path,
    boundary_session_output_dir,
    load_boundary_inputs,
    load_param_free_fields_npz,
    resolve_boundary_input_paths,
    resolve_response_fields_npz,
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
from analysis.receptive_field_mapping.data.rf_contour_params_io import (
    contour_params_path,
    load_gesture_contour_params,
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
    *,
    boundary_method: str,
    method_params: dict,
    min_overlap_pct: float = 25.0,
    force_processing: bool = False,
    median_filter_size: int | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    iff_metric: str = "mean",
    flip_u: bool = False,
    contour_color: str = "red",
    circular_crop_margin: float = 0.0,
    contour_params_dir: Path | None = None,
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    Runs exactly ONE boundary method (``boundary_method``) across every session.
    For each session it produces one PNG per gesture subset (all, stroke, tap,
    stroke_proximal, stroke_distal) plus two composite PNGs (scatter and
    interpolated) under
    ``4_analysed/spatial_extract_boundaries/iff_<metric>/<session_id>/<method>/``.
    ``stroke`` is a virtual subset combining stroke_proximal + stroke_distal.

    Composites use a global colour scale and UV axis range across all sessions
    so they are directly comparable.

    The per-method run directory (``<session>/<method>/``) holds this run's
    figures, monolithic response-fields NPZ and staleness sentinel; the boundary
    contour NPZ itself is written to the same folder by the IO layer (keyed off the
    session root), so the discovery reader ``load_boundary_contours(<session>)``
    finds each method regardless of which method this call computed.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used by
        ``run_single_touch_rf_mapping``.
    output_dir:
        Root output directory for this task
        (e.g. ``database_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES / iff_<metric>``).
    boundary_method:
        Registry name of the single method to run (``registry.get_method`` resolves
        it, fail-fast on an unknown name).
    method_params:
        Generic ``{schema_key: value}`` mapping for the active method's declared
        ``params_schema`` (the fan-out flow reads it from the method node's DAG
        options). Only the keys the active method declares are forwarded; a plain
        ``method_params.get('inflection_sigma')`` also seeds the monolithic NPZ
        scalar (``None`` for methods that do not declare it).
    min_overlap_pct:
        Minimum percentage of touches that must contact a vertex for it to be
        included in the heatmap (default 25 %).
    force_processing:
        If True, reprocess sessions even when the sentinel file exists.
    iff_metric:
        Which IFF aggregation NPZ to consume — ``"mean"`` (default) or
        ``"max"``.  Must be one of ``IFF_METRICS``.
    flip_u:
        If True, negate the U-axis (column 0) of the aligned forearm UV
        coordinates after PCA alignment. Mirrors the heatmap and all
        boundary metrics along the vertical axis of the output plots.
    contour_params_dir:
        Root of the per-(session, gesture) tuned contour-params JSONs
        (``contour_params_root(database_path, iff_metric)``) when
        ``use_tuned_params`` is on; ``None`` when off. When set, each session
        loads its per-gesture params ``strict``ly (raising, naming the combo, if
        any required JSON is absent) and applies them in place of the global
        scalars; when ``None`` the global BoundaryParams scalars are used
        unchanged (byte-identical to the pre-tuning pipeline).
    """
    validate_boundary_params(iff_metric, boundary_method)

    # Algorithm-specific knobs no longer live on BoundaryParams — they flow into
    # the selected method via its declared ``params_schema`` as the generic
    # ``method_params`` mapping. ``extract_session_boundaries`` forwards only the
    # keys the active method declares (so a radial-only knob is dropped for a
    # non-radial method). This function is now method-blind: the *caller* (the
    # per-method fan-out flow) chooses ``boundary_method`` and supplies the matching
    # ``method_params`` read generically from that method node's DAG options.
    if method_params is None:
        raise ValueError(
            "run_population_response_field_extraction: method_params must be a "
            "mapping of the active method's schema params (got None)."
        )

    # ``inflection_sigma`` survives on BoundaryParams only as the monolithic
    # response-fields NPZ scalar — read it generically from method_params (present
    # only when the inflection method is active; ``None`` otherwise). No branch on
    # method identity: this is a plain dict lookup.
    inflection_sigma = method_params.get("inflection_sigma")

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
    )

    # ---- Phase A: per-session verify -> load -> prepare -> process -> persist ----
    bundles: List[_SessionBundle] = []

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        # Session root owns the per-method contour folders (<session>/<method>/),
        # discovered downstream via load_boundary_contours(<session>). Everything
        # this run writes (figures, monolithic NPZ, sentinel) lives under the
        # per-method run dir so concurrent methods never overwrite each other and
        # each method's staleness gate is independent.
        session_root_dir = boundary_session_output_dir(output_dir, session_id)
        session_output_dir = boundary_method_output_dir(session_root_dir, boundary_method)
        sentinel = boundary_sentinel_path(session_output_dir, session_id)

        input_paths = resolve_boundary_input_paths(
            csv_path, database_path, session_id, iff_metric,
        )

        # --- Shared param-free response fields (spatial_build_response_fields) ---
        # The param-free per-gesture arrays + mesh are materialised once upstream
        # and consumed here (single source of truth). Fail-fast if the upstream
        # task has not run for this session.
        response_fields_npz = resolve_response_fields_npz(database_path, session_id)
        if not response_fields_npz.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: shared response-fields "
                f"NPZ missing: {response_fields_npz}. Enable "
                "'spatial_build_response_fields' in the DAG config and re-run."
            )

        # --- Per-(session, gesture) tuned contour params (use_tuned_params) ---
        # Off (contour_params_dir is None): per_gesture_params stays None and the
        # stage reads the global scalars unchanged. On: enumerate the session's
        # gesture subsets from the shared response-fields NPZ — the same combo set
        # the tuner GUI writes JSONs for — and load them strictly.
        per_gesture_params = None
        override_json_paths: List[Path] = []
        if contour_params_dir is not None:
            with np.load(response_fields_npz, allow_pickle=True) as d:
                gesture_keys = [str(g) for g in d["gesture_types"]]
            per_gesture_params = load_gesture_contour_params(
                contour_params_dir, session_id, gesture_keys, params, strict=True,
            )
            # strict=True guarantees each JSON exists; feed them to the staleness
            # gate so editing a tuned JSON marks the stage stale (mtime gate).
            override_json_paths = [
                contour_params_path(contour_params_dir, session_id, gtype)
                for gtype in gesture_keys
            ]

        # The shared response-fields NPZ is a staleness input: regenerating it
        # (newer mtime) marks the boundary stage stale and forces a re-run.
        if boundary_stage_is_up_to_date(
            input_paths, sentinel, force_processing,
            extra_input_paths=[response_fields_npz] + override_json_paths,
        ):
            print(f"[Population Response Fields] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Population Response Fields] {session_id}: processing...")

        inputs = load_boundary_inputs(
            session_id, session_output_dir, sentinel, input_paths,
        )
        # Consume the param-free per-gesture fields from the shared NPZ (byte-
        # identical to recomputing them from `inputs`). SLIM vertex colours + the
        # PCA/interpolation mesh are still derived from `inputs` inside prepare.
        param_free = load_param_free_fields_npz(response_fields_npz)
        prepared = prepare_session_boundary_data(
            inputs, params, per_gesture_params, param_free=param_free,
        )

        if prepared is None:
            session_output_dir.mkdir(parents=True, exist_ok=True)
            write_boundary_sentinel(sentinel, session_id, produced=[])
            continue

        session_output_dir.mkdir(parents=True, exist_ok=True)
        (session_output_dir / "aggregated").mkdir(parents=True, exist_ok=True)
        inspection_dir = session_output_dir / "inspection"
        inspection_dir.mkdir(parents=True, exist_ok=True)

        results = extract_session_boundaries(prepared, params, method_params=method_params)
        vertex_data_npz = save_boundary_outputs_npz(
            prepared, results, params, session_output_dir=session_root_dir,
        )
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
            vertex_data_npz=vertex_data_npz,
        )
