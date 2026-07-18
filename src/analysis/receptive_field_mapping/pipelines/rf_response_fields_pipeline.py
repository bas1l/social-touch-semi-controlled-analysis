"""Per-session pipeline for the ``spatial_build_response_fields`` stage.

Materialises the *parameter-free* per-(session, gesture) response fields once,
upstream of both the RF-contour tuner and the boundary extractor. The fields are
computed purely from upstream artifacts (the single-touch RF NPZ + the SLIM UV
cache + the merged CSV/PLY) using only production functions
(``compute_rf_heatmap`` / ``compute_unique_touch_count`` + the SLIM<->raw map),
so identical upstream inputs yield byte-identical fields.

    load    -> data.rf_boundary_io          (BoundaryInputs)
    build   -> data.rf_boundary_preparation (build_param_free_fields)
    persist -> ``<session>_response_fields.npz``

Output: ``4_analysed/spatial_build_response_fields/<session>/<session>_response_fields.npz``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import TOUCH_PREPARE_SESSIONS
from analysis.pipeline.shared_constants import session_id_from_path
from analysis.receptive_field_mapping.data.rf_boundary_io import (
    load_boundary_inputs,
    resolve_boundary_input_paths,
)
from analysis.receptive_field_mapping.data.rf_boundary_preparation import (
    build_param_free_fields,
)
from analysis.receptive_field_mapping.data.rf_boundary_types import BoundaryInputs
from analysis.receptive_field_mapping.data.rf_stroke_axis_io import (
    load_stroke_axis,
    stroke_axis_path,
    stroke_axis_root,
)
from analysis.receptive_field_mapping.metrics.rf_stroke_axis import (
    compute_stroke_uv_motion,
    load_slim_uv_cache,
    relabel_strokes,
)
from analysis.receptive_field_mapping.pipelines.rf_boundary_verification import (
    boundary_stage_is_up_to_date,
)

logger = logging.getLogger(__name__)


def build_session_response_fields(inputs: BoundaryInputs) -> dict:
    """Compose the param-free fields + mesh into a savable NPZ ``data_dict``.

    Keys (dtypes match ``save_boundary_outputs_npz``):

    * ``heatmap_pre_threshold_<g>`` — float64 SLIM-mapped raw heatmap per gesture,
    * ``unique_count_<g>``          — int64 SLIM-mapped unique-touch count,
    * ``n_touches_<g>``             — int64 touch count,
    * ``forearm_uv``                — float64 raw (un-aligned) SLIM UV,
    * ``forearm_faces``             — int32 SLIM faces,
    * ``forearm_V``                 — float64 SLIM vertices,
    * ``gesture_types``             — object array of gesture keys (canonical order),
    * ``session_id``                — object scalar.

    ``forearm_uv`` is the raw SLIM-cache UV (not PCA-aligned): alignment depends
    on the thresholded ``'all'`` heatmap and is therefore parameter-dependent, so
    it is applied downstream in the extractor/tuner, not here.

    Raises ``ValueError`` if no gesture subset had any touches (a degenerate
    session with zero touches — an upstream data problem).
    """
    param_free = build_param_free_fields(inputs)
    if not param_free:
        raise ValueError(
            f"[Response Fields] {inputs.session_id}: no gesture subset had any "
            "touches — cannot build response fields. Check upstream RF data."
        )

    data_dict: dict = {
        'forearm_uv': inputs.forearm_uv.astype(np.float64),
        'forearm_faces': inputs.slim_faces.astype(np.int32),
        'forearm_V': inputs.slim_V.astype(np.float64),
        'session_id': np.array(inputs.session_id, dtype=object),
        'gesture_types': np.array(list(param_free.keys()), dtype=object),
    }

    for gtype, (slim_raw, slim_unique_count, n_touches) in param_free.items():
        data_dict[f'heatmap_pre_threshold_{gtype}'] = slim_raw.astype(np.float64)
        data_dict[f'unique_count_{gtype}'] = slim_unique_count.astype(np.int64)
        data_dict[f'n_touches_{gtype}'] = np.int64(n_touches)

    return data_dict


def apply_manual_stroke_axis(
    inputs: BoundaryInputs,
    prepared_csv: Path,
    slim_cache_path: Path,
    stroke_axis_json: Path,
) -> None:
    """Overwrite ``inputs.pop_data.gesture_types`` from the manual stroke axis.

    Loads the session's SLIM-UV cache and per-frame ``_prepared.csv``, derives the
    per-stroke UV motion via :func:`compute_stroke_uv_motion`, loads the researcher's
    :class:`StrokeAxisConfig` from ``stroke_axis_json`` (drawn by the
    ``spatial_configure_stroke_axis`` GUI), and re-decides every proximal/distal
    label via :func:`relabel_strokes` — mutating ``inputs.pop_data.gesture_types``
    in place so the per-gesture NPZ heatmaps built downstream carry the correction.

    Uses the single shared ``metrics.rf_stroke_axis`` compute module (same as the
    GUI preview) so the pipeline consumer and the viewer can never drift.

    Fail-fast: raises ``FileNotFoundError`` if the prepared CSV, the SLIM cache, or
    the stroke-axis JSON is absent (no silent fallback).
    """
    prepared_csv = Path(prepared_csv)
    slim_cache_path = Path(slim_cache_path)
    stroke_axis_json = Path(stroke_axis_json)

    if not prepared_csv.exists():
        raise FileNotFoundError(
            f"apply_manual_stroke_axis: prepared CSV not found: {prepared_csv}. "
            "Run 'touch_prepare_sessions' first."
        )
    if not slim_cache_path.exists():
        raise FileNotFoundError(
            f"apply_manual_stroke_axis: SLIM UV cache not found: {slim_cache_path}."
        )

    slim_cache = load_slim_uv_cache(slim_cache_path)
    prepared_df = pd.read_csv(prepared_csv)
    motion = compute_stroke_uv_motion(slim_cache, prepared_df)
    config = load_stroke_axis(stroke_axis_json)  # fail-fast on a missing/malformed JSON

    inputs.pop_data.gesture_types = relabel_strokes(
        inputs.pop_data.gesture_types,
        inputs.pop_data.touch_triple_keys,
        motion,
        config,
    )


def run_response_field_generation(
    input_items: list,
    force_processing: bool,
    output_dir: Path,
    iff_metric: str = "mean",
    use_manual_stroke_axis: bool = False,
) -> None:
    """Generate ``<session>_response_fields.npz`` for each session.

    Mirrors ``run_population_response_field_extraction``'s per-session loop and
    ``should_process_task`` mtime skip (via ``boundary_stage_is_up_to_date``),
    reusing ``resolve_boundary_input_paths`` / ``load_boundary_inputs``. Fails
    fast (``FileNotFoundError``) on any missing upstream input.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    force_processing:
        Reprocess even when the NPZ is newer than every upstream input.
    output_dir:
        Root output directory for this task
        (``database_path / '4_analysed' / SPATIAL_BUILD_RESPONSE_FIELDS``).
    iff_metric:
        Which IFF aggregation NPZ to consume — ``"mean"`` or ``"max"``.
    use_manual_stroke_axis:
        When True, overwrite each session's proximal/distal labels from the
        manually drawn UV stroke axis (``spatial_configure_stroke_axis``) before
        building the per-gesture fields, via :func:`apply_manual_stroke_axis`. The
        per-session stroke-axis JSON is fed to ``boundary_stage_is_up_to_date`` as
        an extra staleness input so editing it re-triggers this stage. Raises
        ``FileNotFoundError`` if the toggle is on but a session's JSON is absent
        (no silent fallback to the upstream 3D labels).
    """
    for csv_path, database_path in input_items:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        session_output_dir = output_dir / session_id
        npz_path = session_output_dir / f'{session_id}_response_fields.npz'

        input_paths = resolve_boundary_input_paths(
            csv_path, database_path, session_id, iff_metric,
        )

        stroke_axis_json = stroke_axis_path(
            stroke_axis_root(database_path), session_id,
        )
        extra_input_paths: list = []
        if use_manual_stroke_axis:
            if not stroke_axis_json.exists():
                raise FileNotFoundError(
                    f"[Response Fields] {session_id}: use_manual_stroke_axis is on "
                    f"but the stroke-axis JSON is absent: {stroke_axis_json}. Run "
                    "'spatial_configure_stroke_axis' first (no silent fallback)."
                )
            extra_input_paths.append(stroke_axis_json)

        if boundary_stage_is_up_to_date(
            input_paths, npz_path, force_processing,
            extra_input_paths=extra_input_paths,
        ):
            print(f"[Response Fields] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Response Fields] {session_id}: processing...")

        inputs = load_boundary_inputs(
            session_id, session_output_dir, npz_path, input_paths,
        )

        if use_manual_stroke_axis:
            prepared_csv = (
                database_path / '4_analysed' / TOUCH_PREPARE_SESSIONS
                / f'{session_id}_prepared.csv'
            )
            # input_paths == [series_csv, forearm_ply, single_touch_npz, slim_cache].
            slim_cache_path = input_paths[3]
            apply_manual_stroke_axis(
                inputs, prepared_csv, slim_cache_path, stroke_axis_json,
            )

        data_dict = build_session_response_fields(inputs)

        session_output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, **data_dict)
        logger.info(
            "[Response Fields] %s: saved response fields NPZ → %s",
            session_id, npz_path.name,
        )
        print(f"[Response Fields] {session_id}: done — saved {npz_path.name}.")
