"""Per-touch receptive field mapping pipeline.

For each session, reads all single touches from the prepared CSV via
``load_playback_data()``, accumulates per-vertex neuron values (IFF or spike)
using the same pattern as the Touch Playback Explorer, and saves two sparse
``.npz`` files per session: ``single_touch_rf_maps_mean.npz`` (per-vertex
mean IFF) and ``single_touch_rf_maps_max.npz`` (per-vertex max IFF). Both
accumulators run in a single pass for efficiency.

Depends on: ``touch_prepare_sessions`` (reads ``<session>_prepared.csv``).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.data.vertex_accumulator import (
    accumulate_vertex_values_into,
    empty_accumulator,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (
    PlaybackData,
    TouchEvent,
    load_playback_data,
)
from analysis.pipeline.shared_constants import (
    session_id_from_path,
    NEURON_MODES,
    single_touch_npz_filename,
)
from _vendor.should_process_task import should_process_task

logger = logging.getLogger(__name__)

_VALID_NEURON_MODES = NEURON_MODES  # re-exported for backward compatibility


def _compute_touch_rf(
    touch: TouchEvent,
    n_vertices: int,
    neuron_mode: str,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Compute mean and max RF maps for a single touch event.

    Accumulates per-vertex neuron values through the shared reduction in
    ``data.vertex_accumulator`` (sum for the mean, running max for the max),
    and returns only vertices that were contacted at least once as
    ``(vertex_idx, value)`` pairs.

    Every contact point currently carries weight ``1.0``: a frame's neuron
    value is credited in full to every vertex it touched, however deeply. The
    weights are passed explicitly rather than assumed so that the reduction has
    no notion of an "unweighted" default.

    ``touch.frame_vertex_indices`` now holds the ``vertex_id`` recorded by the
    contact-depth-field sidecar for each contact point, not a nearest-vertex
    answer computed here. Credit therefore lands on the vertex the producing
    pipeline measured, which is a different vertex from the one the retired
    KDTree picked at patch boundaries.

    Parameters
    ----------
    touch:
        ``TouchEvent`` from ``load_playback_data()``.
    n_vertices:
        Total number of vertices in the forearm PLY mesh (for accumulator size).
    neuron_mode:
        ``"iff"`` → use ``frame_iff``; ``"spike"`` → use ``frame_spikes``.

    Returns
    -------
    Tuple of ``(mean_pairs, max_pairs)``, each a list of
    ``(vertex_idx, value)`` pairs for all contacted vertices with valid data.
    Vertices where all frames are NaN are excluded from both lists.
    """
    accum = empty_accumulator(n_vertices)

    n_frames = len(touch.frame_vertex_indices)
    if neuron_mode == "iff":
        neuron_values = touch.frame_iff
    elif neuron_mode == "spike":
        neuron_values = touch.frame_spikes.astype(np.float64)
    else:
        raise ValueError(
            f"_compute_touch_rf: unknown neuron_mode {neuron_mode!r}. "
            f"Expected one of {_VALID_NEURON_MODES}."
        )

    for fi in range(n_frames):
        verts = touch.frame_vertex_indices[fi]
        if len(verts) == 0:
            continue
        accumulate_vertex_values_into(
            accum,
            verts,
            neuron_values[fi],
            np.ones(len(verts), dtype=np.float64),
        )

    val_sum = accum.value_sum
    val_max = accum.value_max
    contact_count = accum.weight_sum

    contacted_mask = contact_count > 0
    if not contacted_mask.any():
        return [], []

    contacted_indices = np.where(contacted_mask)[0]
    mean_values = val_sum[contacted_indices] / contact_count[contacted_indices]
    max_values = val_max[contacted_indices]

    # Drop vertices whose mean is NaN. NaN propagates through ``np.add.at`` once
    # any frame's ``neuron_values`` is NaN (e.g. unit not held during the touch
    # window — see ST13-03 blocks 5–8). Keeping NaN pairs would yield invisible
    # heatmaps but a non-zero vertex count, masking the "no neural data" state.
    # The same NaN mask is applied to max values: if mean is NaN then all frames
    # at that vertex were NaN, so max is also meaningless.
    valid = ~np.isnan(mean_values)
    if not valid.any():
        return [], []
    contacted_indices = contacted_indices[valid]
    mean_values = mean_values[valid]
    max_values = max_values[valid]
    mean_pairs = [(int(idx), float(val)) for idx, val in zip(contacted_indices, mean_values)]
    max_pairs = [(int(idx), float(val)) for idx, val in zip(contacted_indices, max_values)]
    return mean_pairs, max_pairs


# Config keys that say where a session's contact-depth-field sidecars live. Both are
# required: contact points take their vertex identity from those sidecars, so there is
# no run without them and no default that could quietly pick the wrong stage.
_BLOCKS_STAGE_DIR_KEY = "blocks_stage_dir"
_BLOCK_CSV_STEM_SUFFIX_KEY = "block_csv_stem_suffix"


def _require_depth_field_config(contact_depth_field: Optional[dict]) -> Tuple[str, str]:
    """Validate the ``contact_depth_field`` option block and return its two values.

    Raises ``ValueError`` when the block is absent or either key is missing. There is
    deliberately no default: guessing a blocks stage would silently decide which
    vertices every contact point is credited to.
    """
    if not isinstance(contact_depth_field, dict):
        raise ValueError(
            "run_single_touch_rf_mapping: 'contact_depth_field' options block is "
            f"required, got {contact_depth_field!r}. Add it under "
            "tasks.spatial_map_single_touch.options in the processing DAG config with "
            f"the keys {_BLOCKS_STAGE_DIR_KEY!r} and {_BLOCK_CSV_STEM_SUFFIX_KEY!r}."
        )
    missing = [
        key for key in (_BLOCKS_STAGE_DIR_KEY, _BLOCK_CSV_STEM_SUFFIX_KEY)
        if key not in contact_depth_field
    ]
    if missing:
        raise ValueError(
            f"run_single_touch_rf_mapping: 'contact_depth_field' is missing key(s) "
            f"{missing}. Present keys: {sorted(contact_depth_field)}."
        )
    blocks_stage_dir = str(contact_depth_field[_BLOCKS_STAGE_DIR_KEY])
    if not blocks_stage_dir:
        raise ValueError(
            f"run_single_touch_rf_mapping: {_BLOCKS_STAGE_DIR_KEY!r} is empty."
        )
    return blocks_stage_dir, str(contact_depth_field[_BLOCK_CSV_STEM_SUFFIX_KEY])


def run_single_touch_rf_mapping(
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    force: bool = False,
    neuron_mode: str = "iff",
    preparation_dir: Optional[Path] = None,
    contact_depth_field: Optional[dict] = None,
) -> List[Path]:
    """Compute per-touch RF maps for all sessions and save as ``.npz`` files.

    For each session in ``input_items``:
    1. Resolves the prepared CSV from ``preparation_dir``.
    2. Loads per-touch playback data via ``load_playback_data()``.
    3. Accumulates per-vertex neuron values for every ``TouchEvent``
       (both mean and max in a single pass).
    4. Saves ``single_touch_rf_maps_mean.npz``, ``single_touch_rf_maps_max.npz``
       and ``single_touch_rf_summary.json`` (sentinel for idempotency)
       under ``output_dir / <session_id> /``.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples from the DAG runner.
    output_dir:
        Root output directory.  Session results go under
        ``output_dir / <session_id> /``.
    force:
        If ``True``, reprocess even if outputs are up-to-date.
    neuron_mode:
        ``"iff"`` (default) or ``"spike"`` — determines which signal is
        accumulated per vertex.
    preparation_dir:
        Directory containing ``<session>_prepared.csv`` files produced by
        ``touch_prepare_sessions``.  If ``None``, raises ``ValueError`` immediately
        since prepared CSVs are a hard dependency.
    contact_depth_field:
        Options block naming the blocks stage subdirectory holding the depth-field
        parquet sidecars and the stem suffix that stage uses. Required — contact
        points take their vertex identity from those sidecars.

    Returns
    -------
    List of paths to produced ``.npz`` files.
    """
    if neuron_mode not in _VALID_NEURON_MODES:
        raise ValueError(
            f"run_single_touch_rf_mapping: invalid neuron_mode {neuron_mode!r}. "
            f"Expected one of {_VALID_NEURON_MODES}."
        )

    if preparation_dir is None:
        raise ValueError(
            "run_single_touch_rf_mapping: 'preparation_dir' is required — "
            "this task depends on touch_prepare_sessions output."
        )

    blocks_stage_dir, block_csv_stem_suffix = _require_depth_field_config(
        contact_depth_field
    )

    preparation_dir = Path(preparation_dir)
    if not preparation_dir.exists():
        raise ValueError(
            f"run_single_touch_rf_mapping: preparation_dir does not exist: {preparation_dir}"
        )

    produced: List[Path] = []

    for csv_path, _ in input_items:
        session_id = session_id_from_path(csv_path)
        session_out = output_dir / session_id
        sentinel = session_out / 'single_touch_rf_summary.json'

        mean_npz_path = session_out / single_touch_npz_filename('mean')
        if not should_process_task(
            input_paths=[csv_path],
            output_paths=[sentinel],
            force=force,
        ):
            print(f"[Single-Touch RF] {session_id}: up-to-date, skipping.")
            produced.append(mean_npz_path)
            continue

        print(f"[Single-Touch RF] {session_id}: processing (neuron_mode={neuron_mode!r})...")

        # --- Resolve prepared CSV ---
        prepared_csv = preparation_dir / f'{session_id}_prepared.csv'
        if not prepared_csv.exists():
            raise ValueError(
                f"[Single-Touch RF] {session_id}: prepared CSV not found at "
                f"{prepared_csv}. Run touch_prepare_sessions first."
            )

        # --- Resolve forearm PLY ---
        forearm_ply = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply is None:
            raise ValueError(
                f"[Single-Touch RF] {session_id}: forearm PLY not found in "
                f"{csv_path.parent}. Expected '{session_id}_forearm.ply'."
            )

        # --- Resolve the depth-field blocks directory ---
        # ``csv_path.parent`` is the session's merged output root — the directory the
        # DAG resolved the aggregated session CSV from, and the same one
        # ``resolve_forearm_ply`` is given above. The stage subdirectory under it comes
        # from config; nothing here is composed from repo knowledge of the layout.
        depth_blocks_dir = csv_path.parent / blocks_stage_dir
        if not depth_blocks_dir.is_dir():
            raise ValueError(
                f"[Single-Touch RF] {session_id}: contact-depth-field blocks directory "
                f"not found: {depth_blocks_dir} (session merged root "
                f"{csv_path.parent}, {_BLOCKS_STAGE_DIR_KEY}={blocks_stage_dir!r})."
            )

        # --- Load playback data (CSV parsing, sidecar vertex_id join, caching) ---
        playback: PlaybackData = load_playback_data(
            series_csv_path=prepared_csv,
            forearm_ply_path=forearm_ply,
            depth_blocks_dir=depth_blocks_dir,
            block_csv_stem_suffix=block_csv_stem_suffix,
            session_id=session_id,
        )

        n_vertices = len(playback.session_data.forearm_vertices)

        # --- Iterate all touches and compute RF maps ---
        touch_id_map: dict = {}
        rf_data_mean: dict = {}
        rf_data_max: dict = {}
        incremental_id = 0
        total_touches = 0

        for block_id in playback.block_order_ids:
            for trial_id in playback.trial_ids_by_block[block_id]:
                for touch in playback.touches_by_block_trial[(block_id, trial_id)]:
                    key = (touch.block_order_id, touch.trial_id, touch.single_touch_id)
                    touch_id_map[key] = incremental_id
                    mean_pairs, max_pairs = _compute_touch_rf(touch, n_vertices, neuron_mode)
                    rf_data_mean[incremental_id] = mean_pairs
                    rf_data_max[incremental_id] = max_pairs
                    incremental_id += 1
                    total_touches += 1

        print(
            f"[Single-Touch RF] {session_id}: {total_touches} touches processed, "
            f"{n_vertices} vertices in mesh."
        )

        # --- Save .npz outputs (one per IFF metric) ---
        session_out.mkdir(parents=True, exist_ok=True)
        for iff_metric, rf_data in (('mean', rf_data_mean), ('max', rf_data_max)):
            npz_path = session_out / single_touch_npz_filename(iff_metric)
            np.savez(
                npz_path,
                touch_id_map=touch_id_map,
                rf_data=rf_data,
                neuron_mode=neuron_mode,
            )
            print(f"[Single-Touch RF] {session_id}: saved -> {npz_path.name}")
        produced.append(mean_npz_path)

        # --- Save sentinel ---
        with open(sentinel, 'w') as f:
            json.dump(
                {
                    'session_id': session_id,
                    'neuron_mode': neuron_mode,
                    'n_touches': total_touches,
                    'n_vertices': n_vertices,
                    # Which depth-field sidecar supplied each block's vertex
                    # identities, and the coordinate space each one *declared* in its
                    # parquet metadata (never inferred from the directory name). A
                    # passthrough session — 'pca_calibrated' or 'kinect_space_1' in a
                    # terminal blocks directory — is legal and shows up here rather
                    # than being absorbed.
                    'contact_depth_field': {
                        'blocks_dir': str(depth_blocks_dir),
                        'block_csv_stem_suffix': block_csv_stem_suffix,
                        'blocks': [
                            {
                                'source_block_file': pv.source_block_file,
                                'sidecar_path': pv.sidecar_path,
                                'coordinate_space': pv.coordinate_space,
                            }
                            for pv in playback.depth_field_provenance
                        ],
                    },
                },
                f,
                indent=2,
            )

    return produced
