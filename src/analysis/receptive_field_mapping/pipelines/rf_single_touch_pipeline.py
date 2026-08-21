"""Per-touch receptive field mapping pipeline.

For each session, reads all single touches from the prepared CSV via
``load_playback_data()``, accumulates per-vertex neuron values (IFF or spike)
using the same pattern as the Touch Playback Explorer, and saves two sparse
``.npz`` files per session: ``single_touch_rf_maps_mean.npz`` (per-vertex
mean IFF) and ``single_touch_rf_maps_max.npz`` (per-vertex max IFF). Both
accumulators run in a single pass for efficiency.

The **mean is depth-weighted**: each contact point is credited in proportion to
how fully it was pressed relative to the deepest point of its own frame, raised
to ``depth_weight_alpha``. The **max is not weighted**, because a weighted
maximum has no meaning. ``depth_weight_alpha = 0`` reproduces the unweighted
maps exactly, through the same code path with every weight equal to ``1.0``.
Both ``.npz`` files also carry the confidence channel — per-vertex ``weight_sum``
and Kish ``n_eff`` — which display code consumes and which never alters the
estimate.

Depends on: ``touch_prepare_sessions`` (reads ``<session>_prepared.csv``).
"""

import json
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.data.vertex_accumulator import (
    accumulate_vertex_values_into,
    empty_accumulator,
)
from analysis.receptive_field_mapping.data.touch_frame_weights import (
    touch_frame_weights,
    touch_label as _touch_label,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (
    PLAYBACK_CACHE_SCHEMA_VERSION,
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


class TouchRFMaps(NamedTuple):
    """One touch's per-vertex results: two estimates and their confidence channel.

    All four lists cover the **same vertices, in the same order**, so they can be
    zipped positionally. The confidence channel describes the evidence behind
    ``mean_pairs``; it never modifies it.
    """

    mean_pairs: List[Tuple[int, float]]
    max_pairs: List[Tuple[int, float]]
    weight_sum_pairs: List[Tuple[int, float]]
    n_eff_pairs: List[Tuple[int, float]]


def _compute_touch_rf(
    touch: TouchEvent,
    n_vertices: int,
    neuron_mode: str,
    depth_weight_alpha: float,
) -> TouchRFMaps:
    """Compute depth-weighted mean and unweighted max RF maps for one touch event.

    The mean is a **weighted mean**::

        value[v] = sum_frames(w * neuron_value) / sum_frames(w)

    where ``w`` is how fully vertex ``v`` was pressed relative to the deepest
    point of that same frame (see ``data.vertex_weights``). The denominator is
    the **weight sum**, never the frame count: dividing by the count would leave
    the weight in the answer as a scale factor and the number would stop being a
    firing rate. A vertex touched three times at 100 Hz must report 100 Hz
    whatever its weights were.

    The max is **unweighted**, deliberately and permanently. A weighted maximum
    has no meaning — scaling a sample by ``w`` does not make it "less of a
    maximum", it makes it a different number in different units. The max map
    answers "what is the strongest response ever seen at this vertex", which is
    a statement about the observed values themselves. The mean map is weighted;
    the max map is not; that inconsistency is intentional rather than an
    oversight, which is why it is written here.

    ``depth_weight_alpha = 0`` reproduces the unweighted behaviour **exactly**,
    because every weight is then exactly ``1.0`` and this same code path runs.
    There is no ``if alpha == 0`` branch: a branch would test the old code
    instead of proving that the new code reduces to it.

    ``touch.frame_vertex_indices`` holds the ``vertex_id`` recorded by the
    contact-depth-field sidecar for each contact point, not a nearest-vertex
    answer computed here, and ``touch.frame_depths`` holds ``signed_depth_mm``
    off the same rows — negative = penetrating. The sign is flipped once, by
    ``penetration_from_signed_mm``.

    Parameters
    ----------
    touch:
        ``TouchEvent`` from ``load_playback_data()``.
    n_vertices:
        Total number of vertices in the forearm PLY mesh (for accumulator size).
    neuron_mode:
        ``"iff"`` → use ``frame_iff``; ``"spike"`` → use ``frame_spikes``.
    depth_weight_alpha:
        Depth-weighting exponent, **required** — there is no default at any
        level, so an un-passed alpha is a ``TypeError`` rather than a silent
        ``1.0``.

    Returns
    -------
    ``TouchRFMaps`` — mean, max, ``weight_sum`` and Kish ``n_eff`` as
    ``(vertex_idx, value)`` pairs over the same vertices in the same order.
    Vertices where all frames are NaN are excluded from all four lists.

    Raises
    ------
    ValueError
        A frame's depth array does not align with its vertex array, or a frame is
        entirely grazing (``d_max == 0``) — both raised by ``touch_frame_weights``,
        the conversion this shares with the playback viewer; or a contacted vertex
        ends with ``sum(w) == 0``.
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

    touched: List[np.ndarray] = []
    for fi in range(n_frames):
        verts = touch.frame_vertex_indices[fi]
        if len(verts) == 0:
            continue
        # Shared with ``gui.touch_playback_explorer``: the screen and the saved
        # ``.npz`` must be produced by the same depth -> weight conversion, not by
        # two spellings of it that can drift apart.
        weights = touch_frame_weights(touch, fi, depth_weight_alpha)
        # ``neuron_values[fi]`` is a scalar — a frame has exactly one
        # instantaneous firing frequency however many vertices it touched. The
        # accumulator broadcasts it across the frame and multiplies by the
        # per-point weights, so the credited product is a ``(K_i,)`` array.
        accumulate_vertex_values_into(accum, verts, neuron_values[fi], weights)
        touched.append(np.asarray(verts))

    val_sum = accum.value_sum
    val_max = accum.value_max
    weight_sum = accum.weight_sum
    weight_sq_sum = accum.weight_sq_sum

    if not touched:
        return TouchRFMaps([], [], [], [])

    # The contacted set comes from the vertex lists, not from ``weight_sum > 0``.
    # Those two agree at alpha = 0 but part company above it: a vertex that was
    # genuinely contacted, but only ever grazingly, accumulates zero total weight.
    # That is a 0/0 in the estimator and it must raise rather than reach the map
    # as NaN — and it must be distinguishable from a vertex that was simply never
    # touched, which is not an error at all.
    contacted_indices = np.unique(np.concatenate(touched))

    starved = contacted_indices[weight_sum[contacted_indices] <= 0.0]
    if starved.size:
        raise ValueError(
            f"_compute_touch_rf: touch {_touch_label(touch)}: vertex/vertices "
            f"{starved.tolist()[:10]} were contacted but accumulated a total "
            f"weight of 0 at depth_weight_alpha={depth_weight_alpha!r}, so their "
            f"weighted mean is 0/0. Every frame that touched them was grazing. "
            f"There is no fallback to the unweighted mean: that would silently "
            f"mix two different estimators in one map."
        )

    mean_values = val_sum[contacted_indices] / weight_sum[contacted_indices]
    max_values = val_max[contacted_indices]
    weight_sum_values = weight_sum[contacted_indices]
    # Kish effective sample size. Well defined wherever ``weight_sum > 0``,
    # because a positive weight has a positive square. Flat weights give
    # ``n_eff`` equal to the number of contributing frames however small those
    # weights are — being shallow costs no evidence, only being *inconsistently*
    # shallow does.
    n_eff_values = (weight_sum_values * weight_sum_values) / weight_sq_sum[contacted_indices]

    # Drop vertices whose mean is NaN. NaN propagates through ``np.add.at`` once
    # any frame's ``neuron_values`` is NaN (e.g. unit not held during the touch
    # window — see ST13-03 blocks 5–8). Keeping NaN pairs would yield invisible
    # heatmaps but a non-zero vertex count, masking the "no neural data" state.
    # The same NaN mask is applied to max values: if mean is NaN then all frames
    # at that vertex were NaN, so max is also meaningless. The confidence channel
    # is masked with it too, so all four lists stay vertex-aligned.
    valid = ~np.isnan(mean_values)
    if not valid.any():
        return TouchRFMaps([], [], [], [])
    contacted_indices = contacted_indices[valid]
    mean_values = mean_values[valid]
    max_values = max_values[valid]
    weight_sum_values = weight_sum_values[valid]
    n_eff_values = n_eff_values[valid]
    return TouchRFMaps(
        mean_pairs=[(int(i), float(v)) for i, v in zip(contacted_indices, mean_values)],
        max_pairs=[(int(i), float(v)) for i, v in zip(contacted_indices, max_values)],
        weight_sum_pairs=[
            (int(i), float(v)) for i, v in zip(contacted_indices, weight_sum_values)
        ],
        n_eff_pairs=[(int(i), float(v)) for i, v in zip(contacted_indices, n_eff_values)],
    )


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
    depth_weight_alpha: float,
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
    depth_weight_alpha:
        Depth-weighting exponent handed to ``vertex_weights``.  **Required and
        positional** — no level of this call chain supplies a default, so an
        un-passed alpha is a ``TypeError`` rather than a silently-assumed value.
        ``0.0`` reproduces the unweighted maps exactly.
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
        rf_weight_sum: dict = {}
        rf_n_eff: dict = {}
        incremental_id = 0
        total_touches = 0

        for block_id in playback.block_order_ids:
            for trial_id in playback.trial_ids_by_block[block_id]:
                for touch in playback.touches_by_block_trial[(block_id, trial_id)]:
                    key = (touch.block_order_id, touch.trial_id, touch.single_touch_id)
                    touch_id_map[key] = incremental_id
                    maps = _compute_touch_rf(
                        touch, n_vertices, neuron_mode, depth_weight_alpha
                    )
                    rf_data_mean[incremental_id] = maps.mean_pairs
                    rf_data_max[incremental_id] = maps.max_pairs
                    rf_weight_sum[incremental_id] = maps.weight_sum_pairs
                    rf_n_eff[incremental_id] = maps.n_eff_pairs
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
            # ``rf_weight_sum`` / ``rf_n_eff`` are the confidence channel: they
            # describe the evidence behind each estimate and never modify it.
            # They are vertex-aligned with this file's own ``rf_data``, so a
            # viewer can dim thin-evidence vertices on whichever map it draws.
            # ``depth_weight_alpha`` travels with them because a weight sum is
            # uninterpretable without the exponent that produced it.
            np.savez(
                npz_path,
                touch_id_map=touch_id_map,
                rf_data=rf_data,
                neuron_mode=neuron_mode,
                rf_weight_sum=rf_weight_sum,
                rf_n_eff=rf_n_eff,
                depth_weight_alpha=depth_weight_alpha,
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
                    # The exponent that produced the weighted means in this run.
                    # Recorded here so two runs at different alphas are
                    # distinguishable on disk, and so a config change makes the
                    # sentinel disagree with the config that would produce it.
                    'depth_weight_alpha': float(depth_weight_alpha),
                    # Which playback-cache layout the vertex identities and depths
                    # behind these maps were read through. A cache written under an
                    # older layout is rejected rather than reused, so a summary
                    # naming a superseded version identifies maps that predate a
                    # loader change.
                    'playback_cache_schema_version': int(
                        PLAYBACK_CACHE_SCHEMA_VERSION
                    ),
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
