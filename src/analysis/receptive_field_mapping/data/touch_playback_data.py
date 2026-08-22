"""Per-touch data model and loader for the Touch Playback Explorer GUI."""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .contact_depth_field_io import (
    SIGNED_DEPTH_COLUMN,
    TERMINAL_RF_CENTERED_SPACES,
    VERTEX_ID_COLUMN,
    DepthField,
    declared_coordinate_space,
    depth_field_path_for_csv,
    load_depth_field,
)
from .rf_data_loader import load_forearm_vertex_colors, load_forearm_vertices
from analysis.pipeline.shared_constants import NERVE_SPIKE_COL, NERVE_FREQ_COL, CONTACT_POINTS_COL

logger = logging.getLogger(__name__)

# v5: every contact frame additionally carries its ``signed_depth_mm`` values, one
# per contact point, read off the same sidecar rows that supplied ``vertex_id``.
# A v4 cache has no depth at all, so reusing one would hand the weighting stage a
# depth-free touch that still looks structurally valid.
#
# v4: contact-point vertex identity is read off the contact-depth-field sidecar
# instead of re-derived by nearest-vertex snapping, and the sidecar provenance is
# stored alongside. A v3 cache holds KDTree-derived vertex indices, which are a
# different (and wrong) answer, so it must not be reused.
#
# Public, not ``_``-prefixed: ``rf_single_touch_pipeline`` records it in
# ``single_touch_rf_summary.json`` so a saved run states which cache layout its
# vertex identities and depths came off.
PLAYBACK_CACHE_SCHEMA_VERSION = 5

_FRAME_INDEX_COL = "frame_index"
_SOURCE_BLOCK_FILE_COL = "source_block_file"

_REQUIRED_COLUMNS = (
    CONTACT_POINTS_COL,
    "block_order_id",
    "trial_id",
    "single_touch_id",
    NERVE_SPIKE_COL,
    NERVE_FREQ_COL,
    "gesture_type",
    # The only exact join key the merged CSV and the depth-field parquet share.
    _FRAME_INDEX_COL,
    # The basename of the block CSV this row came from; the sidecar is resolved from
    # it, never synthesised from ``block_order_id`` (which is a zero-padded string
    # parsed by regex and may be None, and whose two spellings — ``block-order02``
    # and ``block-order-02`` — are never converted into one another upstream).
    _SOURCE_BLOCK_FILE_COL,
)

# Forward-filled together, in ONE statement — see ``load_playback_data``.
_FFILL_COLUMNS = (CONTACT_POINTS_COL, _FRAME_INDEX_COL)

_bracket_re = re.compile(r'\[([^\[\]]+)\]')


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ------------------------------------------------------------------
# Contact-point parsing
# ------------------------------------------------------------------

def _parse_contact_points_strict(cell: str) -> np.ndarray:
    """Parse one ``contact_points`` cell into an ``(K, 3)`` float64 array.

    Raises ``ValueError`` on any bracketed group that is not exactly three floats.
    A malformed triplet is **never** skipped: the k-th point of a frame is joined
    to the k-th sidecar row of that same frame (ordered correspondence), so one
    dropped point shifts every later point of the frame onto the wrong row and
    yields a plausible-looking but wrong map. The per-frame count assertion in
    ``_BlockVertexSource`` is the second line of defence, not the first.
    """
    groups = _bracket_re.findall(cell)
    pts: list[list[float]] = []
    for group in groups:
        parts = group.split()
        if len(parts) != 3:
            raise ValueError(
                f"_parse_contact_points_strict: bracketed group {group!r} has "
                f"{len(parts)} field(s), expected exactly 3 (x y z). Full cell: "
                f"{cell!r}. Dropping it would shift every later point of this frame "
                f"onto the wrong contact-depth-field row."
            )
        try:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError as exc:
            raise ValueError(
                f"_parse_contact_points_strict: bracketed group {group!r} is not "
                f"three floats ({exc}). Full cell: {cell!r}."
            ) from exc
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


# ------------------------------------------------------------------
# Contact-depth-field sidecar: the ordered-correspondence vertex source
# ------------------------------------------------------------------

class _FrameRows(NamedTuple):
    """One frame's sidecar rows, in file order, unpacked into parallel arrays.

    ``vertex_ids`` and ``signed_depth_mm`` are read off the **same rows** and are
    aligned element-for-element: element k of both belongs to the k-th contact point
    listed in that frame's CSV ``contact_points`` cell. There is no second lookup and
    no vertex-keyed mapping anywhere — a ``.get(vertex_id, ...)`` would silently
    reintroduce the matching step ordered correspondence exists to avoid.

    ``signed_depth_mm`` is the stored column **verbatim**: negative is penetrating,
    and positive values above the grazing epsilon are real grazing contacts. It is
    not negated, clamped or rescaled here — :func:`penetration_mm` is the single
    documented negation point, and it belongs to the weighting stage, not to
    transport.
    """

    vertex_ids: np.ndarray       # (K,) int64
    signed_depth_mm: np.ndarray  # (K,) float64


class _BlockVertexSource:
    """One block's ``vertex_id`` and ``signed_depth_mm``, by ``frame_index`` value.

    This is the whole of the join, and it is deliberately narrow.

    **Ordered correspondence.** Within a single ``frame_index``, the k-th point
    listed in that frame's CSV ``contact_points`` cell is the k-th parquet row
    carrying that ``frame_index``. ``frame_index`` is the only exact join key the
    two artifacts share, and the producing pipeline enforces the per-frame
    row-count agreement after every stage (``assert_row_counts_agree_with_csv``).

    **This is NOT row-position joining.** Pairing parquet row *i* with CSV row *i*
    across a whole file is unsafe and forbidden: the CSV is at nerve rate with a
    variable number of points per frame, the parquet is at frame rate. Locate the
    frame by ``frame_index`` **value** first; only then index by position *inside*
    that frame. Anyone who "generalises" the second into the first breaks the join
    silently, which is why the distinction is spelled out here and not only in the
    plan.

    Rows are held in **file order**, never sorted by ``vertex_id`` — sorting is what
    ``DepthField.frame()`` does for aggregating callers, and it would hand every
    contact point the wrong vertex *and* the wrong depth here.

    Depth travels on the same rows as vertex identity, so one join answers both
    questions. Anything that reads depth by looking a ``vertex_id`` up in a
    per-frame mapping has stopped using ordered correspondence and has quietly
    reintroduced the matching step it exists to avoid.
    """

    def __init__(self, depth_field: DepthField, source_block_file: str) -> None:
        if not depth_field.has_vertex_id:
            stage_dir = depth_field.sidecar_path.parent.name
            raise ValueError(
                f"_BlockVertexSource: sidecar {depth_field.sidecar_path} has no "
                f"{VERTEX_ID_COLUMN!r} column. Stage directory {stage_dir!r} is a "
                f"pre-projection stage — only 'blocks_projected', "
                f"'blocks_pca_calibrated' and 'blocks_rf_centered' carry "
                f"{VERTEX_ID_COLUMN!r} (schema version 2). Point the blocks-stage "
                f"config key at one of those. There is NO fallback: nearest-vertex "
                f"snapping on this data was measured overshooting 29 mm off-surface "
                f"at touch boundaries, which is exactly what this loader stopped doing."
            )
        self.source_block_file = source_block_file
        self.sidecar_path = depth_field.sidecar_path
        self.coordinate_space = depth_field.space
        self._rows_by_frame = depth_field.frame_row_positions_in_file_order()
        # int32 as the schema stores it — a whole block's column at 4 bytes a row.
        self._vertex_id = depth_field.frames[VERTEX_ID_COLUMN].to_numpy(dtype=np.int32)
        # float64 as the schema stores it, and carried verbatim: the stored value must
        # stay bit-identical to the one the CSV's ``contact_depth`` was derived from.
        self._signed_depth_mm = depth_field.frames[SIGNED_DEPTH_COLUMN].to_numpy(
            dtype=np.float64
        )
        self._n_forearm_vertices = int(len(depth_field.forearm_vertices))

    def rows_for_frame(self, frame_index: int, n_points: int) -> _FrameRows:
        """Return frame *frame_index*'s ``vertex_id`` and depth, in file order.

        *n_points* is the number of contact points parsed from the CSV for this
        frame. It **must** equal the number of sidecar rows carrying this
        ``frame_index``; a mismatch raises. That assertion is the entire reason
        ordered correspondence is safe, and it mirrors the producer's own
        ``assert_row_counts_agree_with_csv``.

        A frame absent from the sidecar has zero rows, which means *no contact in
        that frame* — legal, and it must then have zero parsed points too.

        Two further things are asserted here, at the loader boundary, because this
        is the last place where the file, the frame and the vertex are all still in
        hand:

        * every ``vertex_id`` lies inside the reference forearm;
        * no depth is ``NaN``. Zero is *not* absent — a grazing contact is a real
          measurement of 0.0 mm — and a *missing row* is already impossible by
          construction because the per-frame count assertion above has run, so NaN is
          the only remaining form of absence and it raises rather than turning into a
          zero weight indistinguishable from a grazing touch.

        **A repeated ``vertex_id`` within one frame is legal and is passed through
        unchanged.** The producer's contract permits it and the mechanism is benign:
        the XY dedup runs in ``blocks_deduped`` *before* ``vertex_id`` is assigned in
        ``blocks_projected``, so it guarantees distinct positions, not distinct
        vertices — two points further apart than ``dedup_epsilon`` can still snap to
        the same vertex where the mesh is coarser than epsilon. Both rows are returned
        and the accumulator's ``np.add.at`` sums them::

            numerator   += IFF_f * w1 + IFF_f * w2  ==  IFF_f * (w1 + w2)
            denominator += w1 + w2

        A frame's IFF is one **scalar**, shared by every contact point of that frame,
        so it factors out for *any* weights: the vertex is credited with ``IFF_f`` at
        weight ``w1 + w2``, and its weighted mean is exactly what a single row of
        weight ``w1 + w2`` would have produced. An earlier version of this loader
        raised here, on the stated premise that the cancellation held "only while
        every depth weight is 1"; that premise was false — the algebra above never
        depended on the weights — and the check was removed rather than softened.

        The consequence, which is intended: such a vertex carries **more than unit
        weight** for that frame. The weighting deliberately does not normalise per
        frame, so a frame's total weight already scales with how much of the patch it
        covers, and a vertex that caught two contact points genuinely had more finger
        on it. Note this is a statement about the *mean* only — the Kish ``n_eff``
        computed downstream does distinguish the two forms, because two rows are two
        contributions.

        Nothing is deduplicated, averaged, or collapsed deepest-wins here: each
        sidecar row is one contact point that landed on that vertex, and picking a
        survivor is a decision this loader is not entitled to make on the producer's
        behalf.
        """
        positions = self._rows_by_frame.get(int(frame_index))
        n_rows = 0 if positions is None else int(len(positions))
        if n_rows != n_points:
            raise ValueError(
                f"_BlockVertexSource: per-frame contact-point count mismatch for "
                f"frame_index={int(frame_index)}: the CSV parsed {n_points} contact "
                f"point(s) but the sidecar carries {n_rows} row(s) for that frame. "
                f"Sidecar: {self.sidecar_path}. Block CSV: {self.source_block_file}. "
                f"Ordered correspondence is only safe while these agree — a single "
                f"dropped or extra point shifts every later point of the frame onto "
                f"the wrong row."
            )
        if n_rows == 0:
            return _FrameRows(
                vertex_ids=np.empty(0, dtype=np.int64),
                signed_depth_mm=np.empty(0, dtype=np.float64),
            )
        vertex_ids = self._vertex_id[positions].astype(np.int64)
        # Same rows, same order, one indexing operation each. Depth is NOT looked up
        # by vertex_id: that would be a second join, and a wrong one.
        depths = self._signed_depth_mm[positions].astype(np.float64)
        lo = int(vertex_ids.min())
        hi = int(vertex_ids.max())
        if lo < 0 or hi >= self._n_forearm_vertices:
            raise ValueError(
                f"_BlockVertexSource: frame_index={int(frame_index)} of "
                f"{self.sidecar_path} references {VERTEX_ID_COLUMN} in [{lo}, {hi}] "
                f"but the reference forearm has only {self._n_forearm_vertices} "
                f"vertices."
            )

        nan_mask = np.isnan(depths)
        if nan_mask.any():
            bad_positions = np.flatnonzero(nan_mask)
            raise ValueError(
                f"_BlockVertexSource: {SIGNED_DEPTH_COLUMN} is NaN on "
                f"{int(nan_mask.sum())} of {len(depths)} row(s) of "
                f"frame_index={int(frame_index)}. Contact-point position(s) within "
                f"the frame: {bad_positions.tolist()}; {VERTEX_ID_COLUMN}(s): "
                f"{vertex_ids[nan_mask].tolist()}. "
                f"Sidecar: {self.sidecar_path}. Block CSV: {self.source_block_file}. "
                f"A depth of 0.0 is a grazing contact — a real measurement — while "
                f"NaN is an absent one, and the two must not collapse into the same "
                f"zero weight downstream. There is no fallback value."
            )

        # Returned in file order, one element per contact point, duplicates included.
        # A ``vertex_id`` appearing twice here means two contact points of this frame
        # landed on the same vertex; downstream that vertex is credited with this
        # frame's IFF at the *summed* weight ``w1 + w2`` (which may exceed 1), because
        # the frame's IFF is a scalar and factors out of both numerator and
        # denominator. See this method's docstring: transport, never reduction.
        return _FrameRows(vertex_ids=vertex_ids, signed_depth_mm=depths)


def _load_block_vertex_source(
    depth_blocks_dir: Path,
    source_block_file: str,
    session_id: str,
    forearm_ply_path: Path,
) -> _BlockVertexSource:
    """Resolve and load the depth-field sidecar for one block.

    The path is composed from two things and nothing else: *depth_blocks_dir* comes
    from config, and *source_block_file* comes from the CSV's own
    ``source_block_file`` column. This module composes no path fragment from repo
    knowledge, hardcodes no directory or stage name, and performs no stem surgery --
    the ``source_block_file`` basename already carries whatever suffix its producing
    stage gave it. The CSV-name -> parquet-name rule lives in
    ``depth_field_path_for_csv`` and is not reimplemented here.

    The coordinate space is read from the parquet **metadata**, never inferred from
    the directory name, and asserted against :data:`TERMINAL_RF_CENTERED_SPACES`.
    """
    stage_csv = Path(depth_blocks_dir) / Path(source_block_file).name
    sidecar_path = depth_field_path_for_csv(stage_csv)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"_load_block_vertex_source: depth-field sidecar not found: {sidecar_path}"
            f" | blocks dir (config): {depth_blocks_dir}"
            f" | source_block_file (CSV column): {source_block_file}."
            f" A missing file means the artifact was never produced; zero rows would "
            f"mean 'no contact' and is a different thing entirely."
        )

    # Read the declared space from the file, then assert it against the closed set of
    # what a terminal blocks stage can legitimately carry, then hand back what was
    # actually found. This is an assertion, not a fallback: the three values are the
    # full enumeration of the producer's documented outcomes (normal RF centring, the
    # RF-centring passthrough when no RF cluster was found, and the additional ICP
    # passthrough when the block had no registration snapshot). Depth weighting
    # consumes exactly two things — depth magnitude and vertex identity — and neither
    # is spatial, so the operation is genuinely space-agnostic. A fourth, undocumented
    # value raises, and whichever value was found is recorded in run provenance so a
    # surprise surfaces in the run record instead of being absorbed.
    space = declared_coordinate_space(sidecar_path)
    if space not in TERMINAL_RF_CENTERED_SPACES:
        raise ValueError(
            f"_load_block_vertex_source: sidecar {sidecar_path} declares "
            f"coordinate_space={space!r}, which is not one of the documented terminal "
            f"values {list(TERMINAL_RF_CENTERED_SPACES)}. Either the blocks-stage "
            f"config key points at an intermediate stage, or the producer emitted a "
            f"space this consumer has never been told about."
        )

    depth_field = load_depth_field(
        stage_csv,
        expect_space=space,
        session_id=session_id,
        forearm_ply_path=forearm_ply_path,
    )
    return _BlockVertexSource(depth_field, source_block_file)


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class PlaybackSessionData:
    forearm_vertices: np.ndarray              # (N, 3) raw coordinates
    forearm_vertex_colors: Optional[np.ndarray]  # (N, 3) uint8 RGB, or None


@dataclass
class TouchEvent:
    block_order_id: str
    trial_id: int
    single_touch_id: int
    gesture_type: str                       # 'tap', 'stroke_proximal', etc.
    # Per 1kHz row:
    frame_contact_pts: list                 # list of (K_i, 3) raw contact coords
    frame_vertex_indices: list              # list of (K_i,) vertex_id per contact pt,
                                            # read off the contact-depth-field sidecar
                                            # row that the point corresponds to
    frame_depths: list                      # list of (K_i,) float64 signed_depth_mm,
                                            # off the SAME sidecar rows and aligned
                                            # element-for-element with the line above.
                                            # Verbatim: negative = penetrating. Not
                                            # negated, clamped or normalised here —
                                            # ``penetration_mm`` is the single
                                            # documented negation point and it lives
                                            # in the weighting stage, not in transport
    frame_spikes: np.ndarray               # (n_frames,) bool — per-row Nerve_spike
    frame_iff: np.ndarray                  # (n_frames,) float64 — per-row Nerve_freq (Hz)


@dataclass(frozen=True)
class DepthFieldProvenance:
    """Which depth-field sidecar supplied one block's vertex identities.

    ``coordinate_space`` is the value **declared in the parquet metadata**, not
    inferred from the directory name. It is carried out of the loader so the run
    record shows which of the documented terminal spaces each block was actually in
    — a passthrough session surfaces in the summary instead of being absorbed.
    """

    source_block_file: str
    sidecar_path: str
    coordinate_space: str


@dataclass
class PlaybackData:
    session_data: PlaybackSessionData
    block_order_ids: list                   # sorted unique strings (by numeric value)
    trial_ids_by_block: dict                # block_order_id -> sorted list[int]
    touches_by_block_trial: dict            # (block_order_id, trial_id) -> sorted list[TouchEvent]
    depth_field_provenance: list            # sorted list[DepthFieldProvenance], one per block


# ------------------------------------------------------------------
# Sidecar cache helpers
# ------------------------------------------------------------------

def _playback_cache_path(series_csv_path: Path) -> Path:
    """Return the .npz sidecar cache path for *series_csv_path*."""
    return series_csv_path.parent / f"{series_csv_path.stem}_playback_cache.npz"


def _save_playback_cache(
    series_csv_path: Path,
    data: PlaybackData,
) -> None:
    """Persist *data* to a compressed .npz sidecar next to *series_csv_path*.

    The nested per-trial / per-touch / per-frame structure is flattened using
    offset/count arrays so the whole dataset fits in a single .npz.

    Write failures are logged as warnings and NOT raised, so callers stay on
    the happy path. That pre-existing leniency covers the **write** only, and the
    depth path does not inherit it: every consistency check on the depth arrays
    runs *before* the ``try`` and raises. A misaligned depth channel is a wrong
    answer, not a missing cache, and a caller that lost only its cache still
    recomputes the same correct data next time.
    """
    cache_path = _playback_cache_path(series_csv_path)

    # Collect all touches in a stable order (block → trial → touch_id).
    all_touches: list[TouchEvent] = []
    for bid in data.block_order_ids:
        for tid in data.trial_ids_by_block[bid]:
            all_touches.extend(data.touches_by_block_trial[(bid, tid)])

    n_touches = len(all_touches)

    touch_keys = np.array(
        [[t.trial_id, t.single_touch_id] for t in all_touches],
        dtype=np.int64,
    )  # (n_touches, 2)
    touch_block_ids = np.array([t.block_order_id for t in all_touches], dtype=str)

    gesture_types = np.array([t.gesture_type for t in all_touches], dtype=str)

    # frame_spikes: concatenate all (n_frames_i,) bool arrays.
    frame_spikes_counts = np.array([len(t.frame_spikes) for t in all_touches], dtype=np.int64)
    frame_spikes_data = np.concatenate(
        [t.frame_spikes for t in all_touches]
    ).astype(bool) if n_touches > 0 else np.array([], dtype=bool)

    # frame_iff: concatenate all (n_frames_i,) float64 arrays (same counts as frame_spikes).
    frame_iff_data = np.concatenate(
        [t.frame_iff for t in all_touches]
    ).astype(np.float64) if n_touches > 0 else np.array([], dtype=np.float64)

    # Contact points: deduplicated format.
    # The CSV parser already reuses the same ndarray object for consecutive
    # identical contact_points strings within a touch event.  We exploit
    # object identity (id()) to store each unique (pts, vtx) pair only once.
    #
    # cp_unique_pts     : (total_unique_pts, 3) float32 — unique contact coords
    # cp_unique_vtx     : (total_unique_pts,)   int32   — unique vertex indices
    # cp_unique_offsets : (n_groups + 1,)       int32   — cumulative group sizes
    # cp_frame_group    : (n_contact_frames,)   int32   — group index per frame
    # cp_frame_touch    : (n_contact_frames,)   int32   — touch index per frame
    # cp_frame_fi       : (n_contact_frames,)   int32   — frame-within-touch index
    #
    # Depth is emphatically NOT deduplicated with the group.  The group key is
    # object identity of the vertex array, which the parser shares only across rows
    # that agree on BOTH the contact-point text and the frame index — but depth is
    # not implied by the contact-point string, and two Kinect frames can press the
    # same coordinates to different depths.  Storing depth per unique group would
    # therefore broadcast one frame's depths over every frame sharing it, and it
    # would do so silently, producing a plausible map that is wrong.  Depth is
    # stored **per contact-frame**, in the same order as cp_frame_group, so the
    # correctness of the depth channel does not depend on the parser's reuse key
    # staying what it is today:
    #
    # cp_frame_depth_data    : (total_frame_pts,)     float64 — depth per contact pt
    # cp_frame_depth_offsets : (n_contact_frames + 1,) int64  — cumulative per frame
    seen: dict[int, int] = {}          # id(vtx_array) -> group index
    unique_pts_parts: list[np.ndarray] = []
    unique_vtx_parts: list[np.ndarray] = []
    group_sizes: list[int] = []
    frame_group_list: list[int] = []
    frame_touch_list: list[int] = []
    frame_fi_list: list[int] = []
    frame_depth_parts: list[np.ndarray] = []
    frame_depth_sizes: list[int] = []

    for ti, touch in enumerate(all_touches):
        n_frames_pts = len(touch.frame_contact_pts)
        if not (len(touch.frame_vertex_indices) == len(touch.frame_depths) == n_frames_pts):
            raise ValueError(
                f"_save_playback_cache: touch (block={touch.block_order_id}, "
                f"trial={touch.trial_id}, touch={touch.single_touch_id}) has "
                f"{n_frames_pts} contact-point frame(s), "
                f"{len(touch.frame_vertex_indices)} vertex frame(s) and "
                f"{len(touch.frame_depths)} depth frame(s); all three are per-frame "
                f"lists and must be the same length."
            )
        for fi, (pts, vtx, depths) in enumerate(
            zip(touch.frame_contact_pts, touch.frame_vertex_indices, touch.frame_depths)
        ):
            vtx_arr = np.asarray(vtx, dtype=np.int64)
            depth_arr = np.asarray(depths, dtype=np.float64)
            k = len(vtx_arr)
            if len(depth_arr) != k:
                raise ValueError(
                    f"_save_playback_cache: frame {fi} of touch "
                    f"(block={touch.block_order_id}, trial={touch.trial_id}, "
                    f"touch={touch.single_touch_id}) carries {k} vertex "
                    f"index/indices but {len(depth_arr)} depth value(s). They come "
                    f"off the same sidecar rows and must be aligned "
                    f"element-for-element."
                )
            if k == 0:
                continue
            obj_id = id(vtx)
            if obj_id not in seen:
                group_idx = len(seen)
                seen[obj_id] = group_idx
                unique_pts_parts.append(np.asarray(pts, dtype=np.float32))
                unique_vtx_parts.append(vtx_arr.astype(np.int32))
                group_sizes.append(k)
            else:
                group_idx = seen[obj_id]
            frame_group_list.append(group_idx)
            frame_touch_list.append(ti)
            frame_fi_list.append(fi)
            frame_depth_parts.append(depth_arr)
            frame_depth_sizes.append(k)

    if unique_pts_parts:
        cp_unique_pts = np.concatenate(unique_pts_parts, axis=0)
        cp_unique_vtx = np.concatenate(unique_vtx_parts)
        cp_unique_offsets = np.zeros(len(group_sizes) + 1, dtype=np.int32)
        cp_unique_offsets[1:] = np.cumsum(group_sizes, dtype=np.int32)
        cp_frame_group = np.array(frame_group_list, dtype=np.int32)
        cp_frame_touch = np.array(frame_touch_list, dtype=np.int32)
        cp_frame_fi = np.array(frame_fi_list, dtype=np.int32)
        cp_frame_depth_data = np.concatenate(frame_depth_parts).astype(np.float64)
        cp_frame_depth_offsets = np.zeros(len(frame_depth_sizes) + 1, dtype=np.int64)
        cp_frame_depth_offsets[1:] = np.cumsum(frame_depth_sizes, dtype=np.int64)
    else:
        cp_unique_pts = np.empty((0, 3), dtype=np.float32)
        cp_unique_vtx = np.empty(0, dtype=np.int32)
        cp_unique_offsets = np.zeros(1, dtype=np.int32)
        cp_frame_group = np.empty(0, dtype=np.int32)
        cp_frame_touch = np.empty(0, dtype=np.int32)
        cp_frame_fi = np.empty(0, dtype=np.int32)
        cp_frame_depth_data = np.empty(0, dtype=np.float64)
        cp_frame_depth_offsets = np.zeros(1, dtype=np.int64)

    # Depth-field provenance, one entry per block. Persisted so a cache hit still
    # reports which sidecar and which declared coordinate space produced these vertex
    # identities; without it a warm cache would silently drop the provenance the run
    # record is supposed to carry.
    provenance = list(data.depth_field_provenance)
    depth_provenance_block_files = np.array(
        [pv.source_block_file for pv in provenance], dtype=str
    )
    depth_provenance_sidecar_paths = np.array(
        [pv.sidecar_path for pv in provenance], dtype=str
    )
    depth_provenance_spaces = np.array(
        [pv.coordinate_space for pv in provenance], dtype=str
    )

    optional_arrays: dict = {}
    if data.session_data.forearm_vertex_colors is not None:
        optional_arrays["forearm_vertex_colors"] = data.session_data.forearm_vertex_colors

    try:
        np.savez_compressed(
            cache_path,
            cache_schema_version=np.array(PLAYBACK_CACHE_SCHEMA_VERSION, dtype=np.int64),
            touch_keys=touch_keys,
            touch_block_ids=touch_block_ids,
            gesture_types=gesture_types,
            frame_spikes_data=frame_spikes_data,
            frame_spikes_counts=frame_spikes_counts,
            frame_iff_data=frame_iff_data,
            cp_unique_pts=cp_unique_pts,
            cp_unique_vtx=cp_unique_vtx,
            cp_unique_offsets=cp_unique_offsets,
            cp_frame_group=cp_frame_group,
            cp_frame_touch=cp_frame_touch,
            cp_frame_fi=cp_frame_fi,
            cp_frame_depth_data=cp_frame_depth_data,
            cp_frame_depth_offsets=cp_frame_depth_offsets,
            forearm_vertices=data.session_data.forearm_vertices,
            depth_provenance_block_files=depth_provenance_block_files,
            depth_provenance_sidecar_paths=depth_provenance_sidecar_paths,
            depth_provenance_spaces=depth_provenance_spaces,
            **optional_arrays,
        )
    except Exception as exc:
        logger.warning(
            "_save_playback_cache: could not write cache %s — %s", cache_path, exc
        )


def _load_playback_cache(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> Optional[PlaybackData]:
    """Load the .npz sidecar cache for *series_csv_path* when it is fresh.

    Returns a ``PlaybackData`` on a valid cache hit, or ``None`` when:
    - the cache file does not exist,
    - the cache is older than either source file (mtime check).

    Raises ``ValueError`` on shape mismatches (corrupt cache) so that stale
    data does not silently propagate to the GUI.
    """
    cache_path = _playback_cache_path(series_csv_path)

    if not cache_path.exists():
        return None

    cache_mtime = cache_path.stat().st_mtime
    csv_mtime = series_csv_path.stat().st_mtime
    ply_mtime = forearm_ply_path.stat().st_mtime

    if cache_mtime < max(csv_mtime, ply_mtime):
        logger.debug(
            "_load_playback_cache: stale cache for %s — recomputing",
            series_csv_path.name,
        )
        return None

    npz = np.load(cache_path, allow_pickle=True)

    # Validate schema version.
    if "cache_schema_version" not in npz or int(npz["cache_schema_version"]) != PLAYBACK_CACHE_SCHEMA_VERSION:
        logger.debug(
            "_load_playback_cache: schema version mismatch for %s — recomputing",
            series_csv_path.name,
        )
        return None

    # Validate that all expected keys are present (old-format guard).
    required_keys = (
        "touch_keys", "touch_block_ids", "gesture_types",
        "frame_spikes_data", "frame_spikes_counts",
        "frame_iff_data",
        "cp_unique_pts", "cp_unique_vtx", "cp_unique_offsets",
        "cp_frame_group", "cp_frame_touch", "cp_frame_fi",
        "forearm_vertices",
        # Bumped in the same change as ``PLAYBACK_CACHE_SCHEMA_VERSION`` — a version check that
        # passes while the payload is missing is exactly the failure this guards.
        "depth_provenance_block_files",
        "depth_provenance_sidecar_paths",
        "depth_provenance_spaces",
        # v5, added in the same change as the version bump for the same reason.
        "cp_frame_depth_data",
        "cp_frame_depth_offsets",
    )
    missing_keys = [k for k in required_keys if k not in npz]
    if missing_keys:
        logger.debug(
            "_load_playback_cache: old-format cache (missing %s) for %s — recomputing",
            missing_keys, series_csv_path.name,
        )
        return None

    # Reject caches that still contain the old tangent_rotation key — they were
    # produced by the pre-refactor loader and must be regenerated.
    if "tangent_rotation" in npz:
        logger.debug(
            "_load_playback_cache: old-format cache (contains tangent_rotation) for %s "
            "— recomputing",
            series_csv_path.name,
        )
        return None

    touch_keys = npz["touch_keys"]
    touch_block_ids = npz["touch_block_ids"]
    gesture_types = npz["gesture_types"]
    frame_spikes_data = npz["frame_spikes_data"]
    frame_spikes_counts = npz["frame_spikes_counts"]
    frame_iff_data = npz["frame_iff_data"]
    cp_unique_pts = npz["cp_unique_pts"]
    cp_unique_vtx = npz["cp_unique_vtx"]
    cp_unique_offsets = npz["cp_unique_offsets"]
    cp_frame_group = npz["cp_frame_group"]
    cp_frame_touch = npz["cp_frame_touch"]
    cp_frame_fi = npz["cp_frame_fi"]
    cp_frame_depth_data = npz["cp_frame_depth_data"]
    cp_frame_depth_offsets = npz["cp_frame_depth_offsets"]
    forearm_vertices = npz["forearm_vertices"]
    depth_provenance_block_files = npz["depth_provenance_block_files"]
    depth_provenance_sidecar_paths = npz["depth_provenance_sidecar_paths"]
    depth_provenance_spaces = npz["depth_provenance_spaces"]
    # forearm_vertex_colors is optional — missing means PLY had no colours.
    forearm_vertex_colors: Optional[np.ndarray] = (
        npz["forearm_vertex_colors"] if "forearm_vertex_colors" in npz else None
    )

    # Shape validation — raise on corruption.
    if touch_keys.ndim != 2 or touch_keys.shape[1] != 2:
        raise ValueError(
            f"_load_playback_cache: 'touch_keys' has shape {touch_keys.shape} "
            f"(expected (n_touches, 2)): {cache_path}"
        )
    n_touches = len(touch_keys)

    if touch_block_ids.ndim != 1 or len(touch_block_ids) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'touch_block_ids' has shape {touch_block_ids.shape}, "
            f"expected ({n_touches},): {cache_path}"
        )

    if gesture_types.ndim != 1 or len(gesture_types) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'gesture_types' has shape {gesture_types.shape}, "
            f"expected (n_touches={n_touches},): {cache_path}"
        )
    if frame_spikes_counts.ndim != 1 or len(frame_spikes_counts) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'frame_spikes_counts' has shape "
            f"{frame_spikes_counts.shape}, expected ({n_touches},): {cache_path}"
        )
    expected_spikes_len = int(frame_spikes_counts.sum())
    if frame_spikes_data.ndim != 1 or len(frame_spikes_data) != expected_spikes_len:
        raise ValueError(
            f"_load_playback_cache: 'frame_spikes_data' has length "
            f"{len(frame_spikes_data)}, expected {expected_spikes_len}: {cache_path}"
        )
    if frame_iff_data.ndim != 1 or len(frame_iff_data) != expected_spikes_len:
        raise ValueError(
            f"_load_playback_cache: 'frame_iff_data' has length "
            f"{len(frame_iff_data)}, expected {expected_spikes_len}: {cache_path}"
        )
    if cp_unique_pts.ndim != 2 or cp_unique_pts.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_pts' has shape {cp_unique_pts.shape} "
            f"(expected (total_unique_pts, 3)): {cache_path}"
        )
    total_unique_pts = len(cp_unique_pts)
    if cp_unique_vtx.ndim != 1 or len(cp_unique_vtx) != total_unique_pts:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_vtx' has shape {cp_unique_vtx.shape}, "
            f"expected ({total_unique_pts},): {cache_path}"
        )
    if cp_unique_offsets.ndim != 1 or len(cp_unique_offsets) < 1:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_offsets' has shape "
            f"{cp_unique_offsets.shape} (expected (n_groups+1,)): {cache_path}"
        )
    n_groups = len(cp_unique_offsets) - 1
    if cp_frame_group.ndim != 1:
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_group' has shape "
            f"{cp_frame_group.shape} (expected (n_contact_frames,)): {cache_path}"
        )
    n_contact_frames = len(cp_frame_group)
    for arr_name, arr in [
        ("cp_frame_touch", cp_frame_touch),
        ("cp_frame_fi", cp_frame_fi),
    ]:
        if arr.ndim != 1 or len(arr) != n_contact_frames:
            raise ValueError(
                f"_load_playback_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({n_contact_frames},): {cache_path}"
            )
    if n_contact_frames and (
        int(cp_frame_group.min()) < 0 or int(cp_frame_group.max()) >= n_groups
    ):
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_group' references group(s) outside "
            f"[0, {n_groups}) — range is "
            f"[{int(cp_frame_group.min())}, {int(cp_frame_group.max())}]: {cache_path}"
        )

    # Depth is stored per contact-frame, never per dedup group, so its offsets are
    # indexed by the contact-frame position i — the same i that indexes
    # cp_frame_group / cp_frame_touch / cp_frame_fi — and NOT by the group index.
    if cp_frame_depth_offsets.ndim != 1 or len(cp_frame_depth_offsets) != n_contact_frames + 1:
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_depth_offsets' has shape "
            f"{cp_frame_depth_offsets.shape}, expected "
            f"({n_contact_frames + 1},): {cache_path}"
        )
    if cp_frame_depth_data.ndim != 1:
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_depth_data' has shape "
            f"{cp_frame_depth_data.shape} (expected 1-D): {cache_path}"
        )
    if int(cp_frame_depth_offsets[0]) != 0:
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_depth_offsets' starts at "
            f"{int(cp_frame_depth_offsets[0])}, expected 0: {cache_path}"
        )
    if int(cp_frame_depth_offsets[-1]) != len(cp_frame_depth_data):
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_depth_offsets' ends at "
            f"{int(cp_frame_depth_offsets[-1])} but 'cp_frame_depth_data' holds "
            f"{len(cp_frame_depth_data)} value(s): {cache_path}"
        )
    depth_frame_sizes = np.diff(cp_frame_depth_offsets.astype(np.int64))
    group_sizes_cached = (
        np.diff(cp_unique_offsets.astype(np.int64))[cp_frame_group.astype(np.int64)]
        if n_contact_frames
        else np.empty(0, dtype=np.int64)
    )
    mismatched = np.flatnonzero(depth_frame_sizes != group_sizes_cached)
    if mismatched.size:
        i = int(mismatched[0])
        raise ValueError(
            f"_load_playback_cache: contact frame {i} (touch {int(cp_frame_touch[i])}, "
            f"frame {int(cp_frame_fi[i])}) holds {int(depth_frame_sizes[i])} depth "
            f"value(s) but {int(group_sizes_cached[i])} contact point(s); "
            f"{mismatched.size} contact frame(s) disagree in total: {cache_path}. "
            f"Depth is stored per contact-frame precisely so that it cannot be "
            f"broadcast across the frames sharing a deduplicated contact-point "
            f"group, and this is the check that says so."
        )
    if np.isnan(cp_frame_depth_data).any():
        n_nan = int(np.isnan(cp_frame_depth_data).sum())
        raise ValueError(
            f"_load_playback_cache: 'cp_frame_depth_data' holds {n_nan} NaN "
            f"value(s) out of {len(cp_frame_depth_data)}: {cache_path}. NaN depth "
            f"is rejected at the loader boundary, and a cache is a loader boundary; "
            f"delete the cache and recompute rather than letting an absent "
            f"measurement become a zero weight."
        )
    if forearm_vertices.ndim != 2 or forearm_vertices.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'forearm_vertices' has shape "
            f"{forearm_vertices.shape} (expected (N, 3)): {cache_path}"
        )

    n_provenance = len(depth_provenance_block_files)
    for arr_name, arr in [
        ("depth_provenance_sidecar_paths", depth_provenance_sidecar_paths),
        ("depth_provenance_spaces", depth_provenance_spaces),
    ]:
        if arr.ndim != 1 or len(arr) != n_provenance:
            raise ValueError(
                f"_load_playback_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({n_provenance},): {cache_path}"
            )
    unknown_spaces = sorted(
        {str(v) for v in depth_provenance_spaces} - set(TERMINAL_RF_CENTERED_SPACES)
    )
    if unknown_spaces:
        raise ValueError(
            f"_load_playback_cache: cached depth-field provenance declares coordinate "
            f"space(s) {unknown_spaces}, none of which is one of the documented "
            f"terminal values {list(TERMINAL_RF_CENTERED_SPACES)}: {cache_path}"
        )
    depth_field_provenance = [
        DepthFieldProvenance(
            source_block_file=str(depth_provenance_block_files[i]),
            sidecar_path=str(depth_provenance_sidecar_paths[i]),
            coordinate_space=str(depth_provenance_spaces[i]),
        )
        for i in range(n_provenance)
    ]

    # Build a lookup: frame_lookup[ti][fi] = (group_idx, contact_frame_row)
    # so we can reconstruct per-frame pts/vtx slices from the deduplicated groups
    # and per-frame depth slices from the contact-frame row — two different
    # indexings on purpose, because depth is not shared across a group.
    frame_lookup: dict[int, dict[int, tuple[int, int]]] = {}
    for i in range(n_contact_frames):
        ti_val = int(cp_frame_touch[i])
        fi_val = int(cp_frame_fi[i])
        g_val = int(cp_frame_group[i])
        frame_lookup.setdefault(ti_val, {})[fi_val] = (g_val, i)

    # Reconstruct touch events.
    frames_offset = 0
    touches_by_block_trial: dict[tuple[str, int], list[TouchEvent]] = {}

    for ti in range(n_touches):
        block_order_id = str(touch_block_ids[ti])
        trial_id = int(touch_keys[ti, 0])
        single_touch_id = int(touch_keys[ti, 1])
        gesture = str(gesture_types[ti])
        n_frames = int(frame_spikes_counts[ti])
        spikes = frame_spikes_data[frames_offset: frames_offset + n_frames].astype(bool)
        iff = frame_iff_data[frames_offset: frames_offset + n_frames].astype(np.float64)
        frames_offset += n_frames

        # Reconstruct per-frame contact pts and vertex indices via group offsets.
        ti_lookup = frame_lookup.get(ti, {})
        frame_pts_list: list[np.ndarray] = []
        frame_vtx_list: list[np.ndarray] = []
        frame_depth_list: list[np.ndarray] = []
        for fi in range(n_frames):
            if fi in ti_lookup:
                g, cf = ti_lookup[fi]
                start = int(cp_unique_offsets[g])
                end = int(cp_unique_offsets[g + 1])
                frame_pts_list.append(cp_unique_pts[start:end].astype(np.float64))
                frame_vtx_list.append(cp_unique_vtx[start:end].astype(np.int64))
                d_start = int(cp_frame_depth_offsets[cf])
                d_end = int(cp_frame_depth_offsets[cf + 1])
                frame_depth_list.append(
                    cp_frame_depth_data[d_start:d_end].astype(np.float64)
                )
            else:
                frame_pts_list.append(np.empty((0, 3), dtype=np.float64))
                frame_vtx_list.append(np.empty(0, dtype=np.int64))
                frame_depth_list.append(np.empty(0, dtype=np.float64))

        event = TouchEvent(
            block_order_id=block_order_id,
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture,
            frame_contact_pts=frame_pts_list,
            frame_vertex_indices=frame_vtx_list,
            frame_depths=frame_depth_list,
            frame_spikes=spikes,
            frame_iff=iff,
        )
        touches_by_block_trial.setdefault((block_order_id, trial_id), []).append(event)

    for key in touches_by_block_trial:
        touches_by_block_trial[key].sort(key=lambda e: e.single_touch_id)

    trial_ids_by_block: dict[str, list[int]] = {}
    for bid, tid in touches_by_block_trial:
        trial_ids_by_block.setdefault(bid, []).append(tid)
    for bid in trial_ids_by_block:
        trial_ids_by_block[bid].sort()

    block_order_ids = sorted(trial_ids_by_block.keys(), key=int)

    session_data = PlaybackSessionData(
        forearm_vertices=forearm_vertices,
        forearm_vertex_colors=forearm_vertex_colors,
    )
    return PlaybackData(
        session_data=session_data,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
        depth_field_provenance=depth_field_provenance,
    )


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_playback_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
    *,
    depth_blocks_dir: Path,
    session_id: str,
) -> PlaybackData:
    """Load per-touch data at 1kHz frame rate from *series_csv_path* + *forearm_ply_path*.

    No coordinate transforms, no deduplication, no distance filtering. Each CSV row
    becomes one frame, ``contact_points`` and ``frame_index`` are forward-filled
    together within each touch group, empty frames are kept as ``np.empty((0, 3))``,
    and ``Nerve_spike`` is taken directly per row.

    **Vertex identity comes from the contact-depth-field sidecar, not from geometry.**
    For each frame the loader takes the sidecar rows carrying that ``frame_index``, in
    file order, and reads ``vertex_id`` off the k-th row for the k-th parsed contact
    point (*ordered correspondence*; see ``_BlockVertexSource``). The nearest-vertex
    KDTree that used to answer this question was **removed, not reconciled**: it
    assigns a different vertex from the one the sidecar recorded, and that discrepancy
    *is* the measured 29 mm off-surface error, not a symptom of some other bug.

    **Depth rides the same rows.** ``TouchEvent.frame_depths[i]`` holds the
    ``signed_depth_mm`` of exactly the rows that supplied
    ``frame_vertex_indices[i]``, aligned element-for-element, carried **verbatim**
    (negative is penetrating). Nothing here negates, clamps or normalises it; the
    weighting stage owns that, and ``penetration_mm`` is the one place the sign is
    flipped.

    Parameters
    ----------
    depth_blocks_dir:
        Directory holding the block stage CSVs and their depth-field parquet
        sidecars, already composed by the caller from the session merged root and the
        configured blocks-stage subdirectory. This module composes no path fragment
        from repo knowledge and hardcodes no stage name.
    session_id:
        Session identifier, passed through to the depth-field loader for its
        diagnostics.

    Raises ``ValueError`` on missing required columns, empty data after filtering,
    forearm loading failure, a malformed contact-point triplet, or a per-frame
    contact-point count that disagrees with the sidecar; ``FileNotFoundError`` when a
    block sidecar is absent.
    """
    t_session = time.perf_counter()

    # --- Cache check ---
    depth_blocks_dir = Path(depth_blocks_dir)
    cached = _load_playback_cache(series_csv_path, forearm_ply_path)
    if cached is not None:
        print(
            f"{_ts()} | [Touch Playback] [{series_csv_path.stem}]: cache hit "
            f"({time.perf_counter() - t_session:.2f}s)",
            flush=True,
        )
        return cached

    tag = series_csv_path.stem
    print(f"{_ts()} | [Touch Playback] [{tag}]: computing (cache miss)...", flush=True)

    # --- Read CSV ---
    t = time.perf_counter()
    df = pd.read_csv(series_csv_path)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   CSV read: {time.perf_counter() - t:.1f}s  "
        f"rows={len(df)}",
        flush=True,
    )

    # Validate required columns.
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"load_playback_data: required column(s) missing from {series_csv_path}: "
            f"{missing_cols}"
        )

    # Forward-fill contact_points AND frame_index in ONE statement, over the same
    # groupby and the same column list. Two statements would fill the same gaps today
    # and could drift apart tomorrow; one statement makes desynchronisation of the
    # frame index from the contact points structurally impossible rather than merely
    # tested for. The depth join reads vertex identity off the sidecar rows of
    # ``frame_index``, so a frame index that no longer belongs to its contact points
    # would silently attribute every point of the frame to the wrong vertices.
    _ffill_cols = list(_FFILL_COLUMNS)
    df[_ffill_cols] = df.groupby(["trial_id", "single_touch_id"])[_ffill_cols].ffill()

    # Filter rows: keep only real touches (trial_id > 0 AND single_touch_id > 0).
    df = df[(df["trial_id"] > 0) & (df["single_touch_id"] > 0)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"load_playback_data: no rows with trial_id > 0 and single_touch_id > 0 "
            f"in {series_csv_path}"
        )

    nan_block_mask = df["block_order_id"].isna()
    if nan_block_mask.any():
        raise ValueError(
            f"load_playback_data: block_order_id is NaN for {nan_block_mask.sum()} rows "
            f"with trial_id > 0 and single_touch_id > 0 in {series_csv_path}"
        )

    print(
        f"{_ts()} | [Touch Playback] [{tag}]   after filter: {len(df)} rows",
        flush=True,
    )

    # --- Load forearm mesh ---
    t = time.perf_counter()
    print(f"{_ts()} | [Touch Playback] [{tag}]   loading forearm mesh...", flush=True)
    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_playback_data: could not load forearm vertices from {forearm_ply_path}"
        )
    vertex_colors = load_forearm_vertex_colors(forearm_ply_path)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   forearm mesh: {time.perf_counter() - t:.1f}s  "
        f"verts={len(vertices)}  colors={'yes' if vertex_colors is not None else 'none'}",
        flush=True,
    )

    # --- Group by (block_order_id, trial_id, single_touch_id) and build TouchEvents ---
    #
    # There is no KDTree here any more. Vertex identity is read off the depth-field
    # sidecar row that each contact point corresponds to.
    t = time.perf_counter()
    touches_by_block_trial: dict[tuple[str, int], list[TouchEvent]] = {}

    # Provenance for every block seen, and a one-entry cache of the loaded sidecar.
    # ``groupby(sort=True)`` orders by block first, so all of a block's touches are
    # consecutive and one slot is enough to avoid reloading a 5-90 MB parquet per
    # touch. Only the compact per-frame index and the int32 vertex_id column are
    # retained; the sidecar DataFrame is released with the DepthField.
    provenance_by_block_file: dict[str, DepthFieldProvenance] = {}
    block_source: Optional[_BlockVertexSource] = None

    group_keys = ["block_order_id", "trial_id", "single_touch_id"]
    for (block_order_id, trial_id, single_touch_id), group_df in df.groupby(group_keys, sort=True):
        block_order_id = str(block_order_id)
        trial_id = int(trial_id)
        single_touch_id = int(single_touch_id)

        source_block_files = group_df[_SOURCE_BLOCK_FILE_COL].astype(str).unique()
        if len(source_block_files) != 1:
            raise ValueError(
                f"load_playback_data: touch (block={block_order_id}, trial={trial_id}, "
                f"touch={single_touch_id}) spans {len(source_block_files)} distinct "
                f"{_SOURCE_BLOCK_FILE_COL!r} values {sorted(source_block_files)} in "
                f"{series_csv_path}. One touch must come from exactly one block, or "
                f"its contact points cannot be joined to a single depth-field sidecar."
            )
        source_block_file = str(source_block_files[0])

        if block_source is None or block_source.source_block_file != source_block_file:
            block_source = _load_block_vertex_source(
                depth_blocks_dir=depth_blocks_dir,
                source_block_file=source_block_file,
                session_id=session_id,
                forearm_ply_path=forearm_ply_path,
            )
            provenance_by_block_file[source_block_file] = DepthFieldProvenance(
                source_block_file=source_block_file,
                sidecar_path=str(block_source.sidecar_path),
                coordinate_space=block_source.coordinate_space,
            )

        cp_strings = group_df[CONTACT_POINTS_COL].values
        frame_index_values = group_df[_FRAME_INDEX_COL].to_numpy()
        spikes_arr = group_df[NERVE_SPIKE_COL].to_numpy(dtype=bool)
        iff_arr = group_df[NERVE_FREQ_COL].to_numpy(dtype=np.float64)
        gesture_type = str(group_df["gesture_type"].iloc[0])

        frame_contact_pts: list[np.ndarray] = []
        frame_vertex_indices: list[np.ndarray] = []
        frame_depths: list[np.ndarray] = []
        frame_spikes: list[bool] = []
        frame_iff: list[float] = []

        # Reuse the parsed arrays for consecutive rows that share BOTH the
        # contact-point text and the frame index. The frame index is part of the key
        # on purpose: two Kinect frames can carry identical contact-point text while
        # addressing different sidecar rows, so keying on the text alone would
        # broadcast one frame's vertex identities across the other. Sharing the array
        # object within a frame is also what ``_save_playback_cache`` deduplicates on.
        prev_key: Optional[tuple] = None
        prev_pts: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_vtx: np.ndarray = np.empty(0, dtype=np.int64)
        prev_depths: np.ndarray = np.empty(0, dtype=np.float64)

        for row_idx, cell in enumerate(cp_strings):
            cell_str = str(cell)
            raw_frame_index = frame_index_values[row_idx]
            frame_missing = pd.isna(raw_frame_index)
            frame_index = None if frame_missing else int(raw_frame_index)
            key = (cell_str, frame_index)

            if key == prev_key:
                frame_contact_pts.append(prev_pts)
                frame_vertex_indices.append(prev_vtx)
                frame_depths.append(prev_depths)
            else:
                pts_arr = _parse_contact_points_strict(cell_str)
                if frame_missing:
                    # ``frame_index`` and ``contact_points`` are filled by the same
                    # statement, so a missing frame index can only mean "no contact
                    # geometry has been seen yet in this touch". Contact points
                    # without a frame index would mean the two columns have drifted
                    # apart, which is unrecoverable rather than something to patch up.
                    if len(pts_arr) > 0:
                        raise ValueError(
                            f"load_playback_data: {len(pts_arr)} contact point(s) on a "
                            f"row whose {_FRAME_INDEX_COL!r} is null "
                            f"(block={block_order_id}, trial={trial_id}, "
                            f"touch={single_touch_id}) in {series_csv_path}. The two "
                            f"columns are forward-filled by one statement and must "
                            f"never disagree; without a frame index these points "
                            f"cannot be joined to any depth-field row."
                        )
                    prev_pts = np.empty((0, 3), dtype=np.float64)
                    prev_vtx = np.empty(0, dtype=np.int64)
                    prev_depths = np.empty(0, dtype=np.float64)
                else:
                    prev_vtx, prev_depths = block_source.rows_for_frame(
                        frame_index, len(pts_arr)
                    )
                    prev_pts = pts_arr

                prev_key = key
                frame_contact_pts.append(prev_pts)
                frame_vertex_indices.append(prev_vtx)
                frame_depths.append(prev_depths)

            frame_spikes.append(bool(spikes_arr[row_idx]))
            frame_iff.append(float(iff_arr[row_idx]))

        event = TouchEvent(
            block_order_id=block_order_id,
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture_type,
            frame_contact_pts=frame_contact_pts,
            frame_vertex_indices=frame_vertex_indices,
            frame_depths=frame_depths,
            frame_spikes=np.array(frame_spikes, dtype=bool),
            frame_iff=np.array(frame_iff, dtype=np.float64),
        )
        touches_by_block_trial.setdefault((block_order_id, trial_id), []).append(event)

    print(
        f"{_ts()} | [Touch Playback] [{tag}]   grouping+parse: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    if not touches_by_block_trial:
        raise ValueError(
            f"load_playback_data: no touch events found after filtering "
            f"for {series_csv_path}"
        )

    for key in touches_by_block_trial:
        touches_by_block_trial[key].sort(key=lambda e: e.single_touch_id)

    trial_ids_by_block: dict[str, list[int]] = {}
    for bid, tid in touches_by_block_trial:
        trial_ids_by_block.setdefault(bid, []).append(tid)
    for bid in trial_ids_by_block:
        trial_ids_by_block[bid].sort()

    block_order_ids = sorted(trial_ids_by_block.keys(), key=int)

    session_data = PlaybackSessionData(
        forearm_vertices=vertices,
        forearm_vertex_colors=vertex_colors,
    )

    result = PlaybackData(
        session_data=session_data,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
        depth_field_provenance=[
            provenance_by_block_file[key] for key in sorted(provenance_by_block_file)
        ],
    )

    # --- Save cache ---
    t = time.perf_counter()
    _save_playback_cache(series_csv_path, result)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   cache saved: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    total_touches = sum(len(v) for v in touches_by_block_trial.values())
    print(
        f"{_ts()} | [Touch Playback] [{tag}]: done in "
        f"{time.perf_counter() - t_session:.1f}s  "
        f"blocks={len(block_order_ids)}  touches={total_touches}",
        flush=True,
    )
    return result
