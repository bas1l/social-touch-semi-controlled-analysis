"""Reader for the per-vertex contact-depth parquet sidecars.

Every stage CSV produced by the ``social-touch-semi-controlled`` pipeline (from
``blocks_filtered/`` onward) is accompanied by a parquet *sidecar* holding one row
per ``(frame, contact vertex)`` — the full per-vertex signed penetration field the
merged CSV reduces to the single ``contact_depth`` scalar. The authoritative
description of that artifact lives in ``docs/data-contracts/contact-depth-field.md``;
this module is its reader, and every rule stated there is enforced here rather than
left to the caller.

Three traps the contract calls out, and how this module removes them:

* **The folder name lies about the coordinate space.** Two documented passthrough
  branches (RF centring, ICP registration) copy their input unchanged, so a sidecar
  sitting in ``blocks_rf_centered/`` may legitimately declare ``pca_calibrated``.
  :func:`load_depth_field` therefore reads ``coordinate_space`` from the parquet
  file-level *metadata*, never from the directory, and requires the caller to state
  the space it expects via *expect_space*.
* **The join key is ``vertex_id``, not the coordinates.** ``x``/``y``/``z`` are
  float32 that have been through several transforms with rounding at three separate
  points; the vertex index has not. :meth:`DepthField.xyz_for` resolves geometry
  through ``vertex_id`` against the reference forearm PLY.
* **Rows are not positionally aligned with the CSV.** The merged CSV is upsampled
  ~33x to the nerve rate and not uniformly, so :meth:`DepthField.frame` selects by
  ``frame_index`` *value*.

Nothing is coerced: column names, order and dtypes are all exact, and any deviation
raises :class:`DepthFieldSchemaError`.

**Residency.** Sidecars run roughly 5-90 MB per block and the producer offers no
streaming reader — every read is a whole-file ``pq.read_table(...).to_pandas()``.
This module deliberately keeps **no in-memory cache** of loaded sidecars: callers
iterating over many blocks must load lazily and bound residency themselves rather
than accumulating :class:`DepthField` objects.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.receptive_field_mapping.data.rf_data_loader import load_forearm_vertices

logger = logging.getLogger(__name__)


# The six required columns, in exact file order. Names *and* order are part of the
# contract — a reordered file is a malformed file, not something to reindex.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "time_s",
    "x",
    "y",
    "z",
    "signed_depth_mm",
)

# The optional seventh column, present only from ``blocks_projected/`` onward.
VERTEX_ID_COLUMN: str = "vertex_id"

# Exact dtype per column. ``signed_depth_mm`` is float64 on purpose (it must stay
# bit-identical to the value the CSV's ``contact_depth`` was derived from); x/y/z are
# float32 because their CSV counterpart is already quantised to 0.1 mm.
_EXPECTED_DTYPES: dict[str, np.dtype] = {
    "frame_index": np.dtype(np.int32),
    "time_s": np.dtype(np.float64),
    "x": np.dtype(np.float32),
    "y": np.dtype(np.float32),
    "z": np.dtype(np.float32),
    "signed_depth_mm": np.dtype(np.float64),
    VERTEX_ID_COLUMN: np.dtype(np.int32),
}

# Every coordinate space the pipeline can legally declare in the metadata.
COORDINATE_SPACES: tuple[str, ...] = (
    "kinect_space_1",
    "icp_registered",
    "pca_calibrated",
    "rf_centered",
)

# The three coordinate spaces a *terminal* ``blocks_rf_centered/`` sidecar can
# legitimately declare. ``rf_centered`` is the normal case; ``pca_calibrated`` means the
# RF-centring stage took its documented passthrough branch (no RF cluster found, so the
# file is a byte-identical copy of ``blocks_pca_calibrated/`` and the RF centre was never
# subtracted); ``kinect_space_1`` means the block additionally had no ICP snapshot.
# Asserting membership in this closed set is NOT a fallback: it is the full enumeration of
# what the producer can honestly write there, and anything outside it is a contract breach.
TERMINAL_RF_CENTERED_SPACES: tuple[str, ...] = (
    "rf_centered",
    "pca_calibrated",
    "kinect_space_1",
)

# The three post-projection forearm PLYs that are transformed *in place, in file
# order*, with no reordering, filtering or count change — so ``vertex_id`` indexes
# all of them identically. Any other blocks directory has no documented in-place
# forearm counterpart and must be given an explicit path by the caller.
_BLOCKS_TO_FOREARM_DIR: dict[str, str] = {
    "blocks_deduped": "forearm_deduped",
    "blocks_pca_calibrated": "forearm_pca_calibrated",
    "blocks_rf_centered": "forearm_rf_centered",
}

# The sidecar naming rule is a *substitution* of this marker, not the stripping of a
# trailing suffix — that is what keeps it valid after the PCA stage renames the stem.
_MERGED_DATA_MARKER: str = "_merged_data"
_DEPTH_FIELD_MARKER: str = "_contact_depth_field"


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class CoordinateSpaceError(ValueError):
    """The sidecar declares a coordinate space other than the one expected."""


class DepthFieldSchemaError(ValueError):
    """The sidecar's column layout, order or dtypes violate the contract."""


class ReferencePlyMismatchError(ValueError):
    """The forearm PLY is not the reference the ``vertex_id`` column indexes."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def depth_field_path_for_csv(csv_path: Path) -> Path:
    """Return the depth-field sidecar path that accompanies *csv_path*.

    The sidecar sits in the **same directory** as the stage CSV and differs from it
    only by one name segment and the extension::

        <block>_merged_data.csv          ->  <block>_contact_depth_field.parquet
        <block>_merged_data_pca-xyz.csv  ->  <block>_contact_depth_field_pca-xyz.parquet

    The rule is substitution of the ``_merged_data`` marker with
    ``_contact_depth_field``, *not* stripping a trailing suffix — that is what keeps
    it correct after the PCA calibration stage renames the stem to
    ``..._merged_data_pca-xyz``.

    Raises ``ValueError`` when the stem of *csv_path* contains no ``_merged_data``
    marker, since no sidecar name can be derived from it.
    """
    csv_path = Path(csv_path)
    stem = csv_path.stem
    if _MERGED_DATA_MARKER not in stem:
        raise ValueError(
            f"depth_field_path_for_csv: stem {stem!r} contains no "
            f"{_MERGED_DATA_MARKER!r} marker, so no sidecar name can be derived: "
            f"{csv_path}. Pass a stage CSV such as "
            f"'<block>_merged_data.csv' or '<block>_merged_data_pca-xyz.csv'."
        )
    sidecar_stem = stem.replace(_MERGED_DATA_MARKER, _DEPTH_FIELD_MARKER)
    return csv_path.parent / f"{sidecar_stem}.parquet"


def declared_coordinate_space(sidecar_path: Path) -> str:
    """Return the ``coordinate_space`` *declared in the parquet file metadata*.

    Reads the schema only — no row group is touched — so this is cheap enough to call
    before deciding what to expect of the file.

    This exists because :func:`load_depth_field` requires an ``expect_space``, while a
    terminal ``blocks_rf_centered/`` sidecar has **three** legitimate declared values
    (:data:`TERMINAL_RF_CENTERED_SPACES`). The caller reads the declared value here,
    asserts it against its own closed set, and then passes what it actually found. The
    trap this module closes is *inferring* the space from a directory name; reading it
    from the file and checking it against an enumeration is the opposite of that.

    Raises ``FileNotFoundError`` when the sidecar is absent and ``ValueError`` when the
    file carries no metadata or no ``coordinate_space`` key.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"declared_coordinate_space: depth-field sidecar not found: {sidecar_path}"
        )
    schema = pq.read_schema(sidecar_path)
    raw = schema.metadata
    if not raw:
        raise ValueError(
            f"declared_coordinate_space: sidecar {sidecar_path} carries no file-level "
            f"metadata, so its coordinate space cannot be read. The directory name is "
            f"NEVER consulted as a substitute — it is documented to lie for 8 of 11 "
            f"sessions. Regenerate the sidecar."
        )
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
    return _require_meta(meta, "coordinate_space", sidecar_path)


def forearm_ply_path_for_sidecar(sidecar_path: Path, session_id: str) -> Path:
    """Return the reference forearm PLY that matches *sidecar_path*'s stage.

    Maps the sidecar's parent directory name through the three documented
    post-projection stages, returning
    ``<merged_root>/<forearm_dir>/<session_id>_forearm.ply``. Only
    ``blocks_deduped``, ``blocks_pca_calibrated`` and ``blocks_rf_centered`` have a
    forearm PLY that is transformed in place in file order, so only those are mapped.

    Raises ``ValueError`` naming the directory for any other stage — the caller must
    then pass ``forearm_ply_path=`` explicitly rather than have a forearm guessed.
    """
    sidecar_path = Path(sidecar_path)
    if not session_id:
        raise ValueError(
            "forearm_ply_path_for_sidecar: session_id must be a non-empty string, "
            f"got {session_id!r} (sidecar: {sidecar_path})."
        )

    blocks_dir = sidecar_path.parent.name
    if blocks_dir not in _BLOCKS_TO_FOREARM_DIR:
        raise ValueError(
            f"forearm_ply_path_for_sidecar: directory {blocks_dir!r} has no documented "
            f"in-place forearm PLY (sidecar: {sidecar_path}). Only "
            f"{sorted(_BLOCKS_TO_FOREARM_DIR)} are transformed in place in file order. "
            f"Pass forearm_ply_path= explicitly for this stage."
        )

    merged_root = sidecar_path.parent.parent
    forearm_dir = _BLOCKS_TO_FOREARM_DIR[blocks_dir]
    return merged_root / forearm_dir / f"{session_id}_forearm.ply"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _arrow_type_for(column: str) -> pa.DataType:
    """Return the exact arrow type expected for *column*."""
    return pa.from_numpy_dtype(_EXPECTED_DTYPES[column])


def validate_depth_field_schema(table) -> None:
    """Assert that a loaded parquet *table* satisfies the depth-field contract.

    Fail-fast validator called at the fan-in boundary, before any row is touched.
    Exactly two column layouts are legal: the six required columns, or those six
    followed by ``vertex_id``. Names, order and dtypes are all exact — nothing is
    coerced, reordered or cast.

    *table* is a :class:`pyarrow.Table` (the reader needs the arrow layer for
    file-level metadata, so validation happens there too).

    Raises :class:`DepthFieldSchemaError` when *table* is not a
    :class:`pyarrow.Table`, when the column names or their order differ from either
    legal layout, or when any column's type differs from the one the contract fixes.
    """
    if not isinstance(table, pa.Table):
        raise DepthFieldSchemaError(
            f"validate_depth_field_schema: expected a pyarrow.Table, got "
            f"{type(table).__name__}"
        )

    found = tuple(table.schema.names)
    with_vertex = _REQUIRED_COLUMNS + (VERTEX_ID_COLUMN,)
    if found not in (_REQUIRED_COLUMNS, with_vertex):
        raise DepthFieldSchemaError(
            f"validate_depth_field_schema: column layout {list(found)} is not a legal "
            f"depth-field layout. Expected exactly {list(_REQUIRED_COLUMNS)} or "
            f"{list(with_vertex)}, in that order."
        )

    for column in found:
        expected = _arrow_type_for(column)
        actual = table.schema.field(column).type
        if actual != expected:
            raise DepthFieldSchemaError(
                f"validate_depth_field_schema: column {column!r} has dtype {actual} "
                f"but the contract fixes it at {expected} "
                f"({_EXPECTED_DTYPES[column]}). Depth-field columns are never coerced "
                f"on read — regenerate the sidecar with the correct dtype."
            )


# ---------------------------------------------------------------------------
# Depth-field DTO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthField:
    """One block's per-vertex contact-depth field, with its reference forearm.

    Built only by :func:`load_depth_field`, which has already validated the schema,
    the declared coordinate space, the units and sign convention, and the reference
    PLY's vertex count. Holding the forearm vertices on the DTO is deliberate: a
    ``vertex_id`` join is meaningless without the exact point cloud it indexes.
    """

    frames: pd.DataFrame
    space: str
    meta: Mapping[str, str]
    sidecar_path: Path
    forearm_path: Path
    forearm_vertices: np.ndarray

    @property
    def has_vertex_id(self) -> bool:
        """Whether the sidecar carries the optional ``vertex_id`` join key.

        ``False`` for any sidecar written before ``blocks_projected/``.
        """
        return VERTEX_ID_COLUMN in self.frames.columns

    def frame(self, frame_index: int) -> pd.DataFrame:
        """Return the rows of Kinect frame *frame_index*.

        Selection is by ``frame_index`` **value**, never by row position: the merged
        CSV is upsampled ~33x to the nerve rate and not uniformly, so row *i* of the
        sidecar has no relationship to row *i* of the CSV. Rows are sorted
        defensively by ``vertex_id`` when that column is present, because the schema
        promises no ordering.

        A frame with no rows returns an **empty** DataFrame with the full column
        layout. That means *no contact in that frame*, which is legal and is not an
        error — do not confuse it with a missing sidecar file.
        """
        selected = self.frames.loc[self.frames["frame_index"] == frame_index]
        if self.has_vertex_id:
            selected = selected.sort_values(VERTEX_ID_COLUMN, kind="stable")
        return selected

    def frame_row_positions_in_file_order(self) -> dict[int, np.ndarray]:
        """Map each ``frame_index`` **value** to its row positions, in **file order**.

        This is the index that makes *ordered correspondence* possible: within one
        ``frame_index``, the k-th row of this file is the k-th point listed in that
        frame's CSV ``contact_points`` cell. The producer enforces that agreement after
        every stage (``assert_row_counts_agree_with_csv``), and it is the only exact
        pairing the two artifacts have.

        Deliberately **unsorted within a frame**, unlike :meth:`frame`, which sorts by
        ``vertex_id`` defensively for callers that only aggregate. Sorting here would
        silently destroy the correspondence and hand every contact point the wrong
        vertex, so the two accessors are kept separate rather than parameterised.

        This is *not* a licence to join row *i* of this file to row *i* of the CSV
        across the whole file: the CSV is at nerve rate with a variable number of points
        per frame while this file is at frame rate. Locate the frame by ``frame_index``
        value first; only then index by position *inside* that frame.
        """
        return {
            int(key): np.asarray(positions, dtype=np.int64)
            for key, positions in
            self.frames.groupby("frame_index", sort=False).indices.items()
        }

    def xyz_for(self, frame_index: int) -> np.ndarray:
        """Return the ``(N, 3)`` contact coordinates of frame *frame_index*.

        Coordinates are resolved through ``vertex_id`` against
        :attr:`forearm_vertices`, not read from the sidecar's ``x``/``y``/``z``: the
        stored coordinates are float32 that have been through several transforms with
        rounding at three separate points, while the vertex index has not. The result
        is expressed in :attr:`space`.

        Raises ``ValueError`` when the sidecar has no ``vertex_id`` column — such a
        sidecar predates ``blocks_projected/`` and there is **no fallback**.
        Nearest-vertex snapping on this data was measured overshooting 29 mm
        off-surface at touch boundaries, so one must not be invented here.
        Raises ``IndexError`` when a ``vertex_id`` falls outside the forearm.
        """
        if not self.has_vertex_id:
            raise ValueError(
                f"DepthField.xyz_for: sidecar {self.sidecar_path} has no "
                f"{VERTEX_ID_COLUMN!r} column, so coordinates cannot be resolved "
                f"against the forearm. This sidecar predates 'blocks_projected/'. "
                f"There is NO fallback: nearest-vertex snapping on this data was "
                f"measured overshooting 29 mm off-surface at touch boundaries. Load a "
                f"sidecar from 'blocks_projected/' or later instead."
            )

        vertex_ids = self.frame(frame_index)[VERTEX_ID_COLUMN].to_numpy()
        n_vertices = len(self.forearm_vertices)
        if vertex_ids.size and (vertex_ids.min() < 0 or vertex_ids.max() >= n_vertices):
            raise IndexError(
                f"DepthField.xyz_for: frame {frame_index} references vertex ids in "
                f"[{int(vertex_ids.min())}, {int(vertex_ids.max())}] but the forearm "
                f"{self.forearm_path} has only {n_vertices} vertices."
            )
        return self.forearm_vertices[vertex_ids]

    def deepest_per_vertex(self) -> pd.Series:
        """Return the deepest positive penetration in mm, per ``vertex_id``.

        Indexed by ``vertex_id``, valued as a **positive** penetration magnitude
        (``-signed_depth_mm``), reduced with ``max`` over every frame in the block.
        The reduction is explicit because projection carries no uniqueness
        constraint: two rows of one frame may address the same vertex, and the
        pipeline's own viewer takes the deeper.

        Raises ``ValueError`` when the sidecar has no ``vertex_id`` column — see
        :meth:`xyz_for` for why no fallback exists.
        """
        if not self.has_vertex_id:
            raise ValueError(
                f"DepthField.deepest_per_vertex: sidecar {self.sidecar_path} has no "
                f"{VERTEX_ID_COLUMN!r} column, so rows cannot be aggregated per "
                f"vertex. This sidecar predates 'blocks_projected/' and there is NO "
                f"fallback — aggregating on float coordinate tuples splits one "
                f"physical vertex across several keys."
            )

        return (
            self.frames.assign(penetration_mm=penetration_mm(self.frames))
            .groupby(VERTEX_ID_COLUMN)["penetration_mm"]
            .max()
        )


def penetration_mm(frames: pd.DataFrame) -> np.ndarray:
    """Return the positive penetration magnitude of *frames*, in millimetres.

    Storage is signed millimetres with ``negative = penetrating``; this is
    ``-signed_depth_mm``, the magnitude used for display and for depth weighting.

    Negate **exactly once**, at the boundary of your code, over the whole column —
    this function is that boundary. Because negation reverses order, a colour range
    built from the signed extremes becomes ``(-signed_max, -signed_min)``, **not**
    ``(-min, -max)``.

    Values above ``epsilon = 1e-5`` mm positive in the stored column are grazing
    contacts, not errors, and are returned as small negative penetrations rather than
    clipped. Raises ``KeyError`` when *frames* has no ``signed_depth_mm`` column.
    """
    if "signed_depth_mm" not in frames.columns:
        raise KeyError(
            f"penetration_mm: frames has no 'signed_depth_mm' column; got "
            f"{list(frames.columns)}"
        )
    return -frames["signed_depth_mm"].to_numpy()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _decode_metadata(table, sidecar_path: Path) -> dict[str, str]:
    """Decode the parquet file-level metadata of *table* into a ``str -> str`` dict.

    Raises ``ValueError`` when the file carries no file-level metadata at all: the
    coordinate space, units and sign convention all live there, and a sidecar without
    them cannot be interpreted.
    """
    raw = table.schema.metadata
    if not raw:
        raise ValueError(
            f"load_depth_field: sidecar {sidecar_path} carries no parquet file-level "
            f"metadata, so 'coordinate_space', 'units' and 'sign_convention' cannot be "
            f"read. The space must never be inferred from the directory name — "
            f"regenerate the sidecar with its metadata."
        )
    return {k.decode(): v.decode() for k, v in raw.items()}


def _require_meta(meta: Mapping[str, str], key: str, sidecar_path: Path) -> str:
    """Return ``meta[key]``, raising ``ValueError`` when it is absent."""
    if key not in meta:
        raise ValueError(
            f"load_depth_field: sidecar {sidecar_path} has no {key!r} metadata key; "
            f"present keys are {sorted(meta)}."
        )
    return meta[key]


def _check_coordinate_space(
    meta: Mapping[str, str],
    expect_space: str,
    sidecar_path: Path,
) -> str:
    """Validate the declared coordinate space against *expect_space*.

    Raises ``ValueError`` when *expect_space* is not one of :data:`COORDINATE_SPACES`
    (a caller programming error, distinct from a data mismatch), and
    :class:`CoordinateSpaceError` when the sidecar declares a different space.
    """
    if expect_space not in COORDINATE_SPACES:
        raise ValueError(
            f"load_depth_field: expect_space={expect_space!r} is not a known "
            f"coordinate space. Valid values are {list(COORDINATE_SPACES)}. This is a "
            f"caller error, not a property of {sidecar_path}."
        )

    declared = _require_meta(meta, "coordinate_space", sidecar_path)
    if declared == expect_space:
        return declared

    detail = (
        "The pipeline has documented passthrough branches that copy their input "
        "unchanged and then honestly declare the space the data is actually in, so "
        "the directory name can disagree with the metadata."
    )
    if declared == "pca_calibrated" and expect_space == "rf_centered":
        detail += (
            " This session takes the RF-centring passthrough: no receptive-field "
            "cluster was found, so 'blocks_rf_centered/' is a copy of "
            "'blocks_pca_calibrated/' and THE RF CENTRE WAS NEVER SUBTRACTED — the "
            "origin is not the RF centre. Handle or skip this session explicitly."
        )
    elif declared == "kinect_space_1" and expect_space == "icp_registered":
        detail += (
            " This session takes the ICP passthrough: it has no registration "
            "transforms, so the data is still in raw Kinect Space 1."
        )
    raise CoordinateSpaceError(
        f"load_depth_field: sidecar {sidecar_path} declares "
        f"coordinate_space={declared!r} but {expect_space!r} was expected. {detail}"
    )


def _check_invariants(meta: Mapping[str, str], sidecar_path: Path) -> None:
    """Assert the units and sign convention, which are invariants not preferences.

    Raises ``ValueError`` when ``units != 'mm'`` or
    ``sign_convention != 'negative_is_penetrating'``.
    """
    units = _require_meta(meta, "units", sidecar_path)
    if units != "mm":
        raise ValueError(
            f"load_depth_field: sidecar {sidecar_path} declares units={units!r}, but "
            f"'mm' is an invariant of this pipeline — there is no unit conversion "
            f"anywhere in it. A value of 0.03 where 30 was expected is the "
            f"metres-vs-millimetres tell. Regenerate the sidecar."
        )

    sign_convention = _require_meta(meta, "sign_convention", sidecar_path)
    if sign_convention != "negative_is_penetrating":
        raise ValueError(
            f"load_depth_field: sidecar {sidecar_path} declares "
            f"sign_convention={sign_convention!r}, but 'negative_is_penetrating' is an "
            f"invariant of this pipeline. Reading it under the wrong convention "
            f"silently inverts every depth — regenerate the sidecar."
        )


def _load_reference_forearm(
    forearm_path: Path,
    meta: Mapping[str, str],
    has_vertex_id: bool,
    sidecar_path: Path,
) -> np.ndarray:
    """Load the reference forearm vertices and validate them against *meta*.

    Reuses
    :func:`analysis.receptive_field_mapping.data.rf_data_loader.load_forearm_vertices`
    so the ``.npy`` mtime-guarded cache is shared with the rest of the data layer.
    That function returns ``None`` for a missing or empty PLY; under the fail-fast
    rule that is converted into a raised ``FileNotFoundError`` here.

    The vertex-count check is **eager**, by deliberate design: a wrong-forearm join
    must fail at load time, not deep inside an analysis. It is skipped only when
    ``reference_ply_vertex_count`` is absent *and* the sidecar has no ``vertex_id``
    column, which is the legal pre-projection case.

    Raises ``FileNotFoundError`` for a missing/empty PLY, ``ValueError`` when
    ``vertex_id`` is present without the provenance key, and
    :class:`ReferencePlyMismatchError` on a count mismatch.
    """
    vertices = load_forearm_vertices(forearm_path)
    if vertices is None:
        raise FileNotFoundError(
            f"load_depth_field: reference forearm PLY is missing or holds no points: "
            f"{forearm_path} (for sidecar {sidecar_path}). Pass forearm_ply_path= "
            f"explicitly if the forearm lives elsewhere."
        )

    count_key = "reference_ply_vertex_count"
    if count_key not in meta:
        if has_vertex_id:
            raise ValueError(
                f"load_depth_field: sidecar {sidecar_path} carries a "
                f"{VERTEX_ID_COLUMN!r} column but no {count_key!r} metadata key, so "
                f"the reference forearm cannot be validated before the join. Present "
                f"keys are {sorted(meta)}. Only pre-projection sidecars may omit it."
            )
        return vertices

    expected_count = int(meta[count_key])
    if len(vertices) != expected_count:
        raise ReferencePlyMismatchError(
            f"load_depth_field: forearm {forearm_path} has {len(vertices)} vertices "
            f"but sidecar {sidecar_path} declares {count_key}={expected_count}. This "
            f"is not the reference PLY the vertex ids index — a re-deduplication at a "
            f"different epsilon renumbers every vertex and nothing upstream would "
            f"notice. Point forearm_ply_path= at the forearm produced with "
            f"dedup_epsilon={meta.get('dedup_epsilon', '<unrecorded>')!r}."
        )
    return vertices


def load_depth_field(
    csv_path: Path,
    *,
    expect_space: str,
    session_id: str,
    forearm_ply_path: Path | None = None,
) -> DepthField:
    """Load the depth-field sidecar that accompanies the stage CSV *csv_path*.

    Every check the data contract names is applied here, in order, and every one of
    them raises rather than degrading:

    1. the sidecar path is derived from *csv_path* by the documented marker
       substitution, and its absence raises ``FileNotFoundError``;
    2. the parquet file-level metadata must exist and is decoded to ``str -> str``;
    3. the column layout, order and dtypes must match the contract exactly;
    4. *expect_space* must be a known space, and the space **declared in the
       metadata** must equal it — the directory name is never consulted;
    5. ``units`` and ``sign_convention`` must hold their invariant values;
    6. the reference forearm PLY is loaded eagerly and its vertex count validated
       against ``reference_ply_vertex_count`` *before* any ``vertex_id`` join.

    *expect_space* is the coordinate space the caller's analysis assumes, one of
    :data:`COORDINATE_SPACES`. *session_id* names the session and is used to derive
    the forearm filename. *forearm_ply_path* overrides the derived forearm path and
    is required for any stage outside the three post-projection ones.

    Raises ``FileNotFoundError`` for a missing sidecar or forearm, ``ValueError`` for
    a bogus *expect_space*, missing metadata or a broken invariant,
    :class:`DepthFieldSchemaError` for a malformed schema,
    :class:`CoordinateSpaceError` for a space mismatch, and
    :class:`ReferencePlyMismatchError` for the wrong reference forearm.
    """
    csv_path = Path(csv_path)
    sidecar_path = depth_field_path_for_csv(csv_path)

    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"load_depth_field: depth-field sidecar not found: {sidecar_path} "
            f"(derived from {csv_path}). A missing file means the artifact was never "
            f"produced — note that 'blocks_merged/' has NO sidecar BY CONSTRUCTION, "
            f"since it is written before the neural-quality filter whose output lands "
            f"in 'blocks_filtered/'; that is expected, not missing. Zero rows would "
            f"mean 'no contact' and is a different thing entirely."
        )

    table = pq.read_table(sidecar_path)
    meta = _decode_metadata(table, sidecar_path)
    validate_depth_field_schema(table)

    space = _check_coordinate_space(meta, expect_space, sidecar_path)
    _check_invariants(meta, sidecar_path)

    frames = table.to_pandas()
    has_vertex_id = VERTEX_ID_COLUMN in frames.columns

    if forearm_ply_path is None:
        forearm_path = forearm_ply_path_for_sidecar(sidecar_path, session_id)
    else:
        forearm_path = Path(forearm_ply_path)

    vertices = _load_reference_forearm(forearm_path, meta, has_vertex_id, sidecar_path)

    logger.debug(
        "Loaded depth field %s: %d rows, space=%s, forearm=%s (%d vertices)",
        sidecar_path.name, len(frames), space, forearm_path.name, len(vertices),
    )

    return DepthField(
        frames=frames,
        space=space,
        meta=MappingProxyType(dict(meta)),
        sidecar_path=sidecar_path,
        forearm_path=forearm_path,
        forearm_vertices=vertices,
    )
