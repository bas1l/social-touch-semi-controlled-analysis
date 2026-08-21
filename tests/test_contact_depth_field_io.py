"""Unit tests for ``contact_depth_field_io`` — the per-vertex depth-field reader.

Covers the rules stated in ``docs/data-contracts/contact-depth-field.md``: the
sidecar naming rule (marker substitution, both stems), the exact 6- or 7-column
schema (names, order, dtypes), the metadata-driven coordinate-space check including
the RF-centring passthrough, the ``units`` / ``sign_convention`` invariants, the
eager ``reference_ply_vertex_count`` provenance check, ``vertex_id``-based geometry
resolution, and the single-negation sign convention.

Every fixture is synthetic and built in ``tmp_path``: a parquet written with
``pq.write_table`` carrying file-level metadata, and a small open3d point cloud
standing in for the reference forearm. No real data file is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.contact_depth_field_io import (  # noqa: E402
    COORDINATE_SPACES,
    VERTEX_ID_COLUMN,
    CoordinateSpaceError,
    DepthField,
    DepthFieldSchemaError,
    ReferencePlyMismatchError,
    depth_field_path_for_csv,
    forearm_ply_path_for_sidecar,
    load_depth_field,
    penetration_mm,
    validate_depth_field_schema,
)

SESSION = "2022-06-15_ST14-02"
BLOCK_STEM = f"{SESSION}_semicontrolled_block-order-01"
N_FOREARM_VERTICES = 50


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _default_rows() -> dict[str, np.ndarray]:
    """Three frames, six rows, with a repeated vertex so ``max`` is exercised."""
    return {
        "frame_index": np.array([10, 10, 11, 11, 12, 12], dtype=np.int32),
        "time_s": np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2], dtype=np.float64),
        "x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        "y": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "signed_depth_mm": np.array(
            [-3.4, -1.25, -5.0, -0.5, -2.0, 2e-5], dtype=np.float64
        ),
        VERTEX_ID_COLUMN: np.array([7, 3, 7, 11, 3, 42], dtype=np.int32),
    }


def _default_metadata(with_vertex_id: bool = True) -> dict[str, str]:
    meta = {
        "schema_version": "2",
        "coordinate_space": "rf_centered",
        "units": "mm",
        "sign_convention": "negative_is_penetrating",
        "source_recording": f"{BLOCK_STEM}.mkv",
        "produced_by": "synthetic-test-fixture",
        "pipeline_stage": "postprocessing",
    }
    if with_vertex_id:
        meta["reference_ply"] = f"{SESSION}_forearm.ply"
        meta["reference_ply_vertex_count"] = str(N_FOREARM_VERTICES)
        meta["dedup_epsilon"] = "0.001"
    return meta


def _build_table(
    columns: dict[str, np.ndarray] | None = None,
    metadata: dict[str, str] | None = None,
    dtype_overrides: dict[str, pa.DataType] | None = None,
) -> pa.Table:
    """Build a depth-field arrow table with file-level metadata."""
    from analysis.receptive_field_mapping.data.contact_depth_field_io import (
        _EXPECTED_DTYPES,
    )

    columns = _default_rows() if columns is None else columns
    metadata = _default_metadata() if metadata is None else metadata
    dtype_overrides = dtype_overrides or {}

    fields = []
    arrays = []
    for name, values in columns.items():
        arrow_type = dtype_overrides.get(
            name, pa.from_numpy_dtype(_EXPECTED_DTYPES[name])
        )
        fields.append(pa.field(name, arrow_type))
        arrays.append(pa.array(values, type=arrow_type))

    schema = pa.schema(fields).with_metadata(metadata)
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_forearm(ply_path: Path, n_vertices: int = N_FOREARM_VERTICES) -> np.ndarray:
    """Write a synthetic forearm point cloud and return its vertices."""
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.column_stack(
        [
            np.arange(n_vertices, dtype=np.float64),
            np.arange(n_vertices, dtype=np.float64) * 0.5,
            np.arange(n_vertices, dtype=np.float64) * -0.25,
        ]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    return verts


def _make_block(
    tmp_path: Path,
    *,
    blocks_dir: str = "blocks_rf_centered",
    csv_name: str = f"{BLOCK_STEM}_merged_data_pca-xyz.csv",
    columns: dict[str, np.ndarray] | None = None,
    metadata: dict[str, str] | None = None,
    dtype_overrides: dict[str, pa.DataType] | None = None,
    n_forearm_vertices: int = N_FOREARM_VERTICES,
    write_sidecar: bool = True,
    forearm_dir: str = "forearm_rf_centered",
) -> tuple[Path, np.ndarray]:
    """Lay out one synthetic merged-root block. Returns (csv_path, forearm verts)."""
    merged_root = tmp_path / SESSION
    csv_path = merged_root / blocks_dir / csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("placeholder\n", encoding="utf-8")

    if write_sidecar:
        table = _build_table(columns, metadata, dtype_overrides)
        pq.write_table(table, depth_field_path_for_csv(csv_path))

    verts = _write_forearm(
        merged_root / forearm_dir / f"{SESSION}_forearm.ply", n_forearm_vertices
    )
    return csv_path, verts


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_depth_field_path_for_csv_plain_stem(tmp_path: Path) -> None:
    csv_path = tmp_path / f"{BLOCK_STEM}_merged_data.csv"
    assert depth_field_path_for_csv(csv_path) == (
        tmp_path / f"{BLOCK_STEM}_contact_depth_field.parquet"
    )


def test_depth_field_path_for_csv_pca_stem(tmp_path: Path) -> None:
    """The marker is substituted, not stripped, so the pca-xyz suffix survives."""
    csv_path = tmp_path / f"{BLOCK_STEM}_merged_data_pca-xyz.csv"
    assert depth_field_path_for_csv(csv_path) == (
        tmp_path / f"{BLOCK_STEM}_contact_depth_field_pca-xyz.parquet"
    )


def test_depth_field_path_for_csv_without_marker_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / f"{BLOCK_STEM}_something_else.csv"
    with pytest.raises(ValueError, match=r"contains no '_merged_data' marker"):
        depth_field_path_for_csv(csv_path)


@pytest.mark.parametrize(
    ("blocks_dir", "forearm_dir"),
    [
        ("blocks_deduped", "forearm_deduped"),
        ("blocks_pca_calibrated", "forearm_pca_calibrated"),
        ("blocks_rf_centered", "forearm_rf_centered"),
    ],
)
def test_forearm_ply_path_for_sidecar_maps_known_dirs(
    tmp_path: Path, blocks_dir: str, forearm_dir: str
) -> None:
    sidecar = tmp_path / SESSION / blocks_dir / f"{BLOCK_STEM}_contact_depth_field.parquet"
    assert forearm_ply_path_for_sidecar(sidecar, SESSION) == (
        tmp_path / SESSION / forearm_dir / f"{SESSION}_forearm.ply"
    )


def test_forearm_ply_path_for_sidecar_unmapped_dir_raises(tmp_path: Path) -> None:
    sidecar = (
        tmp_path / SESSION / "blocks_registered"
        / f"{BLOCK_STEM}_contact_depth_field.parquet"
    )
    with pytest.raises(ValueError, match=r"'blocks_registered'.*forearm_ply_path="):
        forearm_ply_path_for_sidecar(sidecar, SESSION)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_round_trip_with_vertex_id(tmp_path: Path) -> None:
    csv_path, verts = _make_block(tmp_path)

    field = load_depth_field(
        csv_path, expect_space="rf_centered", session_id=SESSION
    )

    assert isinstance(field, DepthField)
    assert field.space == "rf_centered"
    assert field.has_vertex_id
    assert field.meta["schema_version"] == "2"
    assert field.sidecar_path == depth_field_path_for_csv(csv_path)
    assert len(field.forearm_vertices) == N_FOREARM_VERTICES

    frame = field.frame(10)
    assert list(frame["frame_index"]) == [10, 10]
    # Sorted defensively by vertex_id: the file order was [7, 3].
    assert list(frame[VERTEX_ID_COLUMN]) == [3, 7]

    xyz = field.xyz_for(10)
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz, verts[[3, 7]])


def test_deepest_per_vertex_is_positive_and_takes_max(tmp_path: Path) -> None:
    csv_path, _ = _make_block(tmp_path)
    field = load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)

    deepest = field.deepest_per_vertex()

    assert deepest.index.name == VERTEX_ID_COLUMN
    # vertex 7 appears with -3.4 and -5.0 -> deepest penetration is +5.0.
    assert deepest.loc[7] == pytest.approx(5.0)
    # vertex 3 appears with -1.25 and -2.0 -> +2.0.
    assert deepest.loc[3] == pytest.approx(2.0)
    assert deepest.loc[11] == pytest.approx(0.5)
    # The grazing positive stays a small negative penetration, not clipped.
    assert deepest.loc[42] == pytest.approx(-2e-5)
    assert (deepest.drop(index=42) > 0).all()


def test_round_trip_without_vertex_id(tmp_path: Path) -> None:
    """A pre-projection 6-column sidecar loads, but resolves no geometry."""
    columns = _default_rows()
    del columns[VERTEX_ID_COLUMN]
    metadata = _default_metadata(with_vertex_id=False)
    metadata["coordinate_space"] = "icp_registered"
    metadata["schema_version"] = "1"

    csv_path, _ = _make_block(
        tmp_path,
        blocks_dir="blocks_deduped",
        forearm_dir="forearm_deduped",
        csv_name=f"{BLOCK_STEM}_merged_data.csv",
        columns=columns,
        metadata=metadata,
    )

    field = load_depth_field(
        csv_path, expect_space="icp_registered", session_id=SESSION
    )

    assert not field.has_vertex_id
    assert field.space == "icp_registered"
    assert len(field.frames) == 6

    with pytest.raises(ValueError, match=r"predates 'blocks_projected/'.*NO fallback"):
        field.xyz_for(10)
    with pytest.raises(ValueError, match=r"cannot be aggregated per vertex"):
        field.deepest_per_vertex()


def test_absent_frame_returns_empty_not_raise(tmp_path: Path) -> None:
    """A frame_index present in the CSV but absent from the sidecar = no contact."""
    csv_path, _ = _make_block(tmp_path)
    field = load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)

    empty = field.frame(999)

    assert len(empty) == 0
    assert list(empty.columns) == list(field.frames.columns)
    assert field.xyz_for(999).shape == (0, 3)


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------

def test_penetration_mm_negates_exactly_once(tmp_path: Path) -> None:
    csv_path, _ = _make_block(tmp_path)
    field = load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)

    values = penetration_mm(field.frames)

    # -3.4 stored -> +3.4 penetration, exactly (IEEE-754 negation is exact).
    assert values[0] == 3.4
    # A grazing positive above epsilon = 1e-5 stays representable, not clipped.
    assert values[-1] == -2e-5
    assert np.allclose(values, -field.frames["signed_depth_mm"].to_numpy())


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

def test_rf_centred_passthrough_raises_coordinate_space_error(tmp_path: Path) -> None:
    """The folder says blocks_rf_centered, the metadata says pca_calibrated."""
    metadata = _default_metadata()
    metadata["coordinate_space"] = "pca_calibrated"
    csv_path, _ = _make_block(tmp_path, metadata=metadata)

    with pytest.raises(CoordinateSpaceError) as excinfo:
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)

    message = str(excinfo.value)
    assert "passthrough" in message
    assert "RF CENTRE WAS NEVER SUBTRACTED" in message
    assert "'pca_calibrated'" in message
    assert "'rf_centered'" in message
    assert str(depth_field_path_for_csv(csv_path)) in message


def test_bogus_expect_space_is_a_caller_error(tmp_path: Path) -> None:
    csv_path, _ = _make_block(tmp_path)

    with pytest.raises(ValueError, match=r"expect_space='banana' is not a known"):
        load_depth_field(csv_path, expect_space="banana", session_id=SESSION)


def test_every_documented_space_is_accepted(tmp_path: Path) -> None:
    assert COORDINATE_SPACES == (
        "kinect_space_1",
        "icp_registered",
        "pca_calibrated",
        "rf_centered",
    )


# ---------------------------------------------------------------------------
# Missing artifacts and invariants
# ---------------------------------------------------------------------------

def test_missing_sidecar_raises_file_not_found(tmp_path: Path) -> None:
    csv_path, _ = _make_block(tmp_path, write_sidecar=False)

    with pytest.raises(FileNotFoundError, match=r"BY CONSTRUCTION"):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)


def test_wrong_units_raises(tmp_path: Path) -> None:
    metadata = _default_metadata()
    metadata["units"] = "m"
    csv_path, _ = _make_block(tmp_path, metadata=metadata)

    with pytest.raises(ValueError, match=r"units='m'.*invariant"):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)


def test_wrong_sign_convention_raises(tmp_path: Path) -> None:
    metadata = _default_metadata()
    metadata["sign_convention"] = "positive_is_penetrating"
    csv_path, _ = _make_block(tmp_path, metadata=metadata)

    with pytest.raises(
        ValueError, match=r"sign_convention='positive_is_penetrating'.*invariant"
    ):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)


def test_reference_ply_vertex_count_mismatch_raises(tmp_path: Path) -> None:
    csv_path, _ = _make_block(tmp_path, n_forearm_vertices=N_FOREARM_VERTICES + 7)

    with pytest.raises(ReferencePlyMismatchError, match=r"re-deduplication"):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)


def test_vertex_id_without_provenance_key_raises(tmp_path: Path) -> None:
    metadata = _default_metadata()
    del metadata["reference_ply_vertex_count"]
    csv_path, _ = _make_block(tmp_path, metadata=metadata)

    with pytest.raises(
        ValueError, match=r"carries a 'vertex_id' column but no "
                          r"'reference_ply_vertex_count'"
    ):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)


# ---------------------------------------------------------------------------
# Schema violations
# ---------------------------------------------------------------------------

def test_schema_wrong_column_order_raises(tmp_path: Path) -> None:
    rows = _default_rows()
    swapped = {
        "frame_index": rows["frame_index"],
        "x": rows["x"],
        "time_s": rows["time_s"],
        "y": rows["y"],
        "z": rows["z"],
        "signed_depth_mm": rows["signed_depth_mm"],
        VERTEX_ID_COLUMN: rows[VERTEX_ID_COLUMN],
    }
    with pytest.raises(DepthFieldSchemaError, match=r"is not a legal .*layout"):
        validate_depth_field_schema(_build_table(swapped))


def test_schema_missing_required_column_raises(tmp_path: Path) -> None:
    rows = _default_rows()
    del rows["time_s"]
    with pytest.raises(DepthFieldSchemaError, match=r"is not a legal .*layout"):
        validate_depth_field_schema(_build_table(rows))


def test_schema_extra_column_raises(tmp_path: Path) -> None:
    from analysis.receptive_field_mapping.data.contact_depth_field_io import (
        _EXPECTED_DTYPES,
    )

    rows = _default_rows()
    fields = [
        pa.field(name, pa.from_numpy_dtype(_EXPECTED_DTYPES[name]))
        for name in rows
    ] + [pa.field("unexpected", pa.float64())]
    arrays = [
        pa.array(values, type=pa.from_numpy_dtype(_EXPECTED_DTYPES[name]))
        for name, values in rows.items()
    ] + [pa.array(np.zeros(6), type=pa.float64())]
    table = pa.Table.from_arrays(
        arrays, schema=pa.schema(fields).with_metadata(_default_metadata())
    )

    with pytest.raises(DepthFieldSchemaError, match=r"unexpected.*is not a legal"):
        validate_depth_field_schema(table)


def test_schema_wrong_dtype_raises(tmp_path: Path) -> None:
    """signed_depth_mm must stay float64 — a float32 file is malformed."""
    csv_path, _ = _make_block(
        tmp_path, dtype_overrides={"signed_depth_mm": pa.float32()}
    )

    with pytest.raises(
        DepthFieldSchemaError,
        match=r"column 'signed_depth_mm' has dtype float but the contract fixes it at "
              r"double",
    ):
        load_depth_field(csv_path, expect_space="rf_centered", session_id=SESSION)
