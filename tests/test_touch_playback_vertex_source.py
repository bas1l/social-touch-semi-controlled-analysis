"""Contact-point vertex identity comes from the depth-field sidecar, not a KDTree.

Covers Phase 2.5 of ``docs/development/plans/active/depth-weighted-iff-attribution.md``:
the ``frame_index``-value + ordered-position-within-that-frame join, the per-frame
count assertion that makes it safe, the ``coordinate_space`` policy, and the failure
modes each of them must produce.

Every fixture is synthetic and built in ``tmp_path`` — a parquet written with
``pq.write_table`` carrying file-level metadata, a small open3d point cloud standing in
for the reference forearm, and a hand-written CSV. No real data file is touched, and
the experimental database is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.pipeline.shared_constants import (  # noqa: E402
    CONTACT_POINTS_COL,
    NERVE_FREQ_COL,
    NERVE_SPIKE_COL,
)
from analysis.receptive_field_mapping.data.contact_depth_field_io import (  # noqa: E402
    VERTEX_ID_COLUMN,
    _EXPECTED_DTYPES,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    TouchEvent,
    _CACHE_SCHEMA_VERSION,
    _playback_cache_path,
    load_playback_data,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (  # noqa: E402
    parse_contact_points,
)

SESSION = "2022-06-15_ST14-02"
BLOCK_FILE = f"{SESSION}_semicontrolled_block-order-01_merged_data.csv"
STEM_SUFFIX = "_pca-xyz"
BLOCKS_STAGE_DIR = "blocks_rf_centered"
N_FOREARM_VERTICES = 40


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _forearm_vertices(n: int = N_FOREARM_VERTICES) -> np.ndarray:
    """A line of vertices at x = 0, 1, 2, ... so nearest-vertex is trivially x."""
    return np.column_stack(
        [np.arange(n, dtype=np.float64), np.zeros(n), np.zeros(n)]
    )


def _write_forearm(ply_path: Path, n: int = N_FOREARM_VERTICES) -> np.ndarray:
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    verts = _forearm_vertices(n)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud(str(ply_path), cloud)
    return verts


def _metadata(space: str = "rf_centered", with_vertex_id: bool = True) -> dict[str, str]:
    meta = {
        "schema_version": "2",
        "coordinate_space": space,
        "units": "mm",
        "sign_convention": "negative_is_penetrating",
        "source_recording": f"{SESSION}_semicontrolled_block-order-01.mkv",
        "produced_by": "synthetic-test-fixture",
        "pipeline_stage": "postprocessing",
    }
    if with_vertex_id:
        meta["reference_ply"] = f"{SESSION}_forearm.ply"
        meta["reference_ply_vertex_count"] = str(N_FOREARM_VERTICES)
        meta["dedup_epsilon"] = "0.001"
    return meta


def _write_sidecar(
    blocks_dir: Path,
    frame_index: list[int],
    vertex_id: list[int] | None,
    *,
    space: str = "rf_centered",
    depths: list[float] | None = None,
    block_file: str = BLOCK_FILE,
) -> Path:
    """Write a depth-field parquet sidecar for one block into *blocks_dir*."""
    blocks_dir.mkdir(parents=True, exist_ok=True)
    n = len(frame_index)
    columns: dict[str, np.ndarray] = {
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "time_s": np.arange(n, dtype=np.float64) * 0.033,
        "x": np.zeros(n, dtype=np.float32),
        "y": np.zeros(n, dtype=np.float32),
        "z": np.zeros(n, dtype=np.float32),
        "signed_depth_mm": np.asarray(
            depths if depths is not None else [-1.0] * n, dtype=np.float64
        ),
    }
    if vertex_id is not None:
        columns[VERTEX_ID_COLUMN] = np.asarray(vertex_id, dtype=np.int32)

    fields, arrays = [], []
    for name, values in columns.items():
        arrow_type = pa.from_numpy_dtype(_EXPECTED_DTYPES[name])
        fields.append(pa.field(name, arrow_type))
        arrays.append(pa.array(values, type=arrow_type))
    schema = pa.schema(fields).with_metadata(
        _metadata(space=space, with_vertex_id=vertex_id is not None)
    )
    table = pa.Table.from_arrays(arrays, schema=schema)

    stem = f"{Path(block_file).stem}{STEM_SUFFIX}".replace(
        "_merged_data", "_contact_depth_field"
    )
    sidecar_path = blocks_dir / f"{stem}.parquet"
    pq.write_table(table, sidecar_path)
    return sidecar_path


def _cell(xs) -> str:
    """Render contact-point coordinates the way the merged CSV stores them."""
    if len(xs) == 0:
        return "[]"
    return "[" + " ".join(f"[{x:.1f} 0.0 0.0]" for x in xs) + "]"


def _write_series_csv(
    csv_path: Path,
    rows: list[dict],
    *,
    block_file: str = BLOCK_FILE,
) -> Path:
    """Write a prepared/series CSV. Each dict is one 1 kHz row."""
    frame = pd.DataFrame(
        [
            {
                CONTACT_POINTS_COL: row.get("cell"),
                "block_order_id": row.get("block_order_id", 1),
                "trial_id": row.get("trial_id", 1),
                "single_touch_id": row.get("single_touch_id", 1),
                NERVE_SPIKE_COL: row.get("spike", 0),
                NERVE_FREQ_COL: row.get("iff", 10.0),
                "gesture_type": row.get("gesture_type", "stroke_proximal"),
                "frame_index": row.get("frame_index"),
                "source_block_file": row.get("source_block_file", block_file),
            }
            for row in rows
        ]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return csv_path


def _load(tmp_path: Path, csv_path: Path, ply_path: Path, blocks_dir: Path):
    return load_playback_data(
        csv_path,
        ply_path,
        depth_blocks_dir=blocks_dir,
        block_csv_stem_suffix=STEM_SUFFIX,
        session_id=SESSION,
    )


@pytest.fixture()
def env(tmp_path: Path):
    """A session merged root with a forearm PLY and an empty blocks stage dir."""
    merged_root = tmp_path / "3_merged" / SESSION
    blocks_dir = merged_root / BLOCKS_STAGE_DIR
    blocks_dir.mkdir(parents=True)
    ply_path = merged_root / f"{SESSION}_forearm.ply"
    verts = _write_forearm(ply_path)
    csv_path = tmp_path / "4_analysed" / f"{SESSION}_series_augmented.csv"
    return {
        "merged_root": merged_root,
        "blocks_dir": blocks_dir,
        "ply_path": ply_path,
        "csv_path": csv_path,
        "forearm_vertices": verts,
        "tmp_path": tmp_path,
    }


# ---------------------------------------------------------------------------
# The join itself
# ---------------------------------------------------------------------------

class TestVertexIdentityComesFromTheSidecar:
    """The sidecar's ``vertex_id`` wins over any geometric answer."""

    def test_sidecar_vertex_id_beats_nearest_vertex(self, env):
        # Contact coordinates sit exactly on vertices 0, 1, 2 — a KDTree over the
        # forearm would answer 0, 1, 2. The sidecar says 30, 31, 32. The sidecar's
        # answer is the one that must reach the map; the discrepancy IS the error the
        # retired nearest-vertex snapping made, not a symptom of another bug.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[100, 100, 100],
            vertex_id=[30, 31, 32],
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 100}],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(
            touch.frame_vertex_indices[0], np.array([30, 31, 32])
        )

    def test_ordered_position_within_the_frame_not_sorted_by_vertex_id(self, env):
        # The frame's rows are in DESCENDING vertex_id order in the file. Ordered
        # correspondence means point k takes row k *in file order*; a defensive sort
        # by vertex_id (what ``DepthField.frame`` does for aggregating callers) would
        # reverse the assignment and is exactly the bug this asserts against.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[7, 7, 7],
            vertex_id=[20, 12, 5],
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 7}],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(
            touch.frame_vertex_indices[0], np.array([20, 12, 5])
        )

    def test_frames_are_located_by_value_not_row_position(self, env):
        # Frame 5 is the *second* group in the file and frame 9 the first. Joining by
        # whole-file row position would hand frame 5 the rows of frame 9.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[9, 9, 5],
            vertex_id=[1, 2, 33],
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0]), "frame_index": 5},
                {"cell": _cell([0.0, 1.0]), "frame_index": 9},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(touch.frame_vertex_indices[0], np.array([33]))
        np.testing.assert_array_equal(touch.frame_vertex_indices[1], np.array([1, 2]))

    def test_identical_contact_text_in_two_frames_gets_each_frames_own_vertices(self, env):
        # Two frames carry byte-identical ``contact_points`` text but different
        # sidecar rows. Reusing the parsed result across frames — keying the
        # optimisation on the text alone — would broadcast frame 1's vertices onto
        # frame 2. The reuse key is (text, frame_index) for exactly this reason.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2, 2],
            vertex_id=[10, 11, 25, 26],
        )
        same_text = _cell([0.0, 1.0])
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": same_text, "frame_index": 1},
                {"cell": same_text, "frame_index": 2},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(touch.frame_vertex_indices[0], np.array([10, 11]))
        np.testing.assert_array_equal(touch.frame_vertex_indices[1], np.array([25, 26]))

    def test_frame_absent_from_the_sidecar_means_no_contact(self, env):
        _write_sidecar(env["blocks_dir"], frame_index=[1], vertex_id=[4])
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0]), "frame_index": 1},
                {"cell": "[]", "frame_index": 2},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        assert len(touch.frame_vertex_indices[1]) == 0


# ---------------------------------------------------------------------------
# The per-frame count assertion (2.5.6)
# ---------------------------------------------------------------------------

class TestPerFrameCountAssertion:

    def test_too_few_sidecar_rows_raises_with_full_context(self, env):
        sidecar = _write_sidecar(
            env["blocks_dir"], frame_index=[3, 3], vertex_id=[1, 2]
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 3}],
        )
        with pytest.raises(ValueError) as exc:
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        message = str(exc.value)
        assert str(sidecar) in message
        assert "frame_index=3" in message
        assert "3 contact point" in message
        assert "2 row" in message

    def test_too_many_sidecar_rows_raises(self, env):
        _write_sidecar(env["blocks_dir"], frame_index=[3, 3, 3], vertex_id=[1, 2, 3])
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 3}]
        )
        with pytest.raises(ValueError, match="count mismatch"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

    def test_contact_points_with_no_sidecar_rows_at_all_raises(self, env):
        _write_sidecar(env["blocks_dir"], frame_index=[1], vertex_id=[1])
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 99}]
        )
        with pytest.raises(ValueError, match="frame_index=99"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])


# ---------------------------------------------------------------------------
# coordinate_space policy (2.5.4)
# ---------------------------------------------------------------------------

class TestCoordinateSpacePolicy:

    @pytest.mark.parametrize(
        "space", ["rf_centered", "pca_calibrated", "kinect_space_1"]
    )
    def test_three_terminal_spaces_load_and_are_recorded(self, env, space):
        _write_sidecar(
            env["blocks_dir"], frame_index=[1], vertex_id=[3], space=space
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        assert [pv.coordinate_space for pv in data.depth_field_provenance] == [space]
        assert [pv.source_block_file for pv in data.depth_field_provenance] == [BLOCK_FILE]

    def test_undocumented_space_raises(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1],
            vertex_id=[3],
            space="icp_registered",
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError, match="icp_registered"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])


# ---------------------------------------------------------------------------
# Distinct failures for distinct causes
# ---------------------------------------------------------------------------

class TestFailureModesAreDistinguishable:

    def test_missing_sidecar_is_a_file_not_found_not_a_count_mismatch(self, env):
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        with pytest.raises(FileNotFoundError) as exc:
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        assert "sidecar not found" in str(exc.value)

    def test_pre_projection_sidecar_names_the_stage(self, env):
        # A schema-version-1 sidecar (blocks_filtered / registered / deduped) has no
        # vertex_id column at all. The loader must say so and name the stage, not
        # surface a downstream KeyError and not snap to the nearest vertex.
        _write_sidecar(env["blocks_dir"], frame_index=[1], vertex_id=None)
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError) as exc:
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        message = str(exc.value)
        assert VERTEX_ID_COLUMN in message
        assert BLOCKS_STAGE_DIR in message
        assert "NO fallback" in message

    def test_malformed_triplet_raises_rather_than_being_dropped(self, env):
        # A dropped point would shift every later point of the frame onto the wrong
        # sidecar row and yield a plausible-looking, wrong map.
        _write_sidecar(env["blocks_dir"], frame_index=[1, 1], vertex_id=[3, 4])
        _write_series_csv(
            env["csv_path"],
            [{"cell": "[[0.0 0.0] [1.0 0.0 0.0]]", "frame_index": 1}],
        )
        with pytest.raises(ValueError, match="expected exactly 3"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

    def test_vertex_id_beyond_the_forearm_raises(self, env):
        _write_sidecar(
            env["blocks_dir"], frame_index=[1], vertex_id=[N_FOREARM_VERTICES + 5]
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError, match="only 40"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])


# ---------------------------------------------------------------------------
# The single ffill statement (2.5.2)
# ---------------------------------------------------------------------------

class TestFrameIndexAndContactPointsFillTogether:

    def test_gaps_in_both_columns_yield_matched_pairs(self, env):
        # Kinect geometry arrives at ~30 Hz and is forward-filled up to the 1 kHz
        # nerve rate. ``frame_index`` and ``contact_points`` are filled by ONE
        # statement, so every filled row keeps the frame index its geometry came
        # from — here every one of the six rows resolves to its own frame's vertices.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2, 2],
            vertex_id=[10, 11, 20, 21],
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0, 1.0]), "frame_index": 1},
                {"cell": None, "frame_index": None},
                {"cell": None, "frame_index": None},
                {"cell": _cell([2.0, 3.0]), "frame_index": 2},
                {"cell": None, "frame_index": None},
                {"cell": None, "frame_index": None},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        assert len(touch.frame_vertex_indices) == 6
        for fi in range(3):
            np.testing.assert_array_equal(
                touch.frame_vertex_indices[fi], np.array([10, 11])
            )
        for fi in range(3, 6):
            np.testing.assert_array_equal(
                touch.frame_vertex_indices[fi], np.array([20, 21])
            )

    def test_leading_rows_before_any_geometry_stay_empty(self, env):
        _write_sidecar(env["blocks_dir"], frame_index=[1], vertex_id=[10])
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": None, "frame_index": None},
                {"cell": _cell([0.0]), "frame_index": 1},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        assert len(touch.frame_vertex_indices[0]) == 0
        np.testing.assert_array_equal(touch.frame_vertex_indices[1], np.array([10]))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCacheCarriesTheSidecarAnswer:

    def test_round_trip_preserves_vertex_ids_and_provenance(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2],
            vertex_id=[30, 31, 32],
            space="pca_calibrated",
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0, 1.0]), "frame_index": 1},
                {"cell": _cell([2.0]), "frame_index": 2},
            ],
        )
        cold = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        assert _playback_cache_path(env["csv_path"]).exists()
        warm = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        cold_touch = cold.touches_by_block_trial[("1", 1)][0]
        warm_touch = warm.touches_by_block_trial[("1", 1)][0]
        for a, b in zip(cold_touch.frame_vertex_indices, warm_touch.frame_vertex_indices):
            np.testing.assert_array_equal(a, b)
        assert warm.depth_field_provenance == cold.depth_field_provenance
        assert warm.depth_field_provenance[0].coordinate_space == "pca_calibrated"

    def test_a_v3_cache_is_not_reused(self, env):
        # v3 caches hold KDTree-derived vertex indices — a different, wrong answer.
        # The version bump and the provenance keys landed in the same change, so a
        # cache cannot pass the version check while lacking the payload.
        assert _CACHE_SCHEMA_VERSION == 4
        _write_sidecar(env["blocks_dir"], frame_index=[1], vertex_id=[30])
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        cache_path = _playback_cache_path(env["csv_path"])
        payload = dict(np.load(cache_path, allow_pickle=True))
        payload["cache_schema_version"] = np.array(3, dtype=np.int64)
        payload["cp_unique_vtx"] = np.array([0], dtype=np.int32)  # the KDTree answer
        np.savez_compressed(cache_path, **payload)

        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(touch.frame_vertex_indices[0], np.array([30]))


# ---------------------------------------------------------------------------
# The canonical parser (hazard 8)
# ---------------------------------------------------------------------------

class TestCanonicalParserRaises:

    def test_two_field_group_raises(self):
        with pytest.raises(ValueError, match="expected exactly 3"):
            parse_contact_points("[[1.0 2.0] [1.0 2.0 3.0]]")

    def test_non_numeric_group_raises(self):
        with pytest.raises(ValueError, match="not three floats"):
            parse_contact_points("[[1.0 2.0 abc]]")

    def test_empty_cell_is_still_empty(self):
        assert parse_contact_points("[]") == []
        assert parse_contact_points(float("nan")) == []

    def test_well_formed_cell_is_unchanged(self):
        assert parse_contact_points("[[1.0 2.0 3.0] [4.0 5.0 6.0]]") == [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ]


# ---------------------------------------------------------------------------
# Upstream column restore (2.5.1)
# ---------------------------------------------------------------------------

def test_preparation_pipeline_no_longer_drops_frame_index():
    from analysis.touch_analytics.preparation_pipeline import _DROP_COLUMNS

    assert "frame_index" not in _DROP_COLUMNS, (
        "frame_index is the only exact join key the merged CSV and the depth-field "
        "sidecars share; dropping it severs the join."
    )


# ---------------------------------------------------------------------------
# Magnitude of the reassignment (2.5.7)
# ---------------------------------------------------------------------------

class TestMagnitudeOfTheVertexReassignment:
    """How much the map moves when the KDTree answer is replaced by the sidecar's.

    The real magnitude is a property of the real data and is measured by
    ``scripts/diagnose_vertex_reassignment.py``; this repo's tests never read the
    experimental database. What is pinned here is the *mechanism*: on a controlled
    fixture where the two assignments disagree, the RF map moves by exactly the
    reassignment and nothing else, and the peak lands on the sidecar's vertex.
    """

    def test_credit_lands_on_the_sidecar_vertex_not_the_nearest_one(self, env):
        from scipy.spatial import cKDTree

        from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (
            _compute_touch_rf,
        )

        # Contact coordinates on vertices 0, 1, 2 across three frames of rising IFF.
        # The sidecar records vertices 20, 21, 22 for the same points — an offset the
        # nearest-vertex answer cannot reproduce from the coordinates.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            vertex_id=[20, 21, 22, 20, 21, 22, 20, 21, 22],
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 1, "iff": 50.0},
                {"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 2, "iff": 100.0},
                {"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 3, "iff": 40.0},
            ],
        )
        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = data.touches_by_block_trial[("1", 1)][0]
        n_vertices = len(data.session_data.forearm_vertices)

        new_mean, _ = _compute_touch_rf(touch, n_vertices, "iff")
        new_vertices = sorted(idx for idx, _ in new_mean)
        assert new_vertices == [20, 21, 22]

        # Reconstruct the retired nearest-vertex assignment to size the change. This
        # is a measurement, never a fallback: the KDTree answers a different vertex,
        # and that discrepancy IS the error the sidecar join removes.
        tree = cKDTree(data.session_data.forearm_vertices)
        kdtree_indices = [
            tree.query(np.asarray(pts, dtype=np.float64))[1].astype(np.int64)
            if len(pts) else np.empty(0, dtype=np.int64)
            for pts in touch.frame_contact_pts
        ]
        old_touch = TouchEvent(
            block_order_id=touch.block_order_id,
            trial_id=touch.trial_id,
            single_touch_id=touch.single_touch_id,
            gesture_type=touch.gesture_type,
            frame_contact_pts=touch.frame_contact_pts,
            frame_vertex_indices=kdtree_indices,
            frame_spikes=touch.frame_spikes,
            frame_iff=touch.frame_iff,
        )
        old_mean, _ = _compute_touch_rf(old_touch, n_vertices, "iff")
        old_vertices = sorted(idx for idx, _ in old_mean)
        assert old_vertices == [0, 1, 2]

        # Every contact point moved, so the two maps share no vertex at all: the
        # per-vertex |delta| is the full value on 6 vertices and undefined nowhere
        # else. The *values* are identical between the two maps — only their vertex
        # identity changed — which is the signature this phase should produce.
        new_values = [value for _, value in sorted(new_mean)]
        old_values = [value for _, value in sorted(old_mean)]
        assert new_values == old_values
        assert set(new_vertices).isdisjoint(old_vertices)
