"""Per-contact-point penetration depth travels from the sidecar to ``TouchEvent``.

Covers Phase 3 of ``docs/development/plans/active/depth-weighted-iff-attribution.md``,
which is **transport only**: the join itself landed in Phase 2.5 and depth comes off
the *same* sidecar rows that supplied ``vertex_id``. What is tested here is that the
values survive the trip — through the forward fill, through the run-expansion that
repeats one Kinect frame across ~33 nerve-rate rows, and through the deduplicating
on-disk cache — unaltered and still attached to the right frame.

The load-bearing test is :class:`TestHazardOneDepthIsNotBroadcastAcrossADedupGroup`.
``_save_playback_cache`` deduplicates contact-point groups by ``id()`` of the vertex
array, exploiting the parser's array reuse. Depth is **not** implied by the
contact-point string: two frames can press identical coordinates to different depths.
If depth were stored per dedup group rather than per contact-frame, one frame's depths
would be silently broadcast over the other, and the resulting map would look entirely
plausible. That is the failure this file exists to make impossible.

No weighting, no alpha, no normalisation is exercised here — none exists yet. Depth is
asserted to arrive as ``signed_depth_mm`` **verbatim**, negative and all.

Every fixture is synthetic and built in ``tmp_path``. No real data file is touched and
the experimental database is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    DepthFieldProvenance,
    PlaybackData,
    PlaybackSessionData,
    TouchEvent,
    PLAYBACK_CACHE_SCHEMA_VERSION,
    _load_playback_cache,
    _playback_cache_path,
    _save_playback_cache,
)

from test_touch_playback_vertex_source import (  # noqa: E402
    BLOCK_FILE,
    _cell,
    _load,
    _write_sidecar,
    _write_series_csv,
    env,  # noqa: F401 — pytest fixture, imported for its side effect
)


def _touch(data):
    """The single touch every fixture in this file builds."""
    return data.touches_by_block_trial[("1", 1)][0]


# ---------------------------------------------------------------------------
# 3.1 / 3.2 — depth arrives off the same rows as vertex_id
# ---------------------------------------------------------------------------

class TestDepthComesOffTheSameRows:

    def test_depth_is_signed_depth_mm_verbatim(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[100, 100, 100],
            vertex_id=[30, 31, 32],
            depths=[-3.25, -1.5, -0.125],
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 100}],
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(
            touch.frame_depths[0], np.array([-3.25, -1.5, -0.125])
        )
        assert touch.frame_depths[0].dtype == np.float64

    def test_grazing_positive_depth_survives_untransformed(self, env):
        # Values above the grazing epsilon are positive in the stored column and are
        # real measurements, not errors. Transport must not clamp, negate or drop
        # them: the clamp belongs to the weight function, which does not exist yet.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[5, 5, 5],
            vertex_id=[10, 11, 12],
            depths=[0.004, 0.0, -2.0],
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 5}],
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(
            touch.frame_depths[0], np.array([0.004, 0.0, -2.0])
        )

    def test_depth_follows_file_order_not_vertex_id_order(self, env):
        # The frame's rows are in DESCENDING vertex_id order in the file. Ordered
        # correspondence means point k takes row k *in file order*; a defensive sort
        # by vertex_id would pair every point with another point's depth.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[7, 7, 7],
            vertex_id=[20, 12, 5],
            depths=[-1.0, -2.0, -3.0],
        )
        _write_series_csv(
            env["csv_path"],
            [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 7}],
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_vertex_indices[0], [20, 12, 5])
        np.testing.assert_array_equal(touch.frame_depths[0], [-1.0, -2.0, -3.0])

    def test_two_frames_get_their_own_depths(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2],
            vertex_id=[30, 31, 32],
            depths=[-1.0, -2.0, -9.5],
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0, 1.0]), "frame_index": 1},
                {"cell": _cell([2.0]), "frame_index": 2},
            ],
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_depths[0], [-1.0, -2.0])
        np.testing.assert_array_equal(touch.frame_depths[1], [-9.5])


# ---------------------------------------------------------------------------
# 3.1 — alignment survives the ffill and the run expansion
# ---------------------------------------------------------------------------

class TestAlignmentSurvivesFfillAndRunExpansion:
    """``len(frame_depths[i]) == len(frame_vertex_indices[i])`` on every frame."""

    def _assert_aligned(self, touch):
        assert len(touch.frame_depths) == len(touch.frame_vertex_indices)
        assert len(touch.frame_depths) == len(touch.frame_contact_pts)
        assert len(touch.frame_depths) == len(touch.frame_iff)
        for i, (vtx, depths, pts) in enumerate(
            zip(touch.frame_vertex_indices, touch.frame_depths, touch.frame_contact_pts)
        ):
            assert len(depths) == len(vtx), f"frame {i}"
            assert len(depths) == len(pts), f"frame {i}"

    def test_run_of_identical_rows_repeats_the_same_depths(self, env):
        # One Kinect frame is re-emitted across many nerve-rate rows. Every row of the
        # run must carry that frame's depths, and only that frame's.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2, 2],
            vertex_id=[10, 11, 12, 13],
            depths=[-1.0, -2.0, -3.0, -4.0],
        )
        rows = (
            [{"cell": _cell([0.0, 1.0]), "frame_index": 1}] * 4
            + [{"cell": _cell([2.0, 3.0]), "frame_index": 2}] * 3
        )
        _write_series_csv(env["csv_path"], rows)
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        self._assert_aligned(touch)
        for i in range(4):
            np.testing.assert_array_equal(touch.frame_depths[i], [-1.0, -2.0])
        for i in range(4, 7):
            np.testing.assert_array_equal(touch.frame_depths[i], [-3.0, -4.0])

    def test_forward_filled_gaps_and_leading_empty_rows(self, env):
        # Leading rows before any contact geometry have a null frame_index and null
        # contact points; they must yield an empty depth array, not a missing entry.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[9, 9, 9],
            vertex_id=[4, 5, 6],
            depths=[-0.5, -1.5, -2.5],
        )
        rows = [
            {"cell": None, "frame_index": None},
            {"cell": None, "frame_index": None},
            {"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 9},
            {"cell": None, "frame_index": None},  # ffilled from the row above
            {"cell": None, "frame_index": None},
        ]
        _write_series_csv(env["csv_path"], rows)
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        self._assert_aligned(touch)
        assert len(touch.frame_depths[0]) == 0
        assert len(touch.frame_depths[1]) == 0
        for i in (2, 3, 4):
            np.testing.assert_array_equal(touch.frame_depths[i], [-0.5, -1.5, -2.5])

    def test_alignment_holds_through_the_cache(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 2, 2, 2],
            vertex_id=[10, 11, 12, 13, 14],
            depths=[-1.0, -2.0, -3.0, -4.0, -5.0],
        )
        rows = (
            [{"cell": None, "frame_index": None}]
            + [{"cell": _cell([0.0, 1.0]), "frame_index": 1}] * 3
            + [{"cell": _cell([2.0, 3.0, 4.0]), "frame_index": 2}] * 2
        )
        _write_series_csv(env["csv_path"], rows)
        cold = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        warm = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        self._assert_aligned(_touch(cold))
        self._assert_aligned(_touch(warm))
        for a, b in zip(_touch(cold).frame_depths, _touch(warm).frame_depths):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 3.4 / HAZARD 1 — the whole point of this phase
# ---------------------------------------------------------------------------

class TestHazardOneDepthIsNotBroadcastAcrossADedupGroup:
    """Identical ``contact_points`` text, different depths, must stay different.

    ``_save_playback_cache`` stores each unique contact-point group once and points
    several contact-frames at it. Depth is not a property of that group — it is a
    property of the frame — so it is stored per contact-frame. If this test ever fails,
    depth has been folded into the dedup and every affected frame is reporting another
    frame's penetration.
    """

    def _fixture(self, env):
        # Frames 1, 2 and 3 all press the SAME three coordinates (byte-identical
        # contact_points cells) to three DIFFERENT sets of depths.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 1, 2, 2, 2, 3, 3, 3],
            vertex_id=[20, 21, 22, 20, 21, 22, 20, 21, 22],
            depths=[-1.0, -2.0, -3.0, -10.0, -20.0, -30.0, -100.0, -200.0, -300.0],
        )
        cell = _cell([0.0, 1.0, 2.0])
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": cell, "frame_index": 1, "iff": 50.0},
                {"cell": cell, "frame_index": 2, "iff": 100.0},
                {"cell": cell, "frame_index": 3, "iff": 40.0},
            ],
        )
        return cell

    def test_cold_load_keeps_the_depths_apart(self, env):
        self._fixture(env)
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_depths[0], [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(touch.frame_depths[1], [-10.0, -20.0, -30.0])
        np.testing.assert_array_equal(touch.frame_depths[2], [-100.0, -200.0, -300.0])

    def test_cache_round_trip_keeps_the_depths_apart(self, env):
        """THE hazard-1 test: write the cache, read it back, depths still differ."""
        self._fixture(env)
        cold = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        assert _playback_cache_path(env["csv_path"]).exists()
        warm = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        warm_touch = _touch(warm)
        np.testing.assert_array_equal(warm_touch.frame_depths[0], [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(warm_touch.frame_depths[1], [-10.0, -20.0, -30.0])
        np.testing.assert_array_equal(warm_touch.frame_depths[2], [-100.0, -200.0, -300.0])

        # ... and the warm read agrees with the cold one frame by frame.
        for a, b in zip(_touch(cold).frame_depths, warm_touch.frame_depths):
            np.testing.assert_array_equal(a, b)

        # Structurally: depth is indexed per contact-frame, so the offsets array is
        # sized by contact-frames (3) and not by dedup groups. Were depth folded into
        # the group, this shape would be the first thing to change.
        payload = np.load(_playback_cache_path(env["csv_path"]), allow_pickle=True)
        assert len(payload["cp_frame_group"]) == 3
        assert len(payload["cp_frame_depth_data"]) == 9
        np.testing.assert_array_equal(payload["cp_frame_depth_offsets"], [0, 3, 6, 9])

    def test_a_shared_dedup_group_still_carries_per_frame_depths(self, env):
        """The hazard reproduced directly, at the level where it is expressible.

        Today the loader's parse-reuse key is ``(contact_points text, frame_index)``,
        so two *different* frames never share a vertex-array object and therefore
        never share a dedup group — which means the cold-load tests above cannot, on
        their own, distinguish per-group storage from per-frame storage. That mitigation
        lives in a different function from the one that dedups, and a future change to
        the reuse key would remove it silently.

        So the shared group is constructed here on purpose: two contact-frames handed
        the **same ndarray object** for their vertex indices — exactly what ``id()``
        dedup collapses — while carrying different depths. Per-group depth storage
        returns one frame's depths for both and this test fails; per-contact-frame
        storage keeps them apart.
        """
        shared_vtx = np.array([20, 21, 22], dtype=np.int64)
        shared_pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
        )
        shallow = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        deep = np.array([-100.0, -200.0, -300.0], dtype=np.float64)
        touch = TouchEvent(
            block_order_id="1",
            trial_id=1,
            single_touch_id=1,
            gesture_type="stroke_proximal",
            # One object, listed twice: the dedup sees a single group.
            frame_contact_pts=[shared_pts, shared_pts],
            frame_vertex_indices=[shared_vtx, shared_vtx],
            frame_depths=[shallow, deep],
            frame_spikes=np.array([True, False], dtype=bool),
            frame_iff=np.array([50.0, 100.0], dtype=np.float64),
        )
        data = PlaybackData(
            session_data=PlaybackSessionData(
                forearm_vertices=env["forearm_vertices"],
                forearm_vertex_colors=None,
            ),
            block_order_ids=["1"],
            trial_ids_by_block={"1": [1]},
            touches_by_block_trial={("1", 1): [touch]},
            depth_field_provenance=[
                DepthFieldProvenance(
                    source_block_file="block.csv",
                    sidecar_path="block.parquet",
                    coordinate_space="rf_centered",
                )
            ],
        )
        # The CSV only has to exist for the cache's mtime freshness check.
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0]), "frame_index": 1}]
        )
        _save_playback_cache(env["csv_path"], data)

        payload = np.load(_playback_cache_path(env["csv_path"]), allow_pickle=True)
        # The two contact-frames really do share ONE deduplicated group here.
        assert len(payload["cp_frame_group"]) == 2
        assert len(set(payload["cp_frame_group"].tolist())) == 1
        assert len(payload["cp_unique_vtx"]) == 3
        # ... and depth is still stored twice, once per contact-frame.
        assert len(payload["cp_frame_depth_data"]) == 6

        restored = _load_playback_cache(env["csv_path"], env["ply_path"])
        assert restored is not None
        rt = restored.touches_by_block_trial[("1", 1)][0]
        np.testing.assert_array_equal(rt.frame_depths[0], shallow)
        np.testing.assert_array_equal(rt.frame_depths[1], deep)
        # The contact points, which genuinely ARE shared, still round-trip shared.
        np.testing.assert_array_equal(rt.frame_vertex_indices[0], shared_vtx)
        np.testing.assert_array_equal(rt.frame_vertex_indices[1], shared_vtx)

    def test_identical_text_across_frames_shares_contact_points_but_not_depth(self, env):
        # The coordinates themselves are genuinely shared and identical; only depth
        # differs. That asymmetry is exactly what makes the hazard invisible to a
        # contact-point-based check.
        self._fixture(env)
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_contact_pts[0], touch.frame_contact_pts[1])
        np.testing.assert_array_equal(touch.frame_vertex_indices[0], touch.frame_vertex_indices[1])
        assert not np.array_equal(touch.frame_depths[0], touch.frame_depths[1])


# ---------------------------------------------------------------------------
# 3.4 — cache schema version and payload guard
# ---------------------------------------------------------------------------

class TestCacheSchemaVersion:

    def test_version_is_pinned_at_five(self):
        # Phase 2.5 took the schema 3 -> 4 (retired KDTree indices). Phase 3 takes it
        # 4 -> 5 (depth). Pinned once, here, so a bump that forgets to extend
        # ``required_keys`` in the same change is caught by the tests below.
        assert PLAYBACK_CACHE_SCHEMA_VERSION == 5

    def test_a_v4_cache_is_not_reused(self, env):
        """A depth-free v4 cache must be rejected, not silently served."""
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 31],
            depths=[-7.0, -8.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        # Degrade the cache to exactly what v4 wrote: stamp version 4 and drop the
        # depth payload entirely.
        cache_path = _playback_cache_path(env["csv_path"])
        payload = dict(np.load(cache_path, allow_pickle=True))
        payload["cache_schema_version"] = np.array(4, dtype=np.int64)
        del payload["cp_frame_depth_data"]
        del payload["cp_frame_depth_offsets"]
        np.savez_compressed(cache_path, **payload)

        data = _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        touch = _touch(data)
        # Recomputed from the sidecar, with depth, rather than served depth-free.
        np.testing.assert_array_equal(touch.frame_depths[0], [-7.0, -8.0])

    def test_current_version_without_the_depth_keys_is_rejected(self, env):
        # The version bump and ``required_keys`` moved in one change; this asserts the
        # second half. A cache stamped v5 but missing the payload must not pass.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 31],
            depths=[-7.0, -8.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        cache_path = _playback_cache_path(env["csv_path"])
        payload = dict(np.load(cache_path, allow_pickle=True))
        assert int(payload["cache_schema_version"]) == PLAYBACK_CACHE_SCHEMA_VERSION
        del payload["cp_frame_depth_data"]
        del payload["cp_frame_depth_offsets"]
        np.savez_compressed(cache_path, **payload)

        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_depths[0], [-7.0, -8.0])

    def test_a_corrupt_depth_payload_raises_rather_than_loading(self, env):
        # A misaligned depth channel is a wrong answer, not a stale cache, so it
        # raises instead of quietly recomputing.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 1],
            vertex_id=[30, 31, 32],
            depths=[-1.0, -2.0, -3.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 1}]
        )
        _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        cache_path = _playback_cache_path(env["csv_path"])
        payload = dict(np.load(cache_path, allow_pickle=True))
        payload["cp_frame_depth_data"] = np.array([-1.0, -2.0], dtype=np.float64)
        payload["cp_frame_depth_offsets"] = np.array([0, 2], dtype=np.int64)
        np.savez_compressed(cache_path, **payload)

        with pytest.raises(ValueError, match="depth value"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

    def test_nan_in_a_cached_depth_payload_raises(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 31],
            depths=[-1.0, -2.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

        cache_path = _playback_cache_path(env["csv_path"])
        payload = dict(np.load(cache_path, allow_pickle=True))
        payload["cp_frame_depth_data"] = np.array([-1.0, np.nan], dtype=np.float64)
        np.savez_compressed(cache_path, **payload)

        with pytest.raises(ValueError, match="NaN"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])


# ---------------------------------------------------------------------------
# 3.3 — duplicate (frame_index, vertex_id) raises. Assert, do not reduce.
# ---------------------------------------------------------------------------

class TestDuplicateFrameVertexPairsRaise:

    def test_duplicate_vertex_id_in_one_frame_raises(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1, 1],
            vertex_id=[30, 31, 30],
            depths=[-1.0, -2.0, -3.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError, match="duplicate"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

    def test_the_message_names_the_frame_and_the_vertex(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[42, 42, 42],
            vertex_id=[7, 7, 7],
            depths=[-1.0, -2.0, -3.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 42}]
        )
        with pytest.raises(ValueError) as exc:
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        message = str(exc.value)
        assert "frame_index=42" in message
        assert "[7]" in message
        assert ".parquet" in message

    def test_the_same_vertex_in_two_different_frames_is_fine(self, env):
        # Uniqueness is per frame, not per block: a vertex touched in consecutive
        # frames is the normal case and must not raise.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 2],
            vertex_id=[30, 30],
            depths=[-1.0, -5.0],
        )
        _write_series_csv(
            env["csv_path"],
            [
                {"cell": _cell([0.0]), "frame_index": 1},
                {"cell": _cell([0.0]), "frame_index": 2},
            ],
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_depths[0], [-1.0])
        np.testing.assert_array_equal(touch.frame_depths[1], [-5.0])

    def test_duplicates_are_not_reduced_to_one_row(self, env):
        # Guard against a future "fix" that dedups instead of raising: the deepest-wins
        # reduction the producer's own viewer uses would silently change the answer.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 30],
            depths=[-1.0, -9.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError, match="asserted, never reduced"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])


# ---------------------------------------------------------------------------
# 3.5 — NaN depth raises with file / frame / vertex context
# ---------------------------------------------------------------------------

class TestNanDepthRaises:

    def test_nan_depth_raises(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 31],
            depths=[-1.0, float("nan")],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        with pytest.raises(ValueError, match="NaN"):
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])

    def test_the_message_carries_file_frame_and_vertex(self, env):
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[77, 77, 77],
            vertex_id=[3, 4, 5],
            depths=[-1.0, float("nan"), -2.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0, 2.0]), "frame_index": 77}]
        )
        with pytest.raises(ValueError) as exc:
            _load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"])
        message = str(exc.value)
        assert "frame_index=77" in message      # frame
        assert "[4]" in message                  # vertex_id
        assert "[1]" in message                  # position within the frame
        assert ".parquet" in message             # file
        assert BLOCK_FILE in message             # the block it came from

    def test_zero_depth_is_not_absent(self, env):
        # A grazing contact measured at exactly 0.0 mm is a real measurement and must
        # load. Only NaN is absence.
        _write_sidecar(
            env["blocks_dir"],
            frame_index=[1, 1],
            vertex_id=[30, 31],
            depths=[0.0, -2.0],
        )
        _write_series_csv(
            env["csv_path"], [{"cell": _cell([0.0, 1.0]), "frame_index": 1}]
        )
        touch = _touch(_load(env["tmp_path"], env["csv_path"], env["ply_path"], env["blocks_dir"]))
        np.testing.assert_array_equal(touch.frame_depths[0], [0.0, -2.0])
