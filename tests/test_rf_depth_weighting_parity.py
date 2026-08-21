"""``depth_weight_alpha = 0`` is an exact no-op — and the viewer draws the saved map.

What these tests prove
----------------------
At ``depth_weight_alpha = 0`` every contact weight is exactly ``1.0``, and the
weighted estimator therefore reproduces the unweighted one **byte for byte** —
not to a tolerance, not to ``rtol``, but under ``np.array_equal``. That is
achievable only because there is no ``if alpha == 0`` fast path anywhere: the
same reduction call runs, over the same values, in the same order, with a
weights array whose entries happen to be ones. The comparison is made against
the frozen pre-weighting transcription of the estimator
(``test_vertex_accumulator._legacy_compute_touch_rf``) at the function level and
against a legacy-backed run of the real stage at the **artifact** level, so the
claim covers ``single_touch_rf_maps_mean.npz`` as written to disk and not only
the arithmetic behind it.

What these tests do **not** prove
--------------------------------
They say **nothing** about agreement with maps produced before Phase 2.5. That
phase deleted the KDTree vertex snapping in ``touch_playback_data`` and took each
contact point's ``vertex_id`` off the depth-field sidecar row instead, which
moves credit between vertices **on purpose**. The baseline these tests compare
against is the one re-pinned after that change (``tests/rf_accumulator_fixtures``
— see its "Baseline status after Phase 2.5" note): it pins the *reduction*, and
its vertex indices are sidecar ``vertex_id`` values.

So a passing suite here means "the weighting machinery is a genuine no-op at
alpha = 0". It does not mean "these are the maps the pipeline used to produce",
and setting ``depth_weight_alpha: 0.0`` in config is therefore a **baseline**,
not a rollback of the branch. How far the Phase 2.5 vertex fix actually moved
the maps is a property of the real data and is measured by
``scripts/diagnose_vertex_reassignment.py``, which must be run on a session
before any result from this branch is trusted.

The last class closes the gap the plan's manual-verification item names — "open a
viewer window and the saved map for the same touch; confirm they now agree". It
is checked here headlessly instead, at four alphas, so it cannot silently rot.

Nothing here reads the experimental database. Every array is built in code.
"""

import inspect
import json
from typing import List, Tuple

import numpy as np
import pytest

import rf_accumulator_fixtures as fixtures

# The frozen pre-weighting transcription of the estimator. It is imported rather
# than copied: a second copy is a second thing to keep frozen, and the whole
# value of a characterization baseline is that exactly one of it exists.
from test_vertex_accumulator import _legacy_compute_touch_rf

from analysis.receptive_field_mapping.data.touch_frame_weights import touch_frame_weights
from analysis.receptive_field_mapping.data.touch_playback_data import (
    DepthFieldProvenance,
    PlaybackData,
    PlaybackSessionData,
    TouchEvent,
)
from analysis.receptive_field_mapping.data.vertex_weights import vertex_weights
from analysis.receptive_field_mapping.gui.touch_playback_explorer import (
    TouchPlaybackExplorer,
    mean_heatmap_scalars,
    replay_touch_frames,
)
from analysis.receptive_field_mapping.pipelines import rf_single_touch_pipeline as pipe
from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (
    TouchRFMaps,
    _compute_touch_rf,
)


# ----------------------------------------------------------------------
# Exact-equality helpers (modelled on tests/test_rf_response_fields_parity.py)
# ----------------------------------------------------------------------

def _assert_exactly_equal(name: str, a, b) -> None:
    """Byte-for-byte for arrays, ``==`` for scalars; never a tolerance.

    ``equal_nan=True``: a NaN in the same cell counts as identical. NaN is a
    legitimate value in these maps (a unit not held during the touch window),
    and it must land in exactly the same cells as before.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), (
            f"{name} differs (array not byte-identical)"
        )
    else:
        assert a == b, f"{name} differs: {a!r} != {b!r}"


def _split_pairs(pairs: List[Tuple[int, float]]):
    """``[(idx, value), ...]`` -> ``(int64 indices, float64 values)``.

    Comparing the two channels as arrays rather than as lists of tuples is what
    lets ``np.array_equal(..., equal_nan=True)`` treat NaN as identical while
    keeping every other comparison exact.
    """
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx = np.array([p[0] for p in pairs], dtype=np.int64)
    val = np.array([p[1] for p in pairs], dtype=np.float64)
    return idx, val


def _assert_pairs_exactly_equal(name: str, got, want) -> None:
    got_idx, got_val = _split_pairs(got)
    want_idx, want_val = _split_pairs(want)
    _assert_exactly_equal(f"{name} vertex indices", got_idx, want_idx)
    _assert_exactly_equal(f"{name} values", got_val, want_val)


_TOUCHES = [
    ("worked_example", fixtures.worked_example_touch, fixtures.WORKED_EXAMPLE_N_VERTICES),
    (
        "nan_and_duplicate",
        fixtures.nan_and_duplicate_touch,
        fixtures.NAN_AND_DUPLICATE_N_VERTICES,
    ),
]


# ----------------------------------------------------------------------
# 6.1 — the estimator itself
# ----------------------------------------------------------------------

class TestAlphaZeroReproducesTheBaselineExactly:
    """The weighted estimator at ``alpha = 0`` == the frozen unweighted estimator."""

    @pytest.mark.parametrize("neuron_mode", ["iff", "spike"])
    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_mean_and_max_are_byte_identical(
        self, name, make_touch, n_verts, neuron_mode
    ):
        touch = make_touch()
        maps = _compute_touch_rf(touch, n_verts, neuron_mode, 0.0)
        want_mean, want_max = _legacy_compute_touch_rf(touch, n_verts, neuron_mode)
        _assert_pairs_exactly_equal(f"{name}/{neuron_mode} mean", maps.mean_pairs, want_mean)
        _assert_pairs_exactly_equal(f"{name}/{neuron_mode} max", maps.max_pairs, want_max)

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_every_weight_is_exactly_one(self, name, make_touch, n_verts):
        """The mechanism, not just the result.

        Equality of the *outputs* could in principle survive a subtly wrong
        weight vector (any constant weight cancels between numerator and
        denominator). This asserts the vector itself, so ``alpha = 0`` is
        pinned as "all weights are 1", not merely "the answer came out the
        same".
        """
        touch = make_touch()
        for fi, verts in enumerate(touch.frame_vertex_indices):
            if len(verts) == 0:
                continue
            weights = touch_frame_weights(touch, fi, 0.0)
            _assert_exactly_equal(
                f"{name} frame {fi} weights",
                weights,
                np.ones(len(verts), dtype=np.float64),
            )

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_weight_sum_is_the_contact_count(self, name, make_touch, n_verts):
        """``weight_sum`` at ``alpha = 0`` *is* the old ``contact_count``.

        The field was renamed when its definition changed. At ``alpha = 0`` the
        two definitions coincide, which is what makes the renamed field readable
        against an old figure — and is the only alpha at which that is true.
        """
        touch = make_touch()
        maps = _compute_touch_rf(touch, n_verts, "iff", 0.0)
        counts = np.zeros(n_verts, dtype=np.float64)
        for verts in touch.frame_vertex_indices:
            np.add.at(counts, verts, 1.0)
        idx, weight_sums = _split_pairs(maps.weight_sum_pairs)
        _assert_exactly_equal(f"{name} weight_sum", weight_sums, counts[idx])


# ----------------------------------------------------------------------
# 6.2 — clamped grazing contacts at alpha = 0
# ----------------------------------------------------------------------

def _grazing_touch() -> TouchEvent:
    """Three vertices, two frames, v3 grazing in **both** frames.

    ``signed_depth_mm`` is positive-or-zero for v3 — the point sat on or just
    above the surface — so its penetration is ``<= 0`` and clamps to ``0``.

    * ``alpha = 0``: ``0.0 ** 0.0 == 1.0`` exactly, so v3 still weighs 1 and is
      credited exactly as the unweighted map credits it.
    * ``alpha > 0``: v3's weights are all 0, its ``sum(w)`` is 0, and the
      estimator raises rather than emitting ``0/0``.

    Both halves are asserted below; the second is what makes the first a
    statement about IEEE arithmetic rather than an accident.
    """
    frame_vertices = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    ]
    return TouchEvent(
        block_order_id="7",
        trial_id=2,
        single_touch_id=5,
        gesture_type="stroke_distal",
        frame_contact_pts=[
            np.zeros((3, 3), dtype=np.float64),
            np.zeros((3, 3), dtype=np.float64),
        ],
        frame_vertex_indices=frame_vertices,
        # Negative = penetrating. v3 is +0.5 mm (above the surface) then exactly
        # 0.0 mm (touching it): a grazing contact and a zero-depth contact, both
        # of which clamp to zero penetration.
        frame_depths=[
            np.array([-2.0, -1.0, 0.5], dtype=np.float64),
            np.array([-4.0, -2.0, 0.0], dtype=np.float64),
        ],
        frame_spikes=np.array([True, False], dtype=bool),
        frame_iff=np.array([10.0, 30.0], dtype=np.float64),
    )


class TestClampedGrazingContactsAtAlphaZero:
    """``max(p, 0)`` is ``0.0`` and ``0.0 ** 0 == 1.0`` — grazing still weighs 1."""

    def test_zero_penetration_weighs_exactly_one(self):
        weights = vertex_weights(np.array([2.0, 1.0, 0.0, -0.5]), 0.0)
        _assert_exactly_equal(
            "grazing weights", weights, np.ones(4, dtype=np.float64)
        )

    def test_a_grazing_vertex_is_credited_exactly_as_before(self):
        """End to end: the grazing vertex's mean equals the unweighted mean.

        ``vertex_weights`` returning ones is a unit fact; this is the one that
        matters — a grazing contact reaches the saved map at ``alpha = 0`` with
        the same value it always had, rather than being dropped or zeroed.
        """
        touch = _grazing_touch()
        maps = _compute_touch_rf(touch, 3, "iff", 0.0)
        want_mean, want_max = _legacy_compute_touch_rf(touch, 3, "iff")
        _assert_pairs_exactly_equal("grazing mean", maps.mean_pairs, want_mean)
        _assert_pairs_exactly_equal("grazing max", maps.max_pairs, want_max)
        # The grazing vertex is present and carries the plain average of the two
        # frames — 20 Hz, exactly as the unweighted estimator gives it.
        assert dict(maps.mean_pairs)[2] == 20.0

    def test_the_same_vertex_raises_above_alpha_zero(self):
        """``alpha = 0`` is the *only* exponent at which grazing is harmless.

        Above it the vertex accumulates zero total weight, and the estimator
        must say so rather than fall back to the unweighted mean — mixing two
        estimators in one map is the failure this raise exists to prevent.
        """
        touch = _grazing_touch()
        with pytest.raises(ValueError, match="accumulated a total"):
            _compute_touch_rf(touch, 3, "iff", 1.0)


# ----------------------------------------------------------------------
# 6.1 — the same claim at the artifact level
# ----------------------------------------------------------------------

def _fake_playback(n_vertices: int, touches, sidecar: str) -> PlaybackData:
    """A ``PlaybackData`` over *touches*, spread across two (block, trial) keys.

    Two keys rather than one so ``touch_id_map`` is non-trivial: the stage
    numbers touches in block/trial iteration order, and a single-entry map would
    not notice if that ordering moved.
    """
    return PlaybackData(
        session_data=PlaybackSessionData(
            forearm_vertices=np.zeros((n_vertices, 3), dtype=np.float64),
            forearm_vertex_colors=None,
        ),
        block_order_ids=[t.block_order_id for t in touches],
        trial_ids_by_block={t.block_order_id: [t.trial_id] for t in touches},
        touches_by_block_trial={
            (t.block_order_id, t.trial_id): [t] for t in touches
        },
        depth_field_provenance=[
            DepthFieldProvenance(
                source_block_file=(
                    "ST99-01_semicontrolled_block-order-02_merged_data.csv"
                ),
                sidecar_path=sidecar,
                coordinate_space="rf_centered",
            )
        ],
    )


def _legacy_backed_compute(touch, n_vertices, neuron_mode, depth_weight_alpha):
    """Stand-in for ``_compute_touch_rf`` that runs the **frozen** estimator.

    The confidence channel did not exist before the weighting, so it is returned
    empty and is not part of the comparison: the parity claim is about
    ``rf_data``, the array the maps themselves live in.
    """
    assert depth_weight_alpha == 0.0, (
        "the legacy estimator has no alpha; this shim is only valid at 0.0"
    )
    mean_pairs, max_pairs = _legacy_compute_touch_rf(touch, n_vertices, neuron_mode)
    return TouchRFMaps(mean_pairs, max_pairs, [], [])


@pytest.fixture
def rf_stage(tmp_path, monkeypatch):
    """Run the real stage over a synthetic session and hand back the saved maps.

    The playback loader and the PLY resolver are replaced, so no experimental
    data is read and what is exercised is the stage itself: the loop over
    touches, the ``.npz`` writer, and the sentinel.
    """
    session_id = "ST99-01"
    merged_root = tmp_path / "3_merged" / session_id
    (merged_root / "blocks_rf_centered").mkdir(parents=True)
    csv_path = merged_root / f"{session_id}_semicontrolled_aggregated.csv"
    csv_path.write_text("placeholder\n", encoding="utf-8")

    preparation_dir = tmp_path / "4_analysed" / "preparation"
    preparation_dir.mkdir(parents=True)
    (preparation_dir / f"{session_id}_prepared.csv").write_text("x\n", encoding="utf-8")

    ply = merged_root / f"{session_id}_forearm.ply"
    ply.write_text("ply\n", encoding="utf-8")
    sidecar = str(merged_root / "blocks_rf_centered" / "block-order-02.parquet")

    n_vertices = fixtures.NAN_AND_DUPLICATE_N_VERTICES
    touches = [fixtures.worked_example_touch(), fixtures.nan_and_duplicate_touch()]

    monkeypatch.setattr(pipe, "resolve_forearm_ply", lambda *_a, **_k: ply)
    monkeypatch.setattr(
        pipe,
        "load_playback_data",
        lambda **_k: _fake_playback(n_vertices, touches, sidecar),
    )

    def _run(alpha: float, run_name: str, legacy: bool = False) -> dict:
        output_dir = tmp_path / "4_analysed" / run_name
        if legacy:
            monkeypatch.setattr(pipe, "_compute_touch_rf", _legacy_backed_compute)
        pipe.run_single_touch_rf_mapping(
            input_items=[(csv_path, tmp_path)],
            output_dir=output_dir,
            depth_weight_alpha=alpha,
            force=True,
            preparation_dir=preparation_dir,
            contact_depth_field={
                "blocks_stage_dir": "blocks_rf_centered",
                "block_csv_stem_suffix": "_pca-xyz",
            },
        )
        if legacy:
            monkeypatch.undo()
            monkeypatch.setattr(pipe, "resolve_forearm_ply", lambda *_a, **_k: ply)
            monkeypatch.setattr(
                pipe,
                "load_playback_data",
                lambda **_k: _fake_playback(n_vertices, touches, sidecar),
            )
        out = {}
        for metric in ("mean", "max"):
            npz = output_dir / session_id / f"single_touch_rf_maps_{metric}.npz"
            with np.load(npz, allow_pickle=True) as data:
                out[metric] = {
                    "touch_id_map": data["touch_id_map"].item(),
                    "rf_data": data["rf_data"].item(),
                    "rf_weight_sum": data["rf_weight_sum"].item(),
                }
        out["summary"] = json.loads(
            (output_dir / session_id / "single_touch_rf_summary.json").read_text(
                encoding="utf-8"
            )
        )
        return out

    return _run


class TestSavedMapsAreByteIdenticalAtAlphaZero:
    """``single_touch_rf_maps_*.npz`` at ``alpha = 0`` == the unweighted stage's."""

    def test_rf_data_is_byte_identical(self, rf_stage):
        weighted = rf_stage(0.0, "weighted_alpha0")
        legacy = rf_stage(0.0, "legacy_baseline", legacy=True)

        # Non-vacuity: prove the frozen estimator really replaced the weighted one
        # for the baseline run. The frozen one predates the confidence channel and
        # emits none, so an empty ``rf_weight_sum`` is the fingerprint of the
        # substitution having taken effect. Without this, a monkeypatch that
        # silently failed to apply would make every assertion below compare the
        # weighted run against itself and pass.
        assert all(not v for v in legacy["mean"]["rf_weight_sum"].values()), (
            "the legacy estimator was not substituted — this comparison is vacuous"
        )
        assert any(v for v in weighted["mean"]["rf_weight_sum"].values()), (
            "the weighted run emitted no confidence channel at all"
        )

        for metric in ("mean", "max"):
            assert weighted[metric]["touch_id_map"] == legacy[metric]["touch_id_map"], (
                f"{metric}: touch_id_map differs"
            )
            got, want = weighted[metric]["rf_data"], legacy[metric]["rf_data"]
            assert sorted(got) == sorted(want), f"{metric}: touch ids differ"
            for touch_id in sorted(want):
                _assert_pairs_exactly_equal(
                    f"{metric} touch {touch_id}", got[touch_id], want[touch_id]
                )

    def test_the_run_is_identifiable_as_a_baseline_on_disk(self, rf_stage):
        """A parity run must be distinguishable from a shipped run.

        ``alpha`` is in the sentinel precisely so that "these maps are
        byte-identical to the unweighted ones" is a recorded fact about the run
        rather than an inference someone makes later from the numbers.
        """
        assert rf_stage(0.0, "recorded")["summary"]["depth_weight_alpha"] == 0.0

    def test_alpha_one_actually_moves_the_map(self, rf_stage):
        """The negative control.

        Without this, every assertion above would also pass against an
        implementation that ignored ``alpha`` entirely.
        """
        at_zero = rf_stage(0.0, "control_alpha0")["mean"]["rf_data"]
        at_one = rf_stage(1.0, "control_alpha1")["mean"]["rf_data"]
        differing = [
            tid
            for tid in at_zero
            if not np.array_equal(
                _split_pairs(at_zero[tid])[1],
                _split_pairs(at_one[tid])[1],
                equal_nan=True,
            )
        ]
        assert differing, "alpha = 1 produced the alpha = 0 map; weights are inert"


# ----------------------------------------------------------------------
# The viewer draws the saved map
# ----------------------------------------------------------------------

class TestPlaybackViewerAgreesWithTheSavedMap:
    """The heatmap on screen and the map on disk are the same numbers.

    The plan's manual-verification item — "open a viewer window and the saved map
    for the same touch; confirm they now agree" — is checked here instead, at
    four alphas and with no ``QApplication``. It is worth automating because the
    divergence it guards against is invisible: the window would simply draw a
    plausible, wrong map.

    One documented difference remains, and it is a presentation choice rather
    than a numeric one: ``_compute_touch_rf`` *drops* vertices whose mean is NaN
    (so a "no neural data" touch reports zero vertices instead of an invisible
    heatmap), while the viewer leaves them NaN in the scalar array and paints
    them grey. The comparison below is therefore over the vertices the saved map
    contains; the NaN fixture is included so that difference is exercised rather
    than avoided.
    """

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_replayed_heatmap_equals_the_pipeline_mean(
        self, name, make_touch, n_verts, alpha
    ):
        touch = make_touch()
        n_frames = len(touch.frame_iff)

        iff_accum, _ = replay_touch_frames(touch, n_frames, n_verts, alpha)
        drawn = mean_heatmap_scalars(iff_accum.value_sum, iff_accum.weight_sum)

        maps = _compute_touch_rf(touch, n_verts, "iff", alpha)
        idx, saved = _split_pairs(maps.mean_pairs)
        _assert_exactly_equal(f"{name} alpha={alpha}", drawn[idx], saved)

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_the_viewers_weight_sum_is_the_saved_weight_sum(self, alpha):
        """The confidence channel agrees too, so a dimmed vertex on screen is a
        thin-evidence vertex in the file."""
        touch = fixtures.worked_example_touch()
        n_verts = fixtures.WORKED_EXAMPLE_N_VERTICES
        iff_accum, spike_accum = replay_touch_frames(
            touch, len(touch.frame_iff), n_verts, alpha
        )
        # Both displayed channels are fed the same weights, which is what makes
        # ``iff_accum.weight_sum`` usable as *the* weight sum for either.
        _assert_exactly_equal(
            "channel weight sums", iff_accum.weight_sum, spike_accum.weight_sum
        )
        maps = _compute_touch_rf(touch, n_verts, "iff", alpha)
        idx, saved = _split_pairs(maps.weight_sum_pairs)
        _assert_exactly_equal(
            f"weight_sum alpha={alpha}", iff_accum.weight_sum[idx], saved
        )

    def test_the_window_cannot_be_opened_without_an_alpha(self):
        """No default, at the window boundary too.

        A default here would be the exact reintroduction of the bug this closes:
        the window would open, draw the unweighted map, and disagree with the
        file without anyone being told.
        """
        params = inspect.signature(TouchPlaybackExplorer.__init__).parameters
        alpha = params["depth_weight_alpha"]
        assert alpha.default is inspect.Parameter.empty
        assert alpha.kind is inspect.Parameter.KEYWORD_ONLY
