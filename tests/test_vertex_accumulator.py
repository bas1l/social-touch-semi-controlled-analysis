"""Characterization + unit tests for the shared per-vertex accumulator.

Two jobs, in this order of importance:

1. **Characterization.** Nine copy-pasted accumulator blocks were merged into
   ``data/vertex_accumulator.py``. Every ``_legacy_*`` function below is a
   verbatim transcription of the code that call site ran *before* the merge;
   each test asserts the merged code reproduces it with ``np.array_equal``, not
   ``assert_allclose``. Exact equality is the right invariant because the merge
   is supposed to change nothing at all: any difference, even in the last
   mantissa bit, means the reduction now blocks its summation differently and
   the depth-weighting parity test (``alpha = 0`` reproduces today's map) would
   have nothing solid to stand on.

   The ``_legacy_*`` functions were validated against the real pre-merge call
   sites before the merge was applied; they are frozen and must never be
   "fixed" to match new behaviour. If one of them disagrees with the shared
   accumulator, the shared accumulator is wrong.

2. **Unit.** The accumulator's own contract: the four channels it reports, the
   fact that ``value_max`` is unweighted, and the inputs it refuses.

Three of the nine sites are Qt window methods that draw while they compute. For
those, the numeric core was lifted to a module-level function in the same
module (``vertex_value_mean``, ``compute_contact_point_heatmap``,
``accumulate_touch_frame`` / ``replay_touch_frames`` / ``mean_heatmap_scalars``)
and the test pins that function, since instantiating the window would need a
QApplication and a render surface. The Qt methods are then one-line delegations
with no arithmetic left in them.
"""

import numpy as np
import pytest

import rf_accumulator_fixtures as fixtures
from analysis.receptive_field_mapping.data.vertex_accumulator import (
    AccumResult,
    accumulate_vertex_values,
    accumulate_vertex_values_into,
    empty_accumulator,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import compute_rf_heatmap
from analysis.receptive_field_mapping.gui.rf_feature_space_explorer import vertex_value_mean
from analysis.receptive_field_mapping.gui.touch_population_explorer import (
    compute_contact_point_heatmap,
)
from analysis.receptive_field_mapping.gui.touch_playback_explorer import (
    accumulate_touch_frame,
    mean_heatmap_scalars,
    replay_touch_frames,
)
from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (
    _compute_touch_rf,
)


def _assert_exactly_equal(name: str, a, b) -> None:
    """Exact equality: byte-for-byte for arrays, ``==`` for scalars/pairs.

    ``equal_nan=True`` so that NaN in the same cell counts as identical — the
    RF maps carry NaN for uncontacted vertices and for frames where the unit
    was not held, and both must land in exactly the same cells as before.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), (
            f"{name} differs (array not byte-identical)"
        )
    elif isinstance(a, (tuple, list)):
        assert list(a) == list(b), f"{name} differs: {a!r} != {b!r}"
    else:
        assert a == b, f"{name} differs: {a!r} != {b!r}"


# ======================================================================
# Frozen pre-merge implementations — DO NOT EDIT to match new behaviour
# ======================================================================

def _legacy_compute_touch_rf(touch, n_vertices, neuron_mode):
    """Site 1 — ``rf_single_touch_pipeline._compute_touch_rf``."""
    val_sum = np.zeros(n_vertices, dtype=np.float64)
    val_max = np.full(n_vertices, -np.inf, dtype=np.float64)
    contact_count = np.zeros(n_vertices, dtype=np.float64)

    n_frames = len(touch.frame_vertex_indices)
    if neuron_mode == "iff":
        neuron_values = touch.frame_iff
    elif neuron_mode == "spike":
        neuron_values = touch.frame_spikes.astype(np.float64)
    else:
        raise ValueError(neuron_mode)

    for fi in range(n_frames):
        verts = touch.frame_vertex_indices[fi]
        if len(verts) == 0:
            continue
        np.add.at(val_sum, verts, neuron_values[fi])
        np.maximum.at(val_max, verts, neuron_values[fi])
        np.add.at(contact_count, verts, 1.0)

    contacted_mask = contact_count > 0
    if not contacted_mask.any():
        return [], []

    contacted_indices = np.where(contacted_mask)[0]
    mean_values = val_sum[contacted_indices] / contact_count[contacted_indices]
    max_values = val_max[contacted_indices]

    valid = ~np.isnan(mean_values)
    if not valid.any():
        return [], []
    contacted_indices = contacted_indices[valid]
    mean_values = mean_values[valid]
    max_values = max_values[valid]
    mean_pairs = [(int(i), float(v)) for i, v in zip(contacted_indices, mean_values)]
    max_pairs = [(int(i), float(v)) for i, v in zip(contacted_indices, max_values)]
    return mean_pairs, max_pairs


def _legacy_compute_rf_heatmap(touch_indices, rf_vertex_indices, rf_values, n_verts):
    """Site 2 — ``rf_population_heatmap.compute_rf_heatmap``."""
    result = np.zeros(n_verts, dtype=np.float64)
    count = np.zeros(n_verts, dtype=np.int64)
    for idx in touch_indices:
        verts = rf_vertex_indices[idx]
        vals = rf_values[idx]
        if len(verts) == 0:
            continue
        np.add.at(result, verts, vals)
        np.add.at(count, verts, 1)
    nonzero = count > 0
    result[nonzero] /= count[nonzero]
    result[~nonzero] = np.nan
    return result


def _legacy_population_heatmap(cp_vertex_idx, cp_iff, cp_spike, cp_mask, n_verts, mode):
    """Site 3 — ``TouchPopulationExplorer._compute_heatmap`` (bincount-based)."""
    active_verts = cp_vertex_idx[cp_mask]
    contact_count = np.bincount(active_verts, minlength=n_verts).astype(float)

    if mode == "mean_iff":
        val = (
            np.bincount(active_verts, weights=cp_iff[cp_mask], minlength=n_verts)
            / np.maximum(contact_count, 1)
        )
    elif mode == "cumulative_iff":
        val = np.bincount(active_verts, weights=cp_iff[cp_mask], minlength=n_verts)
    elif mode == "spike_density":
        val = (
            np.bincount(
                active_verts,
                weights=cp_spike[cp_mask].astype(float),
                minlength=n_verts,
            )
            / np.maximum(contact_count, 1)
        )
    else:
        raise ValueError(mode)

    result = val.astype(float)
    result[contact_count == 0] = np.nan
    return result


def _legacy_feature_space_vertex_mean(vertex_idx, values, n_verts):
    """Sites 4 and 5 — the two identical bincount blocks in the feature-space explorer."""
    vertex_val_sum = np.bincount(vertex_idx, weights=values, minlength=n_verts)
    vertex_n_contacts = np.bincount(vertex_idx, minlength=n_verts).astype(float)
    vertex_density = np.divide(
        vertex_val_sum, vertex_n_contacts,
        out=np.zeros(n_verts), where=vertex_n_contacts > 0,
    )
    return vertex_density, vertex_n_contacts


def _legacy_replay(touch, n_frames, n_verts):
    """Sites 6 and 7 — the two replay loops in the playback explorer."""
    spike_sum = np.zeros(n_verts, dtype=np.float64)
    iff_sum = np.zeros(n_verts, dtype=np.float64)
    contact_count = np.zeros(n_verts, dtype=np.float64)
    for fi in range(n_frames):
        verts = touch.frame_vertex_indices[fi]
        np.add.at(spike_sum, verts, float(touch.frame_spikes[fi]))
        np.add.at(iff_sum, verts, float(touch.frame_iff[fi]))
        np.add.at(contact_count, verts, 1.0)
    return spike_sum, iff_sum, contact_count


def _legacy_accumulate_frame(spike_sum, iff_sum, contact_count, touch, frame_idx):
    """Site 8 — the incremental O(K) single-frame update in ``_render_frame``."""
    verts_this_frame = touch.frame_vertex_indices[frame_idx]
    spike_this_frame = float(touch.frame_spikes[frame_idx])
    iff_this_frame = float(touch.frame_iff[frame_idx])
    np.add.at(spike_sum, verts_this_frame, spike_this_frame)
    np.add.at(iff_sum, verts_this_frame, iff_this_frame)
    np.add.at(contact_count, verts_this_frame, 1.0)


def _legacy_export_scalars(value_sum, contact_count):
    """Site 9 — the inline mean in the render-to-video loop."""
    has_contact = contact_count > 0
    return np.where(
        has_contact,
        value_sum / np.where(has_contact, contact_count, 1.0),
        np.nan,
    )


# ======================================================================
# Characterization — one class per call site
# ======================================================================

_TOUCHES = [
    ("worked_example", fixtures.worked_example_touch, fixtures.WORKED_EXAMPLE_N_VERTICES),
    ("nan_and_duplicate", fixtures.nan_and_duplicate_touch, fixtures.NAN_AND_DUPLICATE_N_VERTICES),
]


class TestSite1SingleTouchPipeline:
    """``rf_single_touch_pipeline._compute_touch_rf``."""

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    @pytest.mark.parametrize("neuron_mode", ["iff", "spike"])
    def test_matches_pre_merge(self, name, make_touch, n_verts, neuron_mode):
        touch = make_touch()
        # alpha = 0 is the baseline: every weight is exactly 1.0 and the
        # weighted code path must reproduce the pre-merge arrays byte for byte.
        got_mean, got_max, _, _ = _compute_touch_rf(touch, n_verts, neuron_mode, 0.0)
        want_mean, want_max = _legacy_compute_touch_rf(touch, n_verts, neuron_mode)
        _assert_exactly_equal(f"{name}/{neuron_mode} mean_pairs", got_mean, want_mean)
        _assert_exactly_equal(f"{name}/{neuron_mode} max_pairs", got_max, want_max)

    def test_worked_example_baseline_values(self):
        """The unweighted means the plan's worked example predicts.

        Hand-computed, not read back from the implementation: v2 is the peak at
        75.0 Hz even though the true receptive field is at v3. That misplaced
        peak is the whole reason depth weighting is being added, so it is
        pinned here as the thing the change is expected to move.
        """
        mean_pairs = _compute_touch_rf(
            fixtures.worked_example_touch(),
            fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff",
            0.0,
        ).mean_pairs
        got = dict(mean_pairs)
        assert got[0] == 50.0                      # v1, one frame at 50 Hz
        assert got[1] == 75.0                      # v2, flank -> today's peak
        assert got[2] == pytest.approx(190.0 / 3)  # v3, the true RF centre
        assert got[3] == 70.0                      # v4, flank
        assert got[4] == 40.0                      # v5, one frame at 40 Hz
        assert got[5] == pytest.approx(190.0 / 3)  # v6, always shallow
        assert got[6] == pytest.approx(190.0 / 3)  # v7, always deep
        assert max(got, key=got.get) == 1          # the peak sits on the wrong vertex


class TestSite2PopulationRfHeatmap:
    """``rf_population_heatmap.compute_rf_heatmap``."""

    def test_matches_pre_merge(self):
        m = fixtures.random_rf_touch_maps()
        got = compute_rf_heatmap(
            m["touch_indices"], m["rf_vertex_indices"], m["rf_values"], m["n_verts"]
        )
        want = _legacy_compute_rf_heatmap(
            m["touch_indices"], m["rf_vertex_indices"], m["rf_values"], m["n_verts"]
        )
        _assert_exactly_equal("compute_rf_heatmap", got, want)

    def test_uncontacted_vertices_stay_nan(self):
        got = compute_rf_heatmap(
            [0], [np.array([2], dtype=np.int64)], [np.array([7.5])], 5
        )
        assert got[2] == 7.5
        assert np.isnan(got[[0, 1, 3, 4]]).all()


class TestSite3PopulationExplorerHeatmap:
    """``touch_population_explorer.compute_contact_point_heatmap`` (was bincount-based).

    The pre-merge code reduced with ``np.bincount``; the shared accumulator
    reduces with ``np.add.at``. Those are different C loops, so this test is
    the evidence that they agree bit for bit on a long, badly-conditioned
    addition chain rather than an assumption that they must.
    """

    @pytest.mark.parametrize("mode", ["mean_iff", "cumulative_iff", "spike_density"])
    def test_matches_pre_merge(self, mode):
        c = fixtures.random_contact_points()
        got = compute_contact_point_heatmap(
            c["cp_vertex_idx"], c["cp_iff"], c["cp_spike"], c["cp_mask"],
            c["n_verts"], mode,
        )
        want = _legacy_population_heatmap(
            c["cp_vertex_idx"], c["cp_iff"], c["cp_spike"], c["cp_mask"],
            c["n_verts"], mode,
        )
        _assert_exactly_equal(f"population heatmap/{mode}", got, want)

    def test_unknown_mode_raises(self):
        c = fixtures.random_contact_points(n_points=10, n_verts=4)
        with pytest.raises(ValueError, match="unknown heatmap mode"):
            compute_contact_point_heatmap(
                c["cp_vertex_idx"], c["cp_iff"], c["cp_spike"], c["cp_mask"],
                c["n_verts"], "no_such_mode",
            )


class TestSites4And5FeatureSpaceExplorer:
    """``rf_feature_space_explorer.vertex_value_mean`` — also bincount-based before."""

    def test_render_3d_block_matches_pre_merge(self):
        c = fixtures.random_contact_points(seed=99)
        values = c["cp_iff"]
        got_mean, got_n = vertex_value_mean(c["cp_vertex_idx"], values, c["n_verts"])
        want_mean, want_n = _legacy_feature_space_vertex_mean(
            c["cp_vertex_idx"], values, c["n_verts"]
        )
        _assert_exactly_equal("vertex_density", got_mean, want_mean)
        _assert_exactly_equal("vertex_n_contacts", got_n, want_n)

    def test_filtered_block_matches_pre_merge(self):
        c = fixtures.random_contact_points(seed=99)
        mask = c["cp_mask"]
        active = c["cp_vertex_idx"][mask]
        values = c["cp_spike"][mask].astype(float)
        got_mean, got_n = vertex_value_mean(active, values, c["n_verts"])
        want_mean, want_n = _legacy_feature_space_vertex_mean(
            active, values, c["n_verts"]
        )
        _assert_exactly_equal("vertex_ratio", got_mean, want_mean)
        _assert_exactly_equal("vertex_n_contacts", got_n, want_n)


class TestSites6And7PlaybackReplay:
    """``touch_playback_explorer.replay_touch_frames`` — both replay loops.

    Site 6 replays ``[0, slider_value)`` and site 7 replays
    ``[0, current_frame]`` inclusive; the bounds are the callers' business, so
    both bounds are exercised here against the same frozen loop.
    """

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_all_bounds_match_pre_merge(self, name, make_touch, n_verts):
        touch = make_touch()
        for n_frames in range(len(touch.frame_iff) + 1):
            iff_accum, spike_accum = replay_touch_frames(touch, n_frames, n_verts)
            want_spike, want_iff, want_count = _legacy_replay(touch, n_frames, n_verts)
            _assert_exactly_equal(f"{name}[{n_frames}] iff_sum", iff_accum.value_sum, want_iff)
            _assert_exactly_equal(f"{name}[{n_frames}] spike_sum", spike_accum.value_sum, want_spike)
            _assert_exactly_equal(f"{name}[{n_frames}] count", iff_accum.weight_sum, want_count)
            _assert_exactly_equal(
                f"{name}[{n_frames}] count (spike channel)",
                spike_accum.weight_sum, want_count,
            )


class TestSite8PlaybackIncrementalFrame:
    """``touch_playback_explorer.accumulate_touch_frame`` — the O(K) incremental path.

    This is the site where a "reduce into a fresh array, then add" rewrite would
    silently stop being bit-identical: a frame whose vertex list repeats an
    index adds ``state + x + x`` here, whereas a batched reduction would add
    ``state + (x + x)``. The fixture contains such a frame on purpose.
    """

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_incremental_walk_matches_pre_merge(self, name, make_touch, n_verts):
        touch = make_touch()
        iff_accum = empty_accumulator(n_verts)
        spike_accum = empty_accumulator(n_verts)
        spike_sum = np.zeros(n_verts, dtype=np.float64)
        iff_sum = np.zeros(n_verts, dtype=np.float64)
        contact_count = np.zeros(n_verts, dtype=np.float64)

        for fi in range(len(touch.frame_iff)):
            accumulate_touch_frame(iff_accum, spike_accum, touch, fi)
            _legacy_accumulate_frame(spike_sum, iff_sum, contact_count, touch, fi)
            _assert_exactly_equal(f"{name} after frame {fi} iff", iff_accum.value_sum, iff_sum)
            _assert_exactly_equal(f"{name} after frame {fi} spike", spike_accum.value_sum, spike_sum)
            _assert_exactly_equal(f"{name} after frame {fi} count", iff_accum.weight_sum, contact_count)

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_incremental_equals_replay(self, name, make_touch, n_verts):
        """Stepping frame by frame must land exactly where a full replay lands.

        The window switches between the two paths (scrub the slider vs let it
        play), so a divergence here would make the same touch draw two
        different heatmaps depending on how the user got there.
        """
        touch = make_touch()
        n_frames = len(touch.frame_iff)
        iff_accum = empty_accumulator(n_verts)
        spike_accum = empty_accumulator(n_verts)
        for fi in range(n_frames):
            accumulate_touch_frame(iff_accum, spike_accum, touch, fi)
        replay_iff, replay_spike = replay_touch_frames(touch, n_frames, n_verts)
        _assert_exactly_equal(f"{name} iff", iff_accum.value_sum, replay_iff.value_sum)
        _assert_exactly_equal(f"{name} spike", spike_accum.value_sum, replay_spike.value_sum)


class TestSite9PlaybackExportLoop:
    """``touch_playback_explorer.mean_heatmap_scalars`` — the export loop's inline mean.

    Also the mean used by ``_update_heatmap_scalars``; before the merge the
    same expression was written out twice.
    """

    @pytest.mark.parametrize("name,make_touch,n_verts", _TOUCHES)
    def test_matches_pre_merge(self, name, make_touch, n_verts):
        touch = make_touch()
        iff_accum = empty_accumulator(n_verts)
        spike_accum = empty_accumulator(n_verts)
        spike_sum = np.zeros(n_verts, dtype=np.float64)
        iff_sum = np.zeros(n_verts, dtype=np.float64)
        contact_count = np.zeros(n_verts, dtype=np.float64)

        for fi in range(len(touch.frame_iff)):
            accumulate_touch_frame(iff_accum, spike_accum, touch, fi)
            _legacy_accumulate_frame(spike_sum, iff_sum, contact_count, touch, fi)
            _assert_exactly_equal(
                f"{name} frame {fi} iff scalars",
                mean_heatmap_scalars(iff_accum.value_sum, iff_accum.weight_sum),
                _legacy_export_scalars(iff_sum, contact_count),
            )
            _assert_exactly_equal(
                f"{name} frame {fi} spike scalars",
                mean_heatmap_scalars(spike_accum.value_sum, iff_accum.weight_sum),
                _legacy_export_scalars(spike_sum, contact_count),
            )


class TestReductionEquivalence:
    """``np.bincount`` and ``np.add.at`` must agree bit for bit.

    Three of the nine sites reduced with ``np.bincount`` and six with
    ``np.add.at``; the merge puts all nine on ``np.add.at``. Both are
    sequential unbuffered float64 additions in array order, so they should be
    identical — but "should be" is not evidence, and the depth-weighting parity
    test depends on it, so it is asserted directly here on a chain long enough
    and badly conditioned enough to expose any difference.
    """

    def test_bincount_matches_add_at(self):
        c = fixtures.random_contact_points(seed=7, n_points=200_000, n_verts=311)
        idx, vals = c["cp_vertex_idx"], c["cp_iff"]
        via_bincount = np.bincount(idx, weights=vals, minlength=c["n_verts"])
        via_add_at = np.zeros(c["n_verts"], dtype=np.float64)
        np.add.at(via_add_at, idx, vals)
        _assert_exactly_equal("bincount vs add.at", via_bincount, via_add_at)


# ======================================================================
# Unit — the accumulator's own contract
# ======================================================================

class TestAccumulatorContract:
    def test_reports_all_four_channels(self):
        idx = np.array([0, 0, 2, 2, 2], dtype=np.int64)
        vals = np.array([10.0, 30.0, 1.0, 2.0, 3.0])
        wgts = np.array([0.5, 0.25, 1.0, 0.5, 0.25])
        r = accumulate_vertex_values(idx, vals, wgts, 4)

        # Hand-computed, not read back from the implementation.
        assert r.value_sum[0] == 10.0 * 0.5 + 30.0 * 0.25
        assert r.value_sum[2] == 1.0 * 1.0 + 2.0 * 0.5 + 3.0 * 0.25
        assert r.weight_sum[0] == 0.75
        assert r.weight_sum[2] == 1.75
        assert r.weight_sq_sum[0] == 0.25 + 0.0625
        assert r.weight_sq_sum[2] == 1.0 + 0.25 + 0.0625
        # Untouched vertices report nothing, and are identified by weight_sum.
        assert r.value_sum[1] == 0.0
        assert r.weight_sum[1] == 0.0
        assert r.value_max[1] == -np.inf

    def test_value_max_is_unweighted(self):
        """A weighted maximum has no meaning, so weights must not touch it."""
        idx = np.array([0, 0, 0], dtype=np.int64)
        vals = np.array([10.0, 40.0, 25.0])
        heavy = accumulate_vertex_values(idx, vals, np.ones(3), 1)
        light = accumulate_vertex_values(idx, vals, np.full(3, 1e-6), 1)
        assert heavy.value_max[0] == 40.0
        assert light.value_max[0] == 40.0
        # ...while the weighted sum does respond to the weights.
        assert light.value_sum[0] != heavy.value_sum[0]

    def test_weighted_mean_is_hand_computable(self):
        """sum(w*x)/sum(w) stays in the units of x — a rate stays a rate.

        Three observations of 100 Hz at wildly different weights must still
        report 100 Hz; if the divisor were the count rather than the weight
        sum, the weight would survive in the answer as a scale factor.
        """
        idx = np.zeros(3, dtype=np.int64)
        vals = np.full(3, 100.0)
        wgts = np.array([0.13, 1.0, 0.42])
        r = accumulate_vertex_values(idx, vals, wgts, 1)
        assert r.value_sum[0] / r.weight_sum[0] == pytest.approx(100.0, rel=1e-12)

    def test_kish_n_eff_from_weight_sq_sum(self):
        """Flat weights give n_eff ~ N; one dominant weight gives n_eff ~ 1."""
        idx = np.zeros(4, dtype=np.int64)
        vals = np.ones(4)

        flat = accumulate_vertex_values(idx, vals, np.full(4, 0.7), 1)
        n_eff_flat = flat.weight_sum[0] ** 2 / flat.weight_sq_sum[0]
        assert n_eff_flat == pytest.approx(4.0, rel=1e-12)

        spiky = accumulate_vertex_values(idx, vals, np.array([1.0, 1e-6, 1e-6, 1e-6]), 1)
        n_eff_spiky = spiky.weight_sum[0] ** 2 / spiky.weight_sq_sum[0]
        assert n_eff_spiky == pytest.approx(1.0, abs=1e-4)

    def test_scalar_value_broadcasts_over_the_batch(self):
        """One frame carries one firing frequency however many vertices it touched."""
        idx = np.array([1, 3, 3], dtype=np.int64)
        scalar = accumulate_vertex_values(idx, 12.5, np.ones(3), 5)
        explicit = accumulate_vertex_values(idx, np.full(3, 12.5), np.ones(3), 5)
        _assert_exactly_equal("value_sum", scalar.value_sum, explicit.value_sum)
        _assert_exactly_equal("value_max", scalar.value_max, explicit.value_max)

    def test_empty_batch_is_a_no_op(self):
        r = empty_accumulator(3)
        before = r.value_sum.copy()
        accumulate_vertex_values_into(
            r, np.array([], dtype=np.int64), np.array([]), np.array([])
        )
        _assert_exactly_equal("value_sum", r.value_sum, before)
        assert (r.weight_sum == 0.0).all()

    def test_duplicate_indices_accumulate_sequentially(self):
        r = accumulate_vertex_values(
            np.array([2, 2, 2], dtype=np.int64), np.array([1.0, 2.0, 4.0]), np.ones(3), 3
        )
        assert r.value_sum[2] == 7.0
        assert r.weight_sum[2] == 3.0


class TestAccumulatorRejects:
    """Fail-fast: every rejected input is one that would otherwise corrupt a map."""

    def test_weights_are_required(self):
        with pytest.raises(TypeError):
            accumulate_vertex_values(np.array([0], dtype=np.int64), np.array([1.0]), 4)

    def test_scalar_weights_rejected(self):
        """``weights=1.0`` would reintroduce the implicit unweighted default."""
        with pytest.raises(ValueError, match="weights must have shape"):
            accumulate_vertex_values(np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0]), 1.0, 3)

    def test_mismatched_weight_length_rejected(self):
        with pytest.raises(ValueError, match="weights must have shape"):
            accumulate_vertex_values(
                np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0]), np.ones(3), 3
            )

    def test_mismatched_value_length_rejected(self):
        with pytest.raises(ValueError, match="values must be a scalar"):
            accumulate_vertex_values(
                np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0, 3.0]), np.ones(2), 3
            )

    def test_float_vertex_index_rejected(self):
        """Float indices would be truncated toward zero and credit the wrong vertex."""
        with pytest.raises(TypeError, match="integer"):
            accumulate_vertex_values(np.array([0.0, 1.9]), np.array([1.0, 2.0]), np.ones(2), 3)

    def test_negative_vertex_index_rejected(self):
        """Numpy would wrap -1 to the last vertex and corrupt it silently."""
        with pytest.raises(IndexError, match="outside"):
            accumulate_vertex_values(np.array([-1], dtype=np.int64), np.array([1.0]), np.ones(1), 3)

    def test_out_of_range_vertex_index_rejected(self):
        with pytest.raises(IndexError, match="outside"):
            accumulate_vertex_values(np.array([3], dtype=np.int64), np.array([1.0]), np.ones(1), 3)

    def test_two_dimensional_index_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            accumulate_vertex_values(
                np.array([[0, 1]], dtype=np.int64), np.array([1.0, 2.0]), np.ones(2), 3
            )

    def test_non_positive_n_vertices_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            empty_accumulator(0)

    def test_non_integer_n_vertices_rejected(self):
        with pytest.raises(TypeError, match="integer"):
            empty_accumulator(3.0)

    def test_result_must_be_an_accum_result(self):
        with pytest.raises(TypeError, match="AccumResult"):
            accumulate_vertex_values_into(
                np.zeros(3), np.array([0], dtype=np.int64), np.array([1.0]), np.ones(1)
            )

    def test_accum_result_arrays_cannot_be_swapped(self):
        r = empty_accumulator(3)
        with pytest.raises(Exception):
            r.value_sum = np.ones(3)
        assert isinstance(r, AccumResult)
