"""Tests for depth weighting: the weight function and the weighted estimator.

Two halves, deliberately kept in one file because they are two halves of one
claim:

1. ``data.vertex_weights.vertex_weights`` turns one frame's penetration depths
   into that frame's weights, and every degenerate case raises.
2. ``rf_single_touch_pipeline._compute_touch_rf`` uses those weights to produce a
   genuine **weighted mean** — and the plan's 7-vertex / 3-frame worked example
   comes out with the peak on the true receptive-field centre.

Every expected number in the estimator tests is **hand-computed from the plan's
published weight table**, never read back from the implementation. That
distinction matters more here than almost anywhere else in this repo: a wrong
weighting still returns a perfectly plausible firing rate in Hz, so a
self-comparison would pass while the map was wrong.

Nothing here reads the experimental database. Every array is a literal.
"""

import numpy as np
import pytest

import rf_accumulator_fixtures as fixtures
from analysis.receptive_field_mapping.data.contact_depth_field_io import (
    penetration_from_signed_mm,
)
from analysis.receptive_field_mapping.data.touch_frame_weights import (
    touch_frame_weights,
)
from analysis.receptive_field_mapping.data.touch_playback_data import TouchEvent
from analysis.receptive_field_mapping.data.vertex_accumulator import (
    accumulate_vertex_values_into,
    empty_accumulator,
)
from analysis.receptive_field_mapping.data.vertex_weights import vertex_weights
from analysis.receptive_field_mapping.pipelines.rf_single_touch_pipeline import (
    _compute_touch_rf,
)


ALL_ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 7.0]


# ======================================================================
# Part 1 — the weight function
# ======================================================================


class TestAlphaZeroIsTheBaseline:
    """``alpha = 0`` must give exactly ones, through the same arithmetic."""

    def test_all_ones(self):
        w = vertex_weights(np.array([0.9, 3.0, 1.26, 0.39, 2.76]), 0.0)
        assert np.array_equal(w, np.ones(5))

    def test_clamped_grazing_vertices_still_get_one(self):
        """A grazing point clamps to ``0.0``, and ``0.0 ** 0 == 1.0`` exactly.

        This is the case that quietly breaks if the clamp is replaced by a mask
        or if the exponent is applied before the normalisation. At ``alpha = 0``
        a vertex that never pressed in must still carry weight 1, because that
        is precisely what the unweighted baseline did.
        """
        w = vertex_weights(np.array([-0.4, 0.0, 2.0, -1e-9]), 0.0)
        assert np.array_equal(w, np.ones(4))

    def test_ones_are_bit_identical_to_np_ones(self):
        """Not merely close — the parity test needs byte-identical arrays."""
        rng = np.random.default_rng(20260821)
        depths = rng.random(500) * 10.0 - 2.0
        depths[0] = 5.0  # guarantee a positive maximum
        w = vertex_weights(depths, 0)
        assert np.array_equal(w, np.ones(500))
        assert w.dtype == np.float64


class TestAlphaOne:
    """The plan's tabulated example: depths 0.2 0.8 1.5 0.7 0.1."""

    DEPTHS = np.array([0.2, 0.8, 1.5, 0.7, 0.1])

    def test_matches_the_plans_rounded_table(self):
        w = vertex_weights(self.DEPTHS, 1.0)
        assert np.round(w, 2).tolist() == [0.13, 0.53, 1.0, 0.47, 0.07]

    def test_matches_the_exact_ratios(self):
        """The rounded table above cannot distinguish 1.5 from 1.4999 as the max."""
        w = vertex_weights(self.DEPTHS, 1.0)
        expected = self.DEPTHS / 1.5
        assert np.allclose(w, expected, rtol=0.0, atol=0.0) or np.allclose(
            w, expected, rtol=1e-15
        )


class TestDeepestVertex:
    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_receives_exactly_one(self, alpha):
        depths = np.array([0.31, 4.75, 0.02, 1.9])
        w = vertex_weights(depths, alpha)
        assert w[1] == 1.0
        assert w.max() == 1.0

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_weights_stay_in_the_unit_interval(self, alpha):
        w = vertex_weights(np.array([-0.5, 0.0, 0.31, 4.75, 1.9]), alpha)
        assert w.min() >= 0.0
        assert w.max() <= 1.0


class TestScaleInvariance:
    """Doubling every depth in a frame must leave the weights unchanged.

    This is the real reason to normalise at all: depth magnitudes may differ
    between sessions and the weight must not.
    """

    DEPTHS = np.array([0.9, 3.0, 1.26, 0.39, 2.76])

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_doubling_is_bit_identical(self, alpha):
        # Multiplying by exactly 2 is exact in binary floating point, so the
        # ratio is not merely close — it is the same float.
        assert np.array_equal(
            vertex_weights(self.DEPTHS, alpha),
            vertex_weights(self.DEPTHS * 2.0, alpha),
        )

    @pytest.mark.parametrize("factor", [0.001, 3.7, 1000.0])
    def test_arbitrary_rescaling(self, factor):
        assert np.allclose(
            vertex_weights(self.DEPTHS, 1.0),
            vertex_weights(self.DEPTHS * factor, 1.0),
            rtol=1e-14,
        )


class TestGrazingContacts:
    """A negative penetration is a real grazing contact, not an error."""

    @pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0, 7.0])
    def test_clamps_to_zero_and_never_goes_negative(self, alpha):
        w = vertex_weights(np.array([-0.4, -1e-9, 0.0, 2.0]), alpha)
        assert w[0] == 0.0
        assert w[1] == 0.0
        assert w[2] == 0.0
        assert w[3] == 1.0
        assert (w >= 0.0).all()

    def test_grazing_never_subtracts_from_the_weight_sum(self):
        """An unclamped negative depth would *reduce* ``sum(w)`` at odd alpha.

        With alpha = 1 and no clamp, a grazing point at -0.4 mm against a 2.0 mm
        maximum would contribute -0.2 to the denominator, so the weight sum
        would be smaller than the deep point's weight alone. The clamp is what
        makes ``sum(w)`` monotone in the number of contacts.
        """
        deep_only = vertex_weights(np.array([2.0]), 1.0).sum()
        with_grazing = vertex_weights(np.array([2.0, -0.4, -1.9]), 1.0).sum()
        assert with_grazing == deep_only


class TestDegenerateFramesRaise:
    def test_every_vertex_grazing_raises(self):
        with pytest.raises(ValueError, match="grazing"):
            vertex_weights(np.array([-0.4, -1e-9, -2.0]), 1.0)

    def test_all_zero_depths_raise(self):
        with pytest.raises(ValueError, match="grazing"):
            vertex_weights(np.zeros(4), 1.0)

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_d_max_zero_raises_at_every_alpha_including_zero(self, alpha):
        """``alpha = 0`` must not be rescued by ``nan ** 0 == 1.0``.

        NumPy would happily return ones here. That is IEEE trivia, not a design:
        a frame in which nothing was pressed is a statement about the data and
        must reach a human, whatever alpha happens to be.
        """
        with pytest.raises(ValueError, match="grazing"):
            vertex_weights(np.zeros(3), alpha)

    def test_nan_depth_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            vertex_weights(np.array([1.0, np.nan, 2.0]), 1.0)

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_nan_depth_raises_at_every_alpha(self, alpha):
        with pytest.raises(ValueError, match="non-finite"):
            vertex_weights(np.array([1.0, np.nan, 2.0]), alpha)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_infinite_depth_raises(self, bad):
        with pytest.raises(ValueError, match="non-finite"):
            vertex_weights(np.array([1.0, bad, 2.0]), 1.0)

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            vertex_weights(np.array([], dtype=np.float64), 1.0)

    def test_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            vertex_weights(np.ones((2, 3)), 1.0)


class TestAlphaIsRequiredAndValidated:
    def test_missing_alpha_is_a_type_error(self):
        """The whole point of the fail-fast rule: never a silent 1.0."""
        with pytest.raises(TypeError):
            vertex_weights(np.array([1.0, 2.0]))

    def test_alpha_is_positional(self):
        assert np.array_equal(
            vertex_weights(np.array([1.0, 2.0]), 1.0),
            np.array([0.5, 1.0]),
        )

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            vertex_weights(np.array([1.0, 2.0]), -1.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_alpha_raises(self, bad):
        with pytest.raises(ValueError, match="finite"):
            vertex_weights(np.array([1.0, 2.0]), bad)

    @pytest.mark.parametrize("bad", ["1.0", None, [1.0], True])
    def test_non_numeric_alpha_is_a_type_error(self, bad):
        with pytest.raises(TypeError, match="real number"):
            vertex_weights(np.array([1.0, 2.0]), bad)

    def test_integer_alpha_is_accepted(self):
        assert np.array_equal(
            vertex_weights(np.array([1.0, 2.0]), 1),
            np.array([0.5, 1.0]),
        )


class TestNoAlphaZeroBranch:
    """``alpha = 0`` must run the *same* arithmetic, not a shortcut.

    A ``if alpha == 0: return ones`` branch would be control coupling, and it
    would make the Phase 6 parity test vacuous — it would prove the old code
    still works rather than that the new code reduces to it. This is asserted on
    the source text because it is a structural claim that no numeric test can
    make: a branch returning ones is numerically indistinguishable from the
    arithmetic that produces them.
    """

    def test_the_source_contains_no_alpha_equality_branch(self):
        import inspect

        body = inspect.getsource(vertex_weights)
        # Strip the docstring so its prose about "alpha = 0" is not matched.
        body = body.replace(vertex_weights.__doc__ or "", "")
        assert "alpha == 0" not in body
        assert "alpha_f == 0" not in body

    def test_normalisation_happens_before_the_exponent(self):
        """The sum-normalised alternative gives ``1/K`` at alpha 0, not 1.

        Pinning the rejected formula's answer here is what makes the choice
        legible: the two differ *only* at alpha != 1, and only the chosen one
        preserves the baseline.
        """
        depths = np.array([0.2, 0.8, 1.5, 0.7, 0.1])
        chosen = vertex_weights(depths, 0.0)
        sum_normalised = depths**0.0 / np.sum(depths**0.0)
        assert np.array_equal(chosen, np.ones(5))
        assert np.allclose(sum_normalised, np.full(5, 0.2))

    def test_the_two_forms_agree_at_alpha_one_up_to_a_constant(self):
        depths = np.array([0.2, 0.8, 1.5, 0.7, 0.1])
        chosen = vertex_weights(depths, 1.0)
        sum_normalised = depths / depths.sum()
        assert np.allclose(chosen / chosen.sum(), sum_normalised, rtol=1e-14)


# ======================================================================
# Part 2 — the weighted estimator
# ======================================================================
#
# The plan's worked example, hand-computed. Seven vertices, three frames, IFF
# 50 / 100 / 40 Hz, the true receptive field at v3.
#
#   frame 0 (50 Hz)  v1 v2 v3 v6 v7   weights 0.30 1.00 0.42 0.13 0.92
#   frame 1 (100 Hz)    v2 v3 v4 v6 v7  weights 0.47 1.00 0.40 0.13 0.93
#   frame 2 (40 Hz)        v3 v4 v5 v6 v7  weights 0.50 1.00 0.30 0.10 0.95
#
# The weights below are the plan's own published table, not values read out of
# the implementation. They agree with the fixture's depths to ~1e-16 relative
# (0.39 / 3.00 is 0.13 up to the last mantissa bit), which is why the assertions
# use rtol=1e-12 rather than exact equality.

_W = {
    0: [(0.30, 50.0)],
    1: [(1.00, 50.0), (0.47, 100.0)],
    2: [(0.42, 50.0), (1.00, 100.0), (0.50, 40.0)],
    3: [(0.40, 100.0), (1.00, 40.0)],
    4: [(0.30, 40.0)],
    5: [(0.13, 50.0), (0.13, 100.0), (0.10, 40.0)],
    6: [(0.92, 50.0), (0.93, 100.0), (0.95, 40.0)],
}


def _hand_weighted_mean(vertex: int) -> float:
    terms = _W[vertex]
    return sum(w * iff for w, iff in terms) / sum(w for w, _ in terms)


def _hand_weight_sum(vertex: int) -> float:
    return sum(w for w, _ in _W[vertex])


def _hand_n_eff(vertex: int) -> float:
    ws = [w for w, _ in _W[vertex]]
    return sum(ws) ** 2 / sum(w * w for w in ws)


UNWEIGHTED = {0: 50.0, 1: 75.0, 2: 190.0 / 3, 3: 70.0, 4: 40.0, 5: 190.0 / 3, 6: 190.0 / 3}


def _mean_map(touch, alpha, n_verts=fixtures.WORKED_EXAMPLE_N_VERTICES):
    return dict(_compute_touch_rf(touch, n_verts, "iff", alpha).mean_pairs)


def _touch(frame_vertices, frame_signed_depths, frame_iff, gesture="stroke_proximal"):
    """Build a minimal ``TouchEvent`` from per-frame literals."""
    verts = [np.asarray(v, dtype=np.int64) for v in frame_vertices]
    depths = [np.asarray(d, dtype=np.float64) for d in frame_signed_depths]
    pts = [
        np.stack([v.astype(np.float64), np.zeros(len(v)), np.ones(len(v))], axis=1)
        for v in verts
    ]
    iff = np.asarray(frame_iff, dtype=np.float64)
    return TouchEvent(
        block_order_id="1",
        trial_id=1,
        single_touch_id=1,
        gesture_type=gesture,
        frame_contact_pts=pts,
        frame_vertex_indices=verts,
        frame_depths=depths,
        frame_spikes=np.zeros(len(iff), dtype=bool),
        frame_iff=iff,
    )


class TestWorkedExampleAtAlphaOne:
    """The full 7-vertex / 3-frame example from the plan's Problem Statement."""

    @pytest.fixture()
    def got(self):
        return _mean_map(fixtures.worked_example_touch(), 1.0)

    @pytest.mark.parametrize("vertex", sorted(_W))
    def test_every_vertex_matches_the_hand_computed_weighted_mean(self, vertex, got):
        assert got[vertex] == pytest.approx(_hand_weighted_mean(vertex), rel=1e-12)

    def test_the_peak_moves_from_v2_to_v3(self):
        """The single claim the whole feature exists to make.

        Unweighted, the peak sits on v2 — a flank that received its neighbour's
        best moment at full credit. Weighted, it sits on v3, the true receptive
        field centre.
        """
        touch = fixtures.worked_example_touch()
        unweighted = _mean_map(touch, 0.0)
        weighted = _mean_map(touch, 1.0)
        assert max(unweighted, key=unweighted.get) == 1
        assert max(weighted, key=weighted.get) == 2

    @pytest.mark.parametrize("vertex", [0, 4])
    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_single_frame_vertices_are_invariant_at_every_alpha(self, vertex, alpha):
        """A vertex touched in exactly one frame: the weight cancels exactly.

        ``w*x / w == x`` for any non-zero ``w``, so v1 stays at 50.0 and v5 at
        40.0 however the dial is turned. Anything else means the denominator is
        not the weight sum.
        """
        got = _mean_map(fixtures.worked_example_touch(), alpha)
        assert got[vertex] == pytest.approx(UNWEIGHTED[vertex], rel=1e-12)

    def test_flat_weight_vertices_barely_move_while_the_rf_centre_moves(self):
        """v6 (always shallow) and v7 (always deep) both stay put; v3 does not.

        This is the assertion that proves the estimator is a genuine weighted
        mean and not a dimmer switch. A roughly-constant weight cancels between
        numerator and denominator, so *absolute depth level is irrelevant* —
        only frame-to-frame variation in a vertex's own depth does anything. An
        estimator that divided by the frame count instead would drag v6 down to
        about 8 Hz and inflate v7, moving the peak to v7 rather than v3.
        """
        touch = fixtures.worked_example_touch()
        unweighted = _mean_map(touch, 0.0)
        weighted = _mean_map(touch, 1.0)

        v6_shift = abs(weighted[5] - unweighted[5])
        v7_shift = abs(weighted[6] - unweighted[6])
        v3_shift = abs(weighted[2] - unweighted[2])

        assert v6_shift < 2.0, "v6 (flat weights 0.13/0.13/0.10) must not move much"
        assert v7_shift < 0.2, "v7 (flat weights 0.92/0.93/0.95) must barely move"
        assert v3_shift > 10.0, "v3 (varying weights 0.42/1.00/0.50) must move"
        assert v3_shift > 5.0 * v6_shift

    def test_flat_weight_vertices_match_their_hand_computed_values(self):
        """Pinned exactly, so "barely moves" cannot drift into "does nothing"."""
        got = _mean_map(fixtures.worked_example_touch(), 1.0)
        assert got[5] == pytest.approx(23.5 / 0.36, rel=1e-12)   # v6, 65.28 Hz
        assert got[6] == pytest.approx(177.0 / 2.80, rel=1e-12)  # v7, 63.21 Hz
        assert got[2] == pytest.approx(141.0 / 1.92, rel=1e-12)  # v3, 73.44 Hz


class TestAlphaZeroReproducesTheUnweightedMap:
    @pytest.mark.parametrize("neuron_mode", ["iff", "spike"])
    def test_byte_identical_to_the_unweighted_baseline(self, neuron_mode):
        """Same code path, weights all exactly 1.0 — not merely close."""
        touch = fixtures.worked_example_touch()
        maps = _compute_touch_rf(
            touch, fixtures.WORKED_EXAMPLE_N_VERTICES, neuron_mode, 0.0
        )
        expected_values = (
            touch.frame_iff if neuron_mode == "iff"
            else touch.frame_spikes.astype(np.float64)
        )
        n = fixtures.WORKED_EXAMPLE_N_VERTICES
        val_sum = np.zeros(n)
        count = np.zeros(n)
        for verts, value in zip(touch.frame_vertex_indices, expected_values):
            np.add.at(val_sum, verts, value)
            np.add.at(count, verts, 1.0)
        want = val_sum / count
        got = dict(maps.mean_pairs)
        assert np.array_equal(
            np.array([got[i] for i in range(n)]), want
        )

    def test_hand_written_unweighted_values(self):
        got = _mean_map(fixtures.worked_example_touch(), 0.0)
        assert got[0] == 50.0
        assert got[1] == 75.0
        assert got[2] == pytest.approx(190.0 / 3)
        assert got[3] == 70.0
        assert got[4] == 40.0

    def test_weight_sum_is_the_contact_count_at_alpha_zero(self):
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 0.0,
        )
        assert dict(maps.weight_sum_pairs) == {
            0: 1.0, 1: 2.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 3.0, 6: 3.0
        }

    def test_n_eff_is_the_contact_count_at_alpha_zero(self):
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 0.0,
        )
        assert dict(maps.n_eff_pairs) == {
            0: 1.0, 1: 2.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 3.0, 6: 3.0
        }


class TestConfidenceChannel:
    def test_weight_sum_matches_the_hand_computed_sums(self):
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 1.0,
        )
        got = dict(maps.weight_sum_pairs)
        for vertex in sorted(_W):
            assert got[vertex] == pytest.approx(_hand_weight_sum(vertex), rel=1e-12)

    def test_n_eff_matches_the_hand_computed_kish_values(self):
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 1.0,
        )
        got = dict(maps.n_eff_pairs)
        for vertex in sorted(_W):
            assert got[vertex] == pytest.approx(_hand_n_eff(vertex), rel=1e-12)

    def test_flat_weights_retain_almost_all_the_evidence(self):
        """v6 keeps 2.96 of its 3 frames despite weights of 0.1.

        Being shallow costs no evidence; only being *inconsistently* shallow
        does. This is the number the plan quotes, and it is what makes the
        "confidence is a separate channel" argument concrete.
        """
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 1.0,
        )
        got = dict(maps.n_eff_pairs)
        assert got[5] == pytest.approx(2.96, abs=0.01)  # v6, always shallow
        assert got[6] == pytest.approx(3.00, abs=0.01)  # v7, always deep

    def test_one_dominant_frame_collapses_n_eff_to_one(self):
        """Three observations of v0, only one of which carries any weight.

        Weights are per *frame*, so dominance has to come from within-frame
        depth ratios: v0 is the deepest point of frame 0 and a near-grazing
        flank of frames 1 and 2, giving it weights 1.0, 1e-6, 1e-6.
        """
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1], [0, 1]],
            frame_signed_depths=[
                [-2.0, -0.1],
                [-2e-6, -2.0],
                [-2e-6, -2.0],
            ],
            frame_iff=[50.0, 100.0, 40.0],
        )
        maps = _compute_touch_rf(touch, 2, "iff", 1.0)
        assert dict(maps.n_eff_pairs)[0] == pytest.approx(1.0, abs=1e-4)
        # ...and the estimate is dominated by that frame, but stays in Hz.
        assert dict(maps.mean_pairs)[0] == pytest.approx(50.0, abs=1e-3)

    def test_flat_weights_give_n_eff_equal_to_the_frame_count(self):
        """v0 sits at a constant 0.3 of the frame maximum in all four frames."""
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1], [0, 1], [0, 1]],
            frame_signed_depths=[[-0.3, -1.0]] * 4,
            frame_iff=[10.0, 20.0, 30.0, 40.0],
        )
        maps = _compute_touch_rf(touch, 2, "iff", 1.0)
        assert dict(maps.n_eff_pairs)[0] == pytest.approx(4.0, rel=1e-12)
        # A flat weight cancels: the estimate is the plain mean, 25 Hz.
        assert dict(maps.mean_pairs)[0] == pytest.approx(25.0, rel=1e-12)

    def test_all_four_lists_cover_the_same_vertices_in_the_same_order(self):
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 1.0,
        )
        keys = [i for i, _ in maps.mean_pairs]
        assert [i for i, _ in maps.max_pairs] == keys
        assert [i for i, _ in maps.weight_sum_pairs] == keys
        assert [i for i, _ in maps.n_eff_pairs] == keys

    def test_the_nan_filter_keeps_all_four_lists_aligned(self):
        """A NaN IFF drops its vertices from the estimate *and* the channel."""
        maps = _compute_touch_rf(
            fixtures.nan_and_duplicate_touch(),
            fixtures.NAN_AND_DUPLICATE_N_VERTICES,
            "iff",
            1.0,
        )
        keys = [i for i, _ in maps.mean_pairs]
        assert keys  # the fixture has at least one NaN-free vertex
        assert [i for i, _ in maps.weight_sum_pairs] == keys
        assert [i for i, _ in maps.n_eff_pairs] == keys
        assert not any(np.isnan(v) for _, v in maps.mean_pairs)


class TestMaxMapIsUnweighted:
    """A weighted maximum changes its units and has no meaning."""

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_max_pairs_are_identical_at_every_alpha(self, alpha):
        touch = fixtures.worked_example_touch()
        n = fixtures.WORKED_EXAMPLE_N_VERTICES
        baseline = _compute_touch_rf(touch, n, "iff", 0.0).max_pairs
        assert _compute_touch_rf(touch, n, "iff", alpha).max_pairs == baseline

    def test_max_is_the_raw_observed_value(self):
        """v6 is always shallow; its max is still the full 100 Hz it saw."""
        maps = _compute_touch_rf(
            fixtures.worked_example_touch(), fixtures.WORKED_EXAMPLE_N_VERTICES,
            "iff", 1.0,
        )
        assert dict(maps.max_pairs)[5] == 100.0


class TestEstimatorGuards:
    def test_alpha_is_required(self):
        with pytest.raises(TypeError):
            _compute_touch_rf(
                fixtures.worked_example_touch(),
                fixtures.WORKED_EXAMPLE_N_VERTICES,
                "iff",
            )

    def test_a_contacted_vertex_with_zero_total_weight_is_excluded_and_counted(self):
        """Vertex 1 grazes in **both** frames that touch it.

        Frame 0: depths ``(-2.0, +0.4)`` -> penetrations ``(2.0, -0.4)``, clamped
        to ``(2.0, 0.0)``, ``d_max = 2.0``, weights ``(1.0, 0.0)``.
        Frame 1: depths ``(-3.0, 0.0)`` -> penetrations ``(3.0, 0.0)``,
        ``d_max = 3.0``, weights ``(1.0, 0.0)``.
        So ``sum(w)`` is ``2.0`` at v0 and exactly ``0.0`` at v1.

        ``sum(w) == 0`` is a 0/0: the vertex carries **no penetration evidence**
        at this alpha, so the weighted estimator is undefined there. The answer
        is "no estimate", not "stop" — the run originally raised here and that
        blocked real sessions. It is excluded from all four lists (never
        emitted as ``0.0`` or ``NaN``, never backfilled from the unweighted mean,
        which would mix two estimators in one picture) and it is **counted**, so
        the exclusion is visible in the artifact rather than silent.

        v0 is unaffected: ``(1.0*50 + 1.0*100) / 2.0 == 75.0``.
        """
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1]],
            frame_signed_depths=[[-2.0, 0.4], [-3.0, 0.0]],
            frame_iff=[50.0, 100.0],
        )
        maps = _compute_touch_rf(touch, 2, "iff", 1.0)
        assert [i for i, _ in maps.mean_pairs] == [0]
        assert dict(maps.mean_pairs)[0] == 75.0
        assert [i for i, _ in maps.max_pairs] == [0]
        assert [i for i, _ in maps.weight_sum_pairs] == [0]
        assert [i for i, _ in maps.n_eff_pairs] == [0]
        assert maps.n_no_estimate_zero_weight == 1
        assert maps.n_no_estimate_nan_value == 0

    def test_a_partly_grazing_vertex_still_gets_a_normal_weighted_mean(self):
        """Only an **all**-grazing vertex is excluded. One deep frame is enough.

        v1 grazes in frame 0 (weight 0) and is the deepest point of frame 1
        (weight 1.0), so ``sum(w) = 1.0`` and its mean is the plain 100 Hz of the
        one frame that pressed in — the grazing frame contributes 0 to both the
        numerator and the denominator and therefore cannot dilute it.
        v0: weights ``(1.0, 1.0/3.0)``, mean ``(50 + 100/3) / (4/3) = 62.5``.
        """
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1]],
            frame_signed_depths=[[-2.0, 0.4], [-1.0, -3.0]],
            frame_iff=[50.0, 100.0],
        )
        maps = _compute_touch_rf(touch, 2, "iff", 1.0)
        got = dict(maps.mean_pairs)
        assert sorted(got) == [0, 1]
        assert got[1] == 100.0
        assert got[0] == pytest.approx(62.5, rel=1e-12)
        assert dict(maps.weight_sum_pairs)[1] == 1.0
        assert maps.n_no_estimate_zero_weight == 0
        assert maps.n_no_estimate_nan_value == 0

    def test_the_same_vertex_is_fine_at_alpha_zero(self):
        """A grazing vertex is only weightless above 0 — and never at 0.

        This is the invariant the exclusion cannot touch: at ``alpha = 0`` every
        weight is exactly ``1.0`` (``0.0 ** 0.0 == 1.0``), so a contacted vertex
        can never reach ``sum(w) == 0`` and nothing is ever excluded for that
        reason.
        """
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1]],
            frame_signed_depths=[[-2.0, 0.4], [-3.0, 0.0]],
            frame_iff=[50.0, 100.0],
        )
        got = dict(_compute_touch_rf(touch, 2, "iff", 0.0).mean_pairs)
        assert got[1] == 75.0

    def test_an_untouched_vertex_is_not_an_error(self):
        """Never contacted and zero weight are different facts.

        Both leave ``weight_sum == 0``, which is why the contacted set is taken
        from the vertex lists rather than from the weight array.
        """
        touch = _touch(
            frame_vertices=[[0], [0]],
            frame_signed_depths=[[-2.0], [-3.0]],
            frame_iff=[50.0, 100.0],
        )
        maps = _compute_touch_rf(touch, 12, "iff", 1.0)
        assert [i for i, _ in maps.mean_pairs] == [0]

    def test_an_entirely_grazing_frame_raises_with_touch_and_frame_context(self):
        touch = _touch(
            frame_vertices=[[0, 1], [0, 1]],
            frame_signed_depths=[[-2.0, -1.0], [0.5, 0.25]],
            frame_iff=[50.0, 100.0],
        )
        with pytest.raises(ValueError) as excinfo:
            _compute_touch_rf(touch, 2, "iff", 1.0)
        message = str(excinfo.value)
        assert "frame 1" in message
        assert "single_touch=1" in message
        assert "grazing" in message

    def test_a_nan_depth_raises_with_frame_context(self):
        touch = _touch(
            frame_vertices=[[0, 1]],
            frame_signed_depths=[[-2.0, np.nan]],
            frame_iff=[50.0],
        )
        with pytest.raises(ValueError, match="non-finite"):
            _compute_touch_rf(touch, 2, "iff", 1.0)

    def test_a_depth_array_that_does_not_match_its_vertex_array_raises(self):
        touch = _touch(
            frame_vertices=[[0, 1, 2]],
            frame_signed_depths=[[-2.0, -1.0]],
            frame_iff=[50.0],
        )
        with pytest.raises(ValueError, match="contact points but"):
            _compute_touch_rf(touch, 3, "iff", 1.0)

    def test_a_touch_with_no_contact_points_returns_four_empty_lists(self):
        """No contact points means nothing to exclude, so both counts are 0.

        Zero excluded is the honest answer here: the touch contacted no vertex,
        so no vertex failed to get an estimate. It is *not* the same fact as
        "every contacted vertex was grazing", which the counts report as > 0.
        """
        touch = _touch(
            frame_vertices=[[], []],
            frame_signed_depths=[[], []],
            frame_iff=[50.0, 100.0],
        )
        maps = _compute_touch_rf(touch, 4, "iff", 1.0)
        assert maps == ([], [], [], [], 0, 0)


class TestUniformDepthReproducesTodaysMap:
    """The falsification path, in miniature.

    If depth is roughly uniform across each contact patch, this feature produces
    today's map at every alpha. That is not a bug — it is the premise being
    false on that data, and it is why Phase 6.3 measures per-vertex weight
    variation before any result is trusted.
    """

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_identical_depths_give_the_unweighted_answer(self, alpha):
        touch = _touch(
            frame_vertices=[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            frame_signed_depths=[[-1.7] * 3, [-1.7] * 3, [-1.7] * 3],
            frame_iff=[50.0, 100.0, 40.0],
        )
        got = dict(_compute_touch_rf(touch, 5, "iff", alpha).mean_pairs)
        want = dict(_compute_touch_rf(touch, 5, "iff", 0.0).mean_pairs)
        assert got == want


class TestADuplicatedVertexIsCreditedAtTheSummedWeight:
    """Two contact points of one frame landing on one vertex sum their weights.

    The loader used to reject a repeated ``(frame_index, vertex_id)``, on the stated
    premise that duplicates "cancel between numerator and denominator only while
    every depth weight is 1". That premise is false, and this class is what replaced
    the guard. A frame's IFF is one **scalar**, shared by every contact point of that
    frame, so for a vertex hit twice at weights ``w1``, ``w2``::

        numerator   += IFF_f * w1 + IFF_f * w2 == IFF_f * (w1 + w2)
        denominator += w1 + w2

    ``IFF_f`` factors out for *any* weights. The load-bearing claim below is the
    consequence: the per-vertex weighted mean is **identical** — bit for bit, not to
    a tolerance — to the mean obtained by replacing the two rows with a single row of
    weight ``w1 + w2``.

    The fixture, and every expected number, computed by hand at ``alpha = 1``:

    ==========  =======  ===================  ====================  =========
    frame       IFF Hz   penetration mm       weights (d/d_max)     duplicate
    ==========  =======  ===================  ====================  =========
    0           40       v2 3, v5 4, v2 3     0.75, 1.00, 0.75      v2 twice
    1           120      v2 1, v5 2           0.50, 1.00            --
    ==========  =======  ===================  ====================  =========

    * v2 weight in frame 0 = 0.75 + 0.75 = **1.5**, i.e. *above* 1. Intended: the
      weighting deliberately does not normalise per frame, so a frame's total weight
      already scales with how much of the patch it covers, and a vertex that caught
      two contact points genuinely had more finger on it.
    * v2 mean = (40*0.75 + 40*0.75 + 120*0.5) / (0.75 + 0.75 + 0.5)
      = (30 + 30 + 60) / 2.0 = 120 / 2.0 = **60.0 Hz**, which is exactly
      (40*1.5 + 120*0.5) / (1.5 + 0.5) — the collapsed form.
    * v5 mean = (40*1.0 + 120*1.0) / 2.0 = 160 / 2.0 = **80.0 Hz**.
    * v2 Kish n_eff = 2.0**2 / (0.75**2 + 0.75**2 + 0.5**2) = 4 / 1.375 = **32/11**
      = 2.909..., which the collapsed form does **not** reproduce (4 / 2.5 = 1.6).
      The equivalence is a statement about the *mean* only: two rows really are two
      contributions, and the evidence channel is entitled to say so.

    Every value in the fixture is a dyadic rational, so all of this arithmetic is
    exact in binary floating point and the equality assertions carry no tolerance.
    """

    ALPHA = 1.0
    N_VERTS = 6

    @pytest.fixture()
    def touch(self):
        return _touch(
            frame_vertices=[[2, 5, 2], [2, 5]],
            frame_signed_depths=[[-3.0, -4.0, -3.0], [-1.0, -2.0]],
            frame_iff=[40.0, 120.0],
        )

    def test_the_frame_weights_are_the_hand_computed_ones(self, touch):
        np.testing.assert_array_equal(
            touch_frame_weights(touch, 0, self.ALPHA), [0.75, 1.0, 0.75]
        )
        np.testing.assert_array_equal(
            touch_frame_weights(touch, 1, self.ALPHA), [0.5, 1.0]
        )

    def test_the_duplicated_vertex_carries_more_than_unit_weight_in_that_frame(
        self, touch
    ):
        weights = touch_frame_weights(touch, 0, self.ALPHA)
        # Positions 0 and 2 of frame 0 are both vertex 2.
        assert weights[0] + weights[2] == 1.5
        assert weights[0] + weights[2] > 1.0

    def test_the_weighted_means_are_the_hand_computed_ones(self, touch):
        mean = dict(_compute_touch_rf(touch, self.N_VERTS, "iff", self.ALPHA).mean_pairs)
        assert sorted(mean) == [2, 5]
        assert mean[2] == 60.0
        assert mean[5] == 80.0

    def test_the_weight_sum_counts_both_rows(self, touch):
        weight_sum = dict(
            _compute_touch_rf(touch, self.N_VERTS, "iff", self.ALPHA).weight_sum_pairs
        )
        # v2: 0.75 + 0.75 + 0.5 ; v5: 1.0 + 1.0
        assert weight_sum[2] == 2.0
        assert weight_sum[5] == 2.0

    def test_the_mean_equals_one_row_of_the_summed_weight(self, touch):
        """The load-bearing claim: collapsing the two rows changes nothing.

        The reference is built by handing the accumulator the *collapsed* batch
        explicitly — v2 once, at weight ``w1 + w2 = 1.5`` — so it is not the
        implementation compared against itself, and its answer is the hand-computed
        60.0 / 80.0 above.
        """
        got = dict(_compute_touch_rf(touch, self.N_VERTS, "iff", self.ALPHA).mean_pairs)

        collapsed = empty_accumulator(self.N_VERTS)
        accumulate_vertex_values_into(
            collapsed,
            np.array([2, 5], dtype=np.int64),
            40.0,
            np.array([1.5, 1.0], dtype=np.float64),
        )
        accumulate_vertex_values_into(
            collapsed,
            np.array([2, 5], dtype=np.int64),
            120.0,
            np.array([0.5, 1.0], dtype=np.float64),
        )
        want = {
            v: collapsed.value_sum[v] / collapsed.weight_sum[v] for v in (2, 5)
        }
        assert want[2] == 60.0
        assert want[5] == 80.0
        assert got[2] == want[2]
        assert got[5] == want[5]

    def test_n_eff_does_distinguish_the_two_rows(self, touch):
        n_eff = dict(
            _compute_touch_rf(touch, self.N_VERTS, "iff", self.ALPHA).n_eff_pairs
        )
        # 4 / (0.75**2 + 0.75**2 + 0.5**2) = 4 / 1.375 = 32/11
        assert n_eff[2] == pytest.approx(32.0 / 11.0, rel=1e-15)
        # The collapsed form would give 4 / (1.5**2 + 0.5**2) = 1.6. The mean is
        # invariant; the evidence count is not, and must not silently claim to be.
        assert n_eff[2] != pytest.approx(1.6, rel=1e-9)
        assert n_eff[5] == 2.0

    def test_the_max_map_ignores_weights_entirely(self, touch):
        max_map = dict(
            _compute_touch_rf(touch, self.N_VERTS, "iff", self.ALPHA).max_pairs
        )
        # Unweighted, so the duplicate cannot move it: both vertices saw 120 Hz.
        assert max_map[2] == 120.0
        assert max_map[5] == 120.0

    @pytest.mark.parametrize("alpha", ALL_ALPHAS)
    def test_the_scalar_iff_factors_out_at_every_alpha(self, alpha):
        """A vertex hit twice in the *only* frame that touches it reports that IFF.

        Whatever the two weights are, the mean is ``IFF * (w1 + w2) / (w1 + w2)``.
        This is the algebraic statement the removed guard denied, swept over the
        whole alpha dial rather than pinned at one point.
        """
        touch = _touch(
            frame_vertices=[[1, 3, 1]],
            frame_signed_depths=[[-0.5, -2.0, -1.25]],
            frame_iff=[37.5],
        )
        mean = dict(_compute_touch_rf(touch, 4, "iff", alpha).mean_pairs)
        assert mean[1] == pytest.approx(37.5, rel=1e-15)
        assert mean[3] == pytest.approx(37.5, rel=1e-15)


class TestSignConvention:
    """The sidecar stores signed depth; the weight function consumes penetration."""

    def test_the_negation_happens_exactly_once(self):
        signed = fixtures.WORKED_EXAMPLE_SIGNED_DEPTHS_MM[0]
        assert np.array_equal(
            penetration_from_signed_mm(signed), fixtures.WORKED_EXAMPLE_DEPTHS_MM[0]
        )

    def test_forgetting_the_negation_would_be_caught_not_absorbed(self):
        """Feeding signed depths straight in raises rather than inverting the map.

        Every entry is negative, so every point clamps to zero and ``d_max`` is
        0. That is the whole reason the weight function takes penetration and
        says so in its argument name: a silent sign error here would put the
        peak on the shallowest vertex.
        """
        signed = fixtures.WORKED_EXAMPLE_SIGNED_DEPTHS_MM[0]
        with pytest.raises(ValueError, match="grazing"):
            vertex_weights(signed, 1.0)
