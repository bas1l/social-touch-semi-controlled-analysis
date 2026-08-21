"""The shared estimator: where a weighted mean exists, and where it does not.

``vertex_estimate`` is the third and last of the shared steps between the saved
maps and the on-screen heatmap (after ``touch_frame_weights`` and
``vertex_accumulator``).  It exists because the two sides disagreed: the pipeline
*raised* on a contacted vertex whose weights all came out zero, while the
playback viewer quietly painted the same vertex grey — so the viewer rendered
sessions the pipeline refused to process.

These tests pin the policy itself.  The end-to-end agreement between the two
callers is pinned separately, in
``tests/test_rf_depth_weighting_parity.py::TestPlaybackViewerAgreesWithTheSavedMap``.
"""

import numpy as np
import pytest

from analysis.receptive_field_mapping.data.vertex_estimate import (
    NoEstimateCounts,
    count_no_estimate,
    has_estimate,
    weighted_mean_or_nan,
)


class TestWeightedMeanOrNan:
    def test_the_quotient_is_the_plain_division_where_a_weight_exists(self):
        """No guard may perturb a vertex that has an estimate.

        The alpha = 0 byte-identity claim rests on this: the substituted divisor
        applies only where the quotient would be 0/0, so every vertex with any
        weight gets exactly the IEEE result of ``value_sum / weight_sum``.
        """
        values = np.array([3.0, 100.0, -7.5, 1e-18])
        weights = np.array([7.0, 3.0, 0.125, 1e18])
        got = weighted_mean_or_nan(values, weights)
        want = values / weights
        assert got.tolist() == want.tolist()

    def test_zero_weight_is_nan_not_zero(self):
        """0/0 is *no estimate*, never a measured zero.

        Emitting ``0.0`` would paint "no penetration evidence" as "no response",
        which is a cold-but-real value at the edge of a receptive field — the
        single most misleading number this module can produce.
        """
        got = weighted_mean_or_nan(np.array([0.0, 5.0]), np.array([0.0, 2.0]))
        assert np.isnan(got[0])
        assert got[1] == 2.5

    def test_a_nan_numerator_is_nan(self):
        """No neural measurement is also *no estimate*, by the same exit."""
        got = weighted_mean_or_nan(np.array([np.nan, 5.0]), np.array([2.0, 2.0]))
        assert np.isnan(got[0])
        assert got[1] == 2.5

    def test_there_is_no_fallback_to_the_unweighted_mean(self):
        """A zero-weight entry does not borrow the mean of its neighbours.

        Falling back would put two different estimators in one map: the
        neighbouring vertices would be weighted means and this one would not, and
        nothing on disk would say which was which.
        """
        # ``value_sum`` at the zero-weight entry holds the *unweighted* total of
        # the frames that touched it (each contributed ``0 * iff``, so it is 0);
        # a fallback would divide by the frame count instead and emit a number.
        got = weighted_mean_or_nan(np.array([0.0, 40.0]), np.array([0.0, 2.0]))
        assert np.isnan(got[0])
        assert not np.isfinite(got[0])
        # The neighbour is untouched by the guard and stays a weighted mean.
        assert got[1] == 20.0

    def test_a_shape_mismatch_raises(self):
        """Sliced differently means one vertex divided by another's weight."""
        with pytest.raises(ValueError, match="same vertices in the same order"):
            weighted_mean_or_nan(np.zeros(3), np.zeros(4))

    def test_a_two_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            weighted_mean_or_nan(np.zeros((2, 2)), np.zeros((2, 2)))


class TestHasEstimate:
    def test_it_is_the_complement_of_the_nan_mask(self):
        values = np.array([0.0, np.nan, 6.0])
        weights = np.array([0.0, 2.0, 3.0])
        assert has_estimate(values, weights).tolist() == [False, False, True]


class TestCountNoEstimate:
    def test_the_two_causes_are_counted_apart(self):
        """"The finger only grazed" and "the electrode dropped out" are different.

        Both are excluded from the map, but collapsing them into one number would
        make a data-quality problem indistinguishable from a geometry one.
        """
        values = np.array([0.0, np.nan, 6.0, 0.0, np.nan])
        weights = np.array([0.0, 2.0, 3.0, 0.0, 5.0])
        counts = count_no_estimate(values, weights)
        assert counts == NoEstimateCounts(zero_weight=2, nan_value=2)
        assert counts.total == 4

    def test_a_zero_weight_entry_is_counted_once_not_twice(self):
        """It is NaN for *both* reasons; it belongs to the zero-weight bucket.

        Otherwise ``total`` would exceed the number of excluded vertices and the
        summary JSON would over-report.
        """
        counts = count_no_estimate(np.array([np.nan]), np.array([0.0]))
        assert counts == NoEstimateCounts(zero_weight=1, nan_value=0)
        assert counts.total == 1

    def test_all_weights_one_gives_no_zero_weight_exclusions(self):
        """The alpha = 0 invariant, at the level of this module.

        At ``depth_weight_alpha = 0`` every weight is exactly ``1.0``, so no
        contacted vertex can reach ``sum(w) == 0`` and this count is necessarily
        0 — which is why the new exclusion policy cannot move an alpha = 0 map.
        """
        counts = count_no_estimate(np.arange(6.0), np.ones(6))
        assert counts.zero_weight == 0

    def test_an_empty_selection_counts_nothing(self):
        counts = count_no_estimate(np.empty(0), np.empty(0))
        assert counts == NoEstimateCounts(zero_weight=0, nan_value=0)
