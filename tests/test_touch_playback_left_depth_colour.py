"""The Touch Playback Explorer's LEFT panel is coloured by penetration depth.

The left panel used to paint every contacted vertex of the current frame one flat
red: a "where is contact" mask that threw away the shape of the indentation the
data already carried. It now colours those same points by
``TouchEvent.frame_depths`` in millimetres, on a scale fixed for the whole of the
selected touch.

Four claims are pinned here, and each of them is a way the panel could look
plausible while being wrong:

* **The scale is per touch, not per frame.** A per-frame scale would pin every
  frame's own deepest point to full brightness, so a stroke that pressed twice as
  hard halfway through would look identical throughout — destroying exactly the
  comparison the panel exists to make.
* **The sign is right.** Storage is ``signed_depth_mm`` with *negative =
  penetrating*, and ``penetration_from_signed_mm`` is the single documented
  negation point. A second negation elsewhere would render the indentation
  inside-out and still look like a smooth patch.
* **Grazing is shown, not clipped.** A touch that only grazed carries small
  negative penetration. Clamping the displayed range at 0 would quietly assert
  that the touch pressed in everywhere it registered.
* **It does not depend on ``depth_weight_alpha``.** The left panel shows the raw
  *input* to the weighting; the right panel shows its output. The two must be
  able to disagree, and the left one must not move when alpha does. That is the
  cleanest single statement of what this panel is, so it gets its own class.

Everything here is the numeric core: module-level, Qt-free, no plotter, no window.
The rendering itself (the colourbar actor, its placement, the ``cool`` colormap on
screen) is not exercised — VTK/Qt cannot be opened in this environment.

Every fixture is synthetic. No experimental data file is read.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.contact_depth_field_io import (  # noqa: E402
    penetration_from_signed_mm,
)
from analysis.receptive_field_mapping.data.touch_playback_data import (  # noqa: E402
    TouchEvent,
)
from analysis.receptive_field_mapping.gui.touch_playback_explorer import (  # noqa: E402
    DEPTH_CMAP,
    DEPTH_SCALAR_BAR_TITLE,
    DEPTH_SCALARS_NAME,
    depth_scale_text,
    frame_penetration_mm,
    mean_heatmap_scalars,
    replay_touch_frames,
    touch_penetration_clim,
)

from rf_accumulator_fixtures import (  # noqa: E402
    WORKED_EXAMPLE_DEPTHS_MM,
    WORKED_EXAMPLE_N_VERTICES,
    nan_and_duplicate_touch,
    worked_example_touch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch_from_signed(signed_frames, *, gesture_type="stroke_proximal") -> TouchEvent:
    """A synthetic touch whose frames carry exactly *signed_frames*.

    Contact points and vertex indices are generated to match each frame's length,
    so the only thing under test is the depth channel.
    """
    frames = [np.asarray(f, dtype=np.float64) for f in signed_frames]
    vertex_indices = [np.arange(len(f), dtype=np.int64) for f in frames]
    contact_pts = [
        np.stack(
            [np.arange(len(f), dtype=np.float64), np.zeros(len(f)), np.ones(len(f))],
            axis=1,
        )
        for f in frames
    ]
    n = len(frames)
    return TouchEvent(
        block_order_id="1",
        trial_id=1,
        single_touch_id=7,
        gesture_type=gesture_type,
        frame_contact_pts=contact_pts,
        frame_vertex_indices=vertex_indices,
        frame_depths=frames,
        frame_spikes=np.zeros(n, dtype=bool),
        frame_iff=np.full(n, 25.0, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# The scale is fixed per touch, over every frame of that touch
# ---------------------------------------------------------------------------

class TestScaleIsPerTouchNotPerFrame:

    def test_clim_spans_every_frame_of_the_touch(self):
        touch = worked_example_touch()
        all_depths = np.concatenate(WORKED_EXAMPLE_DEPTHS_MM)
        assert touch_penetration_clim(touch) == (
            float(all_depths.min()),
            float(all_depths.max()),
        )

    def test_the_shallowest_frames_maximum_does_not_reach_the_top_of_the_scale(self):
        """Frame 1 tops out at 2.0 mm while the touch reaches 4.0 mm.

        This is the whole point of a per-touch scale. Under a per-frame scale
        frame 1's deepest point and frame 2's deepest point would both render at
        full brightness, and a viewer could not tell that the finger pressed
        twice as hard in frame 2.
        """
        touch = worked_example_touch()
        lo, hi = touch_penetration_clim(touch)

        frame1 = frame_penetration_mm(touch, 1)
        frame2 = frame_penetration_mm(touch, 2)

        assert float(frame1.max()) == pytest.approx(2.0)
        assert float(frame2.max()) == pytest.approx(4.0)

        # Position on the fixed scale, which is what the colormap actually sees.
        assert (float(frame1.max()) - lo) / (hi - lo) < 1.0
        assert (float(frame2.max()) - lo) / (hi - lo) == pytest.approx(1.0)

    def test_the_scale_is_a_property_of_the_touch_alone(self):
        """Two touches with different presses get different scales.

        The window recomputes it in ``_load_touch``, i.e. on every Session /
        Block / Trial / Touch change; this pins that the value it caches actually
        differs between touches rather than being a session-wide constant.
        """
        light = _touch_from_signed([[-0.5, -1.0]])
        heavy = _touch_from_signed([[-0.5, -9.0]])
        assert touch_penetration_clim(light) == (0.5, 1.0)
        assert touch_penetration_clim(heavy) == (0.5, 9.0)

    def test_empty_frames_are_skipped_without_collapsing_the_scale(self):
        """A touch whose first frame has no contact still has a scale.

        ``nan_and_duplicate_touch``'s frame 0 is empty. An empty frame is not a
        depth of zero and must not drag the minimum down to it.
        """
        touch = nan_and_duplicate_touch()
        lo, hi = touch_penetration_clim(touch)
        assert (lo, hi) == (0.125, 4.0)
        assert len(frame_penetration_mm(touch, 0)) == 0


# ---------------------------------------------------------------------------
# Sign convention — one negation, in the documented place
# ---------------------------------------------------------------------------

class TestPenetrationSign:

    def test_stored_negative_depth_becomes_positive_penetration(self):
        touch = _touch_from_signed([[-3.25, -1.5, -0.125]])
        assert frame_penetration_mm(touch, 0).tolist() == [3.25, 1.5, 0.125]

    def test_it_is_the_shared_negation_point_verbatim(self):
        """No second ``-x``: the values equal ``penetration_from_signed_mm``'s.

        If this module ever grew its own negation the two would agree today and
        drift the first time the convention was revisited, which is precisely the
        failure the single-negation-point rule exists to prevent.
        """
        touch = worked_example_touch()
        for frame_idx in range(len(touch.frame_depths)):
            np.testing.assert_array_equal(
                frame_penetration_mm(touch, frame_idx),
                penetration_from_signed_mm(touch.frame_depths[frame_idx]),
            )

    def test_deeper_press_is_a_larger_number(self):
        """Ordering, stated directly — an inverted sign would fail here loudly."""
        touch = _touch_from_signed([[-1.0, -5.0]])
        pen = frame_penetration_mm(touch, 0)
        assert pen[1] > pen[0]
        assert touch_penetration_clim(touch)[1] == pen[1]


# ---------------------------------------------------------------------------
# Grazing contact is displayed, not clipped
# ---------------------------------------------------------------------------

class TestGrazingIsShownHonestly:

    def test_a_negative_penetration_lowers_the_scale_below_zero(self):
        """Storage +0.4 mm is a grazing contact: penetration -0.4 mm."""
        touch = _touch_from_signed([[0.4, -2.0, -5.0]])
        lo, hi = touch_penetration_clim(touch)
        assert lo == pytest.approx(-0.4)
        assert hi == pytest.approx(5.0)

    def test_the_lower_bound_is_not_clamped_at_zero(self):
        touch = _touch_from_signed([[0.25, -1.0]])
        assert touch_penetration_clim(touch)[0] < 0.0

    def test_the_label_says_what_below_zero_means(self):
        assert "<0 = not pressed in" in depth_scale_text((-0.4, 5.0))

    def test_a_touch_that_pressed_in_everywhere_gets_no_such_caveat(self):
        text = depth_scale_text((0.5, 5.0))
        assert "<0" not in text
        assert "mm" in text

    def test_the_label_reports_millimetres_not_a_ratio(self):
        """The reader must be able to read indentation off the panel, not 0..1."""
        text = depth_scale_text(touch_penetration_clim(worked_example_touch()))
        assert "0.26" in text
        assert "4 mm" in text

    def test_the_colourbar_title_carries_the_unit(self):
        assert "mm" in DEPTH_SCALAR_BAR_TITLE


# ---------------------------------------------------------------------------
# Independent of depth_weight_alpha — what this panel *is*
# ---------------------------------------------------------------------------

class TestIndependentOfDepthWeightAlpha:
    """The left panel shows the raw input to the weighting, never its output.

    The comparison is deliberately made against a right panel that *does* move:
    asserting only that the left numbers are stable would pass even if the whole
    alpha machinery were inert, and would prove nothing.
    """

    def _left(self, touch):
        return (
            touch_penetration_clim(touch),
            [
                frame_penetration_mm(touch, fi)
                for fi in range(len(touch.frame_depths))
            ],
        )

    def _right(self, touch, alpha):
        iff_accum, _ = replay_touch_frames(
            touch, len(touch.frame_depths), WORKED_EXAMPLE_N_VERTICES, alpha
        )
        return mean_heatmap_scalars(iff_accum.value_sum, iff_accum.weight_sum)

    def test_the_right_panel_really_does_move_with_alpha(self):
        touch = worked_example_touch()
        assert not np.allclose(self._right(touch, 0.0), self._right(touch, 1.0))

    def test_the_left_panel_is_byte_identical_at_alpha_0_and_alpha_1(self):
        touch = worked_example_touch()

        clim_a, frames_a = self._left(touch)
        self._right(touch, 0.0)  # exercise the alpha path in between
        clim_b, frames_b = self._left(touch)
        self._right(touch, 1.0)
        clim_c, frames_c = self._left(touch)

        assert clim_a == clim_b == clim_c
        for fa, fb, fc in zip(frames_a, frames_b, frames_c):
            np.testing.assert_array_equal(fa, fb)
            np.testing.assert_array_equal(fa, fc)

    @pytest.mark.parametrize("func", [frame_penetration_mm, touch_penetration_clim])
    def test_the_depth_colour_functions_take_no_alpha(self, func):
        """Structural half of the claim: alpha cannot even be handed to them.

        An assertion about values could be satisfied by a function that took
        alpha and happened to ignore it; the next edit would then be free to use
        it. Absence from the signature is what keeps that from happening.
        """
        params = inspect.signature(func).parameters
        assert not any("alpha" in name for name in params), sorted(params)


# ---------------------------------------------------------------------------
# Fail-fast: no fallback to flat colour
# ---------------------------------------------------------------------------

class TestFailFast:

    def test_a_depth_array_shorter_than_the_vertex_array_raises(self):
        touch = _touch_from_signed([[-1.0, -2.0, -3.0]])
        touch.frame_depths[0] = np.array([-1.0, -2.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"3 contact vertices but 2 depths"):
            frame_penetration_mm(touch, 0)

    def test_the_alignment_error_names_the_touch(self):
        touch = _touch_from_signed([[-1.0, -2.0, -3.0]])
        touch.frame_depths[0] = np.array([-1.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"single_touch=7"):
            touch_penetration_clim(touch)

    def test_more_contact_points_than_depths_raises(self):
        """The panel colours the plotted points, so those two must align too."""
        touch = _touch_from_signed([[-1.0, -2.0]])
        touch.frame_contact_pts[0] = np.zeros((5, 3), dtype=np.float64)
        touch.frame_vertex_indices[0] = np.arange(2, dtype=np.int64)
        with pytest.raises(ValueError, match=r"plots 5 contact point\(s\)"):
            frame_penetration_mm(touch, 0)

    def test_a_touch_with_no_contact_in_any_frame_raises(self):
        """No depth at all is a missing channel, not a reason to paint flat."""
        touch = _touch_from_signed([[], [], []])
        with pytest.raises(ValueError, match=r"no contact point in any"):
            touch_penetration_clim(touch)

    def test_a_nan_depth_raises_rather_than_destroying_the_scale(self):
        touch = _touch_from_signed([[-1.0, np.nan, -3.0]])
        with pytest.raises(ValueError, match=r"non-finite penetration"):
            touch_penetration_clim(touch)

    def test_mismatched_per_frame_list_lengths_raise(self):
        touch = _touch_from_signed([[-1.0], [-2.0]])
        touch.frame_depths.pop()
        with pytest.raises(ValueError, match=r"must be the same length"):
            touch_penetration_clim(touch)


# ---------------------------------------------------------------------------
# The two panels must not read as one quantity
# ---------------------------------------------------------------------------

class TestColormapIsDistinctFromTheRightPanel:

    def test_it_is_not_the_right_panels_colormap(self):
        assert DEPTH_CMAP != "inferno"

    def test_it_is_a_real_colormap_with_no_hue_shared_with_inferno(self):
        """``cool`` is cyan -> magenta; inferno is black -> red -> yellow.

        Compared in RGB at both ends rather than by name, so a future swap of
        ``DEPTH_CMAP`` to something warm fails here instead of quietly making the
        two panels look like one quantity.
        """
        matplotlib = pytest.importorskip("matplotlib")
        depth = matplotlib.colormaps[DEPTH_CMAP]
        heat = matplotlib.colormaps["inferno"]
        for position in (0.0, 0.5, 1.0):
            d = np.asarray(depth(position)[:3])
            h = np.asarray(heat(position)[:3])
            assert np.linalg.norm(d - h) > 0.4, position

    def test_the_scalars_have_their_own_array_name(self):
        """Not "heatmap": a mesh handed to the wrong plotter must not colour."""
        assert DEPTH_SCALARS_NAME != "heatmap"
