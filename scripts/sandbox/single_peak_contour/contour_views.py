"""Contour-overlay subclasses of the preprocessing-explorer view widgets.

These subclass :class:`DualSurfaceView` and :class:`CrossSectionView` so the shared
explorer widgets stay untouched (no regression risk).  They add the ability to
overlay one or more foot contours (plus the seed peak and any snapped-back
vertices) on the 3D surfaces and to mark where each contour crosses the current 2D
cut.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, map_coordinates

from ..contour_explorer.cross_section_view import CrossSectionView
from ..contour_explorer.dual_surface_view import DualSurfaceView

# Human-readable legend labels keyed by the overlay entry name.
_LEGEND_LABELS = {
    "radial": "radial (snapped to data edge)",
    "radial_unsnapped": "radial (un-snapped λmax foot)",
    "radial_enveloped": "radial (2D footprint envelope)",
    "region": "region (concave basin)",
}

# Marker sizes in screen pixels. Markers render as point sprites
# (``render_points_as_spheres``), which are screen-space, so these stay round at
# any zoom/tilt and need no Z-scale correction.
_CLIP_MARKER_PX = 11.0
_SEED_MARKER_PX = 17.0


def _legend_label(name: str) -> str:
    return _LEGEND_LABELS.get(name, name)


# A contour overlay entry is a dict:
#   {"name": str, "color": str, "rc": (N, 2) ndarray,
#    "marker_rc": (M, 2) ndarray | None}   # e.g. snapped-back vertices


def _rc_to_uv(rc: np.ndarray, grid_u: np.ndarray, grid_v: np.ndarray) -> np.ndarray:
    """Map (row, col) pixel coords to (U, V) world coords (linear, matches grids)."""
    n_rows, n_cols = grid_u.shape
    u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
    v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
    u = u_min + rc[:, 0] * (u_max - u_min) / (n_rows - 1)
    v = v_min + rc[:, 1] * (v_max - v_min) / (n_cols - 1)
    return np.column_stack([u, v])


def _sample_z(field: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """Bilinearly sample *field* at (row, col) points.

    NaN cells are filled by nearest-valid extrapolation so overlay vertices that
    fall in the masked halo (e.g. an un-snapped foot reaching past the raw
    footprint) follow the nearest surface height instead of plunging to the floor.
    """
    nan = np.isnan(field)
    if nan.any() and not nan.all():
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        filled = field[tuple(idx)]
    else:
        filled = np.where(nan, 0.0, field)
    return map_coordinates(filled, [rc[:, 0], rc[:, 1]], order=1, mode="nearest")


class ContourSurfaceView(DualSurfaceView):
    """Dual 3D surfaces that also overlay foot contours, the seed, and clip flags."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._contours: list[dict] = []
        self._seed_rc: tuple[int, int] | None = None

    # ------------------------------------------------------------ camera
    def view_top_xy(self):
        """Look straight down the +Z axis on both surfaces (heatmap-style top view)."""
        if not self._initialized:
            return
        for plotter in (self._plotter_raw, self._plotter_proc):
            plotter.view_xy()
            plotter.render()

    # ------------------------------------------------------------ overlay state
    def set_contours(self, contours: list[dict], seed_rc: tuple[int, int] | None):
        """Store overlays and redraw both surfaces so they appear immediately."""
        self._contours = list(contours)
        self._seed_rc = seed_rc
        if self._initialized:
            self.render_raw()
            self.render_processed()

    # ------------------------------------------------------------ rendering
    def render_raw(self):
        super().render_raw()
        self._draw_overlays(self._plotter_raw, self._raw)

    def render_processed(self):
        super().render_processed()
        self._draw_overlays(self._plotter_proc, self._processed)

    def _draw_overlays(self, plotter, field):
        if self._grid_u is None or field is None:
            return

        legend_entries: list[list] = []
        for entry in self._contours:
            rc = np.asarray(entry["rc"], dtype=float)
            if rc.shape[0] < 2:
                continue
            uv = _rc_to_uv(rc, self._grid_u, self._grid_v)
            z = _sample_z(field, rc)
            # Close the loop. Z is left at true value so the post-render axis
            # scaling lays the polyline onto the surface (same as the mesh).
            pts = np.column_stack([uv[:, 0], uv[:, 1], z])
            pts = np.vstack([pts, pts[0]])
            line = pv.lines_from_points(pts)
            plotter.add_mesh(
                line, color=entry["color"], line_width=entry.get("line_width", 4),
                name=f"contour_{entry['name']}", render=False,
            )
            legend_entries.append([_legend_label(entry["name"]), entry["color"]])

            marker_rc = entry.get("marker_rc")
            if marker_rc is not None and len(marker_rc) > 0:
                self._add_markers(
                    plotter, field, np.asarray(marker_rc, dtype=float),
                    _CLIP_MARKER_PX, entry["color"], f"clip_{entry['name']}",
                )

        if self._seed_rc is not None:
            self._add_markers(
                plotter, field, np.asarray([self._seed_rc], dtype=float),
                _SEED_MARKER_PX, "lime", "seed_marker",
            )
            legend_entries.append(["peak (seed)", "lime"])

        if legend_entries:
            plotter.add_legend(
                legend_entries, bcolor=(0.12, 0.12, 0.12), border=False,
                size=(0.32, 0.14), loc="upper right",
            )
        plotter.render()

    def _add_markers(self, plotter, field, rc, point_size, color, name):
        """Place screen-space round markers at *rc* as a single point cloud.

        Rendered as one ``pv.PolyData`` drawn with ``render_points_as_spheres`` so
        every marker costs a single vertex instead of a full sphere mesh — the
        per-vertex spheres are what made rotation crawl. Point sprites are
        screen-space, so they stay round under the Z-axis scaling without any
        pre-division. The marker Z stays at its true value so the renderer's
        post-draw axis scaling lays each point onto the surface (same as the mesh).
        """
        uv = _rc_to_uv(rc, self._grid_u, self._grid_v)
        z = _sample_z(field, rc)
        finite = np.isfinite(z)
        if not np.any(finite):
            plotter.remove_actor(name, render=False)
            return
        pts = np.column_stack([uv[finite, 0], uv[finite, 1], z[finite]])
        plotter.add_mesh(
            pv.PolyData(pts), color=color, name=name, render=False,
            render_points_as_spheres=True, point_size=point_size,
        )


class ContourCrossSectionView(CrossSectionView):
    """Cross-sections that also mark where each contour crosses the current cut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._contours: list[dict] = []
        self._seed_rc: tuple[int, int] | None = None

    def set_contours(self, contours: list[dict], seed_rc: tuple[int, int] | None = None):
        self._contours = list(contours)
        self._seed_rc = seed_rc
        if self._configured:
            self.refresh()

    def refresh(self):
        super().refresh()
        if self._grid_u is None or (not self._contours and self._seed_rc is None):
            return
        j = int(self._slider_u.value())  # V column index → top profile (along U)
        i = int(self._slider_v.value())  # U row index    → bottom profile (along V)
        n_rows, n_cols = self._raw.shape
        for entry in self._contours:
            rc = np.asarray(entry["rc"], dtype=float)
            if rc.shape[0] < 2:
                continue
            color = entry["color"]
            # Top plot is along U at fixed column j: mark U where the contour
            # crosses col == j.  Bottom is along V at fixed row i: mark V at row==i.
            for u_row in self._crossings(rc, axis=1, value=j):
                u_val = self._axis_value(u_row, self._grid_u[:, 0])
                self._ax_u.axvline(u_val, color=color, ls="--", lw=1.5, alpha=0.9)
                ri = int(np.clip(round(u_row), 0, n_rows - 1))
                yval = self._raw[ri, j]
                if np.isfinite(yval):
                    self._ax_u.plot([u_val], [float(yval)], marker="o",
                                    color=color, ms=6, zorder=5)
            for v_col in self._crossings(rc, axis=0, value=i):
                v_val = self._axis_value(v_col, self._grid_v[0, :])
                self._ax_v.axvline(v_val, color=color, ls="--", lw=1.5, alpha=0.9)
                ci = int(np.clip(round(v_col), 0, n_cols - 1))
                yval = self._raw[i, ci]
                if np.isfinite(yval):
                    self._ax_v.plot([v_val], [float(yval)], marker="o",
                                    color=color, ms=6, zorder=5)

        handles = [
            Line2D([0], [0], color=e["color"], lw=1.5, label=_legend_label(e["name"]))
            for e in self._contours
            if np.asarray(e["rc"]).shape[0] >= 2
        ]

        # The seed lies on a profile only when that profile's fixed index passes
        # through it (top: column j == seed col; bottom: row i == seed row).  With
        # the cut auto-centred on the peak both hold, so the green dot appears on
        # both; if the user moves the cut off the peak it drops, like the crossings.
        seed_drawn = False
        if self._seed_rc is not None:
            pr, pc = int(self._seed_rc[0]), int(self._seed_rc[1])
            peak_val = self._raw[pr, pc]
            if np.isfinite(peak_val):
                if j == pc:
                    u_val = self._axis_value(pr, self._grid_u[:, 0])
                    self._ax_u.plot([u_val], [float(peak_val)], marker="o",
                                    color="lime", ms=8, mec="black", mew=0.5, zorder=6)
                    seed_drawn = True
                if i == pr:
                    v_val = self._axis_value(pc, self._grid_v[0, :])
                    self._ax_v.plot([v_val], [float(peak_val)], marker="o",
                                    color="lime", ms=8, mec="black", mew=0.5, zorder=6)
                    seed_drawn = True
        if seed_drawn:
            handles.append(Line2D([0], [0], color="lime", marker="o", ls="none",
                                  label="peak (seed)"))

        if handles:
            self._ax_u.legend(handles=handles, loc="upper right", fontsize=7,
                              framealpha=0.7, title="contour crossings")
        self._canvas.draw_idle()

    @staticmethod
    def _crossings(rc: np.ndarray, axis: int, value: float) -> list[float]:
        """Return the *other*-axis coords where the closed polygon crosses ``axis==value``."""
        other = 1 - axis
        out: list[float] = []
        pts = np.vstack([rc, rc[0]])  # close the loop
        a = pts[:, axis]
        b = pts[:, other]
        for k in range(len(pts) - 1):
            a0, a1 = a[k], a[k + 1]
            if (a0 - value) * (a1 - value) > 0 or a0 == a1:
                continue  # no straddle (or degenerate edge)
            t = (value - a0) / (a1 - a0)
            out.append(float(b[k] + t * (b[k + 1] - b[k])))
        return out

    @staticmethod
    def _axis_value(index: float, axis_vals: np.ndarray) -> float:
        n = len(axis_vals)
        lo, hi = float(axis_vals[0]), float(axis_vals[-1])
        return lo + index * (hi - lo) / (n - 1)
