"""Right panel: the production ``spatial_extract_boundaries`` heatmap figure, live.

Reproduces the right-hand ("Interpolated heatmap") panel of
:func:`render_population_rf_map` — grey forearm mesh background + the interpolated
grid drawn with ``pcolormesh`` + the foot contours as closed polylines + the peak
seed marker — embedded in a Qt widget via ``FigureCanvasQTAgg`` so it updates in
lockstep with the 3D surfaces and the 2D cross-sections.

Unlike the production figure (a single selected boundary in red), this panel draws
*all* active sandbox contours in their own colours so the delineations being tuned
can be compared directly against the figure that ships.
"""

from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    _draw_forearm_mesh_background,
)

from .contour_views import _rc_to_uv


class ForearmHeatmapView(QWidget):
    """Live reproduction of the production interpolated-heatmap + contour figure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid_u = None
        self._grid_v = None
        self._grid_z = None
        self._forearm_uv = None
        self._forearm_faces = None
        self._contours: list[dict] = []
        self._seed_rc: tuple[int, int] | None = None
        self._cbar = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(5, 6), facecolor="black")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        # Pan/zoom toolbar so the user can navigate the heatmap (pan, box-zoom,
        # home/reset). Without it the embedded canvas is non-interactive.
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas, stretch=1)

    # ------------------------------------------------------------ data hand-off
    def set_data(self, grid_u, grid_v, grid_z, forearm_uv, forearm_faces):
        """Store the grids + forearm mesh. Does NOT redraw (the window drives that)."""
        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._grid_z = np.asarray(grid_z, dtype=float)
        self._forearm_uv = np.asarray(forearm_uv, dtype=float)
        self._forearm_faces = np.asarray(forearm_faces)

    def set_contours(self, contours: list[dict], seed_rc: tuple[int, int] | None):
        """Store overlays and redraw the panel so they appear immediately."""
        self._contours = list(contours)
        self._seed_rc = seed_rc
        self.refresh()

    # ------------------------------------------------------------ rendering
    def refresh(self):
        if self._grid_z is None or self._forearm_uv is None:
            return

        # Rebuild the figure from scratch each refresh. Clearing the whole figure
        # (rather than ``ax.clear()`` + ``colorbar.remove()``) avoids the stale
        # subplotspec / progressive axes-shrink problems that a re-created colorbar
        # otherwise causes across session switches and repeated recomputes.
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax

        # --- forearm mesh background (grey: no skin texture in the sandbox) ---
        _draw_forearm_mesh_background(
            ax, self._forearm_uv, self._forearm_faces, vertex_colors=None
        )

        # --- interpolated heatmap (same recipe as render_population_rf_map) ---
        display_z = np.where(self._grid_z > 0, self._grid_z, np.nan)
        finite = display_z[np.isfinite(display_z)]
        if finite.size == 0:
            raise ValueError(
                "No positive finite heatmap values to display — cannot set colour scale."
            )
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))
        im = ax.pcolormesh(
            self._grid_u, self._grid_v, display_z,
            cmap="inferno", norm=norm, shading="auto",
        )
        self._cbar = self._fig.colorbar(im, ax=ax, label="Mean IFF / spike", shrink=0.8)
        self._cbar.ax.yaxis.set_tick_params(color="white")
        self._cbar.ax.yaxis.label.set_color("white")
        for lbl in self._cbar.ax.yaxis.get_ticklabels():
            lbl.set_color("white")

        # --- contour overlays (all active, each in its own colour) ---
        for entry in self._contours:
            rc = np.asarray(entry["rc"], dtype=float)
            if rc.shape[0] < 2:
                continue
            uv = _rc_to_uv(rc, self._grid_u, self._grid_v)
            closed = np.vstack([uv, uv[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=entry["color"], lw=1.5, zorder=6)

        # --- peak seed marker (lime, matching the other views) ---
        if self._seed_rc is not None:
            seed_uv = _rc_to_uv(
                np.asarray([self._seed_rc], dtype=float), self._grid_u, self._grid_v
            )
            ax.plot(seed_uv[0, 0], seed_uv[0, 1], marker="o", color="lime",
                    ms=8, mec="black", mew=0.5, zorder=7)

        # --- black-on-white axis styling, matching the production figure ---
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_aspect("equal")
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.set_title("Interpolated heatmap", color="white", fontsize=10)

        # The axes were re-created, so the toolbar's pan/zoom history points at a
        # dead axes — reset it so "home"/back/forward track the current view.
        self._toolbar.update()
        self._canvas.draw_idle()
