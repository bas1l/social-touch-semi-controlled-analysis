"""Right panel: two 2D cross-sections through the heatmap.

Top: a profile along U (slider picks the V index); bottom: a profile along V
(slider picks the U index).  Each plot overlays the *raw* field on the left Y-axis
and the *processed* field on the right Y-axis (``twinx``).  A dashed guide-line on
each profile marks the crossing point of the two cuts — the currently selected
pixel.

This widget OWNS the (i, j) selection state: its two sliders are the authoritative
source.  Any change (slider drag or :pymeth:`set_selection` from a 3D pick) redraws
both profiles and emits :pysig:`selection_changed` so the surfaces can place their
pick markers.

NOTE: matplotlib is embedded via FigureCanvasQTAgg, which renders into a Qt widget
and does not need an interactive backend (the analysis package forces Agg at import
time — harmless here).
"""

from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class CrossSectionView(QWidget):
    """Two twinned matplotlib profiles + the index sliders that drive them."""

    selection_changed = pyqtSignal(int, int)  # (row i, col j) of the selected pixel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._configured = False
        self._grid_u = None
        self._grid_v = None
        self._raw = None
        self._processed = None

        lay = QVBoxLayout(self)

        self._fig = Figure(figsize=(4, 6), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax_u = self._fig.add_subplot(211)
        self._ax_u2 = self._ax_u.twinx()
        self._ax_v = self._fig.add_subplot(212)
        self._ax_v2 = self._ax_v.twinx()
        lay.addWidget(self._canvas, stretch=1)

        # Top section: profile along U, slider picks the V column index.
        row_u = QHBoxLayout()
        row_u.addWidget(QLabel("V idx"))
        self._slider_u = QSlider(Qt.Horizontal)
        self._slider_u.valueChanged.connect(self.refresh)
        row_u.addWidget(self._slider_u)
        lay.addLayout(row_u)

        # Bottom section: profile along V, slider picks the U row index.
        row_v = QHBoxLayout()
        row_v.addWidget(QLabel("U idx"))
        self._slider_v = QSlider(Qt.Horizontal)
        self._slider_v.valueChanged.connect(self.refresh)
        row_v.addWidget(self._slider_v)
        lay.addLayout(row_v)

    # ------------------------------------------------------------ data hand-off
    def set_data(self, grid_u, grid_v, raw, processed):
        """Store the grids + both fields. First call configures the slider ranges.

        Does NOT redraw — the window calls :pymeth:`refresh` when the surfaces are
        ready, so markers are only placed after the 3D views are initialised.
        """
        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._raw = np.asarray(raw, dtype=float)
        self._processed = np.asarray(processed, dtype=float)

        if not self._configured:
            n_rows, n_cols = self._raw.shape
            self._slider_u.blockSignals(True)
            self._slider_u.setRange(0, n_cols - 1)
            self._slider_u.setValue(n_cols // 2)
            self._slider_u.blockSignals(False)
            self._slider_v.blockSignals(True)
            self._slider_v.setRange(0, n_rows - 1)
            self._slider_v.setValue(n_rows // 2)
            self._slider_v.blockSignals(False)
            self._configured = True

    def set_selection(self, i: int, j: int):
        """Move both sliders to (row i, col j) and refresh once."""
        self._slider_v.blockSignals(True)
        self._slider_u.blockSignals(True)
        self._slider_v.setValue(int(i))  # U row index
        self._slider_u.setValue(int(j))  # V column index
        self._slider_u.blockSignals(False)
        self._slider_v.blockSignals(False)
        self.refresh()

    # ------------------------------------------------------------ rendering
    def refresh(self):
        j = int(self._slider_u.value())  # V column index
        i = int(self._slider_v.value())  # U row index

        # Coordinate value at each slice index (grid_v constant down a column,
        # grid_u constant across a row) — lets the user locate the cut on the 3D view.
        v_val = float(self._grid_v[0, j])
        u_val = float(self._grid_u[i, 0])

        # Top: profile along U at fixed V (column j). The guide-line sits at the
        # picked U coordinate (= where the bottom V-cut crosses this profile).
        self._draw_section(
            self._ax_u, self._ax_u2,
            x=self._grid_u[:, j],
            raw_line=self._raw[:, j], proc_line=self._processed[:, j],
            xlabel="U", title=f"Profile along U  (V idx = {j}, V = {v_val:.2f})",
            marker_x=u_val,
        )
        # Bottom: profile along V at fixed U (row i). Guide-line at the picked V.
        self._draw_section(
            self._ax_v, self._ax_v2,
            x=self._grid_v[i, :],
            raw_line=self._raw[i, :], proc_line=self._processed[i, :],
            xlabel="V", title=f"Profile along V  (U idx = {i}, U = {u_val:.2f})",
            marker_x=v_val,
        )
        self._canvas.draw_idle()

        # Mirror the current cut intersection onto the 3D surfaces.
        self.selection_changed.emit(i, j)

    @staticmethod
    def _draw_section(ax, ax2, x, raw_line, proc_line, xlabel, title, marker_x):
        ax.clear()
        ax2.clear()
        # clear() resets a twinned axis back to the left side; force the processed
        # axis (label + ticks) onto the right, where its values live.
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax.plot(x, raw_line, color="tab:blue", lw=1.5)
        ax2.plot(x, proc_line, color="tab:red", lw=1.5)
        ax.axvline(marker_x, color="0.6", ls="--", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("raw", color="tab:blue")
        ax2.set_ylabel("processed", color="tab:red")
        ax.tick_params(axis="y", colors="tab:blue")
        ax2.tick_params(axis="y", colors="tab:red")
        ax.set_title(title, fontsize=9)
