"""Per-(session, gesture) RF-contour parameter tuning GUI.

Interactive PyQt5/PyVista window for dialling in the five radial foot-of-mountain
contour-detection parameters (``min_overlap_pct``, ``median_filter_size``,
``radial_gauss_sigma``, ``radial_hess_sigma``, ``radial_envelope_smooth_sigma``)
independently for each (session, gesture) combination, then persisting the tuned
values as one JSON per combination.

The window fuses two proven templates:

  * the *persistence/lifecycle* model of
    :class:`~analysis.receptive_field_mapping.gui.slim_uv_config_viewer.SlimUvConfigViewer`
    — combo navigation, load-on-select, Validate persistence, green-highlighting of
    completed items, and the deferred VTK/PyVista init that avoids a Windows init
    crash; and
  * the *interaction* model of the sandbox single-peak contour explorer — the live
    raw -> smoothed -> λmax -> contour chain overlaid on a 3D surface and on 2D
    heatmaps.

The math is *not* re-implemented here: the grid is rebuilt with the Phase-1
production helper (``build_session_grid``) and the stages are computed with the
Phase-2 single-source-of-truth ``compute_radial_foot_stages`` so the preview and
the ``spatial_extract_boundaries`` pipeline produce identical contours.

Fail-fast with explicit graceful degradation (never a silent blank view): an
all-NaN / empty heatmap or an invalid parameter set is the "no data" case and
still raises loudly, surfaced via a modal error dialog. But when the field IS
computable and only the *contour* cannot be traced (border peak, or radial-
extraction / envelope failure), the raw / smoothed / λmax panels, the cross-
section, and the 3D surface are still rendered and a non-fatal ``QMessageBox``
pop-up states why the contour was omitted — only the contour polyline, peak
marker, and cross-section contour-crossings are skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from matplotlib import cm as mpl_cm
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, map_coordinates
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    ContourParamToggles,
    GestureContourParams,
)
from analysis.receptive_field_mapping.data.rf_contour_params_io import (
    build_session_grid,
    contour_params_path,
    defaults_from_boundary_params,
    load_contour_params,
    load_session_arrays,
    save_contour_params,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_rgba,
)
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    compute_radial_foot_stages,
)

logger = logging.getLogger(__name__)

_GREEN_BG = QColor("#90EE90")
_CONTOUR_COLOR = "deepskyblue"
_PEAK_COLOR = "lime"

# Inline status LED colours (see ``_set_led``): green = contour drawn, red = no
# contour, grey = no recompute has run yet for the current (session, gesture).
_LED_GREEN = "#2ecc71"
_LED_RED = "#e74c3c"
_LED_GREY = "#888888"

# The five tunable parameters mapped to their widget-panel labels, in canonical
# order (matches ``GestureContourParams`` field order).
_PARAM_LABELS: tuple[tuple[str, str], ...] = (
    ("min_overlap_pct", "min overlap %"),
    ("median_filter_size", "median filter size"),
    ("radial_gauss_sigma", "radial gauss σ"),
    ("radial_hess_sigma", "radial hess σ"),
    ("radial_envelope_smooth_sigma", "envelope smooth σ"),
)

_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ("raw", "Raw"),
    ("smoothed", "Smoothed"),
    ("lmax", "λmax"),
)

# The four toggleable parameters (each gets an enable/disable checkbox). The
# fifth tunable, ``radial_hess_sigma``, is deliberately excluded — it is the
# core detector scale and always applies (see plan
# ``rf-contour-tuning-param-toggles-and-help.md``).
_TOGGLEABLE_PARAM_KEYS: tuple[str, ...] = (
    "min_overlap_pct",
    "median_filter_size",
    "radial_gauss_sigma",
    "radial_envelope_smooth_sigma",
)

# One-line tooltip per parameter spinbox (accurate to the actual math — see the
# "?" help dialog, ``_HELP_TEXT``, for the long-form version).
_PARAM_TOOLTIPS: dict[str, str] = {
    "min_overlap_pct": (
        "Minimum % of overlapping touches a forearm vertex must accumulate to "
        "stay in the response field."
    ),
    "median_filter_size": (
        "Side length (odd) of the median window that denoises the interpolated "
        "grid before contour detection."
    ),
    "radial_gauss_sigma": (
        "Sigma of the Gaussian pre-smoothing applied before the Hessian "
        "curvature step."
    ),
    "radial_hess_sigma": (
        "Sigma of the Hessian kernel used to find the λmax curvature ring "
        "(core detector; always applied, no toggle)."
    ),
    "radial_envelope_smooth_sigma": (
        "Sigma of the light smoothing applied to the final contour to remove "
        "the marching-squares staircase."
    ),
}

# One-line tooltip per toggle checkbox — what unticking it does.
_TOGGLE_TOOLTIPS: dict[str, str] = {
    "min_overlap_pct": "Unchecked: no threshold — every contacted vertex is kept.",
    "median_filter_size": "Unchecked: no median filtering (raw interpolated grid).",
    "radial_gauss_sigma": (
        "Unchecked: no pre-smoothing (Hessian runs on the raw normalised field)."
    ),
    "radial_envelope_smooth_sigma": "Unchecked: no contour smoothing.",
}

# Long-form help dialog content (accurate to the math in ``rf_boundary_types.py``
# / ``rf_boundary_extraction.py`` / ``rf_radial_foot_boundary.py``).
_HELP_TEXT = """<h3>Parameters</h3>
<p>
<b>min overlap %</b> — Minimum percentage of overlapping touches a forearm
vertex must accumulate to be kept in the response field (the threshold). Lower
keeps more of the field; higher keeps only strongly-overlapping regions.
<i>Checkbox off:</i> no threshold — every contacted vertex is kept.
</p>
<p>
<b>median filter size</b> — Side length (odd) of the square median window that
denoises the interpolated response grid before contour detection.
<i>Checkbox off:</i> no median filtering (raw interpolated grid).
</p>
<p>
<b>radial gauss &sigma;</b> — Sigma of the Gaussian pre-smoothing applied to
the response field before the Hessian curvature step.
<i>Checkbox off:</i> no pre-smoothing (Hessian runs on the raw normalised
field).
</p>
<p>
<b>radial hess &sigma;</b> — Sigma (scale) of the Hessian derivative kernel
used to find the &lambda;max curvature ring — the "foot of the mountain" that
defines the boundary. This is the core detector and is always applied (no
checkbox).
</p>
<p>
<b>envelope smooth &sigma;</b> — Light circular Gaussian smoothing (in
contour-vertex units) applied to the final footprint-clipped contour to
remove the marching-squares staircase.
<i>Checkbox off:</i> no contour smoothing.
</p>
<h3>Buttons</h3>
<p>
<b>Default</b> — Reset all five values and the four checkboxes to the global
boundary defaults (from the DAG config).
</p>
<p>
<b>Recompute</b> — Re-run the preview (raw &rarr; smoothed &rarr; &lambda;max
&rarr; contour) with the current parameters and toggles. Updates the status
indicator below; does not save.
</p>
<p>
<b>Validate</b> — Save the current parameters and toggle state to this
(session, gesture)'s JSON, so the batch pipeline uses them.
</p>
"""


# ---------------------------------------------------------------------------
# Small geometry helpers (ported from the sandbox contour views — production
# may not import scripts/sandbox, so the tiny pieces we need live here).
# ---------------------------------------------------------------------------


def _rc_to_uv(rc: np.ndarray, grid_u: np.ndarray, grid_v: np.ndarray) -> np.ndarray:
    """Map (row, col) pixel coords to (U, V) world coords (linear, matches grids)."""
    n_rows, n_cols = grid_u.shape
    u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
    v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
    u = u_min + rc[:, 0] * (u_max - u_min) / (n_rows - 1)
    v = v_min + rc[:, 1] * (v_max - v_min) / (n_cols - 1)
    return np.column_stack([u, v])


def _sample_z(field: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """Bilinearly sample *field* at (row, col) points, filling NaN by nearest valid."""
    nan = np.isnan(field)
    if nan.any() and not nan.all():
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        filled = field[tuple(idx)]
    else:
        filled = np.where(nan, 0.0, field)
    return map_coordinates(filled, [rc[:, 0], rc[:, 1]], order=1, mode="nearest")


def _segment_polygon_crossings(
    p0: np.ndarray, p1: np.ndarray, poly: np.ndarray
) -> list[float]:
    """Return the ``t`` in [0, 1] where segment ``p0->p1`` crosses closed ``poly``.

    ``p0`` / ``p1`` and the polygon vertices are in the same (row, col) frame.
    Each ``t`` is the parametric position along the segment; multiply by the
    segment length to get an arc-length position.
    """
    d = p1 - p0
    out: list[float] = []
    closed = np.vstack([poly, poly[0]])
    for k in range(len(closed) - 1):
        q0 = closed[k]
        q1 = closed[k + 1]
        e = q1 - q0
        denom = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(denom) < 1e-12:
            continue  # parallel / degenerate
        diff = q0 - p0
        t = (diff[0] * (-e[1]) - diff[1] * (-e[0])) / denom
        s = (d[0] * diff[1] - d[1] * diff[0]) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
            out.append(float(t))
    return sorted(out)


# ---------------------------------------------------------------------------
# 2D stages canvas (raw / smoothed / λmax heatmaps + arbitrary cross-section)
# ---------------------------------------------------------------------------


class _Stages2DCanvas(QWidget):
    """Four-panel matplotlib canvas: raw / smoothed / λmax + a user-drawn cut.

    The user click-drags an arbitrarily-oriented segment on any of the three
    heatmap panels; on release the raw / smoothed / λmax values are sampled along
    the segment into the fourth panel, with vertical markers where the current
    contour crosses the cut.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._raw: Optional[np.ndarray] = None
        self._smoothed: Optional[np.ndarray] = None
        self._lmax: Optional[np.ndarray] = None
        self._contour_rc: Optional[np.ndarray] = None
        self._peak_rc: Optional[tuple[int, int]] = None

        # Parameter-independent forearm background layer (raw SLIM UV frame).
        self._grid_u: Optional[np.ndarray] = None
        self._grid_v: Optional[np.ndarray] = None
        self._forearm_uv: Optional[np.ndarray] = None
        self._forearm_faces: Optional[np.ndarray] = None
        self._forearm_colors: Optional[np.ndarray] = None

        self._drag_start: Optional[tuple[float, float]] = None  # (row, col)
        self._rubber: Optional[Line2D] = None
        self._cut: Optional[tuple[np.ndarray, np.ndarray]] = None  # (p0, p1) rc

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(9, 7))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas, stretch=1)

        self._ax_raw = self._fig.add_subplot(2, 2, 1)
        self._ax_smoothed = self._fig.add_subplot(2, 2, 2)
        self._ax_lmax = self._fig.add_subplot(2, 2, 3)
        self._ax_cut = self._fig.add_subplot(2, 2, 4)
        self._image_axes = (self._ax_raw, self._ax_smoothed, self._ax_lmax)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------ data hand-off
    def set_stages(
        self,
        raw: np.ndarray,
        smoothed: np.ndarray,
        lmax: np.ndarray,
        contour_rc: Optional[np.ndarray],
        peak_rc: Optional[tuple[int, int]],
        grid_u: Optional[np.ndarray] = None,
        grid_v: Optional[np.ndarray] = None,
        forearm_uv: Optional[np.ndarray] = None,
        forearm_faces: Optional[np.ndarray] = None,
        forearm_colors: Optional[np.ndarray] = None,
    ) -> None:
        """Hand off the stage fields; ``contour_rc``/``peak_rc`` may be ``None``.

        On a partial recompute (field computable but no drawable contour) both
        overlays are ``None`` — the heatmaps and cross-section still render, only
        the contour polyline / peak marker / contour-crossing markers are skipped.

        ``grid_u``/``grid_v`` and the forearm mesh (``forearm_uv``/
        ``forearm_faces``/``forearm_colors``) are the parameter-independent
        background layer.  They share the raw SLIM UV frame with the grid, so the
        forearm registers exactly under the heatmap; ``forearm_colors`` (RGBA per
        vertex) is optional — a neutral-grey silhouette is drawn without it.
        """
        self._raw = np.asarray(raw, dtype=float)
        self._smoothed = np.asarray(smoothed, dtype=float)
        self._lmax = np.asarray(lmax, dtype=float)
        self._contour_rc = None if contour_rc is None else np.asarray(contour_rc, dtype=float)
        self._peak_rc = None if peak_rc is None else (int(peak_rc[0]), int(peak_rc[1]))
        self._grid_u = None if grid_u is None else np.asarray(grid_u, dtype=float)
        self._grid_v = None if grid_v is None else np.asarray(grid_v, dtype=float)
        self._forearm_uv = None if forearm_uv is None else np.asarray(forearm_uv, dtype=float)
        self._forearm_faces = None if forearm_faces is None else np.asarray(forearm_faces)
        self._forearm_colors = None if forearm_colors is None else np.asarray(forearm_colors, dtype=float)
        self._redraw_images()
        self._redraw_cut()

    # ------------------------------------------------------------ rendering
    def _draw_forearm_background(self, ax, n_rows: int, n_cols: int) -> None:
        """Draw the forearm mesh behind the heatmap, registered to its pixel frame.

        The grid and ``forearm_uv`` share the raw SLIM UV frame, so a forearm
        vertex ``(U, V)`` maps into the heatmap pixel frame by the same linear map
        ``_rc_to_uv`` inverts (row = U along axis 0, col = V along axis 1).  Faces
        are filled flat by their mean vertex RGB (the PLY-sourced per-vertex skin
        colour from ``load_forearm_vertex_rgba``), or a neutral-grey silhouette
        when no colours exist (geometry-only anatomical context — explicit
        handling, not a colour fallback).
        """
        if (
            self._forearm_uv is None or self._forearm_faces is None
            or self._grid_u is None or self._grid_v is None
        ):
            return
        u_min, u_max = float(self._grid_u[0, 0]), float(self._grid_u[-1, 0])
        v_min, v_max = float(self._grid_v[0, 0]), float(self._grid_v[0, -1])
        if u_max == u_min or v_max == v_min:
            return
        u = self._forearm_uv[:, 0]
        v = self._forearm_uv[:, 1]
        px_row = (u - u_min) * (n_rows - 1) / (u_max - u_min)  # vertical (U)
        px_col = (v - v_min) * (n_cols - 1) / (v_max - v_min)  # horizontal (V)
        faces = self._forearm_faces
        verts = np.stack([px_col[faces], px_row[faces]], axis=-1)  # (n_faces, 3, 2)
        if self._forearm_colors is not None:
            face_rgb = np.clip(
                self._forearm_colors[faces][:, :, :3].mean(axis=1), 0.0, 1.0
            )
            pc = PolyCollection(verts, facecolors=face_rgb, edgecolors="none", zorder=0)
        else:
            pc = PolyCollection(verts, facecolors="0.6", edgecolors="none", zorder=0)
        ax.add_collection(pc)

    def _redraw_images(self) -> None:
        cmap = mpl_cm.inferno.copy()
        cmap.set_bad(alpha=0.0)  # NaN cells transparent -> forearm shows through
        fields = (self._raw, self._smoothed, self._lmax)
        for ax, field, (_key, label) in zip(self._image_axes, fields, _STAGE_LABELS):
            ax.clear()
            n_rows, n_cols = field.shape
            # Forearm background FIRST, then the NaN-transparent heatmap ON TOP.
            self._draw_forearm_background(ax, n_rows, n_cols)
            ax.imshow(
                np.ma.masked_invalid(field), origin="upper", cmap=cmap,
                aspect="equal", zorder=2,
            )
            # Lock the frame to the heatmap pixel extent so forearm registers exactly.
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_ylim(n_rows - 0.5, -0.5)
            if self._contour_rc is not None and self._contour_rc.shape[0] >= 2:
                closed = np.vstack([self._contour_rc, self._contour_rc[0]])
                ax.plot(closed[:, 1], closed[:, 0], color=_CONTOUR_COLOR, lw=1.5, zorder=6)
            if self._peak_rc is not None:
                ax.plot(
                    self._peak_rc[1], self._peak_rc[0], marker="o", color=_PEAK_COLOR,
                    ms=7, mec="black", mew=0.5, zorder=7,
                )
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("col (V)")
            ax.set_ylabel("row (U)")
        self._canvas.draw_idle()

    def _redraw_cut(self) -> None:
        ax = self._ax_cut
        ax.clear()
        ax.set_title("Cross-section along cut", fontsize=10)
        ax.set_xlabel("arc length (grid cells)")
        ax.set_ylabel("normalised value (0–1)")
        if self._cut is None or self._raw is None:
            ax.text(
                0.5, 0.5, "click-drag a cut on any heatmap",
                ha="center", va="center", transform=ax.transAxes, color="gray",
            )
            self._canvas.draw_idle()
            return

        p0, p1 = self._cut
        seg_len = float(np.hypot(*(p1 - p0)))
        n = max(int(round(seg_len)) + 1, 2)
        ts = np.linspace(0.0, 1.0, n)
        rows = p0[0] + ts * (p1[0] - p0[0])
        cols = p0[1] + ts * (p1[1] - p0[1])
        arc = ts * seg_len

        for field, (_key, label), color in zip(
            (self._raw, self._smoothed, self._lmax), _STAGE_LABELS,
            ("#d62728", "#1f77b4", "#2ca02c"),
        ):
            vals = map_coordinates(
                np.where(np.isnan(field), np.nan, field),
                [rows, cols], order=1, mode="constant", cval=np.nan,
            )
            # Per-stage min–max normalisation to [0, 1] (this axis only — the
            # heatmap panels and 3D surface keep true values). NaN-aware: NaN
            # samples stay NaN so the line breaks at gaps. Explicit degenerate
            # handling (no silent data fallback): an all-NaN cut skips the curve,
            # a flat curve (nanmax == nanmin) plots as a flat line at 0.
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmax > vmin:
                norm_vals = (vals - vmin) / (vmax - vmin)
            else:
                norm_vals = np.where(finite, 0.0, np.nan)
            ax.plot(arc, norm_vals, color=color, lw=1.5, label=label)

        handles = [
            Line2D([0], [0], color=c, lw=1.5, label=lbl)
            for (_k, lbl), c in zip(_STAGE_LABELS, ("#d62728", "#1f77b4", "#2ca02c"))
        ]
        if self._contour_rc is not None and self._contour_rc.shape[0] >= 2:
            crossings = _segment_polygon_crossings(p0, p1, self._contour_rc)
            for i, t in enumerate(crossings):
                ax.axvline(
                    t * seg_len, color=_CONTOUR_COLOR, ls="--", lw=1.5,
                    label="contour" if i == 0 else None,
                )
            if crossings:
                handles.append(
                    Line2D([0], [0], color=_CONTOUR_COLOR, ls="--", lw=1.5, label="contour")
                )
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
        self._canvas.draw_idle()

    # ------------------------------------------------------------ interaction
    def _on_press(self, event) -> None:
        if event.inaxes not in self._image_axes or self._raw is None:
            return
        if self._toolbar.mode:  # pan/zoom active — don't hijack the drag
            return
        if event.xdata is None or event.ydata is None:
            return
        self._drag_start = (float(event.ydata), float(event.xdata))
        self._rubber = Line2D(
            [event.xdata, event.xdata], [event.ydata, event.ydata],
            color="white", lw=1.2, ls="--",
        )
        event.inaxes.add_line(self._rubber)
        self._canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._drag_start is None or self._rubber is None:
            return
        if event.inaxes is not self._rubber.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0 = self._drag_start[1]
        y0 = self._drag_start[0]
        self._rubber.set_data([x0, event.xdata], [y0, event.ydata])
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag_start is None:
            return
        rubber = self._rubber
        start = self._drag_start
        self._drag_start = None
        self._rubber = None
        if rubber is not None:
            try:
                rubber.remove()
            except (ValueError, NotImplementedError):
                pass
        if (
            event.inaxes not in self._image_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._canvas.draw_idle()
            return
        p0 = np.array(start, dtype=float)
        p1 = np.array([float(event.ydata), float(event.xdata)], dtype=float)
        if float(np.hypot(*(p1 - p0))) < 1.0:
            self._canvas.draw_idle()
            return
        self._cut = (p0, p1)
        self._redraw_cut()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class RFContourTuningViewer(QMainWindow):
    """Per-(session, gesture) RF-contour parameter tuning GUI.

    Constructor argument shape (consumed by the Phase-4 launcher
    ``launch_rf_contour_tuning_viewer``):

    ``sessions``
        List of dicts, one per session, with keys:
            - ``session_id`` (str)
            - ``npz_path`` (pathlib.Path) — the session's
              ``<session>_population_response_fields.npz`` under
              ``spatial_extract_boundaries/iff_<metric>/<session>/``.
            - ``forearm_ply_path`` (pathlib.Path) — the session's forearm PLY
              (RF-centered space); the colour source for the 2D forearm
              background (same always-present source as the stroke-axis viewer).

    ``boundary_params``
        Global :class:`BoundaryParams` used to seed the five sliders (via
        :func:`defaults_from_boundary_params`) when a (session, gesture) has no
        saved JSON yet.

    ``contour_params_root``
        Root directory under which per-combination JSONs are read/written:
        ``<db>/4_analysed/spatial_tune_rf_contours/iff_<metric>/``.
    """

    def __init__(
        self,
        sessions: list[dict],
        boundary_params: BoundaryParams,
        contour_params_root: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not sessions:
            raise ValueError("RFContourTuningViewer: sessions list is empty.")
        for i, sess in enumerate(sessions):
            for key in ("session_id", "npz_path", "forearm_ply_path"):
                if key not in sess:
                    raise ValueError(
                        f"RFContourTuningViewer: sessions[{i}] missing required key "
                        f"{key!r}."
                    )
        if not isinstance(boundary_params, BoundaryParams):
            raise ValueError(
                "RFContourTuningViewer: boundary_params must be a BoundaryParams, "
                f"got {type(boundary_params).__name__}."
            )

        self._sessions = sessions
        self._boundary_params = boundary_params
        self._root = Path(contour_params_root)
        self._initialized = False

        # Per-session gesture lists, read once up front so the combos, the
        # green-highlighting, and the Defined table stay consistent.
        self._session_gestures: dict[str, list[str]] = {}
        for sess in sessions:
            self._session_gestures[sess["session_id"]] = _read_gesture_types(
                sess["npz_path"]
            )
        self._all_gestures = _ordered_union(
            self._session_gestures[s["session_id"]] for s in sessions
        )

        self._arrays: Optional[dict] = None
        self._forearm_colors: Optional[np.ndarray] = None
        self._grid_u: Optional[np.ndarray] = None
        self._grid_v: Optional[np.ndarray] = None
        self._grid_z: Optional[np.ndarray] = None
        self._stages: Optional[dict] = None
        self._stage_choice = "raw"
        self._plotter: Optional[QtInteractor] = None

        self.setWindowTitle("RF Contour Tuning")

        # --- top bar: session + gesture combos -------------------------------
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("Session: "))
        self._session_combo = QComboBox()
        for sess in sessions:
            self._session_combo.addItem(sess["session_id"])
        toolbar.addWidget(self._session_combo)
        toolbar.addWidget(QLabel("  Gesture: "))
        self._gesture_combo = QComboBox()
        toolbar.addWidget(self._gesture_combo)
        self._toggle_table_btn = QPushButton("Show Defined Table")
        toolbar.addWidget(self._toggle_table_btn)
        self.addToolBar(toolbar)

        # --- left param panel + right views ----------------------------------
        self._param_panel = self._build_param_panel()

        self._canvas2d = _Stages2DCanvas()

        self._viewer_placeholder = QFrame()
        self._viewer_placeholder.setFrameShape(QFrame.StyledPanel)
        ph_layout = QVBoxLayout(self._viewer_placeholder)
        ph_layout.addWidget(
            QLabel("Initialising 3D viewer..."), alignment=Qt.AlignCenter
        )

        right_split = QSplitter(Qt.Horizontal)
        right_split.addWidget(self._canvas2d)
        right_split.addWidget(self._build_3d_panel())
        # 2D canvas and 3D view share equal width (start 50/50, resize evenly).
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 1)
        right_split.setSizes([1000, 1000])

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(self._param_panel)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([320, 1400])

        # --- Defined table page ----------------------------------------------
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._stack = QStackedWidget()
        self._stack.addWidget(main_split)  # index 0
        self._stack.addWidget(self._table)  # index 1
        self.setCentralWidget(self._stack)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # --- signals ----------------------------------------------------------
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._gesture_combo.currentIndexChanged.connect(self._on_gesture_changed)
        self._default_btn.clicked.connect(self._on_default_clicked)
        self._recompute_btn.clicked.connect(self._on_recompute_clicked)
        self._validate_btn.clicked.connect(self._on_validate_clicked)
        self._help_btn.clicked.connect(self._on_help_clicked)
        self._toggle_table_btn.clicked.connect(self._on_toggle_table)
        for btn in self._stage_buttons.buttons():
            btn.toggled.connect(self._on_stage_changed)

        self._populate_gesture_combo(self._sessions[0]["session_id"])
        self._update_combo_backgrounds()

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _build_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Contour parameters")
        form = QFormLayout(group)
        self._param_spins: dict[str, QWidget] = {}
        self._param_enabled: dict[str, QCheckBox] = {}

        overlap = QDoubleSpinBox()
        overlap.setRange(0.0, 100.0)
        overlap.setSingleStep(1.0)
        overlap.setDecimals(1)
        overlap.setToolTip(_PARAM_TOOLTIPS["min_overlap_pct"])
        self._param_spins["min_overlap_pct"] = overlap

        mfs = QSpinBox()
        mfs.setRange(1, 199)
        mfs.setSingleStep(2)
        mfs.setToolTip(_PARAM_TOOLTIPS["median_filter_size"])
        mfs.valueChanged.connect(self._on_mfs_changed)
        self._param_spins["median_filter_size"] = mfs

        for key in ("radial_gauss_sigma", "radial_hess_sigma"):
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 50.0)
            spin.setSingleStep(0.5)
            spin.setDecimals(2)
            spin.setToolTip(_PARAM_TOOLTIPS[key])
            self._param_spins[key] = spin

        env = QDoubleSpinBox()
        env.setRange(0.0, 20.0)
        env.setSingleStep(0.5)
        env.setDecimals(2)
        env.setToolTip(_PARAM_TOOLTIPS["radial_envelope_smooth_sigma"])
        self._param_spins["radial_envelope_smooth_sigma"] = env

        for key, label in _PARAM_LABELS:
            spin = self._param_spins[key]
            if key in _TOGGLEABLE_PARAM_KEYS:
                checkbox = QCheckBox()
                checkbox.setToolTip(_TOGGLE_TOOLTIPS[key])
                checkbox.toggled.connect(
                    lambda checked, k=key: self._on_param_toggled(k, checked)
                )
                self._param_enabled[key] = checkbox
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addWidget(checkbox)
                row.addWidget(spin, stretch=1)
                row_widget = QWidget()
                row_widget.setLayout(row)
                form.addRow(label + ":", row_widget)
            else:
                form.addRow(label + ":", spin)
        layout.addWidget(group)

        actions = QHBoxLayout()
        self._default_btn = QPushButton("Default")
        self._default_btn.setToolTip(
            "Reset all values and toggles to the global boundary defaults."
        )
        self._recompute_btn = QPushButton("Recompute")
        self._recompute_btn.setToolTip(
            "Re-run the preview with the current parameters and toggles "
            "(does not save)."
        )
        self._validate_btn = QPushButton("Validate")
        self._validate_btn.setToolTip(
            "Save the current parameters and toggles for this session/gesture "
            "so the batch pipeline uses them."
        )
        self._help_btn = QPushButton("?")
        self._help_btn.setToolTip("Show parameter and button descriptions.")
        self._help_btn.setFixedWidth(28)
        actions.addWidget(self._default_btn)
        actions.addWidget(self._recompute_btn)
        actions.addWidget(self._validate_btn)
        actions.addWidget(self._help_btn)
        layout.addLayout(actions)

        status_row = QHBoxLayout()
        self._status_led = QLabel()
        self._status_led.setFixedSize(14, 14)
        self._set_led(_LED_GREY)
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        status_row.addWidget(self._status_led, alignment=Qt.AlignTop)
        status_row.addWidget(self._info_label, stretch=1)
        layout.addLayout(status_row)

        layout.addStretch(1)
        return panel

    def _build_3d_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        radio_row = QHBoxLayout()
        radio_row.addWidget(QLabel("3D surface: "))
        self._stage_buttons = QButtonGroup(panel)
        for key, label in _STAGE_LABELS:
            rb = QRadioButton(label)
            rb.setProperty("stage_key", key)
            if key == self._stage_choice:
                rb.setChecked(True)
            self._stage_buttons.addButton(rb)
            radio_row.addWidget(rb)
        radio_row.addStretch(1)
        lay.addLayout(radio_row)

        lay.addWidget(self._viewer_placeholder, stretch=1)
        return panel

    def _set_led(self, color: str) -> None:
        """Restyle the inline status LED as a small filled circle in *color*."""
        self._status_led.setStyleSheet(
            f"background-color: {color}; border-radius: 7px;"
        )

    def _on_help_clicked(self) -> None:
        """Show the "?" help dialog explaining the parameters and buttons."""
        box = QMessageBox(self)
        box.setWindowTitle("Contour parameters — help")
        box.setTextFormat(Qt.RichText)
        box.setText(_HELP_TEXT)
        box.setIcon(QMessageBox.Information)
        box.exec_()

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        # Create the PyVista interactor lazily — eager init in __init__ has been
        # known to crash on some Windows/Mesa configurations.
        self._plotter = QtInteractor(self)
        self._plotter.set_background("black")

        parent_layout = self._viewer_placeholder.layout()
        while parent_layout.count():
            item = parent_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        parent_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(self._plotter.interactor)

        try:
            self._plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self._plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self._plotter.render_window.SetSize(sz.width(), sz.height())

        self._load_current(recompute=True)

    def closeEvent(self, event):  # noqa: N802
        if self._plotter is not None:
            self._plotter.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Selection / loading
    # ------------------------------------------------------------------

    def _current_session(self) -> dict:
        return self._sessions[self._session_combo.currentIndex()]

    def _current_gesture(self) -> Optional[str]:
        gesture = self._gesture_combo.currentText()
        return gesture if gesture else None

    def _on_session_changed(self, index: int) -> None:
        if not (0 <= index < len(self._sessions)):
            return
        self._populate_gesture_combo(self._sessions[index]["session_id"])
        self._load_current(recompute=self._initialized)

    def _on_gesture_changed(self, index: int) -> None:
        if index < 0:
            return
        self._load_current(recompute=self._initialized)

    def _populate_gesture_combo(self, session_id: str) -> None:
        gestures = self._session_gestures[session_id]
        self._gesture_combo.blockSignals(True)
        self._gesture_combo.clear()
        for g in gestures:
            self._gesture_combo.addItem(g)
        self._gesture_combo.setCurrentIndex(0 if gestures else -1)
        self._gesture_combo.blockSignals(False)
        self._update_gesture_backgrounds()

    def _load_current(self, recompute: bool) -> None:
        """Load arrays for the current (session, gesture) and seed the sliders."""
        session = self._current_session()
        gesture = self._current_gesture()
        if gesture is None:
            return

        try:
            self._arrays = load_session_arrays(session["npz_path"], gesture)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            QMessageBox.critical(self, "Load error", f"{type(exc).__name__}: {exc}")
            return

        # Real forearm skin colour for the 2D background — loaded from the forearm
        # PLY (the always-present single source the stroke-axis viewer uses), NOT
        # the response-fields NPZ (whose optional colour key may be absent on an
        # NPZ built before it existed). Per SLIM vertex (forearm_V), KDTree-
        # transferred from the PLY vertex colours. A PLY that genuinely carries no
        # vertex colours (ValueError) is the one sanctioned grey-silhouette case.
        try:
            self._forearm_colors = load_forearm_vertex_rgba(
                session["forearm_ply_path"], self._arrays["forearm_V"]
            )
        except ValueError as exc:
            self._forearm_colors = None
            logger.warning(
                "forearm colours unavailable for %s (grey silhouette): %s",
                session["session_id"], exc,
            )

        json_path = contour_params_path(self._root, session["session_id"], gesture)
        if json_path.exists():
            params = load_contour_params(json_path)
            loaded_note = "  (JSON loaded)"
        else:
            params = defaults_from_boundary_params(self._boundary_params)
            loaded_note = "  (defaults)"
        self._set_widgets_from_params(params)

        self._status.showMessage(
            f"{session['session_id']}  ·  {gesture}{loaded_note}"
        )
        if recompute:
            self._recompute()

    # ------------------------------------------------------------------
    # Parameter widgets <-> GestureContourParams
    # ------------------------------------------------------------------

    def _on_mfs_changed(self, value: int) -> None:
        # median_filter_size must stay a positive odd int.
        if value % 2 == 0:
            spin = self._param_spins["median_filter_size"]
            spin.blockSignals(True)
            spin.setValue(value + 1)
            spin.blockSignals(False)

    def _on_param_toggled(self, key: str, checked: bool) -> None:
        """Grey/un-grey the paired spinbox and refresh the preview.

        Not modal: a checkbox toggle is a light preview refresh, not a
        session/gesture switch, so the failure pop-up is suppressed
        (``notify_no_contour=False``) — the LED/status text still reflect the
        result.
        """
        self._param_spins[key].setEnabled(checked)
        self._recompute(notify_no_contour=False)

    def _set_widgets_from_params(self, params: GestureContourParams) -> None:
        values = {
            "min_overlap_pct": float(params.min_overlap_pct),
            "median_filter_size": int(params.median_filter_size),
            "radial_gauss_sigma": float(params.radial_gauss_sigma),
            "radial_hess_sigma": float(params.radial_hess_sigma),
            "radial_envelope_smooth_sigma": float(params.radial_envelope_smooth_sigma),
        }
        for key, spin in self._param_spins.items():
            spin.blockSignals(True)
            spin.setValue(values[key])
            spin.blockSignals(False)

        toggles = {
            "min_overlap_pct": params.toggles.min_overlap_pct,
            "median_filter_size": params.toggles.median_filter_size,
            "radial_gauss_sigma": params.toggles.radial_gauss_sigma,
            "radial_envelope_smooth_sigma": params.toggles.radial_envelope_smooth_sigma,
        }
        for key in _TOGGLEABLE_PARAM_KEYS:
            checked = toggles[key]
            checkbox = self._param_enabled[key]
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
            self._param_spins[key].setEnabled(checked)

    def _build_params_from_widgets(self) -> GestureContourParams:
        toggles = ContourParamToggles(
            **{
                key: self._param_enabled[key].isChecked()
                for key in _TOGGLEABLE_PARAM_KEYS
            }
        )
        return GestureContourParams(
            min_overlap_pct=float(self._param_spins["min_overlap_pct"].value()),
            median_filter_size=int(self._param_spins["median_filter_size"].value()),
            radial_gauss_sigma=float(self._param_spins["radial_gauss_sigma"].value()),
            radial_hess_sigma=float(self._param_spins["radial_hess_sigma"].value()),
            radial_envelope_smooth_sigma=float(
                self._param_spins["radial_envelope_smooth_sigma"].value()
            ),
            toggles=toggles,
        )

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _on_default_clicked(self) -> None:
        """Reset the five spinners to the pre-refactor global contour defaults.

        Reads from the same single seed source the viewer uses when a combo has no
        saved JSON — ``defaults_from_boundary_params(self._boundary_params)`` — so
        the revert values can never drift from the launch defaults (they are not
        re-hard-coded here).  This reverts regardless of any saved JSON for the
        current combo.  It only sets the widgets: no auto-save and no
        auto-recompute (matching the panel's convention — the user then clicks
        Recompute / Validate).
        """
        params = defaults_from_boundary_params(self._boundary_params)
        self._set_widgets_from_params(params)

    def _on_recompute_clicked(self) -> None:
        """Recompute-button slot: never pops the "no contour" modal.

        The researcher is actively dialling parameters here, so only the LED /
        status text should update — see ``_recompute``'s ``notify_no_contour``.
        """
        self._recompute(notify_no_contour=False)

    def _recompute(self, notify_no_contour: bool = True) -> None:
        if self._arrays is None:
            return
        try:
            params = self._build_params_from_widgets()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid parameters", str(exc))
            return

        try:
            grid_u, grid_v, grid_z, _title, n_touches = build_session_grid(
                self._arrays,
                params.effective_min_overlap_pct(),
                params.effective_median_filter_size(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Grid error", f"{type(exc).__name__}: {exc}")
            return

        # allow_partial=True: the "no data" case (None / all-NaN grid) still
        # raises loudly below; a computable field whose *contour* cannot be traced
        # comes back as a partial dict (contour_uv is None + an ``error`` message)
        # so we still render raw / smoothed / λmax and pop up the reason.
        # The effective_*() resolvers (single source of truth, shared with the
        # spatial_extract_boundaries pipeline) turn a disabled toggle into its
        # concrete skip value; radial_hess_sigma has no toggle and is used raw.
        try:
            stages = compute_radial_foot_stages(
                grid_u, grid_v, grid_z,
                gauss_sigma=params.effective_gauss_sigma(),
                hess_sigma=params.radial_hess_sigma,
                envelope_smooth_sigma=params.effective_envelope_smooth_sigma(),
                allow_partial=True,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Contour error", str(exc))
            return

        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._grid_z = np.asarray(grid_z, dtype=float)
        self._stages = stages

        partial = stages["contour_uv"] is None
        self._canvas2d.set_stages(
            self._grid_z, stages["smoothed"], stages["lmax"],
            stages["contour_rc"], stages["peak_rc"],
            grid_u=self._grid_u, grid_v=self._grid_v,
            forearm_uv=self._arrays["forearm_uv"],
            forearm_faces=self._arrays["forearm_faces"],
            forearm_colors=self._forearm_colors,
        )
        self._render_3d()

        valid = int(np.count_nonzero(~np.isnan(self._grid_z)))
        if partial:
            self._set_led(_LED_RED)
            short_reason = str(stages["error"]).split(".")[0]
            self._info_label.setText(
                f"n_touches={n_touches}  valid cells={valid}  "
                f"contour: none — {short_reason}"
            )
            # The peak (if located) is still drawn on the panels / 3D via the
            # non-None peak_rc handed to set_stages / _render_3d above; only the
            # contour polyline and cross-section crossings stay omitted. The
            # detailed modal is shown only on a session/gesture switch
            # (``notify_no_contour=True``); a Recompute-button / checkbox-toggle
            # refresh stays quiet — the LED + text above are the only feedback.
            if notify_no_contour:
                hint = _classify_contour_error(str(stages["error"]))
                QMessageBox.warning(
                    self, "Contour not drawn",
                    f"Contour could not be drawn: {stages['error']}.\n\n"
                    f"{hint}\n\n"
                    "Showing raw / smoothed / λmax only.",
                )
        else:
            self._set_led(_LED_GREEN)
            self._info_label.setText(
                f"n_touches={n_touches}  valid cells={valid}  "
                f"contour points={len(stages['contour_rc'])}"
            )

    # ------------------------------------------------------------------
    # 3D rendering
    # ------------------------------------------------------------------

    def _on_stage_changed(self, checked: bool) -> None:
        if not checked:
            return
        btn = self._stage_buttons.checkedButton()
        if btn is None:
            return
        self._stage_choice = str(btn.property("stage_key"))
        self._render_3d()

    def _stage_field(self) -> Optional[np.ndarray]:
        if self._stage_choice == "raw":
            return self._grid_z
        if self._stages is None:
            return None
        return self._stages[self._stage_choice]

    def _render_3d(self) -> None:
        if self._plotter is None or self._grid_u is None:
            return
        field = self._stage_field()
        if field is None:
            return

        self._plotter.clear()
        self._plotter.set_background("black")
        grid = pv.StructuredGrid(self._grid_u, self._grid_v, np.asarray(field, dtype=float))
        grid["value"] = np.asarray(field, dtype=float).ravel(order="F")
        surface = grid.threshold(scalars="value")
        self._plotter.add_mesh(
            surface, scalars="value", cmap="inferno", show_scalar_bar=True,
            scalar_bar_args={"title": self._stage_choice, "color": "white"},
        )

        x_span = float(np.nanmax(self._grid_u) - np.nanmin(self._grid_u))
        y_span = float(np.nanmax(self._grid_v) - np.nanmin(self._grid_v))
        z_span = (
            float(np.nanmax(field) - np.nanmin(field))
            if np.any(np.isfinite(field)) else 0.0
        )
        target = min(x_span, y_span)
        zscale = target / z_span if z_span > 0 and target > 0 else 1.0
        self._plotter.set_scale(zscale=zscale, render=False)

        # On a partial recompute contour_rc / peak_rc are None (field renders,
        # overlays are skipped), so guard every overlay on their presence.
        stages = self._stages
        if stages is not None and stages["contour_rc"] is not None:
            contour_rc = np.asarray(stages["contour_rc"], dtype=float)
            if contour_rc.shape[0] >= 2:
                uv = _rc_to_uv(contour_rc, self._grid_u, self._grid_v)
                z = _sample_z(np.asarray(field, dtype=float), contour_rc)
                pts = np.column_stack([uv[:, 0], uv[:, 1], z])
                pts = np.vstack([pts, pts[0]])
                line = pv.lines_from_points(pts)
                self._plotter.add_mesh(
                    line, color=_CONTOUR_COLOR, line_width=4, name="contour",
                )
        if stages is not None and stages["peak_rc"] is not None:
            peak_rc = stages["peak_rc"]
            peak_uv = _rc_to_uv(
                np.asarray([peak_rc], dtype=float), self._grid_u, self._grid_v
            )
            peak_z = _sample_z(
                np.asarray(field, dtype=float), np.asarray([peak_rc], dtype=float)
            )
            if np.isfinite(peak_z[0]):
                radius = 0.02 * target
                sphere = pv.Sphere(
                    radius=radius,
                    center=(peak_uv[0, 0], peak_uv[0, 1], float(peak_z[0])),
                )
                sphere.points[:, 2] = (
                    float(peak_z[0]) + (sphere.points[:, 2] - float(peak_z[0])) / zscale
                )
                self._plotter.add_mesh(sphere, color=_PEAK_COLOR, name="peak")

        self._plotter.show_bounds(
            xlabel="U", ylabel="V", zlabel=self._stage_choice, color="white"
        )
        self._plotter.view_isometric()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Validate / persistence
    # ------------------------------------------------------------------

    def _on_validate_clicked(self) -> None:
        session = self._current_session()
        gesture = self._current_gesture()
        if gesture is None:
            raise ValueError("RFContourTuningViewer: no gesture selected to validate.")
        try:
            params = self._build_params_from_widgets()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid parameters", str(exc))
            return

        json_path = contour_params_path(self._root, session["session_id"], gesture)
        created_at = ""
        if json_path.exists():
            created_at = load_contour_params_created_at(json_path)
        save_contour_params(
            json_path, params, session["session_id"], gesture, created_at=created_at
        )
        self._update_combo_backgrounds()
        self._update_gesture_backgrounds()
        if self._stack.currentIndex() == 1:
            self._refresh_table()
        self._status.showMessage(f"Saved {json_path}")
        QMessageBox.information(self, "Saved", f"Contour params saved to:\n{json_path}")

    # ------------------------------------------------------------------
    # Combo / table green highlighting
    # ------------------------------------------------------------------

    def _has_json(self, session_id: str, gesture: str) -> bool:
        return contour_params_path(self._root, session_id, gesture).exists()

    def _session_fully_defined(self, session_id: str) -> bool:
        gestures = self._session_gestures[session_id]
        return bool(gestures) and all(self._has_json(session_id, g) for g in gestures)

    def _update_combo_backgrounds(self) -> None:
        combo = self._session_combo
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                session_id = self._sessions[i]["session_id"]
                if self._session_fully_defined(session_id):
                    combo.setItemData(i, _GREEN_BG, Qt.BackgroundRole)
                else:
                    combo.setItemData(i, None, Qt.BackgroundRole)
        finally:
            combo.blockSignals(False)

    def _update_gesture_backgrounds(self) -> None:
        combo = self._gesture_combo
        session_id = self._current_session()["session_id"]
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                gesture = combo.itemText(i)
                if self._has_json(session_id, gesture):
                    combo.setItemData(i, _GREEN_BG, Qt.BackgroundRole)
                else:
                    combo.setItemData(i, None, Qt.BackgroundRole)
        finally:
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Defined table
    # ------------------------------------------------------------------

    def _on_toggle_table(self) -> None:
        if self._stack.currentIndex() == 0:
            self._refresh_table()
            self._stack.setCurrentIndex(1)
            self._toggle_table_btn.setText("Show Tuning View")
        else:
            self._stack.setCurrentIndex(0)
            self._toggle_table_btn.setText("Show Defined Table")

    def _refresh_table(self) -> None:
        sessions = self._sessions
        gestures = self._all_gestures
        self._table.clear()
        self._table.setRowCount(len(sessions))
        self._table.setColumnCount(len(gestures))
        self._table.setHorizontalHeaderLabels(gestures)
        self._table.setVerticalHeaderLabels([s["session_id"] for s in sessions])
        for r, sess in enumerate(sessions):
            session_id = sess["session_id"]
            valid_gestures = set(self._session_gestures[session_id])
            for c, gesture in enumerate(gestures):
                item = QTableWidgetItem()
                if gesture not in valid_gestures:
                    item.setText("–")
                elif self._has_json(session_id, gesture):
                    item.setText("✓")
                    item.setBackground(_GREEN_BG)
                else:
                    item.setText("")
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classify_contour_error(error: str) -> str:
    """Map a partial ``compute_radial_foot_stages`` error to a plain-language
    cause + the specific slider to try.

    Substring-matched against the ``error`` string the shared stages helper
    returns on a partial result.  Unrecognised text falls through to the raw
    error so nothing is ever hidden (fail-loud, not a silent fallback).
    """
    if "no λmax foot plateau" in error:
        return (
            "The response field has no clear curvature ring around the peak "
            "(too flat, or the scale is off). Try raising radial_hess_sigma, and "
            "adjusting radial_gauss_sigma; lowering min_overlap_pct can also help "
            "so the field extends past the peak."
        )
    if "no contour encloses the peak" in error or "seed is not inside" in error:
        return (
            "The contour could not be clipped to the visible footprint. Try "
            "adjusting radial_envelope_smooth_sigma, or a slightly larger (odd) "
            "median_filter_size."
        )
    if "find_peak_location returned None" in error:
        return (
            "No response peak could be located — the field is essentially flat. "
            "Lower min_overlap_pct to keep more of the field."
        )
    if "peak at border" in error:
        return (
            "The response peak is at the edge of the mapped data. Lower "
            "min_overlap_pct, or this RF may genuinely sit at the region boundary."
        )
    return error


def _read_gesture_types(npz_path: Path) -> list[str]:
    """Read the ``gesture_types`` list from a boundary NPZ (fail-fast)."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"boundary NPZ not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as d:
        if "gesture_types" not in d:
            raise KeyError(f"missing 'gesture_types' in NPZ: {npz_path}")
        return [str(g) for g in d["gesture_types"]]


def _ordered_union(iterables) -> list[str]:
    """Ordered de-duplicated union preserving first-seen order."""
    seen: dict[str, None] = {}
    for it in iterables:
        for x in it:
            if x not in seen:
                seen[x] = None
    return list(seen.keys())


def load_contour_params_created_at(path: Path) -> str:
    """Read only the ``created_at`` stamp from an existing params JSON.

    Used so a re-Validate preserves the original creation time (``save`` then
    refreshes ``modified_at``). Fail-fast on a missing file / malformed key.
    """
    import json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contour params JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "created_at" not in data:
        raise ValueError(f"Contour params JSON missing 'created_at': {path}")
    return str(data["created_at"])
