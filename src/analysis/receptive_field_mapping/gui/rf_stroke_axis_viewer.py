"""Per-session manual stroke-axis GUI (``spatial_configure_stroke_axis``).

Interactive PyQt5 window for inspecting, swapping, and redrawing the UV stroke
axis that decides each single-touch stroke's proximal/distal label. For a given
session it renders every valid stroke's SLIM-UV motion as an arrow, split across
**two side-by-side panels** — left = strokes *currently* classified proximal,
right = distal — each drawn over the same forearm SLIM-UV mesh filled with its real
per-triangle skin colour (from the forearm PLY's vertex colours) and the
same editable stroke **axis** line (auto-initialised from the sign-aligned mean of
the per-stroke motion vectors). The heavy per-session load (SLIM cache + prepared
CSV + UV motion) runs on a background thread so the window never freezes. It lets
the researcher:

  * **Swap labels** — flip the proximal/distal assignment without redrawing;
  * **Recompute axis** — re-run the auto-init from the current labels;
  * drag a new axis with a rubber-band on the canvas;
  * **Validate** — persist one ``<session>_stroke_axis.json`` per session.

The window mirrors the *persistence/lifecycle* and *interaction* conventions of
:class:`~analysis.receptive_field_mapping.gui.rf_contour_tuning_viewer.RFContourTuningViewer`
— combo navigation, load-on-select, a ``QStackedWidget`` (arrows canvas / Defined
table), green-highlighting of completed sessions on Validate, and the
rubber-band press/motion/release ``draw_idle`` interaction model of its
``_Stages2DCanvas``. **This GUI is 2D matplotlib only — there is no PyVista / VTK
view and none of the deferred-VTK init.**

The math is *not* re-implemented here: per-stroke UV motion comes from the
single-source-of-truth :func:`compute_stroke_uv_motion`, the auto-init axis from
:func:`default_stroke_axis_config` / :func:`initialize_stroke_axis`, and the live
recolouring from :func:`relabel_strokes`, so the GUI preview and the
``spatial_build_response_fields`` consumer can never drift.

Fail-fast, no silent fallback: a *missing* JSON is a legitimate first-run state
and seeds the auto-init axis, but a *corrupt/invalid* JSON, an unreadable SLIM
cache / prepared CSV, a missing/uncoloured forearm PLY, or a session with zero
valid stroke vectors all surface a loud modal (never a silent default; the
coloured forearm has no grey fallback). A session with no auto-initialisable axis
requires the researcher to draw one manually before Validate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from analysis.receptive_field_mapping.data.rf_boundary_types import (
    StrokeAxisConfig,
    StrokeUvMotion,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_rgba,
)
from analysis.receptive_field_mapping.data.rf_stroke_axis_io import (
    default_stroke_axis_config,
    load_stroke_axis,
    load_stroke_axis_created_at,
    save_stroke_axis,
    stroke_axis_path,
    stroke_axis_snapshot_path,
)
from analysis.receptive_field_mapping.metrics.rf_stroke_axis import (
    compute_stroke_uv_motion,
    load_slim_uv_cache,
    relabel_strokes,
)

logger = logging.getLogger(__name__)

_GREEN_BG = QColor("#90EE90")

# Fixed per-panel arrow colours (proximal / distal).
_PROXIMAL_COLOR = "#1f77b4"  # blue
_DISTAL_COLOR = "#d62728"    # red
_AXIS_COLOR = "#111111"      # near-black editable axis line
_PROXIMAL_END_COLOR = "#2ca02c"  # marker at the proximal (axis_end_uv) end


# ---------------------------------------------------------------------------
# Background load worker (SLIM cache + prepared CSV + UV motion, off the GUI thread)
# ---------------------------------------------------------------------------


class _StrokeLoadWorker(QThread):
    """Load a session's SLIM cache, prepared CSV, and UV motion off the GUI thread.

    ``load_slim_uv_cache`` + ``pd.read_csv`` + :func:`compute_stroke_uv_motion` +
    the forearm PLY read / KDTree colour transfer (:func:`load_forearm_vertex_rgba`)
    are the heavy synchronous cost that used to freeze the Qt main thread. Running
    them here keeps the window responsive. Only plain data crosses the thread
    boundary (a :class:`StrokeUvMotion` plus the mesh ``uv`` / ``F`` arrays and the
    per-vertex ``mesh_rgba`` needed for the coloured forearm background) — no
    matplotlib artist is ever created in :meth:`run`.

    A missing forearm PLY, or one without vertex colours, raises inside
    :func:`load_forearm_vertex_rgba` and is routed through the ``ok=False`` branch
    (no silent grey fallback — the coloured forearm is required).

    Signals
    -------
    result_ready(object)
        Emitted exactly once. Payload dict with keys ``ok`` (bool),
        ``request_id`` (int, echoed so the window can drop stale results after a
        session switch), ``session_id`` (str), ``motion`` (:class:`StrokeUvMotion`
        or ``None``), ``mesh_uv`` / ``mesh_F`` (ndarrays or ``None``),
        ``mesh_rgba`` ((len V, 4) ndarray or ``None``), and ``error``
        (``"<Type>: <msg>"`` string or ``None``). Named ``result_ready`` (not
        ``finished``) to avoid colliding with ``QThread.finished``.
    """

    result_ready = pyqtSignal(object)

    def __init__(
        self,
        slim_cache_path: Path,
        prepared_csv: Path,
        forearm_ply_path: Optional[Path],
        request_id: int,
        session_id: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._slim_cache_path = slim_cache_path
        self._prepared_csv = prepared_csv
        self._forearm_ply_path = forearm_ply_path
        self._request_id = request_id
        self._session_id = session_id

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            slim_cache = load_slim_uv_cache(self._slim_cache_path)
            prepared_df = pd.read_csv(self._prepared_csv)
            motion = compute_stroke_uv_motion(slim_cache, prepared_df)
            # Per-vertex skin RGBA aligned to the cleaned mesh vertices ``V``; the
            # heavy PLY read + KDTree transfer belong here, off the GUI thread.
            mesh_rgba = load_forearm_vertex_rgba(
                self._forearm_ply_path, slim_cache.V
            )
        except Exception as exc:  # noqa: BLE001 - surfaced on the main thread, no fallback
            logger.exception("stroke-axis background load failed")
            self.result_ready.emit({
                "ok": False,
                "request_id": self._request_id,
                "session_id": self._session_id,
                "motion": None,
                "mesh_uv": None,
                "mesh_F": None,
                "mesh_rgba": None,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return
        # Keep only the mesh arrays the background needs; drop the rest of the cache.
        self.result_ready.emit({
            "ok": True,
            "request_id": self._request_id,
            "session_id": self._session_id,
            "motion": motion,
            "mesh_uv": np.asarray(slim_cache.uv, dtype=np.float64),
            "mesh_F": np.asarray(slim_cache.F),
            "mesh_rgba": mesh_rgba,
            "error": None,
        })


# ---------------------------------------------------------------------------
# 2D UV arrows canvas (two panels: proximal / distal, shared axis line + mesh)
# ---------------------------------------------------------------------------


class _DualArrowsUVCanvas(QWidget):
    """Two side-by-side UV panels sharing one axis view + editable stroke axis.

    * **Left panel** draws every stroke *currently* classified ``stroke_proximal``
      (from the live :func:`relabel_strokes` result), **right panel** every
      ``stroke_distal`` one. Each stroke is a ``quiver`` arrow from its ``start_uv``
      to its ``end_uv``. Strokes with no proximal/distal label (unknown / invalid)
      appear in neither panel.
    * Both panels draw the same forearm SLIM-UV mesh underneath the arrows, filled
      per-triangle with the real skin colour transferred from the forearm PLY's
      vertex colours (``zorder=0``) so each arrow can be related to the surface,
      and the same editable stroke **axis** line (+ proximal-end marker).
    * The two panels share x/y limits and equal aspect, so arrow positions are
      directly comparable. A ``swap`` visibly moves arrows between the panels.

    Redraw cost: the coloured forearm mesh is built **once per session load**
    (:meth:`set_session_data`) and left untouched; only the arrow + axis artists
    are rebuilt on each interaction (:meth:`render_labels`). Interaction mirrors
    the contour tuner's ``_Stages2DCanvas``: click-drag on *either* panel lays a
    dashed rubber-band in UV *data* coordinates and, on release, hands the two
    endpoints to :attr:`on_axis_drawn`. Pan/zoom is respected per panel.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._motion: Optional[StrokeUvMotion] = None

        # Rebuilt-on-interaction artists (arrows, axis line, markers, legends).
        self._dynamic_artists: list = []

        # Endpoints of the drag, stored in UV data coordinates + the panel it began
        # on (drags are confined to their originating axes).
        self._drag_start: Optional[tuple[float, float]] = None  # (u, v)
        self._rubber: Optional[Line2D] = None
        self._drag_ax = None

        # Callback set by the parent window: on_axis_drawn(start_uv, end_uv).
        self.on_axis_drawn = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(figsize=(12, 7))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        lay.addWidget(self._toolbar)
        lay.addWidget(self._canvas, stretch=1)

        # Shared x/y so positions stay comparable and pan/zoom keeps them in sync.
        self._ax_prox = self._fig.add_subplot(1, 2, 1)
        self._ax_dist = self._fig.add_subplot(
            1, 2, 2, sharex=self._ax_prox, sharey=self._ax_prox
        )

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------ session hand-off
    def set_session_data(
        self,
        motion: StrokeUvMotion,
        mesh_uv: np.ndarray,
        mesh_F: np.ndarray,
        mesh_rgba: np.ndarray,
    ) -> None:
        """Install a new session: rebuild the coloured forearm mesh background
        (once) and set the shared square UV limits. Does not draw arrows — the
        parent follows with :meth:`render_labels`.

        ``mesh_rgba`` is the per-vertex (len V, 4) skin RGBA aligned to the SLIM
        mesh vertices; each triangle is filled with the mean of its three vertex
        colours.
        """
        self._motion = motion
        # Any in-progress drag belongs to the previous session — drop it.
        self._drag_start = None
        self._rubber = None
        self._drag_ax = None
        self._dynamic_artists = []

        # Per-face skin colour = mean of the face's three vertex colours.
        face_rgba = mesh_rgba[mesh_F].mean(axis=1)  # (n_F, 4)
        tri_verts = mesh_uv[mesh_F]  # (n_F, 3, 2)

        for ax, label in (
            (self._ax_prox, "stroke_proximal"),
            (self._ax_dist, "stroke_distal"),
        ):
            ax.clear()
            ax.set_xlabel("U")
            ax.set_ylabel("V")
            # adjustable="box" (not "datalim"): the panels are sharex/sharey, and
            # modern matplotlib raises at draw time for equal-aspect datalim on
            # shared axes. "box" keeps equal aspect with the explicit limits below.
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label} (0)", fontsize=10)
            # Real skin colour, filled per-triangle underneath everything;
            # rasterized + a single PolyCollection so per-interaction redraws stay
            # light. Built once per session load (kept across render_labels).
            mesh_coll = PolyCollection(
                tri_verts, facecolors=face_rgba, edgecolors="none",
                zorder=0, rasterized=True,
            )
            ax.add_collection(mesh_coll)

        # Square shared limits (centred on the mesh) so both panels frame the same
        # region; with adjustable="box" the equal aspect fits inside these limits.
        umin, umax = float(mesh_uv[:, 0].min()), float(mesh_uv[:, 0].max())
        vmin, vmax = float(mesh_uv[:, 1].min()), float(mesh_uv[:, 1].max())
        half = 0.5 * max(umax - umin, vmax - vmin) * 1.05
        ucen, vcen = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
        self._ax_prox.set_xlim(ucen - half, ucen + half)
        self._ax_prox.set_ylim(vcen - half, vcen + half)
        for ax in (self._ax_prox, self._ax_dist):
            ax.set_autoscale_on(False)
        self._canvas.draw_idle()

    # ------------------------------------------------------------ rendering
    def render_labels(
        self,
        valid_labels: np.ndarray,
        config: Optional[StrokeAxisConfig],
    ) -> None:
        """Redraw only the arrows + axis line for the *valid strokes'* display
        labels (``valid_labels`` aligned to ``motion.valid_mask`` order). The mesh
        background is preserved. ``config`` is ``None`` when no axis exists yet.
        """
        motion = self._motion
        if motion is None:
            return

        # Drop the previous interaction artists; keep the mesh wireframe intact.
        for art in self._dynamic_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._dynamic_artists = []

        valid_labels = np.asarray(valid_labels, dtype=object)
        vmask = motion.valid_mask
        starts = motion.start_uv[vmask]
        vecs = motion.vectors[vmask]

        prox_mask = valid_labels == "stroke_proximal"
        dist_mask = valid_labels == "stroke_distal"
        n_prox = int(np.count_nonzero(prox_mask))
        n_dist = int(np.count_nonzero(dist_mask))

        for ax, mask, color in (
            (self._ax_prox, prox_mask, _PROXIMAL_COLOR),
            (self._ax_dist, dist_mask, _DISTAL_COLOR),
        ):
            if np.any(mask):
                q = ax.quiver(
                    starts[mask, 0], starts[mask, 1],
                    vecs[mask, 0], vecs[mask, 1],
                    color=color, angles="xy", scale_units="xy", scale=1.0,
                    width=0.005, zorder=4,
                )
                self._dynamic_artists.append(q)

        # Editable axis line (start -> proximal end) on BOTH panels, if defined.
        if config is not None:
            start = np.asarray(config.axis_start_uv, dtype=float)
            end = np.asarray(config.axis_end_uv, dtype=float)
            for ax in (self._ax_prox, self._ax_dist):
                (line,) = ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color=_AXIS_COLOR, lw=2.0, ls="-", zorder=6,
                )
                (marker,) = ax.plot(
                    end[0], end[1], marker="o", color=_PROXIMAL_END_COLOR,
                    ms=9, mec="black", mew=0.6, zorder=7,
                )
                self._dynamic_artists.extend((line, marker))

        self._ax_prox.set_title(f"stroke_proximal ({n_prox})", fontsize=10)
        self._ax_dist.set_title(f"stroke_distal ({n_dist})", fontsize=10)
        self._canvas.draw_idle()

    # ------------------------------------------------------------ snapshot
    def save_snapshot(self, path: Path) -> None:
        """Save the current two-panel figure (mesh + arrows + axis) to ``path``.

        Called by Validate to persist a visual record of the validated
        proximal/distal split alongside the JSON. Renders at the current view.
        """
        self._fig.savefig(path, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------ interaction
    def _on_press(self, event) -> None:
        if self._motion is None:
            return
        if event.inaxes not in (self._ax_prox, self._ax_dist):
            return
        if self._toolbar.mode:  # pan/zoom active — don't hijack the drag
            return
        if event.xdata is None or event.ydata is None:
            return
        self._drag_ax = event.inaxes
        self._drag_start = (float(event.xdata), float(event.ydata))
        self._rubber = Line2D(
            [event.xdata, event.xdata], [event.ydata, event.ydata],
            color=_AXIS_COLOR, lw=1.5, ls="--",
        )
        self._drag_ax.add_line(self._rubber)
        self._canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._drag_start is None or self._rubber is None:
            return
        if event.inaxes is not self._drag_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        u0, v0 = self._drag_start
        self._rubber.set_data([u0, event.xdata], [v0, event.ydata])
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag_start is None:
            return
        rubber = self._rubber
        start = self._drag_start
        drag_ax = self._drag_ax
        self._drag_start = None
        self._rubber = None
        self._drag_ax = None
        if rubber is not None:
            try:
                rubber.remove()
            except (ValueError, NotImplementedError):
                pass
        if (
            event.inaxes is not drag_ax
            or event.xdata is None
            or event.ydata is None
        ):
            self._canvas.draw_idle()
            return
        p0 = np.array(start, dtype=float)
        p1 = np.array([float(event.xdata), float(event.ydata)], dtype=float)
        if float(np.hypot(*(p1 - p0))) <= 0.0:
            # Degenerate (zero-length) drag — a StrokeAxisConfig would reject it;
            # ignore the click and keep the current axis (no silent default).
            self._canvas.draw_idle()
            return
        if self.on_axis_drawn is not None:
            self.on_axis_drawn(
                (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))
            )


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class RFStrokeAxisViewer(QMainWindow):
    """Per-session manual stroke-axis GUI.

    Constructor argument shape (consumed by the Phase-3 launcher
    ``launch_stroke_axis_viewer``):

    ``sessions``
        List of dicts, one per session, each with keys:
            - ``session_id`` (str)
            - ``prepared_csv`` (pathlib.Path) — the session's ``_prepared.csv``
              carrying per-frame ``contact_location_{x,y,z}`` + ``gesture_type``;
            - ``slim_cache_path`` (pathlib.Path) — the session's SLIM-UV ``.npz``
              cache read by :func:`load_slim_uv_cache`;
            - ``forearm_ply_path`` (pathlib.Path) — the session's forearm PLY
              (RF-centered space) whose vertex colours fill the mesh background.
        All three paths must exist (fail-fast); a ``None`` ``forearm_ply_path``
        is an error (the coloured forearm is required — no grey fallback).

    ``stroke_axis_root``
        Root directory under which per-session JSONs are read/written:
        ``<db>/4_analysed/spatial_configure_stroke_axis/``.
    """

    def __init__(
        self,
        sessions: list[dict],
        stroke_axis_root: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not sessions:
            raise ValueError("RFStrokeAxisViewer: sessions list is empty.")
        for i, sess in enumerate(sessions):
            for key in (
                "session_id", "prepared_csv", "slim_cache_path", "forearm_ply_path",
            ):
                if key not in sess:
                    raise ValueError(
                        f"RFStrokeAxisViewer: sessions[{i}] missing required key "
                        f"{key!r}."
                    )
            if not sess["session_id"]:
                raise ValueError(
                    f"RFStrokeAxisViewer: sessions[{i}] has an empty 'session_id'."
                )
            # The coloured forearm is required — a None PLY path is a fail-fast
            # error (no silent grey fallback).
            if sess["forearm_ply_path"] is None:
                raise FileNotFoundError(
                    f"RFStrokeAxisViewer: sessions[{i}] forearm PLY not found for "
                    f"session {sess['session_id']!r} — a coloured forearm is "
                    "required."
                )
            for key in ("prepared_csv", "slim_cache_path", "forearm_ply_path"):
                p = Path(sess[key])
                if not p.exists():
                    raise FileNotFoundError(
                        f"RFStrokeAxisViewer: sessions[{i}] {key} does not exist: {p}"
                    )

        self._sessions = sessions
        self._root = Path(stroke_axis_root)
        self._initialized = False

        # Current-session working state (rebuilt on every _load_current).
        self._motion: Optional[StrokeUvMotion] = None
        self._config: Optional[StrokeAxisConfig] = None

        # Background-load bookkeeping. ``_load_request_id`` monotonically increases
        # so a result from a superseded session switch can be dropped as stale;
        # in-flight workers are held in a set so a QThread is never GC'd mid-run.
        self._load_request_id = 0
        self._load_workers: set[_StrokeLoadWorker] = set()

        self.setWindowTitle("Configure Stroke Axis")

        # --- top bar: session combo + Defined-table toggle -------------------
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("Session: "))
        self._session_combo = QComboBox()
        for sess in sessions:
            self._session_combo.addItem(sess["session_id"])
        toolbar.addWidget(self._session_combo)
        self._toggle_table_btn = QPushButton("Show Defined Table")
        toolbar.addWidget(self._toggle_table_btn)
        self.addToolBar(toolbar)

        # --- left action panel + central canvas ------------------------------
        self._info_panel = self._build_info_panel()
        self._canvas = _DualArrowsUVCanvas()
        self._canvas.on_axis_drawn = self._on_axis_drawn

        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(self._info_panel)
        main_split.addWidget(self._canvas)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([320, 1200])

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
        self._swap_btn.clicked.connect(self._on_swap_clicked)
        self._recompute_btn.clicked.connect(self._on_reset_axis_clicked)
        self._reset_default_btn.clicked.connect(self._on_reset_default_clicked)
        self._validate_btn.clicked.connect(self._on_validate_clicked)
        self._finish_btn.clicked.connect(self.close)
        self._toggle_table_btn.clicked.connect(self._on_toggle_table)

        self._update_combo_backgrounds()

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        group = QGroupBox("Stroke axis")
        info_layout = QVBoxLayout(group)
        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        info_layout.addWidget(self._info_label)
        layout.addWidget(group)

        actions = QVBoxLayout()
        self._swap_btn = QPushButton("Swap labels")
        self._recompute_btn = QPushButton("Recompute axis")
        self._reset_default_btn = QPushButton("Reset to default (computed)")
        self._validate_btn = QPushButton("Validate")
        actions.addWidget(self._swap_btn)
        actions.addWidget(self._recompute_btn)
        actions.addWidget(self._reset_default_btn)
        actions.addWidget(self._validate_btn)
        layout.addLayout(actions)

        hint = QLabel(
            "Click-drag on the canvas to redraw the axis. The arrow head end "
            "(green marker) is the proximal end."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch(1)

        # Finish: close the window to end the task. Kept apart from the editing
        # actions (below the stretch) and never disabled during a load, so the
        # researcher can always exit cleanly — ``closeEvent`` waits on any
        # in-flight background worker before the window is torn down.
        self._finish_btn = QPushButton("Finish")
        layout.addWidget(self._finish_btn)
        return panel

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
        # Loading the SLIM cache + prepared CSV is heavy; defer off the constructor
        # so the window paints first, then run the load on a background thread.
        self._load_current()

    def closeEvent(self, event):  # noqa: N802
        # Bump the request id so any late result is dropped, then wait for the
        # in-flight worker threads so none outlives the window (a running QThread
        # torn down with its C++ object would crash).
        self._load_request_id += 1
        for worker in list(self._load_workers):
            worker.wait()
        self._load_workers.clear()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Selection / loading
    # ------------------------------------------------------------------

    def _current_session(self) -> dict:
        return self._sessions[self._session_combo.currentIndex()]

    def _on_session_changed(self, index: int) -> None:
        if not (0 <= index < len(self._sessions)):
            return
        if self._initialized:
            self._load_current()

    def _load_current(self) -> None:
        """Kick off the current session's heavy load on a background thread.

        ``load_slim_uv_cache`` + ``pd.read_csv`` + :func:`compute_stroke_uv_motion`
        run off the GUI thread (:class:`_StrokeLoadWorker`) so the window never
        freezes. The result is applied in :meth:`_on_load_finished` on the main
        thread; a session switch mid-load supersedes the request (stale results are
        dropped by ``request_id``). Axis-config resolution and all fail-fast modals
        stay on the main thread.
        """
        session = self._current_session()
        session_id = session["session_id"]

        self._load_request_id += 1
        request_id = self._load_request_id
        self._set_busy(True)
        self._status.showMessage(f"Loading {session_id}…")

        worker = _StrokeLoadWorker(
            Path(session["slim_cache_path"]),
            Path(session["prepared_csv"]),
            Path(session["forearm_ply_path"]),
            request_id,
            session_id,
        )
        worker.result_ready.connect(self._on_load_finished)
        # Hold a strong ref until the thread is done (a GC'd running QThread crashes).
        self._load_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._load_workers.discard(w))
        worker.start()

    def _on_load_finished(self, payload: dict) -> None:
        """Apply a background-load result on the main thread (or drop it if stale).

        A *missing* JSON seeds the auto-init axis (a legitimate first-run state,
        not a silent fallback); an unreadable SLIM cache / prepared CSV, a
        *corrupt* JSON, or a session with no auto-initialisable axis all surface a
        loud modal (no silent default).
        """
        if payload["request_id"] != self._load_request_id:
            return  # a later session switch superseded this request — ignore.

        # Only re-enable the buttons once the *current* request has landed.
        self._set_busy(False)

        session_id = payload["session_id"]
        if not payload["ok"]:
            self._motion = None
            self._config = None
            QMessageBox.critical(self, "Load error", payload["error"])
            self._status.showMessage(f"{session_id}  ·  load failed")
            return

        motion = payload["motion"]
        self._motion = motion
        self._canvas.set_session_data(
            motion, payload["mesh_uv"], payload["mesh_F"], payload["mesh_rgba"]
        )

        json_path = stroke_axis_path(self._root, session_id)
        if json_path.exists():
            try:
                config = load_stroke_axis(json_path)
            except Exception as exc:  # noqa: BLE001 - corrupt JSON must fail loudly
                self._config = None
                QMessageBox.critical(
                    self, "Invalid stroke-axis JSON",
                    f"{type(exc).__name__}: {exc}",
                )
                self._render()
                self._status.showMessage(f"{session_id}  ·  invalid JSON")
                return
            note = "(JSON loaded)"
        else:
            try:
                config = default_stroke_axis_config(session_id, motion)
            except ValueError as exc:
                # Zero auto-initialisable stroke vectors: no silent default axis —
                # surface the reason and require a manual draw.
                self._config = None
                QMessageBox.warning(
                    self, "No auto-init axis",
                    f"{exc}\n\nDraw the axis manually by click-dragging on the "
                    f"canvas.",
                )
                self._render()
                self._status.showMessage(
                    f"{session_id}  ·  draw the axis manually"
                )
                return
            note = "(auto-init)"

        self._config = config
        self._render()
        self._status.showMessage(f"{session_id}  {note}")

    def _set_busy(self, busy: bool) -> None:
        """Disable the action buttons while a background load is in flight.

        The session combo stays enabled so a switch can supersede the load; the
        stale result is then dropped by ``request_id`` in :meth:`_on_load_finished`.
        """
        for btn in (
            self._swap_btn, self._recompute_btn,
            self._reset_default_btn, self._validate_btn,
        ):
            btn.setEnabled(not busy)

    def _render(self) -> None:
        """Reclassify the valid strokes from the current axis (shared compute) and
        split the arrows across the proximal / distal panels. With no axis yet, the
        arrows keep their current 3D labels so the researcher still has something to
        draw against.
        """
        motion = self._motion
        if motion is None:
            return

        valid = motion.valid_mask
        if self._config is None:
            valid_labels = motion.current_labels[valid]
        else:
            # Reclassify only the *valid* strokes so relabel_strokes never trips its
            # "labelled stroke but no valid UV motion" guard on a UV-invalid stroke.
            valid_labels = relabel_strokes(
                motion.current_labels[valid], motion.touch_keys[valid],
                motion, self._config,
            )

        self._canvas.render_labels(valid_labels, self._config)
        self._update_info(valid_labels)

    def _update_info(self, valid_labels: np.ndarray) -> None:
        motion = self._motion
        if motion is None:
            self._info_label.setText("")
            return
        n_strokes = int(len(motion.valid_mask))
        n_valid = int(np.count_nonzero(motion.valid_mask))
        n_prox = int(np.count_nonzero(valid_labels == "stroke_proximal"))
        n_dist = int(np.count_nonzero(valid_labels == "stroke_distal"))
        n_unknown = n_valid - n_prox - n_dist
        lines = [
            f"session: {self._current_session()['session_id']}",
            f"strokes: {n_strokes}  (valid: {n_valid})",
            f"proximal: {n_prox}   distal: {n_dist}   unknown: {n_unknown}",
        ]
        if self._config is None:
            lines.append("axis: not defined (draw it)")
        else:
            s = self._config.axis_start_uv
            e = self._config.axis_end_uv
            lines.append(
                f"axis: ({s[0]:.3f}, {s[1]:.3f}) -> ({e[0]:.3f}, {e[1]:.3f})"
            )
            lines.append(f"swap_proximal_distal: {self._config.swap_proximal_distal}")
        self._info_label.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Axis edits
    # ------------------------------------------------------------------

    def _on_axis_drawn(
        self, start_uv: tuple[float, float], end_uv: tuple[float, float]
    ) -> None:
        """Rubber-band release: build a fresh axis config and re-render (which
        re-runs the shared relabel + recolours the arrows). The swap flag is
        preserved across a redraw; a degenerate axis raises and is surfaced.
        """
        session_id = self._current_session()["session_id"]
        swap = self._config.swap_proximal_distal if self._config is not None else False
        try:
            self._config = StrokeAxisConfig(
                session_id=session_id,
                axis_start_uv=start_uv,
                axis_end_uv=end_uv,
                swap_proximal_distal=swap,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid axis", str(exc))
            return
        self._render()
        self._status.showMessage(f"{session_id}  ·  axis redrawn")

    def _on_swap_clicked(self) -> None:
        """Flip ``swap_proximal_distal`` and re-render (arrows swap colours)."""
        if self._config is None:
            QMessageBox.warning(
                self, "No axis",
                "There is no stroke axis to swap yet — draw one first.",
            )
            return
        self._config = StrokeAxisConfig(
            session_id=self._config.session_id,
            axis_start_uv=self._config.axis_start_uv,
            axis_end_uv=self._config.axis_end_uv,
            swap_proximal_distal=not self._config.swap_proximal_distal,
        )
        self._render()

    def _on_reset_axis_clicked(self) -> None:
        """Recompute the axis *geometry* from the auto-init while PRESERVING the
        current swap flag (snap the axis line back to the data without discarding
        the researcher's proximal/distal swap). A session with zero valid stroke
        vectors raises — surfaced as a modal, never a silent default. Use
        **Reset to default** to also clear the swap.
        """
        motion = self._motion
        if motion is None:
            return
        session_id = self._current_session()["session_id"]
        swap = self._config.swap_proximal_distal if self._config is not None else False
        try:
            default = default_stroke_axis_config(session_id, motion)
        except ValueError as exc:
            QMessageBox.warning(
                self, "Cannot recompute axis",
                f"{exc}\n\nDraw the axis manually by click-dragging on the canvas.",
            )
            return
        self._config = StrokeAxisConfig(
            session_id=session_id,
            axis_start_uv=default.axis_start_uv,
            axis_end_uv=default.axis_end_uv,
            swap_proximal_distal=swap,
        )
        self._render()
        self._status.showMessage(f"{session_id}  ·  axis recomputed (swap kept)")

    def _on_reset_default_clicked(self) -> None:
        """Reset to the pristine computed default: auto-init axis geometry AND
        ``swap_proximal_distal=False``, discarding every manual edit (a drawn axis
        and/or a swap). A session with zero valid stroke vectors raises — surfaced
        as a modal, never a silent default.
        """
        motion = self._motion
        if motion is None:
            return
        session_id = self._current_session()["session_id"]
        try:
            self._config = default_stroke_axis_config(session_id, motion)
        except ValueError as exc:
            QMessageBox.warning(
                self, "Cannot reset to default",
                f"{exc}\n\nDraw the axis manually by click-dragging on the canvas.",
            )
            return
        self._render()
        self._status.showMessage(f"{session_id}  ·  reset to default (computed)")

    # ------------------------------------------------------------------
    # Validate / persistence
    # ------------------------------------------------------------------

    def _on_validate_clicked(self) -> None:
        if self._config is None:
            QMessageBox.warning(
                self, "No axis",
                "There is no stroke axis to save yet — draw or recompute one first.",
            )
            return
        session_id = self._current_session()["session_id"]
        json_path = stroke_axis_path(self._root, session_id)
        created_at = ""
        if json_path.exists():
            created_at = load_stroke_axis_created_at(json_path)
        save_stroke_axis(json_path, self._config, created_at=created_at)

        # Also persist a snapshot image of the validated split alongside the JSON.
        snapshot_path = stroke_axis_snapshot_path(self._root, session_id)
        self._canvas.save_snapshot(snapshot_path)

        self._update_combo_backgrounds()
        if self._stack.currentIndex() == 1:
            self._refresh_table()
        self._status.showMessage(f"Saved {json_path}  +  {snapshot_path.name}")
        QMessageBox.information(
            self, "Saved",
            f"Stroke axis saved to:\n{json_path}\n\nSnapshot saved to:\n{snapshot_path}",
        )

    # ------------------------------------------------------------------
    # Combo / table green highlighting
    # ------------------------------------------------------------------

    def _has_json(self, session_id: str) -> bool:
        return stroke_axis_path(self._root, session_id).exists()

    def _update_combo_backgrounds(self) -> None:
        combo = self._session_combo
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                session_id = self._sessions[i]["session_id"]
                if self._has_json(session_id):
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
            self._toggle_table_btn.setText("Show Arrows View")
        else:
            self._stack.setCurrentIndex(0)
            self._toggle_table_btn.setText("Show Defined Table")

    def _refresh_table(self) -> None:
        sessions = self._sessions
        self._table.clear()
        self._table.setRowCount(len(sessions))
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["stroke axis"])
        self._table.setVerticalHeaderLabels([s["session_id"] for s in sessions])
        for r, sess in enumerate(sessions):
            item = QTableWidgetItem()
            if self._has_json(sess["session_id"]):
                item.setText("✓")
                item.setBackground(_GREEN_BG)
            else:
                item.setText("")
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 0, item)
        self._table.resizeColumnsToContents()
