"""Interactive preprocessing explorer for the population-response heatmap.

Sibling sandbox to ``sandbox_gradient_ridge_boundary.py``.  Instead of running a
single fixed boundary algorithm, this tool gives the researcher an *interactive*
GUI to build and tune a **preprocessing pipeline** on the same ``grid_z`` heatmap
and inspect its effect live — before any boundary algorithm is chosen.

Layout
------
  Left   — pipeline builder: a palette of preprocessing operators (grouped by
           derivative order) that you drag into an ordered queue.  Queue items
           reorder by drag, open a parameter popup on double-click, and remove via
           right-click / Delete.  "Apply pipeline" runs the queue.
  Middle — two linked 3D surface views (PyVista): top = raw ``grid_z``,
           bottom = processed field.  The cameras are synchronised, so rotating or
           zooming one mirrors the other.
  Right  — two 2D cross-sections.  Top: a profile along U (slider picks the V
           index); bottom: a profile along V (slider picks the U index).  Each plot
           overlays the *raw* field on the left Y-axis and the *processed* field on
           the right Y-axis (``twinx``), so you can compare them at the same cut.
           A dashed guide-line on each profile marks the crossing point of the two
           cuts — i.e. the currently selected pixel.

Right-click either 3D surface to target the (U, V) cell under the cursor: both
cross-sections jump to that pixel and a cyan marker on both surfaces confirms the
location.  (The two index sliders still drive the cuts manually.)

The heatmap is the NaN-masked ``(150, 150)`` ``grid_z`` over a UV meshgrid.  Every
operator is NaN-aware (it never invents data outside the contacted region) and
raises ``ValueError`` on degenerate parameters — no silent fallbacks (CLAUDE.md).

The concerns of this explorer are split across the ``contour_explorer`` package:
numerical operators (``operators``), their schemas (``operator_registry``) and help
text (``operator_descriptions``), the pipeline model (``pipeline``), session loading
(``data_loader``), the Qt dialogs/queue (``param_dialog`` / ``description_dialog`` /
``queue_list``) and the three cooperating panels (``pipeline_builder`` /
``dual_surface_view`` / ``cross_section_view``).  This module is the thin window
that composes them and orchestrates load → apply → render.

Usage
-----
    python scripts/sandbox_preprocessing_explorer.py

Edit the hardcoded ``DATA_ROOT`` / ``SESSIONS`` / ``GESTURE`` block at the top of
``run()``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .cross_section_view import CrossSectionView
from .data_loader import load_session_grid
from .dual_surface_view import DualSurfaceView
from .pipeline import run_pipeline
from .pipeline_builder import PipelineBuilderPanel
from .pipeline_config import load_pipeline, save_pipeline

# Fixed location for the saved pipeline config, in a dedicated subfolder beside the
# sandbox source.  Save/Load are automatic (no file dialog): they always write/read
# this single shared file.
_PIPELINE_CONFIG_DIR = Path(__file__).resolve().parents[2] / "sandbox" / "saved_pipelines"
_PIPELINE_CONFIG_PATH = _PIPELINE_CONFIG_DIR / "pipeline.yaml"


class PreprocessingExplorer(QMainWindow):
    def __init__(self, sessions, gesture, min_overlap_pct, median_filter_size, parent=None):
        super().__init__(parent)
        self._sessions = list(sessions)  # list of (label, Path)
        self._gesture = gesture
        self._min_overlap_pct = min_overlap_pct
        self._median_filter_size = median_filter_size
        self._session_index = 0
        self._initialized = False
        self._load_session(0)  # populates grids + title for the first file
        self.setWindowTitle(f"Preprocessing Explorer — {self._title}")
        self._build_ui()

    def _load_session(self, index: int):
        """Load the NPZ at *index* and set the raw grids. Raises on bad input."""
        label, path = self._sessions[index]
        grid_u, grid_v, grid_z, title, n_touches = load_session_grid(
            path, self._gesture, self._min_overlap_pct, self._median_filter_size
        )
        self._session_index = index
        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._raw = np.asarray(grid_z, dtype=float)
        self._processed = self._raw.copy()
        self._title = title
        valid = int(np.count_nonzero(~np.isnan(self._raw)))
        print(f"Loaded {path.name}  ({self._gesture}, n_touches={n_touches}, valid={valid})")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addLayout(self._build_session_bar())

        self._builder = PipelineBuilderPanel()
        self._dual = DualSurfaceView()
        self._cross = CrossSectionView()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._builder)
        splitter.addWidget(self._dual)
        splitter.addWidget(self._cross)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([320, 800, 560])
        outer.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        # Wire the panels together. The chain pick → set_selection →
        # selection_changed → draw_pick_markers terminates (markers are never fed
        # back into the cross-section).
        self._builder.apply_requested.connect(self._on_apply)
        self._builder.save_requested.connect(self._on_save_pipeline)
        self._builder.load_requested.connect(self._on_load_pipeline)
        self._dual.cell_picked.connect(self._cross.set_selection)
        self._cross.selection_changed.connect(self._dual.draw_pick_markers)

        # Push the already-loaded data down to both views (rendered at first show).
        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._cross.set_data(self._grid_u, self._grid_v, self._raw, self._processed)

    def _build_session_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(self._bold_label("Session file:"))
        self._session_combo = QComboBox()
        self._session_combo.blockSignals(True)
        for label, _path in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.setCurrentIndex(self._session_index)
        self._session_combo.blockSignals(False)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        bar.addWidget(self._session_combo)
        bar.addStretch(1)
        return bar

    @staticmethod
    def _bold_label(text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setBold(True)
        lbl.setFont(f)
        return lbl

    # --------------------------------------------------------------- actions
    def _on_session_changed(self, index: int):
        if index < 0:
            return
        try:
            self._load_session(index)
        except Exception as exc:  # fail loudly to the user
            QMessageBox.critical(self, "Load error", f"{type(exc).__name__}: {exc}")
            return
        self.setWindowTitle(f"Preprocessing Explorer — {self._title}")
        if not self._initialized:
            return
        # Re-render the raw view and re-run the current queue on the new data
        # (an empty queue just mirrors raw into the processed view).
        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._dual.render_raw()
        self._on_apply()

    def _on_apply(self):
        steps = self._builder.steps()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # silence skimage/scipy RuntimeWarnings
                self._processed = run_pipeline(self._raw, steps)
        except Exception as exc:  # fail loudly to the user, not silently
            QMessageBox.critical(self, "Pipeline error", f"{type(exc).__name__}: {exc}")
            return
        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._dual.render_processed()
        self._cross.set_data(self._grid_u, self._grid_v, self._raw, self._processed)
        self._cross.refresh()

    # --------------------------------------------------- pipeline config I/O
    def _on_save_pipeline(self):
        steps = self._builder.steps()
        if not steps:
            QMessageBox.information(self, "Save pipeline", "The pipeline queue is empty.")
            return
        try:
            save_pipeline(_PIPELINE_CONFIG_PATH, steps)
        except Exception as exc:  # fail loudly to the user
            QMessageBox.critical(self, "Save error", f"{type(exc).__name__}: {exc}")
            return
        QMessageBox.information(
            self, "Save pipeline", f"Saved {len(steps)} step(s) to\n{_PIPELINE_CONFIG_PATH}"
        )

    def _on_load_pipeline(self):
        if not _PIPELINE_CONFIG_PATH.exists():
            QMessageBox.information(
                self, "Load pipeline", f"No saved pipeline found at\n{_PIPELINE_CONFIG_PATH}"
            )
            return
        try:
            steps = load_pipeline(_PIPELINE_CONFIG_PATH)
        except Exception as exc:  # fail loudly to the user
            QMessageBox.critical(self, "Load error", f"{type(exc).__name__}: {exc}")
            return
        self._builder.set_steps(steps)
        # Re-run the freshly loaded queue so the views reflect it immediately.
        self._on_apply()

    # ------------------------------------------------------- VTK deferred init
    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self):
        # Surfaces first (init + render + camera link + picking), then the 2D
        # cut, whose refresh emits selection_changed so the markers can be drawn
        # onto the now-initialised surfaces.
        self._dual.initialize_vtk()
        self._cross.refresh()

    def closeEvent(self, event):  # noqa: N802
        self._dual.close_plotters()
        super().closeEvent(event)


def run() -> None:
    # ----------------------- HARDCODED INPUT (edit me) -----------------------
    DATA_ROOT = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/"
        "semi-controlled/4_analysed/spatial_extract_boundaries/iff_mean/"
    )
    # (dropdown label, NPZ path relative to DATA_ROOT) — navigate these in the GUI.
    SESSIONS = [
        ("ST13-01", "2022-06-14_ST13-01/2022-06-14_ST13-01_population_response_fields.npz"),
        ("ST13-02", "2022-06-14_ST13-02/2022-06-14_ST13-02_population_response_fields.npz"),
        ("ST13-03", "2022-06-14_ST13-03/2022-06-14_ST13-03_population_response_fields.npz"),

        ("ST14-01", "2022-06-15_ST14-01/2022-06-15_ST14-01_population_response_fields.npz"),
        ("ST14-02", "2022-06-15_ST14-02/2022-06-15_ST14-02_population_response_fields.npz"),
        ("ST14-04", "2022-06-15_ST14-04/2022-06-15_ST14-04_population_response_fields.npz"),
        
        ("ST16-02", "2022-06-17_ST16-02/2022-06-17_ST16-02_population_response_fields.npz"),
        ("ST16-03", "2022-06-17_ST16-03/2022-06-17_ST16-03_population_response_fields.npz"),
        ("ST16-05", "2022-06-17_ST16-05/2022-06-17_ST16-05_population_response_fields.npz"),
        
        ("ST18-01", "2022-06-22_ST18-01/2022-06-22_ST18-01_population_response_fields.npz"),
        ("ST18-04", "2022-06-22_ST18-04/2022-06-22_ST18-04_population_response_fields.npz"),
    ]
    GESTURE = "all"  # one of: all, stroke, tap, stroke_proximal, stroke_distal

    MIN_OVERLAP_PCT: float = 5.0
    MEDIAN_FILTER_SIZE: int | None = 5
    # -------------------------------------------------------------------------

    sessions = [(label, DATA_ROOT / rel) for label, rel in SESSIONS]

    app = QApplication.instance() or QApplication(sys.argv)
    win = PreprocessingExplorer(sessions, GESTURE, MIN_OVERLAP_PCT, MEDIAN_FILTER_SIZE)
    win.show()
    sys.exit(app.exec_())
