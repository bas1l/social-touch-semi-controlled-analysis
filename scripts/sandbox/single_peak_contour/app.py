"""Single-peak λmax foot-contour explorer.

A stripped-down sibling of ``sandbox_preprocessing_explorer.py``.  The preprocessing
pipeline is **fixed** — Gaussian smooth (σ) → Hessian λmax (σ) — so there is no
left-hand pipeline-builder column.  Instead, two foot-contour delineations of the
single dominant peak are computed live and overlaid on both the 3D surfaces and the
2D cross-sections:

  * **radial** — first-plateau-peak of λmax along each ray from the raw peak;
  * **region** — connected concave (λmax ≤ level) basin around the raw peak.

Layout
------
  Top bar    — session selector.
  Tool bar   — method parameters + a "Recompute" button.
  Middle     — two linked 3D surfaces (PyVista): top = raw ``grid_z``,
               bottom = the λmax field; both carry the two contour overlays, the
               seed marker (green) and any snapped-back vertices.
  Right      — two 2D cross-sections; vertical lines mark where each contour
               crosses the current cut.

Right-click either surface to retarget the cross-sections (same as the explorer).

Usage
-----
    python scripts/sandbox_single_peak_contour.py

Edit the hardcoded ``DATA_ROOT`` / ``SESSIONS`` / ``GESTURE`` block in ``run()``.
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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..contour_explorer.data_loader import build_session_grid, load_session_arrays
from ..contour_explorer.pipeline import PreprocStep, run_pipeline
from .contour_views import ContourCrossSectionView, ContourSurfaceView
from .forearm_heatmap_view import ForearmHeatmapView
from .delineation import (
    envelope_contour_to_footprint,
    extract_radial_plateau_foot,
    extract_region_growth_foot,
    find_peak_location,
)

_RADIAL_COLOR = "deepskyblue"
_REGION_COLOR = "orange"
_UNSNAPPED_COLOR = "magenta"
_ENVELOPE_COLOR = "white"


class SinglePeakContourExplorer(QMainWindow):
    def __init__(self, sessions, gesture, min_overlap_pct, median_filter_size, parent=None):
        super().__init__(parent)
        self._sessions = list(sessions)
        self._gesture = gesture
        self._min_overlap_pct = min_overlap_pct
        self._median_filter_size = median_filter_size
        self._session_index = 0
        self._initialized = False
        self._processed = None
        self._load_session(0)
        self.setWindowTitle(f"Single-Peak Contour — {self._title}")
        self._build_ui()

    # ------------------------------------------------------------ data loading
    def _load_session(self, index: int):
        label, path = self._sessions[index]
        self._arrays = load_session_arrays(path, self._gesture)
        self._session_index = index
        self._session_name = path.name
        # Initial build uses the starting constants; the toolbar controls (which
        # don't exist yet during __init__) drive every later rebuild via _recompute.
        self._rebuild_grid(self._min_overlap_pct, clean_islands=True)

    def _rebuild_grid(self, min_overlap_pct, clean_islands):
        """(Re)threshold and interpolate the cached arrays into the working grid."""
        grid_u, grid_v, grid_z, title, n_touches = build_session_grid(
            self._arrays, min_overlap_pct, self._median_filter_size,
            clean_islands=clean_islands,
        )
        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._raw = np.asarray(grid_z, dtype=float)
        self._processed = self._raw.copy()
        self._title = title
        valid = int(np.count_nonzero(~np.isnan(self._raw)))
        print(
            f"Built {self._session_name}  ({self._gesture}, n_touches={n_touches}, "
            f"min_overlap={min_overlap_pct}, clean_islands={clean_islands}, valid={valid})"
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addLayout(self._build_session_bar())
        outer.addLayout(self._build_tool_bar())

        self._dual = ContourSurfaceView()
        self._cross = ContourCrossSectionView()
        self._heatmap = ForearmHeatmapView()

        # The profiles and the heatmap share the right-hand space as tabs — only
        # one is visible at a time.
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(self._cross, "Profiles")
        self._right_tabs.addTab(self._heatmap, "Heatmap")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._dual)
        splitter.addWidget(self._right_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([480, 960])
        outer.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

        # Keep the explorer's right-click retargeting behaviour.
        self._dual.cell_picked.connect(self._cross.set_selection)
        self._cross.selection_changed.connect(self._dual.draw_pick_markers)

        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._cross.set_data(self._grid_u, self._grid_v, self._raw, self._processed)
        self._heatmap.set_data(
            self._grid_u, self._grid_v, self._raw,
            self._arrays["forearm_uv"], self._arrays["forearm_faces"],
        )

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

    def _build_tool_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        self._spin_overlap = self._dspin(0.0, 100.0, 1.0, 17.5, decimals=1)
        self._chk_islands = QCheckBox("clean islands")
        self._chk_islands.setChecked(True)
        self._spin_gauss = self._dspin(0.1, 50.0, 0.5, 8.0)
        self._spin_hess = self._dspin(0.1, 50.0, 0.5, 5.0)
        self._spin_angles = self._ispin(8, 2000, 360)
        self._spin_plateau = self._ispin(1, 50, 1)
        self._spin_prom = self._dspin(0.0, 1e9, 0.1, 0.0, decimals=4)
        self._spin_savgol = self._ispin(0, 999, 31, step=2)
        self._chk_savgol = QCheckBox("smooth")
        self._chk_savgol.setChecked(False)  # off by default — blue line follows pixels
        # Spinbox only matters when smoothing is on; mirror that in its enabled state.
        self._spin_savgol.setEnabled(False)
        self._chk_savgol.toggled.connect(self._spin_savgol.setEnabled)
        self._chk_positive = QCheckBox("require λmax>0")
        self._chk_positive.setChecked(True)
        self._spin_level = self._dspin(-1e9, 1e9, 0.01, 0.0, decimals=4)
        self._chk_radial = QCheckBox("radial")
        self._chk_radial.setChecked(True)
        self._chk_envelope = QCheckBox("envelope")
        self._chk_envelope.setChecked(True)  # 2D footprint envelope on by default
        # Very light circular smoothing (σ in vertices) to de-tooth the envelope.
        self._spin_envsmooth = self._dspin(0.0, 20.0, 0.5, 1.5, decimals=1)
        self._chk_region = QCheckBox("region")
        self._chk_region.setChecked(False)  # region off by default

        bar.addWidget(QLabel("min overlap %")); bar.addWidget(self._spin_overlap)
        bar.addWidget(self._chk_islands)
        bar.addWidget(QLabel("gauss σ")); bar.addWidget(self._spin_gauss)
        bar.addWidget(QLabel("hess σ")); bar.addWidget(self._spin_hess)
        bar.addWidget(QLabel("n_angles")); bar.addWidget(self._spin_angles)
        bar.addWidget(QLabel("plateau")); bar.addWidget(self._spin_plateau)
        bar.addWidget(QLabel("prom")); bar.addWidget(self._spin_prom)
        bar.addWidget(self._chk_savgol)
        bar.addWidget(QLabel("savgol")); bar.addWidget(self._spin_savgol)
        bar.addWidget(self._chk_positive)
        bar.addWidget(QLabel("region level")); bar.addWidget(self._spin_level)

        # Dedicated area to activate/disable the contour processings.
        proc_box = QGroupBox("Processings")
        proc_lay = QHBoxLayout(proc_box)
        proc_lay.setContentsMargins(6, 2, 6, 2)
        proc_lay.addWidget(self._chk_radial)
        proc_lay.addWidget(self._chk_envelope)
        proc_lay.addWidget(QLabel("smooth σ"))
        proc_lay.addWidget(self._spin_envsmooth)
        proc_lay.addWidget(self._chk_region)
        bar.addWidget(proc_box)

        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self._recompute)
        bar.addWidget(recompute)

        top_view = QPushButton("Top view (XY)")
        top_view.clicked.connect(lambda: self._dual.view_top_xy())
        bar.addWidget(top_view)

        bar.addStretch(1)
        return bar

    @staticmethod
    def _dspin(lo, hi, step, val, decimals=2) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    @staticmethod
    def _ispin(lo, hi, val, step=1) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

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
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"{type(exc).__name__}: {exc}")
            return
        self.setWindowTitle(f"Single-Peak Contour — {self._title}")
        if not self._initialized:
            return
        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._cross.set_data(self._grid_u, self._grid_v, self._raw, self._processed)
        self._heatmap.set_data(
            self._grid_u, self._grid_v, self._raw,
            self._arrays["forearm_uv"], self._arrays["forearm_faces"],
        )
        self._recompute()

    def _run_pipeline(self) -> np.ndarray:
        steps = [
            PreprocStep("gaussian", {"sigma": float(self._spin_gauss.value())}),
            PreprocStep("hessian_eigval", {"sigma": float(self._spin_hess.value())}),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence skimage/scipy RuntimeWarnings
            return run_pipeline(self._raw, steps)

    def _recompute(self):
        """Run the fixed pipeline, extract both feet, and push overlays to the views."""
        try:
            self._rebuild_grid(
                float(self._spin_overlap.value()), self._chk_islands.isChecked()
            )
        except Exception as exc:
            QMessageBox.critical(self, "Grid error", f"{type(exc).__name__}: {exc}")
            return

        try:
            self._processed = self._run_pipeline()
        except Exception as exc:
            QMessageBox.critical(self, "Pipeline error", f"{type(exc).__name__}: {exc}")
            return

        peak_rc = find_peak_location(self._raw)
        if peak_rc is None:
            QMessageBox.critical(self, "No peak", "Raw heatmap is entirely NaN.")
            return

        contours: list[dict] = []
        errors: list[str] = []

        if self._chk_radial.isChecked():
            try:
                if self._chk_savgol.isChecked():
                    savgol = int(self._spin_savgol.value())
                    savgol_window = savgol if savgol > 0 else None
                else:
                    savgol_window = None
                prom = float(self._spin_prom.value())
                res = extract_radial_plateau_foot(
                    self._raw, self._processed, peak_rc,
                    n_angles=int(self._spin_angles.value()),
                    plateau_size=int(self._spin_plateau.value()),
                    prominence=prom if prom > 0 else None,
                    savgol_window=savgol_window,
                    require_positive=self._chk_positive.isChecked(),
                )
                clip = res["clipped_mask"]
                contours.append({
                    "name": "radial", "color": _RADIAL_COLOR,
                    "rc": res["contour_rc"],
                    "marker_rc": res["contour_rc"][clip] if np.any(clip) else None,
                })
                # Show the un-snapped λmax foot too, but only where it differs
                # (i.e. some rays overshot the raw footprint and were clipped).
                if np.any(clip):
                    contours.append({
                        "name": "radial_unsnapped", "color": _UNSNAPPED_COLOR,
                        "rc": res["contour_unsnapped_rc"], "marker_rc": None,
                        "line_width": 2,
                    })
            except Exception as exc:
                errors.append(f"radial: {type(exc).__name__}: {exc}")
            else:
                # 2D footprint envelope: clip the *un-snapped* λmax foot straight
                # to the painted heatmap. The per-ray radial snap is bypassed here
                # — the 2D footprint intersection is the only snapping, so the line
                # can no longer bulge across grey (non-painted) cells.
                if self._chk_envelope.isChecked():
                    try:
                        env_sigma = float(self._spin_envsmooth.value())
                        env = envelope_contour_to_footprint(
                            res["contour_unsnapped_rc"], self._raw, peak_rc,
                            smooth_sigma=env_sigma if env_sigma > 0 else None,
                        )
                        contours.append({
                            "name": "radial_enveloped", "color": _ENVELOPE_COLOR,
                            "rc": env["contour_rc"], "marker_rc": None,
                        })
                    except Exception as exc:
                        errors.append(f"envelope: {type(exc).__name__}: {exc}")

        if self._chk_region.isChecked():
            try:
                res = extract_region_growth_foot(
                    self._raw, self._processed, peak_rc,
                    level=float(self._spin_level.value()),
                )
                contours.append({
                    "name": "region", "color": _REGION_COLOR,
                    "rc": res["contour_rc"], "marker_rc": None,
                })
            except Exception as exc:
                errors.append(f"region: {type(exc).__name__}: {exc}")

        self._dual.set_fields(self._grid_u, self._grid_v, self._raw, self._processed)
        self._cross.set_data(self._grid_u, self._grid_v, self._raw, self._processed)
        self._heatmap.set_data(
            self._grid_u, self._grid_v, self._raw,
            self._arrays["forearm_uv"], self._arrays["forearm_faces"],
        )
        self._dual.set_contours(contours, peak_rc)
        self._cross.set_contours(contours, peak_rc)
        self._heatmap.set_contours(contours, peak_rc)
        # Centre both cuts on the peak so the perimeter crossings are immediately
        # visible (the default centre cut usually misses an off-centre peak).
        self._cross.set_selection(peak_rc[0], peak_rc[1])

        if errors:
            QMessageBox.critical(self, "Extraction error", "\n".join(errors))

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
        self._dual.initialize_vtk()
        self._recompute()

    def closeEvent(self, event):  # noqa: N802
        self._dual.close_plotters()
        super().closeEvent(event)


def run() -> None:
    # ----------------------- HARDCODED INPUT (edit me) -----------------------
    DATA_ROOT = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/"
        "semi-controlled/4_analysed/spatial_extract_boundaries/iff_mean/"
    )
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

    MIN_OVERLAP_PCT: float = 17.5
    MEDIAN_FILTER_SIZE: int | None = 5
    # -------------------------------------------------------------------------

    sessions = [(label, DATA_ROOT / rel) for label, rel in SESSIONS]

    app = QApplication.instance() or QApplication(sys.argv)
    win = SinglePeakContourExplorer(sessions, GESTURE, MIN_OVERLAP_PCT, MEDIAN_FILTER_SIZE)
    win.show()
    sys.exit(app.exec_())
