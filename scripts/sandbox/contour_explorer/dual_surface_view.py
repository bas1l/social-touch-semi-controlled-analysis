"""Middle panel: two linked 3D surface views (PyVista).

Top = raw ``grid_z``, bottom = processed field.  The cameras are synchronised, so
rotating or zooming one mirrors the other.  Right-clicking either surface targets
the (U, V) cell under the cursor and emits :pysig:`cell_picked`; the window routes
that to the cross-sections, which echo back the selection so a cyan marker is drawn
at the chosen pixel on both surfaces via :pymeth:`draw_pick_markers`.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class DualSurfaceView(QWidget):
    """Two camera-linked 3D surfaces (raw + processed) with right-click picking."""

    cell_picked = pyqtSignal(int, int)  # (row i, col j) under the cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing = False
        self._initialized = False
        self._grid_u = None
        self._grid_v = None
        self._raw = None
        self._processed = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Vertical)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self._bold_label("Raw heatmap"))
        pick_hint = QLabel("Right-click either surface to retarget the cross-sections")
        pick_hint.setStyleSheet("color: gray; font-size: 10px;")
        tl.addWidget(pick_hint)
        self._plotter_raw = QtInteractor(top)
        tl.addWidget(self._plotter_raw.interactor)
        vsplit.addWidget(top)

        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self._bold_label("Processed field"))
        self._plotter_proc = QtInteractor(bot)
        bl.addWidget(self._plotter_proc.interactor)
        vsplit.addWidget(bot)

        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 1)
        lay.addWidget(vsplit)

    # ------------------------------------------------------------ data hand-off
    def set_fields(self, grid_u, grid_v, raw, processed):
        """Store the grids + both fields the renderer and markers need."""
        self._grid_u = np.asarray(grid_u, dtype=float)
        self._grid_v = np.asarray(grid_v, dtype=float)
        self._raw = np.asarray(raw, dtype=float)
        self._processed = np.asarray(processed, dtype=float)

    # ------------------------------------------------------- VTK deferred init
    def initialize_vtk(self):
        """One-shot interactor init + first render + camera link + pick setup."""
        for plotter in (self._plotter_raw, self._plotter_proc):
            try:
                plotter.interactor.Initialize()
            except Exception:
                pass
            sz = plotter.interactor.size()
            if sz.width() > 0 and sz.height() > 0:
                plotter.render_window.SetSize(sz.width(), sz.height())
        self._initialized = True
        self.render_raw()
        self.render_processed()
        self._setup_camera_link()
        self._setup_click_picking()

    def close_plotters(self):
        self._plotter_raw.close()
        self._plotter_proc.close()

    # ------------------------------------------------------------ rendering
    def render_raw(self):
        self._render_surface(self._plotter_raw, self._raw, "Raw")

    def render_processed(self):
        self._render_surface(self._plotter_proc, self._processed, "Processed")

    def _render_surface(self, plotter, z, title):
        plotter.clear()
        plotter.set_background("black")
        grid = pv.StructuredGrid(self._grid_u, self._grid_v, z)
        grid["value"] = z.ravel(order="F")
        surface = grid.threshold(scalars="value")
        plotter.add_mesh(
            surface,
            scalars="value",
            cmap="inferno",
            show_scalar_bar=True,
            scalar_bar_args={"title": title, "color": "white"},
        )
        # Scale the Z axis so it visually occupies the same span as the smaller of
        # the X/Y extents (equal visual size, NOT equal value range — the axis tick
        # labels still report the true values).
        x_span = float(np.nanmax(self._grid_u) - np.nanmin(self._grid_u))
        y_span = float(np.nanmax(self._grid_v) - np.nanmin(self._grid_v))
        z_span = float(np.nanmax(z) - np.nanmin(z)) if np.any(np.isfinite(z)) else 0.0
        target = min(x_span, y_span)
        zscale = target / z_span if z_span > 0 and target > 0 else 1.0
        # Stash for the pick marker: it must pre-divide its own Z extent by this
        # factor so the post-render axis scaling leaves the sphere visually round.
        plotter._explorer_zscale = zscale
        plotter.set_scale(zscale=zscale, render=False)
        plotter.show_bounds(xlabel="U", ylabel="V", zlabel=title, color="white")
        plotter.view_isometric()
        plotter.render()

    def _setup_camera_link(self):
        cam_raw = self._plotter_raw.camera
        cam_proc = self._plotter_proc.camera

        def _sync(src, dst, dst_plotter):
            def _cb(_obj, _event):
                if self._syncing:
                    return
                self._syncing = True
                dst.SetPosition(src.GetPosition())
                dst.SetFocalPoint(src.GetFocalPoint())
                dst.SetViewUp(src.GetViewUp())
                dst_plotter.render()
                self._syncing = False
            return _cb

        cam_raw.AddObserver("ModifiedEvent", _sync(cam_raw, cam_proc, self._plotter_proc))
        cam_proc.AddObserver("ModifiedEvent", _sync(cam_proc, cam_raw, self._plotter_raw))

    # --------------------------------------------------------- click picking
    def _setup_click_picking(self):
        """Right-click on either surface targets the (U, V) cell under the cursor."""
        for plotter in (self._plotter_raw, self._plotter_proc):
            plotter.iren.add_observer(
                "RightButtonPressEvent", self._make_pick_callback(plotter)
            )

    def _make_pick_callback(self, plotter):
        def _cb(interactor, _event):
            x, y = interactor.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            # Pick against the real surface mesh (cell picker hits the geometry,
            # unlike a focal-plane world picker which is wrong for tilted views).
            if not picker.Pick(x, y, 0, plotter.renderer):
                return  # clicked off the surface — leave the current cut
            wx, wy, _wz = picker.GetPickPosition()
            i, j = self._world_to_indices(wx, wy)
            self.cell_picked.emit(i, j)

        return _cb

    def _world_to_indices(self, wx: float, wy: float) -> tuple[int, int]:
        """Nearest grid (row i, col j) to world (U=wx, V=wy).

        The Z axis is the only scaled axis, so world X/Y equal grid_u/grid_v.
        grid_u varies down a column (axis 0) and grid_v across a row (axis 1).
        """
        u_axis = self._grid_u[:, 0]
        v_axis = self._grid_v[0, :]
        i = int(np.argmin(np.abs(u_axis - wx)))
        j = int(np.argmin(np.abs(v_axis - wy)))
        return i, j

    def draw_pick_markers(self, i: int, j: int):
        """Place a round cyan marker at the selected pixel on both 3D surfaces."""
        if not self._initialized:
            return
        u = float(self._grid_u[i, j])
        v = float(self._grid_v[i, j])
        x_span = float(np.nanmax(self._grid_u) - np.nanmin(self._grid_u))
        y_span = float(np.nanmax(self._grid_v) - np.nanmin(self._grid_v))
        radius = 0.02 * min(x_span, y_span)
        for plotter, z in ((self._plotter_raw, self._raw),
                           (self._plotter_proc, self._processed)):
            val = z[i, j]
            if not np.isfinite(val):
                # Selected cell is masked on this field — drop a stale marker.
                plotter.remove_actor("pick_marker", render=True)
                continue
            val = float(val)
            sphere = pv.Sphere(radius=radius, center=(u, v, val))
            # The renderer scales Z by ``zscale`` at draw time; pre-divide the
            # marker's Z extent by it so the sphere renders visually round.
            zscale = plotter._explorer_zscale
            sphere.points[:, 2] = val + (sphere.points[:, 2] - val) / zscale
            plotter.add_mesh(sphere, color="cyan", name="pick_marker", render=True)

    @staticmethod
    def _bold_label(text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setBold(True)
        lbl.setFont(f)
        return lbl
