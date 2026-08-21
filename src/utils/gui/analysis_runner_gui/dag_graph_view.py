"""DAG graph view: QGraphicsView subclass with grandalf Sugiyama layout."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from PyQt5.QtCore import QLineF, QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QKeySequence, QPen
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QShortcut

logger = logging.getLogger(__name__)

from grandalf.graphs import Edge as GEdge
from grandalf.graphs import Graph
from grandalf.graphs import Vertex
from grandalf.layouts import SugiyamaLayout

from analysis.pipeline.execution_events import TaskStatus
from utils.gui.analysis_runner_gui.dag_graph_items import GRID_SIZE, DagEdge, DagTaskNode
from utils.pipeline.dag_config_model import DagConfigModel

_SPACING_FACTOR = 1.4
_SCENE_MARGIN_FRACTION = 0.5

#: Colour of the background grid a dragged node snaps to.  Faint on purpose: it
#: is an alignment aid, not content.
_COLOR_GRID = QColor("#ececec")

#: Below this on-screen spacing (device pixels) the grid is not drawn.  Zoomed
#: far out, one grid cell is smaller than a pixel: the lines would fuse into a
#: flat wash while costing one ``QLineF`` per cell of the exposed rect.
_MIN_GRID_DEVICE_PX = 6.0

#: Quoted in the error a malformed layout sidecar raises, so the message names
#: the shape the reader expects rather than only the shape it got.
EXPECTED_LAYOUT_SHAPE = '{"nodes": {"<task name>": [x, y], ...}, "view": {...}}'


class _VertexView:
    """Minimal view object required by grandalf layout engine."""

    def __init__(self, w: float, h: float) -> None:
        self.w = w
        self.h = h
        self.xy = (0.0, 0.0)


class DagGraphView(QGraphicsView):
    """Interactive DAG graph view with zoom, pan, and embedded checkboxes.

    Signals
    -------
    node_clicked(task_name)
    enabled_changed(task_name, new_value)
    force_changed(task_name, new_value)
    bypass_changed(task_name, new_value)
    """

    node_clicked = pyqtSignal(str)
    enabled_changed = pyqtSignal(str, bool)
    force_changed = pyqtSignal(str, bool)
    bypass_changed = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)

        self._nodes: dict[str, DagTaskNode] = {}
        self._edges: list[DagEdge] = []
        self._panning = False
        self._pan_start = QPoint()
        self._config_path: Path | None = None
        self._pending_view: dict | None = None
        # Re-entrancy latch for _on_node_moved.  Loading a layout calls setPos on
        # every node in turn; each one re-enters this handler, which updates the
        # scene rect, which can move the viewport and settle further positions.
        # Without the latch that recursion overflows the stack at startup.
        self._in_node_moved = False

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_layout)

        fit_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        fit_shortcut.activated.connect(self.fit_all)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate(self, model: DagConfigModel) -> None:
        """Clear and rebuild the graph from *model*.

        Every node is rebuilt from scratch, so any run status painted on the
        previous graph is discarded with it: a freshly populated graph is
        uniformly :attr:`TaskStatus.PENDING`.
        """
        self._save_timer.stop()
        self._scene.clear()
        self._nodes = {}
        self._edges = []
        self._config_path = model.path

        task_names = model.get_task_names()
        if not task_names:
            return

        # Create nodes first so each one measures its own text-fitted width.
        pre_nodes: dict[str, DagTaskNode] = {
            name: DagTaskNode(name, model, category=model._get_task(name).get("category", "none"))
            for name in task_names
        }

        vertices: dict[str, Vertex] = {}
        for name, node in pre_nodes.items():
            v = Vertex(name)
            r = node.rect()
            v.view = _VertexView(r.width(), r.height())
            vertices[name] = v

        gedges: list[GEdge] = []
        for name in task_names:
            for dep in model.get_task_dependencies(name):
                if dep in vertices:
                    gedges.append(GEdge(vertices[dep], vertices[name]))

        g = Graph(list(vertices.values()), gedges)

        x_offset = 0.0
        for component in g.C:
            layout = SugiyamaLayout(component)
            layout.init_all(optimize=True)
            layout.draw()

            comp_min_x = min(v.view.xy[0] for v in component.sV)
            for v in component.sV:
                raw_x, raw_y = v.view.xy
                shifted_x = (raw_x - comp_min_x + x_offset) * _SPACING_FACTOR
                shifted_y = raw_y * _SPACING_FACTOR
                v.view.xy = (shifted_x, shifted_y)

            comp_max_x = max(v.view.xy[0] for v in component.sV)
            comp_max_w = max(pre_nodes[v.data].rect().width() for v in component.sV)
            x_offset = comp_max_x + comp_max_w * _SPACING_FACTOR * 1.5

        for name in task_names:
            node = pre_nodes[name]
            x, y = vertices[name].view.xy
            node.setPos(x, y)
            self._scene.addItem(node)
            self._nodes[name] = node

            node.signals.node_clicked.connect(self.node_clicked)
            node.signals.enabled_changed.connect(self.enabled_changed)
            node.signals.force_changed.connect(self.force_changed)
            node.signals.bypass_changed.connect(self.bypass_changed)
            node.signals.position_changed.connect(self._on_node_moved)

        for name in task_names:
            for dep in model.get_task_dependencies(name):
                if dep in self._nodes:
                    edge = DagEdge(self._nodes[dep], self._nodes[name])
                    self._scene.addItem(edge)
                    self._edges.append(edge)

        if not self._load_layout():
            self.fit_all()
        else:
            self._update_scene_rect()
            self._apply_view()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_all(self) -> None:
        self._update_scene_rect()
        rect = self._scene.itemsBoundingRect()
        if rect.isNull():
            return
        margin_x = rect.width() * 0.05
        margin_y = rect.height() * 0.05
        self.fitInView(rect.adjusted(-margin_x, -margin_y, margin_x, margin_y), Qt.KeepAspectRatio)

    def _update_scene_rect(self) -> None:
        rect = self._scene.itemsBoundingRect()
        if rect.isNull():
            return
        margin_x = rect.width() * _SCENE_MARGIN_FRACTION
        margin_y = rect.height() * _SCENE_MARGIN_FRACTION
        self._scene.setSceneRect(rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))

    def select_task(self, task_name: str) -> None:
        for node in self._nodes.values():
            node.setSelected(False)
        if task_name in self._nodes:
            self._nodes[task_name].setSelected(True)

    def update_from_model(self, model: DagConfigModel) -> None:
        for node in self._nodes.values():
            node.update_from_model(model)

    # ------------------------------------------------------------------
    # Run status
    # ------------------------------------------------------------------

    def set_task_status(self, task_name: str, status: TaskStatus) -> None:
        """Paint *task_name*'s node with *status*; no-op if it has no node.

        An unknown name is tolerated on purpose: statuses arrive asynchronously
        from a child process, and the user may have switched workflow while it
        was still running.  Raising here would raise inside a Qt slot and take
        the window down over a stale name.  Run status is transient view state
        and is never persisted, so this method must not touch the layout
        sidecar — it deliberately does not start the save timer.
        """
        node = self._nodes.get(task_name)
        if node is None:
            return
        node.set_status(status)

    def clear_task_statuses(self) -> None:
        """Return every node to :attr:`TaskStatus.PENDING`."""
        for node in self._nodes.values():
            node.set_status(TaskStatus.PENDING)

    # ------------------------------------------------------------------
    # Background grid
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect) -> None:
        """Paint the alignment grid a dragged node snaps to.

        The pitch is ``dag_graph_items.GRID_SIZE``, imported from the item that
        does the snapping so the painted grid and the snap can never drift
        apart.  Only the exposed *rect* is covered, and the lines go out in one
        batched call with a zero-width cosmetic pen so they stay hairline-thin
        at every zoom level.
        """
        super().drawBackground(painter, rect)

        if self.transform().m11() * GRID_SIZE < _MIN_GRID_DEVICE_PX:
            return

        first_x = math.floor(rect.left() / GRID_SIZE) * GRID_SIZE
        first_y = math.floor(rect.top() / GRID_SIZE) * GRID_SIZE

        lines: list[QLineF] = []
        x = first_x
        while x < rect.right():
            lines.append(QLineF(x, rect.top(), x, rect.bottom()))
            x += GRID_SIZE
        y = first_y
        while y < rect.bottom():
            lines.append(QLineF(rect.left(), y, rect.right(), y))
            y += GRID_SIZE

        pen = QPen(_COLOR_GRID)
        pen.setWidth(0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLines(lines)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    def _layout_path(self) -> Path | None:
        if self._config_path is None:
            return None
        return self._config_path.with_suffix(".layout.json")

    def _load_layout(self) -> bool:
        """Apply saved node positions and stash the saved view transform.

        Returns True if any node position was loaded, and False only when no
        sidecar exists yet — the one benign reason not to load one, and the case
        that legitimately falls through to auto-layout.

        A sidecar that *does* exist but cannot be read raises.  Tolerating that
        is what let a flat-to-nested schema change destroy hand-placed positions
        in silence: every launch auto-laid-out the graph and the first drag
        overwrote the file with the machine layout.  An unreadable layout file
        is a fail-fast condition, not a fallback to auto-layout.

        The view (zoom/pan) is stashed on ``self._pending_view`` for
        :meth:`_apply_view` to restore once the scene rect is established.  It
        is genuinely optional — a sidecar may pin node positions without pinning
        a viewport — whereas ``"nodes"`` is the file's entire reason to exist.
        """
        path = self._layout_path()
        if path is None or not path.exists():
            return False

        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"layout sidecar {path} exists but could not be read as JSON: {exc}"
            ) from exc

        if not isinstance(saved, dict):
            raise ValueError(
                f"layout sidecar {path} must be a JSON object carrying a 'nodes' "
                f"mapping, got {type(saved).__name__}"
            )
        if "nodes" not in saved:
            raise ValueError(
                f"layout sidecar {path} has no 'nodes' key; expected shape "
                f"{EXPECTED_LAYOUT_SHAPE}"
            )
        nodes = saved["nodes"]
        if not isinstance(nodes, dict):
            raise ValueError(
                f"layout sidecar {path}: 'nodes' must be a mapping of task name "
                f"to [x, y], got {type(nodes).__name__}"
            )

        self._pending_view = saved["view"] if "view" in saved else None

        loaded_any = False
        for name, position in nodes.items():
            if not (isinstance(position, (list, tuple)) and len(position) == 2):
                raise ValueError(
                    f"layout sidecar {path}: node '{name}' must be an [x, y] "
                    f"pair, got {position!r}"
                )
            if name in self._nodes:
                self._nodes[name].setPos(float(position[0]), float(position[1]))
                loaded_any = True
        return loaded_any

    def _capture_view(self) -> dict:
        """Current zoom scale and scene-space center of the viewport."""
        center = self.mapToScene(self.viewport().rect().center())
        return {"scale": self.transform().m11(), "center": [center.x(), center.y()]}

    def _apply_view(self) -> None:
        """Restore the zoom/pan stashed by :meth:`_load_layout`, if any."""
        view = self._pending_view
        self._pending_view = None
        if not view:
            return
        scale = view["scale"]
        if scale <= 0:
            return
        cx, cy = view["center"]
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(cx, cy)

    def _save_layout(self) -> None:
        path = self._layout_path()
        if path is None:
            return
        data = {
            "nodes": {name: [node.pos().x(), node.pos().y()] for name, node in self._nodes.items()},
            "view": self._capture_view(),
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save layout to %s", path)

    def _on_node_moved(self, _task_name: str, _x: float, _y: float) -> None:
        # Guarded against re-entry: _update_scene_rect can settle item positions,
        # which re-emits position_changed straight back into this handler.  That
        # is a stack overflow at startup, when _load_layout setPos-es every node.
        if self._in_node_moved:
            return
        self._in_node_moved = True
        try:
            for edge in self._edges:
                edge.update_path()
            self._update_scene_rect()
            self._save_timer.start()
        finally:
            self._in_node_moved = False

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._save_timer.start()

    # ------------------------------------------------------------------
    # Middle-click pan
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            self._save_timer.start()
        else:
            super().mouseReleaseEvent(event)
