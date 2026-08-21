"""DAG graph items: node (DagTaskNode) and directed edge (DagEdge)."""

from __future__ import annotations

import math

from PyQt5.QtCore import QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QCheckBox,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from analysis.pipeline.execution_events import TaskStatus
from utils.pipeline.dag_config_model import DagConfigModel

_NODE_H = 90
_MIN_NODE_W = 180
_LABEL_FONT_SIZE = 13
_H_PADDING = 40   # insets + margins + badge clearance
_CORNER_RADIUS = 6
_INSET = 6

_CATEGORY_COLORS: dict[str, QColor] = {
    "processing":           QColor("#d0e8ff"),
    "viewer":               QColor("#e8d0ff"),
    "viewer_required":      QColor("#ffe0b0"),
    "viewer_support":       QColor("#d0ffe8"),
    "none":                 QColor("#f0f0f0"),
    "foundation":           QColor("#e0e0e0"),
    "spatial_sensitivity":  QColor("#d0e8ff"),
    "stimulus_sensitivity": QColor("#ffe0d0"),
    "cross_domain":         QColor("#d0f0d0"),
}
_CATEGORY_BORDER_COLORS: dict[str, QColor] = {
    "processing":           QColor("#2255aa"),
    "viewer":               QColor("#6622aa"),
    "viewer_required":      QColor("#aa6600"),
    "viewer_support":       QColor("#006633"),
    "none":                 QColor("#666666"),
    "foundation":           QColor("#888888"),
    "spatial_sensitivity":  QColor("#2255aa"),
    "stimulus_sensitivity": QColor("#aa5522"),
    "cross_domain":         QColor("#226633"),
}
_COLOR_BG_DISABLED = QColor("#e8e8e8")
_COLOR_BORDER_DISABLED = QColor("#888888")
_COLOR_SELECTION = QColor("#ff8800")
_COLOR_EDGE = QColor("#555555")

_STATUS_GLYPH_BOX = 20   # side of the square the status glyph is centred in
_STATUS_GLYPH_INSET = 22  # from rect.right(); the category badge keeps 12

#: How each run status repaints a node, as
#: ``(fill | None, border | None, border_width, dim, glyph)``.
#:
#: ``fill``/``border`` of ``None`` mean "keep the node's own category colour",
#: which is how ``PENDING`` leaves an untouched graph looking exactly as it did
#: before this feature existed.  Status is an *overlay*: it never replaces the
#: category badge (drawn at ``rect.right() - 12``) nor the selection stroke,
#: which stay on their own visual channels.
#:
#: ``BYPASSED`` is violet and **undimmed** on purpose.  A bypassed task ran
#: nothing, yet it is accounted for — it unlocks its dependents — so it must
#: read as neither "done" (green) nor "skipped" (dimmed amber).  Reusing either
#: would let a run report claim work that never happened.
_STATUS_OVERLAY: dict[TaskStatus, tuple[QColor | None, QColor | None, float, bool, str]] = {
    TaskStatus.PENDING:              (None, None, 1.5, False, ""),
    TaskStatus.RUNNING:              (None, QColor("#1565c0"), 3.0, False, "▶"),
    TaskStatus.COMPLETED:            (None, QColor("#2e7d32"), 2.5, False, "✓"),
    TaskStatus.BYPASSED:             (QColor("#e3e0f0"), QColor("#5c4b99"), 2.0, False, "»"),
    TaskStatus.FAILED:               (None, QColor("#c62828"), 2.5, False, "✗"),
    TaskStatus.SKIPPED_DISABLED:     (QColor("#f0e0c0"), QColor("#aa6600"), 1.5, True, "–"),
    TaskStatus.SKIPPED_DEP:          (QColor("#f0e0c0"), QColor("#aa6600"), 1.5, True, "–"),
    TaskStatus.SKIPPED_GUARD:        (QColor("#ffe0a0"), QColor("#aa6600"), 2.0, True, "!"),
    TaskStatus.SKIPPED_UNREGISTERED: (QColor("#e0e0e0"), QColor("#666666"), 1.5, True, "?"),
    TaskStatus.ABORTED:              (QColor("#f0d0d0"), QColor("#888888"), 1.5, True, "⊘"),
}

#: Widest status border (``RUNNING``), used to grow the item's bounding rect so
#: a 3 px stroke is not clipped at the node's edge.
_MAX_STATUS_BORDER_W = max(width for _, _, width, _, _ in _STATUS_OVERLAY.values())


class DagTaskNode(QGraphicsRectItem):
    """Graph node representing one DAG task."""

    class _Signals(QObject):
        node_clicked = pyqtSignal(str)
        enabled_changed = pyqtSignal(str, bool)
        force_changed = pyqtSignal(str, bool)
        position_changed = pyqtSignal(str, float, float)  # task_name, x, y

    def __init__(
        self,
        task_name: str,
        model: DagConfigModel,
        category: str = "none",
        parent: QGraphicsItem | None = None,
    ) -> None:
        _lf = QFont()
        _lf.setBold(True)
        _lf.setPointSize(_LABEL_FONT_SIZE)
        node_w = max(_MIN_NODE_W, QFontMetrics(_lf).horizontalAdvance(task_name) + _H_PADDING)

        super().__init__(0, 0, node_w, _NODE_H, parent)
        self._task_name = task_name
        self._category = category
        self._updating = False
        self._status = TaskStatus.PENDING

        self.signals = DagTaskNode._Signals()

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self._enabled = model.is_task_enabled(task_name)
        force_val = model.get_task_option(task_name, "force_processing")
        self._has_force = force_val is not None

        self._apply_colors()

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(2)

        self._label = QLabel(task_name)
        self._label.setStyleSheet("font-weight: bold; font-size: 13pt;")
        self._label.setWordWrap(False)
        self._label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents)
        inner_layout.addWidget(self._label)

        cb_row = QWidget()
        cb_row.setStyleSheet("background: transparent;")
        cb_layout = QHBoxLayout(cb_row)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        cb_layout.setSpacing(6)

        self._cb_enabled = QCheckBox("Enabled")
        self._cb_enabled.setChecked(self._enabled)
        self._cb_enabled.stateChanged.connect(self._on_enabled_changed)
        cb_layout.addWidget(self._cb_enabled)

        self._cb_force: QCheckBox | None = None
        if self._has_force:
            self._cb_force = QCheckBox("Force")
            self._cb_force.setChecked(bool(force_val))
            self._cb_force.stateChanged.connect(self._on_force_changed)
            cb_layout.addWidget(self._cb_force)

        cb_layout.addStretch()
        inner_layout.addWidget(cb_row)

        proxy = QGraphicsProxyWidget(self)
        proxy.setWidget(inner)
        proxy.setPos(_INSET, _INSET)
        proxy.resize(node_w - 2 * _INSET, _NODE_H - 2 * _INSET)

    # ------------------------------------------------------------------
    # Visual helpers
    # ------------------------------------------------------------------

    def _apply_colors(self) -> None:
        if self._enabled:
            bg = _CATEGORY_COLORS.get(self._category, _CATEGORY_COLORS["none"])
            border = _CATEGORY_BORDER_COLORS.get(self._category, _CATEGORY_BORDER_COLORS["none"])
        else:
            bg = _COLOR_BG_DISABLED
            border = _COLOR_BORDER_DISABLED
        self.setBrush(QBrush(bg))
        self.setPen(QPen(border, 1.5))

    # ------------------------------------------------------------------
    # Run status
    # ------------------------------------------------------------------

    def set_status(self, status: TaskStatus) -> None:
        """Repaint this node for *status* — the run's view of the task.

        The status is transient run state: it is never written to the model and
        never to the layout sidecar.
        """
        if status not in _STATUS_OVERLAY:
            raise KeyError(
                f"no overlay defined for status {status!r}; "
                f"known: {sorted(member.name for member in _STATUS_OVERLAY)}"
            )
        if status is self._status:
            return
        self._status = status
        self.update()

    @property
    def status(self) -> TaskStatus:
        """The run status this node currently paints."""
        return self._status

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        fill, border, border_w, dim_status, glyph = _STATUS_OVERLAY[self._status]

        # A node with no status yet keeps the pre-run appearance entirely:
        # category (or disabled) fill and border, dimmed while disabled.
        neutral = fill is None and border is None
        brush = self.brush() if fill is None else QBrush(fill)
        pen = self.pen() if border is None else QPen(border, border_w)
        if dim_status or (neutral and not self._enabled):
            painter.setOpacity(0.55)

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(rect, _CORNER_RADIUS, _CORNER_RADIUS)

        painter.fillPath(path, brush)
        painter.strokePath(path, pen)

        if option.state & QStyle.State_Selected:
            sel_pen = QPen(_COLOR_SELECTION, 2.0)
            painter.strokePath(path, sel_pen)

        badge_color = _CATEGORY_BORDER_COLORS.get(self._category, _CATEGORY_BORDER_COLORS["none"])
        painter.fillRect(QRectF(rect.right() - 12, rect.top() + 2, 10, 10), badge_color)

        if glyph:
            # Full opacity even on a dimmed node: the status is the one thing
            # the user is scanning the graph for.
            painter.setOpacity(1.0)
            glyph_font = QFont()
            glyph_font.setBold(True)
            glyph_font.setPointSize(_LABEL_FONT_SIZE)
            painter.setFont(glyph_font)
            painter.setPen(QPen(border if border is not None else badge_color))
            painter.drawText(
                QRectF(
                    rect.right() - _STATUS_GLYPH_INSET,
                    rect.top() + 2,
                    _STATUS_GLYPH_BOX,
                    _STATUS_GLYPH_BOX,
                ),
                Qt.AlignCenter,
                glyph,
            )

        painter.setOpacity(1.0)

    def boundingRect(self) -> QRectF:
        # Half of the widest status border falls outside the rect; round up so
        # the RUNNING stroke is never clipped.
        margin = math.ceil(_MAX_STATUS_BORDER_W / 2.0) + 1
        return self.rect().adjusted(-margin, -margin, margin, margin)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.signals.node_clicked.emit(self._task_name)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.signals.position_changed.emit(self._task_name, value.x(), value.y())
        return super().itemChange(change, value)

    # ------------------------------------------------------------------
    # Checkbox handlers
    # ------------------------------------------------------------------

    def _on_enabled_changed(self, _state: int) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._enabled = self._cb_enabled.isChecked()
            self._apply_colors()
            self.update()
            self.signals.enabled_changed.emit(self._task_name, self._enabled)
        finally:
            self._updating = False

    def _on_force_changed(self, _state: int) -> None:
        if self._updating:
            return
        if self._cb_force is None:
            return
        self._updating = True
        try:
            self.signals.force_changed.emit(self._task_name, self._cb_force.isChecked())
        finally:
            self._updating = False

    # ------------------------------------------------------------------
    # Model sync
    # ------------------------------------------------------------------

    def update_from_model(self, model: DagConfigModel) -> None:
        """Re-read checkbox states from *model* without triggering signals."""
        self._updating = True
        try:
            self._enabled = model.is_task_enabled(self._task_name)
            self._cb_enabled.setChecked(self._enabled)
            self._apply_colors()
            self.update()
            if self._cb_force is not None:
                force_val = model.get_task_option(self._task_name, "force_processing")
                if force_val is not None:
                    self._cb_force.setChecked(bool(force_val))
        finally:
            self._updating = False


class DagEdge(QGraphicsPathItem):
    """Directed arrow edge from *source* node to *target* node."""

    _ARROW_SIZE = 8

    def __init__(
        self,
        source: DagTaskNode,
        target: DagTaskNode,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._target = target

        pen = QPen(_COLOR_EDGE, 1.5)
        pen.setStyle(Qt.SolidLine)
        self.setPen(pen)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)

        self.update_path()

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def update_path(self) -> None:
        src_rect = self._source.sceneBoundingRect()
        tgt_rect = self._target.sceneBoundingRect()

        src_pt = QPointF(src_rect.right(), src_rect.center().y())
        tgt_pt = QPointF(tgt_rect.left(), tgt_rect.center().y())

        ctrl_offset = 60.0
        c1 = QPointF(src_pt.x() + ctrl_offset, src_pt.y())
        c2 = QPointF(tgt_pt.x() - ctrl_offset, tgt_pt.y())

        path = QPainterPath(src_pt)
        path.cubicTo(c1, c2, tgt_pt)

        self._append_arrowhead(path, tgt_pt)
        self.setPath(path)

    def _append_arrowhead(self, path: QPainterPath, tip: QPointF) -> None:
        half = self._ARROW_SIZE / 2.0
        base_x = tip.x() - self._ARROW_SIZE

        p1 = QPointF(base_x, tip.y() - half)
        p2 = QPointF(tip.x(), tip.y())
        p3 = QPointF(base_x, tip.y() + half)

        path.moveTo(p1)
        path.lineTo(p2)
        path.lineTo(p3)
        path.closeSubpath()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.pen())
        path = self.path()
        painter.drawPath(path)

        painter.setBrush(QBrush(_COLOR_EDGE))
        painter.setPen(Qt.NoPen)
        tgt_rect = self._target.sceneBoundingRect()
        tip = self.mapFromScene(QPointF(tgt_rect.left(), tgt_rect.center().y()))
        half = self._ARROW_SIZE / 2.0
        base_x = tip.x() - self._ARROW_SIZE

        arrow = QPolygonF([
            QPointF(base_x, tip.y() - half),
            QPointF(tip.x(), tip.y()),
            QPointF(base_x, tip.y() + half),
        ])
        painter.drawPolygon(arrow)
