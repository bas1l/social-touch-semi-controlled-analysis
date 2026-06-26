"""Pipeline queue widget (drag-and-drop)."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
)

from .operator_registry import OPERATORS
from .param_dialog import ParamDialog
from .pipeline import PreprocStep


class QueueList(QListWidget):
    """Ordered pipeline queue. Accepts palette drops and internal reordering."""

    def __init__(self, palette: QListWidget, parent=None):
        super().__init__(parent)
        self._palette = palette
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.itemDoubleClicked.connect(self._edit_item)

    # -- step <-> item helpers --
    def _insert_step(self, row: int, step: PreprocStep):
        item = QListWidgetItem(step.label())
        item.setData(Qt.UserRole, step)
        if row < 0 or row >= self.count():
            self.addItem(item)
        else:
            self.insertItem(row, item)
        self.setCurrentItem(item)

    def steps(self) -> list[PreprocStep]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def set_steps(self, steps: list[PreprocStep]):
        """Replace the whole queue with *steps* (used when loading a config)."""
        self.clear()
        for step in steps:
            self._insert_step(-1, step)

    # -- drag and drop --
    def dragEnterEvent(self, event):  # noqa: N802
        if event.source() is self._palette or event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        if event.source() is self._palette or event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        drop_row = self.indexAt(event.pos()).row()
        if event.source() is self._palette:
            current = self._palette.currentItem()
            op_id = current.data(Qt.UserRole) if current is not None else None
            if op_id is None:  # dropped a group header — ignore
                event.ignore()
                return
            step = PreprocStep(op_id, OPERATORS[op_id].default_params())
            self._insert_step(drop_row, step)
            event.acceptProposedAction()
        else:
            # Internal reorder — move the dragged item manually to keep its
            # attached PreprocStep object intact.
            src_row = self.currentRow()
            if src_row < 0:
                event.ignore()
                return
            if drop_row < 0:
                drop_row = self.count()
            item = self.takeItem(src_row)
            if src_row < drop_row:
                drop_row -= 1
            self.insertItem(drop_row, item)
            self.setCurrentItem(item)
            event.acceptProposedAction()

    # -- editing / removal --
    def _edit_item(self, item: QListWidgetItem):
        step: PreprocStep = item.data(Qt.UserRole)
        spec = step.spec()
        if not spec.params:
            return
        dlg = ParamDialog(spec, step.params, self)
        if dlg.exec_() == QDialog.Accepted:
            step.params = dlg.values()
            item.setText(step.label())

    def contextMenuEvent(self, event):  # noqa: N802
        item = self.itemAt(event.pos())
        if item is None:
            return
        menu = QMenu(self)
        act_remove = menu.addAction("Remove step")
        if menu.exec_(event.globalPos()) is act_remove:
            self.takeItem(self.row(item))

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            row = self.currentRow()
            if row >= 0:
                self.takeItem(row)
            return
        super().keyPressEvent(event)
