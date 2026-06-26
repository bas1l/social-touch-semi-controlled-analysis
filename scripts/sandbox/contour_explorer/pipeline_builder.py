"""Left panel: the pipeline builder (operator palette + queue + Apply button).

A palette of preprocessing operators (grouped by derivative order) that the user
drags into an ordered queue.  Double-click a palette entry for a maths/behaviour
popup; double-click a queue step to edit its parameters; right-click / Delete to
remove.  ``apply_requested`` fires when the user clicks "Apply pipeline".
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .description_dialog import DescriptionDialog
from .operator_descriptions import OPERATOR_DESCRIPTIONS
from .operator_registry import OPERATOR_LIST, OPERATORS
from .pipeline import PreprocStep
from .queue_list import QueueList


class PipelineBuilderPanel(QWidget):
    """Operator palette + ordered pipeline queue + Apply button."""

    apply_requested = pyqtSignal()
    save_requested = pyqtSignal()
    load_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        lay.addWidget(self._bold_label("Available operators (drag → · double-click for help)"))
        self._palette = QListWidget()
        self._palette.setDragEnabled(True)
        self._palette.setDragDropMode(QAbstractItemView.DragOnly)
        self._palette.itemDoubleClicked.connect(self._show_operator_description)
        self._populate_palette()
        lay.addWidget(self._palette, stretch=1)

        lay.addWidget(self._bold_label("Pipeline queue (drop / reorder)"))
        self._queue = QueueList(self._palette)
        lay.addWidget(self._queue, stretch=1)

        apply_btn = QPushButton("Apply pipeline")
        apply_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        apply_btn.clicked.connect(self.apply_requested)
        lay.addWidget(apply_btn)

        config_row = QHBoxLayout()
        save_btn = QPushButton("Save pipeline")
        save_btn.clicked.connect(self.save_requested)
        load_btn = QPushButton("Load pipeline")
        load_btn.clicked.connect(self.load_requested)
        config_row.addWidget(save_btn)
        config_row.addWidget(load_btn)
        lay.addLayout(config_row)

        hint = QLabel("Double-click a step to edit · right-click / Del to remove")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def steps(self) -> list[PreprocStep]:
        return self._queue.steps()

    def set_steps(self, steps: list[PreprocStep]):
        self._queue.set_steps(steps)

    def _populate_palette(self):
        current_group = None
        for spec in OPERATOR_LIST:
            if spec.group != current_group:
                current_group = spec.group
                header = QListWidgetItem(spec.group)
                header.setFlags(Qt.NoItemFlags)
                f = QFont()
                f.setBold(True)
                header.setFont(f)
                self._palette.addItem(header)
            item = QListWidgetItem(f"   {spec.label}")
            item.setData(Qt.UserRole, spec.op_id)
            self._palette.addItem(item)

    def _show_operator_description(self, item: QListWidgetItem):
        op_id = item.data(Qt.UserRole)
        if op_id is None:  # group header — nothing to describe
            return
        spec = OPERATORS[op_id]
        DescriptionDialog(spec, OPERATOR_DESCRIPTIONS[op_id], self).exec_()

    @staticmethod
    def _bold_label(text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setBold(True)
        lbl.setFont(f)
        return lbl
