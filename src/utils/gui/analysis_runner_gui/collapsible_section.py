"""Collapsible titled section widget for grouping task options."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """A labelled section whose body can be folded away by clicking its header.

    The header is a checkable :class:`QToolButton` showing a down/right arrow;
    the body is a content widget hosting added child widgets in a vertical
    layout. Use :meth:`add_widget` to append children.
    """

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._button.setStyleSheet(
            "QToolButton {"
            " font-weight: bold;"
            " padding: 4px 8px;"
            " background: #e8e8e8;"
            " border: none;"
            " border-bottom: 1px solid #c0c0c0;"
            " text-align: left;"
            "}"
        )
        self._button.toggled.connect(self._on_toggled)
        outer.addWidget(self._button)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 0, 4)
        self._content_layout.setSpacing(8)
        self._content.setVisible(expanded)
        outer.addWidget(self._content)

    def add_widget(self, widget: QWidget) -> None:
        """Append *widget* to the section body."""
        self._content_layout.addWidget(widget)

    def _on_toggled(self, checked: bool) -> None:
        self._button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
