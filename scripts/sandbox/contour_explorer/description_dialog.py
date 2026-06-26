"""Operator description popup: read-only maths/behaviour explanation."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

from .operator_registry import OperatorSpec


class DescriptionDialog(QDialog):
    """Read-only popup explaining what an operator does and the maths behind it."""

    def __init__(self, spec: OperatorSpec, body_html: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About — {spec.label}")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(580, 540)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(
            f"<p style='color:gray; margin:0'><i>{spec.group}</i></p>{body_html}"
        )
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
