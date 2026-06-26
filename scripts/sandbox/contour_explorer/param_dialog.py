"""Parameter popup dialog: builds a form from an OperatorSpec's param schema."""

from __future__ import annotations

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .operator_registry import OperatorSpec, ParamSpec


class ParamDialog(QDialog):
    """Build a parameter form from an OperatorSpec's param schema."""

    def __init__(self, spec: OperatorSpec, values: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Parameters — {spec.label}")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._spec = spec
        self._widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        for p in spec.params:
            w = self._make_widget(p, values.get(p.name, p.default))
            self._widgets[p.name] = w
            form.addRow(p.name, w)
        layout.addLayout(form)

        self._error = QLabel()
        self._error.setStyleSheet("color: red;")
        self._error.setVisible(False)
        layout.addWidget(self._error)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_widget(self, p: ParamSpec, value):
        if p.kind == "float":
            w = QDoubleSpinBox()
            w.setRange(p.minimum, p.maximum)
            w.setSingleStep(p.step)
            w.setDecimals(p.decimals)
            w.setValue(float(value))
            return w
        if p.kind == "int":
            w = QSpinBox()
            w.setRange(int(p.minimum), int(p.maximum))
            w.setSingleStep(int(p.step))
            w.setValue(int(value))
            return w
        if p.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            return w
        if p.kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in p.choices])
            w.setCurrentText(str(value))
            return w
        raise ValueError(f"unknown param kind: {p.kind}")

    def values(self) -> dict:
        out = {}
        for p in self._spec.params:
            w = self._widgets[p.name]
            if p.kind == "float":
                out[p.name] = float(w.value())
            elif p.kind == "int":
                out[p.name] = int(w.value())
            elif p.kind == "bool":
                out[p.name] = bool(w.isChecked())
            elif p.kind == "choice":
                out[p.name] = w.currentText()
        return out

    def _on_ok(self):
        # Validate by dry-running the operator's parameter checks on a tiny field.
        try:
            probe = np.ones((4, 4), dtype=float)
            self._spec.fn(probe, **self.values())
        except ValueError as exc:
            self._error.setText(str(exc))
            self._error.setVisible(True)
            return
        except Exception:
            # Non-parameter errors (e.g. degenerate tiny-field math) are not the
            # dialog's concern — they surface at Apply time. Accept the values.
            pass
        self.accept()
