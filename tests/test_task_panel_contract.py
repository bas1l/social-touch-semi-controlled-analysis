"""Contract tests for the centre column (``task_panel``).

Phase 6 moved the per-task options editor out of the centre column and into a
tab in the right-hand column.  What makes that a *contract* rather than a
rearrangement is that the centre column no longer knows a detail panel exists:
it reports which task is selected and stops there.

Widget placement itself cannot be asserted headlessly — this repo constructs no
``QApplication`` in tests and has no ``pytest-qt`` — so the contract is pinned
where it is observable: the module's import graph and its declared signals.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from utils.gui.analysis_runner_gui import task_panel as tp  # noqa: E402
from utils.gui.analysis_runner_gui.task_panel import TaskPanel  # noqa: E402

_FORBIDDEN_MODULE = "task_detail_panel"


def _imported_module_names(source_path: Path) -> set[str]:
    """Every module name ``source_path`` imports, in dotted form."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # ``node.module`` is None for a bare relative import (``from . import x``).
            module = node.module if node.module is not None else ""
            names.add(module)
            names.update(f"{module}.{alias.name}".lstrip(".") for alias in node.names)
    return names


def test_task_panel_does_not_import_task_detail_panel() -> None:
    """The centre column must not depend on where options are rendered."""
    source_path = Path(tp.__file__)
    offenders = sorted(
        name
        for name in _imported_module_names(source_path)
        if _FORBIDDEN_MODULE in name.split(".")
    )
    assert not offenders, (
        f"{source_path.name} imports {offenders} — the centre column must stay "
        "ignorant of the detail panel; report the selection via task_selected "
        "and let the owning window render it."
    )


def test_task_panel_declares_task_selected_signal() -> None:
    """``task_selected`` is the replacement contract for the removed call."""
    signal = getattr(TaskPanel, "task_selected", None)
    assert signal is not None, "TaskPanel must declare a task_selected signal"
    assert isinstance(signal, tp.pyqtSignal), (
        f"TaskPanel.task_selected must be a pyqtSignal, got {type(signal)!r}"
    )


def test_task_selected_carries_only_the_task_name() -> None:
    """The signal must not smuggle the model across the column boundary.

    ``runner_window`` already owns the ``DagConfigModel``; a second copy
    arriving over the signal would make the centre column a competing source of
    truth for which config is loaded.
    """
    source_path = Path(tp.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
    emits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "emit"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "task_selected"
    ]
    assert emits, "task_selected is declared but never emitted"
    for call in emits:
        assert len(call.args) == 1, (
            "task_selected must be emitted with exactly the task name "
            f"(line {call.lineno} passes {len(call.args)} arguments)"
        )
