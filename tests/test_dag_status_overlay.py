"""The DAG node's status-overlay table must cover the status vocabulary.

``DagTaskNode.paint`` indexes ``_STATUS_OVERLAY`` directly, so a ``TaskStatus``
member missing from the table is not a mild degradation: it is a ``KeyError``
raised inside a Qt paint event, in the GUI, on whichever status nobody happened
to exercise.  These tests are the exhaustiveness proof, and they cost nothing —
no ``QApplication``, no widget, module-level data only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5")

from analysis.pipeline.execution_events import TaskStatus  # noqa: E402
from utils.gui.analysis_runner_gui import dag_graph_items  # noqa: E402

_OVERLAY = dag_graph_items._STATUS_OVERLAY


def test_overlay_covers_every_task_status() -> None:
    missing = sorted(member.name for member in TaskStatus if member not in _OVERLAY)
    assert not missing, (
        "TaskStatus member(s) with no _STATUS_OVERLAY entry — DagTaskNode.paint "
        f"would raise KeyError on: {missing}"
    )


def test_overlay_has_no_keys_outside_task_status() -> None:
    strays = [key for key in _OVERLAY if not isinstance(key, TaskStatus)]
    assert not strays, f"_STATUS_OVERLAY has non-TaskStatus key(s): {strays}"


@pytest.mark.parametrize("status", list(TaskStatus), ids=lambda s: s.name)
def test_overlay_rows_have_the_declared_shape(status: TaskStatus) -> None:
    """Each row is ``(fill | None, border | None, width, dim, glyph)``."""
    from PyQt5.QtGui import QColor

    fill, border, width, dim, glyph = _OVERLAY[status]

    assert fill is None or isinstance(fill, QColor)
    assert border is None or isinstance(border, QColor)
    assert isinstance(width, float) and width > 0
    assert isinstance(dim, bool)
    assert isinstance(glyph, str)


def test_pending_is_the_neutral_row() -> None:
    """PENDING must not override anything: an untouched graph looks untouched."""
    fill, border, _width, dim, glyph = _OVERLAY[TaskStatus.PENDING]

    assert fill is None
    assert border is None
    assert dim is False
    assert glyph == ""


def test_bypassed_is_visually_distinct_from_completed_and_undimmed() -> None:
    """A bypassed task ran nothing; the report must never paint it as done.

    Violet and undimmed: neither green (which would claim work happened) nor
    dimmed (which would read as skipped, though it does unlock its dependents).
    """
    by_fill, by_border, _bw, by_dim, by_glyph = _OVERLAY[TaskStatus.BYPASSED]
    ok_fill, ok_border, _ow, _ok_dim, ok_glyph = _OVERLAY[TaskStatus.COMPLETED]

    assert by_fill != ok_fill
    assert by_border != ok_border
    assert by_glyph != ok_glyph
    assert by_dim is False, "a bypassed node is accounted for, not skipped"
    assert by_fill is not None and by_border is not None, (
        "BYPASSED must override the category colours, or it would be "
        "indistinguishable from a task that has not run yet"
    )


def test_bounding_margin_clears_the_widest_status_border() -> None:
    """The 3 px RUNNING stroke straddles the rect edge and must not be clipped."""
    assert dag_graph_items._MAX_STATUS_BORDER_W == max(
        width for _, _, width, _, _ in _OVERLAY.values()
    )
    assert dag_graph_items._MAX_STATUS_BORDER_W >= 3.0
