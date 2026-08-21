"""The GUI's reader side of the stdout status channel.

``parse_status_line`` is the single point at which a line of child-process
output becomes either a structured event or console text.  Its two failure
modes are opposite and both damaging: treating ordinary output as an event
(a task's log line hijacking the graph) and treating an event as ordinary
output (a node stuck pending for the rest of the run).  These tests pin both
edges, plus the round-trip that keeps the writer and the reader honest.
"""

from __future__ import annotations

import pytest

from analysis.pipeline.execution_events import (
    SENTINEL_PREFIX,
    ConsoleLine,
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
    encode_event,
)
from utils.gui.analysis_runner_gui.status_channel import parse_status_line


# ----------------------------------------------------------------------
# Ordinary output
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "",
        "Loading session data …",
        "✅ Task 'kinect_extract' marked as completed.",
        "  0%|          | 0/42 [00:00<?, ?it/s]",
        "{\"event\": \"TaskFinished\"}",  # JSON without the prefix
    ],
)
def test_ordinary_line_is_not_an_event(line: str) -> None:
    assert parse_status_line(line) is None


@pytest.mark.parametrize(
    "line",
    [
        f"  {SENTINEL_PREFIX}" + '{"event": "ConsoleLine", "text": "x"}',
        f"the child printed {SENTINEL_PREFIX}" + '{"event": "ConsoleLine", "text": "x"}',
        f"ERROR parsing {SENTINEL_PREFIX}...",
    ],
)
def test_prefix_must_anchor_at_position_zero(line: str) -> None:
    """A line that merely *contains* the sentinel is ordinary output.

    Anchoring is what stops a traceback, an indented log line, or a task
    echoing its own output from injecting statuses into the graph.
    """
    assert parse_status_line(line) is None


# ----------------------------------------------------------------------
# Valid sentinels
# ----------------------------------------------------------------------


def test_valid_sentinel_returns_the_event_with_fields_intact() -> None:
    line = SENTINEL_PREFIX + (
        '{"event": "TaskFinished", "name": "kinect_extract", '
        '"status": "skipped_dep", "message": "waiting on kinect_load"}'
    )

    event = parse_status_line(line)

    assert isinstance(event, TaskFinished)
    assert event.name == "kinect_extract"
    assert event.status is TaskStatus.SKIPPED_DEP
    assert event.message == "waiting on kinect_load"


# ----------------------------------------------------------------------
# Malformed sentinels
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        "",  # nothing at all
        '{"event": "TaskFinished", "name": "a"',  # truncated JSON
        '{"event": "Nonsense", "name": "a"}',  # unknown event type
        '{"event": "TaskFinished", "name": "a", "status": "gremlin", "message": ""}',
        '{"event": "TaskFinished", "name": "a"}',  # missing fields
        '{"name": "a"}',  # no discriminator
        "[1, 2, 3]",  # not an object
    ],
)
def test_malformed_sentinel_raises_value_error(payload: str) -> None:
    """A corrupt sentinel is a producer bug and is never silently dropped."""
    with pytest.raises(ValueError):
        parse_status_line(SENTINEL_PREFIX + payload)


def test_non_string_line_raises_type_error() -> None:
    with pytest.raises(TypeError):
        parse_status_line(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Writer/reader round trip
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        ConsoleLine(text="plain narration"),
        TaskStarted(idx=3, total=35, name="rf_extract"),
        *[
            TaskFinished(name="rf_extract", status=status, message=f"m-{status.value}")
            for status in TaskStatus
        ],
        RunFinished(aborted=False, failed=(), skipped=()),
        RunFinished(aborted=True, failed=("a", "b"), skipped=("c",)),
    ],
    ids=lambda event: type(event).__name__
    + (f"-{event.status.value}" if isinstance(event, TaskFinished) else ""),
)
def test_encoded_event_parses_back_to_an_equal_event(event: object) -> None:
    """Pins the producer and the consumer together across the process boundary."""
    assert parse_status_line(encode_event(event)) == event
