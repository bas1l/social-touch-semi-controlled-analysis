"""Contract tests for the DAG event vocabulary and its one-line wire codec.

These pin the producer/consumer boundary: whatever the child process encodes,
the GUI must decode back to an equal event, and anything malformed must raise
rather than be silently dropped.
"""

import json

import pytest

from analysis.pipeline.execution_events import (
    SENTINEL_PREFIX,
    ConsoleLine,
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
    decode_event,
    encode_event,
)


def _round_trip(event):
    """Encode *event*, strip the sentinel prefix, decode it back."""
    line = encode_event(event)
    assert line.startswith(SENTINEL_PREFIX)
    return decode_event(line[len(SENTINEL_PREFIX):])


@pytest.mark.parametrize("status", list(TaskStatus))
def test_every_status_round_trips(status):
    event = TaskFinished(name="some_task", status=status, message="why")
    decoded = _round_trip(event)
    assert decoded == event
    assert decoded.status is status


@pytest.mark.parametrize(
    "event",
    [
        ConsoleLine(text="plain narration"),
        TaskStarted(idx=3, total=35, name="spatial_extract_boundaries"),
        TaskFinished(name="touch_prepare_sessions", status=TaskStatus.COMPLETED),
        TaskFinished(name="a", status=TaskStatus.FAILED, message="boom: ValueError"),
        RunFinished(aborted=False, failed=("a", "b"), skipped=()),
        RunFinished(aborted=True, failed=(), skipped=("c",)),
    ],
)
def test_each_event_round_trips_with_fields_intact(event):
    decoded = _round_trip(event)
    assert decoded == event
    assert type(decoded) is type(event)


def test_tuple_fields_come_back_as_tuples_not_lists():
    decoded = _round_trip(RunFinished(aborted=False, failed=("a",), skipped=("b", "c")))
    assert decoded.failed == ("a",)
    assert decoded.skipped == ("b", "c")
    assert isinstance(decoded.failed, tuple)


@pytest.mark.parametrize(
    "event",
    [
        ConsoleLine(text="line one\nline two\r\nline three"),
        TaskFinished(name="a", status=TaskStatus.FAILED, message="trace:\n  at x\n  at y"),
        TaskStarted(idx=1, total=1, name="tab\tand\nnewline"),
    ],
)
def test_an_encoded_event_is_exactly_one_line(event):
    line = encode_event(event)
    assert "\n" not in line
    assert "\r" not in line
    assert len(line.splitlines()) == 1
    # ...and the embedded line breaks survive the round trip as data.
    assert _round_trip(event) == event


def test_non_ascii_payload_round_trips():
    event = TaskFinished(name="a", status=TaskStatus.FAILED, message="échec ❌")
    assert _round_trip(event) == event


def test_encode_rejects_a_non_event():
    with pytest.raises(TypeError, match="not a DAG event"):
        encode_event({"event": "TaskStarted"})


def test_decode_raises_on_truncated_json():
    line = encode_event(TaskStarted(idx=1, total=2, name="a"))
    payload = line[len(SENTINEL_PREFIX):][:-3]
    with pytest.raises(ValueError, match="malformed event payload"):
        decode_event(payload)


def test_decode_raises_on_a_non_object_payload():
    with pytest.raises(ValueError, match="must be a JSON object"):
        decode_event("[1, 2, 3]")


def test_decode_raises_when_the_discriminator_is_absent():
    with pytest.raises(ValueError, match="has no 'event' key"):
        decode_event(json.dumps({"name": "a"}))


def test_decode_raises_on_an_unknown_event_type():
    with pytest.raises(ValueError, match="unknown event type 'TaskExploded'"):
        decode_event(json.dumps({"event": "TaskExploded", "name": "a"}))


def test_decode_raises_on_an_unknown_status_string():
    payload = json.dumps(
        {"event": "TaskFinished", "name": "a", "status": "mostly_fine", "message": ""}
    )
    with pytest.raises(ValueError, match="unknown task status 'mostly_fine'"):
        decode_event(payload)


def test_decode_raises_on_a_missing_field():
    payload = json.dumps({"event": "TaskStarted", "idx": 1, "name": "a"})
    with pytest.raises(ValueError, match=r"missing field\(s\): \['total'\]"):
        decode_event(payload)


def test_decode_raises_on_an_unexpected_field():
    payload = json.dumps(
        {"event": "ConsoleLine", "text": "hi", "colour": "red"}
    )
    with pytest.raises(ValueError, match=r"unexpected field\(s\): \['colour'\]"):
        decode_event(payload)


def test_decode_raises_on_a_wrong_field_type():
    payload = json.dumps({"event": "TaskStarted", "idx": "one", "total": 2, "name": "a"})
    with pytest.raises(ValueError, match="TaskStarted.idx: expected an integer"):
        decode_event(payload)


def test_decode_rejects_a_bool_where_an_int_is_declared():
    payload = json.dumps({"event": "TaskStarted", "idx": True, "total": 2, "name": "a"})
    with pytest.raises(ValueError, match="TaskStarted.idx: expected an integer"):
        decode_event(payload)


def test_events_are_frozen():
    event = TaskFinished(name="a", status=TaskStatus.COMPLETED)
    with pytest.raises(Exception):
        event.name = "b"


def test_sentinel_prefix_is_stable():
    # The GUI parser anchors on this exact string; changing it is a protocol break.
    assert SENTINEL_PREFIX == "##DAG-EVENT "
