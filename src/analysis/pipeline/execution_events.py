"""Event contract between the DAG execution core and any observer of a run.

This module is the lowest layer of the execution stack: it defines the task
status vocabulary, the immutable event records a run emits, and the
one-event-to-one-line wire encoding used to carry those events out of the
workflow child process on stdout.

It knows nothing about Qt, subprocesses, Prefect, the filesystem or YAML — it
is pure data plus a codec, so both the producer (``dag_execution``) and the
consumer (the GUI's status channel) can depend on it without depending on each
other.

Wire format
-----------
``encode_event`` renders exactly one line::

    ##DAG-EVENT {"event": "TaskFinished", "name": "...", "status": "failed", ...}

The sentinel prefix must anchor at position 0 of a line for a consumer to treat
it as an event.  ``decode_event`` takes the JSON *payload* — the text after the
prefix — and raises ``ValueError`` on anything it cannot decode exactly.  A
malformed sentinel is a producer bug and is never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, Dict, Tuple, Type


class TaskStatus(Enum):
    """The status vocabulary of a single DAG task within one run.

    ``PENDING`` and ``RUNNING`` are transient.  Every other member is a
    *terminal* status: exactly one is reached per task per run.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BYPASSED = "bypassed"
    FAILED = "failed"
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_DEP = "skipped_dep"
    SKIPPED_GUARD = "skipped_guard"
    SKIPPED_UNREGISTERED = "skipped_unregistered"
    ABORTED = "aborted"


#: Statuses that mean "this task was deliberately not run for a declared
#: reason".  None of them is a failure, so none of them affects the exit code.
SKIPPED_STATUSES: Tuple[TaskStatus, ...] = (
    TaskStatus.SKIPPED_DISABLED,
    TaskStatus.SKIPPED_DEP,
    TaskStatus.SKIPPED_GUARD,
    TaskStatus.SKIPPED_UNREGISTERED,
)

#: Prefix that marks a stdout line as a structured event.  Must anchor at
#: position 0 of the line; a line that merely contains it is ordinary output.
SENTINEL_PREFIX = "##DAG-EVENT "


@dataclass(frozen=True)
class ConsoleLine:
    """A line of human-readable narration produced by the execution core."""

    text: str


@dataclass(frozen=True)
class TaskStarted:
    """A task passed every gate and its work is about to be invoked."""

    idx: int
    total: int
    name: str


@dataclass(frozen=True)
class TaskFinished:
    """A task reached a terminal status.  Emitted exactly once per task."""

    name: str
    status: TaskStatus
    message: str = ""


@dataclass(frozen=True)
class RunFinished:
    """The run ended.  ``failed`` and ``skipped`` are in execution order."""

    aborted: bool
    failed: Tuple[str, ...]
    skipped: Tuple[str, ...]


_EVENT_TYPES: Dict[str, Type[Any]] = {
    cls.__name__: cls for cls in (ConsoleLine, TaskStarted, TaskFinished, RunFinished)
}

_EVENT_KEY = "event"


# --- field codecs ------------------------------------------------------------
#
# Decoding is table-driven and strict: every field of the target dataclass must
# be present in the payload, no extra key is tolerated, and each value is
# converted by the codec declared for it.  A module-level consistency check at
# the bottom of this file asserts the table and the dataclasses cannot drift.


def _to_str(field_path: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_path}: expected a string, got {type(value).__name__}")
    return value


def _to_int(field_path: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_path}: expected an integer, got {type(value).__name__}")
    return value


def _to_bool(field_path: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_path}: expected a boolean, got {type(value).__name__}")
    return value


def _to_status(field_path: str, value: Any) -> TaskStatus:
    text = _to_str(field_path, value)
    try:
        return TaskStatus(text)
    except ValueError:
        known = ", ".join(member.value for member in TaskStatus)
        raise ValueError(
            f"{field_path}: unknown task status {text!r} (known: {known})"
        ) from None


def _to_str_tuple(field_path: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_path}: expected a list, got {type(value).__name__}")
    return tuple(
        _to_str(f"{field_path}[{index}]", item) for index, item in enumerate(value)
    )


_FIELD_DECODERS: Dict[str, Dict[str, Callable[[str, Any], Any]]] = {
    "ConsoleLine": {"text": _to_str},
    "TaskStarted": {"idx": _to_int, "total": _to_int, "name": _to_str},
    "TaskFinished": {"name": _to_str, "status": _to_status, "message": _to_str},
    "RunFinished": {
        "aborted": _to_bool,
        "failed": _to_str_tuple,
        "skipped": _to_str_tuple,
    },
}


def _encode_value(value: Any) -> Any:
    """Render one dataclass field value as a JSON-native value."""
    if isinstance(value, TaskStatus):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


def encode_event(event: object) -> str:
    """Render *event* as exactly one sentinel-prefixed line.

    Raises
    ------
    TypeError
        If *event* is not one of the four event dataclasses.
    ValueError
        If the rendered line would contain a newline or carriage return, which
        would let a consumer split one event across two lines.
    """
    type_name = type(event).__name__
    if _EVENT_TYPES.get(type_name) is not type(event):
        known = ", ".join(sorted(_EVENT_TYPES))
        raise TypeError(f"not a DAG event: {type_name} (known: {known})")

    payload: Dict[str, Any] = {_EVENT_KEY: type_name}
    for field in fields(event):
        payload[field.name] = _encode_value(getattr(event, field.name))

    line = SENTINEL_PREFIX + json.dumps(payload, ensure_ascii=True, sort_keys=False)
    if "\n" in line or "\r" in line:
        raise ValueError(f"encoded event contains a line break: {line!r}")
    return line


def decode_event(payload: str) -> object:
    """Decode the JSON *payload* of a sentinel line back into an event.

    *payload* is the text **after** ``SENTINEL_PREFIX``; stripping the prefix is
    the caller's job (see the GUI's status channel).

    Raises
    ------
    ValueError
        On malformed JSON, a non-object payload, a missing or unknown ``event``
        discriminator, a missing or unexpected field, or an unknown status
        string.  Never returns a partially populated event.
    """
    if not isinstance(payload, str):
        raise ValueError(f"event payload must be a string, got {type(payload).__name__}")

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed event payload {payload!r}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"event payload must be a JSON object, got {type(raw).__name__}: {payload!r}"
        )
    if _EVENT_KEY not in raw:
        raise ValueError(f"event payload has no {_EVENT_KEY!r} key: {payload!r}")

    type_name = raw[_EVENT_KEY]
    if type_name not in _EVENT_TYPES:
        known = ", ".join(sorted(_EVENT_TYPES))
        raise ValueError(f"unknown event type {type_name!r} (known: {known})")

    decoders = _FIELD_DECODERS[type_name]
    supplied = set(raw) - {_EVENT_KEY}
    missing = sorted(set(decoders) - supplied)
    if missing:
        raise ValueError(f"{type_name} payload is missing field(s): {missing}")
    unexpected = sorted(supplied - set(decoders))
    if unexpected:
        raise ValueError(f"{type_name} payload has unexpected field(s): {unexpected}")

    kwargs = {
        name: decode(f"{type_name}.{name}", raw[name]) for name, decode in decoders.items()
    }
    return _EVENT_TYPES[type_name](**kwargs)


def _assert_decoder_table_matches_dataclasses() -> None:
    """Fail at import if a dataclass field has no codec, or vice versa."""
    for type_name, cls in _EVENT_TYPES.items():
        declared = {field.name for field in fields(cls)}
        tabled = set(_FIELD_DECODERS[type_name])
        if declared != tabled:
            raise AssertionError(
                f"_FIELD_DECODERS[{type_name!r}] is out of sync with the dataclass: "
                f"fields only on the class {sorted(declared - tabled)}, "
                f"fields only in the table {sorted(tabled - declared)}"
            )
    orphans = sorted(set(_FIELD_DECODERS) - set(_EVENT_TYPES))
    if orphans:
        raise AssertionError(f"_FIELD_DECODERS has entries for unknown events: {orphans}")


_assert_decoder_table_matches_dataclasses()


__all__ = [
    "TaskStatus",
    "SKIPPED_STATUSES",
    "SENTINEL_PREFIX",
    "ConsoleLine",
    "TaskStarted",
    "TaskFinished",
    "RunFinished",
    "encode_event",
    "decode_event",
]
