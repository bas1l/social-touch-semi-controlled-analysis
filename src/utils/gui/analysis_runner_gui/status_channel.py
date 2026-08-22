"""Decode the workflow child process's structured status channel.

The workflow child writes two kinds of thing to the same stdout stream:
ordinary human-readable output, and one-line JSON *sentinels* carrying the
structured execution events defined in
:mod:`analysis.pipeline.execution_events`.  This module is the reader side of
that channel: it classifies a single already-de-newlined line as either an
event or ordinary output.

It is deliberately Qt-free and stateless — one line in, one event or ``None``
out — so the GUI's transport (a ``QThread`` today) and the graph's rendering
can both be replaced without touching the protocol, and so a *past* run log can
be replayed through the very same function.

The prefix must **anchor at position 0**.  A line that merely contains the
sentinel text — a traceback quoting it, a task echoing its own log — is
ordinary output.  Anchoring is what keeps the channel unspoofable by task
output that happens to be indented.
"""

from __future__ import annotations

from typing import Optional

from analysis.pipeline.execution_events import SENTINEL_PREFIX, decode_event


def parse_status_line(line: str) -> Optional[object]:
    """Return the event carried by *line*, or ``None`` for ordinary output.

    Parameters
    ----------
    line
        One line of child-process output, with any trailing newline already
        stripped by the transport.

    Returns
    -------
    object or None
        One of the event dataclasses of
        :mod:`analysis.pipeline.execution_events` when *line* starts with
        :data:`~analysis.pipeline.execution_events.SENTINEL_PREFIX`, otherwise
        ``None``.

    Raises
    ------
    TypeError
        If *line* is not a string.
    ValueError
        If *line* carries the sentinel prefix but its payload cannot be decoded
        exactly.  That is a producer bug — a corrupted or out-of-date event
        writer — and is never silently downgraded to ordinary output, because
        doing so would drop a task's terminal status and leave its node
        permanently pending.
    """
    if not isinstance(line, str):
        raise TypeError(f"status line must be a string, got {type(line).__name__}")
    if not line.startswith(SENTINEL_PREFIX):
        return None
    return decode_event(line[len(SENTINEL_PREFIX) :])


__all__ = ["parse_status_line"]
