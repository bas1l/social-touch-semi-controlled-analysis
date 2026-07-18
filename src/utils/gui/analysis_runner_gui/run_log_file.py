"""Per-run log file that mirrors the GUI console, including tqdm overwrites.

The console widget shows a live view capped at a fixed number of lines; this
class persists the *complete* stdout/stderr stream of a single pipeline run to
disk.  Carriage-return (tqdm-style) updates are written with a leading ``\\r``
and no trailing newline so a terminal pager collapses successive progress
frames exactly as the live console does.
"""

from __future__ import annotations

from pathlib import Path


class RunLogFile:
    """Append-only text log for one pipeline run.

    The file is opened on construction and must be closed with :meth:`close`.
    """

    def __init__(self, path: Path, header: str) -> None:
        self._path = path
        self._handle = path.open("w", encoding="utf-8")
        self._at_cr = False
        self._handle.write(header)
        if not header.endswith("\n"):
            self._handle.write("\n")
        self._handle.flush()

    @property
    def path(self) -> Path:
        return self._path

    def write_line(self, text: str) -> None:
        """Write a full line, terminating any pending carriage-return frame."""
        if self._at_cr:
            self._handle.write("\n")
            self._at_cr = False
        self._handle.write(text + "\n")
        self._handle.flush()

    def write_cr(self, text: str) -> None:
        """Write a carriage-return progress frame (no trailing newline)."""
        self._handle.write("\r" + text)
        self._at_cr = True
        self._handle.flush()

    def close(self) -> None:
        """Terminate any pending frame and close the file handle."""
        if self._handle.closed:
            return
        if self._at_cr:
            self._handle.write("\n")
            self._at_cr = False
        self._handle.close()
