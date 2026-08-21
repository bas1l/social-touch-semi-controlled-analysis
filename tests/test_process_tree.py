"""Aborting a run must leave nothing behind — and must not kill the aborter.

The reference implementation this was ported from calls ``os.killpg`` on a
child it never detached from its own process group, so on POSIX it would have
signalled the GUI itself.  The grandchild test below is the regression guard
for exactly that defect: it asserts both halves — the tree dies *and* this
process survives.

No ``QApplication`` is constructed here; :mod:`process_tree` is Qt-free.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from utils.gui.analysis_runner_gui.process_tree import (
    kill_process_tree,
    popen_group_kwargs,
)

ON_WINDOWS = os.name == "nt"

# A child that spawns a detached grandchild, announces the grandchild's PID on
# stdout, and then blocks.  Killing only the direct child leaves the grandchild
# behind; killing the group reaps both.
_CHILD_SOURCE = """
import subprocess
import sys
import time

grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(grandchild.pid, flush=True)
time.sleep(120)
"""


def test_popen_group_kwargs_returns_the_platform_correct_key():
    kwargs = popen_group_kwargs()
    if ON_WINDOWS:
        assert kwargs == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        assert kwargs == {"start_new_session": True}


def test_popen_group_kwargs_is_never_empty():
    """An empty mapping would silently create a tree that cannot be killed."""
    assert popen_group_kwargs()


def test_kill_process_tree_on_an_exited_process_is_a_no_op(monkeypatch):
    proc = subprocess.Popen([sys.executable, "-c", ""], **popen_group_kwargs())
    proc.wait()
    returncode_before = proc.returncode

    def _fail(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("kill_process_tree acted on an already-exited process")

    monkeypatch.setattr(subprocess, "run", _fail)
    if not ON_WINDOWS:
        monkeypatch.setattr(os, "killpg", _fail)

    kill_process_tree(proc)

    assert proc.returncode == returncode_before


def test_kill_process_tree_rejects_a_non_positive_grace_period():
    proc = subprocess.Popen([sys.executable, "-c", ""], **popen_group_kwargs())
    proc.wait()
    with pytest.raises(ValueError, match="grace_seconds"):
        kill_process_tree(proc, grace_seconds=0)


@pytest.mark.skipif(ON_WINDOWS, reason="process groups and killpg are POSIX-only")
def test_kill_process_tree_reaps_a_grandchild_without_killing_this_process():
    received_sigterm = []
    previous_handler = signal.signal(
        signal.SIGTERM, lambda *_: received_sigterm.append(True)
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SOURCE],
        stdout=subprocess.PIPE,
        text=True,
        **popen_group_kwargs(),
    )
    try:
        grandchild_pid = int(proc.stdout.readline().strip())
        assert _process_is_alive(grandchild_pid)

        kill_process_tree(proc, grace_seconds=5.0)

        assert proc.poll() is not None, "the direct child survived the tree kill"
        assert _wait_until_gone(grandchild_pid, timeout_seconds=5.0), (
            "the grandchild outlived the tree kill"
        )
        assert not received_sigterm, (
            "kill_process_tree signalled this process's own group"
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        if proc.poll() is None:  # pragma: no cover - only on a failed kill
            proc.kill()
            proc.wait()
        proc.stdout.close()


@pytest.mark.skipif(ON_WINDOWS, reason="process groups are POSIX-only")
def test_kill_process_tree_refuses_a_child_sharing_this_process_group():
    """The reference implementation's defect, made loud rather than fatal."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        with pytest.raises(RuntimeError, match="popen_group_kwargs"):
            kill_process_tree(proc)
    finally:
        proc.kill()
        proc.wait()


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until_gone(pid: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_is_alive(pid):
            return True
        time.sleep(0.05)
    return False
