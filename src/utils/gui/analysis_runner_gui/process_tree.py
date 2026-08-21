"""Create and destroy a workflow child process together with its descendants.

The GUI spawns exactly one child — the workflow script — but that child runs a
Prefect DAG and spawns flow subprocesses of its own.  ``Popen.terminate()``
signals only the direct child, so aborting a run used to leave grandchildren
running with no owner and no console.

Killing a whole tree needs cooperation at *creation* time on POSIX: a signal
can only be addressed to a process **group**, and a child inherits its parent's
group unless it is explicitly detached.  :func:`popen_group_kwargs` supplies
that detachment, and :func:`kill_process_tree` relies on it.  The two functions
are a pair; using the second without the first on POSIX is a bug that
:func:`kill_process_tree` refuses to commit — it would signal the GUI itself.

This module is deliberately Qt-free and DAG-free.  It knows about
:class:`subprocess.Popen` objects and about platforms; it knows nothing about
widgets, tasks, or run semantics.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)

#: How long a tree is given to die from the polite signal before the forceful
#: one is sent (POSIX) or before the tree kill is declared failed (Windows).
DEFAULT_GRACE_SECONDS = 5.0

_ON_WINDOWS = os.name == "nt"


def popen_group_kwargs() -> dict[str, object]:
    """Return the :class:`subprocess.Popen` kwargs that isolate a child's tree.

    On Windows the child becomes the root of a new *process group*; on POSIX it
    becomes the leader of a new *session* (and therefore of a new process
    group) via ``setsid``.  Either way the child and everything it spawns can
    afterwards be addressed as a unit, without the addressing reaching back to
    this process.

    Returns
    -------
    dict
        Keyword arguments to splat into a ``Popen`` call.  Never empty — a
        caller that receives ``{}`` would silently create an un-killable tree.
    """
    if _ON_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(
    proc: subprocess.Popen, grace_seconds: float = DEFAULT_GRACE_SECONDS
) -> None:
    """Kill *proc* and every process descended from it.

    Parameters
    ----------
    proc
        A process created with ``**popen_group_kwargs()``.  Passing a process
        created without them is a programming error and raises on POSIX.
    grace_seconds
        Seconds allowed for the tree to exit before escalating (POSIX) or
        before the tree kill is reported as failed (Windows).

    Notes
    -----
    Returns immediately when *proc* has already exited, so an abort racing a
    normally-finishing run is a no-op rather than a signal to a recycled PID.
    """
    if grace_seconds <= 0:
        raise ValueError(f"grace_seconds must be positive, got {grace_seconds!r}")
    if proc.poll() is not None:
        return
    if _ON_WINDOWS:
        _kill_tree_windows(proc, grace_seconds)
    else:
        _kill_tree_posix(proc, grace_seconds)


def _kill_tree_windows(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Kill the tree with ``taskkill /F /T``, which walks the parent-PID chain.

    ``taskkill`` output is captured rather than inherited: the GUI's console
    widget mirrors the child's stdout, and a stray "SUCCESS: ..." line would
    read as workflow output.  ``CREATE_NO_WINDOW`` keeps a console from
    flashing over the GUI, matching how the Prefect server is spawned.
    """
    completed = subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if completed.returncode != 0:
        # taskkill reports 128 for "no such process", i.e. the tree died
        # between our poll() and this call.  Anything else means the tree kill
        # did not happen and descendants may survive: say so loudly, then at
        # least take the direct child down so the GUI is not left waiting on a
        # process it can no longer control.
        logger.error(
            "taskkill /F /T on PID %d failed with exit code %d; descendants may "
            "survive. stdout=%r stderr=%r. Killing the direct child only.",
            proc.pid,
            completed.returncode,
            completed.stdout.strip(),
            completed.stderr.strip(),
        )
        proc.kill()
    _wait_for_exit(proc, grace_seconds)


def _kill_tree_posix(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Signal the child's whole process group, escalating after the grace period."""
    pgid = os.getpgid(proc.pid)
    if pgid == os.getpgid(0):
        raise RuntimeError(
            f"refusing to signal process group {pgid}: child PID {proc.pid} shares "
            "this process's group, so killing it would kill the GUI. The child must "
            "be created with **popen_group_kwargs()."
        )
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(
            "process group %d ignored SIGTERM for %.1fs; escalating to SIGKILL",
            pgid,
            grace_seconds,
        )
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def _wait_for_exit(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Reap *proc*, logging loudly if it outlives the grace period."""
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        logger.error(
            "PID %d was still alive %.1fs after the tree kill; it is now orphaned "
            "from the GUI's point of view.",
            proc.pid,
            grace_seconds,
        )


__all__ = ["DEFAULT_GRACE_SECONDS", "kill_process_tree", "popen_group_kwargs"]
