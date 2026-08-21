"""Qt-free DAG execution core: walk a :class:`DagPlan`, report every outcome.

``execute_dag`` is the whole feature.  It owns the *ladder* — the ordered set of
conditions that decide what happens to a task — and guarantees that every task
it visits reaches exactly one terminal status and emits exactly one
:class:`TaskFinished`.

The ladder, evaluated per task in this order, stopping at the first match:

===  ====================================  ========================  ===============
 #   Condition                              Status                    Marks completed
===  ====================================  ========================  ===============
 0   ``is_aborted()``                       ``ABORTED``               no
 1   name absent from the DAG registry      ``SKIPPED_UNREGISTERED``  no
 2   ``not enabled``                        ``SKIPPED_DISABLED``      no
 3   ``bypass``                             ``BYPASSED``              **yes**
 4   a dependency did not complete          ``SKIPPED_DEP``           no
 5   ``min_inputs``/``max_inputs`` violated  ``SKIPPED_GUARD``        no
 6   ran; returned / raised                 ``COMPLETED`` / ``FAILED`` yes / no
===  ====================================  ========================  ===============

Row order carries meaning.  ``enabled`` stays the master switch (2 before 3), so
a disabled task's ``bypass`` is inert.  A bypass asserts something about *the
disk*, not about this run, so it is immune to upstream failure and to the input
guard (3 before 4 and 5).  Abort beats everything (0 first).

**No filesystem access occurs on the bypass path.**  Row 3 marks the plan and
emits an event; it touches the disk in no way whatsoever.  That is a reviewable
property of this file: it imports nothing that can reach the filesystem, so the
bypass branch provably cannot check whether the outputs it vouches for exist.

**Failure isolation is preserved.**  Row 6 records ``FAILED`` and moves on; the
task's dependents cascade to ``SKIPPED_DEP`` at row 4 and unrelated branches
still run.  What this module adds is that the failure is now *reported* — in the
event stream, in the returned :class:`RunOutcome`, and in its exit code.

The core depends only on the event contract.  ``run_task``, ``emit`` and
``is_aborted`` are injected, so the whole ladder is testable with plain
duck-typed recorders: no Prefect, no subprocess, no Qt, no stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .dag_plan import DagPlan
from .execution_events import (
    SKIPPED_STATUSES,
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
)


@dataclass(frozen=True)
class RunOutcome:
    """The immutable verdict of one run: what every visited task ended as."""

    statuses: Mapping[str, TaskStatus]
    aborted: bool

    @property
    def exit_code(self) -> int:
        """1 if and only if at least one task failed, else 0.

        A task skipped for a declared reason — disabled, dependency, guard,
        bypass — is not a failure and does not colour the exit code.  An abort
        is a user decision, not a failure, and likewise does not.
        """
        return 1 if self.failed_tasks else 0

    @property
    def failed_tasks(self) -> Tuple[str, ...]:
        return self._names_with(TaskStatus.FAILED)

    @property
    def completed_tasks(self) -> Tuple[str, ...]:
        return self._names_with(TaskStatus.COMPLETED)

    @property
    def bypassed_tasks(self) -> Tuple[str, ...]:
        return self._names_with(TaskStatus.BYPASSED)

    @property
    def skipped_tasks(self) -> Tuple[str, ...]:
        """Names skipped for any declared reason, in execution order."""
        return tuple(
            name for name, status in self.statuses.items() if status in SKIPPED_STATUSES
        )

    @property
    def status_counts(self) -> Mapping[TaskStatus, int]:
        """How many tasks ended in each status that occurred at all."""
        counts: Dict[TaskStatus, int] = {}
        for status in self.statuses.values():
            counts[status] = counts.get(status, 0) + 1
        return MappingProxyType(counts)

    def _names_with(self, status: TaskStatus) -> Tuple[str, ...]:
        return tuple(name for name, value in self.statuses.items() if value is status)


def _never_aborted() -> bool:
    return False


def _execution_sequence(
    plan: DagPlan, task_names: Optional[Sequence[str]]
) -> Tuple[str, ...]:
    """Order the names to visit: registered ones topologically, then the rest.

    Passing ``None`` runs the whole plan.  Passing an explicit sequence — the
    adapter passes its stage-descriptor names — runs that subset, still in the
    plan's dependency order rather than in the order the caller happened to list
    them, so a mis-ordered stage list cannot silently stall the DAG.  Names with
    no DAG entry cannot be ordered topologically and are visited last, where
    they are reported as ``SKIPPED_UNREGISTERED``.
    """
    if task_names is None:
        return plan.order

    requested = tuple(task_names)
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"task_names lists duplicate task(s): {duplicates}")

    registered = tuple(name for name in plan.order if name in set(requested))
    unregistered = tuple(name for name in requested if name not in plan.tasks)
    return registered + unregistered


def execute_dag(
    plan: DagPlan,
    run_task: Callable[[str], int],
    emit: Callable[[object], None],
    is_aborted: Callable[[], bool] = _never_aborted,
    *,
    input_count: Optional[int] = None,
    task_names: Optional[Sequence[str]] = None,
) -> RunOutcome:
    """Walk *plan* through the ladder and return the run's verdict.

    Parameters
    ----------
    plan:
        The validated DAG and the run state it accumulates.
    run_task:
        Called as ``run_task(name)`` for a task that passed every gate.  Returns
        an integer result: ``0`` is success, anything else is a failure.  An
        exception propagating out of it is caught here and recorded as
        ``FAILED`` — the run continues either way.
    emit:
        Called with each event in the order it occurs: ``TaskStarted`` for a task
        that is actually invoked, one ``TaskFinished`` per visited task, and one
        final ``RunFinished``.
    is_aborted:
        Polled once at the top of every task, and again after a task returns.
        Once it is true, everything remaining is ``ABORTED``.
    input_count:
        Number of discovered input items, against which ``min_inputs`` /
        ``max_inputs`` are checked.  ``None`` means the caller has no count to
        offer; a task that declares a guard then raises rather than being
        silently waved through.
    task_names:
        Subset of task names to visit.  ``None`` visits the whole plan.  See
        :func:`_execution_sequence` for the ordering rule.
    """
    sequence = _execution_sequence(plan, task_names)
    total = len(sequence)
    statuses: Dict[str, TaskStatus] = {}

    def finish(name: str, status: TaskStatus, message: str = "") -> None:
        """Record the one terminal status of *name* and announce it."""
        if name in plan.tasks:
            if status is TaskStatus.COMPLETED:
                plan.mark_completed(name)
            elif status is TaskStatus.BYPASSED:
                plan.mark_bypassed(name)
            else:
                plan.set_status(name, status)
        statuses[name] = status
        emit(TaskFinished(name=name, status=status, message=message))

    for index, name in enumerate(sequence, start=1):
        # Row 0 — abort beats every other consideration.
        if is_aborted():
            finish(name, TaskStatus.ABORTED, "run aborted before this task started")
            continue

        # Row 1 — a stage descriptor the DAG config knows nothing about.
        if name not in plan.tasks:
            finish(
                name,
                TaskStatus.SKIPPED_UNREGISTERED,
                f"task '{name}' is not declared in the DAG config",
            )
            continue

        task = plan.task(name)

        # Row 2 — enabled is the master switch, so a disabled task's bypass is inert.
        if not task.enabled:
            finish(name, TaskStatus.SKIPPED_DISABLED, f"task '{name}' is disabled")
            continue

        # Row 3 — bypass asserts something about the disk, not about this run:
        # nothing is invoked and nothing is verified, but dependents unblock.
        if task.bypass:
            finish(
                name,
                TaskStatus.BYPASSED,
                f"task '{name}' bypassed — nothing ran and nothing was verified",
            )
            continue

        # Row 4 — dependency gating, naming what blocked.
        unmet = plan.unmet_dependencies(name)
        if unmet:
            finish(
                name,
                TaskStatus.SKIPPED_DEP,
                f"task '{name}' skipped — unmet dependenc{'y' if len(unmet) == 1 else 'ies'}: "
                + ", ".join(unmet),
            )
            continue

        # Row 5 — input-count guard.
        if task.has_input_guard:
            if input_count is None:
                raise ValueError(
                    f"task '{name}' declares an input guard but execute_dag was "
                    "given no input_count to check it against"
                )
            violation = plan.guard_violation(name, input_count)
            if violation:
                finish(name, TaskStatus.SKIPPED_GUARD, violation)
                continue

        # Row 6 — run it.
        plan.set_status(name, TaskStatus.RUNNING)
        emit(TaskStarted(idx=index, total=total, name=name))
        try:
            result = run_task(name)
        except Exception as exc:  # noqa: BLE001 — the declared per-task error boundary
            finish(name, TaskStatus.FAILED, f"Task '{name}' failed: {exc}")
            continue

        if is_aborted():
            finish(name, TaskStatus.ABORTED, f"task '{name}' was aborted while running")
        elif result == 0:
            finish(name, TaskStatus.COMPLETED)
        else:
            finish(name, TaskStatus.FAILED, f"Task '{name}' returned result {result!r}")

    outcome = RunOutcome(
        statuses=MappingProxyType(dict(statuses)), aborted=bool(is_aborted())
    )
    emit(
        RunFinished(
            aborted=outcome.aborted,
            failed=outcome.failed_tasks,
            skipped=outcome.skipped_tasks,
        )
    )
    return outcome


def format_run_summary(outcome: RunOutcome) -> str:
    """Render *outcome* as a human-readable end-of-run block.

    A pure function of the DTO — it writes nothing.  Both workflow entry scripts
    print the result, so the two runs report identically.
    """
    lines: List[str] = ["=" * 62, "DAG RUN SUMMARY", "-" * 62]

    if not outcome.statuses:
        lines.append("  no tasks were visited")
    else:
        counts = outcome.status_counts
        for status in TaskStatus:
            if status in counts:
                lines.append(f"  {status.value:<22} {counts[status]:>3}")

    if outcome.failed_tasks:
        lines.append(f"  failed  : {', '.join(outcome.failed_tasks)}")
    if outcome.skipped_tasks:
        lines.append(f"  skipped : {', '.join(outcome.skipped_tasks)}")
    if outcome.bypassed_tasks:
        lines.append(f"  bypassed: {', '.join(outcome.bypassed_tasks)}")
    if outcome.aborted:
        lines.append("  run was ABORTED")

    lines.append("-" * 62)
    lines.append(f"exit code: {outcome.exit_code}")
    lines.append("=" * 62)
    return "\n".join(lines)


__all__ = ["RunOutcome", "execute_dag", "format_run_summary"]
