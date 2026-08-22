"""Adapter binding the workflow entry scripts to the Qt-free execution core.

``run_pipeline_stages`` is the only place where the pure core
(:mod:`analysis.pipeline.dag_execution`) meets the concrete machinery of a real
run: the stage descriptors that name the Prefect flows, the DAG config the GUI
edits, the ``PipelineMonitor`` that writes the Excel report, and stdout.  The
core stays testable because none of that leaks into it — it receives three
callables and returns a verdict.

The adapter supplies:

* **the plan** — built from the ``DagConfigHandler``'s already-loaded ``tasks``
  mapping, so there is exactly one YAML read per run and no second parser;
* **``run_task``** — opens one stage's work: lazily evaluates its ``params``
  lambda, injects the two common kwargs, and calls the flow.  An exception is
  printed with its traceback (as before) and re-raised so the core records
  ``FAILED`` with the message;
* **``emit``** — prints each event as a one-line stdout sentinel for the GUI
  *and* forwards the run/success/failure vocabulary to ``PipelineMonitor`` so
  the existing status report keeps working unchanged.

Stage descriptor schema (dict) — unchanged:
    name   : str          — DAG task name; must match the key in the DAG YAML.
    func   : callable     — the Prefect ``@flow`` function to invoke.
    params : callable     — zero-argument lambda that returns a dict of
                            task-specific kwargs.  Evaluated **lazily**: only a
                            task that actually runs may evaluate it, because
                            several lambdas raise by design when a required
                            option is absent.

What changed relative to the ``TaskExecutor`` this supersedes: a failed task is
now reported rather than swallowed (``TaskExecutor.__exit__`` returned ``True``),
so the returned :class:`RunOutcome` carries a truthful exit code.  The vendored
``PipelineDependencyError`` popup branch is not carried over: no code in this
repository raises that exception, and this module must stay Qt-free.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

from _vendor import DagConfigHandler, PipelineMonitor

from .dag_execution import RunOutcome, execute_dag
from .dag_plan import DagPlan
from .execution_events import (
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
    encode_event,
)

#: Terminal statuses that map onto the ``PipelineMonitor`` status vocabulary.
#: Statuses with no entry produce no monitor update, matching the historical
#: behaviour in which a skipped task reported nothing to the monitor at all.
_MONITOR_STATUS: Mapping[TaskStatus, str] = {
    TaskStatus.COMPLETED: "SUCCESS",
    TaskStatus.FAILED: "FAILURE",
}


def _index_stages(pipeline_stages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Key the stage descriptors by task name, rejecting malformed input."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for position, stage in enumerate(pipeline_stages):
        missing = [key for key in ("name", "func", "params") if key not in stage]
        if missing:
            raise ValueError(
                f"stage descriptor #{position} is missing key(s) {missing}: {stage!r}"
            )
        name = stage["name"]
        if name in indexed:
            raise ValueError(f"stage descriptor '{name}' is declared more than once")
        indexed[name] = stage
    return indexed


def run_pipeline_stages(
    pipeline_stages: List[Dict[str, Any]],
    dag_handler: DagConfigHandler,
    monitor: PipelineMonitor,
    items_to_process: List[Tuple[Path, Path]],
    block_id_prefix: str,
) -> RunOutcome:
    """Run every stage in *pipeline_stages* through the DAG execution core.

    Parameters
    ----------
    pipeline_stages:
        Ordered list of stage descriptors.  The order is informational only —
        execution order comes from the DAG's topology, so a mis-ordered list
        cannot stall the run.
    dag_handler:
        ``DagConfigHandler`` already loaded with the workflow's DAG YAML.  Only
        its ``tasks`` mapping and ``get_task_options`` are used.
    monitor:
        ``PipelineMonitor`` receiving RUNNING / SUCCESS / FAILURE updates.
    items_to_process:
        ``(aggregated_csv_path, database_path)`` tuples from
        ``discover_input_items``.  Passed verbatim to every flow as
        ``input_items``, and its length is the input-guard count.
    block_id_prefix:
        Prefix for the ``block_name`` reported to the monitor.  Use
        ``"batch_run_processing"`` or ``"batch_run_viewers"``.

    Returns
    -------
    RunOutcome
        The per-task statuses and the truthful ``exit_code``.
    """
    stages_by_name = _index_stages(pipeline_stages)
    plan = DagPlan.from_tasks(dag_handler.tasks)

    def block_name_of(task_name: str) -> str:
        return f"{block_id_prefix}_{task_name}"

    def run_task(task_name: str) -> int:
        stage = stages_by_name[task_name]
        options = dag_handler.get_task_options(task_name)
        kwargs: Dict[str, Any] = stage["params"]()
        kwargs["input_items"] = items_to_process
        kwargs["force_processing"] = options.get("force_processing", False)
        try:
            stage["func"](**kwargs)
        except Exception as exc:
            print(
                f"❌ Task '{task_name}' failed: {exc}\n{traceback.format_exc()}",
                flush=True,
            )
            raise
        return 0

    def emit(event: object) -> None:
        print(encode_event(event), flush=True)

        if isinstance(event, TaskStarted):
            print(
                f"[{block_name_of(event.name)}] ==> Running task: {event.name}",
                flush=True,
            )
            monitor.update(block_name_of(event.name), event.name, "RUNNING")
            return

        if isinstance(event, TaskFinished):
            _report_finished(event, block_name_of, monitor)
            return

        if isinstance(event, RunFinished):
            return

        raise TypeError(f"stage runner received an unknown event: {event!r}")

    return execute_dag(
        plan,
        run_task,
        emit,
        input_count=len(items_to_process),
        task_names=list(stages_by_name),
    )


def _report_finished(
    event: TaskFinished,
    block_name_of: Callable[[str], str],
    monitor: PipelineMonitor,
) -> None:
    """Narrate one terminal status on the console and to the monitor."""
    if event.status is TaskStatus.COMPLETED:
        print(f"✅ Task '{event.name}' marked as completed.", flush=True)
    elif event.status is TaskStatus.SKIPPED_UNREGISTERED:
        logging.warning(event.message)
    elif event.status is not TaskStatus.FAILED and event.message:
        # A failed task already printed its traceback inside ``run_task``.
        print(f"⏭️  {event.message}", flush=True)

    monitor_status = _MONITOR_STATUS.get(event.status)
    if monitor_status is not None:
        monitor.update(
            block_name_of(event.name), event.name, monitor_status, event.message
        )
