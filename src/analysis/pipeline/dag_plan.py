"""Validated, ordered representation of a DAG config's ``tasks`` mapping.

``DagPlan`` is the state machine a run walks.  It owns three things and nothing
else:

* **Topology** — the parsed, immutable ``DagTask`` records and the Kahn
  topological order they must be visited in.
* **Validation** — every task declares exactly the required keys, every
  dependency names a real task, and the graph is acyclic.  Every violation
  raises, naming the task and the offending key or member; nothing is defaulted.
* **Run state** — which tasks have completed (whether by running or by being
  bypassed) and each task's current :class:`TaskStatus`.

It knows nothing about Qt, Prefect, subprocesses, stdout or file paths: it is
handed an already-loaded mapping and answers questions about it.

Task schema
-----------
Required keys — ``category``, ``enabled``, ``bypass``, ``options``,
``depends_on``.  ``bypass`` is deliberately *required* rather than defaulted:
a flag whose failure mode is "silently runs downstream work against stale
outputs" is the wrong place to be lenient, so a config that omits it fails at
load rather than at run time.

Optional keys — ``min_inputs`` / ``max_inputs``, non-negative integer guards on
the number of discovered input items.  A task declaring neither can never be
guard-skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .execution_events import TaskStatus

#: Keys every task mapping must declare.
REQUIRED_TASK_KEYS: Tuple[str, ...] = (
    "category",
    "enabled",
    "bypass",
    "options",
    "depends_on",
)

#: Keys a task mapping may declare in addition to the required ones.
OPTIONAL_TASK_KEYS: Tuple[str, ...] = ("min_inputs", "max_inputs")


@dataclass(frozen=True)
class DagTask:
    """One immutable task declaration from the DAG config."""

    name: str
    category: str
    enabled: bool
    bypass: bool
    options: Mapping[str, Any]
    depends_on: Tuple[str, ...]
    min_inputs: Optional[int]
    max_inputs: Optional[int]

    @property
    def has_input_guard(self) -> bool:
        """True when the task constrains the number of discovered input items."""
        return self.min_inputs is not None or self.max_inputs is not None


def _require_bool(task_name: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"task '{task_name}': key '{key}' must be a boolean, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _require_str(task_name: str, key: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"task '{task_name}': key '{key}' must be a string, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _require_options(task_name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"task '{task_name}': key 'options' must be a mapping, "
            f"got {type(value).__name__} ({value!r})"
        )
    return MappingProxyType(dict(value))


def _require_dependency_list(task_name: str, value: Any) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"task '{task_name}': key 'depends_on' must be a sequence of task "
            f"names, got {type(value).__name__} ({value!r})"
        )
    names: List[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ValueError(
                f"task '{task_name}': every 'depends_on' entry must be a string, "
                f"got {type(entry).__name__} ({entry!r})"
            )
        names.append(entry)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"task '{task_name}': 'depends_on' lists duplicate target(s): {duplicates}"
        )
    return tuple(names)


def _require_optional_count(task_name: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"task '{task_name}': key '{key}' must be an integer, "
            f"got {type(value).__name__} ({value!r})"
        )
    if value < 0:
        raise ValueError(f"task '{task_name}': key '{key}' must be >= 0, got {value}")
    return value


def _parse_task(name: str, raw: Any) -> DagTask:
    """Parse and validate one task mapping, or raise naming *name*."""
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"task '{name}': task entry must be a mapping, got {type(raw).__name__}"
        )

    supplied = set(raw)
    missing = [key for key in REQUIRED_TASK_KEYS if key not in supplied]
    if missing:
        raise ValueError(f"task '{name}': missing required key(s): {missing}")
    unknown = sorted(supplied - set(REQUIRED_TASK_KEYS) - set(OPTIONAL_TASK_KEYS))
    if unknown:
        allowed = sorted(set(REQUIRED_TASK_KEYS) | set(OPTIONAL_TASK_KEYS))
        raise ValueError(
            f"task '{name}': unknown key(s) {unknown} (allowed: {allowed})"
        )

    min_inputs = (
        _require_optional_count(name, "min_inputs", raw["min_inputs"])
        if "min_inputs" in supplied
        else None
    )
    max_inputs = (
        _require_optional_count(name, "max_inputs", raw["max_inputs"])
        if "max_inputs" in supplied
        else None
    )
    if min_inputs is not None and max_inputs is not None and min_inputs > max_inputs:
        raise ValueError(
            f"task '{name}': min_inputs ({min_inputs}) exceeds max_inputs ({max_inputs})"
        )

    return DagTask(
        name=name,
        category=_require_str(name, "category", raw["category"]),
        enabled=_require_bool(name, "enabled", raw["enabled"]),
        bypass=_require_bool(name, "bypass", raw["bypass"]),
        options=_require_options(name, raw["options"]),
        depends_on=_require_dependency_list(name, raw["depends_on"]),
        min_inputs=min_inputs,
        max_inputs=max_inputs,
    )


def _find_cycle(
    tasks: Mapping[str, DagTask], authoring_order: Sequence[str]
) -> Tuple[str, ...]:
    """Return one dependency cycle as an ordered tuple of member names.

    Depth-first search in authoring order, so the reported cycle is a
    deterministic function of the config rather than of dict iteration luck.
    Returns an empty tuple when the graph is acyclic.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {name: WHITE for name in authoring_order}
    stack: List[str] = []

    def visit(name: str) -> Tuple[str, ...]:
        colour[name] = GREY
        stack.append(name)
        for dependency in tasks[name].depends_on:
            if colour[dependency] == GREY:
                return tuple(stack[stack.index(dependency):]) + (dependency,)
            if colour[dependency] == WHITE:
                found = visit(dependency)
                if found:
                    return found
        stack.pop()
        colour[name] = BLACK
        return ()

    for name in authoring_order:
        if colour[name] == WHITE:
            cycle = visit(name)
            if cycle:
                return cycle
    return ()


def _topological_order(
    tasks: Mapping[str, DagTask], authoring_order: Sequence[str]
) -> Tuple[str, ...]:
    """Kahn's algorithm with YAML authoring order as the tie-break.

    Among the tasks whose dependencies are all satisfied, the one declared
    earliest in the config is emitted first.  The result is therefore a pure
    function of the config text, so two independent builds order identically.
    """
    position = {name: index for index, name in enumerate(authoring_order)}
    unmet = {name: set(tasks[name].depends_on) for name in authoring_order}
    dependents: Dict[str, List[str]] = {name: [] for name in authoring_order}
    for name in authoring_order:
        for dependency in tasks[name].depends_on:
            dependents[dependency].append(name)

    ready = sorted((name for name in authoring_order if not unmet[name]), key=position.get)
    order: List[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        released: List[str] = []
        for dependent in dependents[name]:
            unmet[dependent].discard(name)
            if not unmet[dependent]:
                released.append(dependent)
        if released:
            ready = sorted(ready + released, key=position.get)

    if len(order) != len(authoring_order):
        cycle = _find_cycle(tasks, authoring_order)
        raise ValueError(
            "DAG config contains a dependency cycle: " + " -> ".join(cycle)
        )
    return tuple(order)


class DagPlan:
    """A validated DAG plus the run state of one execution of it."""

    def __init__(self, tasks: Mapping[str, DagTask], order: Tuple[str, ...]):
        """Prefer :meth:`from_tasks`; this takes already-validated inputs."""
        self._tasks: Mapping[str, DagTask] = MappingProxyType(dict(tasks))
        self._order: Tuple[str, ...] = tuple(order)
        self._completed: Set[str] = set()
        self._statuses: Dict[str, TaskStatus] = {
            name: TaskStatus.PENDING for name in self._tasks
        }

    # --- construction --------------------------------------------------------

    @classmethod
    def from_tasks(cls, tasks: Mapping[str, Any]) -> "DagPlan":
        """Build a plan from a DAG config's ``tasks`` mapping.

        *tasks* is a snapshot of already-loaded config — typically
        ``DagConfigHandler.tasks``.  Authoring order is the mapping's iteration
        order, which ruamel/PyYAML both preserve from the YAML document.
        """
        if not isinstance(tasks, Mapping):
            raise ValueError(
                f"DAG tasks must be a mapping, got {type(tasks).__name__}"
            )

        authoring_order = tuple(tasks)
        for name in authoring_order:
            if not isinstance(name, str):
                raise ValueError(
                    f"DAG task names must be strings, got {type(name).__name__} ({name!r})"
                )

        parsed = {name: _parse_task(name, tasks[name]) for name in authoring_order}

        for name in authoring_order:
            for dependency in parsed[name].depends_on:
                if dependency not in parsed:
                    raise ValueError(
                        f"task '{name}': depends_on names unknown task '{dependency}'"
                    )

        return cls(parsed, _topological_order(parsed, authoring_order))

    # --- topology ------------------------------------------------------------

    @property
    def tasks(self) -> Mapping[str, DagTask]:
        """The parsed tasks, keyed by name, in authoring order."""
        return self._tasks

    @property
    def order(self) -> Tuple[str, ...]:
        """Topological execution order, authoring order breaking ties."""
        return self._order

    def task(self, name: str) -> DagTask:
        """Return the task named *name*, or raise ``KeyError``."""
        self._require_known(name)
        return self._tasks[name]

    # --- run state -----------------------------------------------------------

    @property
    def completed_tasks(self) -> frozenset:
        """Names that count as done for dependency purposes this run."""
        return frozenset(self._completed)

    @property
    def statuses(self) -> Mapping[str, TaskStatus]:
        """Current status of every task in the plan."""
        return MappingProxyType(dict(self._statuses))

    def status_of(self, name: str) -> TaskStatus:
        self._require_known(name)
        return self._statuses[name]

    def set_status(self, name: str, status: TaskStatus) -> None:
        """Record a status that does **not** satisfy dependents."""
        self._require_known(name)
        if not isinstance(status, TaskStatus):
            raise ValueError(
                f"task '{name}': status must be a TaskStatus, got {type(status).__name__}"
            )
        self._statuses[name] = status

    def mark_completed(self, name: str) -> None:
        """Record that *name* ran successfully; its dependents are released."""
        self._require_known(name)
        self._completed.add(name)
        self._statuses[name] = TaskStatus.COMPLETED

    def mark_bypassed(self, name: str) -> None:
        """Record that *name* was bypassed: nothing ran, dependents released.

        Both halves are the contract — the name enters ``completed_tasks`` so
        downstream tasks unblock, *and* the status is ``BYPASSED`` so no report
        ever claims work that did not happen.
        """
        self._require_known(name)
        self._completed.add(name)
        self._statuses[name] = TaskStatus.BYPASSED

    # --- gating --------------------------------------------------------------

    def unmet_dependencies(self, name: str) -> Tuple[str, ...]:
        """Dependencies of *name* that have not completed, in declared order."""
        self._require_known(name)
        return tuple(
            dependency
            for dependency in self._tasks[name].depends_on
            if dependency not in self._completed
        )

    def can_run(self, name: str) -> bool:
        """True when *name* is enabled and every dependency has completed.

        This mirrors the historical ``DagConfigHandler.can_run`` predicate and
        is the coarse answer.  The execution ladder uses the finer-grained
        :meth:`unmet_dependencies` so it can name *which* dependency blocked.
        """
        self._require_known(name)
        return self._tasks[name].enabled and not self.unmet_dependencies(name)

    def guard_violation(self, name: str, item_count: int) -> Optional[str]:
        """Return why *item_count* violates *name*'s input guard, else ``None``.

        A task declaring neither ``min_inputs`` nor ``max_inputs`` is unbounded
        and never violates.
        """
        self._require_known(name)
        if isinstance(item_count, bool) or not isinstance(item_count, int):
            raise ValueError(
                f"task '{name}': item_count must be an integer, "
                f"got {type(item_count).__name__}"
            )

        task = self._tasks[name]
        minimum, maximum = task.min_inputs, task.max_inputs
        if minimum is None and maximum is None:
            return None
        if minimum is not None and maximum is not None:
            if minimum <= item_count <= maximum:
                return None
            expectation = (
                f"exactly {minimum}"
                if minimum == maximum
                else f"between {minimum} and {maximum}"
            )
        elif minimum is not None:
            if item_count >= minimum:
                return None
            expectation = f"at least {minimum}"
        else:
            if item_count <= maximum:
                return None
            expectation = f"at most {maximum}"

        return (
            f"task '{name}' requires {expectation} input item(s); got {item_count}"
        )

    # --- internals -----------------------------------------------------------

    def _require_known(self, name: str) -> None:
        if name not in self._tasks:
            raise KeyError(f"unknown task '{name}' — not declared in the DAG config")


__all__ = [
    "DagTask",
    "DagPlan",
    "REQUIRED_TASK_KEYS",
    "OPTIONAL_TASK_KEYS",
]
