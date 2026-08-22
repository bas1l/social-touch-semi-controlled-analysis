"""Canonical display metadata for DAG task ids.

One owner for the human-facing ``label`` and ``description`` of every task that
appears in a workflow DAG.  The registry is *display metadata only*: it says
nothing about where a task writes, which stays with
:mod:`analysis.pipeline.output_dirs`, and nothing about how a task is wired,
which stays in the DAG config YAML.

The registry is a config file rather than a Python table so a task's wording can
be corrected without touching source, matching the repo-wide rule that
user-facing values live in ``configs/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ruamel.yaml import YAML

#: Canonical location of the registry file, resolved from this module so the
#: registry is found regardless of the process working directory.
#: ``parents[3]`` is the repository root (``src/analysis/pipeline/`` → root).
REGISTRY_PATH: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "analysis_task_registry.yaml"
)


@dataclass(frozen=True)
class TaskMeta:
    """Human-facing metadata for one canonical task id."""

    task_id: str
    label: str
    description: str


_REQUIRED_KEYS: frozenset[str] = frozenset({"label", "description"})


def _parse_entry(task_id: str, entry: object, path: Path) -> TaskMeta:
    """Validate one registry row and build its :class:`TaskMeta`.

    Every failure names the registry file, because the only way to fix any of
    them is to edit that file.
    """
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"task '{task_id}' in {path} must map to "
            f"{sorted(_REQUIRED_KEYS)}, got {type(entry).__name__}"
        )
    keys = set(entry)
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"task '{task_id}' in {path} is missing key(s) {sorted(missing)}")
    unknown = keys - _REQUIRED_KEYS
    if unknown:
        raise ValueError(f"task '{task_id}' in {path} has unknown key(s) {sorted(unknown)}")

    values: dict[str, str] = {}
    for key in sorted(_REQUIRED_KEYS):
        value = entry[key]
        if not isinstance(value, str):
            raise ValueError(
                f"task '{task_id}' key '{key}' in {path} must be a string, "
                f"got {type(value).__name__}"
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"task '{task_id}' key '{key}' in {path} is empty")
        values[key] = stripped

    return TaskMeta(task_id=task_id, label=values["label"], description=values["description"])


@lru_cache(maxsize=None)
def _load_registry_cached(path: Path) -> Mapping[str, TaskMeta]:
    """Parse and validate the registry at *path*; memoised on the resolved path."""
    if not path.is_file():
        raise FileNotFoundError(f"task registry not found at {path}")

    raw = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{path} must contain a mapping of task id -> "
            f"{sorted(_REQUIRED_KEYS)}, got {type(raw).__name__}"
        )
    if not raw:
        raise ValueError(f"{path} declares no tasks")

    return MappingProxyType(
        {str(task_id): _parse_entry(str(task_id), entry, path) for task_id, entry in raw.items()}
    )


def load_registry(path: Path = REGISTRY_PATH) -> Mapping[str, TaskMeta]:
    """Return the parsed registry, read once per file and cached thereafter.

    The path is resolved before caching so ``load_registry()`` and
    ``load_registry(REGISTRY_PATH)`` share one parsed copy.  That copy is an
    immutable mapping, so no caller can mutate what every other caller sees.
    """
    return _load_registry_cached(path.resolve())


def get_task_meta(task_id: str, path: Path = REGISTRY_PATH) -> TaskMeta:
    """Return the :class:`TaskMeta` for *task_id*.

    Raises ``KeyError`` naming the registry file when *task_id* is not declared:
    a task that reaches the GUI without display metadata is an authoring gap in
    the registry, not a condition to paper over with a placeholder.
    """
    registry = load_registry(path)
    if task_id not in registry:
        raise KeyError(
            f"task '{task_id}' is not declared in {path}; add an entry there "
            f"({len(registry)} task id(s) currently registered)"
        )
    return registry[task_id]
