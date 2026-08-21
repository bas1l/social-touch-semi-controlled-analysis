"""Conformance tests for the canonical task registry and the graph-view polish.

Headless throughout: no ``QApplication`` is constructed anywhere.  The widget
half touches module-level constants and pure helpers only, and is skipped when
PyQt5 is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from analysis.pipeline.task_registry import (
    REGISTRY_PATH,
    TaskMeta,
    get_task_meta,
    load_registry,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

#: The DAG configs the AnalysisRunnerGUI actually offers, read from the GUI's
#: own workflow catalogue rather than listed here, so a workflow added to the
#: GUI is covered by this conformance test without editing it.
_RUNNER_CONFIG = _CONFIGS_DIR / "analysis_runner_gui.yaml"


def _live_dag_config_paths() -> list[Path]:
    """Absolute paths of every ``dag_config`` declared in the GUI catalogue."""
    catalogue = YAML(typ="safe").load(_RUNNER_CONFIG.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for category in catalogue["categories"]:
        for workflow in category["workflows"]:
            if "dag_config" in workflow:
                paths.append(_PROJECT_ROOT / workflow["dag_config"])
    if not paths:
        raise AssertionError(f"{_RUNNER_CONFIG} declares no dag_config to check")
    return paths


def _task_ids_of(dag_config: Path) -> list[str]:
    loaded = YAML(typ="safe").load(dag_config.read_text(encoding="utf-8"))
    return [str(name) for name in loaded["tasks"]]


def _write_registry(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "analysis_task_registry.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Conformance against the live DAG configs
# ---------------------------------------------------------------------------


def test_live_dag_configs_are_discoverable() -> None:
    """Guard the guard: the catalogue must actually name some DAG configs."""
    paths = _live_dag_config_paths()
    assert len(paths) >= 2
    for path in paths:
        assert path.is_file(), path


@pytest.mark.parametrize("dag_config", _live_dag_config_paths(), ids=lambda p: p.stem)
def test_every_live_task_id_resolves_in_the_registry(dag_config: Path) -> None:
    registry = load_registry()
    unregistered = [name for name in _task_ids_of(dag_config) if name not in registry]
    assert not unregistered, (
        f"{dag_config.name} declares task(s) with no entry in "
        f"{REGISTRY_PATH.name}: {unregistered}"
    )


def test_every_live_task_id_has_non_empty_label_and_description() -> None:
    for dag_config in _live_dag_config_paths():
        for task_id in _task_ids_of(dag_config):
            meta = get_task_meta(task_id)
            assert isinstance(meta, TaskMeta)
            assert meta.task_id == task_id
            assert meta.label.strip()
            assert meta.description.strip()


def test_labels_are_distinct_across_the_registry() -> None:
    """Two tasks sharing a label would be indistinguishable in the GUI."""
    registry = load_registry()
    seen: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []
    for task_id, meta in registry.items():
        if meta.label in seen:
            collisions.append((meta.label, seen[meta.label], task_id))
        seen[meta.label] = task_id
    assert not collisions, f"duplicate label(s) in {REGISTRY_PATH.name}: {collisions}"


# ---------------------------------------------------------------------------
# Lookup contract
# ---------------------------------------------------------------------------


def test_unregistered_id_raises_key_error_naming_the_registry_file() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_task_meta("no_such_task_id")
    # KeyError's str() reprs its argument, escaping the Windows path separators;
    # the raised message itself is args[0].
    message = excinfo.value.args[0]
    assert "no_such_task_id" in message
    assert str(REGISTRY_PATH) in message


def test_task_meta_is_frozen() -> None:
    meta = get_task_meta(_task_ids_of(_live_dag_config_paths()[0])[0])
    with pytest.raises(Exception):
        meta.label = "mutated"  # type: ignore[misc]


def test_registry_mapping_is_not_mutable_by_callers() -> None:
    registry = load_registry()
    with pytest.raises(TypeError):
        registry["injected"] = None  # type: ignore[index]


def test_loader_is_cached_per_path() -> None:
    assert load_registry() is load_registry()
    # The default and the explicit path must not parse into two separate copies.
    assert load_registry() is load_registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Loader validation — every failure names the offending file
# ---------------------------------------------------------------------------


def test_empty_description_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        "some_task:\n  label: Some Task\n  description: '   '\n",
    )
    with pytest.raises(ValueError) as excinfo:
        load_registry(path)
    message = str(excinfo.value)
    assert "some_task" in message
    assert "description" in message
    assert str(path) in message


def test_missing_description_key_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, "some_task:\n  label: Some Task\n")
    with pytest.raises(ValueError, match="description"):
        load_registry(path)


def test_missing_label_key_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, "some_task:\n  description: Does a thing.\n")
    with pytest.raises(ValueError, match="label"):
        load_registry(path)


def test_unknown_key_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        "some_task:\n  label: Some Task\n  description: Does a thing.\n  output_dir: nope\n",
    )
    with pytest.raises(ValueError, match="output_dir"):
        load_registry(path)


def test_non_string_description_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, "some_task:\n  label: Some Task\n  description: 42\n")
    with pytest.raises(ValueError, match="description"):
        load_registry(path)


def test_empty_registry_raises_at_load(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, "{}\n")
    with pytest.raises(ValueError, match="declares no tasks"):
        load_registry(path)


def test_missing_registry_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "absent.yaml")


# ---------------------------------------------------------------------------
# Grid snap and click-vs-drag — module-level constants and pure helpers only
# ---------------------------------------------------------------------------


def _graph_items():
    pytest.importorskip("PyQt5")
    from utils.gui.analysis_runner_gui import dag_graph_items

    return dag_graph_items


def test_grid_size_is_shared_by_items_and_view() -> None:
    items = _graph_items()
    from utils.gui.analysis_runner_gui import dag_graph_view

    assert dag_graph_view.GRID_SIZE is items.GRID_SIZE, (
        "the view must import the snapping constant, never redeclare it"
    )


def test_grid_size_is_a_positive_integer() -> None:
    items = _graph_items()
    assert isinstance(items.GRID_SIZE, int)
    assert items.GRID_SIZE > 0


def test_snap_coordinate_rounds_to_the_nearest_grid_multiple() -> None:
    items = _graph_items()
    step = items.GRID_SIZE
    assert items.snap_coordinate(0.0) == 0
    assert items.snap_coordinate(step * 3.0) == step * 3
    assert items.snap_coordinate(step * 3 + 1.0) == step * 3
    assert items.snap_coordinate(step * 3 - 1.0) == step * 3
    assert items.snap_coordinate(step * 3 + step * 0.6) == step * 4
    assert items.snap_coordinate(-step * 3 - 1.0) == -step * 3


def test_snap_coordinate_is_idempotent() -> None:
    items = _graph_items()
    once = items.snap_coordinate(137.4)
    assert items.snap_coordinate(once) == once


def test_snap_to_grid_snaps_both_axes() -> None:
    items = _graph_items()
    from PyQt5.QtCore import QPointF

    step = items.GRID_SIZE
    snapped = items.snap_to_grid(QPointF(step * 2 + 1.0, step * 5 - 1.0))
    assert snapped.x() == step * 2
    assert snapped.y() == step * 5


def test_drag_threshold_is_strict_and_manhattan() -> None:
    items = _graph_items()
    from PyQt5.QtCore import QPointF

    threshold = items.DRAG_THRESHOLD
    assert not items.exceeds_drag_threshold(QPointF(0.0, 0.0))
    # Exactly at the threshold is still a click: the comparison is strict.
    assert not items.exceeds_drag_threshold(QPointF(float(threshold), 0.0))
    assert not items.exceeds_drag_threshold(QPointF(threshold / 2.0, threshold / 2.0))
    assert items.exceeds_drag_threshold(QPointF(float(threshold) + 1.0, 0.0))
    # Manhattan, not Euclidean: the two axes add up.
    assert items.exceeds_drag_threshold(
        QPointF(threshold / 2.0 + 1.0, threshold / 2.0 + 1.0)
    )
    # Sign-independent.
    assert items.exceeds_drag_threshold(QPointF(0.0, -(float(threshold) + 1.0)))


def test_drag_threshold_is_a_positive_number() -> None:
    items = _graph_items()
    assert items.DRAG_THRESHOLD > 0


# ---------------------------------------------------------------------------
# The layout sidecar is read strictly — no silent fallback to auto-layout
# ---------------------------------------------------------------------------


def test_shipped_layout_sidecars_have_a_nodes_mapping() -> None:
    """The two migrated sidecars must still satisfy the strict reader."""
    import json

    sidecars = sorted(_CONFIGS_DIR.glob("*.layout.json"))
    assert sidecars, "expected at least one committed layout sidecar"
    for path in sidecars:
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(saved, dict), path
        assert "nodes" in saved, f"{path} would now raise on load"
        assert isinstance(saved["nodes"], dict), path
        for name, position in saved["nodes"].items():
            assert isinstance(position, list) and len(position) == 2, (path, name)
