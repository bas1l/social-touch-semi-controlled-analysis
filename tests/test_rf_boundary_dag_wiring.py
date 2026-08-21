"""DAG wiring conformance for the pluggable boundary fan-out (Phase 4).

Asserts that the processing DAG + layout are consistent with the boundary-method
registry, so the wiring self-updates when a method is added/removed:

* every registered method has a ``spatial_extract_boundary__<method>`` node whose
  ``options`` carry that method's declared ``params_schema`` keys + the shared
  visualization/neuron options, and whose ``depends_on`` includes
  ``spatial_build_response_fields`` (radial additionally the RF-contour tuner);
* the retained ``spatial_extract_boundaries`` node is the fan-in barrier — its
  ``depends_on`` is exactly the set of per-method node names;
* every method node (and the barrier) has a layout coordinate.

Parametrized off ``boundary_registry.all_methods()`` so the suite tracks the
registry rather than a hardcoded method list.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.boundary import registry as boundary_registry  # noqa: E402

_REPO = Path(__file__).parent.parent
_DAG_YAML = _REPO / "configs" / "analyse_workflow_processing_dag.yaml"
_LAYOUT_JSON = _REPO / "configs" / "analyse_workflow_processing_dag.layout.json"

_BARRIER_NODE = "spatial_extract_boundaries"
_SHARED_OPTION_KEYS = {
    "force_processing",
    "neuron_mode",
    "iff_metric",
    "min_overlap_pct",
    "median_filter_size",
    "heatmap_space",
    "cmap",
    "flip_u",
    "contour_color",
    "circular_crop_margin",
    "use_tuned_params",
}


def _node_name(method_name: str) -> str:
    return f"spatial_extract_boundary__{method_name}"


def _load_tasks() -> dict:
    yaml = YAML()
    with open(_DAG_YAML, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return data["tasks"]


def _load_layout() -> dict:
    with open(_LAYOUT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _layout_nodes() -> dict:
    """Task name -> ``[x, y]`` from the layout sidecar.

    The sidecar written by ``DagGraphView._save_layout`` is
    ``{"nodes": {task: [x, y]}, "view": {...}}``; coordinates live under
    ``"nodes"`` and nowhere else, so the document itself is never keyed by
    task name.
    """
    if "nodes" not in _LAYOUT:
        raise KeyError(
            f"{_LAYOUT_JSON} declares no 'nodes' mapping. Expected the schema written "
            'by DagGraphView._save_layout: {"nodes": {task: [x, y]}, "view": {...}}'
        )
    return _LAYOUT["nodes"]


_TASKS = _load_tasks()
_LAYOUT = _load_layout()
_METHODS = list(boundary_registry.all_methods())
_METHOD_IDS = [m.name for m in _METHODS]


def test_registry_is_non_empty():
    assert _METHODS, "the boundary registry must register at least one method"


@pytest.mark.parametrize("method", _METHODS, ids=_METHOD_IDS)
def test_method_node_exists_and_enabled(method):
    node = _node_name(method.name)
    assert node in _TASKS, f"missing DAG node {node!r} for registered method {method.name!r}"
    spec = _TASKS[node]
    assert spec.get("enabled") is True, f"{node} must be enabled"
    assert str(spec.get("category", "")).strip(), f"{node} must declare a category"


@pytest.mark.parametrize("method", _METHODS, ids=_METHOD_IDS)
def test_method_node_depends_on_response_fields(method):
    node = _node_name(method.name)
    depends_on = list(_TASKS[node]["depends_on"])
    assert "spatial_build_response_fields" in depends_on, (
        f"{node} must depend on spatial_build_response_fields"
    )
    # radial consumes the RF-contour tuner's per-gesture JSONs.
    if method.name == "radial":
        assert "spatial_tune_rf_contours" in depends_on, (
            "spatial_extract_boundary__radial must depend on spatial_tune_rf_contours"
        )


@pytest.mark.parametrize("method", _METHODS, ids=_METHOD_IDS)
def test_method_node_options_cover_schema_and_shared(method):
    node = _node_name(method.name)
    options = _TASKS[node]["options"]
    option_keys = set(options.keys())

    # Every declared schema key must be present as an option.
    schema_keys = {spec.key for spec in method.params_schema}
    missing_schema = schema_keys - option_keys
    assert not missing_schema, f"{node} options missing schema keys: {sorted(missing_schema)}"

    # All shared visualization/neuron options must be present.
    missing_shared = _SHARED_OPTION_KEYS - option_keys
    assert not missing_shared, f"{node} options missing shared keys: {sorted(missing_shared)}"

    # No OTHER method's schema-specific keys should leak in (isolation).
    other_keys = set()
    for other in _METHODS:
        if other.name == method.name:
            continue
        other_keys |= {spec.key for spec in other.params_schema}
    leaked = (other_keys - schema_keys) & option_keys
    assert not leaked, f"{node} options contain another method's schema keys: {sorted(leaked)}"


@pytest.mark.parametrize("method", _METHODS, ids=_METHOD_IDS)
def test_method_node_has_layout(method):
    node = _node_name(method.name)
    nodes = _layout_nodes()
    assert node in nodes, f"layout.json missing coordinate for {node}"
    coord = nodes[node]
    assert isinstance(coord, list) and len(coord) == 2, f"{node} layout must be [x, y]"


def test_barrier_depends_on_exactly_the_method_nodes():
    assert _BARRIER_NODE in _TASKS, f"missing barrier node {_BARRIER_NODE!r}"
    expected = {_node_name(m.name) for m in _METHODS}
    actual = set(_TASKS[_BARRIER_NODE]["depends_on"])
    assert actual == expected, (
        f"{_BARRIER_NODE}.depends_on must equal the per-method node set.\n"
        f"  expected: {sorted(expected)}\n  actual:   {sorted(actual)}"
    )


def test_barrier_carries_no_method_specific_options():
    # The barrier performs no extraction; it must not carry method/visualization
    # option knobs (only force_processing + iff_metric to locate run sentinels).
    options = set(_TASKS[_BARRIER_NODE]["options"].keys())
    all_schema_keys = set()
    for m in _METHODS:
        all_schema_keys |= {spec.key for spec in m.params_schema}
    leaked = all_schema_keys & options
    assert not leaked, f"barrier must not carry method schema options: {sorted(leaked)}"
    assert options <= {"force_processing", "iff_metric"}, (
        f"barrier options should be a subset of {{force_processing, iff_metric}}, got {sorted(options)}"
    )


def test_barrier_has_layout():
    assert _BARRIER_NODE in _layout_nodes(), (
        "layout.json missing coordinate for the barrier node"
    )


_ALL_DAG_YAMLS = sorted((_REPO / "configs").glob("analyse_workflow*_dag.yaml"))


def test_every_dag_config_is_discovered():
    # Guards the glob above: a renamed config must not silently drop out of the
    # bypass conformance check below.
    assert len(_ALL_DAG_YAMLS) == 3, [p.name for p in _ALL_DAG_YAMLS]


@pytest.mark.parametrize("dag_yaml", _ALL_DAG_YAMLS, ids=lambda p: p.name)
def test_every_task_declares_bypass(dag_yaml):
    """``bypass`` is a *required* key, so a config that omits it fails at load.

    A branch that adds a task node merges cleanly as text but produces a config
    the plan parser rejects; this test makes that failure surface in CI rather
    than at run time.
    """
    yaml = YAML()
    with open(dag_yaml, "r", encoding="utf-8") as f:
        tasks = yaml.load(f)["tasks"]

    missing = [name for name, task in tasks.items() if "bypass" not in task]
    assert not missing, f"{dag_yaml.name}: task(s) with no 'bypass' key: {missing}"

    non_boolean = [
        name for name, task in tasks.items() if not isinstance(task["bypass"], bool)
    ]
    assert not non_boolean, (
        f"{dag_yaml.name}: task(s) whose 'bypass' is not a boolean: {non_boolean}"
    )


@pytest.mark.parametrize("dag_yaml", _ALL_DAG_YAMLS, ids=lambda p: p.name)
def test_bypass_sits_immediately_after_enabled(dag_yaml):
    yaml = YAML()
    with open(dag_yaml, "r", encoding="utf-8") as f:
        tasks = yaml.load(f)["tasks"]

    misplaced = {
        name: list(task.keys())[:3]
        for name, task in tasks.items()
        if list(task.keys())[:3] != ["category", "enabled", "bypass"]
    }
    assert not misplaced, f"{dag_yaml.name}: unexpected leading key order: {misplaced}"


def test_downstream_consumers_still_depend_on_barrier_only():
    # Phase 4 must not touch downstream depends_on: each still points at the barrier.
    for consumer in (
        "spatial_compare_boundaries",
        "spatial_compare_proximal_distal",
        "spatial_compare_tap_stroke",
        "spatial_extract_rf_profiles",
    ):
        assert consumer in _TASKS, f"expected downstream consumer {consumer!r} in DAG"
        assert list(_TASKS[consumer]["depends_on"]) == [_BARRIER_NODE], (
            f"{consumer}.depends_on must remain [{_BARRIER_NODE}] (unchanged by Phase 4)"
        )
