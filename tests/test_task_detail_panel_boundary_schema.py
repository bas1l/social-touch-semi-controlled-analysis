"""Schema-resolution unit tests for the registry-driven boundary GUI (Phase 6).

These tests exercise the *pure* schema-resolution helpers of
``task_detail_panel`` — no Qt widget is constructed. They self-cover any newly
registered boundary method because every case iterates
``boundary_registry.all_methods()``.
"""

from __future__ import annotations

import pytest

from analysis.receptive_field_mapping.boundary import registry as boundary_registry
from utils.gui.analysis_runner_gui import task_detail_panel as tdp


def _node_name(method_name: str) -> str:
    return f"{tdp._BOUNDARY_METHOD_NODE_PREFIX}{method_name}"


def test_registry_has_methods() -> None:
    """Guard: the registry is populated (import registered the concrete methods)."""
    assert boundary_registry.all_methods(), "no boundary methods registered"


@pytest.mark.parametrize(
    "method", boundary_registry.all_methods(), ids=lambda m: m.name
)
def test_node_name_resolves_method_schema(method) -> None:
    """(a) Resolving a method-node name yields exactly that method's ParamSpec list."""
    node = _node_name(method.name)

    assert tdp.boundary_method_of_task(node) == method.name

    resolved = tdp.boundary_param_specs(node)
    expected = list(method.params_schema)

    # Same keys, same order.
    assert list(resolved.keys()) == [spec.key for spec in expected]
    # Same ParamSpec objects (identity — single source of truth is the registry).
    for spec in expected:
        assert resolved[spec.key] is spec


@pytest.mark.parametrize(
    "method", boundary_registry.all_methods(), ids=lambda m: m.name
)
def test_every_param_resolves_a_group(method) -> None:
    """(b) Every ParamSpec of every method resolves a group with no ValueError."""
    node = _node_name(method.name)
    for spec in method.params_schema:
        assert tdp.option_group_of(node, spec.key) == spec.group


@pytest.mark.parametrize(
    "method", boundary_registry.all_methods(), ids=lambda m: m.name
)
def test_group_catalogue_covers_every_param_group(method) -> None:
    """The rendered group catalogue contains every ParamSpec group of the node."""
    node = _node_name(method.name)
    catalogue_ids = {gid for gid, _label in tdp.TaskDetailPanel._group_catalogue(node)}
    for spec in method.params_schema:
        assert spec.group in catalogue_ids


def test_no_boundary_method_enum_remains() -> None:
    """(c) The hardcoded ``boundary_method`` enum is gone from _OPTION_ENUMS."""
    assert "boundary_method" not in tdp._OPTION_ENUMS


def test_no_per_method_boundary_option_in_group_map() -> None:
    """No per-method boundary param is hardcoded in the shared group map."""
    method_param_keys = {
        spec.key
        for method in boundary_registry.all_methods()
        for spec in method.params_schema
    }
    assert method_param_keys.isdisjoint(tdp._OPTION_GROUP_OF)


def test_unknown_boundary_method_node_fails_fast() -> None:
    """A boundary node naming an unregistered method raises (fail-fast)."""
    with pytest.raises(ValueError):
        tdp.boundary_method_of_task(_node_name("does_not_exist"))


def test_non_boundary_task_is_not_a_boundary_method_node() -> None:
    assert not tdp._is_boundary_method_task("spatial_extract_boundaries")
    assert tdp._is_grouped_task(_node_name("radial"))
