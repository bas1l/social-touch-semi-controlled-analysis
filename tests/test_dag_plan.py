"""Validation, ordering and gating tests for ``DagPlan``.

Everything here is pure: a task mapping in, a plan or an exception out.  No
YAML is written, no flow is invoked, and the one test that reads the shipped
processing config only asserts topology, never the tie-break.
"""

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from analysis.pipeline.dag_plan import (
    OPTIONAL_TASK_KEYS,
    REQUIRED_TASK_KEYS,
    DagPlan,
)
from analysis.pipeline.execution_events import TaskStatus

PROCESSING_DAG = (
    Path(__file__).resolve().parents[1] / "configs" / "analyse_workflow_processing_dag.yaml"
)


def _task(**overrides):
    """A minimal valid task mapping, with *overrides* applied."""
    task = {
        "category": "foundation",
        "enabled": True,
        "bypass": False,
        "options": {},
        "depends_on": [],
    }
    task.update(overrides)
    return task


def _linear_plan():
    """``a -> b -> c``, all enabled, nothing bypassed."""
    return DagPlan.from_tasks(
        {
            "a": _task(),
            "b": _task(depends_on=["a"]),
            "c": _task(depends_on=["b"]),
        }
    )


# --- schema validation -------------------------------------------------------


@pytest.mark.parametrize("key", REQUIRED_TASK_KEYS)
def test_missing_required_key_raises_naming_it(key):
    raw = _task()
    del raw[key]
    with pytest.raises(ValueError) as excinfo:
        DagPlan.from_tasks({"a": raw})
    assert "missing required key(s)" in str(excinfo.value)
    assert key in str(excinfo.value)
    assert "task 'a'" in str(excinfo.value)


def test_bypass_is_among_the_required_keys():
    # Guards the deliberate decision that bypass must never default silently.
    assert "bypass" in REQUIRED_TASK_KEYS
    assert "bypass" not in OPTIONAL_TASK_KEYS


def test_unknown_key_raises_naming_it():
    raw = _task()
    raw["depend_on"] = ["a"]
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['depend_on'\]"):
        DagPlan.from_tasks({"a": raw})


@pytest.mark.parametrize("value", ["false", 0, None, [], "true"])
def test_non_boolean_bypass_raises(value):
    with pytest.raises(ValueError, match="key 'bypass' must be a boolean"):
        DagPlan.from_tasks({"a": _task(bypass=value)})


@pytest.mark.parametrize("value", ["true", 1, None])
def test_non_boolean_enabled_raises(value):
    with pytest.raises(ValueError, match="key 'enabled' must be a boolean"):
        DagPlan.from_tasks({"a": _task(enabled=value)})


def test_non_mapping_options_raises():
    with pytest.raises(ValueError, match="key 'options' must be a mapping"):
        DagPlan.from_tasks({"a": _task(options=["force_processing"])})


def test_string_depends_on_raises_rather_than_being_iterated_per_character():
    with pytest.raises(ValueError, match="'depends_on' must be a sequence"):
        DagPlan.from_tasks({"a": _task(), "b": _task(depends_on="a")})


def test_duplicate_dependency_raises():
    with pytest.raises(ValueError, match="duplicate target"):
        DagPlan.from_tasks({"a": _task(), "b": _task(depends_on=["a", "a"])})


def test_unknown_dependency_raises_naming_it():
    with pytest.raises(ValueError, match="depends_on names unknown task 'ghost'"):
        DagPlan.from_tasks({"a": _task(depends_on=["ghost"])})


def test_self_loop_raises_naming_the_member():
    with pytest.raises(ValueError) as excinfo:
        DagPlan.from_tasks({"a": _task(depends_on=["a"])})
    message = str(excinfo.value)
    assert "cycle" in message
    assert "a -> a" in message


def test_two_node_cycle_raises_naming_both_members():
    with pytest.raises(ValueError) as excinfo:
        DagPlan.from_tasks(
            {"a": _task(depends_on=["b"]), "b": _task(depends_on=["a"])}
        )
    message = str(excinfo.value)
    assert "cycle" in message
    assert "a" in message and "b" in message


def test_a_task_downstream_of_a_cycle_does_not_hide_the_cycle():
    with pytest.raises(ValueError) as excinfo:
        DagPlan.from_tasks(
            {
                "a": _task(depends_on=["b"]),
                "b": _task(depends_on=["a"]),
                "c": _task(depends_on=["a"]),
            }
        )
    assert "cycle" in str(excinfo.value)


def test_min_inputs_greater_than_max_inputs_raises():
    with pytest.raises(ValueError, match=r"min_inputs \(5\) exceeds max_inputs \(2\)"):
        DagPlan.from_tasks({"a": _task(min_inputs=5, max_inputs=2)})


@pytest.mark.parametrize("key", OPTIONAL_TASK_KEYS)
def test_negative_guard_raises(key):
    with pytest.raises(ValueError, match=f"key '{key}' must be >= 0"):
        DagPlan.from_tasks({"a": _task(**{key: -1})})


@pytest.mark.parametrize("key", OPTIONAL_TASK_KEYS)
def test_non_integer_guard_raises(key):
    with pytest.raises(ValueError, match=f"key '{key}' must be an integer"):
        DagPlan.from_tasks({"a": _task(**{key: "3"})})


def test_guards_are_optional():
    plan = DagPlan.from_tasks({"a": _task()})
    assert plan.task("a").min_inputs is None
    assert plan.task("a").max_inputs is None
    assert plan.task("a").has_input_guard is False


def test_disabled_and_bypassed_is_legal_and_inert():
    # Asserted by *not* raising: the flag persists untouched on a disabled task.
    plan = DagPlan.from_tasks({"a": _task(enabled=False, bypass=True)})
    assert plan.task("a").enabled is False
    assert plan.task("a").bypass is True


def test_empty_tasks_mapping_is_legal():
    plan = DagPlan.from_tasks({})
    assert plan.order == ()
    assert plan.tasks == {}


def test_non_mapping_input_raises():
    with pytest.raises(ValueError, match="DAG tasks must be a mapping"):
        DagPlan.from_tasks(["a", "b"])


def test_parsed_options_are_not_a_live_view_of_the_config():
    raw_options = {"force_processing": False}
    plan = DagPlan.from_tasks({"a": _task(options=raw_options)})
    raw_options["force_processing"] = True
    assert plan.task("a").options["force_processing"] is False


# --- ordering ----------------------------------------------------------------


def test_ordering_is_topological():
    plan = DagPlan.from_tasks(
        {
            "c": _task(depends_on=["b"]),
            "a": _task(),
            "b": _task(depends_on=["a"]),
        }
    )
    assert plan.order == ("a", "b", "c")


def test_authoring_order_breaks_ties():
    plan = DagPlan.from_tasks({"z": _task(), "m": _task(), "a": _task()})
    assert plan.order == ("z", "m", "a")


def test_ordering_is_deterministic_across_two_independent_builds():
    raw = {
        "root": _task(),
        "left": _task(depends_on=["root"]),
        "right": _task(depends_on=["root"]),
        "join": _task(depends_on=["left", "right"]),
    }
    first = DagPlan.from_tasks(dict(raw)).order
    second = DagPlan.from_tasks(dict(raw)).order
    assert first == second
    assert first[0] == "root"
    assert first[-1] == "join"


def _load_shipped_processing_tasks():
    yaml = YAML()
    with PROCESSING_DAG.open("r", encoding="utf-8") as handle:
        return yaml.load(handle)["tasks"]


def test_shipped_processing_config_builds_a_plan_over_the_same_task_set():
    raw = _load_shipped_processing_tasks()
    plan = DagPlan.from_tasks(raw)
    assert set(plan.order) == set(raw)
    assert len(plan.order) == len(raw)


@pytest.mark.parametrize(
    "upstream,downstream",
    [
        ("touch_prepare_sessions", "spatial_map_single_touch"),
        ("spatial_map_single_touch", "spatial_build_response_fields"),
        ("spatial_build_response_fields", "spatial_extract_boundary__radial"),
        ("spatial_build_response_fields", "spatial_extract_boundary__gradient"),
        ("spatial_build_response_fields", "spatial_extract_boundary__inflection"),
        ("spatial_extract_boundary__radial", "spatial_extract_boundaries"),
        ("spatial_extract_boundary__gradient", "spatial_extract_boundaries"),
        ("spatial_extract_boundary__inflection", "spatial_extract_boundaries"),
        ("spatial_extract_boundaries", "spatial_compare_boundaries"),
        ("spatial_extract_boundaries", "spatial_compare_proximal_distal"),
        ("spatial_extract_boundaries", "spatial_compare_tap_stroke"),
        ("spatial_extract_boundaries", "spatial_extract_rf_profiles"),
        ("spatial_extract_boundaries", "spatial_tuning_rf_metrics"),
    ],
)
def test_shipped_processing_config_orders_the_boundary_chain(upstream, downstream):
    # Constrains topology without over-fitting the authoring-order tie-break.
    order = list(DagPlan.from_tasks(_load_shipped_processing_tasks()).order)
    assert order.index(upstream) < order.index(downstream)


# --- run state ---------------------------------------------------------------


def test_a_fresh_plan_has_every_task_pending_and_nothing_completed():
    plan = _linear_plan()
    assert plan.completed_tasks == frozenset()
    assert set(plan.statuses.values()) == {TaskStatus.PENDING}


def test_mark_completed_sets_both_halves():
    plan = _linear_plan()
    plan.mark_completed("a")
    assert "a" in plan.completed_tasks
    assert plan.status_of("a") is TaskStatus.COMPLETED


def test_mark_bypassed_sets_both_halves():
    plan = _linear_plan()
    plan.mark_bypassed("a")
    assert "a" in plan.completed_tasks
    assert plan.status_of("a") is TaskStatus.BYPASSED


def test_a_bypassed_dependency_unblocks_its_dependent():
    plan = _linear_plan()
    assert plan.can_run("b") is False
    plan.mark_bypassed("a")
    assert plan.can_run("b") is True


def test_set_status_does_not_satisfy_dependents():
    plan = _linear_plan()
    plan.set_status("a", TaskStatus.SKIPPED_GUARD)
    assert "a" not in plan.completed_tasks
    assert plan.unmet_dependencies("b") == ("a",)
    assert plan.can_run("b") is False


def test_set_status_rejects_a_non_status():
    plan = _linear_plan()
    with pytest.raises(ValueError, match="must be a TaskStatus"):
        plan.set_status("a", "completed")


def test_can_run_is_false_for_a_disabled_task_with_satisfied_dependencies():
    plan = DagPlan.from_tasks({"a": _task(), "b": _task(enabled=False, depends_on=["a"])})
    plan.mark_completed("a")
    assert plan.unmet_dependencies("b") == ()
    assert plan.can_run("b") is False


def test_unmet_dependencies_reports_in_declared_order():
    plan = DagPlan.from_tasks(
        {"a": _task(), "b": _task(), "c": _task(depends_on=["b", "a"])}
    )
    assert plan.unmet_dependencies("c") == ("b", "a")
    plan.mark_completed("b")
    assert plan.unmet_dependencies("c") == ("a",)


@pytest.mark.parametrize(
    "call",
    [
        lambda plan: plan.can_run("ghost"),
        lambda plan: plan.mark_completed("ghost"),
        lambda plan: plan.mark_bypassed("ghost"),
        lambda plan: plan.guard_violation("ghost", 1),
        lambda plan: plan.task("ghost"),
        lambda plan: plan.status_of("ghost"),
        lambda plan: plan.unmet_dependencies("ghost"),
    ],
)
def test_unknown_task_name_raises_key_error(call):
    plan = _linear_plan()
    with pytest.raises(KeyError, match="unknown task 'ghost'"):
        call(plan)


# --- input guards ------------------------------------------------------------


def test_an_unbounded_task_never_violates():
    plan = DagPlan.from_tasks({"a": _task()})
    for count in (0, 1, 10_000):
        assert plan.guard_violation("a", count) is None


def test_equal_min_and_max_collapses_to_exactly():
    plan = DagPlan.from_tasks({"a": _task(min_inputs=3, max_inputs=3)})
    assert plan.guard_violation("a", 3) is None
    message = plan.guard_violation("a", 2)
    assert "exactly 3" in message
    assert "got 2" in message
    assert "between" not in message


def test_min_only_guard():
    plan = DagPlan.from_tasks({"a": _task(min_inputs=2)})
    assert plan.guard_violation("a", 2) is None
    assert plan.guard_violation("a", 99) is None
    assert "at least 2" in plan.guard_violation("a", 1)


def test_max_only_guard():
    plan = DagPlan.from_tasks({"a": _task(max_inputs=2)})
    assert plan.guard_violation("a", 0) is None
    assert plan.guard_violation("a", 2) is None
    assert "at most 2" in plan.guard_violation("a", 3)


def test_range_guard():
    plan = DagPlan.from_tasks({"a": _task(min_inputs=2, max_inputs=4)})
    assert plan.guard_violation("a", 3) is None
    assert "between 2 and 4" in plan.guard_violation("a", 5)
    assert "between 2 and 4" in plan.guard_violation("a", 1)


def test_guard_violation_rejects_a_non_integer_count():
    plan = DagPlan.from_tasks({"a": _task(min_inputs=1)})
    with pytest.raises(ValueError, match="item_count must be an integer"):
        plan.guard_violation("a", "3")
