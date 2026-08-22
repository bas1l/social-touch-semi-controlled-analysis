"""End-to-end test of the ``run_pipeline_stages`` adapter.

Exercises the real ``DagConfigHandler`` against a fixture DAG written to
``tmp_path``, with stub flow callables and a stub monitor.  No Prefect, no Qt,
no subprocess — this is the first coverage the workflow driver has had.
"""

import textwrap
from pathlib import Path

import pytest

from _vendor import DagConfigHandler
from analysis.pipeline.execution_events import (
    SENTINEL_PREFIX,
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
    decode_event,
)
from analysis.pipeline.stage_runner import run_pipeline_stages

FIXTURE_DAG = """
parameters:
  kinect_configs: []
tasks:
  root:
    category: foundation
    enabled: true
    bypass: false
    options:
      force_processing: true
    depends_on: []
  middle:
    category: foundation
    enabled: true
    bypass: false
    options:
      force_processing: false
      knob: 7
    depends_on: [root]
  leaf:
    category: foundation
    enabled: true
    bypass: false
    options:
      force_processing: false
    depends_on: [middle]
  sibling:
    category: foundation
    enabled: true
    bypass: false
    options:
      force_processing: false
    depends_on: []
"""


class _StubMonitor:
    def __init__(self):
        self.updates = []

    def update(self, dataset, stage, status, message=""):
        self.updates.append((dataset, stage, status, message))


class _StubFlow:
    """Records the kwargs it was called with; optionally raises."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error


class _CountingParams:
    """A ``params`` lambda that records how often it was evaluated."""

    def __init__(self, payload=None, error=None):
        self.payload = dict(payload or {})
        self.error = error
        self.evaluations = 0

    def __call__(self):
        self.evaluations += 1
        if self.error is not None:
            raise self.error
        return dict(self.payload)


def _write_dag(tmp_path: Path, text: str = FIXTURE_DAG) -> DagConfigHandler:
    path = tmp_path / "fixture_dag.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return DagConfigHandler(path)


def _items():
    return [(Path("a.csv"), Path("db")), (Path("b.csv"), Path("db"))]


def _stages(names, flows=None, params=None):
    flows = {} if flows is None else flows
    params = {} if params is None else params
    return [
        {
            "name": name,
            "func": flows.setdefault(name, _StubFlow()),
            "params": params.setdefault(name, _CountingParams()),
        }
        for name in names
    ]


def _sentinel_events(captured_stdout):
    return [
        decode_event(line[len(SENTINEL_PREFIX):])
        for line in captured_stdout.splitlines()
        if line.startswith(SENTINEL_PREFIX)
    ]


# --- happy path --------------------------------------------------------------


def test_dependency_gating_orders_the_run_regardless_of_descriptor_order(tmp_path):
    handler = _write_dag(tmp_path)
    flows, params = {}, {}
    stages = _stages(["leaf", "sibling", "middle", "root"], flows, params)

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")

    assert outcome.exit_code == 0
    assert set(outcome.completed_tasks) == {"root", "middle", "leaf", "sibling"}
    for flow in flows.values():
        assert len(flow.calls) == 1


def test_common_kwargs_are_injected_and_task_params_preserved(tmp_path):
    handler = _write_dag(tmp_path)
    flows, params = {}, {}
    params["middle"] = _CountingParams({"knob": 7})
    stages = _stages(["root", "middle"], flows, params)
    items = _items()

    run_pipeline_stages(stages, handler, _StubMonitor(), items, "unit")

    assert flows["middle"].calls[0] == {
        "knob": 7,
        "input_items": items,
        "force_processing": False,
    }
    assert flows["root"].calls[0]["force_processing"] is True


def test_params_are_evaluated_lazily_only_for_tasks_that_run(tmp_path):
    handler = _write_dag(
        tmp_path, FIXTURE_DAG.replace("  middle:\n    category: foundation\n    enabled: true",
                                      "  middle:\n    category: foundation\n    enabled: false")
    )
    params = {
        "middle": _CountingParams(error=ValueError("required option absent")),
        "leaf": _CountingParams(error=ValueError("required option absent")),
    }
    stages = _stages(["root", "middle", "leaf", "sibling"], {}, params)

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")

    assert params["middle"].evaluations == 0
    assert params["leaf"].evaluations == 0
    assert outcome.statuses["middle"] is TaskStatus.SKIPPED_DISABLED
    assert outcome.statuses["leaf"] is TaskStatus.SKIPPED_DEP
    assert outcome.exit_code == 0


# --- bypass ------------------------------------------------------------------


def test_a_bypassed_node_unlocks_its_dependent_without_running(tmp_path):
    handler = _write_dag(
        tmp_path,
        FIXTURE_DAG.replace(
            "  middle:\n    category: foundation\n    enabled: true\n    bypass: false",
            "  middle:\n    category: foundation\n    enabled: true\n    bypass: true",
        ),
    )
    flows, params = {}, {}
    params["middle"] = _CountingParams(error=AssertionError("bypassed task must not run"))
    stages = _stages(["root", "middle", "leaf"], flows, params)

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")

    assert outcome.statuses["middle"] is TaskStatus.BYPASSED
    assert outcome.statuses["leaf"] is TaskStatus.COMPLETED
    assert flows["middle"].calls == []
    assert len(flows["leaf"].calls) == 1
    assert outcome.exit_code == 0


# --- failure -----------------------------------------------------------------


def test_a_failed_node_cascades_and_makes_the_exit_code_non_zero(tmp_path):
    handler = _write_dag(tmp_path)
    flows = {"middle": _StubFlow(error=RuntimeError("flow blew up"))}
    stages = _stages(["root", "middle", "leaf", "sibling"], flows, {})

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")

    assert outcome.statuses["middle"] is TaskStatus.FAILED
    assert outcome.statuses["leaf"] is TaskStatus.SKIPPED_DEP
    assert outcome.statuses["sibling"] is TaskStatus.COMPLETED
    assert outcome.failed_tasks == ("middle",)
    assert outcome.exit_code == 1


def test_the_monitor_still_receives_the_historical_status_vocabulary(tmp_path):
    handler = _write_dag(tmp_path)
    flows = {"middle": _StubFlow(error=RuntimeError("flow blew up"))}
    monitor = _StubMonitor()
    stages = _stages(["root", "middle"], flows, {})

    run_pipeline_stages(stages, handler, monitor, _items(), "unit")

    by_stage = {(stage, status) for _, stage, status, _ in monitor.updates}
    assert ("root", "RUNNING") in by_stage
    assert ("root", "SUCCESS") in by_stage
    assert ("middle", "RUNNING") in by_stage
    assert ("middle", "FAILURE") in by_stage
    assert {dataset for dataset, _, _, _ in monitor.updates} == {
        "unit_root",
        "unit_middle",
    }
    assert {status for _, _, status, _ in monitor.updates} <= {
        "RUNNING",
        "SUCCESS",
        "FAILURE",
    }


# --- the stdout event stream -------------------------------------------------


def test_every_task_emits_exactly_one_terminal_sentinel_on_stdout(tmp_path, capsys):
    handler = _write_dag(tmp_path)
    stages = _stages(["root", "middle", "leaf", "sibling"], {}, {})

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")
    events = _sentinel_events(capsys.readouterr().out)

    finished = [e for e in events if isinstance(e, TaskFinished)]
    assert sorted(e.name for e in finished) == ["leaf", "middle", "root", "sibling"]
    assert {e.status for e in finished} == {TaskStatus.COMPLETED}
    assert [e.name for e in events if isinstance(e, TaskStarted)] == [
        "root",
        "middle",
        "leaf",
        "sibling",
    ]
    assert isinstance(events[-1], RunFinished)
    assert events[-1].failed == outcome.failed_tasks


def test_a_stage_descriptor_absent_from_the_dag_is_reported(tmp_path):
    handler = _write_dag(tmp_path)
    stages = _stages(["root", "orphan"], {}, {})

    outcome = run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")

    assert outcome.statuses["orphan"] is TaskStatus.SKIPPED_UNREGISTERED
    assert outcome.exit_code == 0


# --- descriptor validation ---------------------------------------------------


def test_a_duplicate_stage_descriptor_raises(tmp_path):
    handler = _write_dag(tmp_path)
    stages = _stages(["root"], {}, {}) + _stages(["root"], {}, {})
    with pytest.raises(ValueError, match="declared more than once"):
        run_pipeline_stages(stages, handler, _StubMonitor(), _items(), "unit")


def test_a_malformed_stage_descriptor_raises(tmp_path):
    handler = _write_dag(tmp_path)
    with pytest.raises(ValueError, match=r"missing key\(s\) \['params'\]"):
        run_pipeline_stages(
            [{"name": "root", "func": _StubFlow()}],
            handler,
            _StubMonitor(),
            _items(),
            "unit",
        )


def test_a_config_without_bypass_is_rejected_at_plan_build(tmp_path):
    handler = _write_dag(tmp_path, FIXTURE_DAG.replace("    bypass: false\n", "", 1))
    with pytest.raises(ValueError, match="missing required key"):
        run_pipeline_stages(
            _stages(["root"], {}, {}), handler, _StubMonitor(), _items(), "unit"
        )
