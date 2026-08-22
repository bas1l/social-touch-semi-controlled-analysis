"""Ladder, isolation and exit-code tests for ``execute_dag``.

``run_task`` and ``emit`` are supplied by a plain duck-typed recorder: no
mocking library, no Prefect, no subprocess, no Qt.  Every row of the ladder and
every boundary between two rows gets its own case, because the row *order* is
the part of the design that is easy to break silently.
"""

import builtins
import pathlib

import pytest

from analysis.pipeline.dag_execution import RunOutcome, execute_dag, format_run_summary
from analysis.pipeline.dag_plan import DagPlan
from analysis.pipeline.execution_events import (
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
)


class _Recorder:
    """Duck-typed ``run_task`` / ``emit`` pair that records what it was asked."""

    def __init__(self, results=None, raises=None):
        self.results = dict(results or {})
        self.raises = dict(raises or {})
        self.invoked = []
        self.events = []

    def run_task(self, name):
        self.invoked.append(name)
        if name in self.raises:
            raise self.raises[name]
        return self.results.get(name, 0)

    def emit(self, event):
        self.events.append(event)

    # --- convenience views ---------------------------------------------------

    def finished(self, name):
        matches = [
            event
            for event in self.events
            if isinstance(event, TaskFinished) and event.name == name
        ]
        assert len(matches) == 1, f"{name}: expected 1 TaskFinished, got {len(matches)}"
        return matches[0]

    @property
    def started_names(self):
        return [e.name for e in self.events if isinstance(e, TaskStarted)]

    @property
    def run_finished(self):
        matches = [e for e in self.events if isinstance(e, RunFinished)]
        assert len(matches) == 1
        return matches[0]


class _Aborter:
    """``is_aborted`` that flips to True after *after_calls* polls."""

    def __init__(self, aborted=False, after_calls=None):
        self._aborted = aborted
        self._after_calls = after_calls
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._after_calls is not None and self.calls > self._after_calls:
            self._aborted = True
        return self._aborted


def _task(**overrides):
    task = {
        "category": "test",
        "enabled": True,
        "bypass": False,
        "options": {},
        "depends_on": [],
    }
    task.update(overrides)
    return task


def _plan(**tasks):
    return DagPlan.from_tasks(tasks)


def _assert_exactly_one_terminal_event_per_task(recorder, names):
    finished = [e for e in recorder.events if isinstance(e, TaskFinished)]
    assert [e.name for e in finished] == list(names)


# --- happy path --------------------------------------------------------------


def test_happy_path_runs_in_dependency_order_and_exits_zero():
    plan = _plan(c=_task(depends_on=["b"]), a=_task(), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert recorder.invoked == ["a", "b", "c"]
    assert outcome.statuses == {
        "a": TaskStatus.COMPLETED,
        "b": TaskStatus.COMPLETED,
        "c": TaskStatus.COMPLETED,
    }
    assert outcome.exit_code == 0
    assert outcome.aborted is False
    _assert_exactly_one_terminal_event_per_task(recorder, ["a", "b", "c"])
    assert recorder.run_finished == RunFinished(aborted=False, failed=(), skipped=())


def test_task_started_is_emitted_once_per_invoked_task_with_a_stable_total():
    plan = _plan(a=_task(), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    execute_dag(plan, recorder.run_task, recorder.emit)

    started = [e for e in recorder.events if isinstance(e, TaskStarted)]
    assert [(e.idx, e.total, e.name) for e in started] == [(1, 2, "a"), (2, 2, "b")]


def test_run_finished_is_the_last_event():
    plan = _plan(a=_task())
    recorder = _Recorder()
    execute_dag(plan, recorder.run_task, recorder.emit)
    assert isinstance(recorder.events[-1], RunFinished)


# --- row 6: failure isolation ------------------------------------------------


@pytest.mark.parametrize(
    "recorder_kwargs",
    [
        {"raises": {"b": RuntimeError("boom")}},
        {"results": {"b": 1}},
    ],
    ids=["raised", "non-zero-result"],
)
def test_a_failing_task_cascades_but_does_not_stop_the_run(recorder_kwargs):
    plan = _plan(
        a=_task(),
        b=_task(depends_on=["a"]),
        c=_task(depends_on=["b"]),
        unrelated=_task(),
    )
    recorder = _Recorder(**recorder_kwargs)

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert outcome.statuses["b"] is TaskStatus.FAILED
    assert outcome.statuses["c"] is TaskStatus.SKIPPED_DEP
    assert outcome.statuses["unrelated"] is TaskStatus.COMPLETED
    assert outcome.exit_code == 1
    assert "c" not in recorder.invoked
    assert "b" in recorder.finished("c").message


def test_a_failing_task_carries_its_exception_text_into_the_event():
    plan = _plan(a=_task())
    recorder = _Recorder(raises={"a": ValueError("depth_weight_alpha missing")})

    execute_dag(plan, recorder.run_task, recorder.emit)

    assert "depth_weight_alpha missing" in recorder.finished("a").message


def test_a_keyboard_interrupt_is_not_swallowed_as_a_task_failure():
    plan = _plan(a=_task())
    recorder = _Recorder(raises={"a": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        execute_dag(plan, recorder.run_task, recorder.emit)


# --- row 2 vs row 3: disabled beats bypass -----------------------------------


def test_disabled_beats_bypass():
    plan = _plan(a=_task(enabled=False, bypass=True), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert outcome.statuses["a"] is TaskStatus.SKIPPED_DISABLED
    assert "a" not in plan.completed_tasks
    assert outcome.statuses["b"] is TaskStatus.SKIPPED_DEP
    assert recorder.invoked == []
    assert outcome.exit_code == 0


def test_a_disabled_task_leaves_its_bypass_flag_untouched():
    plan = _plan(a=_task(enabled=False, bypass=True))
    recorder = _Recorder()
    execute_dag(plan, recorder.run_task, recorder.emit)
    assert plan.task("a").bypass is True


# --- row 3: bypass -----------------------------------------------------------


def test_bypass_runs_nothing_completes_the_task_and_unlocks_its_dependent():
    plan = _plan(a=_task(bypass=True), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert recorder.invoked == ["b"]
    assert "a" in plan.completed_tasks
    assert outcome.statuses["a"] is TaskStatus.BYPASSED
    assert outcome.statuses["b"] is TaskStatus.COMPLETED
    assert outcome.exit_code == 0


def test_bypass_emits_no_task_started():
    plan = _plan(a=_task(bypass=True), b=_task(depends_on=["a"]))
    recorder = _Recorder()
    execute_dag(plan, recorder.run_task, recorder.emit)
    assert recorder.started_names == ["b"]


def test_bypass_survives_a_failed_upstream_and_still_releases_its_dependent():
    # Row 3 before row 4: a bypass asserts something about the disk, not the run.
    plan = _plan(
        upstream=_task(),
        bypassed=_task(bypass=True, depends_on=["upstream"]),
        downstream=_task(depends_on=["bypassed"]),
    )
    recorder = _Recorder(raises={"upstream": RuntimeError("boom")})

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert outcome.statuses["upstream"] is TaskStatus.FAILED
    assert outcome.statuses["bypassed"] is TaskStatus.BYPASSED
    assert outcome.statuses["downstream"] is TaskStatus.COMPLETED
    assert outcome.exit_code == 1


def test_bypass_precedes_the_input_guard():
    # Row 3 before row 5: a guard that would fire is never consulted.
    plan = _plan(a=_task(bypass=True, min_inputs=10))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit, input_count=0)

    assert outcome.statuses["a"] is TaskStatus.BYPASSED


def test_bypass_on_a_root_and_on_a_leaf():
    plan = _plan(
        root=_task(bypass=True),
        middle=_task(depends_on=["root"]),
        leaf=_task(bypass=True, depends_on=["middle"]),
    )
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert recorder.invoked == ["middle"]
    assert outcome.statuses["root"] is TaskStatus.BYPASSED
    assert outcome.statuses["leaf"] is TaskStatus.BYPASSED
    assert outcome.exit_code == 0


def test_every_task_bypassed_invokes_nothing_and_says_so():
    plan = _plan(a=_task(bypass=True), b=_task(bypass=True, depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert recorder.invoked == []
    assert outcome.bypassed_tasks == ("a", "b")
    assert outcome.completed_tasks == ()
    assert "bypassed: a, b" in format_run_summary(outcome)


def test_the_bypass_path_performs_no_filesystem_access(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("the bypass path must not touch the filesystem")

    plan = _plan(a=_task(bypass=True), b=_task(bypass=True, depends_on=["a"]))
    recorder = _Recorder()

    monkeypatch.setattr(builtins, "open", _forbidden)
    for method in ("exists", "is_dir", "is_file", "stat", "glob", "iterdir", "open"):
        monkeypatch.setattr(pathlib.Path, method, _forbidden)

    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    assert outcome.bypassed_tasks == ("a", "b")


# --- row 5: input guards -----------------------------------------------------


def test_a_guard_violation_is_a_loud_skip_that_never_invokes_the_task():
    plan = _plan(a=_task(min_inputs=5), b=_task(depends_on=["a"]), other=_task())
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit, input_count=2)

    assert outcome.statuses["a"] is TaskStatus.SKIPPED_GUARD
    assert "at least 5" in recorder.finished("a").message
    assert "a" not in recorder.invoked
    assert outcome.statuses["b"] is TaskStatus.SKIPPED_DEP
    assert outcome.statuses["other"] is TaskStatus.COMPLETED
    assert outcome.exit_code == 0


def test_a_satisfied_guard_lets_the_task_run():
    plan = _plan(a=_task(min_inputs=1, max_inputs=3))
    recorder = _Recorder()
    outcome = execute_dag(plan, recorder.run_task, recorder.emit, input_count=2)
    assert outcome.statuses["a"] is TaskStatus.COMPLETED


def test_a_declared_guard_without_an_input_count_raises_rather_than_waving_it_through():
    plan = _plan(a=_task(min_inputs=1))
    recorder = _Recorder()
    with pytest.raises(ValueError, match="declares an input guard"):
        execute_dag(plan, recorder.run_task, recorder.emit)


def test_an_unguarded_task_needs_no_input_count():
    plan = _plan(a=_task())
    recorder = _Recorder()
    assert execute_dag(plan, recorder.run_task, recorder.emit).exit_code == 0


# --- row 1: unregistered stage descriptors -----------------------------------


def test_a_stage_descriptor_absent_from_the_dag_is_reported_not_run():
    plan = _plan(a=_task())
    recorder = _Recorder()

    outcome = execute_dag(
        plan, recorder.run_task, recorder.emit, task_names=["a", "orphan"]
    )

    assert outcome.statuses["orphan"] is TaskStatus.SKIPPED_UNREGISTERED
    assert "orphan" not in recorder.invoked
    assert "not declared in the DAG config" in recorder.finished("orphan").message
    assert outcome.exit_code == 0


def test_task_names_selects_a_subset_still_ordered_by_the_dag():
    plan = _plan(c=_task(depends_on=["b"]), a=_task(), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(
        plan, recorder.run_task, recorder.emit, task_names=["c", "b", "a"]
    )

    assert recorder.invoked == ["a", "b", "c"]
    assert set(outcome.statuses) == {"a", "b", "c"}


def test_duplicate_task_names_raise():
    plan = _plan(a=_task())
    recorder = _Recorder()
    with pytest.raises(ValueError, match="duplicate task"):
        execute_dag(plan, recorder.run_task, recorder.emit, task_names=["a", "a"])


# --- row 0: abort ------------------------------------------------------------


def test_abort_beats_bypass():
    plan = _plan(a=_task(bypass=True), b=_task(depends_on=["a"]))
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit, _Aborter(aborted=True))

    assert outcome.statuses["a"] is TaskStatus.ABORTED
    assert "a" not in plan.completed_tasks
    assert outcome.statuses["b"] is TaskStatus.ABORTED
    assert recorder.invoked == []
    assert outcome.aborted is True
    assert outcome.exit_code == 0


def test_abort_marks_the_running_task_and_every_pending_task():
    plan = _plan(a=_task(), b=_task(depends_on=["a"]), c=_task(depends_on=["b"]))
    recorder = _Recorder()
    # Poll 1 = row 0 for "a"; poll 2 = the post-run check for "a", which trips.
    aborter = _Aborter(after_calls=1)

    outcome = execute_dag(plan, recorder.run_task, recorder.emit, aborter)

    assert recorder.invoked == ["a"]
    assert outcome.statuses == {
        "a": TaskStatus.ABORTED,
        "b": TaskStatus.ABORTED,
        "c": TaskStatus.ABORTED,
    }
    assert outcome.aborted is True
    assert recorder.run_finished.aborted is True
    assert outcome.exit_code == 0


def test_an_aborted_run_still_emits_one_terminal_event_per_task():
    plan = _plan(a=_task(), b=_task(), c=_task())
    recorder = _Recorder()
    execute_dag(plan, recorder.run_task, recorder.emit, _Aborter(aborted=True))
    _assert_exactly_one_terminal_event_per_task(recorder, ["a", "b", "c"])


# --- outcome -----------------------------------------------------------------


def test_exit_code_is_zero_when_tasks_are_skipped_for_declared_reasons():
    plan = _plan(
        disabled=_task(enabled=False),
        dependent=_task(depends_on=["disabled"]),
        guarded=_task(max_inputs=0),
        bypassed=_task(bypass=True),
    )
    recorder = _Recorder()

    outcome = execute_dag(plan, recorder.run_task, recorder.emit, input_count=3)

    assert outcome.failed_tasks == ()
    assert set(outcome.skipped_tasks) == {"disabled", "dependent", "guarded"}
    assert outcome.exit_code == 0


def test_every_task_disabled_is_a_clean_zero_exit():
    plan = _plan(a=_task(enabled=False), b=_task(enabled=False))
    recorder = _Recorder()
    outcome = execute_dag(plan, recorder.run_task, recorder.emit)
    assert outcome.exit_code == 0
    assert recorder.invoked == []


def test_an_empty_plan_runs_cleanly():
    recorder = _Recorder()
    outcome = execute_dag(_plan(), recorder.run_task, recorder.emit)
    assert outcome.statuses == {}
    assert outcome.exit_code == 0
    assert "no tasks were visited" in format_run_summary(outcome)


def test_run_outcome_is_frozen():
    outcome = RunOutcome(statuses={}, aborted=False)
    with pytest.raises(Exception):
        outcome.aborted = True


def test_status_counts_tally_the_run():
    plan = _plan(a=_task(), b=_task(enabled=False), c=_task(bypass=True))
    recorder = _Recorder()
    outcome = execute_dag(plan, recorder.run_task, recorder.emit)
    assert outcome.status_counts == {
        TaskStatus.COMPLETED: 1,
        TaskStatus.SKIPPED_DISABLED: 1,
        TaskStatus.BYPASSED: 1,
    }


def test_format_run_summary_names_failures_and_the_exit_code():
    plan = _plan(a=_task(), b=_task(depends_on=["a"]))
    recorder = _Recorder(raises={"a": RuntimeError("boom")})
    outcome = execute_dag(plan, recorder.run_task, recorder.emit)

    summary = format_run_summary(outcome)
    assert "failed  : a" in summary
    assert "skipped : b" in summary
    assert "exit code: 1" in summary
