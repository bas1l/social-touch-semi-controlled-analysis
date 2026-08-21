# Plan: DAG Run Observability, Truthful Failure Reporting, and Task Bypass

**Date:** 2026-08-21
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/depth-weighted-iff-attribution`
**Branch:** `feature/gui-dag-observability`

> **Base-branch note.** This feature branches from
> `feature/depth-weighted-iff-attribution` rather than `dev`, at the user's direction.
> Every premise this plan rests on — the 35-task processing DAG, the
> `spatial_extract_boundary__{radial,gradient,inflection}` fan-out and its
> `spatial_extract_boundaries` barrier, `boundary/registry.py`,
> `tests/test_rf_boundary_dag_wiring.py`, and the deferred TODO at
> `task_detail_panel.py:193-202` — arrives with
> `feature/rf-boundary-plugin-architecture` (one commit ahead of `dev`) and is absent from
> `dev`. Basing on `dev` would make Phase 0 void and the barrier-stall success criteria
> unverifiable. The PR therefore targets `feature/depth-weighted-iff-attribution`, and this
> work lands only once that branch does.

---

## Overview

Port the DAG-observability and task-gating mechanisms that were developed downstream in
`population-synthetic` back into this repo's `AnalysisRunnerGUI` and DAG execution engine.
The workflow subprocess gains a structured per-task event stream, the GUI paints each DAG
node with its live run status, a run that contains a failed task stops reporting itself as
successful, and a new per-task `bypass` flag lets a downstream slice be re-run without
re-deriving upstream outputs that already exist on disk.

## Problem Statement

The GUI currently runs the whole DAG as one opaque subprocess and scrapes its stdout as
undifferentiated text. Four concrete consequences, all verified in the current tree:

1. **A failed run reports success.** `_vendor_task_executor.py:47` returns `True` from
   `__exit__`, suppressing every exception once a task has started.
   `stage_runner.run_pipeline_stages` discards the resulting `executor.error_msg`, returns
   `None`, and `analysis_workflow_processing.main()` never calls `sys.exit`. A run in which
   ten tasks failed exits 0, and `runner_window._poll_process` prints
   `"Finished (exit code 0)"`. The only way to learn otherwise is to read the console
   backlog for `❌` lines.
2. **No per-task visibility.** The DAG graph — the natural place for a run report — is
   static during a run. `PipelineMonitor` is constructed with `data_queue=Queue()`
   (`analysis_workflow_processing.py:1859`), which sets `_is_client = True` and returns at
   `pipeline_monitor.py:42`: no coordinator thread, no `DataManager`, no Excel report.
   `show_dashboard()` is dead code, called from nowhere in the repo. The single artefact
   that reaches the GUI is the `print(f"[{dataset}] -> {stage}: {status}")` at
   `pipeline_monitor.py:135`.
3. **Silent dependency stalls.** Disabling one `spatial_extract_boundary__*` node makes the
   `spatial_extract_boundaries` barrier permanently unrunnable — `can_run` needs every
   `depends_on` entry in `completed_tasks`, and a disabled task is never marked completed.
   Four downstream stages then no-op **with no message at all**, because
   `DagConfigHandler.can_run` prints nothing on the disabled/unmet path. This is documented
   as a deferred follow-up at `task_detail_panel.py:193-202`.
4. **No way to re-run a downstream slice.** `enabled: false` is the only "don't run this"
   control, and it starves dependents. Re-running one late stage means re-running its whole
   upstream chain, or hand-invoking the script outside the GUI — which loses the GUI's
   option translation and any guard.

Separately, `_on_abort` calls `self._process.terminate()` (`runner_window.py:438`), which
signals only the direct child. That child spawns Prefect flow subprocesses; aborting a run
leaves grandchildren alive.

## Goals

### In Scope

1. A Qt-free DAG execution core in the child process that walks tasks in dependency order
   and emits typed events for every task, with a thin adapter binding it to the real
   Prefect flows.
2. A structured status channel from the workflow subprocess to the GUI over stdout, and a
   Qt-free parser for it.
3. Live per-node run-status rendering on `DagGraphView` (colour, border, glyph, dimming),
   plus an end-of-run summary and a status-bar outcome that reflects task outcomes.
4. A truthful process exit code: non-zero when any task failed.
5. Process-tree kill on abort, on both Windows and POSIX, including the process-group
   setup at `Popen` time that makes the POSIX branch actually work.
6. A per-task `bypass` flag: persisted in the DAG YAML, editable from the graph node and
   the task table, honoured by the execution core (marks completed, runs nothing, verifies
   nothing), and never applied without a confirmation modal that defaults to Cancel.
7. `set_task_dependencies` on `DagConfigModel` — the missing mutation API — supplied but
   deliberately **not** wired to the enabled-toggle (see Alternatives).
8. Optional `min_inputs` / `max_inputs` per-task guards on the number of discovered input
   items, producing a loud skip rather than a malformed run.
9. A task registry owning `label` and `description` per canonical task id, consumed by the
   GUI detail panel and validated against the DAG configs by a conformance test.
10. GUI polish ported from the reference: grid-snap dragging with a matching background
    grid, a click-vs-drag threshold, non-clearing checkbox interlocks, and node width that
    accounts for the checkbox row.
11. Headless pytest coverage of dependency gating, bypass, guards, failure isolation,
    abort, exit-code derivation, and event encode/decode round-trip.

### Out of Scope

- Replacing or removing `_vendor/monitoring/PipelineMonitor`. It stays as-is; this plan
  neither revives `show_dashboard()` nor deletes it. (Noted as a follow-up.)
- Migrating output-directory ownership into the task registry. `analysis/pipeline/output_dirs.py`
  keeps owning paths; the registry owns display metadata only. Re-pointing 35 tasks at a
  registry-derived folder is a separate migration with data-layout consequences.
- Persisting run statuses across GUI restarts. Statuses are transient by design.
- Any change to `src/_vendor/`. `TaskExecutor` is consumed only by `stage_runner.py`, so it
  can be superseded without editing vendored files; `DagConfigHandler` has three other
  consumers (`scripts/analysis_workflow_viewers.py`, `scripts/scan_edge_rfs.py`,
  `tests/test_depth_weight_alpha_config.py`) and keeps its current behaviour untouched.
- Parallel task execution, retries, scheduling, or a run queue.
- A "start from this node" run mode. The per-node bypass primitive composes into it later.
- Bypass for the viewers DAG's interactive tasks beyond adding the key for schema uniformity.
- Any `QApplication`-instantiating test. The repo has no `pytest-qt` and this plan does not
  introduce it.

## Success Criteria

- [ ] A run in which at least one task raises exits non-zero, and the GUI status bar reads
      `Failed` naming the count of failed tasks.
- [ ] Every task in the DAG receives exactly one terminal status event per run, and every
      node on the graph is repainted exactly once accordingly.
- [ ] Disabling a `spatial_extract_boundary__*` node produces a visible `SKIPPED_DEP`
      cascade — amber dimmed nodes plus a named console line for each — instead of a silent
      no-op.
- [ ] Setting `bypass: true` on that same node lets the barrier and all four downstream
      stages run, with the bypassed node painted violet and never green.
- [ ] Running with any enabled+bypassed task shows a modal listing them, defaulting to
      Cancel; cancelling leaves the previous run's node statuses on screen untouched.
- [ ] Abort terminates the whole process tree: no `python.exe` descendant of the aborted run
      survives, verified on Windows via `taskkill` and asserted in a unit test on POSIX via
      a spawned grandchild.
- [ ] `pytest tests/ -k "dag_execution or execution_events or status_channel or bypass"`
      passes with no `QApplication` constructed anywhere.
- [ ] `tests/test_rf_boundary_dag_wiring.py` passes in full, including the four assertions
      that fail today.
- [ ] Every task in all three DAG configs declares `bypass`, asserted by a conformance test.
- [ ] No filesystem access whatsoever on the bypass code path — greppable: no `exists`,
      `is_dir`, `stat`, `glob`, or `open` between the bypass branch and its `continue`.

## Definitions

- **Bypassed:** all three of — zero subprocess/flow invocations for that task, the task's
  name present in `completed_tasks`, and its status equal to `BYPASSED`. Any one of the
  three missing means the feature is not implemented.
- **No verification:** the bypass path performs no filesystem access at all. Not "checks
  cheaply", not "warns if missing" — nothing is read, stat'ed, or globbed. This is a
  reviewable property of the diff, not an aspiration.
- **Inert flag:** a `bypass` value that is read, found irrelevant (because the task is
  disabled), and neither acted upon nor mutated. A disabled task's persisted `bypass: true`
  survives untouched on disk and in the widget.
- **Truthful outcome:** the process exit code is non-zero if and only if at least one task
  reached `FAILED`. A task skipped for a declared reason (disabled, dependency, guard,
  bypass) is not a failure.
- **Terminal status:** one of `COMPLETED`, `BYPASSED`, `FAILED`, `SKIPPED_DISABLED`,
  `SKIPPED_DEP`, `SKIPPED_GUARD`, `SKIPPED_UNREGISTERED`, `ABORTED`. Exactly one is emitted
  per task per run.
- **Qt-free:** the module's transitive import graph contains no `PyQt5`. Enforced by a test
  that imports the module with `PyQt5` absent from `sys.modules` and asserts it stays absent.
- **Failure isolation:** a task raising does not stop the run; its dependents cascade to
  `SKIPPED_DEP` and unrelated branches continue. This existing behaviour is preserved —
  what changes is that the outcome is now reported rather than swallowed.

---

## Technical Design

### Approach

The reference implementation runs its execution core **inside the GUI process** and spawns
one subprocess per task. This repo cannot copy that shape: the GUI spawns a single child
(`analysis_workflow_processing.py`) that runs the entire Prefect DAG in-process. The port
therefore relocates the core rather than transplanting it.

```
  reference (population-synthetic)          this repo (after this plan)
  ────────────────────────────────          ──────────────────────────────
  GUI process                               GUI process
    execute_workflow()  ── Qt-free            status_channel.parse_line()  ── Qt-free
      └─ run_cmd -> Popen per task            └─ DagGraphView.set_task_status
                                                        ▲
                                            ────────────┼──── stdout sentinel lines
                                                        │
                                            child process (one per run)
                                              execute_dag()  ── Qt-free
                                                └─ run_task -> Prefect flow (in-process)
```

The Qt-free-core / thin-adapter split is preserved exactly; only the process boundary moves.
The core (`execute_dag`) takes injected `run_task`, `emit`, and `is_aborted` callables, so it
is testable with plain duck-typed recorders and no Prefect, no subprocess, and no Qt — the
same trick that makes three of the reference's four test files display-free.

Events cross the process boundary as one-line JSON sentinels on stdout. This reuses the
channel that already exists and already works (`ProcessOutputReader` streams the child's
merged stdout/stderr into the GUI thread over a Qt signal), rather than standing up a second
IPC mechanism.

**Guide alignment.** Per `~/.claude/knowledge/data-pipeline-engineering/01`, this is Axis-3
DAG orchestration, whose stated reason to exist includes *"operational monitoring"* — so
task status is designed as an explicit state machine, not as a by-product of whether a
function returned. Guide `02` §8 — *"Decide deliberately where errors are fatal and where
they are tolerated, and make both explicit. Default to failing loudly on unexpected
conditions rather than silently substituting a default"* — is the direct grounding for the
exit-code change: a runner reporting exit 0 despite a failed node is the orchestration
equivalent of a pipeline emitting a silently-wrong number. Guide `02` §3 justifies frozen
dataclasses as the event contract; §2's one-way import rule is what "Qt-free" means
concretely. Guide `05`'s DIP is why the core depends only on the event abstraction. Guide
`03` (statistical software) does not apply — this feature computes nothing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Status channel: JSON sentinel lines on stdout** | Reuses the working `ProcessOutputReader` path; survives in the run log for post-hoc replay; trivially testable as a pure string→event function | Must not collide with ordinary output; interleaves with tqdm `\r` frames | **Chosen** |
| Status channel: revive `PipelineMonitor`'s `multiprocessing.Queue` | Machinery already written | The GUI does not own the child's queue; client mode is inert by construction; would need a second transport to reach the GUI anyway | Rejected |
| Status channel: a status JSON file polled by the GUI | Simple; no parsing of a text stream | Adds disk I/O per transition; racy on partial writes; loses ordering against console output | Rejected |
| **Fix the barrier stall with `bypass`** | Fixes the user's actual need (re-run a downstream slice); leaves `depends_on` static | Requires a new YAML key on every task | **Chosen** |
| Fix the barrier stall by auto-rewriting the barrier's `depends_on` when a method node is toggled off | Matches the literal shape of the deferred TODO | `tests/test_rf_boundary_dag_wiring.py::test_barrier_depends_on_exactly_the_method_nodes` asserts **exact set equality** of the barrier's `depends_on` against the registry's method nodes — pruning on disk breaks a green, correct test. It also couples barrier/method node identity into generic GUI code, which is precisely what the TODO warned against. And it mutates the DAG's declared topology to express a per-run choice | Rejected |
| Make `can_run` treat disabled dependencies as satisfied | One-line fix | Silently reinterprets "disabled" as "assume done" for every task in the repo, with no confirmation and no distinct status. Exactly the silent degradation the fail-fast policy forbids | Rejected |
| **Report `SKIPPED_DEP` loudly and leave the topology alone** | Turns a silent stall into a diagnosable one; costs nothing | Does not by itself let the user proceed | **Chosen, alongside bypass** — visibility and remedy are separate concerns |
| `bypass` as a **required** YAML key | An absent key is a genuine config error; matches fail-fast policy; makes a bad merge fail at load | A branch that adds a task node merges cleanly as text but produces a config that raises — a semantic conflict git cannot see | **Chosen**, with a conformance test so the failure surfaces in CI rather than at run time |
| `bypass` optional, defaulting to `false` | No merge hazard | A silent default on a flag whose failure mode is "silently runs against stale data" is the wrong place to be lenient | Rejected |
| Reuse `COMPLETED` for bypassed tasks | No new enum member | Destroys the audit trail — the run report would claim work happened that did not, indistinguishable in console and graph | Rejected |
| Transient, session-only bypass | Cannot be forgotten across runs | Inconsistent with `enabled`/`force`, which both persist; a multi-day sweep would re-tick every run; the round-trip model exists to persist node state | Rejected |
| Cheap output-directory existence check before honouring a bypass | Feels safer | Dispatch granularity means a non-empty directory says nothing about *which* sessions are present — reassurance without a guarantee, which is worse than none. Mitigation is the modal, the banner, and the violet node | Rejected |
| Edit `_vendor/_vendor_task_executor.py` in place | Smallest diff | Diverges from the parent repo's vendored copy for a change that is repo-specific | Rejected — `TaskExecutor` has exactly one consumer, so a superseding non-vendored core is cleaner |
| Port the reference's `os.killpg` branch verbatim | Copy-paste | **It is broken there.** Neither Popen call site sets `start_new_session`/`creationflags`, so `os.getpgid(child)` returns the GUI's own group and `killpg` would signal the GUI itself. Unnoticed because that workstation is Windows-only | Rejected — port the intent, fix the defect |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `analysis/pipeline/execution_events.py` (new) | Define `TaskStatus` and the frozen event dataclasses; encode/decode one event ↔ one sentinel line | event ↔ `str` | Qt, subprocess, Prefect, filesystem, YAML |
| `analysis/pipeline/dag_plan.py` (new) | Parse a DAG config mapping into a validated, immutable plan: task keys, `depends_on`, `bypass`, guards; Kahn topological order with authoring-order tie-break; `can_run` / `mark_completed` / `mark_bypassed` / `guard_violation` | `Mapping` snapshot → `DagPlan` | Qt, Prefect, subprocess, stdout, file paths |
| `analysis/pipeline/dag_execution.py` (new) | Walk the plan's ladder, emit exactly one terminal event per task, aggregate a `RunOutcome` | `(DagPlan, run_task, emit, is_aborted)` → `RunOutcome` | Qt, Prefect, subprocess, stdout, YAML |
| `analysis/pipeline/stage_runner.py` (rewritten) | Adapter: bind stage descriptors + `PipelineMonitor` + Prefect flows to the core; print sentinels; return the outcome | `(stages, DagConfigHandler, monitor, items, prefix)` → `RunOutcome` | Qt, GUI widgets, node coordinates |
| `analysis/pipeline/task_registry.py` (new) | Own `label`/`description` per canonical task id; fail loudly on an unregistered id | `task_id` → `TaskMeta` | Qt, DAG config, output paths |
| `utils/gui/analysis_runner_gui/status_channel.py` (new) | Turn one stdout line into an event or `None`; raise on a malformed sentinel | `str` → `event \| None` | Qt, widgets, subprocess, the DAG |
| `utils/gui/analysis_runner_gui/process_tree.py` (new) | Create a process-group-isolated child; kill a whole tree | `Popen` → `None`; kwargs for `Popen` | Qt, the DAG, task semantics |
| `dag_graph_items.py` / `dag_graph_view.py` (modified) | Paint a node for a given `TaskStatus`; expose `set_task_status` / `clear_task_statuses` | `(name, TaskStatus)` → repaint | How the status was transported; subprocess; YAML |
| `runner_window.py` (modified) | Wire the parser into the reader signal; own the confirm modal, the run summary, and abort | signals → widget calls | Ladder semantics, event encoding |

```
src/analysis/pipeline/
├── execution_events.py      # TaskStatus, ConsoleLine, TaskStarted, TaskFinished,
│                            # RunFinished, encode_event(), decode_event()
├── dag_plan.py              # DagTask, DagPlan (parse/validate/order/gate/guard)
├── dag_execution.py         # execute_dag(plan, run_task, emit, is_aborted) -> RunOutcome
├── task_registry.py         # TaskMeta, get_task_meta(id), load_registry()
└── stage_runner.py          # run_pipeline_stages(...) -> RunOutcome   [rewritten adapter]

src/utils/gui/analysis_runner_gui/
├── status_channel.py        # parse_status_line(line) -> event | None
└── process_tree.py          # popen_group_kwargs(), kill_process_tree(proc)

configs/
└── analysis_task_registry.yaml   # canonical task id -> {label, description}
```

Key signatures:

```python
# execution_events.py
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BYPASSED = "bypassed"
    FAILED = "failed"
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_DEP = "skipped_dep"
    SKIPPED_GUARD = "skipped_guard"
    SKIPPED_UNREGISTERED = "skipped_unregistered"
    ABORTED = "aborted"

SENTINEL_PREFIX = "##DAG-EVENT "

@dataclass(frozen=True)
class TaskStarted:  idx: int; total: int; name: str
@dataclass(frozen=True)
class TaskFinished: name: str; status: TaskStatus; message: str = ""
@dataclass(frozen=True)
class RunFinished:  aborted: bool; failed: tuple[str, ...]; skipped: tuple[str, ...]

def encode_event(event: object) -> str: ...          # -> "##DAG-EVENT {json}"
def decode_event(payload: str) -> object: ...        # raises ValueError on malformed

# dag_execution.py
@dataclass(frozen=True)
class RunOutcome:
    statuses: Mapping[str, TaskStatus]
    aborted: bool
    @property
    def exit_code(self) -> int: ...   # 1 iff any status is FAILED, else 0

def execute_dag(plan, run_task, emit, is_aborted=lambda: False) -> RunOutcome: ...

# status_channel.py
def parse_status_line(line: str) -> object | None:
    """Return the decoded event, or None when *line* is ordinary output.

    Raises ValueError when the line carries the sentinel prefix but the payload
    is malformed — a producer bug that must not be swallowed.
    """
```

**The ladder** (evaluated per task, in order, `continue` after each; exactly one terminal
event emitted per row):

| # | Condition | Status | Marks completed |
|---|-----------|--------|-----------------|
| 0 | `is_aborted()` | `ABORTED` | no |
| 1 | task name absent from the DAG registry | `SKIPPED_UNREGISTERED` | no |
| 2 | `not enabled` | `SKIPPED_DISABLED` | no |
| 3 | `bypass` | `BYPASSED` | **yes** |
| 4 | a dependency did not complete | `SKIPPED_DEP` | no |
| 5 | `min_inputs`/`max_inputs` violated | `SKIPPED_GUARD` | no |
| 6 | ran; exit/exception | `COMPLETED` / `FAILED` | yes / no |

Ordering rationale, carried over from the reference and worth restating because each row
boundary gets its own test: `enabled` stays the master switch (2 before 3), so a disabled
task's `bypass` is inert. A bypass asserts something about **the disk**, not about this run,
so it is immune to upstream failure and to the input guard (3 before 4 and 5). Abort beats
everything (0 first). Row 1 preserves today's `logging.warning` for a stage descriptor with
no DAG entry, but now as a reported status rather than a bare log line.

**Failure isolation is preserved.** Row 6 catches the exception, records `FAILED`, and
continues to the next task — the existing per-task resilience. What changes: the failure is
now recorded in `RunOutcome`, surfaced as an event, and reflected in the exit code.

**Malformed-sentinel policy.** `decode_event` raises. `runner_window`'s slot catches that one
`ValueError`, writes a visible `!! MALFORMED STATUS EVENT: …` line to the console, marks the
run as errored in the status bar, and continues consuming output. A hard crash inside a Qt
slot mid-run would destroy the console backlog the user needs to diagnose it; this is the
deliberate error boundary that guide `02` §8 asks to be made explicit, not a silent fallback
— the condition is loud, visible, and changes the reported outcome.

**Console filtering.** Sentinel lines are consumed by the parser and **not** appended to
`ConsoleWidget`. They *are* written verbatim to the run log by `RunLogFile`, so a past run
can be replayed into the graph later.

**Bypass is GUI-and-config only.** It emits no CLI flag and no script change; scripts remain
unaware of it, exactly as `enabled` is today.

---

## Implementation Plan

### Phase 0: Repair the failing conformance tests
**Goal:** Start from green, since later phases extend this file.

- [x] 0.1 — Fix `tests/test_rf_boundary_dag_wiring.py::_load_layout` to return the `"nodes"`
      sub-mapping (or fix the four assertions to index it), resolving
      `test_method_node_has_layout[radial|gradient|inflection]` and `test_barrier_has_layout`.
- [x] 0.2 — Confirm the full suite's pre-existing status and record the baseline in the
      commit message, so later phases are not blamed for unrelated failures.

**Files Modified:**
- `tests/test_rf_boundary_dag_wiring.py` — index `_LAYOUT["nodes"]`.

**Dependencies:** None

### Phase 1: Qt-free execution core and event contract
**Goal:** The child process walks the DAG through a testable pure core and reports a truthful exit code.

- [x] 1.1 — Add `execution_events.py`: `TaskStatus`, the four frozen event dataclasses,
      `SENTINEL_PREFIX`, `encode_event`, `decode_event`.
- [x] 1.2 — Add `dag_plan.py`: `DagTask`, `DagPlan` with strict key validation (required:
      `category`, `enabled`, `bypass`, `options`, `depends_on`; optional: `min_inputs`,
      `max_inputs`), unknown-key rejection, unknown-dependency rejection, cycle detection
      naming the members, Kahn ordering with YAML authoring order as tie-break,
      `can_run` / `mark_completed` / `mark_bypassed` / `guard_violation`.
- [x] 1.3 — Add `dag_execution.py`: `RunOutcome` and `execute_dag` implementing the 7-row
      ladder, emitting exactly one terminal event per task.
- [x] 1.4 — Rewrite `stage_runner.run_pipeline_stages` as the adapter: build a `DagPlan`
      from the `DagConfigHandler`'s already-loaded mapping, inject a `run_task` that opens
      the existing per-task work (lazy `params()`, common kwargs, Prefect flow call) and
      converts an exception into a non-zero result, inject an `emit` that prints
      `encode_event(...)` with `flush=True` **and** forwards to `PipelineMonitor.update` so
      the existing status vocabulary keeps working. Return the `RunOutcome`.
- [x] 1.5 — `analysis_workflow_processing.main()` and `analysis_workflow_viewers.main()`
      return `outcome.exit_code`; both entry guards become `sys.exit(main())`. Print a
      final summary block (counts per status; named failed and skipped tasks).
- [x] 1.6 — Add `bypass: false` to all 35 tasks in `configs/analyse_workflow_processing_dag.yaml`,
      all 8 in `analyse_workflow_viewers_dag.yaml`, and all 15 in the unused
      `analyse_workflow_dag.yaml`, inserted immediately after `enabled` to preserve the
      repo-wide `category, enabled, bypass, options, depends_on` key order.
- [x] 1.7 — Add `min_inputs`/`max_inputs` support in the plan parser (no config declares
      them yet; the keys are optional).

**Files Modified:**
- `src/analysis/pipeline/execution_events.py` — new.
- `src/analysis/pipeline/dag_plan.py` — new.
- `src/analysis/pipeline/dag_execution.py` — new.
- `src/analysis/pipeline/stage_runner.py` — rewritten as a thin adapter.
- `src/analysis/pipeline/__init__.py` — export the new names.
- `scripts/analysis_workflow_processing.py` — `main()` returns an exit code; `sys.exit(main())`; summary block.
- `scripts/analysis_workflow_viewers.py` — same.
- `configs/analyse_workflow_processing_dag.yaml`, `configs/analyse_workflow_viewers_dag.yaml`, `configs/analyse_workflow_dag.yaml` — add `bypass` to every task.

**Dependencies:** Phase 0

### Phase 2: GUI status channel and live node rendering
**Goal:** The DAG graph becomes the run report.

- [ ] 2.1 — Add `status_channel.parse_status_line`.
- [ ] 2.2 — `DagTaskNode`: add `self._status`, `set_status()`, a `_STATUS_OVERLAY` table
      keyed by `TaskStatus` giving `(fill|None, border, width, dim, glyph)`, and paint it —
      keeping the existing category badge and the selection stroke on separate visual
      channels (status glyph at `rect.right() - 22`, category badge stays at `- 12`).
      Widen `boundingRect` for the 3px `RUNNING` border.
- [ ] 2.3 — `DagGraphView`: add `set_task_status(name, status)` (forgiving `.get`, no-op on
      unknown) and `clear_task_statuses()`. `populate()` resets statuses, since it rebuilds
      `_nodes`. Statuses must never touch `_save_layout`.
- [ ] 2.4 — `TaskPanel`: pass-through `set_task_status` / `clear_task_statuses` to the graph;
      show the status as a new read-only column in the table view.
- [ ] 2.5 — `runner_window`: connect `reader.line_received` to a `_on_output_line` slot that
      calls the parser first — sentinel lines route to the graph and are withheld from the
      console; everything else goes to the console as today. Clear statuses at run start.
      Handle the malformed-sentinel `ValueError` per the policy above.
- [ ] 2.6 — `_poll_process`: derive the status-bar message from the `RunFinished` event when
      one was seen, falling back to the exit code. Report `Failed — N task(s) failed` and
      name them in the console summary.

**Files Modified:**
- `src/utils/gui/analysis_runner_gui/status_channel.py` — new.
- `src/utils/gui/analysis_runner_gui/dag_graph_items.py` — status field, overlay table, paint.
- `src/utils/gui/analysis_runner_gui/dag_graph_view.py` — status API.
- `src/utils/gui/analysis_runner_gui/task_panel.py` — pass-through + status column.
- `src/utils/gui/analysis_runner_gui/runner_window.py` — parser wiring, run summary.

**Dependencies:** Phase 1

### Phase 3: Abort with process-tree kill
**Goal:** Aborting a run leaves nothing behind.

- [ ] 3.1 — Add `process_tree.py`: `popen_group_kwargs()` returning
      `{"creationflags": CREATE_NEW_PROCESS_GROUP}` on Windows and
      `{"start_new_session": True}` on POSIX; `kill_process_tree(proc, grace_seconds)` using
      `taskkill /F /T /PID` on Windows and `killpg(SIGTERM)` then `killpg(SIGKILL)` after the
      grace period on POSIX, with an early-out when `proc.poll()` is not None.
- [ ] 3.2 — `_on_run` creates the child with `**popen_group_kwargs()`.
- [ ] 3.3 — `_on_abort` calls `kill_process_tree`.
- [ ] 3.4 — `closeEvent` uses the same path before the discard dialog, so a grandchild never
      outlives the window.

**Files Modified:**
- `src/utils/gui/analysis_runner_gui/process_tree.py` — new.
- `src/utils/gui/analysis_runner_gui/runner_window.py` — `_on_run`, `_on_abort`, `closeEvent`.

**Dependencies:** Phase 2 (shares the run lifecycle edits; independent in principle)

### Phase 4: Bypass, end to end
**Goal:** A downstream slice can be re-run without re-deriving its upstream.

- [ ] 4.1 — `DagConfigModel`: add `is_task_bypassed` / `set_task_bypassed`, using
      `CommentedMap.insert` so a newly written key lands after `enabled` rather than at the
      end of the task map. Add `set_task_dependencies` (writes a `CommentedSeq`, preserving
      flow style where the existing value used it).
- [ ] 4.2 — `DagTaskNode`: add the `Bypass` checkbox and a `bypass_changed` signal;
      `_apply_interlocks` greys `Bypass` when `Enabled` is off and greys `Force` when
      `Bypass` is on — **`setEnabled` only, never `setChecked`**, so an inert flag survives
      untouched. Re-measure the node width from `inner.sizeHint()` after assembling the
      checkbox row, since three checkboxes exceed `_MIN_NODE_W = 180`.
- [ ] 4.3 — `DagGraphView`: re-emit `bypass_changed`.
- [ ] 4.4 — `TaskPanel`: add the table-view `Bypass` checkbox and `_make_bypass_handler`;
      **replace the `cb.text() == "Force"` / else-is-enabled dispatch in
      `_sync_table_checkboxes` with an explicit role-keyed dict** — with a third checkbox the
      current `else` branch would overwrite `Bypass` with the enabled state.
- [ ] 4.5 — `runner_window`: `_confirm_bypasses` — a modal listing every enabled+bypassed
      task, stating plainly that nothing is checked, with Cancel as the default button.
      Called before the statuses are cleared, so cancelling preserves the prior run report.
- [ ] 4.6 — Verify by inspection and by test that the bypass path in `dag_execution`
      performs no filesystem access.

**Files Modified:**
- `src/utils/pipeline/dag_config_model.py` — bypass accessors, `set_task_dependencies`.
- `src/utils/gui/analysis_runner_gui/dag_graph_items.py` — Bypass checkbox, interlocks, width.
- `src/utils/gui/analysis_runner_gui/dag_graph_view.py` — signal pass-through.
- `src/utils/gui/analysis_runner_gui/task_panel.py` — table checkbox, sync fix.
- `src/utils/gui/analysis_runner_gui/runner_window.py` — confirmation modal.

**Dependencies:** Phase 1 (core honours `bypass`), Phase 2 (violet node rendering)

### Phase 5: Task registry and GUI polish
**Goal:** Task descriptions come from one owner; the graph handles like the reference.

- [ ] 5.1 — Add `configs/analysis_task_registry.yaml` mapping canonical task id →
      `{label, description}` for all tasks across both live DAGs.
- [ ] 5.2 — Add `task_registry.py` with an `lru_cache`d loader (the reference re-parses on
      every call — do not copy that), a frozen `TaskMeta`, and `get_task_meta(id)` raising
      `KeyError` naming the registry file on an unregistered id.
- [ ] 5.3 — `TaskDetailPanel`: pin a read-only, mouse-selectable description label under the
      header in `__init__`, so it survives the clear loop and the option-less early return.
      Source it from the registry.
- [ ] 5.4 — Grid-snap dragging: shared `GRID_SIZE = 20`, snap in `itemChange`
      (`ItemPositionChange`, returning the corrected point), and a matching faint background
      grid drawn in `DagGraphView.drawBackground` with a zero-width cosmetic pen, batched
      `drawLines`, over the exposed rect only.
- [ ] 5.5 — Click-vs-drag: a 4-unit Manhattan threshold in scene coordinates; emit
      `node_clicked` on release only when undragged, latching `_dragged` once tripped.
      Today `node_clicked` fires on press, so nudging a node steals focus to the detail panel.
- [ ] 5.6 — Add the `_in_node_moved` re-entrancy guard in `_on_node_moved`; the reference
      shipped this as a fix for a startup `RecursionError` when `_load_layout` sets positions.
- [ ] 5.7 — Declare `min_inputs`/`max_inputs` on the tasks that need them (candidates
      identified during implementation; none is mandatory to ship the mechanism).
- [ ] 5.8 — Replace `DagGraphView._load_layout`'s `saved.get("nodes", {})` with a loud
      failure on a sidecar that has no `"nodes"` mapping. That silent default is what turned
      the flat-to-nested schema change into invisible loss of hand-placed node positions:
      the GUI auto-laid-out every launch and overwrote the file on the first drag. A layout
      file that exists but cannot be read is a fail-fast condition, not a fallback to
      auto-layout. (Repairing the data was Phase 0; this closes the hole that hid it.)

**Files Modified:**
- `configs/analysis_task_registry.yaml` — new.
- `src/analysis/pipeline/task_registry.py` — new.
- `src/utils/gui/analysis_runner_gui/task_detail_panel.py` — description label.
- `src/utils/gui/analysis_runner_gui/dag_graph_items.py` — grid snap, drag threshold.
- `src/utils/gui/analysis_runner_gui/dag_graph_view.py` — background grid, re-entrancy guard, loud layout-load failure.

**Dependencies:** Phase 2

---

## Testing Plan

All new tests are headless. No `QApplication` is constructed anywhere, matching the existing
convention (`tests/test_task_detail_panel_boundary_schema.py` exercises only pure/static
helpers; the repo has no `pytest-qt`). Widget-module tests import module-level data only,
guarded by `pytest.importorskip("PyQt5")`.

### Unit Tests

`tests/test_execution_events.py`
- [ ] Every `TaskStatus` member round-trips through `encode_event`/`decode_event`.
- [ ] Each event dataclass round-trips with fields intact.
- [ ] An encoded event is exactly one line — no embedded newline can split a sentinel.
- [ ] `decode_event` raises `ValueError` on truncated JSON, on an unknown event type, and on
      an unknown status string.

`tests/test_dag_plan.py`
- [ ] Missing required key raises, naming the key — one case per required key, `bypass` included.
- [ ] Unknown key raises, naming it (typo `depend_on`).
- [ ] Non-boolean `bypass` raises.
- [ ] Unknown `depends_on` target raises, naming it.
- [ ] Two-node cycle and self-loop raise, naming the members.
- [ ] `min_inputs > max_inputs` raises.
- [ ] `enabled: false` + `bypass: true` is legal and inert (asserted by *not* raising).
- [ ] Ordering is deterministic across two independent builds of the same config.
- [ ] Ordering against the real shipped `analyse_workflow_processing_dag.yaml`: exact task-set
      equality, then pairwise `index(a) < index(b)` for the boundary chain and each barrier
      consumer — constraining topology without over-fitting the tie-break.
- [ ] `mark_bypassed` puts the name in `completed_tasks` **and** sets status `BYPASSED` (both halves).
- [ ] A guard skip does not mark completed, and dependents stay blocked.
- [ ] Unknown task name raises `KeyError` from all four of `can_run`, `mark_completed`,
      `mark_bypassed`, `guard_violation`.
- [ ] Guard message shapes: `min == max` collapses to "exactly"; min-only; max-only; unbounded never violates.

`tests/test_dag_execution.py` — with a `_Recorder` supplying duck-typed `run_task`/`emit`, no Prefect, no subprocess
- [ ] Happy path: dependency-ordered completion, one `TaskFinished` per task, `exit_code == 0`.
- [ ] A failing task yields `FAILED`, its dependents `SKIPPED_DEP` naming the unmet dep, an
      unrelated branch still `COMPLETED`, and `exit_code == 1`.
- [ ] Disabled beats bypass (row 2 before 3): status is `SKIPPED_DISABLED`, not in `completed_tasks`.
- [ ] Bypass runs nothing, is in `completed_tasks`, status `BYPASSED`, and unlocks its dependent.
- [ ] Bypass emits no `TaskStarted`.
- [ ] Bypass survives a failed upstream and still releases its dependent (row 3 before 4).
- [ ] Bypass precedes the input guard (row 3 before 5).
- [ ] Abort beats bypass (row 0 before 3).
- [ ] Abort marks the running and all pending tasks `ABORTED`; `RunFinished.aborted is True`.
- [ ] Guard violation is a loud skip; the run continues; the task is never invoked.
- [ ] A stage descriptor absent from the DAG registry yields `SKIPPED_UNREGISTERED`.
- [ ] `exit_code` is 0 when tasks are skipped for declared reasons but none failed.
- [ ] The bypass branch performs no filesystem access (assert via a `Path` method patch that
      raises, or an `open` guard, for the duration of a bypass-only run).

`tests/test_status_channel.py`
- [ ] An ordinary console line returns `None`.
- [ ] A line that merely contains the prefix mid-string returns `None` (prefix must anchor).
- [ ] A valid sentinel returns the event with fields intact.
- [ ] A malformed sentinel raises `ValueError`.
- [ ] End-to-end: `encode_event` output parses back to an equal event — pins the two modules together.

`tests/test_dag_status_overlay.py`
- [ ] `_STATUS_OVERLAY` covers every `TaskStatus` member (reporting missing names, not a bare False).
- [ ] It has no keys outside `TaskStatus`.
- [ ] `BYPASSED` is visually distinct from `COMPLETED` in fill, border and glyph, and is undimmed.

`tests/test_process_tree.py`
- [ ] `popen_group_kwargs()` returns the platform-correct key.
- [ ] POSIX (skipped on Windows): a child that spawns a grandchild is fully reaped by
      `kill_process_tree`, and the test process itself survives — the regression guard for
      the defect in the reference implementation.
- [ ] `kill_process_tree` on an already-exited process is a no-op.

`tests/test_dag_config_model_bypass.py`
- [ ] `set_task_bypassed` round-trips through save/reload with comments, key order and value
      types preserved, and the key placed immediately after `enabled`.
- [ ] `set_task_dependencies` round-trips and preserves flow style where the original used it.
- [ ] Mutations set `dirty`.

`tests/test_task_registry.py`
- [ ] Every task id in both live DAG configs resolves in the registry.
- [ ] An unregistered id raises `KeyError` naming the registry file.
- [ ] A registry entry with an empty `description` raises at load.

### Integration Tests

- [ ] `tests/test_rf_boundary_dag_wiring.py` gains: every task in every DAG config declares
      `bypass` (the CI-visible guard for the required-key merge hazard). The existing
      exact-set-equality assertion on the barrier's `depends_on` must remain green — proof
      that this plan does not rewrite topology.
- [ ] `run_pipeline_stages` against a fixture DAG config in `tmp_path` with stub callables:
      dependency gating, a bypassed node unlocking its dependent, a failed node cascading,
      and the returned `RunOutcome.exit_code`. This is the first end-to-end test of the
      driver the repo has ever had.
- [ ] A "Qt-free" test: import each of `execution_events`, `dag_plan`, `dag_execution`,
      `status_channel` with `PyQt5` removed from `sys.modules` and assert it is still absent
      afterwards.

### Manual Verification

- [ ] Launch the GUI, run the processing DAG, and watch nodes transition
      pending → running → completed live.
- [ ] Disable one boundary method node; confirm the barrier and its four consumers paint
      amber `SKIPPED_DEP` with named console lines, and the run exits 0 (nothing failed).
- [ ] Set `bypass: true` on that node instead; confirm the modal appears, defaults to Cancel,
      and on Ok the node paints violet while the barrier and consumers run.
- [ ] Cancel the modal; confirm the previous run's statuses are still on screen.
- [ ] Force a task to raise; confirm the node paints red, the summary names it, the status bar
      says `Failed`, and the process exit code is 1.
- [ ] Abort mid-run; confirm via Task Manager that no `python.exe` descendant survives.
- [ ] Drag a node; confirm grid snap, that the layout file is written once ~800 ms after the
      drag ends, and that the detail panel does **not** switch tasks on a nudge.
- [ ] Toggle `Enabled` off and back on with `Bypass` ticked; confirm `Bypass` greys but keeps
      its tick, and the YAML value is unchanged.

### Edge Cases

- [ ] Every task bypassed — the run completes having invoked nothing, and says so.
- [ ] Bypass on a DAG root (no dependencies) and on a leaf (no dependents).
- [ ] `bypass: true` together with `force_processing: true` — bypass wins; `Force` is greyed.
- [ ] A task whose output contains a line that looks like a sentinel prefix — must not be
      parsed as an event unless it anchors at position 0 of the line.
- [ ] A tqdm `\r` frame arriving mid-sentinel — the reader must not split a sentinel line.
- [ ] Empty `tasks:` mapping, and a DAG where every task is disabled.
- [ ] Abort pressed after the child has already exited but before the poll timer fires.
- [ ] Switching workflow mid-run (statuses belong to the old graph; `set_task_status` must
      no-op on an unknown node rather than raise into a Qt slot).

---

## Documentation Plan

- [ ] Update `CLAUDE.md` / `docs/claude/architecture.md` with the new
      `analysis/pipeline/` execution modules and the sentinel status protocol.
- [ ] Document the sentinel line format and the `TaskStatus` vocabulary in
      `docs/guides/dag-status-protocol.md`, including how to replay a run log into the graph.
- [ ] Document `bypass` semantics — persisted, unverified, unlocks dependents — in the DAG
      config reference, with the warning that a forgotten `bypass: true` runs dependents
      against stale outputs.
- [ ] Add a changelog entry under `docs/changelogs/`.
- [ ] Note in the plan's completion record that `PipelineMonitor.show_dashboard()` remains
      dead code and why it was left alone.

---

## Rollback Plan

1. **Before deployment:** the work is phase-committed; each phase is independently
   revertable. Phases 3 and 5 are self-contained and can be dropped without touching the rest.
2. **Data considerations:** no data migration. The only on-disk change is the `bypass` key in
   three DAG YAMLs and the new registry file. `.layout.json` is untouched (statuses are never
   persisted). Grid snap changes node coordinates on the next drag only; the existing layout
   file loads unchanged.
3. **Rollback procedure:**
   - Revert the phase commits in reverse order.
   - **Revert the YAML with the code.** A `bypass:` key left in the config against reverted
     code raises `unknown key(s) ['bypass']` at load — loud in both directions, which is the
     intended behaviour, but it means config and code must move together.
   - `sys.exit(main())` reverting to `main()` restores the always-0 exit code; any CI or
     script that started keying on the exit code must revert with it.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A persisted `bypass: true` is forgotten and a later run reports green against stale outputs | High | High | Three independent mitigations, all mandatory: the Cancel-defaulted modal listing every bypassed task; a permanent console banner naming them; and the node painted **violet, never green**, so the run report never claims work that did not happen |
| `bypass` as a required key produces a clean text merge with a broken config when another branch adds a task node | High | Medium | A conformance test asserting every task in every DAG config declares `bypass` — the failure surfaces in CI, not at run time. Three branches currently in flight touch DAG configs |
| Non-zero exit breaks something that assumed exit 0 | Medium | Medium | Grep for callers of both entry scripts before Phase 1.5; the GUI is the only known caller and is updated in the same phase |
| Prefect spawns processes that escape the new process group | Medium | High | Phase 3's Windows path uses `taskkill /F /T`, which walks the parent-PID tree and does not rely on the group at all; POSIX uses a real `setsid` group. Verified manually via Task Manager in the acceptance checks |
| Sentinel lines interleave with tqdm `\r` frames and get corrupted | Medium | Medium | Sentinels are printed with `flush=True` and contain no `\r`; the parser anchors on position 0 and a malformed sentinel is loudly reported rather than dropped. Covered by an edge-case test |
| Rewriting `stage_runner` regresses the 35-task processing DAG | Medium | High | It has no tests today, which is the point of the integration test in this plan. The adapter keeps the existing stage-descriptor shape, lazy `params()` evaluation, and common-kwarg injection byte-for-byte |
| Adding a third checkbox breaks `_sync_table_checkboxes`'s `cb.text() == "Force"` dispatch | High (certain if unaddressed) | Medium | Explicitly scheduled as task 4.4; the else-branch would silently overwrite `Bypass` with the enabled state |
| Scope creep from the task registry into output-path ownership | Medium | Medium | Explicitly out of scope; the registry owns display metadata only, and `output_dirs.py` keeps owning paths |
| Node width overflow with three checkboxes | High | Low | Re-measure from `inner.sizeHint()` after assembling the row (task 4.2) — the current width formula measures only the label |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 0 — repair failing tests | ~0.5 h | None |
| Phase 1 — execution core, events, exit code | ~1.5 days | Phase 0 |
| Phase 2 — status channel, live rendering | ~1 day | Phase 1 |
| Phase 3 — process-tree kill | ~0.5 day | Phase 2 |
| Phase 4 — bypass end to end | ~1 day | Phases 1, 2 |
| Phase 5 — registry and polish | ~1 day | Phase 2 |

---

## References

- Origin of the current GUI module layout: `docs/development/plans/completed/port-pipeline-gui-launcher.md`
  (its Out-of-Scope section defers "new GUI features beyond the functional port" — this plan is that follow-on).
- Established option-rendering pattern to reuse rather than reinvent (`_OPTION_ENUMS`,
  `_OPTION_VISIBILITY`, `self._sections`): `docs/development/plans/completed/dag-launcher-boundary-method-dropdown.md`.
- Reference implementation, all mechanisms ported here:
  `F:\GitHub\clinical_projects\population-synthetic\src\population_synthetic\gui\`
  (`workflow_state.py`, `workflow_runner.py`, `execution.py`,
  `widgets/workflow_graph_items.py`, `widgets/workflow_graph_view.py`, `main_window.py`).
- Reference design record for the bypass flag, including its rejected alternatives:
  `F:\GitHub\clinical_projects\population-synthetic\docs\development\plans\completed\gui-workflow-bypass-flag.md`.
- Reverse-engineered spec of *this* repo's GUI, written by the reference repo when it ported
  from here: `F:\GitHub\clinical_projects\population-synthetic\docs\architecture\gui-dag-launcher-reference.md`.
- Pipeline engineering guides: `~/.claude/knowledge/data-pipeline-engineering/` (`01` Axis 3
  DAG orchestration; `02` §2 layering, §3 DTOs, §8 error boundaries; `05` DIP, cohesion).
- Deferred TODO this plan resolves: `src/utils/gui/analysis_runner_gui/task_detail_panel.py:193-202`.

---
