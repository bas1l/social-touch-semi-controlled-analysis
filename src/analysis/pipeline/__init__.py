"""Shared bootstrap helpers for the analysis workflow entry scripts.

Provides session-discovery utilities and a generic stage-runner dispatcher
so ``analysis_workflow_processing.py`` and ``analysis_workflow_viewers.py``
can each be self-contained without duplicating infrastructure code.

Also re-exports domain constants and utility functions from
``shared_constants`` so callers can do ``from analysis.pipeline import
GESTURE_TYPES`` etc.
"""

from .session_discovery import collect_unique_session_dirs, discover_input_items
from .execution_events import (
    SENTINEL_PREFIX,
    ConsoleLine,
    RunFinished,
    TaskFinished,
    TaskStarted,
    TaskStatus,
    decode_event,
    encode_event,
)
from .dag_plan import DagPlan, DagTask
from .dag_execution import RunOutcome, execute_dag, format_run_summary
from .task_registry import TaskMeta, get_task_meta, load_registry
from .stage_runner import run_pipeline_stages
from .shared_constants import (
    GESTURE_TYPES,
    TOUCH_ID_COLS,
    TOUCH_ID_COLS_WITH_SESSION,
    NERVE_SPIKE_COL,
    CONTACT_POINTS_COL,
    NERVE_FREQ_COL,
    LOCATION_SHARED_COLS,
    LOCATION_BASE_COLS,
    NEURON_MODES,
    session_id_from_path,
    filter_enabled_profiles,
)

__all__ = [
    "collect_unique_session_dirs",
    "discover_input_items",
    "run_pipeline_stages",
    "SENTINEL_PREFIX",
    "ConsoleLine",
    "RunFinished",
    "TaskFinished",
    "TaskStarted",
    "TaskStatus",
    "decode_event",
    "encode_event",
    "DagPlan",
    "DagTask",
    "RunOutcome",
    "execute_dag",
    "format_run_summary",
    "TaskMeta",
    "get_task_meta",
    "load_registry",
    "GESTURE_TYPES",
    "TOUCH_ID_COLS",
    "TOUCH_ID_COLS_WITH_SESSION",
    "NERVE_SPIKE_COL",
    "CONTACT_POINTS_COL",
    "NERVE_FREQ_COL",
    "LOCATION_SHARED_COLS",
    "LOCATION_BASE_COLS",
    "NEURON_MODES",
    "session_id_from_path",
    "filter_enabled_profiles",
]
