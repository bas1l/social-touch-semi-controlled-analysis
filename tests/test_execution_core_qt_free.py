"""The DAG execution core must never drag Qt into the workflow child process.

"Qt-free" is defined as: the module's transitive import graph contains no
``PyQt5``.  The check runs in a fresh interpreter so that a ``PyQt5`` already
imported by another test in this session cannot mask a real dependency.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

QT_FREE_MODULES = (
    "analysis.pipeline.execution_events",
    "analysis.pipeline.dag_plan",
    "analysis.pipeline.dag_execution",
    "analysis.pipeline.stage_runner",
    # The GUI's reader side of the same channel: it lives under the GUI package
    # but must stay importable without Qt, so a run log can be replayed by a
    # headless tool.
    "utils.gui.analysis_runner_gui.status_channel",
)

_PROBE = """
import importlib
import sys

module_name = sys.argv[1]
assert "PyQt5" not in sys.modules, "PyQt5 was already imported before the probe ran"
importlib.import_module(module_name)
leaked = sorted(name for name in sys.modules if name.split(".")[0] == "PyQt5")
if leaked:
    raise SystemExit("PyQt5 leaked into the import graph: " + ", ".join(leaked))
"""


@pytest.mark.parametrize("module_name", QT_FREE_MODULES)
def test_module_imports_without_pulling_in_pyqt5(module_name):
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, module_name],
        cwd=str(REPO_ROOT),
        env={
            **_inherited_env(),
            "PYTHONPATH": str(SRC),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"{module_name} is not Qt-free:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _inherited_env():
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env
