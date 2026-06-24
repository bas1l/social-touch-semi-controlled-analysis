"""Entry point for AnalysisRunnerGUI.

Launch with::

    python scripts/launch_analysis_runner_gui.py
"""

import logging
import sys
import traceback
from pathlib import Path

# CuPy import guard — must precede any preprocessing imports (project convention)
try:
    import cupy  # noqa: F401
except Exception:
    pass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from utils.gui.analysis_runner_gui.runner_config import parse_runner_config
from utils.gui.analysis_runner_gui.runner_window import AnalysisRunnerGUI

# High-DPI awareness — must be set BEFORE QApplication is constructed, otherwise
# showMaximized() on Windows with display scaling renders a ~half-screen window
# with a title bar drawn at the unscaled size.
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


_logger = logging.getLogger(__name__)


def _install_exception_hook() -> None:
    """Replace sys.excepthook so that unhandled exceptions in Qt slots are
    logged instead of silently crashing the application."""

    def _hook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _logger.critical("Unhandled exception in Qt callback:\n%s", msg)
        print(msg, file=sys.stderr, flush=True)

    sys.excepthook = _hook


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _install_exception_hook()

    project_root = Path(".").resolve()
    runner_yaml = project_root / "configs" / "analysis_runner_gui.yaml"

    if not runner_yaml.is_file():
        print(f"Error: AnalysisRunnerGUI config not found at {runner_yaml}")
        sys.exit(1)

    try:
        entries = parse_runner_config(runner_yaml, project_root)
    except Exception as exc:
        print(f"Error: failed to parse {runner_yaml}: {exc}")
        sys.exit(1)

    configs_dir = project_root / "configs"
    app = QApplication(sys.argv)
    window = AnalysisRunnerGUI(entries, configs_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
