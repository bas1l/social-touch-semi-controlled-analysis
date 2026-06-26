"""Launcher for the interactive preprocessing explorer.

The explorer itself lives in the ``sandbox.contour_explorer`` package (one concern
per module).  This thin shim only puts ``src/`` (for ``analysis.*``) and ``scripts/``
(so ``sandbox`` is importable as a top-level package) on ``sys.path``, then runs the
app.  See ``sandbox/contour_explorer/app.py`` for the Layout/Usage notes and the
hardcoded ``DATA_ROOT`` / ``SESSIONS`` / ``GESTURE`` block.

Usage
-----
    python scripts/sandbox_preprocessing_explorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "src"))   # analysis.*
sys.path.insert(0, str(_HERE.parent))               # makes `sandbox` importable

from sandbox.contour_explorer.app import run

if __name__ == "__main__":
    run()
