"""Unit tests for the boundary-extraction verification layer.

Covers ``validate_boundary_params`` (allowed-set guards) and
``boundary_stage_is_up_to_date`` (mtime staleness gate over the four upstream
inputs vs the per-session sentinel).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup and stubs (import the single module without the heavy package init)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stub(dotted: str, **attrs) -> None:
    if dotted in sys.modules:
        return
    mod = types.ModuleType(dotted)
    parts = dotted.split(".")
    pkg_dir = _SRC / Path(*parts)
    if pkg_dir.exists():
        mod.__path__ = [str(pkg_dir)]
    mod.__package__ = dotted
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[dotted] = mod
    if "." in dotted:
        parent, child = dotted.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, sys.modules[dotted])


_stub("utils")
_stub("_vendor")
_stub("analysis")
_stub("analysis.pipeline")
_stub("analysis.receptive_field_mapping")
_stub("analysis.receptive_field_mapping.pipelines")

from analysis.receptive_field_mapping.pipelines.rf_boundary_verification import (  # noqa: E402
    boundary_stage_is_up_to_date,
    validate_boundary_params,
)


def _write(path: Path, mtime: float) -> Path:
    path.write_text("x")
    os.utime(path, (mtime, mtime))
    return path


def _make_inputs(tmp_path: Path, mtime: float) -> list[Path]:
    return [
        _write(tmp_path / f"input_{i}.dat", mtime)
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# validate_boundary_params
# ---------------------------------------------------------------------------

class TestValidateParams:
    def test_valid_combo_does_not_raise(self):
        validate_boundary_params("mean", "radial")
        validate_boundary_params("max", "gradient")
        validate_boundary_params("mean", "inflection")

    def test_invalid_iff_metric_raises(self):
        with pytest.raises(ValueError, match="invalid iff_metric"):
            validate_boundary_params("bogus", "gradient")

    def test_invalid_boundary_method_raises(self):
        with pytest.raises(ValueError, match="invalid boundary_method"):
            validate_boundary_params("mean", "bogus")


# ---------------------------------------------------------------------------
# boundary_stage_is_up_to_date
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_fresh_output_is_up_to_date(self, tmp_path):
        inputs = _make_inputs(tmp_path, mtime=1000)
        sentinel = _write(tmp_path / "done.json", mtime=2000)
        assert boundary_stage_is_up_to_date(inputs, sentinel, force=False) is True

    def test_stale_output_is_not_up_to_date(self, tmp_path):
        inputs = _make_inputs(tmp_path, mtime=3000)
        sentinel = _write(tmp_path / "done.json", mtime=2000)
        assert boundary_stage_is_up_to_date(inputs, sentinel, force=False) is False

    def test_missing_sentinel_is_not_up_to_date(self, tmp_path):
        inputs = _make_inputs(tmp_path, mtime=1000)
        sentinel = tmp_path / "done.json"  # never created
        assert boundary_stage_is_up_to_date(inputs, sentinel, force=False) is False

    def test_force_overrides_freshness(self, tmp_path):
        inputs = _make_inputs(tmp_path, mtime=1000)
        sentinel = _write(tmp_path / "done.json", mtime=2000)
        assert boundary_stage_is_up_to_date(inputs, sentinel, force=True) is False

    def test_missing_input_raises(self, tmp_path):
        inputs = _make_inputs(tmp_path, mtime=1000)
        inputs.append(tmp_path / "missing_upstream.npz")  # does not exist
        sentinel = _write(tmp_path / "done.json", mtime=2000)
        with pytest.raises(FileNotFoundError):
            boundary_stage_is_up_to_date(inputs, sentinel, force=False)
