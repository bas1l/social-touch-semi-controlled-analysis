"""Round-trip + isolation tests for the generic per-method boundary IO.

Phase 3 of the "Pluggable Multi-Algorithm RF Boundary Extraction" plan replaces
the monolithic per-method NPZ blocks with one generic writer/reader over the
:class:`BoundaryContour` contract, writing each method into its own ``<method>/``
folder. These tests assert:

* a contour (incl. a ``diagnostic_fields`` overlay) survives a write -> discovery
  read round-trip with its 11 geometric fields + diagnostic intact;
* two methods write non-overlapping folders and are discovered independently;
* re-running one method overwrites only its own folder (idempotency);
* a partial/corrupt method folder (boundary NPZ without ``run_metadata.json``)
  fails fast on read.

The writer projects UV -> XYZ through ``uv_points_to_xyz``, so the fixture uses a
trivial planar SLIM mesh (V = (u, v, 0)) large enough to contain the contour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.boundary.contract import (  # noqa: E402
    BoundaryContour,
)
from analysis.receptive_field_mapping.data.rf_boundary_io import (  # noqa: E402
    boundary_contour_npz_path,
    boundary_run_metadata_path,
    load_boundary_contours,
    save_boundary_contours_npz,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A single big planar quad covering [-5, 5]^2 in UV; V embeds UV as (u, v, 0) so
# uv_points_to_xyz maps any contained UV point to (u, v, 0).
_FOREARM_UV = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
_FOREARM_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
_FOREARM_V = np.array(
    [[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]]
)


def _circle_contour(n: int = 24, radius: float = 1.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def _make_contour(method_name: str, *, with_diag: bool = False, radius: float = 1.0) -> BoundaryContour:
    fields = dict(
        contour_uv=_circle_contour(radius=radius),
        area_uv=np.pi * radius**2,
        perimeter_uv=2.0 * np.pi * radius,
        circularity=1.0,
        centroid_uv=(0.0, 0.0),
        peak_uv=(0.1, 0.2),
        pca_major_uv=2.0 * radius,
        pca_minor_uv=2.0 * radius,
        pca_orientation_deg=30.0,
        mean_iff_on_contour=5.0,
        iff_at_centroid=7.0,
        method_name=method_name,
    )
    if with_diag:
        fields["diagnostic_fields"] = {"lmax": np.arange(9.0).reshape(3, 3)}
    return BoundaryContour(**fields)


_SESSION = "S_IO"
_CONFIG = {"boundary_method": "radial", "note": "unit-test"}


def _save(tmp_path: Path, method_name: str, contours: dict) -> Path:
    return save_boundary_contours_npz(
        contours,
        method_name=method_name,
        session_output_dir=tmp_path,
        session_id=_SESSION,
        forearm_uv=_FOREARM_UV,
        forearm_faces=_FOREARM_FACES,
        forearm_V=_FOREARM_V,
        config_snapshot=_CONFIG,
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_contour_roundtrips_with_diagnostic(tmp_path: Path) -> None:
    original = _make_contour("radial", with_diag=True)
    _save(tmp_path, "radial", {"all": original})

    discovered = load_boundary_contours(tmp_path)
    assert set(discovered) == {"radial"}
    assert set(discovered["radial"]) == {"all"}

    loaded = discovered["radial"]["all"]
    assert loaded.method_name == "radial"

    # The 11 geometric fields survive.
    np.testing.assert_array_equal(loaded.contour_uv, original.contour_uv)
    for name in (
        "area_uv", "perimeter_uv", "circularity", "pca_major_uv", "pca_minor_uv",
        "pca_orientation_deg", "mean_iff_on_contour", "iff_at_centroid",
    ):
        assert getattr(loaded, name) == pytest.approx(getattr(original, name))
    assert tuple(loaded.centroid_uv) == pytest.approx(tuple(original.centroid_uv))
    assert tuple(loaded.peak_uv) == pytest.approx(tuple(original.peak_uv))

    # The diagnostic overlay survives, keyed by name (method-blind channel).
    assert loaded.has_diagnostic("lmax")
    np.testing.assert_array_equal(
        loaded.diagnostic_fields["lmax"], original.diagnostic_fields["lmax"]
    )

    # Provenance sidecar is written alongside the NPZ.
    assert boundary_run_metadata_path(tmp_path, "radial").exists()


def test_two_methods_write_non_overlapping_folders(tmp_path: Path) -> None:
    _save(tmp_path, "radial", {"all": _make_contour("radial", with_diag=True)})
    _save(tmp_path, "gradient", {"all": _make_contour("gradient")})

    radial_npz = boundary_contour_npz_path(tmp_path, "radial", _SESSION)
    gradient_npz = boundary_contour_npz_path(tmp_path, "gradient", _SESSION)
    assert radial_npz.exists() and gradient_npz.exists()
    assert radial_npz.parent != gradient_npz.parent

    discovered = load_boundary_contours(tmp_path)
    assert set(discovered) == {"radial", "gradient"}
    assert discovered["radial"]["all"].has_diagnostic("lmax")
    assert not discovered["gradient"]["all"].has_diagnostic("lmax")


def test_rerun_overwrites_only_own_folder(tmp_path: Path) -> None:
    _save(tmp_path, "radial", {"all": _make_contour("radial", radius=1.0)})
    _save(tmp_path, "gradient", {"all": _make_contour("gradient", radius=1.0)})

    # Re-run radial with a different contour; gradient must be untouched.
    _save(tmp_path, "radial", {"all": _make_contour("radial", radius=3.0)})

    discovered = load_boundary_contours(tmp_path)
    assert discovered["radial"]["all"].area_uv == pytest.approx(np.pi * 9.0)
    assert discovered["gradient"]["all"].area_uv == pytest.approx(np.pi * 1.0)


def test_none_contour_recorded_by_omission(tmp_path: Path) -> None:
    _save(tmp_path, "radial", {"all": _make_contour("radial"), "tap": None})
    discovered = load_boundary_contours(tmp_path)
    assert set(discovered["radial"]) == {"all"}


def test_discovery_ignores_non_method_subdirs(tmp_path: Path) -> None:
    _save(tmp_path, "radial", {"all": _make_contour("radial")})
    (tmp_path / "aggregated").mkdir()
    (tmp_path / "inspection").mkdir()
    discovered = load_boundary_contours(tmp_path)
    assert set(discovered) == {"radial"}


def test_partial_folder_missing_metadata_fails_fast(tmp_path: Path) -> None:
    _save(tmp_path, "radial", {"all": _make_contour("radial")})
    boundary_run_metadata_path(tmp_path, "radial").unlink()
    with pytest.raises(FileNotFoundError):
        load_boundary_contours(tmp_path)
