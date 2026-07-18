"""Unit tests for rf_radial_foot_boundary and clean_heatmap_islands.

Tests use synthetic Gaussian grids on a known UV domain so that geometric
properties (circularity, centroid, PCA axes, mean IFF) can be validated.

The radial "foot of mountain" method casts rays outward from the peak and snaps
each radius to the last valid raw sample. The grids therefore carry a finite
footprint (NaN outside a disk) so the snap has a real data edge to land on. The
foot ring lands farther out than the inflection ring, so we assert robust
geometric properties (encloses the peak, circularity, mean IFF < centroid IFF)
rather than the inflection-specific exp(-1) relation.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup and stubs
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
_stub("analysis")
_stub("analysis.receptive_field_mapping")

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (  # noqa: E402
    RadialFootBoundary,
    compute_radial_foot_boundary,
    compute_radial_foot_stages,
    radial_foot_boundary_to_dict,
    _select_enclosing_contour,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (  # noqa: E402
    clean_heatmap_islands,
)
from analysis.receptive_field_mapping.data.rf_boundary_types import (  # noqa: E402
    ContourParamToggles,
    ContourExtractionError,
    ContourFailureBranch,
    ContourFailureDiagnostics,
    GestureContourParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_grid(
    size: int = 120,
    sigma_u: float = 12.0,
    sigma_v: float = 12.0,
    center_frac: tuple[float, float] = (0.5, 0.5),
    amplitude: float = 100.0,
    uv_range: float = 1.0,
    footprint_px: float | None = 45.0,
    footprint_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid_u, grid_v, grid_z) for a 2D Gaussian in pixel-sigma units.

    sigma_u and sigma_v are in grid pixels; the UV domain spans [0, uv_range].
    When ``footprint_px`` is given, cells outside an ellipse of semi-axes
    ``footprint_px`` (axis 0) and ``footprint_px * footprint_ratio`` (axis 1) are
    NaN'd, giving the radial-foot snap a real data edge to land on. A
    ``footprint_ratio`` != 1 makes the footprint itself elongated so the snapped
    foot reflects the Gaussian's anisotropy.
    """
    u = np.linspace(0.0, uv_range, size)
    v = np.linspace(0.0, uv_range, size)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")

    cu = uv_range * center_frac[0]
    cv = uv_range * center_frac[1]
    pixel_scale = uv_range / (size - 1)

    sigma_u_uv = sigma_u * pixel_scale
    sigma_v_uv = sigma_v * pixel_scale

    grid_z = amplitude * np.exp(
        -0.5 * ((grid_u - cu) ** 2 / sigma_u_uv ** 2 + (grid_v - cv) ** 2 / sigma_v_uv ** 2)
    )

    if footprint_px is not None:
        rows = np.arange(size)[:, None]
        cols = np.arange(size)[None, :]
        center_row = (size - 1) * center_frac[0]
        center_col = (size - 1) * center_frac[1]
        a = footprint_px
        b = footprint_px * footprint_ratio
        norm = ((rows - center_row) / a) ** 2 + ((cols - center_col) / b) ** 2
        grid_z[norm > 1.0] = np.nan

    return grid_u, grid_v, grid_z


def _point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (polygon = (N, 2) UV vertices)."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Test: circular Gaussian — boundary found, approximately circular, encloses peak
# ---------------------------------------------------------------------------


class TestCircularGaussian:
    """Radial-foot boundary on a circular Gaussian hill."""

    @pytest.fixture(scope="class")
    def boundary(self) -> RadialFootBoundary | None:
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=120, sigma_u=12.0, sigma_v=12.0
        )
        return compute_radial_foot_boundary(grid_u, grid_v, grid_z)

    def test_boundary_is_not_none(self, boundary: RadialFootBoundary | None) -> None:
        assert boundary is not None

    def test_contour_uv_is_2d_array(self, boundary: RadialFootBoundary) -> None:
        assert isinstance(boundary.contour_uv, np.ndarray)
        assert boundary.contour_uv.ndim == 2
        assert boundary.contour_uv.shape[1] == 2
        assert boundary.contour_uv.shape[0] > 0

    def test_circularity_close_to_one(self, boundary: RadialFootBoundary) -> None:
        assert boundary.circularity >= 0.7, (
            f"Circularity {boundary.circularity:.3f} too low for circular Gaussian"
        )

    def test_centroid_near_grid_center(self, boundary: RadialFootBoundary) -> None:
        cu, cv = boundary.centroid_uv
        tolerance = 0.05
        assert abs(cu - 0.5) < tolerance, f"U centroid {cu:.4f} far from 0.5"
        assert abs(cv - 0.5) < tolerance, f"V centroid {cv:.4f} far from 0.5"

    def test_peak_near_grid_center(self, boundary: RadialFootBoundary) -> None:
        pu, pv = boundary.peak_uv
        tolerance = 0.05
        assert abs(pu - 0.5) < tolerance, f"U peak {pu:.4f} far from 0.5"
        assert abs(pv - 0.5) < tolerance, f"V peak {pv:.4f} far from 0.5"

    def test_contour_encloses_peak(self, boundary: RadialFootBoundary) -> None:
        assert _point_in_polygon(boundary.peak_uv, boundary.contour_uv), (
            "radial-foot contour does not enclose the peak"
        )

    def test_area_positive(self, boundary: RadialFootBoundary) -> None:
        assert boundary.area_uv > 0.0

    def test_perimeter_positive(self, boundary: RadialFootBoundary) -> None:
        assert boundary.perimeter_uv > 0.0

    def test_mean_iff_below_centroid_iff(self, boundary: RadialFootBoundary) -> None:
        # The foot ring is farther out than the centroid, so the IFF on the
        # contour is below the IFF at the (peak-like) centroid; both finite.
        assert math.isfinite(boundary.mean_iff_on_contour)
        assert math.isfinite(boundary.iff_at_centroid)
        assert boundary.mean_iff_on_contour < boundary.iff_at_centroid, (
            f"mean_iff_on_contour {boundary.mean_iff_on_contour:.3f} not below "
            f"iff_at_centroid {boundary.iff_at_centroid:.3f}"
        )

    def test_peak_uv_type(self, boundary: RadialFootBoundary) -> None:
        assert isinstance(boundary.peak_uv, tuple)
        assert len(boundary.peak_uv) == 2
        assert isinstance(boundary.peak_uv[0], (float, int))
        assert isinstance(boundary.peak_uv[1], (float, int))


# ---------------------------------------------------------------------------
# Test: elliptical Gaussian — PCA axes reflect elongation
# ---------------------------------------------------------------------------


class TestEllipticalGaussian:
    """PCA axes of the radial-foot boundary reflect the Gaussian sigma ratio."""

    @pytest.fixture(scope="class")
    def boundary(self) -> RadialFootBoundary | None:
        # Elongate the footprint along with the Gaussian so the snapped foot
        # reflects the anisotropy rather than a circular data edge.  PCA of a
        # contour's vertices understates the true aspect ratio (the same caveat
        # the inflection test notes), so an elongated footprint is used to clear
        # the > 1.5 PCA threshold for an anisotropic blob.
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=150,
            sigma_u=32.0,
            sigma_v=10.0,
            footprint_px=68.0,
            footprint_ratio=0.31,
        )
        return compute_radial_foot_boundary(grid_u, grid_v, grid_z)

    def test_boundary_found(self, boundary: RadialFootBoundary | None) -> None:
        assert boundary is not None

    def test_pca_major_larger_than_minor(self, boundary: RadialFootBoundary) -> None:
        assert boundary.pca_major_uv > boundary.pca_minor_uv

    def test_pca_anisotropy_reflects_elongation(self, boundary: RadialFootBoundary) -> None:
        actual_ratio = boundary.pca_major_uv / boundary.pca_minor_uv
        assert actual_ratio > 1.5, (
            f"PCA ratio {actual_ratio:.2f} not > 1.5 for a 2:1 sigma ellipse"
        )


# ---------------------------------------------------------------------------
# Test: None / degenerate cases
# ---------------------------------------------------------------------------


class TestNoneCases:
    """compute_radial_foot_boundary returns None (not raises) for degenerate inputs."""

    def test_flat_surface_returns_none(self) -> None:
        u = np.linspace(0, 1, 120)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.ones((120, 120)) * 5.0
        result = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert result is None

    def test_all_nan_returns_none(self) -> None:
        u = np.linspace(0, 1, 120)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.full((120, 120), np.nan)
        result = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert result is None

    def test_none_grid_z_returns_none(self) -> None:
        u = np.linspace(0, 1, 120)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        result = compute_radial_foot_boundary(grid_u, grid_v, None)
        assert result is None

    def test_near_border_peak_is_attempted_not_vetoed(self) -> None:
        # Intent change: the positional "peak within 2 cells of the edge" veto was
        # removed, so a near-border peak is now ATTEMPTED rather than rejected.
        # This single-spike grid can still form a foot, so extraction succeeds and
        # a boundary is returned (previously this returned None purely because of
        # the border veto).
        size = 120
        u = np.linspace(0, 1, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.zeros((size, size))
        grid_z[1, size // 2] = 100.0  # peak within 2 cells of the top edge (row 1)
        result = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert result is not None


# ---------------------------------------------------------------------------
# Test: compute_radial_foot_stages allow_partial — graceful degradation contract
# ---------------------------------------------------------------------------


class TestStagesAllowPartial:
    """``allow_partial=True`` degrades gracefully; ``False`` stays fail-fast.

    A monotone ramp keeps a fully computable field (raw/smoothed/λmax) and a
    locatable peak (the corner) while the contour stage cannot be traced (a plane
    has no curvature ring / no footprint contour encloses the seed), so it
    exercises exactly the partial branch: the field stages come back populated,
    the contour stages are ``None`` but ``peak_rc`` still carries the located peak
    (Change B), and an ``error`` message explains why — whereas the default
    fail-fast path raises the same underlying ``ValueError``.
    """

    @staticmethod
    def _no_contour_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # A monotone ramp: fully computable field, peak locatable at the corner,
        # but no closed foot contour can be traced around it.
        size = 60
        u = np.linspace(0, 1, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = grid_u + grid_v
        return grid_u, grid_v, grid_z

    def test_allow_partial_returns_field_and_peak_without_contour(self) -> None:
        grid_u, grid_v, grid_z = self._no_contour_grid()
        stages = compute_radial_foot_stages(
            grid_u, grid_v, grid_z, allow_partial=True
        )
        # Field stages present...
        assert stages["smoothed"] is not None
        assert stages["lmax"] is not None
        assert stages["smoothed"].shape == grid_z.shape
        assert stages["lmax"].shape == grid_z.shape
        # ...contour stages omitted, with a loud reason...
        assert stages["contour_uv"] is None
        assert stages["contour_rc"] is None
        assert isinstance(stages.get("error"), str) and stages["error"]
        # ...but the LOCATED peak is still returned (Change B), so the GUI can
        # still mark it even though no contour is drawn.
        assert stages["peak_rc"] is not None
        assert tuple(stages["peak_rc"]) == (grid_z.shape[0] - 1, grid_z.shape[1] - 1)

    def test_default_fail_fast_raises_on_same_input(self) -> None:
        grid_u, grid_v, grid_z = self._no_contour_grid()
        with pytest.raises(ValueError):
            compute_radial_foot_stages(grid_u, grid_v, grid_z)

    def test_all_nan_raises_even_with_allow_partial(self) -> None:
        # The "no data" case must still raise loudly under allow_partial=True.
        u = np.linspace(0, 1, 120)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.full((120, 120), np.nan)
        with pytest.raises(ValueError):
            compute_radial_foot_stages(grid_u, grid_v, grid_z, allow_partial=True)

    def test_full_return_has_no_error_key(self) -> None:
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=120, sigma_u=12.0, sigma_v=12.0
        )
        stages = compute_radial_foot_stages(
            grid_u, grid_v, grid_z, allow_partial=True
        )
        assert stages["contour_uv"] is not None
        assert "error" not in stages


# ---------------------------------------------------------------------------
# Test: 2D footprint envelope — contour never bulges past the painted footprint
# ---------------------------------------------------------------------------


class TestFootprintEnvelope:
    """The enveloped contour encloses only footprint (``grid_z > 0``) cells."""

    @pytest.fixture(scope="class")
    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _make_gaussian_grid(size=120, sigma_u=12.0, sigma_v=12.0)

    def test_interior_within_footprint(self, grid) -> None:
        from matplotlib.path import Path as MplPath

        grid_u, grid_v, grid_z = grid
        boundary = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert boundary is not None

        pts = np.column_stack([grid_u.ravel(), grid_v.ravel()])
        inside = MplPath(boundary.contour_uv).contains_points(pts).reshape(grid_z.shape)
        footprint = grid_z > 0
        leak = int(np.count_nonzero(inside & ~footprint))
        assert leak <= 2, (
            f"{leak} cells inside the enveloped contour fall outside the "
            "painted footprint (grid_z > 0)"
        )

    def test_smooth_sigma_none_is_valid(self, grid) -> None:
        grid_u, grid_v, grid_z = grid
        boundary = compute_radial_foot_boundary(
            grid_u, grid_v, grid_z, envelope_smooth_sigma=None
        )
        assert boundary is not None
        assert boundary.area_uv > 0.0
        assert _point_in_polygon(boundary.peak_uv, boundary.contour_uv), (
            "unsmoothed enveloped contour does not enclose the peak"
        )


# ---------------------------------------------------------------------------
# Test: serialization
# ---------------------------------------------------------------------------


_SHARED_KEYS = {
    "contour_uv",
    "area_uv",
    "perimeter_uv",
    "circularity",
    "centroid_uv",
    "peak_uv",
    "pca_major_uv",
    "pca_minor_uv",
    "pca_orientation_deg",
    "mean_iff_on_contour",
    "iff_at_centroid",
}


class TestSerialization:
    """radial_foot_boundary_to_dict produces a JSON-safe dict (no lmax)."""

    @pytest.fixture(scope="class")
    def boundary(self) -> RadialFootBoundary:
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=120, sigma_u=12.0, sigma_v=12.0
        )
        result = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert result is not None, "Fixture requires a valid boundary"
        return result

    def test_round_trip_is_json_serializable(self, boundary: RadialFootBoundary) -> None:
        d = radial_foot_boundary_to_dict(boundary)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_expected_keys_present(self, boundary: RadialFootBoundary) -> None:
        d = radial_foot_boundary_to_dict(boundary)
        assert set(d.keys()) == _SHARED_KEYS
        assert "lmax" not in d

    def test_no_nan_in_dict(self, boundary: RadialFootBoundary) -> None:
        d = radial_foot_boundary_to_dict(boundary)
        for key, val in d.items():
            if key == "contour_uv":
                for pair in val:
                    assert not (math.isnan(pair[0]) or math.isnan(pair[1])), (
                        "NaN in contour_uv"
                    )
            elif key in ("centroid_uv", "peak_uv"):
                for v in val:
                    if v is not None:
                        assert not math.isnan(v), f"NaN in {key}"
            elif val is not None and isinstance(val, float):
                assert not math.isnan(val), f"NaN in field '{key}'"

    def test_no_numpy_types_in_dict(self, boundary: RadialFootBoundary) -> None:
        d = radial_foot_boundary_to_dict(boundary)
        for key, val in d.items():
            if key == "contour_uv":
                for pair in val:
                    assert isinstance(pair[0], float), f"contour_uv u is {type(pair[0])}"
                    assert isinstance(pair[1], float), f"contour_uv v is {type(pair[1])}"
            elif key in ("centroid_uv", "peak_uv"):
                for v in val:
                    assert v is None or isinstance(v, float), (
                        f"{key} entry is {type(v)}"
                    )
            elif val is not None:
                assert isinstance(val, (float, int, list)), (
                    f"Field '{key}' has numpy type {type(val)}"
                )

    def test_contour_uv_is_list_of_pairs(self, boundary: RadialFootBoundary) -> None:
        d = radial_foot_boundary_to_dict(boundary)
        assert isinstance(d["contour_uv"], list)
        assert all(len(pair) == 2 for pair in d["contour_uv"])


# ---------------------------------------------------------------------------
# Test: island cleaning
# ---------------------------------------------------------------------------


class TestCleanHeatmapIslands:
    """clean_heatmap_islands keeps only the peak's connected component."""

    @staticmethod
    def _make_two_blob_grid() -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        """Return (grid_z, peak_rc, secondary_mask).

        Main blob (the higher peak) on the left half, a separate secondary blob
        on the right half, separated by a NaN wall column so the two are not
        4-connected. ``secondary_mask`` marks the secondary blob's valid cells.
        """
        size = 60
        grid_z = np.full((size, size), np.nan)

        rows = np.arange(size)[:, None]
        cols = np.arange(size)[None, :]

        # Main blob centred at (30, 15), amplitude 100.
        main_dist = np.sqrt((rows - 30) ** 2 + (cols - 15) ** 2)
        main_mask = main_dist <= 8
        grid_z[main_mask] = 100.0 * np.exp(-0.5 * (main_dist[main_mask] / 4.0) ** 2)

        # Secondary blob centred at (30, 45), amplitude 50 (lower than main).
        sec_dist = np.sqrt((rows - 30) ** 2 + (cols - 45) ** 2)
        sec_mask = sec_dist <= 6
        grid_z[sec_mask] = 50.0 * np.exp(-0.5 * (sec_dist[sec_mask] / 3.0) ** 2)

        # NaN wall: column 30 separates the two halves (already NaN by default).
        grid_z[:, 28:33] = np.nan
        sec_mask = sec_mask & ~np.isnan(grid_z)

        peak_rc = np.unravel_index(np.nanargmax(grid_z), grid_z.shape)
        return grid_z, peak_rc, sec_mask

    def test_secondary_blob_nan_after_cleaning(self) -> None:
        grid_z, peak_rc, sec_mask = self._make_two_blob_grid()
        assert np.any(sec_mask), "test setup: secondary blob must have valid cells"
        cleaned = clean_heatmap_islands(grid_z)
        assert np.all(np.isnan(cleaned[sec_mask])), (
            "secondary blob cells should be NaN after island cleaning"
        )

    def test_peak_component_preserved(self) -> None:
        grid_z, peak_rc, _sec_mask = self._make_two_blob_grid()
        cleaned = clean_heatmap_islands(grid_z)
        # The peak's component must retain valid (non-NaN) cells.
        assert not np.isnan(cleaned[peak_rc]), "peak cell should remain valid"
        # The main blob should retain more than just one cell.
        assert np.count_nonzero(~np.isnan(cleaned)) > 1

    def test_peak_cell_value_unchanged(self) -> None:
        grid_z, peak_rc, _sec_mask = self._make_two_blob_grid()
        original_peak_value = grid_z[peak_rc]
        cleaned = clean_heatmap_islands(grid_z)
        assert cleaned[peak_rc] == original_peak_value, (
            "global-peak cell value must be unchanged by cleaning"
        )

    def test_returns_copy(self) -> None:
        grid_z, _peak_rc, _sec_mask = self._make_two_blob_grid()
        original = grid_z.copy()
        clean_heatmap_islands(grid_z)
        assert np.array_equal(grid_z, original, equal_nan=True), (
            "clean_heatmap_islands must not mutate its input"
        )

    def test_all_nan_raises_value_error(self) -> None:
        grid_z = np.full((40, 40), np.nan)
        with pytest.raises(ValueError):
            clean_heatmap_islands(grid_z)


# ---------------------------------------------------------------------------
# Test: toggle parity -- the GUI preview entry point (compute_radial_foot_stages)
# and the pipeline entry point (compute_radial_foot_boundary) produce identical
# contours for the same GestureContourParams.effective_*()-resolved values.
# ---------------------------------------------------------------------------


class TestTogglesParity:
    """``effective_*()`` feeds both entry points identically, ON and OFF.

    ``min_overlap_pct`` / ``median_filter_size`` act earlier, in grid building
    (``rf_contour_params_io.build_session_grid``) -- out of scope here (see
    ``docs/development/plans/active/rf-contour-tuning-param-toggles-and-help.md``,
    "Testing Plan"). This covers the two toggles that map directly onto
    ``compute_radial_foot_stages`` / ``compute_radial_foot_boundary``
    parameters: ``radial_gauss_sigma`` -> ``gauss_sigma`` (skip value ``0.0``)
    and ``radial_envelope_smooth_sigma`` -> ``envelope_smooth_sigma`` (skip
    value ``0.0``). ``radial_hess_sigma`` has no toggle and is passed through
    raw in every case.
    """

    @staticmethod
    def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _make_gaussian_grid(size=120, sigma_u=12.0, sigma_v=12.0)

    @staticmethod
    def _params(**toggle_overrides) -> GestureContourParams:
        toggles = ContourParamToggles(
            min_overlap_pct=toggle_overrides.get("min_overlap_pct", True),
            median_filter_size=toggle_overrides.get("median_filter_size", True),
            radial_gauss_sigma=toggle_overrides.get("radial_gauss_sigma", True),
            radial_envelope_smooth_sigma=toggle_overrides.get(
                "radial_envelope_smooth_sigma", True
            ),
        )
        return GestureContourParams(
            min_overlap_pct=25.0,
            median_filter_size=5,
            radial_gauss_sigma=8.0,
            radial_hess_sigma=5.0,
            radial_envelope_smooth_sigma=1.5,
            toggles=toggles,
        )

    def _assert_stages_matches_boundary(self, params: GestureContourParams) -> None:
        grid_u, grid_v, grid_z = self._grid()
        stages = compute_radial_foot_stages(
            grid_u, grid_v, grid_z,
            gauss_sigma=params.effective_gauss_sigma(),
            hess_sigma=params.radial_hess_sigma,
            envelope_smooth_sigma=params.effective_envelope_smooth_sigma(),
        )
        boundary = compute_radial_foot_boundary(
            grid_u, grid_v, grid_z,
            gauss_sigma=params.effective_gauss_sigma(),
            hess_sigma=params.radial_hess_sigma,
            envelope_smooth_sigma=params.effective_envelope_smooth_sigma(),
        )
        assert boundary is not None, "expected a boundary for the parity fixture"
        assert stages["contour_uv"] is not None
        np.testing.assert_array_equal(stages["contour_uv"], boundary.contour_uv)

    def test_all_toggles_on(self) -> None:
        params = self._params()
        self._assert_stages_matches_boundary(params)

    def test_radial_gauss_sigma_off(self) -> None:
        params = self._params(radial_gauss_sigma=False)
        assert params.effective_gauss_sigma() == 0.0
        self._assert_stages_matches_boundary(params)

    def test_radial_envelope_smooth_sigma_off(self) -> None:
        params = self._params(radial_envelope_smooth_sigma=False)
        assert params.effective_envelope_smooth_sigma() == 0.0
        self._assert_stages_matches_boundary(params)


# ---------------------------------------------------------------------------
# Test: per-branch failure diagnostics (known-answer)
#
# Each fixture below is a minimal synthetic grid crafted to FORCE exactly one of
# the three ``ContourFailureBranch`` values through the public partial entry
# ``compute_radial_foot_stages(allow_partial=True)`` (the same path the tuner GUI
# uses). Every numeric field is asserted against a hand-computed value — a wrong
# formula would still return a plausible number, so we pin the exact quantities
# (guide 03: known-answer tests). The fields belonging to the OTHER two branches
# must stay ``None`` on each payload, keeping "zero" (a measured count of 0)
# distinct from "absent" (not applicable to this branch).
# ---------------------------------------------------------------------------


class TestContourFailureDiagnostics:
    """Force each failure branch and assert its diagnostics payload numerically."""

    # -- Branch fixtures -----------------------------------------------------

    @staticmethod
    def _no_plateau_grid(
        size: int = 120, k: float = 0.05, amp: float = 1000.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-grid strictly-concave paraboloid: λmax stays negative on every ray.

        ``grid_z = amp - k * r^2`` has a constant negative-definite Hessian, so no
        ray forms a *positive* λmax plateau (``require_positive`` rejects them all)
        and ``found.sum() == 0`` -> ``NO_RADIAL_PLATEAU``. There is deliberately no
        NaN footprint mask: a footprint edge would inject a positive λmax ring and
        let a plateau form. ``amp`` keeps the whole grid > 0 so the peak is at the
        centre and the field is not the all-NaN "no data" case.
        """
        u = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        rows = np.arange(size)[:, None].astype(float)
        cols = np.arange(size)[None, :].astype(float)
        cr = cc = (size - 1) * 0.5
        r2 = (rows - cr) ** 2 + (cols - cc) ** 2
        grid_z = amp - k * r2
        return grid_u, grid_v, grid_z

    @staticmethod
    def _seed_outside_footprint_grid(
        size: int = 120, sigma: float = 12.0, offset: float = 200.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A Gaussian dome pushed entirely below zero: the footprint is empty.

        The Gaussian's curvature ring still forms a λmax plateau (a contour IS
        traced), but ``footprint = grid_z > 0`` is empty everywhere, so the seed
        cell is not in ``contour ∩ footprint`` -> ``SEED_OUTSIDE_FOOTPRINT``.
        ``offset`` (> amplitude 100) drives every finite cell negative; the disk
        NaN mask keeps a real edge so the field is not all-NaN.
        """
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=size, sigma_u=sigma, sigma_v=sigma
        )
        grid_z = grid_z - offset
        return grid_u, grid_v, grid_z

    @staticmethod
    def _no_enclosing_grid(
        size: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A monotone plane ramp: a plateau forms but no contour encloses the seed.

        ``grid_z = grid_u + grid_v`` locates the peak at the far corner. A λmax
        plateau does form (so extraction reaches the envelope), but the
        seed-connected footprint component has no marching-squares contour that
        encloses the corner seed -> ``NO_ENCLOSING_CONTOUR``.
        """
        u = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = grid_u + grid_v
        return grid_u, grid_v, grid_z

    # -- NO_RADIAL_PLATEAU ---------------------------------------------------

    def test_no_radial_plateau_branch_and_numbers(self) -> None:
        grid_u, grid_v, grid_z = self._no_plateau_grid()
        assert np.nanmin(grid_z) > 0.0, "fixture must keep the whole grid > 0"

        stages = compute_radial_foot_stages(
            grid_u, grid_v, grid_z, allow_partial=True, n_angles=180
        )
        diag = stages["diagnostics"]
        assert isinstance(diag, ContourFailureDiagnostics)
        assert diag.branch is ContourFailureBranch.NO_RADIAL_PLATEAU

        # Hand-computed: no ray formed a plateau, n_angles echoes the arg, and
        # peak_lmax is the finite nanmax of the very λmax field returned.
        assert diag.found_count == 0
        assert diag.n_angles == 180
        assert math.isfinite(diag.peak_lmax)
        assert diag.peak_lmax == pytest.approx(float(np.nanmax(stages["lmax"])))

        # Other branches' fields absent (zero-vs-absent kept distinct).
        assert diag.footprint_at_seed is None
        assert diag.footprint_cells is None
        assert diag.n_candidate_contours is None

    # -- SEED_OUTSIDE_FOOTPRINT ---------------------------------------------

    def test_seed_outside_footprint_branch_and_numbers(self) -> None:
        grid_u, grid_v, grid_z = self._seed_outside_footprint_grid()
        # Hand-computed footprint size: every finite cell is negative -> 0 painted.
        expected_cells = int(np.count_nonzero(grid_z > 0))
        assert expected_cells == 0, "fixture must have an empty grid_z > 0 footprint"

        stages = compute_radial_foot_stages(grid_u, grid_v, grid_z, allow_partial=True)
        diag = stages["diagnostics"]
        assert isinstance(diag, ContourFailureDiagnostics)
        assert diag.branch is ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT

        assert diag.footprint_at_seed is False
        assert diag.footprint_cells == expected_cells  # == 0

        # Other branches' fields absent.
        assert diag.n_angles is None
        assert diag.found_count is None
        assert diag.peak_lmax is None
        assert diag.n_candidate_contours is None

    # -- NO_ENCLOSING_CONTOUR -----------------------------------------------

    def test_no_enclosing_contour_branch_and_numbers(self) -> None:
        grid_u, grid_v, grid_z = self._no_enclosing_grid()
        stages = compute_radial_foot_stages(grid_u, grid_v, grid_z, allow_partial=True)
        diag = stages["diagnostics"]
        assert isinstance(diag, ContourFailureDiagnostics)
        assert diag.branch is ContourFailureBranch.NO_ENCLOSING_CONTOUR

        # At least one candidate contour was traced (none enclosed the seed).
        assert diag.n_candidate_contours is not None
        assert diag.n_candidate_contours >= 1

        # Other branches' fields absent.
        assert diag.n_angles is None
        assert diag.found_count is None
        assert diag.peak_lmax is None
        assert diag.footprint_at_seed is None
        assert diag.footprint_cells is None

    # -- Cross-branch sanity -------------------------------------------------

    def test_three_fixtures_reach_three_distinct_branches(self) -> None:
        """The three fixtures cover all three branches (no accidental overlap)."""
        branches = set()
        for builder, na in (
            (self._no_plateau_grid, 180),
            (self._seed_outside_footprint_grid, 360),
            (self._no_enclosing_grid, 360),
        ):
            gu, gv, gz = builder()
            stages = compute_radial_foot_stages(
                gu, gv, gz, allow_partial=True, n_angles=na
            )
            branches.add(stages["diagnostics"].branch)
        assert branches == {
            ContourFailureBranch.NO_RADIAL_PLATEAU,
            ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT,
            ContourFailureBranch.NO_ENCLOSING_CONTOUR,
        }


# ---------------------------------------------------------------------------
# Test: ContourExtractionError / ContourFailureDiagnostics unit coverage
# (supplementary — the branch-forcing tests above are the primary coverage).
# ---------------------------------------------------------------------------


class TestContourExtractionErrorUnit:
    """Direct unit coverage of the diagnostics DTO and its carrying exception."""

    def test_is_value_error_subclass_and_carries_payload(self) -> None:
        diag = ContourFailureDiagnostics(
            branch=ContourFailureBranch.NO_RADIAL_PLATEAU,
            n_angles=360,
            found_count=0,
            peak_lmax=0.5,
        )
        err = ContourExtractionError("boom", diag)
        # Subclass of ValueError so the existing ``except ValueError`` still catches.
        assert isinstance(err, ValueError)
        assert err.diagnostics is diag
        assert str(err) == "boom"

    def test_wrong_diagnostics_type_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourExtractionError("bad", diagnostics="not-a-payload")

    def test_diagnostics_optional_fields_default_to_none(self) -> None:
        diag = ContourFailureDiagnostics(
            branch=ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT
        )
        assert diag.n_angles is None
        assert diag.found_count is None
        assert diag.peak_lmax is None
        assert diag.footprint_at_seed is None
        assert diag.footprint_cells is None
        assert diag.n_candidate_contours is None

    def test_select_enclosing_contour_raises_with_branch(self) -> None:
        # A filled blob in one corner and a peak far from it: contour(s) exist but
        # none enclose the peak -> NO_ENCLOSING_CONTOUR with n_candidate_contours>=1.
        binary = np.zeros((40, 40), dtype=bool)
        binary[5:12, 5:12] = True
        with pytest.raises(ContourExtractionError) as excinfo:
            _select_enclosing_contour(binary, (30, 30))
        diag = excinfo.value.diagnostics
        assert diag.branch is ContourFailureBranch.NO_ENCLOSING_CONTOUR
        assert diag.n_candidate_contours is not None
        assert diag.n_candidate_contours >= 1
