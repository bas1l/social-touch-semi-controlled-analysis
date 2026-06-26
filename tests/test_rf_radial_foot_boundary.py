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
    radial_foot_boundary_to_dict,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (  # noqa: E402
    clean_heatmap_islands,
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

    def test_peak_at_border_returns_none(self) -> None:
        size = 120
        u = np.linspace(0, 1, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.zeros((size, size))
        # Peak placed within 2 cells of an edge (row 1).
        grid_z[1, size // 2] = 100.0
        result = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
        assert result is None


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
