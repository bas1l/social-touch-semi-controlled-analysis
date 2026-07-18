"""Byte-identical parity: recomputed param-free fields vs. shared-NPZ round-trip.

Phase 7 rewires the boundary extractor to *consume* the param-free per-(session,
gesture) fields from the shared ``spatial_build_response_fields`` NPZ instead of
recomputing them inline. The only thing that changes between the two code paths
is where ``param_free`` comes from; everything downstream (thresholding,
interpolation, PCA alignment, contour detection) is unchanged. So the parity
invariant reduces to: the fields read back from the NPZ are *byte-identical* to
the fields recomputed from the same inputs — which holds because float64/int64
arrays round-trip through ``np.savez`` / ``np.load`` exactly.

These tests build a synthetic :class:`BoundaryInputs`, run the real writer
(``build_session_response_fields``) and reader (``load_param_free_fields_npz``),
and assert exact equality of both the param-free fields and the thresholded
per-gesture results that the extractor derives from them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_boundary_io import (  # noqa: E402
    load_param_free_fields_npz,
)
from analysis.receptive_field_mapping.data.rf_boundary_preparation import (  # noqa: E402
    _build_gesture_results,
    build_param_free_fields,
)
from analysis.receptive_field_mapping.data.rf_boundary_types import (  # noqa: E402
    BoundaryInputs,
    BoundaryParams,
)
from analysis.receptive_field_mapping.pipelines.rf_response_fields_pipeline import (  # noqa: E402
    build_session_response_fields,
)
from analysis.receptive_field_mapping.boundary.methods.radial import (  # noqa: E402
    RadialFootMethod,
)
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (  # noqa: E402
    compute_radial_foot_boundary,
)


def _make_inputs(tmp_path: Path) -> BoundaryInputs:
    """A small synthetic session covering all gesture subsets.

    12 vertices; 6 touches (2 tap, 2 stroke_proximal, 2 stroke_distal) so that
    ``build_param_free_fields`` produces every canonical subset including the
    synthetic ``'stroke'`` = stroke_proximal + stroke_distal. ``slim_V`` equals
    ``forearm_vertices`` so the SLIM->raw nearest-neighbour map is the identity.
    """
    n_verts = 12
    rng = np.random.default_rng(7)
    forearm_vertices = rng.random((n_verts, 3)).astype(np.float64)

    gesture_types = np.array(
        ["tap", "tap", "stroke_proximal", "stroke_proximal",
         "stroke_distal", "stroke_distal"]
    )
    touch_triple_keys = [f"t{i}" for i in range(6)]

    rf_vertex_indices = [
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([4, 5]),
        np.array([6, 7]),
        np.array([7, 8]),
    ]
    rf_values = [
        np.array([10.0, 20.0]),
        np.array([30.0, 40.0]),
        np.array([5.0, 6.0]),
        np.array([7.0, 8.0]),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
    ]

    cp_vertex_idx = np.array([0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8])
    cp_touch_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    pop_data = SimpleNamespace(
        forearm_vertices=forearm_vertices,
        gesture_types=gesture_types,
        touch_triple_keys=touch_triple_keys,
        cp_vertex_idx=cp_vertex_idx,
        cp_touch_idx=cp_touch_idx,
    )
    rf_data = SimpleNamespace(
        rf_vertex_indices=rf_vertex_indices,
        rf_values=rf_values,
    )

    forearm_uv = rng.random((n_verts, 2)).astype(np.float64)
    slim_V = forearm_vertices.copy()
    slim_faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)

    return BoundaryInputs(
        session_id="S_PARITY",
        session_output_dir=tmp_path,
        sentinel=tmp_path / "sentinel.json",
        pop_data=pop_data,
        rf_data=rf_data,
        forearm_uv=forearm_uv,
        slim_V=slim_V,
        slim_faces=slim_faces,
        raw_vertex_colors=None,
        forearm_ply_path=tmp_path / "forearm.ply",
        input_paths=[],
    )


def test_param_free_fields_roundtrip_byte_identical(tmp_path: Path) -> None:
    """Fields read from the shared NPZ equal the recomputed fields exactly."""
    inputs = _make_inputs(tmp_path)

    recomputed = build_param_free_fields(inputs)
    data_dict = build_session_response_fields(inputs)

    npz_path = tmp_path / "S_PARITY_response_fields.npz"
    np.savez(npz_path, **data_dict)
    loaded = load_param_free_fields_npz(npz_path)

    # Same gesture keys, in the same canonical order (incl. synthetic 'stroke').
    assert list(loaded.keys()) == list(recomputed.keys())
    assert "stroke" in loaded

    for gtype in recomputed:
        raw_r, cnt_r, n_r = recomputed[gtype]
        raw_l, cnt_l, n_l = loaded[gtype]

        # Byte-identical raw heatmap (float64) and unique count (int64).
        np.testing.assert_array_equal(raw_r, raw_l)
        assert raw_l.dtype == np.float64
        np.testing.assert_array_equal(cnt_r, cnt_l)
        assert cnt_l.dtype == np.int64
        assert int(n_r) == int(n_l)


def test_gesture_results_identical_recompute_vs_npz(tmp_path: Path) -> None:
    """The thresholded per-gesture results are identical for both sources.

    ``_build_gesture_results`` is the first downstream consumer of ``param_free``;
    identical results here imply byte-identical PCA alignment, grids, contours,
    and the persisted boundary NPZ (all pure functions of these arrays).
    """
    inputs = _make_inputs(tmp_path)
    params = BoundaryParams(neuron_mode="iff", min_overlap_pct=25.0)

    recomputed = build_param_free_fields(inputs)
    data_dict = build_session_response_fields(inputs)
    npz_path = tmp_path / "S_PARITY_response_fields.npz"
    np.savez(npz_path, **data_dict)
    loaded = load_param_free_fields_npz(npz_path)

    res_recompute, raw_recompute, cnt_recompute = _build_gesture_results(
        recomputed, params, None
    )
    res_npz, raw_npz, cnt_npz = _build_gesture_results(loaded, params, None)

    assert list(res_recompute.keys()) == list(res_npz.keys())
    for gtype in res_recompute:
        heat_r, n_r, thr_r = res_recompute[gtype]
        heat_n, n_n, thr_n = res_npz[gtype]
        np.testing.assert_array_equal(heat_r, heat_n)
        assert int(n_r) == int(n_n)
        assert int(thr_r) == int(thr_n)
        np.testing.assert_array_equal(raw_recompute[gtype], raw_npz[gtype])
        np.testing.assert_array_equal(cnt_recompute[gtype], cnt_npz[gtype])


# ---------------------------------------------------------------------------
# Radial method wrap parity (Phase 2 of the boundary plug-in architecture)
# ---------------------------------------------------------------------------
#
# Guards the wrap-not-rewrite invariant: ``RadialFootMethod.compute`` must forward
# to the untouched ``compute_radial_foot_boundary`` core and merely re-shape its
# output into a ``BoundaryContour``. So the geometric fields produced through the
# method must be byte-for-byte identical (exact array equality, not approximate)
# to the fields produced by calling the core directly on the same input.


def _make_radial_parity_grid(
    size: int = 120,
    sigma_px: float = 12.0,
    footprint_px: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A circular Gaussian bump on a finite disk footprint (NaN outside).

    Mirrors ``tests/test_rf_radial_foot_boundary._make_gaussian_grid`` for the
    default circular case — a configuration known to yield a non-``None`` radial
    foot boundary, so the parity comparison exercises the real success path.
    """
    lin = np.linspace(0.0, 1.0, size)
    grid_u, grid_v = np.meshgrid(lin, lin, indexing="ij")
    center = 0.5
    pixel_scale = 1.0 / (size - 1)
    sigma_uv = sigma_px * pixel_scale
    grid_z = 100.0 * np.exp(
        -0.5 * ((grid_u - center) ** 2 + (grid_v - center) ** 2) / sigma_uv ** 2
    )
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    center_idx = (size - 1) * center
    norm = ((rows - center_idx) / footprint_px) ** 2 + (
        (cols - center_idx) / footprint_px
    ) ** 2
    grid_z[norm > 1.0] = np.nan
    return grid_u, grid_v, grid_z


def _assert_exactly_equal(name: str, a, b) -> None:
    """Exact equality: byte-for-byte for arrays, ``==`` for scalars/pairs."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # equal_nan=True: NaN in the same cell counts as identical (the core's
        # lmax field is NaN outside the footprint); everything else must match
        # exactly (no approximate tolerance).
        assert np.array_equal(a, b, equal_nan=True), (
            f"{name} differs (array not byte-identical)"
        )
    elif isinstance(a, (tuple, list)):
        assert tuple(a) == tuple(b), f"{name} differs: {a!r} != {b!r}"
    else:
        assert a == b, f"{name} differs: {a!r} != {b!r}"


def test_radial_method_wrap_is_byte_identical_to_core() -> None:
    """``RadialFootMethod.compute`` fields == direct core fields (exact equality)."""
    grid_u, grid_v, grid_z = _make_radial_parity_grid()

    direct = compute_radial_foot_boundary(grid_u, grid_v, grid_z)
    assert direct is not None, "fixture grid must yield a radial boundary"

    contour = RadialFootMethod().compute(grid_u, grid_v, grid_z)
    assert contour is not None, "wrapped method must yield the same boundary"

    assert contour.method_name == "radial"

    # The 11 geometric fields must be byte-for-byte identical to the core output.
    for field_name in (
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
    ):
        _assert_exactly_equal(
            field_name, getattr(contour, field_name), getattr(direct, field_name)
        )

    # The method-specific ``lmax`` array is demoted to the diagnostic channel,
    # byte-identical to the core's ``lmax``.
    assert contour.has_diagnostic("lmax")
    _assert_exactly_equal("lmax", contour.diagnostic_fields["lmax"], direct.lmax)
