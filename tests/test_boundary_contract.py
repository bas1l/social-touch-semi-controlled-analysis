"""Contract + registry foundation tests for the pluggable boundary seam.

Covers Phase 1 of the "Pluggable Multi-Algorithm RF Boundary Extraction" plan:

- ``validate_contour`` accepts a well-formed contour and raises (fail-fast) on
  every missing/malformed/wrong-shape field.
- ``ParamSpec`` fails fast on a missing ``group`` and other malformed specs.
- The registry factory ``get_method`` raises on unknown names; ``register_method``
  / ``all_methods`` / ``params_schema_of`` round-trip.
- A **shared contract conformance suite**, parametrised over the methods under
  test. It runs on a self-contained ``DummyBoundaryMethod`` now and will
  automatically cover the three real methods once they register in Phase 2
  (``_methods_under_test`` splices ``registry.all_methods()`` in).
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup and stubs (mirror tests/test_rf_radial_foot_boundary.py so the
# heavy analysis.receptive_field_mapping.__init__ is not executed).
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


_stub("analysis")
_stub("analysis.receptive_field_mapping")

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.boundary import (  # noqa: E402
    BoundaryContour,
    BoundaryMethod,
    ParamSpec,
    all_methods,
    get_method,
    params_schema_of,
    register_method,
    registered_names,
    validate_contour,
)
from analysis.receptive_field_mapping.boundary import registry as _registry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_square_contour(n: int = 32) -> np.ndarray:
    """A closed, finite (N, 2) UV contour (unit circle sampling)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)]).astype(float)


def _make_valid_contour(method_name: str = "dummy", **overrides) -> BoundaryContour:
    """Build a well-formed BoundaryContour, overriding individual fields."""
    fields = dict(
        contour_uv=_make_square_contour(),
        area_uv=np.pi,
        perimeter_uv=2.0 * np.pi,
        circularity=1.0,
        centroid_uv=(0.0, 0.0),
        peak_uv=(0.0, 0.0),
        pca_major_uv=2.0,
        pca_minor_uv=2.0,
        pca_orientation_deg=0.0,
        mean_iff_on_contour=0.5,
        iff_at_centroid=1.0,
        method_name=method_name,
    )
    fields.update(overrides)
    return BoundaryContour(**fields)


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global registry so tests don't leak state."""
    saved = dict(_registry._REGISTRY)
    try:
        yield _registry
    finally:
        _registry._REGISTRY.clear()
        _registry._REGISTRY.update(saved)


class DummyBoundaryMethod(BoundaryMethod):
    """A self-contained conforming method used to exercise the shared suite.

    Independent of the real detectors so the conformance suite is meaningful even
    while the production registry is empty (Phase 1).
    """

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def params_schema(self):
        return (
            ParamSpec("sigma", float, 4.0, range=(0.0, 20.0), tunable=True, group="smoothing"),
            ParamSpec("n_angles", int, 360, range=(8, 1440), group="extraction"),
            ParamSpec("require_positive", bool, True, group="extraction"),
            ParamSpec(
                "mode", str, "a", choices=("a", "b", "c"), group="extraction"
            ),
        )

    def compute(self, grid_u, grid_v, grid_z, **params) -> BoundaryContour | None:
        # Grid-agnostic: always returns a conforming contour carrying a
        # diagnostic overlay so the diagnostic-channel path is exercised too.
        return _make_valid_contour(
            method_name=self.name,
            diagnostic_fields={"dummy_field": np.zeros_like(grid_z)},
        )


def _synthetic_grid(size: int = 60):
    """A Gaussian bump on a finite disk footprint (NaN outside)."""
    lin = np.linspace(-1.0, 1.0, size)
    grid_u, grid_v = np.meshgrid(lin, lin, indexing="ij")
    grid_z = np.exp(-(grid_u**2 + grid_v**2) / (2.0 * 0.25**2))
    grid_z[grid_u**2 + grid_v**2 > 0.9**2] = np.nan
    return grid_u, grid_v, grid_z


def _methods_under_test() -> list[BoundaryMethod]:
    """Every registered method plus the always-present dummy.

    When Phase 2 registers the three real methods they are included here
    automatically, giving the conformance suite full coverage with no edits.
    """
    return [*all_methods(), DummyBoundaryMethod()]


# ---------------------------------------------------------------------------
# validate_contour: acceptance
# ---------------------------------------------------------------------------


def test_validate_contour_accepts_well_formed():
    contour = _make_valid_contour()
    assert validate_contour(contour) is contour


def test_validate_contour_allows_nan_scalar_metrics():
    # A degenerate polygon may legitimately yield NaN circularity/metrics.
    contour = _make_valid_contour(circularity=float("nan"), iff_at_centroid=float("nan"))
    assert validate_contour(contour) is contour


def test_diagnostic_fields_default_is_empty_and_immutable():
    contour = _make_valid_contour()
    assert dict(contour.diagnostic_fields) == {}
    with pytest.raises(TypeError):
        contour.diagnostic_fields["x"] = np.zeros(3)  # type: ignore[index]


def test_diagnostic_fields_capability_check():
    contour = _make_valid_contour(diagnostic_fields={"lmax": np.zeros((4, 4))})
    assert contour.has_diagnostic("lmax")
    assert not contour.has_diagnostic("absent")
    assert validate_contour(contour) is contour


def test_contour_is_frozen():
    contour = _make_valid_contour()
    with pytest.raises(dataclasses.FrozenInstanceError):
        contour.area_uv = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_contour: rejection (fail-fast on each malformed field)
# ---------------------------------------------------------------------------


def test_validate_contour_rejects_non_contour():
    with pytest.raises(TypeError):
        validate_contour(object())


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"contour_uv": None}, id="contour_uv_none"),
        pytest.param({"contour_uv": [[0, 0], [1, 1], [2, 2]]}, id="contour_uv_list"),
        pytest.param({"contour_uv": np.zeros((5, 3))}, id="contour_uv_wrong_cols"),
        pytest.param({"contour_uv": np.zeros((2, 2))}, id="contour_uv_too_few_pts"),
        pytest.param({"contour_uv": np.zeros((5,))}, id="contour_uv_1d"),
        pytest.param(
            {"contour_uv": np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]])},
            id="contour_uv_nan",
        ),
        pytest.param(
            {"contour_uv": np.array([[0, 0], [1, 1], [2, 2]], dtype=int)},
            id="contour_uv_int_dtype",
        ),
        pytest.param({"area_uv": "big"}, id="area_uv_str"),
        pytest.param({"circularity": None}, id="circularity_none"),
        pytest.param({"pca_orientation_deg": [1.0]}, id="pca_orientation_list"),
        pytest.param({"centroid_uv": (0.0,)}, id="centroid_wrong_len"),
        pytest.param({"centroid_uv": (0.0, "x")}, id="centroid_bad_component"),
        pytest.param({"peak_uv": 3.0}, id="peak_not_pair"),
        pytest.param({"method_name": ""}, id="method_name_empty"),
        pytest.param({"method_name": 123}, id="method_name_not_str"),
    ],
)
def test_validate_contour_rejects_malformed_field(overrides):
    contour = _make_valid_contour(**overrides)
    with pytest.raises((TypeError, ValueError)):
        validate_contour(contour)


def test_validate_contour_rejects_bad_diagnostic_value():
    contour = _make_valid_contour(diagnostic_fields={"x": [1, 2, 3]})
    with pytest.raises(TypeError):
        validate_contour(contour)


# ---------------------------------------------------------------------------
# ParamSpec: fail-fast construction
# ---------------------------------------------------------------------------


def test_paramspec_requires_group():
    with pytest.raises(ValueError):
        ParamSpec("sigma", float, 4.0)


def test_paramspec_valid():
    spec = ParamSpec("sigma", float, 4.0, range=(0.0, 20.0), tunable=True, group="g")
    assert spec.key == "sigma"
    assert spec.group == "g"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(key="", type=float, default=1.0, group="g"), id="empty_key"),
        pytest.param(dict(key="k", type="float", default=1.0, group="g"), id="type_not_type"),
        pytest.param(dict(key="k", type=float, default=1.0, group=""), id="empty_group"),
        pytest.param(dict(key="k", type=float, default=1.0, range=(5.0, 1.0), group="g"), id="bad_range_order"),
        pytest.param(dict(key="k", type=float, default=1.0, range=(1.0,), group="g"), id="bad_range_len"),
        pytest.param(dict(key="k", type=str, default="z", choices=("a", "b"), group="g"), id="default_not_in_choices"),
        pytest.param(dict(key="k", type=str, default="a", choices=(), group="g"), id="empty_choices"),
    ],
)
def test_paramspec_rejects_malformed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ParamSpec(**kwargs)


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def test_get_method_raises_on_unknown():
    with pytest.raises(ValueError):
        get_method("no_such_method")


def test_params_schema_of_raises_on_unknown():
    with pytest.raises(ValueError):
        params_schema_of("no_such_method")


def test_register_and_get_roundtrip(clean_registry):
    method = DummyBoundaryMethod()
    returned = register_method(method)
    assert returned is method
    assert get_method("dummy") is method
    assert "dummy" in registered_names()
    assert method in all_methods()
    schema = params_schema_of("dummy")
    assert all(isinstance(s, ParamSpec) for s in schema)


def test_register_rejects_non_method(clean_registry):
    with pytest.raises(TypeError):
        register_method(object())  # type: ignore[arg-type]


def test_register_rejects_duplicate(clean_registry):
    register_method(DummyBoundaryMethod())
    with pytest.raises(ValueError):
        register_method(DummyBoundaryMethod())
    # overwrite=True is allowed.
    register_method(DummyBoundaryMethod(), overwrite=True)


# ---------------------------------------------------------------------------
# Shared contract conformance suite (auto-covers Phase 2 methods)
# ---------------------------------------------------------------------------


@pytest.fixture(params=_methods_under_test(), ids=lambda m: m.name)
def method_under_test(request) -> BoundaryMethod:
    return request.param


def test_conformance_name_is_nonempty_str(method_under_test):
    assert isinstance(method_under_test.name, str)
    assert method_under_test.name.strip()


def test_conformance_params_schema_is_paramspecs(method_under_test):
    schema = list(method_under_test.params_schema)
    assert all(isinstance(s, ParamSpec) for s in schema)
    keys = [s.key for s in schema]
    assert len(keys) == len(set(keys)), "params_schema keys must be unique"
    # Every declared option resolves a group (guards the historical missing-group gap).
    assert all(isinstance(s.group, str) and s.group for s in schema)


def test_conformance_compute_satisfies_contract(method_under_test):
    grid_u, grid_v, grid_z = _synthetic_grid()
    result = method_under_test.compute(grid_u, grid_v, grid_z)
    # The contract is: compute returns a contract-satisfying contour OR None.
    if result is None:
        return
    validate_contour(result)
    assert result.method_name == method_under_test.name


def test_dummy_positive_path_produces_valid_contour():
    # Guarantees the positive (non-None) conformance path is always exercised.
    grid_u, grid_v, grid_z = _synthetic_grid()
    result = DummyBoundaryMethod().compute(grid_u, grid_v, grid_z)
    assert result is not None
    validate_contour(result)
    assert result.has_diagnostic("dummy_field")
