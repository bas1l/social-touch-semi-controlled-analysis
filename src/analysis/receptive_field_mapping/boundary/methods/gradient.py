"""``GradientMethod`` — the gradient-ridge detector as a strategy.

Adapts the existing :func:`compute_gradient_ridge` core (untouched numerics) into
the :class:`BoundaryMethod` contract. The gradient core consumes a
Gaussian-smoothed field; this method reproduces the pipeline's derivation of that
field via :func:`compute_laplacian_arrays` (single source of truth for the
NaN-aware smoothing) and forwards it, then maps the returned
:class:`GradientBoundary` onto :class:`BoundaryContour`, demoting the extra
``gradient_magnitude`` array into the ``diagnostic_fields`` overlay channel.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from analysis.receptive_field_mapping.boundary.contract import BoundaryContour
from analysis.receptive_field_mapping.boundary.method_base import (
    BoundaryMethod,
    ParamSpec,
)
from analysis.receptive_field_mapping.boundary.methods._common import resolve_params
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    compute_gradient_ridge,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_laplacian_arrays,
)

__all__ = ["GradientMethod"]


# Declared parameter schema. ``gradient_gauss_sigma`` seeds the NaN-aware
# Gaussian smoothing that produces the field the gradient ridge is traced on
# (mirroring the pipeline, which builds ``smoothed`` from
# ``compute_laplacian_arrays``); ``gradient_n_angles`` / ``gradient_savgol_window``
# forward to the radial-profiling controls of ``compute_gradient_ridge``. Defaults
# mirror those cores' signatures.
_SCHEMA: tuple[ParamSpec, ...] = (
    ParamSpec(
        "gradient_gauss_sigma", float, 4.0,
        range=(0.0, 50.0), tunable=False, group="smoothing",
    ),
    ParamSpec(
        "gradient_n_angles", int, 360,
        range=(8, 3600), tunable=False, group="extraction",
    ),
    ParamSpec(
        "gradient_savgol_window", int, 31,
        range=(3, 999), tunable=False, group="extraction",
    ),
)


class GradientMethod(BoundaryMethod):
    """Gradient-magnitude ridge boundary, wrapping the existing core."""

    @property
    def name(self) -> str:
        return "gradient"

    @property
    def params_schema(self) -> Sequence[ParamSpec]:
        return _SCHEMA

    def compute(
        self,
        grid_u: np.ndarray,
        grid_v: np.ndarray,
        grid_z: np.ndarray,
        **params: Any,
    ) -> BoundaryContour | None:
        p = resolve_params(_SCHEMA, params)
        # The gradient ridge is traced on the NaN-aware Gaussian-smoothed field,
        # derived exactly as the pipeline does (compute_laplacian_arrays[0]).
        smoothed = compute_laplacian_arrays(grid_z, p["gradient_gauss_sigma"])[0]
        boundary = compute_gradient_ridge(
            grid_u,
            grid_v,
            grid_z,
            smoothed,
            n_angles=p["gradient_n_angles"],
            savgol_window=p["gradient_savgol_window"],
        )
        if boundary is None:
            return None
        return BoundaryContour(
            contour_uv=boundary.contour_uv,
            area_uv=boundary.area_uv,
            perimeter_uv=boundary.perimeter_uv,
            circularity=boundary.circularity,
            centroid_uv=boundary.centroid_uv,
            peak_uv=boundary.peak_uv,
            pca_major_uv=boundary.pca_major_uv,
            pca_minor_uv=boundary.pca_minor_uv,
            pca_orientation_deg=boundary.pca_orientation_deg,
            mean_iff_on_contour=boundary.mean_iff_on_contour,
            iff_at_centroid=boundary.iff_at_centroid,
            method_name=self.name,
            diagnostic_fields={"gradient_magnitude": boundary.gradient_magnitude},
        )
