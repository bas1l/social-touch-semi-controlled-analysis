"""``InflectionMethod`` — the Laplacian zero-crossing detector as a strategy.

Adapts the existing :func:`compute_inflection_boundary` core (untouched numerics)
into the :class:`BoundaryMethod` contract. The inflection detector carries no
method-specific array beyond the 11 geometric fields, so the mapped
:class:`BoundaryContour` populates no ``diagnostic_fields``.
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
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
)

__all__ = ["InflectionMethod"]


# Declared parameter schema. ``inflection_sigma`` is the NaN-aware Gaussian
# pre-smoothing sigma of the Laplacian zero-crossing detector; the default mirrors
# the ``compute_inflection_boundary`` signature.
_SCHEMA: tuple[ParamSpec, ...] = (
    ParamSpec(
        "inflection_sigma", float, 4.0,
        range=(0.0, 50.0), tunable=False, group="smoothing",
    ),
)


class InflectionMethod(BoundaryMethod):
    """Laplacian zero-crossing (inflection) boundary, wrapping the existing core."""

    @property
    def name(self) -> str:
        return "inflection"

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
        boundary = compute_inflection_boundary(
            grid_u,
            grid_v,
            grid_z,
            gaussian_sigma=p["inflection_sigma"],
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
        )
