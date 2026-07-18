"""``RadialFootMethod`` — the radial foot-of-mountain detector as a strategy.

Adapts the existing :func:`compute_radial_foot_boundary` core (untouched source
of truth for the numerics) into the pluggable :class:`BoundaryMethod` contract.
The method does **not** re-implement any math: it forwards the schema-resolved
parameters to the core orchestrator and maps the returned
:class:`RadialFootBoundary` dataclass onto the 11-field
:class:`BoundaryContour`, demoting the extra ``lmax`` array into the contour's
``diagnostic_fields`` overlay channel (no computation stage reads it).
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
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    compute_radial_foot_boundary,
)

__all__ = ["RadialFootMethod"]


# Declared parameter schema. Defaults mirror the existing
# ``compute_radial_foot_boundary`` signature exactly so a params-free
# ``compute(...)`` is byte-for-byte equivalent to calling the core directly.
# ``tunable=True`` marks the parameters the current ``ContourParamToggles``
# exposes (radial_gauss_sigma / radial_envelope_smooth_sigma / radial_prominence);
# radial_hess_sigma (always-applied core scale) and radial_plateau_size are not
# toggleable and so are declared non-tunable.
_SCHEMA: tuple[ParamSpec, ...] = (
    ParamSpec(
        "radial_gauss_sigma", float, 8.0,
        range=(0.0, 50.0), tunable=True, group="smoothing",
    ),
    ParamSpec(
        "radial_hess_sigma", float, 5.0,
        range=(0.0, 50.0), tunable=False, group="smoothing",
    ),
    ParamSpec(
        "radial_envelope_smooth_sigma", float, 1.5,
        range=(0.0, 50.0), tunable=True, group="envelope",
    ),
    ParamSpec(
        "radial_prominence", float, None,
        tunable=True, group="plateau_gate",
    ),
    ParamSpec(
        "radial_plateau_size", int, 1,
        range=(1, 999), tunable=False, group="plateau_gate",
    ),
)


class RadialFootMethod(BoundaryMethod):
    """Radial Hessian-λmax foot boundary, wrapping the existing core."""

    @property
    def name(self) -> str:
        return "radial"

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
        boundary = compute_radial_foot_boundary(
            grid_u,
            grid_v,
            grid_z,
            gauss_sigma=p["radial_gauss_sigma"],
            hess_sigma=p["radial_hess_sigma"],
            envelope_smooth_sigma=p["radial_envelope_smooth_sigma"],
            prominence=p["radial_prominence"],
            plateau_size=p["radial_plateau_size"],
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
            diagnostic_fields={"lmax": boundary.lmax},
        )
