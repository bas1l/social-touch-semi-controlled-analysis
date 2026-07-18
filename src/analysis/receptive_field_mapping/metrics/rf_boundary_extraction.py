"""Processing layer for the ``spatial_extract_boundaries`` stage.

Runs the *selected* boundary method over the prepared per-gesture grids and
returns a :class:`BoundaryResults`. Dispatch is registry-driven: the method is
resolved by name through
:func:`~analysis.receptive_field_mapping.boundary.registry.get_method` and its
:meth:`~analysis.receptive_field_mapping.boundary.method_base.BoundaryMethod.compute`
produces a :class:`~analysis.receptive_field_mapping.boundary.contract.BoundaryContour`.
There is **no** ``if method == "..."`` branch here — algorithm identity lives only
in the registry (Strategy + Factory).

Each method resolves its own parameters from its declared ``params_schema``. The
caller supplies a generic ``method_params`` mapping; only the keys the selected
method declares are forwarded (the rest are ignored — e.g. a radial-only knob is
dropped for the gradient method). Per-(session, gesture) tuned overrides (the
radial contour tuner) are applied by *parameter capability* (the schema declares
the key) rather than by branching on the method name.

The full per-method fan-out FLOW — running several methods and rendering per
method — is wired in Phase 4; here the stage runs the one active method and stores
its contours generically under ``BoundaryResults.boundaries[method_name]``.
"""

import logging

from analysis.receptive_field_mapping.boundary.contract import validate_contour
from analysis.receptive_field_mapping.boundary.registry import get_method
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)

logger = logging.getLogger(__name__)


# Maps a radial-tuner ``GestureContourParams`` onto the radial method's schema
# keys via its ``effective_*`` resolvers. Keyed by *parameter name* (capability),
# not by method identity: an override is applied only when the active method's
# schema declares that key, so a non-radial method silently receives none.
_TUNED_OVERRIDE_RESOLVERS = {
    "radial_gauss_sigma": lambda gp: gp.effective_gauss_sigma(),
    "radial_hess_sigma": lambda gp: gp.radial_hess_sigma,
    "radial_envelope_smooth_sigma": lambda gp: gp.effective_envelope_smooth_sigma(),
    "radial_prominence": lambda gp: gp.effective_prominence(),
    "radial_plateau_size": lambda gp: gp.effective_plateau_size(),
}


def _tuned_overrides(gesture_params, schema_keys: set[str]) -> dict:
    """Return the tuned overrides whose key the active method's schema declares."""
    return {
        key: resolver(gesture_params)
        for key, resolver in _TUNED_OVERRIDE_RESOLVERS.items()
        if key in schema_keys
    }


def extract_session_boundaries(
    prepared: PreparedBoundaryData,
    params: BoundaryParams,
    *,
    method_params: dict | None = None,
) -> BoundaryResults:
    """Detect RF boundaries for every gesture grid of one session.

    Resolves the active method from ``params.boundary_method`` through the
    registry (fail-fast on an unknown name), forwards the schema-declared subset
    of ``method_params`` (plus any per-gesture tuned overrides), and stores the
    resulting contours under ``results.boundaries[method_name][gtype]``.
    """
    method = get_method(params.boundary_method)
    schema_keys = {spec.key for spec in method.params_schema}
    base_params = {
        key: value
        for key, value in (method_params or {}).items()
        if key in schema_keys
    }

    results = BoundaryResults(boundary_method=method.name)
    per_gesture_contours: dict = results.boundaries.setdefault(method.name, {})

    for gtype, (grid_u, grid_v, grid_z) in prepared.per_gesture_grids.items():
        gesture_params = dict(base_params)
        if prepared.per_gesture_params is not None:
            gesture_params.update(
                _tuned_overrides(prepared.per_gesture_params[gtype], schema_keys)
            )
        contour = method.compute(grid_u, grid_v, grid_z, **gesture_params)
        if contour is not None:
            validate_contour(contour)  # fail-fast at the fan-in boundary
        per_gesture_contours[gtype] = contour

    return results
