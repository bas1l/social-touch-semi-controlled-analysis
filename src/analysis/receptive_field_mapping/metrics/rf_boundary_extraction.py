"""Processing layer for the ``spatial_extract_boundaries`` stage.

Runs the boundary detectors over the prepared per-gesture grids and returns a
:class:`BoundaryResults`. The ``smoothed`` / ``laplacian`` / ``gradient_mag``
fields are computed once here and carried in the result so persistence never
recomputes them.

Which detectors run mirrors the historical behaviour exactly (the NPZ contract
and downstream comparison stages depend on the per-method fields):

* inflection + gradient run whenever ``inflection_sigma`` is set, and
* radial runs when ``boundary_method == "radial"``;

``gesture_boundaries`` then holds whichever method ``boundary_method`` selects.
"""

import logging
from pathlib import Path

import numpy as np

from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    compute_gradient_magnitude,
    compute_gradient_ridge,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
    compute_laplacian_arrays,
)
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    compute_radial_foot_boundary,
)

logger = logging.getLogger(__name__)


def extract_session_boundaries(
    prepared: PreparedBoundaryData,
    params: BoundaryParams,
    inspection_dir: Path,
) -> BoundaryResults:
    """Detect RF boundaries for every gesture grid of one session."""
    inflection_sigma = params.inflection_sigma
    boundary_method = params.boundary_method
    contour_color = params.contour_color

    results = BoundaryResults(boundary_method=boundary_method)

    for gtype, (grid_u, grid_v, grid_z) in prepared.per_gesture_grids.items():
        # --- Inflection (Laplacian zero-crossing) ---
        boundary = (
            compute_inflection_boundary(
                grid_u, grid_v, grid_z, inflection_sigma,
                snapshot_dir=inspection_dir, snapshot_label=gtype,
                contour_color=contour_color,
            )
            if inflection_sigma is not None
            else None
        )
        results.gesture_inflection_boundaries[gtype] = boundary

        # --- Gradient ridge (shares the smoothed field, carried for persistence) ---
        if inflection_sigma is not None:
            smoothed, laplacian = compute_laplacian_arrays(grid_z, inflection_sigma)
            results.per_gesture_smoothed[gtype] = smoothed
            results.per_gesture_laplacian[gtype] = laplacian
            results.per_gesture_gradient_mag[gtype] = compute_gradient_magnitude(
                smoothed, np.isnan(grid_z)
            )
            grad_boundary = compute_gradient_ridge(
                grid_u, grid_v, grid_z, smoothed,
                snapshot_dir=inspection_dir, snapshot_label=gtype,
            )
        else:
            grad_boundary = None
        results.gesture_gradient_boundaries[gtype] = grad_boundary

        # --- Radial foot (only when selected) ---
        if boundary_method == "radial":
            # Per-(session, gesture) tuned sigmas when use_tuned_params is on;
            # otherwise the global BoundaryParams sigmas (unchanged default path).
            if prepared.per_gesture_params is not None:
                gp = prepared.per_gesture_params[gtype]
                radial_gauss_sigma = gp.effective_gauss_sigma()
                radial_hess_sigma = gp.radial_hess_sigma
                radial_envelope_smooth_sigma = gp.effective_envelope_smooth_sigma()
                radial_prominence = gp.effective_prominence()
                radial_plateau_size = gp.effective_plateau_size()
            else:
                radial_gauss_sigma = params.radial_gauss_sigma
                radial_hess_sigma = params.radial_hess_sigma
                radial_envelope_smooth_sigma = params.radial_envelope_smooth_sigma
                radial_prominence = params.radial_prominence
                radial_plateau_size = params.radial_plateau_size
            radial_boundary = compute_radial_foot_boundary(
                grid_u, grid_v, grid_z,
                gauss_sigma=radial_gauss_sigma,
                hess_sigma=radial_hess_sigma,
                envelope_smooth_sigma=radial_envelope_smooth_sigma,
                prominence=radial_prominence,
                plateau_size=radial_plateau_size,
                snapshot_dir=inspection_dir, snapshot_label=gtype,
                contour_color=contour_color,
            )
        else:
            radial_boundary = None
        results.gesture_radial_boundaries[gtype] = radial_boundary

        # --- Select the active boundary ---
        if boundary_method == "gradient":
            results.gesture_boundaries[gtype] = grad_boundary
        elif boundary_method == "radial":
            results.gesture_boundaries[gtype] = radial_boundary
        else:
            results.gesture_boundaries[gtype] = boundary

    return results
