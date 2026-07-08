"""Shared ray-sampling helpers for radial RF profiles and raycast figures.

These utilities cast a ray from the RF peak/centre through the response field
and locate the "foot of the mountain" pick (first positive-lambda_max plateau).
They are the single source of truth reused by both the documentation figure
generator (``scripts/generate_boundary_workflow_figures.py``) and the RF profile
extraction pipeline's raycast renderer.

Mirrors the detector logic in
``metrics/rf_radial_foot_boundary._extract_radial_plateau_foot``.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks


def uv_field_interpolator(
    field: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
) -> RegularGridInterpolator:
    """Linear interpolator of a (R, C) field over UV mm (axis0=U, axis1=V).

    Axes are flipped to strictly ascending if needed (``RegularGridInterpolator``
    requires it); NaN outside the footprint propagates as NaN (no fill fallback).
    """
    u_axis = np.asarray(grid_u[:, 0], dtype=float)
    v_axis = np.asarray(grid_v[0, :], dtype=float)
    fld = np.asarray(field, dtype=float)
    if u_axis[0] > u_axis[-1]:
        u_axis = u_axis[::-1]
        fld = fld[::-1, :]
    if v_axis[0] > v_axis[-1]:
        v_axis = v_axis[::-1]
        fld = fld[:, ::-1]
    return RegularGridInterpolator(
        (u_axis, v_axis), fld, bounds_error=False, fill_value=np.nan
    )


def ray_contour_crossing(
    peak_uv,
    direction,
    contour_uv: np.ndarray,
) -> float:
    """Smallest positive radius where the ray ``peak + t·direction`` meets the contour.

    Fail-fast: raises ``ValueError`` if the ray never crosses the closed contour
    (no fallback radius is invented).
    """
    ox, oy = float(peak_uv[0]), float(peak_uv[1])
    dx, dy = float(direction[0]), float(direction[1])
    closed = np.vstack([contour_uv, contour_uv[:1]])
    ts = []
    for i in range(len(closed) - 1):
        ax_, ay = float(closed[i, 0]), float(closed[i, 1])
        bx, by = float(closed[i + 1, 0]), float(closed[i + 1, 1])
        ex, ey = bx - ax_, by - ay
        det = ex * dy - dx * ey
        if abs(det) < 1e-12:
            continue
        t = (-(ax_ - ox) * ey + ex * (ay - oy)) / det
        s = (dx * (ay - oy) - dy * (ax_ - ox)) / det
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
            ts.append(t)
    if not ts:
        raise ValueError(
            "ray_contour_crossing: chosen ray never crosses the RF contour — "
            "cannot mark the foot location."
        )
    return min(ts)


def representative_ray_direction(peak_uv, contour_uv: np.ndarray):
    """Unit direction from the peak toward the farthest contour vertex.

    Deterministic and guaranteed to yield a long, clean foot crossing, matching
    the ray chosen in ``bd_04_ray_section.png``.

    Returns
    -------
    direction : np.ndarray, shape (2,)
        Unit vector in UV mm.
    far_point : np.ndarray, shape (2,)
        The farthest contour vertex the ray points at.
    """
    pu, pv = float(peak_uv[0]), float(peak_uv[1])
    contour = np.asarray(contour_uv, dtype=float)
    d_to_peak = np.hypot(contour[:, 0] - pu, contour[:, 1] - pv)
    far = contour[int(np.argmax(d_to_peak))]
    vec = np.array([far[0] - pu, far[1] - pv], dtype=float)
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-12:
        raise ValueError(
            "representative_ray_direction: contour is degenerate at the peak — "
            "cannot define a ray direction."
        )
    return vec / norm, far


def first_positive_plateau_radius(
    radii: np.ndarray,
    lmax_prof: np.ndarray,
):
    """Radius of the first lambda_max plateau whose crest is above zero.

    This is the detector's actual "foot of mountain" pick: the centre of the
    first ``find_peaks(plateau_size=1)`` plateau with a positive crest value.

    Returns
    -------
    float or None
        The radius (same units as ``radii``) of the pick, or None if no
        positive plateau exists along this ray.
    """
    radii = np.asarray(radii, dtype=float)
    lmax_prof = np.asarray(lmax_prof, dtype=float)
    finite = np.isfinite(lmax_prof)
    pk, props = find_peaks(np.where(finite, lmax_prof, -np.inf), plateau_size=1)
    if len(pk) == 0:
        return None
    left = np.asarray(props.get("left_edges", pk))
    right = np.asarray(props.get("right_edges", pk))
    centres = ((left + right) // 2).astype(int)
    for k, c in enumerate(centres):
        if lmax_prof[pk[k]] > 0.0:
            return float(radii[c])
    return None


def lmax_zero_crossing_radius(
    radii: np.ndarray,
    lmax_prof: np.ndarray,
):
    """Radius of the first upward lambda_max = 0 crossing (the inflection).

    This is the reference the foot pick deliberately walks past. Linear
    interpolation between the straddling samples. Returns None if the profile
    never crosses zero upward.
    """
    radii = np.asarray(radii, dtype=float)
    lmax_prof = np.asarray(lmax_prof, dtype=float)
    for i in range(1, len(lmax_prof)):
        a, b = lmax_prof[i - 1], lmax_prof[i]
        if np.isfinite(a) and np.isfinite(b) and a <= 0 < b:
            t = (0.0 - a) / (b - a)
            return float(radii[i - 1] + t * (radii[i] - radii[i - 1]))
    return None
