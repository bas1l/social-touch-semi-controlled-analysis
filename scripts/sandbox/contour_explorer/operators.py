"""Preprocessing operator implementations  (fn(field, **params) -> field).

Every operator is NaN-aware (it never invents data outside the contacted region)
and raises ``ValueError`` on degenerate parameters — no silent fallbacks
(CLAUDE.md).  Gaussian / Laplacian / gradient-magnitude operators delegate to the
existing analysis metric functions so the math stays identical to the pipeline.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, median_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, meijering, prewitt, sato, scharr, sobel

# Reused, single-sourced pipeline math -----------------------------------------
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_laplacian_arrays,
)
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    compute_gradient_magnitude,
)

from .nan_aware import _extrapolate, _nan_aware


def _require_sigma(sigma: float) -> None:
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")


def op_gaussian(fieldarr, sigma):
    _require_sigma(sigma)
    return compute_laplacian_arrays(fieldarr, sigma)[0]


def op_median(fieldarr, size):
    size = int(size)
    if size < 1 or size % 2 == 0:
        raise ValueError(f"median size must be a positive odd integer, got {size}")
    mask = np.isnan(fieldarr)
    out = median_filter(_extrapolate(fieldarr), size=size)
    out = out.astype(float)
    out[mask] = np.nan
    return out


def op_gradient_magnitude(fieldarr, sigma):
    _require_sigma(sigma)
    smoothed = compute_laplacian_arrays(fieldarr, sigma)[0]
    return compute_gradient_magnitude(smoothed, np.isnan(fieldarr))


def op_sobel(fieldarr):
    return _nan_aware(fieldarr, sobel)


def op_scharr(fieldarr):
    return _nan_aware(fieldarr, scharr)


def op_prewitt(fieldarr):
    return _nan_aware(fieldarr, prewitt)


def op_directional(fieldarr, sigma, angle_deg):
    _require_sigma(sigma)
    mask = np.isnan(fieldarr)
    ex = _extrapolate(fieldarr)
    d_u = gaussian_filter(ex, sigma=sigma, order=(1, 0))  # d/d axis0 (U)
    d_v = gaussian_filter(ex, sigma=sigma, order=(0, 1))  # d/d axis1 (V)
    angle = math.radians(angle_deg)
    out = d_u * math.cos(angle) + d_v * math.sin(angle)
    out[mask] = np.nan
    return out


def op_laplacian(fieldarr, sigma):
    _require_sigma(sigma)
    return compute_laplacian_arrays(fieldarr, sigma)[1]


def op_log(fieldarr, sigma):
    _require_sigma(sigma)
    return _nan_aware(fieldarr, lambda x: gaussian_laplace(x, sigma=sigma))


def _hessian_eigvals(ex: np.ndarray, sigma: float):
    """Return (lambda_max, lambda_min) eigenvalue arrays of the Hessian at *sigma*."""
    try:
        h = hessian_matrix(ex, sigma=sigma, order="rc", use_gaussian_derivatives=True)
    except TypeError:
        # Older scikit-image without the use_gaussian_derivatives kwarg.
        h = hessian_matrix(ex, sigma=sigma, order="rc")
    eigs = hessian_matrix_eigvals(h)  # sorted descending: eigs[0] >= eigs[1]
    return eigs[0], eigs[1]


def op_hessian_eigval(fieldarr, sigma):
    _require_sigma(sigma)
    mask = np.isnan(fieldarr)
    lmax, _lmin = _hessian_eigvals(_extrapolate(fieldarr), sigma)
    lmax = np.asarray(lmax, dtype=float)
    lmax[mask] = np.nan
    return lmax


def op_doh(fieldarr, sigma):
    _require_sigma(sigma)
    mask = np.isnan(fieldarr)
    lmax, lmin = _hessian_eigvals(_extrapolate(fieldarr), sigma)
    out = np.asarray(lmax * lmin, dtype=float)
    out[mask] = np.nan
    return out


def _ridge_sigmas(sigma_min, sigma_max):
    if not (sigma_min > 0):
        raise ValueError(f"sigma_min must be > 0, got {sigma_min}")
    if sigma_max < sigma_min:
        raise ValueError(f"sigma_max ({sigma_max}) must be >= sigma_min ({sigma_min})")
    return np.linspace(sigma_min, sigma_max, 5)


def op_frangi(fieldarr, sigma_min, sigma_max, black_ridges):
    sigmas = _ridge_sigmas(sigma_min, sigma_max)
    return _nan_aware(
        fieldarr, lambda x: frangi(x, sigmas=sigmas, black_ridges=bool(black_ridges))
    )


def op_sato(fieldarr, sigma_min, sigma_max, black_ridges):
    sigmas = _ridge_sigmas(sigma_min, sigma_max)
    return _nan_aware(
        fieldarr, lambda x: sato(x, sigmas=sigmas, black_ridges=bool(black_ridges))
    )


def op_meijering(fieldarr, sigma_min, sigma_max, black_ridges):
    sigmas = _ridge_sigmas(sigma_min, sigma_max)
    return _nan_aware(
        fieldarr, lambda x: meijering(x, sigmas=sigmas, black_ridges=bool(black_ridges))
    )


def op_gaussian_nth(fieldarr, sigma, order_u, order_v):
    _require_sigma(sigma)
    order_u, order_v = int(order_u), int(order_v)
    if not (0 <= order_u <= 4 and 0 <= order_v <= 4):
        raise ValueError(f"orders must be in 0..4, got ({order_u}, {order_v})")
    return _nan_aware(
        fieldarr, lambda x: gaussian_filter(x, sigma=sigma, order=(order_u, order_v))
    )


def op_dog(fieldarr, sigma_low, sigma_high):
    if not (sigma_low > 0 and sigma_high > 0):
        raise ValueError(f"both sigmas must be > 0, got ({sigma_low}, {sigma_high})")
    return _nan_aware(
        fieldarr,
        lambda x: gaussian_filter(x, sigma=sigma_low) - gaussian_filter(x, sigma=sigma_high),
    )
