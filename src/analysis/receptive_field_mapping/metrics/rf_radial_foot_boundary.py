"""Radial "foot of mountain" boundary detection for population RF heatmaps.

Delineates the closed contour where the response dome meets its flat surround —
the *foot* of the soft peak — on the **Hessian λmax** field of the
Gaussian-smoothed IFF heatmap.  On a soft bump λmax (the largest Hessian
eigenvalue) is negative over the concave-down dome and turns positive at the
concave-up foot; that positive ring is the perimeter delineated here.

The contour is extracted via radial profiling: ``n_angles`` rays are cast from
the peak outward and, along each ray, the centre of the first encountered
positive λmax plateau gives the foot radius.  The radius is then **snapped** to
the last valid raw sample so no contour vertex sits on a NaN cell of the raw
heatmap.

This mirrors the structure of :mod:`rf_gradient_boundary`
(dataclass / ``compute_*`` orchestrator / ``*_to_dict`` / optional snapshot
saver), reusing the polygon + sampling helpers from
:mod:`rf_inflection_boundary` and porting only the λmax + radial-snap math from
the sandbox prototype (production must not import from ``scripts/sandbox``).

Fail-fast (CLAUDE.md): degenerate inputs raise ``ValueError`` in the inner
extractor; the public orchestrator catches that single ``ValueError`` and
returns ``None`` (mirroring the gradient/inflection orchestrators) — no silent
fallbacks, no default rings.
"""

from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.signal import find_peaks, savgol_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_contour_pca,
    compute_laplacian_arrays,
    compute_polygon_area,
    compute_polygon_centroid,
    compute_polygon_perimeter,
    contour_pixels_to_uv,
    find_peak_location,
    sample_grid_along_contour,
    sample_grid_at_uv_point,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RadialFootBoundary dataclass
# ---------------------------------------------------------------------------


@dataclass
class RadialFootBoundary:
    """Geometric metrics for the radial-foot boundary of a population RF map."""

    contour_uv: np.ndarray
    """(N, 2) UV coordinates of the radial-foot contour."""

    area_uv: float
    """Polygon area in UV space (shoelace formula)."""

    perimeter_uv: float
    """Polygon perimeter in UV space."""

    circularity: float
    """Shape compactness: 4pi * area / perimeter^2.  1.0 = perfect circle."""

    centroid_uv: tuple[float, float]
    """Polygon centroid in UV space."""

    peak_uv: tuple[float, float]
    """UV coordinates of the global response maximum (grid-cell resolution)."""

    pca_major_uv: float
    """PCA major axis length (2 sigma) in UV space."""

    pca_minor_uv: float
    """PCA minor axis length (2 sigma) in UV space."""

    pca_orientation_deg: float
    """Major axis orientation in degrees."""

    mean_iff_on_contour: float
    """Mean IFF value sampled along the contour path."""

    iff_at_centroid: float
    """IFF value sampled at the polygon centroid via bilinear interpolation."""

    lmax: np.ndarray
    """(R, C) Hessian λmax field used for this boundary."""


# ---------------------------------------------------------------------------
# NaN-aware extrapolation (ported from scripts/sandbox/contour_explorer/nan_aware.py)
# ---------------------------------------------------------------------------


def _extrapolate(fieldarr: np.ndarray) -> np.ndarray:
    """Fill NaN cells by nearest-neighbour extrapolation. Raises if all-NaN."""
    mask = np.isnan(fieldarr)
    if not mask.any():
        return fieldarr.astype(float, copy=True)
    if mask.all():
        raise ValueError("operator received an all-NaN field")
    _distances, indices = distance_transform_edt(mask, return_indices=True)
    out = fieldarr.astype(float, copy=True)
    out[mask] = fieldarr[indices[0][mask], indices[1][mask]]
    return out


# ---------------------------------------------------------------------------
# Hessian λmax field (ported from scripts/sandbox/contour_explorer/operators.py)
# ---------------------------------------------------------------------------


def _hessian_eigvals(ex: np.ndarray, sigma: float):
    """Return (lambda_max, lambda_min) eigenvalue arrays of the Hessian at *sigma*.

    ``hessian_matrix_eigvals`` returns eigenvalues sorted **descending**, so
    ``eigs[0] >= eigs[1]`` and ``eigs[0]`` is λmax.
    """
    try:
        h = hessian_matrix(ex, sigma=sigma, order="rc", use_gaussian_derivatives=True)
    except TypeError:
        # Older scikit-image without the use_gaussian_derivatives kwarg.
        h = hessian_matrix(ex, sigma=sigma, order="rc")
    eigs = hessian_matrix_eigvals(h)  # sorted descending: eigs[0] >= eigs[1]
    return eigs[0], eigs[1]


def _compute_hessian_lmax(
    grid_z: np.ndarray,
    gauss_sigma: float = 8.0,
    hess_sigma: float = 5.0,
) -> np.ndarray:
    """Largest Hessian eigenvalue (λmax) of the Gaussian-smoothed IFF field.

    Pipeline: NaN-aware Gaussian smoothing (``compute_laplacian_arrays``) →
    nearest-neighbour extrapolation of the smoothed field → Hessian at
    ``hess_sigma`` → descending eigenvalues, keep the larger.  The result is
    re-masked to the original NaN cells of ``grid_z``.

    Returns
    -------
    (R, C) λmax array, NaN where *grid_z* is NaN.
    """
    smoothed = compute_laplacian_arrays(grid_z, gauss_sigma)[0]
    ex = _extrapolate(smoothed)
    lmax, _lmin = _hessian_eigvals(ex, hess_sigma)
    lmax = np.asarray(lmax, dtype=float)
    lmax[np.isnan(grid_z)] = np.nan
    return lmax


# ---------------------------------------------------------------------------
# Radial-snap extraction (ported from
# scripts/sandbox/single_peak_contour/delineation.py — snapped path only)
# ---------------------------------------------------------------------------


def _smooth_radii_circular(
    best_radii: np.ndarray,
    savgol_window: int | None,
    savgol_polyorder: int = 3,
) -> np.ndarray:
    """Circular Savitzky-Golay smoothing of a per-angle radius profile.

    Mirrors the smoothing block in
    ``rf_gradient_boundary._extract_ridge_via_radial_profiling``: circular padding
    by ``savgol_window // 2`` on each side, then ``savgol_filter``.
    """
    if savgol_window is not None and len(best_radii) >= savgol_window:
        pad = savgol_window // 2
        padded = np.concatenate([best_radii[-pad:], best_radii, best_radii[:pad]])
        smoothed = savgol_filter(padded, savgol_window, savgol_polyorder)
        best_radii = smoothed[pad:-pad]
    return best_radii


def _extract_radial_plateau_foot(
    grid_z: np.ndarray,
    lmax: np.ndarray,
    peak_rc: tuple[int, int],
    n_angles: int = 360,
    plateau_size: int = 1,
    prominence: float | None = None,
    savgol_window: int | None = 31,
    require_positive: bool = True,
) -> dict:
    """Radial first-plateau-peak foot contour on the λmax field.

    For each of ``n_angles`` rays from ``peak_rc`` the λmax profile is sampled
    outward over its full finite extent and the vertex radius is the **centre of
    the first plateau peak** (``scipy.signal.find_peaks`` with
    ``plateau_size``/``prominence``).  That gives the **un-snapped** foot — which
    may land in the Gaussian-smoothed λmax halo just beyond the raw footprint.
    The **snapped** foot then clips each ray to its last valid raw sample so no
    vertex sits on a NaN raw cell; rays that were clipped are flagged.

    Returns ``{"contour_rc": (N, 2),          # snapped (on the raw footprint)
                "contour_unsnapped_rc": (N, 2), # natural λmax foot (may overshoot)
                "clipped_mask": (N,) bool}``.

    Raises ``ValueError`` if no ray yields any λmax foot plateau.
    """
    if n_angles < 8:
        raise ValueError(f"n_angles must be >= 8, got {n_angles}")
    n_rows, n_cols = lmax.shape
    peak_r, peak_c = peak_rc
    max_radius = int(math.ceil(math.hypot(n_rows, n_cols)))

    # NaN-fill λmax with -inf so masked cells can never be selected as a peak.
    lmax_filled = np.where(np.isnan(lmax), -np.inf, lmax)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    natural_radii = np.empty(n_angles, dtype=np.float64)
    snapped_radii = np.empty(n_angles, dtype=np.float64)
    clipped = np.zeros(n_angles, dtype=bool)
    found = np.zeros(n_angles, dtype=bool)

    for i, angle in enumerate(angles):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        radii = np.arange(1, max_radius, dtype=np.float64)
        rows = peak_r + radii * cos_a
        cols = peak_c + radii * sin_a
        inside = (rows >= 0) & (rows < n_rows - 1) & (cols >= 0) & (cols < n_cols - 1)
        if not np.any(inside):
            natural_radii[i] = snapped_radii[i] = 1.0
            clipped[i] = True
            continue
        radii = radii[inside]
        coords = np.array([rows[inside], cols[inside]])

        # Snap-back limit: index of the last valid (non-NaN) RAW sample on this ray
        # (nearest sampling so the limit is the true last contacted cell).
        raw_samp = map_coordinates(grid_z, coords, order=0, mode="constant", cval=np.nan)
        raw_valid = np.nonzero(~np.isnan(raw_samp))[0]
        raw_limit = int(raw_valid[-1]) if raw_valid.size else 0

        # Sample λmax (linear) over its FULL finite extent (which can reach past
        # the raw footprint into the Gaussian halo) — this is the un-snapped foot.
        lmax_samp = map_coordinates(lmax_filled, coords, order=1, mode="nearest")
        finite = np.isfinite(lmax_samp)
        lmax_valid = np.nonzero(finite)[0]
        if lmax_valid.size == 0:
            natural_radii[i] = snapped_radii[i] = float(radii[raw_limit])
            clipped[i] = raw_limit < (len(radii) - 1)
            continue

        peaks, props = find_peaks(
            np.where(finite, lmax_samp, -np.inf),
            plateau_size=max(1, plateau_size),
            prominence=prominence,
        )
        # Plateau centre for every detected peak (find_peaks reports the right
        # edge as the peak index; recover the centre from the plateau edges).
        left = props.get("left_edges", peaks)
        right = props.get("right_edges", peaks)
        centres = ((np.asarray(left) + np.asarray(right)) // 2).astype(int)

        natural_idx = None
        for k, centre in enumerate(centres):
            if require_positive and lmax_samp[peaks[k]] <= 0.0:
                continue
            natural_idx = int(centre)
            break
        if natural_idx is None:
            # No qualifying foot: fall back to the λmax extent edge (still flagged).
            natural_idx = int(lmax_valid[-1])
        else:
            found[i] = True

        snapped_idx = min(natural_idx, raw_limit)
        clipped[i] = snapped_idx < natural_idx
        natural_radii[i] = float(radii[natural_idx])
        snapped_radii[i] = float(radii[snapped_idx])

    if not np.any(found):
        raise ValueError(
            "radial-plateau: no λmax foot plateau found on any ray "
            "(field too flat or peak runs off the data footprint)"
        )

    snapped_radii = _smooth_radii_circular(snapped_radii, savgol_window)
    natural_radii = _smooth_radii_circular(natural_radii, savgol_window)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    contour_rc = np.column_stack([peak_r + snapped_radii * cos_a,
                                  peak_c + snapped_radii * sin_a])
    unsnapped_rc = np.column_stack([peak_r + natural_radii * cos_a,
                                    peak_c + natural_radii * sin_a])
    return {
        "contour_rc": contour_rc,
        "contour_unsnapped_rc": unsnapped_rc,
        "clipped_mask": clipped,
    }


# ---------------------------------------------------------------------------
# Snapshot rendering
# ---------------------------------------------------------------------------


def _render_radial_snapshot(
    grid_z: np.ndarray,
    lmax: np.ndarray,
    contour_rc: np.ndarray,
    peak_rc: tuple[int, int],
    contour_color: str = "green",
) -> "plt.Figure":
    """Hessian λmax field with the radial-foot contour overlaid."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_z, ax_lmax) = plt.subplots(1, 2, figsize=(12, 5))

    ax_z.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax_z.plot(contour_rc[:, 1], contour_rc[:, 0], "-", color=contour_color, linewidth=1.5)
    ax_z.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax_z.set_title("IFF field + radial foot")

    lmax_abs = float(np.nanmax(np.abs(lmax)))
    lmax_abs = max(lmax_abs, 1e-12)
    ax_lmax.imshow(
        np.ma.masked_invalid(lmax), cmap="RdBu_r",
        origin="upper", vmin=-lmax_abs, vmax=lmax_abs,
    )
    ax_lmax.plot(contour_rc[:, 1], contour_rc[:, 0], "-", color=contour_color, linewidth=1.5)
    ax_lmax.plot(peak_rc[1], peak_rc[0], "rx", markersize=10, markeredgewidth=2)
    ax_lmax.set_title("Hessian λmax")

    fig.suptitle("Radial foot boundary", fontsize=10)
    fig.tight_layout()
    return fig


def _render_radial_uv(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    contour_uv: np.ndarray,
    centroid_uv: tuple[float, float],
    area_uv: float,
    circularity: float,
    contour_color: str = "green",
) -> "plt.Figure":
    """Radial-foot contour in UV space with metric annotations."""
    import matplotlib.pyplot as plt

    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(grid_u, grid_v, np.ma.masked_invalid(grid_z), cmap=jet, shading="auto")

    closed = np.vstack([contour_uv, contour_uv[:1]])
    ax.plot(closed[:, 0], closed[:, 1], "-", color=contour_color, linewidth=1.5)

    cu, cv = centroid_uv
    ax.plot(cu, cv, "+", color=contour_color, markersize=12, markeredgewidth=2)

    ax.text(
        0.02, 0.98,
        f"area={area_uv:.4f}  circ={circularity:.3f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_aspect("equal")
    ax.set_title("UV-space radial foot contour")

    fig.tight_layout()
    return fig


def _save_radial_snapshots(
    grid_z: np.ndarray,
    lmax: np.ndarray,
    contour_rc: np.ndarray,
    contour_uv: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    peak_rc: tuple[int, int],
    centroid_uv: tuple[float, float],
    area_uv: float,
    circularity: float,
    snapshot_dir: pathlib.Path,
    snapshot_label: str,
    contour_color: str = "green",
) -> None:
    """Save diagnostic PNGs for the radial-foot boundary pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = f"radial_{snapshot_label}" if snapshot_label else "radial"

    fig1 = _render_radial_snapshot(grid_z, lmax, contour_rc, peak_rc, contour_color)
    path1 = pathlib.Path(snapshot_dir) / f"{prefix}_foot.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    fig2 = _render_radial_uv(
        grid_u, grid_v, grid_z, contour_uv, centroid_uv,
        area_uv, circularity, contour_color,
    )
    path2 = pathlib.Path(snapshot_dir) / f"{prefix}_foot_uv.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    logger.debug("radial_snapshots: saved %s, %s", path1, path2)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compute_radial_foot_boundary(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    grid_z: np.ndarray,
    gauss_sigma: float = 8.0,
    hess_sigma: float = 5.0,
    n_angles: int = 360,
    savgol_window: int | None = 31,
    plateau_size: int = 1,
    prominence: float | None = None,
    require_positive: bool = True,
    snapshot_dir: pathlib.Path | None = None,
    snapshot_label: str = "",
    contour_color: str = "green",
) -> RadialFootBoundary | None:
    """Detect the radial "foot of mountain" boundary on a 2D IFF heatmap.

    The foot is the concave-up ring where λmax (largest Hessian eigenvalue of the
    Gaussian-smoothed field) turns positive — the perimeter where the response
    dome meets its surround.  The contour is extracted by casting ``n_angles``
    rays from the peak outward and taking, per ray, the centre of the first
    positive λmax plateau; each radius is then snapped to the last valid raw
    sample so no vertex sits on a NaN cell.

    Parameters
    ----------
    grid_u, grid_v:
        (R, C) coordinate grids (axis 0 = U, axis 1 = V).
    grid_z:
        (R, C) raw IFF values; NaN where no data exists.
    gauss_sigma:
        Sigma for the NaN-aware Gaussian pre-smoothing of grid_z.
    hess_sigma:
        Sigma for the Hessian derivative kernel.
    n_angles:
        Number of radial rays for contour extraction.
    savgol_window:
        Savitzky-Golay window for contour smoothing; ``None`` to skip.
    plateau_size:
        Minimum plateau size passed to ``scipy.signal.find_peaks``.
    prominence:
        Optional prominence passed to ``scipy.signal.find_peaks``.
    require_positive:
        Require the selected λmax plateau peak to be > 0 (the concave-up foot).
    snapshot_dir:
        Directory for diagnostic PNGs; ``None`` to skip.
    snapshot_label:
        Label appended to snapshot filenames.
    contour_color:
        Color for the contour in diagnostic snapshots.

    Returns
    -------
    RadialFootBoundary or None if no valid boundary can be extracted.
    """
    if grid_z is None:
        logger.warning("radial_foot: grid_z is None")
        return None

    if np.all(np.isnan(grid_z)):
        logger.debug("radial_foot: all-NaN grid, shape=%s", grid_z.shape)
        return None

    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        logger.warning("radial_foot: find_peak_location returned None")
        return None

    n_rows, n_cols = grid_z.shape
    pr, pc = peak_rc
    if pr < 2 or pr >= n_rows - 2 or pc < 2 or pc >= n_cols - 2:
        logger.warning(
            "radial_foot: peak at border — peak_rc=(%d, %d), grid=%dx%d",
            pr, pc, n_rows, n_cols,
        )
        return None

    lmax = _compute_hessian_lmax(grid_z, gauss_sigma, hess_sigma)

    try:
        res = _extract_radial_plateau_foot(
            grid_z, lmax, peak_rc,
            n_angles=n_angles, plateau_size=plateau_size,
            prominence=prominence, savgol_window=savgol_window,
            require_positive=require_positive,
        )
    except ValueError as exc:
        logger.warning("radial_foot: extractor raised — %s", exc)
        return None

    contour_rc = res["contour_rc"]
    contour_uv = contour_pixels_to_uv(contour_rc, grid_u, grid_v)

    area_uv = compute_polygon_area(contour_uv)
    perimeter_uv = compute_polygon_perimeter(contour_uv)
    circularity = (
        4.0 * math.pi * area_uv / (perimeter_uv ** 2)
        if perimeter_uv > 0 else float("nan")
    )
    centroid_uv = compute_polygon_centroid(contour_uv)

    peak_rc_arr = np.array([[peak_rc[0], peak_rc[1]]], dtype=float)
    peak_uv_arr = contour_pixels_to_uv(peak_rc_arr, grid_u, grid_v)
    peak_uv = (float(peak_uv_arr[0, 0]), float(peak_uv_arr[0, 1]))

    pca_major, pca_minor, pca_orientation_deg = compute_contour_pca(contour_uv)
    mean_iff = sample_grid_along_contour(grid_z, contour_rc)
    iff_at_centroid = sample_grid_at_uv_point(grid_z, grid_u, grid_v, centroid_uv)

    if snapshot_dir is not None:
        _save_radial_snapshots(
            grid_z=grid_z, lmax=lmax,
            contour_rc=contour_rc, contour_uv=contour_uv,
            grid_u=grid_u, grid_v=grid_v, peak_rc=peak_rc,
            centroid_uv=centroid_uv, area_uv=area_uv,
            circularity=circularity,
            snapshot_dir=snapshot_dir, snapshot_label=snapshot_label,
            contour_color=contour_color,
        )

    logger.info(
        "radial_foot: SUCCESS — contour_pts=%d, area_uv=%.4f, "
        "circularity=%.3f, centroid_uv=(%.3f, %.3f), iff_at_centroid=%.4f",
        len(contour_uv), area_uv, circularity,
        centroid_uv[0], centroid_uv[1], iff_at_centroid,
    )

    return RadialFootBoundary(
        contour_uv=contour_uv,
        area_uv=area_uv,
        perimeter_uv=perimeter_uv,
        circularity=circularity,
        centroid_uv=centroid_uv,
        peak_uv=peak_uv,
        pca_major_uv=pca_major,
        pca_minor_uv=pca_minor,
        pca_orientation_deg=pca_orientation_deg,
        mean_iff_on_contour=mean_iff,
        iff_at_centroid=iff_at_centroid,
        lmax=lmax,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def radial_foot_boundary_to_dict(boundary: RadialFootBoundary) -> dict:
    """Serialize RadialFootBoundary to a JSON-safe dict.

    ``lmax`` is excluded (too large for JSON); everything else follows the same
    format as ``inflection_boundary_to_dict``.
    """
    def _safe(val: float) -> float | None:
        if isinstance(val, float) and math.isnan(val):
            return None
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val

    return {
        "contour_uv": [[float(pt[0]), float(pt[1])] for pt in boundary.contour_uv],
        "area_uv": _safe(boundary.area_uv),
        "perimeter_uv": _safe(boundary.perimeter_uv),
        "circularity": _safe(boundary.circularity),
        "centroid_uv": [_safe(boundary.centroid_uv[0]), _safe(boundary.centroid_uv[1])],
        "peak_uv": [_safe(boundary.peak_uv[0]), _safe(boundary.peak_uv[1])],
        "pca_major_uv": _safe(boundary.pca_major_uv),
        "pca_minor_uv": _safe(boundary.pca_minor_uv),
        "pca_orientation_deg": _safe(boundary.pca_orientation_deg),
        "mean_iff_on_contour": _safe(boundary.mean_iff_on_contour),
        "iff_at_centroid": _safe(boundary.iff_at_centroid),
    }
