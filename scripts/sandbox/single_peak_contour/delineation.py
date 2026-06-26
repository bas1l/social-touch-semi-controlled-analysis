"""Foot-contour delineation of a single soft peak on the Hessian λmax field.

Both extractors take the **raw** heatmap ``grid_z`` (NaN outside the contacted
region) and the **processed** field ``lmax`` = the largest Hessian eigenvalue of
the Gaussian-smoothed raw field (computed by the fixed pipeline in :mod:`app`).
On a soft bump λmax is negative over the dome (concave-down) and turns positive at
the foot (concave-up); that positive ring is the perimeter we delineate.

Two definitions are provided so they can be compared on real data:

``extract_radial_plateau_foot``
    Cast ``n_angles`` rays from the seed and, along each ray, take the **peak of
    the first encountered plateau** of λmax (it rises, plateaus, then falls).

``extract_region_growth_foot``
    Flood the connected concave region (``λmax <= level``) containing the seed —
    bounded by the positive foot ridge — and trace its outer boundary.

Snap-back rule (both): a vertex may never land outside the visible (``grid_z > 0``)
heatmap — the bright region painted in the Heatmap tab (below-threshold ``-1.0``
and NaN cells are both excluded). The radial method clips each ray to its last
positive raw sample; the region method intersects the grown region with the
positive raw mask.

Fail-fast (CLAUDE.md): degenerate inputs raise ``ValueError`` — no silent
fallbacks, no default rings.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter1d, label, map_coordinates
from scipy.signal import find_peaks, savgol_filter
from skimage.measure import find_contours

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    contour_pixels_to_uv,
    find_peak_location,
)

__all__ = [
    "find_peak_location",
    "contour_pixels_to_uv",
    "extract_radial_plateau_foot",
    "extract_region_growth_foot",
    "envelope_contour_to_footprint",
]


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


def _smooth_closed_contour(contour_rc: np.ndarray, sigma: float | None) -> np.ndarray:
    """Light circular Gaussian smoothing of a closed (row, col) contour.

    Smooths the marching-squares staircase ("teeth") by Gaussian-filtering the row
    and col coordinate sequences along the loop with ``mode="wrap"`` so the seam is
    continuous. ``sigma`` is in units of vertices; ``None``/``<= 0`` is a no-op.
    The duplicated closing vertex is dropped before filtering and re-appended after,
    so it is not double-weighted at the seam.
    """
    if sigma is None or sigma <= 0 or len(contour_rc) < 4:
        return contour_rc
    pts = contour_rc
    closed = bool(np.allclose(pts[0], pts[-1]))
    if closed:
        pts = pts[:-1]
    rows = gaussian_filter1d(pts[:, 0], sigma, mode="wrap")
    cols = gaussian_filter1d(pts[:, 1], sigma, mode="wrap")
    out = np.column_stack([rows, cols])
    if closed:
        out = np.vstack([out, out[0]])
    return out


def _select_enclosing_contour(
    binary: np.ndarray,
    peak_rc: tuple[int, int],
) -> np.ndarray:
    """Return the largest marching-squares contour of *binary* that encloses the peak.

    Raises ``ValueError`` if no contour encloses the peak (fail-fast).
    """
    contours = find_contours(binary.astype(float), 0.5)
    peak_r, peak_c = peak_rc
    enclosing: list[tuple[float, np.ndarray]] = []
    for c in contours:
        if len(c) < 4:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if not poly.contains_point((peak_r, peak_c)):
            continue
        area = 0.5 * abs(
            np.dot(c[:, 0], np.roll(c[:, 1], 1))
            - np.dot(c[:, 1], np.roll(c[:, 0], 1))
        )
        enclosing.append((area, c))
    if not enclosing:
        raise ValueError("region-growth: no contour encloses the peak")
    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def extract_radial_plateau_foot(
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
    The **snapped** foot then clips each ray to its last positive raw sample so no
    vertex sits outside the visible (``grid_z > 0``) heatmap; rays that were
    clipped are flagged.

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

        # Snap-back limit: index of the last positive RAW sample on this ray
        # (nearest sampling so the limit is the true last visible cell). The
        # heatmap tab paints only grid_z > 0, so clip on that same criterion;
        # below-threshold (-1.0) and NaN cells are both excluded (NaN > 0 is False).
        raw_samp = map_coordinates(grid_z, coords, order=0, mode="constant", cval=np.nan)
        raw_valid = np.nonzero(raw_samp > 0)[0]
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


def extract_region_growth_foot(
    grid_z: np.ndarray,
    lmax: np.ndarray,
    peak_rc: tuple[int, int],
    level: float = 0.0,
) -> dict:
    """Region-growth foot contour: connected concave basin around the seed.

    The region is ``(λmax <= level) & (grid_z > 0)``; the connected component
    containing the seed is bounded by the positive foot ridge. Its outer boundary
    (largest contour enclosing the seed) is the foot contour. Constraining the
    region to the positive (``grid_z > 0``) raw mask is the region method's snap-back.

    Returns ``{"contour_rc": (N, 2), "region_mask": (R, C) bool}``.

    Raises ``ValueError`` if the seed is not in the concave region, or no
    enclosing boundary can be traced.
    """
    peak_r, peak_c = peak_rc
    valid = grid_z > 0
    concave = np.zeros(lmax.shape, dtype=bool)
    concave[valid & ~np.isnan(lmax)] = lmax[valid & ~np.isnan(lmax)] <= level

    if not concave[peak_r, peak_c]:
        seed_val = lmax[peak_r, peak_c]
        raise ValueError(
            f"region-growth: seed λmax={seed_val:.4e} > level={level:.4e} — "
            "peak is not in a concave (λmax<=level) cell; lower the level"
        )

    labeled, _n = label(concave)
    seed_label = int(labeled[peak_r, peak_c])
    region_mask = labeled == seed_label

    contour_rc = _select_enclosing_contour(region_mask, peak_rc)
    return {"contour_rc": contour_rc, "region_mask": region_mask}


def envelope_contour_to_footprint(
    contour_rc: np.ndarray,
    grid_z: np.ndarray,
    peak_rc: tuple[int, int],
    smooth_sigma: float | None = None,
) -> dict:
    """2D (XY-only) envelope: clip a contour to the visible heatmap footprint.

    The radial foot is a star-convex, one-radius-per-angle curve anchored at the
    seed, so its straight chords bulge into non-painted (grey) cells wherever the
    footprint is concave or split into detached islands. This post-pass removes
    that overshoot using only the 2D XY footprint, ignoring all field heights:

      1. ``footprint = grid_z > 0`` — the exact mask the Heatmap tab paints
         (below-threshold ``-1.0`` and NaN both excluded, since ``NaN > 0`` is
         False).
      2. Rasterise the closed ``contour_rc`` polygon into a filled (row, col)
         mask and intersect it with the footprint.
      3. Keep the connected component containing the seed.
      4. Re-trace its outer boundary (largest contour enclosing the seed).

      5. Optionally apply a very light circular Gaussian smoothing
         (``smooth_sigma``, in vertices) to take the staircase ("teeth") off the
         marching-squares trace.

    Where the input contour spilled past the footprint the boundary now follows
    the footprint edge; where it sat inside, it follows the input contour. The
    marching-squares trace **redefines the point group** of the contour.

    Returns ``{"contour_rc": (M, 2)}``.

    Raises ``ValueError`` (fail-fast, no fallback ring) if the polygon does not
    enclose the seed, or the polygon ∩ footprint is empty at the seed.
    """
    peak_r, peak_c = peak_rc
    n_rows, n_cols = grid_z.shape

    rr, cc = np.mgrid[0:n_rows, 0:n_cols]
    pts = np.column_stack([rr.ravel(), cc.ravel()])
    inside = MplPath(contour_rc).contains_points(pts).reshape(n_rows, n_cols)

    footprint = grid_z > 0
    region = inside & footprint

    if not region[peak_r, peak_c]:
        raise ValueError(
            "envelope: seed is not inside the contour ∩ footprint intersection — "
            "cannot envelope an empty region (check the radial contour and seed)"
        )

    labeled, _n = label(region)
    seed_label = int(labeled[peak_r, peak_c])
    region_mask = labeled == seed_label

    contour_rc = _select_enclosing_contour(region_mask, peak_rc)
    contour_rc = _smooth_closed_contour(contour_rc, smooth_sigma)
    return {"contour_rc": contour_rc}
