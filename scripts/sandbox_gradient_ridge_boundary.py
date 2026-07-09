"""Proof-of-concept sandbox for the gradient-ridge RF boundary method.

The gradient-ridge boundary (``rf_gradient_boundary.compute_gradient_ridge``)
recently replaced the Laplacian inflection boundary as the canonical RF
boundary, but its results are not satisfying.  This standalone script
isolates that method on a *single, hardcoded* session so the researcher can
see — step by step, in 2D and 3D — what the method is sensitive to and how
that compares to where the response-field signal actually lives.

It does **not** touch the pipeline.  It loads the already-computed population
response-field heatmap (``grid_z``) from the session's
``*_population_response_fields.npz`` and re-runs the gradient method *live*,
so every tunable (Gaussian sigma, number of rays, Savitzky-Golay window) can
be changed at the top of ``main()`` and the effect inspected immediately.

What each figure shows
----------------------
  Fig 1  Pipeline steps (2D): raw heatmap → smoothed → Laplacian →
         gradient magnitude → both contours overlaid on the heatmap.
  Fig 2  Radial-profiling diagnostic — the heart of the method.  Rays cast
         from the peak, and, for a few angles, the |grad z| profile (what the
         method maximises) next to the normalised response profile (where the
         signal of interest actually decays).  This reveals that the ridge
         sits on the *steepest slope*, not at the RF edge.
  Fig 3  Sigma sensitivity sweep — gradient-ridge contour for several smoothing
         sigmas, plus boundary area vs sigma.
  Fig 4  3D response surface with both contours drawn at their true height —
         shows the ridge clinging to the flank of the bell.
  Fig 5  3D forearm mesh coloured by the heatmap, with the gradient-ridge
         contour back-projected onto the surface (real anatomical view).
  Fig 6  "Foot of the mountain" candidates vs the gradient ridge.  Three
         foot detectors are prototyped to trace the *base* of the response
         (where the flat background bends upward into the peak), rather than
         the inflection ring the gradient ridge catches:
           - curvature foot — per-ray maximum upward (concave-up) profile
             curvature beyond the inflection (√3·σ ≈ 1.73σ for a Gaussian);
           - kneedle foot — inline chord-deviation knee on the decreasing
             response profile (dependency-free Kneedle);
           - threshold foot — a non-radial iso-level reference ring at a low
             fraction (~20%) of the peak.
         Left panel overlays all four contours on grid_z with the data
         boundary; right panel shows sample-ray z(r) profiles with each
         method's foot marked plus the z''(r) curvature trace; a text panel
         reports per-method mean radius and height (% of peak).

Display
-------
By default (``USE_PLT_SHOW = True``) every figure stays open at the end and you
navigate between the windows with the keyboard: ←/→ (or n/p) flips to the
previous/next figure, q closes them all.  This needs a real GUI backend, so run
it from a plain terminal.  Under VS Code's debugger set ``USE_PLT_SHOW = False``
— debugpy forces matplotlib to the Agg backend and ``plt.show`` hangs; in that
mode each figure is saved as a PNG and opened with ``os.startfile`` instead.
Either way the PNGs are written to the output folder.

Usage
-----
    python code/scripts/sandbox_gradient_ridge_boundary.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

# --- Make the analysis package importable without installing -----------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import matplotlib.pyplot as plt
# NOTE: the analysis package imports below call matplotlib.use("Agg") at import
# time, so the interactive backend cannot be selected here — it is re-asserted at
# runtime in main() via _ensure_interactive_backend(), which must run AFTER them.
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401  (registers 3d)
from scipy.ndimage import gaussian_laplace, label as ndi_label, map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from skimage.measure import find_contours
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
    compute_laplacian_arrays,
    contour_pixels_to_uv,
    find_peak_location,
)
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    _extract_ridge_via_radial_profiling,
    compute_gradient_magnitude,
    compute_gradient_ridge,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import uv_points_to_xyz
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    compute_threshold_from_ratio,
)
from analysis.receptive_field_mapping.data.rf_grid_interpolation import (
    compute_interpolated_grid,
)


# =============================================================================
# Display helper
# =============================================================================

# True  → keep every figure open and navigate between them with the arrow keys
#         at the end (interactive; needs a real GUI backend, e.g. a terminal run).
# False → save each PNG and open it with os.startfile (debugger-safe; VS Code's
#         debugpy forces the Agg backend and plt.show hangs — set this False there).
USE_PLT_SHOW = True

# Figures kept alive for interactive navigation: list of (figure, title) pairs.
_FIGS: list[tuple["plt.Figure", str]] = []

# Set in main(): True only when USE_PLT_SHOW and an interactive backend was secured.
_INTERACTIVE = False


def _ensure_interactive_backend() -> bool:
    """Force an interactive matplotlib backend, returning True on success.

    The analysis package imports call ``matplotlib.use("Agg")`` at import time,
    so we must switch back to a GUI backend at runtime (after those imports).
    Tries Qt then Tk; returns False if none is available (e.g. under debugpy),
    in which case the caller falls back to saving + opening PNGs.
    """
    for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            print(f"  interactive backend: {backend}")
            return True
        except Exception:
            continue
    print("  WARNING: no interactive backend available — opening saved PNGs instead.")
    return False


def _emit(fig: "plt.Figure", out_dir: Path, name: str) -> None:
    """Save the figure, then either keep it open for navigation or open the PNG."""
    path = out_dir / name
    fig.savefig(path, dpi=140)
    print(f"  saved {path.name}")
    if _INTERACTIVE:
        # Keep the figure alive; main() wires navigation and calls plt.show().
        _FIGS.append((fig, path.stem))
    else:
        plt.close(fig)
        try:
            os.startfile(str(path))  # noqa: B606  (Windows-only, intentional)
        except Exception as exc:  # pragma: no cover - platform dependent
            print(f"  (could not auto-open {path.name}: {exc})")


def _wire_navigation() -> None:
    """Let the user flip between all open figure windows with the keyboard.

    ←/→ (or n/p) raises the previous/next figure; q closes them all. Window
    titles are numbered so the current position is always visible.
    """
    figs = [f for f, _ in _FIGS]
    n = len(figs)
    if n == 0:
        return

    for i, (fig, title) in enumerate(_FIGS):
        try:
            fig.canvas.manager.set_window_title(
                f"[{i + 1}/{n}] {title}   (←/→ or n/p to navigate, q to close)"
            )
        except Exception:
            pass

    def _raise(idx: int) -> None:
        fig = figs[idx % n]
        mgr = getattr(fig.canvas, "manager", None)
        win = getattr(mgr, "window", None)
        if win is None:
            return
        for attr in ("activateWindow", "raise_", "lift", "focus_force"):
            method = getattr(win, attr, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

    def _on_key(event) -> None:
        if event.canvas.figure not in figs:
            return
        cur = figs.index(event.canvas.figure)
        if event.key in ("right", "n", "pagedown"):
            _raise(cur + 1)
        elif event.key in ("left", "p", "pageup"):
            _raise(cur - 1)
        elif event.key == "q":
            plt.close("all")

    for fig in figs:
        fig.canvas.mpl_connect("key_press_event", _on_key)


# =============================================================================
# Data loading
# =============================================================================


def load_grid(npz_path: Path, gtype: str):
    """Load the interpolated heatmap grid + mesh + stored boundaries for one gesture.

    Returns a dict with grid_u/grid_v/grid_z, the forearm mesh arrays, the
    stored inflection_sigma the pipeline used, and the stored gradient/inflection
    contours (for reference comparison).
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)

    available = list(d["gesture_types"])
    if gtype not in available:
        raise ValueError(
            f"gesture '{gtype}' not in NPZ. Available: {available}"
        )

    def _get(stem: str):
        key = f"{stem}_{gtype}"
        if key not in d:
            raise KeyError(f"missing NPZ key '{key}'")
        return d[key]

    out = {
        "grid_u": _get("grid_u"),
        "grid_v": _get("grid_v"),
        "grid_z": _get("grid_z"),
        "forearm_uv": d["forearm_uv"],
        "forearm_faces": d["forearm_faces"],
        "forearm_V": d["forearm_V"],
        "heatmap_pre_threshold": _get("heatmap_pre_threshold"),
        "unique_count": _get("unique_count"),
        "n_touches": int(d[f"n_touches_{gtype}"]),
        "pipeline_sigma": float(d["inflection_sigma"]),
        "pipeline_boundary_method": str(d["boundary_method"]) if "boundary_method" in d else "?",
        "pipeline_min_overlap_pct": float(d["min_overlap_pct"]),
        "gesture_types": available,
        "session_id": str(d["session_id"]),
    }
    # Stored reference contours (whatever the pipeline produced).
    out["stored_gradient_contour_uv"] = (
        d[f"gradient_contour_uv_{gtype}"] if f"gradient_contour_uv_{gtype}" in d else None
    )
    out["stored_inflection_contour_uv"] = (
        d[f"inflection_contour_uv_{gtype}"] if f"inflection_contour_uv_{gtype}" in d else None
    )
    return out


# =============================================================================
# Faithful radial-profile sampler (mirrors _extract_ridge_via_radial_profiling)
# =============================================================================


def sample_ray(field: np.ndarray, peak_rc, angle: float):
    """Sample *field* along a single ray from peak_rc at *angle*.

    Mirrors the sampling used inside ``_extract_ridge_via_radial_profiling``
    so the diagnostic faithfully reflects what the method sees.

    Returns (radii, values) — both 1D, values bilinearly interpolated with
    NaN cells treated as 0 (as the method does).
    """
    n_rows, n_cols = field.shape
    peak_r, peak_c = peak_rc
    max_radius = int(math.ceil(math.hypot(n_rows, n_cols)))
    filled = np.where(np.isnan(field), 0.0, field)

    radii = np.arange(1, max_radius, dtype=np.float64)
    rows = peak_r + radii * math.cos(angle)
    cols = peak_c + radii * math.sin(angle)
    valid = (rows >= 0) & (rows < n_rows - 1) & (cols >= 0) & (cols < n_cols - 1)
    radii, rows, cols = radii[valid], rows[valid], cols[valid]
    if radii.size == 0:
        return np.array([]), np.array([])
    values = map_coordinates(filled, np.array([rows, cols]), order=1, mode="nearest")
    return radii, values


# =============================================================================
# "Foot of the mountain" extractors (Phase 1)
# =============================================================================
#
# Each extractor mirrors the structure of
# ``rf_gradient_boundary._extract_ridge_via_radial_profiling``: cast ``n_angles``
# rays from the peak, choose one radius per ray, smooth the radii circularly with
# Savitzky-Golay, and return an (N, 2) array of (row, col) contour points (the
# UV conversion is done by the caller with ``contour_pixels_to_uv``, exactly as
# the gradient-ridge orchestrator does).  Only the *per-ray selection rule*
# differs.  Per the fail-fast convention, an all-degenerate field (no ray yields
# a foot beyond the inflection) raises ``ValueError`` rather than returning a
# default ring; per-ray degeneracy mirrors the ridge code (radius floored to a
# sentinel and flagged), and if *every* ray is degenerate the whole field is
# rejected.


def _smooth_radii_circular(
    best_radii: np.ndarray,
    savgol_window: int | None,
    savgol_polyorder: int = 3,
) -> np.ndarray:
    """Circular Savitzky-Golay smoothing of a per-angle radius profile.

    Identical to the smoothing block in ``_extract_ridge_via_radial_profiling``:
    circular padding by ``savgol_window // 2`` on each side, then ``savgol_filter``.
    """
    if savgol_window is not None and len(best_radii) >= savgol_window:
        pad = savgol_window // 2
        padded = np.concatenate([best_radii[-pad:], best_radii, best_radii[:pad]])
        smoothed_radii = savgol_filter(padded, savgol_window, savgol_polyorder)
        best_radii = smoothed_radii[pad:-pad]
    return best_radii


def extract_foot_curvature(
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    peak_rc,
    n_angles: int = 360,
    savgol_window: int | None = 31,
    profile_savgol_window: int = 11,
    profile_savgol_polyorder: int = 3,
) -> np.ndarray:
    """Foot via maximum *upward* (concave-up) profile curvature beyond the inflection.

    For a Gaussian ``A·exp(-r²/2σ²)`` the radial profile ``z(r)`` has its steepest
    slope (inflection, max ``|z'|``) at ``r = σ`` and its maximum positive (concave
    up) curvature at ``r = √3·σ ≈ 1.73σ`` — the "toe of the slope" where the flank
    flattens into the background.  This selects, per ray, the radius of maximum
    ``z''(r)`` restricted to radii *beyond* the per-ray inflection.

    ``z''(r)`` is computed with ``savgol_filter(deriv=2)`` on the ``smoothed``
    profile sampled along each ray.  Rays are clipped at the last valid (non-NaN
    in ``grid_z``) sample so a foot is never placed on the data-mask edge.

    Returns (N, 2) (row, col) contour points.  Raises ``ValueError`` if no ray
    yields a valid foot.
    """
    if profile_savgol_window % 2 == 0:
        raise ValueError(
            f"profile_savgol_window must be odd, got {profile_savgol_window}"
        )

    peak_r, peak_c = peak_rc
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_radii = np.zeros(n_angles)
    found = np.zeros(n_angles, dtype=bool)

    for i, angle in enumerate(angles):
        radii, zvals = sample_ray(smoothed, peak_rc, angle)
        if radii.size == 0:
            continue

        # Clip at the last valid (non-NaN) sample of the RAW field so the foot is
        # never placed on the Gaussian halo beyond the data mask.
        _, zraw = sample_ray(grid_z, peak_rc, angle)
        valid_raw = ~np.isnan(zraw)
        if not np.any(valid_raw):
            continue
        last_valid = int(np.max(np.nonzero(valid_raw)[0]))
        radii = radii[: last_valid + 1]
        zvals = zvals[: last_valid + 1]

        if radii.size < profile_savgol_window:
            continue

        # First and second derivatives of z(r) along the ray.
        dz = savgol_filter(
            zvals, profile_savgol_window, profile_savgol_polyorder, deriv=1
        )
        d2z = savgol_filter(
            zvals, profile_savgol_window, profile_savgol_polyorder, deriv=2
        )

        # Per-ray inflection = radius of steepest descent (max |z'|).  On the
        # decreasing flank z' < 0, so the inflection is argmin(dz).
        infl_idx = int(np.argmin(dz))

        # Beyond the inflection, pick the maximum positive (concave-up) curvature.
        beyond = np.arange(infl_idx + 1, len(d2z))
        if beyond.size == 0:
            continue
        d2z_beyond = d2z[beyond]
        if float(np.max(d2z_beyond)) <= 0.0:
            # No concave-up shoulder beyond the inflection on this ray.
            continue
        sel = beyond[int(np.argmax(d2z_beyond))]
        best_radii[i] = float(radii[sel])
        found[i] = True

    if not np.any(found):
        raise ValueError(
            "extract_foot_curvature: no ray yielded a concave-up foot beyond the "
            "inflection — field too flat/degenerate to define a foot of the mountain"
        )

    # Fill degenerate rays with the median found radius so the contour stays closed
    # (the radii are then circularly smoothed, mirroring the ridge code).
    best_radii[~found] = float(np.median(best_radii[found]))
    best_radii = _smooth_radii_circular(best_radii, savgol_window)

    contour_rows = peak_r + best_radii * np.cos(angles)
    contour_cols = peak_c + best_radii * np.sin(angles)
    return np.column_stack([contour_rows, contour_cols])


def extract_foot_kneedle(
    grid_z: np.ndarray,
    peak_rc,
    n_angles: int = 360,
    savgol_window: int | None = 31,
) -> np.ndarray:
    """Foot via an inline Kneedle knee-point on the decreasing response profile.

    For each ray, take the monotone-decreasing portion of ``z(r)`` from the peak,
    normalise both ``r`` and ``z`` to ``[0, 1]``, subtract the straight chord from
    first→last sample, and take the radius of maximum deviation (``argmax``) — the
    knee of a convex, decreasing curve.  This is the dependency-free Kneedle rule
    (no ``kneed`` import).

    Rays are clipped at the last valid (non-NaN) sample of ``grid_z`` so the knee
    cannot land on the data-mask edge.  Returns (N, 2) (row, col) contour points;
    raises ``ValueError`` if no ray yields a valid knee.
    """
    peak_r, peak_c = peak_rc
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    best_radii = np.zeros(n_angles)
    found = np.zeros(n_angles, dtype=bool)

    for i, angle in enumerate(angles):
        radii, zvals = sample_ray(grid_z, peak_rc, angle)
        if radii.size == 0:
            continue

        valid = ~np.isnan(zvals)
        if not np.any(valid):
            continue
        last_valid = int(np.max(np.nonzero(valid)[0]))
        radii = radii[: last_valid + 1]
        zvals = zvals[: last_valid + 1]

        # Monotone-decreasing portion from the peak: keep up to (and including)
        # the first sample that reaches the profile minimum, so the chord-deviation
        # is taken over a convex, decreasing curve.
        end = int(np.argmin(zvals)) + 1
        if end < 3:
            continue
        r_seg = radii[:end]
        z_seg = zvals[:end]

        r_span = float(r_seg[-1] - r_seg[0])
        z_span = float(z_seg[0] - z_seg[-1])
        if r_span <= 0.0 or z_span <= 0.0:
            continue

        r_norm = (r_seg - r_seg[0]) / r_span
        z_norm = (z_seg - z_seg[-1]) / z_span  # 1 at peak end, 0 at far end.
        # Straight chord from first (1.0) to last (0.0): chord = 1 - r_norm.
        chord = 1.0 - r_norm
        deviation = z_norm - chord
        if float(np.max(deviation)) <= 0.0:
            continue
        knee = int(np.argmax(deviation))
        best_radii[i] = float(r_seg[knee])
        found[i] = True

    if not np.any(found):
        raise ValueError(
            "extract_foot_kneedle: no ray yielded a knee — response profiles are "
            "not convex-decreasing enough to define a foot of the mountain"
        )

    best_radii[~found] = float(np.median(best_radii[found]))
    best_radii = _smooth_radii_circular(best_radii, savgol_window)

    contour_rows = peak_r + best_radii * np.cos(angles)
    contour_cols = peak_c + best_radii * np.sin(angles)
    return np.column_stack([contour_rows, contour_cols])


def extract_foot_threshold(
    grid_z: np.ndarray,
    peak_rc,
    frac: float = 0.2,
) -> np.ndarray:
    """Foot as a simple iso-level reference ring at ``frac * peak``.

    Not radial: fills NaNs to the field minimum, runs
    ``skimage.measure.find_contours`` at ``frac * peak``, and selects the contour
    whose polygon encloses the peak.  A reference ring only.

    Returns (N, 2) (row, col) contour points; raises ``ValueError`` if no
    iso-contour at the requested level encloses the peak.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError(f"frac must be in (0, 1), got {frac}")

    peak_r, peak_c = peak_rc
    peak_val = float(np.nanmax(grid_z))
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        raise ValueError(
            f"extract_foot_threshold: peak value non-positive/non-finite "
            f"({peak_val}) — cannot threshold"
        )

    level = frac * peak_val
    filled = np.where(np.isnan(grid_z), np.nanmin(grid_z), grid_z)
    contours = find_contours(filled, level=level)
    if not contours:
        raise ValueError(
            f"extract_foot_threshold: find_contours found no contour at level "
            f"{level:.4f} ({100 * frac:.0f}% of peak)"
        )

    # Select the contour whose polygon encloses the peak (point-in-polygon),
    # preferring the largest enclosing one if several qualify.
    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            f"extract_foot_threshold: no iso-contour at {100 * frac:.0f}% of peak "
            f"encloses the peak at (row,col)={peak_rc}"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def extract_foot_watershed(
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    peak_rc,
    min_peak_distance: int = 15,
    peak_threshold_rel: float = 0.1,
) -> np.ndarray:
    """Foot via the watershed catchment of the target peak, cut at saddles.

    Non-radial.  Treats the smoothed response surface as a topography and floods
    its *inverted* form (``-smoothed``) from a set of markers, one per maximum.
    The basin growing from the target peak's marker stops along the watershed
    lines (valleys/saddles) where it meets a neighbour peak's basin — that
    catchment line is the "foot of the mountain" against competing peaks, and the
    NaN/low-response region elsewhere.  Unlike the radial foot extractors this
    needs no per-ray casting, so it represents irregular, multi-lobed extents
    faithfully.

    Algorithm
    ---------
    1. Find competing maxima with ``skimage.feature.peak_local_max`` on
       ``smoothed`` (``min_distance=min_peak_distance``, ``threshold_rel=
       peak_threshold_rel``), restricted to the valid (non-NaN ``grid_z``) region.
    2. Build an integer ``markers`` array: the target peak (the
       ``peak_local_max`` detection nearest ``peak_rc``, snapped to it if absent)
       is label ``1``; every other detected maximum gets its own distinct label
       (``2, 3, ...``); the low/background region (valid pixels below
       ``peak_threshold_rel`` of the peak) seeds a single background label so the
       target basin is also bounded on its open (non-neighbour) flanks.
    3. Run ``watershed(-smoothed, markers, mask=~np.isnan(grid_z))`` and take the
       region ``labels == 1``.
    4. ``find_contours((labels == 1).astype(float), 0.5)`` and select the contour
       whose polygon encloses ``peak_rc`` (largest, if several qualify) — the
       same point-in-polygon rule used by ``extract_foot_threshold``.

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if the field is all-NaN/degenerate, if no local maximum
    is detected, if the target catchment is empty, if ``find_contours`` returns
    nothing, or if no contour encloses the peak.  Never returns a default ring or
    sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if not (0 < min_peak_distance):
        raise ValueError(
            f"min_peak_distance must be positive, got {min_peak_distance}"
        )
    if not (0.0 <= peak_threshold_rel < 1.0):
        raise ValueError(
            f"peak_threshold_rel must be in [0, 1), got {peak_threshold_rel}"
        )

    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))
    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_watershed: grid_z is all-NaN — no surface to flood"
        )

    peak_val = float(np.nanmax(smoothed))
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        raise ValueError(
            f"extract_foot_watershed: smoothed peak value non-positive/non-finite "
            f"({peak_val}) — field too flat/degenerate to flood"
        )

    # NaNs cannot drive peak detection or flooding; replace with the field
    # minimum so they read as the lowest topography (deepest "sea").
    fill_low = float(np.nanmin(smoothed))
    surface = np.where(np.isnan(smoothed), fill_low, smoothed)

    # --- 1. Competing maxima (and the target) via peak_local_max -------------
    coords = peak_local_max(
        surface,
        min_distance=min_peak_distance,
        threshold_rel=peak_threshold_rel,
        labels=valid.astype(int),
        exclude_border=False,
    )
    if coords.shape[0] == 0:
        raise ValueError(
            "extract_foot_watershed: peak_local_max found no maxima — field too "
            "flat/degenerate to define a watershed catchment"
        )

    # --- 2. Markers: target peak = 1, other maxima = 2.., background = last ---
    markers = np.zeros(surface.shape, dtype=np.int32)

    # Identify which detected maximum is the target (nearest to peak_rc).
    dists = np.hypot(coords[:, 0] - peak_r, coords[:, 1] - peak_c)
    target_idx = int(np.argmin(dists))
    target_coord = coords[target_idx]

    markers[int(target_coord[0]), int(target_coord[1])] = 1
    next_label = 2
    for i, (r, c) in enumerate(coords):
        if i == target_idx:
            continue
        markers[int(r), int(c)] = next_label
        next_label += 1

    # Background marker: valid, low-response pixels (below the peak fraction).
    # This bounds the target basin on flanks that face no neighbour peak.
    background = valid & (surface <= peak_threshold_rel * peak_val)
    # Do not overwrite any maximum marker already placed.
    background &= markers == 0
    if np.any(background):
        markers[background] = next_label
        next_label += 1

    # --- 3. Flood the inverted surface within the valid region ---------------
    labels = watershed(-surface, markers, mask=valid)

    target_region = labels == 1
    if not np.any(target_region):
        raise ValueError(
            "extract_foot_watershed: the target peak's catchment is empty after "
            "watershed — marker placement or mask is degenerate"
        )

    # --- 4. Contour of the target catchment, enclosing the peak --------------
    contours = find_contours(target_region.astype(float), level=0.5)
    if not contours:
        raise ValueError(
            "extract_foot_watershed: find_contours found no boundary for the "
            "target catchment region"
        )

    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            f"extract_foot_watershed: no watershed-catchment contour encloses the "
            f"peak at (row,col)={peak_rc}"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def extract_foot_prominence(
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    peak_rc,
    min_prominence_frac: float = 0.1,
) -> np.ndarray:
    """Foot via topological prominence: the saddle/col where the peak merges.

    Non-radial.  Treats the smoothed response surface as a topography and runs
    an inline 0-D persistent-homology flood (union-find).  Pixels are added one
    at a time from highest to lowest value; each new pixel joins the connected
    components of its already-added 8-neighbours.  When two components touch, the
    *elder rule* keeps the older (taller-seeded) component alive and "kills" the
    younger one — the value at which the younger dies is the **saddle level**
    (the col).  The target peak's prominence is ``peak_value - saddle_level``.
    The iso-contour of ``smoothed`` at that saddle level is the parameter-light,
    non-radial extent of the one mountain whose summit is the target peak — it is
    exactly the highest closed contour that still encircles the target peak alone
    before it spills over the col into a taller neighbour.

    Algorithm
    ---------
    1. Restrict to valid pixels (``~np.isnan(grid_z)``); sort them descending by
       ``smoothed`` value.
    2. Union-find over valid pixels.  Add pixels one at a time in that order.
       Each component remembers its *seed* (the highest pixel that started it).
       For every already-added 8-neighbour, union: the component with the higher
       seed value (the elder) absorbs the other.  At a merge of two *distinct*
       live components, the younger (lower-seed) component dies at the current
       pixel's value — that is a saddle level for the younger summit.
    3. The target peak is the valid maximum nearest ``peak_rc``.  Record the
       saddle level at which the target peak's component is first absorbed into an
       elder (taller) component — but only accept that merge if its prominence
       (``peak_value - saddle_level``) exceeds ``min_prominence_frac * peak_value``;
       otherwise it is a shallow noise merge and flooding continues, so the
       saddle is taken at the first *significant* merge.
    4. Boundary = ``find_contours(smoothed, saddle_level)`` selecting the contour
       whose polygon encloses the peak (largest, if several qualify) — the same
       point-in-polygon rule used by ``extract_foot_threshold``.

    The ``min_prominence_frac`` knob guards against shallow noise merges: a peak
    that merges into a neighbour over a negligible col (prominence below the
    floor) is treated as the same mountain and flooding continues to the next,
    deeper col.

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if ``grid_z`` is all-NaN, if the smoothed peak value is
    non-positive/non-finite, if the target peak never merges into a taller
    component at any level whose prominence clears the floor (e.g. it is the
    global summit and no significant col exists), if ``find_contours`` returns
    nothing at the saddle level, or if no contour encloses the peak.  Never
    returns a default ring or sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if not (0.0 <= min_prominence_frac < 1.0):
        raise ValueError(
            f"min_prominence_frac must be in [0, 1), got {min_prominence_frac}"
        )

    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_prominence: grid_z is all-NaN — no surface to flood"
        )

    peak_val = float(np.nanmax(smoothed))
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        raise ValueError(
            f"extract_foot_prominence: smoothed peak value non-positive/non-finite "
            f"({peak_val}) — field too flat/degenerate to flood"
        )

    n_rows, n_cols = smoothed.shape
    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))

    # --- Identify the target pixel: the valid maximum nearest peak_rc ---------
    # Use the exact (snapped) peak pixel if it is valid; otherwise fall back to
    # the nearest valid pixel by Euclidean distance, weighted toward high values.
    if valid[peak_r, peak_c]:
        target_flat = peak_r * n_cols + peak_c
    else:
        vr, vc = np.nonzero(valid)
        d = (vr - peak_r) ** 2 + (vc - peak_c) ** 2
        nearest = int(np.argmin(d))
        target_flat = int(vr[nearest]) * n_cols + int(vc[nearest])

    # --- Sort valid pixels descending by smoothed value -----------------------
    flat_valid = np.flatnonzero(valid)
    order = flat_valid[np.argsort(-smoothed.ravel()[flat_valid], kind="stable")]

    # --- Inline union-find over the pixel grid --------------------------------
    NONE = -1
    parent = np.full(smoothed.size, NONE, dtype=np.int64)   # NONE = not yet added
    seed = np.full(smoothed.size, NONE, dtype=np.int64)     # representative -> seed pixel

    def _find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # 8-connectivity neighbour offsets.
    neigh = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    saddle_level = None  # set when the target's component dies into a taller one

    for flat in order:
        r = flat // n_cols
        c = flat % n_cols
        # Birth: the pixel starts its own component, seeded by itself.
        parent[flat] = flat
        seed[flat] = flat

        for dr, dc in neigh:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                continue
            nflat = nr * n_cols + nc
            if parent[nflat] == NONE:
                continue  # neighbour not added yet (lower value, or invalid)

            root_a = _find(flat)
            root_b = _find(nflat)
            if root_a == root_b:
                continue  # already the same component

            # Elder rule: higher-seed component (taller summit) is the elder and
            # survives; the younger dies at this pixel's value (the saddle level).
            val_a = float(smoothed.ravel()[seed[root_a]])
            val_b = float(smoothed.ravel()[seed[root_b]])
            if val_a >= val_b:
                elder, younger = root_a, root_b
            else:
                elder, younger = root_b, root_a

            target_root = _find(target_flat) if parent[target_flat] != NONE else NONE

            if younger == target_root:
                # The target peak's component is being absorbed into a taller one.
                level = float(smoothed.ravel()[flat])
                prominence = peak_val - level
                if prominence > min_prominence_frac * peak_val:
                    saddle_level = level
                    break

            # Merge younger into elder; the elder keeps its (taller) seed.
            elder_seed = seed[elder]
            parent[younger] = elder
            seed[elder] = elder_seed

        if saddle_level is not None:
            break

    if saddle_level is None:
        raise ValueError(
            "extract_foot_prominence: the target peak never merged into a taller "
            f"component above the prominence floor (min_prominence_frac="
            f"{min_prominence_frac}) — it is the global summit or no significant "
            "col exists; cannot define a foot of the mountain"
        )

    # --- Iso-contour at the saddle level, enclosing the peak ------------------
    filled = np.where(np.isnan(smoothed), float(np.nanmin(smoothed)), smoothed)
    contours = find_contours(filled, level=saddle_level)
    if not contours:
        raise ValueError(
            f"extract_foot_prominence: find_contours found no contour at the saddle "
            f"level {saddle_level:.4f}"
        )

    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            f"extract_foot_prominence: no iso-contour at the saddle level "
            f"{saddle_level:.4f} encloses the peak at (row,col)={peak_rc}"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def extract_foot_curvature_field(
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    peak_rc,
    sigma: float = 5.0,
) -> np.ndarray:
    """Foot via the non-radial outer convex-up break of the curvature field.

    Non-radial analog of ``extract_foot_curvature``.  The geomorphological
    *footslope* is the convex-up break where the mountain flank flattens into the
    background — in curvature terms the transition from the concave summit cap
    (Laplacian < 0) outward to the convex-up toe (Laplacian > 0).  Instead of
    casting one ray per angle, this traces that break directly on the 2D
    Laplacian field and selects the closed break that wraps the whole peak basin,
    so it represents irregular, multi-lobed extents faithfully.

    Algorithm
    ---------
    1. Recompute the NaN-aware smoothed field and its Laplacian with
       ``compute_laplacian_arrays(grid_z, sigma)`` (REUSED — same convention as
       the inflection boundary; Laplacian < 0 is concave-up summit, > 0 is the
       convex-up basin/footslope).
    2. **Peak (inflection) basin** = the connected component of the concave
       region (Laplacian < 0) that contains the peak pixel.  Its outer rim is the
       inflection ring; the footslope must lie strictly *outside* it.
    3. Extract every zero-crossing of the Laplacian via
       ``find_contours(laplacian, 0.0)`` — each is a candidate convex-up break.
    4. Among breaks whose polygon **encloses the peak**, drop any that lies inside
       (does not strictly contain) the inflection basin's own rim, then take the
       innermost remaining break — the *first* convex-up break outside the
       inflection basin (the foot of the mountain, not a distant background
       ripple).  Selection uses the same point-in-polygon rule as
       ``extract_foot_threshold``.

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if ``grid_z`` is all-NaN, if the peak pixel is not in a
    concave region (no summit cap to bound), if ``find_contours`` yields no
    zero-crossing, or if no convex-up break both encloses the peak and lies
    outside the inflection basin.  Never returns a default ring or sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if not (sigma > 0.0):
        raise ValueError(f"sigma must be positive, got {sigma}")

    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_curvature_field: grid_z is all-NaN — no surface to "
            "analyse"
        )

    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))

    # --- 1. NaN-aware smoothed field + Laplacian (REUSE the inflection helper) -
    _smoothed_sig, laplacian = compute_laplacian_arrays(grid_z, sigma)

    # --- 2. Connected concave (Laplacian < 0) basin containing the peak --------
    # The summit cap is concave-up (negative Laplacian).  Restrict to valid
    # pixels so the data-mask halo cannot leak into the basin.
    concave = (laplacian < 0.0) & valid
    if not concave[peak_r, peak_c]:
        raise ValueError(
            "extract_foot_curvature_field: the peak pixel is not in a concave "
            "(Laplacian<0) region — no summit cap to bound a footslope against"
        )

    from scipy.ndimage import label as _ndlabel

    labels_cc, _n = _ndlabel(concave)
    peak_label = int(labels_cc[peak_r, peak_c])
    if peak_label == 0:
        raise ValueError(
            "extract_foot_curvature_field: failed to label the peak's concave "
            "basin"
        )
    peak_basin = labels_cc == peak_label

    # Outer rim of the inflection basin (the concave summit cap around the peak).
    basin_contours = find_contours(peak_basin.astype(float), level=0.5)
    if not basin_contours:
        raise ValueError(
            "extract_foot_curvature_field: could not contour the peak's "
            "inflection basin"
        )

    from matplotlib.path import Path as MplPath

    # Largest basin contour enclosing the peak = the inflection rim.
    basin_enclosing = []
    for c in basin_contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            basin_enclosing.append((area, c))
    if not basin_enclosing:
        raise ValueError(
            "extract_foot_curvature_field: no inflection-basin rim encloses the "
            f"peak at (row,col)={peak_rc}"
        )
    basin_enclosing.sort(key=lambda t: t[0])
    basin_rim = basin_enclosing[-1][1]
    basin_poly = MplPath(np.column_stack([basin_rim[:, 0], basin_rim[:, 1]]))

    # --- 3. Zero-crossings of the Laplacian = candidate convex-up breaks -------
    # NaN cells block contouring; fill them with the field minimum so the break
    # is found only inside the valid region.
    lap_filled = np.where(np.isnan(laplacian), float(np.nanmin(laplacian)), laplacian)
    breaks = find_contours(lap_filled, level=0.0)
    if not breaks:
        raise ValueError(
            "extract_foot_curvature_field: find_contours found no Laplacian "
            "zero-crossing — field too flat to define a convex-up break"
        )

    # --- 4. Innermost break that encloses the peak yet lies outside the basin --
    # "Outside the inflection basin" = the break is not contained within the
    # basin rim; we require the basin centroid/peak to be inside the break (so it
    # wraps the whole summit) and the break to be larger than (extend beyond) the
    # basin rim.  Choose the smallest such break — the *first* footslope outward.
    basin_area = 0.5 * abs(
        np.dot(basin_rim[:, 0], np.roll(basin_rim[:, 1], 1))
        - np.dot(basin_rim[:, 1], np.roll(basin_rim[:, 0], 1))
    )

    candidates = []
    for c in breaks:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if not poly.contains_point((peak_r, peak_c)):
            continue
        area = 0.5 * abs(
            np.dot(c[:, 0], np.roll(c[:, 1], 1))
            - np.dot(c[:, 1], np.roll(c[:, 0], 1))
        )
        # Footslope must lie strictly outside the inflection basin: it has to be
        # bigger than the basin rim and contain the rim's points.
        if area <= basin_area:
            continue
        if not np.all(poly.contains_points(
            np.column_stack([basin_rim[:, 0], basin_rim[:, 1]])
        )):
            continue
        candidates.append((area, c))

    if not candidates:
        raise ValueError(
            "extract_foot_curvature_field: no convex-up break both encloses the "
            f"peak at (row,col)={peak_rc} and lies outside the inflection basin "
            "— field has no footslope break beyond the summit cap"
        )

    # Innermost (smallest-area) qualifying break = the first footslope outward.
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def extract_foot_spill_point(
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    peak_rc,
    n_levels: int = 200,
    detrend: bool = False,
) -> np.ndarray:
    """Foot via the spill point: the level where the peak's basin floods outward.

    Non-radial superlevel-set method.  Imagine slowly lowering a water line over
    the smoothed response surface from just below the peak toward the background.
    The connected component that holds the peak grows slowly while the water sits
    on the steep flank of the mountain, then *suddenly* engulfs a large flat area
    the moment the water drops below the surrounding col/shoulder — that jump is
    the **spill point**, the level at which the peak's catchment spills into the
    background plateau.  The iso-contour of ``smoothed`` at that spill level is the
    non-radial foot of the mountain: it wraps the whole basin (irregular,
    multi-lobed extents included) rather than assuming a star-convex single peak.

    Algorithm
    ---------
    1. Work on ``smoothed`` restricted to ``valid = ~np.isnan(grid_z)``.  Optionally
       remove a planar background first (``detrend=True``): least-squares fit
       ``z ≈ a·row + b·col + c`` over valid pixels and subtract it, so a tilted
       background does not smear the spill knee.  The plane fit is fail-fast: a
       rank-deficient design matrix (e.g. all-collinear valid pixels) raises.
    2. Sweep a threshold ``t`` over ``n_levels`` values from just below the peak
       value down toward the background (valid-region) minimum.  At each ``t``,
       label the superlevel set ``field >= t`` with ``ndi_label`` (8-connectivity),
       take the component containing ``peak_rc``, and record its pixel area.
    3. The spill level is the ``t`` of maximum growth rate ``dArea/d(-t)`` — the
       discrete forward difference of the recorded areas as ``t`` decreases.  This
       is the knee where the basin abruptly floods the background.
    4. Boundary = ``find_contours(field, spill_level)`` selecting the contour whose
       polygon encloses the peak (largest, if several qualify) — the same
       point-in-polygon rule used by ``extract_foot_threshold``.

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if ``grid_z`` is all-NaN, if the smoothed peak value is
    non-positive/non-finite, if ``detrend`` is requested but the plane fit is
    degenerate, if the peak's component never grows across the sweep (no spill),
    if the growth-rate knee is undefined, if ``find_contours`` returns nothing at
    the spill level, or if no contour encloses the peak.  Never returns a default
    ring or sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if not (n_levels >= 2):
        raise ValueError(f"n_levels must be >= 2, got {n_levels}")

    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_spill_point: grid_z is all-NaN — no surface to flood"
        )

    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))
    n_rows, n_cols = smoothed.shape

    # --- 1. Field to sweep: smoothed, optionally planar-detrended --------------
    field = np.array(smoothed, dtype=np.float64, copy=True)
    if detrend:
        vr, vc = np.nonzero(valid)
        zv = smoothed[vr, vc]
        finite = np.isfinite(zv)
        vr, vc, zv = vr[finite], vc[finite], zv[finite]
        if vr.size < 3:
            raise ValueError(
                "extract_foot_spill_point: fewer than 3 finite valid pixels — "
                "cannot fit a planar background to detrend"
            )
        design = np.column_stack([vr.astype(np.float64), vc.astype(np.float64),
                                  np.ones(vr.size)])
        # Fail-fast on a degenerate (rank-deficient) plane fit.
        rank = int(np.linalg.matrix_rank(design))
        if rank < 3:
            raise ValueError(
                "extract_foot_spill_point: planar background fit is degenerate "
                f"(design-matrix rank {rank} < 3) — valid pixels are collinear; "
                "cannot detrend"
            )
        coeffs, *_ = np.linalg.lstsq(design, zv, rcond=None)
        a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        rows_idx = np.arange(n_rows)[:, None]
        cols_idx = np.arange(n_cols)[None, :]
        plane = a * rows_idx + b * cols_idx + c
        field = field - plane

    # NaN cells must not join the peak's component or seed contours; drop them to
    # the valid-region minimum so they read as the lowest topography.
    field = np.where(valid, field, np.nan)
    valid_vals = field[valid]
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    if valid_vals.size == 0:
        raise ValueError(
            "extract_foot_spill_point: no finite valid samples after detrend — "
            "cannot define a spill sweep"
        )
    field = np.where(np.isnan(field), float(valid_vals.min()), field)

    peak_val = float(field[peak_r, peak_c]) if valid[peak_r, peak_c] else float(valid_vals.max())
    if not np.isfinite(peak_val):
        raise ValueError(
            "extract_foot_spill_point: peak sample is non-finite — cannot sweep"
        )
    bg_val = float(valid_vals.min())
    if not (peak_val > bg_val):
        raise ValueError(
            f"extract_foot_spill_point: peak value ({peak_val:.4f}) does not exceed "
            f"the background minimum ({bg_val:.4f}) — field too flat/degenerate to "
            "define a spill point"
        )

    # --- 2. Sweep thresholds from just below the peak toward the background ----
    # Descending levels so the peak's component only grows as the sweep advances.
    span = peak_val - bg_val
    levels = np.linspace(peak_val - span / n_levels, bg_val, n_levels)
    areas = np.empty(n_levels, dtype=np.float64)
    for k, t in enumerate(levels):
        superlevel = (field >= t) & valid
        labels_sl, _n = ndi_label(superlevel)
        peak_label = int(labels_sl[peak_r, peak_c])
        if peak_label == 0:
            areas[k] = 0.0
        else:
            areas[k] = float(np.count_nonzero(labels_sl == peak_label))

    if float(np.max(areas)) <= 0.0:
        raise ValueError(
            "extract_foot_spill_point: the peak's superlevel component never grew "
            "across the sweep — peak pixel may sit on the data mask or field is "
            "degenerate"
        )

    # --- 3. Spill level = argmax of the growth rate dArea/d(-t) ----------------
    # t decreases along the sweep, so d(-t) > 0; the forward difference of areas
    # is the growth as the water line drops one step.
    d_area = np.diff(areas)
    dt = -np.diff(levels)  # positive step in (-t)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(dt > 0, d_area / dt, 0.0)
    if not np.any(np.isfinite(growth)) or float(np.nanmax(growth)) <= 0.0:
        raise ValueError(
            "extract_foot_spill_point: the basin growth-rate knee is undefined "
            "(no positive dArea/d(-t)) — no spill detected"
        )
    knee = int(np.nanargmax(growth))
    # The spill happens between levels[knee] and levels[knee+1]; take the lower
    # (already-flooded) level as the iso-contour level for the basin extent.
    spill_level = float(levels[knee + 1])

    # --- 4. Iso-contour at the spill level, enclosing the peak -----------------
    contours = find_contours(field, level=spill_level)
    if not contours:
        raise ValueError(
            f"extract_foot_spill_point: find_contours found no contour at the spill "
            f"level {spill_level:.4f}"
        )

    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            f"extract_foot_spill_point: no iso-contour at the spill level "
            f"{spill_level:.4f} encloses the peak at (row,col)={peak_rc}"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def extract_foot_log_blob(
    grid_z: np.ndarray,
    peak_rc,
    sigma_min: float = 2.0,
    sigma_max: float = 30.0,
    n_sigma: int = 20,
) -> np.ndarray:
    """Foot via scale-space blob detection: the normalized-LoG characteristic size.

    Non-radial scale-space method.  A bell-shaped response is a *blob*, and blob
    detection answers "how big is this blob?" by finding the Gaussian scale at
    which a normalized Laplacian-of-Gaussian (LoG) filter responds most strongly
    at the blob centre.  The scale-normalized LoG ``σ²·∇²(G_σ * z)`` has, for a
    Gaussian blob of width ``s``, an extremum in σ at ``σ* ∝ s`` — the *intrinsic*
    characteristic radius of the mountain, read directly off the data rather than
    assumed.  The LoG **zero-crossing ring** at that best scale traces the
    inflection circle of the matched blob, which is taken here as the foot of the
    mountain: it wraps the whole peak basin (irregular, multi-lobed extents
    included) without star-convex per-ray casting.

    Algorithm
    ---------
    1. Build a NaN-filled copy of ``grid_z`` (NaNs → the valid-region minimum, the
       flat "background" floor) so ``gaussian_laplace`` does not propagate NaN.
       This NaN-fill is a documented *model assumption* — the masked region is
       treated as flat background — NOT a silent fallback; it asserts against an
       all-NaN field.
    2. For each σ on a log-spaced grid in ``[sigma_min, sigma_max]``, compute the
       scale-normalized LoG response ``resp = σ²·gaussian_laplace(filled, σ)``.  A
       bright blob gives a strongly *negative* LoG at its centre, so the matched
       scale is ``σ* = argmax_σ (-resp[peak_rc])``.  Record
       ``log_characteristic_sigma = σ*``.
    3. Boundary = LoG zero-crossing ring of ``resp`` at ``σ*``:
       ``find_contours(resp_sigma_star, 0.0)``, selecting the contour whose polygon
       encloses the peak (largest, if several qualify) — the same point-in-polygon
       rule used by ``extract_foot_threshold``.

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if ``grid_z`` is all-NaN, if the LoG response at the peak
    never has a negative-going extremum within ``[sigma_min, sigma_max]`` (i.e. the
    best ``-resp[peak_rc]`` is non-positive, or the extremum sits at a grid
    endpoint so the characteristic scale lies outside the sampled range), if
    ``find_contours`` returns no zero-crossing at ``σ*``, or if no zero-crossing
    ring encloses the peak.  Never returns a default ring or sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if not (sigma_min > 0.0):
        raise ValueError(f"sigma_min must be positive, got {sigma_min}")
    if not (sigma_max > sigma_min):
        raise ValueError(
            f"sigma_max ({sigma_max}) must exceed sigma_min ({sigma_min})"
        )
    if not (n_sigma >= 2):
        raise ValueError(f"n_sigma must be >= 2, got {n_sigma}")

    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_log_blob: grid_z is all-NaN — no surface to analyse"
        )

    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))

    # --- 1. NaN-filled copy: masked region read as the flat background floor ----
    # Model assumption (NOT a fallback): the data mask is flat background, so NaNs
    # take the valid-region minimum.  ``gaussian_laplace`` would otherwise spread
    # NaN across the whole response.
    fill_low = float(np.nanmin(grid_z))
    filled = np.where(np.isnan(grid_z), fill_low, grid_z).astype(np.float64)

    # --- 2. Scale sweep: σ* maximizes the negative-going LoG at the peak --------
    sigmas = np.geomspace(sigma_min, sigma_max, n_sigma)
    neg_resp_at_peak = np.empty(n_sigma, dtype=np.float64)
    responses = []
    for k, sigma in enumerate(sigmas):
        resp = (sigma ** 2) * gaussian_laplace(filled, sigma)
        responses.append(resp)
        neg_resp_at_peak[k] = -float(resp[peak_r, peak_c])

    best_k = int(np.argmax(neg_resp_at_peak))
    if neg_resp_at_peak[best_k] <= 0.0:
        raise ValueError(
            "extract_foot_log_blob: the normalized-LoG response at the peak is "
            "never negative-going within "
            f"[{sigma_min}, {sigma_max}] — no blob signature at the peak (field too "
            "flat or peak sits on the background floor)"
        )
    if best_k == 0 or best_k == n_sigma - 1:
        raise ValueError(
            "extract_foot_log_blob: the LoG response peaks at a σ-grid endpoint "
            f"(σ*={float(sigmas[best_k]):.3f}) — the characteristic scale lies "
            f"outside the sampled range [{sigma_min}, {sigma_max}]; widen it"
        )

    log_characteristic_sigma = float(sigmas[best_k])  # noqa: F841 (recorded scale)
    resp_sigma_star = responses[best_k]

    # --- 3. LoG zero-crossing ring at σ*, enclosing the peak -------------------
    contours = find_contours(resp_sigma_star, level=0.0)
    if not contours:
        raise ValueError(
            "extract_foot_log_blob: find_contours found no LoG zero-crossing at "
            f"σ*={log_characteristic_sigma:.3f}"
        )

    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            "extract_foot_log_blob: no LoG zero-crossing ring at "
            f"σ*={log_characteristic_sigma:.3f} encloses the peak at "
            f"(row,col)={peak_rc}"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def _elliptical_gaussian_2d(
    coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset
):
    """Rotated 2-D elliptical Gaussian, raveled for ``curve_fit``.

    ``coords`` is a ``(rows, cols)`` pair of equally-shaped index arrays; the
    surface ``offset + amplitude·exp(-(a·dx² + 2b·dx·dy + c·dy²))`` is evaluated
    and returned flattened.  ``theta`` rotates the principal axes.
    """
    r, c = coords
    dr = r - x0
    dc = c - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = (cos_t ** 2) / (2 * sigma_x ** 2) + (sin_t ** 2) / (2 * sigma_y ** 2)
    b = (-np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (
        4 * sigma_y ** 2
    )
    cc = (sin_t ** 2) / (2 * sigma_x ** 2) + (cos_t ** 2) / (2 * sigma_y ** 2)
    surface = offset + amplitude * np.exp(
        -(a * dr ** 2 + 2 * b * dr * dc + cc * dc ** 2)
    )
    return surface.ravel()


def extract_foot_iso_fit(
    grid_z: np.ndarray,
    peak_rc,
    iso_level: float = 0.5,
    model: str = "gaussian",
) -> np.ndarray:
    """Foot as an iso-level ring of a fitted 2-D elliptical-Gaussian model.

    Parametric reference method.  Rather than tracing the (noisy) data surface
    directly, fit a smooth analytic bell — a rotated 2-D elliptical Gaussian
    ``offset + A·exp(-(...))`` with free amplitude, centre ``(x0, y0)``, widths
    ``(σx, σy)``, rotation ``θ`` and ``offset`` — to the valid pixels via
    ``scipy.optimize.curve_fit``, then read the foot off the *model* at a chosen
    iso-level.  Because the model is reproducible and smooth, the resulting ring
    is a clean parametric reference: at ``iso_level=0.5`` it is the half-maximum
    (FWHM) contour of the fitted mountain.

    Algorithm
    ---------
    1. Collect the valid (non-NaN ``grid_z``) pixels as ``(row, col, z)`` samples.
       Seed the fit from the data: ``offset = nanmin``, ``amplitude = nanmax -
       nanmin``, ``(x0, y0) = peak_rc``, ``σx = σy =`` a rough width (a fraction
       of the valid-region extent), ``θ = 0``.
    2. Fit the elliptical Gaussian with ``curve_fit`` over the valid samples.  A
       failure to converge raises (caught and re-raised as ``ValueError``).
    3. Evaluate the *fitted* surface on the full grid and contour it at
       ``offset + iso_level·amplitude`` with ``find_contours``; select the ring
       whose polygon encloses the peak (largest, if several qualify) — the same
       point-in-polygon rule used by ``extract_foot_threshold``.
    4. Compute the fit coefficient of determination ``R²`` over the valid samples
       for diagnostics (returned only via the figure text panel, not the ring).

    ``model`` selects the analytic bell.  Only ``"gaussian"`` is implemented; a
    ``"super_gaussian"`` branch is reserved and raises ``NotImplementedError``
    (fail-fast — it must NOT silently fall back to the Gaussian).

    Fail-fast contract
    ------------------
    Raises ``ValueError`` if ``model`` is unknown, if ``grid_z`` is all-NaN or has
    too few valid samples to fit seven parameters, if the peak value is
    non-positive/non-finite, if ``curve_fit`` fails to converge, if
    ``find_contours`` returns nothing at the iso-level, or if no contour encloses
    the peak.  Raises ``NotImplementedError`` for the reserved ``"super_gaussian"``
    model.  Never returns a default ring or sentinel.

    Returns (N, 2) (row, col) contour points.
    """
    if model == "super_gaussian":
        raise NotImplementedError(
            "extract_foot_iso_fit: model='super_gaussian' is reserved but not "
            "implemented — pass model='gaussian' (no silent fallback)"
        )
    if model != "gaussian":
        raise ValueError(
            f"extract_foot_iso_fit: unknown model '{model}' — expected 'gaussian' "
            "or 'super_gaussian'"
        )
    if not (0.0 < iso_level < 1.0):
        raise ValueError(f"iso_level must be in (0, 1), got {iso_level}")

    valid = ~np.isnan(grid_z)
    if not np.any(valid):
        raise ValueError(
            "extract_foot_iso_fit: grid_z is all-NaN — no surface to fit"
        )

    n_rows, n_cols = grid_z.shape
    peak_r, peak_c = int(round(peak_rc[0])), int(round(peak_rc[1]))

    # --- 1. Valid samples + data-seeded initial parameters --------------------
    vr, vc = np.nonzero(valid)
    zv = grid_z[vr, vc].astype(np.float64)
    if vr.size < 7:
        raise ValueError(
            f"extract_foot_iso_fit: only {vr.size} valid pixels — fewer than the 7 "
            "free parameters of the elliptical Gaussian; cannot fit"
        )

    z_min = float(np.nanmin(grid_z))
    z_max = float(np.nanmax(grid_z))
    amplitude0 = z_max - z_min
    if not np.isfinite(amplitude0) or amplitude0 <= 0.0:
        raise ValueError(
            f"extract_foot_iso_fit: peak amplitude non-positive/non-finite "
            f"({amplitude0}) — field too flat/degenerate to fit a bell"
        )

    # Rough width seed: a fraction of the valid-region row/col extent.
    width0_r = max(float(vr.max() - vr.min()) / 6.0, 1.0)
    width0_c = max(float(vc.max() - vc.min()) / 6.0, 1.0)
    p0 = (
        amplitude0,
        float(peak_r),
        float(peak_c),
        width0_r,
        width0_c,
        0.0,
        z_min,
    )

    # --- 2. Fit the elliptical Gaussian over the valid samples ----------------
    try:
        popt, _pcov = curve_fit(
            _elliptical_gaussian_2d,
            (vr.astype(np.float64), vc.astype(np.float64)),
            zv,
            p0=p0,
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"extract_foot_iso_fit: curve_fit failed to converge on the elliptical "
            f"Gaussian model — {exc}"
        ) from exc

    fit_amplitude = float(popt[0])
    fit_offset = float(popt[6])
    if not np.isfinite(fit_amplitude) or fit_amplitude <= 0.0:
        raise ValueError(
            f"extract_foot_iso_fit: fitted amplitude non-positive/non-finite "
            f"({fit_amplitude}) — degenerate fit, no iso-level to contour"
        )

    # --- Fit R² over the valid samples (diagnostic) ---------------------------
    z_pred = _elliptical_gaussian_2d(
        (vr.astype(np.float64), vc.astype(np.float64)), *popt
    )
    ss_res = float(np.sum((zv - z_pred) ** 2))
    ss_tot = float(np.sum((zv - zv.mean()) ** 2))
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    # --- 3. Iso-level ring of the FITTED surface, enclosing the peak ----------
    rows_idx = np.arange(n_rows)[:, None]
    cols_idx = np.arange(n_cols)[None, :]
    fitted_surface = _elliptical_gaussian_2d(
        (np.broadcast_to(rows_idx, (n_rows, n_cols)).astype(np.float64),
         np.broadcast_to(cols_idx, (n_rows, n_cols)).astype(np.float64)),
        *popt,
    ).reshape(n_rows, n_cols)

    level = fit_offset + iso_level * fit_amplitude
    contours = find_contours(fitted_surface, level=level)
    if not contours:
        raise ValueError(
            f"extract_foot_iso_fit: find_contours found no contour at the iso-level "
            f"{level:.4f} ({100 * iso_level:.0f}% of fitted amplitude)"
        )

    from matplotlib.path import Path as MplPath

    enclosing = []
    for c in contours:
        if len(c) < 3:
            continue
        poly = MplPath(np.column_stack([c[:, 0], c[:, 1]]))
        if poly.contains_point((peak_r, peak_c)):
            area = 0.5 * abs(
                np.dot(c[:, 0], np.roll(c[:, 1], 1))
                - np.dot(c[:, 1], np.roll(c[:, 0], 1))
            )
            enclosing.append((area, c))

    if not enclosing:
        raise ValueError(
            f"extract_foot_iso_fit: no fitted iso-contour at {100 * iso_level:.0f}% "
            f"of amplitude encloses the peak at (row,col)={peak_rc} "
            f"(fit R²={fit_r2:.3f})"
        )

    enclosing.sort(key=lambda t: t[0])
    return enclosing[-1][1]


def contour_height(grid_z: np.ndarray, contour_rc: np.ndarray) -> np.ndarray:
    """Bilinearly sample grid_z at contour (row, col) points (NaN→nanmean fill)."""
    filled = np.where(np.isnan(grid_z), np.nanmean(grid_z), grid_z)
    return map_coordinates(
        filled, np.array([contour_rc[:, 0], contour_rc[:, 1]]), order=1, mode="nearest"
    )


# =============================================================================
# Figures
# =============================================================================


def fig1_pipeline_steps(g, smoothed, laplacian, grad_mag, peak_rc,
                        grad_contour_rc, infl_contour_rc, out_dir):
    """Five-panel 2D walk-through of the gradient-ridge pipeline."""
    grid_z = g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_raw, ax_sm, ax_lap), (ax_grad, ax_overlay, ax_text) = axes

    ax_raw.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    ax_raw.plot(peak_rc[1], peak_rc[0], "rx", ms=11, mew=2)
    ax_raw.set_title("1. Raw heatmap grid_z (peak ✕)")

    ax_sm.imshow(np.ma.masked_invalid(smoothed), cmap=jet, origin="upper")
    ax_sm.set_title("2. NaN-aware Gaussian smoothed")

    vabs = max(float(np.nanmax(np.abs(laplacian))), 1e-12)
    ax_lap.imshow(laplacian, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="upper")
    ax_lap.contour(np.where(np.isnan(laplacian), 0.0, laplacian),
                   levels=[0.0], colors="black", linewidths=0.6)
    ax_lap.set_title("3. Laplacian (zero-crossing = inflection)")

    vmax_g = max(float(np.nanmax(grad_mag)), 1e-12)
    ax_grad.imshow(np.ma.masked_invalid(grad_mag), cmap="inferno",
                   vmin=0, vmax=vmax_g, origin="upper")
    if grad_contour_rc is not None:
        ax_grad.plot(grad_contour_rc[:, 1], grad_contour_rc[:, 0], "-", color="lime", lw=1.6)
    ax_grad.plot(peak_rc[1], peak_rc[0], "cx", ms=11, mew=2)
    ax_grad.set_title("4. |∇z| gradient magnitude + ridge\n(THIS is what the method maximises)")

    ax_overlay.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")
    # Expectation reference: iso-contours of the response itself.
    zf = np.where(np.isnan(grid_z), np.nanmin(grid_z), grid_z)
    ax_overlay.contour(zf, levels=6, colors="white", linewidths=0.5, alpha=0.6)
    if grad_contour_rc is not None:
        ax_overlay.plot(grad_contour_rc[:, 1], grad_contour_rc[:, 0], "-",
                        color="lime", lw=2.0, label="gradient ridge (new)")
    if infl_contour_rc is not None:
        ax_overlay.plot(infl_contour_rc[:, 1], infl_contour_rc[:, 0], "-",
                        color="red", lw=2.0, label="Laplacian inflection (old)")
    ax_overlay.plot(peak_rc[1], peak_rc[0], "kx", ms=11, mew=2)
    ax_overlay.legend(loc="upper right", fontsize=8)
    ax_overlay.set_title("5. Both boundaries vs response iso-contours")

    # Text panel: quantitative comparison.
    ax_text.axis("off")
    lines = ["Method sensitivity summary", "-" * 30]
    peak_val = float(np.nanmax(grid_z))
    lines.append(f"peak grid_z value : {peak_val:.3f}")
    for label, c_rc in (("gradient ridge", grad_contour_rc),
                        ("inflection", infl_contour_rc)):
        if c_rc is None:
            lines.append(f"{label:>14}: <none>")
            continue
        h = contour_height(grid_z, c_rc)
        r = np.hypot(c_rc[:, 0] - peak_rc[0], c_rc[:, 1] - peak_rc[1])
        lines.append(
            f"{label:>14}: mean r={r.mean():5.1f}px  "
            f"height={h.mean():.3f} ({100*h.mean()/peak_val:4.0f}% of peak)"
        )
    lines += ["", "If the ridge sits at a high % of peak,",
              "it hugs the hotspot, not the RF edge."]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                 family="monospace", fontsize=10, transform=ax_text.transAxes)

    fig.suptitle(f"{g['session_id']} — gradient-ridge pipeline steps", fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig1_pipeline_steps.png")


def fig2_radial_profiling(g, grad_mag, peak_rc, grad_contour_rc, n_angles,
                          out_dir, n_show=6):
    """Show the rays and, for a few angles, |grad z| profile vs response profile."""
    grid_z = g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")
    peak_val = float(np.nanmax(grid_z))

    fig = plt.figure(figsize=(16, 8))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_prof = fig.add_subplot(1, 2, 2)

    ax_map.imshow(np.ma.masked_invalid(grad_mag), cmap="inferno", origin="upper",
                  vmin=0, vmax=max(float(np.nanmax(grad_mag)), 1e-12))

    # Overlay the grid_z NaN boundary as a white dashed line so the researcher
    # can confirm the lime-green gradient-ridge contour sits *inside* the data
    # boundary rather than tracing it.
    nan_boundary_segments = find_contours(
        (~np.isnan(grid_z)).astype(float), level=0.5
    )
    for i, seg in enumerate(nan_boundary_segments):
        label = "data boundary" if i == 0 else "_nolegend_"
        ax_map.plot(seg[:, 1], seg[:, 0], "--", color="white",
                    lw=0.8, alpha=0.6, label=label)

    if grad_contour_rc is not None:
        closed = np.vstack([grad_contour_rc, grad_contour_rc[:1]])
        ax_map.plot(closed[:, 1], closed[:, 0], "-", color="lime", lw=1.8,
                    label="gradient ridge")
    ax_map.plot(peak_rc[1], peak_rc[0], "cx", ms=12, mew=2, label="peak")
    ax_map.legend(fontsize=7, loc="upper right")

    show_angles = np.linspace(0, 2 * np.pi, n_show, endpoint=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    for angle, col in zip(show_angles, colors):
        radii, gvals = sample_ray(grad_mag, peak_rc, angle)
        if radii.size == 0:
            continue
        # Draw the ray on the map.
        rr = peak_rc[0] + radii * math.cos(angle)
        cc = peak_rc[1] + radii * math.sin(angle)
        ax_map.plot(cc, rr, "-", color=col, lw=0.9, alpha=0.8)

        # Profiles: gradient magnitude (solid) vs normalised response (dashed).
        _, zvals = sample_ray(grid_z, peak_rc, angle)
        gnorm = gvals / max(gvals.max(), 1e-12)
        znorm = zvals / max(peak_val, 1e-12)
        deg = int(round(math.degrees(angle)))
        ax_prof.plot(radii, gnorm, "-", color=col, lw=1.3, label=f"|∇z| {deg}°")
        ax_prof.plot(radii[: len(znorm)], znorm, "--", color=col, lw=1.0, alpha=0.7)
        # Mark the chosen ridge radius (argmax of |grad z|).
        idx = int(np.argmax(gvals))
        ax_prof.plot(radii[idx], gnorm[idx], "o", color=col, ms=6)

    ax_map.set_title(f"Rays from peak ({n_angles} used in method, {n_show} shown)")
    ax_prof.set_xlabel("radius from peak (px)")
    ax_prof.set_ylabel("normalised value")
    ax_prof.set_title("Solid = |∇z| (method target, ● = chosen radius)\n"
                      "Dashed = response grid_z (where signal actually decays)")
    ax_prof.legend(fontsize=7, ncol=2)
    ax_prof.grid(alpha=0.3)

    fig.suptitle(f"{g['session_id']} — radial-profiling: what the method is sensitive to",
                 fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig2_radial_profiling.png")


def fig3_sigma_sweep(g, sigmas, n_angles, savgol_window, out_dir):
    """Gradient-ridge contour for several smoothing sigmas + area-vs-sigma."""
    grid_u, grid_v, grid_z = g["grid_u"], g["grid_v"], g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")

    fig, (ax_map, ax_area) = plt.subplots(1, 2, figsize=(15, 6.5))
    ax_map.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")

    colors = plt.cm.viridis(np.linspace(0, 1, len(sigmas)))
    areas = []
    for sigma, col in zip(sigmas, colors):
        smoothed, _ = compute_laplacian_arrays(grid_z, sigma)
        b = compute_gradient_ridge(grid_u, grid_v, grid_z, smoothed,
                                   n_angles=n_angles, savgol_window=savgol_window)
        if b is None:
            areas.append(np.nan)
            continue
        areas.append(b.area_uv)
        # Convert UV contour back to pixel space for the imshow overlay.
        n_rows, n_cols = grid_z.shape
        u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
        v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
        rr = (b.contour_uv[:, 0] - u_min) / (u_max - u_min) * (n_rows - 1)
        cc = (b.contour_uv[:, 1] - v_min) / (v_max - v_min) * (n_cols - 1)
        rr = np.append(rr, rr[0]); cc = np.append(cc, cc[0])
        ax_map.plot(cc, rr, "-", color=col, lw=1.6, label=f"σ={sigma:g}")

    ax_map.legend(fontsize=8, loc="upper right")
    ax_map.set_title("Gradient-ridge contour vs Gaussian σ")

    ax_area.plot(sigmas, areas, "o-", color="green")
    ax_area.set_xlabel("Gaussian σ")
    ax_area.set_ylabel("boundary area (UV²)")
    ax_area.set_title("Boundary area vs σ (parameter sensitivity)")
    ax_area.grid(alpha=0.3)

    fig.suptitle(f"{g['session_id']} — σ sensitivity sweep", fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig3_sigma_sweep.png")


def fig4_surface_3d(g, grad_contour_rc, infl_contour_rc, out_dir):
    """3D response surface with both contours drawn at their true height."""
    grid_u, grid_v, grid_z = g["grid_u"], g["grid_v"], g["grid_z"]
    Z = np.ma.masked_invalid(grid_z)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(grid_v, grid_u, np.where(np.isnan(grid_z), np.nan, grid_z),
                    cmap="jet", linewidth=0, antialiased=True, alpha=0.85)

    for c_rc, col, lbl in ((grad_contour_rc, "lime", "gradient ridge"),
                           (infl_contour_rc, "red", "inflection")):
        if c_rc is None:
            continue
        uv = contour_pixels_to_uv(c_rc, grid_u, grid_v)
        h = contour_height(grid_z, c_rc)
        closed_u = np.append(uv[:, 0], uv[0, 0])
        closed_v = np.append(uv[:, 1], uv[0, 1])
        closed_h = np.append(h, h[0])
        ax.plot(closed_v, closed_u, closed_h, "-", color=col, lw=3, label=lbl)

    ax.set_xlabel("V"); ax.set_ylabel("U"); ax.set_zlabel("response (IFF)")
    ax.set_title(f"{g['session_id']} — 3D response surface + boundaries\n"
                 "(does the green ridge cling to the flank instead of the base?)")
    ax.legend()
    fig.tight_layout()
    _emit(fig, out_dir, "fig4_surface_3d.png")


def fig5_forearm_3d(g, grad_contour_uv, infl_contour_uv, out_dir):
    """3D forearm mesh coloured by heatmap with the contour back-projected."""
    forearm_uv = g["forearm_uv"]
    faces = g["forearm_faces"]
    V = g["forearm_V"]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Light wireframe-ish surface via trisurf of the forearm mesh.
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=faces,
                    color="lightgrey", alpha=0.35, linewidth=0, shade=True)

    for c_uv, col, lbl in ((grad_contour_uv, "lime", "gradient ridge"),
                           (infl_contour_uv, "red", "inflection")):
        if c_uv is None:
            continue
        xyz = uv_points_to_xyz(c_uv, forearm_uv, faces, V)
        xyz = np.vstack([xyz, xyz[:1]])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", color=col, lw=3, label=lbl)

    ax.set_title(f"{g['session_id']} — boundaries back-projected on the forearm")
    ax.legend()
    try:
        ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2])))
    except Exception:
        pass
    fig.tight_layout()
    _emit(fig, out_dir, "fig5_forearm_3d.png")


def fig6_foot_methods(g, smoothed, peak_rc, grad_contour_rc, n_angles,
                      savgol_window, out_dir, n_show=6,
                      profile_savgol_window=11, profile_savgol_polyorder=3,
                      threshold_frac=0.2,
                      watershed_min_peak_distance=15,
                      watershed_peak_threshold_rel=0.1,
                      prominence_min_prominence_frac=0.1,
                      curvature_field_sigma=5.0,
                      spill_point_n_levels=200,
                      spill_point_detrend=False,
                      log_blob_sigma_min=2.0,
                      log_blob_sigma_max=30.0,
                      log_blob_n_sigma=20,
                      iso_fit_iso_level=0.5,
                      iso_fit_model="gaussian"):
    """Compare the three "foot of the mountain" candidates vs the gradient ridge.

    Left panel: ``grid_z`` heatmap + the white-dashed data boundary (reused from
    fig2) + four overlaid pixel-space contours — gradient ridge (lime, the
    reference inflection ring), curvature foot, kneedle foot, threshold foot —
    plus the peak marker and a legend.

    Right panel: ``n_show`` sample-ray response profiles ``z(r)`` (from
    ``sample_ray`` on ``grid_z``) with each method's chosen foot radius marked on
    the profile, and the ``z''(r)`` curvature trace (the curvature method's
    target) drawn on a twin axis.

    Text annotation: per-method mean radius (px) and mean height as % of peak
    (via ``contour_height``).  Expected: foot heights ~20-25% of peak vs the
    gradient ridge ~60%.

    The original radial foot extractors are called directly; if a field is
    degenerate they raise ``ValueError`` (fail-fast) and that propagates out of
    this figure, exactly as the other sandbox figures let their errors propagate.
    The non-radial extractors (watershed, prominence, curvature field, and the
    single-peak delineation trio spill point / LoG blob / iso-fit) are each
    guarded by a try/except so a single method that legitimately fails does not
    abort the whole comparison figure: a warning is printed and that row is
    skipped (display robustness only — the extractors themselves remain fail-fast
    and never return a default ring).
    """
    grid_z = g["grid_z"]
    jet = plt.cm.jet.copy()
    jet.set_bad("lightgrey")
    peak_val = float(np.nanmax(grid_z))

    # --- Extract the radial foot candidates (fail-fast, no silent skip) -------
    foot_curv_rc = extract_foot_curvature(
        grid_z, smoothed, peak_rc, n_angles=n_angles, savgol_window=savgol_window,
        profile_savgol_window=profile_savgol_window,
        profile_savgol_polyorder=profile_savgol_polyorder,
    )
    foot_knee_rc = extract_foot_kneedle(
        grid_z, peak_rc, n_angles=n_angles, savgol_window=savgol_window,
    )
    foot_thr_rc = extract_foot_threshold(grid_z, peak_rc, frac=threshold_frac)

    # --- Extract the non-radial candidates, guarded so one failure does not ---
    #     abort the comparison figure (display robustness; the extractors stay
    #     fail-fast and never return a default ring — we just skip the row).
    def _try_extract(name, fn):
        try:
            return fn()
        except ValueError as exc:
            print(f"  WARNING: {name} skipped in fig6 — {exc}")
            return None

    foot_ws_rc = _try_extract(
        "watershed",
        lambda: extract_foot_watershed(
            grid_z, smoothed, peak_rc,
            min_peak_distance=watershed_min_peak_distance,
            peak_threshold_rel=watershed_peak_threshold_rel,
        ),
    )
    foot_prom_rc = _try_extract(
        "prominence",
        lambda: extract_foot_prominence(
            grid_z, smoothed, peak_rc,
            min_prominence_frac=prominence_min_prominence_frac,
        ),
    )
    foot_cfield_rc = _try_extract(
        "curvature field",
        lambda: extract_foot_curvature_field(
            grid_z, smoothed, peak_rc, sigma=curvature_field_sigma,
        ),
    )

    # --- Single-peak delineation candidates (methods 4.2/4.3/4.4), each ------
    #     non-radial and guarded the same way as the other non-radial methods.
    foot_spill_rc = _try_extract(
        "spill point",
        lambda: extract_foot_spill_point(
            grid_z, smoothed, peak_rc,
            n_levels=spill_point_n_levels, detrend=spill_point_detrend,
        ),
    )
    foot_log_rc = _try_extract(
        "LoG blob",
        lambda: extract_foot_log_blob(
            grid_z, peak_rc,
            sigma_min=log_blob_sigma_min, sigma_max=log_blob_sigma_max,
            n_sigma=log_blob_n_sigma,
        ),
    )
    foot_iso_rc = _try_extract(
        "iso-fit",
        lambda: extract_foot_iso_fit(
            grid_z, peak_rc,
            iso_level=iso_fit_iso_level, model=iso_fit_model,
        ),
    )

    # (label, contour_rc, color) — gradient ridge first as the reference ring.
    methods = [
        ("gradient ridge", grad_contour_rc, "lime"),
        ("curvature foot", foot_curv_rc, "deepskyblue"),
        ("kneedle foot", foot_knee_rc, "magenta"),
        (f"threshold foot ({100*threshold_frac:.0f}%)", foot_thr_rc, "orange"),
        ("watershed", foot_ws_rc, "yellow"),
        ("prominence", foot_prom_rc, "white"),
        ("curvature field", foot_cfield_rc, "springgreen"),
        ("spill point", foot_spill_rc, "yellow"),
        ("LoG blob", foot_log_rc, "white"),
        (f"iso-fit ({100*iso_fit_iso_level:.0f}%)", foot_iso_rc, "springgreen"),
    ]

    fig = plt.figure(figsize=(16, 8))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_prof = fig.add_subplot(1, 2, 2)

    # ----------------------------- Left panel --------------------------------
    ax_map.imshow(np.ma.masked_invalid(grid_z), cmap=jet, origin="upper")

    # White-dashed data boundary (same overlay pattern as fig2).
    nan_boundary_segments = find_contours(
        (~np.isnan(grid_z)).astype(float), level=0.5
    )
    for i, seg in enumerate(nan_boundary_segments):
        label = "data boundary" if i == 0 else "_nolegend_"
        ax_map.plot(seg[:, 1], seg[:, 0], "--", color="white",
                    lw=0.8, alpha=0.6, label=label)

    for label, c_rc, col in methods:
        if c_rc is None:
            continue
        closed = np.vstack([c_rc, c_rc[:1]])
        ax_map.plot(closed[:, 1], closed[:, 0], "-", color=col, lw=1.8, label=label)
    ax_map.plot(peak_rc[1], peak_rc[0], "kx", ms=12, mew=2, label="peak")
    ax_map.legend(fontsize=7, loc="upper right")
    ax_map.set_title("Foot-of-mountain candidates vs gradient ridge\n"
                     "(feet should sit outside the lime ring, inside the data boundary)")

    # ----------------------------- Right panel -------------------------------
    show_angles = np.linspace(0, 2 * np.pi, n_show, endpoint=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    ax_curv = ax_prof.twinx()

    # Per-method foot radius as a function of angle, so each ray's foot can be
    # marked on its own profile.  (Radii are recovered from the contour points.)
    def _radii_of(c_rc):
        if c_rc is None:
            return None
        return np.hypot(c_rc[:, 0] - peak_rc[0], c_rc[:, 1] - peak_rc[1])

    # The radial methods produce one point per ray angle (n_angles, matching the
    # method angle grid); map a shown angle to its nearest radial index.
    method_angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    for angle, col in zip(show_angles, colors):
        radii, zvals = sample_ray(grid_z, peak_rc, angle)
        if radii.size == 0:
            continue
        valid = ~np.isnan(zvals)
        if not np.any(valid):
            continue
        last_valid = int(np.max(np.nonzero(valid)[0]))
        radii = radii[: last_valid + 1]
        zvals = zvals[: last_valid + 1]
        znorm = zvals / max(peak_val, 1e-12)
        deg = int(round(math.degrees(angle)))
        ax_prof.plot(radii, znorm, "-", color=col, lw=1.2, label=f"z(r) {deg}°")

        # z''(r) curvature trace (the curvature method's target) on the twin axis.
        _, zsm = sample_ray(smoothed, peak_rc, angle)
        zsm = zsm[: last_valid + 1]
        if zsm.size >= profile_savgol_window:
            d2z = savgol_filter(
                zsm, profile_savgol_window, profile_savgol_polyorder, deriv=2
            )
            ax_curv.plot(radii[: len(d2z)], d2z, ":", color=col, lw=0.9, alpha=0.7)

        # Mark each radial method's foot radius on this ray's profile.
        ang_idx = int(np.argmin(np.abs(method_angles - angle)))
        for _label, c_rc, mcol in methods:
            r_of = _radii_of(c_rc)
            if r_of is None:
                continue
            # Threshold foot is not radial (its point count != n_angles); skip
            # the per-ray marker for it (it has no per-ray radius along this ray).
            if len(r_of) != n_angles:
                continue
            r_foot = float(r_of[ang_idx])
            # Sample the profile height at that radius for the marker y-value.
            if radii.size and radii[0] <= r_foot <= radii[-1]:
                z_at = float(np.interp(r_foot, radii, znorm))
                ax_prof.plot(r_foot, z_at, "o", color=mcol, ms=6,
                             mec="black", mew=0.5)

    ax_prof.axhline(0.0, color="grey", lw=0.6, alpha=0.5)
    ax_prof.set_xlabel("radius from peak (px)")
    ax_prof.set_ylabel("normalised response z(r)")
    ax_curv.set_ylabel("z''(r) curvature (dotted)")
    ax_prof.set_title("Solid = response z(r); dotted = z''(r) curvature trace\n"
                      "● = each radial method's foot on that ray")
    ax_prof.legend(fontsize=7, ncol=2, loc="upper right")
    ax_prof.grid(alpha=0.3)

    # --------------------------- Text annotation -----------------------------
    lines = [f"peak grid_z value : {peak_val:.3f}", ""]
    for label, c_rc, _col in methods:
        if c_rc is None:
            lines.append(f"{label:>22}: <none>")
            continue
        h = contour_height(grid_z, c_rc)
        r = np.hypot(c_rc[:, 0] - peak_rc[0], c_rc[:, 1] - peak_rc[1])
        lines.append(
            f"{label:>22}: mean r={r.mean():5.1f}px  "
            f"height={100 * h.mean() / max(peak_val, 1e-12):4.0f}% of peak"
        )
    ax_map.text(0.02, 0.02, "\n".join(lines), va="bottom", ha="left",
                family="monospace", fontsize=8, color="white",
                transform=ax_map.transAxes,
                bbox=dict(boxstyle="round", fc="black", alpha=0.55))

    fig.suptitle(f"{g['session_id']} — foot-of-mountain boundary candidates",
                 fontsize=13)
    fig.tight_layout()
    _emit(fig, out_dir, "fig6_foot_methods.png")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # ----------------------- HARDCODED INPUT (edit me) -----------------------
    ST13_03 = "2022-06-14_ST13-03/2022-06-14_ST13-03_population_response_fields.npz"
    ST14_01 = "2022-06-15_ST14-01/2022-06-15_ST14-01_population_response_fields.npz"
    ST14_02 = "2022-06-15_ST14-02/2022-06-15_ST14-02_population_response_fields.npz"
    ST14_04 = "2022-06-15_ST14-04/2022-06-15_ST14-04_population_response_fields.npz"

    NPZ_PATH = Path(
        "F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/"
        "semi-controlled/4_analysed/spatial_extract_boundaries/iff_mean/" +
        ST13_03
    )
    
    
    GESTURE = "all"        # one of: all, stroke, tap, stroke_proximal, stroke_distal

    # ----------------------- TUNABLE METHOD PARAMETERS -----------------------
    MIN_OVERLAP_PCT: float = 5.0       # % of gesture touches a vertex must be contacted by
    INFLECTION_SIGMA = 5.0              # Gaussian smoothing sigma (matches pipeline inflection_sigma)
    MEDIAN_FILTER_SIZE: int | None = 5  # median filter on grid_z (None = no extra filtering)
    N_ANGLES = 360                      # radial rays
    SAVGOL_WINDOW = 31     # contour smoothing window (None to disable)
    SIGMA_SWEEP = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    # ---- Figure windows (enable/disable each window independently) -------
    # Each entry's "show" flag gates whether that figure window is generated
    # (and, in interactive mode, opened).  Set "show": False to skip it.
    FIGURES = {
        "fig1_pipeline_steps":   {"show": True},   # smoothing / Laplacian / gradient steps
        "fig2_radial_profiling": {"show": True},   # gradient ridge radial profiling
        "fig3_sigma_sweep":      {"show": True},   # ridge vs Gaussian sigma sweep
        "fig4_surface_3d":       {"show": True},   # grid_z surface with contours
        "fig5_forearm_3d":       {"show": False},  # back-projection onto forearm (slow)
        "fig6_foot_methods":     {"show": True},   # foot-of-mountain candidates vs ridge
    }

    # ---- Foot-of-mountain method parameters (Fig 6) ----------------------
    # Nested by method so each knob's owner is explicit.  The two radial foot
    # methods (curvature, kneedle) also use the shared N_ANGLES and
    # SAVGOL_WINDOW defined above; only method-specific knobs live here.
    FOOT_PARAMS = {
        "curvature": {                       # foot = max upward z'' beyond the inflection
            "profile_savgol_window": 11,     # on-ray z'/z'' smoothing window (odd)
            "profile_savgol_polyorder": 3,   # on-ray Savitzky-Golay poly order
        },
        "kneedle": {                         # foot = chord-deviation knee
            # no method-specific knob (uses shared N_ANGLES / SAVGOL_WINDOW)
        },
        "threshold": {                       # foot = iso-level reference ring
            "frac": 0.2,                     # iso-level as fraction of peak
        },
        "watershed": {                       # foot = catchment cut at saddles (non-radial)
            "min_peak_distance": 15,         # min separation between competing maxima (px)
            "peak_threshold_rel": 0.1,       # maxima/background floor as fraction of peak
        },
        "prominence": {                      # foot = saddle/col iso-contour (non-radial)
            "min_prominence_frac": 0.1,      # min prominence (frac of peak) to accept a merge
        },
        "curvature_field": {                 # foot = outer convex-up break (non-radial)
            "sigma": 5.0,                    # Gaussian sigma for the Laplacian field
        },
        "spill_point": {                     # foot = superlevel-set flood knee (non-radial)
            "n_levels": 200,                 # threshold sweep resolution
            "detrend": False,                # remove a planar background before sweeping
        },
        "log_blob": {                        # foot = scale-space LoG ring (non-radial)
            "sigma_min": 2.0,                # smallest blob scale to probe (px)
            "sigma_max": 30.0,               # largest blob scale to probe (px)
            "n_sigma": 20,                   # log-spaced scales between the bounds
        },
        "iso_fit": {                         # foot = fitted-Gaussian iso-level ring (non-radial)
            "iso_level": 0.5,                # iso-level as fraction of fitted amplitude (0.5=FWHM)
            "model": "gaussian",             # analytic bell; 'super_gaussian' reserved (raises)
        },
        "display": {                         # Fig 6 right-panel rendering
            "n_show": 6,                     # sample rays drawn in the right panel
        },
    }

    out_dir = Path(__file__).resolve().parent / "_sandbox_gradient_ridge_out"
    out_dir.mkdir(exist_ok=True)

    # Secure an interactive backend now (after the analysis imports forced Agg).
    global _INTERACTIVE
    if USE_PLT_SHOW:
        _INTERACTIVE = _ensure_interactive_backend()

    print(f"Loading {NPZ_PATH.name}  (gesture='{GESTURE}')")
    g = load_grid(NPZ_PATH, GESTURE)
    print(f"  session={g['session_id']}  pipeline sigma={g['pipeline_sigma']}  "
          f"pipeline method={g['pipeline_boundary_method']}  "
          f"pipeline min_overlap_pct={g['pipeline_min_overlap_pct']}")
    print(f"  available gestures: {g['gesture_types']}")

    threshold = compute_threshold_from_ratio(MIN_OVERLAP_PCT, g["n_touches"])
    slim_heatmap = apply_vertex_threshold(g["heatmap_pre_threshold"], g["unique_count"], threshold)
    grid_u, grid_v, grid_z = compute_interpolated_grid(
        g["forearm_uv"], g["forearm_faces"], g["forearm_V"],
        slim_heatmap, median_filter_size=MEDIAN_FILTER_SIZE,
    )
    print(f"  threshold={threshold} ({MIN_OVERLAP_PCT}% of {g['n_touches']} touches)")
    g["grid_u"] = grid_u
    g["grid_v"] = grid_v
    g["grid_z"] = grid_z

    # --- Re-run the methods LIVE with the chosen parameters ------------------
    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        raise RuntimeError("grid_z is all-NaN — nothing to analyse.")
    print(f"  peak at (row,col)={peak_rc}, peak value={np.nanmax(grid_z):.3f}")

    smoothed, laplacian = compute_laplacian_arrays(grid_z, INFLECTION_SIGMA)
    grad_mag = compute_gradient_magnitude(smoothed, np.isnan(grid_z))

    grad_contour_rc = _extract_ridge_via_radial_profiling(
        grad_mag, peak_rc, n_angles=N_ANGLES, savgol_window=SAVGOL_WINDOW,
    )
    grad_boundary = compute_gradient_ridge(
        grid_u, grid_v, grid_z, smoothed,
        n_angles=N_ANGLES, savgol_window=SAVGOL_WINDOW,
    )
    infl_boundary = compute_inflection_boundary(grid_u, grid_v, grid_z, INFLECTION_SIGMA)

    # Inflection contour in pixel space (from its UV contour) for overlays.
    infl_contour_rc = None
    if infl_boundary is not None:
        uv = infl_boundary.contour_uv
        n_rows, n_cols = grid_z.shape
        u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
        v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
        rr = (uv[:, 0] - u_min) / (u_max - u_min) * (n_rows - 1)
        cc = (uv[:, 1] - v_min) / (v_max - v_min) * (n_cols - 1)
        infl_contour_rc = np.column_stack([rr, cc])

    print("Rendering figures...")
    if FIGURES["fig1_pipeline_steps"]["show"]:
        fig1_pipeline_steps(g, smoothed, laplacian, grad_mag, peak_rc,
                            grad_contour_rc, infl_contour_rc, out_dir)
    if FIGURES["fig2_radial_profiling"]["show"]:
        fig2_radial_profiling(g, grad_mag, peak_rc, grad_contour_rc, N_ANGLES, out_dir)
    if FIGURES["fig3_sigma_sweep"]["show"]:
        fig3_sigma_sweep(g, SIGMA_SWEEP, N_ANGLES, SAVGOL_WINDOW, out_dir)
    if FIGURES["fig4_surface_3d"]["show"]:
        fig4_surface_3d(g, grad_contour_rc, infl_contour_rc, out_dir)
    if FIGURES["fig5_forearm_3d"]["show"]:
        fig5_forearm_3d(
            g,
            grad_boundary.contour_uv if grad_boundary is not None else None,
            infl_boundary.contour_uv if infl_boundary is not None else None,
            out_dir,
        )
    if FIGURES["fig6_foot_methods"]["show"]:
        fig6_foot_methods(
            g, smoothed, peak_rc, grad_contour_rc, N_ANGLES,
            SAVGOL_WINDOW, out_dir,
            n_show=FOOT_PARAMS["display"]["n_show"],
            profile_savgol_window=FOOT_PARAMS["curvature"]["profile_savgol_window"],
            profile_savgol_polyorder=FOOT_PARAMS["curvature"]["profile_savgol_polyorder"],
            threshold_frac=FOOT_PARAMS["threshold"]["frac"],
            watershed_min_peak_distance=FOOT_PARAMS["watershed"]["min_peak_distance"],
            watershed_peak_threshold_rel=FOOT_PARAMS["watershed"]["peak_threshold_rel"],
            prominence_min_prominence_frac=FOOT_PARAMS["prominence"]["min_prominence_frac"],
            curvature_field_sigma=FOOT_PARAMS["curvature_field"]["sigma"],
            spill_point_n_levels=FOOT_PARAMS["spill_point"]["n_levels"],
            spill_point_detrend=FOOT_PARAMS["spill_point"]["detrend"],
            log_blob_sigma_min=FOOT_PARAMS["log_blob"]["sigma_min"],
            log_blob_sigma_max=FOOT_PARAMS["log_blob"]["sigma_max"],
            log_blob_n_sigma=FOOT_PARAMS["log_blob"]["n_sigma"],
            iso_fit_iso_level=FOOT_PARAMS["iso_fit"]["iso_level"],
            iso_fit_model=FOOT_PARAMS["iso_fit"]["model"],
        )

    print(f"Done. Figures in: {out_dir}")
    if _INTERACTIVE:
        _wire_navigation()
        print("Interactive: Left/Right (or n/p) to navigate between figures, q to close all.")
        plt.show()


if __name__ == "__main__":
    main()
