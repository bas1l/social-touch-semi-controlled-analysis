"""Figures for ``docs/spatial_extract_boundaries`` — the radial-foot workflow.

The pipeline is configured to run the **radial "foot of mountain"** boundary
(``boundary_method: radial`` in the checked-in DAG), so these figures document
that single method as an ordered walk through its subprocesses, using the actual
DAG parameter values.

Loads the population heatmap grid of one worked example (session
``2022-06-17_ST16-02``, gesture ``all``) and reproduces
``compute_radial_foot_boundary`` on that ``grid_z``:

* ``bd_01_workflow.png`` -- a flow diagram of the subprocess sequence with the
                            actual sigma values annotated at each step.
* ``bd_02_fields.png``   -- the field progression: grid_z -> Gaussian-smoothed
                            (sigma=8) -> Hessian lambda-max (sigma=5) with contour
                            -> final contour on the heatmap.
* ``bd_03_result.png``   -- the final radial RF boundary on the heatmap with the
                            derived metrics annotated.
* ``bd_04_ray_section.png`` -- one of the 360 cast rays: the Hessian lambda-max
                            heatmap with the ray drawn, beside the lambda-max-vs-radius
                            profile whose first positive plateau (the concave-up foot)
                            is where the traced contour sits.

Uses the checked-in DAG parameters for ``spatial_extract_boundaries``
(radial_gauss_sigma=8.0, radial_hess_sigma=5.0, radial_envelope_smooth_sigma=1.5).

Fail-fast per repo convention: a missing NPZ raises rather than being skipped.

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_boundary_workflow_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from _vendor.path_tools import get_project_data_root
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_laplacian_arrays,
    find_peak_location,
)
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    _compute_hessian_lmax,
    compute_radial_foot_boundary,
)

SESSION = "2022-06-17_ST16-02"
GTYPE = "all"

# checked-in DAG params for spatial_extract_boundaries (radial method)
RADIAL_GAUSS_SIGMA = 8.0
RADIAL_HESS_SIGMA = 5.0
RADIAL_ENV_SMOOTH = 1.5

NAVY = "#1F2A44"
ACCENT = "#E06A2B"
RADIAL_C = "#D62728"  # red — the active contour colour (contour_color: red)
INK = "#222222"

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG = REPO_ROOT / "docs" / "spatial_extract_boundaries" / "figures"


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"generate_boundary_workflow_figures: missing: {p}")
    return p


def _load_grid():
    data_root = Path(get_project_data_root())
    npz_path = _require(
        data_root / "4_analysed" / "spatial_extract_boundaries" / "iff_mean"
        / SESSION / f"{SESSION}_population_response_fields.npz"
    )
    d = np.load(npz_path, allow_pickle=True)
    return d[f"grid_u_{GTYPE}"], d[f"grid_v_{GTYPE}"], d[f"grid_z_{GTYPE}"]


def _compute_radial(grid_u, grid_v, grid_z):
    """Reproduce the radial-foot subprocesses; return the fields + the boundary."""
    smoothed = compute_laplacian_arrays(grid_z, RADIAL_GAUSS_SIGMA)[0]
    lmax = _compute_hessian_lmax(grid_z, RADIAL_GAUSS_SIGMA, RADIAL_HESS_SIGMA)
    boundary = compute_radial_foot_boundary(
        grid_u, grid_v, grid_z,
        gauss_sigma=RADIAL_GAUSS_SIGMA, hess_sigma=RADIAL_HESS_SIGMA,
        envelope_smooth_sigma=RADIAL_ENV_SMOOTH,
    )
    if boundary is None:
        raise ValueError(
            "generate_boundary_workflow_figures: compute_radial_foot_boundary "
            "returned None — cannot render the workflow figures."
        )
    return {"smoothed": smoothed, "lmax": lmax, "boundary": boundary}


def _uv_extent(grid_u, grid_v):
    return [float(grid_u.min()), float(grid_u.max()),
            float(grid_v.min()), float(grid_v.max())]


def _imshow_uv(ax, grid_u, grid_v, field, cmap, vmin=None, vmax=None):
    """imshow a (R,C) field in UV coordinates (origin lower, U->x, V->y)."""
    ext = _uv_extent(grid_u, grid_v)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e8e8e8")
    return ax.imshow(np.ma.masked_invalid(field.T), origin="lower", extent=ext,
                     cmap=cm, aspect="equal", vmin=vmin, vmax=vmax)


def _closed(c):
    return np.vstack([c, c[:1]])


def _plot_contour(ax, c, color, label, lw=2.4):
    cc = _closed(c)
    ax.plot(cc[:, 0], cc[:, 1], "-", color=color, linewidth=lw, label=label,
            path_effects=[pe.Stroke(linewidth=lw + 1.3, foreground="white"), pe.Normal()])


def _zoom_to_contour(ax, contour, margin=18.0):
    ax.set_xlim(contour[:, 0].min() - margin, contour[:, 0].max() + margin)
    ax.set_ylim(contour[:, 1].min() - margin, contour[:, 1].max() + margin)


def _peak_uv(grid_u, grid_v, grid_z):
    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        raise ValueError("generate_boundary_workflow_figures: all-NaN grid (no peak)")
    return grid_u[peak_rc], grid_v[peak_rc]


# ---------------------------------------------------------------------------
# bd_01 — the subprocess workflow, as a flow diagram
# ---------------------------------------------------------------------------

def fig_workflow():
    """A flow diagram of the radial-foot subprocess sequence with sigma values."""
    fig, ax = plt.subplots(figsize=(13.0, 5.3))
    ax.set_xlim(0, 13); ax.set_ylim(1.2, 6.6); ax.axis("off")

    ax.text(0.3, 6.32, "spatial_extract_boundaries  ·  boundary_method: radial",
            fontsize=14, weight="bold", color=NAVY)
    ax.text(0.3, 6.0, "The subprocesses, in execution order — session 2022-06-17_ST16-02",
            fontsize=10.5, color="#666", style="italic")

    GB = "#3B5C86"   # build lambda-max field
    GT = "#2E7D5B"   # trace the foot
    GC = "#8C5A2B"   # clip & measure
    W, H = 2.95, 1.05
    XS = [0.3, 3.4, 6.5, 9.6]      # four columns
    Y1, Y2 = 4.45, 2.6            # two rows

    # Each step: (x, y, title, subtitle, param-or-None, group-color)
    steps = [
        (XS[0], Y1, "1 · grid_z", "150x150 mean-IFF\nheatmap (input)", None, NAVY),
        (XS[1], Y1, "2 · Gaussian smooth", "NaN-aware — the\nfirst step", "σ = 8.0", GB),
        (XS[2], Y1, "3 · Extrapolate", "nearest-neighbour\nfill of NaNs", None, GB),
        (XS[3], Y1, "4 · Hessian → λmax", "principal-curvature\nfield", "σ = 5.0", GB),

        (XS[3], Y2, "5 · Cast 360 rays", "first λmax > 0 plateau\n= foot radius", "n_angles = 360", GT),
        (XS[2], Y2, "6 · Snap + smooth", "snap to footprint;\ncircular Savitzky-Golay", "window 31 · poly 3", GT),

        (XS[1], Y2, "7 · Clip to footprint", "envelope the curve\nto grid_z > 0", "σ = 1.5", GC),
        (XS[0], Y2, "8 · Metrics + save", "area, perimeter,\ncircularity, ellipse", "→ NPZ boundary_", GC),
    ]

    boxes = {}
    for x, y, title, sub, param, color in steps:
        box = FancyBboxPatch((x, y), W, H,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.6, edgecolor=color, facecolor="#f7f9fc")
        ax.add_patch(box)
        ax.add_patch(plt.Rectangle((x, y), 0.13, H, color=color))
        ax.text(x + 0.30, y + H - 0.26, title, fontsize=11, weight="bold", color=NAVY)
        ax.text(x + 0.30, y + H - 0.56, sub, fontsize=8.4, color=INK, va="top")
        if param:
            ax.text(x + 0.30, y + 0.13, param, fontsize=8.8, color=ACCENT,
                    family="monospace", weight="bold")
        boxes[title[:1]] = (x, y, W, H)

    def arrow(a, b):
        (ax_, ay, aw, ah) = boxes[a]
        (bx, by, bw, bh) = boxes[b]
        if abs(ay - by) < 0.2:  # same row
            if bx > ax_:
                p0 = (ax_ + aw, ay + ah / 2); p1 = (bx, by + bh / 2)
            else:
                p0 = (ax_, ay + ah / 2); p1 = (bx + bw, by + bh / 2)
        else:  # drop down between rows (aligned columns)
            p0 = (ax_ + aw / 2, ay); p1 = (bx + bw / 2, by + bh)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                     color="#5a6b86", linewidth=1.6))

    arrow("1", "2"); arrow("2", "3"); arrow("3", "4")
    arrow("4", "5")                      # down the right edge
    arrow("5", "6"); arrow("6", "7"); arrow("7", "8")

    # group captions — directly above row 1, directly below row 2
    ax.text(XS[1], Y1 + H + 0.12, "Build the λmax curvature field",
            fontsize=9.5, color=GB, weight="bold")
    ax.text(XS[2], Y2 - 0.34, "Trace the foot per ray",
            fontsize=9.5, color=GT, weight="bold")
    ax.text(XS[0], Y2 - 0.34, "Clip to footprint & measure",
            fontsize=9.5, color=GC, weight="bold")

    ax.text(0.3, 1.55,
            "The foot is the crest of the first positive-λmax plateau (a curvature landmark, a "
            "step past the λmax=0 inflection) —\nnot a fixed IFF threshold, so the edge adapts to "
            "each neuron's dome. Any degenerate step returns None (fail-fast), never a default ring.",
            fontsize=9, color="#555", style="italic")

    fig.tight_layout()
    out = FIG / "bd_01_workflow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[gen ] bd_01_workflow.png")


# ---------------------------------------------------------------------------
# bd_02 — the field progression
# ---------------------------------------------------------------------------

def fig_fields(grid_u, grid_v, grid_z, res):
    contour = np.asarray(res["boundary"].contour_uv, dtype=float)
    smoothed = res["smoothed"]
    lmax = res["lmax"]

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2))

    # 1 · raw grid_z
    _imshow_uv(axes[0], grid_u, grid_v, grid_z, "inferno")
    axes[0].set_title("1 · grid_z\nmean-IFF heatmap (input)",
                      fontsize=11.5, color=NAVY, weight="bold")

    # 2 · Gaussian-smoothed (sigma=8); shown masked to the footprint so the
    # smoothing effect is comparable to panel 1 (the halo beyond the edge is
    # filled by step 3 and used only by the Hessian).
    smoothed_disp = np.where(np.isnan(grid_z), np.nan, smoothed)
    _imshow_uv(axes[1], grid_u, grid_v, smoothed_disp, "inferno")
    axes[1].set_title("2 · Gaussian smooth\nσ = 8.0 (radial_gauss_sigma)",
                      fontsize=11.5, color=NAVY, weight="bold")

    # 3 · Hessian lambda-max (sigma=5), diverging + zero + contour
    a = max(float(np.nanmax(np.abs(lmax))), 1e-12)
    _imshow_uv(axes[2], grid_u, grid_v, lmax, "RdBu_r", vmin=-a, vmax=a)
    axes[2].contour(np.ma.masked_invalid(lmax.T), levels=[0.0],
                    extent=_uv_extent(grid_u, grid_v), colors="k",
                    linewidths=0.7, linestyles="--")
    _plot_contour(axes[2], contour, RADIAL_C, "radial contour")
    axes[2].set_title("3 · Hessian λmax\nσ = 5.0 · blue<0 dome, red>0 flank",
                      fontsize=11.5, color=NAVY, weight="bold")

    # 4 · final contour on the heatmap
    _imshow_uv(axes[3], grid_u, grid_v, grid_z, "inferno")
    _plot_contour(axes[3], contour, RADIAL_C, "RF boundary")
    pu, pv = _peak_uv(grid_u, grid_v, grid_z)
    axes[3].plot(pu, pv, "x", color="white", markersize=12, markeredgewidth=2.4)
    axes[3].set_title("4 · Foot contour\nclipped to the painted footprint",
                      fontsize=11.5, color=NAVY, weight="bold")

    for ax in axes:
        _zoom_to_contour(ax, contour)
        ax.set_xlabel("U (mm)"); ax.set_ylabel("V (mm)")

    fig.suptitle("Radial foot-of-mountain — one grid_z through the subprocess chain "
                 f"({SESSION}, gesture = {GTYPE})",
                 fontsize=14, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    out = FIG / "bd_02_fields.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[gen ] bd_02_fields.png")


# ---------------------------------------------------------------------------
# bd_03 — the final result + metrics
# ---------------------------------------------------------------------------

def fig_result(grid_u, grid_v, grid_z, res):
    b = res["boundary"]
    contour = np.asarray(b.contour_uv, dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = _imshow_uv(ax, grid_u, grid_v, grid_z, "inferno")
    _plot_contour(ax, contour, RADIAL_C, "RF boundary (radial foot)")
    ax.plot(b.peak_uv[0], b.peak_uv[1], "x", color="white", markersize=13,
            markeredgewidth=2.6, label="peak")
    ax.plot(b.centroid_uv[0], b.centroid_uv[1], "+", color="#7CFC00",
            markersize=14, markeredgewidth=2.4, label="centroid")
    _zoom_to_contour(ax, contour)
    ax.set_xlabel("U (mm)"); ax.set_ylabel("V (mm)")
    ax.set_title("Final radial RF boundary\n"
                 f"{SESSION} · gesture = {GTYPE}", fontsize=13, weight="bold",
                 color=NAVY)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("mean IFF (Hz)")

    metrics = (
        f"area        {b.area_uv:8.1f} mm2 (UV)\n"
        f"perimeter   {b.perimeter_uv:8.1f} mm\n"
        f"circularity {b.circularity:8.3f}\n"
        f"ellipse maj {b.pca_major_uv:8.1f} mm\n"
        f"ellipse min {b.pca_minor_uv:8.1f} mm\n"
        f"orientation {b.pca_orientation_deg:8.1f} deg"
    )
    ax.text(0.02, 0.02, metrics, transform=ax.transAxes, fontsize=9.5,
            family="monospace", va="bottom", ha="left", color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88,
                      edgecolor=NAVY))

    fig.tight_layout()
    out = FIG / "bd_03_result.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[gen ] bd_03_result.png")


# ---------------------------------------------------------------------------
# bd_04 — one of the 360 rays: the 1-D λmax profile behind the 2-D contour
# ---------------------------------------------------------------------------

def _uv_interp(field, grid_u, grid_v):
    """Linear interpolator of a (R, C) field over UV mm (axis0=U, axis1=V).

    Mirrors ``_imshow_uv``'s convention. Axes are flipped to strictly ascending
    if needed (``RegularGridInterpolator`` requires it); NaN outside the
    footprint propagates as NaN.
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
    return RegularGridInterpolator((u_axis, v_axis), fld,
                                   bounds_error=False, fill_value=np.nan)


def _ray_contour_crossing(peak_uv, direction, contour):
    """Smallest positive radius where the ray peak+t·direction meets the contour.

    Fail-fast: raises if the ray never crosses the closed contour (no fallback).
    """
    ox, oy = float(peak_uv[0]), float(peak_uv[1])
    dx, dy = float(direction[0]), float(direction[1])
    closed = np.vstack([contour, contour[:1]])
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
            "generate_boundary_workflow_figures: chosen ray never crosses the "
            "RF contour — cannot mark the foot location."
        )
    return min(ts)


def fig_ray_section(grid_u, grid_v, grid_z, res):
    """One of the 360 rays: λmax heatmap + the λmax-vs-radius profile."""
    b = res["boundary"]
    lmax = res["lmax"]
    contour = np.asarray(b.contour_uv, dtype=float)
    pu, pv = _peak_uv(grid_u, grid_v, grid_z)

    # A representative ray: toward the contour vertex farthest from the peak —
    # deterministic, and guaranteed to yield a long, clean foot crossing.
    d_to_peak = np.hypot(contour[:, 0] - pu, contour[:, 1] - pv)
    far = contour[int(np.argmax(d_to_peak))]
    vec = np.array([far[0] - pu, far[1] - pv], dtype=float)
    direction = vec / np.hypot(vec[0], vec[1])

    foot_r = _ray_contour_crossing((pu, pv), direction, contour)

    # Sample λmax and the IFF dome along the ray, in UV mm.
    lmax_interp = _uv_interp(lmax, grid_u, grid_v)
    iff_interp = _uv_interp(grid_z, grid_u, grid_v)
    r_max = foot_r * 1.7
    radii = np.linspace(0.0, r_max, 260)
    pts = np.column_stack([pu + radii * direction[0], pv + radii * direction[1]])
    lmax_prof = lmax_interp(pts)
    iff_prof = iff_interp(pts)

    # The detector's actual pick: centre of the first plateau that peaks above 0.
    finite = np.isfinite(lmax_prof)
    pk, props = find_peaks(np.where(finite, lmax_prof, -np.inf), plateau_size=1)
    left = np.asarray(props.get("left_edges", pk))
    right = np.asarray(props.get("right_edges", pk))
    centres = ((left + right) // 2).astype(int)
    plateau_r = None
    for k, c in enumerate(centres):
        if lmax_prof[pk[k]] > 0.0:
            plateau_r = float(radii[c])
            break

    BLUE = "#4A6FA5"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 6.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.25]})

    # ---- Left: the λmax heatmap with the ray drawn on it ----
    a = max(float(np.nanmax(np.abs(lmax))), 1e-12)
    _imshow_uv(axL, grid_u, grid_v, lmax, "RdBu_r", vmin=-a, vmax=a)
    axL.contour(np.ma.masked_invalid(lmax.T), levels=[0.0],
                extent=_uv_extent(grid_u, grid_v), colors="k",
                linewidths=0.7, linestyles="--")
    _plot_contour(axL, contour, RADIAL_C, "RF boundary")
    axL.plot([pu, pu + direction[0] * foot_r], [pv, pv + direction[1] * foot_r],
             "-", color=INK, linewidth=2.0,
             path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    axL.plot(pu, pv, "x", color="white", markersize=13, markeredgewidth=2.6)
    footp = (pu + direction[0] * foot_r, pv + direction[1] * foot_r)
    axL.plot(*footp, "o", color="#FFD400", markersize=11, markeredgecolor=INK,
             markeredgewidth=1.4, label="ray ∩ contour (foot)")
    _zoom_to_contour(axL, contour)
    axL.set_xlabel("U (mm)"); axL.set_ylabel("V (mm)")
    axL.set_title("Hessian λmax + one cast ray\nblue λ<0 dome · red λ>0 flank",
                  fontsize=12, color=NAVY, weight="bold")
    axL.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ---- Right: the λmax profile along that ray ----
    axR.axhline(0.0, color="#555", linewidth=1.0, linestyle="--")
    axR.plot(radii, lmax_prof, "-", color="#222", linewidth=2.0, label="λmax(ray)")
    axR.fill_between(radii, lmax_prof, 0.0, where=(lmax_prof < 0),
                     color=BLUE, alpha=0.30, interpolate=True,
                     label="λ<0 concave-down dome")
    axR.fill_between(radii, lmax_prof, 0.0, where=(lmax_prof > 0),
                     color=RADIAL_C, alpha=0.25, interpolate=True,
                     label="λ>0 concave-up flank")
    # the λmax=0 inflection the detector's pick deliberately walks past
    _zc = None
    for _i in range(1, len(lmax_prof)):
        if (np.isfinite(lmax_prof[_i - 1]) and np.isfinite(lmax_prof[_i])
                and lmax_prof[_i - 1] <= 0 < lmax_prof[_i]):
            _t = (0.0 - lmax_prof[_i - 1]) / (lmax_prof[_i] - lmax_prof[_i - 1])
            _zc = float(radii[_i - 1] + _t * (radii[_i] - radii[_i - 1]))
            break
    if _zc is not None:
        axR.axvline(_zc, color="#888", linewidth=1.3, linestyle="--",
                    label="λ=0 inflection (pick walks past)")
    axR.axvline(foot_r, color="#FFD400", linewidth=2.4)
    axR.axvline(foot_r, color=INK, linewidth=1.0, linestyle=":")
    axR.annotate("traced contour\nsits here", xy=(foot_r, 0.0),
                 xytext=(foot_r * 0.62, a * 0.55), fontsize=9, color=INK,
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))
    if plateau_r is not None:
        axR.plot([plateau_r], [0.0], "^", color="#2E7D5B", markersize=11,
                 markeredgecolor="white", markeredgewidth=1.0,
                 label="first λmax>0 plateau (detector's pick)", zorder=5)
    axR.set_xlabel("distance from peak along ray (mm)")
    axR.set_ylabel("λmax  (principal curvature)")
    axR.set_xlim(0, r_max)
    axR.set_title("λmax along the ray → foot = first positive plateau",
                  fontsize=12, color=NAVY, weight="bold")

    # IFF dome on a twin axis to show it flattening toward the plain as λmax crests.
    ax2 = axR.twinx()
    ax2.plot(radii, iff_prof, "-", color="#999", linewidth=1.6, alpha=0.9,
             label="mean IFF (dome)")
    ax2.set_ylabel("mean IFF (Hz)", color="#777")
    ax2.tick_params(axis="y", labelcolor="#777")

    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.suptitle("Radial foot-of-mountain — one of the 360 cast rays "
                 f"({SESSION}, gesture = {GTYPE})",
                 fontsize=14, weight="bold", color=NAVY, y=1.01)
    fig.tight_layout()
    out = FIG / "bd_04_ray_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[gen ] bd_04_ray_section.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    grid_u, grid_v, grid_z = _load_grid()
    res = _compute_radial(grid_u, grid_v, grid_z)
    fig_workflow()
    fig_fields(grid_u, grid_v, grid_z, res)
    fig_result(grid_u, grid_v, grid_z, res)
    fig_ray_section(grid_u, grid_v, grid_z, res)
    print(f"\nDone. Figures written to {FIG}")


if __name__ == "__main__":
    main()
