"""Rendering functions for 1D RF cross-section profiles and boundary crossings."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from analysis.receptive_field_mapping.metrics.rf_ray_sampling import (
    first_positive_plateau_radius,
    lmax_zero_crossing_radius,
    ray_contour_crossing,
    representative_ray_direction,
    uv_field_interpolator,
)

_BG = 'white'
_AX_BG = 'white'
_SPINE_COLOR = '#333333'


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors='black', which='both')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')


def _tangent_half_length(coords: np.ndarray) -> float:
    """Return a moderate tangent half-length scaled to ~8% of the coordinate range."""
    span = float(np.nanmax(coords) - np.nanmin(coords))
    return span * 0.02 if span > 0 else 1.0


def render_center_axis_profile(
    coords: np.ndarray,
    iff_values: np.ndarray,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "IFF (Hz)",
    contour_color: str = "red",
    gradient_crossings: np.ndarray | None = None,
    gradient_color: str = "#2ca02c",
) -> None:
    """Render a 1D line plot of IFF along one axis at a given center point.

    Parameters
    ----------
    gradient_crossings:
        Optional second set of boundary crossings from the gradient ridge
        boundary.  Rendered as diamond markers in *gradient_color*.
    gradient_color:
        Color for gradient boundary markers (default matplotlib green).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(iff_values)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
    else:
        ax.plot(coords, iff_values, color='#0077b6', linewidth=1.2)

        valid = ~np.isnan(iff_values)
        c_valid = coords[valid]
        v_valid = iff_values[valid]
        half = _tangent_half_length(coords)

        for xc in boundary_crossings:
            yc = float(np.interp(xc, c_valid, v_valid))
            ax.plot(xc, yc, 'o', color=contour_color, markersize=6, zorder=5)

            dx = np.gradient(c_valid)
            dy = np.gradient(v_valid)
            slope = float(np.interp(xc, c_valid, dy / dx))
            x0, x1 = xc - half, xc + half
            y0, y1 = yc - half * slope, yc + half * slope
            ax.plot([x0, x1], [y0, y1], color=contour_color, linewidth=1.2,
                    alpha=0.9, zorder=4)

        if gradient_crossings is not None:
            for xc in gradient_crossings:
                yc = float(np.interp(xc, c_valid, v_valid))
                ax.plot(xc, yc, 'D', color=gradient_color, markersize=5,
                        zorder=5, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_laplacian_context(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    laplacian: np.ndarray,
    smoothed: np.ndarray,
    contour_uv: np.ndarray | None,
    scanline_coord: float,
    scanline_axis: str,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    contour_color: str = "red",
) -> None:
    """Render a 2D view of the smoothed field and its Laplacian with the profile scanline overlaid.

    Parameters
    ----------
    grid_u, grid_v:
        (R, C) coordinate grids (axis 0 = U, axis 1 = V).
    laplacian:
        (R, C) Laplacian array (NaN outside mesh).
    smoothed:
        (R, C) Gaussian-smoothed IFF field (NaN outside mesh).
    contour_uv:
        (N, 2) inflection contour in UV space, or None.
    scanline_coord:
        The fixed coordinate value of the 1D profile scanline.
    scanline_axis:
        ``"U"`` for a constant-V scanline (profile along U),
        ``"V"`` for a constant-U scanline (profile along V).
    boundary_crossings:
        Sorted crossing coordinates along the profile axis.
    output_path:
        Path for the PNG output. SVG is written alongside.
    title:
        Figure suptitle.
    contour_color:
        Color for the inflection contour and crossing markers.
    """
    fig, (ax_smooth, ax_lap) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax_smooth)
    _style_ax(ax_lap)

    u_min, u_max = float(grid_u[0, 0]), float(grid_u[-1, 0])
    v_min, v_max = float(grid_v[0, 0]), float(grid_v[0, -1])
    extent = [u_min, u_max, v_min, v_max]

    jet = plt.cm.jet.copy()
    jet.set_bad('white')
    smoothed_t = np.ma.masked_invalid(smoothed.T)
    ax_smooth.imshow(smoothed_t, cmap=jet, origin='lower', extent=extent, aspect='auto')
    ax_smooth.set_title("Smoothed IFF", color='black')
    ax_smooth.set_xlabel("U (mm)")
    ax_smooth.set_ylabel("V (mm)")

    vabs = float(np.nanmax(np.abs(laplacian)))
    vabs = max(vabs, 1e-12)
    laplacian_t = np.ma.masked_invalid(laplacian.T)
    ax_lap.imshow(laplacian_t, cmap='RdBu_r', origin='lower', extent=extent,
                  vmin=-vabs, vmax=vabs, aspect='auto')
    ax_lap.set_title("Laplacian", color='black')
    ax_lap.set_xlabel("U (mm)")
    ax_lap.set_ylabel("V (mm)")

    for ax in (ax_smooth, ax_lap):
        if contour_uv is not None and len(contour_uv) > 0:
            closed = np.vstack([contour_uv, contour_uv[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=contour_color,
                    linewidth=1.2, zorder=5)

        if scanline_axis == "U":
            ax.axhline(scanline_coord, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.7, zorder=4)
            for xc in boundary_crossings:
                ax.plot(xc, scanline_coord, 'o', color=contour_color,
                        markersize=5, zorder=6)
        else:
            ax.axvline(scanline_coord, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.7, zorder=4)
            for yc in boundary_crossings:
                ax.plot(scanline_coord, yc, 'o', color=contour_color,
                        markersize=5, zorder=6)

    fig.suptitle(title, color='black', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_laplacian_profile(
    coords: np.ndarray,
    lap_values: np.ndarray,
    boundary_crossings: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    contour_color: str = "red",
) -> None:
    """Render a 1D Laplacian cross-section with zero-crossing reference line.

    The Laplacian curve is colored by sign: blue (negative, concave basin)
    below zero, red-orange (positive, convex flank) above zero. Boundary
    crossings mark the negative-to-positive transitions where the inflection
    contour intersects the scanline.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(lap_values)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
    else:
        valid = ~np.isnan(lap_values)
        c_valid = coords[valid]
        v_valid = lap_values[valid]

        ax.fill_between(
            c_valid, v_valid, 0,
            where=v_valid <= 0, interpolate=True,
            color='#0077b6', alpha=0.25, label='negative (concave)',
        )
        ax.fill_between(
            c_valid, v_valid, 0,
            where=v_valid >= 0, interpolate=True,
            color='#e63946', alpha=0.25, label='positive (convex)',
        )
        ax.plot(coords, lap_values, color='#333333', linewidth=1.2)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5, zorder=3)

        for xc in boundary_crossings:
            ax.plot(xc, 0, 'o', color=contour_color, markersize=6, zorder=5)
            ax.axvline(xc, color=contour_color, linewidth=0.6,
                       linestyle=':', alpha=0.6, zorder=4)

        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Laplacian")
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_gradient_profile(
    coords: np.ndarray,
    grad_values: np.ndarray,
    gradient_crossings: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    gradient_color: str = "#2ca02c",
) -> None:
    """Render a 1D gradient magnitude profile with peak markers.

    Shows |dIFF/dx| along the profile axis.  The gradient ridge boundary
    corresponds to the peaks of this curve on either side of the RF centre.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(grad_values)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
    else:
        valid = ~np.isnan(grad_values)
        c_valid = coords[valid]
        v_valid = grad_values[valid]

        ax.fill_between(c_valid, v_valid, 0, alpha=0.2, color=gradient_color)
        ax.plot(coords, grad_values, color='#333333', linewidth=1.2)

        for xc in gradient_crossings:
            yc = float(np.interp(xc, c_valid, v_valid))
            ax.plot(xc, yc, 'D', color=gradient_color, markersize=6,
                    zorder=5, markeredgecolor='black', markeredgewidth=0.5)
            ax.axvline(xc, color=gradient_color, linewidth=0.6,
                       linestyle=':', alpha=0.6, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("|∇IFF| (Hz/mm)")
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def _zero_crossings_1d(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the coordinates where a 1D curve crosses zero (linear interpolation)."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    out = []
    for i in range(1, len(values)):
        a, b = values[i - 1], values[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (a <= 0 < b) or (b <= 0 < a):
            t = (0.0 - a) / (b - a)
            out.append(coords[i - 1] + t * (coords[i] - coords[i - 1]))
    return np.asarray(out, dtype=float)


def render_combined_profile(
    coords: np.ndarray,
    iff_raw: np.ndarray,
    iff_smoothed: np.ndarray,
    lmax_values: np.ndarray,
    boundary_crossings: np.ndarray,
    peak_coord: float,
    output_path: Path,
    title: str,
    xlabel: str,
    contour_color: str = "red",
) -> None:
    """Overlay the raw + smoothed IFF and Hessian λmax curves on one figure.

    The primary axis carries the IFF: the raw slice as a faint dashed line and
    the Gaussian-smoothed slice as the bold primary line.  A twin axis carries
    the Hessian λmax (principal curvature) with blue (<0, concave dome) and
    red (>0, convex flank) sign-fill.

    Values of interest are marked: boundary crossings (●, active-contour color)
    on the smoothed curve, the λmax=0 inflections (faint vertical ticks), and the
    RF centre / peak (× on a dashed vertical).
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    if np.all(np.isnan(iff_raw)) and np.all(np.isnan(iff_smoothed)):
        ax.text(
            0.5, 0.5, "No valid data",
            transform=ax.transAxes, ha='center', va='center', color='black',
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IFF (Hz)")
        ax.set_title(title, color='black')
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    # --- Primary axis: IFF (raw faint, smoothed bold) ---
    ax.plot(coords, iff_raw, color='#0077b6', linewidth=1.0, linestyle='--',
            alpha=0.45, label='raw IFF', zorder=2)
    ax.plot(coords, iff_smoothed, color='#0077b6', linewidth=2.0,
            label='smoothed IFF', zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IFF (Hz)", color='#0077b6')
    ax.tick_params(axis='y', labelcolor='#0077b6')

    smooth_valid = ~np.isnan(iff_smoothed)
    c_sv = coords[smooth_valid]
    v_sv = iff_smoothed[smooth_valid]

    # Boundary crossings on the smoothed curve.
    for xc in boundary_crossings:
        if len(c_sv) == 0:
            break
        yc = float(np.interp(xc, c_sv, v_sv))
        ax.plot(xc, yc, 'o', color=contour_color, markersize=7, zorder=6,
                markeredgecolor='black', markeredgewidth=0.5,
                label='boundary crossing' if xc == boundary_crossings[0] else None)

    # RF centre / peak.
    ax.axvline(peak_coord, color='black', linewidth=1.0, linestyle='--',
               alpha=0.7, zorder=4)
    ax.plot([peak_coord], [np.nanmax(v_sv) if len(v_sv) else 0.0], 'x',
            color='black', markersize=9, markeredgewidth=2.0, zorder=6,
            label='RF centre (peak ×)')

    # --- Twin axis: Hessian λmax with sign fill ---
    ax2 = ax.twinx()
    lmax_valid = ~np.isnan(lmax_values)
    if np.any(lmax_valid):
        c_lv = coords[lmax_valid]
        l_lv = lmax_values[lmax_valid]
        vabs = max(float(np.nanmax(np.abs(lmax_values))), 1e-12)
        ax2.axhline(0, color='#888888', linewidth=0.8, linestyle='--', zorder=2)
        ax2.plot(coords, lmax_values, color='#333333', linewidth=1.4,
                 label='Hessian λmax', zorder=3)
        ax2.fill_between(c_lv, l_lv, 0, where=l_lv < 0, interpolate=True,
                         color='#4A6FA5', alpha=0.22, zorder=1)
        ax2.fill_between(c_lv, l_lv, 0, where=l_lv > 0, interpolate=True,
                         color='#e63946', alpha=0.18, zorder=1)
        ax2.set_ylim(-vabs * 1.1, vabs * 1.1)

        # λmax=0 inflections.
        for xc in _zero_crossings_1d(coords, lmax_values):
            ax2.axvline(xc, color='#888888', linewidth=0.7, linestyle=':',
                        alpha=0.7, zorder=2)
    ax2.set_ylabel("λmax (principal curvature)", color='#333333')
    ax2.tick_params(axis='y', labelcolor='#333333')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7.5, framealpha=0.85)
    ax.set_title(title, color='black')

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def render_raycast_section(
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    lmax: np.ndarray,
    grid_z: np.ndarray,
    smoothed: np.ndarray,
    contour_uv: np.ndarray,
    peak_uv: np.ndarray,
    output_path: Path,
    title: str,
    contour_color: str = "#D62728",
) -> None:
    """Render a 2-panel raycast, mirroring the ``hessian_explainer.html`` view.

    Left: the Hessian λmax field (diverging) with the λmax=0 ring, the RF
    boundary contour, and one representative ray cast from the centre toward the
    farthest contour vertex (× centre, ● foot).

    Right: the along-ray profile — λmax on the left axis (blue<0 / red>0 sign
    fill), the raw and smoothed IFF dome on a twin axis — with the λmax=0
    inflection tick, the ray∩contour foot (yellow line), and the detector's
    first-positive-plateau foot pick (▲).
    """
    contour = np.asarray(contour_uv, dtype=float)
    direction, _far = representative_ray_direction(peak_uv, contour)
    foot_r = ray_contour_crossing(peak_uv, direction, contour)

    pu, pv = float(peak_uv[0]), float(peak_uv[1])
    lmax_interp = uv_field_interpolator(lmax, grid_u, grid_v)
    raw_interp = uv_field_interpolator(grid_z, grid_u, grid_v)
    smooth_interp = uv_field_interpolator(smoothed, grid_u, grid_v)

    r_max = foot_r * 1.7
    radii = np.linspace(0.0, r_max, 260)
    pts = np.column_stack([pu + radii * direction[0], pv + radii * direction[1]])
    lmax_prof = lmax_interp(pts)
    raw_prof = raw_interp(pts)
    smooth_prof = smooth_interp(pts)

    plateau_r = first_positive_plateau_radius(radii, lmax_prof)
    zc_r = lmax_zero_crossing_radius(radii, lmax_prof)

    extent = [float(grid_u.min()), float(grid_u.max()),
              float(grid_v.min()), float(grid_v.max())]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15.0, 6.0), gridspec_kw={"width_ratios": [1.05, 1.25]})
    fig.patch.set_facecolor(_BG)
    _style_ax(axL)
    _style_ax(axR)

    # ---- Left: λmax heatmap + ray ----
    a = max(float(np.nanmax(np.abs(lmax))), 1e-12)
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('#e8e8e8')
    axL.imshow(np.ma.masked_invalid(lmax.T), origin='lower', extent=extent,
               cmap=cm, aspect='equal', vmin=-a, vmax=a)
    axL.contour(np.ma.masked_invalid(lmax.T), levels=[0.0], extent=extent,
                colors='k', linewidths=0.7, linestyles='--')
    closed = np.vstack([contour, contour[:1]])
    axL.plot(closed[:, 0], closed[:, 1], '-', color=contour_color, linewidth=2.0,
             label='RF boundary')
    axL.plot([pu, pu + direction[0] * foot_r], [pv, pv + direction[1] * foot_r],
             '-', color='#222222', linewidth=2.0, zorder=5)
    axL.plot(pu, pv, 'x', color='white', markersize=13, markeredgewidth=2.6, zorder=6)
    footp = (pu + direction[0] * foot_r, pv + direction[1] * foot_r)
    axL.plot(*footp, 'o', color='#FFD400', markersize=11, markeredgecolor='#222222',
             markeredgewidth=1.4, zorder=6, label='ray ∩ contour (foot)')
    margin = 18.0
    axL.set_xlim(contour[:, 0].min() - margin, contour[:, 0].max() + margin)
    axL.set_ylim(contour[:, 1].min() - margin, contour[:, 1].max() + margin)
    axL.set_xlabel("U (mm)")
    axL.set_ylabel("V (mm)")
    axL.set_title("Hessian λmax + one cast ray\nblue λ<0 dome · red λ>0 flank",
                  color='black', fontsize=11)
    axL.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

    # ---- Right: along-ray profile ----
    axR.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
    axR.plot(radii, lmax_prof, '-', color='#222222', linewidth=2.0, label='λmax(ray)')
    axR.fill_between(radii, lmax_prof, 0.0, where=(lmax_prof < 0), color='#4A6FA5',
                     alpha=0.30, interpolate=True, label='λ<0 dome')
    axR.fill_between(radii, lmax_prof, 0.0, where=(lmax_prof > 0), color=contour_color,
                     alpha=0.25, interpolate=True, label='λ>0 flank')
    if zc_r is not None:
        axR.axvline(zc_r, color='#888888', linewidth=1.3, linestyle='--',
                    label='λ=0 inflection')
    axR.axvline(foot_r, color='#FFD400', linewidth=2.4)
    axR.axvline(foot_r, color='#222222', linewidth=1.0, linestyle=':')
    if plateau_r is not None:
        axR.plot([plateau_r], [0.0], '^', color='#2E7D5B', markersize=12,
                 markeredgecolor='white', markeredgewidth=1.0, zorder=5,
                 label="foot pick (first λmax>0 plateau)")
    axR.set_xlabel("distance from centre along ray (mm)")
    axR.set_ylabel("λmax (principal curvature)")
    axR.set_xlim(0, r_max)
    axR.set_title("λmax along the ray → foot = first positive plateau",
                  color='black', fontsize=11)

    ax2 = axR.twinx()
    ax2.plot(radii, raw_prof, '-', color='#7fb3d5', linewidth=1.2, alpha=0.7,
             label='raw IFF (dome)')
    ax2.plot(radii, smooth_prof, '-', color='#999999', linewidth=1.8,
             label='smoothed IFF (dome)')
    ax2.set_ylabel("IFF (Hz)", color='#777777')
    ax2.tick_params(axis='y', labelcolor='#777777')

    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8.0, framealpha=0.92)

    fig.suptitle(title, color='black', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    svg_path = output_path.with_suffix('.svg')
    fig.savefig(svg_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
