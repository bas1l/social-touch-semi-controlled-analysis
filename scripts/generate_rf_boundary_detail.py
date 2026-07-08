"""Deep-dive figures for Stage 3 — the radial "foot of mountain" boundary.

Opens the same worked example (session ``2022-06-17_ST16-02``, gesture ``all``),
reproduces ``compute_radial_foot_boundary`` step by step, and renders three
figures that explain the two hard parts newcomers ask about:

* ``05a_hessian_lmax.png`` -- how the Hessian λmax field is built and what it
                             means (λmax < 0 over the concave-down dome, λmax > 0
                             at the concave-up foot ring).
* ``05b_ray_profile.png``  -- the contour-selection rule along ONE ray: the first
                             positive λmax plateau centre = the foot radius, with
                             the raw-footprint snap limit.
* ``05c_envelope.png``     -- the un-snapped star-convex radial curve vs the final
                             contour after clipping to the painted footprint.

Reproduces the production functions in
``metrics/rf_radial_foot_boundary.py`` with the DAG's parameters
(gauss_sigma=8, hess_sigma=5, n_angles=360, savgol_window=31,
envelope_smooth_sigma=1.5).

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_rf_boundary_detail.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from _vendor.path_tools import get_project_data_root
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import find_peak_location
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    _compute_hessian_lmax,
    _extract_radial_plateau_foot,
    envelope_contour_to_footprint,
)

SESSION = "2022-06-17_ST16-02"
GAUSS_SIGMA, HESS_SIGMA = 8.0, 5.0
N_ANGLES, SAVGOL, ENV_SMOOTH = 360, 31, 1.5

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG = REPO_ROOT / "docs" / "receptive_field_workflow" / "figures"


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"generate_rf_boundary_detail: missing: {p}")
    return p


def _jet():
    cm = plt.cm.jet.copy()
    cm.set_bad("lightgrey")
    return cm


def fig_lmax(grid_z, lmax, peak, unsnapped_rc, final_rc):
    peak_r, peak_c = peak
    fig, (axz, axl) = plt.subplots(1, 2, figsize=(13, 5.6))

    # left: IFF dome + a sample of the radial rays
    axz.imshow(np.ma.masked_invalid(grid_z), cmap=_jet(), origin="upper")
    axz.plot(peak_c, peak_r, "kx", markersize=11, markeredgewidth=2)
    for i in range(0, len(unsnapped_rc), max(1, len(unsnapped_rc) // 16)):
        axz.plot([peak_c, unsnapped_rc[i, 1]], [peak_r, unsnapped_rc[i, 0]],
                 "-", color="white", linewidth=0.7, alpha=0.7)
    axz.set_title("IFF dome + radial rays from the peak", fontsize=12)
    axz.set_xticks([]); axz.set_yticks([])

    # right: lmax diverging, with zero-crossing and final contour
    a = max(float(np.nanmax(np.abs(lmax))), 1e-12)
    im = axl.imshow(np.ma.masked_invalid(lmax), cmap="RdBu_r", origin="upper",
                    vmin=-a, vmax=a)
    axl.contour(np.ma.masked_invalid(lmax), levels=[0.0], colors="k",
                linewidths=0.8, linestyles="--")
    axl.plot(final_rc[:, 1], final_rc[:, 0], "-", color="lime", linewidth=2.0)
    axl.plot(peak_c, peak_r, "kx", markersize=11, markeredgewidth=2)
    axl.set_title("Hessian λmax  ·  blue λ<0 = dome,  red λ>0 = foot", fontsize=12)
    axl.set_xticks([]); axl.set_yticks([])
    cb = fig.colorbar(im, ax=axl, fraction=0.046, pad=0.02)
    cb.set_label("λmax (curvature)")

    fig.suptitle("① Hessian λmax field — the RF edge is where curvature flips sign",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / "05a_hessian_lmax.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[gen ] 05a_hessian_lmax.png")


def _sample_ray(grid_z, lmax, peak, angle):
    """Reproduce the per-ray sampling from _extract_radial_plateau_foot."""
    n_rows, n_cols = lmax.shape
    peak_r, peak_c = peak
    max_radius = int(math.ceil(math.hypot(n_rows, n_cols)))
    radii = np.arange(1, max_radius, dtype=float)
    rows = peak_r + radii * math.cos(angle)
    cols = peak_c + radii * math.sin(angle)
    inside = (rows >= 0) & (rows < n_rows - 1) & (cols >= 0) & (cols < n_cols - 1)
    radii = radii[inside]
    coords = np.array([rows[inside], cols[inside]])
    lmax_filled = np.where(np.isnan(lmax), -np.inf, lmax)
    lmax_samp = map_coordinates(lmax_filled, coords, order=1, mode="nearest")
    raw_samp = map_coordinates(grid_z, coords, order=0, mode="constant", cval=np.nan)
    z_samp = map_coordinates(np.where(np.isnan(grid_z), np.nan, grid_z), coords,
                             order=1, mode="nearest")
    raw_valid = np.nonzero(~np.isnan(raw_samp))[0]
    raw_limit = int(raw_valid[-1]) if raw_valid.size else 0
    finite = np.isfinite(lmax_samp)
    peaks, props = find_peaks(np.where(finite, lmax_samp, -np.inf), plateau_size=1)
    left = props.get("left_edges", peaks)
    right = props.get("right_edges", peaks)
    centres = ((np.asarray(left) + np.asarray(right)) // 2).astype(int)
    foot_idx = None
    for k, centre in enumerate(centres):
        if lmax_samp[peaks[k]] > 0.0:
            foot_idx = int(centre)
            break
    return radii, lmax_samp, z_samp, foot_idx, raw_limit


def fig_ray_profile(grid_z, lmax, peak, unsnapped_rc):
    # Choose a *pedagogically clear* ray: one whose natural foot is interior to
    # the footprint (not clipped by the data edge) and where λmax visibly peaks
    # then declines — so the "first positive plateau" is obvious. Score by the
    # drop after the peak; fall back to the longest ray if none is interior.
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    best = None
    fallback = None
    for ang in angles:
        radii, lmax_s, z_s, foot, raw_lim = _sample_ray(grid_z, lmax, peak, ang)
        if foot is None:
            continue
        if fallback is None or radii[foot] > fallback[0]:
            fallback = (radii[foot], ang, radii, lmax_s, z_s, foot, raw_lim)
        # interior foot with room to show the decline afterwards
        tail_end = min(foot + 10, len(lmax_s))
        if foot < raw_lim - 1 and tail_end - foot >= 4:
            drop = float(lmax_s[foot] - np.min(lmax_s[foot:tail_end]))
            score = drop
            if best is None or score > best[0]:
                best = (score, ang, radii, lmax_s, z_s, foot, raw_lim)
    chosen = best or fallback
    if chosen is None:
        raise ValueError("generate_rf_boundary_detail: no ray with a positive foot")
    _s, ang, radii, lmax_s, z_s, foot, raw_lim = chosen

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    m = radii <= (radii[foot] * 1.6)
    ax.axhline(0, color="#999999", linewidth=0.9)
    ax.fill_between(radii[m], lmax_s[m], 0, where=(lmax_s[m] < 0),
                    color="#c9dbf0", alpha=0.7, label="λ<0 (concave-down dome)")
    ax.fill_between(radii[m], lmax_s[m], 0, where=(lmax_s[m] >= 0),
                    color="#f2c9c0", alpha=0.7, label="λ>0 (concave-up foot)")
    ax.plot(radii[m], lmax_s[m], "-", color="#8b1a0a", linewidth=1.8, label="λmax along ray")
    ax.plot(radii[foot], lmax_s[foot], "o", color="green", markersize=11,
            label="foot = 1st positive plateau")
    ax.axvline(radii[foot], color="green", linestyle="--", linewidth=1.2)
    ax.axvline(radii[raw_lim], color="#555555", linestyle=":", linewidth=1.2,
               label="raw-footprint snap limit")
    ax.set_xlabel("radius from peak (grid cells)")
    ax.set_ylabel("λmax (curvature)", color="#8b1a0a")
    ax.tick_params(axis="y", labelcolor="#8b1a0a")

    ax2 = ax.twinx()
    ax2.plot(radii[m], z_s[m], "-", color="#1F2A44", linewidth=1.3, alpha=0.6)
    ax2.set_ylabel("IFF along ray (Hz)", color="#1F2A44")
    ax2.tick_params(axis="y", labelcolor="#1F2A44")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("② Contour selection along one ray — the foot is the first λmax>0 plateau",
                 fontsize=12)
    fig.tight_layout()
    out = FIG / "05b_ray_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen ] 05b_ray_profile.png  (ray angle {math.degrees(ang):.0f}°)")


def fig_envelope(grid_z, peak, unsnapped_rc, final_rc):
    peak_r, peak_c = peak
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax in (a1, a2):
        ax.imshow(np.ma.masked_invalid(grid_z), cmap=_jet(), origin="upper")
        ax.plot(peak_c, peak_r, "kx", markersize=10, markeredgewidth=2)
        ax.set_xticks([]); ax.set_yticks([])

    uns = np.vstack([unsnapped_rc, unsnapped_rc[:1]])
    a1.plot(uns[:, 1], uns[:, 0], "-", color="darkorange", linewidth=2.0)
    a1.set_title("Un-snapped radial curve\n(one radius per angle — bulges into grey)",
                 fontsize=11)

    fin = np.vstack([final_rc, final_rc[:1]]) if not np.allclose(final_rc[0], final_rc[-1]) else final_rc
    a2.plot(fin[:, 1], fin[:, 0], "-", color="lime", linewidth=2.2)
    a2.set_title("Envelope-clipped to footprint\n(grid_z > 0, seed component, re-traced)",
                 fontsize=11)

    fig.suptitle("③ Contour selection — clip the star-convex curve to the painted footprint",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / "05c_envelope.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[gen ] 05c_envelope.png")


def main() -> None:
    data_root = Path(get_project_data_root())
    npz_path = _require(
        data_root / "4_analysed" / "spatial_extract_boundaries" / "iff_mean"
        / SESSION / f"{SESSION}_population_response_fields.npz"
    )
    d = np.load(npz_path, allow_pickle=True)
    grid_z = d["grid_z_all"]

    peak = find_peak_location(grid_z)
    if peak is None:
        raise ValueError("generate_rf_boundary_detail: all-NaN grid")

    lmax = _compute_hessian_lmax(grid_z, GAUSS_SIGMA, HESS_SIGMA)
    res = _extract_radial_plateau_foot(
        grid_z, lmax, peak, n_angles=N_ANGLES, plateau_size=1,
        savgol_window=SAVGOL, require_positive=True,
    )
    env = envelope_contour_to_footprint(
        res["contour_unsnapped_rc"], grid_z, peak, smooth_sigma=ENV_SMOOTH,
    )

    FIG.mkdir(parents=True, exist_ok=True)
    fig_lmax(grid_z, lmax, peak, res["contour_unsnapped_rc"], env["contour_rc"])
    fig_ray_profile(grid_z, lmax, peak, res["contour_unsnapped_rc"])
    fig_envelope(grid_z, peak, res["contour_unsnapped_rc"], env["contour_rc"])
    print(f"\nDone. Figures written to {FIG}")


if __name__ == "__main__":
    main()
