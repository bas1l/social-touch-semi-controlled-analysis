"""Generate the example figures for ``docs/receptive_field_workflow/README.md``.

The receptive-field walkthrough illustrates how one session/neuron's RF is built
across four stages (preprocessing -> single-touch reduction -> aggregation ->
boundary definition).  Most stages already have PNGs written by the processing
pipeline; this script curates those into ``docs/receptive_field_workflow/figures/``
under stable, numbered names and fills the two stages that have no static figure
today:

* ``01_raw_input.png``    -- forearm surface + all contact points, spike touches
                             highlighted (preprocessing).
* ``03_single_touch.png`` -- one representative touch reduced to a sparse
                             per-vertex mean-IFF map, shown both in the camera
                             view and on the SLIM UV unwrap (single-touch stage).

Worked example: session ``2022-06-17_ST16-02`` (ST16-02).

Fail-fast per repo convention: every expected source artifact is required.  A
missing PNG/NPZ raises ``FileNotFoundError`` rather than being skipped.

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_rf_workflow_doc_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.spatial import cKDTree

from _vendor.path_tools import get_project_data_root
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_colors,
    load_forearm_vertices,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.rf_extraction_io import load_rf_camera_rotation
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import load_slim_uv_cache

NAVY = "#1F2A44"

SESSION = "2022-06-17_ST16-02"
CMAP = "inferno"  # match rf_population_map_renderer

# Repo-relative output directory for the committed figures.
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "docs" / "receptive_field_workflow" / "figures"


def _require(path: Path) -> Path:
    """Return *path*, raising if it does not exist (no silent skipping)."""
    if not path.exists():
        raise FileNotFoundError(
            f"generate_rf_workflow_doc_figures: required source artifact missing: {path}"
        )
    return path


def _copy_curated(analysed: Path) -> None:
    """Copy the PNGs the pipeline already wrote into ``figures/`` with stable names."""
    boundaries = analysed / "spatial_extract_boundaries" / "iff_mean" / SESSION
    slim_uv = analysed / "spatial_slim_uv" / SESSION
    profiles = analysed / "spatial_extract_rf_profiles" / "iff_mean" / SESSION

    # source -> curated destination name
    copy_map = {
        slim_uv / f"{SESSION}_slim_uv_qc.png": "02_slim_unwrap.png",
        boundaries / "aggregated" / f"{SESSION}_rf_population_all.png": "04_aggregate.png",
        boundaries / "inspection" / "radial_all_foot.png": "05_boundary_steps.png",
        boundaries
        / f"{SESSION}_rf_population_all_circular_contour_center_clean_rfboundary.png": "06_rf_outline.png",
        profiles / f"{SESSION}_rf_gradient_1d_u_centroid_all.png": "07_profile_1d.png",
    }

    for src, dst_name in copy_map.items():
        shutil.copyfile(_require(src), FIGURES_DIR / dst_name)
        print(f"[copy] {dst_name} <- {src}")


def _csv_snapshot_rows(merged_csv: Path):
    """Return (columns, rows) — a few contact frames from the initial merged CSV."""
    cols = ["time", "contact_detected", "contact_area", "contact_depth",
            "contact_points", "Nerve_spike", "Nerve_freq",
            "trial_id", "single_touch_id", "type_metadata"]
    df = pd.read_csv(merged_csv, usecols=cols)
    # A window during an actual touch: rows where a contact was detected.
    contact = df[(df["contact_detected"] == 1) & (df["trial_id"] > 0)]
    if contact.empty:
        raise ValueError(f"generate_rf_workflow_doc_figures: no contact rows in {merged_csv}")
    window = contact.iloc[200:207]  # a mid-session touch, 7 frames

    def cell(col, v):
        if pd.isna(v):
            return "NaN"
        if col == "time":
            return f"{v:.3f}"
        if col == "contact_points":
            s = str(v)
            return (s[:16] + "…]") if len(s) > 18 else s
        if col in ("contact_area", "contact_depth", "Nerve_freq"):
            return f"{float(v):.1f}"
        if col in ("contact_detected", "Nerve_spike", "trial_id", "single_touch_id"):
            return str(int(v))
        return str(v)

    header = ["time", "cont?", "area", "depth", "contact_points",
              "spike", "IFF", "trial", "touch", "type"]
    body = [[cell(c, r[c]) for c in cols] for _, r in window.iterrows()]
    return header, body


def _draw_csv_table(ax, header, body):
    ax.axis("off")
    ax.set_title("① Initial merged CSV  (1 kHz rows)", fontsize=12,
                 weight="bold", color=NAVY, loc="left", pad=8)
    tbl = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for (row, _), c in tbl.get_celld().items():
        c.set_edgecolor("#cccccc")
        if row == 0:
            c.set_facecolor(NAVY)
            c.get_text().set_color("white")
            c.get_text().set_weight("bold")


def _draw_pointcloud(ax, xyz, colors, R):
    ax.set_title("② Input forearm point cloud", fontsize=12, weight="bold",
                 color=NAVY, loc="left", pad=8)
    pts = (xyz @ R.T) if R is not None else xyz
    step = max(1, len(pts) // 15000)
    p = pts[::step]
    if colors is not None:
        c = colors[::step].astype(np.float64) / 255.0
    else:
        c = "#c98b6b"
    ax.scatter(p[:, 0], p[:, 1], c=c, s=2, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{len(xyz):,} vertices (RF-centred)", fontsize=9, color="#555555")


def _draw_workflow(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("③ Processing steps", fontsize=12, weight="bold",
                 color=NAVY, loc="left", pad=8)
    steps = [
        "Merged CSV — 1 kHz rows",
        "Prepare · NaN-interp + block/trial IDs",
        "Series · velocity, pressure, MoS",
        "Filter · keep real touches only",
        "Group · by (block, trial, touch)",
        "Contacts · 30 Hz dedup + KD-tree snap",
        "→ per-touch contacts on the surface",
    ]
    n = len(steps)
    h = 0.9 / n
    for i, txt in enumerate(steps):
        y = 0.93 - i * h
        is_out = i == n - 1
        box = FancyBboxPatch(
            (0.04, y - h * 0.42), 0.92, h * 0.72,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.1,
            edgecolor=NAVY,
            facecolor=("#eaf0f6" if not is_out else "#E06A2B"),
        )
        ax.add_patch(box)
        ax.text(0.5, y, txt, ha="center", va="center", fontsize=8.5,
                color=("white" if is_out else "#1a1a1a"),
                weight=("bold" if is_out else "normal"))
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (0.5, y - h * 0.42), (0.5, y - h * 0.58),
                arrowstyle="-|>", mutation_scale=9, color=NAVY, linewidth=1.1))


def _draw_stats(ax, pop_data):
    ax.axis("off")
    ax.set_title("④ Session summary", fontsize=12, weight="bold",
                 color=NAVY, loc="left", pad=8)
    T = len(pop_data.touch_triple_keys)
    C = len(pop_data.cp_vertex_idx)
    V = len(pop_data.forearm_vertices)
    spike_touches = int(pop_data.spike_elicited.sum())
    spike_touch_idx = np.where(pop_data.spike_elicited)[0]
    mask = np.isin(pop_data.cp_touch_idx, spike_touch_idx)
    unique_spike_vtx = int(np.unique(pop_data.cp_vertex_idx[mask]).shape[0])
    gest, counts = np.unique(
        np.array([str(x) for x in pop_data.gesture_types]), return_counts=True)

    lines = [
        f"Total touches (T):       {T}",
        f"Contact points (C):      {C:,}",
        f"Spike touches:           {spike_touches}",
        f"Unique spike vertices:   {unique_spike_vtx:,}",
        f"Forearm vertices (V):    {V:,}",
        "",
        "Gesture breakdown:",
    ] + [f"   {g}: {c}" for g, c in zip(gest, counts)]
    ax.text(0.03, 0.92, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", fontsize=10.5, family="monospace", color="#1a1a1a")


def _generate_raw_input(pop_data, merged_csv: Path, colors, camera_R) -> None:
    """Stage-0 figure: initial CSV + forearm point cloud + workflow + summary stats."""
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25],
                          hspace=0.28, wspace=0.22)
    header, body = _csv_snapshot_rows(merged_csv)
    _draw_csv_table(fig.add_subplot(gs[0, :]), header, body)
    _draw_pointcloud(fig.add_subplot(gs[1, 0]), pop_data.forearm_vertices, colors, camera_R)
    _draw_workflow(fig.add_subplot(gs[1, 1]))
    _draw_stats(fig.add_subplot(gs[1, 2]), pop_data)
    fig.suptitle(f"Stage 0 — inputs & preprocessing  ·  {SESSION}",
                 fontsize=15, weight="bold", color=NAVY, y=0.98)
    out = FIGURES_DIR / "01_raw_input.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[gen ] 01_raw_input.png")


def _generate_single_touch(pop_data, rf_data, cache, camera_R: np.ndarray | None) -> None:
    """Stage-1 figure: one touch reduced to a sparse per-vertex mean-IFF map.

    Left panel  -- camera view of the forearm with the touch's contacted
                   vertices coloured by mean IFF.
    Right panel -- the same vertices placed on the SLIM UV unwrap.
    """
    forearm_xyz = pop_data.forearm_vertices  # (V, 3)

    # Pick a *representative* stroke touch: a proximal stroke sweeps a clean
    # proximal->distal stripe, and the median-footprint one avoids both the
    # sparse taps and the rare oversized outliers (the widest touch covers a
    # third of the mesh and reads like an aggregate, not a single touch).
    counts = np.array([len(v) for v in rf_data.rf_vertex_indices])
    gestures = np.array([str(x) for x in pop_data.gesture_types])
    candidate_idx = np.where((gestures == "stroke_proximal") & (counts > 0))[0]
    if candidate_idx.size == 0:
        raise ValueError(
            "generate_rf_workflow_doc_figures: no stroke_proximal touch with "
            f"contacted vertices in the single-touch RF maps for {SESSION}."
        )
    median_count = np.median(counts[candidate_idx])
    ti = int(candidate_idx[np.argmin(np.abs(counts[candidate_idx] - median_count))])
    vtx = np.asarray(rf_data.rf_vertex_indices[ti], dtype=np.int64)
    vals = np.asarray(rf_data.rf_values[ti], dtype=np.float64)
    gesture = str(pop_data.gesture_types[ti])

    # Raw forearm vertex -> nearest SLIM/cleaned mesh vertex -> UV coordinate.
    slim_tree = cKDTree(cache.V)
    _, slim_idx = slim_tree.query(forearm_xyz[vtx])
    touch_uv = cache.uv[slim_idx]  # (n_touch_vtx, 2)

    # Camera-view positions (match the raw-input figure orientation).
    pts3d = forearm_xyz[vtx]
    bg3d = forearm_xyz
    if camera_R is not None:
        pts3d = pts3d @ camera_R.T
        bg3d = bg3d @ camera_R.T

    fig, (ax3d, axuv) = plt.subplots(1, 2, figsize=(13, 6))
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    # --- left: camera view ---
    ax3d.scatter(bg3d[:, 0], bg3d[:, 1], c="#dddddd", s=2, linewidths=0, alpha=0.5)
    sc = ax3d.scatter(
        pts3d[:, 0], pts3d[:, 1], c=vals, cmap=CMAP, vmin=vmin, vmax=vmax,
        s=18, linewidths=0,
    )
    ax3d.set_aspect("equal")
    ax3d.set_xlabel("horiz. (mm)")
    ax3d.set_ylabel("vert. (mm)")
    ax3d.set_title("Camera view — one touch on the forearm")
    fig.colorbar(sc, ax=ax3d, shrink=0.8, label="mean IFF (Hz)")

    # --- right: SLIM UV unwrap ---
    axuv.scatter(cache.uv[:, 0], cache.uv[:, 1], c="#eeeeee", s=2, linewidths=0)
    sc2 = axuv.scatter(
        touch_uv[:, 0], touch_uv[:, 1], c=vals, cmap=CMAP, vmin=vmin, vmax=vmax,
        s=18, linewidths=0,
    )
    axuv.set_aspect("equal")
    axuv.set_xlabel("U")
    axuv.set_ylabel("V")
    axuv.set_title("SLIM UV unwrap — same touch")
    fig.colorbar(sc2, ax=axuv, shrink=0.8, label="mean IFF (Hz)")

    fig.suptitle(
        f"Single touch reduced to {len(vtx)} contacted vertices "
        f"(touch #{ti}, gesture: {gesture})",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "03_single_touch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen ] 03_single_touch.png  (touch #{ti}, {len(vtx)} vertices, {gesture})")


def main() -> None:
    data_root = get_project_data_root()
    if data_root is None:
        raise RuntimeError(
            "generate_rf_workflow_doc_figures: project data root not resolved."
        )
    data_root = Path(data_root)
    analysed = _require(data_root / "4_analysed")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Curate the PNGs the pipeline already wrote ---
    _copy_curated(analysed)

    # --- Load the inputs needed for the two generated figures ---
    series_csv = _require(
        analysed / "touch_compute_series" / f"{SESSION}_series_augmented.csv"
    )
    session_dir = data_root / "3_merged" / SESSION / "forearm_rf_centered"
    forearm_ply = resolve_forearm_ply(session_dir, SESSION)
    if forearm_ply is None:
        raise FileNotFoundError(
            f"generate_rf_workflow_doc_figures: RF-centred forearm PLY not found in "
            f"{session_dir}"
        )

    merged_csv = _require(
        data_root / "3_merged" / SESSION
        / f"{SESSION}_semicontrolled_aggregated_session.csv"
    )
    forearm_colors = load_forearm_vertex_colors(forearm_ply)

    pop_data = load_population_data(series_csv, forearm_ply)
    n_verts = len(pop_data.forearm_vertices)

    mean_npz = _require(
        analysed / "spatial_map_single_touch" / SESSION / "single_touch_rf_maps_mean.npz"
    )
    rf_data = load_population_rf_data(mean_npz, pop_data.touch_triple_keys, n_verts)

    slim_cache_path = _require(
        analysed / "spatial_slim_uv" / SESSION / f"{SESSION}_slim_uv.npz"
    )
    cache = load_slim_uv_cache(slim_cache_path)

    # Camera rotation is purely for view orientation; fall back to the raw axes
    # if this session was never assigned a camera (documented, cosmetic only).
    camera_dir = analysed / "spatial_set_camera"
    try:
        camera_R = load_rf_camera_rotation(camera_dir, SESSION)
    except ValueError as exc:
        print(f"[warn] no camera settings for {SESSION}; using raw axes ({exc})")
        camera_R = None

    _generate_raw_input(pop_data, merged_csv, forearm_colors, camera_R)
    _generate_single_touch(pop_data, rf_data, cache, camera_R)

    print(f"\nDone. Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
