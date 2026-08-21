"""Deep-dive figures for the single-touch reduction stage (Stage 1).

Takes the *same* worked example as ``generate_rf_workflow_doc_figures.py`` --
session ``2022-06-17_ST16-02``, touch #166 = (block 3, trial 7, single-touch 81),
a median-footprint ``stroke_proximal`` -- and shows, step by step, how the raw
1 kHz frames become one sparse per-vertex **mean-IFF** map:

* ``03a_touch_frames.png``       -- the touch's contact frames (contacts sweep
                                    across the arm; colour = that frame's IFF).
* ``03b_grouping_averaging.png`` -- the reduction itself: Σ IFF, contact count,
                                    and mean = Σ / count (reproduces
                                    ``_compute_touch_rf``, mean branch only).
* ``03c_attributes.png``         -- this touch's stimulus / kinematic / mechanical
                                    / neural attributes.

Frame reduction reproduces ``rf_single_touch_pipeline._compute_touch_rf`` exactly
(iterate every 1 kHz frame, ``val_sum += IFF``, ``count += 1``, ``mean = sum/count``).

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_rf_single_touch_detail.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from _vendor.path_tools import get_project_data_root
from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.data.rf_extraction_io import load_rf_camera_rotation
from analysis.receptive_field_mapping.data.touch_playback_data import load_playback_data

SESSION = "2022-06-17_ST16-02"
BLOCK, TRIAL, TOUCH = 3, 7, 81  # touch #166 in the population ordering
# Depth-field sidecar location, mirroring the DAG config keys of the same names.
BLOCKS_STAGE_DIR = 'blocks_rf_centered'
BLOCK_CSV_STEM_SUFFIX = '_pca-xyz'
CMAP = "inferno"

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG = REPO_ROOT / "docs" / "receptive_field_workflow" / "figures"


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"generate_rf_single_touch_detail: missing: {p}")
    return p


def _project(xyz: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    """Camera-view 2-D projection (drop Z) for display."""
    return (xyz @ R.T)[:, :2] if R is not None else xyz[:, :2]


def _frame_groups(ev):
    """Group the touch's 1 kHz frames into runs of identical contact config.

    Returns list of (row_start, row_end, vertex_indices, mean_iff) -- one entry
    per unique contact frame (contacts update at ~30 Hz within a 1 kHz touch).
    """
    n = len(ev.frame_vertex_indices)
    groups = []
    start = 0
    for fi in range(1, n + 1):
        changed = (
            fi == n
            or not np.array_equal(ev.frame_vertex_indices[fi], ev.frame_vertex_indices[start])
        )
        if changed:
            verts = np.asarray(ev.frame_vertex_indices[start], dtype=np.int64)
            if len(verts) > 0:
                mean_iff = float(np.nanmean(ev.frame_iff[start:fi]))
                groups.append((start, fi, verts, mean_iff))
            start = fi
    return groups


def _bg_scatter(ax, bg2d):
    ax.scatter(bg2d[:, 0], bg2d[:, 1], c="#e2e2e2", s=1.5, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_frames(ev, forearm_xyz, R) -> None:
    groups = _frame_groups(ev)
    bg2d = _project(forearm_xyz, R)
    vmin = min(g[3] for g in groups)
    vmax = max(g[3] for g in groups)
    norm = Normalize(vmin=vmin, vmax=vmax)

    ncol = len(groups)
    fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 3.2))
    if ncol == 1:
        axes = [axes]
    for k, (r0, r1, verts, mean_iff) in enumerate(groups):
        ax = axes[k]
        _bg_scatter(ax, bg2d)
        p2d = _project(forearm_xyz[verts], R)
        ax.scatter(p2d[:, 0], p2d[:, 1], c=[mean_iff] * len(verts), cmap=CMAP,
                   norm=norm, s=6, linewidths=0)
        ax.set_title(f"Frame {k + 1}\nIFF ≈ {mean_iff:.0f} Hz", fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("frame mean IFF (Hz)")
    fig.suptitle(
        f"One touch, frame by frame — contacts sweep proximally; the neuron peaks "
        f"at frame {int(np.argmax([g[3] for g in groups])) + 1}",
        fontsize=13,
    )
    out = FIG / "03a_touch_frames.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen ] 03a_touch_frames.png  ({ncol} frames)")


def generate_grouping_averaging(ev, forearm_xyz, R) -> None:
    """Reproduce _compute_touch_rf (mean branch) and show Σ / count / mean."""
    n_vertices = len(forearm_xyz)
    val_sum = np.zeros(n_vertices)
    count = np.zeros(n_vertices)
    for fi in range(len(ev.frame_vertex_indices)):
        verts = ev.frame_vertex_indices[fi]
        if len(verts) == 0:
            continue
        np.add.at(val_sum, verts, ev.frame_iff[fi])
        np.add.at(count, verts, 1.0)

    contacted = np.where(count > 0)[0]
    mean = val_sum[contacted] / count[contacted]
    valid = ~np.isnan(mean)
    contacted = contacted[valid]
    mean = mean[valid]
    sum_c = val_sum[contacted]
    cnt_c = count[contacted]

    bg2d = _project(forearm_xyz, R)
    p2d = _project(forearm_xyz[contacted], R)

    panels = [
        ("Σ IFF  (val_sum)", sum_c, "summed IFF (Hz)"),
        ("contact count", cnt_c, "frames touching vertex"),
        ("mean = Σ / count", mean, "mean IFF (Hz)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, vals, clabel) in zip(axes, panels):
        _bg_scatter(ax, bg2d)
        sc = ax.scatter(p2d[:, 0], p2d[:, 1], c=vals, cmap=CMAP, s=7, linewidths=0)
        ax.set_title(title, fontsize=12)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(clabel, fontsize=9)

    fig.suptitle(
        f"Reducing the touch: every contacted vertex accumulates IFF across the "
        f"{int(cnt_c.max())} frames it is touched, then Σ ÷ count → {len(contacted)} "
        f"per-vertex mean values",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG / "03b_grouping_averaging.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gen ] 03b_grouping_averaging.png  ({len(contacted)} vertices, "
          f"max count {int(cnt_c.max())})")


def _fmt(v, unit=""):
    return f"{v:.3g}{unit}"


def generate_attributes(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    g = df[(df.block_order_id == BLOCK) & (df.trial_id == TRIAL)
           & (df.single_touch_id == TOUCH)].reset_index(drop=True)
    if g.empty:
        raise ValueError(
            f"generate_rf_single_touch_detail: touch ({BLOCK},{TRIAL},{TOUCH}) "
            f"not found in {csv_path}"
        )

    def m(col):
        return float(np.nanmean(pd.to_numeric(g[col], errors="coerce")))

    dur = float(g["time"].iloc[-1] - g["time"].iloc[0])
    signed_v = m("hand_velocity_signed")
    direction = "proximal (toward elbow)" if signed_v > 0 else "distal (toward wrist)"

    sections = [
        ("Identity", [
            ("session", SESSION),
            ("block · trial · touch", f"{BLOCK} · {TRIAL} · {TOUCH}"),
            ("gesture", f"{g['gesture_type'].iloc[0]}  ({g['gesture_broad_type'].iloc[0]})"),
            ("duration", f"{dur * 1e3:.0f} ms  ({len(g)} × 1 kHz frames)"),
        ]),
        ("Instructed stimulus (metadata)", [
            ("type", str(g["type_metadata"].iloc[0])),
            ("speed", _fmt(float(g["speed_metadata"].iloc[0]), " cm/s")),
            ("contactor", str(g["contact_area_metadata"].iloc[0])),
            ("force", str(g["force_metadata"].iloc[0])),
        ]),
        ("Measured kinematics", [
            ("stroke direction", direction),
            ("velocity |v|", _fmt(m("hand_velocity_amplitude"), " mm/s")),
            ("contact depth", _fmt(m("contact_depth"), " mm")),
            ("contact area", _fmt(m("contact_area"), " mm²")),
            ("pressure", _fmt(m("pressure"))),
        ]),
        ("Skin mechanics (MoS)", [
            ("strain", _fmt(m("mos_strain"))),
            ("stress", _fmt(m("mos_stress_kpa"), " kPa")),
            ("strain rate", _fmt(m("mos_strain_rate"), " /s")),
        ]),
        ("Neural response", [
            ("mean IFF", _fmt(m("Nerve_freq"), " Hz")),
            ("peak IFF", _fmt(float(np.nanmax(pd.to_numeric(g["Nerve_freq"], errors="coerce"))), " Hz")),
            ("spiking frames", f"{int(g['Nerve_spike'].astype(bool).sum())} / {len(g)}"),
        ]),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.axis("off")
    ax.set_title(f"Single-touch attributes — touch #166 "
                 f"({g['gesture_type'].iloc[0]})", fontsize=14, weight="bold", pad=14)
    y = 0.98
    for header, rows in sections:
        ax.text(0.02, y, header, fontsize=12, weight="bold", color="#1F2A44",
                transform=ax.transAxes, va="top")
        y -= 0.055
        for label, value in rows:
            ax.text(0.05, y, f"{label}", fontsize=11, color="#555555",
                    transform=ax.transAxes, va="top")
            ax.text(0.46, y, f"{value}", fontsize=11, color="#111111",
                    transform=ax.transAxes, va="top", family="monospace")
            y -= 0.042
        y -= 0.018

    out = FIG / "03c_attributes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[gen ] 03c_attributes.png")


def main() -> None:
    data_root = Path(get_project_data_root())
    analysed = _require(data_root / "4_analysed")
    csv_path = _require(analysed / "touch_compute_series" / f"{SESSION}_series_augmented.csv")
    ply = resolve_forearm_ply(data_root / "3_merged" / SESSION / "forearm_rf_centered", SESSION)
    if ply is None:
        raise FileNotFoundError("generate_rf_single_touch_detail: RF-centred PLY not found")

    # Vertex identity for every contact point is read off the contact-depth-field
    # parquet sidecars, joined on frame_index. This figure script names the stage
    # explicitly (it has no DAG config to read); the pipeline gets the same two values
    # from tasks.spatial_map_single_touch.options.contact_depth_field.
    pb = load_playback_data(
        csv_path,
        ply,
        depth_blocks_dir=_require(data_root / '3_merged' / SESSION / BLOCKS_STAGE_DIR),
        block_csv_stem_suffix=BLOCK_CSV_STEM_SUFFIX,
        session_id=SESSION,
    )
    forearm_xyz = pb.session_data.forearm_vertices
    events = pb.touches_by_block_trial[(str(BLOCK), TRIAL)]
    matches = [e for e in events if e.single_touch_id == TOUCH]
    if not matches:
        raise ValueError(
            f"generate_rf_single_touch_detail: touch ({BLOCK},{TRIAL},{TOUCH}) not in playback data"
        )
    ev = matches[0]

    try:
        R = load_rf_camera_rotation(analysed / "spatial_set_camera", SESSION)
    except ValueError as exc:
        print(f"[warn] no camera settings; using raw axes ({exc})")
        R = None

    FIG.mkdir(parents=True, exist_ok=True)
    generate_frames(ev, forearm_xyz, R)
    generate_grouping_averaging(ev, forearm_xyz, R)
    generate_attributes(csv_path)
    print(f"\nDone. Figures written to {FIG}")


if __name__ == "__main__":
    main()
