"""Session NPZ loading: build the interpolated population-response heatmap grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    clean_heatmap_islands,
    compute_threshold_from_ratio,
)
from analysis.receptive_field_mapping.data.rf_grid_interpolation import (
    compute_interpolated_grid,
)


def load_session_arrays(npz_path: Path, gesture: str) -> dict:
    """Read the raw per-session arrays needed to build a heatmap grid.

    This is the disk-bound, parameter-free part of session loading; cache the
    returned dict and feed it to :func:`build_session_grid` to re-threshold or
    re-interpolate without re-reading the NPZ. Raises (fail-fast) on a missing
    file, an unknown gesture, or a missing NPZ key.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    available = list(d["gesture_types"])
    if gesture not in available:
        raise ValueError(f"gesture '{gesture}' not in NPZ. Available: {available}")

    def _get(stem: str):
        key = f"{stem}_{gesture}"
        if key not in d:
            raise KeyError(f"missing NPZ key '{key}'")
        return d[key]

    return {
        "gesture": gesture,
        "heatmap_pre_threshold": _get("heatmap_pre_threshold"),
        "unique_count": _get("unique_count"),
        "forearm_uv": d["forearm_uv"],
        "forearm_faces": d["forearm_faces"],
        "forearm_V": d["forearm_V"],
        "n_touches": int(d[f"n_touches_{gesture}"]),
        "session_id": str(d["session_id"]),
    }


def build_session_grid(arrays, min_overlap_pct, median_filter_size, clean_islands=False):
    """Threshold and interpolate cached session arrays into a heatmap grid.

    Mirrors the ``spatial_extract_boundaries`` pipeline: vertex threshold from
    ``min_overlap_pct`` → interpolation → optional island cleaning (keep only the
    peak's connected component). Returns (grid_u, grid_v, grid_z, title, n_touches).
    """
    n_touches = arrays["n_touches"]
    threshold = compute_threshold_from_ratio(min_overlap_pct, n_touches)
    slim_heatmap = apply_vertex_threshold(
        arrays["heatmap_pre_threshold"], arrays["unique_count"], threshold
    )
    grid_u, grid_v, grid_z = compute_interpolated_grid(
        arrays["forearm_uv"], arrays["forearm_faces"], arrays["forearm_V"],
        slim_heatmap, median_filter_size=median_filter_size,
    )
    if clean_islands:
        grid_z = clean_heatmap_islands(grid_z)
    title = f"{arrays['session_id']}  ·  {arrays['gesture']}"
    return grid_u, grid_v, grid_z, title, n_touches


def load_session_grid(npz_path: Path, gesture: str, min_overlap_pct, median_filter_size):
    """Load one session NPZ and build its interpolated heatmap grid.

    Returns (grid_u, grid_v, grid_z, title, n_touches). Raises (fail-fast) on a
    missing file, an unknown gesture, or a missing NPZ key.
    """
    arrays = load_session_arrays(npz_path, gesture)
    return build_session_grid(arrays, min_overlap_pct, median_filter_size)
