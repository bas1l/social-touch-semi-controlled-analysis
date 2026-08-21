"""Data loader for receptive field mapping.

Joins the unified touch summary CSV with per-trial raw CSVs to produce
grouped spatial data (spike counts and total counts per 3D contact point),
partitioned by user-chosen grouping variables (e.g. touch type, direction,
depth bin).
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d  # type: ignore
import pandas as pd
from scipy.spatial import KDTree

from analysis.touch_analytics.touch_config import DISCRETIZATION_CONFIG

from analysis.receptive_field_mapping.config import GroupedSpatialData, RFMappingConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PLY path resolution
# ------------------------------------------------------------------

def load_forearm_vertices(ply_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load forearm point-cloud vertices with a .npy sidecar cache.

    Cache path: ``{ply_path.parent}/{ply_path.stem}_vertices.npy``.
    Returns ``None`` when *ply_path* is ``None`` or the file does not exist.
    On cache hit (sidecar mtime ≥ PLY mtime) loads the .npy directly.
    On cache miss, loads via open3d, saves the sidecar, then returns the array.
    Raises ``ValueError`` for a cached .npy with the wrong shape.
    """
    if ply_path is None or not ply_path.exists():
        return None

    npy_path = ply_path.parent / f"{ply_path.stem}_vertices.npy"

    if npy_path.exists() and npy_path.stat().st_mtime >= ply_path.stat().st_mtime:
        arr = np.load(npy_path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"load_forearm_vertices: cached .npy has unexpected shape {arr.shape} "
                f"(expected (N, 3)): {npy_path}"
            )
        return arr

    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None

    try:
        np.save(npy_path, pts.astype(np.float64))
    except Exception:
        logger.warning("Could not save forearm vertices cache: %s", npy_path)

    return pts


def load_forearm_vertex_colors(ply_path: Optional[Path]) -> Optional[np.ndarray]:
    """Load forearm PLY vertex colors with a .npy sidecar cache.

    Cache path: ``{ply_path.parent}/{ply_path.stem}_colors.npy``.
    Returns an (N, 3) uint8 RGB array, or ``None`` when the PLY has no
    colors or does not exist.
    """
    if ply_path is None or not ply_path.exists():
        return None

    npy_path = ply_path.parent / f"{ply_path.stem}_colors.npy"

    if npy_path.exists() and npy_path.stat().st_mtime >= ply_path.stat().st_mtime:
        arr = np.load(npy_path)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        logger.warning("Color cache has unexpected shape %s, reloading PLY.", arr.shape)

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if not pcd.has_colors():
        return None

    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    try:
        np.save(npy_path, colors)
    except Exception:
        logger.warning("Could not save forearm colors cache: %s", npy_path)

    return colors


def resolve_forearm_ply(session_dir: Path, session_id: str) -> Optional[Path]:
    """Resolve the forearm PLY path for a session in RF-centered coordinate space.

    The aggregated session CSV always contains contact points in RF-centered
    space (produced by ``blocks_rf_centered/``).  The forearm PLY must be in
    the same space to avoid misalignment.  Returns ``None`` if the
    RF-centered PLY does not exist.
    """
    rf_centered = session_dir / f'{session_id}_forearm.ply'
    if rf_centered.exists():
        return rf_centered

    return None


def _transfer_ply_colors(
    ply_vertices: np.ndarray,
    ply_colors_uint8: np.ndarray,
    mesh_vertices: np.ndarray,
) -> np.ndarray:
    """Map PLY RGB colours to mesh vertices via KDTree nearest-neighbour.

    Returns (N_mesh, 4) float64 RGBA in [0, 1] suitable for matplotlib. This is
    the single source of truth for the PLY→mesh colour transfer, reused by both
    :func:`load_forearm_vertex_rgba` and the SLIM precompute in
    ``surface.forearm_slim_uv``.
    """
    tree = KDTree(ply_vertices)
    _, indices = tree.query(mesh_vertices)
    rgb = ply_colors_uint8[indices].astype(np.float64) / 255.0
    return np.column_stack([rgb, np.ones(len(rgb), dtype=np.float64)])


def load_forearm_vertex_rgba(
    forearm_ply_path: Optional[Path],
    mesh_vertices: np.ndarray,
) -> np.ndarray:
    """Load forearm PLY vertex colours transferred onto SLIM mesh vertices.

    Composes forearm vertex load → vertex-colour load → nearest-neighbour
    transfer onto ``mesh_vertices`` (typically a SLIM cache's cleaned ``V``),
    returning per-vertex RGBA in [0, 1].

    Parameters
    ----------
    forearm_ply_path:
        Path to the session's forearm PLY (RF-centered space). Resolve it with
        :func:`resolve_forearm_ply` before calling.
    mesh_vertices:
        (M, 3) mesh vertices to colour (e.g. ``SlimUvCache.V``).

    Returns
    -------
    np.ndarray
        (M, 4) float64 RGBA in [0, 1], one row per mesh vertex.

    Raises
    ------
    FileNotFoundError
        If ``forearm_ply_path`` is ``None`` or does not exist — a coloured
        forearm requires the PLY (no silent grey fallback).
    ValueError
        If the PLY has no vertices (empty/unreadable) or carries no vertex
        colours.
    """
    if forearm_ply_path is None:
        raise FileNotFoundError(
            "load_forearm_vertex_rgba: forearm PLY path is None — a coloured "
            "forearm is required (resolve_forearm_ply returned None)."
        )
    forearm_ply_path = Path(forearm_ply_path)
    if not forearm_ply_path.exists():
        raise FileNotFoundError(
            f"load_forearm_vertex_rgba: forearm PLY does not exist: "
            f"{forearm_ply_path}"
        )

    raw_verts = load_forearm_vertices(forearm_ply_path)
    if raw_verts is None:
        raise ValueError(
            f"load_forearm_vertex_rgba: forearm PLY has no vertices (empty or "
            f"unreadable): {forearm_ply_path}"
        )

    ply_colors_u8 = load_forearm_vertex_colors(forearm_ply_path)
    if ply_colors_u8 is None:
        raise ValueError(
            f"load_forearm_vertex_rgba: forearm PLY has no vertex colours: "
            f"{forearm_ply_path}. A coloured forearm is required (no grey "
            "fallback)."
        )

    return _transfer_ply_colors(
        raw_verts, ply_colors_u8, np.asarray(mesh_vertices, dtype=np.float64)
    )


# ------------------------------------------------------------------
# Contact-point parsing (canonical copy)
# ------------------------------------------------------------------

def parse_contact_points(point_str: str) -> List[Tuple[float, float, float]]:
    """Parse a string representation of 3D points.

    Expected format: ``[[x1 y1 z1] [x2 y2 z2]]`` or ``"[]"``.
    Returns a list of ``(x, y, z)`` tuples.

    Every bracketed group **must** parse as exactly three floats. A group that
    does not raises ``ValueError`` — it is never skipped. Silently dropping a
    malformed triplet is the single most dangerous failure mode this parser has:
    the k-th point of a frame is joined to the k-th row of that frame in the
    contact-depth-field parquet sidecar (ordered correspondence, see
    ``docs/data-contracts/contact-depth-field.md``), so one dropped point shifts
    every later point of the frame onto the **wrong** sidecar row and produces a
    plausible-looking but wrong map. Raising here is the first line of defence;
    the per-frame count assertion against the sidecar is the second.
    """
    if pd.isna(point_str) or not isinstance(point_str, str) or point_str.strip() == "[]":
        return []

    points: List[Tuple[float, float, float]] = []
    matches = re.findall(r'\[([^\]]+)\]', point_str)

    for match in matches:
        parts = match.strip().lstrip("[").split()
        if len(parts) != 3:
            raise ValueError(
                f"parse_contact_points: bracketed group {match!r} has {len(parts)} "
                f"field(s), expected exactly 3 (x y z). Full cell: {point_str!r}. "
                f"A malformed triplet is never skipped — dropping it would shift "
                f"every later point of this frame onto the wrong depth-field row."
            )
        try:
            pt = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise ValueError(
                f"parse_contact_points: bracketed group {match!r} is not three "
                f"floats ({exc}). Full cell: {point_str!r}."
            ) from exc
        points.append(pt)

    return points


# ------------------------------------------------------------------
# Discretization helpers
# ------------------------------------------------------------------

def _discretize_column(
    series: pd.Series,
    q: int,
) -> pd.Series:
    """Discretize a continuous series into *q* quantile bins.

    Returns a Series of string labels.  Rows with NaN values are kept as
    ``"unknown"``.
    """
    try:
        binned = pd.qcut(series, q=q, duplicates="drop")
        return binned.astype(str).fillna("unknown")
    except (ValueError, TypeError):
        # All identical values or not enough distinct values for q bins.
        logger.warning(
            "Could not discretize column '%s' into %d bins; using 'all' as label.",
            series.name,
            q,
        )
        return pd.Series("all", index=series.index)


def _build_group_lookup(
    summary_df: pd.DataFrame,
    grouping_columns: List[str],
    trial_id_col: str,
    touch_id_col: str,
) -> Dict[Tuple, str]:
    """Build a ``(trial_id, single_touch_id) -> group_label`` lookup dict.

    Continuous columns are discretized according to
    ``DISCRETIZATION_CONFIG``; categorical columns are used as-is.
    """
    continuous_cfg = DISCRETIZATION_CONFIG.get("continuous_vars", {})
    categorical_vars = set(DISCRETIZATION_CONFIG.get("categorical_vars", []))

    # Validate that requested grouping columns exist in the summary
    available = set(summary_df.columns)
    valid_columns: List[str] = []
    for col in grouping_columns:
        if col not in available:
            logger.warning(
                "Grouping column '%s' not found in summary CSV (available: %s). "
                "Skipping this column.",
                col,
                ", ".join(sorted(available)),
            )
        else:
            valid_columns.append(col)

    if not valid_columns:
        logger.warning(
            "No valid grouping columns remain. All data will be assigned "
            "to a single group ('all')."
        )
        return {
            (row[trial_id_col], row[touch_id_col]): "all"
            for _, row in summary_df.iterrows()
        }

    # Prepare label series for each grouping column
    label_parts: Dict[str, pd.Series] = {}
    for col in valid_columns:
        if col in continuous_cfg:
            q = continuous_cfg[col].get("q", 3)
            label_parts[col] = _discretize_column(summary_df[col], q)
        elif col in categorical_vars:
            label_parts[col] = summary_df[col].astype(str).fillna("unknown")
        else:
            # Unknown column type: treat as categorical
            label_parts[col] = summary_df[col].astype(str).fillna("unknown")

    # Concatenate labels into a single group string per row
    combined_labels = pd.DataFrame(label_parts)
    group_strings = combined_labels.apply(lambda row: "_".join(row.values), axis=1)

    lookup: Dict[Tuple, str] = {}
    for idx, group_label in group_strings.items():
        trial_id = summary_df.at[idx, trial_id_col]
        touch_id = summary_df.at[idx, touch_id_col]
        lookup[(trial_id, touch_id)] = group_label

    return lookup


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_grouped_spatial_data(
    raw_csv_paths: List[Path],
    summary_csv_path: Path,
    grouping_columns: List[str],
    config: RFMappingConfig,
) -> Dict[str, GroupedSpatialData]:
    """Load and group spatial contact-point data for RF mapping.

    Parameters
    ----------
    raw_csv_paths:
        Paths to per-trial raw CSVs containing frame-level rows with
        ``single_touch_id``, ``Nerve_spike``, and ``contact_points``
        columns.
    summary_csv_path:
        Path to the unified touch summary CSV that contains one row per
        touch with the grouping metadata (e.g. ``type_metadata``,
        ``direction``, ``max_depth``).
    grouping_columns:
        Column names from the summary CSV to group touches by.
        Continuous columns are discretized via ``pd.qcut``; categorical
        columns are used as-is.
    config:
        ``RFMappingConfig`` providing column-name mappings.

    Returns
    -------
    dict mapping group label strings to ``GroupedSpatialData`` objects.
    """
    col = config.columns

    # --- Read summary CSV ---
    if not summary_csv_path.exists():
        logger.error("Summary CSV not found: %s", summary_csv_path)
        return {}

    try:
        summary_df = pd.read_csv(summary_csv_path)
    except Exception:
        logger.exception("Failed to read summary CSV: %s", summary_csv_path)
        return {}

    if summary_df.empty:
        logger.warning("Summary CSV is empty: %s", summary_csv_path)
        return {}

    # Ensure key columns exist in summary
    for required in (col.trial_id, col.touch_id):
        if required not in summary_df.columns:
            logger.error(
                "Required column '%s' missing from summary CSV.", required
            )
            return {}

    # --- Build group lookup ---
    group_lookup = _build_group_lookup(
        summary_df,
        grouping_columns,
        trial_id_col=col.trial_id,
        touch_id_col=col.touch_id,
    )

    # --- Initialize grouped data containers ---
    groups: Dict[str, GroupedSpatialData] = {}
    for label in set(group_lookup.values()):
        groups[label] = GroupedSpatialData(group_label=label)

    # Track unique (trial_id, touch_id) pairs seen per group for touch_count
    group_touch_sets: Dict[str, set] = {label: set() for label in groups}

    # --- Iterate raw CSVs ---
    for csv_path in raw_csv_paths:
        if not csv_path.exists():
            logger.warning("Raw CSV not found, skipping: %s", csv_path)
            continue

        try:
            points_col = col.points
            needed_cols = [col.trial_id, col.touch_id, col.spike, points_col]
            # Only read columns that actually exist (trial_id may be absent
            # in very old CSVs — handle gracefully)
            available_cols = set(pd.read_csv(csv_path, nrows=0).columns)
            missing = [c for c in needed_cols if c not in available_cols]
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s",
                    csv_path.name,
                    missing,
                )
                continue

            df = pd.read_csv(csv_path, usecols=needed_cols)
        except KeyError as exc:
            logger.warning(
                "Skipping %s: column resolution failed (%s)", csv_path.name, exc
            )
            continue
        except Exception:
            logger.exception("Error reading %s", csv_path.name)
            continue

        if df.empty:
            continue

        trial_ids = df[col.trial_id].to_numpy()
        touch_ids = df[col.touch_id].to_numpy()
        spikes = df[col.spike].to_numpy()
        raw_points = df[points_col]

        for idx in range(len(df)):
            # Skip background / noise rows
            if touch_ids[idx] == 0:
                continue

            parsed = parse_contact_points(raw_points.iat[idx])
            if not parsed:
                continue

            key = (trial_ids[idx], touch_ids[idx])
            group_label = group_lookup.get(key)
            if group_label is None:
                # Touch not present in summary — skip silently
                continue

            gsd = groups[group_label]
            gsd.total_counts.update(parsed)

            if spikes[idx] == 1:
                gsd.spike_counts.update(parsed)

            group_touch_sets[group_label].add(key)

    # --- Finalise touch counts ---
    for label, touch_set in group_touch_sets.items():
        groups[label].touch_count = len(touch_set)

    # Log summary
    total_touches = sum(g.touch_count for g in groups.values())
    total_points = sum(len(g.total_counts) for g in groups.values())
    logger.info(
        "Loaded %d groups, %d total unique touches, %d total unique spatial points.",
        len(groups),
        total_touches,
        total_points,
    )

    return groups
