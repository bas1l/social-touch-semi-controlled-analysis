"""Dataclass contracts exchanged between the boundary-extraction layers.

The ``spatial_extract_boundaries`` stage flows as:

    verify -> load (BoundaryInputs) -> prepare (PreparedBoundaryData)
          -> process (BoundaryResults) -> render / persist

Each dataclass is the hand-off between two adjacent layers. Keeping them in one
module (with no heavy imports of their own) avoids import cycles between the
``data`` / ``metrics`` / ``rendering`` sub-packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:  # avoid importing heavy loaders at module import time
    from analysis.receptive_field_mapping.data.touch_population_data import (
        PopulationData,
        PopulationRFData,
    )


@dataclass
class BoundaryParams:
    """All per-run options for the boundary-extraction stage (from the DAG)."""

    neuron_mode: str
    min_overlap_pct: float = 25.0
    median_filter_size: int | None = None
    inflection_sigma: float | None = None
    heatmap_space: str = "linear"
    cmap: str = "inferno"
    iff_metric: str = "mean"
    flip_u: bool = False
    contour_color: str = "red"
    circular_crop_margin: float = 0.0
    boundary_method: str = "gradient"
    radial_gauss_sigma: float = 8.0
    radial_hess_sigma: float = 5.0
    radial_envelope_smooth_sigma: float = 1.5


@dataclass
class BoundaryInputs:
    """Raw loaded per-session data plus the input paths used for the staleness gate."""

    session_id: str
    session_output_dir: Path
    sentinel: Path
    pop_data: "PopulationData"
    rf_data: "PopulationRFData"
    forearm_uv: np.ndarray
    slim_V: np.ndarray
    slim_faces: np.ndarray
    raw_vertex_colors: np.ndarray | None   # per-PLY-vertex RGB uint8; mapped to SLIM in prep
    forearm_ply_path: Path
    input_paths: List[Path]


@dataclass
class PreparedBoundaryData:
    """Everything derived from the raw inputs *before* boundary detection."""

    session_id: str
    output_dir: Path
    sentinel: Path
    forearm_uv: np.ndarray                # aligned (+ optional U-flip)
    slim_faces: np.ndarray
    slim_V: np.ndarray
    results: dict                         # {gtype: (slim_heatmap, n_touches, threshold)}
    per_gesture_grids: dict               # {gtype: (grid_u, grid_v, grid_z)}
    per_gesture_slim_raw: dict            # {gtype: heatmap[nearest_orig_for_slim]}
    per_gesture_slim_unique_count: dict   # {gtype: unique_count[nearest_orig_for_slim]}
    session_vmax: float
    session_vmin: float
    min_overlap_pct: float
    alignment_center: np.ndarray
    alignment_rotation_matrix: np.ndarray
    alignment_angle_deg: float
    flip_u: bool
    slim_vertex_colors: np.ndarray | None
    forearm_ply_path: Path | None = None


@dataclass
class BoundaryResults:
    """Boundary detector outputs for one session, carried forward to render + persist.

    ``gesture_boundaries`` holds the *active* boundary (selected by
    ``boundary_method``). The per-method dicts and derived fields
    (``per_gesture_smoothed`` etc.) are carried so persistence never has to
    recompute them.
    """

    boundary_method: str
    gesture_boundaries: dict = field(default_factory=dict)
    gesture_inflection_boundaries: dict = field(default_factory=dict)
    gesture_gradient_boundaries: dict = field(default_factory=dict)
    gesture_radial_boundaries: dict = field(default_factory=dict)
    per_gesture_smoothed: dict = field(default_factory=dict)      # {gtype: smoothed array}
    per_gesture_laplacian: dict = field(default_factory=dict)     # {gtype: laplacian array}
    per_gesture_gradient_mag: dict = field(default_factory=dict)  # {gtype: |grad| array}
