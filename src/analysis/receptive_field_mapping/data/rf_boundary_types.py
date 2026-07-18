"""Dataclass contracts exchanged between the boundary-extraction layers.

The ``spatial_extract_boundaries`` stage flows as:

    verify -> load (BoundaryInputs) -> prepare (PreparedBoundaryData)
          -> process (BoundaryResults) -> render / persist

Each dataclass is the hand-off between two adjacent layers. Keeping them in one
module (with no heavy imports of their own) avoids import cycles between the
``data`` / ``metrics`` / ``rendering`` sub-packages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
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
    # Plateau-detection gate defaults for the radial foot detector. ``None`` /
    # ``1`` reproduce the pre-feature hardcoded behaviour (no prominence gate;
    # ``find_peaks`` plateau_size floor of 1). Seeded into per-gesture
    # GestureContourParams by ``defaults_from_boundary_params``.
    radial_prominence: float | None = None
    radial_plateau_size: int = 1


@dataclass(frozen=True)
class ContourParamToggles:
    """Enable/disable flags for the four *toggleable* contour-detection steps.

    Mirrors the ``SlimUvCleanSteps`` precedent (``surface/slim_uv_config_io.py``):
    a frozen map of named booleans, persisted alongside the values they gate.
    ``radial_hess_sigma`` has **no** flag here — it is the core Hessian-λmax
    detector scale and is always applied.

    This dataclass knows only ON/OFF state: it does not know about JSON file
    paths, GUI widgets, or the concrete "skip" value each disabled step
    resolves to (that resolution lives on :class:`GestureContourParams`'s
    ``effective_*()`` methods). Frozen so a loaded toggle set cannot be mutated
    in place as it threads through the GUI and pipeline layers.

    Defaults (a combination with no saved flags): ``min_overlap_pct`` and the
    two ``radial_*`` toggles start **ON**; ``median_filter_size`` and
    ``prominence`` start **OFF** (explicit user decisions — the median filter is
    opt-in, and the pre-feature detector imposed no prominence gate, i.e.
    ``prominence=None``).
    """

    min_overlap_pct: bool = True
    median_filter_size: bool = False
    radial_gauss_sigma: bool = True
    radial_envelope_smooth_sigma: bool = True
    prominence: bool = False

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, bool):
                raise ValueError(
                    f"ContourParamToggles.{f.name} must be a bool, got {value!r}"
                )

    def to_dict(self) -> dict[str, bool]:
        """Return the flags as a plain ``{name: bool}`` dict (JSON-ready)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ContourParamToggles":
        """Build toggles from a mapping, defaulting any absent field.

        Backward-compatible with a flag-less JSON: each field is read via
        ``data.get(name, <field default>)``, so an empty (or partially
        populated) mapping resolves to the documented default state. Any
        *unknown* sub-key still raises loudly (fail-fast on malformed input;
        only absence of a *known* key is treated as "not yet saved").
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"param_enabled must be a mapping, got {type(data).__name__}"
            )
        expected = {f.name: f.default for f in fields(cls)}
        extra = set(data.keys()) - set(expected.keys())
        if extra:
            raise ValueError(
                f"param_enabled contains unknown key(s): {sorted(extra)}. "
                f"Valid keys: {sorted(expected.keys())}"
            )
        return cls(
            **{name: data.get(name, default) for name, default in expected.items()}
        )


@dataclass(frozen=True)
class GestureContourParams:
    """The five per-(session, gesture) tunable contour-detection parameters.

    These override the global scalars on :class:`BoundaryParams` for one gesture
    subset when ``use_tuned_params`` is on. The tuner GUI writes one JSON per
    (session, gesture); the boundary stage reads them back into this frozen
    contract. Immutable so a loaded parameter set cannot be mutated in place as
    it is threaded through the prepare/process layers.

    ``median_filter_size`` is the side of a square median window and must be a
    positive odd integer. The three ``radial_*`` sigmas feed the Gaussian ->
    Hessian-λmax -> envelope chain of the ``radial`` foot-of-mountain detector.

    ``prominence`` and ``plateau_size`` tune the per-ray plateau-detection gate
    (``scipy.signal.find_peaks``) inside the radial detector. ``prominence`` is
    optional (``None`` = no prominence requirement) and *toggleable*: when its
    toggle is off it resolves to ``None`` regardless of the stored value.
    ``plateau_size`` is a plain minimum-plateau length (``>= 1``); ``1`` is the
    natural "off" (no plateau-length constraint), so it carries no toggle.

    ``toggles`` carries the enable/disable state for the *toggleable* parameters
    (see :class:`ContourParamToggles`); it does not change the validation of the
    value fields above, which stay valid numbers even when their toggle is off.
    The ``effective_*()`` methods below are the *single* place a disabled step is
    resolved to its concrete "skip" value — both the GUI preview and the pipeline
    call these rather than re-deriving the skip decision.
    """

    min_overlap_pct: float
    median_filter_size: int
    radial_gauss_sigma: float
    radial_hess_sigma: float
    radial_envelope_smooth_sigma: float
    prominence: float | None = None
    plateau_size: int = 1
    toggles: ContourParamToggles = field(default_factory=ContourParamToggles)

    def __post_init__(self) -> None:
        # min_overlap_pct is a percentage of overlapping touches -> [0, 100].
        if isinstance(self.min_overlap_pct, bool) or not isinstance(
            self.min_overlap_pct, (int, float)
        ):
            raise ValueError(
                f"min_overlap_pct must be a real number, got "
                f"{self.min_overlap_pct!r}"
            )
        if not (0.0 <= float(self.min_overlap_pct) <= 100.0):
            raise ValueError(
                f"min_overlap_pct must be in [0, 100], got {self.min_overlap_pct}"
            )

        # median_filter_size is a discrete window side: positive odd int only.
        if isinstance(self.median_filter_size, bool) or not isinstance(
            self.median_filter_size, int
        ):
            raise ValueError(
                f"median_filter_size must be an int, got "
                f"{self.median_filter_size!r}"
            )
        if self.median_filter_size < 1 or self.median_filter_size % 2 == 0:
            raise ValueError(
                f"median_filter_size must be a positive odd integer, got "
                f"{self.median_filter_size}"
            )

        # gauss/hess sigmas drive Gaussian smoothing and Hessian scale: > 0.
        for name in ("radial_gauss_sigma", "radial_hess_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number, got {value!r}")
            if not float(value) > 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")

        # envelope smoothing may be 0 (means "no envelope smoothing").
        env = self.radial_envelope_smooth_sigma
        if isinstance(env, bool) or not isinstance(env, (int, float)):
            raise ValueError(
                f"radial_envelope_smooth_sigma must be a real number, got {env!r}"
            )
        if not float(env) >= 0.0:
            raise ValueError(
                f"radial_envelope_smooth_sigma must be >= 0, got {env}"
            )

        # plateau_size is a discrete find_peaks gate: genuine int >= 1 (1 = off).
        if isinstance(self.plateau_size, bool) or not isinstance(
            self.plateau_size, int
        ):
            raise ValueError(
                f"plateau_size must be an int, got {self.plateau_size!r}"
            )
        if self.plateau_size < 1:
            raise ValueError(
                f"plateau_size must be >= 1, got {self.plateau_size}"
            )

        # prominence is optional (None = no prominence gate); when set it must
        # be a positive real number.
        if self.prominence is not None:
            if isinstance(self.prominence, bool) or not isinstance(
                self.prominence, (int, float)
            ):
                raise ValueError(
                    f"prominence must be a real number or None, got "
                    f"{self.prominence!r}"
                )
            if not float(self.prominence) > 0.0:
                raise ValueError(
                    f"prominence must be > 0 when set, got {self.prominence}"
                )

        if not isinstance(self.toggles, ContourParamToggles):
            raise ValueError(
                f"toggles must be a ContourParamToggles, got "
                f"{type(self.toggles).__name__}"
            )

    def effective_min_overlap_pct(self) -> float:
        """Return the min-overlap threshold, or ``0.0`` (weakest) when disabled.

        ``compute_threshold_from_ratio(0.0, n)`` floors to threshold ``1``, so
        every contacted vertex is kept — the "no thresholding" skip state.
        """
        return float(self.min_overlap_pct) if self.toggles.min_overlap_pct else 0.0

    def effective_median_filter_size(self) -> int | None:
        """Return the median-filter window, or ``None`` (skip) when disabled.

        ``compute_interpolated_grid(..., median_filter_size=None)`` skips the
        median-filter step entirely.
        """
        if self.toggles.median_filter_size:
            return int(self.median_filter_size)
        return None

    def effective_gauss_sigma(self) -> float:
        """Return the pre-smoothing Gaussian sigma, or ``0.0`` when disabled.

        ``scipy.ndimage.gaussian_filter(sigma=0.0)`` is the identity, so the
        Hessian detector then runs on the raw normalised field (no
        pre-smoothing).
        """
        return float(self.radial_gauss_sigma) if self.toggles.radial_gauss_sigma else 0.0

    def effective_envelope_smooth_sigma(self) -> float:
        """Return the envelope-smoothing sigma, or ``0.0`` when disabled.

        ``0.0`` is the documented no-op sentinel for ``_smooth_closed_contour``
        (the contour is returned unchanged).
        """
        if self.toggles.radial_envelope_smooth_sigma:
            return float(self.radial_envelope_smooth_sigma)
        return 0.0

    def effective_prominence(self) -> float | None:
        """Return the plateau-detection prominence, or ``None`` when disabled.

        ``None`` is the pre-feature default: no prominence requirement is passed
        to ``scipy.signal.find_peaks``. When the ``prominence`` toggle is off (or
        the stored value is ``None``), the detector runs without a prominence
        gate. This is the *single* place that decision is resolved.
        """
        if self.toggles.prominence and self.prominence is not None:
            return float(self.prominence)
        return None

    def effective_plateau_size(self) -> int:
        """Return the minimum plateau length (``>= 1``); ``1`` is the "off" state.

        ``plateau_size`` is not toggleable — ``1`` (the natural no-op) already
        imposes no plateau-length constraint on ``find_peaks``.
        """
        return int(self.plateau_size)


class ContourFailureBranch(Enum):
    """The three ring-present failure modes of the radial-foot contour detector.

    Each value names exactly one site inside ``rf_radial_foot_boundary`` where a
    contour cannot be produced *despite* a visible λmax curvature ring. The branch
    is set at the raising site (never re-inferred downstream), so it is a stable,
    explicit contract the GUI maps to plain-language cause + which-knob advice.

    - ``NO_RADIAL_PLATEAU`` — no ray produced a qualifying λmax plateau
      (``contour_unsnapped_rc`` never produced; ``found.sum() == 0``).
    - ``SEED_OUTSIDE_FOOTPRINT`` — a contour was traced but the seed cell is not
      painted in the footprint (``footprint[peak] is False``).
    - ``NO_ENCLOSING_CONTOUR`` — contour(s) traced but none enclose the peak.
    """

    NO_RADIAL_PLATEAU = "no_radial_plateau"
    SEED_OUTSIDE_FOOTPRINT = "seed_outside_footprint"
    NO_ENCLOSING_CONTOUR = "no_enclosing_contour"


@dataclass(frozen=True)
class ContourFailureDiagnostics:
    """Structured, presentation-free numbers behind one contour-extraction failure.

    The compute layer emits this DATA (not a formatted sentence); the GUI owns the
    branch → wording map. Optional fields stay ``None`` when the branch did not
    measure them, so "zero" (a measured count of 0) and "absent" (not applicable
    to this branch) stay distinct in the payload. Each field surfaces the exact
    quantity the failing branch tested on.

    Fields
    ------
    branch:
        Which :class:`ContourFailureBranch` was hit (set at the raising site).
    n_angles, found_count, peak_lmax:
        Site 3 (``NO_RADIAL_PLATEAU``): the number of rays cast, how many formed a
        qualifying plateau (``0`` on this branch), and the peak λmax
        (``nanmax``; ``> 0`` proves a curvature ridge is present).
    footprint_at_seed, footprint_cells:
        Site 5 (``SEED_OUTSIDE_FOOTPRINT``): whether the seed cell is painted in
        the ``grid_z > 0`` footprint, and how many cells the footprint contains.
    n_candidate_contours:
        Site 4 (``NO_ENCLOSING_CONTOUR``): how many candidate contours were traced
        (none of which enclosed the peak).
    """

    branch: ContourFailureBranch
    n_angles: int | None = None
    found_count: int | None = None
    peak_lmax: float | None = None
    footprint_at_seed: bool | None = None
    footprint_cells: int | None = None
    n_candidate_contours: int | None = None


class ContourExtractionError(ValueError):
    """Raised at a radial-foot contour-extraction failure, carrying diagnostics.

    Subclasses :class:`ValueError` so the existing ``except ValueError`` in
    ``compute_radial_foot_stages`` (and the public orchestrator) still catches it
    unchanged. The ``diagnostics`` payload lets the ``allow_partial`` path surface
    structured numbers without the compute layer formatting any human wording.
    """

    def __init__(self, message: str, diagnostics: ContourFailureDiagnostics) -> None:
        if not isinstance(diagnostics, ContourFailureDiagnostics):
            raise ValueError(
                f"ContourExtractionError: diagnostics must be a "
                f"ContourFailureDiagnostics, got {type(diagnostics).__name__}"
            )
        super().__init__(message)
        self.diagnostics = diagnostics


def _validate_uv_endpoint(name: str, value) -> tuple[float, float]:
    """Validate a ``(u, v)`` endpoint is a length-2 sequence of finite reals.

    Accepts a tuple/list/ndarray of two numbers; rejects wrong lengths, bools,
    non-numeric or non-finite components (fail-fast, no silent coercion).
    Returns the normalised ``(float, float)`` tuple.
    """
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (tuple, list)):
        seq = list(value)
    else:
        raise ValueError(
            f"StrokeAxisConfig: {name} must be a length-2 (u, v) sequence, got "
            f"{type(value).__name__}"
        )
    if len(seq) != 2:
        raise ValueError(
            f"StrokeAxisConfig: {name} must have exactly 2 elements (u, v), got "
            f"{len(seq)}"
        )
    out: list[float] = []
    for i, comp in enumerate(seq):
        if isinstance(comp, bool) or not isinstance(comp, (int, float)):
            raise ValueError(
                f"StrokeAxisConfig: {name}[{i}] must be a real number, got {comp!r}"
            )
        if not np.isfinite(float(comp)):
            raise ValueError(
                f"StrokeAxisConfig: {name}[{i}] must be finite, got {comp!r}"
            )
        out.append(float(comp))
    return (out[0], out[1])


@dataclass(frozen=True)
class StrokeAxisConfig:
    """Manual stroke-axis definition for one session (persisted as JSON).

    The researcher draws a two-point line in SLIM-UV space; its direction
    (``axis_end_uv - axis_start_uv``) is the axis along which each stroke's UV
    displacement is projected to assign proximal vs distal. ``axis_end_uv`` is
    the proximal-pointing end: a **positive** projection onto
    :meth:`signed_direction` is ``'stroke_proximal'``, a non-positive one is
    ``'stroke_distal'`` — the 2D analogue of the 3D ``classify_gesture_type``
    ``slope > 0`` rule. ``swap_proximal_distal`` flips that assignment without
    redrawing the line.

    Frozen so a loaded axis cannot be mutated in place as it threads through the
    GUI preview and the response-fields consumer (single source of truth). The
    ``__post_init__`` is fail-fast: non-empty ``session_id``, length-2 finite
    endpoints, a genuine ``bool`` swap flag, and a non-degenerate axis (the two
    endpoints must differ).
    """

    session_id: str
    axis_start_uv: tuple[float, float]
    axis_end_uv: tuple[float, float]
    swap_proximal_distal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError(
                f"StrokeAxisConfig: session_id must be a non-empty string, got "
                f"{self.session_id!r}"
            )
        start = _validate_uv_endpoint("axis_start_uv", self.axis_start_uv)
        end = _validate_uv_endpoint("axis_end_uv", self.axis_end_uv)
        if not isinstance(self.swap_proximal_distal, bool):
            raise ValueError(
                f"StrokeAxisConfig: swap_proximal_distal must be a bool, got "
                f"{self.swap_proximal_distal!r}"
            )
        if start == end:
            raise ValueError(
                f"StrokeAxisConfig: degenerate axis — axis_start_uv and axis_end_uv "
                f"are identical ({start}). The endpoints must differ to define a "
                f"direction."
            )
        # Normalise the endpoints to plain float tuples (frozen -> setattr bypass).
        object.__setattr__(self, "axis_start_uv", start)
        object.__setattr__(self, "axis_end_uv", end)

    def direction(self) -> np.ndarray:
        """Return the unit UV vector from ``axis_start_uv`` to ``axis_end_uv``."""
        d = np.asarray(self.axis_end_uv, dtype=np.float64) - np.asarray(
            self.axis_start_uv, dtype=np.float64
        )
        norm = float(np.linalg.norm(d))
        if norm == 0.0:
            raise ValueError(
                "StrokeAxisConfig.direction: axis endpoints coincide (zero-length "
                "axis)."
            )
        return d / norm

    def signed_direction(self) -> np.ndarray:
        """Unit direction toward proximal, negated when ``swap_proximal_distal``.

        Projecting a stroke's UV motion vector onto this returns a positive value
        for ``'stroke_proximal'`` and a non-positive one for ``'stroke_distal'``.
        The swap flag lets the researcher invert the assignment without redrawing.
        """
        return self.direction() * (-1.0 if self.swap_proximal_distal else 1.0)


@dataclass
class StrokeUvMotion:
    """Per-stroke UV motion for one session (shared by the GUI and the consumer).

    Every array is aligned along axis 0 by stroke — one entry per single-touch
    ``type_metadata == 'stroke'`` group, in prepared-CSV group order (``S``
    strokes):

    * ``touch_keys``     — (S, 3) int64 ``(block_order_id, trial_id, single_touch_id)``
    * ``current_labels`` — (S,) object: the current 3D-derived ``gesture_type``
      (``'stroke_proximal'`` / ``'stroke_distal'`` / ``'stroke_unknown'``)
    * ``start_uv``       — (S, 2) float64: degree-1 UV fit evaluated at the first frame
    * ``end_uv``         — (S, 2) float64: degree-1 UV fit evaluated at the last frame
    * ``vectors``        — (S, 2) float64: ``end_uv - start_uv`` (net UV displacement)
    * ``valid_mask``     — (S,) bool: True when >= 2 valid (non-NaN) contact frames
    * ``centroid_uv``    — (S, 2) float64: mean UV over the stroke's valid frames

    Rows with ``valid_mask == False`` carry NaN UV entries (the ``stroke_unknown``
    short/all-NaN case, mirroring the 3D rule). Mutable because the GUI recolours
    the derived labels in place while the axis is dragged; the *persisted*
    contract is the frozen :class:`StrokeAxisConfig`.
    """

    touch_keys: np.ndarray
    current_labels: np.ndarray
    start_uv: np.ndarray
    end_uv: np.ndarray
    vectors: np.ndarray
    valid_mask: np.ndarray
    centroid_uv: np.ndarray


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
    # Per-(session, gesture) tuned contour params, keyed by gesture type, when
    # ``use_tuned_params`` is on; ``None`` when off (the stage then reads the
    # global BoundaryParams scalars unchanged). Populated by the pipeline before
    # ``prepare_session_boundary_data`` and read back in the extract layer.
    per_gesture_params: "dict[str, GestureContourParams] | None" = None


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
