"""Per-(session, gesture) RF-contour parameters: schema I/O + grid helper.

Single source of truth shared by the tuner GUI (writer) and the
``spatial_extract_boundaries`` stage (reader) for the five tunable
contour-detection parameters. Each (session, gesture) combination persists one
JSON under::

    <database>/4_analysed/spatial_tune_rf_contours/iff_<metric>/
        <session>/<session>_<gesture>_contour_params.json

The five parameters live on :class:`GestureContourParams` (defined in
``rf_boundary_types`` so the boundary-stage DTO module owns every contract). All
I/O here is fail-fast: missing files, missing/unknown/mistyped keys, and an
invalid ``median_filter_size`` all raise rather than silently defaulting.

Four of the five parameters are also *toggleable* (see ``TOGGLEABLE_PARAM_KEYS``
and :class:`ContourParamToggles`): the JSON may carry an optional nested
``param_enabled`` object recording their enable/disable state. It is the one
*sanctioned* optional key — absent entirely, it resolves to the documented
default toggle state at this loader boundary (backward-compatible with a
flag-less JSON); present but malformed (wrong type, unknown sub-key, non-bool
flag), it still raises with file context.

The grid helper (``load_session_arrays`` / ``build_session_grid`` /
``load_session_grid``) is the production promotion of the sandbox
``contour_explorer.data_loader`` composition. It builds the interpolated
population-response heatmap grid from a boundary NPZ using only production
functions, so both the GUI preview and the pipeline consume the same math.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from analysis.pipeline.output_dirs import SPATIAL_TUNE_RF_CONTOURS
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    ContourParamToggles,
    GestureContourParams,
)
from analysis.receptive_field_mapping.data.rf_grid_interpolation import (
    compute_interpolated_grid,
)
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    clean_heatmap_islands,
    compute_threshold_from_ratio,
)

# The five tunable parameters, in canonical order. Kept as the authoritative
# key list so save/load and the JSON schema never drift.
PARAM_KEYS: tuple[str, ...] = (
    "min_overlap_pct",
    "median_filter_size",
    "radial_gauss_sigma",
    "radial_hess_sigma",
    "radial_envelope_smooth_sigma",
)

# The subset of PARAM_KEYS that can be individually enabled/disabled. Excludes
# radial_hess_sigma (the core Hessian-λmax detector scale, always applied).
# Kept as the authoritative key list so the toggle UI/JSON never drift from
# ContourParamToggles' fields.
TOGGLEABLE_PARAM_KEYS: tuple[str, ...] = (
    "min_overlap_pct",
    "median_filter_size",
    "radial_gauss_sigma",
    "radial_envelope_smooth_sigma",
)

# Metadata keys stored alongside the parameters (not part of the tuned identity).
_META_KEYS: tuple[str, ...] = ("session_id", "gesture", "created_at", "modified_at")

_REQUIRED_JSON_KEYS: tuple[str, ...] = PARAM_KEYS + _META_KEYS

# Optional top-level keys: sanctioned to be *absent* (backward-compat with a
# flag-less JSON, resolved to the default toggle state at the loader boundary
# below) but, when present, must still be well-formed (fail-fast).
_OPTIONAL_JSON_KEYS: tuple[str, ...] = ("param_enabled",)


# ---------------------------------------------------------------------------
# Path convention
# ---------------------------------------------------------------------------

def contour_params_root(database_path: Path, iff_metric: str) -> Path:
    """Return ``<database>/4_analysed/spatial_tune_rf_contours/iff_<metric>``.

    Mirrors the ``spatial_extract_boundaries`` layout so both stages resolve the
    same per-metric root.
    """
    if not iff_metric:
        raise ValueError("contour_params_root: iff_metric must be a non-empty string.")
    return (
        Path(database_path)
        / "4_analysed"
        / SPATIAL_TUNE_RF_CONTOURS
        / f"iff_{iff_metric}"
    )


def contour_params_path(root: Path, session_id: str, gtype: str) -> Path:
    """Return ``<root>/<session_id>/<session_id>_<gtype>_contour_params.json``."""
    if not session_id:
        raise ValueError("contour_params_path: session_id must be a non-empty string.")
    if not gtype:
        raise ValueError("contour_params_path: gtype must be a non-empty string.")
    return Path(root) / session_id / f"{session_id}_{gtype}_contour_params.json"


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_contour_params(
    path: Path,
    params: GestureContourParams,
    session_id: str,
    gesture: str,
    created_at: str = "",
) -> None:
    """Write one (session, gesture) parameter JSON to ``path``.

    Serialises the five tuned parameters, the ``param_enabled`` toggle state,
    plus ``session_id`` / ``gesture`` / ``created_at`` / ``modified_at``.
    Creates the parent directory, stamps ``modified_at`` to now (UTC), and
    stamps ``created_at`` to now when the passed value is empty (a fresh
    Validate). Round-trips through :func:`load_contour_params`.
    """
    if not isinstance(params, GestureContourParams):
        raise ValueError(
            f"save_contour_params: params must be GestureContourParams, got "
            f"{type(params).__name__}"
        )
    if not session_id:
        raise ValueError("save_contour_params: session_id must be non-empty.")
    if not gesture:
        raise ValueError("save_contour_params: gesture must be non-empty.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now = _utc_now_iso()
    doc = {
        "session_id": str(session_id),
        "gesture": str(gesture),
        "min_overlap_pct": float(params.min_overlap_pct),
        "median_filter_size": int(params.median_filter_size),
        "radial_gauss_sigma": float(params.radial_gauss_sigma),
        "radial_hess_sigma": float(params.radial_hess_sigma),
        "radial_envelope_smooth_sigma": float(params.radial_envelope_smooth_sigma),
        "param_enabled": params.toggles.to_dict(),
        "created_at": created_at if created_at else now,
        "modified_at": now,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_contour_params(path: Path) -> GestureContourParams:
    """Load one (session, gesture) parameter JSON into a frozen params object.

    Fail-fast on every malformed input (no silent defaults):
      * missing file -> ``FileNotFoundError``;
      * non-mapping root, missing required key, or unknown/extra key ->
        ``ValueError``;
      * a parameter of the wrong JSON type (incl. a float or bool where
        ``median_filter_size`` must be an int) -> ``ValueError``;
      * a non-positive or even ``median_filter_size`` -> ``ValueError``
        (also enforced by :class:`GestureContourParams`);
      * a present-but-malformed ``param_enabled`` (not a mapping, unknown
        sub-key, non-bool flag) -> ``ValueError``.

    The optional ``param_enabled`` key is the single sanctioned exception to
    "no silent fallbacks": when *absent entirely*, it resolves to the
    documented default :class:`ContourParamToggles` state at this loader
    boundary (backward-compat with a flag-less JSON) — not a scattered inline
    default, but one resolution point for every caller.

    Metadata keys (``session_id`` etc.) are validated for presence but not
    returned — the object carries only the five tuned parameters (plus the
    resolved toggle state).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contour params JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Contour params root must be a JSON object, got "
            f"{type(data).__name__} in {path}"
        )

    missing = [k for k in _REQUIRED_JSON_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Contour params JSON is missing required key(s) {missing} in {path}"
        )
    extra = [
        k for k in data if k not in _REQUIRED_JSON_KEYS and k not in _OPTIONAL_JSON_KEYS
    ]
    if extra:
        raise ValueError(
            f"Contour params JSON contains unknown key(s) {sorted(extra)} in "
            f"{path}. Valid keys: {list(_REQUIRED_JSON_KEYS)} "
            f"(optional: {list(_OPTIONAL_JSON_KEYS)})"
        )

    # median_filter_size must be a genuine int in the JSON (json.load keeps
    # 5 as int, 5.0 as float, true as bool). Reject anything else loudly here so
    # the message names the file; the dataclass re-checks odd/positivity.
    raw_mfs = data["median_filter_size"]
    if isinstance(raw_mfs, bool) or not isinstance(raw_mfs, int):
        raise ValueError(
            f"median_filter_size must be an integer, got {raw_mfs!r} in {path}"
        )

    for name in (
        "min_overlap_pct",
        "radial_gauss_sigma",
        "radial_hess_sigma",
        "radial_envelope_smooth_sigma",
    ):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a real number, got {value!r} in {path}")

    # param_enabled is the one sanctioned-optional key: absent -> resolved to
    # the default ContourParamToggles state below; present but not a mapping
    # -> raise loudly here (with file context) before handing off to
    # ContourParamToggles.from_dict, which only knows about sub-key shape.
    raw_toggles = data.get("param_enabled", {})
    if not isinstance(raw_toggles, dict):
        raise ValueError(
            f"param_enabled must be a JSON object, got "
            f"{type(raw_toggles).__name__} in {path}"
        )
    try:
        toggles = ContourParamToggles.from_dict(raw_toggles)
    except ValueError as exc:
        raise ValueError(f"{exc} (in {path})") from exc

    try:
        return GestureContourParams(
            min_overlap_pct=float(data["min_overlap_pct"]),
            median_filter_size=int(raw_mfs),
            radial_gauss_sigma=float(data["radial_gauss_sigma"]),
            radial_hess_sigma=float(data["radial_hess_sigma"]),
            radial_envelope_smooth_sigma=float(data["radial_envelope_smooth_sigma"]),
            toggles=toggles,
        )
    except ValueError as exc:
        raise ValueError(f"{exc} (in {path})") from exc


def defaults_from_boundary_params(params: BoundaryParams) -> GestureContourParams:
    """Seed a :class:`GestureContourParams` from the global boundary defaults.

    Used to bootstrap the GUI (and the ``strict=False`` consume path) from the
    DAG-level scalars. Raises if the global ``median_filter_size`` is ``None``
    (the tuner needs a concrete positive odd window to seed from) — a loud
    prerequisite, not a silent fallback. Toggles are seeded to the default
    :class:`ContourParamToggles` state (median OFF, the rest ON).
    """
    if params.median_filter_size is None:
        raise ValueError(
            "defaults_from_boundary_params: global BoundaryParams.median_filter_size "
            "is None; set a positive odd median_filter_size in the DAG config before "
            "bootstrapping per-gesture contour params."
        )
    return GestureContourParams(
        min_overlap_pct=float(params.min_overlap_pct),
        median_filter_size=int(params.median_filter_size),
        radial_gauss_sigma=float(params.radial_gauss_sigma),
        radial_hess_sigma=float(params.radial_hess_sigma),
        radial_envelope_smooth_sigma=float(params.radial_envelope_smooth_sigma),
        toggles=ContourParamToggles(),
    )


def load_gesture_contour_params(
    root: Path,
    session_id: str,
    gesture_keys: Iterable[str],
    params: BoundaryParams,
    strict: bool,
) -> dict[str, GestureContourParams]:
    """Load per-gesture contour params for one session, keyed by gesture type.

    Iterates the supplied ``gesture_keys`` (the session's actual gesture subsets,
    e.g. from the NPZ ``gesture_types`` — includes ``'all'`` and the synthetic
    ``'stroke'`` when present). For each key resolves
    ``<root>/<session>/<session>_<gesture>_contour_params.json`` and:

    * if it exists, loads it via :func:`load_contour_params`;
    * if it is absent and ``strict`` is ``True``, raises ``ValueError`` naming
      the missing (session, gesture) combination;
    * if it is absent and ``strict`` is ``False``, seeds it from
      :func:`defaults_from_boundary_params` (the explicit bootstrap path, gated
      by the caller's flag — not a silent fallback).

    ``'stroke'`` is synthetic (``stroke_proximal + stroke_distal``) but is
    treated identically to any other key: it gets its own JSON.
    """
    keys = list(gesture_keys)
    if not keys:
        raise ValueError(
            f"load_gesture_contour_params: no gesture_keys for session "
            f"'{session_id}'."
        )

    out: dict[str, GestureContourParams] = {}
    for gtype in keys:
        json_path = contour_params_path(root, session_id, gtype)
        if json_path.exists():
            out[gtype] = load_contour_params(json_path)
        elif strict:
            raise ValueError(
                f"load_gesture_contour_params: missing tuned contour params for "
                f"(session={session_id!r}, gesture={gtype!r}). Expected: {json_path}"
            )
        else:
            out[gtype] = defaults_from_boundary_params(params)
    return out


# ---------------------------------------------------------------------------
# Grid helper (production promotion of contour_explorer.data_loader)
# ---------------------------------------------------------------------------

def load_session_arrays(npz_path: Path, gesture: str) -> dict:
    """Read the raw per-session arrays needed to build a heatmap grid.

    The disk-bound, parameter-free part of session loading: cache the returned
    dict and feed it to :func:`build_session_grid` to re-threshold or
    re-interpolate without re-reading the NPZ. Raises (fail-fast) on a missing
    file, an unknown gesture, or a missing NPZ key.
    """
    npz_path = Path(npz_path)
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


def build_session_grid(
    arrays: dict,
    min_overlap_pct: float,
    median_filter_size: int | None,
    clean_islands: bool = False,
):
    """Threshold and interpolate cached session arrays into a heatmap grid.

    Mirrors the ``spatial_extract_boundaries`` pipeline: vertex threshold from
    ``min_overlap_pct`` -> interpolation -> optional island cleaning (keep only
    the peak's connected component). Composes only production functions so the
    GUI preview and the pipeline produce identical grids. Returns
    ``(grid_u, grid_v, grid_z, title, n_touches)``.
    """
    n_touches = arrays["n_touches"]
    threshold = compute_threshold_from_ratio(min_overlap_pct, n_touches)
    slim_heatmap = apply_vertex_threshold(
        arrays["heatmap_pre_threshold"], arrays["unique_count"], threshold
    )
    grid_u, grid_v, grid_z = compute_interpolated_grid(
        arrays["forearm_uv"],
        arrays["forearm_faces"],
        arrays["forearm_V"],
        slim_heatmap,
        median_filter_size=median_filter_size,
    )
    if clean_islands:
        grid_z = clean_heatmap_islands(grid_z)
    title = f"{arrays['session_id']}  ·  {arrays['gesture']}"
    return grid_u, grid_v, grid_z, title, n_touches


def load_session_grid(
    npz_path: Path,
    gesture: str,
    min_overlap_pct: float,
    median_filter_size: int | None,
):
    """Load one session NPZ and build its interpolated heatmap grid.

    Convenience composition of :func:`load_session_arrays` and
    :func:`build_session_grid`. Returns
    ``(grid_u, grid_v, grid_z, title, n_touches)``. Raises (fail-fast) on a
    missing file, an unknown gesture, or a missing NPZ key.
    """
    arrays = load_session_arrays(npz_path, gesture)
    return build_session_grid(arrays, min_overlap_pct, median_filter_size)
