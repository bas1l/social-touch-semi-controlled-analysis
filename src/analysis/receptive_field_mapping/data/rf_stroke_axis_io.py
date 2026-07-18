"""Per-session manual stroke-axis: schema I/O + auto-init helper.

Single source of truth shared by the ``spatial_configure_stroke_axis`` GUI
(writer) and the ``spatial_build_response_fields`` consumer (reader) for the
manually drawn UV stroke axis. Each session persists one JSON under::

    <database>/4_analysed/spatial_configure_stroke_axis/
        <session>/<session>_stroke_axis.json

with the exact schema::

    { "session_id": "...", "axis_start_uv": [u0, v0], "axis_end_uv": [u1, v1],
      "swap_proximal_distal": false, "created_at": "...Z", "modified_at": "...Z" }

Endpoints (not an angle) are stored because the GUI artifact *is* a two-point
rubber-band line — direction is derived at projection time, avoiding drift. The
axis config lives on :class:`StrokeAxisConfig` (defined in ``rf_boundary_types``
so the boundary-stage DTO module owns every contract). All I/O here is fail-fast:
missing files, non-mapping roots, missing/unknown keys, mis-typed endpoints, a
non-bool swap flag, and a degenerate (start == end) axis all raise rather than
silently defaulting. Mirrors the conventions of ``rf_contour_params_io``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.pipeline.output_dirs import SPATIAL_CONFIGURE_STROKE_AXIS
from analysis.receptive_field_mapping.data.rf_boundary_types import StrokeAxisConfig

# The persisted keys, in canonical order. Kept as the authoritative key list so
# save/load and the JSON schema never drift.
_REQUIRED_JSON_KEYS: tuple[str, ...] = (
    "session_id",
    "axis_start_uv",
    "axis_end_uv",
    "swap_proximal_distal",
    "created_at",
    "modified_at",
)


# ---------------------------------------------------------------------------
# Path convention
# ---------------------------------------------------------------------------

def stroke_axis_root(database_path: Path) -> Path:
    """Return ``<database>/4_analysed/spatial_configure_stroke_axis``."""
    return Path(database_path) / "4_analysed" / SPATIAL_CONFIGURE_STROKE_AXIS


def stroke_axis_path(root: Path, session_id: str) -> Path:
    """Return ``<root>/<session_id>/<session_id>_stroke_axis.json``."""
    if not session_id:
        raise ValueError("stroke_axis_path: session_id must be a non-empty string.")
    return Path(root) / session_id / f"{session_id}_stroke_axis.json"


def stroke_axis_snapshot_path(root: Path, session_id: str) -> Path:
    """Return ``<root>/<session_id>/<session_id>_stroke_axis.png``.

    Companion snapshot image saved alongside the JSON on Validate — a visual
    record of the validated proximal/distal split (arrows over the forearm UV
    mesh + the chosen axis). Shares the JSON's per-session directory convention.
    """
    if not session_id:
        raise ValueError(
            "stroke_axis_snapshot_path: session_id must be a non-empty string."
        )
    return Path(root) / session_id / f"{session_id}_stroke_axis.png"


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_stroke_axis(
    path: Path,
    config: StrokeAxisConfig,
    created_at: str = "",
) -> None:
    """Write one session's stroke-axis JSON to ``path``.

    Serialises the axis endpoints + swap flag plus ``session_id`` / ``created_at``
    / ``modified_at``. Creates the parent directory, stamps ``modified_at`` to now
    (UTC), and stamps ``created_at`` to now only when the passed value is empty (a
    fresh Validate). Callers preserving an existing file's ``created_at`` should
    pass :func:`load_stroke_axis_created_at`'s result. Round-trips through
    :func:`load_stroke_axis`.
    """
    if not isinstance(config, StrokeAxisConfig):
        raise ValueError(
            f"save_stroke_axis: config must be StrokeAxisConfig, got "
            f"{type(config).__name__}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now = _utc_now_iso()
    doc = {
        "session_id": str(config.session_id),
        "axis_start_uv": [float(config.axis_start_uv[0]), float(config.axis_start_uv[1])],
        "axis_end_uv": [float(config.axis_end_uv[0]), float(config.axis_end_uv[1])],
        "swap_proximal_distal": bool(config.swap_proximal_distal),
        "created_at": created_at if created_at else now,
        "modified_at": now,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _load_doc(path: Path) -> dict:
    """Read + structurally validate the stroke-axis JSON at ``path``.

    Fail-fast on a missing file, a non-mapping root, and any missing/unknown key.
    Returns the raw dict (value typing is checked by the individual loaders).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stroke-axis JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Stroke-axis root must be a JSON object, got "
            f"{type(data).__name__} in {path}"
        )

    missing = [k for k in _REQUIRED_JSON_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Stroke-axis JSON is missing required key(s) {missing} in {path}"
        )
    extra = [k for k in data if k not in _REQUIRED_JSON_KEYS]
    if extra:
        raise ValueError(
            f"Stroke-axis JSON contains unknown key(s) {sorted(extra)} in {path}. "
            f"Valid keys: {list(_REQUIRED_JSON_KEYS)}"
        )
    return data


def _validate_endpoint(name: str, value, path: Path) -> tuple[float, float]:
    """Validate a JSON endpoint is a length-2 list of real (non-bool) numbers."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(
            f"{name} must be a 2-element [u, v] list, got {value!r} in {path}"
        )
    for i, comp in enumerate(value):
        if isinstance(comp, bool) or not isinstance(comp, (int, float)):
            raise ValueError(
                f"{name}[{i}] must be a real number, got {comp!r} in {path}"
            )
    return (float(value[0]), float(value[1]))


def load_stroke_axis(path: Path) -> StrokeAxisConfig:
    """Load one session's stroke-axis JSON into a frozen :class:`StrokeAxisConfig`.

    Fail-fast on every malformed input (no silent defaults):
      * missing file -> ``FileNotFoundError``;
      * non-mapping root, missing key, or unknown/extra key -> ``ValueError``;
      * an endpoint that is not a 2-element list of real numbers -> ``ValueError``;
      * a non-bool ``swap_proximal_distal`` -> ``ValueError``;
      * a degenerate (start == end) axis -> ``ValueError`` (via
        :class:`StrokeAxisConfig`).

    Metadata keys (``created_at`` / ``modified_at``) are validated for presence
    but not carried on the returned object.
    """
    path = Path(path)
    data = _load_doc(path)

    session_id = data["session_id"]
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(
            f"session_id must be a non-empty string, got {session_id!r} in {path}"
        )

    start = _validate_endpoint("axis_start_uv", data["axis_start_uv"], path)
    end = _validate_endpoint("axis_end_uv", data["axis_end_uv"], path)

    swap = data["swap_proximal_distal"]
    if not isinstance(swap, bool):
        raise ValueError(
            f"swap_proximal_distal must be a bool, got {swap!r} in {path}"
        )

    try:
        return StrokeAxisConfig(
            session_id=session_id,
            axis_start_uv=start,
            axis_end_uv=end,
            swap_proximal_distal=swap,
        )
    except ValueError as exc:
        raise ValueError(f"{exc} (in {path})") from exc


def load_stroke_axis_created_at(path: Path) -> str:
    """Return the ``created_at`` timestamp of an existing stroke-axis JSON.

    Used by the GUI's Validate to preserve the original creation time across a
    re-save. Fail-fast on a missing file / key or a non-string value.
    """
    data = _load_doc(path)
    created_at = data["created_at"]
    if not isinstance(created_at, str) or not created_at:
        raise ValueError(
            f"created_at must be a non-empty string, got {created_at!r} in {path}"
        )
    return created_at


def default_stroke_axis_config(session_id, motion) -> StrokeAxisConfig:
    """Build the seed :class:`StrokeAxisConfig` for a session with no saved JSON.

    Auto-initialises the axis *direction* from the current labels via
    :func:`initialize_stroke_axis` (proximal-pointing unit vector), then spans the
    axis across the valid strokes' UV extent along that direction, centred on
    their mean centroid. ``axis_end_uv`` is the proximal end and
    ``swap_proximal_distal`` is ``False``, so the seed axis reproduces the current
    labels (a no-op) — the researcher then edits it.

    Raises (via :func:`initialize_stroke_axis`) when no valid stroke vector exists
    — the GUI surfaces that and requires a manual draw (no silent default axis).
    """
    # Imported lazily so this IO module stays free of the heavy SLIM/mesh import
    # chain that ``metrics.rf_stroke_axis`` pulls in for the projection helpers.
    import numpy as np

    from analysis.receptive_field_mapping.metrics.rf_stroke_axis import (
        initialize_stroke_axis,
    )

    direction = initialize_stroke_axis(motion)  # (2,) proximal-pointing unit vector

    valid = motion.valid_mask
    center = np.nanmean(motion.centroid_uv[valid], axis=0)
    pts = np.vstack([motion.start_uv[valid], motion.end_uv[valid]])
    proj = (pts - center) @ direction
    lo = float(proj.min())
    hi = float(proj.max())

    start = center + lo * direction  # distal-most extent
    end = center + hi * direction    # proximal-most extent (direction -> proximal)

    return StrokeAxisConfig(
        session_id=str(session_id),
        axis_start_uv=(float(start[0]), float(start[1])),
        axis_end_uv=(float(end[0]), float(end[1])),
        swap_proximal_distal=False,
    )
