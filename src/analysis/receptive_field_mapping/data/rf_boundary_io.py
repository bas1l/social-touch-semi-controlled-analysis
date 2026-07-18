"""I/O layer for the ``spatial_extract_boundaries`` stage.

Owns everything that touches disk for the boundary-extraction stage:

* path resolution + fail-fast existence guards for the upstream artifacts,
* loading the raw per-session inputs into a :class:`BoundaryInputs`,
* persisting the response-fields NPZ and the JSON sentinel.

The persistence functions read the derived arrays (``smoothed`` / ``laplacian``
/ ``gradient_mag``) straight from :class:`BoundaryResults` — they are computed
once in the processing layer and never recomputed here.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping

import numpy as np

from analysis.pipeline.output_dirs import (
    SPATIAL_BUILD_RESPONSE_FIELDS,
    SPATIAL_EXTRACT_BOUNDARIES,
    SPATIAL_MAP_SINGLE_TOUCH,
    SPATIAL_SLIM_UV,
    TOUCH_COMPUTE_SERIES,
)
from analysis.pipeline.shared_constants import (
    session_id_from_path,
    single_touch_npz_filename,
)
from analysis.receptive_field_mapping.boundary.contract import (
    BoundaryContour,
    validate_contour,
)
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryInputs,
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_colors,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    load_slim_uv_cache,
    uv_points_to_xyz,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def boundary_session_output_dir(output_dir: Path, session_id: str) -> Path:
    return output_dir / session_id


def boundary_sentinel_path(session_output_dir: Path, session_id: str) -> Path:
    return session_output_dir / f'{session_id}_population_response_fields_done.json'


# ---------------------------------------------------------------------------
# Per-method boundary output layout
# ---------------------------------------------------------------------------
#
# Each boundary method owns its own subfolder under the session output dir:
#
#   <session_output_dir>/<method>/<session_id>_boundary.npz   (per-gesture contours)
#   <session_output_dir>/<method>/run_metadata.json           (provenance record)
#
# Methods never share a file, so running several methods produces non-overlapping
# trees and re-running one method overwrites only its own folder (idempotent).

def boundary_method_output_dir(session_output_dir: Path, method_name: str) -> Path:
    return Path(session_output_dir) / method_name


def boundary_contour_npz_path(
    session_output_dir: Path, method_name: str, session_id: str,
) -> Path:
    return boundary_method_output_dir(session_output_dir, method_name) / f'{session_id}_boundary.npz'


def boundary_run_metadata_path(session_output_dir: Path, method_name: str) -> Path:
    return boundary_method_output_dir(session_output_dir, method_name) / 'run_metadata.json'


def boundary_response_fields_npz_path(
    session_output_dir: Path, method_name: str, session_id: str,
) -> Path:
    """Per-method monolithic grid/heatmap NPZ path.

    Each method's fan-out run writes the grid/heatmap ``population_response_fields``
    NPZ inside that method's own ``<method>/`` folder (alongside its boundary NPZ).
    The grids themselves are boundary-method-independent, but the file is written
    once per method, so a downstream consumer iterating discovered methods reads the
    copy that lives in the method folder it is currently processing.
    """
    return (
        boundary_method_output_dir(session_output_dir, method_name)
        / f'{session_id}_population_response_fields.npz'
    )


def boundary_session_extract_dir(
    database_path: Path, iff_metric: str, session_id: str,
) -> Path:
    """Resolve one session's ``spatial_extract_boundaries`` output root.

    ``<database>/4_analysed/spatial_extract_boundaries/iff_<metric>/<session_id>`` —
    the directory whose immediate ``<method>/`` subfolders
    :func:`load_boundary_contours` / :func:`load_boundary_records` enumerate. The
    single place this path is spelled out, so downstream consumers never hardcode
    it.
    """
    return (
        Path(database_path) / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
        / f'iff_{iff_metric}' / session_id
    )


# ---------------------------------------------------------------------------
# Input path resolution + existence guards (fail-fast)
# ---------------------------------------------------------------------------

def resolve_boundary_input_paths(
    csv_path: Path,
    database_path: Path,
    session_id: str,
    iff_metric: str,
) -> List[Path]:
    """Resolve the four upstream input artifacts, raising loudly if any is missing.

    Returns ``[series_csv, forearm_ply, single_touch_npz, slim_cache]`` — the
    exact set fed to the staleness gate.
    """
    npz_filename = single_touch_npz_filename(iff_metric)

    series_csv_path = (
        database_path / '4_analysed' / TOUCH_COMPUTE_SERIES
        / f'{session_id}_series_augmented.csv'
    )
    if not series_csv_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: series-augmented CSV not found — "
            f"run 'touch_compute_series' first: {series_csv_path}"
        )

    forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
    if forearm_ply_path is None:
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: forearm PLY not found in "
            f"{csv_path.parent} — RF-centred PLY must exist."
        )

    npz_path = (
        database_path / '4_analysed' / SPATIAL_MAP_SINGLE_TOUCH
        / session_id / npz_filename
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: {npz_filename} not found: "
            f"{npz_path}. Enable 'spatial_map_single_touch' in the DAG config and re-run."
        )

    slim_cache_path = (
        database_path / '4_analysed' / SPATIAL_SLIM_UV
        / session_id / f'{session_id}_slim_uv.npz'
    )
    if not slim_cache_path.exists():
        raise FileNotFoundError(
            f"[Population Response Fields] {session_id}: SLIM UV cache not found: "
            f"{slim_cache_path}. Enable 'spatial_precompute_slim_uv' in the DAG "
            f"config and re-run to generate the cache."
        )

    return [series_csv_path, forearm_ply_path, npz_path, slim_cache_path]


# ---------------------------------------------------------------------------
# Shared response-fields NPZ (spatial_build_response_fields) — consumer side
# ---------------------------------------------------------------------------

def resolve_response_fields_npz(database_path: Path, session_id: str) -> Path:
    """Resolve the shared param-free response-fields NPZ for one session.

    Points at
    ``<database>/4_analysed/spatial_build_response_fields/<session>/<session>_response_fields.npz``
    — the single source of truth written by ``spatial_build_response_fields`` and
    consumed by both the RF-contour tuner and the boundary extractor. The path
    carries **no** iff-metric subfolder (the param-free fields are metric-
    independent).
    """
    return (
        Path(database_path) / '4_analysed' / SPATIAL_BUILD_RESPONSE_FIELDS
        / session_id / f'{session_id}_response_fields.npz'
    )


def load_param_free_fields_npz(npz_path: Path) -> dict[str, tuple]:
    """Read the param-free per-gesture fields from a shared response-fields NPZ.

    Inverse of ``build_session_response_fields``' packing: returns

        {gtype: (slim_raw float64, slim_unique_count int64, n_touches int)}

    in the NPZ's stored gesture order (canonical, incl. the synthetic
    ``'stroke'`` subset when present). float64/int64 arrays round-trip through
    ``np.savez`` / ``np.load`` exactly, so the returned fields are byte-identical
    to recomputing them via ``build_param_free_fields`` on the same inputs.

    Fail-fast (``FileNotFoundError`` / ``KeyError``) on a missing file or any
    missing per-gesture key.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Response-fields NPZ not found: {npz_path}. Run "
            "'spatial_build_response_fields' first."
        )
    fields: dict = {}
    with np.load(npz_path, allow_pickle=True) as d:
        gesture_types = [str(g) for g in d["gesture_types"]]
        for gtype in gesture_types:
            raw_key = f"heatmap_pre_threshold_{gtype}"
            cnt_key = f"unique_count_{gtype}"
            n_key = f"n_touches_{gtype}"
            for key in (raw_key, cnt_key, n_key):
                if key not in d:
                    raise KeyError(
                        f"Response-fields NPZ {npz_path} is missing key '{key}'."
                    )
            fields[gtype] = (
                d[raw_key].astype(np.float64),
                d[cnt_key].astype(np.int64),
                int(d[n_key]),
            )
    return fields


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_boundary_inputs(
    session_id: str,
    session_output_dir: Path,
    sentinel: Path,
    input_paths: List[Path],
) -> BoundaryInputs:
    """Load the raw per-session data from resolved ``input_paths``.

    ``input_paths`` must be ``[series_csv, forearm_ply, single_touch_npz,
    slim_cache]`` as returned by :func:`resolve_boundary_input_paths`.
    """
    series_csv_path, forearm_ply_path, npz_path, slim_cache_path = input_paths

    pop_data = load_population_data(series_csv_path, forearm_ply_path)
    n_verts = len(pop_data.forearm_vertices)
    rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_verts)

    cache = load_slim_uv_cache(slim_cache_path)
    raw_colors = load_forearm_vertex_colors(forearm_ply_path)

    return BoundaryInputs(
        session_id=session_id,
        session_output_dir=session_output_dir,
        sentinel=sentinel,
        pop_data=pop_data,
        rf_data=rf_data,
        forearm_uv=cache.uv,
        slim_V=cache.V,
        slim_faces=cache.F,
        raw_vertex_colors=raw_colors,
        forearm_ply_path=forearm_ply_path,
        input_paths=input_paths,
    )


# ---------------------------------------------------------------------------
# Persistence: response-fields NPZ
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Generic per-method boundary contour serialization
# ---------------------------------------------------------------------------
#
# One generic serializer over the frozen BoundaryContour contract (11 geometric
# fields + their SLIM-UV -> XYZ projections + every entry in the method-blind
# ``diagnostic_fields`` channel, keyed by field name, NOT by method). This
# replaces the former per-method ``_save_boundary_fields`` prefix blocks and the
# three duplicated ``*_to_dict`` serializers — the method is now the *folder*, so
# per-gesture keys carry only a ``_{gtype}`` suffix and no method prefix.

# The 11 contour scalar metrics persisted verbatim (float64), suffixed by gesture.
_CONTOUR_SCALAR_FIELDS = (
    'area_uv',
    'perimeter_uv',
    'circularity',
    'pca_major_uv',
    'pca_minor_uv',
    'pca_orientation_deg',
    'mean_iff_on_contour',
    'iff_at_centroid',
)


def _serialize_contour(
    contour: BoundaryContour,
    gtype: str,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
) -> dict:
    """Serialize one :class:`BoundaryContour` to ``{key}_{gtype}`` NPZ entries.

    Emits the 11 geometric fields, their XYZ projections (via
    :func:`uv_points_to_xyz`), and every ``diagnostic_fields`` array (each keyed
    ``diagnostic_{name}_{gtype}`` with a ``diagnostic_names_{gtype}`` index) so the
    channel round-trips without the IO layer knowing which method produced it.
    """
    out: dict = {}
    out[f'contour_uv_{gtype}'] = contour.contour_uv.astype(np.float64)
    out[f'centroid_uv_{gtype}'] = np.array(contour.centroid_uv, dtype=np.float64)
    out[f'peak_uv_{gtype}'] = np.array(contour.peak_uv, dtype=np.float64)
    for name in _CONTOUR_SCALAR_FIELDS:
        out[f'{name}_{gtype}'] = np.float64(getattr(contour, name))

    contour_xyz = uv_points_to_xyz(
        contour.contour_uv, forearm_uv, forearm_faces, forearm_V,
    )
    centroid_uv_arr = np.array(contour.centroid_uv, dtype=np.float64).reshape(1, 2)
    centroid_xyz = uv_points_to_xyz(
        centroid_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    contour_xyz_closed = np.vstack([contour_xyz, contour_xyz[:1]])
    perimeter_xyz_mm = float(
        np.sum(np.linalg.norm(np.diff(contour_xyz_closed, axis=0), axis=1))
    )

    c = centroid_xyz
    edges_i = contour_xyz[:-1] - c
    edges_j = contour_xyz[1:] - c
    closing_i = contour_xyz[-1] - c
    closing_j = contour_xyz[0] - c
    cross_vecs = np.vstack([
        np.cross(edges_i, edges_j),
        np.cross(closing_i, closing_j).reshape(1, 3),
    ])
    area_xyz_mm2 = 0.5 * float(np.sum(np.linalg.norm(cross_vecs, axis=1)))

    peak_uv_arr = np.array(contour.peak_uv, dtype=np.float64).reshape(1, 2)
    peak_xyz = uv_points_to_xyz(
        peak_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    out[f'contour_xyz_{gtype}'] = contour_xyz.astype(np.float64)
    out[f'centroid_xyz_{gtype}'] = centroid_xyz.astype(np.float64)
    out[f'peak_xyz_{gtype}'] = peak_xyz.astype(np.float64)
    out[f'perimeter_xyz_mm_{gtype}'] = np.float64(perimeter_xyz_mm)
    out[f'area_xyz_mm2_{gtype}'] = np.float64(area_xyz_mm2)

    diag_names = sorted(contour.diagnostic_fields.keys())
    out[f'diagnostic_names_{gtype}'] = np.array(diag_names, dtype=object)
    for name in diag_names:
        out[f'diagnostic_{name}_{gtype}'] = np.asarray(contour.diagnostic_fields[name])

    return out


def _deserialize_contour(data, gtype: str, method_name: str) -> BoundaryContour:
    """Rebuild a validated :class:`BoundaryContour` from ``{key}_{gtype}`` entries.

    Inverse of :func:`_serialize_contour` for the 11 geometric fields + the
    diagnostic channel (the XYZ projections are on-disk extras for downstream, not
    part of the contract). Fail-fast: any missing key raises ``KeyError`` and the
    result is passed through :func:`validate_contour`.
    """
    diag_names = [str(n) for n in data[f'diagnostic_names_{gtype}']]
    contour = BoundaryContour(
        contour_uv=np.asarray(data[f'contour_uv_{gtype}'], dtype=np.float64),
        area_uv=float(data[f'area_uv_{gtype}']),
        perimeter_uv=float(data[f'perimeter_uv_{gtype}']),
        circularity=float(data[f'circularity_{gtype}']),
        centroid_uv=tuple(
            np.asarray(data[f'centroid_uv_{gtype}'], dtype=np.float64).tolist()
        ),
        peak_uv=tuple(
            np.asarray(data[f'peak_uv_{gtype}'], dtype=np.float64).tolist()
        ),
        pca_major_uv=float(data[f'pca_major_uv_{gtype}']),
        pca_minor_uv=float(data[f'pca_minor_uv_{gtype}']),
        pca_orientation_deg=float(data[f'pca_orientation_deg_{gtype}']),
        mean_iff_on_contour=float(data[f'mean_iff_on_contour_{gtype}']),
        iff_at_centroid=float(data[f'iff_at_centroid_{gtype}']),
        method_name=method_name,
        diagnostic_fields={
            name: np.asarray(data[f'diagnostic_{name}_{gtype}'])
            for name in diag_names
        },
    )
    return validate_contour(contour)


def save_boundary_contours_npz(
    contours_by_gesture: Mapping[str, BoundaryContour | None],
    *,
    method_name: str,
    session_output_dir: Path,
    session_id: str,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    config_snapshot: Mapping,
) -> Path:
    """Persist one method's per-gesture contours to its own ``<method>/`` folder.

    Writes ``<session_output_dir>/<method>/<session_id>_boundary.npz`` (the whole
    file is rewritten each call — overwrite, never append, so a re-run leaves no
    stale per-gesture keys) plus a ``run_metadata.json`` provenance record (method
    name, session id, UTC timestamp, gesture list, config snapshot). Gestures whose
    contour is ``None`` (no boundary found) are recorded by omission — only the
    gestures that produced a contour are serialized and listed in ``gesture_types``.
    Each serialized contour is validated (fail-fast) before writing.
    """
    method_dir = boundary_method_output_dir(session_output_dir, method_name)
    method_dir.mkdir(parents=True, exist_ok=True)

    present = [g for g, contour in contours_by_gesture.items() if contour is not None]

    data_dict: dict = {
        'method_name': np.array(method_name, dtype=object),
        'session_id': np.array(session_id, dtype=object),
        'gesture_types': np.array(present, dtype=object),
    }
    for gtype in present:
        contour = validate_contour(contours_by_gesture[gtype])
        data_dict.update(
            _serialize_contour(contour, gtype, forearm_uv, forearm_faces, forearm_V)
        )

    npz_path = boundary_contour_npz_path(session_output_dir, method_name, session_id)
    np.savez(npz_path, **data_dict)

    metadata = {
        'method_name': method_name,
        'session_id': session_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'gesture_types': present,
        'n_contours': len(present),
        'config': dict(config_snapshot),
    }
    with open(boundary_run_metadata_path(session_output_dir, method_name), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "[Population Response Fields] %s: saved %d %s contour(s) → %s",
        session_id, len(present), method_name, npz_path.parent.name + '/' + npz_path.name,
    )
    return npz_path


@dataclass(frozen=True)
class LoadedBoundary:
    """One discovered per-gesture boundary: the contract plus its XYZ projections.

    :attr:`contour` is the validated, method-blind :class:`BoundaryContour` (the 11
    geometric fields + provenance + diagnostic channel). The remaining fields are
    the SLIM-UV -> XYZ projections written alongside it by :func:`_serialize_contour`
    (true-surface metrics downstream consumers need but which are *not* part of the
    abstract contract): the projected polygon/centroid/peak plus the 3D perimeter
    (mm) and 3D surface area (mm^2). Loaded from disk rather than recomputed so the
    written value stays the single source of truth.
    """

    contour: BoundaryContour
    contour_xyz: np.ndarray
    centroid_xyz: np.ndarray
    peak_xyz: np.ndarray
    perimeter_xyz_mm: float
    area_xyz_mm2: float


def _deserialize_record(data, gtype: str, method_name: str) -> LoadedBoundary:
    """Rebuild a :class:`LoadedBoundary` (contract + XYZ projections) for ``gtype``.

    Superset of :func:`_deserialize_contour`: also reads the on-disk XYZ-projection
    keys emitted by :func:`_serialize_contour`. Fail-fast: any missing key raises
    ``KeyError``.
    """
    return LoadedBoundary(
        contour=_deserialize_contour(data, gtype, method_name),
        contour_xyz=np.asarray(data[f'contour_xyz_{gtype}'], dtype=np.float64),
        centroid_xyz=np.asarray(data[f'centroid_xyz_{gtype}'], dtype=np.float64),
        peak_xyz=np.asarray(data[f'peak_xyz_{gtype}'], dtype=np.float64),
        perimeter_xyz_mm=float(data[f'perimeter_xyz_mm_{gtype}']),
        area_xyz_mm2=float(data[f'area_xyz_mm2_{gtype}']),
    )


def load_boundary_records(session_output_dir: Path) -> dict[str, dict[str, LoadedBoundary]]:
    """Discover ``<method>/`` folders for a session and load their boundary records.

    Enumerates the immediate subdirectories of ``session_output_dir``; a directory
    is a boundary-method folder iff it contains a ``*_boundary.npz``. Each such
    folder is loaded into ``{gtype: LoadedBoundary}`` (the contract contour plus its
    XYZ projections), keyed by the folder's method name, returning
    ``{method_name: {gtype: LoadedBoundary}}``.

    This is the discovery entry point for downstream consumers, which loop over the
    returned methods with identical logic (no branch on method identity) and emit
    per-method sub-outputs. Gestures whose contour was ``None`` (no boundary found)
    are absent from the inner mapping, so ``gtype in records[method]`` is the
    capability check that replaces the old ``boundary_contour_uv_<gtype> in npz``.

    Fail-fast on a malformed/partial method folder: a boundary NPZ without its
    ``run_metadata.json`` sidecar, more than one ``*_boundary.npz`` in a folder, a
    folder name that disagrees with the NPZ's stored ``method_name``, or a
    duplicate method all raise.
    """
    session_output_dir = Path(session_output_dir)
    if not session_output_dir.exists():
        raise FileNotFoundError(
            f"load_boundary_records: session output dir not found: {session_output_dir}"
        )

    discovered: dict[str, dict[str, LoadedBoundary]] = {}
    for subdir in sorted(p for p in session_output_dir.iterdir() if p.is_dir()):
        npz_files = sorted(subdir.glob('*_boundary.npz'))
        if not npz_files:
            continue  # not a method folder (e.g. aggregated/, inspection/)
        if len(npz_files) != 1:
            raise ValueError(
                f"load_boundary_records: method folder {subdir} contains "
                f"{len(npz_files)} '*_boundary.npz' files; expected exactly one."
            )
        metadata_path = boundary_run_metadata_path(session_output_dir, subdir.name)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"load_boundary_records: method folder {subdir} has a boundary NPZ "
                f"but no run_metadata.json (partial/corrupt output): {metadata_path}"
            )

        with np.load(npz_files[0], allow_pickle=True) as data:
            method_name = str(data['method_name'])
            if method_name != subdir.name:
                raise ValueError(
                    f"load_boundary_records: folder name {subdir.name!r} disagrees "
                    f"with stored method_name {method_name!r} in {npz_files[0]}"
                )
            gtypes = [str(g) for g in data['gesture_types']]
            records = {
                gtype: _deserialize_record(data, gtype, method_name)
                for gtype in gtypes
            }

        if method_name in discovered:
            raise ValueError(
                f"load_boundary_records: duplicate boundary method {method_name!r} "
                f"discovered under {session_output_dir}"
            )
        discovered[method_name] = records

    return discovered


class BoundaryNpzView:
    """Read-only ``np.load``-like view: a method's boundary geometry + its grid NPZ.

    Wraps the per-method monolithic grid/heatmap NPZ (``grid_*`` / ``heatmap_*`` /
    ``forearm_*`` / ``n_touches_*`` / ``boundary_method`` / …) and re-exposes the
    discovered :class:`LoadedBoundary` geometry under the **legacy** ``boundary_*``
    key names the comparison consumers already read (``boundary_contour_uv_<gtype>``,
    ``boundary_centroid_uv_<gtype>``, ``boundary_area_xyz_mm2_<gtype>``, …). This lets
    each consumer keep its existing per-gesture logic while sourcing geometry from
    the method-blind contract instead of the monolithic NPZ (which no longer carries
    it). The *same* view wraps every method — there is no branch on method identity.

    Presence semantics match the old NPZ exactly: a ``boundary_*_<gtype>`` key exists
    iff that gesture produced a contour (i.e. ``<gtype>`` is in the method's records),
    so ``f'boundary_contour_uv_{gtype}' in view`` is the capability check that
    replaced ``... in npz``.
    """

    def __init__(self, grids_npz, method_records: Mapping[str, "LoadedBoundary"]):
        self._npz = grids_npz
        self._bkeys: dict[str, np.ndarray] = {}
        for gtype, record in method_records.items():
            c = record.contour
            self._bkeys[f'boundary_contour_uv_{gtype}'] = np.asarray(c.contour_uv, dtype=np.float64)
            self._bkeys[f'boundary_centroid_uv_{gtype}'] = np.asarray(c.centroid_uv, dtype=np.float64)
            self._bkeys[f'boundary_peak_uv_{gtype}'] = np.asarray(c.peak_uv, dtype=np.float64)
            self._bkeys[f'boundary_area_uv_{gtype}'] = np.float64(c.area_uv)
            self._bkeys[f'boundary_perimeter_uv_{gtype}'] = np.float64(c.perimeter_uv)
            self._bkeys[f'boundary_circularity_{gtype}'] = np.float64(c.circularity)
            self._bkeys[f'boundary_pca_major_uv_{gtype}'] = np.float64(c.pca_major_uv)
            self._bkeys[f'boundary_pca_minor_uv_{gtype}'] = np.float64(c.pca_minor_uv)
            self._bkeys[f'boundary_pca_orientation_deg_{gtype}'] = np.float64(c.pca_orientation_deg)
            self._bkeys[f'boundary_mean_iff_on_contour_{gtype}'] = np.float64(c.mean_iff_on_contour)
            self._bkeys[f'boundary_iff_at_centroid_{gtype}'] = np.float64(c.iff_at_centroid)
            self._bkeys[f'boundary_area_xyz_mm2_{gtype}'] = np.float64(record.area_xyz_mm2)
            self._bkeys[f'boundary_perimeter_xyz_mm_{gtype}'] = np.float64(record.perimeter_xyz_mm)
            self._bkeys[f'boundary_centroid_xyz_{gtype}'] = np.asarray(record.centroid_xyz, dtype=np.float64)
            self._bkeys[f'boundary_peak_xyz_{gtype}'] = np.asarray(record.peak_xyz, dtype=np.float64)
            self._bkeys[f'boundary_contour_xyz_{gtype}'] = np.asarray(record.contour_xyz, dtype=np.float64)

    def __contains__(self, key: str) -> bool:
        return key in self._bkeys or key in self._npz

    def __getitem__(self, key: str):
        if key in self._bkeys:
            return self._bkeys[key]
        return self._npz[key]

    @property
    def files(self) -> list[str]:
        return list(self._npz.files) + list(self._bkeys.keys())


def discover_boundary_methods(
    session_configs: List[tuple],
    iff_metric: str,
) -> tuple[dict[str, dict[str, dict[str, "LoadedBoundary"]]], list[str]]:
    """Load every session's boundary records once; return them + the sorted methods.

    Shared discovery entry point for the cross-session comparison consumers. Returns
    ``(records_by_session, methods)`` where ``records_by_session`` maps
    ``session_id -> {method: {gtype: LoadedBoundary}}`` (each session read exactly
    once, reused by the per-method passes) and ``methods`` is the sorted union of the
    method names discovered across all sessions. Fail-fast: raises ``ValueError`` if
    no method folder exists under any session.
    """
    records_by_session: dict[str, dict] = {}
    method_names: set[str] = set()
    for csv_path, db_path in session_configs:
        session_id = session_id_from_path(Path(csv_path))
        session_dir = boundary_session_extract_dir(Path(db_path), iff_metric, session_id)
        records = load_boundary_records(session_dir)
        records_by_session[session_id] = records
        method_names.update(records.keys())
    if not method_names:
        raise ValueError(
            f"discover_boundary_methods: no boundary methods discovered across "
            f"{len(session_configs)} session(s) for iff_metric={iff_metric!r} — run "
            f"spatial_extract_boundaries first."
        )
    return records_by_session, sorted(method_names)


def load_boundary_npz_view(
    session_output_dir: Path,
    method_name: str,
    session_id: str,
    method_records: Mapping[str, "LoadedBoundary"],
) -> BoundaryNpzView:
    """Build a :class:`BoundaryNpzView` for one (session, method).

    Loads that method's per-method grid/heatmap NPZ and wraps it together with the
    already-discovered ``method_records`` (``{gtype: LoadedBoundary}``). Fail-fast:
    raises ``FileNotFoundError`` if the grid NPZ is missing.
    """
    npz_path = boundary_response_fields_npz_path(session_output_dir, method_name, session_id)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"load_boundary_npz_view: grid NPZ not found for method {method_name!r}: "
            f"{npz_path}"
        )
    grids_npz = np.load(npz_path, allow_pickle=True)
    return BoundaryNpzView(grids_npz, method_records)


def load_boundary_contours(session_output_dir: Path) -> dict[str, dict[str, BoundaryContour]]:
    """Discover ``<method>/`` folders and load their contract contours.

    Thin projection over :func:`load_boundary_records`: returns
    ``{method_name: {gtype: BoundaryContour}}`` (the contract only, without the XYZ
    projections). Consumers needing the true-surface XYZ metrics call
    :func:`load_boundary_records` instead.
    """
    return {
        method_name: {
            gtype: record.contour for gtype, record in records.items()
        }
        for method_name, records in load_boundary_records(session_output_dir).items()
    }


def save_boundary_outputs_npz(
    prepared: PreparedBoundaryData,
    results: BoundaryResults,
    params: BoundaryParams,
    *,
    session_output_dir: Path,
) -> Path:
    """Persist the session's grid/heatmap NPZ + one ``<method>/`` folder per method.

    Writes two kinds of output:

    * the monolithic ``{session_id}_population_response_fields.npz`` carrying the
      *non-boundary* per-session arrays (thresholded heatmaps, interpolation grids,
      PCA alignment, SLIM vertex colours) plus the active ``boundary_method`` /
      ``inflection_sigma`` scalars — boundary contours are **no longer** written
      here (they moved to per-method folders); and
    * one ``<method>/<session_id>_boundary.npz`` (+ ``run_metadata.json``) per
      method present in ``results.boundaries`` via
      :func:`save_boundary_contours_npz`.

    ``prepared.output_dir`` is the *per-method* run directory (``<session>/<method>``
    under the fan-out flow) — the monolithic NPZ + figures live there.
    ``session_output_dir`` is the session root (``<session>``); the per-method
    boundary contour folder is built under it (``<session>/<method>/``) so the
    discovery reader (:func:`load_boundary_contours`), which enumerates
    ``<session>/*/``, finds every method's contours regardless of which method the
    current run computed.

    Returns the monolithic NPZ path (used as the session's ``vertex_data_npz``).
    """
    session_id = prepared.session_id
    forearm_uv = prepared.forearm_uv
    forearm_faces = prepared.slim_faces
    forearm_V = prepared.slim_V
    inflection_sigma = params.inflection_sigma

    npz_path = prepared.output_dir / f'{session_id}_population_response_fields.npz'

    data_dict: dict = {
        'forearm_uv': forearm_uv.astype(np.float64),
        'forearm_faces': forearm_faces.astype(np.int32),
        'forearm_V': forearm_V.astype(np.float64),
        'session_id': np.array(session_id, dtype=object),
        'neuron_mode': np.array(params.neuron_mode, dtype=object),
        'min_overlap_pct': np.float64(prepared.min_overlap_pct),
        'gesture_types': np.array(list(prepared.results.keys()), dtype=object),
        'inflection_sigma': np.float64(inflection_sigma if inflection_sigma is not None else float('nan')),
        'flip_u': np.bool_(prepared.flip_u),
        'boundary_method': np.array(results.boundary_method, dtype=object),
    }

    for gtype in prepared.results.keys():
        slim_heatmap, n_touches, threshold = prepared.results[gtype]
        grid_u, grid_v, grid_z = prepared.per_gesture_grids[gtype]
        data_dict[f'heatmap_{gtype}'] = slim_heatmap.astype(np.float64)
        data_dict[f'heatmap_pre_threshold_{gtype}'] = prepared.per_gesture_slim_raw[gtype].astype(np.float64)
        data_dict[f'unique_count_{gtype}'] = prepared.per_gesture_slim_unique_count[gtype].astype(np.int64)
        data_dict[f'n_touches_{gtype}'] = np.int64(n_touches)
        data_dict[f'threshold_{gtype}'] = np.int64(threshold)
        data_dict[f'grid_u_{gtype}'] = grid_u.astype(np.float64)
        data_dict[f'grid_v_{gtype}'] = grid_v.astype(np.float64)
        data_dict[f'grid_z_{gtype}'] = grid_z.astype(np.float64)

    data_dict['alignment_center_uv'] = prepared.alignment_center.astype(np.float64)
    data_dict['alignment_rotation_matrix'] = prepared.alignment_rotation_matrix.astype(np.float64)
    data_dict['alignment_rotation_deg'] = np.float64(prepared.alignment_angle_deg)

    if prepared.slim_vertex_colors is not None:
        data_dict['slim_vertex_colors'] = prepared.slim_vertex_colors.astype(np.float64)

    np.savez(npz_path, **data_dict)
    logger.info("[Population Response Fields] %s: saved response fields NPZ → %s", session_id, npz_path.name)

    # ---- Per-method boundary folders (one <method>/ tree per computed method) ----
    config_snapshot = {
        'neuron_mode': params.neuron_mode,
        'min_overlap_pct': float(prepared.min_overlap_pct),
        'iff_metric': params.iff_metric,
        'inflection_sigma': (
            float(inflection_sigma) if inflection_sigma is not None else None
        ),
        'flip_u': bool(prepared.flip_u),
        'boundary_method': params.boundary_method,
    }
    for method_name, contours_by_gesture in results.boundaries.items():
        save_boundary_contours_npz(
            contours_by_gesture,
            method_name=method_name,
            session_output_dir=session_output_dir,
            session_id=session_id,
            forearm_uv=forearm_uv,
            forearm_faces=forearm_faces,
            forearm_V=forearm_V,
            config_snapshot=config_snapshot,
        )

    return npz_path


# ---------------------------------------------------------------------------
# Persistence: JSON sentinel
# ---------------------------------------------------------------------------

def write_boundary_sentinel(
    sentinel: Path,
    session_id: str,
    produced: List[Path],
    vertex_data_npz: Path | None = None,
) -> None:
    """Write the stage's done-marker JSON (PNG manifest + vertex-data NPZ path).

    The boundary geometry itself lives in the per-method ``<method>/`` NPZ folders
    (and in the monolithic grid NPZ referenced by ``vertex_data_npz``), so the
    sentinel no longer embeds serialized boundary dicts — the former
    ``inflection_boundaries`` payload was write-only (never read) and is dropped.
    """
    data = {
        'session_id': session_id,
        'n_pngs': len(produced),
        'pngs': [str(p) for p in produced],
    }
    if vertex_data_npz is not None:
        data['vertex_data_npz'] = str(vertex_data_npz)

    with open(sentinel, 'w') as f:
        json.dump(data, f, indent=2)
