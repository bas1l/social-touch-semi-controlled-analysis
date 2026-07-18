#!/usr/bin/env python
"""Prevalence scan: RF peaks sitting at the mapped-region (valid-data) edge.

Read-only *sizing* tool for the "Data-Edge RF Boundary" feature (Phase 1). It
answers one question over the whole database: **how many (session, gesture)
receptive fields have their response peak at the edge of the mapped forearm
region**, and which of those currently fail contour extraction with
``SEED_OUTSIDE_FOOTPRINT``.

It changes nothing in the contour algorithm. It only reuses production building
blocks:

* grid building        -> ``load_session_arrays`` + ``build_session_grid``
                          (``rf_contour_params_io`` — the same math the tuner GUI
                          and the ``spatial_extract_boundaries`` pipeline use).
* peak location        -> ``find_peak_location`` (``rf_inflection_boundary``).
* actual extraction    -> ``compute_radial_foot_stages(..., allow_partial=True)``
                          (``rf_radial_foot_boundary``), reading the returned
                          ``diagnostics`` branch to detect ``SEED_OUTSIDE_FOOTPRINT``.

Definitions (from the plan, kept exact):

* **Valid-data / mapped-region boundary** = the edge of ``~isnan(grid_z)`` on the
  interpolated response grid — the real physical limit of the mapped forearm
  blob. NOT the footprint (``grid_z > 0``).
* **Edge peak** = a located peak whose distance to the valid-data boundary is
  ``<= k`` cells (via a Euclidean distance transform), OR whose extraction
  currently raises ``SEED_OUTSIDE_FOOTPRINT``.

The distance-to-edge classification is deliberately robust: ``~isnan(grid_z)`` is
set by the interpolation domain, so it is essentially independent of the tuned
threshold. The ``SEED_OUTSIDE_FOOTPRINT`` attempt IS parameter-dependent, so per
(session, gesture) the scan uses the tuned contour-params JSON when one exists
(exactly what the tuner GUI would load) and falls back to the DAG global scalars
otherwise (the sanctioned ``strict=False`` bootstrap, reported per row — not a
silent default).

Config-driven, no hardcoded database path. The data root is resolved by
``path_tools.get_project_data_root()`` (the same resolver every entry script
uses) unless ``--database`` overrides it; the global boundary scalars + iff
metric come from the DAG config's ``spatial_extract_boundaries`` task options.

Fail-fast: a missing iff-metric directory, or a missing/unreadable session NPZ,
raises loudly. A per-(session, gesture) *compute* failure (e.g. an all-NaN
gesture grid) is recorded explicitly in the report and printed as a WARNING —
reported, never silently skipped.

Example
-------
    conda run -n social-touch-analysis python scripts/scan_edge_rfs.py --k 2

    # scan a specific (non-default) boundary output dir / metric:
    conda run -n social-touch-analysis python scripts/scan_edge_rfs.py --k 2 \
        --boundary-subdir spatial_extract_boundaries_bkp --iff-metric mean
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Make the ``src`` layout importable when run as a plain script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from scipy.ndimage import distance_transform_edt

from _vendor import DagConfigHandler
from _vendor import path_tools
from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES
from analysis.receptive_field_mapping.data.rf_boundary_types import (
    ContourFailureBranch,
)
from analysis.receptive_field_mapping.data.rf_contour_params_io import (
    build_session_grid,
    contour_params_path,
    contour_params_root,
    load_contour_params,
    load_session_arrays,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    find_peak_location,
)
from analysis.receptive_field_mapping.metrics.rf_radial_foot_boundary import (
    compute_radial_foot_stages,
)

_BOUNDARY_TASK = "spatial_extract_boundaries"
_POP_NPZ_SUFFIX = "_population_response_fields.npz"


# ---------------------------------------------------------------------------
# Extraction parameters (per session/gesture): tuned JSON if present, else DAG
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionParams:
    """The grid + radial-extraction scalars for one (session, gesture)."""

    min_overlap_pct: float
    median_filter_size: int | None
    radial_gauss_sigma: float
    radial_hess_sigma: float
    radial_envelope_smooth_sigma: float | None
    prominence: float | None
    plateau_size: int
    source: str  # "tuned" | "global"


def _global_params_from_dag(dag_handler: DagConfigHandler) -> dict:
    """Read the global boundary scalars + iff metric from the DAG task options.

    These are the exact scalars the ``spatial_extract_boundaries`` stage uses
    when ``use_tuned_params`` is off — the honest global fallback for a
    (session, gesture) that has no tuned JSON.
    """
    opts = dag_handler.get_task_options(_BOUNDARY_TASK)
    required = (
        "iff_metric",
        "min_overlap_pct",
        "median_filter_size",
        "radial_gauss_sigma",
        "radial_hess_sigma",
        "radial_envelope_smooth_sigma",
        "radial_prominence",
        "radial_plateau_size",
    )
    missing = [k for k in required if k not in opts]
    if missing:
        raise KeyError(
            f"DAG task '{_BOUNDARY_TASK}' options are missing key(s) {missing}. "
            f"Present: {sorted(opts)}"
        )
    return opts


def _resolve_params(
    globals_opts: dict,
    tune_root: Path,
    session_id: str,
    gesture: str,
    ignore_tuned: bool,
) -> ExtractionParams:
    """Pick the tuned per-(session, gesture) JSON if present, else DAG globals.

    Mirrors the tuner GUI: a tuned JSON is consumed through its ``effective_*``
    resolvers (so a disabled toggle becomes its concrete skip value). Absent (or
    ``--ignore-tuned``), the DAG global scalars are used verbatim. The chosen
    source is recorded on the row — an explicit, reported choice.
    """
    json_path = contour_params_path(tune_root, session_id, gesture)
    if not ignore_tuned and json_path.exists():
        p = load_contour_params(json_path)
        return ExtractionParams(
            min_overlap_pct=p.effective_min_overlap_pct(),
            median_filter_size=p.effective_median_filter_size(),
            radial_gauss_sigma=p.effective_gauss_sigma(),
            radial_hess_sigma=p.radial_hess_sigma,
            radial_envelope_smooth_sigma=p.effective_envelope_smooth_sigma(),
            prominence=p.effective_prominence(),
            plateau_size=p.effective_plateau_size(),
            source="tuned",
        )

    mfs = globals_opts["median_filter_size"]
    prom = globals_opts["radial_prominence"]
    return ExtractionParams(
        min_overlap_pct=float(globals_opts["min_overlap_pct"]),
        median_filter_size=(int(mfs) if mfs is not None else None),
        radial_gauss_sigma=float(globals_opts["radial_gauss_sigma"]),
        radial_hess_sigma=float(globals_opts["radial_hess_sigma"]),
        radial_envelope_smooth_sigma=float(
            globals_opts["radial_envelope_smooth_sigma"]
        ),
        prominence=(float(prom) if prom is not None else None),
        plateau_size=int(globals_opts["radial_plateau_size"]),
        source="global",
    )


# ---------------------------------------------------------------------------
# Geometry: distance from the peak to the valid-data boundary
# ---------------------------------------------------------------------------

def _distance_to_valid_edge(grid_z: np.ndarray, peak_rc: tuple[int, int]) -> float:
    """Euclidean distance (in cells) from ``peak_rc`` to the valid-data edge.

    The valid-data region is ``~isnan(grid_z)``. The array is padded with a ring
    of invalid cells so a region touching the raster border still measures its
    distance to that border (the raster edge is treated as a real limit). The
    distance transform gives, for each valid cell, the distance to the nearest
    invalid cell; the peak's value is its distance to the mapped-region boundary.
    """
    valid = ~np.isnan(grid_z)
    if not valid[peak_rc[0], peak_rc[1]]:
        # The global argmax cell is NaN only if the whole grid is NaN, which is
        # handled by the caller before reaching here. Guard loudly regardless.
        raise ValueError(
            f"peak cell {peak_rc} is NaN in grid_z — cannot measure edge distance."
        )
    padded = np.pad(valid, pad_width=1, mode="constant", constant_values=False)
    edt = distance_transform_edt(padded)
    return float(edt[peak_rc[0] + 1, peak_rc[1] + 1])


# ---------------------------------------------------------------------------
# One (session, gesture) scan
# ---------------------------------------------------------------------------

@dataclass
class RFScanRow:
    session_id: str
    gesture: str
    params_source: str
    peak_rc: tuple[int, int] | None
    dist_to_edge: float | None
    is_edge: bool
    contour_ok: bool
    failure_branch: str | None
    error: str | None


def _scan_one(
    npz_path: Path,
    session_id: str,
    gesture: str,
    params: ExtractionParams,
    k: float,
    n_angles: int,
    savgol_window: int | None,
) -> RFScanRow:
    """Build the grid, locate the peak, measure edge distance, attempt extraction."""
    arrays = load_session_arrays(npz_path, gesture)
    grid_u, grid_v, grid_z, _title, _n_touches = build_session_grid(
        arrays, params.min_overlap_pct, params.median_filter_size
    )

    peak_rc = find_peak_location(grid_z)
    if peak_rc is None:
        return RFScanRow(
            session_id=session_id,
            gesture=gesture,
            params_source=params.source,
            peak_rc=None,
            dist_to_edge=None,
            is_edge=False,
            contour_ok=False,
            failure_branch=None,
            error="find_peak_location returned None (all-NaN grid)",
        )

    dist = _distance_to_valid_edge(grid_z, peak_rc)

    stages = compute_radial_foot_stages(
        grid_u,
        grid_v,
        grid_z,
        gauss_sigma=params.radial_gauss_sigma,
        hess_sigma=params.radial_hess_sigma,
        envelope_smooth_sigma=params.radial_envelope_smooth_sigma,
        n_angles=n_angles,
        savgol_window=savgol_window,
        plateau_size=params.plateau_size,
        prominence=params.prominence,
        require_positive=True,
        allow_partial=True,
    )

    contour_ok = stages.get("contour_uv") is not None
    error = stages.get("error")
    diagnostics = stages.get("diagnostics")
    branch = diagnostics.branch if diagnostics is not None else None
    seed_outside = branch == ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT

    return RFScanRow(
        session_id=session_id,
        gesture=gesture,
        params_source=params.source,
        peak_rc=peak_rc,
        dist_to_edge=dist,
        is_edge=bool(dist <= k) or seed_outside,
        contour_ok=contour_ok,
        failure_branch=(branch.value if branch is not None else None),
        error=error,
    )


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------

def _discover_session_npzs(iff_dir: Path) -> list[tuple[str, Path]]:
    """Return ``[(session_id, npz_path), ...]`` under ``iff_dir``, sorted.

    Fail-fast: raises if ``iff_dir`` is absent or contains no session NPZ.
    """
    if not iff_dir.is_dir():
        raise FileNotFoundError(
            f"Boundary iff directory not found: {iff_dir}. Check --boundary-subdir "
            f"/ --iff-metric, or run 'spatial_extract_boundaries' first."
        )
    out: list[tuple[str, Path]] = []
    for session_dir in sorted(p for p in iff_dir.iterdir() if p.is_dir()):
        session_id = session_dir.name
        npz_path = session_dir / f"{session_id}{_POP_NPZ_SUFFIX}"
        if npz_path.exists():
            out.append((session_id, npz_path))
    if not out:
        raise FileNotFoundError(
            f"No '*{_POP_NPZ_SUFFIX}' files found under {iff_dir}."
        )
    return out


def _gesture_keys(npz_path: Path) -> list[str]:
    with np.load(npz_path, allow_pickle=True) as d:
        if "gesture_types" not in d:
            raise KeyError(f"NPZ {npz_path} is missing 'gesture_types'.")
        return [str(g) for g in d["gesture_types"]]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(rows: list[RFScanRow], k: float) -> None:
    total = len(rows)
    edge = [r for r in rows if r.is_edge]
    within_k = [
        r for r in rows if r.dist_to_edge is not None and r.dist_to_edge <= k
    ]
    seed_outside = [
        r for r in rows if r.failure_branch == ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT.value
    ]
    other_errors = [
        r
        for r in rows
        if r.error is not None
        and r.failure_branch != ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT.value
    ]

    print("\n" + "=" * 78)
    print("EDGE-RF PREVALENCE SCAN")
    print("=" * 78)
    print(f"  k (edge threshold, cells)           : {k}")
    print(f"  total RFs (session x gesture)        : {total}")
    print(f"  within k cells of valid-data edge    : {len(within_k)}")
    print(f"  currently SEED_OUTSIDE_FOOTPRINT     : {len(seed_outside)}")
    print(f"  edge peaks (within-k OR seed-outside): {len(edge)}")
    print(f"  other compute failures              : {len(other_errors)}")

    def _fmt(rows_):
        for r in sorted(rows_, key=lambda x: (x.session_id, x.gesture)):
            d = "n/a" if r.dist_to_edge is None else f"{r.dist_to_edge:6.2f}"
            print(
                f"    - {r.session_id:24s} {r.gesture:16s} "
                f"dist={d}  src={r.params_source:6s} "
                f"contour_ok={str(r.contour_ok):5s} "
                f"branch={r.failure_branch or '-'}"
            )

    print("\n  [within k cells of the valid-data edge]")
    _fmt(within_k) if within_k else print("    (none)")
    print("\n  [currently raising SEED_OUTSIDE_FOOTPRINT]")
    _fmt(seed_outside) if seed_outside else print("    (none)")
    if other_errors:
        print("\n  [other per-(session, gesture) compute failures — reported, not skipped]")
        for r in sorted(other_errors, key=lambda x: (x.session_id, x.gesture)):
            print(
                f"    - {r.session_id:24s} {r.gesture:16s} "
                f"branch={r.failure_branch or '-'}  error={r.error}"
            )
    print("=" * 78 + "\n")


def _save_fixtures(
    rows: list[RFScanRow],
    npz_by_session: dict[str, Path],
    params_by_key: dict[tuple[str, str], ExtractionParams],
    out_dir: Path,
    n_fixtures: int,
) -> list[Path]:
    """Save up to ``n_fixtures`` edge-RF grids as compact ``.npz`` fixtures.

    Prioritises the ``SEED_OUTSIDE_FOOTPRINT`` cases (the failures later phases
    must fix), then the remaining edge peaks nearest the valid-data boundary.
    """
    seed_branch = ContourFailureBranch.SEED_OUTSIDE_FOOTPRINT.value
    edge_rows = sorted(
        (r for r in rows if r.is_edge and r.peak_rc is not None),
        key=lambda x: (
            x.failure_branch != seed_branch,  # seed-outside cases first
            x.dist_to_edge if x.dist_to_edge is not None else 1e9,
        ),
    )[:n_fixtures]
    if not edge_rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for r in edge_rows:
        params = params_by_key[(r.session_id, r.gesture)]
        arrays = load_session_arrays(npz_by_session[r.session_id], r.gesture)
        grid_u, grid_v, grid_z, _title, _n = build_session_grid(
            arrays, params.min_overlap_pct, params.median_filter_size
        )
        fixture_path = out_dir / f"rf_edge_{r.session_id}_{r.gesture}.npz"
        np.savez_compressed(
            fixture_path,
            grid_u=grid_u.astype(np.float64),
            grid_v=grid_v.astype(np.float64),
            grid_z=grid_z.astype(np.float64),
            peak_rc=np.array(r.peak_rc, dtype=np.int64),
            dist_to_edge=np.float64(r.dist_to_edge),
            session_id=np.array(r.session_id, dtype=object),
            gesture=np.array(r.gesture, dtype=object),
            failure_branch=np.array(r.failure_branch or "", dtype=object),
        )
        saved.append(fixture_path)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prevalence scan: count/list (session, gesture) RFs whose peak sits "
            "at the mapped-region (valid-data) edge, and which currently fail "
            "contour extraction with SEED_OUTSIDE_FOOTPRINT. Read-only; changes "
            "no algorithm."
        )
    )
    parser.add_argument(
        "--k",
        type=float,
        default=2.0,
        help=(
            "Edge threshold in grid cells: a peak within k cells of the "
            "valid-data boundary counts as an edge peak (default: 2)."
        ),
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=Path("configs/analyse_workflow_processing_dag.yaml"),
        help=(
            "DAG config YAML — source of the global boundary scalars + iff "
            "metric (default: configs/analyse_workflow_processing_dag.yaml)."
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=(
            "Override the data root (…/semi-controlled). When omitted, the root "
            "is resolved by path_tools.get_project_data_root() — the same "
            "resolver the entry scripts use."
        ),
    )
    parser.add_argument(
        "--boundary-subdir",
        type=str,
        default=SPATIAL_EXTRACT_BOUNDARIES,
        help=(
            "Directory under 4_analysed holding the per-session "
            "'*_population_response_fields.npz' (default: "
            f"'{SPATIAL_EXTRACT_BOUNDARIES}'). Override to scan a renamed / "
            "backup / param-sweep variant."
        ),
    )
    parser.add_argument(
        "--iff-metric",
        type=str,
        default=None,
        help="Override the iff metric (default: read from the DAG task option).",
    )
    parser.add_argument("--n-angles", type=int, default=360)
    parser.add_argument("--savgol-window", type=int, default=31)
    parser.add_argument(
        "--ignore-tuned",
        action="store_true",
        help=(
            "Ignore per-(session, gesture) tuned JSONs and use the DAG global "
            "scalars everywhere (pure-global scan)."
        ),
    )
    parser.add_argument(
        "--save-fixtures",
        type=int,
        default=0,
        metavar="N",
        help="Save up to N edge-RF grids as .npz fixtures under --fixtures-dir.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures/rf_edge"),
        help="Directory for saved fixtures (default: tests/fixtures/rf_edge).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save the full per-RF report as JSON.",
    )
    args = parser.parse_args()

    dag_handler = DagConfigHandler(args.dag_config)
    globals_opts = _global_params_from_dag(dag_handler)
    iff_metric = args.iff_metric or str(globals_opts["iff_metric"])

    if args.database is not None:
        database_path = args.database
        if not database_path.is_dir():
            raise FileNotFoundError(
                f"--database path does not exist: {database_path}"
            )
    else:
        database_path = path_tools.get_project_data_root()
        if database_path is None:
            raise RuntimeError(
                "path_tools.get_project_data_root() returned None (no data root "
                "selected). Pass --database explicitly."
            )
        database_path = Path(database_path)

    iff_dir = database_path / "4_analysed" / args.boundary_subdir / f"iff_{iff_metric}"
    tune_root = contour_params_root(database_path, iff_metric)

    print(f"[scan_edge_rfs] database    : {database_path}")
    print(f"[scan_edge_rfs] boundary dir: {iff_dir}")
    print(f"[scan_edge_rfs] tune root   : {tune_root}")
    print(f"[scan_edge_rfs] k = {args.k}   ignore_tuned = {args.ignore_tuned}")

    sessions = _discover_session_npzs(iff_dir)
    print(f"[scan_edge_rfs] {len(sessions)} session NPZ(s) found.\n")

    rows: list[RFScanRow] = []
    npz_by_session: dict[str, Path] = {}
    params_by_key: dict[tuple[str, str], ExtractionParams] = {}

    for session_id, npz_path in sessions:
        npz_by_session[session_id] = npz_path
        gestures = _gesture_keys(npz_path)
        print(f"  {session_id}: {len(gestures)} gesture(s)")
        for gesture in gestures:
            params = _resolve_params(
                globals_opts, tune_root, session_id, gesture, args.ignore_tuned
            )
            params_by_key[(session_id, gesture)] = params
            try:
                row = _scan_one(
                    npz_path,
                    session_id,
                    gesture,
                    params,
                    args.k,
                    args.n_angles,
                    args.savgol_window,
                )
            except Exception as exc:  # noqa: BLE001
                # A per-(session, gesture) compute failure is reported explicitly
                # (a row with the error) so the survey is complete — never a
                # silent skip. Missing NPZ / missing dirs already fail-fast above.
                print(
                    f"    WARNING [{session_id} / {gesture}]: "
                    f"{type(exc).__name__}: {exc}"
                )
                row = RFScanRow(
                    session_id=session_id,
                    gesture=gesture,
                    params_source=params.source,
                    peak_rc=None,
                    dist_to_edge=None,
                    is_edge=False,
                    contour_ok=False,
                    failure_branch=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)

    _print_report(rows, args.k)

    if args.save_fixtures > 0:
        saved = _save_fixtures(
            rows, npz_by_session, params_by_key, args.fixtures_dir, args.save_fixtures
        )
        if saved:
            print(f"[scan_edge_rfs] saved {len(saved)} fixture(s):")
            for p in saved:
                print(f"    - {p}")
        else:
            print("[scan_edge_rfs] no edge RFs to save as fixtures.")

    if args.out is not None:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "k": args.k,
                    "database": str(database_path),
                    "iff_metric": iff_metric,
                    "boundary_subdir": args.boundary_subdir,
                    "rows": [asdict(r) for r in rows],
                },
                fh,
                indent=2,
            )
        print(f"[scan_edge_rfs] full report written to {args.out}")


if __name__ == "__main__":
    main()
