"""Extract 1D RF cross-section profiles and boundary crossings from population heatmap grids."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import IFF_METRICS, session_id_from_path
from analysis.receptive_field_mapping.data.rf_boundary_io import (
    boundary_response_fields_npz_path,
    boundary_session_extract_dir,
    load_boundary_records,
)
from analysis.receptive_field_mapping.rendering.rf_profile_renderer import (
    render_center_axis_profile,
    render_combined_profile,
    render_gradient_profile,
    render_laplacian_context,
    render_laplacian_profile,
    render_raycast_section,
)

logger = logging.getLogger(__name__)


def find_boundary_u_crossings(
    contour_uv: np.ndarray | None,
    v_target: float,
) -> np.ndarray:
    """Return sorted U-intercepts where a closed UV contour crosses a constant-V scanline."""
    if contour_uv is None or len(contour_uv) == 0:
        return np.empty(0, dtype=np.float64)

    closed = np.concatenate([contour_uv, contour_uv[:1]], axis=0)
    u1 = closed[:-1, 0]
    v1 = closed[:-1, 1]
    u2 = closed[1:, 0]
    v2 = closed[1:, 1]

    # One-sided strict inequality avoids double-counting at vertices
    straddles = (v1 <= v_target) & (v2 > v_target) | (v2 <= v_target) & (v1 > v_target)
    if not np.any(straddles):
        return np.empty(0, dtype=np.float64)

    u1_s = u1[straddles]
    v1_s = v1[straddles]
    u2_s = u2[straddles]
    v2_s = v2[straddles]

    u_cross = u1_s + (v_target - v1_s) * (u2_s - u1_s) / (v2_s - v1_s)
    return np.sort(u_cross)


def find_boundary_v_crossings(
    contour_uv: np.ndarray | None,
    u_target: float,
) -> np.ndarray:
    """Return sorted V-intercepts where a closed UV contour crosses a constant-U scanline."""
    if contour_uv is None or len(contour_uv) == 0:
        return np.empty(0, dtype=np.float64)

    closed = np.concatenate([contour_uv, contour_uv[:1]], axis=0)
    u1 = closed[:-1, 0]
    v1 = closed[:-1, 1]
    u2 = closed[1:, 0]
    v2 = closed[1:, 1]

    straddles = (u1 <= u_target) & (u2 > u_target) | (u2 <= u_target) & (u1 > u_target)
    if not np.any(straddles):
        return np.empty(0, dtype=np.float64)

    u1_s, v1_s = u1[straddles], v1[straddles]
    u2_s, v2_s = u2[straddles], v2[straddles]

    v_cross = v1_s + (u_target - u1_s) * (v2_s - v1_s) / (u2_s - u1_s)
    return np.sort(v_cross)


def _extract_profiles_for_gesture(
    session_id: str,
    gtype: str,
    npz: np.lib.npyio.NpzFile,
    contour_uv: np.ndarray | None,
) -> pd.DataFrame:
    """Extract 150 constant-V profiles and boundary crossings for one gesture type.

    ``contour_uv`` is the current method's boundary contour for this gesture (or
    ``None`` when that gesture produced no boundary) — read from the discovered
    :class:`BoundaryContour`, never from a method-named NPZ key.
    """
    grid_u = npz[f'grid_u_{gtype}']
    grid_v = npz[f'grid_v_{gtype}']
    grid_z = npz[f'grid_z_{gtype}']

    n_cols = grid_v.shape[1]
    rows: list[dict] = []

    for j in range(n_cols):
        iff_values = grid_z[:, j]
        v_value = float(grid_v[0, j])

        valid_mask = ~np.isnan(iff_values)
        n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            profile_max = float('nan')
            profile_mean = float('nan')
        else:
            profile_max = float(np.nanmax(iff_values))
            profile_mean = float(np.nanmean(iff_values))

        crossings = find_boundary_u_crossings(contour_uv, v_value)
        n_crossings = len(crossings)

        if n_crossings >= 1:
            boundary_u_left = float(crossings[0])
            boundary_u_right = float(crossings[-1])
        else:
            boundary_u_left = float('nan')
            boundary_u_right = float('nan')

        if n_crossings >= 2:
            boundary_width = boundary_u_right - boundary_u_left
        else:
            boundary_width = float('nan')

        boundary_u_all = ';'.join(f'{u:.6f}' for u in crossings) if n_crossings > 0 else ''

        rows.append({
            'session_id': session_id,
            'gesture_type': gtype,
            'v_index': j,
            'v_value_mm': v_value,
            'n_valid': n_valid,
            'profile_max': profile_max,
            'profile_mean': profile_mean,
            'n_boundary_crossings': n_crossings,
            'boundary_u_left_mm': boundary_u_left,
            'boundary_u_right_mm': boundary_u_right,
            'boundary_width_mm': boundary_width,
            'boundary_u_all': boundary_u_all,
        })

    return pd.DataFrame(rows)


def _methodology_md(boundary_method: str) -> str:
    """Build the (method-blind) methodology markdown for one boundary method.

    ``boundary_method`` is interpolated as free-form provenance text only — the
    prose never branches on which method it is. Overlay figures are described by
    the diagnostic capability they require (the ``lmax`` / ``gradient_magnitude``
    channels), so a method is documented by what it *provides*, not by its name.
    """
    return f"""\
# RF Profile Extraction: Methodology

## What this pipeline produces

1D cross-section profiles extracted from 2D population RF heatmaps, with
boundary crossing annotations derived from the boundary contour of the
`{boundary_method}` method.

## Distinction between profile data and boundary computation

The profiles and the boundary contour are computed from **different
representations** of the same underlying heatmap.

### Profile IFF values (the plotted curve)

The IFF values shown in each profile are sampled directly from `grid_z`,
the **raw interpolated** 150x150 heatmap produced by harmonic interpolation
of per-vertex IFF onto a regular UV grid. No additional smoothing is applied
to these values.

### Boundary contour (the crossing markers)

The crossing markers are the intersections of the `{boundary_method}` method's
closed UV boundary contour with each constant-U / constant-V scan line. The
contour is produced by the boundary-extraction stage and read back here through
the method-blind boundary contract, so this stage makes no assumption about how
the contour was computed.

## Consequence for profile interpretation

The boundary is generally derived from a smoothed / curvature representation of
the field while the profile plots the raw field, so the crossing markers may not
fall exactly at the corresponding feature of the displayed 1D curve. The 2D
analysis (which includes off-axis curvature) does not generally coincide with a
purely 1D analysis along a single slice.

## Capability-gated overlay figures

Beyond the always-produced raw profiles, each figure group is emitted only when
the method's contour carries the diagnostic field that group needs — a
capability check, never a check on the method name:

- `gradient_1d/` — 1D gradient magnitude along the scan axis; emitted when the
  contour carries a `gradient_magnitude` diagnostic field.
- `combined/` and `raycast/` — raw+smoothed+λmax overlays and the λmax raycast
  section; emitted when the contour carries the `lmax` diagnostic (and a smoothed
  field). Methods that do not provide these diagnostics simply skip the groups.

## Output formats

Each profile figure is saved in two formats:
- **PNG** (150 dpi) for quick inspection and screen display.
- **SVG** (vector) for publication-quality figure preparation.

## Output layout

Outputs are grouped per boundary method under each session directory
(`iff_<metric>/<session_id>/{boundary_method}/`); figures are further grouped into
per-category subfolders (`profile_raw/`, `gradient_1d/`, `combined/`, `raycast/`,
…). The per-gesture profile CSV tables and the `*_rf_profiles_done.json` sentinel
stay at that per-method directory root.
"""


def _write_methodology_md(output_dir: Path, boundary_method: str) -> None:
    """Write the methodology markdown for one boundary method at the output root."""
    md_path = output_dir / f'rf_profile_extraction_methodology_{boundary_method}.md'
    md_path.write_text(_methodology_md(boundary_method), encoding='utf-8')


def _render_profiles_for_method(
    *,
    session_id: str,
    method_output_dir: Path,
    sentinel: Path,
    npz: "np.lib.npyio.NpzFile",
    method_records: dict,
) -> None:
    """Render one boundary method's profile CSVs + figures for a session.

    ``method_records`` maps ``gtype -> LoadedBoundary`` for the discovered method (a
    gesture absent from the mapping produced no contour). All boundary geometry is
    read from each gesture's :class:`BoundaryContour`; overlay figure groups are
    gated purely on the diagnostic capabilities of that contour
    (``gradient_magnitude`` / ``lmax`` / ``smoothed`` + ``laplacian``), never on the
    method name — so a method is served by what it *provides*, tolerating absence.
    """
    gesture_types = list(npz['gesture_types'])
    method_output_dir.mkdir(parents=True, exist_ok=True)
    produced_csvs: list[str] = []
    produced_pngs: list[str] = []
    produced_svgs: list[str] = []

    for gtype in gesture_types:
        grid_key = f'grid_u_{gtype}'
        if grid_key not in npz:
            logger.warning(
                "[RF Profile Extraction] %s: grid key %r missing from NPZ, skipping gesture %r.",
                session_id, grid_key, gtype,
            )
            continue

        record = method_records.get(gtype)
        contour = record.contour if record is not None else None
        contour_uv = contour.contour_uv if contour is not None else None

        df = _extract_profiles_for_gesture(session_id, gtype, npz, contour_uv)
        csv_out = method_output_dir / f'{session_id}_rf_profiles_{gtype}.csv'
        df.to_csv(csv_out, index=False)
        produced_csvs.append(str(csv_out))
        logger.info(
            "[RF Profile Extraction] %s/%s: wrote %d profiles to %s",
            session_id, gtype, len(df), csv_out.name,
        )

        grid_u = npz[f'grid_u_{gtype}']
        grid_v = npz[f'grid_v_{gtype}']
        grid_z = npz[f'grid_z_{gtype}']
        u_coords = grid_u[:, 0]
        v_coords = grid_v[0, :]

        centroid_uv = (
            np.asarray(contour.centroid_uv, dtype=np.float64)
            if contour is not None else None
        )
        peak_uv = (
            np.asarray(contour.peak_uv, dtype=np.float64)
            if contour is not None else None
        )

        centers = [('centroid', centroid_uv), ('peak', peak_uv)]
        has_any_center = any(c is not None for _, c in centers)
        if not has_any_center:
            logger.warning(
                "[RF Profile Extraction] %s/%s: no centroid or peak found, skipping profile rendering.",
                session_id, gtype,
            )
            continue

        # Overlay diagnostics — capability checks on this gesture's own contour
        # (tolerating absence), never a check on which method produced it.
        has_laplacian = (
            contour is not None
            and contour.has_diagnostic("smoothed")
            and contour.has_diagnostic("laplacian")
        )
        if has_laplacian:
            smoothed = contour.diagnostic_fields["smoothed"]
            laplacian = contour.diagnostic_fields["laplacian"]

        has_gradient = contour is not None and contour.has_diagnostic("gradient_magnitude")
        if has_gradient:
            gradient_mag = contour.diagnostic_fields["gradient_magnitude"]

        has_hessian = contour is not None and contour.has_diagnostic("lmax")
        if has_hessian:
            radial_lmax = contour.diagnostic_fields["lmax"]

        # Category subdirectories — created only when their category is produced
        # (gated by the same capability flags as the figures), so no empty dirs.
        raw_dir = method_output_dir / 'profile_raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        if has_laplacian:
            smoothed_dir = method_output_dir / 'profile_smoothed'
            smoothed_dir.mkdir(parents=True, exist_ok=True)
            laplacian_1d_dir = method_output_dir / 'laplacian_1d'
            laplacian_1d_dir.mkdir(parents=True, exist_ok=True)
            laplacian_2d_dir = method_output_dir / 'laplacian_2d'
            laplacian_2d_dir.mkdir(parents=True, exist_ok=True)
        if has_gradient:
            gradient_1d_dir = method_output_dir / 'gradient_1d'
            gradient_1d_dir.mkdir(parents=True, exist_ok=True)
        if has_hessian and has_laplacian:
            combined_dir = method_output_dir / 'combined'
            combined_dir.mkdir(parents=True, exist_ok=True)

        for center_type, center in centers:
            if center is None:
                continue

            # U-profile: IFF vs U at constant V = center_v
            j = int(np.argmin(np.abs(v_coords - center[1])))
            iff_u = grid_z[:, j]
            crossings_u = find_boundary_u_crossings(contour_uv, v_coords[j])
            u_png = raw_dir / f'{session_id}_rf_profile_u_{center_type}_{gtype}.png'
            render_center_axis_profile(
                coords=u_coords,
                iff_values=iff_u,
                boundary_crossings=crossings_u,
                output_path=u_png,
                title=f"{session_id} | {gtype} | IFF vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                xlabel="U (mm)",
            )
            produced_pngs.append(str(u_png))
            produced_svgs.append(str(u_png.with_suffix('.svg')))

            if has_laplacian:
                smoothed_u = smoothed[:, j]
                u_smooth_png = smoothed_dir / f'{session_id}_rf_profile_smoothed_u_{center_type}_{gtype}.png'
                render_center_axis_profile(
                    coords=u_coords,
                    iff_values=smoothed_u,
                    boundary_crossings=crossings_u,
                    output_path=u_smooth_png,
                    title=f"{session_id} | {gtype} | Smoothed IFF vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                    xlabel="U (mm)",
                    ylabel="Smoothed IFF (Hz)",
                )
                produced_pngs.append(str(u_smooth_png))
                produced_svgs.append(str(u_smooth_png.with_suffix('.svg')))

                lap_u = laplacian[:, j]
                u_lap1d_png = laplacian_1d_dir / f'{session_id}_rf_laplacian_1d_u_{center_type}_{gtype}.png'
                render_laplacian_profile(
                    coords=u_coords,
                    lap_values=lap_u,
                    boundary_crossings=crossings_u,
                    output_path=u_lap1d_png,
                    title=f"{session_id} | {gtype} | Laplacian vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                    xlabel="U (mm)",
                )
                produced_pngs.append(str(u_lap1d_png))
                produced_svgs.append(str(u_lap1d_png.with_suffix('.svg')))

                u_lap_png = laplacian_2d_dir / f'{session_id}_rf_laplacian_2d_u_{center_type}_{gtype}.png'
                render_laplacian_context(
                    grid_u=grid_u, grid_v=grid_v,
                    laplacian=laplacian, smoothed=smoothed,
                    contour_uv=contour_uv,
                    scanline_coord=float(v_coords[j]),
                    scanline_axis="U",
                    boundary_crossings=crossings_u,
                    output_path=u_lap_png,
                    title=f"{session_id} | {gtype} | Laplacian context — U-profile at {center_type} (V={v_coords[j]:.1f} mm)",
                )
                produced_pngs.append(str(u_lap_png))
                produced_svgs.append(str(u_lap_png.with_suffix('.svg')))

            if has_gradient:
                grad_u = gradient_mag[:, j]
                u_grad_png = gradient_1d_dir / f'{session_id}_rf_gradient_1d_u_{center_type}_{gtype}.png'
                render_gradient_profile(
                    coords=u_coords,
                    grad_values=grad_u,
                    gradient_crossings=crossings_u,
                    output_path=u_grad_png,
                    title=f"{session_id} | {gtype} | |∇IFF| vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                    xlabel="U (mm)",
                )
                produced_pngs.append(str(u_grad_png))
                produced_svgs.append(str(u_grad_png.with_suffix('.svg')))

            if has_hessian and has_laplacian:
                u_combined_png = combined_dir / f'{session_id}_rf_profile_combined_u_{center_type}_{gtype}.png'
                render_combined_profile(
                    coords=u_coords,
                    iff_raw=grid_z[:, j],
                    iff_smoothed=smoothed[:, j],
                    lmax_values=radial_lmax[:, j],
                    boundary_crossings=crossings_u,
                    peak_coord=float(center[0]),
                    output_path=u_combined_png,
                    title=f"{session_id} | {gtype} | raw+smoothed+λmax vs U at {center_type} (V={v_coords[j]:.1f} mm)",
                    xlabel="U (mm)",
                )
                produced_pngs.append(str(u_combined_png))
                produced_svgs.append(str(u_combined_png.with_suffix('.svg')))

            # V-profile: IFF vs V at constant U = center_u
            i = int(np.argmin(np.abs(u_coords - center[0])))
            iff_v = grid_z[i, :]
            crossings_v = find_boundary_v_crossings(contour_uv, u_coords[i])
            v_png = raw_dir / f'{session_id}_rf_profile_v_{center_type}_{gtype}.png'
            render_center_axis_profile(
                coords=v_coords,
                iff_values=iff_v,
                boundary_crossings=crossings_v,
                output_path=v_png,
                title=f"{session_id} | {gtype} | IFF vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                xlabel="V (mm)",
            )
            produced_pngs.append(str(v_png))
            produced_svgs.append(str(v_png.with_suffix('.svg')))

            if has_laplacian:
                smoothed_v = smoothed[i, :]
                v_smooth_png = smoothed_dir / f'{session_id}_rf_profile_smoothed_v_{center_type}_{gtype}.png'
                render_center_axis_profile(
                    coords=v_coords,
                    iff_values=smoothed_v,
                    boundary_crossings=crossings_v,
                    output_path=v_smooth_png,
                    title=f"{session_id} | {gtype} | Smoothed IFF vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                    xlabel="V (mm)",
                    ylabel="Smoothed IFF (Hz)",
                )
                produced_pngs.append(str(v_smooth_png))
                produced_svgs.append(str(v_smooth_png.with_suffix('.svg')))

                lap_v = laplacian[i, :]
                v_lap1d_png = laplacian_1d_dir / f'{session_id}_rf_laplacian_1d_v_{center_type}_{gtype}.png'
                render_laplacian_profile(
                    coords=v_coords,
                    lap_values=lap_v,
                    boundary_crossings=crossings_v,
                    output_path=v_lap1d_png,
                    title=f"{session_id} | {gtype} | Laplacian vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                    xlabel="V (mm)",
                )
                produced_pngs.append(str(v_lap1d_png))
                produced_svgs.append(str(v_lap1d_png.with_suffix('.svg')))

                v_lap_png = laplacian_2d_dir / f'{session_id}_rf_laplacian_2d_v_{center_type}_{gtype}.png'
                render_laplacian_context(
                    grid_u=grid_u, grid_v=grid_v,
                    laplacian=laplacian, smoothed=smoothed,
                    contour_uv=contour_uv,
                    scanline_coord=float(u_coords[i]),
                    scanline_axis="V",
                    boundary_crossings=crossings_v,
                    output_path=v_lap_png,
                    title=f"{session_id} | {gtype} | Laplacian context — V-profile at {center_type} (U={u_coords[i]:.1f} mm)",
                )
                produced_pngs.append(str(v_lap_png))
                produced_svgs.append(str(v_lap_png.with_suffix('.svg')))

            if has_gradient:
                grad_v = gradient_mag[i, :]
                v_grad_png = gradient_1d_dir / f'{session_id}_rf_gradient_1d_v_{center_type}_{gtype}.png'
                render_gradient_profile(
                    coords=v_coords,
                    grad_values=grad_v,
                    gradient_crossings=crossings_v,
                    output_path=v_grad_png,
                    title=f"{session_id} | {gtype} | |∇IFF| vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                    xlabel="V (mm)",
                )
                produced_pngs.append(str(v_grad_png))
                produced_svgs.append(str(v_grad_png.with_suffix('.svg')))

            if has_hessian and has_laplacian:
                v_combined_png = combined_dir / f'{session_id}_rf_profile_combined_v_{center_type}_{gtype}.png'
                render_combined_profile(
                    coords=v_coords,
                    iff_raw=grid_z[i, :],
                    iff_smoothed=smoothed[i, :],
                    lmax_values=radial_lmax[i, :],
                    boundary_crossings=crossings_v,
                    peak_coord=float(center[1]),
                    output_path=v_combined_png,
                    title=f"{session_id} | {gtype} | raw+smoothed+λmax vs V at {center_type} (U={u_coords[i]:.1f} mm)",
                    xlabel="V (mm)",
                )
                produced_pngs.append(str(v_combined_png))
                produced_svgs.append(str(v_combined_png.with_suffix('.svg')))

            # Raycast section (HTML-style): one representative ray from this
            # centre. Requires an lmax + smoothed overlay and a boundary contour.
            if has_hessian and has_laplacian and contour_uv is not None and len(contour_uv) > 0:
                raycast_dir = method_output_dir / 'raycast'
                raycast_dir.mkdir(parents=True, exist_ok=True)
                raycast_png = raycast_dir / f'{session_id}_rf_raycast_{center_type}_{gtype}.png'
                render_raycast_section(
                    grid_u=grid_u, grid_v=grid_v,
                    lmax=radial_lmax, grid_z=grid_z, smoothed=smoothed,
                    contour_uv=contour_uv, peak_uv=center,
                    output_path=raycast_png,
                    title=f"{session_id} | {gtype} | raycast at {center_type} — Hessian λmax + foot pick",
                )
                produced_pngs.append(str(raycast_png))
                produced_svgs.append(str(raycast_png.with_suffix('.svg')))

    sentinel_data = {
        'session_id': session_id,
        'gesture_types': gesture_types,
        'n_csvs': len(produced_csvs),
        'csvs': produced_csvs,
        'n_pngs': len(produced_pngs),
        'pngs': produced_pngs,
        'n_svgs': len(produced_svgs),
        'svgs': produced_svgs,
    }
    with open(sentinel, 'w') as f:
        json.dump(sentinel_data, f, indent=2)

    print(
        f"[RF Profile Extraction] {session_id}: done — "
        f"{len(produced_csvs)} CSVs, {len(produced_pngs)} PNGs, "
        f"{len(produced_svgs)} SVGs written."
    )


def run_rf_profile_extraction(
    session_configs: list[tuple[Path, Path]],
    output_dir: Path,
    force_processing: bool = False,
    iff_metric: str = "mean",
) -> None:
    """Extract 1D RF profiles and boundary crossings from population heatmap grids."""
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"[RF Profile Extraction] Invalid iff_metric {iff_metric!r}. "
            f"Expected one of {IFF_METRICS}."
        )
    if not session_configs:
        raise ValueError("[RF Profile Extraction] session_configs is empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # One methodology markdown per discovered method (written once, first sighting).
    methodology_written: set[str] = set()

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)
        session_id = session_id_from_path(csv_path)

        session_extract_dir = boundary_session_extract_dir(
            database_path, iff_metric, session_id,
        )
        # Discover every boundary method that ran for this session and emit one
        # profile sub-output tree per method — identical per-method logic, no
        # branch on method identity.
        records_by_method = load_boundary_records(session_extract_dir)

        for method_name, method_records in records_by_method.items():
            method_output_dir = (
                output_dir / f"iff_{iff_metric}" / session_id / method_name
            )
            sentinel = method_output_dir / f'{session_id}_rf_profiles_done.json'

            if sentinel.exists() and not force_processing:
                print(
                    f"[RF Profile Extraction] {session_id}/{method_name}: "
                    "up-to-date, skipping."
                )
                continue

            npz_path = boundary_response_fields_npz_path(
                session_extract_dir, method_name, session_id,
            )
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"[RF Profile Extraction] {session_id}/{method_name}: NPZ not "
                    f"found at {npz_path} — run spatial_extract_boundaries first."
                )

            print(
                f"[RF Profile Extraction] {session_id}/{method_name}: processing..."
            )

            npz = np.load(npz_path, allow_pickle=True)

            if method_name not in methodology_written:
                _write_methodology_md(output_dir, method_name)
                methodology_written.add(method_name)

            _render_profiles_for_method(
                session_id=session_id,
                method_output_dir=method_output_dir,
                sentinel=sentinel,
                npz=npz,
                method_records=method_records,
            )
