"""Rendering layer for the ``spatial_extract_boundaries`` stage.

Consumes the already-computed :class:`PreparedBoundaryData` +
:class:`BoundaryResults` and emits the PNG figures. It **never** recomputes
grids or boundaries — the Pass-2 composites read the same grids and active
boundaries that Pass-1 produced.

Two entry points:

* :func:`render_session_figures` — per-session, per-gesture heatmap PNGs
  (session-local colour scale).
* :func:`render_global_composites` — cross-session composites, standalones,
  colorbars and circular crops on a shared global colour scale + UV extent.
  Appends every produced path to each session's ``produced`` list in place.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from analysis.receptive_field_mapping.data.rf_boundary_types import (
    BoundaryParams,
    BoundaryResults,
    PreparedBoundaryData,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_highest_contour_center,
    compute_standalone_figwidth,
    compute_uv_to_mm_scale,
    render_population_rf_circular_crop,
    render_population_rf_colorbar,
    render_population_rf_composite,
    render_population_rf_map,
    render_population_rf_standalone_interpolated,
)

logger = logging.getLogger(__name__)

# A bundle carries one session's prepared data, its boundary results, and the
# mutable list of produced PNG paths (extended in place by the render passes).
SessionBundle = Tuple[PreparedBoundaryData, BoundaryResults, List[Path]]


def render_session_figures(
    prepared: PreparedBoundaryData,
    results: BoundaryResults,
    params: BoundaryParams,
) -> List[Path]:
    """Render per-gesture heatmap PNGs (session-local colour scale) into ``aggregated/``."""
    session_id = prepared.session_id
    aggregated_dir = prepared.output_dir / "aggregated"
    produced: List[Path] = []

    for gtype, (slim_heatmap, n_touches, threshold) in prepared.results.items():
        title = (
            f"{session_id} | {gtype} | {n_touches} touches | "
            f"threshold={threshold} ({prepared.min_overlap_pct:.0f}%)"
        )
        png_path = aggregated_dir / f'{session_id}_rf_population_{gtype}.png'

        print(f"[Population Response Fields] {session_id}: rendering '{gtype}'...")
        grid_u, grid_v, grid_z = prepared.per_gesture_grids[gtype]
        render_population_rf_map(
            forearm_uv=prepared.forearm_uv,
            heatmap_val=slim_heatmap,
            vmax=prepared.session_vmax,
            vmin=prepared.session_vmin,
            title=title,
            output_path=png_path,
            forearm_faces=prepared.slim_faces,
            forearm_V=prepared.slim_V,
            median_filter_size=params.median_filter_size,
            precomputed_grid=(grid_u, grid_v, grid_z),
            boundary=results.gesture_boundaries[gtype],
            heatmap_space=params.heatmap_space,
            cmap=params.cmap,
            vertex_colors=prepared.slim_vertex_colors,
            contour_color=params.contour_color,
        )
        produced.append(png_path)
        print(f"[Population Response Fields] {session_id}: saved {png_path.name}")

    return produced


def render_global_composites(
    bundles: List[SessionBundle],
    params: BoundaryParams,
) -> None:
    """Render composites/standalones/colorbars/crops on a shared global scale.

    Mutates each bundle's ``produced`` list in place.
    """
    if not bundles:
        return

    # Pre-compute all standalone titles and measure the longest to get a shared
    # figure width, so every *_interpolated.png has identical pixel dimensions.
    standalone_titles: dict[tuple[str, str], str] = {}
    for prepared, _results, _produced in bundles:
        for gtype, (_, n_touches, threshold) in prepared.results.items():
            standalone_titles[(prepared.session_id, gtype)] = (
                f"{prepared.session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({prepared.min_overlap_pct:.0f}%)"
            )
    longest_title = max(standalone_titles.values(), key=len) if standalone_titles else ""
    standalone_figwidth = compute_standalone_figwidth(longest_title) if longest_title else 6.0

    global_vmax = max(prepared.session_vmax for prepared, _r, _p in bundles)
    _all_grid_positive = [
        float(grid_z[np.isfinite(grid_z) & (grid_z > 0)].min())
        for prepared, _r, _p in bundles
        for _, (_, _, grid_z) in prepared.per_gesture_grids.items()
        if np.any(np.isfinite(grid_z) & (grid_z > 0))
    ]
    global_vmin = min(_all_grid_positive) if _all_grid_positive else global_vmax * 1e-3
    all_u = np.concatenate([prepared.forearm_uv[:, 0] for prepared, _r, _p in bundles])
    all_v = np.concatenate([prepared.forearm_uv[:, 1] for prepared, _r, _p in bundles])
    uv_margin = 0.02
    u_range = all_u.max() - all_u.min()
    v_range = all_v.max() - all_v.min()
    global_uv_xlim = (float(all_u.min() - uv_margin * u_range),
                      float(all_u.max() + uv_margin * u_range))
    global_uv_ylim = (float(all_v.min() - uv_margin * v_range),
                      float(all_v.max() + uv_margin * v_range))
    del all_u, all_v

    print(
        f"[Population Response Fields] Rendering composites: global_vmax={global_vmax:.2f}, "
        f"U=[{global_uv_xlim[0]:.1f}, {global_uv_xlim[1]:.1f}], "
        f"V=[{global_uv_ylim[0]:.1f}, {global_uv_ylim[1]:.1f}]"
    )

    for bundle in bundles:
        _render_one_session_composites(
            bundle, params,
            global_vmax=global_vmax, global_vmin=global_vmin,
            global_uv_xlim=global_uv_xlim, global_uv_ylim=global_uv_ylim,
            standalone_titles=standalone_titles, standalone_figwidth=standalone_figwidth,
        )


def _render_one_session_composites(
    bundle: SessionBundle,
    params: BoundaryParams,
    *,
    global_vmax: float,
    global_vmin: float,
    global_uv_xlim: tuple[float, float],
    global_uv_ylim: tuple[float, float],
    standalone_titles: dict,
    standalone_figwidth: float,
) -> None:
    prepared, results, produced = bundle
    session_id = prepared.session_id
    aggregated_dir = prepared.output_dir / "aggregated"

    for panel_type in ('scatter', 'interpolated'):
        composite_path = (
            aggregated_dir / f'{session_id}_rf_population_{panel_type}_composite.png'
        )
        print(
            f"[Population Response Fields] {session_id}: "
            f"rendering '{panel_type}' composite..."
        )
        if panel_type == 'interpolated':
            # Reuse the grids + active boundaries computed in prepare/process —
            # no recomputation.
            precomputed_grids = dict(prepared.per_gesture_grids)
            boundaries = {
                gtype: results.gesture_boundaries.get(gtype)
                for gtype in prepared.results
            }
        else:
            precomputed_grids = None
            boundaries = None
        render_population_rf_composite(
            forearm_uv=prepared.forearm_uv,
            results=prepared.results,
            vmax=global_vmax,
            vmin=global_vmin,
            session_id=session_id,
            panel_type=panel_type,
            output_path=composite_path,
            min_overlap_pct=prepared.min_overlap_pct,
            uv_xlim=global_uv_xlim,
            uv_ylim=global_uv_ylim,
            forearm_faces=prepared.slim_faces,
            forearm_V=prepared.slim_V,
            median_filter_size=params.median_filter_size,
            precomputed_grids=precomputed_grids,
            boundaries=boundaries,
            heatmap_space=params.heatmap_space,
            cmap=params.cmap,
            vertex_colors=prepared.slim_vertex_colors,
            contour_color=params.contour_color,
        )
        produced.append(composite_path)
        print(
            f"[Population Response Fields] {session_id}: saved {composite_path.name}"
        )

    for gtype, (grid_u, grid_v, grid_z) in prepared.per_gesture_grids.items():
        boundary = results.gesture_boundaries.get(gtype)
        standalone_path = prepared.output_dir / f'{session_id}_rf_population_{gtype}_interpolated.png'
        render_population_rf_standalone_interpolated(
            u_grid=grid_u,
            v_grid=grid_v,
            interp_grid=grid_z,
            forearm_uv=prepared.forearm_uv,
            boundary_u=boundary.contour_uv[:, 0] if boundary is not None else None,
            boundary_v=boundary.contour_uv[:, 1] if boundary is not None else None,
            output_path=standalone_path,
            vmax=global_vmax,
            vmin=global_vmin,
            title=standalone_titles[(session_id, gtype)],
            figwidth=standalone_figwidth,
            xlim=global_uv_xlim,
            ylim=global_uv_ylim,
            heatmap_space=params.heatmap_space,
            cmap=params.cmap,
            vertex_colors=prepared.slim_vertex_colors,
            forearm_faces=prepared.slim_faces,
            contour_color=params.contour_color,
        )
        produced.append(standalone_path)
        print(f"[Population Response Fields] {session_id}: saved {standalone_path.name}")

    colorbar_path = prepared.output_dir / f'{session_id}_rf_population_colorbar.png'
    render_population_rf_colorbar(
        output_path=colorbar_path, vmax=global_vmax, vmin=global_vmin,
        heatmap_space=params.heatmap_space, cmap=params.cmap,
    )
    produced.append(colorbar_path)
    print(f"[Population Response Fields] {session_id}: saved {colorbar_path.name}")

    colorbar_local_path = prepared.output_dir / f'{session_id}_rf_population_colorbar_local.png'
    render_population_rf_colorbar(
        output_path=colorbar_local_path,
        vmax=prepared.session_vmax,
        vmin=prepared.session_vmin,
        heatmap_space=params.heatmap_space,
        cmap=params.cmap,
    )
    produced.append(colorbar_local_path)
    print(f"[Population Response Fields] {session_id}: saved {colorbar_local_path.name}")

    all_boundary = results.gesture_boundaries.get('all')
    if all_boundary is None:
        raise ValueError(
            "render_population_rf_circular_crop requires a boundary for gesture 'all' "
            "but none is available — ensure inflection_sigma is configured"
        )
    all_grid_u, all_grid_v, all_grid_z = prepared.per_gesture_grids['all']

    centroid_uv = np.array(all_boundary.centroid_uv)
    peak_uv = np.array(all_boundary.peak_uv)
    contour_center_uv = compute_highest_contour_center(
        all_grid_u, all_grid_v, all_grid_z, n_levels=6,
    )

    crop_jobs: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ('centroid', centroid_uv, centroid_uv),
        ('peak', peak_uv, peak_uv),
    ]
    if contour_center_uv is not None:
        crop_jobs.append(('contour_center', contour_center_uv, contour_center_uv))

    radius_mm = 50.0
    scale = compute_uv_to_mm_scale(prepared.forearm_uv, prepared.slim_V, prepared.slim_faces)
    radius_uv = radius_mm / scale
    margin = radius_uv * params.circular_crop_margin
    all_centers = np.array([c for _, c, _ in crop_jobs])
    shared_xlim = (
        float(all_centers[:, 0].min()) - radius_uv - margin,
        float(all_centers[:, 0].max()) + radius_uv + margin,
    )
    shared_ylim = (
        float(all_centers[:, 1].min()) - radius_uv - margin,
        float(all_centers[:, 1].max()) + radius_uv + margin,
    )

    for center_label, center_uv, marker_uv in crop_jobs:
        for vmax_val, vmin_val, suffix in (
            (global_vmax, global_vmin, ''),
            (prepared.session_vmax, prepared.session_vmin, '_local'),
        ):
            for use_marker, marker_suffix in ((True, ''), (False, '_clean')):
                out_path = (
                    prepared.output_dir
                    / f'{session_id}_rf_population_all_circular_{center_label}{marker_suffix}{suffix}.png'
                )
                print(
                    f"[Population Response Fields] {session_id}: "
                    f"rendering 'all' circular {center_label}{marker_suffix}{suffix}..."
                )
                crop_kwargs = dict(
                    u_grid=all_grid_u,
                    v_grid=all_grid_v,
                    interp_grid=all_grid_z,
                    forearm_uv=prepared.forearm_uv,
                    forearm_V=prepared.slim_V,
                    forearm_faces=prepared.slim_faces,
                    center_uv=center_uv,
                    radius_mm=radius_mm,
                    vmax=vmax_val,
                    vmin=vmin_val,
                    vertex_colors=prepared.slim_vertex_colors,
                    heatmap_space=params.heatmap_space,
                    cmap=params.cmap,
                    contour_levels=6,
                    centroid_uv=marker_uv if use_marker else None,
                    xlim=shared_xlim,
                    ylim=shared_ylim,
                )
                # Original crop — preserved unchanged (no RF boundary overlay).
                render_population_rf_circular_crop(output_path=out_path, **crop_kwargs)
                produced.append(out_path)
                print(f"[Population Response Fields] {session_id}: saved {out_path.name}")
                # Duplicate crop with the red closed RF boundary overlaid.
                boundary_path = out_path.with_name(f'{out_path.stem}_rfboundary{out_path.suffix}')
                render_population_rf_circular_crop(
                    output_path=boundary_path,
                    boundary_contour_uv=all_boundary.contour_uv,
                    boundary_color=params.contour_color,
                    **crop_kwargs,
                )
                produced.append(boundary_path)
                print(f"[Population Response Fields] {session_id}: saved {boundary_path.name}")
