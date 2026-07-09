"""Harmonic interpolation of per-vertex heatmap values onto a regular UV grid.

This is a **data-preparation** primitive (not rendering): it turns a per-vertex
population heatmap into the 150x150 ``grid_z`` field that every boundary detector
consumes. It lives in ``data/`` so both the preparation layer and the renderers
depend on it in the correct direction (rendering -> data).
"""

import logging
from collections import defaultdict

import igl
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import generic_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


def _fill_interior_face_holes(
    faces: np.ndarray,
    contacted_face: np.ndarray,
) -> np.ndarray:
    """Include uncontacted face clusters that are fully enclosed by contacted faces.

    Builds face-face adjacency, finds connected components of uncontacted faces,
    and reclassifies components that don't touch the mesh boundary as contacted
    (interior holes where the harmonic solution already has valid values).
    """
    n_faces = len(faces)
    if contacted_face.all():
        return contacted_face

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(n_faces):
        v0, v1, v2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        edge_to_faces[(min(v0, v1), max(v0, v1))].append(fi)
        edge_to_faces[(min(v1, v2), max(v1, v2))].append(fi)
        edge_to_faces[(min(v0, v2), max(v0, v2))].append(fi)

    rows, cols = [], []
    boundary_faces = set()
    for edge, flist in edge_to_faces.items():
        if len(flist) == 1:
            boundary_faces.add(flist[0])
        elif len(flist) == 2:
            rows.extend([flist[0], flist[1]])
            cols.extend([flist[1], flist[0]])

    adj = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_faces, n_faces),
    )

    uncontacted = ~contacted_face
    uncontacted_idx = np.where(uncontacted)[0]
    if len(uncontacted_idx) == 0:
        return contacted_face

    adj_sub = adj[np.ix_(uncontacted_idx, uncontacted_idx)]
    n_components, labels = connected_components(adj_sub, directed=False)

    filled = contacted_face.copy()
    for comp in range(n_components):
        comp_local = np.where(labels == comp)[0]
        comp_global = uncontacted_idx[comp_local]
        if not boundary_faces.intersection(comp_global):
            filled[comp_global] = True

    n_filled = int(filled.sum() - contacted_face.sum())
    if n_filled > 0:
        logger.info(
            "_fill_interior_face_holes: filled %d interior faces across %d hole(s).",
            n_filled,
            sum(
                1 for comp in range(n_components)
                if not boundary_faces.intersection(
                    uncontacted_idx[np.where(labels == comp)[0]]
                )
            ),
        )
    return filled


def interpolate_on_mesh(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    median_filter_size: int | None = None,
) -> np.ndarray:
    above_threshold_mask = np.isfinite(heatmap_val) & (heatmap_val >= 0.0)
    b = np.where(above_threshold_mask)[0]

    if len(b) < 4:
        raise ValueError(
            f"interpolate_on_mesh: harmonic interpolation requires at least 4 "
            f"contacted vertices, got {len(b)}. Gesture type may have too few contacts."
        )

    if median_filter_size is not None and (median_filter_size < 1 or median_filter_size % 2 == 0):
        raise ValueError(
            f"interpolate_on_mesh: median_filter_size must be a positive odd integer, "
            f"got {median_filter_size}."
        )

    bc = heatmap_val[b].reshape(-1, 1)

    filled = igl.harmonic(forearm_V, forearm_faces, b, bc, 1).ravel()

    # Build the unmasked triangulation for CubicTriInterpolator.  Passing a
    # face_mask to mtri.Triangulation triggers matplotlib's
    # TrapezoidMapTriFinder which can reject valid SLIM meshes with many
    # hole-fill faces (RuntimeError: "Triangulation is invalid").  Instead,
    # interpolate on the full mesh and NaN-out grid points that fall in faces
    # with no contacted vertex.
    contacted_face = np.any(above_threshold_mask[forearm_faces], axis=1)
    contacted_face = _fill_interior_face_holes(forearm_faces, contacted_face)

    tri = mtri.Triangulation(
        forearm_uv[:, 0], forearm_uv[:, 1], triangles=forearm_faces,
    )
    interp = mtri.CubicTriInterpolator(tri, filled, kind='min_E')

    grid_z_masked = interp(grid_u, grid_v)
    grid_z = grid_z_masked.data.copy()
    grid_z[grid_z_masked.mask] = np.nan

    trifinder = tri.get_trifinder()
    face_idx = trifinder(grid_u.ravel(), grid_v.ravel()).reshape(grid_u.shape)
    outside = (face_idx < 0) | ~contacted_face[np.clip(face_idx, 0, None)]
    grid_z[outside] = np.nan

    if np.all(np.isnan(grid_z)):
        raise ValueError(
            "interpolate_on_mesh: interpolation produced all-NaN output. "
            "All contacted vertices may fall outside the mesh."
        )

    grid_z = np.clip(grid_z, 0.0, None)

    if median_filter_size is not None:
        nan_mask = np.isnan(grid_z)
        grid_z = generic_filter(grid_z, np.nanmedian, size=median_filter_size)
        grid_z[nan_mask] = np.nan

    return grid_z


def compute_interpolated_grid(
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    heatmap_val: np.ndarray,
    median_filter_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 150x150 interpolation grid over the forearm UV extent and interpolate heatmap values.

    Returns
    -------
    (grid_u, grid_v, grid_z):
        grid_u, grid_v — meshgrid coordinate arrays (150, 150).
        grid_z — interpolated heatmap values (150, 150), NaN outside the mesh.
    """
    u_all = forearm_uv[:, 0]
    v_all = forearm_uv[:, 1]
    margin_u = (u_all.max() - u_all.min()) * 0.05 or 1.0
    margin_v = (v_all.max() - v_all.min()) * 0.05 or 1.0
    grid_u, grid_v = np.mgrid[
        u_all.min() - margin_u : u_all.max() + margin_u : 150j,
        v_all.min() - margin_v : v_all.max() + margin_v : 150j,
    ]
    grid_z = interpolate_on_mesh(
        forearm_uv, forearm_faces, forearm_V, heatmap_val, grid_u, grid_v,
        median_filter_size=median_filter_size,
    )
    return grid_u, grid_v, grid_z
