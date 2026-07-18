"""Parity check for the batched :func:`compute_stroke_uv_motion`.

The batched implementation projects *all* valid strokes' contact frames in a
single :func:`barycentric_uv_lookup` call (one KDTree build per session) instead
of one call per stroke. This test verifies the speedup is byte-for-byte
equivalent to the old per-stroke path by re-deriving the same quantities with an
independent per-stroke reference and asserting array equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.pipeline.shared_constants import TOUCH_ID_COLS  # noqa: E402
from analysis.receptive_field_mapping.metrics.rf_stroke_axis import (  # noqa: E402
    compute_stroke_uv_motion,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (  # noqa: E402
    SlimUvCache,
    barycentric_uv_lookup,
)

_CONTACT_COLS = ("contact_location_x", "contact_location_y", "contact_location_z")


def _synthetic_cache() -> SlimUvCache:
    """A small flat grid mesh with ``uv == (x, y)`` so lookups are well-defined."""
    xs = np.linspace(-1.0, 1.0, 6)
    ys = np.linspace(-1.0, 1.0, 6)
    gx, gy = np.meshgrid(xs, ys)
    V = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]).astype(np.float64)
    uv = V[:, :2].copy()
    # Two triangles per grid cell.
    faces = []
    n = len(xs)
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            b = a + 1
            d = a + n
            e = d + 1
            faces.append([a, b, d])
            faces.append([b, e, d])
    F = np.asarray(faces, dtype=np.int32)
    return SlimUvCache(
        V=V, F=F, uv=uv, center_vid=0, boundary_vid=1,
        ply_mtime=0.0, ply_hash="", rf_npz_mtime=0.0,
        centroid_3d=np.zeros(3, dtype=np.float64),
    )


def _synthetic_df() -> pd.DataFrame:
    """Three strokes: two valid (moving lines), one with a single valid frame."""
    rows = []

    def add(block, trial, touch, gesture, xyzs):
        for (x, y, z) in xyzs:
            rows.append({
                "block_order_id": block, "trial_id": trial,
                "single_touch_id": touch, "gesture_type": gesture,
                "type_metadata": "stroke",
                "contact_location_x": x, "contact_location_y": y,
                "contact_location_z": z,
            })

    # Stroke A — proximal-ish, +x motion.
    add(1, 1, 1, "stroke_proximal",
        [(-0.8, 0.1, 0.0), (-0.2, 0.15, 0.0), (0.4, 0.2, 0.0), (0.7, 0.25, 0.0)])
    # Stroke B — distal-ish, -x motion, with one NaN frame (dropped, still >= 2).
    add(1, 1, 2, "stroke_distal",
        [(0.6, -0.3, 0.0), (np.nan, np.nan, np.nan), (0.0, -0.2, 0.0),
         (-0.5, -0.1, 0.0)])
    # Stroke C — only one valid frame -> invalid (UV 'stroke_unknown').
    add(1, 2, 1, "stroke_proximal",
        [(0.1, 0.1, 0.0), (np.nan, np.nan, np.nan)])
    # A tap (single_touch_id keeps 0 -> excluded) to exercise the filter.
    rows.append({
        "block_order_id": 1, "trial_id": 2, "single_touch_id": 0,
        "gesture_type": "tap", "type_metadata": "tap",
        "contact_location_x": 0.0, "contact_location_y": 0.0,
        "contact_location_z": 0.0,
    })
    return pd.DataFrame(rows)


def _reference_motion(cache: SlimUvCache, df: pd.DataFrame):
    """Independent per-stroke reference (the pre-batching algorithm)."""
    touch_df = df[df["single_touch_id"] != 0]
    stroke_df = touch_df[touch_df["type_metadata"] == "stroke"]
    starts, ends, vectors, centroids, valids = [], [], [], [], []
    nan_uv = [float("nan"), float("nan")]
    for _key, group in stroke_df.groupby(list(TOUCH_ID_COLS), sort=False):
        valid_group = group.dropna(subset=list(_CONTACT_COLS))
        n = len(valid_group)
        if n < 2:
            starts.append(nan_uv[:]); ends.append(nan_uv[:])
            vectors.append(nan_uv[:]); centroids.append(nan_uv[:])
            valids.append(False)
            continue
        xyz = valid_group[list(_CONTACT_COLS)].to_numpy(dtype=np.float64)
        uv = barycentric_uv_lookup(cache, xyz)
        frames = np.arange(n, dtype=np.float64)
        su, iu = np.polyfit(frames, uv[:, 0], 1)
        sv, iv = np.polyfit(frames, uv[:, 1], 1)
        last = float(n - 1)
        start = [float(iu), float(iv)]
        end = [float(iu + su * last), float(iv + sv * last)]
        starts.append(start); ends.append(end)
        vectors.append([end[0] - start[0], end[1] - start[1]])
        centroids.append([float(uv[:, 0].mean()), float(uv[:, 1].mean())])
        valids.append(True)
    return (
        np.asarray(starts), np.asarray(ends), np.asarray(vectors),
        np.asarray(centroids), np.asarray(valids, dtype=bool),
    )


def test_batched_compute_matches_per_stroke_reference():
    cache = _synthetic_cache()
    df = _synthetic_df()

    motion = compute_stroke_uv_motion(cache, df)
    r_start, r_end, r_vec, r_cent, r_valid = _reference_motion(cache, df)

    np.testing.assert_array_equal(motion.valid_mask, r_valid)
    # NaN-aware exact equality (same NaN placement + identical finite values).
    np.testing.assert_array_equal(motion.start_uv, r_start)
    np.testing.assert_array_equal(motion.end_uv, r_end)
    np.testing.assert_array_equal(motion.vectors, r_vec)
    np.testing.assert_array_equal(motion.centroid_uv, r_cent)

    # Structural sanity: 3 strokes, 2 valid, tap excluded.
    assert motion.touch_keys.shape == (3, 3)
    assert int(np.count_nonzero(motion.valid_mask)) == 2


if __name__ == "__main__":
    test_batched_compute_matches_per_stroke_reference()
    print("parity OK")
