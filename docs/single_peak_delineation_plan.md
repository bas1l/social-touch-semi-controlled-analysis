# Delineating the Boundary of a Single Soft Peak — Implementation Plan

**Purpose of this document:** a self-contained brief for whoever implements the algorithm. It states the problem, explains why the obvious terrain-GIS methods do *not* apply, defines four candidate boundary definitions with their algorithms, and proposes a concrete reusable architecture. No prior context is assumed.

---

## 1. Problem statement

Given a 2-D scalar field (a "height" array) that contains **exactly one dominant peak** with **soft, gradual transitions** into the background, produce a **closed contour** that delineates the extent of that peak/hill.

Fixed constraints from the task owner:

- **One peak per input.** Each call receives a single mountain/hill, never a range. There is no second summit to separate from.
- **Soft transitions.** The peak asymptotes into the background; there is generally no cliff, scarp, or sharp break of slope.
- **Reusable algorithm.** Deliverable is a general function/pipeline, not a one-off analysis.
- The input field is treated as a generic 2-D scalar surface. The algorithm must not rely on domain-specific semantics (hydrology, geology, etc.).

---

## 2. The key conceptual point (read this before coding)

**A soft single peak has no objective boundary.** Because the surface decays smoothly into the background rather than terminating at a physical edge, *any* contour we draw is a **definition we impose**, not a feature we detect. Two reasonable people will draw different "feet" for the same hill and both can be correct.

Consequence: the algorithm's real job is **to encode one clear, reproducible definition and apply it consistently.** The engineering question is therefore *which definition*, not *how to find the true edge* (there isn't one).

Four defensible definitions are given in Section 4. They **coincide for a clean, symmetric bump** and **diverge for skewed, flat-topped, or heavy-tailed peaks** — which is exactly where the choice matters.

---

## 3. Methods explicitly ruled out (and why)

These are the standard terrain/landform-delineation approaches. They are documented here so the implementer does not waste time on them.

| Method | Why it does **not** apply here |
|---|---|
| **Topographic prominence / key-col / Morse–Smale peak domains** (Kirmse & de Ferranti) | Defines a peak's territory by the saddle separating it from *higher neighbouring peaks*. With one peak, there is no saddle and no neighbour — the concept is undefined. |
| **Watershed on inverted DEM** (catchment delineation) | Returns drainage *divides* between multiple basins. With a single peak it has nothing to partition against; the boundary would run off to the array edge. |
| **Mathematical morphology — conditional dilation with slope cutoff** (Soille-style, e.g. delete boundary pixels below ~6°) | Keys on an absolute slope threshold to catch a break of slope. A soft hill has no such break, so the boundary wanders and is highly threshold-sensitive. |
| **Geomorphons / TPI per-cell landform classes** | Built for whole-map classification into peak/ridge/slope/footslope/etc. Overkill and ill-posed for a single isolated soft bump. |

The reusable methods in Section 4 are the ones suited to a **single soft peak**.

---

## 4. Candidate boundary definitions (implement all four)

Each subsection gives the **cue**, the **algorithm**, **pros/cons**, and **when it wins**.

### 4.1 Geometric foot — radial curvature inflection
- **Cue:** the toe of the hill, where the flank flattens into the base.
- **Algorithm:** cast radial profiles outward from the peak (e.g. every 1–2°). Along each ray, slope rises, peaks on the flank, then decays; **profile curvature** goes convex near the summit, crosses zero at an inflection, then concave at the toe. Take, per ray, the radius of **maximum concave curvature** (or the convex→concave zero-crossing). Connect the per-ray radii into a closed polygon; smooth.
- **Pros:** most physically meaningful "edge of a hill"; keys on the *shape* of the bend, so it survives soft transitions; no absolute slope value needed.
- **Cons:** noisy on rough data (curvature amplifies noise → needs pre-smoothing); rays can pick spurious inflections on bumpy flanks.
- **Wins when:** the intent is literally "where the hill stops," and the flank-to-base bend is at least mildly expressed.

### 4.2 Spill point — superlevel-set area knee
- **Cue:** the level just before the hill "floods" into the background.
- **Algorithm:** sweep a threshold *t* downward from the summit. At each *t*, take the **connected component containing the peak** and record its area. On the flank, area grows slowly; once *t* passes below the base, area jumps sharply as the component floods outward. The **base level = the *t* at the knee** (max of dArea/d(−t), i.e. the spill point). Boundary = that superlevel-set contour.
- **Pros:** nearly parameter-free; topologically clean; directly encodes "the level below which this stops being a compact hill"; this is the special case of persistence (Section 5) that matters here.
- **Cons:** assumes a reasonably flat background; on a tilted background the flood happens asymmetrically and the knee blurs — **detrend first** (Section 6).
- **Wins when:** background is flattish and you want a robust, defensible, low-parameter default.

### 4.3 Scale-space blob — Laplacian-of-Gaussian (Lindeberg / Marr–Hildreth)
- **Cue:** the characteristic size of the blob, found in scale space.
- **Algorithm:** convolve the field with a **normalized LoG** (σ²∇²G) across a range of scales σ. The σ that **maximizes the LoG response** at the peak gives the blob's characteristic radius. Boundary = the **LoG zero-crossing ring** at that scale (or a k·σ contour).
- **Pros:** image-processing-native, domain-agnostic; **noise-robust** through the Gaussian; hands you a principled *size/radius* for free; well-understood theory.
- **Cons:** Gaussian model biases toward roughly circular blobs; the contour is a smoothed approximation, not the literal toe.
- **Wins when:** the domain is unknown / generic, noise is non-trivial, and a principled characteristic size is more useful than the exact toe. **Good first default for an unknown 2-D field.**

### 4.4 Parametric fit — fixed iso-contour (2-D FWHM)
- **Cue:** a fixed fraction of peak height above background.
- **Algorithm:** fit a parametric bump — 2-D Gaussian, or a **super-Gaussian / generalized radial profile** if the peak is flat-topped or heavy-tailed — then take the contour at **half-maximum** (the 2-D generalization of FWHM) or at n·σ.
- **Pros:** maximally reproducible; sub-pixel; very noise-tolerant; gives clean shape parameters (center, widths, orientation, ellipticity).
- **Cons:** imposes a shape model; a poor fit on an irregular hill yields a misleading contour. Always report fit residual / R².
- **Wins when:** reproducibility and parameterization matter more than honoring every wiggle of the real outline; good as a smooth **reference** contour alongside the others.

---

## 5. Supporting layer — topological persistence

Compute **persistence** on the superlevel sets to enforce the single-peak hypothesis rigorously. Each local maximum has a persistence (its birth-to-death lifetime as the threshold sweeps); the true peak has high persistence, while secondary bumps and noise spikes have low persistence and are **pruned automatically** below a `persistence_min` cutoff. Method 4.2 is the area-based special case of this framework. Use a persistence library (e.g. a cubical-complex / 2-D persistence implementation) or a hand-rolled union–find sweep over sorted pixels.

This makes the pipeline robust to "almost two peaks" inputs without special-casing them.

---

## 6. Recommended architecture

Do **not** hard-commit to one definition up front. Build a single reusable function that computes all candidates and returns them for comparison; the task owner then locks in whichever behaves best on real data.

```
delineate_peak(
    field,                       # 2-D numpy array (the scalar surface)
    peak=None,                   # (row, col); if None, auto = global max after smoothing
    methods=('curvature_foot',   # 4.1
             'spill_point',      # 4.2
             'log_blob',         # 4.3
             'iso_fit'),         # 4.4
    smoothing='auto',            # Gaussian sigma for pre-smoothing; 'auto' from noise est.
    detrend=True,                # remove a planar/low-order background tilt before sweeps
    persistence_min=None,        # prune secondary maxima below this lifetime
    iso_level=0.5,               # half-max for method 4.4
    n_rays=180,                  # angular sampling for method 4.1
) -> {
    'peak': (r, c),
    'contours': {                # one closed polygon (Nx2 array) per method
        'curvature_foot': ...,
        'spill_point': ...,
        'log_blob': ...,
        'iso_fit': ...,
    },
    'diagnostics': {             # for choosing/locking a method
        'log_characteristic_sigma': ...,
        'spill_level': ...,
        'fit_params': ..., 'fit_r2': ...,
        'agreement': ...,        # e.g. pairwise IoU between the four contours
    },
}
```

**Design rules:**
- **Agreement is signal.** When the four contours roughly agree (high pairwise IoU), the hill is clean and any definition is fine. Where they disagree, the disagreement *quantifies the asymmetry/heavy-tailed-ness* of the data — surface this in diagnostics.
- **Detrend before the sweep methods** (4.1, 4.2) if the background may tilt; fit and subtract a low-order plane/surface first.
- **Pre-smooth once**, consistently, and estimate noise (e.g. robust MAD on a background patch, or Laplacian-based noise estimate) to set `smoothing='auto'`.
- Return contours as closed vector polygons (sub-pixel where possible), not just raster masks, so they're reusable downstream.

---

## 7. Suggested implementation stack (Python)

- `numpy`, `scipy.ndimage` — smoothing (`gaussian_filter`), `gaussian_laplace` for 4.3, gradients/curvature for 4.1.
- `scikit-image` — `feature.peak_local_max` (peak auto-detect), `measure.find_contours` (marching squares for the iso/spill/zero-crossing rings), `measure.regionprops`.
- `scipy.optimize.curve_fit` (or `least_squares`) — the parametric fit in 4.4.
- A persistence package for Section 5 (cubical-complex 2-D persistence), or a custom union–find sweep.
- Radial-profile routine for 4.1: bilinear-sample the field along each ray, compute slope and curvature, locate the toe.

---

## 8. Parameters to pin down against real data

Two unknowns drive the defaults; confirm with the task owner or set defensively:

1. **Noise level** — sets the pre-smoothing σ and how aggressively to filter curvature. Default assumption if unspecified: *moderate noise* → light Gaussian pre-smooth.
2. **Background geometry** — flat vs. tilted/sloping. Decides whether the spill-point method (4.2) is reliable as-is or whether detrending is mandatory. Default assumption if unspecified: *possibly tilted* → `detrend=True`.

Defensive defaults (moderate noise + possibly-tilted background) are encoded above so the pipeline runs without these answers.

---

## 9. Deliverables checklist

- [ ] `delineate_peak()` implementing methods 4.1–4.4 with the signature in Section 6.
- [ ] Persistence pruning (Section 5) wired into peak selection.
- [ ] Detrending + noise-adaptive smoothing front-end.
- [ ] Diagnostics: characteristic size, spill level, fit quality, pairwise contour IoU.
- [ ] Overlay visualization: the four candidate contours on the field for visual method selection.
- [ ] Short note recommending which single definition to lock in, based on observed agreement/disagreement on the real data.

---

## 10. Key references

- Kirmse, A. & de Ferranti, J. (2017). *Calculating the prominence and isolation of every mountain in the world.* Progress in Physical Geography. — prominence / Morse–Smale peak domains (ruled out here, but the canonical multi-peak reference).
- Jasiewicz, J. & Stepinski, T. (2013). *Geomorphons — a pattern recognition approach to classification and mapping of landforms.* Geomorphology 182, 147–156. — per-cell landform classification (context).
- Soille, P. & Ansoult, M. (1990) and Soille, P. *Morphological Image Analysis.* — morphological/conditional-dilation delineation (context).
- Lindeberg, T. (1998). *Feature detection with automatic scale selection.* IJCV. — normalized LoG / scale-space blob (method 4.3).
- Marr, D. & Hildreth, E. (1980). *Theory of edge detection.* — LoG zero-crossings (method 4.3).
- Edelsbrunner, Letscher & Zomorodian (2002). *Topological persistence and simplification.* — persistence framework (Section 5).
