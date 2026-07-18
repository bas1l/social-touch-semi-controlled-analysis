# Brainstorm: Data-edge RF boundaries (peak at the region border)

**Started:** 2026-07-18   **Last matured:** 2026-07-18   **Status:** Brainstorming

## Real goal (north star)
Produce a **clean, defensible RF boundary when the peak / RF sits at the edge of
the mapped forearm region**, where the radial foot-of-mountain method structurally
fails (it assumes an interior peak surrounded by field on all sides). User reports
this is "not uncommon." Must NOT fabricate unobserved RF — scientific honesty +
this repo's fail-fast/no-fabrication ethos.

## How the failure arises (grounded)
Radial method shoots rays outward from the peak in all directions, finds the
curvature "foot" per ray, closes the ring. At a border, ~half the rays leave the
data immediately → one-sided/collapsed ring → doesn't enclose the edge peak → the
seed falls outside the traced region → `SEED_OUTSIDE_FOOTPRINT` (Site 5). NOTE the
footprint can be LARGE (observed 7605 cells) — it is the peak's *position* at the
edge, not the footprint's size. So "grow the footprint" advice is wrong here.

## The crux (decides "clean")
Is the mapped-region boundary a **real edge** or an **arbitrary crop**?
- Real edge (surface genuinely ends / unstimulable past it): RF is genuinely
  bounded there; border chord is a REAL boundary; area honest. -> close along border.
- Arbitrary crop (RF continues past edge): any contour is CENSORED/partial; honestly
  represent only the observed part + flag as lower-bound; never invent the rest.
(May vary per session.)

## Alternatives on the table
- **1. Boundary-closed foot arc** — foot on data-bearing rays + close along region
  border. Same boundary definition as interior RFs; honestly clipped. Most consistent.
- **2. Footprint/level-set fallback** — use the min_overlap threshold region (already
  a closed area) clipped at edge, instead of the curvature ring. Reuses machinery
  BUT different boundary definition -> border vs interior RF areas not comparable.
- **3. Mirror-pad then radial** — reflect field to make peak interior. FABRICATES the
  unobserved half. Rejected on honesty grounds unless a symmetry argument exists.
- **4. Flag + censor** — tag RF as border/censored so downstream area/circularity/
  centroid interpret correctly (area = lower bound; circularity meaningless). Not a
  boundary method; the honesty bookkeeping any of 1-2 needs.
Leading guess: (1 or 2) + 4; choice hinges on the crux + cross-RF comparability.

## Open questions
- Crux: real edge vs arbitrary crop (per session?).
- What "clean" means: valid closed polygon / defensible area number / just not-broken.
- Downstream consumers of the contour (area, circularity, centroid, overlap?) — decides
  what the representation must preserve + whether the censored flag matters.
- Prevalence of border RFs (handful vs meaningful fraction).
- Cross-RF comparability requirement: must border-RF areas be comparable to interior?

## Relation to other work
- Follows directly from [[rf-contour-param-sweep]] (the diagnostic that surfaced the
  border case as SEED_OUTSIDE_FOOTPRINT). That feature is implemented (uncommitted on
  `feature/rf-contour-failure-diagnostics`); this is a separate, later problem.
- Possible small tie-in: add a border-aware branch to the diagnostic so the message
  says "peak at data border" instead of "grow the footprint."

## Session log
- 2026-07-18: Spun off from the failure-diagnostic work. Framed the honest-partial-vs-
  estimated-whole crux (real edge vs arbitrary crop); listed 4 candidate forms
  (boundary-closed arc / footprint fallback / mirror-pad [reject] / flag+censor);
  posed pinning questions (clean-definition, downstream metrics, prevalence, comparability).
