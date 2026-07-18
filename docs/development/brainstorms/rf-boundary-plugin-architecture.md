# Brainstorm: Pluggable multi-algorithm RF boundary extraction

**Started:** 2026-07-18   **Last matured:** 2026-07-18   **Status:** Matured — handed off to /plan-create

## Real goal (north star)
Make RF boundary/contour extraction a **plug-in point**: one formalized contract, N interchangeable
algorithms, each independently developable (parallel tasks), each individually toggleable in the GUI as
its own box, all feeding the same downstream consumers — singly or as a list. Adding a 5th/6th algorithm
must require **zero edits to shared code**.

## Where it stands
Grounding map (background agent) revealed the contract already exists *implicitly*: there are already
**three** detectors today — gradient (1st-derivative, older), inflection (Laplacian zero-crossing), and
radial/Hessian (λmax, current default). All share:
- detector signature `(grid_u, grid_v, grid_z) -> boundary | None`
- an identical **11-field** boundary object (`contour_uv`, `area_uv`, `perimeter_uv`, `circularity`,
  `centroid_uv`, `peak_uv`, `pca_major_uv`, `pca_minor_uv`, `pca_orientation_deg`,
  `mean_iff_on_contour`, `iff_at_centroid`)
- a single generic `boundary_` NPZ prefix that **all downstream comparison consumers already read**
  method-blind.

So the work is NOT "invent a contract" — it's **formalize the de-facto contract into a base class +
registry, and delete the ~9 hardcoded method-list coupling points.** This is de-duplication into a
plug-in seam, not new capability.

Confirmed with user: **World 1** — every method present and planned outputs one comparable geometric
object (a boundary curve). No soft-field / multi-lobe / capability-negotiation machinery needed.

## The ~9 coupling points to remove (from map)
1. `metrics/rf_boundary_extraction.py:54-114` — which-detectors-run guards + method-selection if/elif
2. `pipelines/rf_boundary_verification.py:18` — `_VALID_BOUNDARY_METHODS` tuple
3. `data/rf_boundary_types.py` — `BoundaryResults` one-dict-per-method; `BoundaryParams` radial-only scalars; no polymorphic params
4. No boundary base class — 3 dataclasses duplicate the 11 fields + own `*_to_dict`
5. `data/rf_boundary_io.py:284-361` — one persistence block per method
6. GUI `task_detail_panel.py:133-191` — method enum, `_OPTION_VISIBILITY`, `_OPTION_GROUP_OF`, `_GROUPED_TASKS`
7. `scripts/analysis_workflow_processing.py:1920-1937` — literal per-method option wiring
8. Radial-only tuning subsystem — `gui/rf_contour_tuning_viewer.py`, `data/rf_contour_params_io.py`, `GestureContourParams`/`ContourParamToggles`
9. Downstream method-specific NPZ keys — `rf_profile_extraction_pipeline.py:355,394` reads `gradient_contour_uv_`, `radial_lmax_` by literal name

## Decisions locked
- **World 1** — every method outputs one comparable boundary-curve object.
- **Extras (`lmax`, gradient field) are rendering-only** — cut from the compute contract; become an
  optional named *diagnostic field* a method may emit for its own figure overlays. Downstream computation
  never reads them; profile stage stops reaching for `radial_lmax_`/`gradient_contour_uv_` by literal name.
  (pending final user nod — user leaned World 1 / comparable geometric outputs, consistent with cutting)
- **Fan-out, stopping at extraction.** Each method = its own DAG node + GUI box (internal node, NOT a
  leaf), with an output edge into the downstream consumer stages.
- **Downstream consumers stay single nodes** — each discovers whichever `<method>/` output dirs exist and
  loops over them internally, emitting per-method sub-outputs. Adding a method adds exactly ONE box.
- **Per-method output namespacing** — inverts today's single-file model:
  `4_analysed/spatial_extract_boundaries/<session_id>/<method>/<session_id>_boundary.npz`. One generic
  contract-shaped writer called once per method; no shared file, no cross-method overwrite.

## Alternatives on the table
- **(A) Strict core + opaque extras channel** — contract = 11 geometric fields; method-specific scalar
  fields (`lmax`, `gradient_magnitude`) ride in a generic `extras: dict`, read by key only. *Chosen (as
  diagnostic-only).*
- **(B) Typed capabilities** — rejected: World 1 makes it overkill.
- Propagation **(2) fan-out all the way down** (methods × consumers nodes) — rejected: proliferates DAG,
  every new method touches every consumer. Chose **(1) stop at extraction**.

## Threads explored
- World 1 vs World 2 (comparable-curve vs structurally-different output) — **resolved: World 1.**
- Contract already implicit in codebase — **kept**, reframes work as formalization/dedup.

## Decisions locked (cont.)
- **Config registry, explicit not magic.** Single declared `boundary_methods:` list in DAG config is the
  one source of truth. No self-registration decorators. Downstream MAY know method names (it's just
  iterating the list) but the code path is identical per method — **invariant: no `if method == "..."`
  anywhere except the method's own extraction module.**
- **GUI is the primary surface** — renders one box per declared method; the method set + wiring is visible,
  not buried in code.
- **One box per method BECAUSE inputs differ.** Each declared method carries its own **params schema**
  (name/type/default/range/tunable-flag); GUI renders per-method options from that schema, replacing the
  hardcoded `boundary_method` enum + `_OPTION_VISIBILITY`/`_OPTION_GROUP_OF` maps (gaps #6, #8).
- **Downstream edge-wiring** — consumers depend on the boundary-extraction **group/tag**, not on individual
  methods; a new method joins the group and downstream auto-depends. No consumer edit.

## Open questions
- All resolved for this pass. Tuner: **radial-only this pass** — generic param *rendering* for all methods
  is in-scope; the heavyweight live tuning viewer stays wired to radial only, generalize later when a
  method earns it.

## Matured brief (for /plan-create)
Formalize the already-implicit boundary contract and make boundary extraction a config-declared plug-in
point:
1. **Contract** — a boundary base class capturing the 11 uniform geometric fields; the 3 existing
   dataclasses (radial/gradient/inflection) become subclasses. Method-specific scalar fields
   (`lmax`, `gradient_magnitude`) demoted to an optional **diagnostic-field** channel used only for figure
   overlays — no compute path reads them; profile stage stops reading `radial_lmax_`/`gradient_contour_uv_`
   by literal name.
2. **Registry** — single declared `boundary_methods:` list (DAG config) as the one source of truth; each
   entry carries the method's params schema. Removes the ~9 hardcoded method-list coupling points.
   Invariant: no `if method == "..."` outside a method's own extraction module.
3. **Fan-out, stop at extraction** — each method is its own DAG node + GUI box (internal node with an
   output edge to consumers), writing its own tree
   `4_analysed/spatial_extract_boundaries/<session_id>/<method>/<session_id>_boundary.npz`. One generic
   contract-shaped NPZ writer, called once per method (collapses the per-method blocks).
4. **Uniform downstream** — consumers stay single nodes, depend on the boundary-extraction group/tag, and
   discover + loop over whichever `<method>/` dirs exist, emitting per-method sub-outputs. Identical code
   per method.
5. **GUI** — one box per declared method rendering that method's own params schema (replaces the hardcoded
   `boundary_method` enum + visibility/group maps). GUI is the primary surface for seeing the method set.
6. **Tuner** — the live interactive tuning viewer stays radial-only this pass.
Out of scope: choosing/implementing the two future algorithms; generalizing the live tuner.

## Session log
- 2026-07-18 — Mapped current wiring; confirmed contract is already implicit (3 methods, 11 fields,
  generic `boundary_` prefix). User confirmed World 1. Next fork: extras channel vs pure-geometric.
- 2026-07-18 — Resolved every fork: extras→diagnostic-only, list→fan-out stopping at extraction, method=
  node/box with output edge to consumers, per-method output folders, config-declared registry (explicit),
  uniform downstream looping over method dirs, per-method params schemas rendered in GUI, live tuner stays
  radial-only. Status → Matured, handed off to /plan-create.
