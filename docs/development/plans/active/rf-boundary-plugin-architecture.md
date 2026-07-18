# Plan: Pluggable Multi-Algorithm RF Boundary Extraction

**Date:** 2026-07-18
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/rf-boundary-plugin-architecture`

---

## Overview

Turn RF boundary/contour extraction into a **config-declared plug-in point**: one formalized boundary
contract, N interchangeable algorithms (today: gradient, inflection, radial/Hessian), each an independent
DAG node + GUI box that writes to its own output folder, all feeding the same downstream consumers via a
fan-in barrier. The de-facto contract already present in the codebase (three detectors, an identical
11-field boundary object, one generic `boundary_` NPZ prefix that all consumers already read) is
formalized into a base class + code registry, and the ~9 hardcoded method-list coupling points are
removed so a new algorithm can be added by dropping one subclass + one YAML node, editing no shared code.

## Problem Statement

Adding a boundary-extraction algorithm today requires editing ~9 places that hardcode the method list:
the `if/elif` selector in `extract_session_boundaries`, the `_VALID_BOUNDARY_METHODS` tuple, one dict per
method in `BoundaryResults`, one persistence block per method in the NPZ writer, three duplicated
`*_to_dict` serializers, the GUI's `boundary_method` enum + `_OPTION_VISIBILITY` + `_OPTION_GROUP_OF`
maps, and the literal per-method option wiring in the Prefect script. The three existing detectors already
share an identical output shape and are already consumed method-blind downstream — the uniformity exists,
but nothing formalizes or enforces it, and the coupling makes parallel development of new methods (the
stated near-term goal: two more algorithms, chosen later) touch shared code and collide. There is also no
way to run more than one method at a time and compare their outputs, because all methods currently write
into one shared NPZ file under one prefix.

## Goals

### In Scope
1. **Formalize the boundary contract** — a `BoundaryContour` base class (frozen) capturing the 11 uniform
   geometric fields; the three existing dataclasses become subclasses; a single validator enforces the
   contract at the fan-in boundary.
2. **Demote method-specific scalar fields to a diagnostic-only channel** — `lmax` (radial),
   `gradient_magnitude` (gradient) move into an optional `diagnostic_fields: dict[str, np.ndarray]` used
   only for figure overlays; no compute path reads them; the profile stage stops reading `radial_lmax_`
   and `gradient_contour_uv_` by literal name.
3. **Introduce a code registry (`BoundaryMethod` ABC + factory)** — the authoritative set of algorithms,
   each declaring `name`, a **params schema**, a `compute(...) -> BoundaryContour | None`, and its output
   contract. A factory dispatches by name — the only place algorithm identity is branched on.
4. **Fan-out, stopping at extraction** — one generic extraction flow parameterized by `method`, driven
   per method node; each method writes its own tree
   `4_analysed/spatial_extract_boundaries/<session_id>/<method>/<session_id>_boundary.npz` via one generic
   contract-shaped writer; each method renders its own inspection figures into its folder.
5. **Barrier fan-in** — retain `spatial_extract_boundaries` as a light join node depending on all enabled
   per-method nodes; downstream consumers keep `depends_on: [spatial_extract_boundaries]` unchanged.
6. **Uniform downstream** — each consumer stays a single node that discovers whichever `<method>/` dirs
   exist and loops over them, emitting per-method sub-outputs; identical code per method.
7. **Registry-driven GUI + Prefect wiring** — the GUI renders one box per method node from the method's
   declared params schema (replacing the hardcoded enum/visibility/group maps for boundary options);
   `_build_pipeline_stages` iterates the registry to append one descriptor per method node.
8. **Migrate the three existing methods** onto the new architecture with output parity (byte-for-byte
   equivalence of the geometric fields for the radial default).

### Out of Scope
- Choosing or implementing the two future algorithms (this pass delivers the seam; new methods are added
  later against it).
- Generalizing the live interactive tuning viewer — the heavyweight `rf_contour_tuning_viewer` stays
  radial-only this pass; only generic *param rendering* (static controls) is generalized.
- Cross-method global composite figures (comparing methods side-by-side in one figure).
- Activating the DAG `category:` field as a dependency-group mechanism (would touch the shared vendored
  resolver; rejected — see Alternatives).
- Any change to the upstream `spatial_build_response_fields` param-free NPZ contract.

## Success Criteria

- [ ] A new boundary algorithm can be added by (a) adding one `BoundaryMethod` subclass file and (b)
      adding one DAG node + one line to the barrier's `depends_on` + one layout coordinate — **editing no
      other shared code** (verified by grepping that no `if method == "..."` / method-name literal exists
      outside subclass modules and the registry).
- [ ] `grep -rn 'radial_lmax_\|gradient_contour_uv_'` returns no hits in downstream consumer code; the
      profile stage reads only the generic contract + the `diagnostic_fields` channel by capability, not
      by method name.
- [ ] Running two methods simultaneously produces two non-overlapping output trees under
      `.../<session_id>/<method>/`, neither overwriting the other; re-running one method overwrites only
      its own folder.
- [ ] Every registered method passes one shared **contract conformance test-suite** (same tests, each
      concrete method).
- [ ] The radial (default) method produces geometric fields byte-for-byte identical to `main` for a fixed
      fixture session (parity test).
- [x] The GUI shows one box per enabled method node, each rendering that method's declared params; no
      boundary option is hardcoded by method name in `task_detail_panel.py`. _(Phase 6)_
- [ ] `_VALID_BOUNDARY_METHODS`, the `if/elif` selector in `extract_session_boundaries`, and the
      per-method dicts/blocks in `BoundaryResults` / the NPZ writer are removed.

## Definitions

- **Boundary contract**: the frozen `BoundaryContour` base class exposing exactly the 11 geometric fields
  (`contour_uv`, `area_uv`, `perimeter_uv`, `circularity`, `centroid_uv`, `peak_uv`, `pca_major_uv`,
  `pca_minor_uv`, `pca_orientation_deg`, `mean_iff_on_contour`, `iff_at_centroid`) plus provenance
  (`method_name`) and the optional `diagnostic_fields` channel. "Satisfies the contract" = is an instance
  of `BoundaryContour` and passes the contract validator.
- **Diagnostic field**: a named `np.ndarray` a method may emit *solely* for its own figure overlays (e.g.
  radial `lmax`). No computation stage may read it; consumers access it only through a capability check
  ("does this contour carry field X?"), tolerating absence.
- **Uniform downstream**: a consumer contains no branch on method identity; it discovers `<method>/`
  directories and applies identical logic to each. Testable: no method-name string literal in consumer
  modules.
- **Barrier node**: a DAG node that performs no extraction itself; its sole role is to `depends_on` all
  enabled method nodes so downstream (depending on it) runs only after every enabled method has produced
  output.
- **Adding a method with zero shared-code edits**: no edit to any module other than the new subclass file,
  the registry's declared list (if the registry is code-declared) or the DAG (if node-declared), and the
  barrier's dependency line + layout coordinate.

---

## Technical Design

### Approach

Formalize the already-implicit contract and replace runtime method-branching with the **Strategy +
Factory** pattern behind a code registry (per `data-pipeline-engineering/02 §6`: "Add a new variant by
adding one adapter, touching nothing downstream (open/closed)"). The registry is the single authoritative
representation of *what algorithms exist and their params schema* (DRY, `05 §3`); the DAG YAML is the
authoritative representation of *which method nodes are wired and with what values*. Downstream depends on
an abstract data contract, never a concrete algorithm (DIP/LSP, `05 §1`: "any two implementations behind
the same contract must be truly interchangeable"). Per-method output folders are overwrite-idempotent
(`02 §5`: "Overwrite, don't append"), so methods never collide and a single method re-runs cleanly.

The fan-in uses a **barrier node** rather than a group-dependency because the repo has no group mechanism
and `category:` is inert; the barrier is purely additive to the vendored resolver and keeps every
downstream `depends_on` unchanged.

The one place that becomes dynamic is `_build_pipeline_stages`: it iterates the registry to emit one stage
descriptor per method node. Everything else (contracts, IO, GUI schema) is derived from the registry;
nothing is code-generated or hidden behind decorators — the method set is explicit in both the registry
and the DAG.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Barrier fan-in node** (downstream depends on barrier; barrier depends on all method nodes) | Zero downstream churn; strictly additive to vendored resolver; localized to one dependency line | One extra node in the graph | **Chosen** |
| Activate `category:` as a dependency group | Reads elegantly; no barrier node | Edits shared vendored `can_run` used by *every* task; broad blast radius; changes dependency semantics repo-wide | Rejected (violates "don't disturb the codebase") |
| Downstream lists each method node in `depends_on` | No barrier | Adding a method edits every downstream node — the exact coupling being removed | Rejected |
| **Code registry (ABC + factory), params schema in code** | Single authoritative algorithm set; strong typing; contract conformance testable; params schema co-located with algorithm | Method list lives in code, not pure config | **Chosen** (DAG still declares which nodes are wired) |
| Self-registration via decorators (auto-discovery) | One-touch add | "Magic"; hides the method set from the human-readable config; against user's explicit-over-implicit preference | Rejected |
| Full registry-generated DAG nodes/flows/layout | Truly one-touch | Requires a node-generation engine touching the vendored handler + layout + GUI; speculative generality (YAGNI, `05 §6`) | Rejected |
| **Diagnostic fields as opaque `dict` channel** | Contract stays lean (ISP); consumers tolerate absence | Consumers wanting an overlay must capability-check | **Chosen** |
| Typed capability negotiation for extras | Explicit about what each method offers | Overkill under World-1 (all outputs are comparable curves) | Rejected |

### Architecture & Module Contracts

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `BoundaryContour` (base, frozen) | Define the 11-field geometric contract + `method_name` + `diagnostic_fields` | — (data type) | which algorithm produced it; NPZ layout; GUI |
| `BoundaryMethod` (ABC) | Declare `name`, `params_schema`, `compute(grid_u, grid_v, grid_z, **params) -> BoundaryContour \| None` | grids + params → contour | downstream consumers; IO format; DAG/GUI |
| `RadialFootMethod` / `GradientMethod` / `InflectionMethod` | Concrete algorithms (migrated from existing detectors) | grids + own params → own `BoundaryContour` subclass | each other; the registry's consumers |
| `boundary_method_registry` | Authoritative map name → `BoundaryMethod`; factory `get_method(name)`; `all_methods()`; `params_schema_of(name)` | name → method / schema | how methods compute internally |
| `contract validator` | Assert a produced object satisfies `BoundaryContour` at fan-in | contour → contour or raise | algorithm internals |
| generic boundary NPZ writer | Serialize any `BoundaryContour` (+ XYZ projections + diagnostic fields) to `<method>/` | contour → NPZ file | which method; params schema; GUI |
| generic boundary NPZ reader / discovery | Enumerate `<method>/` dirs for a session; load contours | session dir → `dict[method, BoundaryContour]` | algorithm internals; GUI |
| `spatial_extract_boundary_flow(method=...)` | Run one method for the session; write its folder; render its figures | items + method + params → outputs | other methods; downstream |
| barrier flow (`spatial_extract_boundaries`) | Join point; no extraction | depends on method nodes → sentinel | algorithm internals |
| downstream consumers (comparison/profile pipelines) | Loop over discovered methods; identical per-method logic | `dict[method, contour]` → per-method sub-outputs | which methods exist; `if method ==` |
| GUI `task_detail_panel` (boundary path) | Render a method node's options from its registry params schema | node + schema + YAML values → widgets | method-specific option names |
| `_build_pipeline_stages` (boundary path) | Iterate registry → one stage descriptor per method node | registry → stage list entries | algorithm internals |

```
src/analysis/receptive_field_mapping/
  boundary/                                   # new package (or metrics/boundary/)
    contract.py            # BoundaryContour (frozen base) + validator
    method_base.py         # BoundaryMethod ABC + ParamSpec/params-schema type
    registry.py            # registry + get_method() factory + all_methods()
    methods/
      radial.py            # RadialFootMethod  (migrated from rf_radial_foot_boundary.py)
      gradient.py          # GradientMethod    (migrated from rf_gradient_boundary.py)
      inflection.py        # InflectionMethod  (migrated from rf_inflection_boundary.py)
  data/
    rf_boundary_io.py      # generic writer/reader over BoundaryContour + <method>/ layout
    rf_boundary_types.py   # BoundaryParams/Results de-methodized; params come from schema

Output layout:
4_analysed/spatial_extract_boundaries/<session_id>/
  radial/    <session_id>_boundary.npz   + inspection/ figures
  gradient/  <session_id>_boundary.npz   + inspection/ figures
  inflection/<session_id>_boundary.npz   + inspection/ figures
```

**Params schema shape** (per method, declared in code): an ordered list of specs
`ParamSpec(key, type, default, range=None, choices=None, tunable=False, group=...)`. The GUI renders
widget type from `type`/`choices` (not from the runtime Python value), fixing the current
type-by-value-inference fragility and the missing-group `ValueError` gap.

---

## Implementation Plan

### Phase 1: Contract + Registry Foundation
**Goal:** The formal contract, ABC, and registry exist and are unit-tested, with no wiring changes yet.

_Phase 1 completed 2026-07-18_

- [x] 1.1 — Add `boundary/contract.py`: frozen `BoundaryContour` base (11 fields + `method_name` +
      `diagnostic_fields: Mapping[str, np.ndarray]`), plus a `validate_contour()` that raises on any
      missing/malformed field (fail-fast).
- [x] 1.2 — Add `boundary/method_base.py`: `BoundaryMethod` ABC (`name`, `params_schema`, `compute`) and
      the `ParamSpec` params-schema type.
- [x] 1.3 — Add `boundary/registry.py`: registry dict, `get_method(name)` factory (raises on unknown —
      replaces `_VALID_BOUNDARY_METHODS`), `all_methods()`, `params_schema_of(name)`.
- [x] 1.4 — Write the shared **contract conformance test-suite** (parametrized over registered methods)
      and a `BoundaryContour` validator unit test.

**Files Modified:**
- `src/analysis/receptive_field_mapping/boundary/contract.py` — new
- `src/analysis/receptive_field_mapping/boundary/method_base.py` — new
- `src/analysis/receptive_field_mapping/boundary/registry.py` — new
- `tests/test_boundary_contract.py` — new

**Dependencies:** None

### Phase 2: Migrate the Three Methods onto the Contract
**Goal:** Existing detectors become `BoundaryMethod` subclasses emitting `BoundaryContour`; `lmax` /
`gradient_magnitude` move to `diagnostic_fields`.

_Phase 2 completed 2026-07-18_

- [x] 2.1 — `methods/radial.py`: wrap `compute_radial_foot_boundary` / `compute_radial_foot_stages` as
      `RadialFootMethod`; emit `RadialBoundaryContour` with `lmax` in `diagnostic_fields`; declare its
      5-param schema (with `tunable=True` flags matching current `ContourParamToggles`).
- [x] 2.2 — `methods/gradient.py`: wrap `compute_gradient_ridge` as `GradientMethod`; `gradient_magnitude`
      → `diagnostic_fields`; declare schema.
- [x] 2.3 — `methods/inflection.py`: wrap `compute_inflection_boundary` as `InflectionMethod`; declare
      schema. Keep the shared polygon/sampling helpers (currently hosted here) importable by all methods.
- [x] 2.4 — Register all three in `registry.py`.
- [x] 2.5 — Parity test: radial geometric fields byte-for-byte vs `main` on a fixture session.

**Files Modified:**
- `boundary/methods/{radial,gradient,inflection}.py` — new (migrated logic)
- `metrics/rf_radial_foot_boundary.py`, `rf_gradient_boundary.py`, `rf_inflection_boundary.py` — reduce to
  computational cores imported by the subclasses (or move wholesale)
- `tests/test_rf_response_fields_parity.py` — extend for parity

**Dependencies:** Phase 1

### Phase 3: Generic IO + Per-Method Output Layout
**Goal:** One generic writer/reader over `BoundaryContour`; per-method output folders; per-method NPZ
blocks and duplicated `*_to_dict` removed.

_Phase 3 completed 2026-07-18_

- [x] 3.1 — Rewrite `rf_boundary_io.py` writer to serialize any `BoundaryContour` (11 fields + XYZ
      projections + `diagnostic_fields`) to `<session>/<method>/<session>_boundary.npz`; delete the
      per-method `inflection_`/`gradient_`/`radial_` blocks.
- [x] 3.2 — Add a discovery reader: enumerate `<method>/` dirs → `dict[method, BoundaryContour]`.
- [x] 3.3 — De-methodize `rf_boundary_types.py`: remove per-method dicts in `BoundaryResults` and the
      radial-only scalar fields in `BoundaryParams`; params now flow from the schema.
- [x] 3.4 — Update `rf_boundary_verification.py`: drop `_VALID_BOUNDARY_METHODS`; validate via registry.

**Phase 3 implementation notes (for Phase 4/5):**
- New generic IO API in `data/rf_boundary_io.py`: `save_boundary_contours_npz(contours_by_gesture, *,
  method_name, session_output_dir, session_id, forearm_uv, forearm_faces, forearm_V, config_snapshot)`
  writes `<method>/<session>_boundary.npz` (OVERWRITE, never append) + a `run_metadata.json` provenance
  sidecar; `load_boundary_contours(session_output_dir) -> dict[method, dict[gtype, BoundaryContour]]`
  discovers `<method>/` folders (a subdir is a method folder iff it holds a `*_boundary.npz`; a boundary
  NPZ without its metadata sidecar fails fast). The reader **generalizes** the plan's stated
  `dict[method, BoundaryContour]` to carry the gesture axis (real sessions have multiple gestures);
  layout helpers `boundary_method_output_dir` / `boundary_contour_npz_path` / `boundary_run_metadata_path`.
  One generic `_serialize_contour` / `_deserialize_contour` pair replaces `_save_boundary_fields` and the
  three per-method blocks; diagnostics round-trip via `diagnostic_{name}_{gtype}` + a `diagnostic_names_{gtype}` index.
- `BoundaryResults` is now `boundary_method: str` + `boundaries: {method_name: {gtype: BoundaryContour|None}}`
  with a `gesture_boundaries` **property** returning the active method's dict (renderer unchanged). Removed
  the 4 per-method dicts and `per_gesture_smoothed`/`laplacian`/`gradient_mag`. `BoundaryParams` lost the 5
  `radial_*` scalars.
- `extract_session_boundaries(prepared, params, *, method_params=None)` now runs **only the active method**
  via `registry.get_method(params.boundary_method).compute(...)` (no `if/elif`, no derived-array caching).
  **Phase 4 must fan out to all enabled methods as separate nodes.** Gradient reconciliation still pending
  (Phase-2 note): gradient uses its own `gradient_gauss_sigma` schema default, not the DAG `inflection_sigma`.
- **Bridging changes (immediate callers, for consistency — Phase 4/5 will supersede):**
  - `pipelines/rf_population_response_field_pipeline.py`: builds a generic `method_params` dict from its
    `radial_*`/`inflection_sigma` kwargs and passes it to `extract_session_boundaries`; dropped the dead
    `inflection_boundaries=` sentinel arg; `BoundaryParams(...)` no longer sets `radial_*`.
  - `data/rf_contour_params_io.py::defaults_from_boundary_params`: radial defaults now sourced from
    `registry.params_schema_of("radial")` (was `BoundaryParams.radial_*`). Behaviour change: the tuner
    bootstrap now uses the radial schema defaults (identical to the old `BoundaryParams` defaults) rather
    than DAG radial overrides — acceptable this pass (tuner is Phase 6).
  - `data/rf_boundary_io.py::write_boundary_sentinel`: dropped the write-only (never read)
    `inflection_boundaries` JSON payload and the last `inflection_boundary_to_dict` production coupling.
- **Known runtime breakage handed to Phase 5 (imports stay healthy; these are separate DAG stages):** the
  monolithic `<session>_population_response_fields.npz` no longer carries any boundary contour data
  (`boundary_contour_uv_`, `inflection_`/`gradient_`/`radial_` blocks, `radial_lmax_`) nor the derived
  `smoothed_`/`laplacian_`/`gradient_mag_` arrays — those moved to per-method folders / the diagnostic
  channel. Downstream consumers still reading those keys from the monolithic NPZ will break until Phase 5
  repoints them to `load_boundary_contours`: `rf_profile_extraction_pipeline.py`,
  `rf_proximal_distal_comparison_pipeline.py`, `rf_tap_stroke_comparison_pipeline.py`,
  `rf_session_boundary_comparison_pipeline.py`.
- The three metric-module `*_to_dict` serializers (`inflection_boundary_to_dict` etc.) were left in place
  (still imported by their own metric tests); they no longer have any production caller.
- Verified: `test_boundary_contract.py` + `test_rf_response_fields_parity.py` (52) stay green; new
  `tests/test_boundary_io.py` (6) covers the round-trip incl. diagnostics, two-method folder isolation,
  re-run idempotency, and partial-folder fail-fast. Full import health confirmed. The 31 pre-existing
  failures in `test_rf_tap_stroke_comparison.py` / `test_rf_grid_cell_metrics.py` reproduce identically
  with all Phase-3 changes stashed (renderer returning `None`) — unrelated to this phase.

**Files Modified:**
- `data/rf_boundary_io.py`, `data/rf_boundary_types.py`,
  `pipelines/rf_boundary_verification.py`, `metrics/rf_boundary_extraction.py` — remove `if/elif` selector

**Dependencies:** Phase 2

### Phase 4: Fan-Out Flow + Barrier + Registry-Driven Prefect Wiring
**Goal:** One generic extraction flow per method node; barrier join; downstream unchanged.

_Phase 4 completed 2026-07-18_

- [x] 4.1 — Add generic `spatial_extract_boundary_flow(method=..., **params)` that runs one method and
      writes/renders its folder (figures per method).
- [x] 4.2 — Add the barrier flow for `spatial_extract_boundaries` (no extraction; join only).
- [x] 4.3 — Make `_build_pipeline_stages` iterate `registry.all_methods()` to append one descriptor per
      method node, params pulled generically from the node's options (no literal per-`radial_*` keys).
- [x] 4.4 — DAG YAML: add one node per method (`spatial_extract_boundary__<name>`) with that method's
      option values; repoint `spatial_extract_boundaries` to `depends_on` the method nodes; add layout
      coordinates. Downstream consumers unchanged.

**Phase 4 implementation notes (for Phase 5):**
- **New flows** (`scripts/analysis_workflow_processing.py`): generic fan-out
  `spatial_extract_boundary_flow(input_items, method, method_params, *, force_processing, neuron_mode,
  iff_metric, min_overlap_pct, median_filter_size, heatmap_space, cmap, flip_u, contour_color,
  circular_crop_margin, use_tuned_params)` — `@flow(name="spatial_extract_boundary")`, dispatches through
  the registry inside `run_population_response_field_extraction`, never branches on `method`. Barrier
  `spatial_extract_boundaries_flow(input_items, enabled_methods, *, force_processing, iff_metric)` —
  `@flow(name="spatial_extract_boundaries")`, does NO extraction; verifies each enabled method wrote its
  per-session run sentinel and fails fast otherwise.
- **`_build_pipeline_stages`** now splices `*_build_boundary_stage_descriptors(dag_handler)` where the
  single boundary descriptor was. That helper iterates `boundary_registry.all_methods()` → one descriptor
  per `spatial_extract_boundary__<method>` node (params via `_make_boundary_method_params`, which reads the
  shared options + the method's `params_schema` keys generically — no per-`radial_*` literals), then the
  barrier descriptor (passes `enabled_methods` = method nodes whose DAG node is enabled). This registry
  loop is the sole dynamic point.
- **Signature change** `run_population_response_field_extraction(session_configs, neuron_mode, output_dir,
  *, boundary_method, method_params, ...)`: dropped the explicit `radial_*` + `inflection_sigma` args in
  favour of the generic `method_params` dict; `inflection_sigma` (monolithic-NPZ scalar only) is now
  `method_params.get("inflection_sigma")`. `save_boundary_outputs_npz(..., *, session_output_dir)` gained a
  keyword-only session-root arg.
- **PER-METHOD OUTPUT LAYOUT (Phase 5 MUST read this):** each run now writes under
  `4_analysed/spatial_extract_boundaries/iff_<metric>/<session_id>/<method>/` — this per-method dir holds
  the figures, the monolithic `<session>_population_response_fields.npz`, the run sentinel
  (`<session>_population_response_fields_done.json`), **and** the boundary contour NPZ
  (`<session>_boundary.npz` + `run_metadata.json`). The contour NPZ is still written by the IO layer keyed
  off the SESSION root, so `load_boundary_contours(<session_id_dir>)` (where `<session_id_dir> =
  output_dir/iff_<metric>/<session_id>`) discovers every `<method>/` folder → `{method: {gtype:
  BoundaryContour}}`. Downstream consumers must switch from the old shared monolithic NPZ to
  `load_boundary_contours(session_dir)` and loop over the returned methods. The monolithic NPZ is now
  per-method and no longer authoritative for boundary geometry.
- DAG YAML: three nodes `spatial_extract_boundary__{radial,gradient,inflection}` (radial `depends_on`
  includes `spatial_tune_rf_contours`; all include `spatial_build_response_fields`); gradient carries its
  own `gradient_gauss_sigma`/`gradient_n_angles`/`gradient_savgol_window`. `spatial_extract_boundaries` is
  now the barrier: `options` = `{force_processing, iff_metric}`, `depends_on` = the three method nodes.
  Every downstream consumer's `depends_on: [spatial_extract_boundaries]` is UNCHANGED. layout.json gained
  the three method-node coordinates (barrier nudged right to x≈3320).
- **Verified:** `test_boundary_contract.py` + `test_rf_response_fields_parity.py` + `test_boundary_io.py` +
  new `test_rf_boundary_dag_wiring.py` = 75 passed. Full suite collects (449 tests, no import errors).
  Script imports; `_build_pipeline_stages` integration check confirms barrier ordered after all method
  nodes with correct generic params. No `if method ==` in the script/flow.

**Files Modified:**
- `scripts/analysis_workflow_processing.py`, `configs/analyse_workflow_processing_dag.yaml`,
  `configs/analyse_workflow_processing_dag.layout.json`,
  `src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`,
  `src/analysis/receptive_field_mapping/data/rf_boundary_io.py`,
  `tests/test_rf_boundary_dag_wiring.py` (new). `stage_runner.py` needed no change (gates by `depends_on`).

**Dependencies:** Phase 3

### Phase 5: Uniform Downstream Consumers
**Goal:** Consumers loop over discovered methods; no method-name literals; profile stage stops reading
diagnostic fields by name.

_Phase 5 completed 2026-07-18_

- [x] 5.1 — Update comparison pipelines (`rf_session_boundary_comparison`, `rf_proximal_distal_comparison`,
      `rf_tap_stroke_comparison`) to iterate discovered `<method>/` dirs, emitting per-method sub-outputs.
- [x] 5.2 — Update `rf_profile_extraction_pipeline`: read the generic contract; obtain any overlay field
      via the `diagnostic_fields` capability check (drop `radial_lmax_` / `gradient_contour_uv_` literals).

**Phase 5 implementation notes (for Phase 6):**
- **New IO surface in `data/rf_boundary_io.py`** (all method-blind, reused by every consumer):
  `boundary_session_extract_dir(db, iff_metric, session_id)` (the one place the
  `iff_<metric>/<session_id>` path is spelled out), `boundary_response_fields_npz_path(session_dir,
  method, session_id)` (per-method monolithic grid NPZ), `LoadedBoundary` (frozen: the contract
  `BoundaryContour` + its on-disk XYZ projections `contour_xyz`/`centroid_xyz`/`peak_xyz`/
  `perimeter_xyz_mm`/`area_xyz_mm2`), `load_boundary_records(session_dir) -> {method: {gtype:
  LoadedBoundary}}` (the discovery reader consumers loop over; `load_boundary_contours` is now a thin
  projection of it), `discover_boundary_methods(session_configs, iff_metric) -> (records_by_session,
  sorted_methods)` (shared cross-session discovery), and `BoundaryNpzView` / `load_boundary_npz_view(...)`
  — a read-only `np.load`-like adapter that re-exposes a method's `LoadedBoundary` geometry under the
  **legacy `boundary_*` key names** the comparison logic already reads, backed by the per-method grid NPZ
  for everything else. The same view wraps every method (no method-identity branch); a `boundary_*_<gtype>`
  key exists iff that gesture produced a contour, preserving the old `... in npz` capability semantics.
- **Comparison pipelines** each gained a thin registry-blind orchestrator (`run_*` → `discover_boundary_methods`
  → loop → `_run_for_method(output_dir/<method>, method, records_by_session, ...)`). Per-method sub-outputs
  land under `<compare_dir>/iff_<metric>/<method>/` (CSV summary + figures + sentinel), so methods never
  overwrite. Each `_run_for_method` builds a `BoundaryNpzView` per session and is otherwise the original
  body unchanged — geometry now flows from the discovered contracts, grids from the per-method grid NPZ.
  Sessions missing a given method's folder are skipped with a warning (consistent with the existing
  missing-centroid skip).
- **Profile stage** now iterates `load_boundary_records(session_dir)`; per method it writes
  `output_dir/iff_<metric>/<session_id>/<method>/` (CSVs + `profile_raw/`, `gradient_1d/`, `combined/`,
  `raycast/`, `*_rf_profiles_done.json`, one `rf_profile_extraction_methodology_<method>.md`). Overlay
  figure groups are gated purely by capability on each gesture's `BoundaryContour`: `gradient_1d` on
  `has_diagnostic("gradient_magnitude")`, `combined`/`raycast` on `has_diagnostic("lmax")` (+ a smoothed
  overlay), laplacian groups on `has_diagnostic("smoothed"/"laplacian")`. The old cross-method
  `gradient_contour_uv` overlay was dropped (a cross-method composite, out of scope); `radial_lmax_` /
  `gradient_contour_uv_` literals are gone and the method-name-keyed `_BOUNDARY_METHOD_DESCRIPTIONS` dict
  was replaced with method-blind methodology prose (interpolating the name as free-form provenance only).
  Consequence today: only radial emits `lmax` and only gradient emits `gradient_magnitude`; no method emits
  `smoothed`/`laplacian`, so laplacian/combined/raycast groups stay dormant until a method provides those
  diagnostics — forward-compatible, no silent fallback.
- **Grep proof:** `radial_lmax_`/`gradient_contour_uv_` return nothing under `pipelines/`; the four consumer
  files contain no `"radial"`/`"gradient"`/`"inflection"` literal and no `if … method … ==` branch (methods
  appear only as `.items()` loop keys). Import health OK on all four + `rf_boundary_io`.
- **Verification:** required suite `test_boundary_contract` + `test_rf_response_fields_parity` +
  `test_boundary_io` + `test_rf_boundary_dag_wiring` = **75 passed**; full suite still collects (449, no
  import errors). A dedicated end-to-end smoke (two methods, real per-method layout via
  `save_boundary_contours_npz`) confirms all four consumers discover both methods and write non-overlapping
  `<method>/` sub-outputs, with `gradient_1d` produced only for the gradient method.
- **Fixed a latent pre-existing bug** in `_methodology_md`: the output-layout line contained unescaped
  `{metric}`/`{session_id}` in an f-string (`NameError` the moment the profile stage actually ran) — now
  literal `<metric>`/`<session_id>`.
- **Known non-blocker for Phase 6:** `tests/test_rf_tap_stroke_comparison.py` builds the pre-Phase-3
  monolithic fixture (boundary keys inside the response-fields NPZ, no `<method>/` folder), so
  `discover_boundary_methods` now fails fast on it: 19→21 failing (baseline had 19 pre-existing, mostly the
  shift-decomposition mock returning `None`). Migrating that fixture to the per-method layout is a
  test-suite task (the XYZ area/perimeter are now computed by the serializer from the contour, so the
  fixture's injected `*_area_mm2` values must be derived from the contours) — out of the four-consumer file
  scope for this phase.

**Files Modified:**
- `pipelines/rf_session_boundary_comparison_pipeline.py`, `rf_proximal_distal_comparison_pipeline.py`,
  `rf_tap_stroke_comparison_pipeline.py`, `rf_profile_extraction_pipeline.py`,
  `data/rf_boundary_io.py` (new discovery/view/path helpers)

**Dependencies:** Phase 4

### Phase 6: Registry-Driven GUI
**Goal:** One box per method node, options rendered from the method's params schema; radial tuner
unchanged.

_Phase 6 completed 2026-07-18_

- [x] 6.1 — In `task_detail_panel.py`, for boundary-method nodes render options from
      `registry.params_schema_of(method)` (widget type from `ParamSpec.type`/`choices`, grouping from
      `ParamSpec.group`); remove the boundary entries from `_OPTION_ENUMS`, `_OPTION_VISIBILITY`,
      `_OPTION_GROUP_OF`; add each method node to `_GROUPED_TASKS` generically.
- [x] 6.2 — Keep `spatial_tune_rf_contours` (radial live tuner) wired to the radial method unchanged.

**Phase 6 implementation notes:**
- Registry-driven rendering keys off the node-name prefix `spatial_extract_boundary__`. New pure
  (Qt-free) module helpers in `task_detail_panel.py` are the single seam: `boundary_method_of_task`
  (prefix → registry method, fail-fast on unknown), `boundary_param_specs` (node → ordered
  `{key: ParamSpec}` from `registry.params_schema_of`), `option_group_of` (method param → its
  `ParamSpec.group`; every other option → `_OPTION_GROUP_OF`, fail-fast if neither), and
  `_is_grouped_task` (`_GROUPED_TASKS` ∪ any `spatial_extract_boundary__*`). `_build_option_section`
  now renders a method-specific key via `_make_param_spec_section` (widget from `spec.choices`/
  `spec.type`: combobox / bool checkbox / numeric line-edit with schema-driven nullability — empty ⇒
  YAML null only when `spec.default is None`, else revert), and `_insert_grouped` builds its group
  catalogue from `_OPTION_GROUPS` + the node's `ParamSpec.group`s in schema order.
- **Removed from the hardcoded dicts:** the whole `boundary_method` enum from `_OPTION_ENUMS`; the
  `boundary_method` controller block from `_OPTION_VISIBILITY` (kept the `neuron_mode`→`iff_metric`
  block); the `inflection_sigma`/`boundary_method`/`radial_gauss_sigma`/`radial_hess_sigma`/
  `radial_envelope_smooth_sigma` rows from `_OPTION_GROUP_OF` (added the shared, method-agnostic
  `use_tuned_params: "method"`); dropped the single `spatial_extract_boundaries` entry from
  `_GROUPED_TASKS` (barrier now renders flat — only `iff_metric`). No boundary option is hardcoded by
  method name anymore.
- Legacy disabled node `spatial_tuning_rf_metrics` still carries its own `boundary_method`/
  `inflection_sigma` options; with the enum removed these now render as plain scalar line-edits (it is
  not in `_GROUPED_TASKS`, so no group-resolution `ValueError`). Cosmetic only, disabled task.
- **Deferred (see Known Follow-ups):** syncing the barrier's `depends_on` when a method node is toggled
  off. Out of Phase 6's file scope (toggle lives in `task_panel.py`/`dag_graph_view.py`; no
  `DagConfigModel` depends_on-mutation API). Documented with a code comment at the boundary helpers.

**Files Modified:**
- `src/utils/gui/analysis_runner_gui/task_detail_panel.py`
- `tests/test_task_detail_panel_boundary_schema.py` (new)

**Dependencies:** Phase 4 (nodes exist), Phase 1 (schema API)

### Known Follow-ups

- **Barrier `depends_on` toggle sync (GUI).** `spatial_extract_boundaries` lists all three
  `spatial_extract_boundary__*` nodes in `depends_on`. Because `can_run` requires every dependency to be
  `mark_completed`, and a disabled node is never run/completed, toggling any method node OFF in the GUI
  makes the barrier unsatisfiable and stalls all downstream stages. Not introduced by Phase 6 (it is a
  consequence of the Phase 4 barrier wiring), and fixing it cleanly needs (a) a `DagConfigModel` API to
  add/remove a `depends_on` entry and (b) hooking the enabled-toggle handlers in `task_panel.py` /
  `dag_graph_view.py` — without hardcoding barrier/method-node identity into generic GUI code. Deferred.
  Workaround today: leave all method nodes enabled, or hand-edit the barrier's `depends_on` in the YAML.

---

## Testing Plan

### Unit Tests
- [ ] `BoundaryContour` validator rejects missing/malformed fields (fail-fast).
- [ ] Registry `get_method` raises on unknown name; `all_methods()` returns the three.
- [ ] Contract conformance suite parametrized over every registered method (same assertions each).
- [ ] Generic NPZ writer round-trips a contour incl. `diagnostic_fields` and XYZ projections.
- [x] Params-schema → widget-type mapping (enum/float/bool) resolves from `ParamSpec`, not value type.
      _(Phase 6: `tests/test_task_detail_panel_boundary_schema.py`)_

### Integration Tests
- [ ] Two methods enabled → two non-overlapping `<method>/` trees; re-running one overwrites only its own.
- [ ] Downstream consumer discovers N method dirs and emits N per-method sub-outputs from identical code.
- [ ] Barrier gates downstream until all enabled method nodes complete (`can_run` behavior).
- [ ] End-to-end on a tiny fixture session: expected artifacts appear under each enabled method.

### Manual Verification
- [ ] Launch the AnalysisRunnerGUI: each enabled method shows as its own box with its own options; toggling
      one on/off adds/removes only its node's run.
- [ ] Radial default output geometric fields match `main` (parity), figures land in `radial/inspection/`.
- [ ] `grep -rn 'if.*method.*==\|radial_lmax_\|gradient_contour_uv_\|_VALID_BOUNDARY_METHODS'` over
      consumers/GUI/script returns no hits outside subclass modules + registry.

### Edge Cases
- [ ] A method returns `None` for a gesture (no boundary found) — handled per current fail/skip semantics,
      recorded, does not corrupt other methods' output.
- [ ] Degenerate grid (empty / all-NaN / single point) — each method's declared behavior tested against a
      known result rather than emitting a silent `NaN` downstream.
- [ ] Downstream requests a diagnostic overlay from a method that didn't emit it — capability check yields
      "absent", overlay skipped, no crash and no silent wrong number.
- [ ] Only one method enabled — barrier still resolves; downstream emits a single sub-output.

---

## Documentation Plan

- [ ] Update `CLAUDE.md` / `docs/claude/architecture.md` with the boundary plug-in seam and how to add a
      method.
- [ ] Update `docs/spatial_extract_boundaries/` walkthrough to reflect per-method nodes + output layout.
- [ ] Add a short "Adding a boundary method" guide: `docs/guides/adding-a-boundary-method.md` (subclass +
      DAG node + barrier line + layout coord).
- [ ] Add changelog entry: `docs/changelogs/rf-boundary-plugin-architecture.md`.
- [ ] Docstrings on `BoundaryContour`, `BoundaryMethod`, registry, and the diagnostic-fields channel.

---

## Rollback Plan

1. **Before merge:** all work is on `feature/rf-boundary-plugin-architecture`; the plan branch is not
   merged until parity + conformance tests pass. Revert = delete the branch.
2. **Data considerations:** the output layout changes from one shared NPZ to per-`<method>/` NPZs. This is
   a regenerated artifact under `4_analysed/` (no source data). Rollback = re-run the pre-change pipeline,
   which rewrites the old single-file layout. No migration of historical outputs is attempted; document
   that pre-change and post-change output trees are incompatible and must be regenerated.
3. **Rollback procedure:** revert the merge commit; downstream `depends_on` were never changed, so no DAG
   surgery is needed beyond removing the added method nodes + barrier repoint.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Radial parity drifts during migration (byte-for-byte) | Med | High | Phase-2 parity test gates the migration; keep `compute_radial_foot_stages` as the untouched computational core, wrap don't rewrite |
| Vendored `DagConfigHandler` can't express barrier cleanly | Low | Med | Barrier is a normal node with `depends_on`; uses existing `can_run` semantics unchanged — no vendored edit |
| GUI schema-driven rendering misses an option, hits the existing missing-group `ValueError` | Med | Med | Params come from `ParamSpec.group` for every option; add a test that every schema option resolves a group; fixes the current `radial_prominence`/`radial_plateau_size` gap |
| Registry-iterated stage list breaks static ordering assumptions in the runner | Low | Med | Runner gates by `depends_on`/`can_run`, not list order (confirmed); append method descriptors before the barrier descriptor |
| Downstream discovery picks up a stale `<method>/` dir from a prior run | Med | Med | Per-method overwrite-idempotency + a per-folder run-metadata/sentinel (`02 §11`); discovery keys off enabled nodes, not just dirs on disk |
| Scope creep into generalizing the tuner | Med | Med | Explicit Out-of-Scope; tuner stays radial-only; only static param rendering generalized |
| Two future algorithms impose contract needs not yet known | Low | Med | World-1 confirmed (all outputs comparable curves); diagnostic-fields channel absorbs method-specific extras without contract change |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Contract + Registry | S | None |
| Phase 2 — Migrate 3 methods | M | Phase 1 |
| Phase 3 — Generic IO + layout | M | Phase 2 |
| Phase 4 — Flow + barrier + Prefect | M | Phase 3 |
| Phase 5 — Uniform downstream | M | Phase 4 |
| Phase 6 — Registry-driven GUI | S–M | Phase 1, 4 |

---

## References

- Brainstorm: `docs/development/brainstorms/rf-boundary-plugin-architecture.md` (matured brief, locked
  decisions, ~9 coupling points with file:line)
- Pipeline-engineering guides: `~/.claude/knowledge/data-pipeline-engineering/` — `02 §6` (Strategy/
  Factory normalization), `05 §1` (SOLID: OCP/LSP/DIP), `05 §3` (DRY), `02 §5` (idempotency/overwrite),
  `02 §11` (provenance)
- Current wiring anchors: `metrics/rf_boundary_extraction.py`, `data/rf_boundary_types.py`,
  `data/rf_boundary_io.py`, `scripts/analysis_workflow_processing.py` (`_build_pipeline_stages`,
  `spatial_extract_boundaries_flow`), `src/utils/gui/analysis_runner_gui/task_detail_panel.py`,
  `configs/analyse_workflow_processing_dag.yaml`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- configs/analyse_workflow_processing_dag.layout.json
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/rf-boundary-plugin-architecture.md
- scripts/analysis_workflow_processing.py
- src/analysis/receptive_field_mapping/boundary/__init__.py
- src/analysis/receptive_field_mapping/boundary/contract.py
- src/analysis/receptive_field_mapping/boundary/method_base.py
- src/analysis/receptive_field_mapping/boundary/methods/__init__.py
- src/analysis/receptive_field_mapping/boundary/methods/_common.py
- src/analysis/receptive_field_mapping/boundary/methods/gradient.py
- src/analysis/receptive_field_mapping/boundary/methods/inflection.py
- src/analysis/receptive_field_mapping/boundary/methods/radial.py
- src/analysis/receptive_field_mapping/boundary/registry.py
- src/analysis/receptive_field_mapping/data/rf_boundary_io.py
- src/analysis/receptive_field_mapping/data/rf_boundary_types.py
- src/analysis/receptive_field_mapping/data/rf_contour_params_io.py
- src/analysis/receptive_field_mapping/metrics/rf_boundary_extraction.py
- src/analysis/receptive_field_mapping/pipelines/rf_boundary_verification.py
- src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py
- src/analysis/receptive_field_mapping/pipelines/rf_profile_extraction_pipeline.py
- src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py
- src/analysis/receptive_field_mapping/pipelines/rf_tap_stroke_comparison_pipeline.py
- src/utils/gui/analysis_runner_gui/task_detail_panel.py
- tests/test_boundary_contract.py
- tests/test_boundary_io.py
- tests/test_rf_boundary_dag_wiring.py
- tests/test_rf_response_fields_parity.py
- tests/test_task_detail_panel_boundary_schema.py
