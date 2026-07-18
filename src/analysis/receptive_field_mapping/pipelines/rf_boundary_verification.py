"""Verification layer for the ``spatial_extract_boundaries`` stage.

Two responsibilities, both fail-fast:

* validate the run parameters against their allowed sets, and
* decide whether a session's output is stale relative to its upstream inputs,
  using the same ``should_process_task`` mtime checker the sibling RF pipelines
  use (replaces the old bare ``sentinel.exists()`` check that never noticed
  regenerated upstream artifacts).
"""

from pathlib import Path
from typing import List

from _vendor.should_process_task import should_process_task
from analysis.pipeline.shared_constants import IFF_METRICS
from analysis.receptive_field_mapping.boundary.registry import get_method


def validate_boundary_params(iff_metric: str, boundary_method: str) -> None:
    """Raise ``ValueError`` if either option is outside its allowed set.

    ``boundary_method`` is validated through the boundary-method registry: an
    unregistered name raises ``ValueError`` (listing the registered methods). The
    registry is the single source of truth for which algorithms exist, so this
    replaces the former hardcoded ``_VALID_BOUNDARY_METHODS`` tuple.
    """
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"run_population_response_field_extraction: invalid iff_metric "
            f"{iff_metric!r}. Expected one of {IFF_METRICS}."
        )
    get_method(boundary_method)  # fail-fast on an unregistered boundary method


def boundary_stage_is_up_to_date(
    input_paths: List[Path],
    sentinel: Path,
    force: bool,
    extra_input_paths: List[Path] = [],
) -> bool:
    """Return True when the session sentinel is newer than every upstream input.

    Wraps ``should_process_task`` (inputs = the four resolved upstream artifacts,
    output = the per-session sentinel). Raises ``FileNotFoundError`` via
    ``should_process_task`` if any input is missing.

    ``extra_input_paths`` extends the staleness inputs with soft inputs that are
    not part of the loaded 4-tuple — used to feed the existing per-(session,
    gesture) contour-params JSONs when ``use_tuned_params`` is on, so editing a
    tuned JSON (a newer mtime) marks the boundary stage stale and regenerates it.
    """
    return not should_process_task(
        input_paths=list(input_paths) + list(extra_input_paths),
        output_paths=[sentinel],
        force=force,
    )
