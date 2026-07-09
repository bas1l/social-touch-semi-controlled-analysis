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

_VALID_BOUNDARY_METHODS = ("gradient", "inflection", "radial")


def validate_boundary_params(iff_metric: str, boundary_method: str) -> None:
    """Raise ``ValueError`` if either option is outside its allowed set."""
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"run_population_response_field_extraction: invalid iff_metric "
            f"{iff_metric!r}. Expected one of {IFF_METRICS}."
        )
    if boundary_method not in _VALID_BOUNDARY_METHODS:
        raise ValueError(
            f"run_population_response_field_extraction: invalid boundary_method "
            f"{boundary_method!r}. Expected 'gradient', 'inflection', or 'radial'."
        )


def boundary_stage_is_up_to_date(
    input_paths: List[Path],
    sentinel: Path,
    force: bool,
) -> bool:
    """Return True when the session sentinel is newer than every upstream input.

    Wraps ``should_process_task`` (inputs = the four resolved upstream artifacts,
    output = the per-session sentinel). Raises ``FileNotFoundError`` via
    ``should_process_task`` if any input is missing.
    """
    return not should_process_task(
        input_paths=input_paths,
        output_paths=[sentinel],
        force=force,
    )
