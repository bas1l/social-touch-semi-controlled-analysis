"""Shared helpers for :class:`BoundaryMethod` subclasses.

Currently exposes :func:`resolve_params`, the single fail-fast resolution point
that maps a caller-supplied ``**params`` mapping onto a method's declared
``params_schema``: it applies each :class:`ParamSpec` default for omitted keys
and rejects any key not declared in the schema. This keeps the three concrete
methods DRY (``data-pipeline-engineering/05 §3``) and makes an unknown parameter
a loud error rather than a silent no-op (CLAUDE.md fail-fast).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from analysis.receptive_field_mapping.boundary.method_base import ParamSpec

__all__ = ["resolve_params"]


def resolve_params(
    schema: Sequence[ParamSpec],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve caller ``params`` against a method's ``schema`` (fail-fast).

    Returns a ``{key: value}`` dict carrying every schema key, using the caller's
    value where supplied and the :class:`ParamSpec` default otherwise. Any key in
    *params* that the schema does not declare raises ``ValueError`` — there is no
    silent fallback for an unrecognised parameter.
    """
    allowed = {spec.key: spec.default for spec in schema}
    unknown = [key for key in params if key not in allowed]
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)}; "
            f"declared params: {sorted(allowed)}"
        )
    resolved = dict(allowed)
    resolved.update(params)
    return resolved
