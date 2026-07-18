"""Pluggable RF boundary extraction: contract, method ABC, and registry.

This package formalises the de-facto boundary contract shared by the three
existing detectors into an explicit, testable plug-in seam:

- :class:`BoundaryContour` / :func:`validate_contour` — the frozen 11-field
  geometric contract and its fail-fast validator (``contract``).
- :class:`BoundaryMethod` / :class:`ParamSpec` — the Strategy ABC every algorithm
  implements and the declarative parameter schema (``method_base``).
- registry factory — :func:`register_method`, :func:`get_method`,
  :func:`all_methods`, :func:`registered_names`, :func:`params_schema_of`
  (``registry``).

The concrete algorithms live under ``boundary.methods`` and register themselves
via the registry (added in a later phase).
"""

from __future__ import annotations

from analysis.receptive_field_mapping.boundary.contract import (
    BoundaryContour,
    validate_contour,
)
from analysis.receptive_field_mapping.boundary.method_base import (
    BoundaryMethod,
    ParamSpec,
)
from analysis.receptive_field_mapping.boundary.registry import (
    all_methods,
    get_method,
    params_schema_of,
    register_method,
    registered_names,
)

__all__ = [
    "BoundaryContour",
    "validate_contour",
    "BoundaryMethod",
    "ParamSpec",
    "register_method",
    "get_method",
    "all_methods",
    "registered_names",
    "params_schema_of",
]

# Import the methods package for its side effect: instantiating and registering
# the three concrete algorithms (radial / gradient / inflection) with the
# registry, so ``all_methods()`` is populated for every importer of this package.
from analysis.receptive_field_mapping.boundary import methods  # noqa: E402,F401
