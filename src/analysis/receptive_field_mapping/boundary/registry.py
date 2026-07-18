"""Authoritative registry + factory for RF boundary methods.

The registry is the single source of truth for *which algorithms exist* and
*what parameters each declares* (DRY, ``data-pipeline-engineering/05 §3``). It
maps a method name to a :class:`~...boundary.method_base.BoundaryMethod` instance
and exposes a factory (:func:`get_method`) that dispatches by name — the one
place algorithm identity is branched on. This replaces the scattered
``_VALID_BOUNDARY_METHODS`` tuple and the ``if/elif`` method selector.

Registration is explicit (no decorator auto-discovery magic — the method set
stays visible in code): Phase 2 registers each of the three concrete methods with
a single :func:`register_method` call. In this phase the registry starts empty.
"""

from __future__ import annotations

from analysis.receptive_field_mapping.boundary.method_base import (
    BoundaryMethod,
    ParamSpec,
)

__all__ = [
    "register_method",
    "get_method",
    "all_methods",
    "registered_names",
    "params_schema_of",
]


# Ordered map: method name -> BoundaryMethod instance. Insertion order is the
# authoritative method ordering used by the DAG-stage and GUI iteration.
_REGISTRY: dict[str, BoundaryMethod] = {}


def register_method(method: BoundaryMethod, *, overwrite: bool = False) -> BoundaryMethod:
    """Register a boundary method instance under its declared ``name``.

    Fail-fast: rejects non-``BoundaryMethod`` objects, empty names, and duplicate
    registrations (unless ``overwrite=True``). Returns the registered method so it
    can be assigned inline.
    """
    if not isinstance(method, BoundaryMethod):
        raise TypeError(
            f"register_method expects a BoundaryMethod instance, "
            f"got {type(method).__name__}"
        )
    name = method.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"BoundaryMethod.name must be a non-empty string, got {name!r}"
        )
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"a boundary method named {name!r} is already registered "
            f"(pass overwrite=True to replace it)"
        )
    _REGISTRY[name] = method
    return method


def get_method(name: str) -> BoundaryMethod:
    """Return the registered method named *name*.

    Fail-fast: raises ``ValueError`` (listing the known names) on an unknown name.
    This is the factory that replaces the old ``if/elif`` method selector.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ValueError(
            f"unknown boundary method {name!r}; registered methods: {known}"
        )
    return _REGISTRY[name]


def all_methods() -> tuple[BoundaryMethod, ...]:
    """Return every registered method, in registration order."""
    return tuple(_REGISTRY.values())


def registered_names() -> tuple[str, ...]:
    """Return the names of every registered method, in registration order."""
    return tuple(_REGISTRY.keys())


def params_schema_of(name: str) -> tuple[ParamSpec, ...]:
    """Return the ordered params schema of the method named *name*.

    Fail-fast on an unknown name (via :func:`get_method`).
    """
    return tuple(get_method(name).params_schema)
