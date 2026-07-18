"""Concrete boundary-extraction algorithms (``BoundaryMethod`` subclasses).

Each module here wraps one algorithm's computational core as a
:class:`~analysis.receptive_field_mapping.boundary.method_base.BoundaryMethod`.
Importing this package instantiates each concrete method and registers it with
the registry via
:func:`~analysis.receptive_field_mapping.boundary.registry.register_method`, so
that :func:`~analysis.receptive_field_mapping.boundary.registry.all_methods`
returns the three (radial / gradient / inflection) for any importer of the
``boundary`` package.

Registration is explicit and eager (no decorator auto-discovery): the method set
stays visible in this module's import list. Registration order here is the
authoritative ordering used by DAG-stage and GUI iteration.
"""

from __future__ import annotations

from analysis.receptive_field_mapping.boundary.methods.radial import RadialFootMethod
from analysis.receptive_field_mapping.boundary.methods.gradient import GradientMethod
from analysis.receptive_field_mapping.boundary.methods.inflection import (
    InflectionMethod,
)
from analysis.receptive_field_mapping.boundary.registry import register_method

__all__ = ["RadialFootMethod", "GradientMethod", "InflectionMethod"]


# Eager, explicit registration in canonical order.
register_method(RadialFootMethod())
register_method(GradientMethod())
register_method(InflectionMethod())
