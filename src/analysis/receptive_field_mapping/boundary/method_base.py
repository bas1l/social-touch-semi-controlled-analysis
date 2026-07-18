"""Boundary-method abstraction: the ``BoundaryMethod`` ABC and ``ParamSpec``.

A ``BoundaryMethod`` is a Strategy (per ``data-pipeline-engineering/02 §6``): a
single algorithm that turns a smoothed IFF grid into a
:class:`~analysis.receptive_field_mapping.boundary.contract.BoundaryContour` (or
``None`` when no boundary can be traced). Each method declares its own name and a
``params_schema`` — an ordered sequence of :class:`ParamSpec` describing every
tunable/configurable parameter — so that IO, the DAG wiring, and the GUI are all
*derived* from the method rather than hardcoding per-method option names.

A ``BoundaryMethod`` must NOT know about downstream consumers, the on-disk IO
format, or the DAG/GUI. It only maps ``(grids, params) -> contour | None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from analysis.receptive_field_mapping.boundary.contract import BoundaryContour

__all__ = ["ParamSpec", "BoundaryMethod"]


# Sentinel marking a required field that has no default. ``group`` is mandatory
# for every ParamSpec (the GUI groups every option by it); leaving it unset is a
# fail-fast error rather than a silent default.
_REQUIRED = object()


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one algorithm parameter.

    The GUI resolves the widget from ``type``/``choices`` (never by inferring the
    type from a runtime value), fixing the previous type-by-value fragility and
    the missing-group ``ValueError`` gap.

    Attributes
    ----------
    key:
        Parameter name as passed to ``BoundaryMethod.compute(**params)``.
    type:
        The Python type of the parameter's value (e.g. ``float``, ``int``,
        ``bool``, ``str``). Drives GUI widget selection.
    default:
        Default value used when the parameter is not overridden.
    range:
        Optional ``(low, high)`` bounds for numeric parameters (inclusive).
    choices:
        Optional finite set of allowed values (renders as an enum/combobox).
    tunable:
        Whether this parameter is exposed to the interactive tuning workflow.
    group:
        **Mandatory.** The option group this parameter belongs to (GUI grouping).
    """

    key: str
    type: type
    default: Any
    range: tuple[float, float] | None = None
    choices: Sequence[Any] | None = None
    tunable: bool = False
    group: str = _REQUIRED  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("ParamSpec.key must be a non-empty string")
        if not isinstance(self.type, type):
            raise TypeError(
                f"ParamSpec.type must be a type, got {type(self.type).__name__}"
            )
        if self.group is _REQUIRED:
            raise ValueError(
                f"ParamSpec(key={self.key!r}) requires an explicit 'group'"
            )
        if not isinstance(self.group, str) or not self.group.strip():
            raise ValueError(
                f"ParamSpec(key={self.key!r}).group must be a non-empty string"
            )
        if self.range is not None:
            if not isinstance(self.range, tuple) or len(self.range) != 2:
                raise ValueError(
                    f"ParamSpec(key={self.key!r}).range must be a (low, high) "
                    f"tuple, got {self.range!r}"
                )
            low, high = self.range
            if low > high:
                raise ValueError(
                    f"ParamSpec(key={self.key!r}).range low ({low}) must be "
                    f"<= high ({high})"
                )
        if self.choices is not None:
            if len(self.choices) == 0:
                raise ValueError(
                    f"ParamSpec(key={self.key!r}).choices must be non-empty "
                    "when provided"
                )
            if self.default is not None and self.default not in self.choices:
                raise ValueError(
                    f"ParamSpec(key={self.key!r}).default {self.default!r} is "
                    f"not among choices {list(self.choices)!r}"
                )


class BoundaryMethod(ABC):
    """Strategy interface for a single RF boundary extraction algorithm.

    Implementations declare their :attr:`name` and :attr:`params_schema`, and map
    a smoothed IFF grid to a ``BoundaryContour`` via :meth:`compute`. A method is
    a pure algorithm: it must not know about the IO format, the DAG, the GUI, or
    any downstream consumer.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registry name of this algorithm (also the output folder name)."""

    @property
    @abstractmethod
    def params_schema(self) -> Sequence[ParamSpec]:
        """Ordered sequence of :class:`ParamSpec` describing this method's params."""

    @abstractmethod
    def compute(
        self,
        grid_u: np.ndarray,
        grid_v: np.ndarray,
        grid_z: np.ndarray,
        **params: Any,
    ) -> BoundaryContour | None:
        """Extract the boundary contour for a single response field.

        Parameters
        ----------
        grid_u, grid_v:
            (R, C) coordinate grids (axis 0 = U, axis 1 = V).
        grid_z:
            (R, C) IFF values; NaN where no data exists.
        **params:
            Algorithm parameters keyed by :attr:`params_schema` ``key`` values.

        Returns
        -------
        BoundaryContour or None if no valid boundary can be extracted.
        """
