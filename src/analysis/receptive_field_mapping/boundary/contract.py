"""Formal boundary contract for pluggable RF boundary extraction.

This module defines the single authoritative data contract that every boundary
extraction algorithm must satisfy: :class:`BoundaryContour`, a frozen dataclass
exposing exactly the 11 uniform geometric fields already produced identically by
the three existing detectors (radial-foot / gradient-ridge / inflection), plus
provenance (``method_name``) and an optional, immutable ``diagnostic_fields``
channel for method-specific figure overlays.

The contract is deliberately *method-blind*: a ``BoundaryContour`` knows nothing
about which algorithm produced it (beyond the free-form ``method_name`` string),
nothing about the NPZ on-disk layout, and nothing about the GUI. Downstream
consumers depend only on this abstract shape (DIP), so any two algorithms behind
the contract are interchangeable (LSP).

Fail-fast (CLAUDE.md): :func:`validate_contour` raises loudly on any missing,
malformed, or wrong-shape field. There are no silent fallbacks, defaults, or
sentinel returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

__all__ = ["BoundaryContour", "validate_contour"]


# An immutable, shared empty mapping used as the default diagnostic channel.
_EMPTY_DIAGNOSTICS: Mapping[str, np.ndarray] = MappingProxyType({})

# Real numeric scalar types accepted for the scalar geometric fields.
# ``bool`` is a subclass of ``int`` and is explicitly rejected below.
_REAL_SCALAR_TYPES = (int, float, np.integer, np.floating)


@dataclass(frozen=True)
class BoundaryContour:
    """The uniform geometric contract for an RF boundary contour.

    Instances are immutable. The 11 geometric fields are exactly those already
    emitted identically by every existing detector; ``method_name`` records which
    algorithm produced the contour (provenance only — no consumer branches on it);
    ``diagnostic_fields`` is an optional, immutable ``name -> np.ndarray`` channel
    a method may populate *solely* for its own figure overlays (e.g. the radial
    ``lmax`` field or the gradient magnitude field). No computation stage may read
    ``diagnostic_fields``; consumers access it only by capability check, tolerating
    absence.
    """

    contour_uv: np.ndarray
    """(N, 2) UV coordinates of the closed boundary contour."""

    area_uv: float
    """Polygon area in UV space (shoelace formula)."""

    perimeter_uv: float
    """Polygon perimeter in UV space."""

    circularity: float
    """Shape compactness: 4*pi*area / perimeter^2. 1.0 = perfect circle."""

    centroid_uv: tuple[float, float]
    """Polygon centroid in UV space (u, v)."""

    peak_uv: tuple[float, float]
    """UV coordinates of the global response maximum (grid-cell resolution)."""

    pca_major_uv: float
    """PCA major axis length (2 sigma) in UV space."""

    pca_minor_uv: float
    """PCA minor axis length (2 sigma) in UV space."""

    pca_orientation_deg: float
    """Major axis orientation in degrees."""

    mean_iff_on_contour: float
    """Mean IFF value sampled along the contour path."""

    iff_at_centroid: float
    """IFF value sampled at the polygon centroid via bilinear interpolation."""

    method_name: str
    """Provenance: the registry name of the algorithm that produced this contour."""

    diagnostic_fields: Mapping[str, np.ndarray] = field(default=_EMPTY_DIAGNOSTICS)
    """Optional, immutable ``name -> np.ndarray`` overlay channel (default empty)."""

    def __post_init__(self) -> None:
        # Normalise the diagnostic channel to an immutable mapping so the frozen
        # contract cannot be mutated through this reference after construction.
        # This is contract enforcement (immutability), not a silent fallback.
        diag = self.diagnostic_fields
        if not isinstance(diag, MappingProxyType):
            if not isinstance(diag, Mapping):
                raise TypeError(
                    "BoundaryContour.diagnostic_fields must be a Mapping, "
                    f"got {type(diag).__name__}"
                )
            object.__setattr__(
                self, "diagnostic_fields", MappingProxyType(dict(diag))
            )

    def has_diagnostic(self, name: str) -> bool:
        """Capability check: does this contour carry diagnostic field *name*?

        Consumers use this to decide whether a method-specific overlay is
        available, tolerating absence rather than branching on method identity.
        """
        return name in self.diagnostic_fields


def _is_real_scalar(value: object) -> bool:
    """True if *value* is a real numeric scalar (not a bool, not an array)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, _REAL_SCALAR_TYPES)


def _validate_pair(name: str, value: object) -> None:
    """Validate a 2-component UV coordinate field (centroid_uv / peak_uv)."""
    if isinstance(value, np.ndarray):
        if value.shape != (2,):
            raise ValueError(
                f"BoundaryContour.{name} must have shape (2,), got {value.shape}"
            )
        components = (value[0], value[1])
    elif isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"BoundaryContour.{name} must have length 2, got {len(value)}"
            )
        components = (value[0], value[1])
    else:
        raise TypeError(
            f"BoundaryContour.{name} must be a length-2 tuple/list/ndarray, "
            f"got {type(value).__name__}"
        )
    for i, comp in enumerate(components):
        if not _is_real_scalar(comp):
            raise TypeError(
                f"BoundaryContour.{name}[{i}] must be a real scalar, "
                f"got {type(comp).__name__}"
            )


def validate_contour(contour: object) -> BoundaryContour:
    """Assert that *contour* satisfies the :class:`BoundaryContour` contract.

    Fail-fast validator called at the fan-in boundary. Raises ``TypeError`` /
    ``ValueError`` on any missing, wrong-type, or wrong-shape field. Returns the
    validated contour unchanged so it can be used inline.

    NaN scalar metrics are permitted (a degenerate polygon may legitimately yield
    ``circularity = NaN``), but ``contour_uv`` must be a finite (N, 2) float array
    with ``N >= 3``: a geometric contour with fewer than three vertices, or with
    non-finite coordinates, is malformed.
    """
    if not isinstance(contour, BoundaryContour):
        raise TypeError(
            f"expected a BoundaryContour instance, got {type(contour).__name__}"
        )

    # --- contour_uv: (N, 2) finite float array, N >= 3 ---------------------
    cuv = contour.contour_uv
    if not isinstance(cuv, np.ndarray):
        raise TypeError(
            f"BoundaryContour.contour_uv must be an np.ndarray, "
            f"got {type(cuv).__name__}"
        )
    if cuv.ndim != 2 or cuv.shape[1] != 2:
        raise ValueError(
            f"BoundaryContour.contour_uv must have shape (N, 2), got {cuv.shape}"
        )
    if cuv.shape[0] < 3:
        raise ValueError(
            f"BoundaryContour.contour_uv must have at least 3 vertices, "
            f"got {cuv.shape[0]}"
        )
    if not np.issubdtype(cuv.dtype, np.floating):
        raise TypeError(
            f"BoundaryContour.contour_uv must be a floating-point array, "
            f"got dtype {cuv.dtype}"
        )
    if not np.all(np.isfinite(cuv)):
        raise ValueError(
            "BoundaryContour.contour_uv must contain only finite values "
            "(found NaN or inf)"
        )

    # --- scalar geometric metrics ------------------------------------------
    for name in (
        "area_uv",
        "perimeter_uv",
        "circularity",
        "pca_major_uv",
        "pca_minor_uv",
        "pca_orientation_deg",
        "mean_iff_on_contour",
        "iff_at_centroid",
    ):
        value = getattr(contour, name)
        if not _is_real_scalar(value):
            raise TypeError(
                f"BoundaryContour.{name} must be a real scalar, "
                f"got {type(value).__name__}"
            )

    # --- 2-component UV coordinate fields ----------------------------------
    _validate_pair("centroid_uv", contour.centroid_uv)
    _validate_pair("peak_uv", contour.peak_uv)

    # --- provenance --------------------------------------------------------
    if not isinstance(contour.method_name, str):
        raise TypeError(
            f"BoundaryContour.method_name must be a str, "
            f"got {type(contour.method_name).__name__}"
        )
    if not contour.method_name.strip():
        raise ValueError("BoundaryContour.method_name must be a non-empty string")

    # --- diagnostic channel ------------------------------------------------
    diag = contour.diagnostic_fields
    if not isinstance(diag, Mapping):
        raise TypeError(
            f"BoundaryContour.diagnostic_fields must be a Mapping, "
            f"got {type(diag).__name__}"
        )
    for key, value in diag.items():
        if not isinstance(key, str):
            raise TypeError(
                f"BoundaryContour.diagnostic_fields keys must be str, "
                f"got {type(key).__name__}"
            )
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"BoundaryContour.diagnostic_fields['{key}'] must be an "
                f"np.ndarray, got {type(value).__name__}"
            )

    return contour
