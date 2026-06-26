"""Preprocessing pipeline model: an ordered queue of operator steps."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from .operator_registry import OPERATORS, OperatorSpec


@dataclass
class PreprocStep:
    op_id: str
    params: dict = dataclass_field(default_factory=dict)

    def spec(self) -> OperatorSpec:
        return OPERATORS[self.op_id]

    def label(self) -> str:
        spec = self.spec()
        if not self.params:
            return spec.label
        summary = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{spec.label}  [{summary}]"

    def apply(self, fieldarr: np.ndarray) -> np.ndarray:
        return self.spec().fn(fieldarr, **self.params)


def run_pipeline(raw: np.ndarray, steps: list[PreprocStep]) -> np.ndarray:
    """Fold every step over a copy of *raw*, in queue order."""
    fieldarr = raw.astype(float, copy=True)
    for step in steps:
        fieldarr = step.apply(fieldarr)
    return fieldarr
