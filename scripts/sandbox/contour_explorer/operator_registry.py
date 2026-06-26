"""Operator registry: parameter/operator schemas and the operator catalogue.

``ParamSpec`` / ``OperatorSpec`` describe each operator's tunables and which
``op_*`` callable implements it; ``OPERATOR_LIST`` is the ordered palette and
``OPERATORS`` the id -> spec lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

from .operators import (
    op_doh,
    op_dog,
    op_directional,
    op_frangi,
    op_gaussian,
    op_gaussian_nth,
    op_gradient_magnitude,
    op_hessian_eigval,
    op_laplacian,
    op_log,
    op_median,
    op_meijering,
    op_prewitt,
    op_sato,
    op_scharr,
    op_sobel,
)


@dataclass
class ParamSpec:
    name: str
    kind: str  # "float" | "int" | "bool" | "choice"
    default: object
    minimum: float = 0.0
    maximum: float = 100.0
    step: float = 1.0
    decimals: int = 2
    choices: tuple = ()


@dataclass
class OperatorSpec:
    op_id: str
    label: str
    group: str
    fn: object
    params: tuple = ()

    def default_params(self) -> dict:
        return {p.name: p.default for p in self.params}


_SIGMA = lambda default=5.0: ParamSpec("sigma", "float", default, 0.0, 50.0, 0.5, 2)  # noqa: E731


OPERATOR_LIST = [
    # --- 0th order: smoothing ---
    OperatorSpec("gaussian", "Gaussian smooth", "0 · Smoothing",
                 op_gaussian, (_SIGMA(5.0),)),
    OperatorSpec("median", "Median filter", "0 · Smoothing",
                 op_median, (ParamSpec("size", "int", 5, 1, 31, 2),)),
    # --- 1st order: edges / slopes ---
    OperatorSpec("gradient_magnitude", "Gradient magnitude |grad z|", "1 · Edges",
                 op_gradient_magnitude, (_SIGMA(5.0),)),
    OperatorSpec("sobel", "Sobel edges", "1 · Edges", op_sobel, ()),
    OperatorSpec("scharr", "Scharr edges (isotropic)", "1 · Edges", op_scharr, ()),
    OperatorSpec("prewitt", "Prewitt edges", "1 · Edges", op_prewitt, ()),
    OperatorSpec("directional", "Directional derivative", "1 · Edges",
                 op_directional, (_SIGMA(5.0),
                                  ParamSpec("angle_deg", "float", 0.0, -180.0, 180.0, 5.0, 1))),
    # --- 2nd order: curvature / ridges / foot ---
    OperatorSpec("laplacian", "Laplacian (zero-cross = inflection)", "2 · Curvature",
                 op_laplacian, (_SIGMA(5.0),)),
    OperatorSpec("log", "Laplacian of Gaussian (blob/foot)", "2 · Curvature",
                 op_log, (_SIGMA(5.0),)),
    OperatorSpec("hessian_eigval", "Hessian lambda_max (foot)", "2 · Curvature",
                 op_hessian_eigval, (_SIGMA(5.0),)),
    OperatorSpec("doh", "Determinant of Hessian", "2 · Curvature",
                 op_doh, (_SIGMA(5.0),)),
    OperatorSpec("frangi", "Frangi ridge", "2 · Ridges",
                 op_frangi, (ParamSpec("sigma_min", "float", 1.0, 0.0, 50.0, 0.5, 2),
                             ParamSpec("sigma_max", "float", 8.0, 0.0, 50.0, 0.5, 2),
                             ParamSpec("black_ridges", "bool", False))),
    OperatorSpec("sato", "Sato tubeness", "2 · Ridges",
                 op_sato, (ParamSpec("sigma_min", "float", 1.0, 0.0, 50.0, 0.5, 2),
                           ParamSpec("sigma_max", "float", 8.0, 0.0, 50.0, 0.5, 2),
                           ParamSpec("black_ridges", "bool", False))),
    OperatorSpec("meijering", "Meijering neuriteness", "2 · Ridges",
                 op_meijering, (ParamSpec("sigma_min", "float", 1.0, 0.0, 50.0, 0.5, 2),
                                ParamSpec("sigma_max", "float", 8.0, 0.0, 50.0, 0.5, 2),
                                ParamSpec("black_ridges", "bool", False))),
    # --- 3rd order+ : generic Gaussian n-th derivative ---
    OperatorSpec("gaussian_nth", "Gaussian n-th derivative", "3 · Advanced",
                 op_gaussian_nth, (_SIGMA(5.0),
                                   ParamSpec("order_u", "int", 1, 0, 4, 1),
                                   ParamSpec("order_v", "int", 1, 0, 4, 1))),
    # --- scale-space ---
    OperatorSpec("dog", "Difference of Gaussians", "4 · Scale-space",
                 op_dog, (ParamSpec("sigma_low", "float", 3.0, 0.0, 50.0, 0.5, 2),
                          ParamSpec("sigma_high", "float", 5.0, 0.0, 50.0, 0.5, 2))),
]

OPERATORS: dict[str, OperatorSpec] = {o.op_id: o for o in OPERATOR_LIST}
