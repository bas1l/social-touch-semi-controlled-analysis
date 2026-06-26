"""Save / load a preprocessing pipeline to a YAML config file.

A pipeline config is a flat, ordered list of operator steps — exactly the
``PreprocStep`` queue the user has built in the GUI.  Serialising the queue lets a
researcher snapshot a tuned pipeline and reload it later (or share it).

On-disk format (``ruamel.yaml`` round-trip — CLAUDE.md mandates ruamel, not
PyYAML)::

    pipeline:
      - op: gaussian
        params:
          sigma: 5.0
      - op: median
        params:
          size: 5

Loading is **fail-fast** (CLAUDE.md): an unknown operator id, an unknown
parameter name, or a structurally malformed file raises ``ValueError`` — there is
no silent skipping, defaulting, or version-detection fallback.  Parameter values
are coerced to the type declared by the operator's ``ParamSpec`` so a freshly
loaded step is indistinguishable from one built in the GUI.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .operator_registry import OPERATORS, ParamSpec
from .pipeline import PreprocStep


def _make_yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _coerce(p: ParamSpec, value) -> object:
    """Coerce a loaded YAML scalar to the native type declared by *p*."""
    if p.kind == "float":
        return float(value)
    if p.kind == "int":
        return int(value)
    if p.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"param '{p.name}' must be a boolean, got {value!r}")
        return bool(value)
    if p.kind == "choice":
        text = str(value)
        if text not in {str(c) for c in p.choices}:
            raise ValueError(
                f"param '{p.name}' must be one of {p.choices}, got {value!r}"
            )
        return text
    raise ValueError(f"unknown param kind: {p.kind}")


def steps_to_doc(steps: list[PreprocStep]) -> CommentedMap:
    """Build the round-trip YAML document for *steps*."""
    seq = CommentedSeq()
    for step in steps:
        entry = CommentedMap()
        entry["op"] = step.op_id
        params = CommentedMap()
        for name, val in step.params.items():
            params[name] = val
        entry["params"] = params
        seq.append(entry)
    doc = CommentedMap()
    doc["pipeline"] = seq
    return doc


def save_pipeline(path: Path, steps: list[PreprocStep]) -> None:
    """Write *steps* to *path* as a YAML pipeline config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        _make_yaml().dump(steps_to_doc(steps), fh)


def load_pipeline(path: Path) -> list[PreprocStep]:
    """Read a YAML pipeline config from *path* into a list of ``PreprocStep``.

    Raises ``ValueError`` on any structural or schema violation (fail-fast).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = _make_yaml().load(fh)

    if not isinstance(doc, dict) or "pipeline" not in doc:
        raise ValueError(f"{path.name}: not a pipeline config (missing 'pipeline' key)")
    raw_steps = doc["pipeline"]
    if raw_steps is None:
        return []
    if not isinstance(raw_steps, list):
        raise ValueError(f"{path.name}: 'pipeline' must be a list of steps")

    steps: list[PreprocStep] = []
    for i, entry in enumerate(raw_steps):
        if not isinstance(entry, dict) or "op" not in entry:
            raise ValueError(f"{path.name}: step {i} missing 'op' key")
        op_id = str(entry["op"])
        if op_id not in OPERATORS:
            raise ValueError(f"{path.name}: step {i} has unknown operator '{op_id}'")
        spec = OPERATORS[op_id]
        valid_params = {p.name: p for p in spec.params}

        raw_params = entry.get("params") or {}
        if not isinstance(raw_params, dict):
            raise ValueError(f"{path.name}: step {i} 'params' must be a mapping")
        unknown = set(raw_params) - set(valid_params)
        if unknown:
            raise ValueError(
                f"{path.name}: step {i} ('{op_id}') has unknown param(s): "
                f"{', '.join(sorted(unknown))}"
            )

        # Start from the operator defaults so an omitted param is the schema
        # default (not a silent fallback — the schema *is* the source of truth),
        # then overlay every explicitly supplied value, coerced to its type.
        params = spec.default_params()
        for name, val in raw_params.items():
            params[name] = _coerce(valid_params[name], val)
        steps.append(PreprocStep(op_id, params))

    return steps
