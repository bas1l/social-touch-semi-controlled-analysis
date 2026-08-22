"""Round-trip fidelity of ``DagConfigModel``'s bypass and dependency writers.

``DagConfigModel`` is the GUI's *round-trip* editing model: a save must return a
file a human still recognises — same comments, same key order, same quoting,
same flow/block rendering.  These tests exercise the two mutations added for the
bypass feature against that contract, on a fixture written into ``tmp_path`` so
no shipped config is touched.

No ``QApplication`` is constructed here; the model is pure ruamel.yaml.
"""

from pathlib import Path

import pytest
from ruamel.yaml.comments import CommentedSeq

from utils.pipeline.dag_config_model import DagConfigModel


_FIXTURE = """\
# Fixture DAG config — header comment that must survive every save.

parameters:
  # Flow-style sequence, deliberately.
  kinect_configs: [alpha, beta]

tasks:
  # ============================================================
  # Section banner above the first task.
  # ============================================================

  # Root task: no upstream.
  root:
    category: foundation
    enabled: true
    bypass: false
    options:
      force_processing: false   # trailing comment on an option
    depends_on: []

  # Leaf task with a flow-style dependency list.
  leaf:
    category: spatial_sensitivity
    enabled: true
    bypass: false
    options:
      label: "double-quoted value"
    depends_on: [root]

  # A task written before 'bypass' existed: the key must be inserted, not appended.
  legacy:
    category: foundation
    enabled: false
    options:
      force_processing: true
    depends_on:
      - root
      - leaf
"""


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "fixture_dag.yaml"
    path.write_text(_FIXTURE, encoding="utf-8")
    return path


def _saved_lines(model: DagConfigModel) -> list[str]:
    model.save()
    return model.path.read_text(encoding="utf-8").splitlines()


def _task_key_order(path: Path, task_name: str) -> list[str]:
    return list(DagConfigModel(path)._get_task(task_name).keys())


# --- reading -----------------------------------------------------------------


def test_is_task_bypassed_reads_the_persisted_flag(config_path: Path):
    model = DagConfigModel(config_path)
    assert model.is_task_bypassed("root") is False
    assert model.dirty is False


def test_is_task_bypassed_raises_on_a_task_with_no_bypass_key(config_path: Path):
    model = DagConfigModel(config_path)
    with pytest.raises(KeyError, match="legacy"):
        model.is_task_bypassed("legacy")


def test_is_task_bypassed_raises_on_a_non_boolean_value(config_path: Path):
    model = DagConfigModel(config_path)
    model._get_task("root")["bypass"] = "yes"
    with pytest.raises(TypeError, match="non-boolean"):
        model.is_task_bypassed("root")


def test_is_task_bypassed_raises_on_an_unknown_task(config_path: Path):
    model = DagConfigModel(config_path)
    with pytest.raises(KeyError, match="nonexistent"):
        model.is_task_bypassed("nonexistent")


# --- set_task_bypassed: round trip -------------------------------------------


def test_set_task_bypassed_round_trips_through_save_and_reload(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("root", True)
    model.save()

    reloaded = DagConfigModel(config_path)
    assert reloaded.is_task_bypassed("root") is True
    # A real YAML boolean, not the string "True".
    assert reloaded._get_task("root")["bypass"] is True
    assert reloaded.is_task_bypassed("leaf") is False


def test_set_task_bypassed_preserves_comments_and_formatting(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("root", True)
    text = "\n".join(_saved_lines(model))

    assert "# Fixture DAG config — header comment that must survive every save." in text
    assert "# Section banner above the first task." in text
    assert "# Root task: no upstream." in text
    assert "force_processing: false   # trailing comment on an option" in text
    assert "kinect_configs: [alpha, beta]" in text
    assert 'label: "double-quoted value"' in text
    assert "depends_on: [root]" in text


def test_set_task_bypassed_keeps_the_repo_wide_key_order(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("root", True)
    model.save()
    assert _task_key_order(config_path, "root") == [
        "category",
        "enabled",
        "bypass",
        "options",
        "depends_on",
    ]


def test_a_first_write_inserts_bypass_immediately_after_enabled(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("legacy", True)
    model.save()

    assert _task_key_order(config_path, "legacy") == [
        "category",
        "enabled",
        "bypass",
        "options",
        "depends_on",
    ]
    lines = [line.strip() for line in config_path.read_text(encoding="utf-8").splitlines()]
    enabled_idx = lines.index("enabled: false")
    assert lines[enabled_idx + 1] == "bypass: true"


def test_toggling_bypass_off_again_round_trips(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("legacy", True)
    model.set_task_bypassed("legacy", False)
    model.save()

    reloaded = DagConfigModel(config_path)
    assert reloaded.is_task_bypassed("legacy") is False
    assert _task_key_order(config_path, "legacy")[:3] == ["category", "enabled", "bypass"]


def test_set_task_bypassed_raises_on_an_unknown_task(config_path: Path):
    model = DagConfigModel(config_path)
    with pytest.raises(KeyError, match="nonexistent"):
        model.set_task_bypassed("nonexistent", True)


# --- set_task_dependencies ---------------------------------------------------


def test_set_task_dependencies_preserves_a_flow_style_sequence(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_dependencies("leaf", ["root", "legacy"])
    text = "\n".join(_saved_lines(model))

    assert "depends_on: [root, legacy]" in text
    assert DagConfigModel(config_path).get_task_dependencies("leaf") == ["root", "legacy"]


def test_set_task_dependencies_keeps_a_block_style_sequence_in_block_style(
    config_path: Path,
):
    model = DagConfigModel(config_path)
    model.set_task_dependencies("legacy", ["leaf"])
    lines = _saved_lines(model)

    depends_idx = next(
        i for i, line in enumerate(lines) if line.strip() == "depends_on:"
    )
    assert lines[depends_idx + 1].strip() == "- leaf"
    assert DagConfigModel(config_path).get_task_dependencies("legacy") == ["leaf"]


def test_set_task_dependencies_writes_a_commented_seq(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_dependencies("leaf", ["root"])
    assert isinstance(model._get_task("leaf")["depends_on"], CommentedSeq)


def test_set_task_dependencies_can_empty_a_flow_style_list(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_dependencies("leaf", [])
    text = "\n".join(_saved_lines(model))

    assert "depends_on: []" in text
    assert DagConfigModel(config_path).get_task_dependencies("leaf") == []


def test_set_task_dependencies_leaves_the_rest_of_the_file_intact(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_dependencies("leaf", ["root", "legacy"])
    text = "\n".join(_saved_lines(model))

    assert "# Fixture DAG config — header comment that must survive every save." in text
    assert "# A task written before 'bypass' existed" in text
    assert "kinect_configs: [alpha, beta]" in text
    assert 'label: "double-quoted value"' in text


def test_set_task_dependencies_raises_on_an_unknown_task(config_path: Path):
    model = DagConfigModel(config_path)
    with pytest.raises(KeyError, match="nonexistent"):
        model.set_task_dependencies("nonexistent", [])


# --- dirty tracking ----------------------------------------------------------


def test_set_task_bypassed_marks_the_model_dirty(config_path: Path):
    model = DagConfigModel(config_path)
    assert model.dirty is False
    model.set_task_bypassed("root", True)
    assert model.dirty is True
    model.save()
    assert model.dirty is False


def test_a_first_bypass_write_marks_the_model_dirty(config_path: Path):
    model = DagConfigModel(config_path)
    model.set_task_bypassed("legacy", True)
    assert model.dirty is True


def test_set_task_dependencies_marks_the_model_dirty(config_path: Path):
    model = DagConfigModel(config_path)
    assert model.dirty is False
    model.set_task_dependencies("leaf", ["root"])
    assert model.dirty is True


def test_a_read_never_marks_the_model_dirty(config_path: Path):
    model = DagConfigModel(config_path)
    model.is_task_bypassed("root")
    model.get_task_dependencies("leaf")
    assert model.dirty is False


# --- the shipped config -------------------------------------------------------


def test_the_shipped_processing_config_round_trips_a_bypass_toggle(tmp_path: Path):
    """The barrier's two-line flow-style ``depends_on`` is the awkward case."""
    shipped = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "analyse_workflow_processing_dag.yaml"
    )
    working = tmp_path / shipped.name
    working.write_text(shipped.read_text(encoding="utf-8"), encoding="utf-8")

    model = DagConfigModel(working)
    barrier = "spatial_extract_boundaries"
    before = model.get_task_dependencies(barrier)
    model.set_task_bypassed(barrier, True)
    model.save()

    reloaded = DagConfigModel(working)
    assert reloaded.is_task_bypassed(barrier) is True
    assert reloaded.get_task_dependencies(barrier) == before
    assert list(reloaded._get_task(barrier).keys())[:3] == [
        "category",
        "enabled",
        "bypass",
    ]
