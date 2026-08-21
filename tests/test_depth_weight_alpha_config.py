"""Phase 5: ``depth_weight_alpha`` is configurable, required, and recorded.

Phase 4 proved the *arithmetic* of the weighting. These tests cover the thread
that carries the exponent from YAML to the estimator, and the three claims the
plan makes about it:

1. **No default anywhere.** Every level of the call chain — ``vertex_weights``,
   ``_compute_touch_rf``, ``run_single_touch_rf_mapping`` and
   ``spatial_map_single_touch_flow`` — declares ``depth_weight_alpha`` without a
   default, so an un-passed alpha is a ``TypeError`` at the boundary rather than a
   silently-assumed ``1.0`` several frames deeper. A signature test is used
   alongside the call tests because a default added later would make the call
   tests quietly stop failing while still passing.
2. **The DAG owns the value.** The stage registry reads it from config and raises
   naming the key when it is absent; it never falls back.
3. **The run records it.** ``single_touch_rf_summary.json`` carries the alpha, the
   playback-cache schema version, and every sidecar path, so two runs at different
   alphas are distinguishable on disk and a config change makes the stage stale.

The GUI half of the phase is exercised headlessly: only the pure grouping
resolver is called, so no ``QApplication`` is constructed.

Extended in Phase 6 with ``TestViewersDagCarriesTheSameAlpha``. The playback
viewer now applies the same depth weighting the pipeline applies, so it needs the
same exponent — and it reads it from a *different* config file, because the two
entry-point scripts share no module by design. Two configs describing one number
drift silently; that class pins them together.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from _vendor import DagConfigHandler
from analysis.receptive_field_mapping.data.touch_playback_data import (
    PLAYBACK_CACHE_SCHEMA_VERSION,
    DepthFieldProvenance,
    PlaybackData,
    PlaybackSessionData,
)
from analysis.receptive_field_mapping.data.vertex_weights import vertex_weights
from analysis.receptive_field_mapping.pipelines import rf_single_touch_pipeline as pipe
from utils.gui.analysis_runner_gui import task_detail_panel as tdp

import rf_accumulator_fixtures as fixtures

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSING_DAG = REPO_ROOT / "configs" / "analyse_workflow_processing_dag.yaml"
VIEWERS_DAG = REPO_ROOT / "configs" / "analyse_workflow_viewers_dag.yaml"
STAGE = "spatial_map_single_touch"


def _load_round_trip(path: Path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with path.open("r", encoding="utf-8") as fh:
        return yaml.load(fh)


def _workflow_module():
    """Import ``scripts/analysis_workflow_processing.py`` (not on ``pythonpath``)."""
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import analysis_workflow_processing  # noqa: PLC0415 — deliberate late import

    return analysis_workflow_processing


# ----------------------------------------------------------------------
# 1. No default at any level of the chain
# ----------------------------------------------------------------------


def _param(func, name: str) -> inspect.Parameter:
    params = inspect.signature(func).parameters
    assert name in params, f"{func!r} does not declare {name!r}"
    return params[name]


class TestNoDefaultAnywhere:
    """The exponent must never be supplied by omission."""

    def test_vertex_weights_alpha_has_no_default(self):
        assert _param(vertex_weights, "alpha").default is inspect.Parameter.empty

    def test_compute_touch_rf_alpha_has_no_default(self):
        p = _param(pipe._compute_touch_rf, "depth_weight_alpha")
        assert p.default is inspect.Parameter.empty

    def test_run_single_touch_rf_mapping_alpha_has_no_default(self):
        p = _param(pipe.run_single_touch_rf_mapping, "depth_weight_alpha")
        assert p.default is inspect.Parameter.empty

    def test_flow_alpha_is_keyword_only_and_has_no_default(self):
        flow = _workflow_module().spatial_map_single_touch_flow
        p = _param(flow.fn, "depth_weight_alpha")
        assert p.default is inspect.Parameter.empty
        assert p.kind is inspect.Parameter.KEYWORD_ONLY


class TestMissingAlphaIsATypeError:
    """Not a warning, not a default — the call does not happen at all."""

    def test_vertex_weights_without_alpha(self):
        with pytest.raises(TypeError):
            vertex_weights(np.array([1.0, 2.0]))

    def test_compute_touch_rf_without_alpha(self):
        with pytest.raises(TypeError):
            pipe._compute_touch_rf(
                fixtures.worked_example_touch(),
                fixtures.WORKED_EXAMPLE_N_VERTICES,
                "iff",
            )

    def test_run_single_touch_rf_mapping_without_alpha(self, tmp_path):
        """Raises before touching the filesystem: the binding fails, not the work."""
        with pytest.raises(TypeError):
            pipe.run_single_touch_rf_mapping(
                input_items=[],
                output_dir=tmp_path,
            )

    def test_flow_without_alpha(self, tmp_path):
        flow = _workflow_module().spatial_map_single_touch_flow
        with pytest.raises(TypeError):
            flow.fn(input_items=[(tmp_path / "s.csv", tmp_path)])


# ----------------------------------------------------------------------
# 2. Config
# ----------------------------------------------------------------------


class TestProcessingDagConfig:
    def test_stage_declares_depth_weight_alpha(self):
        data = _load_round_trip(PROCESSING_DAG)
        options = data["tasks"][STAGE]["options"]
        assert "depth_weight_alpha" in options
        assert float(options["depth_weight_alpha"]) == 1.0

    def test_alpha_is_a_float_not_an_int(self):
        """The GUI renders it as free text cast by ``type(original_value)``.

        An int in YAML would make the panel reject ``0.5``, so the shipped value
        must parse as a float.
        """
        data = _load_round_trip(PROCESSING_DAG)
        value = data["tasks"][STAGE]["options"]["depth_weight_alpha"]
        assert isinstance(value, float)
        assert not isinstance(value, bool)
        assert tdp.TaskDetailPanel._native_type(value) is float

    def test_the_contact_depth_field_block_survived_the_edit(self):
        """Phase 2.5's nested block and its explanatory comment are still there."""
        data = _load_round_trip(PROCESSING_DAG)
        block = data["tasks"][STAGE]["options"]["contact_depth_field"]
        assert block["blocks_stage_dir"] == "blocks_rf_centered"
        assert "block_csv_stem_suffix" not in block
        text = PROCESSING_DAG.read_text(encoding="utf-8")
        assert "Where the per-vertex contact-depth-field parquet sidecars live" in text

    @pytest.mark.parametrize("path", [PROCESSING_DAG, VIEWERS_DAG], ids=lambda p: p.name)
    def test_both_dag_configs_still_parse(self, path):
        data = _load_round_trip(path)
        assert "tasks" in data and data["tasks"]

    def test_the_combined_reference_dag_does_not_define_this_stage(self):
        """Task 5.6's precondition, asserted rather than remembered.

        ``configs/analyse_workflow_dag.yaml`` carries no ``spatial_map_single_touch``
        task, so there is nothing to mirror the option into. If that ever changes,
        this test fails and the mirror becomes required.
        """
        combined = REPO_ROOT / "configs" / "analyse_workflow_dag.yaml"
        data = _load_round_trip(combined)
        assert STAGE not in data["tasks"]


class TestStageRegistryParams:
    """The registry reads alpha from config and never invents one."""

    @staticmethod
    def _stage_params(config_path: Path) -> dict:
        module = _workflow_module()
        handler = DagConfigHandler(config_path)
        stages = module._build_pipeline_stages(handler, [])
        descriptor = next(s for s in stages if s["name"] == STAGE)
        return descriptor["params"]()

    def test_params_carry_the_configured_alpha(self):
        params = self._stage_params(PROCESSING_DAG)
        assert params["depth_weight_alpha"] == 1.0
        assert isinstance(params["depth_weight_alpha"], float)

    def test_params_match_the_flow_signature(self):
        """Every key the registry emits is a parameter the flow accepts."""
        module = _workflow_module()
        params = self._stage_params(PROCESSING_DAG)
        accepted = set(
            inspect.signature(module.spatial_map_single_touch_flow.fn).parameters
        )
        assert set(params) <= accepted

    def test_a_config_without_the_key_raises_naming_it(self, tmp_path):
        yaml = YAML()
        yaml.preserve_quotes = True
        with PROCESSING_DAG.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh)
        del data["tasks"][STAGE]["options"]["depth_weight_alpha"]
        stripped = tmp_path / "no_alpha.yaml"
        with stripped.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)

        with pytest.raises(ValueError, match="depth_weight_alpha"):
            self._stage_params(stripped)


# ----------------------------------------------------------------------
# 2b. The viewers DAG carries the same number (Phase 6)
# ----------------------------------------------------------------------

VIEWER_STAGE = "explore_touch_playback"


def _viewers_module():
    """Import ``scripts/analysis_workflow_viewers.py`` (not on ``pythonpath``)."""
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import analysis_workflow_viewers  # noqa: PLC0415 — deliberate late import

    return analysis_workflow_viewers


class TestViewersDagCarriesTheSameAlpha:
    """The playback viewer draws the weighted map, so it needs the same exponent.

    Two configs describe one number. They are separate files because the two entry
    points share no module by design, so the only thing holding them together is
    this test — and the comment each config carries pointing at the other.
    """

    def test_the_viewer_stage_declares_alpha(self):
        data = _load_round_trip(VIEWERS_DAG)
        options = data["tasks"][VIEWER_STAGE]["options"]
        assert "depth_weight_alpha" in options
        value = options["depth_weight_alpha"]
        assert isinstance(value, float) and not isinstance(value, bool)

    def test_the_two_configs_agree(self):
        """A viewer at a different alpha from the pipeline draws a different map.

        That is the exact failure the nine-site accumulator merge existed to
        prevent, reintroduced one layer up as a config drift instead of a code
        duplication — so it is pinned here rather than left to a comment.
        """
        processing = _load_round_trip(PROCESSING_DAG)["tasks"][STAGE]["options"]
        viewers = _load_round_trip(VIEWERS_DAG)["tasks"][VIEWER_STAGE]["options"]
        assert float(viewers["depth_weight_alpha"]) == float(
            processing["depth_weight_alpha"]
        )

    def test_the_viewer_flow_has_no_default_for_alpha(self):
        module = _viewers_module()
        params = inspect.signature(
            module.explore_touch_playback_flow.fn
        ).parameters
        alpha = params["depth_weight_alpha"]
        assert alpha.default is inspect.Parameter.empty
        assert alpha.kind is inspect.Parameter.KEYWORD_ONLY

    def test_a_viewers_config_without_the_key_raises_naming_it(self, tmp_path):
        module = _viewers_module()
        yaml = YAML()
        yaml.preserve_quotes = True
        with VIEWERS_DAG.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh)
        del data["tasks"][VIEWER_STAGE]["options"]["depth_weight_alpha"]
        stripped = tmp_path / "no_alpha_viewers.yaml"
        with stripped.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)

        handler = DagConfigHandler(stripped)
        with pytest.raises(ValueError, match="depth_weight_alpha"):
            module._required_option(handler, VIEWER_STAGE, "depth_weight_alpha")

    def test_the_key_is_read_from_config_not_invented(self):
        module = _viewers_module()
        handler = DagConfigHandler(VIEWERS_DAG)
        assert (
            float(module._required_option(handler, VIEWER_STAGE, "depth_weight_alpha"))
            == 1.0
        )


# ----------------------------------------------------------------------
# 3. GUI task detail panel
# ----------------------------------------------------------------------


class TestGuiOptionGrouping:
    """``spatial_map_single_touch`` is a grouped task: every option needs a group.

    Only the pure resolver is exercised — no ``QApplication``, no widgets — which
    is what makes this runnable headlessly. ``_insert_grouped`` calls exactly this
    function for exactly these keys, so an ungrouped option fails here for the same
    reason it would fail the render.
    """

    @staticmethod
    def _option_keys() -> list[str]:
        data = _load_round_trip(PROCESSING_DAG)
        return [k for k in data["tasks"][STAGE]["options"] if k != "force_processing"]

    def test_the_stage_is_grouped(self):
        assert STAGE in tdp._GROUPED_TASKS

    def test_every_option_of_the_stage_resolves_a_group(self):
        keys = self._option_keys()
        assert "depth_weight_alpha" in keys
        for key in keys:
            group = tdp.option_group_of(STAGE, key)
            assert group in {gid for gid, _ in tdp._OPTION_GROUPS}

    def test_an_unknown_option_still_raises(self):
        """The guard is real, not disabled by a catch-all group."""
        with pytest.raises(ValueError, match="has no group"):
            tdp.option_group_of(STAGE, "not_an_option")


# ----------------------------------------------------------------------
# 4. Provenance in the summary JSON
# ----------------------------------------------------------------------


def _fake_playback(n_vertices: int, sidecar: str) -> PlaybackData:
    touch = fixtures.worked_example_touch()
    return PlaybackData(
        session_data=PlaybackSessionData(
            forearm_vertices=np.zeros((n_vertices, 3), dtype=np.float64),
            forearm_vertex_colors=None,
        ),
        block_order_ids=[touch.block_order_id],
        trial_ids_by_block={touch.block_order_id: [touch.trial_id]},
        touches_by_block_trial={(touch.block_order_id, touch.trial_id): [touch]},
        depth_field_provenance=[
            DepthFieldProvenance(
                source_block_file="ST99-01_semicontrolled_block-order-02_merged_data.csv",
                sidecar_path=sidecar,
                coordinate_space="rf_centered",
            )
        ],
    )


@pytest.fixture
def rf_run(tmp_path, monkeypatch):
    """Drive ``run_single_touch_rf_mapping`` over a synthetic session.

    No experimental data is read: the playback loader and the PLY resolver are
    both replaced, so what is under test is the *stage's* bookkeeping — the
    sentinel it writes — not the loader.
    """
    session_id = "ST99-01"
    merged_root = tmp_path / "3_merged" / session_id
    (merged_root / "blocks_rf_centered").mkdir(parents=True)
    csv_path = merged_root / f"{session_id}_semicontrolled_aggregated.csv"
    csv_path.write_text("placeholder\n", encoding="utf-8")

    preparation_dir = tmp_path / "4_analysed" / "preparation"
    preparation_dir.mkdir(parents=True)
    (preparation_dir / f"{session_id}_prepared.csv").write_text("x\n", encoding="utf-8")

    ply = merged_root / f"{session_id}_forearm.ply"
    ply.write_text("ply\n", encoding="utf-8")
    sidecar = str(merged_root / "blocks_rf_centered" / "block-order-02.parquet")

    monkeypatch.setattr(pipe, "resolve_forearm_ply", lambda *_a, **_k: ply)
    monkeypatch.setattr(
        pipe,
        "load_playback_data",
        lambda **_k: _fake_playback(fixtures.WORKED_EXAMPLE_N_VERTICES, sidecar),
    )

    output_dir = tmp_path / "4_analysed" / "spatial_map_single_touch"

    def _run(alpha: float) -> dict:
        pipe.run_single_touch_rf_mapping(
            input_items=[(csv_path, tmp_path)],
            output_dir=output_dir,
            depth_weight_alpha=alpha,
            force=True,
            preparation_dir=preparation_dir,
            contact_depth_field={
                "blocks_stage_dir": "blocks_rf_centered",
            },
        )
        sentinel = output_dir / session_id / "single_touch_rf_summary.json"
        return json.loads(sentinel.read_text(encoding="utf-8"))

    _run.sidecar = sidecar  # type: ignore[attr-defined]
    return _run


class TestSummaryJsonRecordsTheRun:
    def test_alpha_is_recorded(self, rf_run):
        assert rf_run(1.0)["depth_weight_alpha"] == 1.0

    def test_two_alphas_are_distinguishable_on_disk(self, rf_run):
        """A config change must make the two runs different files, not just
        different numbers inside the ``.npz``."""
        assert rf_run(0.0)["depth_weight_alpha"] == 0.0
        assert rf_run(1.0)["depth_weight_alpha"] == 1.0

    def test_cache_schema_version_is_recorded(self, rf_run):
        summary = rf_run(1.0)
        assert (
            summary["playback_cache_schema_version"] == PLAYBACK_CACHE_SCHEMA_VERSION
        )

    def test_sidecar_paths_and_declared_space_are_recorded(self, rf_run):
        summary = rf_run(1.0)
        blocks = summary["contact_depth_field"]["blocks"]
        assert [b["sidecar_path"] for b in blocks] == [rf_run.sidecar]
        assert [b["coordinate_space"] for b in blocks] == ["rf_centered"]

    def test_the_summary_is_json_serialisable_scalars(self, rf_run):
        """No numpy scalars leak in — ``json.dump`` would already have failed,
        but the cast is what keeps the file diffable across runs."""
        summary = rf_run(1.0)
        assert isinstance(summary["depth_weight_alpha"], float)
        assert isinstance(summary["playback_cache_schema_version"], int)
