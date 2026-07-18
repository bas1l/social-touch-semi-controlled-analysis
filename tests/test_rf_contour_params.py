"""Unit tests for ``ContourParamToggles``, ``GestureContourParams.effective_*()``,
and the ``param_enabled`` JSON round-trip in ``rf_contour_params_io``.

Covers the "RF Contour Tuning — Per-Parameter Toggles" plan's Testing Plan ->
Unit Tests section: default toggle state, fail-fast validation (non-bool
fields, unknown sub-keys, non-mapping ``param_enabled``), the
``to_dict``/``from_dict`` round-trip, the four ``effective_*()`` skip
resolvers (tested both ON and OFF per guide `03`), and the
``save_contour_params`` / ``load_contour_params`` JSON round-trip including
backward-compatibility with a flag-less (pre-toggle) JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping.data.rf_boundary_types import (  # noqa: E402
    ContourParamToggles,
    GestureContourParams,
)
from analysis.receptive_field_mapping.data.rf_contour_params_io import (  # noqa: E402
    _META_KEYS,
    PARAM_KEYS,
    load_contour_params,
    save_contour_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_params(**toggle_overrides) -> GestureContourParams:
    """A valid GestureContourParams with explicit toggles for test control."""
    toggles = ContourParamToggles(
        min_overlap_pct=toggle_overrides.get("min_overlap_pct", True),
        median_filter_size=toggle_overrides.get("median_filter_size", True),
        radial_gauss_sigma=toggle_overrides.get("radial_gauss_sigma", True),
        radial_envelope_smooth_sigma=toggle_overrides.get(
            "radial_envelope_smooth_sigma", True
        ),
    )
    return GestureContourParams(
        min_overlap_pct=25.0,
        median_filter_size=5,
        radial_gauss_sigma=8.0,
        radial_hess_sigma=5.0,
        radial_envelope_smooth_sigma=1.5,
        toggles=toggles,
    )


def _required_json_doc(session_id: str = "S1", gesture: str = "tap") -> dict:
    """A hand-written JSON dict with every required key, WITHOUT param_enabled."""
    return {
        "session_id": session_id,
        "gesture": gesture,
        "min_overlap_pct": 25.0,
        "median_filter_size": 5,
        "radial_gauss_sigma": 8.0,
        "radial_hess_sigma": 5.0,
        "radial_envelope_smooth_sigma": 1.5,
        "created_at": "2026-07-18T00:00:00Z",
        "modified_at": "2026-07-18T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# ContourParamToggles: defaults
# ---------------------------------------------------------------------------


class TestContourParamTogglesDefaults:
    def test_default_state(self) -> None:
        toggles = ContourParamToggles()
        assert toggles.min_overlap_pct is True
        assert toggles.median_filter_size is False
        assert toggles.radial_gauss_sigma is True
        assert toggles.radial_envelope_smooth_sigma is True


# ---------------------------------------------------------------------------
# ContourParamToggles: fail-fast validation
# ---------------------------------------------------------------------------


class TestContourParamTogglesValidation:
    def test_non_bool_min_overlap_pct_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles(min_overlap_pct=1)

    def test_non_bool_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles(median_filter_size="no")

    def test_non_bool_radial_gauss_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles(radial_gauss_sigma=0)

    def test_non_bool_radial_envelope_smooth_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles(radial_envelope_smooth_sigma=None)


# ---------------------------------------------------------------------------
# ContourParamToggles: to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestContourParamTogglesRoundTrip:
    def test_to_dict_from_dict_round_trip(self) -> None:
        toggles = ContourParamToggles(
            min_overlap_pct=False,
            median_filter_size=True,
            radial_gauss_sigma=False,
            radial_envelope_smooth_sigma=True,
        )
        d = toggles.to_dict()
        assert d == {
            "min_overlap_pct": False,
            "median_filter_size": True,
            "radial_gauss_sigma": False,
            "radial_envelope_smooth_sigma": True,
        }
        rebuilt = ContourParamToggles.from_dict(d)
        assert rebuilt == toggles

    def test_from_dict_empty_yields_defaults(self) -> None:
        assert ContourParamToggles.from_dict({}) == ContourParamToggles()

    def test_from_dict_partial_fills_remaining_defaults(self) -> None:
        toggles = ContourParamToggles.from_dict({"median_filter_size": True})
        assert toggles == ContourParamToggles(
            min_overlap_pct=True,
            median_filter_size=True,
            radial_gauss_sigma=True,
            radial_envelope_smooth_sigma=True,
        )

    def test_from_dict_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles.from_dict({"radial_hess_sigma": True})

    def test_from_dict_non_bool_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles.from_dict({"min_overlap_pct": "yes"})

    def test_from_dict_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError):
            ContourParamToggles.from_dict(["min_overlap_pct", True])


# ---------------------------------------------------------------------------
# GestureContourParams.effective_*(): ON and OFF for each toggle
# ---------------------------------------------------------------------------


class TestEffectiveResolvers:
    def test_min_overlap_pct_on_returns_stored_value(self) -> None:
        params = _valid_params(min_overlap_pct=True)
        assert params.effective_min_overlap_pct() == pytest.approx(25.0)

    def test_min_overlap_pct_off_returns_zero(self) -> None:
        params = _valid_params(min_overlap_pct=False)
        assert params.effective_min_overlap_pct() == pytest.approx(0.0)

    def test_median_filter_size_on_returns_stored_value(self) -> None:
        params = _valid_params(median_filter_size=True)
        assert params.effective_median_filter_size() == 5

    def test_median_filter_size_off_returns_none(self) -> None:
        params = _valid_params(median_filter_size=False)
        assert params.effective_median_filter_size() is None

    def test_radial_gauss_sigma_on_returns_stored_value(self) -> None:
        params = _valid_params(radial_gauss_sigma=True)
        assert params.effective_gauss_sigma() == pytest.approx(8.0)

    def test_radial_gauss_sigma_off_returns_zero(self) -> None:
        params = _valid_params(radial_gauss_sigma=False)
        assert params.effective_gauss_sigma() == pytest.approx(0.0)

    def test_radial_envelope_smooth_sigma_on_returns_stored_value(self) -> None:
        params = _valid_params(radial_envelope_smooth_sigma=True)
        assert params.effective_envelope_smooth_sigma() == pytest.approx(1.5)

    def test_radial_envelope_smooth_sigma_off_returns_zero(self) -> None:
        params = _valid_params(radial_envelope_smooth_sigma=False)
        assert params.effective_envelope_smooth_sigma() == pytest.approx(0.0)

    def test_radial_hess_sigma_has_no_toggle(self) -> None:
        # radial_hess_sigma is always applied raw -- no effective_*() resolver,
        # no field on ContourParamToggles.
        params = _valid_params()
        assert not hasattr(params, "effective_hess_sigma")
        assert not hasattr(params.toggles, "radial_hess_sigma")
        assert params.radial_hess_sigma == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# save_contour_params / load_contour_params round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trips_toggles(self, tmp_path: Path) -> None:
        params = _valid_params(
            min_overlap_pct=False,
            median_filter_size=True,
            radial_gauss_sigma=False,
            radial_envelope_smooth_sigma=True,
        )
        path = tmp_path / "S1_tap_contour_params.json"
        save_contour_params(path, params, session_id="S1", gesture="tap")

        loaded = load_contour_params(path)

        assert loaded.toggles == params.toggles
        assert loaded.min_overlap_pct == pytest.approx(params.min_overlap_pct)
        assert loaded.median_filter_size == params.median_filter_size
        assert loaded.radial_gauss_sigma == pytest.approx(params.radial_gauss_sigma)
        assert loaded.radial_hess_sigma == pytest.approx(params.radial_hess_sigma)
        assert loaded.radial_envelope_smooth_sigma == pytest.approx(
            params.radial_envelope_smooth_sigma
        )

    def test_saved_json_carries_param_enabled_key(self, tmp_path: Path) -> None:
        params = _valid_params()
        path = tmp_path / "S1_tap_contour_params.json"
        save_contour_params(path, params, session_id="S1", gesture="tap")

        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        assert doc["param_enabled"] == params.toggles.to_dict()


# ---------------------------------------------------------------------------
# Backward-compatibility: flag-less JSON
# ---------------------------------------------------------------------------


class TestBackwardCompatNoParamEnabled:
    def test_json_without_param_enabled_loads_with_default_toggles(
        self, tmp_path: Path
    ) -> None:
        doc = _required_json_doc()
        assert "param_enabled" not in doc
        # Sanity: exactly the required keys, matching the module's own contract.
        assert set(doc.keys()) == set(PARAM_KEYS) | set(_META_KEYS)

        path = tmp_path / "S1_tap_contour_params.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

        loaded = load_contour_params(path)
        assert loaded.toggles == ContourParamToggles()


# ---------------------------------------------------------------------------
# Malformed param_enabled: fail-fast
# ---------------------------------------------------------------------------


class TestMalformedParamEnabled:
    def test_param_enabled_as_list_raises(self, tmp_path: Path) -> None:
        doc = _required_json_doc()
        doc["param_enabled"] = ["min_overlap_pct", True]
        path = tmp_path / "S1_tap_contour_params.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

        with pytest.raises(ValueError):
            load_contour_params(path)

    def test_param_enabled_non_bool_flag_raises(self, tmp_path: Path) -> None:
        doc = _required_json_doc()
        doc["param_enabled"] = {"min_overlap_pct": "yes"}
        path = tmp_path / "S1_tap_contour_params.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

        with pytest.raises(ValueError):
            load_contour_params(path)

    def test_param_enabled_unknown_sub_key_raises(self, tmp_path: Path) -> None:
        doc = _required_json_doc()
        doc["param_enabled"] = {"radial_hess_sigma": True}
        path = tmp_path / "S1_tap_contour_params.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

        with pytest.raises(ValueError):
            load_contour_params(path)
