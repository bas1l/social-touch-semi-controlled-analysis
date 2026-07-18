"""Per-task option detail panel for AnalysisRunnerGUI."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ruamel.yaml.comments import CommentedSeq

from utils.gui.analysis_runner_gui.collapsible_section import CollapsibleSection
from utils.gui.analysis_runner_gui.cluster_group_dialog import ClusterGroupDialog, ClusterGroupReadOnlyDialog
from utils.gui.analysis_runner_gui.feature_combination_dialog import FeatureCombinationDialog
from utils.gui.analysis_runner_gui.grid_group_dialog import GridGroupDialog, GridGroupReadOnlyDialog
from utils.gui.analysis_runner_gui.radar_group_dialog import RadarGroupDialog
from utils.gui.analysis_runner_gui.yaml_edit_dialog import YamlEditDialog
from utils.pipeline.dag_config_model import DagConfigModel

from analysis.receptive_field_mapping.boundary import registry as boundary_registry

_COMPLEX_FG = QColor("#336699")

# Node-name prefix for the per-method boundary-extraction DAG nodes
# (``spatial_extract_boundary__radial`` etc.). Their method-specific options are
# rendered from the registry's params schema — never hardcoded by method name.
_BOUNDARY_METHOD_NODE_PREFIX = "spatial_extract_boundary__"

# Default size used when the median filter is toggled on from a disabled (null) state.
# Must be a positive odd integer (renderer requirement).
_MEDIAN_FILTER_DEFAULT = 3

# Full universe of items for checklist options keyed by option name.
# Without this, unchecking an item removes it from the YAML and it vanishes on restart.
_CHECKLIST_UNIVERSES: dict[str, list[str]] = {
    "extracted_features": [
        "touch_count",
        "max_iff",
        "mean_iff",
        "median_iff",
        "std_iff",
        "iff_range",
        "n_active_vertices",
        "hypsometric_integral",
        "coefficient_of_variation",
        "iff_skewness",
        "iff_kurtosis",
        "gini_coefficient",
        "shannon_entropy",
        "perimeter_mm",
        "circularity",
        "eccentricity",
        "hotspot_area_mm2",
        "threshold_area_mm2",
        "convex_hull_area_mm2",
    ],
    "tuning_features": [
        "contact_area_mean",
        "pressure_mean",
        "hand_velocity_amplitude_mean",
        "contact_depth_mean",
        "hand_velocity_signed_mean",
    ],
}

# Option keys that should render as a dropdown. Values are (display_label, saved_value) pairs.
# None as saved_value writes YAML null (~ ), meaning "use the default".
_OPTION_ENUMS: dict[str, list[tuple[str, object]]] = {
    "projection_method": [
        ("3D (default)", None),
        ("tangent_plane", "tangent_plane"),
        ("cylindrical_unwrap", "cylindrical_unwrap"),
    ],
    "mesh_method": [
        ("Delaunay (default)", "delaunay"),
        ("Ball Pivoting Algorithm", "bpa"),
    ],
    "heatmap_space": [
        ("Linear (default)", "linear"),
        ("Logarithmic", "log"),
    ],
    "cmap": [
        ("Inferno", "inferno"),
        ("Viridis", "viridis"),
        ("Plasma", "plasma"),
        ("Magma", "magma"),
        ("Cividis", "cividis"),
        ("Batlow", "batlow"),
        ("Nipy Spectral", "nipy_spectral"),
        ("Rainbow", "rainbow"),
        ("Turbo", "turbo"),
        ("Jet", "jet"),
    ],
    "neuron_mode": [
        ("IFF (default)", "iff"),
        ("Spike", "spike"),
    ],
    "iff_metric": [
        ("Mean (default)", "mean"),
        ("Max", "max"),
        ("Both", "both"),
    ],
    "response_metric": [
        ("IFF Mean (default)", "iff_mean"),
        ("IFF Max", "iff_max"),
        ("Spike Count Mean", "spike_count_mean"),
        ("All", "all"),
    ],
    "binning_strategy": [
        ("Sliding Window (default)", "sliding_window"),
        ("Raw Dots", "raw_dots"),
    ],
    "contour_color": [
        ("Red (default)", "red"),
        ("Violet", "violet"),
        ("White", "white"),
        ("Cyan", "cyan"),
        ("Lime", "lime"),
        ("Yellow", "yellow"),
        ("Orange", "orange"),
        ("Magenta", "magenta"),
    ],
}

_OPTION_VISIBILITY: dict[str, dict[str, set]] = {
    "neuron_mode":     {"iff_metric": {"iff"}},
}

# Ordered catalogue of option groups (render order + display label).
_OPTION_GROUPS: list[tuple[str, str]] = [
    ("neuron", "Neuron firing"),
    ("method", "Processing & parameters"),
    ("visual", "Visualization"),
]

# Which group each option key belongs to. One group per (shared, method-agnostic)
# option key. Per-method boundary-extraction params are NOT listed here: they are
# classified from their own ``ParamSpec.group`` via :func:`option_group_of`, so
# adding a boundary method touches no hardcoded map in this module.
_OPTION_GROUP_OF: dict[str, str] = {
    # neuron firing
    "neuron_mode": "neuron",
    "iff_metric": "neuron",
    "response_metric": "neuron",
    # preprocessing & methods (shared across boundary methods, method-agnostic)
    "min_overlap_pct": "method",
    "median_filter_size": "method",
    "use_tuned_params": "method",
    "projection_method": "method",
    # visualization
    "heatmap_space": "visual",
    "cmap": "visual",
    "flip_u": "visual",
    "contour_color": "visual",
    "circular_crop_margin": "visual",
    "show_interactive": "visual",
}

# Tasks that opt in to grouped rendering. Every non-force_processing option of a
# grouped task MUST resolve a group (via :func:`option_group_of`, enforced at
# render). The per-method boundary nodes (``spatial_extract_boundary__*``) are
# grouped generically — see :func:`_is_grouped_task` — rather than being listed
# individually, so a new boundary method needs no edit here.
_GROUPED_TASKS: set[str] = {
    "spatial_map_single_touch",
    "spatial_map_baseline",
    "spatial_compare_boundaries",
    "spatial_compare_proximal_distal",
    "spatial_compare_tap_stroke",
    "spatial_extract_rf_profiles",
}


# NOTE (deferred follow-up — not in Phase 6 scope): the barrier node
# ``spatial_extract_boundaries`` keeps every per-method node in its ``depends_on``.
# If a user toggles a ``spatial_extract_boundary__*`` node OFF, that node never
# runs → is never ``mark_completed`` → the barrier's ``can_run`` (which needs all
# deps completed) never satisfies → every downstream stage stalls. Keeping the
# barrier's ``depends_on`` in sync on toggle is NOT a localized change to this
# module: the enabled toggle lives in ``task_panel.py`` / ``dag_graph_view.py``,
# and ``DagConfigModel`` exposes no depends_on mutation API. Left as a follow-up
# (see the plan's "Known Follow-ups"); doing it here would couple barrier/method
# node identity into generic GUI code.
def _is_boundary_method_task(task_name: str) -> bool:
    """Return True for a per-method boundary node (``spatial_extract_boundary__<name>``)."""
    return task_name.startswith(_BOUNDARY_METHOD_NODE_PREFIX)


def boundary_method_of_task(task_name: str) -> str:
    """Return the registry method name encoded in a boundary-method node name.

    Fail-fast: raises ``ValueError`` if *task_name* is not a boundary-method node,
    or if the encoded method is not registered (via ``registry.get_method``).
    """
    if not _is_boundary_method_task(task_name):
        raise ValueError(
            f"'{task_name}' is not a boundary-method node "
            f"(expected prefix '{_BOUNDARY_METHOD_NODE_PREFIX}')"
        )
    method_name = task_name[len(_BOUNDARY_METHOD_NODE_PREFIX):]
    boundary_registry.get_method(method_name)  # validates / raises on unknown
    return method_name


def boundary_param_specs(task_name: str):
    """Return the ordered ``{key: ParamSpec}`` schema for a boundary-method node.

    Pure (no Qt); the single source of truth for which options of a boundary
    node are method-specific and how each renders. Fail-fast via
    :func:`boundary_method_of_task`.
    """
    method_name = boundary_method_of_task(task_name)
    return {
        spec.key: spec
        for spec in boundary_registry.params_schema_of(method_name)
    }


def option_group_of(task_name: str, key: str) -> str:
    """Resolve the collapsible-group id for option *key* under *task_name*.

    A boundary node's method-specific option is grouped by its ``ParamSpec.group``;
    every other (shared) option is grouped by :data:`_OPTION_GROUP_OF`. Fail-fast:
    raises ``ValueError`` for an option that resolves no group.
    """
    if _is_boundary_method_task(task_name):
        specs = boundary_param_specs(task_name)
        if key in specs:
            return specs[key].group
    if key in _OPTION_GROUP_OF:
        return _OPTION_GROUP_OF[key]
    raise ValueError(
        f"Task '{task_name}' option '{key}' has no group "
        f"(not in _OPTION_GROUP_OF and not a boundary method param)"
    )


def _is_grouped_task(task_name: str) -> bool:
    """Return True if *task_name* renders its options inside collapsible groups."""
    return task_name in _GROUPED_TASKS or _is_boundary_method_task(task_name)


def _is_profile_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have a 'method' key."""
    if not isinstance(val, dict) or not val:
        return False
    return all(isinstance(v, dict) and "method" in v for v in val.values())


def _is_feature_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts with 'enabled' but not 'method' or 'features'."""
    if not isinstance(val, dict) or not val:
        return False
    return all(
        isinstance(v, dict) and "enabled" in v and "method" not in v and "features" not in v
        for v in val.values()
    )


def _is_feature_combinations_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have a 'features' key."""
    if not isinstance(val, dict) or not val:
        return False
    return all(isinstance(v, dict) and "features" in v for v in val.values())


def _is_cluster_groups_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict of sub-dicts that each have both 'features' and 'clustering_methods' keys."""
    if not isinstance(val, dict) or not val:
        return False
    return all(
        isinstance(v, dict) and "features" in v and "clustering_methods" in v
        for v in val.values()
    )


def _is_radar_groups_dict(key: str, val: Any) -> bool:
    """Return True if *key* is ``'radar_groups'`` and *val* is a dict.

    Key-based check to avoid ambiguity with ``_is_cluster_groups_dict`` and
    ``_is_feature_combinations_dict``, which both use structural inspection.
    """
    return key == "radar_groups" and isinstance(val, dict)


def _is_grid_groups_dict(val: Any) -> bool:
    """Return True if *val* is a non-empty dict-of-dicts whose every entry has a 'features' dict
    whose values are dicts containing numeric 'min', 'max', 'step', 'span' keys."""
    if not isinstance(val, dict) or not val:
        return False
    for entry in val.values():
        if not isinstance(entry, dict):
            return False
        features = entry.get("features")
        if not isinstance(features, dict) or not features:
            return False
        for bounds in features.values():
            if not isinstance(bounds, dict):
                return False
            if not all(k in bounds for k in ("min", "max", "step", "span")):
                return False
            if not all(isinstance(bounds[k], (int, float)) for k in ("min", "max", "step", "span")):
                return False
    return True


def _cluster_group_summary(spec: dict) -> str:
    features: dict = spec.get("features") or {}
    parts: list[str] = []
    for dtype, aggs in features.items():
        if dtype == "touch_category":
            parts.append("category")
        elif aggs:
            parts.append(f"{dtype}[{','.join(aggs)}]")
        else:
            parts.append(dtype)
    return " · ".join(parts)


def _radar_group_summary(spec: dict) -> str:
    """Return a short human-readable summary of a radar group spec.

    Examples:
        ``"3 data types, mean_during_iff"`` — single aggregation across all types
        ``"5 data types, mean + max"``       — multiple distinct aggregations
    """
    features: dict = spec.get("features") or {}
    n_types = len(features)
    if n_types == 0:
        return "(no features)"

    all_aggs: list[str] = []
    for aggs in features.values():
        for a in (aggs or []):
            if a not in all_aggs:
                all_aggs.append(a)

    type_str = f"{n_types} data type{'s' if n_types != 1 else ''}"
    if not all_aggs:
        return type_str
    if len(all_aggs) == 1:
        return f"{type_str}, {all_aggs[0]}"
    return f"{type_str}, {' + '.join(all_aggs)}"


def _grid_group_summary(spec: dict) -> str:
    n = len(spec.get("features", {}))
    mode = spec.get("neuron_mode", "?")
    return f"{n} feature{'s' if n != 1 else ''}, neuron_mode={mode}"


def _option_header(key: str) -> str:
    return key.replace("_", " ").title()


def _preview_text(val: Any) -> str:
    if isinstance(val, list):
        items = [str(v) for v in val[:3]]
        preview = "[" + ", ".join(items) + (", ..." if len(val) > 3 else "") + "]"
        return f"{preview} ({len(val)} items)"
    if isinstance(val, dict):
        keys = list(val.keys())[:3]
        preview = "{" + ", ".join(str(k) for k in keys) + (", ..." if len(val) > 3 else "") + "}"
        return f"{preview} ({len(val)} keys)"
    return str(val)


class TaskDetailPanel(QWidget):
    """Panel with a pinned task-name header and a scrollable options area.

    Call :meth:`show_task` to populate. Emits :attr:`task_changed` whenever
    any option is modified.
    """

    task_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model: DagConfigModel | None = None
        self._task_name: str | None = None
        self._sections: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Pinned header — always visible above the scroll area
        self._header = QLabel()
        font = QFont()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self._header.setFont(font)
        self._header.setStyleSheet(
            "padding: 4px 8px;"
            "background: #e8e8e8;"
            "border-bottom: 1px solid #c0c0c0;"
        )
        outer.addWidget(self._header)

        # Scroll area for options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        scroll.setWidget(self._content)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_task(self, model: DagConfigModel, task_name: str) -> None:
        """Clear the panel and rebuild for *task_name*."""
        self._model = model
        self._task_name = task_name
        self._header.setText(task_name)

        # Clear option sections — keep only the trailing stretch
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sections.clear()

        # force_processing is rendered in the task list row, not here
        options = {
            k: v for k, v in model.get_task_options(task_name).items()
            if k != "force_processing"
        }

        if not options:
            lbl = QLabel("No options")
            lbl.setStyleSheet("color: #888;")
            self._layout.insertWidget(0, lbl)
            return

        # Build a section widget per option, recording it in self._sections so
        # conditional-visibility toggling can find it regardless of nesting.
        sections: list[tuple[str, QWidget]] = []
        for key, val in options.items():
            widget = self._build_option_section(key, val)
            self._sections[key] = widget
            sections.append((key, widget))

        if _is_grouped_task(task_name):
            self._insert_grouped(task_name, options, sections)
        else:
            for i, (_key, widget) in enumerate(sections):
                self._layout.insertWidget(i, widget)

        for controller, dependents in _OPTION_VISIBILITY.items():
            if controller not in options:
                continue
            current_val = options[controller]
            for dep_key, visible_when in dependents.items():
                if dep_key in self._sections:
                    self._sections[dep_key].setVisible(current_val in visible_when)

    def _build_option_section(self, key: str, val: Any) -> QWidget:
        """Dispatch *key*/*val* to the matching per-option section builder."""
        # Boundary-method nodes render their method-specific options from the
        # registry's ParamSpec (widget from declared type/choices, not the runtime
        # value type). Shared options (median_filter_size, cmap, ...) fall through
        # to the generic dispatch below.
        if self._task_name is not None and _is_boundary_method_task(self._task_name):
            specs = boundary_param_specs(self._task_name)
            if key in specs:
                return self._make_param_spec_section(key, val, specs[key])
        if key == "median_filter_size":
            return self._make_median_filter_section(key, val)
        if key in _OPTION_ENUMS:
            return self._make_enum_section(key, val)
        if key == "camera_angle_mode" and isinstance(val, dict) and "auto" in val:
            return self._make_camera_angle_mode_section(key, val)
        if key == "cluster_groups" and isinstance(val, (list, CommentedSeq)):
            return self._make_downstream_cluster_groups_section(key, val)
        if key in _CHECKLIST_UNIVERSES and isinstance(val, (list, CommentedSeq)):
            return self._make_checklist_section(key, val)
        if key == "cluster_groups" and _is_cluster_groups_dict(val):
            return self._make_cluster_groups_section(key, val)
        if _is_radar_groups_dict(key, val):
            return self._make_radar_groups_section(key, val)
        if _is_grid_groups_dict(val):
            return self._make_grid_groups_section(key, val)
        if _is_profile_dict(val):
            return self._make_profile_section(key, val)
        if _is_feature_combinations_dict(val):
            return self._make_combination_section(key, val)
        if _is_feature_dict(val):
            cols = 1 if len(val) <= 6 else 3
            return self._make_feature_section(key, val, cols=cols)
        return self._make_scalar_section(key, val)

    def _insert_grouped(
        self,
        task_name: str,
        options: dict[str, Any],
        sections: list[tuple[str, QWidget]],
    ) -> None:
        """Insert *sections* into collapsible group containers by option group.

        Every option must resolve a group via :func:`option_group_of`; an
        unclassified option raises (fail-fast). Groups are rendered in catalogue
        order (:data:`_OPTION_GROUPS`, then any boundary ``ParamSpec.group`` in
        schema order); empty groups are omitted; option order within a group
        follows YAML order.
        """
        group_of = {key: option_group_of(task_name, key) for key in options}
        catalogue = self._group_catalogue(task_name)

        widget_of = dict(sections)
        insert_at = 0
        for group_key, group_label in catalogue:
            members = [k for k in options if group_of[k] == group_key]
            if not members:
                continue
            container = CollapsibleSection(group_label, expanded=True)
            for key in members:
                container.add_widget(widget_of[key])
            self._layout.insertWidget(insert_at, container)
            insert_at += 1

    @staticmethod
    def _group_catalogue(task_name: str) -> list[tuple[str, str]]:
        """Ordered ``(group_id, label)`` catalogue for *task_name*.

        The shared :data:`_OPTION_GROUPS` come first; a boundary-method node then
        appends its own ``ParamSpec.group`` ids (in schema order, de-duplicated),
        labelled from the group id. Purely registry-derived — no per-method rows.
        """
        catalogue = list(_OPTION_GROUPS)
        if _is_boundary_method_task(task_name):
            seen = {gid for gid, _ in catalogue}
            for spec in boundary_param_specs(task_name).values():
                if spec.group not in seen:
                    catalogue.append((spec.group, _option_header(spec.group)))
                    seen.add(spec.group)
        return catalogue

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _make_enum_section(self, key: str, val: Any) -> QWidget:
        """QComboBox for options with a fixed set of allowed values."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        entries = _OPTION_ENUMS[key]
        combo = QComboBox()
        current_idx = 0
        for i, (label, saved) in enumerate(entries):
            combo.addItem(label)
            if val == saved:
                current_idx = i
        combo.setCurrentIndex(current_idx)
        combo.currentIndexChanged.connect(self._make_enum_handler(key, entries, combo))

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(combo)
        row_layout.addStretch()
        layout.addWidget(row)
        return box

    def _make_param_spec_section(self, key: str, val: Any, spec) -> QWidget:
        """Render a boundary-method param from its :class:`ParamSpec`.

        Widget type is decided by the declared schema, never by the runtime value:
        a combobox when ``spec.choices`` is set, a checkbox for ``bool``, a numeric
        line edit for ``int``/``float`` (empty renders as YAML ``null`` only when
        the param is nullable, i.e. ``spec.default is None``), else a string edit.
        """
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        if spec.choices is not None:
            entries = [(str(choice), choice) for choice in spec.choices]
            combo = QComboBox()
            current_idx = 0
            for i, (label, saved) in enumerate(entries):
                combo.addItem(label)
                if val == saved:
                    current_idx = i
            combo.setCurrentIndex(current_idx)
            combo.currentIndexChanged.connect(self._make_enum_handler(key, entries, combo))
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(combo)
            row_layout.addStretch()
            layout.addWidget(row)
        elif spec.type is bool:
            cb = QCheckBox("Enabled")
            cb.setChecked(bool(val))
            cb.stateChanged.connect(self._make_bool_handler(key, cb))
            layout.addWidget(cb)
        elif spec.type in (int, float):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit("" if val is None else str(val))
            edit.setFixedWidth(200)
            if spec.default is None:
                edit.setPlaceholderText("null (unset)")
            edit.editingFinished.connect(
                self._make_param_numeric_handler(key, spec, edit)
            )
            row_layout.addWidget(edit)
            row_layout.addStretch()
            layout.addWidget(row_widget)
        else:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit("" if val is None else str(val))
            edit.setFixedWidth(200)
            edit.editingFinished.connect(
                self._make_scalar_edit_handler(key, "" if val is None else val, edit)
            )
            row_layout.addWidget(edit)
            row_layout.addStretch()
            layout.addWidget(row_widget)
        return box

    def _make_median_filter_section(self, key: str, val: Any) -> QWidget:
        """Checkbox-gated numeric field for the median filter size.

        Unchecking the box writes ``null`` (filter disabled); the size field is
        greyed out. Checking it restores a positive-odd-integer size. ``val`` is
        treated as enabled only when it is a positive integer.
        """
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        enabled = isinstance(val, int) and not isinstance(val, bool) and val >= 1

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        cb = QCheckBox("Enabled")
        cb.setChecked(enabled)

        edit = QLineEdit(str(val) if enabled else "")
        edit.setFixedWidth(120)
        edit.setEnabled(enabled)
        edit.setPlaceholderText("odd ≥ 1")
        edit.setToolTip("Median filter window size — must be a positive odd integer.")

        cb.stateChanged.connect(self._make_median_filter_toggle_handler(key, cb, edit))
        edit.editingFinished.connect(self._make_median_filter_edit_handler(key, cb, edit))

        row_layout.addWidget(cb)
        row_layout.addWidget(edit)
        row_layout.addStretch()
        layout.addWidget(row_widget)
        return box

    def _make_scalar_section(self, key: str, val: Any) -> QWidget:
        """Bool checkbox, string/number line edit, or complex clickable label."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)

        if isinstance(val, bool):
            cb = QCheckBox("Enabled")
            cb.setChecked(val)
            cb.stateChanged.connect(self._make_bool_handler(key, cb))
            layout.addWidget(cb)
        elif isinstance(val, str) and val in {"auto", "manual"}:
            cb = QCheckBox("Auto")
            cb.setChecked(val == "auto")
            cb.stateChanged.connect(self._make_mode_toggle_handler(key, cb))
            layout.addWidget(cb)
        elif isinstance(val, (int, float, str)):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(str(val))
            edit.setFixedWidth(200)
            edit.editingFinished.connect(self._make_scalar_edit_handler(key, val, edit))
            row_layout.addWidget(edit)
            row_layout.addStretch()
            layout.addWidget(row_widget)
        else:
            lbl = QLabel(_preview_text(val))
            lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setToolTip("Click to edit in YAML editor")
            lbl.mousePressEvent = self._make_complex_click_handler(key, lbl)
            layout.addWidget(lbl)

        return box

    def _make_camera_angle_mode_section(self, key: str, val: dict) -> QWidget:
        """Group box titled 'Camera Angle Mode Auto' with an Enabled checkbox."""
        box = QGroupBox("Camera Angle Mode Auto")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        auto_cfg = val.get("auto") or {}
        enabled = auto_cfg.get("enabled", False)
        cb = QCheckBox("Enabled")
        cb.setChecked(enabled)
        cb.stateChanged.connect(self._make_camera_angle_mode_handler(key, cb))
        layout.addWidget(cb)
        return box

    def _make_feature_section(self, key: str, val: dict, cols: int = 3) -> QWidget:
        """Checkbox grid with optional '...' param buttons. cols=1 gives expanded per-item layout."""
        box = QGroupBox(_option_header(key))
        flow = QVBoxLayout(box)
        flow.setContentsMargins(6, 4, 6, 4)
        flow.setSpacing(2)

        if cols == 1:
            for i, (feature_name, feature_cfg) in enumerate(val.items()):
                if i > 0:
                    flow.addSpacing(6)

                lbl_row = QWidget()
                lbl_row_layout = QHBoxLayout(lbl_row)
                lbl_row_layout.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(feature_name.replace("_", " ") + ":")
                font = QFont()
                font.setBold(True)
                lbl.setFont(font)
                lbl_row_layout.addWidget(lbl)
                lbl_row_layout.addStretch()
                flow.addWidget(lbl_row)

                for param_key, param_val in feature_cfg.items():
                    if not isinstance(param_val, bool):
                        continue
                    param_row = QWidget()
                    param_layout = QHBoxLayout(param_row)
                    param_layout.setContentsMargins(14, 0, 0, 0)
                    param_layout.setSpacing(4)
                    cb = QCheckBox(param_key.replace("_", " "))
                    if param_key == "enabled":
                        cb.setChecked(
                            self._model.get_profile_enabled(self._task_name, key, feature_name)
                        )
                        cb.stateChanged.connect(self._make_profile_handler(key, feature_name, cb))
                    else:
                        cb.setChecked(bool(param_val))
                        cb.stateChanged.connect(
                            self._make_transform_bool_param_handler(key, feature_name, param_key, cb)
                        )
                    param_layout.addWidget(cb)
                    param_layout.addStretch()
                    flow.addWidget(param_row)

                has_other_params = any(
                    k != "enabled" and not isinstance(v, bool)
                    for k, v in feature_cfg.items()
                )
                if has_other_params:
                    btn_row = QWidget()
                    btn_layout = QHBoxLayout(btn_row)
                    btn_layout.setContentsMargins(14, 0, 0, 0)
                    btn = QPushButton("Parameters…")
                    btn.setFixedWidth(90)
                    btn.clicked.connect(self._make_transform_params_handler(key, feature_name))
                    btn_layout.addWidget(btn)
                    btn_layout.addStretch()
                    flow.addWidget(btn_row)

            return box

        flow.setSpacing(4)
        row_widget: QWidget | None = None
        row_layout: QHBoxLayout | None = None

        for i, (feature_name, feature_cfg) in enumerate(val.items()):
            if i % cols == 0:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                flow.addWidget(row_widget)

            is_enabled = self._model.get_profile_enabled(self._task_name, key, feature_name)
            has_params = any(k != "enabled" for k in feature_cfg)

            cb = QCheckBox(feature_name.replace("_", " "))
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, feature_name, cb))

            if has_params:
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)
                cell_layout.addWidget(cb)
                btn = QPushButton("...")
                btn.setFixedSize(24, 20)
                btn.setToolTip("Edit feature parameters")
                btn.clicked.connect(self._make_feature_params_handler(key, feature_name))
                cell_layout.addWidget(btn)
                row_layout.addWidget(cell)
            else:
                row_layout.addWidget(cb)

        remainder = len(val) % cols
        if remainder != 0 and row_layout is not None:
            for _ in range(cols - remainder):
                row_layout.addStretch()

        return box

    def _make_profile_section(self, key: str, val: dict) -> QWidget:
        """Checkbox per profile with '...' button for method params."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for profile_name in val:
            is_enabled = self._model.get_profile_enabled(self._task_name, key, profile_name)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(profile_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, profile_name, cb))
            row_layout.addWidget(cb)

            btn = QPushButton("...")
            btn.setFixedSize(24, 20)
            btn.setToolTip("Edit profile parameters")
            btn.clicked.connect(self._make_profile_params_handler(key, profile_name))
            row_layout.addWidget(btn)

            row_layout.addStretch()
            layout.addWidget(row)

        return box

    def _make_combination_section(self, key: str, val: dict) -> QWidget:
        """Checkbox + feature preview per combination, plus '+' add button."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for combo_name in val:
            is_enabled = self._model.get_profile_enabled(self._task_name, key, combo_name)
            features = self._model.get_combination_features(self._task_name, key, combo_name)

            row = QWidget()
            row.setContextMenuPolicy(Qt.CustomContextMenu)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(combo_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(self._make_profile_handler(key, combo_name, cb))
            row_layout.addWidget(cb)

            preview = QLabel(f"[{', '.join(features)}]")
            preview.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            preview.setCursor(Qt.PointingHandCursor)
            preview.setToolTip("Click to edit features")
            preview.mousePressEvent = self._make_combination_edit_handler(key, combo_name, preview)
            row_layout.addWidget(preview)

            row_layout.addStretch()

            for _w in (row, cb, preview):
                _w.setContextMenuPolicy(Qt.CustomContextMenu)
                _w.customContextMenuRequested.connect(
                    self._make_combination_context_handler(key, combo_name)
                )

            layout.addWidget(row)

        # "+" add button
        add_row = QWidget()
        add_layout = QHBoxLayout(add_row)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(24, 20)
        add_btn.setToolTip("Add new feature combination")
        add_btn.clicked.connect(self._make_combination_add_handler(key))
        add_layout.addWidget(add_btn)
        add_layout.addStretch()
        layout.addWidget(add_row)

        return box

    def _make_cluster_groups_section(self, key: str, val: dict) -> QWidget:
        """Per-group rows with enabled checkbox, summary label, Edit and Delete buttons."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for group_name in val:
            spec = val[group_name]
            is_enabled = bool(spec.get("enabled", True))
            summary = _cluster_group_summary(spec)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(
                self._make_cluster_group_enabled_handler(key, group_name, cb)
            )
            row_layout.addWidget(cb)

            summary_lbl = QLabel(summary)
            summary_lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            summary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            summary_lbl.setMinimumWidth(0)
            summary_lbl.setToolTip(summary)
            row_layout.addWidget(summary_lbl, stretch=1)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedWidth(48)
            edit_btn.clicked.connect(
                self._make_cluster_group_edit_handler(key, group_name, cb, summary_lbl)
            )
            row_layout.addWidget(edit_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedWidth(56)
            del_btn.clicked.connect(self._make_cluster_group_delete_handler(key, group_name))
            row_layout.addWidget(del_btn)

            layout.addWidget(row)

        new_row = QWidget()
        new_layout = QHBoxLayout(new_row)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_btn = QPushButton("New Group…")
        new_btn.setFixedWidth(100)
        new_btn.clicked.connect(self._make_cluster_group_new_handler(key))
        new_layout.addWidget(new_btn)
        new_layout.addStretch()
        layout.addWidget(new_row)

        return box

    def _make_radar_groups_section(self, key: str, val: dict) -> QWidget:
        """Per-group rows with enabled checkbox, summary label, Edit and Delete buttons."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for group_name in val:
            spec = val[group_name]
            is_enabled = bool(spec.get("enabled", True))
            summary = _radar_group_summary(spec)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(
                self._make_radar_group_enabled_handler(key, group_name, cb)
            )
            row_layout.addWidget(cb)

            summary_lbl = QLabel(summary)
            summary_lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            summary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            summary_lbl.setMinimumWidth(0)
            summary_lbl.setToolTip(summary)
            row_layout.addWidget(summary_lbl, stretch=1)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedWidth(48)
            edit_btn.clicked.connect(
                self._make_radar_group_edit_handler(key, group_name, cb, summary_lbl)
            )
            row_layout.addWidget(edit_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedWidth(56)
            del_btn.clicked.connect(self._make_radar_group_delete_handler(key, group_name))
            row_layout.addWidget(del_btn)

            layout.addWidget(row)

        new_row = QWidget()
        new_layout = QHBoxLayout(new_row)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_btn = QPushButton("New Group…")
        new_btn.setFixedWidth(100)
        new_btn.clicked.connect(self._make_radar_group_new_handler(key))
        new_layout.addWidget(new_btn)
        new_layout.addStretch()
        layout.addWidget(new_row)

        return box

    def _make_grid_groups_section(self, key: str, val: dict) -> QWidget:
        """Per-group rows with enabled checkbox, summary label, Edit and Delete buttons."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for group_name in val:
            spec = val[group_name]
            is_enabled = bool(spec.get("enabled", True))
            summary = _grid_group_summary(spec)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(
                self._make_grid_group_enabled_handler(key, group_name, cb)
            )
            row_layout.addWidget(cb)

            summary_lbl = QLabel(summary)
            summary_lbl.setStyleSheet(f"color: {_COMPLEX_FG.name()};")
            summary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            summary_lbl.setMinimumWidth(0)
            summary_lbl.setToolTip(summary)
            row_layout.addWidget(summary_lbl, stretch=1)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedWidth(48)
            edit_btn.clicked.connect(
                self._make_grid_group_edit_handler(key, group_name, cb, summary_lbl)
            )
            row_layout.addWidget(edit_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedWidth(56)
            del_btn.clicked.connect(self._make_grid_group_delete_handler(key, group_name))
            row_layout.addWidget(del_btn)

            layout.addWidget(row)

        new_row = QWidget()
        new_layout = QHBoxLayout(new_row)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_btn = QPushButton("New Group…")
        new_btn.setFixedWidth(100)
        new_btn.clicked.connect(self._make_grid_group_new_handler(key))
        new_layout.addWidget(new_btn)
        new_layout.addStretch()
        layout.addWidget(new_row)

        return box

    def _make_downstream_cluster_groups_section(self, key: str, val: list) -> QWidget:
        """Checkbox per group defined in stimulus_cluster_touches; checked if this task references it."""
        box = QGroupBox("Cluster Groups")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        sub_lbl = QLabel("From stimulus_cluster_touches:")
        sub_font = QFont()
        sub_font.setItalic(True)
        sub_lbl.setFont(sub_font)
        sub_lbl.setStyleSheet("color: #555;")
        layout.addWidget(sub_lbl)

        try:
            all_groups = self._model.get_profile_names("stimulus_cluster_touches", "cluster_groups")
        except (KeyError, AttributeError):
            all_groups = []

        if not all_groups:
            info = QLabel("No cluster groups defined in stimulus_cluster_touches.")
            info.setStyleSheet("color: #888;")
            layout.addWidget(info)
            return box

        selected: list[str] = list(val)

        for group_name in all_groups:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            cb = QCheckBox(group_name.replace("_", " ").title())
            cb.setChecked(group_name in selected)
            cb.stateChanged.connect(
                self._make_downstream_group_handler(key, group_name, cb, all_groups, selected)
            )
            row_layout.addWidget(cb)

            row_layout.addStretch()

            details_btn = QPushButton("Details…")
            details_btn.setFixedWidth(72)
            details_btn.clicked.connect(
                self._make_downstream_group_details_handler(group_name)
            )
            row_layout.addWidget(details_btn)

            layout.addWidget(row)

        return box

    def _make_checklist_section(self, key: str, val: list) -> QWidget:
        """Checkbox list for a plain list-of-strings option (e.g. extracted_features)."""
        box = QGroupBox(_option_header(key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        all_items: list[str] = _CHECKLIST_UNIVERSES.get(key, list(val))
        selected: list[str] = list(val)

        for item_name in all_items:
            cb = QCheckBox(item_name.replace("_", " "))
            cb.setChecked(item_name in selected)
            cb.stateChanged.connect(
                self._make_checklist_handler(key, item_name, cb, all_items, selected)
            )
            layout.addWidget(cb)

        return box

    # ------------------------------------------------------------------
    # Handler factories
    # ------------------------------------------------------------------

    def _make_checklist_handler(
        self,
        opt_key: str,
        item_name: str,
        cb: QCheckBox,
        all_items: list,
        selected: list,
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            if cb.isChecked():
                if item_name not in selected:
                    selected.append(item_name)
            else:
                if item_name in selected:
                    selected.remove(item_name)
            ordered = [g for g in all_items if g in selected]
            self._model.set_task_option(self._task_name, opt_key, ordered)
            self.task_changed.emit()
        return _handler

    def _make_enum_handler(self, key: str, entries: list, combo: QComboBox):
        def _handler(index: int) -> None:
            if self._model is None or self._task_name is None:
                return
            _, saved_val = entries[index]
            self._model.set_task_option(self._task_name, key, saved_val)
            self.task_changed.emit()
            if key in _OPTION_VISIBILITY:
                for dep_key, visible_when in _OPTION_VISIBILITY[key].items():
                    if dep_key in self._sections:
                        self._sections[dep_key].setVisible(saved_val in visible_when)
        return _handler

    def _make_mode_toggle_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, "auto" if cb.isChecked() else "manual")
            self.task_changed.emit()
        return _handler

    def _make_camera_angle_mode_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, {"auto": {"enabled": cb.isChecked()}})
            self.task_changed.emit()
        return _handler

    def _make_bool_handler(self, key: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_task_option(self._task_name, key, cb.isChecked())
            self.task_changed.emit()
        return _handler

    @staticmethod
    def _native_type(val: Any) -> type:
        """Return the Python builtin type for a YAML scalar value.

        ruamel.yaml round-trip mode may store scalars as ScalarFloat,
        ScalarInt, etc.  Normalise to builtins so that type(text) works
        reliably for user-entered values.
        """
        if isinstance(val, bool):
            return bool
        if isinstance(val, int):
            return int
        if isinstance(val, float):
            return float
        return str

    def _make_scalar_edit_handler(self, key: str, original_val: Any, edit: QLineEdit):
        cast = self._native_type(original_val)

        def _handler() -> None:
            if self._model is None or self._task_name is None:
                return
            try:
                new_val = cast(edit.text())
            except (ValueError, TypeError):
                edit.setText(str(original_val))
                return
            self._model.set_task_option(self._task_name, key, new_val)
            self.task_changed.emit()

        return _handler

    def _make_param_numeric_handler(self, key: str, spec, edit: QLineEdit):
        """Handler for a numeric boundary-param edit driven by *spec* (not value type).

        Casts via ``spec.type``. An empty field writes YAML ``null`` only when the
        param is nullable (``spec.default is None``); otherwise, and on a cast
        failure, the field reverts to the stored value (fail-fast — no silent
        coercion of an invalid entry).
        """
        nullable = spec.default is None

        def _handler() -> None:
            if self._model is None or self._task_name is None:
                return
            text = edit.text().strip()
            current = self._model.get_task_option(self._task_name, key)
            if text == "":
                if nullable:
                    self._model.set_task_option(self._task_name, key, None)
                    self.task_changed.emit()
                else:
                    edit.setText("" if current is None else str(current))
                return
            try:
                new_val = spec.type(text)
            except (ValueError, TypeError):
                edit.setText("" if current is None else str(current))
                return
            self._model.set_task_option(self._task_name, key, new_val)
            self.task_changed.emit()

        return _handler

    @staticmethod
    def _parse_median_filter_size(text: Any) -> int | None:
        """Return a positive odd int parsed from *text*, or None if invalid."""
        try:
            size = int(text)
        except (ValueError, TypeError):
            return None
        if size < 1 or size % 2 == 0:
            return None
        return size

    def _make_median_filter_toggle_handler(self, key: str, cb: QCheckBox, edit: QLineEdit):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            if cb.isChecked():
                size = self._parse_median_filter_size(edit.text())
                if size is None:
                    size = _MEDIAN_FILTER_DEFAULT
                edit.setText(str(size))
                edit.setEnabled(True)
                self._model.set_task_option(self._task_name, key, size)
            else:
                edit.setEnabled(False)
                self._model.set_task_option(self._task_name, key, None)
            self.task_changed.emit()
        return _handler

    def _make_median_filter_edit_handler(self, key: str, cb: QCheckBox, edit: QLineEdit):
        def _handler() -> None:
            if self._model is None or self._task_name is None:
                return
            if not cb.isChecked():
                return
            size = self._parse_median_filter_size(edit.text())
            if size is None:
                current = self._model.get_task_option(self._task_name, key)
                revert = current if self._parse_median_filter_size(current) is not None else _MEDIAN_FILTER_DEFAULT
                edit.setText(str(revert))
                return
            self._model.set_task_option(self._task_name, key, size)
            self.task_changed.emit()
        return _handler

    def _make_complex_click_handler(self, key: str, lbl: QLabel):
        def _handler(_event) -> None:
            if self._model is None or self._task_name is None:
                return
            current_value = self._model.get_task_option(self._task_name, key)
            dlg = YamlEditDialog(self._task_name, key, current_value, self)
            if dlg.exec_() == QDialog.Accepted:
                new_value = dlg.get_value()
                self._model.set_task_option(self._task_name, key, new_value)
                lbl.setText(_preview_text(new_value))
                self.task_changed.emit()
        return _handler

    def _make_profile_handler(self, opt_key: str, profile_name: str, cb: QCheckBox):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(self._task_name, opt_key, profile_name, cb.isChecked())
            self.task_changed.emit()
        return _handler

    def _make_transform_bool_param_handler(
        self, opt_key: str, feature_name: str, param_key: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
            feature[param_key] = cb.isChecked()
            self._model._dirty = True
            self.task_changed.emit()
        return _handler

    def _make_transform_params_handler(self, opt_key: str, feature_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            feature_cfg = dict(opts.get(feature_name) or {})
            params = {
                k: v for k, v in feature_cfg.items()
                if k != "enabled" and not isinstance(v, bool)
            }
            dlg = YamlEditDialog(feature_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
                for k, v in new_params.items():
                    feature[k] = v
                for k in list(params.keys()):
                    if k not in new_params:
                        feature.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_feature_params_handler(self, opt_key: str, feature_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            feature_cfg = dict(opts.get(feature_name) or {})
            params = {k: v for k, v in feature_cfg.items() if k != "enabled"}
            dlg = YamlEditDialog(feature_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                feature = self._model.get_task_option(self._task_name, opt_key)[feature_name]
                for k, v in new_params.items():
                    if k != "enabled":
                        feature[k] = v
                for k in list(feature_cfg.keys()):
                    if k != "enabled" and k not in new_params:
                        feature.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_profile_params_handler(self, opt_key: str, profile_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            profile_cfg = dict(opts.get(profile_name) or {})
            params = {k: v for k, v in profile_cfg.items() if k != "enabled"}
            dlg = YamlEditDialog(profile_name, opt_key, params, self)
            if dlg.exec_() == QDialog.Accepted:
                new_params = dlg.get_value() or {}
                profile = self._model.get_task_option(self._task_name, opt_key)[profile_name]
                for k, v in new_params.items():
                    if k != "enabled":
                        profile[k] = v
                for k in list(profile_cfg.keys()):
                    if k != "enabled" and k not in new_params:
                        profile.pop(k, None)
                self._model._dirty = True
                self.task_changed.emit()
        return _handler

    def _make_combination_edit_handler(self, opt_key: str, combo_name: str, preview_label: QLabel):
        def _handler(_event) -> None:
            if self._model is None or self._task_name is None:
                return
            current_features = self._model.get_combination_features(
                self._task_name, opt_key, combo_name
            )
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            existing = list(opts.keys())
            dlg = FeatureCombinationDialog(
                self._task_name,
                existing_names=[n for n in existing if n != combo_name],
                combo_name=combo_name,
                selected_features=current_features,
                parent=self,
            )
            if dlg.exec_() == QDialog.Accepted:
                new_combo_name = dlg.get_combo_name()
                original_combo_name = dlg.get_original_combo_name()
                new_features = dlg.get_selected_features()

                # Handle rename: remove old, add new
                if new_combo_name != original_combo_name:
                    self._model.remove_combination(self._task_name, opt_key, original_combo_name)
                    seq = CommentedSeq(new_features)
                    seq.fa.set_flow_style()
                    self._model.add_combination(
                        self._task_name, opt_key, new_combo_name, {"enabled": True, "features": seq}
                    )
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_combination_features(
                        self._task_name, opt_key, combo_name, new_features
                    )
                    preview_label.setText(f"[{', '.join(new_features)}]")
                    self.task_changed.emit()
        return _handler

    def _make_combination_add_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            opts = self._model.get_task_option(self._task_name, opt_key) or {}
            existing = list(opts.keys())
            dlg = FeatureCombinationDialog(
                self._task_name,
                existing_names=existing,
                parent=self,
            )
            if dlg.exec_() == QDialog.Accepted:
                combo_name = dlg.get_combo_name()
                features = dlg.get_selected_features()
                seq = CommentedSeq(features)
                seq.fa.set_flow_style()
                self._model.add_combination(
                    self._task_name, opt_key, combo_name, {"enabled": True, "features": seq}
                )
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_combination_context_handler(self, opt_key: str, combo_name: str):
        def _handler(_pos) -> None:
            if self._model is None or self._task_name is None:
                return
            menu = QMenu(self)
            delete_action = menu.addAction(f"Delete '{combo_name}'")
            action = menu.exec_(QCursor.pos())
            if action == delete_action:
                reply = QMessageBox.question(
                    self,
                    "Delete Combination",
                    f"Delete combination '{combo_name}' from '{self._task_name}'?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._model.remove_combination(self._task_name, opt_key, combo_name)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
        return _handler

    def _make_cluster_group_enabled_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(
                self._task_name, opt_key, group_name, cb.isChecked()
            )
            self.task_changed.emit()
        return _handler

    def _make_cluster_group_edit_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox, summary_lbl: QLabel
    ):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            spec = self._model.get_cluster_group_spec(self._task_name, group_name)
            dlg = ClusterGroupDialog(self, name=group_name, spec=spec)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                if new_name != group_name:
                    self._model.remove_combination(self._task_name, opt_key, group_name)
                    self._model.set_cluster_group_spec(self._task_name, new_name, new_spec)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_cluster_group_spec(self._task_name, group_name, new_spec)
                    cb.setChecked(new_spec.get("enabled", True))
                    summary_lbl.setText(_cluster_group_summary(new_spec))
                    self.task_changed.emit()
        return _handler

    def _make_cluster_group_delete_handler(self, opt_key: str, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            reply = QMessageBox.question(
                self,
                "Delete Cluster Group",
                f"Delete cluster group '{group_name}' from '{self._task_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._model.remove_combination(self._task_name, opt_key, group_name)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_cluster_group_new_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            dlg = ClusterGroupDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                self._model.set_cluster_group_spec(self._task_name, new_name, new_spec)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_radar_group_enabled_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(
                self._task_name, opt_key, group_name, cb.isChecked()
            )
            self.task_changed.emit()
        return _handler

    def _make_radar_group_edit_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox, summary_lbl: QLabel
    ):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            spec = self._model.get_radar_group_spec(self._task_name, group_name)
            dlg = RadarGroupDialog(self, name=group_name, spec=spec)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                if new_name != group_name:
                    self._model.remove_combination(self._task_name, opt_key, group_name)
                    self._model.set_radar_group_spec(self._task_name, new_name, new_spec)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_radar_group_spec(self._task_name, group_name, new_spec)
                    cb.setChecked(new_spec.get("enabled", True))
                    summary_lbl.setText(_radar_group_summary(new_spec))
                    self.task_changed.emit()
        return _handler

    def _make_radar_group_delete_handler(self, opt_key: str, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            reply = QMessageBox.question(
                self,
                "Delete Radar Group",
                f"Delete radar group '{group_name}' from '{self._task_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._model.remove_combination(self._task_name, opt_key, group_name)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_radar_group_new_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            dlg = RadarGroupDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                self._model.set_radar_group_spec(self._task_name, new_name, new_spec)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_grid_group_enabled_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            self._model.set_profile_enabled(
                self._task_name, opt_key, group_name, cb.isChecked()
            )
            self.task_changed.emit()
        return _handler

    def _make_grid_group_edit_handler(
        self, opt_key: str, group_name: str, cb: QCheckBox, summary_lbl: QLabel
    ):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            existing_spec = self._model.get_grid_group_spec(
                self._task_name, opt_key, group_name
            )
            dlg = GridGroupDialog(self, name=group_name, existing_spec=existing_spec)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                if new_name != group_name:
                    self._model.remove_combination(self._task_name, opt_key, group_name)
                    self._model.set_grid_group_spec(self._task_name, opt_key, new_name, new_spec)
                    self.task_changed.emit()
                    self.show_task(self._model, self._task_name)
                else:
                    self._model.set_grid_group_spec(self._task_name, opt_key, group_name, new_spec)
                    cb.setChecked(new_spec.get("enabled", True))
                    summary_lbl.setText(_grid_group_summary(new_spec))
                    self.task_changed.emit()
        return _handler

    def _make_grid_group_delete_handler(self, opt_key: str, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            reply = QMessageBox.question(
                self,
                "Delete Grid Group",
                f"Delete grid group '{group_name}' from '{self._task_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._model.remove_combination(self._task_name, opt_key, group_name)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_grid_group_new_handler(self, opt_key: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None or self._task_name is None:
                return
            dlg = GridGroupDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                new_name = dlg.get_group_name()
                new_spec = dlg.get_group_spec()
                self._model.add_combination(self._task_name, opt_key, new_name, new_spec)
                self._model.set_grid_group_spec(self._task_name, opt_key, new_name, new_spec)
                self.task_changed.emit()
                self.show_task(self._model, self._task_name)
        return _handler

    def _make_downstream_group_handler(
        self,
        opt_key: str,
        group_name: str,
        cb: QCheckBox,
        all_groups: list,
        selected: list,
    ):
        def _handler(_state: int) -> None:
            if self._model is None or self._task_name is None:
                return
            if cb.isChecked():
                if group_name not in selected:
                    selected.append(group_name)
            else:
                if group_name in selected:
                    selected.remove(group_name)
            ordered = [g for g in all_groups if g in selected]
            self._model.set_downstream_cluster_group_names(self._task_name, ordered)
            self.task_changed.emit()
        return _handler

    def _make_downstream_group_details_handler(self, group_name: str):
        def _handler(_checked: bool = False) -> None:
            if self._model is None:
                return
            try:
                spec = self._model.get_cluster_group_spec("stimulus_cluster_touches", group_name)
            except KeyError:
                return
            dlg = ClusterGroupReadOnlyDialog(group_name, spec, self)
            dlg.exec_()
        return _handler
