"""Single-object inspector. Drafts never mutate a canvas shape directly."""
from __future__ import annotations

import json
from copy import deepcopy

from qtpy import QtCore, QtWidgets


REVIEW_STATUSES = ("unreviewed", "accepted", "corrected", "rejected", "imported")


def editable_properties(shape):
    data = shape.other_data or {}
    return {
        "label": shape.label or "",
        "description": shape.description or "",
        "attributes": deepcopy(data.get("attributes") or {}),
        "review_status": data.get("review_status", "unreviewed"),
    }


def parse_attributes(text):
    """Accept a JSON object, never Python expressions or non-finite numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate attribute name: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ValueError("Attributes must be a JSON object, for example {\"diameter\": 12}.")
        # Also catches exponent overflow (e.g. 1e999 -> inf).
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"Invalid attributes: {exc}") from exc


class ObjectPropertiesPanel(QtWidgets.QWidget):
    apply_requested = QtCore.Signal()
    reset_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target = None
        self.baseline = None
        self._draft_baseline = None
        self._loading = False
        self._editable = False
        root = QtWidgets.QVBoxLayout(self)
        self.summary = QtWidgets.QLabel("Select one object to inspect its properties.")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        self.readonly = {}
        self.editor = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.class_combo = QtWidgets.QComboBox()
        self.status_combo = QtWidgets.QComboBox()
        self.status_combo.addItems(REVIEW_STATUSES)
        self.description = QtWidgets.QPlainTextEdit()
        self.description.setMaximumHeight(70)
        self.attributes = QtWidgets.QPlainTextEdit()
        self.attributes.setMinimumHeight(90)
        self.attributes.setMaximumHeight(150)
        self.attributes.setPlaceholderText('{"diameter": 12, "checked": true}')
        form.addRow("Class", self.class_combo)
        form.addRow("Review status", self.status_combo)
        form.addRow("Description", self.description)
        form.addRow("Attributes (JSON)", self.attributes)
        root.addWidget(self.editor)
        hint = QtWidgets.QLabel(
            'Attributes are a JSON object, e.g. {"diameter": 12, "checked": true}.\n'
            "Changing review status does not delete an object.")
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        buttons = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.reset_button = QtWidgets.QPushButton("Reset")
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reset_button)
        # Keep the primary actions visible even in a short dock.
        root.insertLayout(1, buttons)
        self.details_toggle = QtWidgets.QToolButton()
        self.details_toggle.setText("Object details (read-only)")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(QtCore.Qt.RightArrow)
        root.addWidget(self.details_toggle)
        self.details = QtWidgets.QWidget()
        details_form = QtWidgets.QFormLayout(self.details)
        details_form.setContentsMargins(0, 0, 0, 0)
        for key, label in (
            ("object_id", "Object ID"), ("class_id", "Class ID"), ("geometry", "Geometry"),
            ("color", "Class/object color"), ("source", "Source"), ("model", "Model"),
            ("model_version", "Model version"), ("device", "Device"), ("image", "Image"),
            ("frame", "Frame"), ("track_id", "Track ID"), ("modified_at", "Modified"),
            ("score", "Model score"),
        ):
            field = QtWidgets.QLineEdit()
            field.setReadOnly(True)
            details_form.addRow(label, field)
            self.readonly[key] = field
        self.metrics = QtWidgets.QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMaximumHeight(100)
        details_form.addRow("Metrics", self.metrics)
        root.addWidget(self.details)
        self.details.hide()
        root.addStretch()
        self.details_toggle.toggled.connect(self._toggle_details)
        self.apply_button.clicked.connect(self.apply_requested)
        self.reset_button.clicked.connect(self.reset_requested)
        self.class_combo.currentTextChanged.connect(self._draft_changed)
        self.status_combo.currentTextChanged.connect(self._draft_changed)
        self.description.textChanged.connect(self._draft_changed)
        self.attributes.textChanged.connect(self._draft_changed)
        self.bind(None)

    def _toggle_details(self, expanded):
        self.details.setVisible(expanded)
        self.details_toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

    def _raw_draft(self):
        return (self.class_combo.currentText(), self.status_combo.currentText(),
                self.description.toPlainText(), self.attributes.toPlainText())

    @property
    def dirty(self):
        return bool(self.target is not None and self._editable
                    and self._draft_baseline is not None
                    and self._raw_draft() != self._draft_baseline)

    def _draft_changed(self, *_):
        if self._loading:
            return
        self.apply_button.setEnabled(self.dirty)
        self.reset_button.setEnabled(self.dirty)
        self.message.setText("Unapplied changes." if self.dirty else "")

    def values(self):
        label = self.class_combo.currentText().strip()
        if not label:
            raise ValueError("Class must not be empty.")
        return {
            "label": label,
            "review_status": self.status_combo.currentText(),
            "description": self.description.toPlainText(),
            "attributes": parse_attributes(self.attributes.toPlainText()),
        }

    def bind(self, shape, *, context=None, classes=(), editable=False,
             allow_new_class=False, summary="", force=False):
        context = context or {}
        if shape is self.target and self.dirty and not force:
            self.update_details(context)
            return
        self._loading = True
        try:
            self.target = shape
            self._editable = editable and shape is not None
            self.baseline = editable_properties(shape) if shape is not None else None
            self.summary.setText(summary or "Select one object to inspect its properties.")
            self.class_combo.clear()
            self.class_combo.setEditable(allow_new_class)
            names = list(dict.fromkeys(classes))
            label = self.baseline["label"] if self.baseline else ""
            if label and label not in names:
                names.append(label)
            self.class_combo.addItems(names)
            self.class_combo.setCurrentText(label)
            self.status_combo.clear()
            self.status_combo.addItems(REVIEW_STATUSES)
            status = self.baseline["review_status"] if self.baseline else "unreviewed"
            if status not in REVIEW_STATUSES:
                self.status_combo.addItem(status)
            self.status_combo.setCurrentText(status)
            self.description.setPlainText(self.baseline["description"] if self.baseline else "")
            self.attributes.setPlainText(
                json.dumps(self.baseline["attributes"], ensure_ascii=False, indent=2)
                if self.baseline else "")
            self.editor.setEnabled(self._editable)
            self._draft_baseline = self._raw_draft()
            self.update_details(context)
        finally:
            self._loading = False
        self._draft_changed()

    def update_details(self, context):
        for key, field in self.readonly.items():
            value = context.get(key)
            text = str(value) if value is not None and value != "" else "-"
            field.setText(text)
            field.setToolTip(text)
            field.setCursorPosition(0)
        metrics = context.get("metrics") or {}
        self.metrics.setPlainText(json.dumps(metrics, ensure_ascii=False, indent=2) if metrics else "-")
