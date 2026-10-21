"""Object inspector integration, validation and transactional canvas edits."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from qtpy import QtCore, QtWidgets

from app.gui.object_properties import ObjectPropertiesPanel, REVIEW_STATUSES, editable_properties


class ObjectPropertiesMixin:
    def _object_edit_signature(self, shape):
        signature = self._shape_signature([shape])[0]
        signature[-1].pop("modified_at", None)
        return deepcopy(signature)

    def _reset_object_edit_tracking(self):
        self._object_edit_signatures = {
            (s.other_data or {}).get("object_id"): self._object_edit_signature(s)
            for s in self.canvas.shapes if not self._is_temporary(s)
        }

    def _update_object_edit_timestamps(self):
        """Only actual edits change time; selection, saves and undo do not."""
        previous = getattr(self, "_object_edit_signatures", {})
        current = {}
        for shape in self.canvas.shapes:
            if self._is_temporary(shape):
                continue
            data = shape.other_data
            identity = data["object_id"]
            signature = self._object_edit_signature(shape)
            if previous.get(identity) != signature:
                data["modified_at"] = datetime.now(timezone.utc).isoformat()
            current[identity] = signature
        self._object_edit_signatures = current

    def editLabel(self, *args, **kwargs):
        if not self._resolve_property_draft():
            return
        result = super().editLabel(*args, **kwargs)
        self._refresh_object_properties()
        return result

    def _install_object_properties(self):
        panel = ObjectPropertiesPanel(self)
        dock = QtWidgets.QDockWidget("Object properties", self)
        dock.setObjectName("MedPhysObjectProperties")
        scroll = QtWidgets.QScrollArea(dock)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        dock.setWidget(scroll)
        dock.setMinimumWidth(290)
        dock.setMinimumHeight(380)
        self._properties_panel, self._properties_dock = panel, dock
        self._properties_applying = False
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.tabifyDockWidget(self.flag_dock, dock)
        self.resizeDocks([dock], [450], QtCore.Qt.Vertical)
        dock.raise_()
        self.menus.view.addAction(dock.toggleViewAction())
        panel.apply_requested.connect(self._apply_object_properties)
        panel.reset_requested.connect(lambda: self._refresh_object_properties(force=True))

    def _object_property_context(self, shape):
        data = shape.other_data or {}
        model = data.get("model_name")
        version = data.get("model_version")
        definition = None
        frame = None
        if self._project is not None:
            definition = next((c for c in self._project.classes if c.name == shape.label), None)
            index = next((i for i, image in enumerate(self._project.images)
                          if image.id == self._project_current), None)
            if index is not None:
                frame = f"{index + 1} / {len(self._project.images)} (project order)"
            record = next((r for r in self._project.model_records
                           if model and r.get("model_name") == model
                           and r.get("model_version") == version), {})
            if record.get("variant"):
                model = f"{model} {record['variant']}"
        if self._is_sam2_prompt(shape):
            source = "temporary prompt"
        else:
            source = data.get("source", "manual")
        return {
            "object_id": data.get("object_id"), "class_id": definition.id if definition else data.get("class_id"),
            "geometry": shape.shape_type, "color": shape.line_color.name(),
            "source": source, "model": model, "model_version": version,
            "device": data.get("device"), "image": Path(self.imagePath).name if self.imagePath else None,
            "frame": frame, "track_id": data.get("track_id"),
            "modified_at": data.get("modified_at"), "score": data.get("sam2_score"),
            "metrics": data.get("metrics"),
        }

    def _refresh_object_properties(self, *, force=False):
        panel = getattr(self, "_properties_panel", None)
        if panel is None or self._initializing:
            return
        selected = [s for s in self.canvas.selectedShapes if s in self.canvas.shapes]
        if len(selected) != 1:
            panel.bind(None, summary=(f"{len(selected)} objects selected. "
                                     "Select one object to edit its properties."
                                     if selected else ""), force=force)
            return
        shape = selected[0]
        temporary = self._is_temporary(shape)
        if self._project is not None:
            classes = [c.name for c in self._project.classes if c.active]
        else:
            classes = sorted({s.label for s in self.canvas.shapes
                              if s.label and not self._is_temporary(s)}
                             | set(self._config.get("labels") or []))
        if self._is_sam2_candidate(shape):
            summary = "SAM2 candidate (read-only). Accept it before editing object properties."
        elif self._is_sam2_prompt(shape):
            summary = "Temporary prompt (read-only). It is not a final annotation."
        else:
            summary = "One final object selected. Apply commits all property changes together."
        panel.bind(shape, context=self._object_property_context(shape), classes=classes,
                   editable=not temporary, allow_new_class=self._project is None,
                   summary=summary, force=force)

    def _resolve_property_draft(self):
        panel = getattr(self, "_properties_panel", None)
        if panel is None or not panel.dirty or getattr(self, "_properties_applying", False):
            return True
        mb = QtWidgets.QMessageBox
        answer = mb.question(
            self, "Unapplied object properties",
            "Apply the object property changes before continuing?",
            mb.Save | mb.Discard | mb.Cancel, mb.Save)
        if answer == mb.Save:
            return self._apply_object_properties()
        if answer == mb.Discard:
            self._refresh_object_properties(force=True)
            return True
        return False

    def _apply_object_properties(self):
        panel = self._properties_panel
        shape = panel.target
        if shape is None or not panel.dirty:
            return True
        try:
            if shape not in self.canvas.shapes or self._is_temporary(shape):
                raise ValueError("This object is no longer editable. Reset the properties.")
            if editable_properties(shape) != panel.baseline:
                raise ValueError("Object properties changed elsewhere. Use Reset before editing again.")
            values = panel.values()
            if values["review_status"] not in REVIEW_STATUSES and (
                    values["review_status"] != panel.baseline["review_status"]):
                raise ValueError("Unknown review status.")
            if self._project is not None:
                definition = next((c for c in self._project.classes
                                   if c.name == values["label"]), None)
                if definition is None:
                    raise ValueError("Unknown project class. Add it in Project classes.")
                if values["label"] != shape.label and not definition.active:
                    raise ValueError("Cannot assign an inactive class.")
                if shape.shape_type not in definition.allowed_geometry:
                    raise ValueError(f"Class {definition.name!r} does not allow {shape.shape_type}.")
            elif not self.validateLabel(values["label"]):
                raise ValueError("This class is not allowed by the current label configuration.")
        except (ValueError, TypeError, RecursionError) as exc:
            panel.message.setText(str(exc))
            QtWidgets.QMessageBox.warning(self, "Invalid object properties", str(exc))
            return False

        if values == editable_properties(shape):
            self._refresh_object_properties(force=True)
            return True
        self._properties_applying = True
        try:
            shape.label = values["label"]
            shape.description = values["description"]
            data = dict(shape.other_data or {})
            data.update(attributes=deepcopy(values["attributes"]),
                        review_status=values["review_status"],
                        modified_at=datetime.now(timezone.utc).isoformat())
            if self._project is not None:
                data["class_id"] = definition.id
            shape.other_data = data
            # Mark the draft clean before setDirty (upstream may autosave).
            self._refresh_object_properties(force=True)
            self.canvas.storeShapes()
            self.setDirty()
            self._refresh_views()
            self.statusBar().showMessage("Object properties applied", 4000)
        finally:
            self._properties_applying = False
        return True
