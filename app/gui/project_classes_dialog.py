"""Class registry editor with explicit migration/deletion choices."""
import json

from qtpy import QtCore, QtWidgets

from project_format.classes import add_class, find_class, remove_class, rename_class
from project_format.models import ClassDefinition, Project, now


class ProjectClassesDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Project classes")
        self.resize(650, 580)
        layout = QtWidgets.QVBoxLayout(self)
        self.listing = QtWidgets.QListWidget()
        self.listing.currentItemChanged.connect(self._selected)
        layout.addWidget(self.listing)
        form = QtWidgets.QFormLayout()
        self.name, self.color = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.description, self.attributes = QtWidgets.QPlainTextEdit(), QtWidgets.QPlainTextEdit()
        self.description.setMaximumHeight(60)
        self.attributes.setMaximumHeight(80)
        self.geometry, self.shortcut = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        self.active = QtWidgets.QCheckBox("Active")
        for label, widget in [
            ("Name", self.name), ("Color #RRGGBB", self.color),
            ("Description", self.description), ("Attributes JSON", self.attributes),
            ("Allowed geometry (comma-separated)", self.geometry),
            ("Class assignment shortcut", self.shortcut), ("State", self.active),
        ]:
            form.addRow(label, widget)
        layout.addLayout(form)
        for text, slot in [
            ("Add class", self._add), ("Apply to whole class", self._apply),
            ("Rename/reclassify selected objects only", self._rename_selected),
            ("Remove / deactivate class...", self._remove),
        ]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, fn=slot: self._safe(fn))
            layout.addWidget(button)
        button = QtWidgets.QPushButton("Close")
        button.clicked.connect(self.accept)
        layout.addWidget(button)
        self._refresh()

    def _safe(self, fn):
        try:
            fn()
        except (ValueError, TypeError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Class change rejected", str(exc))

    def _definition(self):
        item = self.listing.currentItem()
        if item is None:
            raise ValueError("Select a class")
        return find_class(self.window._project, item.data(QtCore.Qt.UserRole))

    def _refresh(self):
        self.listing.clear()
        for definition in self.window._project.classes:
            item = QtWidgets.QListWidgetItem(definition.name + ("" if definition.active else " [inactive]"))
            item.setData(QtCore.Qt.UserRole, definition.id)
            item.setToolTip(definition.id)
            self.listing.addItem(item)
        if self.listing.count():
            self.listing.setCurrentRow(0)

    def _selected(self, item, previous):
        if item is None:
            return
        definition = self._definition()
        self.name.setText(definition.name)
        self.color.setText(definition.color)
        self.description.setPlainText(definition.description)
        self.attributes.setPlainText(json.dumps(definition.attributes, ensure_ascii=False, indent=2))
        self.geometry.setText(", ".join(definition.allowed_geometry))
        self.shortcut.setText(definition.shortcut)
        self.active.setChecked(definition.active)

    def _apply_result(self, result):
        self.window._apply_project_change(result)
        self._refresh()

    def _add(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add class", "Name")
        if ok and name.strip():
            self._apply_result(add_class(self.window._project, ClassDefinition(name=name.strip())))

    def _apply(self):
        current = self._definition()
        attrs = json.loads(self.attributes.toPlainText() or "{}")
        if not isinstance(attrs, dict):
            raise ValueError("Attributes must be a JSON object")
        replacement = ClassDefinition(
            id=current.id, name=self.name.text().strip(), color=self.color.text().strip(),
            description=self.description.toPlainText(), attributes=attrs,
            allowed_geometry=[s.strip() for s in self.geometry.text().split(",") if s.strip()],
            shortcut=self.shortcut.text(), active=self.active.isChecked())
        result = self.window._project.model_copy(deep=True)
        result.classes = [replacement if c.id == current.id else c for c in result.classes]
        result.journal.append({"time": now(), "action": "class_updated", "class_id": current.id})
        self._apply_result(Project.model_validate(result.model_dump(mode="json")))

    def _rename_selected(self):
        definition = self._definition()
        selected = [s.other_data.get("object_id") for s in self.window._get_selected_final_shapes()]
        self._apply_result(rename_class(self.window._project, definition.id,
                                        self.name.text(), selected_ids=selected))

    def _remove(self):
        definition = self._definition()
        count = sum(a.class_id == definition.id for image in self.window._project.images for a in image.annotations)
        options = ["Keep as inactive", "Transfer objects to another class", "Delete class and linked objects"]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Remove class", f"{count} linked objects. Choose the policy:", options, editable=False)
        if not ok:
            return
        target_id = None
        if choice == options[0]:
            policy = "deactivate"
        elif choice == options[1]:
            candidates = [c for c in self.window._project.classes if c.id != definition.id and c.active]
            if not candidates:
                raise ValueError("Create another active class first")
            name, ok = QtWidgets.QInputDialog.getItem(
                self, "Transfer objects", "Target class", [c.name for c in candidates], editable=False)
            if not ok:
                return
            target_id = next(c.id for c in candidates if c.name == name)
            policy = "transfer"
        else:
            if QtWidgets.QMessageBox.question(
                self, "Delete objects",
                f"Delete class '{definition.name}' and {count} linked objects across the whole project?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
            policy = "delete_objects"
        self._apply_result(remove_class(self.window._project, definition.id, policy, target_id))
