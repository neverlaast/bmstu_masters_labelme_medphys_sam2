"""Object inspector with real LabelMe/Qt objects, without loading a model."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from PIL import Image
from qtpy import QtCore, QtWidgets

import test_gui as helpers
from app.gui.object_properties import parse_attributes
from project_format.classes import add_class
from project_format.models import ClassDefinition
from project_format.io import load_project


class AttributeParsingTests(unittest.TestCase):
    def test_json_types_and_unicode(self):
        result = parse_attributes('{"тип": "клетка", "n": 3, "ok": true, "x": null, "nested": [1, {}]}')
        self.assertEqual(result["тип"], "клетка")
        self.assertEqual(result["n"], 3)
        self.assertIs(result["ok"], True)
        self.assertIsNone(result["x"])

    def test_invalid_attributes_are_not_evaluated(self):
        for text in ("", "[]", "null", "3", '{"a": NaN}', '{"a": Infinity}',
                     '{"a": 1e999}', '{"a": 1, "a": 2}', "__import__('os')"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_attributes(text)


class ObjectPropertiesTests(unittest.TestCase):
    setUpClass = classmethod(helpers.GUITests.setUpClass.__func__)
    setUp = helpers.GUITests.setUp
    tearDown = helpers.GUITests.tearDown
    final = helpers.GUITests.final
    prompt = helpers.GUITests.prompt

    @property
    def panel(self):
        return self.window._properties_panel

    def select(self, shape):
        self.window.shapeSelectionChanged([shape])
        return self.panel

    def test_no_selection_and_multiple_selection_are_not_editable(self):
        self.assertIsNone(self.panel.target)
        self.assertFalse(self.panel.editor.isEnabled())
        a, b = self.final(), self.final(offset=35)
        self.window.shapeSelectionChanged([a, b])
        self.assertIn("2 objects selected", self.panel.summary.text())
        self.assertIsNone(self.panel.target)
        self.assertFalse(self.panel.apply_button.isEnabled())

    def test_readonly_details_and_view_menu(self):
        shape = self.final()
        shape.other_data.update(track_id="track-7", source="sam2",
                                model_name="sam2", model_version="sam2.1:hash", device="mps")
        self.select(shape)
        self.assertEqual(self.panel.readonly["object_id"].text(), shape.other_data["object_id"])
        self.assertEqual(self.panel.readonly["track_id"].text(), "track-7")
        self.assertEqual(self.panel.readonly["source"].text(), "sam2")
        self.assertTrue(all(field.isReadOnly() for field in self.panel.readonly.values()))
        self.assertTrue(self.panel.metrics.isReadOnly())
        self.assertIn("Object properties", [a.text() for a in self.window.menus.view.actions()])
        self.assertEqual(self.panel.readonly["image"].text(), self.image_path.name)

    def test_modified_time_changes_only_for_edited_objects_and_undo_restores_it(self):
        a, b = self.final(), self.final(offset=35)
        a.other_data["modified_at"] = "2020-01-01T00:00:00+00:00"
        b.other_data["modified_at"] = "2020-01-02T00:00:00+00:00"
        self.window._reset_object_edit_tracking()
        self.window._history.reset(self.window.canvas.shapes)
        self.select(a)
        self.window.setDirty()
        self.assertEqual(a.other_data["modified_at"], "2020-01-01T00:00:00+00:00")
        a.points[0] += QtCore.QPointF(1, 1)
        self.window.setDirty()
        changed = a.other_data["modified_at"]
        self.assertNotEqual(changed, "2020-01-01T00:00:00+00:00")
        self.assertEqual(b.other_data["modified_at"], "2020-01-02T00:00:00+00:00")
        self.window.undoShapeEdit()
        self.assertEqual(self.window.canvas.shapes[0].other_data["modified_at"],
                         "2020-01-01T00:00:00+00:00")
        self.window.redoShapeEdit()
        self.assertEqual(self.window.canvas.shapes[0].other_data["modified_at"], changed)
        self.window.setDirty()
        self.assertEqual(self.window.canvas.shapes[0].other_data["modified_at"], changed)

    def test_native_label_edit_and_class_hotkey_respect_cancelled_draft(self):
        self.make_project()
        shape = self.window.canvas.shapes[0]
        description = shape.description
        self.select(shape)
        self.panel.description.setPlainText("pending")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.window.editLabel()
            self.window._class_hotkey("does-not-exist")
        self.assertTrue(self.panel.dirty)
        self.assertEqual(shape.description, description)

    def test_review_status_after_manual_mask_correction_is_not_overwritten(self):
        shape = self.final()
        shape.other_data.update(
            source="sam2", model_name="sam2", review_status="accepted",
            mask_outline=[[p.x(), p.y()] for p in shape.points],
            mask_rle={"size": [96, 96], "counts": [9216]},
            raster_source_valid=True, geometry_modified=False,
        )
        self.window.setDirty()
        shape.points[0] += QtCore.QPointF(1, 1)
        self.window.setDirty()
        self.assertEqual(shape.other_data["review_status"], "corrected")
        self.select(shape)
        self.panel.status_combo.setCurrentText("accepted")
        self.assertTrue(self.window._apply_object_properties())
        self.assertEqual(shape.other_data["review_status"], "accepted")
        self.window.setDirty()
        self.assertEqual(shape.other_data["review_status"], "accepted")
        shape.points[0] += QtCore.QPointF(1, 1)
        self.window.setDirty()
        self.assertEqual(shape.other_data["review_status"], "corrected")

    def test_apply_is_atomic_and_preserves_service_metadata_and_geometry(self):
        shape = self.final()
        shape.other_data.update(source="sam2", model_version="test-v", track_id="7",
                                refinement_token="opaque", mask_rle={"opaque": True})
        geometry = deepcopy(shape.points)
        identity = shape.other_data["object_id"]
        self.select(shape)
        self.panel.class_combo.setEditText("particle")
        self.panel.description.setPlainText("A particle")
        self.panel.attributes.setPlainText('{"diameter": 12, "checked": true}')
        self.panel.status_combo.setCurrentText("corrected")
        self.assertEqual(shape.label, "cell")  # draft is not applied yet
        self.panel.apply_button.click()
        self.assertEqual(shape.label, "particle")
        self.assertEqual(shape.description, "A particle")
        self.assertEqual(shape.other_data["attributes"], {"diameter": 12, "checked": True})
        self.assertEqual(shape.other_data["review_status"], "corrected")
        for key, value in {"object_id": identity, "source": "sam2", "model_version": "test-v",
                           "track_id": "7", "refinement_token": "opaque"}.items():
            self.assertEqual(shape.other_data[key], value)
        self.assertEqual(shape.points, geometry)
        self.assertTrue(shape.other_data["modified_at"])
        self.assertFalse(self.panel.dirty)
        self.window.undoShapeEdit()
        restored = self.window.canvas.shapes[0]
        self.assertEqual(restored.label, "cell")
        self.assertNotIn("attributes", restored.other_data)
        self.assertIs(self.panel.target, restored)
        self.window.redoShapeEdit()
        self.assertEqual(self.panel.attributes.toPlainText(),
                         json.dumps({"diameter": 12, "checked": True}, indent=2))
        self.assertEqual(self.window.canvas.shapes[0].label, "particle")

    def test_invalid_json_changes_nothing_and_reset_discards_draft(self):
        shape = self.final()
        before = deepcopy(shape.other_data)
        self.select(shape)
        self.panel.class_combo.setEditText("particle")
        self.panel.attributes.setPlainText('{"bad": NaN}')
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.assertFalse(self.window._apply_object_properties())
        warning.assert_called_once()
        self.assertEqual(shape.label, "cell")
        self.assertEqual(shape.other_data, before)
        self.assertTrue(self.panel.dirty)
        self.panel.reset_button.click()
        self.assertFalse(self.panel.dirty)
        self.assertEqual(self.panel.class_combo.currentText(), "cell")

    def test_semantically_identical_json_is_not_a_new_edit(self):
        shape = self.final()
        shape.other_data.update(attributes={"a": 1}, modified_at="original")
        self.window.setDirty()
        modified_at = shape.other_data["modified_at"]
        self.select(shape)
        self.panel.attributes.setPlainText('{"a":1}')
        history = len(self.window._history._past)
        self.assertTrue(self.window._apply_object_properties())
        self.assertEqual(shape.other_data["modified_at"], modified_at)
        self.assertEqual(len(self.window._history._past), history)
        self.assertFalse(self.panel.dirty)

    def test_candidates_and_prompts_are_readonly_not_accepted_by_status(self):
        candidate = self.window._add_candidate_polygon_to_canvas(
            [[10, 10], [25, 10], [25, 25]], score=.9)
        self.select(candidate)
        self.assertFalse(self.panel.editor.isEnabled())
        self.assertIn("Accept", self.panel.summary.text())
        self.assertEqual(self.panel.readonly["score"].text(), "0.9")
        self.assertTrue(self.window._is_sam2_candidate(candidate))
        prompt = self.prompt("circle")
        self.select(prompt)
        self.assertFalse(self.panel.editor.isEnabled())
        self.assertEqual(self.panel.readonly["source"].text(), "temporary prompt")

    def test_selection_save_discard_and_cancel(self):
        a, b = self.final(), self.final(offset=35)
        self.select(a)
        self.panel.description.setPlainText("keep me")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.select(b)
        self.assertEqual(self.window.canvas.selectedShapes, [a])
        self.assertTrue(self.panel.dirty)
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.select(b)
        self.assertEqual(a.description, "keep me")
        self.assertEqual(self.window.canvas.selectedShapes, [b])
        self.panel.description.setPlainText("discard")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard):
            self.select(a)
        self.assertNotEqual(b.description, "discard")
        self.assertFalse(self.panel.dirty)

    def test_invalid_draft_prevents_selection_and_delete(self):
        a, b = self.final(), self.final(offset=35)
        self.select(a)
        self.panel.attributes.setPlainText("[not json")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save), \
                patch.object(QtWidgets.QMessageBox, "warning"):
            self.select(b)
        self.assertEqual(self.window.canvas.selectedShapes, [a])
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.assertEqual(self.window._remove_shapes([a]), 0)
        self.assertIn(a, self.window.canvas.shapes)

    def test_external_edit_is_not_overwritten(self):
        shape = self.final()
        self.select(shape)
        self.panel.description.setPlainText("draft")
        shape.description = "changed elsewhere"
        with patch.object(QtWidgets.QMessageBox, "warning"):
            self.assertFalse(self.window._apply_object_properties())
        self.assertEqual(shape.description, "changed elsewhere")

    def test_labelme_json_roundtrip(self):
        shape = self.final()
        self.select(shape)
        self.panel.attributes.setPlainText('{"cell_type": "A", "verified": true}')
        self.panel.status_combo.setCurrentText("accepted")
        self.panel.apply_button.click()
        filename = self.image_path.with_suffix(".json")
        self.assertTrue(self.window.saveLabels(str(filename)))
        saved = json.loads(filename.read_text())["shapes"][0]
        self.assertEqual(saved["attributes"], {"cell_type": "A", "verified": True})
        self.assertEqual(saved["review_status"], "accepted")
        self.assertTrue(self.window.loadFile(str(filename)))
        loaded = self.window.canvas.shapes[0]
        self.assertEqual(loaded.other_data["attributes"], saved["attributes"])
        self.assertEqual(loaded.other_data["object_id"], shape.other_data["object_id"])

    def make_project(self):
        self.final()
        path = Path(self.directory.name) / "dataset.medphys.json"
        self.window.create_project(path)
        return path

    def test_project_roundtrip_and_pending_draft_on_save(self):
        path = self.make_project()
        shape = self.window.canvas.shapes[0]
        self.select(shape)
        self.panel.attributes.setPlainText('{"diameter": 12.5}')
        self.panel.status_combo.setCurrentText("corrected")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.assertTrue(self.window._save_project())
        saved = load_project(path).images[0].annotations[0]
        self.assertEqual(saved.attributes, {"diameter": 12.5})
        self.assertEqual(saved.review_status, "corrected")
        self.assertEqual(saved.id, shape.other_data["object_id"])
        self.window.open_project(path)
        self.select(self.window.canvas.shapes[0])
        self.assertEqual(self.panel.status_combo.currentText(), "corrected")
        self.assertEqual(json.loads(self.panel.attributes.toPlainText()), saved.attributes)
        self.assertEqual(self.panel.readonly["class_id"].text(), saved.class_id)
        self.assertEqual(self.panel.readonly["frame"].text(), "1 / 1 (project order)")

    def test_project_class_and_allowed_geometry_validation(self):
        self.make_project()
        project = add_class(self.window._project, ClassDefinition(
            name="point-only", allowed_geometry=["point"]))
        project = add_class(project, ClassDefinition(name="particle", color="#00ff00"))
        self.window._apply_project_change(project)
        shape = self.window.canvas.shapes[0]
        self.select(shape)
        self.assertFalse(self.panel.class_combo.isEditable())
        self.panel.class_combo.setCurrentText("point-only")
        with patch.object(QtWidgets.QMessageBox, "warning"):
            self.assertFalse(self.window._apply_object_properties())
        self.assertEqual(shape.label, "cell")
        self.panel.class_combo.setCurrentText("particle")
        self.assertTrue(self.window._apply_object_properties())
        self.assertEqual(shape.line_color.name(), "#00ff00")
        self.assertEqual(shape.other_data["class_id"], project.classes[-1].id)

    def test_navigation_draft_cancel_and_save(self):
        path = self.make_project()
        second = Path(self.directory.name) / "second.png"
        Image.new("RGB", (96, 96), "white").save(second)
        self.window._add_project_images([str(second)])
        self.select(self.window.canvas.shapes[0])
        self.panel.description.setPlainText("before navigation")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.assertFalse(self.window.loadFile(str(second)))
        self.assertEqual(Path(self.window.imagePath).resolve(), self.image_path.resolve())
        self.assertTrue(self.panel.dirty)
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.assertTrue(self.window.loadFile(str(second)))
        self.assertEqual(Path(self.window.imagePath).resolve(), second.resolve())
        self.assertIsNone(self.panel.target)
        self.assertEqual(load_project(path).images[0].annotations[0].description, "before navigation")
