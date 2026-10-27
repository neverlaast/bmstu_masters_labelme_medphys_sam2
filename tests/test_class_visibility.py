import json
from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import test_gui as helpers
from qtpy import QtCore, QtWidgets
from labelme.shape import Shape


class ClassVisibilityTests(TestCase):
    setUpClass = classmethod(helpers.GUITests.setUpClass.__func__)
    setUp = helpers.GUITests.setUp
    tearDown = helpers.GUITests.tearDown
    final = helpers.GUITests.final
    prompt = helpers.GUITests.prompt

    def test_hide_and_show_are_class_specific_display_only(self):
        a, b, other = self.final(), self.final(offset=30), self.final("particle", 50)
        before = [deepcopy(s.other_data) for s in self.window.canvas.shapes]
        self.window.setClean()
        history_length = len(self.window._history._past)
        self.window.shapeSelectionChanged([a, other])
        self.assertEqual(self.window._set_class_visibility("cell", False), 2)
        self.assertFalse(self.window.canvas.isVisible(a))
        self.assertFalse(self.window.canvas.isVisible(b))
        self.assertTrue(self.window.canvas.isVisible(other))
        self.assertEqual(self.window.canvas.selectedShapes, [other])
        self.assertEqual([s.other_data for s in self.window.canvas.shapes], before)
        self.assertFalse(self.window.dirty)
        self.assertEqual(len(self.window._history._past), history_length)
        self.assertEqual(self.window._set_class_visibility("cell", True), 2)
        self.assertTrue(self.window.canvas.isVisible(a))
        self.assertTrue(self.window.canvas.isVisible(b))
        self.assertTrue(all(i.checkState() == QtCore.Qt.Checked for i in self.window.labelList))

    def test_visibility_survives_refresh_and_undo_redo(self):
        a = self.final()
        self.window._set_class_visibility("cell", False)
        self.window._refresh_views()
        self.assertFalse(self.window.canvas.isVisible(a))
        self.assertEqual(next(iter(self.window.labelList)).checkState(), QtCore.Qt.Unchecked)
        self.final("particle", 30)
        self.window.undoShapeEdit()
        self.assertFalse(self.window.canvas.isVisible(self.window.canvas.shapes[0]))
        self.window.redoShapeEdit()
        self.assertFalse(self.window.canvas.isVisible(self.window.canvas.shapes[0]))
        self.assertTrue(self.window.canvas.isVisible(self.window.canvas.shapes[1]))

    def test_hidden_polygons_are_saved_not_deleted(self):
        self.final()
        self.final("particle", 30)
        self.window._set_class_visibility("cell", False)
        target = Path(self.directory.name) / "annotations.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        data = json.loads(target.read_text())
        self.assertEqual([s["label"] for s in data["shapes"]], ["cell", "particle"])
        self.assertIsNone(data["imageData"])
        self.assertFalse(self.window.canvas.isVisible(self.window.canvas.shapes[0]))

    def test_all_final_geometries_hide_but_prompts_and_candidates_do_not(self):
        self.final()
        prompt = self.prompt()
        candidate = self.window._make_candidate([[5, 5], [20, 5], [20, 20]], 0)
        candidate.label = "cell"
        self.window.canvas.shapes.append(candidate)
        rectangle = Shape(label="cell", shape_type="rectangle")
        rectangle.addPoint(QtCore.QPointF(1, 1))
        rectangle.addPoint(QtCore.QPointF(5, 5))
        self.window.canvas.shapes.append(rectangle)
        self.window._refresh_views()
        self.window._set_class_visibility("cell", False)
        self.assertFalse(self.window.canvas.isVisible(rectangle))
        for shape in (prompt, candidate):
            self.assertTrue(self.window.canvas.isVisible(shape))

    def test_existing_rectangle_json_can_be_hidden_while_adding_clusters(self):
        path = Path(self.directory.name) / "particles.json"
        data = {
            "version": "5.2.1", "flags": {}, "imagePath": self.image_path.name,
            "imageData": None, "imageHeight": 96, "imageWidth": 96,
            "shapes": [
                {"label": "particle", "points": [[5, 5], [15, 15]],
                 "shape_type": "rectangle", "flags": {}, "group_id": None},
                {"label": "particle", "points": [[20, 20], [30, 30]],
                 "shape_type": "rectangle", "flags": {}, "group_id": None},
            ],
        }
        original = json.dumps(data)
        path.write_text(original)
        self.assertTrue(self.window.loadFile(str(path)))
        with patch.object(QtWidgets.QInputDialog, "getItem",
                          return_value=("particle", True)) as chooser:
            self.window._choose_class_visibility(False)
        chooser.assert_called_once()
        self.assertIn("particle", chooser.call_args.args[3])
        self.assertTrue(all(not self.window.canvas.isVisible(s)
                            for s in self.window.canvas.shapes))
        cluster = self.final("cluster", 40)
        self.window._refresh_views()
        self.assertTrue(self.window.canvas.isVisible(cluster))
        self.assertTrue(all(not self.window.canvas.isVisible(s)
                            for s in self.window.canvas.shapes if s.label == "particle"))
        saved = Path(self.directory.name) / "clusters.json"
        self.assertTrue(self.window.saveLabels(str(saved)))
        self.assertEqual([s["label"] for s in json.loads(saved.read_text())["shapes"]],
                         ["particle", "particle", "cluster"])
        self.assertEqual(path.read_text(), original)
        self.assertEqual(self.window._set_class_visibility("particle", True), 2)
        self.assertTrue(all(self.window.canvas.isVisible(s) for s in self.window.canvas.shapes))

    def test_circle_point_line_and_linestrip_are_supported(self):
        shapes = []
        for kind, points in (
            ("circle", [(20, 20), (25, 20)]),
            ("point", [(30, 30)]),
            ("line", [(40, 40), (50, 50)]),
            ("linestrip", [(60, 60), (70, 70), (80, 60)]),
        ):
            shape = Shape(label="particle", shape_type=kind)
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shapes.append(shape)
        self.window.canvas.shapes.extend(shapes)
        self.window._refresh_views()
        self.assertEqual(self.window._set_class_visibility("particle", False), 4)
        self.assertTrue(all(not self.window.canvas.isVisible(s) for s in shapes))
        self.window._refresh_views()
        self.assertTrue(all(not self.window.canvas.isVisible(s) for s in shapes))
        self.assertEqual(self.window._set_class_visibility("particle", True), 4)
        self.assertTrue(all(self.window.canvas.isVisible(s) for s in shapes))

    def test_accept_cluster_then_overwrite_same_json_preserves_particles(self):
        particle = self.final("particle")
        target = Path(self.directory.name) / "combined.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        original_particle = json.loads(target.read_text())["shapes"][0]
        self.window._set_class_visibility("particle", False)
        self.prompt("rectangle")
        candidate = self.window._make_candidate([[40, 40], [60, 40], [60, 60]], 0)
        self.window.canvas.shapes.append(candidate)
        self.window._refresh_views()
        with patch.object(QtWidgets.QInputDialog, "getItem", return_value=("cluster", True)):
            self.window._accept_candidates([candidate])
        self.assertTrue(self.window.canvas.isVisible(candidate))
        self.assertFalse(self.window.canvas.isVisible(particle))
        self.assertTrue(self.window.saveLabels(str(target)))
        saved = json.loads(target.read_text())
        self.assertEqual([s["label"] for s in saved["shapes"]], ["particle", "cluster"])
        self.assertEqual(saved["shapes"][0], original_particle)
        self.assertFalse(any(s.get("flags", {}).get("sam2_prompt") for s in saved["shapes"]))

    def test_dialog_cancel_changes_nothing(self):
        shape = self.final()
        with patch.object(QtWidgets.QInputDialog, "getItem", return_value=("cell", False)):
            self.window._choose_class_visibility(False)
        self.assertTrue(self.window.canvas.isVisible(shape))
        with patch.object(QtWidgets.QInputDialog, "getItem", return_value=("cell", True)):
            self.window._choose_class_visibility(False)
        self.assertFalse(self.window.canvas.isVisible(shape))
        with patch.object(QtWidgets.QInputDialog, "getItem", return_value=("cell", True)):
            self.window._choose_class_visibility(True)
        self.assertTrue(self.window.canvas.isVisible(shape))

    def test_cancelled_property_draft_prevents_hiding_selected_object(self):
        shape = self.final()
        self.window.shapeSelectionChanged([shape])
        self.window._properties_panel.description.setPlainText("pending")
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Cancel):
            self.assertEqual(self.window._set_class_visibility("cell", False), 0)
        self.assertTrue(self.window.canvas.isVisible(shape))
        self.assertTrue(self.window._properties_panel.dirty)

    def test_embedded_image_data_is_base64_image_bytes(self):
        import base64
        self.final()
        self.window._config["store_data"] = True
        target = Path(self.directory.name) / "embedded.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        payload = json.loads(target.read_text())
        self.assertEqual(base64.b64decode(payload["imageData"]), self.window.imageData)
