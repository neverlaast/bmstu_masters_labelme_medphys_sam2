"""Offscreen regression tests using the actual vendored LabelMe/Qt widgets."""
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.paths import PROJECT_ROOT, ensure_labelme

ensure_labelme()

import yaml
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets
from labelme.shape import Shape

from app.gui.labelme_window import MedPhysLabelMeWindow
from app.gui.worker_task import WorkerTask
from app.worker_api.schemas import SegmentationCandidate, SegmentationResponse
from app.worker_api.schemas import CirclePrompt


class GUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.directory.name) / "frame.png"
        Image.new("RGB", (96, 96), color="white").save(self.image_path)
        config = yaml.safe_load((PROJECT_ROOT / "third_party" / "labelme_upstream" /
                                 "labelme" / "config" / "default_config.yaml").read_text())
        config.update(auto_save=False, store_data=False, keep_prev=False)
        self.window = MedPhysLabelMeWindow(config=config)
        self.window.loadFile(str(self.image_path))

    def tearDown(self):
        for task in list(self.window._jobs):
            task.cancel()
            task.wait(3000)
        self.application.processEvents()
        # Do not invoke upstream closeEvent/QSettings or a save-confirm dialog.
        self.window.deleteLater()
        self.application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.application.processEvents()
        self.directory.cleanup()

    def final(self, label="cell", offset=0):
        shape = Shape(label=label, shape_type="polygon")
        for x, y in [(10+offset, 10), (30+offset, 10), (30+offset, 30), (10+offset, 30)]:
            shape.addPoint(QtCore.QPointF(x, y))
        shape.close()
        self.window.canvas.shapes.append(shape)
        self.window.canvas.storeShapes()
        self.window.addLabel(shape)
        self.window.setDirty()
        return shape

    def prompt(self, kind="circle", label=1):
        self.window._activate_sam2_prompt_drawing(kind, label)
        shape = Shape(shape_type=kind)
        shape.addPoint(QtCore.QPointF(40, 40))
        if kind != "point":
            shape.addPoint(QtCore.QPointF(55, 40 if kind == "circle" else 55))
        self.window.canvas.current = shape
        with patch.object(self.window.labelDialog, "popUp",
                          side_effect=AssertionError("Prompts must not open class dialog")):
            self.window.canvas.finalise()
        return shape

    def test_point_mode_stays_active_for_several_points(self):
        self.window._activate_sam2_prompt_drawing("point", 1)
        for x, label in [(30, 1), (45, 1), (60, 0)]:
            if label != self.window._prompt_label:
                self.window._activate_sam2_prompt_drawing("point", label)
            self.assertEqual(self.window.sam2_prompt_type, "point")
            shape = Shape(shape_type="point")
            shape.addPoint(QtCore.QPointF(x, 40))
            self.window.canvas.current = shape
            self.window.canvas.finalise()
            self.assertEqual(self.window.sam2_prompt_type, "point")  # no re-selection needed
        request = self.window._build_segmentation_request()
        self.assertEqual([p.label for p in request.points], [1, 1, 0])
        self.window.toggleDrawMode(True)
        self.assertIsNone(self.window.sam2_prompt_type)

    def test_prompts_bypass_class_dialog_and_form_combined_request(self):
        roi = self.prompt("circle")
        self.prompt("point", 0)
        self.assertTrue(self.window._is_sam2_prompt(roi))
        self.assertEqual(len(self.window.labelList), 0)
        request = self.window._build_segmentation_request()
        self.assertEqual(request.circle.radius, 15)
        self.assertEqual(request.points[0].label, 0)

    def test_normal_annotations_still_use_upstream_label_dialog(self):
        self.window.toggleDrawMode(False, "rectangle")
        shape = Shape(shape_type="rectangle")
        shape.addPoint(QtCore.QPointF(10, 10))
        shape.addPoint(QtCore.QPointF(20, 20))
        self.window.canvas.current = shape
        with patch.object(self.window.labelDialog, "popUp", return_value=("cell", {}, None, "")) as popup:
            self.window.canvas.finalise()
        popup.assert_called_once()
        self.assertFalse(self.window._is_temporary(shape))
        self.assertEqual(shape.label, "cell")

    def test_multiple_regions_and_points_form_independent_requests(self):
        self.prompt("circle")
        second = self.prompt("rectangle")
        second.points = [QtCore.QPointF(65, 65), QtCore.QPointF(90, 90)]
        self.prompt("point", 1)
        negative = self.prompt("point", 0)
        negative.points = [QtCore.QPointF(5, 5)]
        requests = self.window._build_interactive_requests()
        self.assertEqual(len(requests), 2)
        self.assertIsNotNone(requests[0].circle)
        self.assertIsNotNone(requests[1].box)
        self.assertEqual([p.label for p in requests[0].points], [1, 0])
        self.assertEqual([p.label for p in requests[1].points], [0])
        self.assertTrue(all(not r.return_multiple_masks for r in requests))
        self.assertEqual(requests[0].image_path, requests[1].image_path)
        with self.assertRaisesRegex(ValueError, "Object search needs one ROI"):
            self.window._build_segmentation_request("automatic")
        whole = self.window._build_segmentation_request("automatic", whole_image=True)
        self.assertIsNone(whole.circle)
        self.assertIsNone(whole.box)

    def test_positive_point_outside_or_in_overlapping_regions_is_rejected(self):
        self.prompt("circle")
        self.prompt("circle")
        point = self.prompt("point")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.window._build_interactive_requests()
        point.points = [QtCore.QPointF(5, 5)]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.window._build_interactive_requests()

    def test_circle_and_rectangle_modes_stay_active(self):
        for kind in ("circle", "rectangle"):
            self.prompt(kind)
            self.assertEqual(self.window.sam2_prompt_type, kind)
            shape = Shape(shape_type=kind)
            shape.addPoint(QtCore.QPointF(65, 65))
            shape.addPoint(QtCore.QPointF(80, 80))
            self.window.canvas.current = shape
            self.window.canvas.finalise()
            self.assertEqual(self.window.sam2_prompt_type, kind)
        self.window.toggleDrawMode(True)
        self.assertIsNone(self.window.sam2_prompt_type)

    def test_batch_client_keeps_best_per_object_and_refinement_tokens(self):
        from app.worker_api.client import WorkerClient
        self.prompt("circle")
        self.prompt("rectangle")
        requests = self.window._build_interactive_requests()
        a = SegmentationCandidate(score=.9, polygon=[[1, 1], [10, 1], [10, 10]],
                                  refinement_token="a")
        b = a.model_copy(update={"score": .7, "refinement_token": "b"})
        response = SegmentationResponse(
            ok=True, model_name="sam2", model_version="test", device="cpu",
            elapsed_ms=10, candidates=[a, b])
        client = WorkerClient()
        with patch.object(client, "segment", return_value=response) as segment:
            result = client.segment_many(requests)
        self.assertEqual(segment.call_count, 2)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.elapsed_ms, 20)
        self.assertEqual([c.metadata["prompt_index"] for c in result.candidates], [1, 2])
        self.assertEqual([c.refinement_token for c in result.candidates], ["a", "a"])
        self.assertEqual(a.metadata, {})

    def test_batch_cancellation_stops_next_request_and_failure_keeps_candidates(self):
        from app.worker_api.errors import WorkerResponseError
        self.prompt("circle")
        self.prompt("rectangle")
        requests = self.window._build_interactive_requests()
        response = SegmentationResponse(
            ok=True, model_name="sam2", model_version="test", device="cpu", elapsed_ms=1)
        with patch.object(self.window.worker_client, "segment", return_value=response) as segment:
            with self.assertRaisesRegex(WorkerResponseError, "cancelled"):
                self.window.worker_client.segment_many(requests, cancelled=iter([False, True]).__next__)
            self.assertEqual(segment.call_count, 1)
        old = self.window._add_candidate_polygon_to_canvas([[1, 1], [10, 1], [10, 10]])
        with patch.object(self.window.worker_client, "segment",
                          side_effect=[response, WorkerResponseError("second failed")]), \
                patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.window._run_sam2()
            deadline = time.monotonic() + 3
            while self.window._job is not None and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(.005)
            self.assertIsNone(self.window._job)
            self.assertIn(old, self.window.canvas.shapes)
            warning.assert_called_once()

    def test_native_selection_is_bidirectional_and_keeps_candidates(self):
        final = self.final()
        candidate = self.window._add_candidate_polygon_to_canvas([[35, 35], [50, 35], [50, 50]], score=.9)
        self.window.canvas.selectShapes([final, candidate])
        self.assertEqual(len(self.window.canvas.selectedShapes), 2)
        self.assertEqual(len(self.window.labelList.selectedItems()), 1)
        self.assertEqual(len(self.window._sam2_candidate_list.selectedItems()), 1)
        self.window.labelList.clearSelection()
        self.assertEqual(self.window.canvas.selectedShapes, [])
        self.window._sam2_candidate_list.item(0).setSelected(True)
        self.assertEqual(self.window.canvas.selectedShapes, [candidate])

    def test_accept_selected_preserves_unselected_candidates(self):
        a = self.window._add_candidate_polygon_to_canvas([[10, 10], [20, 10], [20, 20]], score=.9)
        b = self.window._add_candidate_polygon_to_canvas([[30, 30], [40, 30], [40, 40]], score=.8)
        self.window.shapeSelectionChanged([a])
        with patch.object(self.window, "_ask_target_label", return_value="particle"):
            self.window._accept_selected_sam2_candidates()
        self.assertEqual(a.label, "particle")
        self.assertFalse(self.window._is_temporary(a))
        self.assertIn(b, self.window.canvas.shapes)
        self.assertTrue(self.window._is_sam2_candidate(b))
        self.assertEqual(len(self.window.labelList), 1)
        self.window.undoShapeEdit()
        self.assertEqual(len(list(self.window._iter_sam2_candidate_shapes())), 2)
        self.window.redoShapeEdit()
        self.assertEqual(len(list(self.window._iter_sam2_candidate_shapes())), 1)

    def test_batch_relabel_delete_confirmation_undo_redo(self):
        a, b = self.final(), self.final(offset=35)
        self.window.shapeSelectionChanged([a, b])
        with patch.object(self.window, "_ask_target_label", return_value="particle"):
            self.window._relabel_selected_final_shapes()
        self.assertEqual([s.label for s in self.window.canvas.shapes], ["particle", "particle"])
        self.window.undoShapeEdit()
        self.assertEqual([s.label for s in self.window.canvas.shapes], ["cell", "cell"])
        self.window.redoShapeEdit()
        self.window._select_all_final()
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No):
            self.window._delete_selected_final_shapes()
        self.assertEqual(len(self.window.canvas.shapes), 2)
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes) as question:
            self.window._delete_selected_final_shapes()
        self.assertIn("2", question.call_args.args[2])
        self.assertEqual(len(self.window.canvas.shapes), 0)
        self.window.undoShapeEdit()
        self.assertEqual(len(self.window.canvas.shapes), 2)
        self.window.redoShapeEdit()
        self.assertEqual(len(self.window.canvas.shapes), 0)

    def test_save_excludes_temporary_and_preserves_original_image(self):
        original = hashlib.sha256(self.image_path.read_bytes()).hexdigest()
        self.final()
        self.prompt("circle")
        self.window._add_candidate_polygon_to_canvas([[35, 35], [50, 35], [50, 50]], score=.9)
        target = Path(self.directory.name) / "frame.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        data = json.loads(target.read_text())
        self.assertEqual(len(data["shapes"]), 1)
        self.assertEqual(data["shapes"][0]["label"], "cell")
        self.assertIn("object_id", data["shapes"][0])
        self.assertEqual(hashlib.sha256(self.image_path.read_bytes()).hexdigest(), original)
        self.window.loadFile(str(target))
        self.assertEqual(len(self.window.canvas.shapes), 1)
        self.assertEqual(self.window.canvas.shapes[0].label, "cell")

    def test_export_snapshot_excludes_prompts_candidates_and_does_not_mutate(self):
        final = self.final()
        self.prompt("circle")
        self.window._add_candidate_polygon_to_canvas([[35, 35], [50, 35], [50, 50]], score=.9)
        before = self.window._shape_signature(self.window.canvas.shapes)
        shapes, width, height, classes = self.window._export_snapshot()
        self.assertEqual((width, height), (96, 96))
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0]["object_id"], final.other_data["object_id"])
        self.assertEqual(before, self.window._shape_signature(self.window.canvas.shapes))

    def test_export_menu_writes_current_image_and_handles_invalid_geometry(self):
        self.final()
        target = Path(self.directory.name) / "exports"
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(target)), \
                patch.object(QtWidgets.QMessageBox, "information") as success:
            self.window._export_current_image("coco")
        success.assert_called_once()
        self.assertEqual(len(list(target.glob("*/annotations.json"))), 1)
        self.window.canvas.shapes[0].points[0] = QtCore.QPointF(-1, 1)
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory") as chooser, \
                patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.window._export_current_image("coco")
        chooser.assert_not_called()
        warning.assert_called_once()

    def test_atomic_save_failure_preserves_previous_json(self):
        self.final()
        target = Path(self.directory.name) / "frame.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        original = target.read_bytes()
        with patch("app.gui.labelme_window.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.window.saveLabels(str(target))
        self.assertEqual(target.read_bytes(), original)

    def test_save_button_handles_io_failure_without_leaking_exception(self):
        self.final()
        target = Path(self.directory.name) / "frame.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        original = target.read_bytes()
        with patch("app.gui.labelme_window.os.replace", side_effect=OSError("disk failure")), \
             patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.assertFalse(self.window._saveFile(str(target)))
        warning.assert_called_once()
        self.assertEqual(target.read_bytes(), original)
        self.assertTrue(self.window.dirty)

    def test_background_request_keeps_event_loop_responsive(self):
        self.prompt("circle")
        response = SegmentationResponse(
            ok=True, model_name="sam2", model_version="test", device="cpu", elapsed_ms=200,
            candidates=[SegmentationCandidate(score=.9, polygon=[[10, 10], [30, 10], [30, 30]])])
        ticks = []
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.append(True))
        timer.start(10)
        def delayed(request):
            time.sleep(.2)
            return response
        with patch.object(self.window.worker_client, "segment", side_effect=delayed):
            self.window._run_sam2()
            self.pump_until(lambda: self.window._job is None)
        timer.stop()
        self.assertGreater(len(ticks), 3)
        self.assertEqual(len(list(self.window._iter_sam2_candidate_shapes())), 1)

    def test_cancel_discards_result(self):
        self.prompt("circle")
        response = SegmentationResponse(
            ok=True, model_name="sam2", model_version="test", device="cpu", elapsed_ms=0,
            candidates=[SegmentationCandidate(score=.9, polygon=[[10, 10], [30, 10], [30, 30]])])
        def delayed(request):
            time.sleep(.15)
            return response
        with patch.object(self.window.worker_client, "segment", side_effect=delayed):
            self.window._run_sam2()
            self.window._cancel_sam2()
            self.pump_until(lambda: self.window._job is None)
        self.assertEqual(len(list(self.window._iter_sam2_candidate_shapes())), 0)

    def test_frame_change_discards_stale_result(self):
        self.prompt("circle")
        response = SegmentationResponse(
            ok=True, model_name="sam2", model_version="test", device="cpu", elapsed_ms=0,
            candidates=[SegmentationCandidate(score=.9, polygon=[[10, 10], [30, 10], [30, 30]])])
        def delayed(request):
            time.sleep(.15)
            return response
        with patch.object(self.window.worker_client, "segment", side_effect=delayed):
            self.window._run_sam2()
            self.window.loadFile(str(self.image_path))
            self.pump_until(lambda: self.window._job is None)
        self.assertEqual(len(list(self.window._iter_sam2_candidate_shapes())), 0)

    def test_reordering_final_list_does_not_delete_candidates(self):
        a = self.final()
        b = self.window._add_candidate_polygon_to_canvas([[35, 35], [50, 35], [50, 50]])
        self.window.labelOrderChanged()
        self.assertIn(a, self.window.canvas.shapes)
        self.assertIn(b, self.window.canvas.shapes)

    def test_copy_paste_uses_distinct_objects_and_ids(self):
        original = self.final()
        self.window.shapeSelectionChanged([original])
        self.window.copySelectedShape()
        self.window.pasteSelectedShape()
        self.window.pasteSelectedShape()
        shapes = self.window.canvas.shapes
        self.assertEqual(len(shapes), 3)
        self.assertEqual(len({id(s) for s in shapes}), 3)
        self.assertEqual(len({s.other_data["object_id"] for s in shapes}), 3)

    def test_native_shape_movement_supports_undo_redo(self):
        shape = self.final()
        before = [(p.x(), p.y()) for p in shape.points]
        shape.moveBy(QtCore.QPointF(4, 3))
        self.window.canvas.storeShapes()
        self.window.canvas.shapeMoved.emit()
        after = [(p.x(), p.y()) for p in shape.points]
        self.window.undoShapeEdit()
        self.assertEqual([(p.x(), p.y()) for p in self.window.canvas.shapes[0].points], before)
        self.window.redoShapeEdit()
        self.assertEqual([(p.x(), p.y()) for p in self.window.canvas.shapes[0].points], after)

    def test_labelme_edge_distance_does_not_use_numpy_cross(self):
        import numpy as np
        from labelme import utils
        from labelme.utils import qt
        from app.core.geometry import point_segment_distance
        self.assertIs(utils.distancetoline, point_segment_distance)
        self.assertIs(qt.distancetoline, point_segment_distance)
        shape = self.final()
        with patch.object(np, "cross", side_effect=AssertionError("2D cross must not be used")):
            self.assertIsNotNone(shape.nearestEdge(QtCore.QPointF(20, 10), 1))
            self.assertAlmostEqual(
                utils.distancetoline(QtCore.QPointF(20, 12),
                                    [QtCore.QPointF(10, 10), QtCore.QPointF(30, 10)]), 2)

    def test_actual_mouse_hover_event_over_polygon_and_candidate(self):
        shape = self.final()
        canvas = self.window.canvas
        canvas.setEditing(True)
        # Send a real Qt event through the event dispatcher, traversing the
        # same Canvas.mouseMoveEvent -> Shape.nearestEdge path as the crash.
        image_point = QtCore.QPointF(20, 10)
        widget_point = (image_point + canvas.offsetToCenter()) * canvas.scale
        event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, widget_point,
            QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
        with patch.object(shape, "nearestEdge", wraps=shape.nearestEdge) as nearest:
            self.application.sendEvent(canvas, event)
            nearest.assert_called()
        self.assertIs(canvas.hShape, shape)
        candidate = self.window._add_candidate_polygon_to_canvas(
            [[35, 35], [60, 35], [60, 60]], score=.9)
        widget_point = (QtCore.QPointF(47, 35) + canvas.offsetToCenter()) * canvas.scale
        event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove, widget_point,
            QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
        with patch.object(candidate, "nearestEdge", wraps=candidate.nearestEdge) as nearest:
            self.application.sendEvent(canvas, event)
            nearest.assert_called()
        self.assertIs(canvas.hShape, candidate)

    def test_accept_notifies_model_and_renders_without_manual_annotation(self):
        candidate = self.window._add_candidate_polygon_to_canvas(
            [[10,10],[30,10],[30,30]], score=.9)
        self.window.shapeSelectionChanged([candidate])
        inserted, reset = [], []
        self.window.labelList.model().rowsInserted.connect(lambda *args: inserted.append(True))
        self.window.labelList.model().modelReset.connect(lambda *args: reset.append(True))
        with patch.object(self.window, "_ask_target_label", return_value="cell"):
            self.window._accept_selected_sam2_candidates()
        self.window.resize(1000,700)
        self.window.shape_dock.show()
        self.window.show()
        self.application.processEvents()
        self.assertTrue(inserted)
        self.assertTrue(reset)
        index = self.window.labelList.model().index(0,0)
        self.assertTrue(self.window.labelList.visualRect(index).isValid())

    def test_roi_selection_uses_centroids(self):
        a, b = self.final(), self.final(offset=40)
        self.prompt("circle")
        # Circle centre 40,40 / radius15 includes neither original centroid.
        # Move the ROI to centre20,20, radius15 to include only the first.
        roi = list(self.window._iter_sam2_prompt_shapes())[0]
        roi.points = [QtCore.QPointF(20,20),QtCore.QPointF(35,20)]
        self.window._select_in_roi()
        self.assertEqual(self.window._get_selected_final_shapes(), [a])

    def test_refine_accepted_object_replaces_geometry_preserving_id(self):
        original = self.final()
        original.other_data["refinement_token"] = "state-1"
        original.other_data["custom_attribute"] = "keep"
        original.group_id = 7
        original.description = "reviewed cell"
        original.flags["difficult"] = True
        original_id = original.other_data["object_id"]
        self.window.shapeSelectionChanged([original])
        self.prompt("point", 1)
        request = self.window._build_segmentation_request(refine=True)
        self.assertEqual(request.refinement_token, "state-1")
        self.assertEqual(request.refinement_target_id, original_id)
        self.assertIsNone(request.circle)
        self.assertIsNone(request.box)
        candidate = self.window._add_candidate_polygon_to_canvas(
            [[10,10],[35,10],[35,35]], score=.9)
        candidate.other_data["refines_object_id"] = original_id
        self.window.shapeSelectionChanged([candidate])
        with patch.object(self.window, "_ask_target_label", return_value="cell"):
            self.window._accept_selected_sam2_candidates()
        finals = [s for s in self.window.canvas.shapes if not self.window._is_temporary(s)]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].other_data["object_id"], original_id)
        self.assertEqual(finals[0].points[1].x(), 35)
        self.assertEqual(finals[0].group_id, 7)
        self.assertEqual(finals[0].description, "reviewed cell")
        self.assertTrue(finals[0].flags["difficult"])
        self.assertEqual(finals[0].other_data["custom_attribute"], "keep")
        self.window.undoShapeEdit()
        finals = [s for s in self.window.canvas.shapes if not self.window._is_temporary(s)]
        self.assertEqual(finals[0].points[1].x(), 30)

    def test_polygon_roi_request_is_automatic_only(self):
        self.window._activate_sam2_prompt_drawing("polygon")
        shape = Shape(shape_type="polygon")
        for x,y in [(10,10),(60,10),(40,60)]:
            shape.addPoint(QtCore.QPointF(x,y))
        self.window.canvas.current = shape
        self.window.canvas.finalise()
        request = self.window._build_segmentation_request("automatic")
        self.assertIsNotNone(request.polygon)
        self.assertEqual(request.generation.mode, "automatic")
        with self.assertRaises(ValueError):
            self.window._build_segmentation_request()

    def test_manual_vector_edit_invalidates_old_raster_and_logits(self):
        shape = self.final()
        shape.other_data.update(
            mask_outline=[[p.x(),p.y()] for p in shape.points],
            mask_rle={"size":[96,96],"counts":[9216]}, raster_source_valid=True,
            refinement_token="old-mask-state",
        )
        shape.moveBy(QtCore.QPointF(2,2))
        self.window.canvas.shapeMoved.emit()
        self.assertIsNone(shape.other_data["mask_rle"])
        self.assertFalse(shape.other_data["raster_source_valid"])
        self.assertIsNone(shape.other_data["refinement_token"])

    def pump_until(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.application.processEvents()
        self.assertTrue(predicate(), "Background task timed out in the test.")


class ImportGUITests(unittest.TestCase):
    setUpClass = GUITests.__dict__["setUpClass"]
    setUp = GUITests.setUp
    tearDown = GUITests.tearDown
    final = GUITests.final

    def _run_import(self, labels, choice):
        original = QtWidgets.QMessageBox.addButton
        buttons = []

        def remember(box, *args):
            button = original(box, *args)
            buttons.append(button)
            return button
        with patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(labels), "")), \
                patch.object(QtWidgets.QMessageBox, "addButton", remember), \
                patch.object(QtWidgets.QMessageBox, "exec_", return_value=0), \
                patch.object(QtWidgets.QMessageBox, "clickedButton",
                             lambda box: buttons[choice] if choice is not None else None), \
                patch.object(QtWidgets.QMessageBox, "information"):
            return self.window._import_annotations("yolo")

    def test_import_yolo_append_replace_cancel_undo(self):
        self.final("cell")
        labels = Path(self.directory.name) / "frame.txt"
        labels.write_text("0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.2 0.3\n")
        (Path(self.directory.name) / "classes.txt").write_text("particle\n")
        self.assertIsNone(self._run_import(labels, None))
        self.assertEqual(len(self.window.canvas.shapes), 1)
        report = self._run_import(labels, 0)
        self.assertEqual(len(report.annotations), 2)
        imported = self.window.canvas.shapes[1:]
        self.assertEqual([s.shape_type for s in imported], ["rectangle", "polygon"])
        self.assertTrue(all(s.label == "particle" and s.flags.get("imported") for s in imported))
        out = Path(self.directory.name) / "frame.json"
        self.window.saveLabels(str(out))
        saved = json.loads(out.read_text())["shapes"]
        self.assertEqual(sum(s.get("source") == "import_yolo" for s in saved), 2)
        self._run_import(labels, 1)
        self.assertEqual([s.label for s in self.window.canvas.shapes], ["particle"] * 2)
        self.window.undoShapeEdit()
        self.assertEqual(len(self.window.canvas.shapes), 3)
