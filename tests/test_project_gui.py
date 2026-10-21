"""Project workflows with actual LabelMe/Qt objects and no source sidecars."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from PIL import Image
from qtpy import QtCore, QtWidgets

import test_gui as gui_helpers
from project_format.classes import rename_class, remove_class
from project_format.models import ClassDefinition
from project_format.classes import add_class
from project_format.io import load_project


class ProjectGUITests(unittest.TestCase):
    setUpClass = classmethod(gui_helpers.GUITests.setUpClass.__func__)
    setUp = gui_helpers.GUITests.setUp
    tearDown = gui_helpers.GUITests.tearDown
    final = gui_helpers.GUITests.final
    prompt = gui_helpers.GUITests.prompt

    def path(self):
        return Path(self.directory.name) / "dataset.medphys.json"

    def test_new_project_roundtrip_and_exclusion_of_temporary_state(self):
        final = self.final()
        self.prompt("circle")
        self.window._add_candidate_polygon_to_canvas([[35, 35], [50, 35], [50, 50]], score=.9)
        self.window.create_project(self.path())
        project = load_project(self.path())
        self.assertEqual(len(project.images[0].annotations), 1)
        self.assertEqual(project.images[0].annotations[0].id, final.other_data["object_id"])
        self.assertEqual(len(self.window.canvas.shapes), 1)
        self.assertFalse(self.image_path.with_suffix(".json").exists())
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.assertTrue(self.window.open_project(self.path()))
        self.assertEqual(self.window.canvas.shapes[0].label, "cell")

    def test_two_frames_save_before_navigation_and_do_not_read_old_sidecar(self):
        self.final()
        second = Path(self.directory.name) / "second.png"
        Image.new("RGB", (96, 96), "white").save(second)
        second.with_suffix(".json").write_text("not a LabelMe file")
        self.window.create_project(self.path(), [self.image_path, second])
        first_id = self.window.canvas.shapes[0].other_data["object_id"]
        self.window.canvas.shapes[0].points[0] = QtCore.QPointF(11, 11)
        self.window.setDirty()
        self.assertTrue(self.window.loadFile(str(second)))
        self.assertEqual(self.window.canvas.shapes, [])
        self.final(offset=35)
        self.assertTrue(self.window.loadFile(str(self.image_path)))
        self.assertEqual(self.window.canvas.shapes[0].other_data["object_id"], first_id)
        self.assertEqual(self.window.canvas.shapes[0].points[0].x(), 11)
        self.assertEqual([len(i.annotations) for i in load_project(self.path()).images], [1, 1])
        self.assertEqual(second.with_suffix(".json").read_text(), "not a LabelMe file")

    def test_same_frame_click_does_not_discard_edits(self):
        self.final()
        self.window.create_project(self.path())
        self.window.canvas.shapes[0].points[0] = QtCore.QPointF(12, 12)
        self.window.setDirty()
        self.assertTrue(self.window.loadFile(str(self.image_path)))
        self.assertEqual(self.window.canvas.shapes[0].points[0].x(), 12)

    def test_class_rename_recolors_exports_and_project_undo_redo(self):
        self.final()
        self.window.create_project(self.path())
        original_id = self.window._project.classes[0].id
        self.window._capture_project_image()
        result = rename_class(self.window._project, original_id, "particle")
        result.classes[0].color = "#123456"
        self.window._apply_project_change(result)
        shape = self.window.canvas.shapes[0]
        self.assertEqual(shape.label, "particle")
        self.assertEqual(shape.line_color.name(), "#123456")
        self.assertEqual(self.window._export_snapshot()[3], ["particle"])
        self.window._undo_project_change()
        self.assertEqual(self.window.canvas.shapes[0].label, "cell")
        self.window._redo_project_change()
        self.assertEqual(self.window.canvas.shapes[0].label, "particle")

    def test_deactivated_class_preserves_old_objects_and_prevents_new_assignment(self):
        self.final()
        self.window.create_project(self.path())
        definition = self.window._project.classes[0]
        self.window._capture_project_image()
        result = remove_class(self.window._project, definition.id, "deactivate")
        self.window._apply_project_change(result)
        self.assertEqual(self.window.canvas.shapes[0].label, "cell")
        self.assertFalse(self.window.validateLabel("cell"))
        self.assertTrue(self.window._save_project())
        self.assertEqual(len(load_project(self.path()).images[0].annotations), 1)

    def test_project_autosave_excludes_tokens_and_makes_backups(self):
        self.final()
        self.window.create_project(self.path())
        shape = self.window.canvas.shapes[0]
        shape.other_data["refinement_token"] = "expired-worker-state"
        shape.description = "updated"
        self.window.setDirty()
        self.window._project_autosave()
        self.window._finish_pending_autosave()
        project = load_project(self.path())
        self.assertEqual(project.images[0].annotations[0].description, "updated")
        self.assertNotIn("refinement_token", project.images[0].annotations[0].metadata)
        self.assertTrue(list(Path(self.directory.name).glob(".*.backups/*.json")))
        before = len(self.window._project.journal)
        self.window._capture_project_image()
        self.assertEqual(len(self.window._project.journal), before)

    def test_save_failure_prevents_navigation_and_close(self):
        self.final()
        second = Path(self.directory.name) / "second.png"
        Image.new("RGB", (96, 96), "white").save(second)
        self.window.create_project(self.path(), [self.image_path, second])
        self.window.canvas.shapes[0].description = "must survive"
        self.window.setDirty()
        with patch("app.gui.project_window.save_project", side_effect=OSError("disk full")), \
                patch.object(QtWidgets.QMessageBox, "warning"):
            self.assertFalse(self.window.loadFile(str(second)))
            with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
                self.assertFalse(self.window.mayContinue())
        self.assertEqual(self.window.canvas.shapes[0].description, "must survive")
        self.assertEqual(Path(self.window.imagePath).resolve(), self.image_path.resolve())

    def test_classes_dialog_edits_registry_without_invalid_migrations(self):
        self.final()
        self.window.create_project(self.path())
        from app.gui.project_classes_dialog import ProjectClassesDialog
        dialog = ProjectClassesDialog(self.window)
        dialog.name.setText("particle")
        dialog.color.setText("#aabbcc")
        dialog._apply()
        self.assertEqual(self.window.canvas.shapes[0].label, "particle")
        self.assertEqual(self.window._project.classes[0].color, "#aabbcc")
        dialog.deleteLater()

    def test_close_project_restores_standalone_mode(self):
        self.final()
        self.window.create_project(self.path())
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.window._close_project()
        self.assertIsNone(self.window._project)
        self.window.loadFile(str(self.image_path))
        self.assertEqual(self.window.canvas.shapes, [])
        self.final()
        target = Path(self.directory.name) / "standalone.json"
        self.assertTrue(self.window.saveLabels(str(target)))
        self.assertEqual(json.loads(target.read_text())["shapes"][0]["label"], "cell")

    def test_class_hotkey_and_reserved_shortcut_are_transactional(self):
        self.final()
        self.window.create_project(self.path())
        result = add_class(self.window._project, ClassDefinition(
            name="particle", shortcut="Ctrl+Alt+8", color="#00aaff"))
        self.window._apply_project_change(result)
        self.window.shapeSelectionChanged([self.window.canvas.shapes[0]])
        self.window._class_hotkey("particle")
        self.assertEqual(self.window.canvas.shapes[0].label, "particle")
        self.window._capture_project_image()
        invalid = self.window._project.model_copy(deep=True)
        invalid.classes[0].shortcut = self.window.actions.save.shortcut().toString()
        with self.assertRaises(ValueError):
            self.window._apply_project_change(invalid)
        self.assertEqual(self.window._project.classes[0].shortcut, "")

    def test_empty_project_and_adding_images(self):
        self.window.resetState()
        self.window.create_project(self.path())
        self.assertEqual(self.window._project.images, [])
        self.window._add_project_images([self.image_path])
        self.assertEqual(Path(self.window.imagePath).resolve(), self.image_path.resolve())
        self.assertTrue(self.window._save_project())
        self.assertEqual(len(load_project(self.path()).images), 1)

    def test_display_and_generation_settings_roundtrip(self):
        self.final()
        self.window.create_project(self.path())
        self.window.setZoom(200)
        self.window.brightnessContrast_values[self.window.filename] = (110, 120)
        from app.worker_api.schemas import GenerationOptions
        self.window._generation_options = GenerationOptions(points_per_side=7)
        self.assertTrue(self.window._save_project())
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.assertTrue(self.window.open_project(self.path()))
        self.assertEqual(self.window.zoomWidget.value(), 200)
        self.assertEqual(self.window._generation_options.points_per_side, 7)
        self.assertEqual(self.window.brightnessContrast_values[self.window.filename], (110, 120))

    def test_class_rename_keeps_live_refinement_token_but_never_serializes_it(self):
        shape = self.final()
        shape.other_data["refinement_token"] = "live-token"
        self.window.create_project(self.path())
        self.window._capture_project_image()
        definition = self.window._project.classes[0]
        self.window._apply_project_change(rename_class(self.window._project, definition.id, "particle"))
        self.assertEqual(self.window.canvas.shapes[0].other_data["refinement_token"], "live-token")
        self.assertTrue(self.window._save_project())
        self.assertNotIn("live-token", self.path().read_text())

    def test_save_as_rebases_paths_and_resets_old_rooted_history(self):
        self.final()
        self.window.create_project(self.path())
        self.window._capture_project_image()
        definition = self.window._project.classes[0]
        self.window._apply_project_change(rename_class(self.window._project, definition.id, "particle"))
        target = Path(self.directory.name) / "nested/copy.medphys.json"
        with patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(target), "")):
            self.window._save_project_as()
        self.assertEqual(self.window._project_path, target.resolve())
        self.assertFalse(self.window._project_history.can_undo)
        from project_format.io import resolve_image
        self.assertEqual(resolve_image(target, load_project(target).images[0]), self.image_path.resolve())

    def test_new_geometry_after_project_undo_invalidates_redo(self):
        self.final()
        self.window.create_project(self.path())
        self.window._capture_project_image()
        definition = self.window._project.classes[0]
        self.window._apply_project_change(rename_class(self.window._project, definition.id, "particle"))
        self.window._undo_project_change()
        self.window.canvas.shapes[0].points[0] = QtCore.QPointF(12, 12)
        self.window.setDirty()
        self.window._redo_project_change()
        self.assertEqual(self.window.canvas.shapes[0].label, "cell")
        self.assertEqual(self.window.canvas.shapes[0].points[0].x(), 12)
        self.assertFalse(self.window._project_history.can_redo)

    def test_project_undo_keeps_persistent_event_journal(self):
        self.final()
        self.window.create_project(self.path())
        self.window._capture_project_image()
        definition = self.window._project.classes[0]
        self.window._apply_project_change(rename_class(self.window._project, definition.id, "particle"))
        before = len(self.window._project.journal)
        self.window._undo_project_change()
        self.assertGreater(len(self.window._project.journal), before)
        self.assertEqual(self.window._project.journal[-1]["action"], "project_state_restored")


class BackgroundIOTests(unittest.TestCase):
    setUpClass = classmethod(gui_helpers.GUITests.setUpClass.__func__)
    setUp = gui_helpers.GUITests.setUp
    tearDown = gui_helpers.GUITests.tearDown
    final = gui_helpers.GUITests.final

    def path(self):
        return Path(self.directory.name) / "bg.medphys.json"

    def drain(self):
        for job in list(self.window._io_jobs_list()):
            job.wait(10000)
        for _ in range(5):
            QtWidgets.QApplication.processEvents()

    def test_autosave_runs_in_background_and_keeps_later_edits_dirty(self):
        self.final()
        self.window.create_project(self.path())
        self.window.setDirty()
        self.window._project_dirty = True
        self.window._autosave_async()
        self.final(offset=35)          # edit while the save is in flight
        self.window.setDirty()
        self.drain()
        self.assertIsNone(self.window._autosave_job)
        self.assertTrue(self.window.dirty or self.window._project_dirty)
        self.assertTrue(self.window._save_project())
        self.assertEqual(len(load_project(self.path()).images[0].annotations), 2)

    def test_manual_save_waits_for_running_autosave(self):
        self.final()
        self.window.create_project(self.path())
        self.window._project_dirty = True
        self.window._autosave_async()
        self.assertTrue(self.window._save_project())     # no stale-hash conflict
        self.assertIsNone(self.window._autosave_job)

    def test_open_project_async_verifies_and_attaches(self):
        self.final()
        self.window.create_project(self.path())
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Discard):
            self.window._close_project()
        self.assertIsNone(self.window._project)
        job = self.window._open_project_async(self.path())
        self.assertIsNotNone(job)
        self.drain()
        self.assertIsNotNone(self.window._project)
        self.assertEqual(len(self.window.canvas.shapes), 1)

    def test_dataset_export_in_background_and_cancel(self):
        from app.gui.io_jobs import OperationCancelled
        self.final()
        self.window.create_project(self.path())
        out = Path(self.directory.name) / "datasets"
        with patch.object(QtWidgets.QDialog, "exec_", return_value=QtWidgets.QDialog.Accepted), \
                patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(out)), \
                patch.object(QtWidgets.QMessageBox, "information") as info:
            job = self.window._export_dataset_dialog()
            self.drain()
        self.assertIsNone(job.error)
        self.assertTrue(info.called)
        self.assertEqual(len([p for p in out.iterdir() if not p.name.startswith(".")]), 1)
        job = self.window._run_io("Cancelled", lambda progress, cancelled: (_ for _ in ()).throw(
            OperationCancelled()), lambda job: self.fail("on_done must not run"))
        self.drain()
        self.assertEqual(self.window._io_jobs_list(), [])


class PropagationGUITests(unittest.TestCase):
    setUpClass = classmethod(gui_helpers.GUITests.setUpClass.__func__)
    setUp = gui_helpers.GUITests.setUp
    tearDown = gui_helpers.GUITests.tearDown
    final = gui_helpers.GUITests.final

    def test_proposals_written_and_manual_work_preserved(self):
        from app.worker_api.schemas import PropagationResponse
        from worker.postprocess.masks import mask_rle
        import numpy as np
        frames = [self.image_path]
        for i in range(2):
            path = Path(self.directory.name) / f"next{i}.png"
            Image.new("RGB", (96, 96), "white").save(path)
            frames.append(path)
        source = self.final()
        path = Path(self.directory.name) / "track.medphys.json"
        self.window.create_project(path, frames)
        track = self.window.canvas.shapes[0].other_data["object_id"]
        mask = np.zeros((96, 96), bool); mask[20:40, 20:40] = True
        polygon = [[20.0, 20.0], [39.0, 20.0], [39.0, 39.0], [20.0, 39.0]]
        entry = {"track_id": track, "label": "cell", "polygon": polygon,
                 "mask_rle": mask_rle(mask), "area": 400}
        response = PropagationResponse(ok=True, frames=[
            {"index": 1, "path": str(frames[1]), "objects": [entry]},
            {"index": 2, "path": str(frames[2]), "objects": [entry]}], lost=[], elapsed_ms=10)
        seen = {}

        def fake(request, timeout=900.0):
            seen["request"] = request
            return response
        with patch.object(self.window.worker_client, "propagate", side_effect=fake), \
                patch.object(QtWidgets.QDialog, "exec_", return_value=QtWidgets.QDialog.Accepted), \
                patch.object(QtWidgets.QMessageBox, "information"):
            self.window._propagate_dialog()
            for job in list(self.window._io_jobs_list()):
                job.wait(10000)
            QtWidgets.QApplication.processEvents()
        self.assertEqual(seen["request"].start_index, 0)
        self.assertEqual(len(seen["request"].frame_paths), 3)
        images = self.window._project.images
        added = images[1].annotations[0]
        self.assertEqual((added.source, added.review_status), ("sam2_propagation", "unreviewed"))
        self.assertEqual(added.metadata["track_id"], track)
        # mark frame 2 proposal reviewed; a second run must not overwrite it
        images[2].annotations[0].review_status = "accepted"
        with patch.object(QtWidgets.QMessageBox, "information"):
            added_count, kept = self.window._apply_propagation(
                response, [i.id for i in images], images[0].id)
        self.assertEqual((added_count, kept), (1, 1))
        images = self.window._project.images
        self.assertEqual(len(images[1].annotations), 1)
        self.assertEqual(images[2].annotations[0].review_status, "accepted")


class DisplayPersistenceTests(unittest.TestCase):
    setUpClass = classmethod(gui_helpers.GUITests.setUpClass.__func__)
    setUp = gui_helpers.GUITests.setUp
    tearDown = gui_helpers.GUITests.tearDown

    def test_display_settings_saved_and_restored(self):
        import numpy as np
        from app.gui import labelme_window as lw
        grey = Path(self.directory.name) / "g16.tif"
        Image.fromarray(np.linspace(0, 4000, 96 * 96).reshape(96, 96).astype(np.uint16)).save(grey)
        path = Path(self.directory.name) / "disp.medphys.json"
        self.window.create_project(path, [grey, self.image_path])
        self.window._apply_display({"mode": "manual", "low": 0, "high": 1000})
        self.assertTrue(self.window._save_project())
        stored = load_project(path).images[0].display["scientific"]
        self.assertEqual((stored["mode"], stored["high"]), ("manual", 1000))
        self.assertNotIn("scientific", load_project(path).images[1].display)
        lw._DISPLAY_SETTINGS.clear()           # simulate an application restart
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Save):
            self.assertTrue(self.window.open_project(path))
        self.assertEqual(self.window._current_display()["high"], 1000)
        from PyQt5 import QtGui
        self.assertEqual(QtGui.QColor(self.window.image.pixel(95, 95)).red(), 255)
