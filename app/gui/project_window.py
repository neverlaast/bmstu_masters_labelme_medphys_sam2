"""Project lifecycle integrated without modifying vendored LabelMe."""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
import os
from pathlib import Path

from app.gui.io_jobs import IOJob, OperationCancelled
from qtpy import QtCore, QtGui, QtWidgets
from labelme.label_file import LabelFile
from labelme.widgets import BrightnessContrastDialog
from labelme import utils

from app.core.export import snapshot_shapes
from worker.postprocess.masks import mask_rle
from app.core.history import SnapshotHistory
from app.worker_api.schemas import GenerationOptions
from project_format.io import (
    file_sha256, image_record, load_project, rebase_paths, relocate_images,
    resolve_image, resource_issues, save_project,
)
from project_format.models import Annotation, ClassDefinition, Project, annotation_payload, now


class ProjectWindowMixin:
    def _record_project_model(self, model_info):
        if self._project is None or not model_info:
            return
        record = dict(model_info)
        if record.get("checkpoint_path"):
            record["checkpoint_path"] = os.path.relpath(record["checkpoint_path"], self._project_path.parent)
        key = (record.get("backend"), record.get("model_name"), record.get("model_version"))
        existing = {(r.get("backend"), r.get("model_name"), r.get("model_version"))
                    for r in self._project.model_records}
        if key not in existing:
            record["added_at"] = now()
            self._project.model_records.append(record)
            self._project_dirty = True

    def _init_project_state(self):
        self._project = self._project_path = self._project_hash = self._project_current = None
        self._project_dirty = self._project_loading = False
        self._project_saved_config = self._project_history = None
        self._class_shortcuts = []
        self._project_tokens = OrderedDict()

    def _validate_class_shortcuts(self, project):
        reserved = {sequence.toString() for action in self.findChildren(QtWidgets.QAction)
                    for sequence in action.shortcuts() if not sequence.isEmpty()}
        used = set()
        for definition in project.classes:
            if not definition.active or not definition.shortcut:
                continue
            sequence = QtGui.QKeySequence(definition.shortcut)
            key = sequence.toString()
            if sequence.isEmpty() or not key or key in reserved or key in used:
                raise ValueError(f"Invalid, duplicated or reserved class shortcut: {definition.shortcut}")
            used.add(key)

    def _clear_class_shortcuts(self):
        for shortcut in self._class_shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._class_shortcuts = []

    def _class_hotkey(self, name):
        if not self._resolve_property_draft():
            return
        shapes = self._get_selected_final_shapes()
        if not shapes:
            self.statusBar().showMessage("Select final objects to assign this class", 3000)
            return
        definition = next(c for c in self._project.classes if c.name == name and c.active)
        if any(s.shape_type not in definition.allowed_geometry for s in shapes):
            QtWidgets.QMessageBox.warning(self, "Class assignment", "Some selected geometries are not allowed for this class.")
            return
        for shape in shapes:
            shape.label = name
            shape.other_data["class_id"] = definition.id
        self._refresh_views()
        self.setDirty()

    def _install_project_menu(self):
        menu = self.menuBar().addMenu("Project")
        for text, slot in [
            ("New project from current image...", self._new_project_dialog),
            ("Open project...", self._open_project_dialog),
            ("Save project", self._save_project),
            ("Save project as...", self._save_project_as),
            ("Add images...", self._add_project_images_dialog),
            ("Import video...", self._import_video_dialog),
            ("Project classes...", self._project_classes_dialog),
            ("Export dataset (train/val/test)...", self._export_dataset_dialog),
            ("Autosave interval...", self._project_autosave_interval),
            ("Undo project/class change", self._undo_project_change),
            ("Redo project/class change", self._redo_project_change),
            ("Close project", self._close_project),
        ]:
            self._action(menu, text, slot)
        self._project_timer = QtCore.QTimer(self)
        self._project_timer.timeout.connect(self._project_autosave)

    def _guard_project_operation(self):
        if not self._resolve_property_draft():
            raise ValueError("Object property editing was cancelled.")
        if self._job is not None:
            raise ValueError("Wait for the model request to finish before changing the project")
        if getattr(self, "_exclusive_io", None) is not None:
            raise ValueError(f"Wait for '{self._exclusive_io.title}' to finish")

    # ----------------------------------------------------------- background I/O
    def _io_jobs_list(self):
        if not hasattr(self, "_io_jobs"):
            self._io_jobs = []
        return self._io_jobs

    def _run_io(self, title, operation, on_done, *, modal=True, exclusive=False):
        """Run file I/O in a thread; ``on_done(job)`` runs once on the GUI thread."""
        if exclusive and getattr(self, "_exclusive_io", None) is not None:
            raise ValueError(f"Wait for '{self._exclusive_io.title}' to finish")
        job = IOJob(title, operation, self)
        self._io_jobs_list().append(job)
        if exclusive:
            self._exclusive_io = job
        dialog = None
        if modal:
            dialog = QtWidgets.QProgressDialog(title, "Cancel", 0, 0, self)
            dialog.setWindowTitle(title)
            dialog.setWindowModality(QtCore.Qt.WindowModal)
            dialog.setMinimumDuration(400)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            dialog.canceled.connect(job.cancel)

            def update(done, total, text):
                dialog.setMaximum(max(total, 1))
                dialog.setValue(min(done, max(total, 1)))
                dialog.setLabelText(text or title)
            job.progressed.connect(update)
        else:
            job.progressed.connect(lambda d, t, text: self.statusBar().showMessage(text, 2000))

        def finished():
            try:
                if dialog is not None:
                    # QProgressDialog.closeEvent emits canceled(); detach first so a
                    # finished job is not mistaken for a cancelled one.
                    dialog.canceled.disconnect(job.cancel)
                    dialog.close()
                    dialog.deleteLater()
                self._finish_io(job, on_done)
            except RuntimeError:   # window already destroyed; nothing to update
                pass
        job.finished.connect(finished)
        job.start()
        return job

    def _finish_io(self, job, on_done):
        if job.handled:
            return
        job.handled = True
        if getattr(self, "_exclusive_io", None) is job:
            self._exclusive_io = None
        jobs = self._io_jobs_list()
        if job in jobs:
            jobs.remove(job)
        if isinstance(job.error, OperationCancelled):
            self.statusBar().showMessage(f"{job.title}: cancelled", 5000)
            return
        on_done(job)
        job.deleteLater()

    def _wait_io(self, job, on_done):
        """Finish a running job synchronously (e.g. before a dependent save)."""
        job.wait()
        self._finish_io(job, on_done)

    def _io_busy(self):
        return any(job.isRunning() for job in self._io_jobs_list())

    def _sync_project_classes(self):
        self._clear_class_shortcuts()
        active = [c.name for c in self._project.classes if c.active]
        self._config.update(labels=active, validate_label="exact", auto_save=False, keep_prev=False)
        self.labelDialog.labelList.clear()
        self.labelDialog.labelList.addItems(active)
        self.uniqLabelList.clear()
        for definition in self._project.classes:
            item = self.uniqLabelList.createItemFromLabel(definition.name)
            self.uniqLabelList.addItem(item)
            color = QtGui.QColor(definition.color)
            self.uniqLabelList.setItemLabel(item, definition.name, (color.red(), color.green(), color.blue()))
            if definition.active and definition.shortcut:
                shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(definition.shortcut), self)
                shortcut.activated.connect(lambda name=definition.name: self._class_hotkey(name))
                self._class_shortcuts.append(shortcut)

    def _set_project_image_list(self):
        blocker = QtCore.QSignalBlocker(self.fileListWidget)
        paths = [str(resolve_image(self._project_path, image)) for image in self._project.images]
        self.fileListWidget.clear()
        self.fileListWidget.addItems(paths)
        index = next((i for i, image in enumerate(self._project.images)
                      if image.id == self._project_current), -1)
        self.fileListWidget.setCurrentRow(index)
        del blocker
        self.actions.openNextImg.setEnabled(bool(self.imageList))
        self.actions.openPrevImg.setEnabled(bool(self.imageList))

    def _annotations_from_canvas(self, project):
        classes, annotations = {c.name: c for c in project.classes}, []
        for payload in snapshot_shapes(self.canvas.shapes, str(self.imagePath)):
            token = payload.get("refinement_token")
            if token and not payload.get("geometry_modified"):
                self._project_tokens[payload["object_id"]] = (
                    str(Path(self.imagePath).resolve()), deepcopy(payload["points"]), token)
                self._project_tokens.move_to_end(payload["object_id"])
                while len(self._project_tokens) > 128:
                    self._project_tokens.popitem(last=False)
            else:
                self._project_tokens.pop(payload["object_id"], None)
            definition = classes.get(payload["label"])
            if definition is None:
                raise ValueError(f"Unknown project class: {payload['label']}. Add it in Project classes.")
            fields = {"object_id", "class_id", "label", "points", "shape_type", "group_id",
                      "description", "flags", "attributes", "modified_at", "review_status", "source"}
            metadata = {k: deepcopy(v) for k, v in payload.items()
                        if k not in fields and k not in {"refinement_token", "refines_object_id"}}
            annotations.append(Annotation(
                id=payload["object_id"], class_id=definition.id,
                shape_type=payload["shape_type"], points=payload["points"],
                group_id=payload.get("group_id"), description=payload.get("description") or "",
                flags=payload.get("flags") or {}, attributes=payload.get("attributes") or {},
                modified_at=payload.get("modified_at") or now(),
                review_status=payload.get("review_status", "unreviewed"),
                source=payload.get("source", "manual"), metadata=metadata))
        return annotations

    def _capture_project_image(self):
        if self._project is None or self._project_current is None or self._project_loading:
            return
        image = next(image for image in self._project.images if image.id == self._project_current)
        annotations = self._annotations_from_canvas(self._project)
        before, after = [a.model_dump(mode="json") for a in image.annotations], [a.model_dump(mode="json") for a in annotations]
        if before != after:
            old = {a["id"]: a for a in before}
            for annotation in annotations:
                if old.get(annotation.id) != annotation.model_dump(mode="json"):
                    annotation.modified_at = now()
                    shape = next((s for s in self.canvas.shapes
                                  if (s.other_data or {}).get("object_id") == annotation.id), None)
                    if shape is not None:
                        shape.other_data["modified_at"] = annotation.modified_at
            self._project.journal.append({
                "time": now(), "action": "image_annotations_changed",
                "image_id": image.id, "objects_before": len(before), "objects_after": len(after)})
            self._project_dirty = True
        image.annotations = annotations
        image.flags = {self.flag_widget.item(i).text():
                       self.flag_widget.item(i).checkState() == QtCore.Qt.Checked
                       for i in range(self.flag_widget.count())}
        display = {"zoom": self.zoomWidget.value(),
                   "brightness_contrast": list(self.brightnessContrast_values.get(self.filename, (None, None)))}
        from app.gui import labelme_window as lw
        scientific = lw._DISPLAY_SETTINGS.get(lw._display_key(self.imagePath)) if self.imagePath else None
        if scientific is not None:
            display["scientific"] = dict(scientific)
        settings = {"generation": self._generation_options.model_dump(),
                    "worker_url": self.worker_client.base_url}
        if image.display != display or any(self._project.sam_settings.get(k) != v for k, v in settings.items()):
            self._project_dirty = True
        image.display = display
        self._project.sam_settings.update(settings)
        if self._project_history is not None:
            # Local geometry edits invalidate project redo just as they do canvas redo.
            if self._project_history.can_redo and before != after:
                self._project_history.commit(self._project)

    def _attach_project(self, project, path, saved_hash):
        self._validate_class_shortcuts(project)
        if project.sam_settings.get("generation"):
            GenerationOptions.model_validate(project.sam_settings["generation"])
        if self._project_saved_config is None:
            self._project_saved_config = deepcopy(self._config)
        self._project, self._project_path, self._project_hash = project, Path(path).resolve(), saved_hash
        self._project_current, self._project_dirty = None, False
        self._sync_project_classes()
        if project.sam_settings.get("generation"):
            self._generation_options = GenerationOptions.model_validate(project.sam_settings["generation"])
        self._project_history = SnapshotHistory(
            clone=deepcopy,
            key=lambda p: p.model_dump(mode="json", exclude={"journal", "modified_at"})
            if isinstance(p, Project) else p, limit=30)
        self._project_history.reset(project)
        self._set_project_image_list()
        self._project_timer.start(project.autosave_seconds * 1000)
        if project.images:
            self._load_project_image(str(resolve_image(path, project.images[0])), capture=False)
        else:
            self.resetState()
            self.canvas.setEnabled(False)
        self.statusBar().showMessage(f"Project: {project.name}", 6000)

    def create_project(self, path, image_paths=None):
        self._guard_project_operation()
        if self._project is not None:
            raise ValueError("Close the current project before creating another")
        path = Path(path).resolve()
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in
                                  (image_paths or ([self.imagePath] if self.imagePath else []))))
        project = Project(name=path.stem, images=[image_record(p, path) for p in paths])
        labels = list(dict.fromkeys([*(self._config.get("labels") or []),
                                    *[s.label for s in self.canvas.shapes if not self._is_temporary(s)]]))
        project.classes = [ClassDefinition(name=label) for label in labels]
        for image in project.images:
            if self.imagePath and resolve_image(path, image) == Path(self.imagePath).resolve():
                image.annotations = self._annotations_from_canvas(project)
        project.sam_settings = {"generation": self._generation_options.model_dump(),
                                "worker_url": self.worker_client.base_url}
        project.journal.append({"time": now(), "action": "project_created"})
        saved_hash = save_project(path, project, require_new=True)
        self._attach_project(project, path, saved_hash)
        return project

    def open_project(self, path, relocate_directory=None):
        self._guard_project_operation()
        project = load_project(path, check_resources=False)
        if relocate_directory:
            project = relocate_images(project, path, relocate_directory)
        issues = resource_issues(path, project)
        if issues:
            raise ValueError("\n".join(issues))
        if not self.mayContinue():
            return False
        self._attach_project(project, path, file_sha256(path))
        if relocate_directory:
            self._project_dirty = True
        return True

    def _restore_scientific_display(self, path, record):
        """Project-stored 16-bit/multichannel display mapping -> renderer registry."""
        from app.gui import labelme_window as lw
        from worker.imaging import normalise_display
        key, stored = lw._display_key(path), record.display.get("scientific")
        if not stored:
            lw._DISPLAY_SETTINGS.pop(key, None)
            return
        try:
            lw._DISPLAY_SETTINGS[key] = normalise_display(stored)
        except (TypeError, ValueError) as exc:
            lw._DISPLAY_SETTINGS.pop(key, None)
            self.statusBar().showMessage(f"Stored display settings ignored: {exc}", 8000)

    def _load_project_image(self, filename, capture=True):
        if not self._resolve_property_draft():
            return False
        path = Path(filename).resolve()
        target = next((image for image in self._project.images
                       if resolve_image(self._project_path, image) == path), None)
        if target is None:
            raise ValueError("Image is not in this project. Use Project > Add images.")
        if capture and target.id == self._project_current:
            return True
        if file_sha256(path) != target.sha256:
            raise ValueError("Image checksum changed; refusing to reuse old coordinates")
        self._restore_scientific_display(path, target)
        image_data = LabelFile.load_image_file(str(path))
        image = QtGui.QImage.fromData(image_data)
        if image.isNull() or (image.width(), image.height()) != (target.width, target.height):
            raise ValueError("Image dimensions/decoder do not match this project")
        if capture and self._project_current is not None and target.id != self._project_current:
            if not self._save_project():
                return False
        self._project_loading = True
        try:
            self.resetState()
            self._project_current = target.id
            self.imagePath = self.filename = str(path)
            self.imageData, self.image, self.labelFile = image_data, image, None
            self.otherData = {}
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
            definitions = {c.id: c for c in self._project.classes}
            payloads = []
            for annotation in target.annotations:
                payload = annotation_payload(annotation, definitions[annotation.class_id])
                transient = self._project_tokens.get(annotation.id)
                if transient and transient[0] == str(path) and transient[1] == payload["points"]:
                    payload["refinement_token"] = transient[2]
                payload["other_data"] = {k: deepcopy(v) for k, v in payload.items()
                                         if k not in {"label", "points", "shape_type", "group_id", "description", "flags"}}
                payloads.append(payload)
            self.loadLabels(payloads)
            self.loadFlags(target.flags)
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            if target.display.get("zoom"):
                self.setZoom(target.display["zoom"])
            values = target.display.get("brightness_contrast", [None, None])
            self.brightnessContrast_values[self.filename] = tuple(values)
            if any(v is not None for v in values):
                dialog = BrightnessContrastDialog(
                    utils.img_data_to_pil(self.imageData), self.onNewBrightnessContrast, parent=self)
                if values[0] is not None:
                    dialog.slider_brightness.setValue(values[0])
                if values[1] is not None:
                    dialog.slider_contrast.setValue(values[1])
                dialog.onNewValue(None)
                dialog.deleteLater()
            self.paintCanvas()
            self.toggleActions(True)
            self.setClean()
            self._history.reset(self.canvas.shapes)
            self._reset_object_edit_tracking()
            self._refresh_views()
            self._set_project_image_list()
            self.canvas.setFocus()
        finally:
            self._project_loading = False
        return True

    def _save_project(self, checked=False):
        if not self._resolve_property_draft():
            return False
        if self._project is None:
            return False
        self._finish_pending_autosave()
        if self._project is None:
            return False
        try:
            self._capture_project_image()
            self._project_hash = save_project(
                self._project_path, self._project, expected_sha256=self._project_hash)
            self._project_dirty = False
            self.setClean()
            self._project_timer.start(self._project.autosave_seconds * 1000)
            self.statusBar().showMessage(f"Project saved: {self._project_path.name}", 4000)
            return True
        except (OSError, ValueError, TypeError) as exc:
            self._project_dirty = self.dirty = True
            self.actions.save.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Project save failed", str(exc)[:8000])
            return False

    def _new_project_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "New project", "", "MedPhys project (*.medphys.json)")
        if path:
            self._project_safe(lambda: self.create_project(path))

    def _open_project_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "MedPhys project (*.medphys.json)")
        if not path:
            return
        return self._open_project_async(path)

    def _open_project_async(self, path, directory=None):
        """Load + verify all image checksums off the GUI thread."""
        try:
            self._guard_project_operation()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Project operation failed", str(exc))
            return None

        def operation(progress, cancelled):
            progress(0, 1, "Reading project...")
            project = load_project(path, check_resources=False)
            if directory:
                project = relocate_images(project, path, directory)
            issues = resource_issues(path, project, progress=progress)
            return project, issues, file_sha256(path)

        def done(job):
            if job.error is not None:
                QtWidgets.QMessageBox.warning(self, "Project operation failed", str(job.error)[:8000])
                return
            project, issues, saved_hash = job.result
            if issues:
                if directory:
                    QtWidgets.QMessageBox.warning(self, "Project resources", "\n".join(issues[:40]))
                    return
                answer = QtWidgets.QMessageBox.question(
                    self, "Project resources", "\n".join(issues[:20]) + "\nChoose a new image folder?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if answer != QtWidgets.QMessageBox.Yes:
                    return
                folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Locate project images")
                if folder:
                    self._open_project_async(path, folder)
                return

            def attach():
                self._guard_project_operation()
                if self._project is not None:
                    raise ValueError("Close the current project before opening another")
                if not self.mayContinue():
                    return False
                self._attach_project(project, path, saved_hash)
                if directory:
                    self._project_dirty = True
                return True
            self._project_safe(attach)
        return self._run_io("Open project", operation, done, exclusive=True)

    def _project_safe(self, operation):
        try:
            return operation()
        except (OSError, ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Project operation failed", str(exc)[:8000])
            return False

    def _save_project_as(self):
        if self._project is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project as", str(self._project_path), "MedPhys project (*.medphys.json)")
        if not path:
            return
        def operation():
            self._capture_project_image()
            if Path(path).resolve() == self._project_path:
                return self._save_project()
            rebased = rebase_paths(self._project, self._project_path, path)
            saved_hash = save_project(path, rebased, require_new=True)
            self._project = rebased
            self._project_path, self._project_hash = Path(path).resolve(), saved_hash
            # Snapshots contain relative paths; don't replay snapshots rooted
            # at the old project directory after Save As.
            self._project_history.reset(self._project)
            self._project_dirty = False
            self.setClean()
            self._set_project_image_list()
        self._project_safe(operation)

    def _add_project_images(self, paths):
        if self._project is None:
            raise ValueError("Open or create a project first")
        self._guard_project_operation()
        self._capture_project_image()
        result = self._project.model_copy(deep=True)
        existing = {resolve_image(self._project_path, image) for image in result.images}
        for path in paths:
            if Path(path).resolve() not in existing:
                result.images.append(image_record(path, self._project_path))
                existing.add(Path(path).resolve())
        result.journal.append({"time": now(), "action": "images_added",
                               "count": len(result.images)-len(self._project.images)})
        self._apply_project_change(result)
        self._set_project_image_list()
        if self._project_current is None and result.images:
            self._load_project_image(str(resolve_image(self._project_path, result.images[0])), capture=False)

    def _add_project_images_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add project images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if paths:
            self._project_safe(lambda: self._add_project_images(paths))

    # ------------------------------------------------------------------ video
    def _import_video_dialog(self, checked=False, path=None):
        from app.core.video import VIDEO_FILTER
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import video", "", VIDEO_FILTER)
        if path:
            return self._project_safe(lambda: self.import_video(path))

    def import_video(self, video_path, frame_range=None, frames_dir=None, project_path=None):
        """Decode a video into PNG frames (background) and add them in temporal order.

        Without an open project a new one is created next to the video, so the
        frames can be propagated with the SAM2 video predictor right away.
        ``frame_range``/``frames_dir`` given -> no dialog (scripts, tests).
        """
        from app.core.video import default_frames_dir, extract_frames, probe_video
        from app.gui.video_import_dialog import VideoImportDialog
        self._guard_project_operation()
        info = probe_video(video_path)
        if self._project is None and project_path is None:
            suggested = Path(video_path).with_suffix("").as_posix() + ".medphys.json"
            project_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "New project for the video", suggested, "MedPhys project (*.medphys.json)")
            if not project_path:
                return None
        target_project = project_path or self._project_path
        if frame_range is None or frames_dir is None:
            dialog = VideoImportDialog(info, Path(frames_dir or default_frames_dir(video_path, target_project)), self)
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return None
            frame_range, frames_dir = dialog.frame_range(), dialog.frames_dir()

        def operation(progress, cancelled):
            try:
                return extract_frames(video_path, frames_dir, frame_range,
                                      progress=progress, cancelled=cancelled)
            except InterruptedError as exc:
                raise OperationCancelled(str(exc)) from exc

        def done(job):
            if job.error is not None:
                QtWidgets.QMessageBox.warning(self, "Video import failed", str(job.error)[:8000])
                return
            frames = [str(p) for p in job.result]

            def attach():
                if self._project is None:
                    if not self.mayContinue():
                        return False
                    self.create_project(project_path, image_paths=frames)
                else:
                    self._add_project_images(frames)
                self._project.journal.append({
                    "time": now(), "action": "video_imported", "source": info.path,
                    "fps": info.fps, "frames": len(frames), "start": frame_range.start,
                    "end": frame_range.end, "step": frame_range.step})
                self._project_dirty = True
                self.statusBar().showMessage(
                    f"Imported {len(frames)} frame(s). Annotate a frame, then "
                    "Segmentation / SAM2 > Propagate objects to other frames.", 10000)
                return True
            self._project_safe(attach)
        return self._run_io("Import video", operation, done, exclusive=True)

    def _apply_project_change(self, project):
        self._guard_project_operation()
        validated = Project.model_validate(project.model_dump(mode="json"))
        self._validate_class_shortcuts(validated)
        self._project_history.commit(self._project)
        self._project = validated
        self._project_history.commit(self._project)
        self._project_dirty = True
        self._sync_project_classes()
        if self._project_current:
            image = next(i for i in self._project.images if i.id == self._project_current)
            self._load_project_image(str(resolve_image(self._project_path, image)), capture=False)
        self.dirty = True
        self.actions.save.setEnabled(True)

    def _undo_project_change(self):
        def operation():
            self._guard_project_operation()
            if self._project_history is None:
                return
            if self.dirty:
                self._capture_project_image()
                self._project_history.commit(self._project)
            result = self._project_history.undo()
            if result is not None:
                self._restore_project_change(result)
        self._project_safe(operation)

    def _redo_project_change(self):
        def operation():
            self._guard_project_operation()
            if self._project_history is None:
                return
            self._capture_project_image()
            result = self._project_history.redo()
            if result is not None:
                self._restore_project_change(result)
        self._project_safe(operation)

    def _restore_project_change(self, project):
        self._validate_class_shortcuts(project)
        journal = deepcopy(self._project.journal)
        journal.append({"time": now(), "action": "project_state_restored",
                        "classes": [c.name for c in project.classes], "images": len(project.images)})
        self._project = project
        self._project.journal = journal
        if self._project_current not in {i.id for i in project.images}:
            self._project_current = project.images[0].id if project.images else None
        self._sync_project_classes()
        self._set_project_image_list()
        if self._project_current:
            image = next(i for i in project.images if i.id == self._project_current)
            self._load_project_image(str(resolve_image(self._project_path, image)), capture=False)
        else:
            self.resetState()
            self.canvas.setEnabled(False)
        self._project_dirty = self.dirty = True
        self.actions.save.setEnabled(True)

    def _export_dataset_dialog(self):
        from app.core.dataset_export import DATASET_FORMATS, DatasetExportError, export_dataset
        if self._project is None:
            QtWidgets.QMessageBox.information(self, "Export dataset", "Open or create a project first.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Export dataset")
        form = QtWidgets.QFormLayout(dialog)
        kind = QtWidgets.QComboBox(); kind.addItems(list(DATASET_FORMATS))
        spins = []
        for value in (80, 10, 10):
            spin = QtWidgets.QSpinBox(); spin.setRange(0, 100); spin.setSuffix(" %"); spin.setValue(value)
            spins.append(spin)
        seed = QtWidgets.QSpinBox(); seed.setRange(0, 2**31 - 1)
        empty = QtWidgets.QCheckBox("Include images without objects"); empty.setChecked(True)
        reviewed = QtWidgets.QCheckBox("Only accepted/corrected objects")
        form.addRow("Format", kind)
        for name, spin in zip(("Train", "Validation", "Test"), spins):
            form.addRow(name, spin)
        form.addRow("Random seed", seed)
        form.addRow(empty)
        form.addRow(reviewed)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        ratios = tuple(spin.value() for spin in spins)
        if sum(ratios) == 0:
            QtWidgets.QMessageBox.warning(self, "Export dataset", "At least one subset must be non-zero.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose dataset parent directory")
        if not directory:
            return
        try:
            self._guard_project_operation()
            self._capture_project_image()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Dataset export failed", str(exc)[:6000])
            return
        # The thread works on a frozen copy; labelling may continue meanwhile.
        snapshot = Project.model_validate(self._project.model_dump(mode="json"))
        project_path, format_name = self._project_path, kind.currentText()
        options = dict(seed=seed.value(), include_empty=empty.isChecked(),
                       review_filter={"accepted", "corrected"} if reviewed.isChecked() else None)

        def operation(progress, cancelled):
            return export_dataset(directory, project_path, snapshot, format_name, ratios,
                                  progress=progress, **options)

        def done(job):
            if isinstance(job.error, DatasetExportError):
                QtWidgets.QMessageBox.warning(
                    self, "Dataset not exported",
                    f"{len(job.error.issues)} problem(s) must be fixed first:\n" + str(job.error)[:6000])
                return
            if job.error is not None:
                QtWidgets.QMessageBox.warning(self, "Dataset export failed", str(job.error)[:6000])
                return
            target, report = job.result
            lines = [f"{name}: {info['images']} images, {sum(info['objects'].values())} objects"
                     for name, info in report["splits"].items()]
            QtWidgets.QMessageBox.information(
                self, "Dataset exported", f"{target}\n\n" + "\n".join(lines + report["warnings"]))
        return self._run_io("Export dataset", operation, done)

    # ------------------------------------------------------------ propagation
    def _propagation_sources(self):
        from app.core.export import annotation_mask
        selected = [s for s in self.canvas.selectedShapes if not self._is_temporary(s)]
        shapes = selected or [s for s in self.canvas.shapes if not self._is_temporary(s)]
        payloads = {p["object_id"]: p for p in snapshot_shapes(shapes, str(self.imagePath))}
        width, height = self.image.width(), self.image.height()
        objects = []
        for shape in shapes:
            payload = payloads[shape.other_data["object_id"]]
            if payload["shape_type"] not in {"polygon", "rectangle", "circle"}:
                continue
            mask = annotation_mask(payload, height, width)
            if not mask.any():
                continue
            track = shape.other_data.setdefault("track_id", payload["object_id"])
            objects.append({"object_id": track, "label": shape.label,
                            "mask_rle": mask_rle(mask)})
        return objects

    def _propagate_dialog(self):
        from app.worker_api.schemas import PropagationRequest
        if self._project is None:
            QtWidgets.QMessageBox.information(
                self, "Propagation",
                "Propagation writes proposals into other frames, which requires a project. "
                "Create one from this folder via Project -> New project.")
            return
        try:
            self._guard_project_operation()
            self._capture_project_image()
            objects = self._propagation_sources()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Propagation", str(exc))
            return
        if not objects:
            QtWidgets.QMessageBox.information(self, "Propagation", "No area objects to propagate.")
            return
        images = self._project.images
        start = next(i for i, image in enumerate(images) if image.id == self._project_current)
        current = images[start]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Propagate objects")
        form = QtWidgets.QFormLayout(dialog)
        direction = QtWidgets.QComboBox(); direction.addItems(["forward", "backward", "both"])
        count = QtWidgets.QSpinBox(); count.setRange(1, 300)
        count.setValue(min(20, max(1, len(images) - start - 1)))
        form.addRow(QtWidgets.QLabel(
            f"{len(objects)} object(s) from frame {start + 1}/{len(images)} "
            f"({'selected' if self.canvas.selectedShapes else 'all'})"))
        form.addRow("Direction", direction)
        form.addRow("Frames per direction", count)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        n, mode = count.value(), direction.currentText()
        low = start - n if mode in {"backward", "both"} else start
        high = start + n if mode in {"forward", "both"} else start
        window = [i for i in range(max(0, low), min(len(images) - 1, high) + 1)
                  if (images[i].width, images[i].height) == (current.width, current.height)]
        if len(window) < 2:
            QtWidgets.QMessageBox.information(
                self, "Propagation", "No neighbouring frames of the same size in that direction.")
            return
        frame_ids = [images[i].id for i in window]
        paths = [str(resolve_image(self._project_path, images[i])) for i in window]
        request = PropagationRequest(
            frame_paths=paths, start_index=window.index(start), objects=objects,
            direction=mode, display=self._current_display())
        client = self.worker_client

        def operation(progress, cancelled):
            progress(0, 1, f"SAM2 is tracking {len(objects)} object(s) over {len(paths)} frames...")
            return client.propagate(request)

        def done(job):
            if job.cancelled:
                self.statusBar().showMessage("Propagation result discarded", 5000)
                return
            if job.error is not None:
                QtWidgets.QMessageBox.warning(self, "Propagation failed", str(job.error)[:6000])
                return
            self._project_safe(lambda: self._apply_propagation(job.result, frame_ids, current.id))
        self._run_io("Propagate objects", operation, done, exclusive=True)

    def _apply_propagation(self, result, frame_ids, source_image_id):
        """Write proposals; replace only earlier unreviewed propagated copies."""
        project = Project.model_validate(self._project.model_dump(mode="json"))
        classes = {c.name: c for c in project.classes}
        by_id = {image.id: image for image in project.images}
        added = kept = 0
        for frame in result.frames:
            image = by_id[frame_ids[frame["index"]]]
            if image.id == source_image_id:
                continue
            for entry in frame["objects"]:
                track = entry["track_id"]
                same = [a for a in image.annotations if a.metadata.get("track_id") == track]
                if any(a.source != "sam2_propagation" or a.review_status != "unreviewed"
                       for a in same):
                    kept += 1           # never overwrite manual or reviewed work
                    continue
                image.annotations = [a for a in image.annotations if a not in same]
                image.annotations.append(Annotation(
                    class_id=classes[entry["label"]].id, shape_type="polygon",
                    points=[tuple(p) for p in entry["polygon"]], source="sam2_propagation",
                    review_status="unreviewed",
                    metadata={"track_id": track, "mask_rle": entry["mask_rle"],
                              "mask_outline": entry["polygon"], "raster_source_valid": True,
                              "geometry_modified": False, "propagated_from_image": source_image_id,
                              "area": entry["area"]}))
                added += 1
        project.journal.append({"time": now(), "action": "sam2_propagation", "added": added,
                                "kept_existing": kept, "lost": len(result.lost)})
        self._apply_project_change(project)
        QtWidgets.QMessageBox.information(
            self, "Propagation complete",
            f"Added {added} proposal(s) marked 'unreviewed' (source sam2_propagation).\n"
            f"Kept {kept} existing manual/reviewed object(s) unchanged.\n"
            f"Object lost on {len(result.lost)} frame/object pair(s).\n"
            f"Time: {result.elapsed_ms / 1000:.1f} s")
        return added, kept

    def _project_classes_dialog(self):
        if self._project is None:
            QtWidgets.QMessageBox.information(self, "Project classes", "Create/open a project first.")
            return
        from app.gui.project_classes_dialog import ProjectClassesDialog
        def operation():
            self._guard_project_operation()
            self._capture_project_image()
            ProjectClassesDialog(self).exec_()
        self._project_safe(operation)

    def _project_autosave_interval(self):
        if self._project is None:
            return
        seconds, ok = QtWidgets.QInputDialog.getInt(
            self, "Autosave", "Interval in seconds", self._project.autosave_seconds, 10, 3600)
        if ok:
            self._project.autosave_seconds = seconds
            self._project_dirty = True
            self._project_timer.start(seconds * 1000)

    def _project_autosave(self):
        if self._project is not None and not self._project_loading and self._job is None:
            if self._project_safe(self._capture_project_image) is False:
                self._project_timer.stop()
                return
            if (self.dirty or self._project_dirty) and getattr(self, "_autosave_job", None) is None:
                self._autosave_async()

    def _autosave_async(self):
        try:
            self._capture_project_image()   # canvas -> project before snapshot
            snapshot = Project.model_validate(self._project.model_dump(mode="json"))
        except (OSError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            self._project_timer.stop()
            QtWidgets.QMessageBox.warning(
                self, "Autosave failed",
                f"{str(exc)[:8000]}\n\nAutosave is paused; fix the object and use Project > Save project.")
            return
        state = snapshot.model_dump_json(exclude={"modified_at"})
        path, expected, project = self._project_path, self._project_hash, self._project

        def operation(progress, cancelled):
            progress(0, 1, "Autosaving project...")
            return save_project(path, snapshot, expected_sha256=expected)

        def done(job):
            self._autosave_job = None
            if self._project is not project:
                return          # project was closed/replaced meanwhile
            if job.error is not None:
                self._project_timer.stop()
                self._project_dirty = True
                QtWidgets.QMessageBox.warning(self, "Autosave failed", str(job.error)[:8000])
                return
            self._project_hash = job.result
            project.modified_at = snapshot.modified_at
            try:
                self._capture_project_image()
            except (OSError, ValueError):
                return
            if project.model_dump_json(exclude={"modified_at"}) == state:
                self._project_dirty = False
                self.setClean()
            self.statusBar().showMessage(f"Autosaved: {path.name}", 3000)
        self._autosave_job = self._run_io("Autosave", operation, done, modal=False)
        self._autosave_done = done

    def _finish_pending_autosave(self):
        job = getattr(self, "_autosave_job", None)
        if job is not None:
            self._wait_io(job, self._autosave_done)

    def _close_project(self):
        if self._project is None:
            return
        def operation():
            self._guard_project_operation()
            if not self.mayContinue():
                return
            self._project_timer.stop()
            self._clear_class_shortcuts()
            self._project_tokens.clear()
            self._project = self._project_current = self._project_history = None
            self._project_path = self._project_hash = None
            self._project_dirty = False
            self._config = self._project_saved_config
            self._project_saved_config = None
            self.labelDialog.labelList.clear()
            self.labelDialog.labelList.addItems(self._config.get("labels") or [])
            self.uniqLabelList.clear()
            blocker = QtCore.QSignalBlocker(self.fileListWidget)
            self.fileListWidget.clear()
            del blocker
            self.resetState()
            self.setClean()
            self.setWindowTitle("LabelMe MedPhys SAM2")
        self._project_safe(operation)
