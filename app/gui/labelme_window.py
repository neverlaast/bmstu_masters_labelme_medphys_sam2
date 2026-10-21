"""Application adapter for unmodified LabelMe 5.2.1.

Temporary prompts/candidates live on the canvas but never in the final label
list. All Qt widgets are accessed on the GUI thread; HTTP runs in a QThread.
"""
from __future__ import annotations

import logging
import base64
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from copy import deepcopy
from math import cos, sin, pi

from app.paths import ensure_labelme

ensure_labelme()

from qtpy import QtCore, QtGui, QtWidgets

from app.core.video import is_video

from app.core.video import is_video
from labelme.app import MainWindow as LabelMeMainWindow
from labelme.shape import Shape
from labelme.label_file import LabelFile

from app.core.history import SnapshotHistory
from app.gui.worker_task import WorkerTask
from app.worker_api.client import WorkerClient
from app.worker_api.schemas import (
    BoxPrompt, CirclePrompt, PointPrompt, PolygonPrompt, GenerationOptions, SegmentationRequest,
)
from app.core.selection import centroid, inside_roi
from app.gui.generation_dialog import GenerationDialog
from app.core.export import snapshot_shapes, validate_annotations, export_image, FORMATS
from app.core.importers import ImportFormatError, find_yolo_label, import_coco, import_yolo
from worker.imaging import (
    DEFAULT_DISPLAY, StackImageError, describe, frame_count, load_rgb8,
    needs_rendering, normalise_display, png_bytes, split_stack,
)

# path -> display settings for images that need an explicit 8-bit rendering
_DISPLAY_SETTINGS = {}
_upstream_load_image_file = LabelFile.load_image_file


def _display_key(filename):
    return str(Path(filename).resolve())


def _load_image_file(filename):
    """16/32-bit, float and multichannel images are rendered, never altered."""
    try:
        if needs_rendering(filename):
            settings = _DISPLAY_SETTINGS.get(_display_key(filename))
            return png_bytes(load_rgb8(filename, settings or dict(DEFAULT_DISPLAY)))
    except StackImageError:
        return None
    except (OSError, ValueError, IndexError) as exc:
        logging.getLogger(__name__).error("Cannot render %s: %s", filename, exc)
        return None
    return _upstream_load_image_file(filename)


LabelFile.load_image_file = staticmethod(_load_image_file)
from app.gui.project_window import ProjectWindowMixin

logger = logging.getLogger(__name__)


class MedPhysLabelMeWindow(ProjectWindowMixin, LabelMeMainWindow):
    SAM2_CANDIDATE_COLORS = [
        (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
        (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 112, 99),
    ]

    def __init__(self, *args, **kwargs):
        self._init_project_state()
        # Upstream may call overridden methods while constructing/loading.
        self._initializing = True
        self._syncing = False
        self._restoring = False
        self._frame_revision = 0
        self._job = None
        self._jobs = set()
        self._sam2_candidate_list = None
        self._sam2_candidate_dock = None
        self._candidate_by_id = {}
        self._generation_options = GenerationOptions()
        self._refinement_token = None
        self._refinement_source_id = None
        self.sam2_prompt_type = None
        self._prompt_label = 1
        self.worker_client = WorkerClient()
        self._history = SnapshotHistory(
            clone=lambda shapes: [s.copy() for s in shapes],
            key=self._shape_signature,
            limit=100,
        )
        super().__init__(*args, **kwargs)
        self._install_medphys_menu()
        self._install_candidates_dock()
        self._install_batch_actions_menu()
        self._install_export_menu()
        self._install_display_menu()
        self._install_project_menu()
        self._install_video_entry()
        self.labelList.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self._show_label_list_context_menu)
        # Upstream already connected itemSelectionChanged to our override.
        self._initializing = False
        self._history.reset(self.canvas.shapes)
        self._refresh_views()

    @staticmethod
    def _shape_signature(shapes):
        return [
            (
                s.label, s.shape_type, s.group_id, s.description,
                tuple((p.x(), p.y()) for p in s.points),
                dict(s.flags or {}), dict(s.other_data or {}),
            )
            for s in shapes
        ]

    @staticmethod
    def _is_sam2_prompt(shape):
        return bool((shape.flags or {}).get("sam2_prompt"))

    @staticmethod
    def _is_sam2_candidate(shape):
        flags = shape.flags or {}
        return bool(flags.get("candidate") and flags.get("sam2") and flags.get("pending"))

    def _is_temporary(self, shape):
        return self._is_sam2_prompt(shape) or self._is_sam2_candidate(shape)

    def _get_selected_shapes(self):
        return list(self.canvas.selectedShapes)

    def _get_selected_final_shapes(self):
        return [s for s in self.canvas.selectedShapes if not self._is_temporary(s)]

    def _get_selected_candidate_shapes(self):
        return [s for s in self.canvas.selectedShapes if self._is_sam2_candidate(s)]

    def _iter_sam2_candidate_shapes(self):
        return (s for s in self.canvas.shapes if self._is_sam2_candidate(s))

    def _iter_sam2_prompt_shapes(self):
        return (s for s in self.canvas.shapes if self._is_sam2_prompt(s))

    def _action(self, menu, text, slot, shortcut=None):
        action = QtWidgets.QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def _install_medphys_menu(self):
        menu = self.menuBar().addMenu("Segmentation / SAM2")
        self._action(menu, "Propagate objects to other frames...", self._propagate_dialog)
        menu.addSeparator()
        self._action(menu, "Check worker", self._check_worker)
        self._action(menu, "Restart worker", self._restart_worker)
        self._action(menu, "Show worker log", self._show_worker_log)
        self._action(menu, "Show model capabilities", lambda: self._start_task(
            self.worker_client.healthcheck, "capabilities"))
        self._action(menu, "Unload current model", lambda: self._start_task(
            self.worker_client.unload_model, "unload_model"))
        menu.addSeparator()
        for text, kind, label in [
            ("Positive point", "point", 1),
            ("Negative point", "point", 0),
            ("Rectangle prompt", "rectangle", 1),
            ("Circle prompt: one object", "circle", 1),
        ]:
            self._action(menu, text, lambda checked=False, k=kind, l=label:
                         self._activate_sam2_prompt_drawing(k, l))
        self._action(menu, "Finish prompt drawing", lambda: self.toggleDrawMode(True))
        self._run_action = self._action(menu, "Run SAM2", self._run_sam2, "Ctrl+Return")
        self._action(menu, "Refine previously selected mask", lambda: self._run_sam2(refine=True))
        self._action(menu, "Polygon ROI: object search", lambda: self._activate_sam2_prompt_drawing("polygon"))
        self._action(menu, "Find objects inside ROI", self._find_objects)
        self._action(menu, "Find objects in whole image", lambda: self._find_objects(whole_image=True))
        self._action(menu, "Object search settings...", self._edit_generation_settings)
        self._action(menu, "Clear worker cache", self._clear_worker_cache)
        self._cancel_action = self._action(menu, "Cancel / discard pending result", self._cancel_sam2)
        self._cancel_action.setEnabled(False)
        menu.addSeparator()
        self._action(menu, "Accept selected candidates", self._accept_selected_sam2_candidates)
        self._action(menu, "Accept all candidates", self._accept_sam2_candidates)
        self._action(menu, "Reject selected candidates", self._reject_selected_sam2_candidates)
        self._action(menu, "Reject all candidates", self._reject_sam2_candidates)
        self._action(menu, "Clear prompts", self._clear_sam2_prompts)

    def _install_batch_actions_menu(self):
        menu = self.menuBar().addMenu("Batch")
        self._action(menu, "Select all final annotations", self._select_all_final, "Ctrl+A")
        self._action(menu, "Select final annotations by class", self._select_by_class)
        self._action(menu, "Select final objects inside ROI (centroids)", self._select_in_roi)
        self._action(menu, "Delete selected annotations", self._delete_selected_final_shapes)
        self._action(menu, "Relabel selected annotations", self._relabel_selected_final_shapes)
        self._redo_action = self._action(menu, "Redo", self.redoShapeEdit, "Ctrl+Shift+Z")
        self._redo_action.setEnabled(False)

    def _install_candidates_dock(self):
        dock = QtWidgets.QDockWidget("SAM2 Candidates", self)
        dock.setObjectName("sam2CandidatesDock")
        widget = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(widget)
        listing = QtWidgets.QListWidget(widget)
        listing.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        listing.itemSelectionChanged.connect(self._on_candidate_selection)
        layout.addWidget(listing)
        for text, slot in [
            ("Accept selected", self._accept_selected_sam2_candidates),
            ("Reject selected", self._reject_selected_sam2_candidates),
            ("Reject all", self._reject_sam2_candidates),
        ]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            layout.addWidget(button)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self._sam2_candidate_dock = dock
        self._sam2_candidate_list = listing
        dock.hide()

    def _install_export_menu(self):
        menu = self.menuBar().addMenu("Validate / Export")
        self._action(menu, "Validate current image annotations", self._validate_current_annotations)
        for name in FORMATS:
            self._action(menu, f"Export current image: {name.upper()}...",
                         lambda checked=False, kind=name: self._export_current_image(kind))
        menu.addSeparator()
        self._action(menu, "Import annotations: COCO JSON...",
                     lambda checked=False: self._import_annotations("coco"))
        self._action(menu, "Import annotations: YOLO TXT...",
                     lambda checked=False: self._import_annotations("yolo"))

    # ------------------------------------------------------ scientific images
    def _install_display_menu(self):
        menu = self.menuBar().addMenu("Image display")
        self._action(menu, "Display range and channels...", self._edit_display)
        self._action(menu, "Reset display to automatic", self._reset_display)
        self._action(menu, "Image information", self._show_image_info)
        menu.addSeparator()
        self._action(menu, "Split multi-page image into frames...", self._split_stack_dialog)

    def _current_display(self):
        if not self.imagePath or not needs_rendering(self.imagePath):
            return None
        return normalise_display(_DISPLAY_SETTINGS.get(_display_key(self.imagePath)))

    def _apply_display(self, settings):
        """Re-render the open image without touching shapes or the file."""
        key = _display_key(self.imagePath)
        previous = _DISPLAY_SETTINGS.get(key)
        _DISPLAY_SETTINGS[key] = normalise_display(settings)
        try:
            data = png_bytes(load_rgb8(self.imagePath, _DISPLAY_SETTINGS[key]))
        except Exception:
            if previous is None:
                _DISPLAY_SETTINGS.pop(key, None)
            else:
                _DISPLAY_SETTINGS[key] = previous
            raise
        image = QtGui.QImage.fromData(data)
        self.imageData, self.image = data, image
        self.canvas.loadPixmap(QtGui.QPixmap.fromImage(image), clear_shapes=False)
        if self._project is not None:
            self._capture_project_image()   # persists display in the project record
            self.setDirty()
        self.statusBar().showMessage("Display updated; SAM2 will use the same rendering.", 4000)

    def _require_rendered_image(self):
        if self.image is None or self.image.isNull() or not self.imagePath:
            QtWidgets.QMessageBox.information(self, "Image display", "Open an image first.")
            return False
        if not needs_rendering(self.imagePath):
            QtWidgets.QMessageBox.information(
                self, "Image display", "This 8-bit image is shown as stored; no display mapping is needed.")
            return False
        return True

    def _edit_display(self):
        if not self._require_rendered_image():
            return
        info = describe(self.imagePath)
        current = normalise_display(_DISPLAY_SETTINGS.get(_display_key(self.imagePath)))
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Display range and channels")
        form = QtWidgets.QFormLayout(dialog)
        form.addRow(QtWidgets.QLabel(
            f"{info['dtype']}, {info['channels']} channel(s), "
            f"values {info['min']:g} .. {info['max']:g}"))
        mode = QtWidgets.QComboBox()
        mode.addItems(["auto", "manual"])
        mode.setCurrentText(current["mode"] if current["mode"] != "native" else "auto")
        low_pct, high_pct = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box, value in ((low_pct, current["low_pct"]), (high_pct, current["high_pct"])):
            box.setRange(0, 100); box.setDecimals(2); box.setValue(float(value))
        low, high = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box, value in ((low, current["low"] if current["low"] is not None else info["min"]),
                           (high, current["high"] if current["high"] is not None else info["max"])):
            box.setRange(-1e12, 1e12); box.setDecimals(4); box.setValue(float(value))
        gamma = QtWidgets.QDoubleSpinBox()
        gamma.setRange(0.1, 10); gamma.setSingleStep(0.1); gamma.setValue(float(current["gamma"]))
        channels = QtWidgets.QLineEdit(
            ",".join(map(str, current["channels"])) if current["channels"] else "")
        channels.setPlaceholderText(f"first three of 0..{info['channels'] - 1}; e.g. 2 or 0,1")
        form.addRow("Mode", mode)
        form.addRow("Low percentile (auto)", low_pct)
        form.addRow("High percentile (auto)", high_pct)
        form.addRow("Low value (manual)", low)
        form.addRow("High value (manual)", high)
        form.addRow("Gamma", gamma)
        form.addRow("Channels -> R,G,B", channels)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            text = channels.text().strip()
            selected = [int(v) for v in text.replace(" ", "").split(",")] if text else None
            settings = {"mode": mode.currentText(), "low_pct": low_pct.value(),
                        "high_pct": high_pct.value(), "gamma": gamma.value(),
                        "channels": selected}
            if mode.currentText() == "manual":
                settings.update(low=low.value(), high=high.value())
            self._apply_display(settings)
        except (ValueError, IndexError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid display settings", str(exc))

    def _reset_display(self):
        if self._require_rendered_image():
            self._apply_display(dict(DEFAULT_DISPLAY))

    def _show_image_info(self):
        if self.image is None or not self.imagePath:
            QtWidgets.QMessageBox.information(self, "Image information", "Open an image first.")
            return
        info = describe(self.imagePath)
        display = self._current_display()
        QtWidgets.QMessageBox.information(self, "Image information", "\n".join(
            [f"{k}: {v}" for k, v in info.items()]
            + [f"display: {display if display else 'native 8-bit'}"]))

    def _offer_split_stack(self, filename, frames):
        answer = QtWidgets.QMessageBox.question(
            self, "Multi-page image",
            f"{Path(filename).name} contains {frames} frames. LabelMe stores one annotation "
            f"file per image, so frames are labelled as separate lossless TIFF files.\n\n"
            f"Split into {Path(filename).stem}_frames/ next to the original and open the folder?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        if answer != QtWidgets.QMessageBox.Yes:
            return False
        return self._split_and_open(filename)

    def _split_and_open(self, filename):
        target = Path(filename).parent / f"{Path(filename).stem}_frames"
        try:
            if target.exists():
                outputs = sorted(target.glob(f"{Path(filename).stem}_f*.tif"))
                if not outputs:
                    raise FileExistsError(f"{target} exists but contains no frames")
            else:
                def operation(progress, cancelled):
                    progress(0, 1, f"Splitting {Path(filename).name}...")
                    return split_stack(filename)

                def done(job):
                    if job.error is not None:
                        QtWidgets.QMessageBox.warning(self, "Cannot split image", str(job.error))
                    else:
                        self._open_split_frames(target, job.result)
                return self._run_io("Split multi-page image", operation, done) is not None
        except (OSError, ValueError, IndexError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot split image", str(exc))
            return False
        return self._open_split_frames(target, outputs)

    def _open_split_frames(self, target, outputs):
        if self._project is not None:
            QtWidgets.QMessageBox.information(
                self, "Frames created",
                f"{len(outputs)} frames written to {target}. Add them via Project -> Add images.")
            return False
        self.importDirImages(str(target))
        return True

    def _split_stack_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Multi-page image", str(Path(self.imagePath).parent) if self.imagePath else "",
            "TIFF stacks (*.tif *.tiff);;All files (*)")
        if not filename:
            return
        if frame_count(filename) <= 1:
            QtWidgets.QMessageBox.information(self, "Split", "The image has a single frame.")
            return
        self._split_and_open(filename)

    def _import_annotations(self, format_name):
        """Add external COCO/YOLO objects to the current image as final shapes."""
        if self.image is None or self.image.isNull() or not self.imagePath:
            QtWidgets.QMessageBox.information(self, "Import", "Open an image first.")
            return
        width, height = self.image.width(), self.image.height()
        image_path = Path(self.imagePath)
        if format_name == "coco":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "COCO annotations", str(image_path.parent), "COCO JSON (*.json)")
        else:
            guess = find_yolo_label(image_path)
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "YOLO labels", str(guess or image_path.parent), "YOLO labels (*.txt)")
        if not path:
            return
        try:
            report = (import_coco(path, image_path.name, width, height) if format_name == "coco"
                      else import_yolo(path, width, height))
        except (ImportFormatError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc)[:4000])
            return
        if not report.annotations:
            QtWidgets.QMessageBox.warning(self, "Nothing imported", report.summary())
            return
        if self._project is not None:
            known = {c.name for c in self._project.classes if c.active}
            missing = sorted(set(report.labels) - known)
            if missing:
                QtWidgets.QMessageBox.warning(
                    self, "Unknown project classes",
                    "Add or activate these classes in Project classes before import:\n"
                    + ", ".join(missing))
                return
        existing = [s for s in self.canvas.shapes if not self._is_temporary(s)]
        replace = False
        if existing:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Import annotations")
            box.setText(f"The image already has {len(existing)} objects. "
                        f"Import {len(report.annotations)} objects:")
            append = box.addButton("Append", QtWidgets.QMessageBox.AcceptRole)
            replace_button = box.addButton("Replace existing", QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() not in (append, replace_button):
                return
            replace = box.clickedButton() is replace_button
        shapes = []
        for data in report.annotations:
            data = dict(data)
            shape = Shape(label=data.pop("label"), shape_type=data.pop("shape_type"),
                          group_id=data.pop("group_id"), description=data.pop("description"))
            for x, y in data.pop("points"):
                shape.addPoint(QtCore.QPointF(float(x), float(y)))
            shape.close()
            shape.flags = data.pop("flags")
            shape.other_data = data
            shapes.append(shape)
        if replace:
            ids = {id(s) for s in existing}
            self.canvas.shapes = [s for s in self.canvas.shapes if id(s) not in ids]
            self.canvas.selectedShapes = []
            self.canvas.hShape = self.canvas.prevhShape = None
            self.canvas.hVertex = self.canvas.prevhVertex = None
            self.canvas.visible = {s: v for s, v in self.canvas.visible.items() if id(s) not in ids}
        self.canvas.shapes.extend(shapes)
        self.canvas.storeShapes()
        self._refresh_views()
        self.setDirty()
        QtWidgets.QMessageBox.information(self, "Import complete", report.summary())
        return report

    def _export_snapshot(self):
        if self.image is None or not self.imagePath:
            raise ValueError("Open an image first.")
        shapes = snapshot_shapes(self.canvas.shapes, str(Path(self.imagePath).resolve()))
        configured = self._config.get("labels")
        if self._project is not None:
            configured = [c.name for c in self._project.classes]
        classes = list(configured) if configured else None
        return shapes, self.image.width(), self.image.height(), classes

    def _validate_current_annotations(self):
        try:
            shapes, width, height, classes = self._export_snapshot()
            issues, masks = validate_annotations(shapes, width, height, classes)
            del masks
            text = ("\n".join(issues[:50]) if issues else
                    f"Validated {len(shapes)} final objects. No geometry/ID errors.")
            if len(issues) > 50:
                text += f"\n... {len(issues)-50} more issues."
            if classes is None:
                text += "\nNo project class registry yet: class names are taken from this image."
            if issues:
                QtWidgets.QMessageBox.warning(self, "Annotation validation", text)
            else:
                QtWidgets.QMessageBox.information(self, "Annotation validation", text)
        except (ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Annotation validation", str(exc))

    def _export_current_image(self, format_name):
        try:
            shapes, width, height, classes = self._export_snapshot()
            issues, masks = validate_annotations(shapes, width, height, classes)
            del masks
            if issues:
                raise ValueError("\n".join(issues[:50]))
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export parent directory")
            if not directory:
                return
            target, report = export_image(directory, self.imagePath, shapes, width, height,
                                          format_name, classes)
            QtWidgets.QMessageBox.information(
                self, "Export complete",
                f"Images: {report['images']}\nObjects: {report['objects']}\nErrors: 0\n{target}")
        except (OSError, ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc)[:8000])

    def addLabel(self, shape):
        shape.flags = shape.flags or {}
        shape.other_data = shape.other_data or {}
        if not self._is_temporary(shape):
            super().addLabel(shape)

    def _get_rgb_by_label(self, label):
        if self._project is not None:
            definition = next((c for c in self._project.classes if c.name == label), None)
            if definition is not None:
                color = QtGui.QColor(definition.color)
                return color.red(), color.green(), color.blue()
        return super()._get_rgb_by_label(label)

    def validateLabel(self, label):
        if self._project is not None:
            return any(c.name == label and c.active for c in self._project.classes)
        return super().validateLabel(label)

    def loadLabels(self, shapes):
        # Old or externally generated LabelMe JSON may contain null flags.
        normalised = [
            dict(s, flags=s.get("flags") or {}, other_data=s.get("other_data") or {})
            for s in shapes
        ]
        return super().loadLabels(normalised)

    def _refresh_views(self):
        selected = [s for s in self.canvas.selectedShapes if s in self.canvas.shapes]
        self._syncing = True
        self._noSelectionSlot = True
        # Model reset/rowsInserted must reach QListView. Blocking the model
        # leaves real rows invisible until an unrelated manual annotation.
        blockers = [QtCore.QSignalBlocker(self.labelList)]
        try:
            self.labelList.clear()
            for shape in self.canvas.shapes:
                self.addLabel(shape)
            listing = self._sam2_candidate_list
            if listing is not None:
                blockers.append(QtCore.QSignalBlocker(listing))
                listing.clear()
                self._candidate_by_id = {}
                for index, shape in enumerate(self._iter_sam2_candidate_shapes()):
                    score = (shape.other_data or {}).get("sam2_score")
                    text = f"Candidate {index + 1}"
                    if score is not None:
                        text += f" | score={score:.4f}"
                    values = (shape.other_data or {}).get("metrics", {})
                    if values:
                        text += f" | A={values.get('area_px', 0)} px²"
                        if values.get("overlap_warning"):
                            text += " | overlaps existing"
                    item = QtWidgets.QListWidgetItem(text)
                    item.setData(QtCore.Qt.UserRole, id(shape))
                    item.setForeground(QtGui.QBrush(shape.line_color))
                    item.setToolTip(json.dumps(values, ensure_ascii=False, indent=2))
                    listing.addItem(item)
                    self._candidate_by_id[id(shape)] = shape
                self._sam2_candidate_dock.setVisible(bool(listing.count()))
        finally:
            del blockers
            self._noSelectionSlot = False
            self._syncing = False
        self.shapeSelectionChanged(selected)
        self._update_history_actions()
        self.canvas.update()

    def shapeSelectionChanged(self, selected_shapes):
        if self._syncing:
            return
        selected = [s for s in selected_shapes if s in self.canvas.shapes]
        self._syncing = True
        self._noSelectionSlot = True
        blockers = [QtCore.QSignalBlocker(self.labelList)]
        try:
            for shape in self.canvas.shapes:
                shape.selected = shape in selected
            self.canvas.selectedShapes = selected
            self.labelList.clearSelection()
            for shape in selected:
                if self._is_temporary(shape):
                    continue
                item = self.labelList.findItemByShape(shape)
                if item is not None:
                    self.labelList.selectItem(item)
            listing = self._sam2_candidate_list
            if listing is not None:
                blockers.append(QtCore.QSignalBlocker(listing))
                for i in range(listing.count()):
                    item = listing.item(i)
                    shape = self._candidate_by_id.get(item.data(QtCore.Qt.UserRole))
                    item.setSelected(shape in selected)
        finally:
            del blockers
            self._noSelectionSlot = False
            self._syncing = False
        finals = [s for s in selected if not self._is_temporary(s)]
        if len(selected) == 1:
            data = selected[0].other_data or {}
            if data.get("refinement_token"):
                self._refinement_token = data["refinement_token"]
                self._refinement_source_id = (
                    data.get("object_id") if not self._is_temporary(selected[0])
                    else data.get("refines_object_id"))
        self.actions.delete.setEnabled(bool(selected))
        self.actions.copy.setEnabled(bool(finals))
        self.actions.duplicate.setEnabled(bool(finals))
        self.actions.edit.setEnabled(len(finals) == 1 and len(selected) == 1)
        self.canvas.update()

    def labelSelectionChanged(self, *args):
        if self._syncing or self._noSelectionSlot or not self.canvas.editing():
            return
        self.shapeSelectionChanged([item.shape() for item in self.labelList.selectedItems()])

    def _on_candidate_selection(self):
        if self._syncing:
            return
        selected = [s for s in self.canvas.selectedShapes if not self._is_sam2_candidate(s)]
        selected += [
            self._candidate_by_id[item.data(QtCore.Qt.UserRole)]
            for item in self._sam2_candidate_list.selectedItems()
        ]
        self.shapeSelectionChanged(selected)

    def _select_shapes_in_canvas(self, shapes):
        self.canvas.selectShapes(list(shapes))

    def _select_all_final(self):
        self._select_shapes_in_canvas([s for s in self.canvas.shapes if not self._is_temporary(s)])

    def _select_by_class(self):
        label = self._ask_target_label()
        if label:
            self._select_shapes_in_canvas([
                s for s in self.canvas.shapes if not self._is_temporary(s) and s.label == label
            ])

    def _activate_sam2_prompt_drawing(self, prompt_type, label=1):
        if not self.imagePath:
            self.statusBar().showMessage("Open an image first.", 4000)
            return
        if prompt_type not in {"point", "rectangle", "circle", "polygon"}:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        # Retain points to allow positive + negative prompts. Only one ROI.
        if prompt_type != "point":
            self._remove_shapes([s for s in self._iter_sam2_prompt_shapes()
                                 if s.shape_type in {"rectangle", "circle", "polygon"}])
            self._refinement_token = self._refinement_source_id = None
        self._prompt_label = label
        self.toggleDrawMode(False, createMode=prompt_type)
        self.sam2_prompt_type = prompt_type
        self.statusBar().showMessage(f"Draw SAM2 {prompt_type}; no class dialog is needed.", 5000)

    def toggleDrawMode(self, edit=True, createMode="polygon"):
        # Native annotation actions must not inherit a previous SAM prompt mode.
        self.sam2_prompt_type = None
        result = super().toggleDrawMode(edit, createMode)
        return result

    def newShape(self):
        shape = self.canvas.shapes[-1]
        # Prompt activation sets the mode after toggleDrawMode.
        if self.sam2_prompt_type == shape.shape_type:
            shape.label = "__sam2_prompt__"
            shape.flags = {"sam2_prompt": True, "temporary": True}
            shape.other_data = {"source": "sam2_prompt", "point_label": self._prompt_label}
            shape.line_color = QtGui.QColor(0, 190, 230) if self._prompt_label else QtGui.QColor(230, 70, 70)
            shape.fill_color = QtGui.QColor(shape.line_color)
            shape.fill_color.setAlpha(50)
            self.canvas.storeShapes()
            self.setDirty()
            self.toggleDrawMode(True)
            self._refresh_views()
        else:
            super().newShape()
            self._refresh_views()

    def _build_segmentation_request(self, mode="interactive", whole_image=False, refine=False):
        prompts = [] if whole_image else list(self._iter_sam2_prompt_shapes())
        # Native final annotations may be used as prompts without conversion
        # into temporary objects or destruction of their label.
        if not prompts and not whole_image:
            prompts = [s for s in self.canvas.selectedShapes
                       if s.shape_type in {"point", "rectangle", "circle", "polygon"}]
        if mode == "automatic" and not whole_image and not any(
                s.shape_type in {"rectangle", "circle", "polygon"} for s in prompts):
            prompts += [s for s in self.canvas.selectedShapes
                        if s.shape_type in {"rectangle", "circle", "polygon"} and s not in prompts]
        if refine:
            # A large search ROI must not inject positive points into other
            # objects while refining one selected mask. GUI refinement uses
            # the mask prior plus explicit points, not the search ROI.
            prompts = [s for s in prompts if s.shape_type == "point"]
        points, box, circle, polygon = [], None, None, None
        for shape in prompts:
            coords = [(float(p.x()), float(p.y())) for p in shape.points]
            if shape.shape_type == "point" and len(coords) == 1:
                x, y = coords[0]
                points.append(PointPrompt(x=x, y=y,
                                          label=(shape.other_data or {}).get("point_label", 1)))
            elif shape.shape_type == "rectangle" and len(coords) == 2:
                if box or circle or polygon:
                    raise ValueError("Use at most one rectangle or circle ROI.")
                (x1, y1), (x2, y2) = coords
                box = BoxPrompt(x1=min(x1, x2), y1=min(y1, y2),
                                x2=max(x1, x2), y2=max(y1, y2))
            elif shape.shape_type == "circle" and len(coords) == 2:
                if box or circle or polygon:
                    raise ValueError("Use at most one rectangle or circle ROI.")
                (cx, cy), (px, py) = coords
                circle = CirclePrompt(center_x=cx, center_y=cy,
                                      radius=((px-cx)**2 + (py-cy)**2)**0.5,
                                      mode="multiple_objects" if mode == "automatic" else "single_object")
            elif shape.shape_type == "polygon" and len(coords) >= 3:
                if box or circle or polygon:
                    raise ValueError("Use at most one ROI.")
                polygon = PolygonPrompt(points=coords)
        if mode == "interactive" and not points and box is None and circle is None:
            if refine:
                raise ValueError("Add a positive or negative point before refining the selected mask.")
            raise ValueError("Draw a prompt, or select a point/rectangle/circle.")
        if mode == "automatic" and not whole_image and box is None and circle is None and polygon is None:
            raise ValueError("Draw or select a rectangle, circle or polygon ROI first.")
        if refine and not self._refinement_token:
            raise ValueError("First select one SAM2 candidate or accepted SAM2 object, then add point prompts.")
        generation = self._generation_options.model_copy(update={"mode": mode})
        existing = []
        for shape in self.canvas.shapes:
            if self._is_temporary(shape):
                continue
            coordinates = [(float(p.x()), float(p.y())) for p in shape.points]
            if shape.shape_type == "polygon":
                existing.append(coordinates)
            elif shape.shape_type == "rectangle" and len(coordinates) == 2:
                (x1, y1), (x2, y2) = coordinates
                existing.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            elif shape.shape_type == "circle" and len(coordinates) == 2:
                (cx,cy), (px,py) = coordinates
                radius = ((px-cx)**2+(py-cy)**2)**.5
                existing.append([(cx+radius*cos(2*pi*i/64), cy+radius*sin(2*pi*i/64))
                                 for i in range(64)])
        return SegmentationRequest(
            image_path=str(Path(self.imagePath).resolve()), points=points,
            box=box, circle=circle, polygon=polygon, generation=generation,
            refinement_token=self._refinement_token if refine else None,
            refinement_target_id=self._refinement_source_id if refine else None,
            existing_polygons=existing,
            display=self._current_display(),
        )

    def _start_task(self, operation, kind, request=None):
        if self._job is not None:
            self.statusBar().showMessage("Wait for the pending request, or cancel it.", 4000)
            return
        task = WorkerTask(operation, self)
        task.frame_revision = self._frame_revision
        task.image_path = self.imagePath
        task.kind = kind
        task.request = request
        task.annotation_state = deepcopy(self._shape_signature(self.canvas.shapes))
        task.succeeded.connect(lambda result, t=task: self._task_succeeded(t, result))
        task.failed.connect(lambda error, t=task: self._task_failed(t, error))
        task.finished.connect(lambda t=task: self._task_finished(t))
        self._jobs.add(task)
        self._job = task
        self._run_action.setEnabled(False)
        self._cancel_action.setEnabled(True)
        self.statusBar().showMessage("SAM2 request running; the interface remains available.")
        task.start()

    def attach_worker_manager(self, manager):
        """Called by app.main; tests construct the window without a manager."""
        self.worker_manager = manager
        manager.client = self.worker_client
        self._worker_status = QtWidgets.QLabel("SAM2: -")
        self.statusBar().addPermanentWidget(self._worker_status)
        manager.stateChanged.connect(self._on_worker_state)
        manager.ensure_running()

    def _on_worker_state(self, state, message):
        self._worker_status.setText(f"SAM2: {state}")
        self._worker_status.setToolTip(message)
        self.statusBar().showMessage(message, 6000)

    def _restart_worker(self):
        manager = getattr(self, "worker_manager", None)
        if manager is None or manager.state == "external":
            self.statusBar().showMessage("Worker is external; restart it manually.", 6000)
            return
        manager.restart()

    def _show_worker_log(self):
        manager = getattr(self, "worker_manager", None)
        text = manager.log_text() if manager else ""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("SAM2 worker log")
        box.setText("Output of the worker process started by the application.")
        box.setDetailedText(text or "(no output: worker is external or not started)")
        box.exec_()

    def _check_worker(self):
        self._start_task(self.worker_client.healthcheck, "health")

    def _run_sam2(self, checked=False, *, refine=False):
        if not self.imagePath:
            return
        try:
            request = self._build_segmentation_request(refine=refine)
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "SAM2 prompt", str(exc))
            return
        self._start_task(lambda: self.worker_client.segment(request), "segment", request)

    def _find_objects(self, checked=False, *, whole_image=False):
        if not self.imagePath:
            return
        try:
            request = self._build_segmentation_request("automatic", whole_image=whole_image)
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "SAM2 object search", str(exc))
            return
        self._start_task(lambda: self.worker_client.segment(request), "segment", request)

    def _edit_generation_settings(self):
        dialog = GenerationDialog(self._generation_options, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._generation_options = dialog.options()

    def _clear_worker_cache(self):
        self._start_task(self.worker_client.clear_cache, "clear_cache")

    def _select_in_roi(self):
        try:
            request = self._build_segmentation_request("automatic")
        except ValueError as exc:
            QtWidgets.QMessageBox.information(self, "ROI selection", str(exc))
            return
        self._select_shapes_in_canvas([
            s for s in self.canvas.shapes if not self._is_temporary(s) and s.points
            and inside_roi(centroid([(p.x(),p.y()) for p in s.points], s.shape_type), request)
        ])

    def _is_current_task(self, task):
        return (not task.cancelled and task.frame_revision == self._frame_revision
                and task.image_path == self.imagePath
                and (task.kind != "segment" or task.annotation_state == self._shape_signature(self.canvas.shapes)))

    def _task_succeeded(self, task, response):
        if not self._is_current_task(task):
            if not task.cancelled:
                self.statusBar().showMessage("Result discarded: image, prompts or annotations changed.", 8000)
            return
        if task.kind in {"health", "clear_cache", "capabilities", "unload_model"}:
            if task.kind in {"clear_cache", "unload_model"}:
                self._refinement_token = self._refinement_source_id = None
                self._project_tokens.clear()
            self._record_project_model(response.model_info)
            if task.kind == "capabilities":
                QtWidgets.QMessageBox.information(
                    self, "Model capabilities",
                    f"{response.model_name} {response.model_version}\n" +
                    "\n".join(f"{key}: {'yes' if value else 'no'}" for key, value in response.capabilities.items()))
            self.statusBar().showMessage(
                f"Worker: device={response.device}, model_loaded={response.model_loaded}, "
                f"cache_entries={response.cache_entries}", 7000)
            return
        if not response.ok:
            self._task_failed(task, response.error_message or "SAM2 returned an error.")
            return
        self._record_project_model(response.model_info)
        if task.request and (
                response.generation_mode != task.request.generation.mode
                or (task.request.refinement_token and response.api_version < 2)):
            self._task_failed(task, "Worker version is outdated. Use Segmentation / SAM2 > Restart worker.")
            return
        # Validate a complete result before replacing the previous candidates.
        candidates = []
        try:
            for index, candidate in enumerate(response.candidates):
                shape = self._make_candidate(candidate.polygon, index, candidate.score)
                shape.other_data.update(model_name=response.model_name,
                                        model_version=response.model_version,
                                        device=response.device, elapsed_ms=response.elapsed_ms)
                shape.other_data.update(candidate.metadata)
                shape.other_data["refinement_token"] = candidate.refinement_token
                candidates.append(shape)
        except (ValueError, TypeError) as exc:
            self._task_failed(task, str(exc))
            return
        self.canvas.shapes = [s for s in self.canvas.shapes if not self._is_sam2_candidate(s)]
        self.canvas.shapes.extend(candidates)
        self._refinement_token = self._refinement_source_id = None
        self.canvas.storeShapes()
        self._refresh_views()
        self.setDirty()
        self.statusBar().showMessage(
            f"SAM2: {len(candidates)} candidate(s); embeddings cache "
            f"{'hit' if response.cache_hit else 'miss'}. " + " ".join(response.warnings), 12000)

    def _task_failed(self, task, error):
        if self._is_current_task(task):
            # Do not log image paths or full requests.
            logger.warning("SAM2 %s request failed (%s)", task.kind, type(error).__name__)
            QtWidgets.QMessageBox.warning(self, "SAM2 request failed", str(error))

    def _task_finished(self, task):
        self._jobs.discard(task)
        if self._job is task:
            self._job = None
            self._run_action.setEnabled(True)
            self._cancel_action.setEnabled(False)
        task.deleteLater()

    def _cancel_sam2(self):
        if self._job is not None:
            self._job.cancel()
            self._cancel_action.setEnabled(False)
            self.statusBar().showMessage(
                "Result cancelled. Worker may finish the calculation; its result will be discarded.", 8000)

    def _make_candidate(self, polygon, index=0, score=None):
        import math
        if len(polygon) < 3:
            raise ValueError("Candidate polygon must have at least 3 vertices.")
        shape = Shape(label=f"SAM2 cand {index + 1}", shape_type="polygon")
        for x, y in polygon:
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                raise ValueError("Non-finite candidate coordinates.")
            if not (0 <= x < self.image.width() and 0 <= y < self.image.height()):
                raise ValueError("Candidate coordinates are outside the image.")
            shape.addPoint(QtCore.QPointF(float(x), float(y)))
        shape.close()
        shape.flags = {"candidate": True, "sam2": True, "pending": True, "temporary": True}
        shape.other_data = {"source": "sam2", "candidate_index": index,
                            "sam2_score": score, "object_id": str(uuid4())}
        color = self.SAM2_CANDIDATE_COLORS[index % len(self.SAM2_CANDIDATE_COLORS)]
        shape.line_color = QtGui.QColor(*color)
        shape.fill_color = QtGui.QColor(*color, 120)
        return shape

    def _add_candidate_polygon_to_canvas(self, polygon, candidate_index=0, score=None):
        shape = self._make_candidate(polygon, candidate_index, score)
        self.canvas.shapes.append(shape)
        self.canvas.storeShapes()
        self._refresh_views()
        self.setDirty()
        return shape

    def _ask_target_label(self, default_label="object"):
        if self._project is not None:
            labels = [c.name for c in self._project.classes if c.active]
            if not labels:
                QtWidgets.QMessageBox.information(self, "Classes", "Add an active class in Project classes.")
                return None
            current = labels.index(default_label) if default_label in labels else 0
            label, ok = QtWidgets.QInputDialog.getItem(
                self, "Class", "Target class", labels, current, editable=False)
            return label if ok else None
        labels = sorted({s.label for s in self.canvas.shapes
                         if not self._is_temporary(s) and s.label}
                        | set(self._config.get("labels") or []) | {default_label})
        text, ok = QtWidgets.QInputDialog.getItem(
            self, "Target class", "Class:", labels, labels.index(default_label), True)
        return text.strip() if ok and text.strip() else None

    def _accept_candidates(self, candidates):
        if not candidates:
            return
        targets = {(s.other_data or {}).get("refines_object_id") for s in candidates}
        targets.discard(None)
        if any(sum((s.other_data or {}).get("refines_object_id") == target for s in candidates) > 1 for target in targets):
            QtWidgets.QMessageBox.information(
                self, "Choose one refinement", "Accept only one alternative per object being refined.")
            return
        overlaps = [s for s in candidates if not (s.other_data or {}).get("refines_object_id")
                    and (s.other_data or {}).get("metrics", {}).get("overlap_warning")]
        if overlaps and QtWidgets.QMessageBox.question(
                self, "Overlap warning", f"{len(overlaps)} candidates overlap existing annotations. Accept anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        old_objects = {s.other_data.get("object_id"): s for s in self.canvas.shapes
                       if not self._is_temporary(s)}
        if any(target not in old_objects for target in targets):
            QtWidgets.QMessageBox.warning(self, "Refinement target missing",
                                         "The original object no longer exists. Run SAM2 again.")
            return
        default = old_objects[next(iter(targets))].label if len(targets) == 1 and next(iter(targets)) in old_objects else "object"
        label = self._ask_target_label(default_label=default)
        if label is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        for shape in candidates:
            shape.label = label
            shape.flags = {k: v for k, v in shape.flags.items()
                           if k not in {"candidate", "sam2", "pending", "temporary"}}
            shape.flags["accepted"] = True
            model_source = "sam2" if shape.other_data.get("model_name", "sam2") == "sam2" else "segmentation_model"
            shape.other_data.update(source=model_source, accepted_label=label,
                                    review_status="accepted", modified_at=now)
            target_id = shape.other_data.get("refines_object_id")
            if target_id in old_objects:
                original = old_objects[target_id]
                merged = dict(original.other_data)
                merged.update(shape.other_data)
                merged.update(object_id=target_id, source=model_source + "_refined", geometry_modified=False)
                shape.other_data = merged
                shape.group_id = original.group_id
                shape.description = original.description
                shape.flags = {**original.flags, **shape.flags}
        self.canvas.shapes = [s for s in self.canvas.shapes
                              if id(s) not in {id(old_objects[t]) for t in targets if t in old_objects}]
        if targets:
            self.canvas.hShape = self.canvas.prevhShape = None
            self.canvas.hVertex = self.canvas.prevhVertex = None
            self.canvas.visible = {s:v for s,v in self.canvas.visible.items() if s in self.canvas.shapes}
        # Unselected candidates remain available; prompts remain for refinement.
        self.canvas.storeShapes()
        self._refresh_views()
        self.shapeSelectionChanged(candidates)
        self.setDirty()

    def _accept_selected_sam2_candidates(self):
        self._accept_candidates(self._get_selected_candidate_shapes())

    def _accept_sam2_candidates(self):
        self._accept_candidates(list(self._iter_sam2_candidate_shapes()))

    def _remove_shapes(self, shapes):
        ids = {id(s) for s in shapes}
        if not ids:
            return 0
        before = len(self.canvas.shapes)
        self.canvas.shapes = [s for s in self.canvas.shapes if id(s) not in ids]
        self.canvas.selectedShapes = [s for s in self.canvas.selectedShapes if id(s) not in ids]
        # Drop stale references held by upstream hover/highlight code.
        self.canvas.hShape = self.canvas.prevhShape = None
        self.canvas.hVertex = self.canvas.prevhVertex = None
        self.canvas.visible = {s: v for s, v in self.canvas.visible.items() if id(s) not in ids}
        self.canvas.storeShapes()
        self._refresh_views()
        self.setDirty()
        return before - len(self.canvas.shapes)

    def _reject_selected_sam2_candidates(self):
        self._remove_shapes(self._get_selected_candidate_shapes())

    def _reject_sam2_candidates(self):
        self._remove_shapes(list(self._iter_sam2_candidate_shapes()))

    def _clear_sam2_prompts(self):
        self._remove_shapes(list(self._iter_sam2_prompt_shapes()))

    def _delete_selected_final_shapes(self):
        selected = self._get_selected_final_shapes()
        if not selected:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete annotations", f"Delete {len(selected)} selected annotation(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            self._remove_shapes(selected)

    def deleteSelectedShape(self):
        finals = self._get_selected_final_shapes()
        if finals:
            self._delete_selected_final_shapes()
        else:
            self._remove_shapes(self.canvas.selectedShapes)

    def _relabel_selected_final_shapes(self):
        selected = self._get_selected_final_shapes()
        if not selected:
            return
        label = self._ask_target_label()
        if label is None:
            return
        for shape in selected:
            shape.label = label
            shape.other_data["modified_at"] = datetime.now(timezone.utc).isoformat()
        self.canvas.storeShapes()
        self._refresh_views()
        self.shapeSelectionChanged(selected)
        self.setDirty()

    def _show_label_list_context_menu(self, pos):
        index = self.labelList.indexAt(pos)
        if not index.isValid():
            return
        item = self.labelList.model().itemFromIndex(index)
        if item not in self.labelList.selectedItems():
            self.shapeSelectionChanged([item.shape()])
        menu = QtWidgets.QMenu(self)
        relabel = menu.addAction("Assign class...")
        delete = menu.addAction("Delete selected...")
        chosen = menu.exec_(self.labelList.viewport().mapToGlobal(pos))
        if chosen is relabel:
            self._relabel_selected_final_shapes()
        elif chosen is delete:
            self._delete_selected_final_shapes()

    def labelOrderChanged(self):
        if self._syncing:
            return
        temporary = [s for s in self.canvas.shapes if self._is_temporary(s)]
        self.canvas.shapes = [item.shape() for item in self.labelList] + temporary
        self.canvas.storeShapes()
        self.setDirty()

    def copySelectedShape(self):
        self._copied_shapes = [s.copy() for s in self._get_selected_final_shapes()]
        self.actions.paste.setEnabled(bool(self._copied_shapes))

    def pasteSelectedShape(self):
        shapes = [s.copy() for s in self._copied_shapes]
        for shape in shapes:
            old_id = (shape.other_data or {}).get("object_id")
            shape.other_data = dict(shape.other_data or {})
            shape.other_data.update(object_id=str(uuid4()), copied_from=old_id)
        self.canvas.loadShapes(shapes, replace=False)
        self._refresh_views()
        self.shapeSelectionChanged(shapes)
        self.setDirty()

    def duplicateSelectedShape(self):
        self.shapeSelectionChanged(self._get_selected_final_shapes())
        super().duplicateSelectedShape()
        self._refresh_views()

    def setDirty(self):
        if not self._initializing and not self._restoring:
            seen_ids = set()
            for shape in self.canvas.shapes:
                data = shape.other_data or {}
                outline = data.get("mask_outline")
                if outline is not None and outline != [[p.x(),p.y()] for p in shape.points]:
                    # A raster prediction must not silently remain ground
                    # truth after manual vector edits.
                    if data.get("refinement_token") == self._refinement_token:
                        self._refinement_token = self._refinement_source_id = None
                    data.setdefault("model_prediction_rle", data.get("mask_rle"))
                    data.update(mask_rle=None, raster_source_valid=False,
                                refinement_token=None, metrics={}, geometry_modified=True)
                    if not self._is_temporary(shape):
                        model_source = "sam2" if data.get("model_name", "sam2") == "sam2" else "model"
                        data.update(source=model_source + "_manually_edited", review_status="corrected")
                if not self._is_temporary(shape):
                    shape.other_data = shape.other_data or {}
                    object_id = shape.other_data.get("object_id")
                    if not object_id or object_id in seen_ids:
                        object_id = str(uuid4())
                        shape.other_data["object_id"] = object_id
                    seen_ids.add(object_id)
                    shape.other_data.setdefault("source", "manual")
                    if self._project is not None:
                        definition = next((c for c in self._project.classes if c.name == shape.label), None)
                        if definition is not None:
                            shape.other_data["class_id"] = definition.id
                        shape.other_data.setdefault("modified_at", datetime.now(timezone.utc).isoformat())
            self._history.commit(self.canvas.shapes)
            if self._project is not None and not self._project_loading:
                self._project_dirty = True
        try:
            super().setDirty()
        except (OSError, ValueError, TypeError) as exc:
            self.dirty = True
            self.actions.save.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Autosave failed", str(exc))
        self._update_history_actions()

    def _update_history_actions(self):
        if not self._initializing:
            self.actions.undo.setEnabled(self._history.can_undo)
            self._redo_action.setEnabled(self._history.can_redo)

    def _restore_history(self, shapes):
        if shapes is None:
            return
        self._restoring = True
        try:
            self.canvas.selectedShapes = []
            self.canvas.visible.clear()
            self.canvas.hShape = self.canvas.prevhShape = None
            self.canvas.hVertex = self.canvas.prevhVertex = None
            self.canvas.current = None
            self.canvas.shapes = shapes
            # Maintain upstream geometry backups without using them as the
            # application history (which also includes labels and metadata).
            self.canvas.shapesBackups = []
            self.canvas.storeShapes()
            self._refresh_views()
            self.setDirty()
        finally:
            self._restoring = False
        self._update_history_actions()

    def undoShapeEdit(self):
        self._restore_history(self._history.undo())

    def redoShapeEdit(self):
        self._restore_history(self._history.redo())

    def resetState(self):
        self._frame_revision += 1
        if self._job:
            self._job.cancel()
        self.sam2_prompt_type = None
        self._refinement_token = self._refinement_source_id = None
        super().resetState()
        self._history.reset([])
        if not self._initializing:
            self._refresh_views()

    def _install_video_entry(self):
        """File > Open video... (also works without a project: one is created)."""
        action = QtWidgets.QAction("Open video...", self)
        action.triggered.connect(self._import_video_dialog)
        file_actions = self.menus.file.actions()
        self.menus.file.insertAction(file_actions[1] if len(file_actions) > 1 else None, action)
        self.actions_open_video = action

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(is_video(u.toLocalFile()) for u in urls):
            event.accept()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        videos = [u.toLocalFile() for u in event.mimeData().urls() if is_video(u.toLocalFile())]
        if videos:
            event.accept()
            if len(videos) > 1:
                self.statusBar().showMessage(
                    "Importing the first dropped video; drop the others one at a time.", 8000)
            self._import_video_dialog(path=videos[0])
            return
        super().dropEvent(event)

    def loadFile(self, filename=None):
        if filename and not self._initializing and is_video(filename) and Path(filename).is_file():
            # File > Open, recent files, drag-and-drop, file list: videos go to the importer.
            QtCore.QTimer.singleShot(0, lambda: self._import_video_dialog(path=str(filename)))
            return False
        if filename and Path(filename).is_file() and not self._initializing:
            try:
                frames = frame_count(filename)
            except Exception:
                frames = 1
            if frames > 1:
                return self._offer_split_stack(filename, frames)
        if self._project is not None and not self._initializing:
            previous = self.imagePath
            try:
                result = self._load_project_image(filename)
                if not result:
                    self.filename = previous
                return result
            except (OSError, ValueError, TypeError) as exc:
                self.filename = previous
                QtWidgets.QMessageBox.warning(self, "Cannot open project image", str(exc))
                return False
        result = super().loadFile(filename)
        self._history.reset(self.canvas.shapes)
        if not self._initializing:
            self._refresh_views()
        return result

    def saveLabels(self, filename):
        if self._project is not None:
            return self._save_project()
        # Serialize only final objects. Own I/O also closes/flushed files
        # explicitly; legacy upstream LabelFile.open lacks that guarantee.
        self._refresh_views()
        target = Path(filename).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shapes = []
        for shape in self.canvas.shapes:
            if self._is_temporary(shape):
                continue
            data = dict(shape.other_data or {})
            data.update(
                label=shape.label, points=[(p.x(), p.y()) for p in shape.points],
                group_id=shape.group_id, description=shape.description,
                shape_type=shape.shape_type, flags=shape.flags or {},
            )
            shapes.append(data)
        flags = {
            self.flag_widget.item(i).text():
            self.flag_widget.item(i).checkState() == QtCore.Qt.Checked
            for i in range(self.flag_widget.count())
        }
        from labelme import __version__
        image_path = os.path.relpath(self.imagePath, target.parent)
        payload = dict(self.otherData or {})
        payload.update(
            version=__version__, shapes=shapes, flags=flags, imagePath=image_path,
            imageData=base64.b64encode(self.imageData).decode("ascii")
            if self._config["store_data"] else None,
            imageHeight=self.image.height(), imageWidth=self.image.width(),
        )
        fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".json",
                                         dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            self.labelFile = LabelFile()
            self.labelFile.filename = str(target)
            self.labelFile.imagePath = image_path
            self.labelFile.imageData = self.imageData
            for item in self.fileListWidget.findItems(self.imagePath, QtCore.Qt.MatchExactly):
                item.setCheckState(QtCore.Qt.Checked)
            return True
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _saveFile(self, filename):
        try:
            return super()._saveFile(filename)
        except (OSError, ValueError, TypeError) as exc:
            # Qt slots must not leak exceptions (PyQt can abort the process).
            self.dirty = True
            self.actions.save.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return False

    def saveFile(self, _value=False):
        if self._project is not None:
            return self._save_project()
        return super().saveFile(_value)

    def saveFileAs(self, _value=False):
        if self._project is not None:
            return self._save_project_as()
        return super().saveFileAs(_value)

    def importDirImages(self, dirpath, pattern=None, load=True):
        if self._project is not None:
            paths = sorted(str(p) for p in Path(dirpath).iterdir()
                           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
            return self._project_safe(lambda: self._add_project_images(paths))
        return super().importDirImages(dirpath, pattern, load)

    def mayContinue(self):
        if self._project is None:
            return super().mayContinue()
        try:
            self._capture_project_image()
        except (OSError, ValueError, TypeError):
            self._project_dirty = True
        if not (self.dirty or self._project_dirty):
            return True
        mb = QtWidgets.QMessageBox
        answer = mb.question(self, "Save project?",
                             "Save the project before leaving? Temporary SAM prompts/candidates are not saved.",
                             mb.Save | mb.Discard | mb.Cancel, mb.Save)
        if answer == mb.Save:
            return self._save_project()
        return answer == mb.Discard

    def closeEvent(self, event):
        # No QThread is destroyed while its request is still running. Timeout
        # limits are finite; model computation is in a separate process.
        if self._io_busy():
            for job in self._io_jobs_list():
                if job is not getattr(self, "_autosave_job", None):
                    job.cancel()
            self.statusBar().showMessage(
                "File operation in progress. Close again after it finishes.", 8000)
            event.ignore()
            return
        if any(job.isRunning() for job in self._jobs):
            for job in self._jobs:
                job.cancel()
            self.statusBar().showMessage(
                "Pending result cancelled. Close again after the request finishes.", 8000)
            event.ignore()
            return
        super().closeEvent(event)
