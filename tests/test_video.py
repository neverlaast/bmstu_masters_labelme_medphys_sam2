"""Video import: decoding, frame selection, sidecar, project integration."""
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

import cv2
import numpy as np
import yaml
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets

from app.core.video import (FrameRange, VideoError, default_frames_dir, extract_frames,
                            is_video, probe_video, SIDECAR)


def make_video(path, frames=12, size=(64, 48), fps=10.0):
    """A dark disc moving right by 3 px per frame; MJPG is available everywhere."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot write MJPG video here")
    for i in range(frames):
        img = np.full((size[1], size[0], 3), 220, np.uint8)
        cv2.circle(img, (10 + 3 * i, 24), 6, (30, 30, 30), -1)
        writer.write(img)
    writer.release()
    return path


class VideoCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.video = make_video(Path(self.tmp.name) / "cells.avi")

    def test_probe(self):
        info = probe_video(self.video)
        self.assertEqual((info.frame_count, info.width, info.height), (12, 64, 48))
        self.assertAlmostEqual(info.fps, 10.0, places=1)
        self.assertAlmostEqual(info.duration_s, 1.2, places=1)

    def test_is_video_and_default_dir(self):
        self.assertTrue(is_video("a/B.MP4"))
        self.assertFalse(is_video("a/b.png"))
        self.assertEqual(default_frames_dir("/v/x.mp4", "/p/proj.medphys.json"),
                         Path("/p/x_frames").resolve())

    def test_extract_range_step_order_and_sidecar(self):
        out = Path(self.tmp.name) / "frames"
        calls = []
        paths = extract_frames(self.video, out, FrameRange(2, 9, 3),
                               progress=lambda d, t, _: calls.append((d, t)))
        self.assertEqual([p.name for p in paths],
                         ["cells_f000002.png", "cells_f000005.png", "cells_f000008.png"])
        self.assertEqual(calls[-1], (3, 3))
        meta = json.loads((out / SIDECAR).read_text())
        self.assertEqual([f["frame_index"] for f in meta["frames"]], [2, 5, 8])
        self.assertAlmostEqual(meta["frames"][1]["time_s"], 0.5)
        # frames are the right ones: the disc centre moves 3 px/frame
        xs = [np.nonzero(np.asarray(Image.open(p).convert("L"))[24] < 100)[0].mean() for p in paths]
        np.testing.assert_allclose(np.diff(xs), [9, 9], atol=1.5)

    def test_invalid_inputs(self):
        with self.assertRaises(VideoError):
            extract_frames(self.video, self.tmp.name, FrameRange(5, 2))
        with self.assertRaises(VideoError):
            extract_frames(self.video, self.tmp.name, FrameRange(step=0))
        bad = Path(self.tmp.name) / "bad.mp4"
        bad.write_bytes(b"not a video")
        with self.assertRaises(VideoError):
            probe_video(bad)

    def test_cancel(self):
        with self.assertRaises(InterruptedError):
            extract_frames(self.video, Path(self.tmp.name) / "c", cancelled=lambda: True)


class VideoGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from app.gui.labelme_window import MedPhysLabelMeWindow
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.video = make_video(self.dir / "cells.avi")
        config = yaml.safe_load((PROJECT_ROOT / "third_party" / "labelme_upstream" /
                                 "labelme" / "config" / "default_config.yaml").read_text())
        config.update(auto_save=False, store_data=False, keep_prev=False)
        self.window = MedPhysLabelMeWindow(config=config)
        self.box = patch.multiple(QtWidgets.QMessageBox, warning=self._fail_box,
                                  information=lambda *a, **k: None)
        self.box.start()
        self.addCleanup(self.box.stop)

    def tearDown(self):
        self.window._project_timer.stop()
        self.window.deleteLater()
        self.application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.application.processEvents()

    def _fail_box(self, _parent, title, text, *a, **k):
        raise AssertionError(f"{title}: {text}")

    def wait_io(self):
        deadline = time.monotonic() + 10
        while self.window._io_busy() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.01)
        for _ in range(5):
            self.application.processEvents()

    def _menu(self, title):
        menu = next(a.menu() for a in self.window.menuBar().actions()
                    if a.text().replace("&", "") == title)
        return [a.text() for a in menu.actions() if a.text()]

    def test_menu_entries_exist_once(self):
        self.assertEqual(self._menu("Project").count("Import video..."), 1)
        file_menu = self._menu("File")
        self.assertEqual(file_menu.count("Open video..."), 1)
        self.assertEqual(file_menu.index("Open video..."), 1)  # right after Open

    def test_open_video_without_project_routes_to_importer(self):
        self.assertIsNone(self.window._project)
        with patch.object(self.window, "_import_video_dialog") as importer:
            self.assertFalse(self.window.loadFile(str(self.video)))
            self.application.processEvents()
        importer.assert_called_once_with(path=str(self.video))

    def test_drop_video(self):
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(self.video))])
        event = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.CopyAction, mime,
                                 QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
        with patch.object(self.window, "_import_video_dialog") as importer:
            self.window.dropEvent(event)
        importer.assert_called_once_with(path=str(self.video))

    def test_import_creates_project_with_ordered_frames(self):
        project = self.dir / "cells.medphys.json"
        with patch.object(self.window, "mayContinue", return_value=True):
            self.window.import_video(self.video, FrameRange(0, 11, 2), self.dir / "fr",
                                     project_path=str(project))
            self.wait_io()
        p = self.window._project
        self.assertIsNotNone(p)
        self.assertEqual([Path(i.path).name for i in p.images],
                         [f"cells_f{n:06d}.png" for n in range(0, 12, 2)])
        self.assertEqual(p.journal[-1]["action"], "video_imported")
        self.assertEqual(p.journal[-1]["frames"], 6)
        self.assertEqual(Path(self.window.imagePath).name, "cells_f000000.png")
        # second video appended to the same project keeps order after existing frames
        second = make_video(self.dir / "more.avi", frames=3)
        self.window.import_video(second, FrameRange(), self.dir / "fr2")
        self.wait_io()
        self.assertEqual(len(self.window._project.images), 9)
        self.assertEqual(Path(self.window._project.images[-1].path).name, "more_f000002.png")

    def test_dialog_defaults_fit_propagation_limit(self):
        from app.core.video import VideoInfo
        from app.gui.video_import_dialog import VideoImportDialog
        info = VideoInfo("x.mp4", 1000, 25.0, 64, 48)
        dialog = VideoImportDialog(info, self.dir / "f")
        self.assertEqual(dialog.step.value(), 4)
        self.assertEqual(dialog.count(), 250)
        dialog.step.setValue(1)
        self.assertEqual(dialog.count(), 1000)
        self.assertTrue(dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled())
        dialog.start.setValue(10); dialog.end.setValue(5)
        self.assertFalse(dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled())


if __name__ == "__main__":
    unittest.main()
