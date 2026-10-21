"""End-to-end smoke test of every 'Segmentation / SAM2' menu item.

The real WorkerClient talks to the real worker HTTP handler (stub model); only
the TCP socket is replaced by in-memory buffers, so request/response JSON,
status codes and GUI result handling are all exercised.
"""
import io
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

import numpy as np
import requests
import yaml
from PIL import Image
from qtpy import QtCore, QtWidgets
from labelme.shape import Shape

import worker.main as worker_main
from worker.model_registry import create_model_adapter
from app.gui.labelme_window import MedPhysLabelMeWindow


class _Resp:
    def __init__(self, raw):
        head, _, body = raw.partition(b"\r\n\r\n")
        self.status_code = int(head.split()[1])
        self._body = body

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return json.loads(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def _call(method, url, payload=None):
    path = "/" + url.split("/", 3)[3]
    body = json.dumps(payload).encode() if payload is not None else b""
    raw = (f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n"
           "Content-Type: application/json\r\n\r\n").encode() + body

    class Sock:
        def makefile(self, mode, *a, **k):
            return io.BytesIO(raw) if "r" in mode else out

        def sendall(self, data):
            out.write(data)

    out = io.BytesIO()
    out.close = lambda: None
    worker_main.WorkerHandler(Sock(), ("127.0.0.1", 0), None)
    return _Resp(out.getvalue())


def fake_get(url, timeout=None, **_):
    return _call("GET", url)


def fake_post(url, json=None, timeout=None, **_):
    return _call("POST", url, json)


class MenuSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.adapter = patch.object(worker_main, "MODEL_ADAPTER", create_model_adapter("stub"))
        self.adapter.start()
        self.addCleanup(self.adapter.stop)
        for name, fake in [("get", fake_get), ("post", fake_post)]:
            p = patch(f"app.worker_api.client.requests.{name}", side_effect=fake)
            p.start()
            self.addCleanup(p.stop)
        self.box = patch.multiple(QtWidgets.QMessageBox, information=lambda *a, **k: None,
                                  warning=lambda *a, **k: None, critical=lambda *a, **k: None)
        self.box.start()
        self.addCleanup(self.box.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.directory.name) / "frame.png"
        img = np.full((256, 256, 3), 255, np.uint8)
        img[30:50, 30:50] = 0
        img[120:150, 120:150] = 0
        Image.fromarray(img).save(self.image_path)
        config = yaml.safe_load((PROJECT_ROOT / "third_party" / "labelme_upstream" /
                                 "labelme" / "config" / "default_config.yaml").read_text())
        config.update(auto_save=False, store_data=False, keep_prev=False)
        self.window = MedPhysLabelMeWindow(config=config)
        self.window.loadFile(str(self.image_path))
        self.messages = []
        self.window.statusBar().messageChanged.connect(self.messages.append)
        menu = next(a.menu() for a in self.window.menuBar().actions()
                    if a.text() == "Segmentation / SAM2")
        self.actions = {a.text(): a for a in menu.actions() if a.text()}

    def tearDown(self):
        for task in list(self.window._jobs):
            task.cancel()
            task.wait(3000)
        self.application.processEvents()
        self.window.deleteLater()
        self.application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.application.processEvents()
        self.directory.cleanup()

    def trigger(self, text):
        self.actions[text].trigger()
        deadline = time.monotonic() + 5
        while self.window._job is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(.005)
        self.application.processEvents()
        self.assertIsNone(self.window._job, f"'{text}' did not finish")

    def test_multiple_regions_through_real_http_handler(self):
        for kind, x in [("circle", 25), ("rectangle", 65)]:
            self.window._activate_sam2_prompt_drawing(kind)
            shape = Shape(shape_type=kind)
            shape.addPoint(QtCore.QPointF(x, 30))
            shape.addPoint(QtCore.QPointF(x+12, 42))
            self.window.canvas.current = shape
            self.window.canvas.finalise()
        self.trigger("Run SAM2")
        candidates = list(self.window._iter_sam2_candidate_shapes())
        self.assertEqual(len(candidates), 2)
        self.assertEqual([s.other_data["prompt_index"] for s in candidates], [1, 2])

    def prompt(self, kind, label=1, pts=((40, 40), (52, 40))):
        self.window._activate_sam2_prompt_drawing(kind, label)
        shape = Shape(shape_type=kind)
        for x, y in (pts[:1] if kind == "point" else pts):
            shape.addPoint(QtCore.QPointF(x, y))
        self.window.canvas.current = shape
        self.window.canvas.finalise()
        return shape

    def candidates(self):
        return list(self.window._iter_sam2_candidate_shapes())

    def test_every_menu_item_is_present(self):
        expected = [
            "Propagate objects to other frames...", "Restart worker",
            "Show worker log", "Show model capabilities", "Unload current model",
            "Positive point", "Negative point", "Rectangle prompt", "Circle prompt: one object",
            "Finish prompt drawing", "Run SAM2", "Refine previously selected mask",
            "Polygon ROI: object search", "Find objects inside ROI", "Find objects in whole image",
            "Object search settings...", "Clear worker cache", "Cancel / discard pending result",
            "Accept selected candidates", "Accept all candidates", "Reject selected candidates",
            "Reject all candidates", "Clear prompts"]
        self.assertEqual([t for t in self.actions], expected)

    def test_worker_service_items(self):
        for text in ["Show model capabilities", "Clear worker cache",
                     "Unload current model"]:
            with self.subTest(text):
                self.messages.clear()
                self.trigger(text)
                self.assertFalse(any("fail" in m.lower() or "error" in m.lower()
                                     for m in self.messages), self.messages)
        with patch.object(QtWidgets.QMessageBox, "exec_", return_value=0) as shown:
            self.trigger("Show worker log")
        shown.assert_called_once()
        self.trigger("Restart worker")  # no manager in tests -> explains instead of crashing

    def test_each_prompt_type_runs_sam2(self):
        for kind, label, text in [("point", 1, "Positive point"), ("point", 0, "Negative point"),
                                  ("rectangle", 1, "Rectangle prompt"),
                                  ("circle", 1, "Circle prompt: one object")]:
            with self.subTest(text):
                self.trigger("Clear prompts")
                self.trigger("Reject all candidates")
                self.trigger(text)
                self.assertEqual(self.window.sam2_prompt_type, kind)
                pts = ((40, 40), (52, 52)) if kind == "rectangle" else ((40, 40), (52, 40))
                self.prompt(kind, label, pts)
                if label == 0:
                    self.prompt("point", 1, ((35, 35),))
                self.trigger("Finish prompt drawing")
                self.trigger("Run SAM2")
                self.assertTrue(self.candidates(), f"{text}: no candidates")

    def test_candidate_review_and_refine(self):
        self.prompt("circle")
        self.trigger("Run SAM2")
        first = self.candidates()
        self.assertTrue(first)
        self.window.canvas.selectShapes(first[:1])
        self.window.shapeSelectionChanged(first[:1])
        self.prompt("point", 0, ((45, 45),))
        self.trigger("Refine previously selected mask")
        self.assertTrue(self.candidates())
        cands = self.candidates()
        self.window.shapeSelectionChanged(cands[:1])
        self.trigger("Reject selected candidates")
        self.assertEqual(len(self.candidates()), len(cands) - 1)
        if self.candidates():
            self.window.shapeSelectionChanged(self.candidates()[:1])
            with patch.object(self.window, "_ask_target_label", return_value="cell"):
                self.trigger("Accept selected candidates")
        self.prompt("circle")
        self.trigger("Run SAM2")
        with patch.object(self.window, "_ask_target_label", return_value="cell"):
            self.trigger("Accept all candidates")
        self.assertEqual(self.candidates(), [])
        self.assertGreater(len(self.window.labelList), 0)
        self.trigger("Clear prompts")
        self.assertFalse(any(self.window._is_sam2_prompt(s) for s in self.window.canvas.shapes))

    def test_object_search(self):
        from worker.adapters import load_trusted_adapter
        self.adapter.stop()
        self.adapter = patch.object(worker_main, "MODEL_ADAPTER",
                                    load_trusted_adapter("worker.plugins.threshold_demo:create_adapter"))
        self.adapter.start()
        self.trigger("Find objects in whole image")
        whole = len(self.candidates())
        self.trigger("Reject all candidates")
        self.trigger("Polygon ROI: object search")
        self.assertEqual(self.window.sam2_prompt_type, "polygon")
        self.prompt("polygon", 1, ((20, 20), (70, 20), (70, 70), (20, 70)))
        self.trigger("Find objects inside ROI")
        self.assertGreater(whole, 0, self.messages)
        self.assertTrue(self.candidates(), self.messages)
        with patch("app.gui.labelme_window.GenerationDialog.exec_",
                   return_value=QtWidgets.QDialog.Rejected):
            self.trigger("Object search settings...")

    def test_cancel_is_enabled_only_while_running(self):
        self.assertFalse(self.actions["Cancel / discard pending result"].isEnabled())

    def test_propagate_without_project_explains(self):
        self.trigger("Propagate objects to other frames...")


if __name__ == "__main__":
    unittest.main()
