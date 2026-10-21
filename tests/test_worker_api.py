import unittest
from unittest.mock import Mock, patch

import requests
from pydantic import ValidationError

from app.worker_api.client import WorkerClient
from app.worker_api.errors import WorkerConnectionError, WorkerResponseError
from app.worker_api.schemas import BoxPrompt, CirclePrompt, PointPrompt, SegmentationRequest


class APITests(unittest.TestCase):
    def test_invalid_geometry_rejected(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.subTest(radius=value), self.assertRaises(ValidationError):
                CirclePrompt(center_x=0, center_y=0, radius=value)
        with self.assertRaises(ValidationError):
            BoxPrompt(x1=10, y1=0, x2=5, y2=10)

    def test_combined_roi_and_points(self):
        request = SegmentationRequest(
            image_path="test.png", circle=CirclePrompt(center_x=30, center_y=30, radius=20),
            points=[PointPrompt(x=10, y=10, label=0)])
        self.assertEqual(request.points[0].label, 0)
        with self.assertRaises(ValidationError):
            SegmentationRequest(image_path="test.png")

    @patch("app.worker_api.client.requests.post")
    def test_worker_error_message_preserved(self, post):
        response = Mock(ok=False)
        response.json.return_value = {
            "ok": False, "model_name": "sam2", "model_version": "sam2.1",
            "device": "cpu", "elapsed_ms": 0, "candidates": [],
            "error_message": "Checkpoint not found",
        }
        post.return_value = response
        with self.assertRaisesRegex(WorkerResponseError, "Checkpoint not found"):
            WorkerClient().segment(SegmentationRequest(
                image_path="test.png", points=[PointPrompt(x=1, y=1, label=1)]))

    @patch("app.worker_api.client.requests.post", side_effect=requests.ConnectionError("offline"))
    def test_connection_failure_has_actionable_message(self, post):
        with self.assertRaisesRegex(WorkerConnectionError, "Restart worker"):
            WorkerClient().segment(SegmentationRequest(
                image_path="test.png", points=[PointPrompt(x=1, y=1, label=1)]))

    @patch("app.worker_api.client.requests.post", side_effect=requests.ReadTimeout("slow"))
    def test_timeout_is_not_misreported_as_missing_worker(self, post):
        with self.assertRaisesRegex(WorkerConnectionError, "timed out"):
            WorkerClient().segment(SegmentationRequest(
                image_path="test.png", points=[PointPrompt(x=1,y=1,label=1)]))

    def test_circle_conversion_includes_background_and_explicit_points(self):
        from worker.main import segmentation_request_to_sam_request
        request = SegmentationRequest(
            image_path="test.png", circle=CirclePrompt(center_x=30, center_y=40, radius=10),
            points=[PointPrompt(x=3, y=4, label=0)])
        converted = segmentation_request_to_sam_request(request)
        self.assertEqual(converted.box.x_min, 20)
        self.assertEqual(converted.box.y_max, 50)
        self.assertEqual(sum(p.label == 1 for p in converted.points), 5)
        self.assertEqual(sum(p.label == 0 for p in converted.points), 5)

    def test_box_keeps_explicit_points(self):
        from worker.main import segmentation_request_to_sam_request
        request = SegmentationRequest(
            image_path="test.png", box=BoxPrompt(x1=0, y1=0, x2=20, y2=20),
            points=[PointPrompt(x=3, y=4, label=0)])
        converted = segmentation_request_to_sam_request(request)
        self.assertIsNotNone(converted.box)
        self.assertEqual(len(converted.points), 1)

    def test_health_does_not_load_model(self):
        from worker import main as server
        from worker.model_registry import RealSam2ModelAdapter
        adapter = RealSam2ModelAdapter(checkpoint_path="missing.pt", config_path="missing.yaml")
        with patch.object(server, "MODEL_ADAPTER", adapter), patch.object(adapter, "load") as load:
            payload = server.build_health_payload()
        load.assert_not_called()
        self.assertFalse(payload.model_loaded)
