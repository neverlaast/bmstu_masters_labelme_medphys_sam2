"""The second backend must work through the same dispatch/HTTP schema."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from worker.adapters import (
    AdapterCapabilities, load_trusted_adapter, register_adapter,
    run_adapter, validate_adapter,
)
from worker.model_registry import InferenceRequest, RealSam2ModelAdapter, create_model_adapter
from app.worker_api.schemas import SegmentationRequest, GenerationOptions, PointPrompt
from worker.main import segmentation_request_to_sam_request, sam_response_to_segmentation_response


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.image = Path(self.directory.name) / "cells.png"
        image = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 30, 30), fill="black")
        draw.ellipse((60, 60, 80, 80), fill="black")
        image.save(self.image)
        self.adapter = load_trusted_adapter("worker.plugins.threshold_demo:create_adapter", device="cpu")

    def tearDown(self):
        self.directory.cleanup()

    def request(self):
        request = SegmentationRequest(image_path=str(self.image),
                                      generation=GenerationOptions(mode="automatic"))
        return segmentation_request_to_sam_request(request)

    def test_alternate_backend_full_dispatch_and_serialization(self):
        response = run_adapter(self.adapter, self.request())
        self.assertEqual(response.model_info.model_name, "threshold_demo")
        self.assertEqual(len(response.candidates), 2)
        payload = sam_response_to_segmentation_response(response)
        self.assertEqual(payload.model_name, "threshold_demo")
        self.assertEqual(payload.generation_mode, "automatic")
        self.assertIn("threshold_demo", payload.model_dump_json())
        self.assertTrue(self.adapter.loaded)
        self.adapter.unload()
        self.assertFalse(self.adapter.loaded)

    def test_registered_factory_and_reserved_names(self):
        import worker.adapters as registry
        with patch.dict(registry._FACTORIES, {}, clear=True):
            register_adapter("other_nn", lambda **kw: self.adapter)
            self.assertIs(create_model_adapter("other_nn"), self.adapter)
            with self.assertRaises(ValueError):
                register_adapter("other_nn", lambda: self.adapter)
            with self.assertRaises(ValueError):
                register_adapter("sam2_real", lambda: self.adapter)

    def test_unsupported_prompts_rejected_before_model_load(self):
        request = segmentation_request_to_sam_request(SegmentationRequest(
            image_path=str(self.image), points=[PointPrompt(x=20, y=20, label=1)]))
        with patch.object(self.adapter, "load") as load:
            with self.assertRaisesRegex(ValueError, "point prompts"):
                run_adapter(self.adapter, request)
        load.assert_not_called()

    def test_model_mismatch_and_invalid_plugins_rejected(self):
        request = self.request()
        request.model_name = "sam2"
        with self.assertRaisesRegex(ValueError, "differs"):
            run_adapter(self.adapter, request)
        with self.assertRaisesRegex(ValueError, "missing methods"):
            validate_adapter(object())
        for reference in ["", "/tmp/evil.py", "module:", "x:y:z"]:
            with self.assertRaises(ValueError):
                load_trusted_adapter(reference)

    def test_sam_contract_advertises_no_video_and_lazy_health(self):
        adapter = validate_adapter(RealSam2ModelAdapter(checkpoint_path="missing", config_path="missing"))
        self.assertTrue(adapter.get_capabilities().embeddings)
        self.assertFalse(adapter.get_capabilities().sequence)
        with self.assertRaises(NotImplementedError):
            adapter.process_sequence(None)
        from worker import main as server
        with patch.object(server, "MODEL_ADAPTER", adapter), patch.object(adapter, "load") as load:
            health = server.build_health_payload()
        load.assert_not_called()
        self.assertEqual(health.capabilities["automatic"], True)
        self.assertFalse(health.model_loaded)

    def test_neutral_records_and_capabilities_are_importable(self):
        from worker.model_registry import Point, Box, MaskCandidate, ModelInfo, InferenceResponse
        self.assertEqual(Point(1, 2, 1).x, 1)
        self.assertEqual(Box(0, 0, 10, 10).x_max, 10)
        self.assertTrue(AdapterCapabilities(automatic=True).to_dict()["automatic"])
