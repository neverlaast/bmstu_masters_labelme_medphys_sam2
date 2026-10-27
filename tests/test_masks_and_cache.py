import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from app.worker_api.schemas import GenerationOptions, CandidateFilters
from worker.cache.embedding_cache import EmbeddingCache
from worker.model_registry import SamInferenceRequest, SamBox, SamModelInfo, SamPoint
from worker.postprocess.masks import PackedMask, mask_iou, roi_mask, grid_points, metrics, mask_rle, polygon_mask
from worker.predictors.sam2_predictor import predict


class CacheTests(unittest.TestCase):
    def test_lru_and_memory_budget(self):
        cache = EmbeddingCache(max_bytes=10, max_items=2)
        cache.put("a", np.zeros(5, np.uint8))
        cache.put("b", np.zeros(5, np.uint8))
        cache.get("a")
        cache.put("c", np.zeros(5, np.uint8))
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))
        self.assertLessEqual(cache.bytes, 10)
        self.assertFalse(cache.put("huge", np.zeros(11, np.uint8)))
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.bytes, 0)


class MaskTests(unittest.TestCase):
    def test_cropped_polygons_match_full_raster_and_exact_iou(self):
        rng = np.random.default_rng(4)
        shape = (73, 91)
        polygons = [
            [[-10, -5], [20, 0], [10, 25]],
            [[12, 20], [40, 20], [40, 45], [12, 45]],
            [[-30, -30], [-20, -20], [-10, -30]],
            [],
        ] + [rng.integers(-15, 105, size=(6, 2)).tolist() for _ in range(30)]
        for points in polygons:
            original = PackedMask.from_mask(polygon_mask(shape, points))
            cropped = PackedMask.from_polygon(shape, points)
            np.testing.assert_array_equal(cropped.unpack(), original.unpack())
            self.assertEqual(cropped.area, original.area)
            self.assertEqual(cropped.bbox, original.bbox)
            other_points = [[10, 10], [65, 15], [60, 55], [5, 45]]
            other_full = PackedMask.from_mask(polygon_mask(shape, other_points))
            other_crop = PackedMask.from_polygon(shape, other_points)
            expected = mask_iou(original, other_full)
            self.assertAlmostEqual(mask_iou(cropped, other_full), expected)
            self.assertAlmostEqual(mask_iou(original, other_crop), expected)
            self.assertAlmostEqual(mask_iou(cropped, other_crop), expected)

    def test_polygon_storage_does_not_scale_with_full_image_area(self):
        shape = (100000, 100000)
        a = PackedMask.from_polygon(shape, [[30, 40], [40, 40], [40, 50], [30, 50]])
        b = PackedMask.from_polygon(shape, [[35, 45], [45, 45], [45, 55], [35, 55]])
        self.assertEqual(a.area, 121)
        self.assertLess(a.bits.nbytes, 32)
        self.assertAlmostEqual(mask_iou(a, b), 36/(121+121-36))

    def test_exact_mask_iou_not_bbox_iou(self):
        a = np.eye(10, dtype=bool)
        b = np.fliplr(a)
        self.assertEqual(mask_iou(PackedMask.from_mask(a), PackedMask.from_mask(b)), 0)
        self.assertEqual(mask_iou(PackedMask.from_mask(a), PackedMask.from_mask(a)), 1)

    def test_roi_grid_stays_inside_circle_and_polygon(self):
        for roi in [
            roi_mask((100,100), circle={"center_x":50,"center_y":50,"radius":30}),
            roi_mask((100,100), polygon=[[10,10],[90,10],[50,90]]),
        ]:
            seeds = grid_points(roi, 8)
            self.assertTrue(seeds)
            self.assertTrue(all(roi[int(y),int(x)] for x,y in seeds))
        self.assertFalse(roi_mask((100,100), box=[-20,-20,-1,-1]).any())

    def test_grid_does_not_repeat_pixel_in_tiny_roi(self):
        roi = np.zeros((10,10), bool)
        roi[5,5] = True
        self.assertEqual(grid_points(roi, 20), [(5.0,5.0)])

    def test_rle_round_trip_preserves_hole(self):
        mask = np.zeros((32,32), bool)
        mask[5:25,5:25] = True
        mask[10:20,10:20] = False
        rle = mask_rle(mask)
        values = []
        for index, count in enumerate(rle["counts"]):
            values.extend([index % 2]*count)
        restored = np.asarray(values).reshape(rle["size"], order="F").astype(bool)
        np.testing.assert_array_equal(mask, restored)

    def test_metrics_and_filter_validation(self):
        mask = np.zeros((64,64), np.uint8)
        cv2.circle(mask, (32,32), 10, 1, -1)
        values = metrics(mask, np.ones_like(mask, bool))
        self.assertGreater(values["roundness"], .75)
        self.assertAlmostEqual(values["aspect_ratio"], 1)
        self.assertFalse(values["touches_image_border"])
        with self.assertRaises(ValueError):
            CandidateFilters(min_area=100, max_area=10)


class FakePredictor:
    model = SimpleNamespace(image_size=1024)

    def __init__(self):
        self.set_count = 0
        self.last_mask_input = None
        self._features = None

    def set_image(self, image):
        self.set_count += 1
        self._orig_hw = [image.shape[:2]]
        self._features = {"image_embed": np.zeros((1,1), np.float32)}
        self._is_image_set, self._is_batch = True, False

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, **kwargs):
        self.last_mask_input = mask_input
        mask = np.ones((128,128), np.uint8)
        if box is not None or mask_input is not None:
            centre = (32,32)
        else:
            x,y = point_coords[0]
            centres = [(32,32),(90,90)]
            centre = next((c for c in centres if (x-c[0])**2+(y-c[1])**2 <= 12**2), None)
        if centre:
            mask[:] = 0
            cv2.circle(mask, centre, 12, 1, -1)
        masks = np.repeat(mask[None], 3, axis=0).astype(bool)
        return masks, np.asarray([.8,.9,.7]), np.ones((3,256,256), np.float32)


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.image = Path(self.directory.name) / "image.png"
        Image.new("RGB", (128,128), "white").save(self.image)
        self.adapter = SimpleNamespace(
            device="cpu", _model_identity="test-model", _predictor=FakePredictor(),
            _embedding_cache=EmbeddingCache(), _mask_inputs=OrderedDict(),
            get_info=lambda: SamModelInfo(backend="sam2_real", model_name="test", loaded=True),
        )

    def tearDown(self):
        self.directory.cleanup()

    def request(self, auto=False, **params):
        options = GenerationOptions(
            mode="automatic" if auto else "interactive", points_per_side=8,
            filters=CandidateFilters(max_area=1000, min_roundness=.5))
        return SamInferenceRequest(
            image_path=str(self.image), prompt_type="auto" if auto else "box",
            box=None if auto else SamBox(15,15,50,50),
            generation_params={
                "generation": options.model_dump(),
                "roi": {"circle":{"center_x":64,"center_y":64,"radius":60}},
                **params,
            },
        )

    def test_auto_detects_two_objects_and_suppresses_duplicates(self):
        result = predict(self.adapter, self.request(auto=True))
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.generation_mode, "automatic")
        self.assertTrue(all(c.score == .9 for c in result.candidates))
        self.assertTrue(all(c.metadata["metrics"]["area_px"] > 100 for c in result.candidates))

    def test_embedding_reuse_and_invalidation(self):
        first = predict(self.adapter, self.request())
        second = predict(self.adapter, self.request())
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(self.adapter._predictor.set_count, 1)
        Image.new("RGB", (128,128), "black").save(self.image)
        self.assertFalse(predict(self.adapter, self.request()).cache_hit)
        self.assertEqual(self.adapter._predictor.set_count, 2)
        self.adapter._model_identity = "different-model"
        self.assertFalse(predict(self.adapter, self.request()).cache_hit)

    def test_explicit_refinement_passes_low_resolution_logits(self):
        first = predict(self.adapter, self.request())
        token = first.candidates[0].metadata["refinement_token"]
        result = predict(self.adapter, self.request(refinement_token=token, refinement_target_id="object-1"))
        self.assertTrue(result.cache_hit)
        self.assertEqual(self.adapter._predictor.last_mask_input.shape, (1,256,256))
        self.assertEqual(result.candidates[0].metadata["refines_object_id"], "object-1")
        Image.new("RGB", (128,128), "black").save(self.image)
        with self.assertRaisesRegex(ValueError, "expired"):
            predict(self.adapter, self.request(refinement_token=token))

    def test_existing_overlap_is_reported(self):
        result = predict(self.adapter, self.request(
            auto=True, existing_polygons=[[(20,20),(44,20),(44,44),(20,44)]]))
        self.assertTrue(result.warnings)
        self.assertTrue(any(c.metadata["metrics"]["overlap_warning"] for c in result.candidates))
