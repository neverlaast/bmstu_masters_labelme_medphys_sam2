"""Regression tests for export validation, raster integrity and publication."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.core.export import (
    ExportValidationError, FORMATS, annotation_mask, decode_rle,
    export_image, validate_annotations,
)
from worker.postprocess.masks import mask_rle


def polygon(identity="one", label="cell", points=None):
    return {"object_id": identity, "label": label, "shape_type": "polygon",
            "points": points or [[2, 2], [12, 2], [12, 12], [2, 12]],
            "flags": {}, "group_id": None}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.image = self.root / "frame.png"
        Image.new("RGB", (32, 32), "white").save(self.image)
        self.original = self.image.read_bytes()

    def tearDown(self):
        self.assertEqual(self.image.read_bytes(), self.original)
        self.directory.cleanup()

    def export(self, format_name, annotations=None, classes=None):
        return export_image(self.root / "exports", self.image,
                            [polygon()] if annotations is None else annotations,
                            32, 32, format_name, classes)

    def test_all_formats_have_report_and_class_map(self):
        for format_name in FORMATS:
            with self.subTest(format_name=format_name):
                target, report = self.export(format_name)
                self.assertEqual(report["objects"], 1)
                self.assertTrue((target / "export_report.json").is_file())
                self.assertEqual(json.loads((target / "classes.json").read_text())[0]["yolo_id"], 0)
                self.assertEqual((target / "images/frame.png").read_bytes(), self.original)

    def test_self_intersection_and_bounds_block_export_before_directory_creation(self):
        bad = polygon(points=[[2, 2], [33, 20], [2, 20], [20, 2]])
        with self.assertRaises(ExportValidationError) as error:
            self.export("coco", [bad])
        self.assertIn("outside", str(error.exception))
        self.assertIn("Self-intersecting", str(error.exception))
        self.assertFalse((self.root / "exports").exists())

    def test_invalid_classes_ids_and_coordinates(self):
        a, b = polygon(), polygon(label="other")
        issues, _ = validate_annotations([a, b], 32, 32, ["cell"])
        self.assertTrue(any("Unknown class" in issue for issue in issues))
        self.assertTrue(any("Duplicate object ID" in issue for issue in issues))
        a["object_id"] = None
        a["points"][0][0] = float("nan")
        issues, _ = validate_annotations([a], 32, 32)
        self.assertTrue(any("non-finite" in issue for issue in issues))
        self.assertTrue(any("invalid object ID" in issue for issue in issues))

    def test_identical_raster_with_different_vertices_is_duplicate(self):
        a = polygon()
        b = polygon("two", points=[[2, 2], [7, 2], [12, 2], [12, 12], [2, 12]])
        issues, _ = validate_annotations([a, b], 32, 32)
        self.assertTrue(any("Duplicate mask" in issue for issue in issues))

    def test_degenerate_and_unsupported_shapes(self):
        for shape in [
            dict(polygon(), points=[[2, 2], [2, 2], [3, 3]]),
            dict(polygon(), points=[[2, 2], [2, 12]], shape_type="rectangle"),
            dict(polygon(), points=[[2, 2]], shape_type="point"),
            dict(polygon(), points=[[2, 2], [5, 2]], shape_type="circle"),
        ]:
            self.assertTrue(validate_annotations([shape], 32, 32)[0])

    def test_rectangle_circle_and_raster_measurements(self):
        rectangle = dict(polygon(), points=[[2, 2], [12, 12]], shape_type="rectangle")
        circle = dict(polygon("two"), points=[[24, 24], [27, 24]], shape_type="circle")
        errors, masks = validate_annotations([rectangle, circle], 32, 32)
        self.assertEqual(errors, [])
        self.assertEqual(int(masks[0].sum()), 121)
        self.assertEqual(int(masks[1].sum()), 29)

    def hole(self):
        mask = np.zeros((32, 32), bool)
        mask[2:13, 2:13] = True
        mask[5:10, 5:10] = False
        annotation = polygon()
        annotation.update(mask_rle=mask_rle(mask), mask_outline=annotation["points"],
                          raster_source_valid=True)
        return annotation, mask

    def test_coco_rle_preserves_holes_and_uses_pixel_bbox(self):
        annotation, expected = self.hole()
        target, _ = self.export("coco", [annotation])
        obj = json.loads((target / "annotations.json").read_text())["annotations"][0]
        np.testing.assert_array_equal(decode_rle(obj["segmentation"], 32, 32), expected)
        self.assertEqual(obj["area"], 96)
        self.assertEqual(obj["bbox"], [2, 2, 11, 11])
        self.assertEqual(obj["iscrowd"], 0)

    def test_yolo_rejects_holes_without_leaving_files(self):
        with self.assertRaises(ExportValidationError):
            self.export("yolo", [self.hole()[0]])
        self.assertFalse((self.root / "exports").exists())

    def test_modified_outline_does_not_export_stale_raster(self):
        annotation, old_mask = self.hole()
        annotation["points"] = [[15, 15], [20, 15], [20, 20], [15, 20]]
        mask = annotation_mask(annotation, 32, 32)
        self.assertEqual(int(mask.sum()), 36)
        self.assertFalse(np.any(mask & old_mask))

    def test_bad_rle_and_empty_mask_block_export(self):
        annotation, _ = self.hole()
        annotation["mask_rle"]["counts"] = [1023]
        self.assertTrue(validate_annotations([annotation], 32, 32)[0])
        annotation["mask_rle"]["counts"] = [1024]
        self.assertIn("Empty mask", "\n".join(validate_annotations([annotation], 32, 32)[0]))

    def test_yolo_normalized_coordinates_and_class_order(self):
        target, _ = self.export("yolo", classes=["particle", "cell"])
        numbers = (target / "labels/frame.txt").read_text().split()
        self.assertEqual(numbers[0], "1")
        self.assertAlmostEqual(float(numbers[1]), 2/32)
        self.assertTrue(all(0 <= float(value) < 1 for value in numbers[1:]))

    def test_palette_png_preserves_class_indices(self):
        target, _ = self.export("masks", classes=["particle", "cell"])
        with Image.open(target / "indexed_mask.png") as image:
            self.assertEqual(image.mode, "P")
            self.assertEqual(image.getpixel((5, 5)), 2)
            self.assertEqual(image.getpixel((0, 0)), 0)

    def test_overlapping_masks_not_silently_overwritten_and_binary_export_allowed(self):
        objects = [polygon(), polygon("two", points=[[8, 8], [18, 8], [18, 18], [8, 18]])]
        with self.assertRaises(ExportValidationError):
            self.export("masks", objects)
        self.assertEqual(list((self.root / "exports").iterdir()), [])
        target, _ = self.export("binary_masks", objects)
        self.assertEqual(len(list((target / "masks").glob("*.png"))), 2)
        self.assertFalse((target / "indexed_mask.png").exists())

    def test_failed_write_cleans_staging_and_success_never_overwrites(self):
        with patch("app.core.export._json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.export("coco")
        self.assertEqual(list((self.root / "exports").iterdir()), [])
        one, _ = self.export("coco")
        original = (one / "annotations.json").read_bytes()
        two, _ = self.export("coco")
        self.assertNotEqual(one, two)
        self.assertEqual((one / "annotations.json").read_bytes(), original)

    def test_empty_image_export_and_csv(self):
        target, report = self.export("csv", [])
        self.assertEqual(report["objects"], 0)
        self.assertEqual(len((target / "objects.csv").read_text(encoding="utf-8-sig").splitlines()), 1)
        target, _ = self.export("csv")
        text = (target / "objects.csv").read_text(encoding="utf-8-sig")
        self.assertIn("area_px", text)
        self.assertIn("121", text)

    def test_memory_budget_and_csv_formula_safety(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_annotations([polygon()], 30000, 30000)
        target, _ = self.export("csv", [polygon(label="=danger()")])
        text = (target / "objects.csv").read_text(encoding="utf-8-sig")
        self.assertIn("'=danger()", text)
