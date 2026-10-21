"""Import of COCO / YOLO annotations: round-trip with export and error reporting."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.export import annotation_mask, export_image, validate_annotations
from app.core.importers import (
    ImportFormatError, decode_compressed_rle, find_yolo_label, import_coco, import_yolo,
)


def square(label="cell", x=2, y=2, s=10):
    return {"object_id": label + str(x), "label": label, "shape_type": "polygon",
            "points": [[x, y], [x + s, y], [x + s, y + s], [x, y + s]],
            "flags": {}, "group_id": None}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "frame.png"
        Image.new("RGB", (32, 24), "white").save(self.image)

    def tearDown(self):
        self.tmp.cleanup()

    def masks(self, annotations):
        errors, masks = validate_annotations(annotations, 32, 24)
        self.assertEqual(errors, [])
        return masks

    def test_coco_rle_round_trip_is_pixel_exact(self):
        original = [square("cell", 2, 2), square("particle", 18, 8, 5)]
        target, _ = export_image(self.root / "out", self.image, original, 32, 24, "coco")
        report = import_coco(target / "annotations.json", "frame.png", 32, 24)
        self.assertEqual(len(report.annotations), 2)
        self.assertEqual(report.skipped, [])
        for a, b in zip(self.masks(original), self.masks(report.annotations)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual({a["label"] for a in report.annotations}, {"cell", "particle"})
        self.assertTrue(all(a["source"] == "import_coco" for a in report.annotations))

    def test_coco_polygon_bbox_and_bad_objects(self):
        data = {"images": [{"id": 7, "file_name": "sub/frame.png", "width": 32, "height": 24}],
                "categories": [{"id": 1, "name": "cell"}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 1,
                     "segmentation": [[1, 1, 10, 1, 10, 10, 1, 10]]},
                    {"id": 2, "image_id": 7, "category_id": 1, "bbox": [4, 4, 6, 3]},
                    {"id": 3, "image_id": 7, "category_id": 9, "bbox": [1, 1, 2, 2]},
                    {"id": 4, "image_id": 7, "category_id": 1, "segmentation": [[1, 1, 99, 1, 5, 5]]},
                    {"id": 5, "image_id": 8, "category_id": 1, "bbox": [1, 1, 2, 2]}]}
        path = self.root / "c.json"
        path.write_text(json.dumps(data))
        report = import_coco(path, "frame.png", 32, 24)
        self.assertEqual([a["shape_type"] for a in report.annotations], ["polygon", "rectangle"])
        self.assertEqual(report.annotations[1]["points"], [[4.0, 4.0], [9.0, 6.0]])
        self.assertEqual(len(report.skipped), 2)
        self.assertIn("category_id", report.skipped[0][1])

    def test_coco_size_mismatch_rejected(self):
        path = self.root / "c.json"
        path.write_text(json.dumps({"images": [{"id": 1, "file_name": "frame.png",
                                                "width": 64, "height": 64}],
                                    "annotations": [], "categories": []}))
        with self.assertRaises(ImportFormatError):
            import_coco(path, "frame.png", 32, 24)

    def test_compressed_rle_matches_reference(self):
        # pycocotools.mask.encode of a 4x4 mask with a 2x2 block at (1,1)
        mask = decode_compressed_rle("52203", 4, 4)
        expected = np.zeros((4, 4), bool)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(mask, expected)

    def test_multipart_polygon_kept_as_exact_raster(self):
        data = {"images": [{"id": 1, "file_name": "frame.png", "width": 32, "height": 24}],
                "categories": [{"id": 1, "name": "cell"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [
                    [1, 1, 5, 1, 5, 5, 1, 5], [20, 10, 25, 10, 25, 15, 20, 15]]}]}
        path = self.root / "c.json"
        path.write_text(json.dumps(data))
        (ann,) = import_coco(path, "frame.png", 32, 24).annotations
        mask = annotation_mask(ann, 24, 32)
        self.assertTrue(mask[3, 3] and mask[12, 22])
        self.assertEqual(ann["mask_components"], 2)

    def test_yolo_round_trip_and_names(self):
        original = [square("cell", 2, 2), square("particle", 18, 8, 5)]
        target, _ = export_image(self.root / "out", self.image, original, 32, 24, "yolo")
        label = target / "labels" / "frame.txt"
        report = import_yolo(label, 32, 24)
        self.assertEqual(report.skipped, [])
        self.assertEqual(sorted(a["label"] for a in report.annotations), ["cell", "particle"])
        for a, b in zip(self.masks(original), self.masks(report.annotations)):
            self.assertGreater((a & b).sum() / (a | b).sum(), 0.95)

    def test_yolo_boxes_and_errors(self):
        path = self.root / "frame.txt"
        path.write_text("0 0.5 0.5 0.25 0.5\n1 0.1 0.1 0.2\n0 1.5 0.5 0.1 0.1\n\n"
                        "0 0 0 1 0 1 1\n")
        report = import_yolo(path, 32, 24, names={0: "cell", 1: "dust"})
        self.assertEqual(report.annotations[0]["shape_type"], "rectangle")
        self.assertEqual(report.annotations[0]["points"], [[12.0, 6.0], [19.0, 17.0]])
        self.assertEqual(report.annotations[1]["shape_type"], "polygon")
        self.assertEqual(len(report.skipped), 2)

    def test_yolo_without_names_uses_ids(self):
        path = self.root / "frame.txt"
        path.write_text("3 0.5 0.5 0.2 0.2\n")
        report = import_yolo(path, 32, 24)
        self.assertEqual(report.annotations[0]["label"], "class_3")
        self.assertTrue(report.warnings)

    def test_find_yolo_label_ultralytics_layout(self):
        (self.root / "ds/images/train").mkdir(parents=True)
        (self.root / "ds/labels/train").mkdir(parents=True)
        image = self.root / "ds/images/train/a.png"
        label = self.root / "ds/labels/train/a.txt"
        image.touch(); label.touch()
        self.assertEqual(find_yolo_label(image), label)

    def test_sources_not_modified(self):
        path = self.root / "frame.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n")
        before = path.read_bytes()
        import_yolo(path, 32, 24)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
