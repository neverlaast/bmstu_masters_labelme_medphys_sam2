"""Whole-project dataset export: split reproducibility, leakage, formats, safety."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from app.core.dataset_export import (
    DatasetExportError, assign_splits, export_dataset, group_key,
)
from app.core.importers import import_coco, import_yolo
from project_format.io import image_record
from project_format.models import Annotation, ClassDefinition, Project


def square(class_id, x, y, s=6):
    return Annotation(class_id=class_id, shape_type="polygon",
                      points=[(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "p.medphys.json"
        self.cell, self.dust = ClassDefinition(name="cell"), ClassDefinition(name="dust")
        images = []
        for i in range(10):
            image = self.root / f"img_{i:02d}.png"
            Image.new("RGB", (40, 30), "white").save(image)
            record = image_record(image, self.path)
            record.annotations = [square(self.cell.id, 2, 2), square(self.dust.id, 20, 10, 4)]
            images.append(record)
        stack = self.root / "stack_frames"
        stack.mkdir()
        for i in range(4):
            image = stack / f"stack_f{i + 1:04d}.png"
            Image.new("L", (40, 30), 0).save(image)
            images.append(image_record(image, self.path))
        self.project = Project(classes=[self.cell, self.dust], images=images)
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_is_deterministic_and_groups_stacks(self):
        keys = [group_key(resolve) for resolve in
                [self.root / i.path for i in self.project.images]]
        self.assertEqual(len(set(keys)), 11)
        a = assign_splits(keys, (0.6, 0.2, 0.2), seed=3)
        self.assertEqual(a, assign_splits(keys, (0.6, 0.2, 0.2), seed=3))
        counts = Counter_values(a, keys)
        self.assertEqual(sum(counts.values()), 14)
        self.assertLessEqual(abs(counts["train"] - 8.4), 4)
        self.assertEqual(len({a[k] for k in keys if "stack_frames" in k}), 1)
        self.assertEqual(set(a.values()), {"train", "val", "test"})

    def test_coco_export_round_trip_and_report(self):
        target, report = export_dataset(self.out, self.path, self.project, "coco",
                                        (0.6, 0.2, 0.2), seed=1)
        manifest = json.loads((target / "manifest.json").read_text())
        stack_splits = {m["split"] for m in manifest if "stack_frames" in m["group"]}
        self.assertEqual(len(stack_splits), 1, "frames of one stack leaked across subsets")
        total = sum(v["images"] for v in report["splits"].values())
        self.assertEqual(total, 14)
        for split, info in report["splits"].items():
            if not info["images"]:
                continue
            coco = target / "annotations" / f"instances_{split}.json"
            data = json.loads(coco.read_text())
            self.assertEqual(len(data["images"]), info["images"])
            name = data["images"][0]["file_name"]
            self.assertTrue((target / "images" / split / name).is_file())
            imported = import_coco(coco, name, 40, 30)
            self.assertEqual(imported.skipped, [])
        self.assertEqual(sum(sum(v["objects"].values()) for v in report["splits"].values()), 20)

    def test_yolo_layout(self):
        target, report = export_dataset(self.out, self.path, self.project, "yolo", seed=0)
        config = yaml.safe_load((target / "data.yaml").read_text())
        self.assertEqual(config["names"], {0: "cell", 1: "dust"})
        labels = sorted((target / "labels").rglob("img_*.txt"))
        self.assertEqual(len(labels), 10)
        imported = import_yolo(labels[0], 40, 30)
        self.assertEqual(sorted(a["label"] for a in imported.annotations), ["cell", "dust"])

    def test_masks_values(self):
        target, _ = export_dataset(self.out, self.path, self.project, "masks", (1, 0, 0))
        semantic = np.array(Image.open(target / "masks/train/img_00_semantic.png"))
        instance = np.array(Image.open(target / "masks/train/img_00_instance.png"))
        self.assertEqual(semantic[4, 4], 1)
        self.assertEqual(semantic[12, 22], 2)
        self.assertEqual(set(np.unique(instance)), {0, 1, 2})

    def test_exclude_empty_and_review_filter(self):
        self.project.images[0].annotations[0].review_status = "accepted"
        _, report = export_dataset(self.out, self.path, self.project, "coco", (1, 0, 0),
                                   include_empty=False, review_filter={"accepted"})
        self.assertEqual(report["splits"]["train"]["images"], 1)
        self.assertEqual(report["splits"]["train"]["objects"], {"cell": 1})

    def test_all_problems_reported_and_nothing_written(self):
        self.project.images[1].annotations.append(Annotation(
            class_id=self.cell.id, shape_type="polygon",
            points=[(0, 0), (10, 10), (10, 0), (0, 10)]))
        (self.root / self.project.images[2].path).write_bytes(b"changed")
        with self.assertRaises(DatasetExportError) as context:
            export_dataset(self.out, self.path, self.project, "coco")
        text = str(context.exception)
        self.assertIn("img_01", text)
        self.assertIn("checksum", text)
        self.assertEqual(list(self.out.iterdir()) if self.out.exists() else [], [])

    def test_name_collisions_get_unique_names(self):
        sub = self.root / "other"
        sub.mkdir()
        Image.new("RGB", (40, 30)).save(sub / "img_00.png")
        self.project.images.append(image_record(sub / "img_00.png", self.path))
        target, _ = export_dataset(self.out, self.path, self.project, "coco", (1, 0, 0))
        names = [m["file"] for m in json.loads((target / "manifest.json").read_text())]
        self.assertEqual(len(names), len(set(names)))

    def test_overlap_blocks_indexed_masks(self):
        self.project.images[0].annotations.append(square(self.dust.id, 3, 3))
        with self.assertRaises(DatasetExportError):
            export_dataset(self.out, self.path, self.project, "masks")


def Counter_values(assignment, keys):
    from collections import Counter
    return Counter(assignment[k] for k in keys)


if __name__ == "__main__":
    unittest.main()
