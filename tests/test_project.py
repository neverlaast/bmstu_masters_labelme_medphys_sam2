"""Project round trips, path recovery, class transactions and failed writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from project_format.models import Annotation, ClassDefinition, Project
from project_format.classes import add_class, remove_class, rename_class
from project_format.io import (
    image_record, load_project, rebase_paths, relocate_images, resolve_image,
    save_project,
)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "dataset.medphys.json"
        self.image = self.root / "frame.png"
        Image.new("RGB", (64, 64), "white").save(self.image)
        self.definition = ClassDefinition(name="cell")
        self.project = Project(
            classes=[self.definition], images=[image_record(self.image, self.path)])
        self.annotation = Annotation(
            class_id=self.definition.id, shape_type="polygon",
            points=[(5, 5), (15, 5), (15, 15), (5, 15)])
        self.project.images[0].annotations.append(self.annotation)
        self.source_bytes = self.image.read_bytes()

    def tearDown(self):
        if self.image.exists():
            self.assertEqual(self.image.read_bytes(), self.source_bytes)
        self.directory.cleanup()

    def test_roundtrip_schema_classes_geometry_and_relative_paths(self):
        sha = save_project(self.path, self.project)
        loaded = load_project(self.path)
        self.assertEqual(loaded.model_dump(), self.project.model_dump())
        self.assertEqual(loaded.images[0].path, "frame.png")
        self.assertEqual(len(sha), 64)

    def test_unknown_version_class_and_duplicate_annotation_ids_rejected(self):
        payload = self.project.model_dump(mode="json")
        for key, value in [("schema_version", 99), ("format", "other")]:
            invalid = dict(payload, **{key: value})
            with self.assertRaises(ValueError):
                Project.model_validate(invalid)
        payload["images"][0]["annotations"][0]["class_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Unknown class"):
            Project.model_validate(payload)
        self.project.images[0].annotations.append(self.annotation.model_copy(deep=True))
        with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
            save_project(self.path, self.project)
        self.assertFalse(self.path.exists())

    def test_unknown_keys_nan_geometry_and_temporary_objects_rejected(self):
        with self.assertRaises(ValueError):
            Project.model_validate(dict(self.project.model_dump(), unexpected=True))
        with self.assertRaises(ValueError):
            Annotation(class_id=self.definition.id, shape_type="polygon",
                       points=[(float("nan"), 5), (15, 5), (15, 15)])
        invalid = self.project.model_copy(deep=True)
        invalid.images[0].annotations[0].points[0] = (-1, 5)
        with self.assertRaisesRegex(ValueError, "outside image"):
            save_project(self.path, invalid)
        with self.assertRaisesRegex(ValueError, "Temporary"):
            Annotation(class_id=self.definition.id, shape_type="point", points=[(5, 5)], flags={"pending": True})

    def test_atomic_failure_preserves_old_project_and_backup(self):
        sha = save_project(self.path, self.project)
        previous = self.path.read_bytes()
        self.project.name = "changed"
        import project_format.io as module
        original = module.os.replace
        def failing(source, target):
            if Path(target).resolve() == self.path.resolve():
                raise OSError("disk full")
            return original(source, target)
        with patch.object(module.os, "replace", side_effect=failing):
            with self.assertRaises(OSError):
                save_project(self.path, self.project, expected_sha256=sha)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(next(self.root.glob(".*.backups/*.json")).read_bytes(), previous)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_stale_writer_and_require_new_refuse_overwrite(self):
        sha = save_project(self.path, self.project)
        self.path.write_text(self.path.read_text() + "\n")
        previous = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            save_project(self.path, self.project, expected_sha256=sha)
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_project(self.path, self.project, require_new=True)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_backup_retention(self):
        self.project.backup_count = 2
        sha = save_project(self.path, self.project)
        for i in range(5):
            self.project.name = f"version {i}"
            sha = save_project(self.path, self.project, expected_sha256=sha)
        self.assertEqual(len(list(self.root.glob(".*.backups/*.json"))), 2)

    def test_image_checksum_and_dimensions_checked_on_load(self):
        save_project(self.path, self.project)
        mutated = json.loads(self.path.read_text())
        mutated["images"][0]["sha256"] = "0"*64
        self.path.write_text(json.dumps(mutated))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_project(self.path)
        mutated["images"][0]["sha256"] = self.project.images[0].sha256
        mutated["images"][0]["width"] = 65
        self.path.write_text(json.dumps(mutated))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            load_project(self.path)

    def test_relocate_and_save_as_rebases_without_changing_original(self):
        save_project(self.path, self.project)
        relocated = self.root / "moved"
        relocated.mkdir()
        self.image.rename(relocated / self.image.name)
        with self.assertRaisesRegex(ValueError, "Missing image"):
            load_project(self.path)
        recovered = relocate_images(self.project, self.path, relocated)
        self.assertEqual(resolve_image(self.path, recovered.images[0]), (relocated / "frame.png").resolve())
        new_path = self.root / "new/location.medphys.json"
        rebased = rebase_paths(recovered, self.path, new_path)
        save_project(new_path, rebased)
        self.assertEqual(resolve_image(new_path, load_project(new_path).images[0]), (relocated / "frame.png").resolve())
        self.assertEqual(self.project.images[0].path, "frame.png")

    def test_relocate_rejects_wrong_or_ambiguous_content(self):
        directory = self.root / "moved"
        directory.mkdir()
        self.image.rename(directory / "frame.png")
        duplicate = directory / "nested"
        duplicate.mkdir()
        (duplicate / "frame.png").write_bytes(self.source_bytes)
        with self.assertRaisesRegex(ValueError, "found 2"):
            relocate_images(self.project, self.path, directory)

    def test_class_rename_preserves_id_and_selected_rename_creates_new_id(self):
        renamed = rename_class(self.project, self.definition.id, "particle")
        self.assertEqual(renamed.classes[0].id, self.definition.id)
        self.assertEqual(renamed.images[0].annotations[0].class_id, self.definition.id)
        selected = rename_class(self.project, self.definition.id, "selected", [self.annotation.id])
        self.assertEqual(len(selected.classes), 2)
        self.assertEqual(selected.images[0].annotations[0].class_id, selected.classes[1].id)
        self.assertEqual(self.project.classes[0].name, "cell")
        with self.assertRaises(ValueError):
            rename_class(self.project, self.definition.id, "cell", [self.annotation.id])

    def test_used_class_requires_explicit_policy_and_transfer_is_atomic(self):
        with self.assertRaisesRegex(ValueError, "in use"):
            remove_class(self.project, self.definition.id)
        inactive = remove_class(self.project, self.definition.id, "deactivate")
        self.assertFalse(inactive.classes[0].active)
        self.assertEqual(len(inactive.images[0].annotations), 1)
        target = ClassDefinition(name="particle")
        expanded = add_class(self.project, target)
        transferred = remove_class(expanded, self.definition.id, "transfer", target.id)
        self.assertEqual(transferred.images[0].annotations[0].class_id, target.id)
        self.assertEqual(len(transferred.classes), 1)
        deleted = remove_class(self.project, self.definition.id, "delete_objects")
        self.assertEqual(deleted.images[0].annotations, [])
        self.assertEqual(deleted.classes, [])

    def test_invalid_class_geometry_migration_does_not_change_source(self):
        target = ClassDefinition(name="point_only", allowed_geometry=["point"])
        project = add_class(self.project, target)
        with self.assertRaisesRegex(ValueError, "Geometry is not allowed"):
            remove_class(project, self.definition.id, "transfer", target.id)
        self.assertEqual(project.images[0].annotations[0].class_id, self.definition.id)

    def test_high_depth_single_frame_accepted_multipage_rejected(self):
        image = self.root / "high.tif"
        Image.new("I;16", (32, 24)).save(image)
        record = image_record(image, self.path)
        self.assertEqual((record.width, record.height), (32, 24))
        stack = self.root / "stack.tif"
        frames = [Image.new("L", (8, 8), i) for i in range(2)]
        frames[0].save(stack, save_all=True, append_images=frames[1:])
        with self.assertRaisesRegex(ValueError, "Split multi-page"):
            image_record(stack, self.path)
