"""Whole-project dataset export with a reproducible train/val/test split.

* All images are validated before anything is written; the output folder is
  published atomically and never overwrites an existing dataset.
* Source files are verified against the project SHA-256 to avoid exporting
  annotations for a silently replaced image.
* Frames split from one stack (``<stem>_frames/``) or sharing a group key stay in
  one subset, preventing near-duplicate leakage between train and test.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from app.core.export import _yolo_polygon, validate_annotations
from project_format.io import file_sha256, resolve_image
from project_format.models import annotation_payload
from worker.postprocess.masks import mask_rle

DATASET_FORMATS = ("coco", "yolo", "masks")
SPLITS = ("train", "val", "test")
_FRAME = re.compile(r"^(?P<stem>.+)_f\d{4,}$")


class DatasetExportError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("\n".join(self.issues[:200]))


def group_key(image_path):
    """Images from one split stack share a key; other images are independent."""
    path = Path(image_path)
    match = _FRAME.match(path.stem)
    if match and path.parent.name == f"{match['stem']}_frames":
        return str(path.parent)
    return str(path)


def assign_splits(keys, ratios=(0.8, 0.1, 0.1), seed=0):
    """Deterministic group-level split; returns {key: split}.

    Groups are shuffled with a fixed seed and assigned greedily so that the
    number of images per subset follows the ratios as closely as possible.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or not sum(ratios) > 0:
        raise ValueError("Ratios must be three non-negative numbers")
    total = float(sum(ratios))
    ratios = [r / total for r in ratios]
    groups = defaultdict(list)
    for key in keys:
        groups[key].append(key)
    sizes = Counter(keys)
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    count = len(keys)
    targets = [r * count for r in ratios]
    filled = [0, 0, 0]
    result = {}
    for key in order:
        # Pick the subset with the largest remaining deficit (ties: train first).
        deficits = [targets[i] - filled[i] if ratios[i] > 0 else -np.inf for i in range(3)]
        index = int(np.argmax(deficits))
        result[key] = SPLITS[index]
        filled[index] += sizes[key]
    return result


def _unique_name(name, used, image_id):
    if name not in used:
        used.add(name)
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = f"{stem}__{image_id[:8]}{suffix}"
    used.add(candidate)
    return candidate


def _json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect(project_path, project, include_empty=True, review_filter=None,
            verify_checksums=True, progress=None):
    """Validate every image; return rows and class order or raise with all issues."""
    classes = {c.id: c for c in project.classes}
    issues, rows = [], []
    for number, image in enumerate(project.images):
        if progress:
            progress(number, len(project.images), f"Checking {Path(image.path).name}")
        path = resolve_image(project_path, image)
        name = Path(image.path).name
        if not path.is_file():
            issues.append(f"{name}: image file is missing ({path})")
            continue
        if verify_checksums and file_sha256(path) != image.sha256:
            issues.append(f"{name}: file content differs from the project checksum")
            continue
        annotations = [annotation_payload(a, classes[a.class_id]) for a in image.annotations
                       if review_filter is None or a.review_status in review_filter]
        if not annotations and not include_empty:
            continue
        errors, masks = validate_annotations(annotations, image.width, image.height)
        issues.extend(f"{name}: {e}" for e in errors)
        rows.append({"image": image, "path": path, "annotations": annotations,
                     "masks": masks, "group": group_key(path)})
    used_ids = {a["class_id"] for row in rows for a in row["annotations"]}
    labels = [c.name for c in project.classes if c.active or c.id in used_ids]
    return rows, labels, issues


def export_dataset(parent, project_path, project, format_name, ratios=(0.8, 0.1, 0.1),
                   seed=0, include_empty=True, review_filter=None, copy_images=True,
                   progress=None):
    """``progress(done, total, text)`` may raise to cancel; nothing is published then."""
    if format_name not in DATASET_FORMATS:
        raise ValueError(f"Unknown dataset format: {format_name}")
    if not project.images:
        raise ValueError("The project has no images")
    check = (lambda d, t, s: progress(d, 2 * t, s)) if progress else None
    rows, labels, issues = collect(project_path, project, include_empty, review_filter,
                                   progress=check)
    if not rows and not issues:
        issues.append("No images match the export filter")
    category = {label: i + 1 for i, label in enumerate(labels)}
    yolo_polygons = {}
    for row in rows:
        name = Path(row["image"].path).name
        if format_name == "yolo":
            for index, (a, m) in enumerate(zip(row["annotations"], row["masks"])):
                try:
                    yolo_polygons[(row["image"].id, index)] = _yolo_polygon(a, m)
                except ValueError as exc:
                    issues.append(f"{name}, object {index + 1}: {exc}")
        if format_name == "masks":
            occupied = np.zeros((row["image"].height, row["image"].width), bool)
            for m in row["masks"]:
                if np.any(occupied & m):
                    issues.append(f"{name}: overlapping objects cannot be stored in one "
                                  f"indexed mask; use COCO")
                    break
                occupied |= m
    if issues:
        raise DatasetExportError(issues)

    splits = assign_splits([row["group"] for row in rows], ratios, seed)
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".medphys-dataset-", dir=parent))
    target = parent / f"medphys_dataset_{format_name}_{uuid4().hex[:8]}"
    stats = {s: {"images": 0, "objects": Counter()} for s in SPLITS}
    try:
        used_names = {s: set() for s in SPLITS}
        coco = {s: {"images": [], "annotations": [],
                    "categories": [{"id": i, "name": n} for n, i in category.items()]}
                for s in SPLITS}
        manifest = []
        object_counter = 0
        for number, row in enumerate(rows):
            image, split = row["image"], splits[row["group"]]
            if progress:
                progress(len(rows) + number, 2 * len(rows), f"Writing {Path(image.path).name}")
            file_name = _unique_name(Path(image.path).name, used_names[split], image.id)
            stats[split]["images"] += 1
            image_dir = staging / "images" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(row["path"], image_dir / file_name)
            manifest.append({"image_id": image.id, "source": str(row["path"]),
                             "file": f"images/{split}/{file_name}", "split": split,
                             "group": row["group"], "sha256": image.sha256,
                             "objects": len(row["annotations"]),
                             "display": image.display.get("scientific")})
            for a in row["annotations"]:
                stats[split]["objects"][a["label"]] += 1
            if format_name == "coco":
                image_number = len(coco[split]["images"]) + 1
                coco[split]["images"].append({"id": image_number, "file_name": file_name,
                                              "width": image.width, "height": image.height,
                                              "medphys_image_id": image.id})
                for a, m in zip(row["annotations"], row["masks"]):
                    object_counter += 1
                    ys, xs = np.nonzero(m)
                    coco[split]["annotations"].append({
                        "id": object_counter, "image_id": image_number,
                        "category_id": category[a["label"]], "segmentation": mask_rle(m),
                        "area": int(m.sum()), "iscrowd": 0,
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                 int(ys.max() - ys.min() + 1)],
                        "object_id": a["object_id"], "source": a.get("source", "manual"),
                        "review_status": a.get("review_status", "unreviewed")})
            elif format_name == "yolo":
                label_dir = staging / "labels" / split
                label_dir.mkdir(parents=True, exist_ok=True)
                lines = [str(category[a["label"]] - 1) + " " + " ".join(
                    f"{v:.10f}" for x, y in yolo_polygons[(image.id, i)]
                    for v in (x / image.width, y / image.height))
                    for i, a in enumerate(row["annotations"])]
                (label_dir / f"{Path(file_name).stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            else:
                mask_dir = staging / "masks" / split
                mask_dir.mkdir(parents=True, exist_ok=True)
                dtype = np.uint8 if len(labels) <= 255 else np.uint16
                semantic = np.zeros((image.height, image.width), dtype)
                instance = np.zeros((image.height, image.width), np.uint16)
                for number, (a, m) in enumerate(zip(row["annotations"], row["masks"]), 1):
                    semantic[m] = category[a["label"]]
                    instance[m] = number
                Image.fromarray(semantic).save(mask_dir / f"{Path(file_name).stem}_semantic.png")
                Image.fromarray(instance).save(mask_dir / f"{Path(file_name).stem}_instance.png")
        if format_name == "coco":
            (staging / "annotations").mkdir()
            for split in SPLITS:
                if coco[split]["images"]:
                    _json(staging / "annotations" / f"instances_{split}.json", coco[split])
        elif format_name == "yolo":
            import yaml
            config = {"path": ".", "names": {i: n for i, n in enumerate(labels)}}
            for split in SPLITS:
                if stats[split]["images"]:
                    config[split] = f"images/{split}"
            (staging / "data.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        _json(staging / "classes.json", [
            {"name": n, "category_id": i, "yolo_id": i - 1, "mask_value": i}
            for n, i in category.items()])
        report = {
            "format": format_name, "created_at": datetime.now(timezone.utc).isoformat(),
            "project_id": project.id, "project_name": project.name,
            "seed": seed, "ratios": list(ratios), "include_empty": include_empty,
            "review_filter": sorted(review_filter) if review_filter else None,
            "splits": {s: {"images": stats[s]["images"],
                           "objects": dict(sorted(stats[s]["objects"].items()))}
                       for s in SPLITS},
            "warnings": [f"Subset '{s}' is empty although its ratio is {r}"
                         for s, r in zip(SPLITS, ratios) if r > 0 and not stats[s]["images"]],
            "split_fingerprint": hashlib.sha256(json.dumps(
                sorted((m["image_id"], m["split"]) for m in manifest)).encode()).hexdigest(),
        }
        _json(staging / "manifest.json", manifest)
        _json(staging / "report.json", report)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target, report
