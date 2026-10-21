"""Validated, non-destructive export of one image's final annotations.

Coordinates describe pixel centres: 0 <= x < width, 0 <= y < height.
SAM raster predictions are authoritative only while their outline is unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL

import cv2
import numpy as np
from PIL import Image

from worker.postprocess.masks import mask_rle

FORMATS = ("labelme", "coco", "yolo", "binary_masks", "masks", "csv")


class ExportValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("\n".join(issues))


def snapshot_shapes(shapes, image_key):
    """Detach Qt shapes and exclude all transient prompts/pending candidates."""
    result = []
    for index, shape in enumerate(shapes):
        flags = dict(shape.flags or {})
        if flags.get("sam2_prompt") or flags.get("temporary") or flags.get("pending"):
            continue
        data = dict(shape.other_data or {})
        result.append({
            **data,
            "label": shape.label,
            "points": [[p.x(), p.y()] for p in shape.points],
            "shape_type": shape.shape_type,
            "group_id": shape.group_id,
            "description": shape.description,
            "flags": flags,
            "object_id": data.get("object_id") or str(uuid5(
                NAMESPACE_URL, f"{image_key}:annotation:{index}")),
        })
    return result


def decode_rle(rle, height, width):
    if not isinstance(rle, dict) or rle.get("size") != [height, width]:
        raise ValueError("Mask dimensions do not match the image")
    counts = rle.get("counts")
    if not isinstance(counts, list) or not counts or any(
        type(n) is not int or n < 0 for n in counts
    ) or sum(counts) != height * width:
        raise ValueError("Malformed uncompressed mask RLE")
    output = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    for index, length in enumerate(counts):
        if index % 2:
            output[offset:offset + length] = 1
        offset += length
    return output.reshape((height, width), order="F").astype(bool)


def _intersects(a, b, c, d):
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

    def on(p, q, r):
        return (min(p[0], q[0])-1e-9 <= r[0] <= max(p[0], q[0])+1e-9
                and min(p[1], q[1])-1e-9 <= r[1] <= max(p[1], q[1])+1e-9)

    values = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if values[0]*values[1] < 0 and values[2]*values[3] < 0:
        return True
    return any(abs(v) < 1e-9 and on(p, q, r) for v, p, q, r in [
        (values[0], a, b, c), (values[1], a, b, d),
        (values[2], c, d, a), (values[3], c, d, b)])


def _polygon_issues(points):
    issues = []
    n = len(points)
    if n < 3:
        return ["Polygon needs at least three vertices"]
    if n > 10000:
        return ["Polygon exceeds the 10000-vertex validation limit"]
    if len(set(map(tuple, points))) != n:
        issues.append("Repeated polygon vertices")
    area = abs(sum(points[i][0]*points[(i+1) % n][1]
                   - points[(i+1) % n][0]*points[i][1] for i in range(n))) / 2
    if area <= 1e-9:
        issues.append("Zero-area polygon")
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if _intersects(points[i], points[(i+1) % n],
                           points[j], points[(j+1) % n]):
                return issues + ["Self-intersecting polygon"]
    return issues


def annotation_mask(annotation, height, width):
    points = annotation["points"]
    rle = annotation.get("mask_rle")
    if (rle is not None and annotation.get("raster_source_valid", True)
            and not annotation.get("geometry_modified")
            and annotation.get("mask_outline") == points):
        return decode_rle(rle, height, width)
    output = np.zeros((height, width), dtype=np.uint8)
    kind = annotation.get("shape_type", "polygon")
    rounded = np.rint(points).astype(np.int32)
    if kind == "polygon":
        cv2.fillPoly(output, [rounded], 1)
    elif kind == "rectangle":
        cv2.rectangle(output, tuple(rounded.min(axis=0)), tuple(rounded.max(axis=0)), 1, -1)
    elif kind == "circle":
        yy, xx = np.ogrid[:height, :width]
        cx, cy = points[0]
        radius = math.dist(points[0], points[1])
        output = ((xx-cx)**2 + (yy-cy)**2 <= radius**2).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported segmentation shape: {kind}")
    return output.astype(bool)


def validate_annotations(annotations, width, height, classes=None):
    """Return errors and per-object masks; never fix or clip user geometry."""
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    if width * height * max(1, len(annotations)) > 512 * 1024 * 1024:
        raise ValueError("Export raster validation exceeds its 512 MiB mask budget")
    errors, masks, ids, geometries = [], [], set(), {}
    for index, annotation in enumerate(annotations):
        prefix = f"Object {index+1}"
        local = []
        label = annotation.get("label")
        if not isinstance(label, str) or not label.strip():
            local.append("Empty class")
        elif classes is not None and label not in classes:
            local.append(f"Unknown class: {label}")
        identity = annotation.get("object_id")
        if not isinstance(identity, str) or not identity.strip():
            local.append("Missing/invalid object ID")
        elif identity in ids:
            local.append("Duplicate object ID")
        ids.add(identity) if isinstance(identity, str) else None
        points = annotation.get("points")
        valid_points = (isinstance(points, list) and bool(points) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in p) for p in points))
        mask = None
        if not valid_points:
            local.append("Missing/non-finite coordinates")
        else:
            if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                local.append("Coordinates outside image bounds")
            kind = annotation.get("shape_type", "polygon")
            if kind == "polygon":
                local.extend(_polygon_issues(points))
            elif kind in {"rectangle", "circle"}:
                if len(points) != 2:
                    local.append(f"{kind} needs two points")
                elif kind == "rectangle" and (points[0][0] == points[1][0] or points[0][1] == points[1][1]):
                    local.append("Zero-area rectangle")
                elif kind == "circle":
                    radius = math.dist(points[0], points[1])
                    cx, cy = points[0]
                    if radius == 0:
                        local.append("Zero-radius circle")
                    if cx-radius < 0 or cy-radius < 0 or cx+radius >= width or cy+radius >= height:
                        local.append("Circle extends outside image")
            else:
                local.append(f"Unsupported segmentation shape: {kind}")
            if not local:
                try:
                    mask = annotation_mask(annotation, height, width)
                    if not mask.any():
                        local.append("Empty mask")
                    digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
                    if digest in geometries:
                        local.append(f"Duplicate mask of object {geometries[digest]}")
                    geometries[digest] = index+1
                except (TypeError, ValueError, KeyError) as exc:
                    local.append(str(exc))
        masks.append(mask)
        errors.extend(f"{prefix}: {issue}" for issue in local)
    return errors, masks


def _json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _csv_text(value):
    # Keep user labels/IDs from becoming spreadsheet formulas on opening CSV.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _yolo_polygon(annotation, mask):
    # A single YOLO polygon cannot faithfully express holes or disconnected parts.
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1 or hierarchy is None or len(contours[0]) < 3:
        raise ValueError("YOLO cannot represent this mask without losing holes/components")
    if annotation.get("shape_type", "polygon") == "polygon" and annotation.get("mask_rle") is None:
        return annotation["points"]
    return contours[0].reshape(-1, 2).tolist()


def export_image(parent, image_path, annotations, width, height, format_name, classes=None):
    """Write a new complete directory, publishing only after all writes succeed."""
    if format_name not in FORMATS:
        raise ValueError("Unknown export format")
    image_path, parent = Path(image_path), Path(parent)
    errors, masks = validate_annotations(annotations, width, height, classes)
    if errors:
        raise ExportValidationError(errors)
    labels = list(dict.fromkeys(classes if classes is not None else sorted(
        {a["label"] for a in annotations})))
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Invalid class list")
    if len(labels) > 65535:
        raise ValueError("Too many classes for indexed PNG")
    indices = {label: index+1 for index, label in enumerate(labels)}
    # Ensure every YOLO object is convertible before touching the filesystem.
    polygons = []
    if format_name == "yolo":
        for index, (annotation, mask) in enumerate(zip(annotations, masks)):
            try:
                polygons.append(_yolo_polygon(annotation, mask))
            except ValueError as exc:
                raise ExportValidationError([f"Object {index+1}: {exc}"]) from exc
    if not image_path.is_file():
        raise ValueError("Original image file is missing")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".medphys-export-", dir=parent))
    target = parent / f"medphys_{format_name}_{uuid4().hex[:12]}"
    try:
        source_dir = staging / "images"
        source_dir.mkdir()
        shutil.copy2(image_path, source_dir / image_path.name)
        _json(staging / "classes.json", [
            {"name": label, "category_id": index+1, "yolo_id": index}
            for index, label in enumerate(labels)])
        if format_name == "labelme":
            _json(staging / "annotations.json", {
                "version": "5.2.1", "flags": {}, "shapes": annotations,
                "imagePath": f"images/{image_path.name}", "imageData": None,
                "imageWidth": width, "imageHeight": height})
        elif format_name == "coco":
            objects = []
            for index, (annotation, mask) in enumerate(zip(annotations, masks)):
                ys, xs = np.nonzero(mask)
                objects.append({
                    "id": index+1, "image_id": 1,
                    "category_id": indices[annotation["label"]],
                    "segmentation": mask_rle(mask), "iscrowd": 0,
                    "area": int(mask.sum()),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)],
                    "object_id": annotation["object_id"]})
            _json(staging / "annotations.json", {
                "images": [{"id": 1, "file_name": image_path.name, "width": width, "height": height}],
                "annotations": objects,
                "categories": [{"id": i+1, "name": label} for i, label in enumerate(labels)]})
        elif format_name == "yolo":
            directory = staging / "labels"
            directory.mkdir()
            lines = [str(indices[a["label"]]-1) + " " + " ".join(
                f"{value:.10f}" for x, y in polygon for value in (x/width, y/height))
                for a, polygon in zip(annotations, polygons)]
            (directory / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            import yaml
            (staging / "data.yaml").write_text(yaml.safe_dump({
                "path": ".", "train": "images",
                "names": {i: label for i, label in enumerate(labels)}},
                allow_unicode=True, sort_keys=False), encoding="utf-8")
        elif format_name in {"masks", "binary_masks"}:
            directory = staging / "masks"
            directory.mkdir()
            indexed = np.zeros((height, width), np.uint16)
            mapping, occupied = [], np.zeros((height, width), bool)
            for index, (annotation, mask) in enumerate(zip(annotations, masks)):
                # Semantic indexed masks cannot encode overlap; don't silently overwrite.
                if format_name == "masks" and np.any(occupied & mask):
                    raise ExportValidationError(["Overlapping objects cannot be represented in one indexed mask; export COCO or individual masks instead"])
                occupied |= mask
                name = f"object_{index+1:06d}.png"
                Image.fromarray(mask.astype(np.uint8)*255).save(directory / name)
                indexed[mask] = indices[annotation["label"]]
                mapping.append({"file": name, "object_id": annotation["object_id"], "label": annotation["label"]})
            if format_name == "binary_masks":
                pass
            elif len(labels) <= 255:
                image = Image.fromarray(indexed.astype(np.uint8)).convert("P")
                palette = [0, 0, 0]
                for index in range(1, 256):
                    palette.extend([(index*67) % 256, (index*131) % 256, (index*197) % 256])
                image.putpalette(palette)
                image.save(staging / "indexed_mask.png")
            else:
                Image.fromarray(indexed).save(staging / "indexed_mask.png")
            _json(staging / "mask_objects.json", mapping)
        else:
            with (staging / "objects.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["object_id", "class", "area_px", "centroid_x_px", "centroid_y_px",
                                 "x_min", "y_min", "width_px", "height_px", "source", "review_status"])
                for annotation, mask in zip(annotations, masks):
                    ys, xs = np.nonzero(mask)
                    writer.writerow([_csv_text(annotation["object_id"]), _csv_text(annotation["label"]), int(mask.sum()),
                                     float(xs.mean()), float(ys.mean()), int(xs.min()), int(ys.min()),
                                     int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1),
                                     _csv_text(annotation.get("source", "manual")),
                                     _csv_text(annotation.get("review_status", ""))])
        report = {
            "format": format_name, "images": 1, "objects": len(annotations), "errors": [],
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "class_registry": "explicit" if classes is not None else "derived_from_current_image",
            "coordinates": "pixel centres", "units": "pixels",
            "notes": ["CSV geometry is measured from the raster mask",
                      "YOLO contours may be rasterized; holes/components are rejected",
                      "COCO uses uncompressed column-major RLE",
                      "No train/validation split is assigned"]}
        _json(staging / "export_report.json", report)
        os.rename(staging, target)
        return target, report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
