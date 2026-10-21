"""Non-destructive import of external COCO / YOLO annotations for one image.

The importer never modifies source files and never silently fixes geometry:
objects that cannot be represented are skipped and listed in the report.
Coordinates use the same convention as export: pixel centres,
0 <= x < width, 0 <= y < height.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.core.export import decode_rle
from worker.postprocess.masks import mask_rle

IMPORT_FORMATS = ("coco", "yolo")
MAX_OBJECTS = 100_000


class ImportFormatError(ValueError):
    """The annotation file as a whole cannot be interpreted."""


@dataclass
class ImportReport:
    source: str
    format_name: str
    annotations: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # (object reference, reason)
    warnings: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def summary(self, limit=20):
        lines = [f"Imported {len(self.annotations)} objects from {Path(self.source).name} "
                 f"({self.format_name.upper()}); skipped {len(self.skipped)}."]
        if self.labels:
            lines.append("Classes: " + ", ".join(self.labels))
        for ref, reason in self.skipped[:limit]:
            lines.append(f"Skipped {ref}: {reason}")
        if len(self.skipped) > limit:
            lines.append(f"... and {len(self.skipped) - limit} more skipped objects")
        lines.extend(self.warnings[:limit])
        return "\n".join(lines)


# ----------------------------------------------------------------- helpers

def decode_compressed_rle(counts: str, height: int, width: int) -> np.ndarray:
    """Decode the pycocotools LEB128-like string RLE without the dependency."""
    values, index, data = [], 0, counts.encode("ascii")
    while index < len(data):
        value, shift, more = 0, 0, True
        while more:
            if index >= len(data):
                raise ValueError("Truncated compressed RLE")
            char = data[index] - 48
            value |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            index += 1
            shift += 1
            if not more and (char & 0x10):
                value |= -1 << (5 * shift)
        if len(values) > 2:
            value += values[-2]
        values.append(value)
    return decode_rle({"size": [height, width], "counts": values}, height, width)


def _finite_points(values):
    points = [[float(values[i]), float(values[i + 1])] for i in range(0, len(values) - 1, 2)]
    if len(values) % 2 or any(not math.isfinite(c) for p in points for c in p):
        raise ValueError("Odd or non-finite coordinate list")
    return points


def _clamp(points, width, height, report, ref):
    """Allow only sub-pixel overshoot (x == width from normalised 1.0)."""
    result, clamped = [], False
    for x, y in points:
        if x < -0.5 or y < -0.5 or x > width + 0.5 or y > height + 0.5:
            raise ValueError("Coordinates are outside the image")
        nx, ny = min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)
        clamped |= (nx, ny) != (x, y)
        result.append([nx, ny])
    if clamped:
        report.warnings.append(f"{ref}: boundary coordinates clamped by <= 0.5 px")
    return result


def _dedupe(points):
    out = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _mask_shape(mask, label, meta):
    """Raster object -> outline polygon plus authoritative RLE."""
    if not mask.any():
        raise ValueError("Empty mask")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    outline = [[float(p[0][0]), float(p[0][1])] for p in largest]
    if len(outline) < 3:
        ys, xs = np.nonzero(mask)
        outline = [[float(xs.min()), float(ys.min())], [float(xs.max()), float(ys.min())],
                   [float(xs.max()), float(ys.max())], [float(xs.min()), float(ys.max())]]
        outline = _dedupe(outline)
        if len(outline) < 3:
            raise ValueError("Mask is too small for a polygon outline")
    return _annotation(label, "polygon", outline, meta,
                       mask_rle=mask_rle(mask), mask_outline=outline,
                       raster_source_valid=True, geometry_modified=False,
                       mask_components=int(len(contours)))


def _annotation(label, shape_type, points, meta, **extra):
    return {"label": label, "points": points, "shape_type": shape_type,
            "group_id": None, "description": "", "flags": {"imported": True},
            "object_id": str(uuid4()), "review_status": "imported", **meta, **extra}


def _polygon_mask(polygons, height, width):
    mask = np.zeros((height, width), np.uint8)
    for points in polygons:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


# -------------------------------------------------------------------- COCO

def _select_coco_image(images, image_name, width, height):
    by_name = [im for im in images if Path(str(im.get("file_name", ""))).name == image_name]
    if len(by_name) > 1:
        raise ImportFormatError(f"Several COCO images are named {image_name}")
    if not by_name and len(images) == 1:
        by_name = images
    if not by_name:
        raise ImportFormatError(f"No COCO image entry matches {image_name}")
    entry = by_name[0]
    if (entry.get("width"), entry.get("height")) not in {(width, height), (None, None)}:
        raise ImportFormatError(
            f"COCO image size {entry.get('width')}x{entry.get('height')} "
            f"differs from the open image {width}x{height}")
    return entry


def import_coco(path, image_name, width, height):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImportFormatError(f"Cannot read COCO JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list) \
            or not isinstance(data.get("images"), list):
        raise ImportFormatError("COCO JSON needs 'images' and 'annotations' lists")
    categories = {c.get("id"): str(c.get("name", "")).strip()
                  for c in data.get("categories", []) if isinstance(c, dict)}
    entry = _select_coco_image(data["images"], image_name, width, height)
    report = ImportReport(str(path), "coco")
    selected = [a for a in data["annotations"]
                if isinstance(a, dict) and a.get("image_id") == entry.get("id")]
    if len(selected) > MAX_OBJECTS:
        raise ImportFormatError(f"More than {MAX_OBJECTS} objects for one image")
    for number, ann in enumerate(selected, 1):
        ref = f"COCO annotation id={ann.get('id', number)}"
        try:
            label = categories.get(ann.get("category_id"))
            if not label:
                raise ValueError(f"Unknown category_id {ann.get('category_id')}")
            meta = {"source": "import_coco", "import_file": str(path),
                    "import_original_id": ann.get("id"),
                    "iscrowd": int(ann.get("iscrowd", 0) or 0)}
            for key in ("object_id", "score"):
                if key in ann:
                    meta["import_" + key] = ann[key]
            seg = ann.get("segmentation")
            if isinstance(seg, dict):
                size = seg.get("size")
                if size != [height, width]:
                    raise ValueError(f"RLE size {size} differs from image")
                counts = seg.get("counts")
                mask = (decode_compressed_rle(counts, height, width) if isinstance(counts, str)
                        else decode_rle(seg, height, width))
                report.annotations.append(_mask_shape(mask, label, meta))
            elif isinstance(seg, list) and seg:
                parts = [_dedupe(_clamp(_finite_points(p), width, height, report, ref))
                         for p in seg]
                parts = [p for p in parts if len(p) >= 3]
                if not parts:
                    raise ValueError("Polygon has fewer than three vertices")
                if len(parts) == 1:
                    report.annotations.append(_annotation(label, "polygon", parts[0], meta))
                else:   # multi-part object: keep exact union as raster
                    report.annotations.append(_mask_shape(
                        _polygon_mask(parts, height, width), label, meta))
            elif isinstance(ann.get("bbox"), list) and len(ann["bbox"]) == 4:
                x, y, w, h = map(float, ann["bbox"])
                if not (w > 0 and h > 0):
                    raise ValueError("Non-positive bbox")
                # COCO bbox spans pixel edges; convert to inclusive pixel centres.
                box = _clamp([[x, y], [x + w - 1, y + h - 1]], width, height, report, ref)
                report.annotations.append(_annotation(label, "rectangle", box, meta))
            else:
                raise ValueError("No segmentation or bbox")
        except (ValueError, TypeError) as exc:
            report.skipped.append((ref, str(exc)))
    report.labels = sorted({a["label"] for a in report.annotations})
    return report


# -------------------------------------------------------------------- YOLO

def load_yolo_names(label_path):
    """Find class names near a YOLO label file: data.yaml / *.yaml / classes.txt."""
    label_path = Path(label_path)
    for directory in [label_path.parent, *label_path.parents][:4]:
        for name in ("data.yaml", "dataset.yaml"):
            candidate = directory / name
            if candidate.is_file():
                import yaml
                names = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}).get("names")
                if isinstance(names, list):
                    return {i: str(n) for i, n in enumerate(names)}
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
        candidate = directory / "classes.txt"
        if candidate.is_file():
            lines = [l.strip() for l in candidate.read_text(encoding="utf-8").splitlines()]
            return {i: n for i, n in enumerate(lines) if n}
    return {}


def import_yolo(path, width, height, names=None):
    path = Path(path)
    names = load_yolo_names(path) if names is None else names
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ImportFormatError(f"Cannot read YOLO labels: {exc}") from exc
    if len(lines) > MAX_OBJECTS:
        raise ImportFormatError(f"More than {MAX_OBJECTS} objects for one image")
    report = ImportReport(str(path), "yolo")
    if not names:
        report.warnings.append("No data.yaml/classes.txt found: classes named class_<id>")
    for number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ref = f"line {number}"
        try:
            tokens = line.split()
            if not tokens[0].isdigit():
                raise ValueError("Class id must be a non-negative integer")
            class_id = int(tokens[0])
            label = names.get(class_id, f"class_{class_id}") if names else f"class_{class_id}"
            if names and class_id not in names:
                raise ValueError(f"Class id {class_id} is missing from class names")
            values = [float(v) for v in tokens[1:]]
            if any(not math.isfinite(v) or v < -1e-6 or v > 1 + 1e-6 for v in values):
                raise ValueError("Normalised coordinates must lie in [0, 1]")
            meta = {"source": "import_yolo", "import_file": str(path),
                    "import_line": number, "yolo_class_id": class_id}
            if len(values) == 4:
                cx, cy, w, h = values
                if not (w > 0 and h > 0):
                    raise ValueError("Non-positive box size")
                box = [[(cx - w / 2) * width, (cy - h / 2) * height],
                       [(cx + w / 2) * width - 1, (cy + h / 2) * height - 1]]
                report.annotations.append(_annotation(
                    label, "rectangle", _clamp(box, width, height, report, ref), meta))
            elif len(values) >= 6 and len(values) % 2 == 0:
                pts = [[x * width, y * height] for x, y in _finite_points(values)]
                pts = _dedupe(_clamp(pts, width, height, report, ref))
                if len(pts) < 3:
                    raise ValueError("Polygon has fewer than three distinct vertices")
                report.annotations.append(_annotation(label, "polygon", pts, meta))
            elif len(values) == 5:
                raise ValueError("Pose/keypoint or confidence column format is not supported")
            else:
                raise ValueError(f"Unexpected number of values ({len(values)})")
        except (ValueError, IndexError) as exc:
            report.skipped.append((ref, str(exc)))
    report.labels = sorted({a["label"] for a in report.annotations})
    return report


def find_yolo_label(image_path):
    """images/x.png -> labels/x.txt (Ultralytics layout) or x.txt beside image."""
    image_path = Path(image_path)
    parts = list(image_path.parts)
    candidates = [image_path.with_suffix(".txt")]
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        candidates.insert(0, Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
    return next((c for c in candidates if c.is_file()), None)
