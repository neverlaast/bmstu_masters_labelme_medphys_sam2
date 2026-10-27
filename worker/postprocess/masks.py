"""Mask-aware ROI, metrics and duplicate suppression (not bbox NMS)."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import cv2
import numpy as np

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


@dataclass
class PackedMask:
    shape: tuple
    bits: np.ndarray
    area: int
    bbox: tuple | None = None
    crop_box: tuple | None = None  # Storage window in full-image coordinates.

    @classmethod
    def from_mask(cls, mask):
        mask = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1) if len(xs) else None
        return cls(mask.shape, np.packbits(mask.reshape(-1)), len(xs), bbox)

    def unpack(self):
        if self.crop_box is None:
            return np.unpackbits(self.bits, count=int(np.prod(self.shape))).reshape(self.shape).astype(bool)
        x1, y1, x2, y2 = self.crop_box
        mask = np.zeros(self.shape, bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = np.unpackbits(
                self.bits, count=(y2-y1)*(x2-x1)).reshape(y2-y1, x2-x1).astype(bool)
        return mask

    @classmethod
    def from_polygon(cls, shape, points):
        """Rasterize only clipped polygon bounds, never an image per object."""
        height, width = shape
        if len(points) < 3:
            return cls(shape, np.empty(0, np.uint8), 0, crop_box=(0, 0, 0, 0))
        vertices = np.rint(points).astype(np.int32)
        x1 = max(0, int(vertices[:, 0].min()))
        y1 = max(0, int(vertices[:, 1].min()))
        x2 = min(width, int(vertices[:, 0].max())+1)
        y2 = min(height, int(vertices[:, 1].max())+1)
        if x2 <= x1 or y2 <= y1:
            return cls(shape, np.empty(0, np.uint8), 0, crop_box=(0, 0, 0, 0))
        local = cls.from_mask(polygon_mask(
            (y2-y1, x2-x1), vertices - np.array([x1, y1], dtype=np.int32)))
        bbox = None if local.bbox is None else (
            local.bbox[0]+x1, local.bbox[1]+y1,
            local.bbox[2]+x1, local.bbox[3]+y1)
        crop = None if (x1, y1, x2, y2) == (0, 0, width, height) else (x1, y1, x2, y2)
        return cls(shape, local.bits, local.area, bbox, crop)


def _bits_in_window(mask, x1, y1, x2, y2):
    sx, sy, ex, _ = mask.crop_box or (0, 0, mask.shape[1], mask.shape[0])
    indices = ((np.arange(y1, y2, dtype=np.int64)-sy)[:, None] * (ex-sx)
               + (np.arange(x1, x2, dtype=np.int64)-sx)[None, :])
    return (mask.bits[indices // 8] >> (7-indices % 8)) & 1


def mask_iou(a, b):
    if a.shape != b.shape:
        raise ValueError("Masks must have identical dimensions.")
    if not a.bbox or not b.bbox:
        return 0.0
    if min(a.bbox[2],b.bbox[2]) <= max(a.bbox[0],b.bbox[0]) or min(a.bbox[3],b.bbox[3]) <= max(a.bbox[1],b.bbox[1]):
        return 0.0
    if a.crop_box is None and b.crop_box is None:
        intersection = int(np.sum(POPCOUNT[np.bitwise_and(a.bits, b.bits)], dtype=np.int64))
    else:
        x1, y1 = max(a.bbox[0], b.bbox[0]), max(a.bbox[1], b.bbox[1])
        x2, y2 = min(a.bbox[2], b.bbox[2]), min(a.bbox[3], b.bbox[3])
        rows = max(1, 65536 // (x2-x1))
        intersection = 0
        for start in range(y1, y2, rows):
            end = min(y2, start+rows)
            intersection += int(np.count_nonzero(
                _bits_in_window(a, x1, start, x2, end)
                & _bits_in_window(b, x1, start, x2, end)))
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def roi_mask(shape, circle=None, box=None, polygon=None):
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if circle is not None:
        yy, xx = np.ogrid[:height, :width]
        return (xx-circle["center_x"])**2 + (yy-circle["center_y"])**2 <= circle["radius"]**2
    if box is not None:
        x1, y1, x2, y2 = box
        start_x, end_x = np.clip([int(np.ceil(x1)), int(np.floor(x2))+1], 0, width)
        start_y, end_y = np.clip([int(np.ceil(y1)), int(np.floor(y2))+1], 0, height)
        if end_x > start_x and end_y > start_y:
            mask[start_y:end_y, start_x:end_x] = 1
    elif polygon is not None:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    else:
        mask[:] = 1
    return mask.astype(bool)


def grid_points(roi, points_per_side):
    ys, xs = np.nonzero(roi)
    if not len(xs):
        return []
    # Sample cell centres to avoid placing points exactly on the ROI boundary.
    x_values = xs.min() + (np.arange(points_per_side)+.5) * (xs.max()-xs.min()+1)/points_per_side
    y_values = ys.min() + (np.arange(points_per_side)+.5) * (ys.max()-ys.min()+1)/points_per_side
    points, seen = [], set()
    for y in y_values:
        for x in x_values:
            xi, yi = min(int(x), roi.shape[1]-1), min(int(y), roi.shape[0]-1)
            if roi[yi, xi] and (xi,yi) not in seen:
                points.append((float(xi), float(yi)))
                seen.add((xi,yi))
    return points


def largest_component(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8), 8)
    if len(stats) <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def metrics(mask, roi):
    mask = np.asarray(mask, bool)
    area = int(np.count_nonzero(mask))
    if not area:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    contour_area = cv2.contourArea(contour)
    rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
    return {
        "area_px": area,
        "diameter_px": 2*sqrt(area/pi),
        "aspect_ratio": max(1.0, max(rotated_width, rotated_height)/max(1e-8, min(rotated_width, rotated_height))),
        "roundness": min(1.0, 4*pi*area/(perimeter*perimeter)) if perimeter else 0.0,
        "convexity": min(1.0, contour_area/hull_area) if hull_area else 0.0,
        "roi_fraction": float(np.count_nonzero(mask & roi))/area,
        "touches_image_border": bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
    }


def passes_filters(values, score, filters):
    if values is None or not np.isfinite(score) or score < filters["min_score"]:
        return False
    area, diameter = values["area_px"], values["diameter_px"]
    return (
        area >= filters["min_area"]
        and not (filters.get("exclude_border", False) and values["touches_image_border"])
        and (not filters["max_area"] or area <= filters["max_area"])
        and diameter >= filters["min_diameter"]
        and (not filters["max_diameter"] or diameter <= filters["max_diameter"])
        and (not filters["max_aspect_ratio"] or values["aspect_ratio"] <= filters["max_aspect_ratio"])
        and values["roundness"] >= filters["min_roundness"]
        and values["convexity"] >= filters["min_convexity"]
        and values["roi_fraction"] >= filters["min_roi_fraction"]
    )


def mask_rle(mask):
    """Uncompressed COCO-compatible RLE preserving disconnected areas/holes."""
    flat = np.asarray(mask, np.uint8).reshape(-1, order="F")
    changes = np.flatnonzero(np.diff(flat)) + 1
    lengths = np.diff(np.concatenate(([0], changes, [len(flat)]))).astype(int).tolist()
    if len(flat) and flat[0]:
        lengths.insert(0, 0)
    return {"size": list(mask.shape), "counts": lengths}


def polygon_mask(shape, points):
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)
