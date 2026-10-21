from __future__ import annotations

import math

import cv2
import numpy as np


def circle_to_box(center_x: float, center_y: float, radius: float) -> list[float]:
    return [
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    ]


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    mask_u8 = (mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return []

    # Douglas-Peucker can fold thin parts of a mask into a self-intersecting
    # outline; fall back to less simplification until the polygon is simple.
    perimeter = cv2.arcLength(largest, True)
    polygon = []
    for factor in (0.002, 0.001, 0.0005, 0.0):
        approx = cv2.approxPolyDP(largest, factor * perimeter, True) if factor else largest
        polygon = _dedupe([[float(pt[0][0]), float(pt[0][1])] for pt in approx])
        if len(polygon) >= 3 and is_simple_polygon(polygon):
            return polygon
    return polygon if len(polygon) >= 3 else []


def _dedupe(points):
    out = []
    for p in points:
        if p not in out:
            out.append(p)
    return out


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 1e-12) - (v < -1e-12)

    def on(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(p1, p2, p3)) or (o2 == 0 and on(p1, p2, p4))
            or (o3 == 0 and on(p3, p4, p1)) or (o4 == 0 and on(p3, p4, p2)))


def is_simple_polygon(points) -> bool:
    """True if no two non-adjacent edges touch (same rule as export QA)."""
    n = len(points)
    if n > 4000:  # O(n^2); raw contours this long are taken as is
        return True
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return False
    return True
