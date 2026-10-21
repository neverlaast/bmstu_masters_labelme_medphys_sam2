"""Small geometry primitives independent of Qt and NumPy."""
from __future__ import annotations

from math import hypot


def point_segment_distance(point, line):
    """Euclidean distance from an x()/y() point to a finite 2D segment.

    Projection handles endpoints and zero-length segments without a cross
    product. The signature matches LabelMe's distancetoline helper.
    """
    start, end = line
    dx = float(end.x() - start.x())
    dy = float(end.y() - start.y())
    px = float(point.x() - start.x())
    py = float(point.y() - start.y())
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return hypot(px, py)
    t = min(1.0, max(0.0, (px * dx + py * dy) / length_squared))
    return hypot(px - t * dx, py - t * dy)
