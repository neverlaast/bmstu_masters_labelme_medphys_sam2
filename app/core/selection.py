"""Region selection using geometry centroids, independent of Qt."""
from math import hypot


def polygon_contains(point, vertices):
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index+1) % len(vertices)]
        # Include the boundary.
        cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
        if abs(cross) < 1e-8 and min(x1,x2)-1e-8 <= x <= max(x1,x2)+1e-8 and min(y1,y2)-1e-8 <= y <= max(y1,y2)+1e-8:
            return True
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            inside = not inside
    return inside


def centroid(points, kind):
    if kind in {"circle", "point"}:
        return points[0]
    if kind == "polygon" and len(points) >= 3:
        twice_area, cx, cy = 0.0, 0.0, 0.0
        for i, (x1, y1) in enumerate(points):
            x2, y2 = points[(i+1) % len(points)]
            cross = x1*y2-x2*y1
            twice_area += cross
            cx += (x1+x2)*cross
            cy += (y1+y2)*cross
        if abs(twice_area) > 1e-8:
            return cx/(3*twice_area), cy/(3*twice_area)
    return sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points)


def inside_roi(point, request):
    x, y = point
    if request.circle:
        circle = request.circle
        return hypot(x-circle.center_x, y-circle.center_y) <= circle.radius
    if request.box:
        box = request.box
        return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2
    if request.polygon:
        return polygon_contains(point, request.polygon.points)
    return True
