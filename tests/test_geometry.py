import math
import unittest

from app.core.geometry import point_segment_distance


class Point:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


class GeometryTests(unittest.TestCase):
    def test_horizontal_vertical_and_diagonal(self):
        cases = [
            (Point(4, 3), [Point(0, 0), Point(10, 0)], 3),
            (Point(3, 4), [Point(0, 0), Point(0, 10)], 3),
            (Point(4, 0), [Point(0, 0), Point(4, 4)], math.sqrt(8)),
        ]
        for point, line, distance in cases:
            with self.subTest(point=point, line=line):
                self.assertAlmostEqual(point_segment_distance(point, line), distance)
                self.assertAlmostEqual(point_segment_distance(point, list(reversed(line))), distance)

    def test_distance_to_segment_endpoints(self):
        line = [Point(0, 0), Point(10, 0)]
        self.assertAlmostEqual(point_segment_distance(Point(-3, 4), line), 5)
        self.assertAlmostEqual(point_segment_distance(Point(13, 4), line), 5)

    def test_zero_length_segment_uses_point_distance(self):
        self.assertAlmostEqual(
            point_segment_distance(Point(4, 5), [Point(1, 1), Point(1, 1)]), 5)

    def test_points_on_segment_and_endpoints(self):
        line = [Point(0, 0), Point(10, 0)]
        for x in [0, 4, 10]:
            self.assertEqual(point_segment_distance(Point(x, 0), line), 0)
