"""Narrow runtime adapters for the pristine LabelMe 5.2.1 dependency.

    No files under third_party are changed. NumPy itself is not patched.
"""
from __future__ import annotations

from app.core.geometry import point_segment_distance


def install_labelme_compatibility():
    # LabelMe exports this function by value through utils.__init__. Both
    # names must refer to the adapter; changing only utils.qt is insufficient.
    from labelme import utils
    from labelme.utils import qt

    utils.distancetoline = point_segment_distance
    qt.distancetoline = point_segment_distance
