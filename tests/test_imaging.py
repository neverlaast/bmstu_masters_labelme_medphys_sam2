"""Scientific image reading, display rendering and lossless stack splitting."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from worker import imaging
from worker.imaging import (
    StackImageError, frame_count, load_rgb8, needs_rendering, normalise_display,
    read_array, render_rgb8, split_stack,
)

needs_tifffile = unittest.skipIf(imaging.tifffile is None, "tifffile is not installed")


class ImagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_8bit_rgb_unchanged(self):
        data = np.random.default_rng(0).integers(0, 255, (12, 16, 3), dtype=np.uint8)
        path = self.root / "a.png"
        Image.fromarray(data).save(path)
        self.assertFalse(needs_rendering(path))
        np.testing.assert_array_equal(load_rgb8(path), data)

    def test_16bit_grey_auto_window(self):
        data = np.linspace(1000, 5000, 20 * 10).reshape(10, 20).astype(np.uint16)
        path = self.root / "g16.tif"
        Image.fromarray(data).save(path)
        self.assertTrue(needs_rendering(path))
        self.assertEqual(read_array(path).dtype, np.uint16)
        rgb = load_rgb8(path)
        self.assertEqual(rgb.shape, (10, 20, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[-1, -1, 0], 255)
        np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])

    def test_manual_window_and_gamma(self):
        data = np.array([[0, 100, 200, 300]], np.uint16)
        rgb = render_rgb8(data, {"mode": "manual", "low": 100, "high": 300})
        self.assertEqual(rgb[0, :, 0].tolist(), [0, 0, 128, 255])
        bright = render_rgb8(data, {"mode": "manual", "low": 100, "high": 300, "gamma": 2.0})
        self.assertGreater(bright[0, 2, 0], 128)

    def test_float_with_nan(self):
        data = np.array([[np.nan, 0.0], [0.5, 1.0]], np.float32)
        rgb = render_rgb8(data, {"mode": "auto", "low_pct": 0, "high_pct": 100})
        self.assertEqual(rgb[1, 1, 0], 255)
        self.assertEqual(rgb[0, 0, 0], 0)

    def test_channel_selection(self):
        data = np.zeros((4, 4, 5), np.uint16)
        data[..., 3] = 1000
        data[0, 0, 3] = 0
        rgb = render_rgb8(data, {"channels": [3], "low_pct": 0, "high_pct": 100})
        self.assertEqual(rgb[1, 1].tolist(), [255, 255, 255])
        two = render_rgb8(data, {"channels": [3, 0], "low_pct": 0, "high_pct": 100})
        self.assertEqual(two[1, 1].tolist(), [255, 0, 0])
        with self.assertRaises(ValueError):
            render_rgb8(data, {"channels": [7]})

    def test_invalid_settings_rejected(self):
        for bad in ({"mode": "x"}, {"low_pct": 90, "high_pct": 10}, {"gamma": 0},
                    {"mode": "manual", "low": 5, "high": 5}, {"unknown": 1},
                    {"channels": [0, 1, 2, 3]}):
            with self.assertRaises(ValueError, msg=bad):
                normalise_display(bad)

    @needs_tifffile
    def test_multichannel_16bit_stack_split_is_lossless(self):
        data = (np.arange(4 * 3 * 8 * 9) * 37 % 65535).astype(np.uint16).reshape(4, 3, 8, 9)
        path = self.root / "stack.tif"
        imaging.tifffile.imwrite(path, data, metadata={"axes": "TCYX"})
        self.assertEqual(frame_count(path), 4)
        with self.assertRaises(StackImageError):
            load_rgb8(path)
        before = path.read_bytes()
        outputs = split_stack(path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(outputs), 4)
        for index, output in enumerate(outputs):
            self.assertEqual(frame_count(output), 1)
            np.testing.assert_array_equal(read_array(output), np.moveaxis(data[index], 0, -1))
        self.assertEqual(load_rgb8(outputs[0]).shape, (8, 9, 3))
        with self.assertRaises(FileExistsError):
            split_stack(path)

    def test_pillow_multipage_split(self):
        path = self.root / "pages.tif"
        frames = [Image.fromarray(np.full((5, 6), i * 50, np.uint8)) for i in range(3)]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        self.assertEqual(frame_count(path), 3)
        outputs = split_stack(path)
        self.assertEqual([int(read_array(o)[0, 0]) for o in outputs], [0, 50, 100])

    def test_single_frame_split_rejected(self):
        path = self.root / "one.png"
        Image.new("L", (4, 4)).save(path)
        with self.assertRaises(ValueError):
            split_stack(path)


if __name__ == "__main__":
    unittest.main()
