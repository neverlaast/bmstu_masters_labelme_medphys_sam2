"""Mask propagation: worker logic with a deterministic fake SAM2 video predictor."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.export import decode_rle
from worker.postprocess.masks import mask_rle
from worker.propagation import propagate, validate_request


class ShiftingPredictor:
    """Moves every mask by +2 px in x per frame forward (-2 backward)."""

    def __init__(self, lose_after=None):
        self.lose_after = lose_after
        self.reset = False

    def init_state(self, **kwargs):
        return {"masks": {}}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        state["masks"][obj_id] = (frame_idx, np.asarray(mask, bool))

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse):
        ids = sorted(state["masks"])
        n = state.setdefault("n", 6)
        step = -1 if reverse else 1
        limit = max_frame_num_to_track or n
        frame, count = start_frame_idx, 0
        while 0 <= frame < n and count <= limit:
            logits = []
            for obj in ids:
                start, mask = state["masks"][obj]
                shifted = np.roll(mask, 2 * (frame - start), axis=1)
                if self.lose_after is not None and abs(frame - start) > self.lose_after and obj == 2:
                    shifted = np.zeros_like(shifted)
                logits.append(np.where(shifted, 5.0, -5.0)[None])
            yield frame, ids, np.stack(logits)
            frame += step
            count += 1

    def reset_state(self, state):
        self.reset = True


class PropagationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(6):
            path = Path(self.tmp.name) / f"f{i}.png"
            Image.new("L", (40, 30), 30 * i).save(path)
            self.paths.append(str(path))
        self.mask = np.zeros((30, 40), bool)
        self.mask[10:18, 10:18] = True
        small = np.zeros((30, 40), bool)
        small[2:6, 30:34] = True
        self.objects = [{"object_id": "a", "label": "cell", "mask_rle": mask_rle(self.mask)},
                        {"object_id": "b", "label": "dust", "mask_rle": mask_rle(small)}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_forward_tracks_identity_and_geometry(self):
        predictor = ShiftingPredictor()
        result = propagate(predictor, self.paths, 1, self.objects, "forward")
        self.assertEqual([f["index"] for f in result["frames"]], [2, 3, 4, 5])
        self.assertTrue(predictor.reset)
        frame = result["frames"][1]                      # index 3: +4 px
        cell = next(o for o in frame["objects"] if o["track_id"] == "a")
        moved = decode_rle(cell["mask_rle"], 30, 40)
        np.testing.assert_array_equal(moved, np.roll(self.mask, 4, axis=1))
        self.assertEqual(cell["label"], "cell")
        self.assertEqual(result["lost"], [])

    def test_backward_and_both(self):
        back = propagate(ShiftingPredictor(), self.paths, 3, self.objects, "backward")
        self.assertEqual([f["index"] for f in back["frames"]], [0, 1, 2])
        both = propagate(ShiftingPredictor(), self.paths, 3, self.objects, "both", max_frames=1)
        self.assertEqual([f["index"] for f in both["frames"]], [2, 4])

    def test_lost_objects_reported(self):
        result = propagate(ShiftingPredictor(lose_after=1), self.paths, 0, self.objects)
        self.assertEqual({l["index"] for l in result["lost"]}, {2, 3, 4, 5})
        self.assertTrue(all(l["track_id"] == "b" for l in result["lost"]))

    def test_validation(self):
        for args in ((self.paths[:1], 0, self.objects, "forward", None),
                     (self.paths, 9, self.objects, "forward", None),
                     (self.paths, 0, [], "forward", None),
                     (self.paths, 0, self.objects, "sideways", None),
                     (self.paths + self.paths[:1], 0, self.objects, "forward", None)):
            with self.assertRaises(ValueError):
                validate_request(*args)

    def test_size_mismatch_and_empty_mask(self):
        Image.new("L", (41, 30)).save(self.paths[4])
        with self.assertRaises(ValueError):
            propagate(ShiftingPredictor(), self.paths, 0, self.objects)
        empty = [{"object_id": "e", "label": "cell", "mask_rle": mask_rle(np.zeros((30, 40), bool))}]
        with self.assertRaises(ValueError):
            propagate(ShiftingPredictor(), self.paths[:3], 0, empty)


if __name__ == "__main__":
    unittest.main()
