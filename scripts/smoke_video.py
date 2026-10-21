"""Video -> frames -> real SAM2 video predictor; reports IoU against ground truth.

A synthetic video with two moving dark discs ("cells") is written, decoded by
app.core.video exactly as the GUI does, the masks of frame 0 are given to the
SAM2 video predictor, and the propagated masks are compared to the known discs.

    .venv/bin/python scripts/smoke_video.py --video my.mp4   # only decode check
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.core.video import FrameRange, extract_frames, probe_video
from worker.postprocess.masks import mask_rle
from worker.propagation import decode_mask, propagate

SIZE, FRAMES, R = (320, 240), 16, 18


def centres(i):
    return [(60 + 8 * i, 80), (250, 60 + 6 * i)]


def disc(centre, shape=(SIZE[1], SIZE[0])):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= R ** 2


def write_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, SIZE)
    rng = np.random.default_rng(0)
    for i in range(FRAMES):
        img = np.clip(rng.normal(200, 8, (SIZE[1], SIZE[0], 3)), 0, 255).astype(np.uint8)
        for c in centres(i):
            cv2.circle(img, c, R, (60, 60, 60), -1)
        writer.write(img)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", default="tiny", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--video", help="Only probe/decode an existing video")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.video:
            info = probe_video(args.video)
            frames = extract_frames(args.video, tmp / "f", FrameRange(0, min(9, info.frame_count - 1)))
            print(json.dumps({"info": info.__dict__, "decoded": len(frames)}, indent=2))
            return
        video = tmp / "cells.avi"
        write_video(video)
        frames = extract_frames(video, tmp / "frames")
        from worker.model_registry import create_model_adapter
        root = Path(__file__).resolve().parents[1]
        suffix = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}[args.variant]
        adapter = create_model_adapter(
            "sam2_real", device=args.device,
            checkpoint_path=str(root / "third_party" / "sam2_upstream" / f"sam2.1_hiera_{args.variant}.pt"),
            config_path=f"configs/sam2.1/sam2.1_hiera_{suffix}.yaml")
        adapter.load()
        objects = [{"object_id": f"cell{k}", "label": "cell", "mask_rle": mask_rle(disc(c))}
                   for k, c in enumerate(centres(0))]
        result = propagate(adapter.video_predictor(), [str(p) for p in frames], 0, objects)
        ious = {}
        for frame in result["frames"]:
            for obj in frame["objects"]:
                k = int(obj["track_id"][-1])
                pred = decode_mask(obj["mask_rle"], SIZE[1], SIZE[0])
                gt = disc(centres(frame["index"])[k])
                ious.setdefault(obj["track_id"], []).append(
                    float((pred & gt).sum() / max((pred | gt).sum(), 1)))
        print(json.dumps({
            "frames": len(frames), "propagated_frames": len(result["frames"]),
            "lost": len(result["lost"]), "elapsed_ms": round(result["elapsed_ms"]),
            "iou_mean": {k: round(float(np.mean(v)), 3) for k, v in ious.items()},
            "iou_min": {k: round(float(np.min(v)), 3) for k, v in ious.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
