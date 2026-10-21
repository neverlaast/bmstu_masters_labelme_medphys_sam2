"""Mask propagation through an ordered frame sequence with the SAM2 video predictor.

Frames are rendered by ``worker.imaging`` exactly as in the GUI (same display
settings) and fed to SAM2 in memory, reproducing SAM2's own JPEG loader
normalisation without a lossy intermediate JPEG.  Each propagated object keeps
``track_id`` = id of the source object, so identity is preserved across frames.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from worker.imaging import load_rgb8
from worker.postprocess.geometry import mask_to_polygon
from worker.postprocess.masks import mask_rle

MAX_FRAMES = 300
MAX_OBJECTS = 200
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def decode_mask(rle, height, width):
    from app.core.export import decode_rle
    return decode_rle(rle, height, width)


def frame_tensor(rgb, image_size):
    import torch
    resized = np.array(Image.fromarray(rgb).resize((image_size, image_size))) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


@contextmanager
def in_memory_frames(frames):
    """Make SAM2's init_state consume our rendered frames instead of a JPEG folder."""
    import torch
    import sam2.sam2_video_predictor as svp
    original = svp.load_video_frames

    def load(video_path, image_size, offload_video_to_cpu, compute_device=None, **_):
        images = torch.stack([frame_tensor(f, image_size) for f in frames])
        if not offload_video_to_cpu and compute_device is not None:
            images = images.to(compute_device)
        return images, frames[0].shape[0], frames[0].shape[1]
    svp.load_video_frames = load
    try:
        yield
    finally:
        svp.load_video_frames = original


def validate_request(frame_paths, start_index, objects, direction, max_frames):
    if not 2 <= len(frame_paths) <= MAX_FRAMES:
        raise ValueError(f"Propagation needs 2..{MAX_FRAMES} frames")
    if not 0 <= start_index < len(frame_paths):
        raise ValueError("start_index is outside the frame list")
    if not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError(f"Propagation needs 1..{MAX_OBJECTS} objects")
    if direction not in {"forward", "backward", "both"}:
        raise ValueError("direction must be forward, backward or both")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if len({str(Path(p).resolve()) for p in frame_paths}) != len(frame_paths):
        raise ValueError("Frame list contains duplicates")


def propagate(predictor, frame_paths, start_index, objects, direction="forward",
              max_frames=None, display=None, min_area=1, progress=None):
    """Return {"frames": [{"index", "path", "objects": [...]}, ...], "lost": [...]}.

    ``objects``: [{"object_id", "label", "mask_rle"}] on the start frame.
    """
    validate_request(frame_paths, start_index, objects, direction, max_frames)
    started = time.perf_counter()
    rendered = [load_rgb8(p, display) for p in frame_paths]
    height, width = rendered[0].shape[:2]
    if any(f.shape[:2] != (height, width) for f in rendered):
        raise ValueError("All frames must have the same size")
    masks = [decode_mask(o["mask_rle"], height, width) for o in objects]
    if any(not m.any() for m in masks):
        raise ValueError("Every source object needs a non-empty mask")

    with in_memory_frames(rendered):
        state = predictor.init_state(video_path="<memory>", offload_video_to_cpu=True,
                                     offload_state_to_cpu=False)
    try:
        for number, mask in enumerate(masks, 1):
            predictor.add_new_mask(state, frame_idx=start_index, obj_id=number, mask=mask)
        runs = {"forward": [False], "backward": [True], "both": [False, True]}[direction]
        results, seen = {}, 0
        total = sum((len(frame_paths) - start_index - 1) if not r else start_index for r in runs)
        if max_frames:
            total = min(total, max_frames * len(runs))
        for reverse in runs:
            for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=start_index, max_frame_num_to_track=max_frames,
                    reverse=reverse):
                if frame_idx == start_index:
                    continue
                seen += 1
                if progress:
                    progress(seen, max(total, 1), f"Frame {frame_idx + 1}/{len(frame_paths)}")
                entries = []
                for obj_id, logit in zip(obj_ids, logits):
                    mask = (logit > 0.0).cpu().numpy().reshape(height, width) \
                        if hasattr(logit, "cpu") else (np.asarray(logit) > 0).reshape(height, width)
                    if int(mask.sum()) < min_area:
                        continue
                    polygon = mask_to_polygon(mask)
                    if len(polygon) < 3:
                        continue
                    source = objects[int(obj_id) - 1]
                    entries.append({"track_id": source["object_id"], "label": source["label"],
                                    "polygon": polygon, "mask_rle": mask_rle(mask),
                                    "area": int(mask.sum())})
                results[frame_idx] = entries
    finally:
        predictor.reset_state(state)
    frames, lost = [], []
    for index in sorted(results):
        frames.append({"index": index, "path": str(frame_paths[index]),
                       "objects": results[index]})
        present = {e["track_id"] for e in results[index]}
        lost.extend({"index": index, "track_id": o["object_id"]}
                    for o in objects if o["object_id"] not in present)
    return {"frames": frames, "lost": lost,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0}
