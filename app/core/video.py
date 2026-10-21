"""Video files -> ordered, lossless frame sequence for SAM2 propagation.

SAM2's video predictor consumes an ordered list of frames; the project stores
frames as images, so a video is decoded once into PNG files (lossless, unlike
re-encoding to JPEG) and the frames are added to the project in temporal order.
A sidecar ``video_frames.json`` keeps the mapping frame file -> source frame
index and timestamp, needed to relate annotations back to the original video.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")
VIDEO_FILTER = "Video (" + " ".join(f"*{e}" for e in VIDEO_EXTENSIONS) + ")"
MAX_EXTRACTED_FRAMES = 5000
SIDECAR = "video_frames.json"


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class FrameRange:
    start: int = 0
    end: int | None = None  # inclusive; None = last frame
    step: int = 1

    def indices(self, frame_count: int) -> range:
        if self.step < 1:
            raise VideoError("Frame step must be >= 1")
        end = frame_count - 1 if self.end is None else min(self.end, frame_count - 1)
        if not 0 <= self.start <= end:
            raise VideoError(f"Invalid frame range {self.start}..{end} for {frame_count} frames")
        return range(self.start, end + 1, self.step)


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _open(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"Cannot open video: {path}")
    return capture


def probe_video(path) -> VideoInfo:
    import cv2
    capture = _open(path)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:  # some containers do not store the frame count
            count = 0
            while capture.grab():
                count += 1
        if count <= 0:
            raise VideoError(f"Video has no decodable frames: {path}")
        return VideoInfo(path=str(Path(path).resolve()), frame_count=count,
                         fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                         width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def default_frames_dir(video_path, project_path=None) -> Path:
    video_path = Path(video_path)
    base = Path(project_path).resolve().parent if project_path else video_path.resolve().parent
    return base / f"{video_path.stem}_frames"


def extract_frames(video_path, output_dir, frame_range=FrameRange(), *,
                   progress=None, cancelled=None) -> list[Path]:
    """Decode selected frames to ``output_dir`` as PNG; returns paths in temporal order.

    Frames are read sequentially (grab/retrieve) rather than by seeking, because
    seeking in inter-frame codecs is often inaccurate to the exact frame.
    """
    import cv2
    info = probe_video(video_path)
    wanted = frame_range.indices(info.frame_count)
    if len(wanted) > MAX_EXTRACTED_FRAMES:
        raise VideoError(f"{len(wanted)} frames selected; limit is {MAX_EXTRACTED_FRAMES}. "
                         "Increase the step or narrow the range.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    wanted_set, last = set(wanted), wanted[-1]
    written, records = [], []
    capture = _open(video_path)
    try:
        index = 0
        while index <= last:
            if cancelled and cancelled():
                raise InterruptedError("Video import cancelled")
            if not capture.grab():
                break
            if index in wanted_set:
                ok, frame = capture.retrieve()
                if not ok:
                    raise VideoError(f"Cannot decode frame {index}")
                target = output_dir / f"{stem}_f{index:06d}.png"
                if not cv2.imwrite(str(target), frame):  # BGR in, PNG is lossless
                    raise VideoError(f"Cannot write {target}")
                written.append(target)
                records.append({"file": target.name, "frame_index": index,
                                "time_s": index / info.fps if info.fps > 0 else None})
                if progress:
                    progress(len(written), len(wanted), f"Frame {index + 1}/{info.frame_count}")
            index += 1
    finally:
        capture.release()
    if not written:
        raise VideoError("No frames were extracted")
    (output_dir / SIDECAR).write_text(json.dumps({
        "source": asdict(info), "range": asdict(frame_range), "frames": records,
    }, indent=2), encoding="utf-8")
    return written
