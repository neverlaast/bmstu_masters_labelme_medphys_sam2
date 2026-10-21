"""Scientific image I/O shared by the GUI and the segmentation worker.

Raw data (8/16/32-bit integer, float, 1..N channels) is never modified.
Labelling and SAM2 operate on an explicit, reproducible 8-bit RGB rendering
controlled by ``display`` settings, which are part of the embedding cache key.
Multi-page stacks are split losslessly into single-frame files because a
LabelMe sidecar stores annotations for exactly one image.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # optional, needed for multi-channel 16-bit TIFF that Pillow cannot read
    import tifffile
except ImportError:  # pragma: no cover - depends on the environment
    tifffile = None

TIFF_SUFFIXES = {".tif", ".tiff"}
MAX_CHANNELS = 64
DEFAULT_DISPLAY = {"mode": "auto", "low_pct": 0.5, "high_pct": 99.5,
                   "low": None, "high": None, "channels": None, "gamma": 1.0}


class StackImageError(ValueError):
    """A multi-page image must be split into frames before labelling."""


# ------------------------------------------------------------------ reading

def _tiff_pages(path):
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes, shape = series.axes, series.shape
        frames = 1
        for axis, size in zip(axes, shape):
            if axis in "TZQI" and size > 1:   # time, depth, unknown/sequence
                frames *= size
        return frames, axes, shape


def frame_count(path):
    path = Path(path)
    if tifffile is not None and path.suffix.lower() in TIFF_SUFFIXES:
        return _tiff_pages(path)[0]
    with Image.open(path) as image:
        return getattr(image, "n_frames", 1)


def _channels_last(array, axes=None):
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported image dimensions {array.shape}")
    if axes and len(axes) == 3 and axes[0] in "CS" and axes[-1] not in "CS":
        return np.moveaxis(array, 0, -1)
    if array.shape[0] <= MAX_CHANNELS < array.shape[-1] or (
            array.shape[0] <= 8 and array.shape[-1] > 8):
        return np.moveaxis(array, 0, -1)
    return array


def read_array(path, frame=0):
    """Return the raw pixel array (H, W) or (H, W, C) in its original dtype."""
    path = Path(path)
    if tifffile is not None and path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes, data = series.axes, series.asarray()
        frame_axes = [i for i, a in enumerate(axes) if a in "TZQI" and data.shape[i] > 1]
        rest = [i for i in range(data.ndim) if i not in frame_axes]
        data = np.transpose(data, frame_axes + rest)
        data = data.reshape((-1,) + data.shape[len(frame_axes):])
        if not 0 <= frame < data.shape[0]:
            raise IndexError(f"Frame {frame} is out of range ({data.shape[0]} frames)")
        data = data[frame]
        axes = "".join(axes[i] for i in rest)
        keep = [i for i, size in enumerate(data.shape) if size > 1 or axes[i] in "YX"]
        data = data.reshape([data.shape[i] for i in keep])
        axes = "".join(axes[i] for i in keep)
        return _channels_last(np.asarray(data), axes)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(frame)
        image = ImageOps.exif_transpose(image)
        if image.mode == "P":
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        elif image.mode in {"LA", "PA"}:
            image = image.convert("RGBA")
        elif image.mode == "CMYK":
            image = image.convert("RGB")
        return np.array(image)


def describe(path):
    array = read_array(path)
    channels = 1 if array.ndim == 2 else array.shape[2]
    finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
    return {"dtype": str(array.dtype), "width": int(array.shape[1]),
            "height": int(array.shape[0]), "channels": int(channels),
            "frames": int(frame_count(path)),
            "min": float(finite.min()) if finite.size else math.nan,
            "max": float(finite.max()) if finite.size else math.nan}


def needs_rendering(path):
    """True when LabelMe/Qt cannot show the file faithfully as is."""
    try:
        if frame_count(path) > 1:
            return True
        with Image.open(path) as image:
            return image.mode not in {"1", "L", "P", "RGB", "RGBA", "LA", "PA"}
    except Exception:
        # Pillow could not open it at all (e.g. multi-channel uint16 TIFF).
        return tifffile is not None and Path(path).suffix.lower() in TIFF_SUFFIXES


# ---------------------------------------------------------------- rendering

def normalise_display(display):
    result = dict(DEFAULT_DISPLAY)
    for key, value in (display or {}).items():
        if key not in DEFAULT_DISPLAY:
            raise ValueError(f"Unknown display setting: {key}")
        result[key] = value
    if result["mode"] not in {"auto", "manual", "native"}:
        raise ValueError("Display mode must be auto, manual or native")
    if not 0 <= float(result["low_pct"]) < float(result["high_pct"]) <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    if result["mode"] == "manual":
        if result["low"] is None or result["high"] is None or \
                not float(result["low"]) < float(result["high"]):
            raise ValueError("Manual display range needs low < high")
    gamma = float(result["gamma"])
    if not (math.isfinite(gamma) and 0.1 <= gamma <= 10):
        raise ValueError("Gamma must be within [0.1, 10]")
    channels = result["channels"]
    if channels is not None:
        if not isinstance(channels, (list, tuple)) or not 1 <= len(channels) <= 3 \
                or any(type(c) is not int or c < 0 for c in channels):
            raise ValueError("Channels must be a list of 1-3 non-negative indices")
        result["channels"] = list(channels)
    return result


def _scale(channel, low, high, gamma):
    channel = channel.astype(np.float64)
    if channel.dtype.kind == "f":
        channel = np.nan_to_num(channel, nan=low, posinf=high, neginf=low)
    span = high - low
    scaled = np.zeros_like(channel) if span <= 0 else np.clip((channel - low) / span, 0, 1)
    if gamma != 1.0:
        scaled = scaled ** (1.0 / gamma)
    return np.rint(scaled * 255).astype(np.uint8)


def render_rgb8(array, display=None):
    """Map a raw array to 8-bit RGB with an explicit, deterministic window."""
    settings = normalise_display(display)
    array = np.asarray(array)
    planes = [array] if array.ndim == 2 else [array[..., i] for i in range(array.shape[2])]
    if settings["mode"] == "native" or (array.dtype == np.uint8 and display is None):
        if array.dtype != np.uint8:
            raise ValueError("Native display is only possible for 8-bit images")
        if len(planes) in (1, 2):
            return np.stack([planes[0]] * 3, axis=-1)
        return np.stack(planes[:3], axis=-1)
    indices = settings["channels"] or list(range(min(3, len(planes))))
    if any(i >= len(planes) for i in indices):
        raise ValueError(f"Image has {len(planes)} channels; requested {indices}")
    rendered = []
    for i in indices:
        plane = planes[i]
        if settings["mode"] == "manual":
            low, high = float(settings["low"]), float(settings["high"])
        else:
            values = plane[np.isfinite(plane)] if plane.dtype.kind == "f" else plane
            if values.size == 0:
                low = high = 0.0
            else:
                low, high = np.percentile(values, [settings["low_pct"], settings["high_pct"]])
                if high <= low:
                    low, high = float(values.min()), float(values.max())
        rendered.append(_scale(plane, float(low), float(high), float(settings["gamma"])))
    if len(rendered) == 1:
        rendered *= 3
    elif len(rendered) == 2:
        rendered.append(np.zeros_like(rendered[0]))
    return np.stack(rendered, axis=-1)


def load_rgb8(path, display=None):
    if frame_count(path) > 1:
        raise StackImageError("Multi-page image: split it into frames first")
    array = read_array(path)
    if display is None and array.dtype == np.uint8:
        planes = 1 if array.ndim == 2 else array.shape[2]
        if planes in (1, 3, 4):
            return render_rgb8(array[..., :3] if planes == 4 else array, None)
    return render_rgb8(array, display)


def png_bytes(rgb):
    with io.BytesIO() as stream:
        Image.fromarray(rgb).save(stream, format="PNG")
        return stream.getvalue()


# ----------------------------------------------------------------- stacks

def split_stack(path, parent=None):
    """Write each frame to <stem>_frames/<stem>_f0001.tif without data loss.

    The output folder is published atomically and never overwritten.
    """
    path = Path(path)
    count = frame_count(path)
    if count <= 1:
        raise ValueError("The image has a single frame")
    parent = Path(parent) if parent else path.parent
    target = parent / f"{path.stem}_frames"
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    staging = Path(tempfile.mkdtemp(prefix=".medphys-split-", dir=parent))
    width = max(4, len(str(count)))
    try:
        outputs = []
        for index in range(count):
            array = read_array(path, index)
            name = f"{path.stem}_f{index + 1:0{width}d}.tif"
            if tifffile is not None:
                if array.ndim == 3 and array.dtype == np.uint8 and array.shape[2] in (3, 4):
                    tifffile.imwrite(staging / name, array, photometric="rgb")
                elif array.ndim == 3:
                    tifffile.imwrite(staging / name, np.moveaxis(array, -1, 0),
                                     photometric="minisblack", metadata={"axes": "CYX"})
                else:
                    tifffile.imwrite(staging / name, array, photometric="minisblack",
                                     metadata={"axes": "YX"})
            else:
                Image.fromarray(array).save(staging / name, format="TIFF")
            outputs.append(target / name)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return outputs
