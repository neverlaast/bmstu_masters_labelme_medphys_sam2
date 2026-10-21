"""Atomic saves, optimistic concurrency, backups and resource verification."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from project_format.models import ImageRecord, Project, now


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_record(path, project_path):
    path = Path(path).resolve()
    from worker.imaging import frame_count, read_array
    if frame_count(path) != 1:
        raise ValueError("Multi-page image: use Image display -> Split multi-page image first")
    height, width = read_array(path).shape[:2]
    return ImageRecord(path=os.path.relpath(path, Path(project_path).resolve().parent),
                       sha256=file_sha256(path), width=width, height=height)


def resolve_image(project_path, image):
    return (Path(project_path).resolve().parent / image.path).resolve()


def resource_issues(project_path, project, progress=None):
    issues = []
    for number, image in enumerate(project.images):
        if progress:
            progress(number, len(project.images), f"Verifying {Path(image.path).name}")
        path = resolve_image(project_path, image)
        if not path.is_file():
            issues.append(f"Missing image: {image.path}")
            continue
        if file_sha256(path) != image.sha256:
            issues.append(f"Image checksum changed: {image.path}")
            continue
        with Image.open(path) as raw:
            width, height = ImageOps.exif_transpose(raw).size
        if (width, height) != (image.width, image.height):
            issues.append(f"Image dimensions changed: {image.path}")
    for record in project.model_records:
        checkpoint = record.get("checkpoint_path")
        if checkpoint and not (Path(project_path).resolve().parent / checkpoint).is_file():
            issues.append(f"Missing model checkpoint: {checkpoint}")
        elif checkpoint and record.get("checkpoint_sha256"):
            absolute = (Path(project_path).resolve().parent / checkpoint).resolve()
            if file_sha256(absolute) != record["checkpoint_sha256"]:
                issues.append(f"Model checkpoint checksum changed: {checkpoint}")
    return issues


def load_project(path, *, check_resources=True):
    path = Path(path).resolve()
    project = Project.model_validate_json(path.read_text(encoding="utf-8"))
    if check_resources:
        issues = resource_issues(path, project)
        if issues:
            raise ValueError("\n".join(issues))
    return project


def _atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_project(path, project, *, expected_sha256=None, require_new=False):
    """Return the saved SHA; reject stale writers before touching the old file."""
    path = Path(path).resolve()
    validated = Project.model_validate(project.model_dump(mode="json"))
    validated.modified_at = now()
    data = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False,
                      indent=2, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if require_new and path.exists():
        raise ValueError("Project file already exists")
    if expected_sha256 is not None and (
        not path.exists() or file_sha256(path) != expected_sha256
    ):
        raise ValueError("Project changed on disk; refusing to overwrite another version")
    if path.exists():
        # Preserve the previous bytes, even if the subsequent save fails.
        backup_dir = path.parent / f".{path.name}.backups"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"{now().replace(':', '-')}_{uuid4().hex[:8]}.json"
        _atomic_write(backup, path.read_bytes())
    _atomic_write(path, data)
    project.modified_at = validated.modified_at
    if path.parent.joinpath(f".{path.name}.backups").exists():
        backups = sorted(path.parent.joinpath(f".{path.name}.backups").glob("*.json"))
        for old in backups[:-validated.backup_count]:
            try:
                old.unlink()
            except OSError:
                import warnings
                warnings.warn("Project saved, but an older backup could not be pruned", RuntimeWarning)
    return hashlib.sha256(data).hexdigest()


def rebase_paths(project, old_path, new_path):
    result = project.model_copy(deep=True)
    for image in result.images:
        image.path = os.path.relpath(resolve_image(old_path, image), Path(new_path).resolve().parent)
    for model in result.model_records:
        if model.get("checkpoint_path"):
            absolute = (Path(old_path).resolve().parent / model["checkpoint_path"]).resolve()
            model["checkpoint_path"] = os.path.relpath(absolute, Path(new_path).resolve().parent)
    return result


def relocate_images(project, project_path, directory):
    """Match basename candidates by checksum; ambiguous matches are errors."""
    directory = Path(directory).resolve()
    result = project.model_copy(deep=True)
    for image in result.images:
        existing = resolve_image(project_path, image)
        if existing.is_file() and file_sha256(existing) == image.sha256:
            continue
        candidates = [p for p in directory.rglob(Path(image.path).name)
                      if p.is_file() and file_sha256(p) == image.sha256]
        if len(candidates) != 1:
            raise ValueError(f"Expected one checksum-matching file for {image.path}, found {len(candidates)}")
        image.path = os.path.relpath(candidates[0], Path(project_path).resolve().parent)
    issues = resource_issues(project_path, result)
    if issues:
        raise ValueError("\n".join(issues))
    return result
