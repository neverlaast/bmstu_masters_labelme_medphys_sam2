"""Strict, versioned project schema; geometry remains in original pixel units."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    return str(uuid4())


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_identifiers(self):
        for key in ("id", "class_id"):
            if hasattr(self, key):
                value = getattr(self, key)
                if not value.strip() or value != value.strip():
                    raise ValueError(f"Invalid {key}")
        return self


class ClassDefinition(Record):
    id: str = Field(default_factory=identity, min_length=1)
    name: str = Field(min_length=1)
    color: str = Field(default="#ff5050", pattern=r"^#[0-9a-fA-F]{6}$")
    description: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    allowed_geometry: list[Literal["polygon", "rectangle", "circle", "line", "linestrip", "point"]] = Field(default_factory=lambda: [
        "polygon", "rectangle", "circle", "line", "linestrip", "point"])
    shortcut: str = ""
    active: bool = True

    @model_validator(mode="after")
    def validate_name(self):
        if self.name != self.name.strip() or not self.name:
            raise ValueError("Class name must be non-empty with no surrounding spaces")
        if not self.allowed_geometry or len(set(self.allowed_geometry)) != len(self.allowed_geometry):
            raise ValueError("Allowed geometry must be a non-empty unique list")
        return self


class Annotation(Record):
    id: str = Field(default_factory=identity, min_length=1)
    class_id: str = Field(min_length=1)
    shape_type: Literal["polygon", "rectangle", "circle", "line", "linestrip", "point"]
    points: list[tuple[float, float]]
    group_id: int | str | None = None
    description: str = ""
    flags: dict[str, bool] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    modified_at: str = Field(default_factory=now)
    review_status: str = "unreviewed"
    source: str = "manual"

    @model_validator(mode="after")
    def check_geometry(self):
        count = len(self.points)
        minimum = {"polygon": 3, "rectangle": 2, "circle": 2,
                   "line": 2, "linestrip": 2, "point": 1}[self.shape_type]
        if count < minimum:
            raise ValueError("Not enough geometry points")
        if self.shape_type in {"rectangle", "circle", "line", "point"} and count != minimum:
            raise ValueError("Unexpected number of geometry points")
        if any(self.flags.get(flag) for flag in ("sam2_prompt", "temporary", "pending")):
            raise ValueError("Temporary annotations do not belong in a project")
        return self


class ImageRecord(Record):
    id: str = Field(default_factory=identity, min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    annotations: list[Annotation] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)
    display: dict[str, Any] = Field(default_factory=dict)
    normalization: dict[str, Any] = Field(default_factory=lambda: {"mode": "RGB"})


class Project(Record):
    format: Literal["medphys-project"] = "medphys-project"
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=identity)
    name: str = "Untitled project"
    created_at: str = Field(default_factory=now)
    modified_at: str = Field(default_factory=now)
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[ClassDefinition] = Field(default_factory=list)
    sam_settings: dict[str, Any] = Field(default_factory=dict)
    export_profiles: dict[str, Any] = Field(default_factory=dict)
    import_profiles: dict[str, Any] = Field(default_factory=dict)
    model_records: list[dict[str, Any]] = Field(default_factory=list)
    journal: list[dict[str, Any]] = Field(default_factory=list)
    cache_policy: dict[str, Any] = Field(default_factory=lambda: {
        "storage": "worker_ram", "max_images": 2, "max_bytes": 128*1024*1024})
    autosave_seconds: int = Field(default=60, ge=10, le=3600)
    backup_count: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def check_relations(self):
        for collection, label in ((self.classes, "class"), (self.images, "image")):
            ids = [entry.id for entry in collection]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} ID")
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate class name")
        classes = {c.id: c for c in self.classes}
        all_ids = set()
        for image in self.images:
            for annotation in image.annotations:
                if annotation.id in all_ids:
                    raise ValueError("Duplicate annotation ID")
                all_ids.add(annotation.id)
                if annotation.class_id not in classes:
                    raise ValueError(f"Unknown class ID: {annotation.class_id}")
                if annotation.shape_type not in classes[annotation.class_id].allowed_geometry:
                    raise ValueError("Geometry is not allowed for its class")
                if any(not (0 <= x < image.width and 0 <= y < image.height)
                       for x, y in annotation.points):
                    raise ValueError("Annotation coordinates outside image")
                # Only structural integrity is enforced here. Geometric quality
                # (self-intersection, zero area, duplicate masks...) is a property
                # of work in progress: it is reported by Validate / Export, which
                # use app.core.export.validate_annotations, and must never make an
                # open project unsaveable.
                needed = {"polygon": 3, "rectangle": 2, "circle": 2}.get(annotation.shape_type)
                if needed and (len(annotation.points) < needed if annotation.shape_type == "polygon"
                               else len(annotation.points) != needed):
                    raise ValueError(f"{annotation.shape_type} has {len(annotation.points)} point(s)")
        return self


def annotation_payload(annotation, class_definition):
    return {
        **annotation.metadata,
        "object_id": annotation.id, "class_id": annotation.class_id,
        "label": class_definition.name,
        "points": [list(p) for p in annotation.points],
        "shape_type": annotation.shape_type, "group_id": annotation.group_id,
        "description": annotation.description, "flags": annotation.flags,
        "attributes": annotation.attributes,
        "modified_at": annotation.modified_at, "review_status": annotation.review_status,
        "source": annotation.source,
    }
