from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PromptType = Literal["positive_point", "negative_point", "box", "circle"]


class APIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class PointPrompt(APIModel):
    x: float
    y: float
    label: Literal[0, 1]


class BoxPrompt(APIModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def check_area(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Rectangle must have positive width and height.")
        return self


class CirclePrompt(APIModel):
    center_x: float
    center_y: float
    radius: float = Field(gt=0)
    mode: Literal["single_object", "multiple_objects"] = "single_object"


class PolygonPrompt(APIModel):
    points: list[tuple[float, float]] = Field(min_length=3, max_length=1024)

    @model_validator(mode="after")
    def check_area(self):
        twice_area = sum(
            self.points[i][0] * self.points[(i+1) % len(self.points)][1]
            - self.points[(i+1) % len(self.points)][0] * self.points[i][1]
            for i in range(len(self.points))
        )
        if abs(twice_area) < 1e-8:
            raise ValueError("Polygon ROI must have non-zero area.")
        return self


class CandidateFilters(APIModel):
    min_area: float = Field(default=20, ge=0)
    max_area: float = Field(default=0, ge=0)  # 0 = unlimited
    min_diameter: float = Field(default=0, ge=0)
    max_diameter: float = Field(default=0, ge=0)
    max_aspect_ratio: float = Field(default=0, ge=0)
    min_roundness: float = Field(default=0, ge=0, le=1)
    min_convexity: float = Field(default=0, ge=0, le=1)
    min_roi_fraction: float = Field(default=.85, ge=0, le=1)
    min_score: float = Field(default=.5, ge=0, le=1)
    duplicate_iou: float = Field(default=.65, gt=0, le=1)
    overlap_warning: float = Field(default=.5, ge=0, le=1)
    exclude_border: bool = True

    @model_validator(mode="after")
    def check_ranges(self):
        if self.max_area and self.max_area < self.min_area:
            raise ValueError("Maximum area must not be below minimum area.")
        if self.max_diameter and self.max_diameter < self.min_diameter:
            raise ValueError("Maximum diameter must not be below minimum diameter.")
        if self.max_aspect_ratio and self.max_aspect_ratio < 1:
            raise ValueError("Aspect ratio must be >= 1, or 0 for no limit.")
        return self


class GenerationOptions(APIModel):
    mode: Literal["interactive", "automatic"] = "interactive"
    points_per_side: int = Field(default=8, ge=2, le=20)
    max_candidates: int = Field(default=64, ge=1, le=128)
    filters: CandidateFilters = Field(default_factory=CandidateFilters)


class SegmentationRequest(APIModel):
    image_path: str
    model_name: str = "auto"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    points: list[PointPrompt] = Field(default_factory=list, max_length=256)
    box: BoxPrompt | None = None
    circle: CirclePrompt | None = None
    polygon: PolygonPrompt | None = None
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    refinement_token: str | None = None
    refinement_target_id: str | None = None
    existing_polygons: list[list[tuple[float, float]]] = Field(default_factory=list)
    return_multiple_masks: bool = True
    display: dict | None = None

    @model_validator(mode="after")
    def check_prompts(self):
        if self.display is not None:
            from worker.imaging import normalise_display
            self.display = normalise_display(self.display)
        if sum(roi is not None for roi in (self.box, self.circle, self.polygon)) > 1:
            raise ValueError("Use only one ROI.")
        if self.generation.mode == "interactive" and self.polygon is not None:
            raise ValueError("Polygon ROI is supported in automatic mode.")
        if self.generation.mode == "interactive" and not self.points and self.box is None and self.circle is None:
            raise ValueError("At least one prompt is required.")
        if self.circle and self.circle.mode == "multiple_objects" and self.generation.mode != "automatic":
            raise ValueError("Multiple-object circle requires automatic mode.")
        if self.generation.mode == "automatic" and self.refinement_token:
            raise ValueError("Refinement is supported in interactive mode.")
        return self


class SegmentationCandidate(APIModel):
    score: float
    polygon: list[list[float]] = Field(default_factory=list)
    source: str = "sam2"
    refinement_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SegmentationResponse(APIModel):
    ok: bool
    model_name: str
    model_version: str
    device: str
    elapsed_ms: float
    candidates: list[SegmentationCandidate] = Field(default_factory=list)
    error_message: str | None = None
    api_version: int = 1
    generation_mode: Literal["interactive", "automatic"] = "interactive"
    cache_hit: bool = False
    warnings: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(APIModel):
    ok: bool
    service: str
    device: str
    model_loaded: bool
    api_version: int = 1
    cache_entries: int = 0
    model_name: str = "sam2"
    model_version: str = ""
    capabilities: dict[str, bool] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)


class PropagationObject(APIModel):
    object_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    mask_rle: dict


class PropagationRequest(APIModel):
    frame_paths: list[str] = Field(min_length=2, max_length=300)
    start_index: int = Field(ge=0)
    objects: list[PropagationObject] = Field(min_length=1, max_length=200)
    direction: Literal["forward", "backward", "both"] = "forward"
    max_frames: int | None = Field(default=None, ge=1, le=300)
    display: dict | None = None
    min_area: int = Field(default=4, ge=1)


class PropagationResponse(APIModel):
    ok: bool
    frames: list[dict] = Field(default_factory=list)
    lost: list[dict] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    error_message: str | None = None
