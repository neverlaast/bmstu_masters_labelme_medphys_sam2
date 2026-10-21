"""Image embeddings, explicit mask refinement and grid-based ROI generation."""
from __future__ import annotations

import hashlib
import json
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps

from app.worker_api.schemas import GenerationOptions
from worker.postprocess.geometry import mask_to_polygon
from worker.postprocess.masks import (
    PackedMask, grid_points, largest_component, mask_iou, mask_rle,
    metrics, passes_filters, polygon_mask, roi_mask,
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(adapter, request):
    # Content, not just the filename/mtime, determines cache validity.
    image_hash = file_hash(request.image_path)
    from worker.imaging import load_rgb8
    # Same deterministic 8-bit rendering as the GUI; settings are in the key.
    image = load_rgb8(request.image_path, request.preprocess_params.get("display"))
    key = hashlib.sha256(json.dumps({
        "image": image_hash, "model": adapter._model_identity, "device": adapter.device,
        "preprocess": request.preprocess_params, "channels": "RGB",
        "shape": image.shape, "input_size": getattr(adapter._predictor.model, "image_size", 1024),
        "orientation": "EXIF_transpose",
    }, sort_keys=True).encode()).hexdigest()
    return image, key


def encode_image(adapter, prepared):
    image, key = prepared
    cached = adapter._embedding_cache.get(key)
    if cached is not None:
        adapter._predictor._features = cached["features"]
        adapter._predictor._orig_hw = list(cached["orig_hw"])
        adapter._predictor._is_image_set = True
        adapter._predictor._is_batch = False
    else:
        adapter._predictor.set_image(image)
        adapter._embedding_cache.put(key, {
            "features": adapter._predictor._features,
            "orig_hw": list(adapter._predictor._orig_hw),
        })
    return image, key, cached is not None


def prepare_image(adapter, request):
    return encode_image(adapter, read_image(adapter, request))


def remember_logits(adapter, image_key, logits):
    token = str(uuid4())
    adapter._mask_inputs[token] = (image_key, np.asarray(logits, np.float32).copy())
    while len(adapter._mask_inputs) > 128:
        adapter._mask_inputs.popitem(last=False)
    return token


def predict(adapter, request, prepared=None):
    from worker.model_registry import SamInferenceResponse, SamMaskCandidate

    started = time.perf_counter()
    generation = GenerationOptions.model_validate(request.generation_params.get("generation", {}))
    mode = generation.mode
    filters = generation.filters.model_dump()
    image, image_key, cache_hit = prepared if prepared is not None else prepare_image(adapter, request)
    height, width = image.shape[:2]
    roi_options = request.generation_params.get("roi", {})
    roi = roi_mask((height, width), **roi_options)
    if not np.any(roi):
        raise ValueError("ROI does not intersect the image.")
    existing = [
        PackedMask.from_mask(polygon_mask((height, width), points))
        for points in request.generation_params.get("existing_polygons", [])
    ]
    warnings = []
    group_id = str(uuid4()) if mode == "interactive" else None
    records = []

    def add(mask, score, logits):
        # A single candidate represents one connected object, not unrelated
        # components. Full topology (including holes) is preserved as RLE.
        if not np.isfinite(score):
            return
        mask = largest_component(mask)
        values = metrics(mask, roi)
        if values is None:
            return
        if mode == "automatic" and not passes_filters(values, score, filters):
            return
        if mode == "automatic":
            raw_roi_fraction = values["roi_fraction"]
            mask = largest_component(mask & roi)
            values = metrics(mask, roi)
            if not passes_filters(values, score, filters):
                return
            values["raw_roi_fraction"] = raw_roi_fraction
            values["clipped_to_roi"] = raw_roi_fraction < 1
        polygon = mask_to_polygon(mask)
        if len(polygon) < 3:
            return
        packed = PackedMask.from_mask(mask)
        if mode == "automatic":
            duplicates = [r for r in records if mask_iou(packed, r["mask"]) >= filters["duplicate_iou"]]
            if any(r["score"] >= score for r in duplicates):
                return
            duplicate_ids = {id(record) for record in duplicates}
            records[:] = [record for record in records if id(record) not in duplicate_ids]
        values["existing_overlap"] = max((mask_iou(packed, old) for old in existing), default=0.0)
        values["overlap_warning"] = bool(
            existing and values["existing_overlap"] >= filters["overlap_warning"])
        records.append({"mask": packed, "score": float(score), "polygon": polygon,
                        "metrics": values, "logits": np.asarray(logits, np.float32).copy()})
        records.sort(key=lambda r: r["score"], reverse=True)
        if len(records) > generation.max_candidates:
            records.pop()
            if "Candidate count limited by max_candidates." not in warnings:
                warnings.append("Candidate count limited by max_candidates.")

    if mode == "automatic":
        seeds = grid_points(roi, generation.points_per_side)
        negative = [(p.x, p.y) for p in request.points if p.label == 0]
        for x, y in seeds:
            coords = np.asarray([(x, y), *negative], np.float32)
            coords[:, 0] = np.clip(coords[:, 0], 0, width-1)
            coords[:, 1] = np.clip(coords[:, 1], 0, height-1)
            labels = np.asarray([1, *([0]*len(negative))], np.int32)
            masks, scores, logits = adapter._predictor.predict(
                point_coords=coords, point_labels=labels, box=None, multimask_output=True)
            for mask, score, logit in zip(masks, scores, logits):
                add(mask, float(score), logit)
    else:
        box = None
        if request.box:
            box = np.asarray([request.box.x_min, request.box.y_min,
                              request.box.x_max, request.box.y_max], np.float32)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width-1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height-1)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("ROI does not intersect the image.")
        coords = labels = None
        if request.points:
            coords = np.asarray([(p.x, p.y) for p in request.points], np.float32)
            coords[:, 0] = np.clip(coords[:, 0], 0, width-1)
            coords[:, 1] = np.clip(coords[:, 1], 0, height-1)
            labels = np.asarray([p.label for p in request.points], np.int32)
        mask_input = None
        token = request.generation_params.get("refinement_token")
        if token:
            stored = adapter._mask_inputs.get(token)
            if stored is None or stored[0] != image_key:
                raise ValueError("Refinement state expired or belongs to another image/model. Use Run SAM2 without refinement.")
            adapter._mask_inputs.move_to_end(token)
            mask_input = stored[1][None, ...]
        masks, scores, logits = adapter._predictor.predict(
            point_coords=coords, point_labels=labels, box=box, mask_input=mask_input,
            multimask_output=bool(request.generation_params.get("return_multiple_masks", True)))
        for mask, score, logit in zip(masks, scores, logits):
            add(mask, float(score), logit)
    candidates = []
    overlaps = 0
    for record in records:
        overlaps += record["metrics"]["overlap_warning"]
        mask = record["mask"].unpack()
        candidates.append(SamMaskCandidate(
            polygon=record["polygon"], score=record["score"], label="sam2_candidate",
            metadata={
                "source": "sam2_real", "generation_mode": mode,
                "alternative_group": group_id, "metrics": record["metrics"],
                "mask_rle": mask_rle(mask),
                "mask_outline": record["polygon"],
                "raster_source_valid": True,
                "refinement_token": remember_logits(adapter, image_key, record["logits"]),
                "refines_object_id": request.generation_params.get("refinement_target_id"),
                "image_key": image_key,
                "prompt_context": {
                    "roi": roi_options,
                    "points": [(p.x,p.y,p.label) for p in request.points],
                    "generation": generation.model_dump(),
                    "used_previous_mask": bool(request.generation_params.get("refinement_token")),
                },
            },
        ))
    if overlaps:
        warnings.append(f"{overlaps} candidate(s) significantly overlap existing annotations.")
    if mode == "automatic" and not records:
        warnings.append("No candidates passed filters. Increase grid density or relax filter thresholds.")
    return SamInferenceResponse(
        ok=True, candidates=candidates, model_info=adapter.get_info(),
        elapsed_ms=(time.perf_counter()-started)*1000,
        warnings=warnings, generation_mode=mode, cache_hit=cache_hit,
    )
