"""Non-neural demonstration of a second backend using the common contract.

It is a contract example, NOT a medical model or a SAM replacement.
"""
import numpy as np
import cv2
from PIL import Image, ImageOps

from worker.adapters import AdapterCapabilities
from worker.model_registry import ModelInfo, MaskCandidate, InferenceResponse
from worker.postprocess.geometry import mask_to_polygon
from worker.postprocess.masks import mask_rle, metrics, passes_filters, roi_mask
from app.worker_api.schemas import GenerationOptions


class ThresholdDemoAdapter:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def clear_cache(self):
        pass

    def get_info(self):
        return ModelInfo(backend="threshold_demo", model_name="threshold_demo",
                         model_version="demo-1", device="cpu", loaded=self.loaded)

    def get_capabilities(self):
        return AdapterCapabilities(automatic=True)

    def prepare_image(self, request):
        with Image.open(request.image_path) as raw:
            if raw.mode in {"I", "F"} or raw.mode.startswith("I;16"):
                raise ValueError("Demo accepts only ordinary 8-bit images")
            return np.asarray(ImageOps.exif_transpose(raw).convert("L"))

    def compute_embedding(self, prepared):
        raise NotImplementedError("No embeddings in the threshold demo")

    def segment_prompt(self, request, prepared):
        raise NotImplementedError("No interactive prompts in the threshold demo")

    def process_sequence(self, request):
        raise NotImplementedError("No sequence support in the threshold demo")

    def generate_masks(self, request, prepared):
        options = GenerationOptions.model_validate(request.generation_params.get("generation", {}))
        roi = roi_mask(prepared.shape, **request.generation_params.get("roi", {}))
        threshold = (prepared < 128) & roi
        _, components = cv2.connectedComponents(threshold.astype(np.uint8), 8)
        candidates = []
        for component in range(1, int(components.max())+1):
            mask = components == component
            values = metrics(mask, roi)
            if not passes_filters(values, 1.0, options.filters.model_dump()):
                continue
            polygon = mask_to_polygon(mask)
            if len(polygon) >= 3:
                candidates.append(MaskCandidate(polygon=polygon, score=1.0, metadata={
                    "source": "threshold_demo", "mask_rle": mask_rle(mask), "mask_outline": polygon,
                    "raster_source_valid": True, "metrics": values,
                    "score_note": "Constant demo score, not a neural model confidence"}))
            if len(candidates) == options.max_candidates:
                break
        return InferenceResponse(ok=True, candidates=candidates, model_info=self.get_info(),
                                 generation_mode="automatic")


def create_adapter(*, device="cpu", **options):
    if device != "cpu":
        raise ValueError("Threshold demo supports CPU only; pass --device cpu")
    return ThresholdDemoAdapter()
