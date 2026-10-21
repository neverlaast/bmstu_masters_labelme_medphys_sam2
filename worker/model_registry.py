from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from collections import OrderedDict

from worker.cache.embedding_cache import EmbeddingCache

from worker.postprocess.geometry import mask_to_polygon


PromptType = Literal["box", "point", "points", "mask", "auto"]
BackendType = str


@dataclass
class SamPoint:
    x: float
    y: float
    label: int  # 1 = positive, 0 = negative


@dataclass
class SamBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class SamMaskCandidate:
    polygon: list[list[float]] = field(default_factory=list)
    score: float | None = None
    logits_score: float | None = None
    label: str = "sam2_candidate"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamModelInfo:
    backend: BackendType
    model_name: str
    model_version: str | None = None
    checkpoint_path: str | None = None
    config_path: str | None = None
    device: str = "cpu"
    loaded: bool = False
    checkpoint_sha256: str | None = None
    variant: str | None = None


# SAM2.1 size variant -> config suffix (sam2.1_hiera_<variant>.pt <-> sam2.1_hiera_<suffix>.yaml)
SAM2_VARIANTS = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}


def sam2_variant_from_checkpoint(path: str | None) -> str | None:
    """'.../sam2.1_hiera_large.pt' -> 'large'; None for unknown names."""
    if not path:
        return None
    stem = Path(path).stem
    for variant in sorted(SAM2_VARIANTS, key=len, reverse=True):
        if stem.endswith("_" + variant):
            return variant
    return None


def sam2_variant_from_config(path: str | None) -> str | None:
    """'configs/sam2.1/sam2.1_hiera_l.yaml' -> 'large'; None for unknown names."""
    if not path:
        return None
    stem = Path(path).stem
    for variant, suffix in SAM2_VARIANTS.items():
        if stem.endswith("_hiera_" + suffix):
            return variant
    return None


def sam2_config_for_variant(variant: str) -> str:
    return f"configs/sam2.1/sam2.1_hiera_{SAM2_VARIANTS[variant]}.yaml"


def resolve_sam2_config(checkpoint_path: str | None, config_path: str | None) -> str:
    """Pick the config matching the checkpoint, or reject a mismatched pair.

    A config describes the architecture and a checkpoint holds weights for it;
    loading weights into another size fails deep inside torch with
    'Missing key(s) in state_dict', so the pair is checked up front.
    """
    ckpt_variant = sam2_variant_from_checkpoint(checkpoint_path)
    if config_path is None:
        if ckpt_variant is None:
            raise ValueError(
                f"Cannot infer SAM2 config from checkpoint name {checkpoint_path!r}; pass --config")
        return sam2_config_for_variant(ckpt_variant)
    cfg_variant = sam2_variant_from_config(config_path)
    if ckpt_variant and cfg_variant and ckpt_variant != cfg_variant:
        raise ValueError(
            f"Checkpoint is SAM2.1 {ckpt_variant} but config is {cfg_variant} ({config_path}); "
            f"use --config {sam2_config_for_variant(ckpt_variant)} or omit --config")
    return config_path


@dataclass
class SamInferenceRequest:
    image_path: str
    prompt_type: PromptType
    backend: BackendType = "stub"
    model_name: str = "sam2"
    device: str = "cpu"

    box: SamBox | None = None
    points: list[SamPoint] = field(default_factory=list)
    mask_input_path: str | None = None

    preprocess_params: dict[str, Any] = field(default_factory=dict)
    generation_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamInferenceResponse:
    ok: bool
    candidates: list[SamMaskCandidate] = field(default_factory=list)
    model_info: SamModelInfo | None = None
    elapsed_ms: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    generation_mode: str = "interactive"
    cache_hit: bool = False


from worker.adapters import SegmentationAdapter

# Backwards-compatible type name now refers to the full neutral contract.
SamModelAdapter = SegmentationAdapter


class StubSamModelAdapter:
    def get_capabilities(self):
        from worker.adapters import AdapterCapabilities
        return AdapterCapabilities(points=True, box=True)

    def clear_cache(self):
        pass

    def prepare_image(self, request):
        return request

    def compute_embedding(self, prepared):
        raise NotImplementedError("Demo adapter does not compute embeddings")

    def segment_prompt(self, request, prepared):
        return self.predict(request)

    def generate_masks(self, request, prepared):
        raise NotImplementedError("Demo adapter does not support automatic segmentation")

    def process_sequence(self, request):
        raise NotImplementedError("Demo adapter does not process sequences")

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def get_info(self) -> SamModelInfo:
        return SamModelInfo(
            backend="stub",
            model_name="sam2_stub",
            model_version="dev",
            device="cpu",
            loaded=True,
        )

    def predict(self, request: SamInferenceRequest) -> SamInferenceResponse:
        if request.generation_params.get("generation", {}).get("mode") == "automatic":
            return SamInferenceResponse(
                ok=False, model_info=self.get_info(), generation_mode="automatic",
                error="Stub backend is a demo only. Use sam2_real for object search.",
            )
        candidate = SamMaskCandidate(
            polygon=[
                [100.0, 100.0],
                [220.0, 100.0],
                [220.0, 220.0],
                [100.0, 220.0],
            ],
            score=0.95,
            label="sam2_candidate",
            metadata={
                "source": "stub",
                "prompt_type": request.prompt_type,
            },
        )
        return SamInferenceResponse(
            ok=True,
            candidates=[candidate],
            model_info=self.get_info(),
            elapsed_ms=15.0,
        )


class RealSam2ModelAdapter:
    def get_capabilities(self):
        from worker.adapters import AdapterCapabilities
        return AdapterCapabilities(points=True, box=True, mask_refinement=True,
                                   automatic=True, embeddings=True, sequence=False,
                                   propagation=True)

    def prepare_image(self, request):
        from worker.predictors.sam2_predictor import read_image
        return read_image(self, request)

    def compute_embedding(self, prepared):
        import torch
        from worker.predictors.sam2_predictor import encode_image
        with torch.inference_mode():
            return encode_image(self, prepared)

    def segment_prompt(self, request, prepared):
        return self.predict(request, prepared=prepared)

    def generate_masks(self, request, prepared):
        return self.predict(request, prepared=prepared)

    def process_sequence(self, request):
        raise NotImplementedError("SAM2 video propagation is not implemented in this adapter yet")

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self._loaded = False
        self._predictor = None
        self._load_error: str | None = None
        self._embedding_cache = EmbeddingCache()
        self._mask_inputs = OrderedDict()
        self._model_identity = None
        self._checkpoint_hash = None

    def load(self) -> None:
        if self._loaded:
            return

        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for sam2_real")
        if not self.config_path:
            raise ValueError("config_path is required for sam2_real")

        import sys

        sam2_repo_root = Path(__file__).resolve().parents[1] / "third_party" / "sam2_upstream"
        if str(sam2_repo_root) not in sys.path:
            sys.path.insert(0, str(sam2_repo_root))

        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            config_file=self.config_path,
            ckpt_path=str(checkpoint),
            device=self.device,
        )
        self._predictor = SAM2ImagePredictor(model)
        from worker.predictors.sam2_predictor import file_hash
        import hashlib
        config_file = sam2_repo_root / "sam2" / self.config_path
        config_digest = file_hash(config_file) if config_file.is_file() else self.config_path
        source_digest = hashlib.sha256()
        for path in sorted((sam2_repo_root / "sam2").rglob("*.py")):
            source_digest.update(str(path.relative_to(sam2_repo_root)).encode())
            source_digest.update(path.read_bytes())
        self._checkpoint_hash = file_hash(checkpoint)
        self._model_identity = hashlib.sha256(
            (self._checkpoint_hash + config_digest + source_digest.hexdigest()).encode()).hexdigest()
        self._loaded = True
        self._load_error = None

    def unload(self) -> None:
        self.clear_cache()
        self._predictor = None
        self._video_predictor = None
        self._loaded = False

    def video_predictor(self):
        """SAM2 video predictor sharing checkpoint/config/device with the image model."""
        if getattr(self, "_video_predictor", None) is None:
            self.load()
            from sam2.build_sam import build_sam2_video_predictor
            self._video_predictor = build_sam2_video_predictor(
                config_file=self.config_path, ckpt_path=str(self.checkpoint_path),
                device=self.device)
        return self._video_predictor

    def clear_cache(self) -> None:
        self._embedding_cache.clear()
        self._mask_inputs.clear()
        if self._predictor is not None:
            self._predictor.reset_predictor()

    def get_info(self) -> SamModelInfo:
        return SamModelInfo(
            backend="sam2_real",
            model_name="sam2",
            model_version=f"sam2.1:{self._model_identity[:12]}" if self._model_identity else "sam2.1",
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_path,
            device=self.device,
            loaded=self._loaded,
            checkpoint_sha256=self._checkpoint_hash,
            variant=sam2_variant_from_checkpoint(self.checkpoint_path),
        )

    def get_load_error(self) -> str | None:
        return self._load_error

    def ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            self.load()
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self._loaded = False
            self._predictor = None
            return False

    def predict(self, request: SamInferenceRequest, prepared=None) -> SamInferenceResponse:
        import time
        started = time.perf_counter()
        if not self.ensure_loaded():
            return SamInferenceResponse(
                ok=False,
                candidates=[],
                model_info=self.get_info(),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=self._load_error or "Failed to load SAM2 model",
            )

        mode = request.generation_params.get("generation", {}).get("mode", "interactive")
        if mode == "interactive" and request.box is None and not request.points:
            return SamInferenceResponse(
                ok=False,
                candidates=[],
                model_info=self.get_info(),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error="sam2_real requires at least one prompt: box or points",
            )

        import torch
        from worker.predictors.sam2_predictor import predict
        with torch.inference_mode():
            response = predict(self, request, prepared=prepared)
        response.elapsed_ms = (time.perf_counter()-started)*1000
        return response


def create_model_adapter(
    backend: BackendType,
    *,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
) -> SamModelAdapter:
    if backend == "stub":
        return StubSamModelAdapter()
    if backend == "sam2_real":
        return RealSam2ModelAdapter(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )
    from worker.adapters import create_registered_adapter
    return create_registered_adapter(
        backend, checkpoint_path=checkpoint_path, config_path=config_path, device=device)


# Neutral public names for plugin authors; legacy names remain compatible.
Point = SamPoint
Box = SamBox
MaskCandidate = SamMaskCandidate
ModelInfo = SamModelInfo
InferenceRequest = SamInferenceRequest
InferenceResponse = SamInferenceResponse
