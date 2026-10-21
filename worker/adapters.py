"""Model-independent contract and explicit trusted-code registration.

Plugins are chosen by the operator at worker startup, never imported from
project files or HTTP requests. A plugin runs Python code with worker rights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class AdapterCapabilities:
    points: bool = False
    box: bool = False
    mask_refinement: bool = False
    automatic: bool = False
    embeddings: bool = False
    sequence: bool = False

    def to_dict(self):
        return asdict(self)


class SegmentationAdapter(Protocol):
    """Prepared data/embeddings are opaque and owned by the chosen model."""

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def clear_cache(self) -> None: ...
    def get_info(self) -> Any: ...
    def get_capabilities(self) -> AdapterCapabilities: ...
    def prepare_image(self, request: Any) -> Any: ...
    def compute_embedding(self, prepared: Any) -> Any: ...
    def segment_prompt(self, request: Any, prepared: Any) -> Any: ...
    def generate_masks(self, request: Any, prepared: Any) -> Any: ...
    def process_sequence(self, request: Any) -> Any: ...


_FACTORIES: dict[str, Callable] = {}
_METHODS = (
    "load", "unload", "clear_cache", "get_info", "get_capabilities",
    "prepare_image", "compute_embedding", "segment_prompt",
    "generate_masks", "process_sequence",
)


def register_adapter(name, factory):
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("Adapter needs a non-empty name and a callable factory")
    if name in {"sam2_real", "stub"} or name in _FACTORIES:
        raise ValueError(f"Adapter name already registered/reserved: {name}")
    _FACTORIES[name] = factory


def validate_adapter(adapter):
    missing = [method for method in _METHODS if not callable(getattr(adapter, method, None))]
    if missing:
        raise ValueError("Invalid adapter: missing methods " + ", ".join(missing))
    if not isinstance(adapter.get_capabilities(), AdapterCapabilities):
        raise ValueError("get_capabilities must return AdapterCapabilities")
    # Verify serializability and model identity without loading weights.
    info = adapter.get_info()
    from worker.model_registry import ModelInfo
    if not isinstance(info, ModelInfo) or not info.model_name:
        raise ValueError("Adapter must return a ModelInfo record")
    return adapter


def create_registered_adapter(name, **options):
    if name not in _FACTORIES:
        raise ValueError(f"Unsupported backend: {name}")
    return validate_adapter(_FACTORIES[name](**options))


def load_trusted_adapter(reference, **options):
    """Explicit CLI opt-in, e.g. my_package.my_model:create_adapter."""
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("Use an installed trusted module:factory reference")
    module, attribute = reference.split(":")
    if not module or not attribute or not all(p.isidentifier() for p in module.split(".")) or not attribute.isidentifier():
        raise ValueError("Invalid module:factory reference")
    factory = getattr(import_module(module), attribute)
    if not callable(factory):
        raise ValueError("Adapter factory is not callable")
    return validate_adapter(factory(**options))


def run_adapter(adapter, request):
    import time
    started = time.perf_counter()
    capabilities = adapter.get_capabilities()
    mode = request.generation_params.get("generation", {}).get("mode", "interactive")
    if request.model_name not in {"auto", adapter.get_info().model_name}:
        raise ValueError("Requested model differs from the worker model")
    if mode == "automatic" and not capabilities.automatic:
        raise ValueError("This model does not support automatic mask generation")
    if mode == "interactive":
        if request.points and not capabilities.points:
            raise ValueError("This model does not support point prompts")
        if request.box and not capabilities.box:
            raise ValueError("This model does not support box prompts")
        if request.generation_params.get("refinement_token") and not capabilities.mask_refinement:
            raise ValueError("This model does not support refinement by previous mask")
    adapter.load()
    prepared = adapter.prepare_image(request)
    if capabilities.embeddings:
        prepared = adapter.compute_embedding(prepared)
    if mode == "automatic":
        response = adapter.generate_masks(request, prepared)
    else:
        response = adapter.segment_prompt(request, prepared)
    if response.generation_mode != mode:
        raise ValueError("Adapter returned an incompatible generation mode")
    response.elapsed_ms = (time.perf_counter() - started) * 1000
    return response
