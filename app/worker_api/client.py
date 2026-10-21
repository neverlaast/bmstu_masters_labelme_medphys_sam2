from __future__ import annotations

from dataclasses import dataclass
import os

import requests

from app.worker_api.errors import WorkerConnectionError, WorkerResponseError
from app.worker_api.schemas import HealthResponse, SegmentationRequest, SegmentationResponse


@dataclass
class WorkerClient:
    base_url: str = "http://127.0.0.1:8765"

    def __post_init__(self):
        self.base_url = os.environ.get("MEDPHYS_WORKER_URL", self.base_url).rstrip("/")

    def clear_cache(self) -> HealthResponse:
        try:
            response = requests.post(f"{self.base_url}/cache/clear", json={}, timeout=(3, 10))
            response.raise_for_status()
            return HealthResponse.model_validate(response.json())
        except Exception as exc:
            raise WorkerResponseError("Cannot clear cache. Restart the worker (Segmentation / SAM2 > Restart worker).") from exc

    def unload_model(self) -> HealthResponse:
        try:
            response = requests.post(f"{self.base_url}/model/unload", json={}, timeout=(3, 10))
            response.raise_for_status()
            return HealthResponse.model_validate(response.json())
        except Exception as exc:
            raise WorkerResponseError("Cannot unload model. Restart the worker (Segmentation / SAM2 > Restart worker).") from exc

    def healthcheck(self, timeout: float = 1.5) -> HealthResponse:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WorkerConnectionError(str(exc)) from exc

        try:
            return HealthResponse.model_validate(response.json())
        except Exception as exc:
            raise WorkerResponseError(str(exc)) from exc

    def segment(self, payload: SegmentationRequest, timeout: float = 180.0) -> SegmentationResponse:
        try:
            response = requests.post(
                f"{self.base_url}/segment",
                json=payload.model_dump(),
                timeout=(3.0, timeout),
            )
        except requests.Timeout as exc:
            raise WorkerConnectionError(
                "SAM2 request timed out. The worker may still be calculating; "
                "reduce the search grid and retry after it finishes."
            ) from exc
        except requests.RequestException as exc:
            raise WorkerConnectionError(
                "Cannot contact SAM2 worker. Restart it via Segmentation / SAM2 > Restart worker, or check the worker log."
            ) from exc

        try:
            result = SegmentationResponse.model_validate(response.json())
            if not response.ok or not result.ok:
                raise WorkerResponseError(result.error_message or "SAM2 worker failed.")
            return result
        except WorkerResponseError:
            raise
        except Exception as exc:
            raise WorkerResponseError("Worker returned an invalid segmentation response.") from exc

    def propagate(self, payload, timeout: float = 900.0):
        from app.worker_api.schemas import PropagationResponse
        try:
            response = requests.post(f"{self.base_url}/propagate",
                                     json=payload.model_dump(), timeout=(3.0, timeout))
        except requests.Timeout as exc:
            raise WorkerConnectionError(
                "Propagation timed out; the worker may still be busy. Use fewer frames.") from exc
        except requests.RequestException as exc:
            raise WorkerConnectionError(f"Cannot reach the SAM2 worker: {exc}") from exc
        try:
            result = PropagationResponse.model_validate(response.json())
        except Exception as exc:
            raise WorkerResponseError(
                "Worker does not support propagation. Restart the updated worker.") from exc
        if not result.ok:
            raise WorkerResponseError(result.error_message or "Propagation failed")
        return result
