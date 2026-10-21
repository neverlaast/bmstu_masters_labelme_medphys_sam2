from __future__ import annotations

import json
import argparse
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from app.worker_api.schemas import (
    HealthResponse,
    SegmentationCandidate,
    SegmentationRequest,
    SegmentationResponse,
)
from worker.device import detect_best_device
from worker.model_registry import (
    RealSam2ModelAdapter,
    SamBox,
    SamInferenceRequest,
    SamPoint,
    create_model_adapter,
)
from worker.postprocess.geometry import circle_to_box
from worker.adapters import run_adapter, load_trusted_adapter


HOST = "127.0.0.1"
PORT = 8765
MODEL_NAME = "sam2"
MODEL_VERSION = "sam2.1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BACKEND = "sam2_real"
DEFAULT_SAM2_CHECKPOINT_PATH = str(
    PROJECT_ROOT / "third_party" / "sam2_upstream" / "sam2.1_hiera_small.pt"
)
DEFAULT_SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_s.yaml"

DEFAULT_DEVICE = detect_best_device()

MODEL_ADAPTER = create_model_adapter(
    DEFAULT_BACKEND,
    checkpoint_path=DEFAULT_SAM2_CHECKPOINT_PATH,
    config_path=DEFAULT_SAM2_CONFIG_PATH,
    device=DEFAULT_DEVICE,
)


def segmentation_request_to_sam_request(
    request: SegmentationRequest,
) -> SamInferenceRequest:
    prompt_type = "auto"
    box = None
    points = []

    if request.circle is not None:
        cx = float(request.circle.center_x)
        cy = float(request.circle.center_y)
        r = float(request.circle.radius)

        x1, y1, x2, y2 = circle_to_box(cx, cy, r)
        box = SamBox(
            x_min=float(x1),
            y_min=float(y1),
            x_max=float(x2),
            y_max=float(y2),
        )

        inner = r * 0.6
        outer = r * 1.2
        points = [
            SamPoint(x=cx, y=cy, label=1),
            SamPoint(x=cx - inner, y=cy, label=1),
            SamPoint(x=cx + inner, y=cy, label=1),
            SamPoint(x=cx, y=cy - inner, label=1),
            SamPoint(x=cx, y=cy + inner, label=1),
            SamPoint(x=cx - outer, y=cy, label=0),
            SamPoint(x=cx + outer, y=cy, label=0),
            SamPoint(x=cx, y=cy - outer, label=0),
            SamPoint(x=cx, y=cy + outer, label=0),
        ]
        prompt_type = "points"

    elif request.box is not None:
        box = SamBox(
            x_min=float(request.box.x1),
            y_min=float(request.box.y1),
            x_max=float(request.box.x2),
            y_max=float(request.box.y2),
        )
        prompt_type = "box"

    if request.points:
        points += [
            SamPoint(
                x=float(p.x),
                y=float(p.y),
                label=int(p.label),
            )
            for p in request.points
        ]
        prompt_type = "points"

    requested_device = request.device
    if requested_device == "auto":
        requested_device = DEFAULT_DEVICE

    return SamInferenceRequest(
        image_path=request.image_path,
        prompt_type=prompt_type,
        backend=DEFAULT_BACKEND,
        model_name=request.model_name or "auto",
        device=requested_device,
        box=box,
        points=points,
        preprocess_params={"display": request.display} if request.display else {},
        generation_params={
            "return_multiple_masks": request.return_multiple_masks,
            "generation": request.generation.model_dump(),
            "roi": (
                {"circle": request.circle.model_dump()} if request.circle else
                {"box": [request.box.x1, request.box.y1, request.box.x2, request.box.y2]} if request.box else
                {"polygon": request.polygon.points} if request.polygon else {}
            ),
            "refinement_token": request.refinement_token,
            "refinement_target_id": request.refinement_target_id,
            "existing_polygons": request.existing_polygons,
        },
    )


def sam_response_to_segmentation_response(
    sam_response,
) -> SegmentationResponse:
    from dataclasses import asdict
    candidates = [
        SegmentationCandidate(
            score=c.score if c.score is not None else 0.0,
            polygon=c.polygon,
            source=str(c.metadata.get("source", c.label)),
            refinement_token=c.metadata.get("refinement_token"),
            metadata=c.metadata,
        )
        for c in sam_response.candidates
    ]

    model_name = MODEL_NAME
    model_version = MODEL_VERSION
    device = DEFAULT_DEVICE

    if sam_response.model_info is not None:
        model_name = sam_response.model_info.model_name
        model_version = sam_response.model_info.model_version or MODEL_VERSION
        device = sam_response.model_info.device

    return SegmentationResponse(
        ok=sam_response.ok,
        model_name=model_name,
        model_version=model_version,
        device=device,
        elapsed_ms=sam_response.elapsed_ms or 0.0,
        candidates=candidates,
        error_message=sam_response.error,
        api_version=2,
        generation_mode=sam_response.generation_mode,
        cache_hit=sam_response.cache_hit,
        warnings=sam_response.warnings,
        model_info=asdict(sam_response.model_info) if sam_response.model_info else {},
    )


def build_health_payload() -> HealthResponse:
    from dataclasses import asdict
    model_loaded = False
    device = DEFAULT_DEVICE

    info = MODEL_ADAPTER.get_info()
    model_loaded = info.loaded
    device = info.device

    return HealthResponse(
        ok=True,
        service="sam2-worker",
        device=device,
        model_loaded=model_loaded,
        api_version=2,
        cache_entries=len(getattr(MODEL_ADAPTER, "_embedding_cache", [])),
        model_name=info.model_name,
        model_version=info.model_version or "",
        capabilities=MODEL_ADAPTER.get_capabilities().to_dict(),
        model_info=asdict(info),
    )


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "MedPhysSAM2Worker/0.2"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/model":
            from dataclasses import asdict
            self._send_json(200, {
                "model": asdict(MODEL_ADAPTER.get_info()),
                "capabilities": MODEL_ADAPTER.get_capabilities().to_dict()})
            return
        if self.path == "/health":
            payload = build_health_payload()
            self._send_json(200, payload.model_dump())
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/cache/clear":
            MODEL_ADAPTER.clear_cache()
            self._send_json(200, build_health_payload().model_dump())
            return
        if self.path == "/model/unload":
            MODEL_ADAPTER.unload()
            self._send_json(200, build_health_payload().model_dump())
            return
        if self.path == "/propagate":
            self._propagate()
            return
        if self.path != "/segment":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 0 < content_length <= 16 * 1024 * 1024:
                self._send_json(413, {"ok": False, "error": "Invalid request size"})
                return
            raw = self.rfile.read(content_length)
            data = json.loads(raw.decode("utf-8"))
            request = SegmentationRequest.model_validate(data)
        except Exception as exc:
            response = SegmentationResponse(
                ok=False,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                device=DEFAULT_DEVICE,
                elapsed_ms=0.0,
                candidates=[],
                error_message=str(exc),
            )
            self._send_json(400, response.model_dump())
            return

        try:
            if request.device not in {"auto", MODEL_ADAPTER.get_info().device}:
                raise ValueError(
                    f"Worker is configured for {MODEL_ADAPTER.get_info().device}. "
                    "Restart it with --device to change the execution device."
                )
            sam_request = segmentation_request_to_sam_request(request)
            sam_response = run_adapter(MODEL_ADAPTER, sam_request)
            response = sam_response_to_segmentation_response(sam_response)

            status = 200 if sam_response.ok else 500
            self._send_json(status, response.model_dump())

        except Exception as exc:
            response = SegmentationResponse(
                ok=False,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                device=DEFAULT_DEVICE,
                elapsed_ms=0.0,
                candidates=[],
                error_message=str(exc),
            )
            self._send_json(500, response.model_dump())

    def _propagate(self) -> None:
        from app.worker_api.schemas import PropagationRequest, PropagationResponse
        from worker.propagation import propagate
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError("Invalid request size")
            request = PropagationRequest.model_validate(json.loads(self.rfile.read(length)))
        except Exception as exc:
            self._send_json(400, PropagationResponse(ok=False, error_message=str(exc)).model_dump())
            return
        factory = getattr(MODEL_ADAPTER, "video_predictor", None)
        if factory is None:
            self._send_json(501, PropagationResponse(
                ok=False, error_message="This backend has no video predictor; use sam2_real").model_dump())
            return
        try:
            result = propagate(factory(), request.frame_paths, request.start_index,
                               [o.model_dump() for o in request.objects], request.direction,
                               request.max_frames, request.display, request.min_area)
            self._send_json(200, PropagationResponse(ok=True, **result).model_dump())
        except Exception as exc:
            self._send_json(500, PropagationResponse(ok=False, error_message=str(exc)).model_dump())

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    global DEFAULT_BACKEND, DEFAULT_DEVICE, MODEL_ADAPTER
    parser = argparse.ArgumentParser(description="Local SAM2 inference worker")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--adapter", help="Explicit trusted installed module:factory, overrides --backend")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--checkpoint", default=DEFAULT_SAM2_CHECKPOINT_PATH)
    parser.add_argument("--config", default=DEFAULT_SAM2_CONFIG_PATH)
    args = parser.parse_args()
    DEFAULT_BACKEND = args.backend
    DEFAULT_DEVICE = detect_best_device() if args.device == "auto" else args.device
    factory = load_trusted_adapter if args.adapter else create_model_adapter
    MODEL_ADAPTER = factory(
        args.adapter or DEFAULT_BACKEND, checkpoint_path=args.checkpoint,
        config_path=args.config, device=DEFAULT_DEVICE)
    DEFAULT_BACKEND = MODEL_ADAPTER.get_info().backend
    print(f"Worker listening on http://{args.host}:{args.port}")
    print(f"Backend: {DEFAULT_BACKEND}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {DEFAULT_DEVICE}")

    server = HTTPServer((args.host, args.port), WorkerHandler)
    # The GUI stops its worker with SIGTERM; turn it into a clean shutdown.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        MODEL_ADAPTER.unload()


if __name__ == "__main__":
    main()
