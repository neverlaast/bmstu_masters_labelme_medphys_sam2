"""Real inference on a generated test image; not a medical accuracy benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
from app.worker_api.schemas import (
    CirclePrompt, SegmentationRequest, PointPrompt, BoxPrompt,
    GenerationOptions, CandidateFilters,
)
from worker.main import segmentation_request_to_sam_request, sam_response_to_segmentation_response
from worker.model_registry import RealSam2ModelAdapter
from worker.adapters import run_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--variant", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "smoke")
    parser.add_argument("--advanced", action="store_true", help="Test cache, mask refinement and ROI object generation")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (215, 215, 215))
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 45, 105, 105), fill=(35, 35, 35))
    draw.ellipse((145, 130, 195, 180), fill=(70, 70, 70))
    image_path = output / "synthetic_particles.png"
    image.save(image_path)
    checkpoint = ROOT / "third_party" / "sam2_upstream" / f"sam2.1_hiera_{args.variant}.pt"
    suffix = {"tiny": "t", "small": "s"}[args.variant]
    adapter = RealSam2ModelAdapter(
        checkpoint_path=str(checkpoint),
        config_path=f"configs/sam2.1/sam2.1_hiera_{suffix}.yaml",
        device=args.device,
    )
    try:
        request = SegmentationRequest(
            image_path=str(image_path), device=args.device,
            circle=CirclePrompt(center_x=75, center_y=75, radius=32))
        result = run_adapter(adapter, segmentation_request_to_sam_request(request))
        # Exercise production endpoint serialisation without a socket.
        sam_response_to_segmentation_response(result).model_dump_json()
        summary = {
            "ok": result.ok, "device": args.device, "variant": args.variant,
            "elapsed_ms": result.elapsed_ms, "error": result.error,
            "candidates": [{"score": c.score, "vertices": len(c.polygon),
                            "area_px": c.metadata.get("metrics", {}).get("area_px")}
                           for c in result.candidates],
        }
        (output / f"result_{args.variant}_{args.device}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if not result.ok or not result.candidates:
            raise SystemExit("Real SAM2 smoke test failed.")
        if args.advanced:
            token = result.candidates[0].metadata["refinement_token"]
            refinement = SegmentationRequest(
                image_path=str(image_path), device=args.device,
                points=[PointPrompt(x=75,y=75,label=1), PointPrompt(x=110,y=110,label=0)],
                refinement_token=token,
            )
            refined = run_adapter(adapter, segmentation_request_to_sam_request(refinement))
            search = SegmentationRequest(
                image_path=str(image_path), device=args.device,
                box=BoxPrompt(x1=20,y1=20,x2=240,y2=240),
                generation=GenerationOptions(
                    mode="automatic", points_per_side=5,
                    filters=CandidateFilters(min_area=100,max_area=5000,min_roundness=.4,min_score=.6)),
            )
            objects = run_adapter(adapter, segmentation_request_to_sam_request(search))
            sam_response_to_segmentation_response(refined).model_dump_json()
            sam_response_to_segmentation_response(objects).model_dump_json()
            advanced = {
                "refinement_ok": refined.ok, "refined_candidates":len(refined.candidates),
                "refinement_cache_hit":refined.cache_hit,
                "search_ok":objects.ok, "objects":len(objects.candidates),
                "search_cache_hit":objects.cache_hit, "cache_entries":len(adapter._embedding_cache),
                "search_elapsed_ms":objects.elapsed_ms,
                "areas_px":[c.metadata["metrics"]["area_px"] for c in objects.candidates],
                "warnings":objects.warnings,
            }
            (output / f"advanced_{args.variant}_{args.device}.json").write_text(
                json.dumps(advanced,indent=2),encoding="utf-8")
            print(json.dumps(advanced,indent=2))
            if not (refined.ok and refined.candidates and refined.cache_hit and objects.ok
                    and len(objects.candidates)>=2 and objects.cache_hit):
                raise SystemExit("Advanced SAM2 smoke test failed.")
    finally:
        adapter.unload()


if __name__ == "__main__":
    main()
