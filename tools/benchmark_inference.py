import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from groundingdino.models.GroundingDINO import ms_deform_attn
from groundingdino.util.inference import load_image, load_model, predict


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GroundingDINO single-image inference.")
    parser.add_argument("--config", required=True, help="Path to GroundingDINO config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--text-prompt", required=True, help="Text prompt used for grounding.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def synchronize(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def detect_ms_deform_attn_backend(device):
    force_fallback = os.environ.get("GROUNDINGDINO_MSDA_FORCE_FALLBACK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if force_fallback:
        return "pytorch_fallback_forced"
    if device.startswith("cuda") and torch.version.hip is not None and getattr(ms_deform_attn, "_C_AVAILABLE", False):
        return "custom_ext_hip_forward"
    if device.startswith("cuda") and torch.version.hip is not None:
        return "pytorch_fallback_rocm"
    if device.startswith("cuda") and getattr(ms_deform_attn, "_C_AVAILABLE", False):
        return "custom_ext_cuda"
    return "pytorch_fallback"


def main():
    args = parse_args()
    model = load_model(args.config, args.checkpoint, device=args.device)
    _, image = load_image(args.image)

    def run_once():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
        synchronize(args.device)
        return boxes, logits, phrases

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    warmup_latencies = []
    boxes, logits, phrases = None, None, None
    for _ in range(args.warmup):
        start = time.perf_counter()
        boxes, logits, phrases = run_once()
        warmup_latencies.append(time.perf_counter() - start)

    latencies = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        boxes, logits, phrases = run_once()
        latencies.append(time.perf_counter() - start)

    assert boxes is not None and logits is not None and phrases is not None
    result = {
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "ms_deform_attn_backend": detect_ms_deform_attn_backend(args.device),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "image": args.image,
        "text_prompt": args.text_prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "num_boxes": int(len(boxes)),
        "boxes": boxes.detach().cpu().tolist(),
        "logits": logits.detach().cpu().tolist(),
        "phrases": phrases,
        "warmup_latency_s": warmup_latencies,
        "latency_s": latencies,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_min_s": min(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "gpu_mem_mb": (
            round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)


if __name__ == "__main__":
    main()
