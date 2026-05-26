import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from profile_inference import (
    classify_event,
    event_time,
    load_image,
    load_model,
    run_inference,
    summarize_categories,
    synchronize,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile FP32 GroundingDINO.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--text-prompt", default="truck")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--compile-targets", default="none,transformer")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active", type=int, default=10)
    parser.add_argument("--output-dir", default="profile_outputs/compile_fp32")
    return parser.parse_args()


def apply_compile(model, target, mode):
    if target == "none":
        return model, {"compiled": False, "error": None}
    try:
        if target == "model":
            return torch.compile(model, mode=mode), {"compiled": True, "error": None}
        if target == "transformer":
            model.transformer = torch.compile(model.transformer, mode=mode)
            return model, {"compiled": True, "error": None}
        if target == "encoder":
            model.transformer.encoder = torch.compile(model.transformer.encoder, mode=mode)
            return model, {"compiled": True, "error": None}
        if target == "encoder_layers":
            for idx, layer in enumerate(model.transformer.encoder.layers):
                model.transformer.encoder.layers[idx] = torch.compile(layer, mode=mode)
            return model, {"compiled": True, "error": None}
    except Exception as exc:
        return model, {"compiled": False, "error": repr(exc)}
    return model, {"compiled": False, "error": f"unknown target: {target}"}


def reset_dynamo():
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
        from torch._dynamo.utils import counters

        counters.clear()
    except Exception:
        pass


def dynamo_summary():
    try:
        from torch._dynamo.utils import counters

        return {
            "graph_break_count": sum(counters["graph_break"].values()),
            "graph_breaks": dict(counters["graph_break"]),
            "unique_graphs": dict(counters.get("unique_graphs", {})),
            "frames": dict(counters.get("frames", {})),
            "inductor": dict(counters.get("inductor", {})),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def summarize_profiler(prof):
    rows = []
    for event in prof.key_averages():
        self_cuda_ms = event_time(event, "self_cuda_time_total") / 1000.0
        if self_cuda_ms <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "self_cuda_time_ms": self_cuda_ms,
                "cuda_time_ms": event_time(event, "cuda_time_total") / 1000.0,
                "cpu_time_ms": event.cpu_time_total / 1000.0,
                "calls": event.count,
            }
        )

    total_self_cuda_ms = sum(row["self_cuda_time_ms"] for row in rows)
    kernel_like_rows = [
        row
        for row in rows
        if not row["name"].startswith("aten::")
        and not row["name"].startswith("torch::")
        and "groundingdino_inference" not in row["name"]
    ]
    return {
        "total_self_cuda_time_ms": total_self_cuda_ms,
        "cuda_event_rows": len(rows),
        "cuda_event_calls": sum(row["calls"] for row in rows),
        "kernel_like_rows": len(kernel_like_rows),
        "kernel_like_calls": sum(row["calls"] for row in kernel_like_rows),
        "category_summary": summarize_categories(rows, total_self_cuda_ms),
        "top_cuda": sorted(rows, key=lambda row: row["self_cuda_time_ms"], reverse=True)[:30],
    }


def run_case(args, target):
    reset_dynamo()
    device = "cuda"
    image = load_image(args.image_path).to(device)
    model = load_model(args.config_file, args.checkpoint_path, device)
    model, compile_info = apply_compile(model, target, args.compile_mode)

    result = None
    compile_failed_at_runtime = None
    latencies = []

    try:
        for _ in range(args.warmup):
            result = run_inference(model, image, args.text_prompt, args.box_threshold, args.text_threshold)
        synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(args.active):
                synchronize()
                start = time.perf_counter()
                with record_function("groundingdino_inference"):
                    result = run_inference(
                        model,
                        image,
                        args.text_prompt,
                        args.box_threshold,
                        args.text_threshold,
                    )
                synchronize()
                latencies.append(time.perf_counter() - start)
                prof.step()
        profiler_summary = summarize_profiler(prof)
    except Exception as exc:
        compile_failed_at_runtime = repr(exc)
        profiler_summary = {}

    summary = {
        "target": target,
        "compile_mode": args.compile_mode,
        **compile_info,
        "runtime_error": compile_failed_at_runtime,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_min_s": min(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "detections": len(result["boxes"]) if result is not None else None,
        "phrases": result["phrases"] if result is not None else [],
        "dynamo": dynamo_summary(),
        **profiler_summary,
    }
    return summary


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA/HIP device is required"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = [target.strip() for target in args.compile_targets.split(",") if target.strip()]
    summary = {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "warmup": args.warmup,
        "active": args.active,
        "cases": [run_case(args, target) for target in targets],
    }

    payload = json.dumps(summary, indent=2)
    print(payload)
    (output_dir / "groundingdino_compile_fp32_summary.json").write_text(payload)


if __name__ == "__main__":
    main()
