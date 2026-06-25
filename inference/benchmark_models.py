"""
RSP-ViT Model Inference Benchmark

Compares Vanilla ViT, Shared ViT, Stem-only ViT, and RSP-ViT on:
  - Parameter count & model file size
  - FLOPs per forward pass
  - GPU memory (peak allocation, if CUDA available)
  - CPU memory usage
  - Inference latency (mean & std)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# ---------- path setup ----------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training import build_vit_model
from config import config as _cfg

# ---------- variant definitions ----------
VARIANT_DEFS = [
    ("Vanilla ViT",   {"share_transformer_weights": False, "patch_embed_variant": "standard"}),
    ("Shared ViT",    {"share_transformer_weights": True,  "patch_embed_variant": "standard"}),
    ("Stem-only ViT", {"share_transformer_weights": False, "patch_embed_variant": "resnet_stem"}),
    ("RSP-ViT",       {"share_transformer_weights": True,  "patch_embed_variant": "resnet_stem"}),
]

# ---------- utils ----------
def format_num(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return f"{n:.2f}"


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def measure_flops(model: nn.Module, dummy_input: torch.Tensor) -> float:
    """Return FLOPs (multiply-add) for a single forward pass using FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy_input)
    total_flops = flop_counter.get_total_flops()
    return total_flops


def measure_latency(
    model: nn.Module,
    dummy_input: torch.Tensor,
    num_warmup: int = 50,
    num_iters: int = 200,
    device: str = "cpu",
) -> dict:
    """Measure inference latency (ms)."""
    model.eval()
    # warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(dummy_input)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                model(dummy_input)
                end = time.perf_counter()
            timings.append((end - start) * 1000)  # ms

    timings = sorted(timings)
    n = len(timings)
    mean = sum(timings) / n
    std = (sum((t - mean) ** 2 for t in timings) / n) ** 0.5
    p50 = timings[n // 2]
    p95 = timings[int(n * 0.95)]
    p99 = timings[int(n * 0.99)]
    return {"mean_ms": mean, "std_ms": std, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


def measure_gpu_memory(model: nn.Module, dummy_input: torch.Tensor) -> dict:
    """Measure peak GPU memory during inference (MB)."""
    if not torch.cuda.is_available():
        return {"peak_mb": float("nan"), "current_mb": float("nan")}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    peak = torch.cuda.max_memory_allocated()
    current = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
    return {"peak_mb": peak / (1024 * 1024), "current_mb": current / (1024 * 1024)}


def measure_cpu_memory() -> dict:
    """Measure current process CPU memory (MB)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
        }
    except ImportError:
        return {"rss_mb": float("nan"), "vms_mb": float("nan")}


def get_model_file_size(checkpoint_path: Path | None) -> float:
    """Return model checkpoint file size in MB."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return float("nan")
    return checkpoint_path.stat().st_size / (1024 * 1024)


def find_best_checkpoint(variant_key: str, search_dirs: list[Path]) -> Path | None:
    """Find the best checkpoint for a given variant in the search directories.

    Looks for a subdirectory whose name starts with the variant key,
    then for 'best_vit_model.pth' inside it.
    """
    for sd in search_dirs:
        if not sd.exists():
            continue
        for sub in sd.iterdir():
            if not sub.is_dir():
                continue
            if sub.name.startswith(variant_key):
                ckpt = sub / "best_vit_model.pth"
                if ckpt.exists():
                    return ckpt
    return None


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Benchmark all 4 ViT variants.")
    parser.add_argument(
        "--checkpoint-dirs",
        type=str,
        nargs="*",
        default=[
            str(_PROJECT_ROOT / "checkpoint" / "paper_ablation_4modes_5noise_seed42"),
            str(_PROJECT_ROOT / "checkpoint"),
        ],
        help="Directories to search for model checkpoints.",
    )
    parser.add_argument("--n-classes", type=int, default=9,
                        help="Number of output classes.")
    parser.add_argument("--img-size", type=int, default=192,
                        help="Input image size (assumed square).")
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to CSV file.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force CPU even if CUDA is available.")
    parser.add_argument("--print-json", action="store_true",
                        help="Also output results as JSON.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Device: {device}")

    search_dirs = [Path(d) for d in args.checkpoint_dirs]
    img_size = args.img_size
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    rows = []
    for name, kwargs in VARIANT_DEFS:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # ---------- build model ----------
        model = build_vit_model(
            n_classes=args.n_classes,
            **kwargs,
        )
        model.to(device)
        model.eval()

        # ---------- parameter count ----------
        param_info = count_parameters(model)
        print(f"  Parameters:     {format_num(param_info['total'])}")
        print(f"  Trainable:      {format_num(param_info['trainable'])}")

        # ---------- checkpoint size ----------
        variant_key = name.lower().replace("-", "_").split()[0]
        ckpt_path = find_best_checkpoint(variant_key, search_dirs)
        if ckpt_path is not None:
            print(f"  Checkpoint:     {ckpt_path}")
            # Load state dict
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  Missing keys:   {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys:{len(unexpected)}")
            model.to(device)
            model.eval()
        else:
            print(f"  Checkpoint:     (not found — using random init)")

        ckpt_size = get_model_file_size(ckpt_path)

        # ---------- FLOPs ----------
        dummy_cpu = torch.randn(1, 3, img_size, img_size)
        model_cpu = model.cpu() if device == "cuda" else model
        model_cpu.eval()
        try:
            flops = measure_flops(model_cpu, dummy_cpu)
            print(f"  FLOPs:          {format_num(flops)}")
        except Exception as e:
            flops = float("nan")
            print(f"  FLOPs:          (error: {e})")
        # Move model back to original device if needed
        if device == "cuda":
            model_cpu.to(device)
            model = model_cpu
        else:
            model = model_cpu
        model.eval()

        # Reset CUDA memory stats
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # ---------- latency ----------
        lat_info = measure_latency(model, dummy, args.num_warmup, args.num_iters, device)
        print(f"  Latency mean:   {lat_info['mean_ms']:.3f} ms")
        print(f"  Latency std:    {lat_info['std_ms']:.3f} ms")
        print(f"  Latency p50:    {lat_info['p50_ms']:.3f} ms")
        print(f"  Latency p95:    {lat_info['p95_ms']:.3f} ms")
        print(f"  Latency p99:    {lat_info['p99_ms']:.3f} ms")

        # ---------- GPU memory ----------
        gpu_mem = measure_gpu_memory(model, dummy)
        if not torch.cuda.is_available() or args.no_cuda:
            print(f"  GPU memory:     N/A (CPU only)")
        else:
            print(f"  GPU peak mem:   {gpu_mem['peak_mb']:.1f} MB")
            print(f"  GPU curr mem:   {gpu_mem['current_mb']:.1f} MB")

        row = {
            "model": name,
            "total_params": param_info["total"],
            "trainable_params": param_info["trainable"],
            "checkpoint_mb": ckpt_size,
            "flops": flops,
            "latency_mean_ms": lat_info["mean_ms"],
            "latency_std_ms": lat_info["std_ms"],
            "latency_p50_ms": lat_info["p50_ms"],
            "latency_p95_ms": lat_info["p95_ms"],
            "latency_p99_ms": lat_info["p99_ms"],
            "gpu_peak_mb": gpu_mem["peak_mb"],
            "gpu_current_mb": gpu_mem["current_mb"],
        }
        rows.append(row)

    # ---------- summary table ----------
    print(f"\n\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    header = (f"{'Model':<16} {'Params':>10} {'FLOPs':>12} "
              f"{'Latency':>10} {'GPU Mem':>10} {'Ckpt':>8}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        lat_str = f"{r['latency_mean_ms']:.2f}ms" if not pd_isnan(r['latency_mean_ms']) else "N/A"
        gpu_str = f"{r['gpu_peak_mb']:.0f}MB" if not pd_isnan(r['gpu_peak_mb']) else "N/A"
        ckpt_str = f"{r['checkpoint_mb']:.1f}MB" if not pd_isnan(r['checkpoint_mb']) else "N/A"
        print(f"{r['model']:<16} {format_num(r['total_params']):>10} "
              f"{format_num(r['flops']):>12} {lat_str:>10} "
              f"{gpu_str:>10} {ckpt_str:>8}")
    print(sep)

    # ---------- save CSV ----------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults saved to {out_path}")

    # ---------- JSON output ----------
    if args.print_json:
        print(f"\nJSON:\n{json.dumps(rows, indent=2, ensure_ascii=False)}")


def pd_isnan(x: float) -> bool:
    """NaN check that works for both float('nan') and torch nan."""
    return x != x


if __name__ == "__main__":
    main()
