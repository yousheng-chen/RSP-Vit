"""
Batch-size sweep for all 4 ViT variants.

Measures:
  - Latency (ms) at batch_size = 1, 2, 4, 8, 16, 32
  - Throughput (images/sec)
  - Peak GPU memory (if CUDA available)

Generates a combined CSV and prints a summary table.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from training import build_vit_model

VARIANT_DEFS = [
    ("Vanilla ViT",   {"share_transformer_weights": False, "patch_embed_variant": "standard"}),
    ("Shared ViT",    {"share_transformer_weights": True,  "patch_embed_variant": "standard"}),
    ("Stem-only ViT", {"share_transformer_weights": False, "patch_embed_variant": "resnet_stem"}),
    ("RSP-ViT",       {"share_transformer_weights": True,  "patch_embed_variant": "resnet_stem"}),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
IMG_SIZE = 192
N_CLASSES = 9
N_WARMUP = 10
N_ITERS = 30


def find_best_checkpoint(variant_key: str) -> Path | None:
    base = _PROJECT_ROOT / "checkpoint" / "paper_ablation_4modes_5noise_seed42"
    if not base.exists():
        return None
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        if sub.name.startswith(variant_key):
            ckpt = sub / "best_vit_model.pth"
            if ckpt.exists():
                return ckpt
    return None


def measure_batch(model: torch.nn.Module, batch_size: int, device: str) -> dict:
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(N_ITERS):
            if device == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                model(dummy)
                t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

    timings.sort()
    n = len(timings)
    mean_ms = sum(timings) / n
    std_ms = (sum((t - mean_ms) ** 2 for t in timings) / n) ** 0.5
    throughput = batch_size * 1000 / mean_ms if mean_ms > 0 else 0
    p50 = timings[n // 2]
    p95 = timings[int(n * 0.95)]
    p99 = timings[int(n * 0.99)]

    gpu_peak = float("nan")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.empty_cache()

    return {
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput": throughput,
        "gpu_peak_mb": gpu_peak,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Num warmup: {N_WARMUP}, Num iters: {N_ITERS}")
    print()

    all_rows = []

    for name, kwargs in VARIANT_DEFS:
        print(f"--- {name} ---")

        model = build_vit_model(d_model=256, n_heads=8, n_layers=6,
                                patch_size=16, n_classes=N_CLASSES,
                                d_ff=1024, **kwargs)

        variant_key = name.lower().replace("-", "_").split()[0]
        ckpt_path = find_best_checkpoint(variant_key)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            print(f"  Loaded checkpoint: {ckpt_path.name}")
        else:
            print(f"  Using random init (no checkpoint found)")

        model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {total_params:,}")

        for bs in BATCH_SIZES:
            res = measure_batch(model, bs, device)
            res["model"] = name
            res["total_params"] = total_params
            all_rows.append(res)

            gpu_str = f" | GPU peak: {res['gpu_peak_mb']:.0f}MB" if device == "cuda" else ""
            print(f"  batch={bs:>2}: {res['mean_ms']:>8.2f} ms | "
                  f"{res['throughput']:>8.0f} img/s{gpu_str}")

        print()

    # --- Summary ---
    print("=" * 100)
    header = (f"{'Model':<16} {'Batch':>5} | {'Latency':>8} {'p50':>8} {'p95':>8} "
              f"| {'Throughput':>10} | {'Params':>8}")
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in all_rows:
        print(f"{r['model']:<16} {r['batch_size']:>5} | "
              f"{r['mean_ms']:>8.2f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} | "
              f"{r['throughput']:>10.0f} | {r['total_params']//1000:>4}K")
    print(sep)

    # --- Save CSV ---
    out_path = _SCRIPT_DIR / "batch_sweep_results.csv"
    fieldnames = ["model", "total_params", "batch_size",
                  "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms",
                  "throughput", "gpu_peak_mb"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nResults saved to {out_path}")

    # --- JSON ---
    json_path = _SCRIPT_DIR / "batch_sweep_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
