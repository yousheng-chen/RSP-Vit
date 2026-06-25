"""RSP-ViT Model Inference Benchmark"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import torch, torch.nn as nn
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from training import build_vit_model

VARIANT_DEFS = [
    ("Vanilla ViT",   {"share_transformer_weights": False, "patch_embed_variant": "standard"}),
    ("Shared ViT",    {"share_transformer_weights": True,  "patch_embed_variant": "standard"}),
    ("Stem-only ViT", {"share_transformer_weights": False, "patch_embed_variant": "resnet_stem"}),
    ("RSP-ViT (Ours)", {"share_transformer_weights": True,  "patch_embed_variant": "resnet_stem"}),
]

VARIANT_KEY_MAP = {
    "Vanilla ViT":    "vanilla_vit",
    "Shared ViT":     "shared_vit",
    "Stem-only ViT":  "stem_only_vit",
    "RSP-ViT (Ours)": "rsp_vit",
}

def format_num(n):
    if n != n or n is None: return "N/A"
    if n >= 1e9: return f"{n/1e9:.2f}G"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return f"{n:.4f}"

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def measure_flops(model, img_size=192):
    total = [0]
    def _h(m, i, o):
        if isinstance(m, nn.Linear): total[0] += 2 * m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d): total[0] += 2 * o.shape[0] * m.out_channels * o.shape[2] * o.shape[3] * m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
    hooks = [mod.register_forward_hook(_h) for mod in model.modules()]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, img_size, img_size))
    for h in hooks: h.remove()
    return float(total[0])

def measure_latency(model, dummy_input, num_warmup=30, num_iters=100, device="cpu"):
    model.eval(); sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    with torch.no_grad():
        for _ in range(num_warmup): model(dummy_input)
    timings = []
    with torch.no_grad():
        for _ in range(num_iters):
            sync(); t0 = time.perf_counter(); model(dummy_input); sync(); t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
    timings.sort(); n = max(len(timings), 1)
    mean = sum(timings) / n
    variance = sum((t - mean)**2 for t in timings) / n
    return {"mean_ms": mean, "std_ms": math.sqrt(variance), "p50_ms": timings[n//2], "p95_ms": timings[int(n*0.95)], "p99_ms": timings[int(n*0.99)], "min_ms": timings[0], "max_ms": timings[-1]}

def measure_gpu_memory(model, dummy_input):
    torch.cuda.reset_peak_memory_stats(); model.eval()
    with torch.no_grad(): _ = model(dummy_input)
    peak = torch.cuda.max_memory_allocated(); reserved = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return {"peak_mb": peak/(1024*1024), "reserved_mb": reserved/(1024*1024)}

def measure_cpu_memory():
    try:
        import psutil, os; m = psutil.Process(os.getpid()).memory_info()
        return {"rss_mb": m.rss/(1024*1024), "vms_mb": m.vms/(1024*1024)}
    except: return {"rss_mb": float("nan"), "vms_mb": float("nan")}

def get_model_file_size(p):
    if p is None or not p.exists(): return float("nan")
    return p.stat().st_size/(1024*1024)

def find_checkpoint_with_metrics(variant_name, search_dirs, noise_label="-10db"):
    vkey = VARIANT_KEY_MAP.get(variant_name)
    if vkey is None: return None, {}
    for sd in search_dirs:
        if not sd.exists(): continue
        subs = sorted([d for d in sd.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        for sub in subs:
            dn = sub.name.lower()
            if not dn.startswith(vkey): continue
            if noise_label and noise_label.lower() not in dn: continue
            ckpt = sub / "best_vit_model.pth"; mf = sub / "metrics.json"
            met = {}
            if mf.exists():
                try: met = json.loads(mf.read_text(encoding="utf-8"))
                except: pass
            if ckpt.exists(): return ckpt, met
    return None, {}

def build_profile_model(**kw):
    return build_vit_model(d_model=256, n_heads=8, n_layers=6, patch_size=16, d_ff=1024, use_depth_embeddings=False, ffn_variant="standard", **kw)

def main():
    parser = argparse.ArgumentParser(description="Benchmark all 4 ViT variants.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--noise-level", type=str, default="-10db")
    parser.add_argument("--checkpoint-dirs", type=str, nargs="*", default=None)
    parser.add_argument("--n-classes", type=int, default=9)
    parser.add_argument("--img-size", type=int, default=192)
    parser.add_argument("--num-warmup", type=int, default=30)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--no-cuda", action="store_true", default=None)
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--skip-flops", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Device: {device}")
    if args.checkpoint_dirs is None:
        d1 = _PROJECT_ROOT / "checkpoint" / "paper_ablation_4modes_5noise_seed42"
        search_dirs = [d1] if d1.exists() else [_PROJECT_ROOT / "checkpoint"]
    else: search_dirs = [Path(d) for d in args.checkpoint_dirs]
    dummy = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    gc = device == "cuda"
    rows = []
    for name, kw in VARIANT_DEFS:
        print(f"\\n{'='*60}"); print(f"  {name}"); print(f"{'='*60}")
        model = build_profile_model(n_classes=args.n_classes, **kw).to(device).eval()
        pi = count_parameters(model)
        print(f"  Parameters:       {format_num(pi['total'])}")
        test_acc = bval = train_time = float("nan"); ckpt_path = None; ckpt_sz = float("nan")
        if not args.profile_only:
            cp, cm = find_checkpoint_with_metrics(name, search_dirs, args.noise_level)
            if cp is not None:
                try:
                    sd = torch.load(cp, map_location=device, weights_only=True)
                    model.load_state_dict(sd, strict=False); model.to(device).eval()
                    ckpt_sz = get_model_file_size(cp)
                    test_acc = float(cm.get("test_acc", float("nan")))
                    bval = float(cm.get("best_val_acc", float("nan")))
                    train_time = float(cm.get("training_time_sec", float("nan")))
                    print(f"  Checkpoint:       {cp.parent.name}")
                    print(f"  Test accuracy:    {test_acc:.2f}%")
                    print(f"  Best val accuracy:{bval:.2f}%")
                    print(f"  Training time:    {train_time:.1f} s")
                except Exception as e: print(f"  [WARN] {e}")
        if ckpt_path is None: print(f"  Checkpoint:       (random init)")
        if args.skip_flops:
            flops = float("nan"); print(f"  FLOPs:            (skipped)")
        else:
            try:
                flops = measure_flops(model, args.img_size)
                print(f"  FLOPs:            {format_num(flops)}")
            except Exception as e:
                flops = float("nan"); print(f"  FLOPs:            (error: {e})")
        if gc: torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        lat = measure_latency(model, dummy, args.num_warmup, args.num_iters, device)
        print(f"  Latency mean:     {lat['mean_ms']:.3f} ms")
        print(f"  Latency P50:      {lat['p50_ms']:.3f} ms")
        print(f"  Latency P95:      {lat['p95_ms']:.3f} ms")
        print(f"  Latency P99:      {lat['p99_ms']:.3f} ms")
        gpu_pk = float("nan")
        if gc:
            gi = measure_gpu_memory(model, dummy); gpu_pk = gi["peak_mb"]
            print(f"  GPU peak memory:  {gpu_pk:.1f} MB")
        else: print(f"  GPU memory:       N/A (CPU mode)")
        cmem = measure_cpu_memory()
        print(f"  CPU RSS memory:   {cmem['rss_mb']:.0f} MB" if not math.isnan(cmem['rss_mb']) else "  CPU RSS memory:   N/A")
        rows.append({"model": name, "noise_level": args.noise_level, "total_params": pi["total"], "trainable_params": pi["trainable"], "checkpoint_mb": ckpt_sz, "flops": flops, "flops_g": flops/1e9 if not math.isnan(flops) else float("nan"), "latency_mean_ms": lat["mean_ms"], "latency_std_ms": lat["std_ms"], "latency_p50_ms": lat["p50_ms"], "latency_p95_ms": lat["p95_ms"], "latency_p99_ms": lat["p99_ms"], "gpu_peak_mb": gpu_pk, "cpu_rss_mb": cmem["rss_mb"], "test_acc": test_acc, "best_val_acc": bval, "training_time_sec": train_time})
    print(f"\\n{'='*80}")
    print(f"  DEPLOYMENT COMPLEXITY SUMMARY (noise_level={args.noise_level})")
    print(f"{'='*80}")
    hdr = f"  {'Model':<18} {'Params':>8} {'FLOPs':>8} {'Latency':>8} {'GPU Mem':>8} {'Ckpt':>7} {'Test Acc':>8}"
    print("-"*len(hdr)); print(hdr); print("-"*len(hdr))
    for r in rows:
        ls = f"{r['latency_mean_ms']:.2f}ms" if not math.isnan(r['latency_mean_ms']) else "N/A"
        gs = f"{r['gpu_peak_mb']:.0f}MB" if not math.isnan(r['gpu_peak_mb']) else "N/A"
        cs = f"{r['checkpoint_mb']:.1f}MB" if not math.isnan(r['checkpoint_mb']) else "N/A"
        ac = f"{r['test_acc']:.2f}%" if not math.isnan(r['test_acc']) else "--"
        print(f"  {r['model']:<18} {format_num(r['total_params']):>8} {format_num(r['flops']):>8} {ls:>8} {gs:>8} {cs:>7} {ac:>8}")
    print("-"*len(hdr))
    if args.output:
        import csv; out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(rows[0].keys())).writeheader(); csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerows(rows)
        print(f"\\nCSV: {out}")
    if args.print_json: print(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\\nDone.")

if __name__ == "__main__": main()
