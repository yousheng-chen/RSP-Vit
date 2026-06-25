import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import to_rgba

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette (same as paper)
COLORS = {
    "Vanilla ViT":   "#6b7280",
    "Shared ViT":    "#1f4e79",
    "Stem-only ViT": "#c75b12",
    "RSP-ViT":       "#1a7f5a",
}
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]

# ---------- data loading ----------
def load_single_results():
    path = SCRIPT_DIR / "benchmark_results.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["total_params"] = int(r["total_params"])
            r["flops"] = float(r["flops"])
            r["latency_mean_ms"] = float(r["latency_mean_ms"])
            r["checkpoint_mb"] = float(r["checkpoint_mb"])
            rows.append(r)
    return rows

def load_batch_results():
    path = SCRIPT_DIR / "batch_sweep_results.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["batch_size"] = int(r["batch_size"])
            r["mean_ms"] = float(r["mean_ms"])
            r["throughput"] = float(r["throughput"])
            r["total_params"] = int(r["total_params"])
            rows.append(r)
    return rows

SINGLE = load_single_results()
BATCH = load_batch_results()

# ============================================================
# 1. Params + Checkpoint Size (grouped bar, dual axis)
# ============================================================
def plot_params_and_ckpt():
    models = [r["model"] for r in SINGLE]
    params = [r["total_params"] for r in SINGLE]
    ckpts = [r["checkpoint_mb"] for r in SINGLE]
    colors = [COLORS[m] for m in models]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    w = 0.35

    bars1 = ax1.bar(x - w/2, [p/1e6 for p in params], w,
                    color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Parameters (M)")
    ax1.set_ylim(0, max(p/1e6 for p in params) * 1.35)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, ckpts, w,
                    color=colors, edgecolor="white", linewidth=0.5,
                    alpha=0.65)
    ax2.set_ylabel("Checkpoint Size (MB)")
    ax2.set_ylim(0, max(ckpts) * 1.35)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_title("Model Size Comparison", fontweight="bold")

    for bar, val in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val/1e6:.2f}M", ha="center", va="bottom", fontsize=8, color="#333")
    for bar, val in zip(bars2, ckpts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.1f}MB", ha="center", va="bottom", fontsize=8, color="#333")

    lines1 = [plt.Rectangle((0,0),1,1, color="gray", ec="white")]
    lines2 = [plt.Rectangle((0,0),1,1, color="gray", alpha=0.65, ec="white")]
    ax1.legend(lines1 + lines2, ["Parameters (M)", "Checkpoint Size (MB)"],
              loc="upper left", framealpha=0.9)

    plt.tight_layout()
    path = OUT_DIR / "params_and_ckpt.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 2. FLOPs bar
# ============================================================
def plot_flops():
    models = [r["model"] for r in SINGLE]
    flops = [r["flops"] for r in SINGLE]
    colors = [COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    bars = ax.bar(x, [f/1e9 for f in flops], 0.5,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("FLOPs (G)")
    ax.set_title("Computational Cost per Forward Pass", fontweight="bold")
    ax.set_ylim(0, max(f/1e9 for f in flops) * 1.25)

    for bar, val in zip(bars, flops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val/1e9:.2f}G", ha="center", va="bottom", fontsize=9, color="#333")

    plt.tight_layout()
    path = OUT_DIR / "flops.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 3. Latency vs Batch Size
# ============================================================
def plot_latency_vs_batch():
    fig, ax = plt.subplots(figsize=(8, 5))
    models_order = ["Vanilla ViT", "Shared ViT", "Stem-only ViT", "RSP-ViT"]
    batch_sizes = sorted(set(r["batch_size"] for r in BATCH))

    for idx, model in enumerate(models_order):
        rows = sorted([r for r in BATCH if r["model"] == model], key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in rows]
        lat = [r["mean_ms"] for r in rows]
        ax.plot(bs, lat, marker=MARKERS[idx], linestyle=LINESTYLES[idx],
                color=COLORS[model], label=model, linewidth=1.8, markersize=7)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency vs. Batch Size", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(framealpha=0.9, loc="upper left")

    plt.tight_layout()
    path = OUT_DIR / "latency_vs_batch.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 4. Throughput vs Batch Size
# ============================================================
def plot_throughput_vs_batch():
    fig, ax = plt.subplots(figsize=(8, 5))
    models_order = ["Vanilla ViT", "Shared ViT", "Stem-only ViT", "RSP-ViT"]
    batch_sizes = sorted(set(r["batch_size"] for r in BATCH))

    for idx, model in enumerate(models_order):
        rows = sorted([r for r in BATCH if r["model"] == model], key=lambda r: r["batch_size"])
        bs = [r["batch_size"] for r in rows]
        th = [r["throughput"] for r in rows]
        ax.plot(bs, th, marker=MARKERS[idx], linestyle=LINESTYLES[idx],
                color=COLORS[model], label=model, linewidth=1.8, markersize=7)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title("Inference Throughput vs. Batch Size", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(framealpha=0.9, loc="lower right")

    plt.tight_layout()
    path = OUT_DIR / "throughput_vs_batch.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 5. Summary table as figure
# ============================================================
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("tight")
    ax.axis("off")

    models = [r["model"] for r in SINGLE]
    params_m = [f"{r['total_params']/1e6:.2f}M" for r in SINGLE]
    flops_g = [f"{r['flops']/1e9:.2f}G" for r in SINGLE]
    lat = [f"{r['latency_mean_ms']:.2f}ms" for r in SINGLE]
    ckpt = [f"{r['checkpoint_mb']:.1f}MB" for r in SINGLE]

    col_labels = ["Model", "Params", "FLOPs", "Latency (bs=1)", "Ckpt Size"]
    cell_text = [[m, p, f, l, c] for m, p, f, l, c in zip(models, params_m, flops_g, lat, ckpt)]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for i, model in enumerate(models):
        color = COLORS[model]
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(to_rgba(color, alpha=0.12))

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary of Inference Benchmarks", fontweight="bold", pad=20)
    plt.tight_layout()
    path = OUT_DIR / "summary_table.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 6. Latency bar (bs=1)
# ============================================================
def plot_latency_bs1():
    fig, ax = plt.subplots(figsize=(7, 4))
    models = [r["model"] for r in SINGLE]
    lat = [r["latency_mean_ms"] for r in SINGLE]
    colors = [COLORS[m] for m in models]

    x = np.arange(len(models))
    bars = ax.bar(x, lat, 0.5, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}ms", ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency (Batch Size = 1)", fontweight="bold")
    ax.set_ylim(0, max(lat) * 1.25)

    plt.tight_layout()
    path = OUT_DIR / "latency_bs1.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# 7. Combined overview: 3 subplots
# ============================================================
def plot_combined_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    models = [r["model"] for r in SINGLE]
    params = [r["total_params"] for r in SINGLE]
    flops = [r["flops"] for r in SINGLE]
    lat = [r["latency_mean_ms"] for r in SINGLE]
    colors = [COLORS[m] for m in models]
    x = np.arange(len(models))
    w = 0.55

    # params
    ax = axes[0]
    bars = ax.bar(x, [p/1e6 for p in params], w, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size", fontweight="bold")
    ax.set_ylim(0, max(p/1e6 for p in params) * 1.3)
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val/1e6:.2f}M", ha="center", va="bottom", fontsize=7)

    # FLOPs
    ax = axes[1]
    bars = ax.bar(x, [f/1e9 for f in flops], w, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("FLOPs (G)")
    ax.set_title("Computational Cost", fontweight="bold")
    ax.set_ylim(0, max(f/1e9 for f in flops) * 1.3)
    for bar, val in zip(bars, flops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val/1e9:.2f}G", ha="center", va="bottom", fontsize=7)

    # latency
    ax = axes[2]
    bars = ax.bar(x, lat, w, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency (bs=1)", fontweight="bold")
    ax.set_ylim(0, max(lat) * 1.3)
    for bar, val in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}ms", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = OUT_DIR / "combined_overview.png"
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    plot_params_and_ckpt()
    plot_flops()
    plot_latency_vs_batch()
    plot_throughput_vs_batch()
    plot_summary_table()
    plot_latency_bs1()
    plot_combined_overview()
    print(f"\nAll figures saved to {OUT_DIR}")
