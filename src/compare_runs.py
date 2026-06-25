from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_history(run_dir: Path) -> dict:
    history_path = run_dir / "training_history.pkl"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training_history.pkl in {run_dir}")
    with history_path.open("rb") as f:
        return pickle.load(f)


def _safe_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return path.stat().st_size / (1024 * 1024)


def _format_millions(value: int) -> float:
    return value / 1_000_000


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "share_transformer_weights",
        "use_depth_embeddings",
        "ffn_variant",
        "total_params_m",
        "trainable_params_m",
        "best_val_acc",
        "test_acc",
        "test_loss",
        "training_time_sec",
        "checkpoint_mb",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_curves(histories: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    for run_name, history in histories.items():
        epochs = list(range(1, len(history["train_losses"]) + 1))
        ax_loss.plot(epochs, history["train_losses"], label=f"{run_name} train")
        ax_loss.plot(epochs, history["val_losses"], linestyle="--", label=f"{run_name} val")

        ax_acc.plot(epochs, history["train_accs"], label=f"{run_name} train")
        ax_acc.plot(epochs, history["val_accs"], linestyle="--", label=f"{run_name} val")

    ax_loss.set_title("Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)
    ax_loss.legend()

    ax_acc.set_title("Accuracy Curves")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True)
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_names = [r["run_name"] for r in rows]
    params_m = [r["total_params_m"] for r in rows]
    test_acc = [r["test_acc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x = list(range(len(run_names)))
    ax1.bar(x, params_m, alpha=0.6, label="Params (M)")
    ax2.plot(x, test_acc, marker="o", color="tab:red", label="Test Acc (%)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(run_names)
    ax1.set_ylabel("Params (Millions)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Capacity vs Performance")
    ax1.grid(True, axis="y", alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ViT runs (paper-style summary).")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoint", help="Root checkpoint directory.")
    parser.add_argument("--runs", nargs="+", default=["shared", "unshared"], help="Run subdirectories to compare.")
    parser.add_argument("--out-dir", type=str, default="checkpoint/compare", help="Where to write comparison outputs.")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    out_dir = Path(args.out_dir)

    rows: list[dict] = []
    histories: dict[str, dict] = {}

    for run_name in args.runs:
        run_dir = checkpoint_root / run_name
        metrics = _load_metrics(run_dir)
        history = _load_history(run_dir)
        ckpt_size_mb = _safe_size_mb(run_dir / "best_vit_model.pth")

        rows.append(
            {
                "run_name": metrics.get("run_name", run_name),
                "share_transformer_weights": metrics.get("share_transformer_weights"),
                "use_depth_embeddings": metrics.get("use_depth_embeddings"),
                "ffn_variant": metrics.get("ffn_variant"),
                "total_params_m": round(_format_millions(int(metrics["total_params"])), 3),
                "trainable_params_m": round(_format_millions(int(metrics["trainable_params"])), 3),
                "best_val_acc": round(float(metrics["best_val_acc"]), 3),
                "test_acc": round(float(metrics["test_acc"]), 3),
                "test_loss": round(float(metrics["test_loss"]), 6),
                "training_time_sec": round(float(metrics["training_time_sec"]), 3),
                "checkpoint_mb": round(float(ckpt_size_mb), 3),
            }
        )
        histories[metrics.get("run_name", run_name)] = history

    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(rows, out_dir / "summary.csv")
    plot_curves(histories, out_dir / "curves.png")
    plot_tradeoff(rows, out_dir / "capacity_vs_acc.png")

    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'curves.png'}")
    print(f"Wrote {out_dir / 'capacity_vs_acc.png'}")


if __name__ == "__main__":
    main()
