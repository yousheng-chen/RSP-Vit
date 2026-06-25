from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

from config import PROJECT_ROOT, config
from training import train_vit


MODE_SPECS = (
    ("share_plain", False),
    ("share_depth", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat shared ViT vs shared ViT + depth embeddings on one dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "noisy", "-10db"),
        help="Dataset directory to train on. Default: data/noisy/-10db",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=config["checkpoint_dir"],
        help="Checkpoint root directory.",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=None,
        help="Output folder name under checkpoint root.",
    )
    parser.add_argument("--repeats", type=int, default=2, help="How many repeats per mode.")
    parser.add_argument("--num-epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 1e-5))
    parser.add_argument("--img-size", type=int, nargs=2, default=config.get("img_size", [192, 192]))
    parser.add_argument("--train-split", type=float, default=config.get("train_split", 0.5))
    parser.add_argument("--val-split", type=float, default=config.get("val_split", 0.25))
    parser.add_argument(
        "--seed",
        type=int,
        default=config.get("split_seed", 42),
        help="Base seed. By default every repeat uses the same seed for fair A/B comparison.",
    )
    parser.add_argument(
        "--vary-seed-by-repeat",
        action="store_true",
        help="Use seed, seed+1, seed+2... across repeats instead of the same seed every time.",
    )
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.parent / f"{path.name}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = path.parent / f"{path.name}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def write_csv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_group_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    summary_rows: list[dict] = []
    for mode, mode_rows in grouped.items():
        test_accs = [float(row["test_acc"]) for row in mode_rows]
        best_val_accs = [float(row["best_val_acc"]) for row in mode_rows]
        times = [float(row["training_time_sec"]) for row in mode_rows]

        def _std(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) >= 2 else 0.0

        summary_rows.append(
            {
                "mode": mode,
                "runs": len(mode_rows),
                "mean_test_acc": round(statistics.mean(test_accs), 4),
                "std_test_acc": round(_std(test_accs), 4),
                "mean_best_val_acc": round(statistics.mean(best_val_accs), 4),
                "std_best_val_acc": round(_std(best_val_accs), 4),
                "mean_training_time_sec": round(statistics.mean(times), 4),
                "std_training_time_sec": round(_std(times), 4),
            }
        )
    return sorted(summary_rows, key=lambda row: row["mode"])


def plot_test_acc(rows: list[dict], out_path: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["mode"], []).append(row)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = {"share_plain": "#1f4e79", "share_depth": "#c75b12"}
    markers = {"share_plain": "o", "share_depth": "s"}
    labels = {"share_plain": "Share", "share_depth": "Share + Depth"}

    for mode, mode_rows in sorted(grouped.items()):
        mode_rows = sorted(mode_rows, key=lambda row: row["repeat_index"])
        xs = [int(row["repeat_index"]) for row in mode_rows]
        ys = [float(row["test_acc"]) for row in mode_rows]
        ax.plot(
            xs,
            ys,
            marker=markers.get(mode, "o"),
            color=colors.get(mode, "#333333"),
            linewidth=2.2,
            markersize=7,
            label=labels.get(mode, mode),
        )
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.1, f"{y:.2f}", fontsize=8, ha="center", color=colors.get(mode, "#333333"))

    ax.set_title("Test Accuracy by Repeat")
    ax.set_xlabel("Repeat")
    ax.set_ylabel("Test Accuracy (%)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xticks(sorted({int(row["repeat_index"]) for row in rows}))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_bar(summary_rows: list[dict], out_path: Path) -> None:
    labels = ["Share", "Share + Depth"]
    values = []
    errors = []
    lookup = {row["mode"]: row for row in summary_rows}
    for mode in ("share_plain", "share_depth"):
        row = lookup.get(mode, {})
        values.append(float(row.get("mean_test_acc", 0.0)))
        errors.append(float(row.get("std_test_acc", 0.0)))

    fig, ax = plt.subplots(figsize=(6.6, 5))
    bars = ax.bar(labels, values, yerr=errors, capsize=6, color=["#1f4e79", "#c75b12"], alpha=0.9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Mean Test Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    benchmark_name = args.benchmark_name or f"share_depth_repeat_{time.strftime('%Y%m%d_%H%M%S')}"
    benchmark_dir = ensure_unique_dir(checkpoint_root / benchmark_name)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    original_checkpoint_dir = config["checkpoint_dir"]
    config["checkpoint_dir"] = str(benchmark_dir)

    rows: list[dict] = []
    try:
        print(f"Benchmark directory: {benchmark_dir.resolve()}")
        print(f"Dataset: {Path(args.data_dir).resolve()}")
        print(f"Repeats per mode: {args.repeats}")
        print(f"Base seed: {args.seed}")
        print(f"Seed strategy: {'vary by repeat' if args.vary_seed_by_repeat else 'fixed same seed'}")

        total_runs = args.repeats * len(MODE_SPECS)
        current_run = 0

        for repeat_index in range(1, args.repeats + 1):
            run_seed = args.seed + repeat_index - 1 if args.vary_seed_by_repeat else args.seed
            for mode, use_depth_embeddings in MODE_SPECS:
                current_run += 1
                run_name = f"repeat{repeat_index:02d}_{mode}_seed{run_seed}"
                print("=" * 88)
                print(
                    f"[{current_run}/{total_runs}] Running {run_name} "
                    f"(share_transformer_weights=True, use_depth_embeddings={use_depth_embeddings})"
                )

                train_vit(
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    img_size=list(args.img_size),
                    train_split=args.train_split,
                    val_split=args.val_split,
                    seed=run_seed,
                    share_transformer_weights=True,
                    use_depth_embeddings=use_depth_embeddings,
                    run_name=run_name,
                    data_dir=args.data_dir,
                    show_plots=args.show_plots,
                )

                run_dir = benchmark_dir / run_name
                metrics = load_metrics(run_dir)
                rows.append(
                    {
                        "repeat_index": repeat_index,
                        "mode": mode,
                        "seed": run_seed,
                        "share_transformer_weights": True,
                        "use_depth_embeddings": use_depth_embeddings,
                        "best_val_acc": metrics.get("best_val_acc"),
                        "test_acc": metrics.get("test_acc"),
                        "test_loss": metrics.get("test_loss"),
                        "training_time_sec": metrics.get("training_time_sec"),
                        "total_params": metrics.get("total_params"),
                        "trainable_params": metrics.get("trainable_params"),
                        "run_name": metrics.get("run_name", run_name),
                        "run_dir": str(run_dir.resolve()),
                        "data_dir": metrics.get("data_dir", args.data_dir),
                    }
                )

    finally:
        config["checkpoint_dir"] = original_checkpoint_dir

    summary_rows = build_group_summary(rows)

    detail_fields = [
        "repeat_index",
        "mode",
        "seed",
        "share_transformer_weights",
        "use_depth_embeddings",
        "best_val_acc",
        "test_acc",
        "test_loss",
        "training_time_sec",
        "total_params",
        "trainable_params",
        "run_name",
        "run_dir",
        "data_dir",
    ]
    summary_fields = [
        "mode",
        "runs",
        "mean_test_acc",
        "std_test_acc",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_training_time_sec",
        "std_training_time_sec",
    ]

    write_csv(rows, benchmark_dir / "summary_detail.csv", detail_fields)
    write_csv(summary_rows, benchmark_dir / "summary_by_mode.csv", summary_fields)
    plot_test_acc(rows, benchmark_dir / "test_acc_by_repeat.png")
    plot_mean_bar(summary_rows, benchmark_dir / "mean_test_acc.png")

    print(f"Wrote {benchmark_dir / 'summary_detail.csv'}")
    print(f"Wrote {benchmark_dir / 'summary_by_mode.csv'}")
    print(f"Wrote {benchmark_dir / 'test_acc_by_repeat.png'}")
    print(f"Wrote {benchmark_dir / 'mean_test_acc.png'}")


if __name__ == "__main__":
    main()
