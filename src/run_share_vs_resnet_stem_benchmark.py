from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PROJECT_ROOT, config
from training import train_vit
from run_vit_noise_benchmark import (
    build_final_dir_name,
    build_temp_run_name,
    discover_noise_dirs,
    ensure_unique_path,
    extract_noise_db,
    load_existing_rows,
    load_metrics,
    save_metrics,
    write_csv,
    write_excel_xml,
    write_markdown_table,
    write_plan,
    write_progress,
)


MODE_ORDER = ("share", "share_resnet_stem")
MODE_LABELS = {
    "share": "Shared ViT",
    "share_resnet_stem": "Shared ViT + ResNet Stem",
}
MODE_COLORS = {
    "share": "#1f4e79",
    "share_resnet_stem": "#c75b12",
}
MODE_MARKERS = {
    "share": "o",
    "share_resnet_stem": "s",
}
MODE_TO_PATCH_EMBED = {
    "share": "standard",
    "share_resnet_stem": "resnet_stem",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared ViT vs shared ViT + ResNet-style stem across selected noisy datasets."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(PROJECT_ROOT) / "data" / "noisy"),
        help="Root directory containing noise subfolders such as -10db, 0db, 10db.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=config["checkpoint_dir"],
        help="Root checkpoint directory. A benchmark subfolder will be created inside it.",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=None,
        help="Optional benchmark folder name. Default: share_vs_resnet_stem_<timestamp>.",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="*",
        default=["-10db", "0db", "10db"],
        help="Subset of noise folder names to run. Default: -10db 0db 10db",
    )
    parser.add_argument("--num-epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 1e-5))
    parser.add_argument("--img-size", type=int, nargs=2, default=config.get("img_size", [192, 192]))
    parser.add_argument("--train-split", type=float, default=config.get("train_split", 0.5))
    parser.add_argument("--val-split", type=float, default=config.get("val_split", 0.25))
    parser.add_argument("--seed", type=int, default=config.get("seed", config.get("split_seed", 42)))
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def filter_noise_dirs(noise_dirs: list[Path], requested_levels: list[str]) -> list[Path]:
    requested = {item.lower() for item in requested_levels}
    filtered = [path for path in noise_dirs if path.name.lower() in requested]
    if not filtered:
        raise FileNotFoundError(
            f"None of the requested noise folders were found: {requested_levels}. "
            f"Available: {[path.name for path in noise_dirs]}"
        )
    return filtered


def build_wide_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["noise_db"], {})[row["mode"]] = row

    wide_rows: list[dict] = []
    for noise_db in sorted(grouped.keys()):
        share_row = grouped[noise_db].get("share")
        stem_row = grouped[noise_db].get("share_resnet_stem")
        wide_rows.append(
            {
                "noise_db": noise_db,
                "share_test_acc": share_row.get("test_acc") if share_row else "",
                "share_resnet_stem_test_acc": stem_row.get("test_acc") if stem_row else "",
                "share_best_val_acc": share_row.get("best_val_acc") if share_row else "",
                "share_resnet_stem_best_val_acc": stem_row.get("best_val_acc") if stem_row else "",
                "share_total_params": share_row.get("total_params") if share_row else "",
                "share_resnet_stem_total_params": stem_row.get("total_params") if stem_row else "",
                "share_training_time_sec": share_row.get("training_time_sec") if share_row else "",
                "share_resnet_stem_training_time_sec": stem_row.get("training_time_sec") if stem_row else "",
                "share_run_dir": share_row.get("run_dir_name") if share_row else "",
                "share_resnet_stem_run_dir": stem_row.get("run_dir_name") if stem_row else "",
            }
        )
    return wide_rows


def _mode_rows(rows: list[dict], mode: str) -> list[dict]:
    return sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["noise_db"])


def _plot_lines(rows: list[dict], value_key: str, out_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for mode in MODE_ORDER:
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue
        xs = [row["noise_db"] for row in mode_rows]
        ys = [row[value_key] for row in mode_rows]
        ax.plot(
            xs,
            ys,
            linewidth=2.4,
            markersize=7,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
        if value_key == "test_acc":
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.18, f"{y:.2f}", fontsize=8, ha="center", color=MODE_COLORS[mode])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Noise level (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_param_compare(rows: list[dict], out_path: Path) -> None:
    mode_to_params: dict[str, int] = {}
    for row in rows:
        mode_to_params.setdefault(row["mode"], row["total_params"])

    modes = [mode for mode in MODE_ORDER if mode in mode_to_params]
    labels = [MODE_LABELS[mode] for mode in modes]
    values = [mode_to_params[mode] / 1_000_000 for mode in modes]
    colors = [MODE_COLORS[mode] for mode in modes]

    fig, ax = plt.subplots(figsize=(6.8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.88, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}M", ha="center", va="bottom", fontsize=9)
    ax.set_title("Parameter Count Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parameters (Millions)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_param_accuracy_tradeoff(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for row in rows:
        mode = row["mode"]
        x = row["total_params"] / 1_000_000
        y = row["test_acc"]
        ax.scatter(x, y, s=75, color=MODE_COLORS[mode], alpha=0.82)
        ax.text(x, y + 0.18, f"{mode} {row['noise_label']}", fontsize=8, ha="center")
    ax.set_title("Parameter-Accuracy Tradeoff", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total parameters (Millions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paper_overview(rows: list[dict], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.24)
    ax_main = fig.add_subplot(gs[0, :])
    ax_param = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    for mode in MODE_ORDER:
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue
        xs = [row["noise_db"] for row in mode_rows]
        ys = [row["test_acc"] for row in mode_rows]
        ax_main.plot(
            xs,
            ys,
            linewidth=2.5,
            markersize=7,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )

    ax_main.set_title("(a) Accuracy Under Noise", fontsize=12, fontweight="bold")
    ax_main.set_xlabel("Noise level (dB)")
    ax_main.set_ylabel("Test accuracy (%)")
    ax_main.grid(True, alpha=0.25, linestyle="--")
    ax_main.legend(frameon=False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    labels = []
    params_m = []
    avg_times = []
    colors = []
    for mode in MODE_ORDER:
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue
        labels.append(MODE_LABELS[mode])
        params_m.append(mode_rows[0]["total_params"] / 1_000_000)
        avg_times.append(float(np.mean([row["training_time_sec"] for row in mode_rows])))
        colors.append(MODE_COLORS[mode])

    bars = ax_param.bar(labels, params_m, color=colors, alpha=0.88, width=0.6)
    for bar, value in zip(bars, params_m):
        ax_param.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}M", ha="center", va="bottom", fontsize=9)
    ax_param.set_title("(b) Parameter Count", fontsize=12, fontweight="bold")
    ax_param.set_ylabel("Parameters (Millions)")
    ax_param.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_param.spines["top"].set_visible(False)
    ax_param.spines["right"].set_visible(False)

    bars = ax_time.bar(labels, avg_times, color=colors, alpha=0.88, width=0.6)
    for bar, value in zip(bars, avg_times):
        ax_time.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax_time.set_title("(c) Average Training Time", fontsize=12, fontweight="bold")
    ax_time.set_ylabel("Time (sec)")
    ax_time.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)

    fig.suptitle("Shared ViT vs Shared ViT + ResNet Stem", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def generate_summary_outputs(rows: list[dict], benchmark_dir: Path) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda row: (row["noise_db"], row["mode"]))
    summary_fields = [
        "run_index",
        "mode",
        "noise_label",
        "noise_db",
        "patch_embed_variant",
        "test_acc",
        "best_val_acc",
        "test_loss",
        "total_params",
        "trainable_params",
        "training_time_sec",
        "data_dir",
        "run_dir_name",
        "run_dir",
    ]
    write_csv(rows, benchmark_dir / "summary.csv", summary_fields)
    write_excel_xml(rows, benchmark_dir / "summary_excel.xml", summary_fields)

    wide_rows = build_wide_rows(rows)
    wide_fields = [
        "noise_db",
        "share_test_acc",
        "share_resnet_stem_test_acc",
        "share_best_val_acc",
        "share_resnet_stem_best_val_acc",
        "share_total_params",
        "share_resnet_stem_total_params",
        "share_training_time_sec",
        "share_resnet_stem_training_time_sec",
        "share_run_dir",
        "share_resnet_stem_run_dir",
    ]
    write_csv(wide_rows, benchmark_dir / "summary_wide.csv", wide_fields)
    write_excel_xml(wide_rows, benchmark_dir / "summary_wide_excel.xml", wide_fields)
    write_markdown_table(rows, benchmark_dir / "summary.md")

    _plot_lines(rows, "test_acc", benchmark_dir / "accuracy_vs_noise.png", "Test Accuracy vs Noise Level", "Test accuracy (%)")
    _plot_lines(rows, "training_time_sec", benchmark_dir / "training_time_vs_noise.png", "Training Time vs Noise Level", "Training time (sec)")
    plot_param_compare(rows, benchmark_dir / "parameter_compare.png")
    plot_param_accuracy_tradeoff(rows, benchmark_dir / "parameter_accuracy_tradeoff.png")
    _plot_lines(rows, "test_acc", benchmark_dir / "paper_robustness_main.png", "Robustness Comparison Under Different Noise Levels", "Test accuracy (%)")
    plot_paper_overview(rows, benchmark_dir / "paper_overview.png")


def run_single_experiment(
    benchmark_name: str,
    benchmark_dir: Path,
    checkpoint_root: Path,
    data_dir: Path,
    mode: str,
    run_index: int,
    args: argparse.Namespace,
) -> dict:
    noise_label = data_dir.name
    temp_run_name = build_temp_run_name(benchmark_name, mode, noise_label, run_index)
    temp_run_dir = checkpoint_root / temp_run_name

    if temp_run_dir.exists():
        shutil.rmtree(temp_run_dir)

    patch_embed_variant = MODE_TO_PATCH_EMBED[mode]
    print("=" * 88)
    print(
        f"Starting run {run_index:02d}: mode={mode}, dataset={data_dir}, "
        f"patch_embed_variant={patch_embed_variant}"
    )

    train_vit(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        img_size=list(args.img_size),
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        share_transformer_weights=True,
        use_depth_embeddings=False,
        ffn_variant="standard",
        patch_embed_variant=patch_embed_variant,
        run_name=temp_run_name,
        data_dir=str(data_dir),
        show_plots=args.show_plots,
    )

    metrics_path = temp_run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training finished but metrics.json is missing: {metrics_path}")

    metrics = load_metrics(metrics_path)
    final_accuracy = float(metrics.get("test_acc", metrics.get("best_val_acc", 0.0)))
    final_dir_name = build_final_dir_name(mode, noise_label, run_index, final_accuracy)
    final_run_dir = ensure_unique_path(benchmark_dir / final_dir_name)
    shutil.move(str(temp_run_dir), str(final_run_dir))

    final_metrics_path = final_run_dir / "metrics.json"
    metrics = load_metrics(final_metrics_path)
    metrics["benchmark_name"] = benchmark_name
    metrics["mode"] = mode
    metrics["noise_label"] = noise_label
    metrics["noise_db"] = extract_noise_db(noise_label)
    metrics["run_index"] = run_index
    metrics["run_name"] = final_dir_name
    metrics["run_dir"] = str(final_run_dir.resolve())
    save_metrics(final_metrics_path, metrics)

    row = {
        "run_index": run_index,
        "mode": mode,
        "noise_label": noise_label,
        "noise_db": float(metrics["noise_db"]),
        "patch_embed_variant": metrics.get("patch_embed_variant", patch_embed_variant),
        "test_acc": float(metrics.get("test_acc", 0.0)),
        "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
        "test_loss": float(metrics.get("test_loss", 0.0)),
        "total_params": int(metrics.get("total_params", 0)),
        "trainable_params": int(metrics.get("trainable_params", 0)),
        "training_time_sec": float(metrics.get("training_time_sec", 0.0)),
        "data_dir": metrics.get("data_dir", str(data_dir)),
        "run_dir_name": final_run_dir.name,
        "run_dir": str(final_run_dir.resolve()),
    }

    print(f"Completed run {run_index:02d}: {final_run_dir.name}")
    return row


def build_run_plan(noise_dirs: list[Path]) -> list[dict]:
    plan: list[dict] = []
    run_index = 1
    for noise_dir in noise_dirs:
        for mode in MODE_ORDER:
            plan.append(
                {
                    "run_index": run_index,
                    "mode": mode,
                    "noise_label": noise_dir.name,
                    "data_dir": noise_dir,
                }
            )
            run_index += 1
    return plan


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(checkpoint_root)

    benchmark_name = args.benchmark_name or f"share_vs_resnet_stem_{time.strftime('%Y%m%d_%H%M%S')}"
    requested_benchmark_dir = checkpoint_root / benchmark_name
    if requested_benchmark_dir.exists():
        benchmark_dir = requested_benchmark_dir if (args.resume or args.benchmark_name is not None) else ensure_unique_path(requested_benchmark_dir)
    else:
        benchmark_dir = requested_benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    noise_dirs = filter_noise_dirs(discover_noise_dirs(data_root), args.noise_levels)
    run_plan = build_run_plan(noise_dirs)
    total_runs = len(run_plan)
    write_plan(benchmark_dir, run_plan)

    print(f"Benchmark directory: {benchmark_dir.resolve()}")
    print("Noise datasets:")
    for noise_dir in noise_dirs:
        print(f"  - {noise_dir.name}")

    rows = load_existing_rows(benchmark_dir)
    failures: list[dict] = []
    completed_keys = {(row["run_index"], row["mode"], row["noise_label"]) for row in rows}
    write_progress(benchmark_dir, rows, failures, total_runs)

    if rows:
        print(f"Resuming benchmark: already completed {len(rows)}/{total_runs} runs.")

    try:
        for item in run_plan:
            run_index = item["run_index"]
            mode = item["mode"]
            noise_label = item["noise_label"]
            data_dir = item["data_dir"]
            progress_prefix = f"[{run_index}/{total_runs}]"

            if (run_index, mode, noise_label) in completed_keys:
                print(f"{progress_prefix} Skip existing run: mode={mode}, noise={noise_label}")
                continue

            try:
                print(f"{progress_prefix} Running: mode={mode}, noise={noise_label}")
                row = run_single_experiment(
                    benchmark_name=benchmark_name,
                    benchmark_dir=benchmark_dir,
                    checkpoint_root=checkpoint_root,
                    data_dir=data_dir,
                    mode=mode,
                    run_index=run_index,
                    args=args,
                )
                rows.append(row)
                completed_keys.add((row["run_index"], row["mode"], row["noise_label"]))
                generate_summary_outputs(rows, benchmark_dir)
                write_progress(benchmark_dir, rows, failures, total_runs)
            except Exception as exc:
                failure = {
                    "run_index": run_index,
                    "mode": mode,
                    "noise_label": noise_label,
                    "error": repr(exc),
                }
                failures.append(failure)
                print(f"Run failed: {failure}")
                write_progress(benchmark_dir, rows, failures, total_runs)
                if not args.continue_on_error:
                    generate_summary_outputs(rows, benchmark_dir)
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Partial results have been kept.")
        generate_summary_outputs(rows, benchmark_dir)
        write_progress(benchmark_dir, rows, failures, total_runs)
        raise

    generate_summary_outputs(rows, benchmark_dir)
    write_progress(benchmark_dir, rows, failures, total_runs)

    if failures:
        failure_path = benchmark_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote failures to: {failure_path}")

    print("=" * 88)
    print("Benchmark complete.")
    print(f"Summary CSV: {benchmark_dir / 'summary.csv'}")
    print(f"Excel XML: {benchmark_dir / 'summary_excel.xml'}")
    print(f"Summary wide CSV: {benchmark_dir / 'summary_wide.csv'}")
    print(f"Accuracy plot: {benchmark_dir / 'accuracy_vs_noise.png'}")
    print(f"Paper main figure: {benchmark_dir / 'paper_robustness_main.png'}")
    print(f"Paper overview figure: {benchmark_dir / 'paper_overview.png'}")
    print(f"Benchmark folder: {benchmark_dir.resolve()}")


if __name__ == "__main__":
    main()
