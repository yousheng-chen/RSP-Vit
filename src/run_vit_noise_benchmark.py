from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PROJECT_ROOT, config
from training import train_vit


NOISE_DIR_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*db", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared vs unshared ViT across all noisy datasets and generate summary charts."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "noisy"),
        help="Root directory that contains noise subfolders such as -10db, -5db, 0db, 5db, 10db.",
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
        help="Optional benchmark folder name. Default: vit_noise_benchmark_<timestamp>.",
    )
    parser.add_argument("--num-epochs", type=int, default=None, help="Override config['num_epochs'].")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config['batch_size'].")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override config['learning_rate'].")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show matplotlib windows during training. Default is save-only.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one run fails, continue the remaining runs and still write a partial summary.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing benchmark folder when --benchmark-name already exists.",
    )
    return parser.parse_args()


def extract_noise_db(folder_name: str) -> float:
    match = NOISE_DIR_PATTERN.search(folder_name)
    if not match:
        raise ValueError(f"Cannot parse noise level from folder name: {folder_name}")
    return float(match.group(1))


def discover_noise_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    noise_dirs = [path for path in data_root.iterdir() if path.is_dir()]
    if not noise_dirs:
        raise FileNotFoundError(f"No dataset folders found under: {data_root}")

    return sorted(noise_dirs, key=lambda path: extract_noise_db(path.name))


def build_temp_run_name(benchmark_name: str, mode_tag: str, noise_label: str, run_index: int) -> str:
    safe_noise = noise_label.replace(" ", "")
    return f"tmp_{benchmark_name}_{mode_tag}_{safe_noise}_run{run_index:02d}"


def build_final_dir_name(mode_tag: str, noise_label: str, run_index: int, accuracy: float) -> str:
    return f"{mode_tag}-{noise_label}-{run_index:02d}\u8f6e-{accuracy:.2f}\u7cbe\u5ea6"


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics_path: Path, metrics: dict) -> None:
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def format_mode(share_transformer_weights: bool) -> str:
    return "share" if share_transformer_weights else "unshare"


def write_csv(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown_table(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# ViT Noise Benchmark Summary",
        "",
        "| mode | noise_db | test_acc | best_val_acc | total_params | trainable_params | training_time_sec | run_dir |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {noise_db:.1f} | {test_acc:.2f} | {best_val_acc:.2f} | {total_params} | {trainable_params} | "
            "{training_time_sec:.2f} | {run_dir_name} |".format(**row)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_excel_xml(rows: list[dict], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:o="urn:schemas-microsoft-com:office:office"',
        ' xmlns:x="urn:schemas-microsoft-com:office:excel"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
        ' <Worksheet ss:Name="Summary">',
        '  <Table>',
        "   <Row>",
    ]
    for field in fieldnames:
        lines.append(f'    <Cell><Data ss:Type="String">{escape(str(field))}</Data></Cell>')
    lines.append("   </Row>")

    for row in rows:
        lines.append("   <Row>")
        for field in fieldnames:
            value = row.get(field, "")
            cell_type = "Number" if isinstance(value, (int, float)) and value != "" else "String"
            lines.append(f'    <Cell><Data ss:Type="{cell_type}">{escape(str(value))}</Data></Cell>')
        lines.append("   </Row>")

    lines.extend(
        [
            "  </Table>",
            " </Worksheet>",
            "</Workbook>",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_wide_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["noise_db"], {})[row["mode"]] = row

    wide_rows: list[dict] = []
    for noise_db in sorted(grouped.keys()):
        share_row = grouped[noise_db].get("share")
        unshare_row = grouped[noise_db].get("unshare")
        wide_rows.append(
            {
                "noise_db": noise_db,
                "share_test_acc": share_row.get("test_acc") if share_row else "",
                "unshare_test_acc": unshare_row.get("test_acc") if unshare_row else "",
                "share_best_val_acc": share_row.get("best_val_acc") if share_row else "",
                "unshare_best_val_acc": unshare_row.get("best_val_acc") if unshare_row else "",
                "share_total_params": share_row.get("total_params") if share_row else "",
                "unshare_total_params": unshare_row.get("total_params") if unshare_row else "",
                "share_training_time_sec": share_row.get("training_time_sec") if share_row else "",
                "unshare_training_time_sec": unshare_row.get("training_time_sec") if unshare_row else "",
                "share_run_dir": share_row.get("run_dir_name") if share_row else "",
                "unshare_run_dir": unshare_row.get("run_dir_name") if unshare_row else "",
            }
        )
    return wide_rows


def plot_accuracy_vs_noise(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, color in [("share", "tab:blue"), ("unshare", "tab:orange")]:
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["noise_db"])
        if not mode_rows:
            continue
        xs = [row["noise_db"] for row in mode_rows]
        ys = [row["test_acc"] for row in mode_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=mode)
        for row in mode_rows:
            ax.text(row["noise_db"], row["test_acc"] + 0.15, f"{row['test_acc']:.2f}", fontsize=8, ha="center")

    ax.set_title("Test Accuracy vs Noise Level")
    ax.set_xlabel("Noise level (dB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_time_vs_noise(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, color in [("share", "tab:green"), ("unshare", "tab:red")]:
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["noise_db"])
        if not mode_rows:
            continue
        xs = [row["noise_db"] for row in mode_rows]
        ys = [row["training_time_sec"] for row in mode_rows]
        ax.plot(xs, ys, marker="s", linewidth=2.0, color=color, label=mode)

    ax.set_title("Training Time vs Noise Level")
    ax.set_xlabel("Noise level (dB)")
    ax.set_ylabel("Training time (sec)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_param_compare(rows: list[dict], out_path: Path) -> None:
    mode_to_params: dict[str, int] = {}
    for row in rows:
        mode_to_params.setdefault(row["mode"], row["total_params"])

    modes = ["share", "unshare"]
    values = [mode_to_params.get(mode, 0) / 1_000_000 for mode in modes]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(modes, values, color=["tab:blue", "tab:orange"], alpha=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}M", ha="center", va="bottom", fontsize=9)

    ax.set_title("Parameter Count Comparison")
    ax.set_ylabel("Total parameters (Millions)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_param_accuracy_tradeoff(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {"share": "tab:blue", "unshare": "tab:orange"}

    for row in rows:
        x = row["total_params"] / 1_000_000
        y = row["test_acc"]
        ax.scatter(x, y, s=70, color=color_map[row["mode"]], alpha=0.8)
        ax.text(x, y + 0.15, f"{row['mode']} {row['noise_label']}", fontsize=8, ha="center")

    ax.set_title("Parameter-Accuracy Tradeoff")
    ax.set_xlabel("Total parameters (Millions)")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _mode_rows(rows: list[dict], mode: str) -> list[dict]:
    return sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["noise_db"])


def _mode_label(mode: str) -> str:
    return "Shared ViT" if mode == "share" else "Vanilla ViT"


def plot_paper_robustness_figure(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    style = {
        "share": {"color": "#1f4e79", "marker": "o"},
        "unshare": {"color": "#c75b12", "marker": "s"},
    }

    for mode in ("share", "unshare"):
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue

        xs = np.array([row["noise_db"] for row in mode_rows], dtype=float)
        ys = np.array([row["test_acc"] for row in mode_rows], dtype=float)
        ax.plot(
            xs,
            ys,
            linewidth=2.4,
            markersize=7,
            label=_mode_label(mode),
            **style[mode],
        )
        ax.fill_between(xs, ys, ys.min() - 0.8, color=style[mode]["color"], alpha=0.08)

        for x, y in zip(xs, ys):
            ax.text(x, y + 0.18, f"{y:.2f}", fontsize=8, ha="center", color=style[mode]["color"])

    ax.set_title("Robustness Comparison Under Different Noise Levels", fontsize=13, fontweight="bold")
    ax.set_xlabel("Noise level (dB)", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_paper_overview_figure(rows: list[dict], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.24)

    ax_main = fig.add_subplot(gs[0, :])
    ax_param = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    style = {
        "share": {"color": "#1f4e79", "marker": "o"},
        "unshare": {"color": "#c75b12", "marker": "s"},
    }

    for mode in ("share", "unshare"):
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
            label=_mode_label(mode),
            **style[mode],
        )

    ax_main.set_title("(a) Accuracy Under Noise", fontsize=12, fontweight="bold")
    ax_main.set_xlabel("Noise level (dB)")
    ax_main.set_ylabel("Test accuracy (%)")
    ax_main.grid(True, alpha=0.25, linestyle="--")
    ax_main.legend(frameon=False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    params_m = []
    labels = []
    colors = []
    for mode in ("share", "unshare"):
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue
        labels.append(_mode_label(mode))
        params_m.append(mode_rows[0]["total_params"] / 1_000_000)
        colors.append(style[mode]["color"])

    bars = ax_param.bar(labels, params_m, color=colors, alpha=0.88, width=0.6)
    for bar, value in zip(bars, params_m):
        ax_param.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}M", ha="center", va="bottom", fontsize=9)
    ax_param.set_title("(b) Parameter Count", fontsize=12, fontweight="bold")
    ax_param.set_ylabel("Parameters (Millions)")
    ax_param.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_param.spines["top"].set_visible(False)
    ax_param.spines["right"].set_visible(False)

    avg_times = []
    for mode in ("share", "unshare"):
        mode_rows = _mode_rows(rows, mode)
        if not mode_rows:
            continue
        avg_times.append(float(np.mean([row["training_time_sec"] for row in mode_rows])))

    bars = ax_time.bar(labels, avg_times, color=colors, alpha=0.88, width=0.6)
    for bar, value in zip(bars, avg_times):
        ax_time.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax_time.set_title("(c) Average Training Time", fontsize=12, fontweight="bold")
    ax_time.set_ylabel("Time (sec)")
    ax_time.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)

    fig.suptitle("Shared vs Unshared ViT: Robustness, Efficiency, and Capacity", fontsize=14, fontweight="bold", y=0.98)
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
        "unshare_test_acc",
        "share_best_val_acc",
        "unshare_best_val_acc",
        "share_total_params",
        "unshare_total_params",
        "share_training_time_sec",
        "unshare_training_time_sec",
        "share_run_dir",
        "unshare_run_dir",
    ]
    write_csv(wide_rows, benchmark_dir / "summary_wide.csv", wide_fields)
    write_excel_xml(wide_rows, benchmark_dir / "summary_wide_excel.xml", wide_fields)
    write_markdown_table(rows, benchmark_dir / "summary.md")

    plot_accuracy_vs_noise(rows, benchmark_dir / "accuracy_vs_noise.png")
    plot_training_time_vs_noise(rows, benchmark_dir / "training_time_vs_noise.png")
    plot_param_compare(rows, benchmark_dir / "parameter_compare.png")
    plot_param_accuracy_tradeoff(rows, benchmark_dir / "parameter_accuracy_tradeoff.png")
    plot_paper_robustness_figure(rows, benchmark_dir / "paper_robustness_main.png")
    plot_paper_overview_figure(rows, benchmark_dir / "paper_overview.png")


def run_single_experiment(
    benchmark_name: str,
    benchmark_dir: Path,
    checkpoint_root: Path,
    data_dir: Path,
    share_transformer_weights: bool,
    run_index: int,
    num_epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    show_plots: bool,
) -> dict:
    mode_tag = format_mode(share_transformer_weights)
    noise_label = data_dir.name
    temp_run_name = build_temp_run_name(benchmark_name, mode_tag, noise_label, run_index)
    temp_run_dir = checkpoint_root / temp_run_name

    if temp_run_dir.exists():
        shutil.rmtree(temp_run_dir)

    print("=" * 88)
    print(f"Starting run {run_index:02d}: mode={mode_tag}, dataset={data_dir}")

    train_vit(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        share_transformer_weights=share_transformer_weights,
        run_name=temp_run_name,
        data_dir=str(data_dir),
        show_plots=show_plots,
    )

    metrics_path = temp_run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training finished but metrics.json is missing: {metrics_path}")

    metrics = load_metrics(metrics_path)
    final_accuracy = float(metrics.get("test_acc", metrics.get("best_val_acc", 0.0)))
    final_dir_name = build_final_dir_name(mode_tag, noise_label, run_index, final_accuracy)
    final_run_dir = ensure_unique_path(benchmark_dir / final_dir_name)
    shutil.move(str(temp_run_dir), str(final_run_dir))

    final_metrics_path = final_run_dir / "metrics.json"
    metrics = load_metrics(final_metrics_path)
    metrics["benchmark_name"] = benchmark_name
    metrics["mode"] = mode_tag
    metrics["noise_label"] = noise_label
    metrics["noise_db"] = extract_noise_db(noise_label)
    metrics["run_index"] = run_index
    metrics["run_name"] = final_dir_name
    metrics["run_dir"] = str(final_run_dir.resolve())
    save_metrics(final_metrics_path, metrics)

    row = {
        "run_index": run_index,
        "mode": mode_tag,
        "noise_label": noise_label,
        "noise_db": float(metrics["noise_db"]),
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
        for share_transformer_weights in (True, False):
            plan.append(
                {
                    "run_index": run_index,
                    "mode": format_mode(share_transformer_weights),
                    "share_transformer_weights": share_transformer_weights,
                    "noise_label": noise_dir.name,
                    "data_dir": noise_dir,
                }
            )
            run_index += 1
    return plan


def load_existing_rows(benchmark_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not benchmark_dir.exists():
        return rows

    for run_dir in benchmark_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        if "run_index" not in metrics or "mode" not in metrics or "noise_label" not in metrics:
            continue
        rows.append(
            {
                "run_index": int(metrics["run_index"]),
                "mode": str(metrics["mode"]),
                "noise_label": str(metrics["noise_label"]),
                "noise_db": float(metrics.get("noise_db", extract_noise_db(str(metrics["noise_label"])))),
                "test_acc": float(metrics.get("test_acc", 0.0)),
                "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
                "test_loss": float(metrics.get("test_loss", 0.0)),
                "total_params": int(metrics.get("total_params", 0)),
                "trainable_params": int(metrics.get("trainable_params", 0)),
                "training_time_sec": float(metrics.get("training_time_sec", 0.0)),
                "data_dir": str(metrics.get("data_dir", "")),
                "run_dir_name": run_dir.name,
                "run_dir": str(run_dir.resolve()),
            }
        )

    rows.sort(key=lambda row: row["run_index"])
    return rows


def write_progress(benchmark_dir: Path, rows: list[dict], failures: list[dict], total_runs: int) -> None:
    progress = {
        "completed_runs": len(rows),
        "total_runs": total_runs,
        "completed_run_indices": [row["run_index"] for row in sorted(rows, key=lambda row: row["run_index"])],
        "failures": failures,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (benchmark_dir / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def write_plan(benchmark_dir: Path, plan: list[dict]) -> None:
    serializable_plan = [
        {
            "run_index": item["run_index"],
            "mode": item["mode"],
            "noise_label": item["noise_label"],
            "data_dir": str(item["data_dir"]),
        }
        for item in plan
    ]
    (benchmark_dir / "run_plan.json").write_text(
        json.dumps(serializable_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(checkpoint_root)

    benchmark_name = args.benchmark_name or f"vit_noise_benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    requested_benchmark_dir = checkpoint_root / benchmark_name
    if requested_benchmark_dir.exists():
        if args.resume or args.benchmark_name is not None:
            benchmark_dir = requested_benchmark_dir
        else:
            benchmark_dir = ensure_unique_path(requested_benchmark_dir)
    else:
        benchmark_dir = requested_benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    noise_dirs = discover_noise_dirs(data_root)
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
            share_transformer_weights = item["share_transformer_weights"]
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
                    share_transformer_weights=share_transformer_weights,
                    run_index=run_index,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    show_plots=args.show_plots,
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
