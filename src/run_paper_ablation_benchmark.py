from __future__ import annotations

import argparse
import csv
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
    load_metrics,
    save_metrics,
    write_csv,
    write_excel_xml,
    write_plan,
    write_progress,
)


MODE_ORDER = ("vanilla_vit", "shared_vit", "stem_only_vit", "rsp_vit")
RUN_MODE_ORDER = MODE_ORDER

MODE_LABELS = {
    "vanilla_vit": "Vanilla ViT",
    "shared_vit": "Shared ViT",
    "stem_only_vit": "Stem-only ViT",
    "rsp_vit": "RSP-ViT",
}

MODE_COLORS = {
    "vanilla_vit": "#6b7280",
    "shared_vit": "#1f4e79",
    "stem_only_vit": "#c75b12",
    "rsp_vit": "#1a7f5a",
}

MODE_MARKERS = {
    "vanilla_vit": "o",
    "shared_vit": "s",
    "stem_only_vit": "^",
    "rsp_vit": "D",
}

MODE_SETTINGS = {
    "vanilla_vit": {"share_transformer_weights": False, "patch_embed_variant": "standard"},
    "shared_vit": {"share_transformer_weights": True, "patch_embed_variant": "standard"},
    "stem_only_vit": {"share_transformer_weights": False, "patch_embed_variant": "resnet_stem"},
    "rsp_vit": {"share_transformer_weights": True, "patch_embed_variant": "resnet_stem"},
}

DEFAULT_BASELINE_SUMMARY = (
    Path(PROJECT_ROOT)
    / "checkpoint"
    / "share与其他模型对比"
    / "share vs unshare"
    / "summary_wide.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the paper ablation benchmark by uniformly running all four ViT variants "
            "across the selected noisy datasets."
        )
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(PROJECT_ROOT) / "data" / "noisy"),
        help="Root directory containing noise subfolders such as -10db, -5db, 0db, 5db, 10db.",
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
        help="Optional benchmark folder name. Default: paper_ablation_<timestamp>.",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="*",
        default=["-10db", "-5db", "0db", "5db", "10db"],
        help="Subset of noise folder names to run. Default: -10db -5db 0db 5db 10db",
    )
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default=str(DEFAULT_BASELINE_SUMMARY),
        help="Existing shared-vs-unshared summary_wide.csv used only when --import-baselines is enabled.",
    )
    parser.add_argument(
        "--import-baselines",
        action="store_true",
        help="Import existing Vanilla ViT and Shared ViT baselines instead of rerunning them.",
    )
    parser.add_argument(
        "--main-noise-level",
        type=str,
        default="-10db",
        help="Noise level used to build the main ablation table. Default: -10db",
    )
    parser.add_argument("--num-epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 1e-5))
    parser.add_argument("--img-size", type=int, nargs=2, default=config.get("img_size", [192, 192]))
    parser.add_argument("--train-split", type=float, default=config.get("train_split", 0.5))
    parser.add_argument("--val-split", type=float, default=config.get("val_split", 0.25))
    parser.add_argument("--seed", type=int, default=42)
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


def format_noise_label(noise_db: float) -> str:
    if float(noise_db).is_integer():
        return f"{int(noise_db)}db"
    return f"{noise_db:g}db"


def mode_rank(mode: str) -> int:
    try:
        return MODE_ORDER.index(mode)
    except ValueError:
        return len(MODE_ORDER)


def normalize_bool_text(value: bool) -> str:
    return "Yes" if value else "No"


def format_metric(value) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.2f}"


def as_float(value, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def as_int(value, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def load_baseline_rows(summary_wide_path: Path, requested_noise_levels: set[str]) -> list[dict]:
    if not summary_wide_path.exists():
        raise FileNotFoundError(f"Baseline summary file does not exist: {summary_wide_path}")

    rows: list[dict] = []
    with summary_wide_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            noise_db = float(record["noise_db"])
            noise_label = format_noise_label(noise_db)
            if noise_label.lower() not in requested_noise_levels:
                continue

            baseline_specs = [
                (
                    "shared_vit",
                    "share_test_acc",
                    "share_best_val_acc",
                    "share_total_params",
                    "share_training_time_sec",
                    "share_run_dir",
                ),
                (
                    "vanilla_vit",
                    "unshare_test_acc",
                    "unshare_best_val_acc",
                    "unshare_total_params",
                    "unshare_training_time_sec",
                    "unshare_run_dir",
                ),
            ]

            for mode, acc_key, val_key, param_key, time_key, run_dir_key in baseline_specs:
                if record.get(acc_key, "") == "":
                    continue
                settings = MODE_SETTINGS[mode]
                run_dir_name = str(record.get(run_dir_key, ""))
                rows.append(
                    {
                        "run_index": 0,
                        "mode": mode,
                        "noise_label": noise_label,
                        "noise_db": noise_db,
                        "share_transformer_weights": settings["share_transformer_weights"],
                        "patch_embed_variant": settings["patch_embed_variant"],
                        "test_acc": as_float(record.get(acc_key)),
                        "best_val_acc": as_float(record.get(val_key)),
                        "test_loss": 0.0,
                        "total_params": as_int(record.get(param_key)),
                        "trainable_params": as_int(record.get(param_key)),
                        "training_time_sec": as_float(record.get(time_key)),
                        "data_dir": "",
                        "run_dir_name": run_dir_name,
                        "run_dir": run_dir_name,
                        "source": "imported_baseline",
                    }
                )

    rows.sort(key=lambda row: (row["noise_db"], mode_rank(row["mode"])))
    return rows


def load_existing_experiment_rows(benchmark_dir: Path) -> list[dict]:
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
                "share_transformer_weights": bool(metrics.get("share_transformer_weights", False)),
                "patch_embed_variant": str(metrics.get("patch_embed_variant", "standard")),
                "test_acc": float(metrics.get("test_acc", 0.0)),
                "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
                "test_loss": float(metrics.get("test_loss", 0.0)),
                "total_params": int(metrics.get("total_params", 0)),
                "trainable_params": int(metrics.get("trainable_params", 0)),
                "training_time_sec": float(metrics.get("training_time_sec", 0.0)),
                "data_dir": str(metrics.get("data_dir", "")),
                "run_dir_name": run_dir.name,
                "run_dir": str(run_dir.resolve()),
                "source": "new_run",
            }
        )

    rows.sort(key=lambda row: row["run_index"])
    return rows


def merge_rows(baseline_rows: list[dict], experiment_rows: list[dict]) -> list[dict]:
    merged = {(row["mode"], row["noise_label"]): dict(row) for row in baseline_rows}
    for row in experiment_rows:
        merged[(row["mode"], row["noise_label"])] = dict(row)
    rows = list(merged.values())
    rows.sort(key=lambda row: (row["noise_db"], mode_rank(row["mode"])))
    return rows


def build_run_plan(noise_dirs: list[Path]) -> list[dict]:
    plan: list[dict] = []
    run_index = 1
    for noise_dir in noise_dirs:
        for mode in RUN_MODE_ORDER:
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


def build_wide_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["noise_db"], {})[row["mode"]] = row

    wide_rows: list[dict] = []
    for noise_db in sorted(grouped.keys()):
        by_mode = grouped[noise_db]
        wide_rows.append(
            {
                "noise_db": noise_db,
                "vanilla_vit_test_acc": by_mode.get("vanilla_vit", {}).get("test_acc", ""),
                "shared_vit_test_acc": by_mode.get("shared_vit", {}).get("test_acc", ""),
                "stem_only_vit_test_acc": by_mode.get("stem_only_vit", {}).get("test_acc", ""),
                "rsp_vit_test_acc": by_mode.get("rsp_vit", {}).get("test_acc", ""),
                "vanilla_vit_best_val_acc": by_mode.get("vanilla_vit", {}).get("best_val_acc", ""),
                "shared_vit_best_val_acc": by_mode.get("shared_vit", {}).get("best_val_acc", ""),
                "stem_only_vit_best_val_acc": by_mode.get("stem_only_vit", {}).get("best_val_acc", ""),
                "rsp_vit_best_val_acc": by_mode.get("rsp_vit", {}).get("best_val_acc", ""),
                "vanilla_vit_total_params": by_mode.get("vanilla_vit", {}).get("total_params", ""),
                "shared_vit_total_params": by_mode.get("shared_vit", {}).get("total_params", ""),
                "stem_only_vit_total_params": by_mode.get("stem_only_vit", {}).get("total_params", ""),
                "rsp_vit_total_params": by_mode.get("rsp_vit", {}).get("total_params", ""),
                "vanilla_vit_training_time_sec": by_mode.get("vanilla_vit", {}).get("training_time_sec", ""),
                "shared_vit_training_time_sec": by_mode.get("shared_vit", {}).get("training_time_sec", ""),
                "stem_only_vit_training_time_sec": by_mode.get("stem_only_vit", {}).get("training_time_sec", ""),
                "rsp_vit_training_time_sec": by_mode.get("rsp_vit", {}).get("training_time_sec", ""),
                "vanilla_vit_run_dir": by_mode.get("vanilla_vit", {}).get("run_dir_name", ""),
                "shared_vit_run_dir": by_mode.get("shared_vit", {}).get("run_dir_name", ""),
                "stem_only_vit_run_dir": by_mode.get("stem_only_vit", {}).get("run_dir_name", ""),
                "rsp_vit_run_dir": by_mode.get("rsp_vit", {}).get("run_dir_name", ""),
            }
        )
    return wide_rows


def build_main_table_rows(rows: list[dict], main_noise_db: float) -> list[dict]:
    filtered = [row for row in rows if abs(row["noise_db"] - main_noise_db) < 1e-9]
    filtered.sort(key=lambda row: mode_rank(row["mode"]))

    main_rows: list[dict] = []
    for row in filtered:
        settings = MODE_SETTINGS[row["mode"]]
        main_rows.append(
            {
                "model": MODE_LABELS[row["mode"]],
                "shared_weights": normalize_bool_text(settings["share_transformer_weights"]),
                "resnet_stem": normalize_bool_text(settings["patch_embed_variant"] == "resnet_stem"),
                "params_m": round(row["total_params"] / 1_000_000, 3),
                "best_val_acc": round(row["best_val_acc"], 2),
                "test_acc": round(row["test_acc"], 2),
            }
        )
    return main_rows


def build_robustness_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[float, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["noise_db"], {})[row["mode"]] = row

    robustness_rows: list[dict] = []
    for noise_db in sorted(grouped.keys()):
        by_mode = grouped[noise_db]
        robustness_rows.append(
            {
                "noise_db": noise_db,
                "vanilla_vit": by_mode.get("vanilla_vit", {}).get("test_acc", ""),
                "shared_vit": by_mode.get("shared_vit", {}).get("test_acc", ""),
                "stem_only_vit": by_mode.get("stem_only_vit", {}).get("test_acc", ""),
                "rsp_vit": by_mode.get("rsp_vit", {}).get("test_acc", ""),
            }
        )
    return robustness_rows


def write_markdown_lines(lines: list[str], out_path: Path) -> None:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Paper Ablation Benchmark Summary",
        "",
        "| mode | noise_db | shared_weights | patch_embed_variant | test_acc | best_val_acc | total_params | training_time_sec | source | run_dir |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {mode_label} | {noise_db:.1f} | {shared} | {patch} | {test_acc:.2f} | {best_val_acc:.2f} | {total_params} | {training_time_sec:.2f} | {source} | {run_dir_name} |".format(
                mode_label=MODE_LABELS.get(row["mode"], row["mode"]),
                noise_db=row["noise_db"],
                shared=normalize_bool_text(bool(row["share_transformer_weights"])),
                patch=row["patch_embed_variant"],
                test_acc=row["test_acc"],
                best_val_acc=row["best_val_acc"],
                total_params=row["total_params"],
                training_time_sec=row["training_time_sec"],
                source=row.get("source", ""),
                run_dir_name=row.get("run_dir_name", ""),
            )
        )
    write_markdown_lines(lines, out_path)


def write_main_table_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Main Ablation Table",
        "",
        "| Model | Shared Weights | ResNet-style Stem | Params (M) | Best Val. (%) | Test Acc. (%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {shared_weights} | {resnet_stem} | {params_m:.3f} | {best_val_acc:.2f} | {test_acc:.2f} |".format(
                **row
            )
        )
    write_markdown_lines(lines, out_path)


def write_robustness_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Robustness Ablation Table",
        "",
        "| Noise level (dB) | Vanilla ViT | Shared ViT | Stem-only ViT | RSP-ViT |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['noise_db']:.1f} | {format_metric(row['vanilla_vit'])} | "
            f"{format_metric(row['shared_vit'])} | {format_metric(row['stem_only_vit'])} | "
            f"{format_metric(row['rsp_vit'])} |"
        )
    write_markdown_lines(lines, out_path)


def plot_robustness_curve(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    for mode in MODE_ORDER:
        mode_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["noise_db"])
        if not mode_rows:
            continue
        xs = np.array([row["noise_db"] for row in mode_rows], dtype=float)
        ys = np.array([row["test_acc"] for row in mode_rows], dtype=float)
        ax.plot(
            xs,
            ys,
            linewidth=2.4,
            markersize=7,
            marker=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.15, f"{y:.2f}", fontsize=8, ha="center", color=MODE_COLORS[mode])

    ax.set_title("Ablation Robustness Under Different Noise Levels", fontsize=13, fontweight="bold")
    ax.set_xlabel("Noise level (dB)", fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=10, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def generate_summary_outputs(rows: list[dict], benchmark_dir: Path, main_noise_db: float) -> None:
    if not rows:
        return

    rows = sorted(rows, key=lambda row: (row["noise_db"], mode_rank(row["mode"])))
    summary_fields = [
        "run_index",
        "mode",
        "noise_label",
        "noise_db",
        "share_transformer_weights",
        "patch_embed_variant",
        "test_acc",
        "best_val_acc",
        "test_loss",
        "total_params",
        "trainable_params",
        "training_time_sec",
        "source",
        "data_dir",
        "run_dir_name",
        "run_dir",
    ]
    write_csv(rows, benchmark_dir / "summary.csv", summary_fields)
    write_excel_xml(rows, benchmark_dir / "summary_excel.xml", summary_fields)
    write_summary_markdown(rows, benchmark_dir / "summary.md")

    wide_rows = build_wide_rows(rows)
    wide_fields = [
        "noise_db",
        "vanilla_vit_test_acc",
        "shared_vit_test_acc",
        "stem_only_vit_test_acc",
        "rsp_vit_test_acc",
        "vanilla_vit_best_val_acc",
        "shared_vit_best_val_acc",
        "stem_only_vit_best_val_acc",
        "rsp_vit_best_val_acc",
        "vanilla_vit_total_params",
        "shared_vit_total_params",
        "stem_only_vit_total_params",
        "rsp_vit_total_params",
        "vanilla_vit_training_time_sec",
        "shared_vit_training_time_sec",
        "stem_only_vit_training_time_sec",
        "rsp_vit_training_time_sec",
        "vanilla_vit_run_dir",
        "shared_vit_run_dir",
        "stem_only_vit_run_dir",
        "rsp_vit_run_dir",
    ]
    write_csv(wide_rows, benchmark_dir / "summary_wide.csv", wide_fields)
    write_excel_xml(wide_rows, benchmark_dir / "summary_wide_excel.xml", wide_fields)

    main_rows = build_main_table_rows(rows, main_noise_db=main_noise_db)
    main_fields = ["model", "shared_weights", "resnet_stem", "params_m", "best_val_acc", "test_acc"]
    write_csv(main_rows, benchmark_dir / "paper_ablation_main_table.csv", main_fields)
    write_excel_xml(main_rows, benchmark_dir / "paper_ablation_main_table_excel.xml", main_fields)
    write_main_table_markdown(main_rows, benchmark_dir / "paper_ablation_main_table.md")

    robustness_rows = build_robustness_rows(rows)
    robustness_fields = ["noise_db", "vanilla_vit", "shared_vit", "stem_only_vit", "rsp_vit"]
    write_csv(robustness_rows, benchmark_dir / "paper_ablation_robustness.csv", robustness_fields)
    write_excel_xml(robustness_rows, benchmark_dir / "paper_ablation_robustness_excel.xml", robustness_fields)
    write_robustness_markdown(robustness_rows, benchmark_dir / "paper_ablation_robustness.md")

    plot_robustness_curve(rows, benchmark_dir / "paper_ablation_robustness.png")


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

    settings = MODE_SETTINGS[mode]
    print("=" * 88)
    print(
        f"Starting run {run_index:02d}: mode={mode}, dataset={data_dir}, "
        f"share_transformer_weights={settings['share_transformer_weights']}, "
        f"patch_embed_variant={settings['patch_embed_variant']}"
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
        share_transformer_weights=settings["share_transformer_weights"],
        use_depth_embeddings=False,
        ffn_variant="standard",
        patch_embed_variant=settings["patch_embed_variant"],
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
        "share_transformer_weights": bool(metrics.get("share_transformer_weights", settings["share_transformer_weights"])),
        "patch_embed_variant": str(metrics.get("patch_embed_variant", settings["patch_embed_variant"])),
        "test_acc": float(metrics.get("test_acc", 0.0)),
        "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
        "test_loss": float(metrics.get("test_loss", 0.0)),
        "total_params": int(metrics.get("total_params", 0)),
        "trainable_params": int(metrics.get("trainable_params", 0)),
        "training_time_sec": float(metrics.get("training_time_sec", 0.0)),
        "data_dir": metrics.get("data_dir", str(data_dir)),
        "run_dir_name": final_run_dir.name,
        "run_dir": str(final_run_dir.resolve()),
        "source": "new_run",
    }

    print(f"Completed run {run_index:02d}: {final_run_dir.name}")
    return row


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(checkpoint_root)

    benchmark_name = args.benchmark_name or f"paper_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    requested_benchmark_dir = checkpoint_root / benchmark_name
    if requested_benchmark_dir.exists():
        benchmark_dir = requested_benchmark_dir if (args.resume or args.benchmark_name is not None) else ensure_unique_path(requested_benchmark_dir)
    else:
        benchmark_dir = requested_benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    noise_dirs = filter_noise_dirs(discover_noise_dirs(data_root), args.noise_levels)
    requested_noise_levels = {path.name.lower() for path in noise_dirs}
    main_noise_db = extract_noise_db(args.main_noise_level)
    baseline_rows = (
        load_baseline_rows(Path(args.baseline_summary), requested_noise_levels)
        if args.import_baselines
        else []
    )
    run_plan = build_run_plan(noise_dirs)
    total_runs = len(run_plan)
    write_plan(benchmark_dir, run_plan)

    print(f"Benchmark directory: {benchmark_dir.resolve()}")
    if args.import_baselines:
        print(f"Baseline summary: {Path(args.baseline_summary).resolve()}")
    print("Noise datasets:")
    for noise_dir in noise_dirs:
        print(f"  - {noise_dir.name}")

    experiment_rows = load_existing_experiment_rows(benchmark_dir)
    failures: list[dict] = []
    completed_keys = {(row["run_index"], row["mode"], row["noise_label"]) for row in experiment_rows}
    combined_rows = merge_rows(baseline_rows, experiment_rows)
    write_progress(benchmark_dir, experiment_rows, failures, total_runs)
    generate_summary_outputs(combined_rows, benchmark_dir, main_noise_db=main_noise_db)

    if experiment_rows:
        print(f"Resuming benchmark: already completed {len(experiment_rows)}/{total_runs} new runs.")

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
                experiment_rows.append(row)
                completed_keys.add((row["run_index"], row["mode"], row["noise_label"]))
                combined_rows = merge_rows(baseline_rows, experiment_rows)
                generate_summary_outputs(combined_rows, benchmark_dir, main_noise_db=main_noise_db)
                write_progress(benchmark_dir, experiment_rows, failures, total_runs)
            except Exception as exc:
                failure = {
                    "run_index": run_index,
                    "mode": mode,
                    "noise_label": noise_label,
                    "error": repr(exc),
                }
                failures.append(failure)
                print(f"Run failed: {failure}")
                write_progress(benchmark_dir, experiment_rows, failures, total_runs)
                if not args.continue_on_error:
                    combined_rows = merge_rows(baseline_rows, experiment_rows)
                    generate_summary_outputs(combined_rows, benchmark_dir, main_noise_db=main_noise_db)
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Partial results have been kept.")
        combined_rows = merge_rows(baseline_rows, experiment_rows)
        generate_summary_outputs(combined_rows, benchmark_dir, main_noise_db=main_noise_db)
        write_progress(benchmark_dir, experiment_rows, failures, total_runs)
        raise

    combined_rows = merge_rows(baseline_rows, experiment_rows)
    generate_summary_outputs(combined_rows, benchmark_dir, main_noise_db=main_noise_db)
    write_progress(benchmark_dir, experiment_rows, failures, total_runs)

    if failures:
        failure_path = benchmark_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote failures to: {failure_path}")

    print("=" * 88)
    print("Paper ablation benchmark complete.")
    print(f"Summary CSV: {benchmark_dir / 'summary.csv'}")
    print(f"Wide summary CSV: {benchmark_dir / 'summary_wide.csv'}")
    print(f"Main ablation table: {benchmark_dir / 'paper_ablation_main_table.csv'}")
    print(f"Robustness table: {benchmark_dir / 'paper_ablation_robustness.csv'}")
    print(f"Robustness plot: {benchmark_dir / 'paper_ablation_robustness.png'}")
    print(f"Benchmark folder: {benchmark_dir.resolve()}")


if __name__ == "__main__":
    main()
