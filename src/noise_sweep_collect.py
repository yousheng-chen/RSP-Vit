"""
Collect existing noise-run metrics.json files into a single Excel-friendly summary.csv.

Use this when you already finished training runs (or a sweep was interrupted) and you
want to re-generate the combined table without re-training.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect noise runs into one summary.csv.")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoint", help="Checkpoint root directory.")
    parser.add_argument(
        "--run-glob",
        type=str,
        required=True,
        help="Glob pattern for run directories under checkpoint-root (e.g. 'noise_sweep_20260321_171608_noise*').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to write summary.csv (e.g. 'checkpoint/noise_sweep_noise_sweep_20260321_171608').",
    )
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    run_glob = args.run_glob
    out_dir = Path(args.out_dir)

    run_dirs = sorted([p for p in checkpoint_root.glob(run_glob) if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories match: {checkpoint_root / run_glob}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = _load_metrics(metrics_path)

        rows.append(
            {
                "run_name": m.get("run_name", run_dir.name),
                "noise_pct": float(m.get("noise_std", 0.0)) * 100.0,
                "noise_std": m.get("noise_std"),
                "noise_prob": m.get("noise_prob"),
                "share_transformer_weights": m.get("share_transformer_weights"),
                "total_params": m.get("total_params"),
                "trainable_params": m.get("trainable_params"),
                "best_val_acc": m.get("best_val_acc"),
                "test_acc": m.get("test_acc"),
                "test_loss": m.get("test_loss"),
                "training_time_sec": m.get("training_time_sec"),
                "data_dir": m.get("data_dir"),
                "img_size": m.get("img_size"),
                "batch_size": m.get("batch_size"),
                "num_epochs": m.get("num_epochs"),
                "learning_rate": m.get("learning_rate"),
                "weight_decay": m.get("weight_decay"),
                "seed": m.get("seed"),
            }
        )

    # Sort by noise level for readability
    rows.sort(key=lambda r: (r.get("noise_pct", 0.0), str(r.get("run_name", ""))))

    out_csv = out_dir / "summary.csv"
    _write_csv(rows, out_csv)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

