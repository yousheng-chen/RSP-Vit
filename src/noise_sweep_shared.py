"""
Run a robustness sweep: train the shared-weight ViT on multiple additive-noise levels
and write a single Excel-friendly summary (CSV).

Noise definition:
- Images are converted with ToTensor() to [0, 1]
- Additive Gaussian noise is applied: x' = clamp(x + N(0, noise_std), 0, 1)
  where noise_std = noise_pct / 100.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path


def _format_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return float("nan")


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


def main() -> None:
    # Import training/config in a way that matches how training.py is normally executed.
    import sys

    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    from config import config  # noqa: E402
    import training  # noqa: E402

    parser = argparse.ArgumentParser(description="Noise robustness sweep (shared ViT).")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root (class subfolders). Defaults to config['data_dir'].")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Checkpoint root. Defaults to config['checkpoint_dir'].")
    parser.add_argument("--run-name", type=str, default=None, help="Base run name for the sweep.")
    parser.add_argument("--noise-levels", nargs="*", type=float, default=[50, 70, 90], help="Noise levels in percent (e.g. 5 10 20).")
    parser.add_argument("--noise-prob", type=float, default=1.0, help="Probability to apply noise per sample (default 1.0).")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else config["data_dir"]
    checkpoint_root = args.checkpoint_root if args.checkpoint_root is not None else config["checkpoint_dir"]

    timestamp = _format_timestamp()
    base_name = args.run_name if args.run_name is not None else f"noise_sweep_{timestamp}"

    rows: list[dict] = []
    sweep_out_dir = Path(checkpoint_root) / f"noise_sweep_{base_name}"
    sweep_out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sweep_out_dir / "summary.csv"

    for lvl in args.noise_levels:
        noise_pct = float(lvl)
        noise_std = noise_pct / 100.0

        # Keep the directory name stable and unique.
        run_name = f"{base_name}_noise{int(round(noise_pct)):02d}_{timestamp}"

        print(f"\n=== Noise {noise_pct:.0f}% (std={noise_std:.3f}) | run_name={run_name} ===")

        training.train_vit(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            share_transformer_weights=True,
            run_name=run_name,
            data_dir=data_dir,
            noise_std=noise_std,
            noise_prob=args.noise_prob,
        )

        run_dir = Path(checkpoint_root) / run_name
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for run {run_name}: {metrics_path}")

        import json

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        ckpt_mb = _safe_size_mb(run_dir / "best_vit_model.pth")

        rows.append(
            {
                "run_name": metrics.get("run_name", run_name),
                "noise_pct": noise_pct,
                "noise_std": noise_std,
                "noise_prob": float(metrics.get("noise_prob", args.noise_prob)),
                "share_transformer_weights": metrics.get("share_transformer_weights"),
                "total_params": metrics.get("total_params"),
                "trainable_params": metrics.get("trainable_params"),
                "best_val_acc": metrics.get("best_val_acc"),
                "test_acc": metrics.get("test_acc"),
                "test_loss": metrics.get("test_loss"),
                "training_time_sec": metrics.get("training_time_sec"),
                "checkpoint_mb": (ckpt_mb if not math.isnan(ckpt_mb) else ""),
                "data_dir": metrics.get("data_dir"),
                "img_size": metrics.get("img_size"),
                "batch_size": metrics.get("batch_size"),
                "num_epochs": metrics.get("num_epochs"),
                "learning_rate": metrics.get("learning_rate"),
                "weight_decay": metrics.get("weight_decay"),
                "seed": metrics.get("seed"),
            }
        )

        # Write partial progress so an interruption won't lose the sweep summary.
        _write_csv(rows, out_csv)
        print(f"Wrote summary CSV (partial): {out_csv}")

    print(f"\nWrote summary CSV: {out_csv}")
    print("Tip: open summary.csv directly with Excel.")


if __name__ == "__main__":
    main()
