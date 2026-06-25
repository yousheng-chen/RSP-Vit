import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _iter_image_files(class_dir: Path):
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _group_files_by_hash(class_dir: Path):
    """
    Group duplicated files by content hash so identical images stay in the same split.
    """
    groups = defaultdict(list)
    for path in _iter_image_files(class_dir):
        groups[_file_md5(path)].append(path)
    return list(groups.values())


def _assign_groups_to_splits(groups, train_ratio: float, val_ratio: float, seed: int):
    """
    Greedy assignment by group size.

    We shuffle deterministically, then place larger duplicate-groups first so
    train/val/test file counts stay close to the requested ratios.
    """
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total_files = sum(len(g) for g in groups)
    targets = {
        "train": train_ratio * total_files,
        "val": val_ratio * total_files,
        "test": total_files - (train_ratio * total_files) - (val_ratio * total_files),
    }
    assigned = {"train": [], "val": [], "test": []}
    current = {"train": 0, "val": 0, "test": 0}

    for group in groups:
        group_size = len(group)

        # Prefer the split that is most under target after considering current counts.
        best_split = min(
            current.keys(),
            key=lambda split: (current[split] + group_size - targets[split], current[split] / max(targets[split], 1e-8)),
        )
        assigned[best_split].append(group)
        current[best_split] += group_size

    return assigned, current, total_files


def _copy_groups(split_root: Path, class_name: str, groups):
    class_out = split_root / class_name
    class_out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for group in groups:
        for src in group:
            shutil.copy2(src, class_out / src.name)
            copied += 1
    return copied


def build_hashsafe_split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")

    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found under: {source_dir}")

    train_root = output_dir / "train"
    val_root = output_dir / "val"
    test_root = output_dir / "test"
    for root in (train_root, val_root, test_root):
        root.mkdir(parents=True, exist_ok=True)

    overall = {"train": 0, "val": 0, "test": 0, "total": 0}

    print("=" * 72)
    print("Creating hash-safe split dataset")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Requested ratios -> train: {train_ratio:.0%}, val: {val_ratio:.0%}, test: {1 - train_ratio - val_ratio:.0%}")
    print(f"Seed: {seed}")
    print("=" * 72)

    for class_name in classes:
        class_dir = source_dir / class_name
        groups = _group_files_by_hash(class_dir)
        assigned, counts, total_files = _assign_groups_to_splits(groups, train_ratio, val_ratio, seed)

        copied_train = _copy_groups(train_root, class_name, assigned["train"])
        copied_val = _copy_groups(val_root, class_name, assigned["val"])
        copied_test = _copy_groups(test_root, class_name, assigned["test"])

        overall["train"] += copied_train
        overall["val"] += copied_val
        overall["test"] += copied_test
        overall["total"] += total_files

        duplicate_groups = sum(1 for g in groups if len(g) > 1)
        duplicate_files_extra = sum(len(g) - 1 for g in groups if len(g) > 1)

        print(f"\nClass: {class_name}")
        print(f"  Total files: {total_files}")
        print(f"  Unique hash groups: {len(groups)}")
        print(f"  Duplicate groups: {duplicate_groups}")
        print(f"  Extra duplicate files: {duplicate_files_extra}")
        print(f"  Train: {copied_train}")
        print(f"  Val:   {copied_val}")
        print(f"  Test:  {copied_test}")

    summary_path = output_dir / "split_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {source_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Ratios: train={train_ratio}, val={val_ratio}, test={1 - train_ratio - val_ratio}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total files: {overall['total']}\n")
        f.write(f"Train files: {overall['train']}\n")
        f.write(f"Val files: {overall['val']}\n")
        f.write(f"Test files: {overall['test']}\n")

    print("\n" + "=" * 72)
    print("Done.")
    print(f"Total files: {overall['total']}")
    print(f"Train: {overall['train']} ({overall['train'] / overall['total']:.1%})")
    print(f"Val:   {overall['val']} ({overall['val'] / overall['total']:.1%})")
    print(f"Test:  {overall['test']} ({overall['test'] / overall['total']:.1%})")
    print(f"Summary written to: {summary_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Create a hash-safe split dataset that keeps duplicate images in the same split.")
    parser.add_argument("--source-dir", required=True, help="Source ImageFolder dataset root.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root; will create train/val/test inside it.")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Train ratio. Default: 0.5")
    parser.add_argument("--val-ratio", type=float, default=0.25, help="Validation ratio. Default: 0.25")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic assignment.")
    args = parser.parse_args()

    build_hashsafe_split_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
