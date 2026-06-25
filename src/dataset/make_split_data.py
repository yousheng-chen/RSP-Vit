"""
Generate an augmented dataset by cropping a center ROI band and tiling square patches.

Default behavior (matches our earlier sizing discussion for 875x656 spectrogram-like images):
- For classes except 'Walk':
  1) Take a center vertical ROI band of height 224 (approx 1/3 of 656, and divisible by 16)
  2) From that ROI, take 192x192 square tiles (divisible by 16)
  3) Slide along width with stride 160 -> for width 875, yields 5 tiles:
     floor((875 - 192) / 160) + 1 = 5
- For 'Walk': do NOT split; just copy images as-is so the class is preserved.

This script only creates a new dataset folder; it does not modify training code.
"""
#160一步，剪切出192×192的图像块，中心区域高度224，其他类都这样处理，Walk类不处理直接复制
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (value // multiple) * multiple)


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def _save_jpeg(img: Image.Image, out_path: Path, quality: int = 95) -> None:
    parent = out_path.parent
    # The parent class directory should already be created once per class.
    # Still, guard against missing parent or a path collision (file with same name).
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    elif not parent.is_dir():
        raise NotADirectoryError(f"Output parent exists but is not a directory: {parent}")
    img.save(out_path, "JPEG", quality=quality)


def _split_one_image(
    img: Image.Image,
    *,
    roi_height: int,
    tile_size: int,
    stride: int,
    multiple: int,
) -> list[Image.Image]:
    """
    Returns a list of PIL images (tiles).
    """
    w, h = img.size

    roi_height = min(roi_height, h)
    roi_height = _round_down_to_multiple(roi_height, multiple)

    tile_size = min(tile_size, w, roi_height)
    tile_size = _round_down_to_multiple(tile_size, multiple)

    if roi_height <= 0 or tile_size <= 0:
        return []

    # 1) Center ROI band (full width, center height slice)
    roi_y0 = max(0, (h - roi_height) // 2)
    roi = img.crop((0, roi_y0, w, roi_y0 + roi_height))

    # 2) Center tile vertically within ROI (ROI is slightly taller than tile)
    tile_y0 = max(0, (roi_height - tile_size) // 2)
    tile_y1 = tile_y0 + tile_size

    tiles: list[Image.Image] = []
    max_x0 = w - tile_size
    if max_x0 < 0:
        return tiles

    x0 = 0
    while x0 <= max_x0:
        tiles.append(roi.crop((x0, tile_y0, x0 + tile_size, tile_y1)))
        x0 += stride

    return tiles


def build_split_dataset(
    input_root: Path,
    output_root: Path,
    *,
    skip_split_classes: set[str],
    roi_height: int,
    tile_size: int,
    stride: int,
    multiple: int,
    jpeg_quality: int,
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subfolders found under: {input_root}")

    total_in = 0
    total_out = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        out_class_dir = output_root / class_name
        if out_class_dir.exists() and not out_class_dir.is_dir():
            raise NotADirectoryError(
                f"Output class path exists but is not a directory: {out_class_dir}"
            )
        out_class_dir.mkdir(parents=True, exist_ok=True)

        in_count = 0
        out_count = 0

        for img_path in _iter_images(class_dir):
            in_count += 1
            total_in += 1

            if class_name in skip_split_classes:
                # Copy as-is (preserve class and avoid over-splitting small square images)
                dst = out_class_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                out_count += 1
                total_out += 1
                continue

            with Image.open(img_path) as im:
                im = im.convert("RGB")
                tiles = _split_one_image(
                    im,
                    roi_height=roi_height,
                    tile_size=tile_size,
                    stride=stride,
                    multiple=multiple,
                )

            stem = img_path.stem
            for idx, tile in enumerate(tiles):
                out_name = f"{stem}_roi{roi_height}_tile{tile_size}_s{stride}_{idx+1}.jpg"
                out_path = out_class_dir / out_name
                if not out_path.exists():
                    _save_jpeg(tile, out_path, quality=jpeg_quality)
                out_count += 1
                total_out += 1

        print(f"{class_name}: in={in_count} out={out_count}")

    print(f"Done. Total in={total_in}, total out={total_out}")
    print(f"Output folder: {output_root}")


def _default_paths() -> tuple[Path, Path]:
    # This file is at: <project_root>/src/dataset/make_split_data.py
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "processed_data", project_root / "data" / "split_data"


def main() -> None:
    default_in, default_out = _default_paths()

    parser = argparse.ArgumentParser(
        description="Create split_data from processed_data using center ROI + square tiling."
    )

    # These are the main knobs you asked to be clearly editable.
    parser.add_argument("--input-dir", type=str, default=str(default_in), help="Input dataset root (class subfolders).")
    parser.add_argument("--output-dir", type=str, default=str(default_out), help="Output dataset root to create.")

    parser.add_argument("--skip-classes", nargs="*", default=["Walk"], help="Class names to copy without splitting.")
    parser.add_argument("--roi-height", type=int, default=224, help="Center ROI band height (pixels).")
    parser.add_argument("--tile-size", type=int, default=192, help="Square tile size (pixels).")
    parser.add_argument("--stride", type=int, default=160, help="Stride along width (pixels).")
    parser.add_argument("--multiple", type=int, default=16, help="Round roi/tile sizes down to a multiple.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG save quality.")

    args = parser.parse_args()

    build_split_dataset(
        Path(args.input_dir),
        Path(args.output_dir),
        skip_split_classes=set(args.skip_classes),
        roi_height=args.roi_height,
        tile_size=args.tile_size,
        stride=args.stride,
        multiple=args.multiple,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
