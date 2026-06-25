import argparse
from pathlib import Path

from PIL import Image


def center_crop_image(img: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    """
    Crop the center region with the requested size.

    PIL image size is (width, height).
    """
    width, height = img.size
    if crop_width > width or crop_height > height:
        raise ValueError(
            f"Requested crop {crop_width}x{crop_height} exceeds image size {width}x{height}."
        )

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return img.crop((left, top, right, bottom))


def build_center_crop_dataset(source_dir: Path, output_dir: Path, crop_width: int, crop_height: int):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    copied = 0

    print("=" * 72)
    print("Creating center-cropped dataset copy")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Crop size: {crop_width}x{crop_height}")
    print("=" * 72)

    for src_path in source_dir.rglob("*"):
        rel_path = src_path.relative_to(source_dir)
        dst_path = output_dir / rel_path

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix.lower() not in image_suffixes:
            dst_path.write_bytes(src_path.read_bytes())
            continue

        with Image.open(src_path) as img:
            cropped = center_crop_image(img, crop_width=crop_width, crop_height=crop_height)
            save_kwargs = {}
            if src_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95
            cropped.save(dst_path, **save_kwargs)
        copied += 1

    summary_path = output_dir / "crop_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {source_dir}\n")
        f.write(f"Output: {output_dir}\n")
        f.write(f"Crop size: {crop_width}x{crop_height}\n")
        f.write(f"Images processed: {copied}\n")

    print(f"Images processed: {copied}")
    print(f"Summary written to: {summary_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Create a new dataset by center-cropping all images.")
    parser.add_argument("--source-dir", required=True, help="Source dataset root.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root.")
    parser.add_argument("--crop-width", type=int, required=True, help="Center crop width.")
    parser.add_argument("--crop-height", type=int, required=True, help="Center crop height.")
    args = parser.parse_args()

    build_center_crop_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        crop_width=args.crop_width,
        crop_height=args.crop_height,
    )


if __name__ == "__main__":
    main()
