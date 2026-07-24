from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-keys-dir", type=Path, default=Path("evaluate/sample_keys"))
    return parser.parse_args()


def read_rows(sample_keys_dir: Path) -> list[dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for csv_path in sorted(sample_keys_dir.glob("partimagenet_*.csv")):
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[(row["sample_id"], row["class_name"])] = row
    return list(rows.values())


def decode_png(payload: bytes, path: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return image


def main() -> None:
    args = parse_args()
    rows = read_rows(args.sample_keys_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    copied_images: set[str] = set()

    with zipfile.ZipFile(args.source_zip) as archive:
        archive_names = set(archive.namelist())
        for index, row in enumerate(rows, start=1):
            split = row["split"]
            sample_id = row["sample_id"]
            image_archive_path = f"PartImageNet/images/{split}/{sample_id}.JPEG"
            annotation_archive_path = f"PartImageNet/annotations/{split}/{sample_id}.png"
            if image_archive_path not in archive_names or annotation_archive_path not in archive_names:
                raise FileNotFoundError(f"Missing source files for {sample_id}")

            image_output_path = args.output_root / row["image_path"]
            if image_archive_path not in copied_images:
                image_output_path.parent.mkdir(parents=True, exist_ok=True)
                image_output_path.write_bytes(archive.read(image_archive_path))
                copied_images.add(image_archive_path)

            labels = decode_png(archive.read(annotation_archive_path), annotation_archive_path)
            source_values = [
                int(value)
                for value in re.split(r"[,;\s]+", row["source_values"].strip())
                if value
            ]
            mask = np.isin(labels, source_values).astype(np.uint8) * 255
            expected_pixels = int(row["mask_pixels"])
            actual_pixels = int(np.count_nonzero(mask))
            if actual_pixels != expected_pixels:
                raise RuntimeError(
                    f"{sample_id}/{row['class_name']}: expected {expected_pixels} mask pixels, "
                    f"got {actual_pixels}"
                )

            mask_output_path = args.output_root / row["mask_path"]
            mask_output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_output_path), mask):
                raise RuntimeError(f"Failed to write {mask_output_path}")

            if index % 250 == 0 or index == len(rows):
                print(f"prepared {index}/{len(rows)}", flush=True)

    print(f"images={len(copied_images)} masks={len(rows)} output={args.output_root}")


if __name__ == "__main__":
    main()
