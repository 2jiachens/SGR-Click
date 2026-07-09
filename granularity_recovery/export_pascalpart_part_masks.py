from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granularity_recovery.evaluate_part_noc85 import load_part_samples, load_sample, resolve_dataset_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--voc-images-dir", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--parts", type=str, default="head,wheel,torso")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--write-overlay", action="store_true", default=False)
    return parser.parse_args()


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix if path.suffix else ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def build_args_proxy(args):
    class Proxy:
        dataset_root = args.dataset_root
        voc_images_dir = args.voc_images_dir
        split_file = args.split_file

    return Proxy()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, voc_images_dir, split_file = resolve_dataset_paths(build_args_proxy(args))
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]

    manifest_rows: List[Dict[str, object]] = []
    summary: Dict[str, int] = {}

    for part_name in parts:
        samples = load_part_samples(dataset_root, voc_images_dir, split_file, part_name)
        summary[part_name] = len(samples)
        for sample in samples:
            image, gt_mask = load_sample(sample)
            mask_uint8 = (gt_mask > 0).astype(np.uint8) * 255
            base_name = f"{sample['sample_id']}_obj{int(sample['object_id']):02d}_part{int(sample['part_id']):02d}"

            mask_path = output_dir / part_name / f"{base_name}_mask.png"
            imwrite_unicode(mask_path, mask_uint8)

            overlay_path = ""
            if args.write_overlay:
                overlay = image.copy()
                overlay[gt_mask > 0] = (0.35 * overlay[gt_mask > 0] + 0.65 * np.array([255, 80, 80])).astype(np.uint8)
                overlay_path_obj = output_dir / part_name / f"{base_name}_overlay.png"
                imwrite_unicode(overlay_path_obj, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                overlay_path = str(overlay_path_obj)

            manifest_rows.append(
                {
                    "part_name": part_name,
                    "sample_id": sample["sample_id"],
                    "object_id": int(sample["object_id"]),
                    "part_id": int(sample["part_id"]),
                    "mask_path": str(mask_path),
                    "overlay_path": overlay_path,
                    "image_path": str(sample["image_path"]),
                    "mat_path": str(sample["mat_path"]),
                }
            )

    manifest_csv_path = output_dir / "manifest.csv"
    with manifest_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "part_name",
                "sample_id",
                "object_id",
                "part_id",
                "mask_path",
                "overlay_path",
                "image_path",
                "mat_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "voc_images_dir": str(voc_images_dir),
                "split_file": str(split_file),
                "parts": summary,
                "total_masks": len(manifest_rows),
                "manifest_csv": str(manifest_csv_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(output_dir), "parts": summary, "total_masks": len(manifest_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
