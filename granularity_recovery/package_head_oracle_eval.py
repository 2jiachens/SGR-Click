from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [
    REPO_ROOT / "outputs" / "head_oracle_eval",
    REPO_ROOT / "outputs" / "head_oracle_eval_repair",
]
PACKAGE_DIR = REPO_ROOT / "outputs" / "packages" / "head_oracle_eval_validation_20260705"


def iter_result_jsons(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.json")
        if not path.name.endswith(".samples.json") and path.name != "skipped_keys.json"
    )


def normalize_sample_key(item: Dict[str, object]) -> str:
    return item.get("sample_key") or "{}:{}:{}".format(
        item.get("sample_id"),
        item.get("object_id"),
        item.get("part_id"),
    )


def load_all_samples() -> tuple[Dict[str, Dict[str, object]], Set[str], List[Dict[str, object]]]:
    samples: Dict[str, Dict[str, object]] = {}
    skipped: Set[str] = set()
    source_manifest: List[Dict[str, object]] = []

    for folder in SOURCE_DIRS:
        folder_records = 0
        folder_samples = 0
        folder_skipped = 0
        for json_path in iter_result_jsons(folder):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
                continue
            folder_records += 1
            for item in data.get("samples", []):
                key = normalize_sample_key(item)
                samples[key] = {**item, "_source_json": str(json_path)}
                folder_samples += 1
            skipped_items = data.get("skipped", [])
            if not isinstance(skipped_items, list):
                skipped_items = []
            for item in skipped_items:
                key = normalize_sample_key(item)
                skipped.add(key)
                folder_skipped += 1
        skipped_path = folder / "skipped_keys.json"
        if skipped_path.exists():
            payload = json.loads(skipped_path.read_text(encoding="utf-8"))
            for item in payload:
                skipped.add(str(item))
        source_manifest.append(
            {
                "folder": str(folder),
                "json_record_files": folder_records,
                "sample_rows_read": folder_samples,
                "skipped_rows_read": folder_skipped,
            }
        )

    skipped -= set(samples.keys())
    return samples, skipped, source_manifest


def compute_metrics(samples: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    threshold = 0.85
    max_clicks = 20
    failures = 0
    noc_sum = 0.0
    iou1_sum = 0.0

    for sample in samples.values():
        ious = [float(x) for x in sample.get("ious", [])]
        first_iou = ious[0] if ious else 0.0
        iou1_sum += first_iou
        reached = None
        for idx, iou in enumerate(ious, start=1):
            if iou >= threshold:
                reached = idx
                break
        if reached is None:
            reached = max_clicks
            failures += 1
        noc_sum += reached

    count = len(samples)
    return {
        "evaluated_samples": count,
        "skipped_samples": None,
        "noc85_mean": noc_sum / count if count else 0.0,
        "iou1_mean": iou1_sum / count if count else 0.0,
        "failures_at_20": failures,
        "iou_threshold": threshold,
        "max_clicks": max_clicks,
    }


def build_sample_rows(samples: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    threshold = 0.85
    max_clicks = 20
    for key, sample in sorted(samples.items()):
        ious = [float(x) for x in sample.get("ious", [])]
        reached = None
        for idx, iou in enumerate(ious, start=1):
            if iou >= threshold:
                reached = idx
                break
        if reached is None:
            reached = max_clicks
        rows.append(
            {
                "sample_key": key,
                "sample_id": str(sample.get("sample_id", "")),
                "object_id": int(sample.get("object_id", -1)),
                "part_id": int(sample.get("part_id", -1)),
                "part_name": str(sample.get("part_name", "")),
                "clicks_recorded": len(ious),
                "iou_at_1": ious[0] if ious else 0.0,
                "iou_at_last": ious[-1] if ious else 0.0,
                "noc85": reached,
                "reached_085": bool(any(iou >= threshold for iou in ious)),
                "selected_granularity": float(sample.get("selected_granularity", 0.0)),
                "source_json": str(sample.get("_source_json", "")),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples, skipped, source_manifest = load_all_samples()
    metrics = compute_metrics(samples)
    metrics["skipped_samples"] = len(skipped)
    metrics["remaining_samples"] = 0
    metrics["target_total_samples"] = len(samples) + len(skipped)
    metrics["package_dir"] = str(PACKAGE_DIR)

    sample_rows = build_sample_rows(samples)
    skipped_rows = [{"sample_key": key} for key in sorted(skipped)]

    summary_path = PACKAGE_DIR / "final_summary.json"
    samples_csv_path = PACKAGE_DIR / "sample_metrics.csv"
    skipped_csv_path = PACKAGE_DIR / "skipped_samples.csv"
    manifest_path = PACKAGE_DIR / "source_manifest.json"

    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source_dirs": [str(path) for path in SOURCE_DIRS],
                "source_manifest": source_manifest,
                "generated_files": [
                    str(summary_path),
                    str(samples_csv_path),
                    str(skipped_csv_path),
                    str(manifest_path),
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(
        samples_csv_path,
        sample_rows,
        [
            "sample_key",
            "sample_id",
            "object_id",
            "part_id",
            "part_name",
            "clicks_recorded",
            "iou_at_1",
            "iou_at_last",
            "noc85",
            "reached_085",
            "selected_granularity",
            "source_json",
        ],
    )
    write_csv(skipped_csv_path, skipped_rows, ["sample_key"])

    print(json.dumps({"summary": str(summary_path), "samples": len(sample_rows), "skipped": len(skipped_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
