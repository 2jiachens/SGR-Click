from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one or more slider-evaluation result directories and compute "
            "per-image best NoC plus max IoU@1, matching the server reporting rule."
        )
    )
    parser.add_argument("result_dirs", nargs="+", help="Target directories containing g*.json files.")
    parser.add_argument("--thresholds", default="0.85,0.90", help="Comma-separated IoU thresholds for NoC.")
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def threshold_key(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def realized_noc(ious: Iterable[float], threshold: float, max_clicks: int) -> int:
    for index, iou in enumerate(ious, start=1):
        if float(iou) >= threshold:
            return index
    return max_clicks


def load_records(result_dirs: List[Path]) -> tuple[Dict[str, Dict[str, list]], List[str]]:
    by_sample: Dict[str, Dict[str, list]] = {}
    sliders: List[str] = []
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("g*.json")):
            slider = path.stem
            if slider not in sliders:
                sliders.append(slider)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("samples", []):
                ious = [float(value) for value in record.get("ious", [])]
                if not ious:
                    continue
                sample_id = str(record.get("sample_id"))
                item = by_sample.setdefault(sample_id, {"ious": []})
                item["ious"].append(ious)
    return by_sample, sorted(sliders)


def summarize(result_dirs: List[Path], thresholds: List[float], max_clicks: int) -> dict:
    by_sample, sliders = load_records(result_dirs)
    summary = {
        "result_dirs": [str(path) for path in result_dirs],
        "sliders": sliders,
        "samples": len(by_sample),
    }
    if not by_sample:
        for threshold in thresholds:
            summary[f"noc{threshold_key(threshold)}"] = None
        summary["iou_at_1"] = None
        return summary

    for threshold in thresholds:
        key = threshold_key(threshold)
        summary[f"noc{key}"] = mean(
            min(realized_noc(ious, threshold, max_clicks) for ious in item["ious"])
            for item in by_sample.values()
        )
    summary["iou_at_1"] = mean(
        max((ious[0] if ious else 0.0) for ious in item["ious"])
        for item in by_sample.values()
    )
    return summary


def main() -> None:
    args = parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    result = summarize([Path(path) for path in args.result_dirs], thresholds, args.max_clicks)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
