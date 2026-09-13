from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.analyze_semantic_trajectories import (
    SUMMARY_METRICS,
    paired_bootstrap_differences,
)
from evaluate.semantic_coverage import rescore_semantic_record


DATASETS = ("SPIN", "PascalPart", "PartImageNet")
FULL_ROOT = Path(
    "outputs/01_evaluations/hierarchical_semantics_sgr_vitb_direct11"
)
FROZEN_ROOT = Path("outputs/02_ablations/semantic_path_frozen_vitb")
OUTPUT_ROOT = Path("outputs/02_ablations/semantic_path_frozen_vitb")


def mean_metrics(records: list[dict[str, object]]) -> dict[str, float]:
    return {
        name: sum(float(record[name]) for record in records) / len(records)
        for name in SUMMARY_METRICS
    }


def main() -> None:
    rows = []
    for dataset in DATASETS:
        full_payload = json.loads(
            (FULL_ROOT / dataset / "semantic_coverage.json").read_text(
                encoding="utf-8"
            )
        )
        frozen_payload = json.loads(
            (FROZEN_ROOT / dataset / "semantic_coverage.json").read_text(
                encoding="utf-8"
            )
        )
        if [
            row["sample_key"] for row in full_payload["records"]
        ] != [
            row["sample_key"] for row in frozen_payload["records"]
        ]:
            raise ValueError(f"sample mismatch for {dataset}")

        full_records = [
            rescore_semantic_record(record, tuple(range(11)))
            for record in full_payload["records"]
        ]
        frozen_records = [
            rescore_semantic_record(record, tuple(range(11)))
            for record in frozen_payload["records"]
        ]
        full_summary = mean_metrics(full_records)
        frozen_summary = mean_metrics(frozen_records)
        differences = paired_bootstrap_differences(
            full_records,
            frozen_records,
            repetitions=10000,
            seed=20260730,
        )
        for metric in SUMMARY_METRICS:
            interval = differences[metric]
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "frozen_hierarchy": frozen_summary[metric],
                    "full_sgr_click": full_summary[metric],
                    "full_minus_frozen": interval["estimate"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "paired_summary.json"
    csv_path = OUTPUT_ROOT / "paired_summary.csv"
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "samples_per_dataset": 1000,
                    "controls": [index / 10 for index in range(11)],
                    "comparison": "full SGR-Click minus frozen hierarchy",
                    "bootstrap_repetitions": 10000,
                    "bootstrap_seed": 20260730,
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
