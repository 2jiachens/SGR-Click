from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.semantic_coverage import (
    rescore_semantic_record,
    summarize_semantic_trajectory_records,
)


RUNS = {
    ("SGR-Click", "ViT-B"): (
        "hierarchical_semantics_sgr_vitb_direct11",
        tuple(range(11)),
    ),
    ("SGR-Click", "ViT-L"): (
        "hierarchical_semantics_sgr_vitl_direct11",
        tuple(range(11)),
    ),
    ("UniGraCo", "ViT-B"): (
        "hierarchical_semantics_unigraco_vitb_exact_5server",
        tuple(range(0, 21, 2)),
    ),
    ("UniGraCo", "ViT-L"): (
        "hierarchical_semantics_unigraco_vitl_exact_5server",
        tuple(range(0, 21, 2)),
    ),
}
VISUALIZATION_RUNS = {
    ("SGR-Click", "ViT-B"): "hierarchical_semantics_sgr_vitb_exact_5server",
    ("SGR-Click", "ViT-L"): "hierarchical_semantics_sgr_vitl_exact_5server",
    ("UniGraCo", "ViT-B"): "hierarchical_semantics_unigraco_vitb_exact_5server",
    ("UniGraCo", "ViT-L"): "hierarchical_semantics_unigraco_vitl_exact_5server",
}
DATASETS = ("SPIN", "PascalPart", "PartImageNet")
SHARED_STATE_INDICES = tuple(range(0, 21, 2))
SUMMARY_METRICS = (
    "lr60",
    "hsr60",
    "lr70",
    "hsr70",
    "m_biou",
    "semantic_order_rate",
    "backward_area_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations-root",
        type=Path,
        default=Path("outputs/01_evaluations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/04_trajectory_diagnostics"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def bootstrap_summary(
    sample_metrics: list[dict[str, float | int]],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    values = {
        name: np.asarray([float(row[name]) for row in sample_metrics])
        for name in SUMMARY_METRICS
    }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(sample_metrics),
        size=(int(repetitions), len(sample_metrics)),
    )
    result = {}
    for name, metric_values in values.items():
        bootstrap_means = metric_values[sampled].mean(axis=1)
        result[name] = {
            "estimate": float(metric_values.mean()),
            "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        }
    return result


def paired_bootstrap_differences(
    first: list[dict[str, float | int]],
    second: list[dict[str, float | int]],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if len(first) != len(second):
        raise ValueError("paired runs must contain the same number of samples")
    differences = {
        name: np.asarray(
            [
                float(first_row[name]) - float(second_row[name])
                for first_row, second_row in zip(first, second)
            ]
        )
        for name in SUMMARY_METRICS
    }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(first), size=(int(repetitions), len(first)))
    result = {}
    for name, values in differences.items():
        bootstrap_means = values[sampled].mean(axis=1)
        result[name] = {
            "estimate": float(values.mean()),
            "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        }
    return result


def main() -> None:
    args = parse_args()
    rows = []
    sample_rows = {}
    for (method, backbone), (run_name, state_indices) in RUNS.items():
        for dataset in DATASETS:
            source_path = (
                args.evaluations_root
                / run_name
                / dataset
                / "semantic_coverage.json"
            )
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            visualization_payload = json.loads(
                (
                    args.evaluations_root
                    / VISUALIZATION_RUNS[(method, backbone)]
                    / dataset
                    / "semantic_coverage.json"
                ).read_text(encoding="utf-8")
            )
            visualization_summary = summarize_semantic_trajectory_records(
                visualization_payload["records"]
            )
            rescored = [
                rescore_semantic_record(record, state_indices)
                for record in payload["records"]
            ]
            bootstrap = bootstrap_summary(
                rescored,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            )
            summary = {
                name: bootstrap[name]["estimate"]
                for name in SUMMARY_METRICS
            }
            sample_rows[(method, backbone, dataset)] = rescored
            rows.append(
                {
                    "method": method,
                    "backbone": backbone,
                    "dataset": dataset,
                    "source": str(source_path),
                    "samples": len(rescored),
                    **summary,
                    "bootstrap": bootstrap,
                    "full_21_state_observable_state_count":
                        visualization_summary["observable_state_count"],
                }
            )

    paired_differences = []
    for backbone in ("ViT-B", "ViT-L"):
        for dataset in DATASETS:
            paired_differences.append(
                {
                    "comparison": "SGR-Click minus UniGraCo",
                    "backbone": backbone,
                    "dataset": dataset,
                    "bootstrap": paired_bootstrap_differences(
                        sample_rows[("SGR-Click", backbone, dataset)],
                        sample_rows[("UniGraCo", backbone, dataset)],
                        repetitions=args.bootstrap_repetitions,
                        seed=args.bootstrap_seed,
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "semantic_trajectory_diagnostics.json"
    csv_path = args.output_dir / "semantic_trajectory_diagnostics.csv"
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "quantitative_controls": [
                        index / 20 for index in SHARED_STATE_INDICES
                    ],
                    "visualization_controls": 21,
                    "bootstrap_repetitions": args.bootstrap_repetitions,
                    "bootstrap_seed": args.bootstrap_seed,
                    "semantic_order_rate":
                        "strictly increasing best-response indices from fine to coarse",
                    "backward_area_rate":
                        "mean adjacent positive area decrease divided by previous area",
                    "observable_state":
                        "unique (area, annotated-target IoU vector) signature; "
                        "a lower bound on distinct binary masks",
                },
                "rows": rows,
                "paired_differences": paired_differences,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        flat_rows = []
        for row in rows:
            flat = {
                key: value
                for key, value in row.items()
                if key != "bootstrap"
            }
            for metric, interval in row["bootstrap"].items():
                flat[f"{metric}_ci95_low"] = interval["ci95_low"]
                flat[f"{metric}_ci95_high"] = interval["ci95_high"]
            flat_rows.append(flat)
        writer = csv.DictWriter(stream, fieldnames=tuple(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
