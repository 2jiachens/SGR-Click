from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.semantic_coverage import rescore_semantic_record


DATASETS = ("PascalPart", "PartImageNet", "SPIN")
METRICS = ("lr60", "hsr60", "lr70", "hsr70", "m_biou", "semantic_order_rate")
ABLATION_MODES = (
    ("full", "Complete"),
    ("frozen_hierarchy", "w/o boundary projection"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/05_revision_round3"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/01_evaluations"),
    )
    parser.add_argument(
        "--flops-summary",
        type=Path,
        default=Path("outputs/flops_sgr_vs_unigraco/flops_summary.csv"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(
    values: Sequence[float],
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def semantic_rows(payload: dict[str, object]) -> dict[str, dict[str, float]]:
    return {
        str(record["sample_key"]): {
            name: float(value)
            for name, value in rescore_semantic_record(
                record,
                range(len(record["prediction_areas"])),
            ).items()
        }
        for record in payload["records"]
    }


def sam_rows(
    payload: dict[str, object],
    mapping_name: str,
) -> dict[str, dict[str, float]]:
    result = {}
    for record in payload["records"]:
        mapping = record["mappings"][mapping_name]
        result[str(record["sample_key"])] = {
            "lr60": float(mapping["lr60"]),
            "hsr60": float(mapping["hsr60"]),
            "lr70": float(mapping["lr70"]),
            "hsr70": float(mapping["hsr70"]),
            "m_biou": float(mapping["m_biou"]),
            "semantic_order_rate": float(mapping["sor"]),
        }
    return result


def summarize_rows(
    rows: dict[str, dict[str, float]],
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    summary: dict[str, object] = {"samples": len(rows)}
    for offset, metric in enumerate(METRICS):
        estimate, low, high = bootstrap_interval(
            [row[metric] for row in rows.values()],
            repetitions,
            seed + offset,
        )
        summary[metric] = estimate
        summary[f"{metric}_ci95_low"] = low
        summary[f"{metric}_ci95_high"] = high
    return summary


def paired_differences(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if set(first) != set(second):
        missing_first = sorted(set(second) - set(first))[:5]
        missing_second = sorted(set(first) - set(second))[:5]
        raise ValueError(
            "paired sample keys differ; "
            f"missing_first={missing_first}, missing_second={missing_second}"
        )
    keys = sorted(first)
    result: dict[str, object] = {"samples": len(keys)}
    for offset, metric in enumerate(METRICS):
        differences = [first[key][metric] - second[key][metric] for key in keys]
        estimate, low, high = bootstrap_interval(
            differences,
            repetitions,
            seed + offset,
        )
        result[metric] = estimate
        result[f"{metric}_ci95_low"] = low
        result[f"{metric}_ci95_high"] = high
    return result


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_b1(args: argparse.Namespace):
    table_rows = []
    difference_rows = []
    sam_by_dataset = {}
    sam_datasets = ["PascalPart", "PartImageNet"]
    optional_spin = (
        args.results_root
        / "B1_SAM_multimask"
        / "SPIN"
        / "semantic_coverage.json"
    )
    if optional_spin.exists():
        sam_datasets.append("SPIN")
    for dataset in sam_datasets:
        sam_payload = load_json(
            args.results_root
            / "B1_SAM_multimask"
            / dataset
            / "semantic_coverage.json"
        )
        sam_by_dataset[dataset] = sam_rows(sam_payload, "direct3")
        for mapping in ("direct3", "area_expanded11"):
            current = sam_rows(sam_payload, mapping)
            table_rows.append(
                {
                    "method": "SAM",
                    "backbone": "ViT-L",
                    "mapping": mapping,
                    "dataset": dataset,
                    **summarize_rows(
                        current,
                        args.bootstrap_repetitions,
                        args.bootstrap_seed,
                    ),
                }
            )

    for backbone, run_name in (
        ("ViT-B", "hierarchical_semantics_sgr_vitb_direct11"),
        ("ViT-L", "hierarchical_semantics_sgr_vitl_direct11"),
    ):
        for dataset in sam_datasets:
            sgr = semantic_rows(
                load_json(
                    args.baseline_root
                    / run_name
                    / dataset
                    / "semantic_coverage.json"
                )
            )
            table_rows.append(
                {
                    "method": "SGR-Click",
                    "backbone": backbone,
                    "mapping": "direct11",
                    "dataset": dataset,
                    **summarize_rows(
                        sgr,
                        args.bootstrap_repetitions,
                        args.bootstrap_seed,
                    ),
                }
            )
            difference_rows.append(
                {
                    "comparison": f"SGR-Click {backbone} minus SAM ViT-L",
                    "dataset": dataset,
                    **paired_differences(
                        sgr,
                        sam_by_dataset[dataset],
                        args.bootstrap_repetitions,
                        args.bootstrap_seed,
                    ),
                }
            )
    return table_rows, difference_rows


def build_b2(args: argparse.Namespace):
    table_rows = []
    difference_rows = []
    for dataset in DATASETS:
        by_mode = {}
        for mode, label in ABLATION_MODES:
            current = semantic_rows(
                load_json(
                    args.results_root
                    / "B2_component_ablation"
                    / dataset
                    / mode
                    / "semantic_coverage.json"
                )
            )
            by_mode[mode] = current
            table_rows.append(
                {
                    "row": label,
                    "mode": mode,
                    "dataset": dataset,
                    **summarize_rows(
                        current,
                        args.bootstrap_repetitions,
                        args.bootstrap_seed,
                    ),
                }
            )
        for mode, label in ABLATION_MODES[1:]:
            difference_rows.append(
                {
                    "comparison": f"Complete minus {label}",
                    "dataset": dataset,
                    **paired_differences(
                        by_mode["full"],
                        by_mode[mode],
                        args.bootstrap_repetitions,
                        args.bootstrap_seed,
                    ),
                }
            )
    return table_rows, difference_rows


def build_b3(args: argparse.Namespace):
    rows = []
    flops_by_backbone: dict[str, dict[str, dict[str, float]]] = {}
    with args.flops_summary.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            flops_by_backbone.setdefault(row["backbone"], {})[row["scenario"]] = {
                "SGR-Click": float(row["sgr_click_gflops"]),
                "UniGraCo": float(row["unigraco_gflops"]),
            }
    paths = (
        "sgr_vitb.json",
        "sgr_vitl.json",
        "unigraco_vitb.json",
        "unigraco_vitl.json",
    )
    sample_keys = []
    for name in paths:
        payload = load_json(args.results_root / "B3_cost_profile" / name)
        method = payload["protocol"]["method"]
        backbone = payload["protocol"]["backbone"]
        method_flops = flops_by_backbone[backbone]
        sample_keys.append(tuple(payload["sample_keys"]))
        rows.append(
            {
                "method": method,
                "backbone": backbone,
                "samples": payload["first_click_latency_ms"]["count"],
                "first_click_median_ms": payload["first_click_latency_ms"]["median"],
                "first_click_p95_ms": payload["first_click_latency_ms"]["p95"],
                "slider_move_median_ms": payload["slider_move_latency_ms"]["median"],
                "slider_move_p95_ms": payload["slider_move_latency_ms"]["p95"],
                "first_click_peak_gpu_memory_gib": payload["peak_gpu_memory_gib"]["first_click_max"],
                "slider_move_peak_gpu_memory_gib": payload["peak_gpu_memory_gib"]["slider_move_max"],
                "frozen_parameters": payload["parameters"]["frozen"],
                "trainable_parameters": payload["parameters"]["trainable"],
                "total_parameters": payload["parameters"]["total"],
                "first_state_gflops": method_flops["First state"][method],
                "one_slider_update_gflops": method_flops["One slider update"][method],
                "ten_state_sweep_gflops": method_flops["10-state sweep"][method],
                "twenty_one_state_sweep_gflops": method_flops["21-state sweep"][method],
                "gpu": payload["protocol"]["gpu"],
            }
        )
    if len(set(sample_keys)) != 1:
        raise ValueError("B3 rows were not measured on identical samples")
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.results_root / "summary"
    b1_rows, b1_differences = build_b1(args)
    b2_rows, b2_differences = build_b2(args)
    b3_rows = build_b3(args)
    write_csv(output_dir / "b1_sam_comparison.csv", b1_rows)
    write_csv(output_dir / "b1_paired_differences.csv", b1_differences)
    write_csv(output_dir / "b2_component_ablation.csv", b2_rows)
    write_csv(output_dir / "b2_paired_differences.csv", b2_differences)
    write_csv(output_dir / "b3_cost_profile.csv", b3_rows)
    summary = {
        "protocol": {
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
            "paired_differences": "first method minus second method on identical sample keys",
        },
        "B1": {"rows": b1_rows, "paired_differences": b1_differences},
        "B2": {"rows": b2_rows, "paired_differences": b2_differences},
        "B3": {"rows": b3_rows},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "revision_round3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
