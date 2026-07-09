from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granularity_recovery.evaluate_part_noc85 import (
    load_part_samples,
    normalize_sample_key,
    resolve_dataset_paths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-granularity-checkpoint", type=str, required=True)
    parser.add_argument("--part-name", type=str, default="head")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--first-click-granularity", type=float, default=0.5)
    parser.add_argument("--oracle-granularity", action="store_true", default=False)
    parser.add_argument("--oracle-granularity-values", type=str, default="0.00,0.08,0.18,0.32,0.50,0.68,0.82,0.92,1.00")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--config-path", type=str, default="config.yml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--voc-images-dir", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--existing-output-dir", type=str, required=True)
    parser.add_argument("--repair-output-dir", type=str, required=True)
    parser.add_argument("--save-summary-json", type=str, default=None)
    return parser.parse_args()


def _iter_result_paths(output_dir: Path) -> Iterable[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if not path.name.endswith(".samples.json") and path.name != "skipped_keys.json"
    )


def load_done_and_skipped(output_dir: Path) -> tuple[Set[str], Set[str]]:
    done_keys: Set[str] = set()
    skipped_keys: Set[str] = set()
    for path in _iter_result_paths(output_dir):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for item in data.get("samples", []):
            done_keys.add(item.get("sample_key") or normalize_sample_key(item))
        for item in data.get("skipped", []):
            skipped_keys.add(item.get("sample_key") or normalize_sample_key(item))
    skipped_path = output_dir / "skipped_keys.json"
    if skipped_path.exists():
        payload = json.loads(skipped_path.read_text(encoding="utf-8"))
        skipped_keys.update(str(item) for item in payload)
    return done_keys, skipped_keys


def write_sample_keys_file(path: Path, samples: List[Dict[str, object]]) -> None:
    payload = [
        {
            "sample_id": sample["sample_id"],
            "object_id": int(sample["object_id"]),
            "part_id": int(sample["part_id"]),
            "part_name": sample["part_name"],
        }
        for sample in samples
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_eval(args, sample_keys_file: Path, json_path: Path, log_path: Path, err_path: Path) -> int:
    cmd = [
        args.python_exe,
        "-u",
        "-m",
        "granularity_recovery.evaluate_part_noc85",
        "--base-granularity-checkpoint",
        args.base_granularity_checkpoint,
        "--part-name",
        args.part_name,
        "--sample-keys-file",
        str(sample_keys_file),
        "--max-clicks",
        str(args.max_clicks),
        "--first-click-granularity",
        str(args.first_click_granularity),
        "--oracle-granularity-values",
        str(args.oracle_granularity_values),
        "--save-json",
        str(json_path),
        "--config-path",
        args.config_path,
    ]
    if args.oracle_granularity:
        cmd.append("--oracle-granularity")
    if args.dataset_root:
        cmd.extend(["--dataset-root", args.dataset_root])
    if args.voc_images_dir:
        cmd.extend(["--voc-images-dir", args.voc_images_dir])
    if args.split_file:
        cmd.extend(["--split-file", args.split_file])

    with open(log_path, "w", encoding="utf-8") as stdout_f, open(err_path, "w", encoding="utf-8") as stderr_f:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
        )
    return int(completed.returncode)


def aggregate_results(output_dirs: List[Path], total_expected: int) -> Dict[str, object]:
    seen: Dict[str, Dict[str, object]] = {}
    skipped: Set[str] = set()
    for output_dir in output_dirs:
        for path in _iter_result_paths(output_dir):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for item in data.get("samples", []):
                key = item.get("sample_key") or normalize_sample_key(item)
                seen[key] = item
            for item in data.get("skipped", []):
                skipped.add(item.get("sample_key") or normalize_sample_key(item))
        skipped_path = output_dir / "skipped_keys.json"
        if skipped_path.exists():
            payload = json.loads(skipped_path.read_text(encoding="utf-8"))
            skipped.update(str(item) for item in payload)

    threshold = 0.85
    max_clicks = 20
    nocs: List[int] = []
    ious1: List[float] = []
    failures = 0
    for sample in seen.values():
        ious = [float(x) for x in sample.get("ious", [])]
        if ious:
            ious1.append(ious[0])
        reached = None
        for idx, iou in enumerate(ious, start=1):
            if iou >= threshold:
                reached = idx
                break
        if reached is None:
            reached = max_clicks
            failures += 1
        nocs.append(reached)

    return {
        "done": len(seen),
        "skipped": len(skipped),
        "remaining": max(0, total_expected - len(seen) - len(skipped)),
        "noc85_mean": (sum(nocs) / len(nocs)) if nocs else 0.0,
        "iou1_mean": (sum(ious1) / len(ious1)) if ious1 else 0.0,
        "failures_at_20": failures,
    }


def main():
    args = parse_args()
    existing_output_dir = (REPO_ROOT / args.existing_output_dir).resolve()
    repair_output_dir = (REPO_ROOT / args.repair_output_dir).resolve()
    repair_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, voc_images_dir, split_file = resolve_dataset_paths(args)
    all_samples = load_part_samples(dataset_root, voc_images_dir, split_file, args.part_name)
    total_expected = len(all_samples)

    existing_done, existing_skipped = load_done_and_skipped(existing_output_dir)
    repair_done, repair_skipped = load_done_and_skipped(repair_output_dir)

    done_keys = existing_done | repair_done
    skipped_keys = existing_skipped | repair_skipped
    missing_samples = [
        sample
        for sample in all_samples
        if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys
    ]
    print(
        f"[repair] part={args.part_name} total={total_expected} "
        f"existing_done={len(existing_done)} existing_skipped={len(existing_skipped)} "
        f"repair_done={len(repair_done)} repair_skipped={len(repair_skipped)} "
        f"missing={len(missing_samples)} oracle_granularity={args.oracle_granularity}",
        flush=True,
    )

    chunk_index = 0
    while missing_samples:
        chunk_samples = missing_samples[: args.chunk_size]
        chunk_tag = f"{chunk_index:04d}"
        keys_path = repair_output_dir / f"repair_chunk_{chunk_tag}.samples.json"
        json_path = repair_output_dir / f"repair_chunk_{chunk_tag}.json"
        log_path = repair_output_dir / f"repair_chunk_{chunk_tag}.log"
        err_path = repair_output_dir / f"repair_chunk_{chunk_tag}.err.log"

        write_sample_keys_file(keys_path, chunk_samples)
        returncode = run_eval(args, keys_path, json_path, log_path, err_path)
        if returncode == 0 and json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for item in data.get("samples", []):
                repair_done.add(item.get("sample_key") or normalize_sample_key(item))
            for item in data.get("skipped", []):
                repair_skipped.add(item.get("sample_key") or normalize_sample_key(item))
            (repair_output_dir / "skipped_keys.json").write_text(
                json.dumps(sorted(repair_skipped), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            done_keys = existing_done | repair_done
            skipped_keys = existing_skipped | repair_skipped
            missing_samples = [
                sample
                for sample in all_samples
                if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys
            ]
            print(
                f"[repair chunk ok] idx={chunk_index} "
                f"repair_done={len(repair_done)} repair_skipped={len(repair_skipped)} "
                f"missing={len(missing_samples)}",
                flush=True,
            )
            chunk_index += 1
            continue

        print(f"[repair chunk fail] idx={chunk_index} rc={returncode} falling back to singles", flush=True)
        for sample in chunk_samples:
            sample_key = normalize_sample_key(sample)
            if sample_key in repair_done or sample_key in repair_skipped or sample_key in existing_done or sample_key in existing_skipped:
                continue
            single_keys_path = repair_output_dir / f"repair_single_{sample_key.replace(':', '_')}.samples.json"
            single_json_path = repair_output_dir / f"repair_single_{sample_key.replace(':', '_')}.json"
            single_log_path = repair_output_dir / f"repair_single_{sample_key.replace(':', '_')}.log"
            single_err_path = repair_output_dir / f"repair_single_{sample_key.replace(':', '_')}.err.log"
            write_sample_keys_file(single_keys_path, [sample])
            single_rc = run_eval(args, single_keys_path, single_json_path, single_log_path, single_err_path)
            if single_rc == 0 and single_json_path.exists():
                data = json.loads(single_json_path.read_text(encoding="utf-8"))
                for item in data.get("samples", []):
                    repair_done.add(item.get("sample_key") or normalize_sample_key(item))
                print(f"[repair single ok] {sample_key}", flush=True)
            else:
                repair_skipped.add(sample_key)
                print(f"[repair single skip] {sample_key} rc={single_rc}", flush=True)
            (repair_output_dir / "skipped_keys.json").write_text(
                json.dumps(sorted(repair_skipped), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        done_keys = existing_done | repair_done
        skipped_keys = existing_skipped | repair_skipped
        missing_samples = [
            sample
            for sample in all_samples
            if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys
        ]
        print(
            f"[repair after fallback] repair_done={len(repair_done)} repair_skipped={len(repair_skipped)} "
            f"missing={len(missing_samples)}",
            flush=True,
        )
        chunk_index += 1

    summary = aggregate_results([existing_output_dir, repair_output_dir], total_expected)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.save_summary_json:
        summary_path = (REPO_ROOT / args.save_summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
