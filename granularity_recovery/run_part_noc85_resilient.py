from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

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
    parser.add_argument("--output-dir", type=str, default="outputs/head_resilient_eval")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--config-path", type=str, default="config.yml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--voc-images-dir", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--seed-skip-sample-ids", type=str, default="2008_002429")
    parser.add_argument("--seed-result-globs", type=str,
                        default="outputs/part_head_all5_chunk_*.json,outputs/part_head_all5_skip429_chunk_*.json",
                        help="Comma-separated JSON glob patterns whose successful samples should be treated as already done.")
    return parser.parse_args()


def load_done_and_skipped(output_dir: Path, seed_globs: List[str]) -> tuple[Set[str], Set[str]]:
    done_keys: Set[str] = set()
    skipped_keys: Set[str] = set()

    candidate_paths: Set[Path] = set(output_dir.glob("chunk_*.json"))
    candidate_paths.update(output_dir.glob("single_*.json"))
    for pattern in seed_globs:
        pattern = pattern.strip()
        if not pattern:
            continue
        candidate_paths.update(REPO_ROOT.glob(pattern))

    for path in sorted(candidate_paths):
        if path.name.endswith(".samples.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for sample in data.get("samples", []):
            done_keys.add(sample.get("sample_key") or normalize_sample_key(sample))
        for sample in data.get("skipped", []):
            skipped_keys.add(sample.get("sample_key") or normalize_sample_key(sample))

    skipped_path = output_dir / "skipped_keys.json"
    if skipped_path.exists():
        payload = json.loads(skipped_path.read_text(encoding="utf-8"))
        skipped_keys.update(str(item) for item in payload)

    return done_keys, skipped_keys


def save_skipped(output_dir: Path, skipped_keys: Set[str]) -> None:
    path = output_dir / "skipped_keys.json"
    path.write_text(json.dumps(sorted(skipped_keys), ensure_ascii=False, indent=2), encoding="utf-8")


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


def main():
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_globs = [item.strip() for item in args.seed_result_globs.split(",") if item.strip()]

    dataset_root, voc_images_dir, split_file = resolve_dataset_paths(args)
    all_samples = load_part_samples(dataset_root, voc_images_dir, split_file, args.part_name)

    done_keys, skipped_keys = load_done_and_skipped(output_dir, seed_globs)
    seed_skip_ids = {item.strip() for item in args.seed_skip_sample_ids.split(",") if item.strip()}
    if seed_skip_ids:
        for sample in all_samples:
            if sample["sample_id"] in seed_skip_ids:
                skipped_keys.add(normalize_sample_key(sample))

    remaining = [sample for sample in all_samples if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys]
    print(f"[resume] part={args.part_name} total={len(all_samples)} done={len(done_keys)} skipped={len(skipped_keys)} "
          f"remaining={len(remaining)} oracle_granularity={args.oracle_granularity}", flush=True)

    chunk_index = 0
    while remaining:
        chunk_samples = remaining[:args.chunk_size]
        chunk_tag = f"{chunk_index:04d}"
        keys_path = output_dir / f"chunk_{chunk_tag}.samples.json"
        json_path = output_dir / f"chunk_{chunk_tag}.json"
        log_path = output_dir / f"chunk_{chunk_tag}.log"
        err_path = output_dir / f"chunk_{chunk_tag}.err.log"

        write_sample_keys_file(keys_path, chunk_samples)
        returncode = run_eval(args, keys_path, json_path, log_path, err_path)
        if returncode == 0 and json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for sample in data.get("samples", []):
                done_keys.add(sample.get("sample_key") or normalize_sample_key(sample))
            for sample in data.get("skipped", []):
                skipped_keys.add(sample.get("sample_key") or normalize_sample_key(sample))
            save_skipped(output_dir, skipped_keys)
            remaining = [sample for sample in all_samples if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys]
            print(f"[chunk ok] idx={chunk_index} done={len(done_keys)} skipped={len(skipped_keys)} remaining={len(remaining)}", flush=True)
            chunk_index += 1
            continue

        print(f"[chunk fail] idx={chunk_index} rc={returncode} falling back to singles", flush=True)
        for sample in chunk_samples:
            sample_key = normalize_sample_key(sample)
            if sample_key in done_keys or sample_key in skipped_keys:
                continue
            single_keys_path = output_dir / f"single_{sample_key.replace(':', '_')}.samples.json"
            single_json_path = output_dir / f"single_{sample_key.replace(':', '_')}.json"
            single_log_path = output_dir / f"single_{sample_key.replace(':', '_')}.log"
            single_err_path = output_dir / f"single_{sample_key.replace(':', '_')}.err.log"
            write_sample_keys_file(single_keys_path, [sample])
            single_rc = run_eval(args, single_keys_path, single_json_path, single_log_path, single_err_path)
            if single_rc == 0 and single_json_path.exists():
                data = json.loads(single_json_path.read_text(encoding="utf-8"))
                for item in data.get("samples", []):
                    done_keys.add(item.get("sample_key") or normalize_sample_key(item))
                print(f"[single ok] {sample_key}", flush=True)
            else:
                skipped_keys.add(sample_key)
                save_skipped(output_dir, skipped_keys)
                print(f"[single skip] {sample_key} rc={single_rc}", flush=True)

        remaining = [sample for sample in all_samples if normalize_sample_key(sample) not in done_keys and normalize_sample_key(sample) not in skipped_keys]
        print(f"[after fallback] done={len(done_keys)} skipped={len(skipped_keys)} remaining={len(remaining)}", flush=True)
        chunk_index += 1

    save_skipped(output_dir, skipped_keys)
    print(f"[complete] part={args.part_name} done={len(done_keys)} skipped={len(skipped_keys)} total={len(all_samples)}", flush=True)


if __name__ == "__main__":
    main()
