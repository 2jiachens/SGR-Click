import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REQUIRED_NPZ_KEYS = {
    "gt_mask",
    "simple_prob",
    "heuristic_masks",
    "corrected_probs",
    "corrective_clicks",
    "positive_click_map",
    "effective_level_mask",
    "effective_level_indices",
    "num_effective_levels",
    "granularities",
    "positive_click",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and optionally repair metadata/npz files for the granularity refine dataset."
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--remove-bad", action="store_true", default=False)
    parser.add_argument("--show-bad", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_npz(npz_path: Path) -> Tuple[bool, str]:
    if not npz_path.exists():
        return False, "npz_missing"
    try:
        with np.load(npz_path, allow_pickle=False) as blob:
            missing = sorted(REQUIRED_NPZ_KEYS - set(blob.files))
            if missing:
                return False, f"npz_missing_keys:{','.join(missing)}"
            _ = blob["corrected_probs"].shape
            _ = blob["heuristic_masks"].shape
            _ = blob["gt_mask"].shape
    except Exception as error:
        return False, f"npz_error:{type(error).__name__}:{error}"
    return True, "ok"


def validate_record(record: Dict) -> Tuple[bool, str]:
    image_path = Path(record["image_path"])
    npz_path = Path(record["npz_path"])
    if not image_path.exists():
        return False, "image_missing"
    return validate_npz(npz_path)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    overall_bad = 0
    for split in ["train", "val", "test"]:
        metadata_path = dataset_root / "metadata" / f"{split}.jsonl"
        records = load_jsonl(metadata_path)
        kept_records: List[Dict] = []
        bad_records: List[Tuple[str, str, str]] = []

        for record in records:
            ok, reason = validate_record(record)
            sample_name = record.get("sample_name", "<unknown>")
            npz_path = record.get("npz_path", "")
            if ok:
                kept_records.append(record)
            else:
                bad_records.append((sample_name, reason, npz_path))

        overall_bad += len(bad_records)
        print(f"[{split}] total={len(records)} bad={len(bad_records)} kept={len(kept_records)}")
        for sample_name, reason, npz_path in bad_records[:args.show_bad]:
            print(f"  - {sample_name} :: {reason} :: {npz_path}")
        if len(bad_records) > args.show_bad:
            print(f"  ... ({len(bad_records) - args.show_bad} more)")

        if args.remove_bad and bad_records:
            removed_npz = 0
            for _, _, npz_path in bad_records:
                path = Path(npz_path)
                if path.exists():
                    path.unlink()
                    removed_npz += 1
            with open(metadata_path, "w", encoding="utf-8") as handle:
                for record in kept_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  repaired metadata={metadata_path} removed_npz={removed_npz}")

    print(f"overall_bad={overall_bad}")


if __name__ == "__main__":
    main()
