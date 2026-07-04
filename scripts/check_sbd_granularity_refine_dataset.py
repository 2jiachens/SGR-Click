import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether the SBD granularity refine dataset is fully generated."
    )
    parser.add_argument(
        "--sbd-root",
        type=str,
        required=True,
        help="Path to benchmark_RELEASE/dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Generated dataset root, for example outputs/SBD_granularity_refine_full.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio used by the generator.",
    )
    parser.add_argument(
        "--min-object-area",
        type=int,
        default=400,
        help="Minimum object area used by the generator.",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="How many missing sample names to print per split.",
    )
    return parser.parse_args()


def read_split_ids(split_file: Path) -> List[str]:
    with open(split_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def deterministic_train_val_split(image_ids: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    import hashlib

    scored = []
    for image_id in image_ids:
        score = int(hashlib.md5(image_id.encode("utf-8")).hexdigest(), 16)
        scored.append((score, image_id))
    scored.sort()
    val_count = max(1, int(round(len(scored) * val_ratio)))
    val_ids = [image_id for _, image_id in scored[:val_count]]
    train_ids = [image_id for _, image_id in scored[val_count:]]
    return train_ids, val_ids


def load_instance_mask(inst_path: Path):
    return loadmat(str(inst_path))["GTinst"][0][0][0]


def expected_sample_names(sbd_root: Path, image_ids: Iterable[str], min_object_area: int) -> Set[str]:
    sample_names: Set[str] = set()
    for image_id in image_ids:
        inst_path = sbd_root / "inst" / f"{image_id}.mat"
        if not inst_path.exists():
            continue
        instances_mask = load_instance_mask(inst_path)
        instance_ids = [int(x) for x in set(instances_mask.reshape(-1).tolist()) if int(x) > 0]
        for instance_id in instance_ids:
            area = int((instances_mask == instance_id).sum())
            if area < min_object_area:
                continue
            sample_names.add(f"{image_id}_inst{instance_id:03d}")
    return sample_names


def load_jsonl_records(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_split(dataset_root: Path, split_name: str, expected_names: Set[str], show_missing: int) -> Dict:
    metadata_path = dataset_root / "metadata" / f"{split_name}.jsonl"
    records_dir = dataset_root / "records" / split_name
    previews_dir = dataset_root / "previews" / split_name

    metadata = load_jsonl_records(metadata_path)
    metadata_names = [item["sample_name"] for item in metadata if "sample_name" in item]
    metadata_name_set = set(metadata_names)
    metadata_duplicates = sorted([name for name, count in Counter(metadata_names).items() if count > 1])

    npz_name_set = {path.stem for path in records_dir.glob("*.npz")} if records_dir.exists() else set()
    preview_name_set = {path.stem for path in previews_dir.glob("*.jpg")} if previews_dir.exists() else set()

    missing_metadata = sorted(expected_names - metadata_name_set)
    missing_npz = sorted(expected_names - npz_name_set)
    metadata_without_npz = sorted(metadata_name_set - npz_name_set)
    npz_without_metadata = sorted(npz_name_set - metadata_name_set)

    preview_covered = len(preview_name_set & expected_names)
    npz_covered = len(npz_name_set & expected_names)

    return {
        "split": split_name,
        "expected": len(expected_names),
        "metadata_records": len(metadata),
        "unique_metadata_names": len(metadata_name_set),
        "npz_files": len(npz_name_set),
        "preview_files": len(preview_name_set),
        "npz_covered": npz_covered,
        "preview_covered": preview_covered,
        "missing_metadata": missing_metadata,
        "missing_npz": missing_npz,
        "metadata_without_npz": metadata_without_npz,
        "npz_without_metadata": npz_without_metadata,
        "metadata_duplicates": metadata_duplicates,
        "complete": not (
            missing_metadata or missing_npz or metadata_without_npz or npz_without_metadata or metadata_duplicates
        ),
        "show_missing": show_missing,
    }


def print_split_summary(result: Dict):
    split_name = result["split"]
    print(f"[{split_name}]")
    print(
        "  "
        f"expected={result['expected']} "
        f"metadata={result['metadata_records']} "
        f"unique_metadata={result['unique_metadata_names']} "
        f"npz={result['npz_files']} "
        f"previews={result['preview_files']}"
    )
    print(
        "  "
        f"covered_npz={result['npz_covered']}/{result['expected']} "
        f"covered_previews={result['preview_covered']}/{result['expected']}"
    )
    print(f"  complete={result['complete']}")

    show_missing = result["show_missing"]
    for label in [
        "missing_metadata",
        "missing_npz",
        "metadata_without_npz",
        "npz_without_metadata",
        "metadata_duplicates",
    ]:
        values = result[label]
        if not values:
            continue
        print(f"  {label}={len(values)}")
        for name in values[:show_missing]:
            print(f"    - {name}")
        if len(values) > show_missing:
            print(f"    ... ({len(values) - show_missing} more)")


def main():
    args = parse_args()
    sbd_root = Path(args.sbd_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    official_train_ids = read_split_ids(sbd_root / "train.txt")
    official_test_ids = read_split_ids(sbd_root / "val.txt")
    train_ids, val_ids = deterministic_train_val_split(official_train_ids, args.val_ratio)
    split_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": official_test_ids,
    }

    results = []
    for split_name, image_ids in split_to_ids.items():
        expected_names = expected_sample_names(sbd_root, image_ids, args.min_object_area)
        results.append(summarize_split(dataset_root, split_name, expected_names, args.show_missing))

    overall_complete = all(item["complete"] for item in results)
    print(f"dataset_root={dataset_root}")
    print(f"overall_complete={overall_complete}")
    for result in results:
        print_split_summary(result)


if __name__ == "__main__":
    main()
