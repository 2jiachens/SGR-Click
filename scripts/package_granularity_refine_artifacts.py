import argparse
import tarfile
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the generated granularity-refine dataset and exported model artifacts."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Generated dataset root directory.")
    parser.add_argument("--exp-root", type=str, required=True, help="Training experiment directory.")
    parser.add_argument("--output", type=str, required=True, help="Output .tar.gz path.")
    parser.add_argument(
        "--include-vis",
        action="store_true",
        default=False,
        help="Include exp-root/vis in the archive.",
    )
    parser.add_argument(
        "--include-preview",
        action="store_true",
        default=False,
        help="Include dataset-root/previews in the archive.",
    )
    parser.add_argument(
        "--dataset-arcname",
        type=str,
        default="dataset",
        help="Top-level archive folder name for the dataset.",
    )
    parser.add_argument(
        "--model-arcname",
        type=str,
        default="model",
        help="Top-level archive folder name for model artifacts.",
    )
    return parser.parse_args()


def iter_model_paths(exp_root: Path, include_vis: bool) -> List[Tuple[Path, str]]:
    paths: List[Tuple[Path, str]] = []
    for name in ["best.pth", "latest.pth", "history.json"]:
        path = exp_root / name
        if path.exists():
            paths.append((path, name))
    vis_dir = exp_root / "vis"
    if include_vis and vis_dir.exists():
        paths.append((vis_dir, "vis"))
    return paths


def iter_dataset_paths(dataset_root: Path, include_preview: bool) -> List[Tuple[Path, str]]:
    names: Iterable[str] = ["records", "metadata", "splits", "README.md"]
    paths: List[Tuple[Path, str]] = []
    for name in names:
        path = dataset_root / name
        if path.exists():
            paths.append((path, name))
    previews_dir = dataset_root / "previews"
    if include_preview and previews_dir.exists():
        paths.append((previews_dir, "previews"))
    return paths


def add_path(archive: tarfile.TarFile, source: Path, arcname: str):
    archive.add(str(source), arcname=arcname, recursive=True)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    exp_root = Path(args.exp_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {exp_root}")
    if output_path.suffixes[-2:] != [".tar", ".gz"] and output_path.suffix != ".tgz":
        raise ValueError("Output path must end with .tar.gz or .tgz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_paths = iter_dataset_paths(dataset_root, args.include_preview)
    model_paths = iter_model_paths(exp_root, args.include_vis)

    if not dataset_paths:
        raise RuntimeError(f"No dataset artifacts found under {dataset_root}")
    if not model_paths:
        raise RuntimeError(f"No model artifacts found under {exp_root}")

    with tarfile.open(output_path, mode="w:gz") as archive:
        for source, rel_name in dataset_paths:
            add_path(archive, source, f"{args.dataset_arcname}/{rel_name}")
        for source, rel_name in model_paths:
            add_path(archive, source, f"{args.model_arcname}/{rel_name}")

    print(f"archive={output_path}")
    print(f"dataset_root={dataset_root}")
    print(f"exp_root={exp_root}")
    print(f"include_preview={args.include_preview}")
    print(f"include_vis={args.include_vis}")


if __name__ == "__main__":
    main()
