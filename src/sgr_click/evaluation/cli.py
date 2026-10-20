"""Reproduce semantic coverage and click-correction evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sgr_click.cli import add_model_arguments, load_model
from .semantic_coverage_data import SemanticHierarchyDataset, load_paper_samples
from .semantic_coverage import hierarchical_semantic_metrics, semantic_trajectory_diagnostics
from .slider_metrics import slider_diagnostics
from .noc_protocols import Branches, load_official, measure


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def subset_samples(dataset, samples, split):
    ordered = sorted(samples, key=lambda s: hashlib.sha256(
        ("compact-mechanism-v1|" + dataset + "|" + s["sample_key"]).encode()
    ).hexdigest())
    if split == "development":
        return ordered[:100]
    if split == "remaining":
        return ordered[100:]
    return samples


def strict_chain(targets):
    return len(targets) >= 2 and all(
        np.any(b & ~a) and not np.any(a & ~b)
        for a, b in zip(targets[:-1], targets[1:])
    )


def semantic_record(controller, image, targets):
    eligible = strict_chain(targets)
    official = load_official()
    clicks = official.Clicker(gt_mask=targets[0].astype(np.int32))
    clicks.make_next_click(np.zeros_like(targets[0], dtype=bool))
    click = clicks.get_clicks()[0]
    x, y = int(click.coords[1]), int(click.coords[0])
    controller.set_image(image)
    controller.set_granularity(0.0)
    start = time.perf_counter()
    try:
        controller.add_click(x, y, True)
        if not controller.granularity_ready:
            raise RuntimeError("No first-click semantic path")
        path = controller.granularity_refiner.path
        masks = []
        for tick in range(101):
            controller.set_granularity(tick / 100)
            masks.append((controller.current_object_prob > 0.5).copy())
    except (ValueError, RuntimeError) as error:
        # Unexpected infrastructure/weight failures must not become model scores.
        if "first semantic mask does not contain" not in str(error):
            raise
        return dict(construction_failed=True, error=str(error), m_biou=0.0,
                    sor=0.0 if eligible else None, sor_eligible=eligible,
                    stagnant_window_rate=None, jump_step_rate=None, click_xy=[x, y])
    metrics = hierarchical_semantic_metrics(masks, targets)
    diagnostics = semantic_trajectory_diagnostics(
        metrics["matched_prediction_indices"], [int(m.sum()) for m in masks],
        metrics["target_prediction_ious"],
    )
    return dict(
        **metrics, **slider_diagnostics(masks), construction_failed=False,
        sor=diagnostics["semantic_order_success"] if eligible else None,
        sor_eligible=eligible, click_xy=[x, y], anchor_ticks=path.anchor_ticks.tolist(),
        elapsed_seconds=time.perf_counter() - start,
    )


def summarize(records, task):
    def mean(values):
        return float(np.mean(values)) if values else None
    if task == "semantic":
        return dict(
            samples=len(records), construction_failures=sum(r["construction_failed"] for r in records),
            m_biou=mean([r["m_biou"] for r in records]),
            sor_samples=sum(r["sor_eligible"] for r in records),
            sor=mean([r["sor"] for r in records if r["sor_eligible"]]),
            stagnant_window_rate=mean([r["stagnant_window_rate"] for r in records if not r["construction_failed"]]),
            jump_step_rate=mean([r["jump_step_rate"] for r in records if not r["construction_failed"]]),
        )
    result = {"samples": len(records), "construction_failures": sum(r.get("construction_failed", False) for r in records)}
    for protocol in ["ten", "best101"]:
        rows = [r[protocol] for r in records if protocol in r]
        if rows:
            result[protocol] = {str(t): {
                "noc": mean([r["noc"][str(t)]["noc"] for r in rows]),
                "failures": sum(not r["noc"][str(t)]["reached"] for r in rows),
                "samples": len(rows),
            } for t in [85, 90]}
    return result


def dataset_args(args):
    import yaml
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Resolve relative data paths against the configuration file, not the shell.
    paths = {}
    for key, value in cfg.items():
        path = Path(value)
        paths[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
    return SimpleNamespace(
        dataset=args.dataset, category=args.category,
        pascalpart_annotations_dir=paths["pascalpart_annotations"],
        pascalpart_images_dir=paths["pascalpart_images"],
        partimagenet_zip=paths["partimagenet_zip"],
        spin_annotations_dir=paths["spin_annotations"],
        partimagenet_whole_source="parts",
    ), paths


def whole_samples(dataset, paths):
    from isegm.inference.utils import get_dataset
    cfg = SimpleNamespace(
        GRABCUT_PATH=paths["grabcut"], BERKELEY_PATH=paths["berkeley"],
        DAVIS_PATH=paths["davis"], SBD_PATH=paths["sbd"], PASCALVOC_PATH=paths["pascalvoc"],
    )
    data = get_dataset(dataset, cfg)
    if len(data) == 0:
        raise ValueError(f"No samples found for {dataset}. Check the dataset layout and configuration.")
    for index in range(len(data)):
        sample = data.get_sample(index)
        for object_id in sample.objects_ids:
            yield (f"{sample.sample_id}:{object_id}", sample.image, sample.gt_mask(object_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["semantic", "noc"])
    add_model_arguments(parser)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--dataset", required=True, choices=[
        "PascalPart", "PartImageNet", "SPIN", "GrabCut", "Berkeley", "DAVIS", "SBD", "PascalVOC"])
    parser.add_argument("--category", choices=["all", "head", "torso", "wheel", "body", "foot"], default="all")
    parser.add_argument("--split", choices=["full", "development", "remaining"], default="remaining")
    parser.add_argument("--protocol", choices=["both", "ten", "best101"], default="both")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Debug sample cap; 0 evaluates all selected targets.")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--validate-branches", action="store_true")
    args = parser.parse_args()
    allowed = {"PascalPart": {"all", "head", "torso", "wheel"},
               "PartImageNet": {"all", "head", "body", "foot"}}
    if args.category not in allowed.get(args.dataset, {"all"}):
        parser.error("Invalid category for this dataset.")
    is_part = args.dataset in {"PascalPart", "PartImageNet", "SPIN"}
    if args.task == "semantic" and not is_part:
        parser.error("Semantic evaluation requires a dataset with hierarchical annotations.")
    if args.task == "noc" and args.dataset == "SPIN":
        parser.error("SPIN is used only for semantic evaluation in the paper.")
    if not 0 <= args.shard_id < args.num_shards or args.limit < 0:
        parser.error("Invalid shard or sample limit.")
    ds_args, paths = dataset_args(args)
    required = {
        "PascalPart": ["pascalpart_annotations", "pascalpart_images"],
        "PartImageNet": ["partimagenet_zip"],
        "SPIN": ["partimagenet_zip", "spin_annotations"],
    }.get(args.dataset, [args.dataset.lower()])
    for key in required:
        if not paths[key].exists():
            raise FileNotFoundError(f"Dataset path does not exist: {key} = {paths[key]}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "output"}
    weights = [args.backbone, args.student if args.model == "lite" else args.dino_weights]
    run_config["weight_sha256"] = {Path(p).name: sha256(p) for p in weights}
    run_config["samples_sha256"] = sha256(Path(__file__).with_name("paper_samples.json"))
    run_config["data_config_sha256"] = sha256(args.config)
    import importlib.metadata
    run_config["environment"] = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image"]},
    }
    run_config["seed"] = 0
    run_config["source_sha256"] = {
        p.relative_to(Path(__file__).parents[2]).as_posix(): sha256(p)
        for package in ["sgr_click", "isegm"]
        for p in sorted((Path(__file__).parents[2] / package).rglob("*.py"))
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != run_config:
            raise ValueError("Output directory belongs to a different run. Choose a fresh --output.")
    else:
        atomic_json(manifest_path, run_config)
    records_path = args.output / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()] if records_path.exists() else []
    done = {r["sample_key"] for r in records}
    dataset = SemanticHierarchyDataset(ds_args) if is_part else None
    if is_part:
        samples = dataset.samples()
        if args.task == "semantic":
            samples = subset_samples(args.dataset, samples, args.split)
        items = ((s["sample_key"], s) for s in samples)
    else:
        items = ((key, (image, target)) for key, image, target in whole_samples(args.dataset, paths))
    controller, source = load_model(args)
    official = load_official()
    processed = 0
    try:
        for index, (key, value) in enumerate(items):
            if index % args.num_shards != args.shard_id:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if key in done:
                continue
            if is_part:
                image, targets, names = dataset.load(value)
                target = targets[0].astype(np.int32)
            else:
                image, target = value
            if args.task == "semantic":
                row = semantic_record(controller, image, targets)
            else:
                try:
                    if args.protocol == "both":
                        row = measure(controller, image, target, is_part, official, args.validate_branches)
                    else:
                        branches = Branches(controller, image, target, official)
                        if args.protocol == "best101":
                            row = {"best101": branches.best_first(), "construction_failed": False}
                        else:
                            primary = branches.ten(0.85 if is_part else 0.90)
                            if is_part:
                                primary["noc"]["90"] = branches.ten(0.90)["noc"]["90"]
                            row = {"ten": primary, "construction_failed": False}
                except (ValueError, RuntimeError) as error:
                    if "first semantic mask does not contain" not in str(error):
                        raise
                    failure = {str(t): {"noc": 20, "reached": False} for t in [85, 90]}
                    row = {"construction_failed": True, "error": str(error)}
                    for protocol in (["ten", "best101"] if args.protocol == "both" else [args.protocol]):
                        row[protocol] = {"noc": failure}
            row.update(sample_key=key, dataset=args.dataset, category=args.category)
            records.append(row)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_json(args.output / "summary.json", summarize(records, args.task))
            print(f"[{processed}] {args.dataset} {key}", flush=True)
        atomic_json(args.output / "summary.json", summarize(records, args.task))
    finally:
        if dataset:
            dataset.close()
        if hasattr(source, "close"):
            source.close()


if __name__ == "__main__":
    main()
