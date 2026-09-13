from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate.semantic_coverage_data import SemanticHierarchyDataset
from evaluate.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["SGR-Click", "UniGraCo"], required=True)
    parser.add_argument("--backbone", choices=["ViT-B", "ViT-L"], required=True)
    parser.add_argument("--dataset", choices=["PascalPart", "PartImageNet", "SPIN"], default="PascalPart")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-path", default="config.yml")
    parser.add_argument("--sample-limit", type=int, default=50)
    parser.add_argument("--warmup-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--official-repo", type=Path, default=REPO_ROOT / "third_party" / "UniGraCo_tmp")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--pascalpart-annotations-dir", type=Path, default=REPO_ROOT / "datasets" / "PascalPart" / "parts" / "Annotations_Part")
    parser.add_argument("--pascalpart-images-dir", type=Path, default=REPO_ROOT / "datasets" / "VOCdevkit" / "VOC2012" / "JPEGImages")
    parser.add_argument("--partimagenet-zip", type=Path, default=REPO_ROOT / "datasets" / "PartImageNet_Seg.zip")
    parser.add_argument("--spin-annotations-dir", type=Path, default=REPO_ROOT / "datasets" / "SPIN" / "spin_jsons_for_coco")
    args = parser.parse_args()
    if args.sample_limit <= args.warmup_samples:
        parser.error("--sample-limit must exceed --warmup-samples")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, operation: Callable[[], object]) -> tuple[float, int]:
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    operation()
    synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return elapsed_ms, peak_bytes


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
    }


def parameter_count(module: torch.nn.Module | None) -> int:
    return 0 if module is None else int(sum(parameter.numel() for parameter in module.parameters()))


def named_parameter_count(module: torch.nn.Module, needle: str) -> int:
    needle = needle.lower()
    return int(
        sum(
            parameter.numel()
            for name, parameter in module.named_parameters()
            if needle in name.lower()
        )
    )


def sgr_runtime(args: argparse.Namespace, device: torch.device):
    from evaluate.run_semantic_coverage import build_sgr_runtime

    checkpoint_name = {
        "ViT-B": "cocolvis_vit_base.pth",
        "ViT-L": "cocolvis_vit_large.pth",
    }[args.backbone]
    runtime_values = vars(args).copy()
    runtime_values.update(
        checkpoint=f"weights/simpleclick_models/{checkpoint_name}",
        ablation_mode="full",
    )
    runtime_args = SimpleNamespace(**runtime_values)
    controller = build_sgr_runtime(runtime_args)
    controller.granularity_refiner.warmup()
    components = {
        "simpleclick_frozen": parameter_count(controller.net),
        "dinov3_frozen": parameter_count(controller.granularity_refiner.encoder),
    }
    components["frozen"] = components["simpleclick_frozen"] + components["dinov3_frozen"]
    components["trainable"] = 0
    components["total"] = components["frozen"]
    return controller, components


def profile_sgr_sample(controller, image, fine_target, device: torch.device):
    from isegm.inference.clicker import Clicker

    clicker = Clicker(gt_mask=np.asarray(fine_target, dtype=np.int32))
    clicker.make_next_click(np.zeros_like(fine_target, dtype=bool))
    first_click = clicker.clicks_list[-1]
    y, x = first_click.coords

    def first_click_operation():
        controller.set_image(image)
        controller._requested_granularity = 0.50
        controller.add_click(int(x), int(y), bool(first_click.is_positive))

    first_ms, first_peak = timed_call(device, first_click_operation)
    slider_ms = []
    slider_peaks = []
    for control in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        elapsed, peak = timed_call(
            device,
            lambda current=control: controller.set_granularity(current),
        )
        slider_ms.append(elapsed)
        slider_peaks.append(peak)
    controller.reset_last_object(update_image=False)
    return first_ms, first_peak, slider_ms, slider_peaks


def unigraco_runtime(args: argparse.Namespace, device: torch.device):
    from evaluate.run_unigraco_semantic_coverage import load_official_runtime

    if args.checkpoint is None:
        args.checkpoint = args.official_repo / "weights" / "simpleclick" / {
            "ViT-B": "sbd_vit_base.pth",
            "ViT-L": "sbd_vit_large.pth",
        }[args.backbone]
    if args.lora_checkpoint is None:
        args.lora_checkpoint = args.official_repo / "weights" / "graco" / {
            "ViT-B": "GraCo_base_lora.pth",
            "ViT-L": "GraCo_large_lora.pth",
        }[args.backbone]
    predictor = load_official_runtime(args)
    model = predictor.net
    lora = named_parameter_count(model, "lora")
    total = parameter_count(model)
    components = {
        "base_frozen": total - lora,
        "lora_trainable": lora,
        "frozen": total - lora,
        "trainable": lora,
        "total": total,
    }
    return predictor, components


def profile_unigraco_sample(predictor, image, fine_target, device: torch.device):
    from isegm.inference.clicker import Clicker

    clicker = Clicker(gt_mask=np.asarray(fine_target, dtype=np.int32))
    clicker.make_next_click(np.zeros_like(fine_target, dtype=bool))

    def first_click_operation():
        predictor.set_input_image(image)
        predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
        predictor.get_prediction(clicker, gra=0.50)

    first_ms, first_peak = timed_call(device, first_click_operation)
    slider_ms = []
    slider_peaks = []
    for control in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        def slider_operation(current=control):
            predictor.prev_prediction = torch.zeros_like(
                predictor.original_image[:, :1, :, :]
            )
            predictor.get_prediction(clicker, gra=current)

        elapsed, peak = timed_call(device, slider_operation)
        slider_ms.append(elapsed)
        slider_peaks.append(peak)
    return first_ms, first_peak, slider_ms, slider_peaks


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = SemanticHierarchyDataset(args)
    samples = dataset.samples()[: args.sample_limit]
    if args.method == "SGR-Click":
        runtime, parameters = sgr_runtime(args, device)
        profile_sample = profile_sgr_sample
    else:
        runtime, parameters = unigraco_runtime(args, device)
        profile_sample = profile_unigraco_sample

    first_click_ms: list[float] = []
    first_click_peak_bytes: list[int] = []
    slider_move_ms: list[float] = []
    slider_move_peak_bytes: list[int] = []
    sample_keys: list[str] = []
    try:
        for index, sample in enumerate(samples):
            image, targets, _ = dataset.load(sample)
            first_ms, first_peak, update_ms, update_peaks = profile_sample(
                runtime,
                image,
                targets[0],
                device,
            )
            if index < args.warmup_samples:
                continue
            sample_keys.append(str(sample["sample_key"]))
            first_click_ms.append(first_ms)
            first_click_peak_bytes.append(first_peak)
            slider_move_ms.extend(update_ms)
            slider_move_peak_bytes.extend(update_peaks)
            print(
                f"[{index + 1}/{len(samples)}] first={first_ms:.2f} ms "
                f"slider_median={np.median(update_ms):.2f} ms",
                flush=True,
            )
    finally:
        dataset.close()

    payload = {
        "protocol": {
            "method": args.method,
            "backbone": args.backbone,
            "dataset": args.dataset,
            "sample_limit": args.sample_limit,
            "warmup_samples_excluded": args.warmup_samples,
            "timing": "wall-clock time with CUDA synchronization",
            "first_click": "image setup plus first positive-click prediction",
            "slider_move": "one unique control update; ten controls per image excluding 0.50",
            "precision": "runtime model default",
            "seed": args.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "parameters": parameters,
        "first_click_latency_ms": distribution(first_click_ms),
        "slider_move_latency_ms": distribution(slider_move_ms),
        "peak_gpu_memory_gib": {
            "first_click_max": max(first_click_peak_bytes, default=0) / (1024 ** 3),
            "slider_move_max": max(slider_move_peak_bytes, default=0) / (1024 ** 3),
        },
        "sample_keys": sample_keys,
        "raw": {
            "first_click_ms": first_click_ms,
            "first_click_peak_bytes": first_click_peak_bytes,
            "slider_move_ms": slider_move_ms,
            "slider_move_peak_bytes": slider_move_peak_bytes,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "raw"}, indent=2))
    print(f"[complete] {args.output}")


if __name__ == "__main__":
    main()
