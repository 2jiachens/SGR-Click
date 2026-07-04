import argparse
import faulthandler
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isegm.inference import clicker, utils
from isegm.inference.dino_granularity import DinoGranularityRefiner
from isegm.inference.predictors import get_predictor


def read_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    faulthandler.dump_traceback_later(30, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("assets/test_imgs/sheep.jpg"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("weights/simpleclick_models/cocolvis_vit_base.pth"))
    parser.add_argument("--dinov3-repo", type=Path,
                        default=Path(r"D:\code\论文代码\INSID3\third_party\dinov3"))
    parser.add_argument("--dinov3-weights", type=Path,
                        default=Path(r"D:\code\论文代码\INSID3\pretrain\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
    parser.add_argument("--dinov3-model", default="dinov3_vitb16")
    parser.add_argument("--click-x", type=int, default=270)
    parser.add_argument("--click-y", type=int, default=190)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    image = read_rgb(args.image)
    print(f"image_loaded={time.perf_counter() - started:.2f}s", flush=True)
    model = utils.load_is_model(args.checkpoint, device, False, cpu_dist_maps=True)
    print(f"simpleclick_loaded={time.perf_counter() - started:.2f}s", flush=True)
    predictor = get_predictor(
        model,
        brs_mode="NoBRS",
        device=device,
        with_flip=False,
        zoom_in_params={
            "skip_clicks": -1,
            "target_size": (448, 448),
            "expansion_ratio": 1.4,
        },
    )
    predictor.set_input_image(image)

    clicks = clicker.Clicker()
    clicks.add_click(clicker.Click(True, coords=(args.click_y, args.click_x)))
    probability = predictor.get_prediction(clicks)
    print(f"simpleclick_predicted={time.perf_counter() - started:.2f}s", flush=True)

    refiner = DinoGranularityRefiner(
        repo_path=str(args.dinov3_repo),
        weights_path=str(args.dinov3_weights),
        device=device,
        model_name=args.dinov3_model,
        image_size=768,
        work_size=512,
        n_segments=1000,
        candidate_threshold=0.50,
        max_hole_ratio=0.04,
        slic_compactness=6.0,
    )
    masks = [refiner.build(image, probability, (args.click_x, args.click_y))]
    print(f"hierarchy_built={time.perf_counter() - started:.2f}s", flush=True)
    masks.extend(refiner.mask_at(value) for value in (0.25, 0.50, 0.75, 1.0))

    areas = [int(mask.sum()) for mask in masks]
    nested = all(np.all(np.logical_or(~first, second)) for first, second in zip(masks, masks[1:]))
    finite_regions = int(np.isfinite(refiner._arrival).sum())
    ties_split = len(refiner._level_costs) == finite_regions
    region_order = np.argsort(refiner._arrival)
    region_order = region_order[np.isfinite(refiner._arrival[region_order])]
    proximity_ordered = True
    for first, second in zip(region_order, region_order[1:]):
        if refiner._primary_arrival[first] == refiner._primary_arrival[second]:
            if refiner._graph_hops[first] > refiner._graph_hops[second]:
                proximity_ordered = False
                break
    candidate = refiner._click_component(probability, (args.click_x, args.click_y), 0.50)
    hole_limit = max(64, int(candidate.sum() * refiner.max_hole_ratio))
    allowed_candidate = refiner._fill_small_holes(candidate, hole_limit)
    within_candidate = bool(np.all(np.logical_or(~masks[-1], allowed_candidate)))
    print(f"device={device}")
    print(f"areas={areas}")
    print(f"strictly_nested={nested}")
    print(f"within_candidate={within_candidate}")
    print(f"semantic_ties_split={ties_split}")
    print(f"proximity_ordered={proximity_ordered}")
    if (not nested or not within_candidate or not ties_split or not proximity_ordered
            or any(first > second for first, second in zip(areas, areas[1:]))):
        raise AssertionError("Granularity masks are not monotonic.")


if __name__ == "__main__":
    main()
