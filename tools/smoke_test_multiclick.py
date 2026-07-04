import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_demo.controller import InteractiveController
from isegm.inference import utils
from isegm.inference.dino_granularity import DinoGranularityRefiner
from tools.smoke_test_granularity import read_rgb


def main():
    device = torch.device("cuda:0")
    model = utils.load_is_model(
        PROJECT_ROOT / "weights/simpleclick_models/cocolvis_vit_base.pth",
        device,
        False,
        cpu_dist_maps=True,
    )
    refiner = DinoGranularityRefiner(
        repo_path=r"D:\code\论文代码\INSID3\third_party\dinov3",
        weights_path=str(PROJECT_ROOT / "weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        device=device,
        model_name="dinov3_vitb16",
    )
    predictor_params = {
        "brs_mode": "NoBRS",
        "prob_thresh": 0.5,
        "zoom_in_params": {
            "skip_clicks": -1,
            "target_size": (448, 448),
            "expansion_ratio": 1.4,
        },
        "predictor_params": {"net_clicks_limit": None, "max_size": 800},
    }
    controller = InteractiveController(
        model,
        device,
        predictor_params,
        update_image_callback=lambda **kwargs: None,
        granularity_refiner=refiner,
    )
    controller.set_image(read_rgb(PROJECT_ROOT / "assets/test_imgs/sheep.jpg"))
    controller.add_click(270, 190, True)
    controller.set_granularity(0.5)
    first_area = int((controller.current_object_prob > 0.5).sum())
    first_mask = controller.current_object_prob > 0.5

    controller.add_click(80, 200, True)
    positive_mask = controller.current_object_prob > 0.5
    positive_area = int(positive_mask.sum())
    assert len(controller.clicker) == 2
    assert np.all(np.logical_or(~first_mask, positive_mask))
    assert not np.array_equal(first_mask, positive_mask)
    assert positive_mask[200, 80]
    candidate_before_negative = controller.granularity_refiner._original_candidate.copy()

    controller.add_click(350, 220, False)
    negative_mask = controller.current_object_prob > 0.5
    negative_area = int(negative_mask.sum())
    assert len(controller.clicker) == 3
    assert controller.granularity_ready
    assert abs(controller.granularity_refiner.current_granularity - 0.5) < 1e-6
    assert np.all(np.logical_or(~negative_mask, positive_mask))
    assert np.array_equal(
        candidate_before_negative,
        controller.granularity_refiner._original_candidate,
    )
    assert negative_mask[190, 270]
    assert negative_mask[200, 80]
    assert not negative_mask[220, 350]

    controller.undo_click()
    undo_area = int((controller.current_object_prob > 0.5).sum())
    assert len(controller.clicker) == 2
    assert controller.granularity_ready

    print(
        f"areas: first={first_area}, after_positive={positive_area}, "
        f"after_negative={negative_area}, undo={undo_area}"
    )
    print("positive_grows_only=True negative_shrinks_only=True undo=True slider_preserved=True")


if __name__ == "__main__":
    main()
