from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isegm.utils import exp
from isegm.inference import utils
from isegm.inference.soft_dino_semantic import SoftDinoSemanticRefiner
from granularity_recovery.gui_app import GranularityRecoveryDemoApp
from granularity_recovery.soft_semantic_controller import SoftSemanticFusionController


def resolve_default_dinov3_repo():
    env_path = os.environ.get("DINOV3_REPO")
    if env_path:
        return env_path
    return str(REPO_ROOT / "third_party" / "dinov3")


def resolve_default_dinov3_weights(model_name):
    env_path = os.environ.get("DINOV3_WEIGHTS")
    if env_path:
        return env_path
    weights_dir = REPO_ROOT / "weights" / "dinov3"
    default_names = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    }
    return str(weights_dir / default_names[model_name])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--limit-longest-size", type=int, default=800)
    parser.add_argument("--cfg", type=str, default="config.yml")
    parser.add_argument("--eval-ritm", action="store_true", default=False)
    parser.add_argument("--dinov3-repo", type=str, default=None)
    parser.add_argument("--dinov3-model", type=str, default="dinov3_vitl16",
                        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"])
    parser.add_argument("--dinov3-weights", type=str, default=None)
    parser.add_argument("--dinov3-image-size", type=int, default=768)
    parser.add_argument("--disable-boundary-projection", action="store_true", default=False,
                        help="Disable SimpleClick boundary projection (component ablation).")
    args = parser.parse_args()

    if args.dinov3_repo is None:
        args.dinov3_repo = resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = resolve_default_dinov3_weights(args.dinov3_model)
    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}")
    cfg = exp.load_config_file(args.cfg, return_edict=True)
    return args, cfg


def main():
    args, cfg = parse_args()
    torch.backends.cudnn.deterministic = True

    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    # Use the native PyTorch distance maps.  The legacy Cython CPU path has a
    # fixed-size queue that is unsafe for longer multi-click trajectories.
    model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm, cpu_dist_maps=False)

    granularity_refiner = SoftDinoSemanticRefiner(
        repo_path=args.dinov3_repo,
        weights_path=args.dinov3_weights,
        device=args.device,
        model_name=args.dinov3_model,
        image_size=args.dinov3_image_size,
    )
    print("Initializing DINOv3 encoder and positional basis...", flush=True)
    granularity_refiner.warmup()
    print("DINOv3 initialization complete.", flush=True)

    root = tk.Tk()
    root.minsize(960, 480)
    app_holder = {}

    controller = SoftSemanticFusionController(
        model,
        args.device,
        predictor_params={
            "brs_mode": "NoBRS",
            "prob_thresh": 0.5,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (448, 448),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": args.limit_longest_size,
            },
            "brs_opt_func_params": {"min_iou_diff": 1e-3},
            "lbfgs_params": {"maxfun": 20},
        },
        update_image_callback=lambda reset_canvas=False: app_holder["app"]._update_image(
            reset_canvas=reset_canvas,
        ),
        granularity_refiner=granularity_refiner,
        enable_boundary_projection=not args.disable_boundary_projection,
    )
    app = GranularityRecoveryDemoApp(root, controller, limit_longest_size=args.limit_longest_size)
    app_holder["app"] = app
    root.deiconify()
    app.mainloop()


if __name__ == "__main__":
    main()
