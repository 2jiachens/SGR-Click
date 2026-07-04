import matplotlib
matplotlib.use('Agg')

import argparse
import os
import tkinter as tk
from pathlib import Path

import torch

from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp


REPO_ROOT = Path(__file__).resolve().parent


def resolve_default_dinov3_repo():
    env_path = os.environ.get('DINOV3_REPO')
    if env_path:
        return env_path
    return str(REPO_ROOT / 'third_party' / 'dinov3')


def resolve_default_dinov3_weights(model_name):
    env_path = os.environ.get('DINOV3_WEIGHTS')
    if env_path:
        return env_path

    weights_dir = REPO_ROOT / 'weights' / 'dinov3'
    default_names = {
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
    }
    return str(weights_dir / default_names[model_name])


def resolve_default_granularity_refine_checkpoint():
    env_path = os.environ.get('GRANULARITY_REFINE_CHECKPOINT')
    if env_path:
        return env_path

    checkpoint = REPO_ROOT / 'experiments' / 'granularity_refine' / 'best.pth'
    if checkpoint.exists():
        return str(checkpoint)
    return None


def main():
    args, cfg = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--no-dinov3-granularity', action='store_false', dest='dinov3_granularity',
                        help='Disable the one-click DINOv3 granularity hierarchy.')
    parser.add_argument('--dinov3-repo', type=str, default=None,
                        help='Path to the local DINOv3 repository.')
    parser.add_argument('--dinov3-model', type=str, default='dinov3_vitl16',
                        choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                        help='DINOv3 architecture matching the selected weights.')
    parser.add_argument('--dinov3-weights', type=str, default=None,
                        help='Path to DINOv3 weights matching --dinov3-model.')
    parser.add_argument('--dinov3-image-size', type=int, default=768,
                        help='Longest input side used for DINOv3 feature extraction.')
    parser.add_argument('--granularity-work-size', type=int, default=512,
                        help='Longest side used for the superpixel region graph.')
    parser.add_argument('--granularity-segments', type=int, default=1000,
                        help='Approximate number of SLIC regions in the hierarchy.')
    parser.add_argument('--granularity-candidate-threshold', type=float, default=0.50,
                        help='SimpleClick probability threshold for the maximum hierarchy extent.')
    parser.add_argument('--granularity-max-hole-ratio', type=float, default=0.04,
                        help='Maximum enclosed hole area, relative to the SimpleClick candidate, to fill.')
    parser.add_argument('--granularity-slic-compactness', type=float, default=6.0,
                        help='SLIC compactness; lower values follow image boundaries more closely.')
    parser.add_argument('--granularity-refine-checkpoint', type=str, default=None,
                        help='Optional checkpoint for the trained granularity refine UNet.')
    parser.add_argument('--no-granularity-refine-model', action='store_true', default=False,
                        help='Disable the trained granularity refine model even if a checkpoint exists.')
    parser.set_defaults(dinov3_granularity=True)

    args = parser.parse_args()
    if args.dinov3_repo is None:
        args.dinov3_repo = resolve_default_dinov3_repo()
    if args.dinov3_weights is None:
        args.dinov3_weights = resolve_default_dinov3_weights(args.dinov3_model)
    if args.granularity_refine_checkpoint is None and not args.no_granularity_refine_model:
        args.granularity_refine_checkpoint = resolve_default_granularity_refine_checkpoint()
    if args.no_granularity_refine_model:
        args.granularity_refine_checkpoint = None
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
