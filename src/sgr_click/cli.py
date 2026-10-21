"""Command-line demo and image inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .model import ModelPaths, build_model


def add_model_arguments(parser):
    parser.add_argument("--model", choices=["tf", "lite"], required=True)
    parser.add_argument("--backbone", default="checkpoints/cocolvis_vit_base.pth")
    parser.add_argument("--student", default="checkpoints/sgr_click_lite.pt")
    parser.add_argument("--dino-repo", default="third_party/dinov3")
    parser.add_argument("--dino-weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)


def load_model(args):
    return build_model(args.model, ModelPaths(
        args.backbone, args.student, args.dino_repo, args.dino_weights,
    ), device=args.device, threads=args.threads)


def main():
    parser = argparse.ArgumentParser(description="SGR-Click: Semantic Granularity Recovery for Interactive Segmentation.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Open the interactive desktop demo.")
    add_model_arguments(demo)
    infer = commands.add_parser("predict", help="Export a mask path from one positive click.")
    add_model_arguments(infer)
    infer.add_argument("--image", required=True, type=Path)
    infer.add_argument("--click", nargs=2, required=True, type=int, metavar=("X", "Y"))
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--granularity", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "predict" and not 0 <= args.granularity <= 1:
        parser.error("--granularity must be in [0, 1]")
    controller, source = load_model(args)
    try:
        if args.command == "demo":
            import tkinter as tk
            from .gui.gui_app import GranularityRecoveryDemoApp
            from tkinter import messagebox
            controller.report_error = messagebox.showerror
            controller.report_warning = messagebox.showwarning
            root = tk.Tk()
            root.minsize(960, 480)
            app = GranularityRecoveryDemoApp(root, controller, limit_longest_size=800)
            controller.update_image_callback = lambda reset_canvas=False: app._update_image(reset_canvas=reset_canvas)
            root.title(f"SGR-Click {args.model.upper()}")
            try:
                app.mainloop()
            finally:
                root.destroy()
        else:
            import numpy as np
            from PIL import Image
            image = np.asarray(Image.open(args.image).convert("RGB")).copy()
            x, y = args.click
            if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                raise ValueError("Click must be inside the image.")
            controller.set_image(image)
            controller.set_granularity(0.0)
            controller.add_click(x, y, True)
            controller.set_granularity(args.granularity)
            mask = controller.current_object_prob > 0.5
            args.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype("uint8") * 255).save(args.output / "mask.png")
            path = controller.granularity_refiner.path
            np.savez_compressed(args.output / "anchors.npz",
                                masks=np.stack(path.masks), granularity=path.anchor_controls)
            (args.output / "prediction.json").write_text(json.dumps({
                "model": args.model, "click_xy": args.click, "granularity": args.granularity,
                "anchor_granularities": path.anchor_controls.tolist(),
            }, indent=2), encoding="utf-8")
    finally:
        if hasattr(source, "close"):
            source.close()


if __name__ == "__main__":
    main()
