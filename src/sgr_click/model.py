"""Construct the two released models with the same inference configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


@dataclass
class ModelPaths:
    backbone: str | Path
    student: str | Path | None = None
    dino_repo: str | Path | None = None
    dino_weights: str | Path | None = None

    def validate(self, model: str):
        required = {"SimpleClick checkpoint": self.backbone}
        if model == "lite":
            required["Lite semantic head"] = self.student
        else:
            required["DINOv3 checkpoint"] = self.dino_weights
            required["DINOv3 hubconf.py"] = (
                Path(self.dino_repo) / "hubconf.py" if self.dino_repo else None
            )
        for label, path in required.items():
            if path is None or not Path(path).is_file():
                raise FileNotFoundError(f"{label} not found: {path}. See docs/weights.md.")


def build_model(model: str, paths: ModelPaths, device="cuda:0", threads=4):
    """Return (interaction_controller, semantic_provider).

    The controller accepts RGB uint8 images, clicks in (x, y) order, and
    granularity values in [0, 1]. Call provider.close() when finished if present.
    """
    if model not in {"tf", "lite"}:
        raise ValueError("model must be 'tf' or 'lite'")
    paths.validate(model)
    import numpy as np
    import torch
    from isegm.inference import utils
    from isegm.inference.continuous_granularity import ContinuousGranularityRefiner
    from isegm.inference.soft_dino_semantic import SoftDinoSemanticRefiner
    from .runtime.boundary import EdgeController
    from .runtime.pruned_original_controller import install as install_prune
    from .runtime.pixel_arc_ticks import install as install_arc
    from .runtime.quadratic_local_prefix import install as install_prefix
    from .runtime.gated_patch_coordinates import install as install_coordinates
    from .runtime.hard_scope_threshold import install as install_threshold
    from .runtime.candidate_hooks import install_correction_probe
    from .runtime.native_consensus_scopes import install as install_native
    from .runtime.simple_distance_path import install as install_distance
    from .runtime.fast_first_click import install as install_cache

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu or install CUDA-enabled PyTorch.")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(False)
    net = utils.load_is_model(str(paths.backbone), device, eval_ritm=False, cpu_dist_maps=False)
    if model == "tf":
        source = SoftDinoSemanticRefiner(
            repo_path=str(paths.dino_repo), weights_path=str(paths.dino_weights),
            device=device, model_name="dinov3_vitl16", image_size=768,
        )
        source.warmup()
    else:
        from .runtime.semantic_student import make_student
        from .runtime.student_refiner import DistilledSemanticRefiner
        student = make_student("context")
        payload = torch.load(str(paths.student), map_location="cpu", weights_only=True)
        student.load_state_dict(payload["state_dict"], strict=True)
        source = DistilledSemanticRefiner(
            net, student, device, image_size=768, cluster_tau=0.60,
            expansion_evidence_floor=0.15, soft_logit_gain=4.5,
            prototype_radius=2.5, similarity_temperature=0.08, positional_debias_mode="none",
        )

    controller = EdgeController(
        net, device,
        predictor_params={
            "brs_mode": "NoBRS", "prob_thresh": 0.5,
            "zoom_in_params": {"skip_clicks": -1, "target_size": (448, 448), "expansion_ratio": 1.4},
            "predictor_params": {"net_clicks_limit": None, "max_size": 800},
            "brs_opt_func_params": {"min_iou_diff": 1e-3}, "lbfgs_params": {"maxfun": 20},
        },
        update_image_callback=lambda reset_canvas=False: None,
        granularity_refiner=ContinuousGranularityRefiner(source),
        prob_thresh=0.5, enable_boundary_projection=True,
    )
    if model == "lite":
        from .runtime.quality_controller import apply_to_controller
        from .runtime.coherent_scope_dedup import install as install_geometry
        from .runtime.gated_geometry_controller import install as install_scope
        apply_to_controller(controller, "corrected")

        class StudentController(type(controller)):
            def add_click(self, x, y, is_positive):
                if len(self.clicker) == 0 and self._init_mask is None:
                    source.begin_first_click(self.image)
                return super().add_click(x, y, is_positive)

        controller.__class__ = StudentController
        geometry = install_geometry(source, "light")
        install_scope(controller, source, {"variant": "base"}, geometry)

    install_prune(controller, {"enabled": True})
    if model == "lite":
        install_native(controller, {"variant": "near80_protected97_nativebest_frag10"})
    install_arc(controller, {"variant": "pixel_arc"})
    install_prefix(controller, {"enabled": True})
    install_coordinates(controller, source, {
        "enabled": True, "tau": 0.60 if model == "tf" else 0.75,
        "original_single_scope_gate": True, "skip_aligned_rebuild": True,
        "prototype_radius": 1.0,
    })
    install_threshold(controller)
    install_correction_probe(controller)
    controller._probe_correction = "native"
    source._external_trajectory.set_cluster_tau(0.60)
    install_distance(controller)
    install_cache(controller, source)
    return controller, source
