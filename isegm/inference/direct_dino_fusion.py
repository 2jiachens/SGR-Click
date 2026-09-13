from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch

from isegm.inference.dino_granularity import DinoGranularityRefiner


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


class DirectDinoFusionRefiner(DinoGranularityRefiner):
    """Evaluation-only baseline that directly corrects SimpleClick logits.

    It uses the same frozen, position-debiased DINOv3 features as SGR-Click but
    deliberately omits component construction, graph traversal, evidence audit,
    multiscale endpoints, and semantic-staircase logic.
    """

    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        device: torch.device,
        model_name: str = "dinov3_vitl16",
        image_size: int = 768,
        positional_debias_components: int = 500,
        positional_debias_mode: str = "zero",
        positional_debias_seed: int = 0,
        prototype_radius: float = 2.5,
        similarity_temperature: float = 0.08,
        logit_gain: float = 4.5,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.device = device
        self.model_name = model_name
        self.image_size = int(image_size)
        self.positional_debias_components = int(positional_debias_components)
        self.positional_debias_mode = positional_debias_mode
        self.positional_debias_seed = int(positional_debias_seed)
        self.prototype_radius = float(prototype_radius)
        self.similarity_temperature = float(similarity_temperature)
        self.logit_gain = float(logit_gain)
        self.encoder = None
        self._positional_basis_cache = {}
        self._prepared_image_key = None
        self._prepared_semantic_feature = None
        self.reset()

    def reset(self):
        self._original_size = None
        self._simple_probability = None
        self._semantic_support = None
        self._root_image_mask = None
        self._current_granularity = 0.0

    @property
    def ready(self) -> bool:
        return self._simple_probability is not None

    @staticmethod
    def _patch_coordinate(
        click_xy: Tuple[int, int], image_shape: Tuple[int, int], grid_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        x, y = click_xy
        image_h, image_w = image_shape
        grid_h, grid_w = grid_shape
        patch_x = int(np.clip(round(x * grid_w / image_w), 0, grid_w - 1))
        patch_y = int(np.clip(round(y * grid_h / image_h), 0, grid_h - 1))
        return patch_x, patch_y

    def build(
        self,
        image: np.ndarray,
        simple_probability: np.ndarray,
        click_xy: Tuple[int, int],
        positive_clicks: Iterable[Tuple[int, int]] = None,
        negative_clicks: Iterable[Tuple[int, int]] = None,
        candidate_override: np.ndarray = None,
        semantic_overdrive: bool = False,
    ) -> np.ndarray:
        del negative_clicks, candidate_override, semantic_overdrive
        self.reset()
        self._original_size = image.shape[:2]
        feature_tensor = self.prepare_image(image)
        features = feature_tensor.permute(1, 2, 0).float().cpu().numpy()
        features /= np.maximum(np.linalg.norm(features, axis=2, keepdims=True), 1e-6)
        grid_h, grid_w = features.shape[:2]
        probability_patch = cv2.resize(
            np.clip(simple_probability.astype(np.float32), 0.0, 1.0),
            (grid_w, grid_h),
            interpolation=cv2.INTER_LINEAR,
        )

        clicks = list(positive_clicks) if positive_clicks is not None else [click_xy]
        yy, xx = np.mgrid[0:grid_h, 0:grid_w]
        weights = np.zeros((grid_h, grid_w), dtype=np.float32)
        radius_sq = max(self.prototype_radius, 0.5) ** 2
        roots = []
        for positive_click in clicks:
            patch_x, patch_y = self._patch_coordinate(
                positive_click, image.shape[:2], (grid_h, grid_w)
            )
            roots.append((patch_y, patch_x))
            spatial = np.exp(
                -((xx - patch_x) ** 2 + (yy - patch_y) ** 2) / (2.0 * radius_sq)
            ).astype(np.float32)
            weights += spatial * (0.15 + 0.85 * np.square(probability_patch))
            weights[patch_y, patch_x] += 2.0

        prototype = (features * weights[..., None]).sum(axis=(0, 1))
        prototype /= max(float(np.linalg.norm(prototype)), 1e-6)
        similarities = np.einsum("hwc,c->hw", features, prototype).astype(np.float32)
        positive_reference = float(
            (similarities * weights).sum() / max(float(weights.sum()), 1e-6)
        )
        low_probability = probability_patch <= float(np.quantile(probability_patch, 0.35))
        negative_reference = float(
            np.quantile(similarities[low_probability], 0.75)
            if np.any(low_probability)
            else np.quantile(similarities, 0.50)
        )
        center = min(
            positive_reference - 0.03,
            0.5 * (positive_reference + negative_reference),
        )
        patch_support = _sigmoid(
            (similarities - center) / max(self.similarity_temperature, 1e-3)
        ).astype(np.float32)

        height, width = image.shape[:2]
        self._semantic_support = np.clip(
            cv2.resize(patch_support, (width, height), interpolation=cv2.INTER_CUBIC),
            0.0,
            1.0,
        ).astype(np.float32)
        self._simple_probability = np.clip(
            simple_probability.astype(np.float32), 0.0, 1.0
        )
        self._root_image_mask = self._click_component(
            self._simple_probability, click_xy, 0.50
        )
        self._current_granularity = 0.0
        return self.mask_at(0.0)

    def root_mask(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The direct DINO fusion baseline has not been built yet.")
        return self._root_image_mask.copy()

    def mask_at(self, granularity: float) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The direct DINO fusion baseline has not been built yet.")
        self._current_granularity = float(np.clip(granularity, 0.0, 1.0))
        simple = np.clip(self._simple_probability, 1e-4, 1.0 - 1e-4)
        simple_logit = np.log(simple) - np.log1p(-simple)
        correction = self._semantic_support - 0.5
        return _sigmoid(
            simple_logit + self.logit_gain * self._current_granularity * correction
        ).astype(np.float32)

    @property
    def semantic_support(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("The direct DINO fusion baseline has not been built yet.")
        return self._semantic_support.copy()
