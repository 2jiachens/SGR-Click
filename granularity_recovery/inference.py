from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from granularity_recovery.model import GranularityRecoveryUNet


def click_to_map(height: int, width: int, click_xy: Tuple[int, int], sigma: float = 10.0) -> np.ndarray:
    x, y = click_xy
    yy, xx = np.mgrid[0:height, 0:width]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    return np.exp(-distance_sq / (2.0 * sigma * sigma)).astype(np.float32)


class GranularityRecoveryInference:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device
        self.model = GranularityRecoveryUNet().to(device)
        self._load_checkpoint()
        self.model.eval()

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Granularity recovery checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (image - mean) / std

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
        anchor_prob: np.ndarray,
        heuristic_mask: np.ndarray,
        click_xy: Tuple[int, int],
        granularity: float,
    ) -> np.ndarray:
        height, width = anchor_prob.shape
        image_tensor = self._normalize_image(image)
        heuristic_mask = heuristic_mask.astype(np.float32)
        click_map = click_to_map(height, width, click_xy)
        g_map = np.full_like(anchor_prob, granularity, dtype=np.float32)

        stacked = np.concatenate([
            image_tensor,
            anchor_prob.astype(np.float32)[None, ...],
            heuristic_mask[None, ...],
            click_map[None, ...],
            g_map[None, ...],
        ], axis=0)

        inputs = torch.from_numpy(stacked).unsqueeze(0).to(self.device, dtype=torch.float32)
        logits = self.model(inputs)
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
