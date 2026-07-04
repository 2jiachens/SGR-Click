from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from granularity_recovery.granularity_mapping import map_display_granularities_for_model


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class GranularityRecoveryDataset(Dataset):
    def __init__(self, dataset_root: str, split: str = "train", normalize: bool = True):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.normalize = normalize
        self.samples: List[Dict] = []

        metadata_path = self.dataset_root / "metadata" / f"{split}.jsonl"
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (image - mean) / std

    def __getitem__(self, index):
        sample = self.samples[index]
        image = imread_unicode(sample["image_path"])
        if image is None:
            raise FileNotFoundError(sample["image_path"])

        blob = np.load(sample["npz_path"])
        image = self._normalize_image(image) if self.normalize else image.astype(np.float32).transpose(2, 0, 1) / 255.0
        simple_prob = blob["simple_prob"].astype(np.float32)
        heuristic_masks = blob["global_level_masks"].astype(np.float32)
        base_teacher_probs = (
            blob["base_teacher_probs"].astype(np.float32)
            if "base_teacher_probs" in blob.files
            else blob["refine_teacher_probs"].astype(np.float32)
        )
        recovery_teacher_probs = (
            blob["recovery_teacher_probs"].astype(np.float32)
            if "recovery_teacher_probs" in blob.files
            else blob["completion_teacher_probs"].astype(np.float32)
        )
        training_teacher_probs = (
            blob["training_teacher_probs"].astype(np.float32)
            if "training_teacher_probs" in blob.files
            else None
        )
        gt_mask = blob["gt_mask"].astype(np.float32)
        click_map = blob["positive_click_map"].astype(np.float32)
        granularities = blob["granularities"].astype(np.float32)
        model_granularities = (
            blob["model_granularities"].astype(np.float32)
            if "model_granularities" in blob.files
            else map_display_granularities_for_model(granularities)
        )
        if "effective_level_mask" in blob.files:
            effective_level_mask = blob["effective_level_mask"].astype(np.float32)
        else:
            effective_level_mask = np.ones_like(granularities, dtype=np.float32)

        variants = []
        teacher_probs = []
        teacher_masks = []
        gt_masks = []
        for level_index, (granularity, model_granularity, heuristic_mask, base_prob, recovery_prob) in enumerate(
            zip(granularities, model_granularities, heuristic_masks, base_teacher_probs, recovery_teacher_probs)
        ):
            teacher_prob = training_teacher_probs[level_index] if training_teacher_probs is not None else (
                base_prob if float(granularity) <= 0.5 else recovery_prob
            )
            anchor_prob = base_prob if float(granularity) > 0.5 else simple_prob
            g_map = np.full_like(simple_prob, model_granularity, dtype=np.float32)
            stacked = np.concatenate([
                image,
                anchor_prob[None, ...],
                heuristic_mask[None, ...],
                click_map[None, ...],
                g_map[None, ...],
            ], axis=0)
            variants.append(stacked.astype(np.float32))
            teacher_probs.append(teacher_prob[None, ...].astype(np.float32))
            teacher_masks.append((teacher_prob >= 0.49).astype(np.float32)[None, ...])
            gt_masks.append(gt_mask[None, ...].astype(np.float32))

        return {
            "inputs": torch.from_numpy(np.stack(variants, axis=0)),
            "teacher_probs": torch.from_numpy(np.stack(teacher_probs, axis=0)),
            "teacher_masks": torch.from_numpy(np.stack(teacher_masks, axis=0)),
            "gt_masks": torch.from_numpy(np.stack(gt_masks, axis=0)),
            "effective_level_mask": torch.from_numpy(effective_level_mask.astype(np.float32)),
            "granularities": torch.from_numpy(granularities.astype(np.float32)),
            "sample_name": sample["sample_name"],
        }
