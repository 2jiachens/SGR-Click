import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class GranularityRefineDataset(Dataset):
    def __init__(self, dataset_root, split="train", normalize=True):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.normalize = normalize
        self.samples = []

        metadata_path = self.dataset_root / "metadata" / f"{split}.jsonl"
        with open(metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = imread_unicode(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        blob = np.load(sample["npz_path"])
        simple_prob = blob["simple_prob"].astype(np.float32)
        heuristic_masks = blob["heuristic_masks"].astype(np.float32)
        corrected_probs = blob["corrected_probs"].astype(np.float32)
        gt_mask = blob["gt_mask"].astype(np.float32)
        click_map = blob["positive_click_map"].astype(np.float32)
        granularities = blob["granularities"].astype(np.float32)
        if "effective_level_mask" in blob.files:
            effective_level_mask = blob["effective_level_mask"].astype(np.float32)
        else:
            effective_level_mask = np.ones_like(granularities, dtype=np.float32)

        image = image.transpose(2, 0, 1)
        if self.normalize:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            image = (image - mean) / std

        variants = []
        targets = []
        hard_targets = []
        gt_targets = []
        for granularity, heuristic_mask, corrected_prob in zip(granularities, heuristic_masks, corrected_probs):
            g_map = np.full_like(simple_prob, granularity, dtype=np.float32)
            stacked = np.concatenate([
                image,
                simple_prob[None, ...],
                heuristic_mask[None, ...],
                click_map[None, ...],
                g_map[None, ...],
            ], axis=0)
            variants.append(stacked)
            targets.append(corrected_prob[None, ...])
            hard_targets.append((corrected_prob >= 0.49).astype(np.float32)[None, ...])
            gt_targets.append(gt_mask[None, ...])

        return {
            "inputs": torch.from_numpy(np.stack(variants, axis=0)),
            "teacher_probs": torch.from_numpy(np.stack(targets, axis=0)),
            "teacher_masks": torch.from_numpy(np.stack(hard_targets, axis=0)),
            "gt_masks": torch.from_numpy(np.stack(gt_targets, axis=0)),
            "effective_level_mask": torch.from_numpy(effective_level_mask),
            "sample_name": sample["sample_name"],
        }
