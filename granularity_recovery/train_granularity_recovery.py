from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from granularity_recovery.dataset import GranularityRecoveryDataset
from granularity_recovery.model import GranularityRecoveryUNet


def imwrite_unicode(path: Path, image: np.ndarray):
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--exp-root", type=str, default="./experiments/granularity_recovery")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-vis", action="store_true", default=False)
    parser.add_argument("--gt-pull-start", type=float, default=0.55)
    parser.add_argument("--gt-pull-power", type=float, default=2.0)
    parser.add_argument("--gt-pull-weight", type=float, default=0.45)
    parser.add_argument("--last-level-gt-weight", type=float, default=1.0)
    parser.add_argument("--base-mimic-weight", type=float, default=0.25)
    parser.add_argument("--base-edge-weight", type=float, default=2.00)
    parser.add_argument("--gt-edge-weight", type=float, default=0.75)
    parser.add_argument("--edge-shape-weight", type=float, default=1.00)
    parser.add_argument("--edge-kernel-size", type=int, default=5)
    return parser.parse_args()


def pad_tensor_spatial(tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_h = target_height - height
    pad_w = target_width - width
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Target size {(target_height, target_width)} is smaller than {(height, width)}")
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))


def collate_batch(batch):
    max_height = max(sample["inputs"].shape[-2] for sample in batch)
    max_width = max(sample["inputs"].shape[-1] for sample in batch)

    collated = {
        "inputs": [],
        "teacher_probs": [],
        "teacher_masks": [],
        "base_teacher_probs": [],
        "base_teacher_masks": [],
        "gt_masks": [],
        "effective_level_mask": [],
        "granularities": [],
        "sample_name": [],
    }

    for sample in batch:
        collated["inputs"].append(pad_tensor_spatial(sample["inputs"], max_height, max_width))
        collated["teacher_probs"].append(pad_tensor_spatial(sample["teacher_probs"], max_height, max_width))
        collated["teacher_masks"].append(pad_tensor_spatial(sample["teacher_masks"], max_height, max_width))
        collated["base_teacher_probs"].append(pad_tensor_spatial(sample["base_teacher_probs"], max_height, max_width))
        collated["base_teacher_masks"].append(pad_tensor_spatial(sample["base_teacher_masks"], max_height, max_width))
        collated["gt_masks"].append(pad_tensor_spatial(sample["gt_masks"], max_height, max_width))
        collated["effective_level_mask"].append(sample["effective_level_mask"])
        collated["granularities"].append(sample["granularities"])
        collated["sample_name"].append(sample["sample_name"])

    collated["inputs"] = torch.stack(collated["inputs"], dim=0)
    collated["teacher_probs"] = torch.stack(collated["teacher_probs"], dim=0)
    collated["teacher_masks"] = torch.stack(collated["teacher_masks"], dim=0)
    collated["base_teacher_probs"] = torch.stack(collated["base_teacher_probs"], dim=0)
    collated["base_teacher_masks"] = torch.stack(collated["base_teacher_masks"], dim=0)
    collated["gt_masks"] = torch.stack(collated["gt_masks"], dim=0)
    collated["effective_level_mask"] = torch.stack(collated["effective_level_mask"], dim=0)
    collated["granularities"] = torch.stack(collated["granularities"], dim=0)
    return collated


def flatten_variants(batch):
    bsz, num_variants, channels, height, width = batch["inputs"].shape
    inputs = batch["inputs"].reshape(bsz * num_variants, channels, height, width)
    teacher_probs = batch["teacher_probs"].reshape(bsz * num_variants, 1, height, width)
    teacher_masks = batch["teacher_masks"].reshape(bsz * num_variants, 1, height, width)
    base_teacher_probs = batch["base_teacher_probs"].reshape(bsz * num_variants, 1, height, width)
    base_teacher_masks = batch["base_teacher_masks"].reshape(bsz * num_variants, 1, height, width)
    gt_masks = batch["gt_masks"].reshape(bsz * num_variants, 1, height, width)
    effective_level_mask = batch["effective_level_mask"]
    granularities = batch["granularities"]
    return (
        inputs,
        teacher_probs,
        teacher_masks,
        base_teacher_probs,
        base_teacher_masks,
        gt_masks,
        effective_level_mask,
        granularities,
        bsz,
        num_variants,
    )


def dice_loss_from_logits(logits, targets, weights=None, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    loss = 1.0 - dice
    if weights is not None:
        weights = weights.reshape(-1)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def weighted_bce_with_logits(logits, targets, weights):
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expanded_weights = torch.broadcast_to(weights, raw.shape)
    weighted = raw * expanded_weights
    return weighted.sum() / expanded_weights.sum().clamp_min(1.0)


def dice_loss_from_probs(probs, targets, weights=None, eps=1e-6):
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
    if weights is not None:
        sample_weights = weights.reshape(-1)
        return (loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
    return loss.mean()


def boundary_band(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def soft_boundary_map(prob: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    dilated = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def compute_loss(
    logits,
    teacher_probs,
    teacher_masks,
    base_teacher_probs,
    base_teacher_masks,
    gt_masks,
    effective_level_mask,
    granularities,
    batch_size,
    num_variants,
    gt_pull_start,
    gt_pull_power,
    gt_pull_weight,
    last_level_gt_weight,
    base_mimic_weight,
    base_edge_weight,
    gt_edge_weight,
    edge_shape_weight,
    edge_kernel_size,
):
    height, width = logits.shape[-2:]
    flat_effective = effective_level_mask.reshape(batch_size * num_variants, 1, 1, 1).to(logits.device, dtype=logits.dtype)

    teacher_bce = weighted_bce_with_logits(logits, teacher_probs, flat_effective)
    teacher_dice = dice_loss_from_logits(logits, teacher_masks, weights=flat_effective)
    teacher_loss = teacher_bce + teacher_dice

    logits_grouped = logits.reshape(batch_size, num_variants, 1, height, width)
    gt_grouped = gt_masks.reshape(batch_size, num_variants, 1, height, width)
    teacher_probs_grouped = teacher_probs.reshape(batch_size, num_variants, 1, height, width)
    effective_grouped = effective_level_mask.to(logits.device, dtype=logits.dtype)
    granularity_grouped = granularities.to(logits.device, dtype=logits.dtype)

    last_level_mask = torch.zeros_like(effective_grouped)
    last_level_mask[:, -1] = 1.0
    last_level_mask = last_level_mask.reshape(batch_size * num_variants, 1, 1, 1)
    instance_bce = weighted_bce_with_logits(logits, gt_masks, last_level_mask)
    instance_dice = dice_loss_from_logits(logits, gt_masks, weights=last_level_mask)
    instance_loss = instance_bce + instance_dice

    pred_probs_grouped = torch.sigmoid(logits_grouped)
    smooth_loss = logits.new_tensor(0.0)
    smooth_pairs = 0.0
    for index in range(num_variants - 1):
        pair_mask = (effective_grouped[:, index] > 0.5) & (effective_grouped[:, index + 1] > 0.5)
        if not torch.any(pair_mask):
            continue
        pred_delta = pred_probs_grouped[pair_mask, index + 1] - pred_probs_grouped[pair_mask, index]
        teacher_delta = teacher_probs_grouped[pair_mask, index + 1] - teacher_probs_grouped[pair_mask, index]
        smooth_loss = smooth_loss + torch.mean(torch.abs(pred_delta - teacher_delta))
        smooth_pairs += 1.0
    if smooth_pairs > 0:
        smooth_loss = smooth_loss / smooth_pairs

    gt_pull_curve = ((granularity_grouped - gt_pull_start) / max(1.0 - gt_pull_start, 1e-6)).clamp(0.0, 1.0)
    gt_pull_curve = gt_pull_curve.pow(gt_pull_power) * effective_grouped
    gt_pull_weights = gt_pull_curve.reshape(batch_size * num_variants, 1, 1, 1)
    gt_pull_bce = weighted_bce_with_logits(logits, gt_masks, gt_pull_weights)
    gt_pull_dice = dice_loss_from_logits(logits, gt_masks, weights=gt_pull_weights)
    gt_pull_loss = gt_pull_bce + gt_pull_dice

    low_granularity_curve = ((0.55 - granularity_grouped) / 0.55).clamp(0.0, 1.0) * effective_grouped
    low_granularity_weights = low_granularity_curve.reshape(batch_size * num_variants, 1, 1, 1)
    base_mimic_bce = weighted_bce_with_logits(logits, base_teacher_probs, low_granularity_weights)
    base_mimic_dice = dice_loss_from_logits(logits, base_teacher_masks, weights=low_granularity_weights)
    base_mimic_loss = base_mimic_bce + base_mimic_dice

    base_boundary = boundary_band(base_teacher_masks, edge_kernel_size)
    base_edge_weights = base_boundary * low_granularity_weights
    base_edge_pixel_loss = weighted_bce_with_logits(logits, base_teacher_masks, base_edge_weights)
    pred_boundary = soft_boundary_map(torch.sigmoid(logits), edge_kernel_size)
    base_edge_shape_loss = dice_loss_from_probs(
        pred_boundary,
        base_boundary,
        weights=low_granularity_weights,
    )
    base_edge_loss = base_edge_pixel_loss + edge_shape_weight * base_edge_shape_loss

    high_granularity_curve = ((granularity_grouped - 0.50) / 0.50).clamp(0.0, 1.0) * effective_grouped
    high_granularity_weights = high_granularity_curve.reshape(batch_size * num_variants, 1, 1, 1)
    gt_boundary = boundary_band(gt_masks, edge_kernel_size)
    gt_edge_weights = gt_boundary * high_granularity_weights
    gt_edge_pixel_loss = weighted_bce_with_logits(logits, gt_masks, gt_edge_weights)
    gt_edge_shape_loss = dice_loss_from_probs(
        pred_boundary,
        gt_boundary,
        weights=high_granularity_weights,
    )
    gt_edge_loss = gt_edge_pixel_loss + edge_shape_weight * gt_edge_shape_loss

    total = (
        teacher_loss
        + last_level_gt_weight * instance_loss
        + 0.2 * smooth_loss
        + gt_pull_weight * gt_pull_loss
        + base_mimic_weight * base_mimic_loss
        + base_edge_weight * base_edge_loss
        + gt_edge_weight * gt_edge_loss
    )
    return total, {
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "instance_loss": float(instance_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
        "gt_pull_loss": float(gt_pull_loss.detach().cpu()),
        "base_mimic_loss": float(base_mimic_loss.detach().cpu()),
        "base_edge_loss": float(base_edge_loss.detach().cpu()),
        "gt_edge_loss": float(gt_edge_loss.detach().cpu()),
        "base_edge_pixel_loss": float(base_edge_pixel_loss.detach().cpu()),
        "base_edge_shape_loss": float(base_edge_shape_loss.detach().cpu()),
        "gt_edge_pixel_loss": float(gt_edge_pixel_loss.detach().cpu()),
        "gt_edge_shape_loss": float(gt_edge_shape_loss.detach().cpu()),
        "mean_gt_pull_weight": float(gt_pull_curve.mean().detach().cpu()),
        "mean_low_granularity_weight": float(low_granularity_curve.mean().detach().cpu()),
    }


def save_preview(exp_root: Path, epoch: int, logits, teacher_masks, gt_masks):
    vis_dir = exp_root / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred = torch.sigmoid(logits[:1]).detach().cpu().numpy()[0, 0]
    tgt = teacher_masks[:1].detach().cpu().numpy()[0, 0]
    gt = gt_masks[:1].detach().cpu().numpy()[0, 0]
    canvas = np.concatenate([
        (pred * 255.0).clip(0, 255).astype(np.uint8),
        (tgt * 255.0).clip(0, 255).astype(np.uint8),
        (gt * 255.0).clip(0, 255).astype(np.uint8),
    ], axis=1)
    imwrite_unicode(vis_dir / f"epoch_{epoch:03d}.png", canvas)


def run_epoch(model, loader, device, optimizer=None, save_vis=False, exp_root=None, epoch=0, args=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    metrics_accum = {
        "teacher_loss": 0.0,
        "instance_loss": 0.0,
        "smooth_loss": 0.0,
        "gt_pull_loss": 0.0,
        "base_mimic_loss": 0.0,
        "base_edge_loss": 0.0,
        "gt_edge_loss": 0.0,
        "base_edge_pixel_loss": 0.0,
        "base_edge_shape_loss": 0.0,
        "gt_edge_pixel_loss": 0.0,
        "gt_edge_shape_loss": 0.0,
        "mean_gt_pull_weight": 0.0,
        "mean_low_granularity_weight": 0.0,
    }

    for batch_index, batch in enumerate(loader):
        (
            inputs,
            teacher_probs,
            teacher_masks,
            base_teacher_probs,
            base_teacher_masks,
            gt_masks,
            effective_level_mask,
            granularities,
            batch_size,
            num_variants,
        ) = flatten_variants(batch)
        inputs = inputs.to(device=device, dtype=torch.float32)
        teacher_probs = teacher_probs.to(device=device, dtype=torch.float32)
        teacher_masks = teacher_masks.to(device=device, dtype=torch.float32)
        base_teacher_probs = base_teacher_probs.to(device=device, dtype=torch.float32)
        base_teacher_masks = base_teacher_masks.to(device=device, dtype=torch.float32)
        gt_masks = gt_masks.to(device=device, dtype=torch.float32)

        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss, metrics = compute_loss(
                logits,
                teacher_probs,
                teacher_masks,
                base_teacher_probs,
                base_teacher_masks,
                gt_masks,
                effective_level_mask,
                granularities,
                batch_size,
                num_variants,
                args.gt_pull_start,
                args.gt_pull_power,
                args.gt_pull_weight,
                args.last_level_gt_weight,
                args.base_mimic_weight,
                args.base_edge_weight,
                args.gt_edge_weight,
                args.edge_shape_weight,
                args.edge_kernel_size,
            )
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        for key, value in metrics.items():
            metrics_accum[key] += value * batch_size

        if save_vis and batch_index == 0 and exp_root is not None:
            save_preview(exp_root, epoch, logits, teacher_masks, gt_masks)

    mean_loss = total_loss / max(total_items, 1)
    mean_metrics = {key: value / max(total_items, 1) for key, value in metrics_accum.items()}
    return mean_loss, mean_metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    exp_root = Path(args.exp_root).expanduser().resolve()
    exp_root.mkdir(parents=True, exist_ok=True)

    train_dataset = GranularityRecoveryDataset(args.dataset_root, split="train")
    val_dataset = GranularityRecoveryDataset(args.dataset_root, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    model = GranularityRecoveryUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, args=args)
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            save_vis=args.save_vis,
            exp_root=exp_root,
            epoch=epoch,
            args=args,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        latest_path = exp_root / "latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            },
            latest_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "args": vars(args),
                },
                exp_root / "best.pth",
            )

    with open(exp_root / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
