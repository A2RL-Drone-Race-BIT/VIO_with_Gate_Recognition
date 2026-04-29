from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import GateMaskDataset
from model import MobileNetV3UNet, count_trainable_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds_flat = preds.flatten(1)
    targets_flat = targets.flatten(1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    pred_sum = preds_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection

    iou = ((intersection + eps) / (union + eps)).mean()
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    return {"iou": float(iou.item()), "dice": float(dice.item())}


def build_datasets(args: argparse.Namespace) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    has_split_files = args.train_split is not None and args.val_split is not None

    if has_split_files:
        train_dataset = GateMaskDataset(
            args.images,
            args.masks,
            image_size=args.size,
            split_file=args.train_split,
            augment=True,
            hflip=args.hflip,
        )
        val_dataset = GateMaskDataset(
            args.images,
            args.masks,
            image_size=args.size,
            split_file=args.val_split,
            augment=False,
        )
        return train_dataset, val_dataset

    base_dataset = GateMaskDataset(args.images, args.masks, image_size=args.size, augment=False)
    if len(base_dataset) < 2:
        raise RuntimeError("Need at least 2 image-mask pairs for a train/val split.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    val_len = max(1, int(round(len(indices) * args.val_ratio)))
    val_len = min(val_len, len(indices) - 1)
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    train_full = GateMaskDataset(
        args.images,
        args.masks,
        image_size=args.size,
        augment=True,
        hflip=args.hflip,
    )
    val_full = GateMaskDataset(args.images, args.masks, image_size=args.size, augment=False)

    return Subset(train_full, train_indices), Subset(val_full, val_indices)


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


@torch.no_grad()
def estimate_pos_weight(loader: DataLoader, device: torch.device) -> float:
    positive = 0.0
    total = 0.0
    for _, masks in tqdm(loader, desc="estimate pos_weight", leave=False):
        masks = masks.to(device, non_blocking=True)
        positive += float(masks.sum().item())
        total += float(masks.numel())

    if positive < 1.0:
        return 1.0

    negative = total - positive
    return float(min(max(negative / positive, 1.0), 50.0))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    args: argparse.Namespace,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
    total_dice_metric = 0.0

    desc = "train" if train else "val"
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss_bce = bce_loss(logits, masks)
                loss_dice = dice_loss_with_logits(logits, masks)
                loss = args.bce_weight * loss_bce + args.dice_weight * loss_dice

            if train:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics = segmentation_metrics(logits.detach(), masks.detach(), threshold=args.threshold)
        total_loss += float(loss.item())
        total_bce += float(loss_bce.item())
        total_dice_loss += float(loss_dice.item())
        total_iou += metrics["iou"]
        total_dice_metric += metrics["dice"]

    count = max(len(loader), 1)
    return {
        "loss": total_loss / count,
        "bce": total_bce / count,
        "dice_loss": total_dice_loss / count,
        "iou": total_iou / count,
        "dice": total_dice_metric / count,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_iou: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
            "args": {**vars(args), "device": str(args.device)},
        },
        path,
    )


def append_history(path: Path, row: Dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-UNet gate mask segmentation.")
    parser.add_argument("--images", type=str, default="data/images")
    parser.add_argument("--masks", type=str, default="data/masks")
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hflip", action="store_true")

    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.device = device

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(
        json.dumps({**vars(args), "device": str(device)}, indent=2),
        encoding="utf-8",
    )

    train_dataset, val_dataset = build_datasets(args)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)

    model = MobileNetV3UNet(pretrained=args.pretrained, dropout=args.dropout).to(device)
    if args.freeze_encoder:
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_iou = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint.get("best_iou", 0.0))

    pos_weight = None
    if args.auto_pos_weight:
        estimate_loader = make_loader(train_dataset, args, shuffle=False)
        pos_weight = estimate_pos_weight(estimate_loader, device)
    elif args.pos_weight > 0:
        pos_weight = args.pos_weight

    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    if pos_weight_tensor is not None:
        print(f"BCE pos_weight: {float(pos_weight_tensor.item()):.3f}")

    history_path = save_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            bce_loss,
            optimizer,
            scaler,
            args,
            train=True,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            device,
            bce_loss,
            optimizer=None,
            scaler=scaler,
            args=args,
            train=False,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_iou": train_stats["iou"],
            "train_dice": train_stats["dice"],
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
            "val_dice": val_stats["dice"],
        }
        append_history(history_path, row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={lr:.2e} "
            f"train_loss={train_stats['loss']:.4f} train_iou={train_stats['iou']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_iou={val_stats['iou']:.4f} "
            f"val_dice={val_stats['dice']:.4f}"
        )

        save_checkpoint(save_dir / "latest.pth", model, optimizer, scheduler, epoch, best_iou, args)
        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            save_checkpoint(save_dir / "best.pth", model, optimizer, scheduler, epoch, best_iou, args)
            print(f"Saved best checkpoint: {save_dir / 'best.pth'} (val_iou={best_iou:.4f})")


if __name__ == "__main__":
    main()
