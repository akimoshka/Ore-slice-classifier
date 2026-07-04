"""Train the binary talc U-Net on the blue-line annotation masks.

Pipeline:
    1. build masks from annotation pairs (idempotent);
    2. split images into train/val;
    3. train the U-Net with BCE + Dice loss and cosine LR;
    4. save the best (by val Dice) weights to ``models/unet_talc.pth`` and the
       epoch history to ``reports/unet_talc_history.csv``.

Usage:
    python -m src.segmentation.train_talc --epochs 80
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.segmentation.dataset import TalcDataset, list_pairs
from src.segmentation.talc_masks import build_all_masks
from src.segmentation.unet import ResNetUNet

ROOT = Path(__file__).resolve().parents[2]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * target).sum(dim=(1, 2, 3)) + eps
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - numerator / denominator).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    dice_total, iou_total, count = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs > 0.5).float()
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        pred_area = preds.sum(dim=(1, 2, 3))
        true_area = masks.sum(dim=(1, 2, 3))
        dice = (2 * intersection + 1) / (pred_area + true_area + 1)
        iou = (intersection + 1) / (pred_area + true_area - intersection + 1)
        dice_total += dice.sum().item()
        iou_total += iou.sum().item()
        count += images.size(0)
    return dice_total / count, iou_total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-frac", type=float, default=0.24)
    parser.add_argument("--warmup", type=int, default=25,
                        help="ignore these first epochs when picking the best checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs_dir = ROOT / "data" / "talc_pairs"
    masks_dir = ROOT / "data" / "talc_masks"
    print("Building talc masks from annotations...")
    build_all_masks(pairs_dir, masks_dir)

    pairs = list_pairs(pairs_dir, masks_dir)
    random.shuffle(pairs)
    n_val = max(4, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"{len(pairs)} images -> {len(train_pairs)} train / {len(val_pairs)} val")

    train_loader = DataLoader(
        TalcDataset(train_pairs, size=args.size, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=len(train_pairs) > args.batch_size,
    )
    val_loader = DataLoader(
        TalcDataset(val_pairs, size=args.size, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    device = pick_device()
    print(f"Device: {device}")
    model = ResNetUNet(num_classes=1, pretrained=True).to(device)

    # Positive class (talc) covers ~20% of pixels -> weight BCE toward it.
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    history_path = reports_dir / "unet_talc_history.csv"
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    weights_path = models_dir / "unet_talc.pth"
    last_path = models_dir / "unet_talc_last.pth"

    best_dice = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        val_dice, val_iou = evaluate(model, val_loader, device)
        history.append((epoch, epoch_loss, val_dice, val_iou))
        torch.save(model.state_dict(), last_path)
        # Only trust the "best" checkpoint once the encoder has warmed up; early
        # epochs score well by predicting one big blob on the tiny val set.
        flag = ""
        if epoch > args.warmup and val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), weights_path)
            flag = " *saved"
        print(f"epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}{flag}")

    # Fall back to the last checkpoint if warmup never produced a "best".
    if not weights_path.exists():
        torch.save(model.state_dict(), weights_path)

    with history_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_iou"])
        writer.writerows(history)

    print(f"\nBest val Dice: {best_dice:.4f}")
    print(f"Weights: {weights_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
