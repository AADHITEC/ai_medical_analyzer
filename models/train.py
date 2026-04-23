"""
models/train.py
Training script for ChestXRayClassifier on NIH ChestX-ray14 or custom dataset.

Dataset structure expected:
  data/
    train/
      Normal/       *.jpg
      Pneumonia/    *.jpg
      COVID-19/     *.jpg
      Tuberculosis/ *.jpg
      Pleural Effusion/ *.jpg
    val/  (same structure)

Usage:
  python models/train.py --data_dir data --epochs 30 --batch_size 32
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from classifier import ChestXRayClassifier, get_transforms, CLASSES


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--save_path",  default="models/chest_xray_resnet50.pth")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=get_transforms("train"),
    )
    val_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=get_transforms("inference"),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"[Train] Classes : {train_ds.classes}")
    print(f"[Train] Train   : {len(train_ds)} samples")
    print(f"[Train] Val     : {len(val_ds)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ChestXRayClassifier(num_classes=len(CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                criterion, device, scaler)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✓ Saved best model  (val_acc={val_acc:.1f}%)")

    print(f"\n[Train] Done. Best val accuracy: {best_val_acc:.1f}%")
    print(f"[Train] Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
