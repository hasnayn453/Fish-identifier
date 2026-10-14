#!/usr/bin/env python3
"""
Fish Species Classification - Optimized Training Script

Lightweight transfer learning with MobileNetV3-Small (~2.5M params, ~10MB weights).
Saves only state_dict to models/ (gitignored) to keep repo under 100MB.
"""

import os
import sys
import argparse
import time
import copy
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from tqdm import tqdm


# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_DATA_DIR = "dataset"
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "mobilenet_v3_small"  # ~2.5M params, ~10MB weights
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_INPUT_SIZE = 224
DEFAULT_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]


# ============================================================================
# Device Detection
# ============================================================================

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[GPU] Apple Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("[CPU] CPU only")
    return device


# ============================================================================
# Data Transforms
# ============================================================================

def get_transforms(input_size: int = DEFAULT_INPUT_SIZE) -> Dict[str, transforms.Compose]:
    """Data augmentation for training, clean resize for validation."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return {"train": train_tf, "val": val_tf}


# ============================================================================
# Dataset & DataLoader
# ============================================================================

def create_dataloaders(
    data_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_split: float = DEFAULT_VAL_SPLIT,
    num_workers: int = DEFAULT_NUM_WORKERS,
    input_size: int = DEFAULT_INPUT_SIZE
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """Create train/val dataloaders from folder-per-class structure."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders in {data_dir}. Expected: dataset/class_name/*.jpg")

    print(f"[Data] Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    tf = get_transforms(input_size)
    full_ds = datasets.ImageFolder(data_dir, transform=tf["train"])
    class_names = full_ds.classes

    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    gen = torch.Generator().manual_seed(DEFAULT_SEED)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=gen)
    val_ds.dataset.transform = tf["val"]

    print(f"[Data] Train: {train_size} | Val: {val_size}")

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True,
                         persistent_workers=num_workers > 0),
    }
    return loaders, {"train": train_size, "val": val_size}, class_names


# ============================================================================
# Model Creation
# ============================================================================

def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create lightweight transfer learning model."""
    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        print(f"[Model] MobileNetV3-Small: {num_classes} classes")

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        print(f"[Model] MobileNetV3-Large: {num_classes} classes")

    elif model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        print(f"[Model] ResNet18: {num_classes} classes")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: {SUPPORTED_MODELS}")

    return model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model size in MB."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "size_mb": total * 4 / 1e6}


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    """Single training epoch."""
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix({"loss": loss_sum/total, "acc": correct/total})

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, epoch, epochs):
    """Single validation epoch."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix({"loss": loss_sum/total, "acc": correct/total})

    return loss_sum / total, correct / total


def train(model, loaders, sizes, criterion, optimizer, scheduler, device, epochs, model_path, class_names, model_name, input_size):
    """Full training loop with early stopping."""
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience = 7
    wait = 0

    print(f"\n{'='*50}")
    print(f"Training {model_name} for {epochs} epochs")
    print(f"{'='*50}\n")

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, optimizer, device, epoch, epochs)
        val_loss, val_acc = eval_epoch(model, loaders["val"], criterion, device, epoch, epochs)
        if scheduler:
            scheduler.step()

        dt = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"{dt:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            wait = 0
            # Save lightweight state_dict only
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "model_name": model_name,
                "input_size": input_size,
                "best_acc": best_acc,
                "epoch": epoch,
            }, model_path)
            print(f"  [OK] Best model saved (val_acc={best_acc:.4f}) -> {model_path}")
        else:
            wait += 1
            if wait >= patience:
                print(f"\n[EarlyStop] No improvement for {patience} epochs")
                break

    model.load_state_dict(best_wts)
    return model, best_acc


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train lightweight fish classifier")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dataset root folder")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Output model directory (gitignored)")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, choices=SUPPORTED_MODELS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--input_size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()

    # Data
    loaders, sizes, class_names = create_dataloaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers, args.input_size
    )

    # Model
    model = create_model(args.model_name, len(class_names), pretrained=not args.no_pretrained).to(device)
    info = get_model_size(model)
    print(f"[Model] Params: {info['total_params']:,} | Size: {info['size_mb']:.1f} MB (FP32)")

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Output path (in gitignored models/ folder)
    Path(args.model_dir).mkdir(exist_ok=True)
    model_path = Path(args.model_dir) / f"{args.model_name}_fish_classifier.pth"

    # Train
    model, best_acc = train(model, loaders, sizes, criterion, optimizer, scheduler,
                            device, args.epochs, str(model_path), class_names, args.model_name, args.input_size)

    print(f"\n{'='*50}")
    print(f"Done. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Repo size impact: ~{info['size_mb']:.1f} MB (gitignored)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()