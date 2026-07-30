"""
Training script for the steel defect classification CNN.

Usage:
    python -m steel_defect.train
    python -m steel_defect.train --epochs 30 --batch_size 64 --lr 0.0005
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from steel_defect.utils import (
    setup_logging,
    DEVICE,
    CHECKPOINT_PATH,
    MODELS_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
)
from steel_defect.dataset import build_file_list, create_splits, create_dataloaders
from steel_defect.preprocessing import build_train_transforms, build_val_transforms
from steel_defect.model import SteelCNN

logger = setup_logging(__name__)


def setup_training(
    model: nn.Module,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, optim.Optimizer]:
    """
    TRAIN-1: Set up the loss function and optimizer.

    Create:
        1. criterion — nn.CrossEntropyLoss()
           This combines LogSoftmax + NLLLoss. It expects:
             - input: raw logits of shape (batch, num_classes)
             - target: class indices of shape (batch,)

        2. optimizer — optim.Adam(model.parameters(), lr=learning_rate)
           Adam is a good default optimizer that adapts learning rates
           per-parameter.

    Args:
        model: The CNN model.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Tuple of (criterion, optimizer).
    """
    # ┌──────────────────────────────────────────────┐
    # │  TRAIN-1: Write your code below              │
    # └──────────────────────────────────────────────┘

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return (criterion, optimizer)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    TRAIN-2: Run one training epoch.

    For each batch in the loader:
        1. Move images and labels to the device
        2. Zero the optimizer gradients:      optimizer.zero_grad()
        3. Forward pass:                       outputs = model(images)
        4. Compute loss:                       loss = criterion(outputs, labels)
        5. Backward pass:                      loss.backward()
        6. Update weights:                     optimizer.step()
        7. Track running loss and correct predictions

    Args:
        model: The CNN model (should be in train mode).
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (Adam).
        device: Device to run on (cpu/cuda/mps).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
        - average_loss: total loss / number of batches
        - accuracy: correct predictions / total samples (float, 0-1)

    Hint:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            ... (steps 2-6) ...
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    """
    # ┌──────────────────────────────────────────────┐
    # │  TRAIN-2: Write your code below              │
    # └──────────────────────────────────────────────┘

    model.train()
    running_loss= 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    average_loss = running_loss / len(loader)
    accuracy = correct / total

    return (average_loss, accuracy)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    TRAIN-3: Run one validation epoch.

    Similar to train_one_epoch but:
        - Set model to eval mode:   model.eval()
        - Wrap in torch.no_grad() context — no gradient computation
        - Do NOT call optimizer.zero_grad(), loss.backward(), or optimizer.step()

    Args:
        model: The CNN model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (average_loss, accuracy) for the validation set.

    Hint:
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                ... forward pass and accumulate metrics ...
    """
    # ┌──────────────────────────────────────────────┐
    # │  TRAIN-3: Write your code below              │
    # └──────────────────────────────────────────────┘


    model.eval()
    with torch.no_grad():   
            
        running_loss= 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        average_loss = running_loss / len(loader)
        accuracy = correct / total

    return (average_loss, accuracy)


# ── Scaffold — main training loop ─────────────────────────────

def train(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> str:
    """
    Full training pipeline.

    Orchestrates data loading, model creation, training loop,
    and checkpoint saving. Students implement the individual
    functions called here.
    """
    logger.info("=" * 60)
    logger.info("Steel Defect CNN Training")
    logger.info("=" * 60)
    logger.info("Epochs:     %d", epochs)
    logger.info("Batch size: %d", batch_size)
    logger.info("LR:         %s", learning_rate)
    logger.info("Device:     %s", DEVICE)
    logger.info("Image size: %s", IMAGE_SIZE)
    logger.info("Classes:    %s", CLASS_NAMES)

    start_time = time.time()

    # ── Data ──────────────────────────────────────────────
    logger.info("Loading dataset...")
    file_list = build_file_list()
    train_list, val_list, test_list = create_splits(file_list)
    logger.info(
        "Splits | train=%d | val=%d | test=%d",
        len(train_list), len(val_list), len(test_list),
    )

    train_transform = build_train_transforms()
    val_transform = build_val_transforms()
    train_loader, val_loader = create_dataloaders(
        train_list, val_list, train_transform, val_transform,
        batch_size=batch_size,
    )

    # ── Model ─────────────────────────────────────────────
    model = SteelCNN(num_classes=NUM_CLASSES).to(DEVICE)
    logger.info("Model: %s", model)

    criterion, optimizer = setup_training(model, learning_rate)

    # ── Training Loop ─────────────────────────────────────
    best_val_acc = 0.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.3f | "
            "val_loss=%.4f val_acc=%.3f | time=%.1fs",
            epoch, epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            epoch_time,
        )

        # ┌──────────────────────────────────────────────┐
        # │  TRAIN-4: Save checkpoint if val_acc improved│
        # │                                              │
        # │  If val_acc > best_val_acc:                  │
        # │    1. Update best_val_acc                    │
        # │    2. Save a checkpoint dict with:           │
        # │       - "model_state_dict": model.state_dict│
        # │       - "optimizer_state_dict": optimizer... │
        # │       - "epoch": epoch                       │
        # │       - "best_val_acc": best_val_acc         │
        # │       - "num_classes": NUM_CLASSES            │
        # │    3. Use torch.save(checkpoint, CHECKPOINT_PATH)│
        # │    4. Log the save event                     │
        # │                                              │
        # │  Hint:                                       │
        # │    if val_acc > best_val_acc:                 │
        # │        best_val_acc = val_acc                 │
        # │        torch.save({...}, CHECKPOINT_PATH)    │
        # │        logger.info("Saved best model ...")   │
        # └──────────────────────────────────────────────┘
        # TRAIN-4: Write your code below

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "num_classes": NUM_CLASSES
            }, CHECKPOINT_PATH)

            logger.info(f"Saved best model on => {CHECKPOINT_PATH}")


    elapsed = time.time() - start_time
    logger.info("Training complete | time=%.1fs | best_val_acc=%.3f", elapsed, best_val_acc)
    return str(CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Train Steel Defect CNN")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)


if __name__ == "__main__":
    main()
