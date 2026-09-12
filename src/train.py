"""
train.py
--------
Trains the CIFAR10CNN on the CIFAR-10 dataset, tracks metrics, saves the
best checkpoint, and generates evaluation plots.

Usage:
    python3 src/train.py --epochs 15 --batch-size 64 --lr 0.001
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CIFAR10CNN
from utils import plot_training_curves, plot_confusion_matrix


def get_dataloaders(batch_size: int, data_dir: str = "data"):
    """Download CIFAR-10 and return train/val/test DataLoaders."""

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Split off 10% of training data for validation
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Run one epoch of training or evaluation. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_test_set(model, loader, device):
    """Collect predictions on the test set for the confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir)

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(args.models_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (val acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model for final test evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    plot_training_curves(history, args.outputs_dir)
    y_true, y_pred = evaluate_test_set(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, args.outputs_dir)


if __name__ == "__main__":
    main()
