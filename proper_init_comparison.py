"""
Fair comparison: BCE vs BCEWithLogitsLoss with proper initialization and lower LR.

This tests whether the two loss functions perform equivalently when:
1. Proper initialization (Kaiming + Xavier) is used
2. Lower learning rate (0.0001) prevents explosion

Setup:
- Model: VGG16 (no BatchNorm)
- Init: Kaiming for ReLU layers, Xavier for final layer
- Optimizer: Adam lr=0.0001
- Dataset: CelebA smile classification
- Epochs: 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from vgg16_bce_investigation import VGG16, init_proper, get_celeba_loaders


def evaluate(model, loader, loss_fn, use_logits):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda()
            labels = labels.float().cuda()
            logits = model(images).flatten()

            if use_logits:
                loss = loss_fn(logits, labels)
                preds = (logits > 0).float()
            else:
                probs = torch.sigmoid(logits)
                loss = loss_fn(probs, labels)
                preds = (probs > 0.5).float()

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, total_loss / total


def train_model(train_loader, test_loader, use_logits, num_epochs, lr, seed):
    loss_name = "BCEWithLogitsLoss" if use_logits else "BCELoss"

    print(f"\n{'='*70}")
    print(f"Training: {loss_name} + Proper Init + Adam lr={lr}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    model = VGG16().cuda()
    init_proper(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if use_logits else nn.BCELoss()

    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Time':<8}")
    print("-" * 55)

    history = []

    for epoch in range(num_epochs):
        model.train()
        start = time.time()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            logits = model(images).flatten()

            if use_logits:
                loss = loss_fn(logits, labels)
                preds = (logits > 0).float()
            else:
                probs = torch.sigmoid(logits)
                loss = loss_fn(probs, labels)
                preds = (probs > 0.5).float()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc, test_loss = evaluate(model, test_loader, loss_fn, use_logits)
        elapsed = time.time() - start

        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} {test_acc:<12.4f} {elapsed:<8.1f}s")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

    return history


def main():
    parser = argparse.ArgumentParser(description="Compare BCE vs BCEWithLogitsLoss with proper init")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*70)
    print("FAIR COMPARISON: BCE vs BCEWithLogitsLoss")
    print("="*70)
    print(f"Init: Proper (Kaiming + Xavier)")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")

    # Load data
    train_loader, test_loader = get_celeba_loaders(args.batch_size, num_workers=4)

    # Train both
    history_bcewl = train_model(
        train_loader, test_loader,
        use_logits=True,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed
    )

    history_bce = train_model(
        train_loader, test_loader,
        use_logits=False,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed
    )

    # Summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Loss Function':<25} {'Final Train Acc':<18} {'Final Test Acc':<18}")
    print("-" * 60)
    print(f"{'BCEWithLogitsLoss':<25} {history_bcewl[-1]['train_acc']:<18.4f} {history_bcewl[-1]['test_acc']:<18.4f}")
    print(f"{'BCELoss':<25} {history_bce[-1]['train_acc']:<18.4f} {history_bce[-1]['test_acc']:<18.4f}")

    diff = history_bcewl[-1]['test_acc'] - history_bce[-1]['test_acc']
    print(f"\nDifference: {diff*100:+.2f} percentage points")

    if abs(diff) < 0.01:
        print("Conclusion: Both losses perform equivalently with proper init + lower LR")
    elif diff > 0:
        print("Conclusion: BCEWithLogitsLoss performs better")
    else:
        print("Conclusion: BCELoss performs better")


if __name__ == "__main__":
    main()
