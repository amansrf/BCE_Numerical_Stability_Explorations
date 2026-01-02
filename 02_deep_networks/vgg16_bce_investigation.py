"""
VGG16 BCE Investigation: Systematic replication of Raschka's findings with detailed logging.

Experiments:
1. BCEWithLogitsLoss + Raschka init (baseline - should work ~92%)
2. Sigmoid + BCELoss + Raschka init (should fail ~50%)
3. BCEWithLogitsLoss + Proper init (Kaiming + Xavier)
4. Sigmoid + BCELoss + Proper init (Kaiming + Xavier) - THE KEY QUESTION

Logging:
- Layer-wise gradient magnitudes every N batches
- Logit/sigmoid distributions
- Loss curves
- Activation statistics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from collections import defaultdict
import time
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import argparse


def check_working_directory():
    """Ensure script is run from the 02_deep_networks directory."""
    cwd = Path.cwd()
    expected_marker = Path("../data/celeba")
    script_name = "vgg16_bce_investigation.py"

    if not expected_marker.exists() or not Path(script_name).exists():
        print(f"Error: This script must be run from the 02_deep_networks directory.")
        print(f"Current directory: {cwd}")
        print(f"\nTo run correctly:")
        print(f"  cd 02_deep_networks")
        print(f"  python {script_name}")
        sys.exit(1)


check_working_directory()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# VGG16 MODEL (Matching Raschka exactly)
# =============================================================================

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x

    def get_all_layers(self):
        """Return list of (name, module) for all parameterized layers."""
        layers = []
        for block_name, block in [('block1', self.block_1), ('block2', self.block_2),
                                   ('block3', self.block_3), ('block4', self.block_4),
                                   ('block5', self.block_5)]:
            conv_idx = 1
            for m in block:
                if isinstance(m, nn.Conv2d):
                    layers.append((f'{block_name}_conv{conv_idx}', m))
                    conv_idx += 1

        fc_idx = 1
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                layers.append((f'fc{fc_idx}', m))
                fc_idx += 1

        return layers


# =============================================================================
# INITIALIZATION STRATEGIES
# =============================================================================

def init_raschka(model):
    """Raschka's initialization: N(0, 0.05) for all weights."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()


def init_proper(model):
    """
    Proper initialization:
    - Kaiming He for ReLU layers (all conv + fc1, fc2)
    - Xavier/Glorot for the final pre-sigmoid layer (fc3)
    """
    layers = model.get_all_layers()

    for _, m in layers[:-1]:  # All layers except the last
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Final layer: Xavier for sigmoid
    _, final_layer = layers[-1]
    nn.init.xavier_normal_(final_layer.weight)
    if final_layer.bias is not None:
        nn.init.zeros_(final_layer.bias)

    print(f"Initialized {len(layers)-1} layers with Kaiming, final layer with Xavier")


# =============================================================================
# DATA LOADING
# =============================================================================

def get_celeba_loaders(batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.CenterCrop((160, 160)),
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def get_smile(attr):
        return attr[31]

    train_dataset = datasets.CelebA(
        root='../data/celeba', split='train', transform=transform,
        target_type='attr', target_transform=get_smile, download=True
    )
    test_dataset = datasets.CelebA(
        root='../data/celeba', split='test', transform=transform,
        target_type='attr', target_transform=get_smile, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset):,} samples, Test: {len(test_dataset):,} samples")
    return train_loader, test_loader


# =============================================================================
# LOGGING
# =============================================================================

class DetailedLogger:
    def __init__(self, run_name, log_dir="runs_vgg16_investigation"):
        self.run_name = run_name
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.history = defaultdict(list)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.history[tag].append((step, value))

    def log_histogram(self, tag, values, step):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
        self.writer.add_histogram(tag, values, step)

    def log_layer_gradients(self, model, step):
        """Log gradient statistics for each layer."""
        for name, layer in model.get_all_layers():
            if layer.weight.grad is not None:
                grad = layer.weight.grad
                self.log_scalar(f'grad/{name}/mean', grad.abs().mean().item(), step)
                self.log_scalar(f'grad/{name}/max', grad.abs().max().item(), step)
                self.log_scalar(f'grad/{name}/norm', grad.norm().item(), step)
                zero_frac = (grad.abs() < 1e-8).float().mean().item()
                self.log_scalar(f'grad/{name}/zero_frac', zero_frac, step)

    def log_initial_state(self, model, sample_images):
        """Log model state before training."""
        model.eval()
        with torch.no_grad():
            logits = model(sample_images.to(DEVICE)).flatten()
            sigmoid_out = torch.sigmoid(logits)

        stats = {
            'logit_mean': logits.mean().item(),
            'logit_std': logits.std().item(),
            'logit_min': logits.min().item(),
            'logit_max': logits.max().item(),
            'sigmoid_mean': sigmoid_out.mean().item(),
            'sigmoid_min': sigmoid_out.min().item(),
            'sigmoid_max': sigmoid_out.max().item(),
            'sigmoid_saturated': ((sigmoid_out < 0.01) | (sigmoid_out > 0.99)).float().mean().item()
        }

        print(f"\n{'='*60}")
        print("INITIAL STATE")
        print(f"{'='*60}")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
            self.log_scalar(f'initial/{k}', v, 0)
        print(f"{'='*60}\n")

        self.log_histogram('initial/logits', logits, 0)
        self.log_histogram('initial/sigmoid', sigmoid_out, 0)

        return stats

    def close(self):
        self.writer.close()
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(dict(self.history), f)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, loss_fn, use_logits, logger, epoch, log_every=50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)
        global_step = epoch * len(loader) + batch_idx

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

        # Detailed logging every N batches
        if batch_idx % log_every == 0:
            logger.log_scalar('batch/loss', loss.item(), global_step)
            logger.log_scalar('batch/logit_mean', logits.mean().item(), global_step)
            logger.log_scalar('batch/logit_std', logits.std().item(), global_step)
            logger.log_scalar('batch/logit_max_abs', logits.abs().max().item(), global_step)

            sigmoid_out = torch.sigmoid(logits).detach()
            sat_frac = ((sigmoid_out < 0.01) | (sigmoid_out > 0.99)).float().mean().item()
            logger.log_scalar('batch/sigmoid_saturated_frac', sat_frac, global_step)

            logger.log_layer_gradients(model, global_step)

            # Histogram every 200 batches
            if batch_idx % 200 == 0:
                logger.log_histogram('batch/logits', logits, global_step)
                logger.log_histogram('batch/sigmoid', sigmoid_out, global_step)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn, use_logits):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_logits = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

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
            all_logits.append(logits.cpu())

    return total_loss / total, correct / total, torch.cat(all_logits)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(init_type, use_logits, num_epochs=4, batch_size=256, lr=0.001, seed=42):
    """Run a single experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loss_name = "BCEWithLogits" if use_logits else "BCE"
    run_name = f"{init_type}_{loss_name}_seed{seed}"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {run_name}")
    print(f"{'='*70}")

    # Create model and apply initialization
    model = VGG16().to(DEVICE)

    if init_type == "raschka":
        init_raschka(model)
    elif init_type == "proper":
        init_proper(model)
    else:
        raise ValueError(f"Unknown init: {init_type}")

    print(f"Model: VGG16 ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Init: {init_type}")
    print(f"Loss: {loss_name}")
    print(f"Device: {DEVICE}")

    # Data
    train_loader, test_loader = get_celeba_loaders(batch_size)

    # Loss and optimizer
    if use_logits:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Logger
    logger = DetailedLogger(run_name)

    # Log initial state
    sample_images, _ = next(iter(train_loader))
    initial_stats = logger.log_initial_state(model, sample_images)

    # Training loop
    results = {'epochs': []}

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, use_logits,
            logger, epoch
        )
        test_loss, test_acc, test_logits = evaluate(model, test_loader, loss_fn, use_logits)

        elapsed = time.time() - start

        logger.log_scalar('epoch/train_loss', train_loss, epoch)
        logger.log_scalar('epoch/train_acc', train_acc, epoch)
        logger.log_scalar('epoch/test_loss', test_loss, epoch)
        logger.log_scalar('epoch/test_acc', test_acc, epoch)
        logger.log_histogram('epoch/test_logits', test_logits, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | Time: {elapsed:.1f}s")

        results['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    logger.close()

    results['config'] = {
        'init_type': init_type,
        'use_logits': use_logits,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'seed': seed
    }
    results['initial_state'] = initial_stats
    results['final_test_acc'] = test_acc

    return results


def run_all_experiments(num_epochs=4, seed=42):
    """Run all 4 experiment combinations."""
    experiments = [
        ("raschka", True),   # Should work (~92%)
        ("raschka", False),  # Should fail (~50%)
        ("proper", True),    # Should work
        ("proper", False),   # KEY QUESTION: Does proper init fix BCE?
    ]

    all_results = []

    for init_type, use_logits in experiments:
        result = run_experiment(init_type, use_logits, num_epochs=num_epochs, seed=seed)
        all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<40} {'Final Test Acc':>15}")
    print("-"*60)
    for r in all_results:
        cfg = r['config']
        loss_name = "BCEWithLogits" if cfg['use_logits'] else "BCE"
        name = f"{cfg['init_type']}_{loss_name}"
        print(f"{name:<40} {r['final_test_acc']*100:>14.2f}%")
    print("="*70)

    # Save all results
    with open("vgg16_investigation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to vgg16_investigation_results.json")
    print("TensorBoard: tensorboard --logdir=runs_vgg16_investigation")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "single"], default="all")
    parser.add_argument("--init", choices=["raschka", "proper"], default="raschka")
    parser.add_argument("--loss", choices=["bcewithlogits", "bce"], default="bcewithlogits")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.mode == "all":
        run_all_experiments(num_epochs=args.epochs, seed=args.seed)
    else:
        use_logits = (args.loss == "bcewithlogits")
        run_experiment(args.init, use_logits, num_epochs=args.epochs, seed=args.seed)
