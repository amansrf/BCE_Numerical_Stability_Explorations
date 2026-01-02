"""
BCE vs BCEWithLogitsLoss: Experiment v2
Fixed methodological issues:
1. Train and test sets have same class distribution (digit grouping)
2. Gradient analysis on training data, not test data
3. Statistical significance tests included
4. All metrics computed during training (no post-hoc analysis)
5. Support for CIFAR-10 as harder alternative
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter


def check_working_directory():
    """Ensure script is run from the 01_shallow_networks directory."""
    cwd = Path.cwd()
    script_name = "experiment_v2.py"

    if not Path(script_name).exists():
        print(f"Error: This script must be run from the 01_shallow_networks directory.")
        print(f"Current directory: {cwd}")
        print(f"\nTo run correctly:")
        print(f"  cd 01_shallow_networks")
        print(f"  python {script_name}")
        sys.exit(1)


check_working_directory()

DEVICE = torch.device("cpu")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA LOADING - Fixed to use same distribution for train and test
# =============================================================================

def load_mnist_by_digit_grouping(
    n_positive_digits: int,
    batch_size: int = 128,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, float]:
    """
    Load MNIST as binary classification by grouping digits.

    n_positive_digits: 1-9, determines which digits are positive class
        1 -> digit 0 is positive (10% positive)
        2 -> digits 0,1 are positive (20% positive)
        3 -> digits 0,1,2 are positive (30% positive)
        ...
        5 -> digits 0-4 are positive (50% positive)

    IMPORTANT: Same grouping applied to BOTH train and test sets.

    Returns: train_loader, test_loader, actual_positive_ratio
    """
    if not 1 <= n_positive_digits <= 9:
        raise ValueError("n_positive_digits must be between 1 and 9")

    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

    positive_digits = list(range(n_positive_digits))  # [0], [0,1], [0,1,2], etc.

    def process_dataset(data):
        X = data.data.float().view(-1, 28*28) / 255.0
        X = (X - 0.1307) / 0.3081  # Normalize
        y = torch.tensor([1 if label.item() in positive_digits else 0
                         for label in data.targets])
        return X, y

    X_train, y_train = process_dataset(train_data)
    X_test, y_test = process_dataset(test_data)

    train_ratio = y_train.float().mean().item()
    test_ratio = y_test.float().mean().item()

    # Note: Ratios are approximate (e.g., 21% instead of 20%) because MNIST
    # digits aren't perfectly uniformly distributed. The key is train/test match.
    print(f"  Train ratio: {train_ratio:.3f}, Test ratio: {test_ratio:.3f}")

    # Verify distributions match
    assert abs(train_ratio - test_ratio) < 0.02, \
        f"Train/test ratio mismatch: {train_ratio:.3f} vs {test_ratio:.3f}"

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, train_ratio


def load_cifar10_by_class_grouping(
    n_positive_classes: int,
    batch_size: int = 128,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, float]:
    """
    Load CIFAR-10 as binary classification by grouping classes.

    CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    n_positive_classes: 1-9, determines which classes are positive
        1 -> class 0 (airplane) is positive (10% positive)
        2 -> classes 0,1 are positive (20% positive)
        ...

    Returns: train_loader, test_loader, actual_positive_ratio
    """
    if not 1 <= n_positive_classes <= 9:
        raise ValueError("n_positive_classes must be between 1 and 9")

    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    positive_classes = list(range(n_positive_classes))

    def process_dataset(data):
        # CIFAR-10: 32x32x3 images
        X = torch.tensor(data.data, dtype=torch.float32) / 255.0
        X = X.permute(0, 3, 1, 2)  # NHWC -> NCHW
        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        X = (X - mean) / std
        X = X.view(X.size(0), -1)  # Flatten to (N, 3072)

        y = torch.tensor([1 if label in positive_classes else 0
                         for label in data.targets])
        return X, y

    X_train, y_train = process_dataset(train_data)
    X_test, y_test = process_dataset(test_data)

    train_ratio = y_train.float().mean().item()
    test_ratio = y_test.float().mean().item()

    print(f"  Train ratio: {train_ratio:.3f}, Test ratio: {test_ratio:.3f}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, train_ratio


# =============================================================================
# MODEL
# =============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

    def get_final_layer(self):
        return self.net[-1]


# =============================================================================
# METRICS COMPUTATION - All computed during training
# =============================================================================

@dataclass
class EpochMetrics:
    """All metrics for a single epoch."""
    # Training metrics
    train_loss: float = 0.0
    train_grad_mean: float = 0.0
    train_grad_std: float = 0.0
    train_grad_max: float = 0.0

    # Test metrics
    test_loss: float = 0.0
    test_accuracy: float = 0.0
    test_acc_pos: float = 0.0
    test_acc_neg: float = 0.0

    # Calibration
    brier_score: float = 0.0

    # Logit statistics
    logit_mean: float = 0.0
    logit_std: float = 0.0
    logit_max_abs: float = 0.0
    logit_pos_mean: float = 0.0  # Mean logit for positive class
    logit_neg_mean: float = 0.0  # Mean logit for negative class

    # Wrong predictions
    total_wrong: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    wrong_extreme_count: int = 0  # |logit| > 17 but wrong
    wrong_mean_logit: float = 0.0

    # Per-class gradient analysis (FROM TRAINING DATA)
    grad_pos_mean: float = 0.0
    grad_pos_std: float = 0.0
    grad_neg_mean: float = 0.0
    grad_neg_std: float = 0.0
    grad_pos_zero_frac: float = 0.0
    grad_neg_zero_frac: float = 0.0


def compute_per_class_gradients_from_training(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    use_bce_with_logits: bool,
    device: torch.device,
    n_samples: int = 200
) -> Dict[str, float]:
    """
    Compute gradient statistics per class FROM TRAINING DATA.
    Samples n_samples from training set for analysis.
    """
    model.train()

    # Collect samples from training data
    all_X, all_y = [], []
    for X, y in train_loader:
        all_X.append(X)
        all_y.append(y)
        if sum(len(x) for x in all_X) >= n_samples * 2:
            break

    all_X = torch.cat(all_X)
    all_y = torch.cat(all_y)

    # Sample balanced subset
    pos_idx = (all_y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (all_y == 0).nonzero(as_tuple=True)[0]

    n_per_class = min(n_samples // 2, len(pos_idx), len(neg_idx))

    pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_per_class]]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_per_class]]

    grad_pos, grad_neg = [], []

    for idx in pos_idx:
        xi = all_X[idx:idx+1].to(device)
        yi = all_y[idx:idx+1].float().to(device)

        model.zero_grad()
        logit = model(xi).squeeze()

        if use_bce_with_logits:
            loss = criterion(logit.unsqueeze(0), yi)
        else:
            prob = torch.sigmoid(logit)
            loss = criterion(prob.unsqueeze(0), yi)

        loss.backward()
        grad_mag = model.get_final_layer().weight.grad.abs().mean().item()
        grad_pos.append(grad_mag)

    for idx in neg_idx:
        xi = all_X[idx:idx+1].to(device)
        yi = all_y[idx:idx+1].float().to(device)

        model.zero_grad()
        logit = model(xi).squeeze()

        if use_bce_with_logits:
            loss = criterion(logit.unsqueeze(0), yi)
        else:
            prob = torch.sigmoid(logit)
            loss = criterion(prob.unsqueeze(0), yi)

        loss.backward()
        grad_mag = model.get_final_layer().weight.grad.abs().mean().item()
        grad_neg.append(grad_mag)

    return {
        'pos_mean': np.mean(grad_pos) if grad_pos else 0,
        'pos_std': np.std(grad_pos) if grad_pos else 0,
        'neg_mean': np.mean(grad_neg) if grad_neg else 0,
        'neg_std': np.std(grad_neg) if grad_neg else 0,
        'pos_zero_frac': np.mean([g < 1e-7 for g in grad_pos]) if grad_pos else 0,
        'neg_zero_frac': np.mean([g < 1e-7 for g in grad_neg]) if grad_neg else 0,
    }


def compute_test_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    use_bce_with_logits: bool,
    device: torch.device
) -> Dict:
    """Compute all test metrics."""
    model.eval()

    all_logits = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.float().to(device)
            logits = model(X).squeeze()

            if use_bce_with_logits:
                total_loss += criterion(logits, y).sum().item()
            else:
                probs = torch.sigmoid(logits)
                total_loss += criterion(probs, y).sum().item()

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_probs = torch.sigmoid(all_logits)
    all_preds = (all_probs > 0.5).float()

    n_samples = len(all_targets)
    pos_mask = all_targets == 1
    neg_mask = all_targets == 0

    # Basic metrics
    correct = (all_preds == all_targets).float()
    accuracy = correct.mean().item()
    acc_pos = correct[pos_mask].mean().item() if pos_mask.sum() > 0 else 0
    acc_neg = correct[neg_mask].mean().item() if neg_mask.sum() > 0 else 0

    # Brier score (calibration)
    brier = ((all_probs - all_targets) ** 2).mean().item()

    # Wrong predictions
    wrong = all_preds != all_targets
    false_pos = (wrong & neg_mask).sum().item()
    false_neg = (wrong & pos_mask).sum().item()
    wrong_extreme = ((all_logits.abs() > 17) & wrong).sum().item()
    wrong_mean_logit = all_logits[wrong].abs().mean().item() if wrong.sum() > 0 else 0

    return {
        'loss': total_loss / n_samples,
        'accuracy': accuracy,
        'acc_pos': acc_pos,
        'acc_neg': acc_neg,
        'brier_score': brier,
        'logit_mean': all_logits.mean().item(),
        'logit_std': all_logits.std().item(),
        'logit_max_abs': all_logits.abs().max().item(),
        'logit_pos_mean': all_logits[pos_mask].mean().item() if pos_mask.sum() > 0 else 0,
        'logit_neg_mean': all_logits[neg_mask].mean().item() if neg_mask.sum() > 0 else 0,
        'total_wrong': wrong.sum().item(),
        'false_positives': false_pos,
        'false_negatives': false_neg,
        'wrong_extreme_count': wrong_extreme,
        'wrong_mean_logit': wrong_mean_logit,
        'logits': all_logits.numpy(),
        'targets': all_targets.numpy(),
        'probs': all_probs.numpy(),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_bce_with_logits: bool,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch, return training metrics."""
    model.train()

    total_loss = 0.0
    n_samples = 0
    grad_magnitudes = []

    for X, y in train_loader:
        X, y = X.to(device), y.float().to(device)
        batch_size = X.size(0)

        optimizer.zero_grad()
        logits = model(X).squeeze()

        if use_bce_with_logits:
            loss = criterion(logits, y).mean()
        else:
            probs = torch.sigmoid(logits)
            loss = criterion(probs, y).mean()

        loss.backward()

        # Track gradient magnitude
        final_layer = model.get_final_layer()
        if final_layer.weight.grad is not None:
            grad_magnitudes.append(final_layer.weight.grad.abs().mean().item())

        optimizer.step()

        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return {
        'loss': total_loss / n_samples,
        'grad_mean': np.mean(grad_magnitudes) if grad_magnitudes else 0,
        'grad_std': np.std(grad_magnitudes) if grad_magnitudes else 0,
        'grad_max': np.max(grad_magnitudes) if grad_magnitudes else 0,
    }


def run_single_experiment(
    dataset: str,
    n_positive_classes: int,
    weight_decay: float,
    seed: int,
    epochs: int = 50,
    lr: float = 1e-3,
    compute_grad_every: int = 5,
    store_logits_every: int = 10,
    log_dir: str = None,
    optimizer_type: str = 'adamw'
) -> Dict:
    """
    Run a single experiment comparing BCE vs BCEWithLogits.

    Returns results for both loss functions.
    """
    set_seed(seed)

    # Load data
    if dataset == 'mnist':
        train_loader, test_loader, actual_ratio = load_mnist_by_digit_grouping(
            n_positive_digits=n_positive_classes, seed=seed
        )
        input_dim = 784
    elif dataset == 'cifar10':
        train_loader, test_loader, actual_ratio = load_cifar10_by_class_grouping(
            n_positive_classes=n_positive_classes, seed=seed
        )
        input_dim = 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    results = {}

    for use_bcewl in [True, False]:
        loss_name = 'BCEWithLogits' if use_bcewl else 'BCE'

        # TensorBoard writer
        writer = None
        if log_dir:
            run_name = f"{dataset}_pos{n_positive_classes}_wd{weight_decay}_seed{seed}_{loss_name}"
            writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

        set_seed(seed)  # Reset seed for fair comparison
        model = MLP(input_dim=input_dim).to(DEVICE)

        # Create optimizer based on type
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        if use_bcewl:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            criterion = nn.BCELoss(reduction='none')

        history = defaultdict(list)
        logit_snapshots = {}

        for epoch in range(epochs):
            # Training
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, use_bcewl, DEVICE
            )

            history['train_loss'].append(train_metrics['loss'])
            history['train_grad_mean'].append(train_metrics['grad_mean'])
            history['train_grad_std'].append(train_metrics['grad_std'])
            history['train_grad_max'].append(train_metrics['grad_max'])

            # Evaluation
            test_metrics = compute_test_metrics(
                model, test_loader, criterion, use_bcewl, DEVICE
            )

            # Store test metrics
            for key in ['loss', 'accuracy', 'acc_pos', 'acc_neg', 'brier_score',
                       'logit_mean', 'logit_std', 'logit_max_abs',
                       'logit_pos_mean', 'logit_neg_mean',
                       'total_wrong', 'false_positives', 'false_negatives',
                       'wrong_extreme_count', 'wrong_mean_logit']:
                history[f'test_{key}'].append(test_metrics[key])

            # Store logit snapshots periodically
            if epoch % store_logits_every == 0 or epoch == epochs - 1:
                logit_snapshots[epoch] = {
                    'logits': test_metrics['logits'],
                    'targets': test_metrics['targets'],
                    'probs': test_metrics['probs'],
                }

            # Per-class gradient analysis FROM TRAINING DATA
            if epoch % compute_grad_every == 0 or epoch == epochs - 1:
                grad_stats = compute_per_class_gradients_from_training(
                    model, train_loader, criterion, use_bcewl, DEVICE
                )
                history['grad_pos_mean'].append(grad_stats['pos_mean'])
                history['grad_pos_std'].append(grad_stats['pos_std'])
                history['grad_neg_mean'].append(grad_stats['neg_mean'])
                history['grad_neg_std'].append(grad_stats['neg_std'])
                history['grad_pos_zero_frac'].append(grad_stats['pos_zero_frac'])
                history['grad_neg_zero_frac'].append(grad_stats['neg_zero_frac'])
                history['grad_epoch'].append(epoch)

            # === TENSORBOARD LOGGING ===
            if writer:
                # Loss
                writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                writer.add_scalar('Loss/test', test_metrics['loss'], epoch)

                # Accuracy
                writer.add_scalar('Accuracy/test', test_metrics['accuracy'], epoch)
                writer.add_scalar('Accuracy/test_positive_class', test_metrics['acc_pos'], epoch)
                writer.add_scalar('Accuracy/test_negative_class', test_metrics['acc_neg'], epoch)
                writer.add_scalar('Accuracy/class_gap',
                                 test_metrics['acc_neg'] - test_metrics['acc_pos'], epoch)

                # Calibration
                writer.add_scalar('Calibration/brier_score', test_metrics['brier_score'], epoch)

                # Logit statistics
                writer.add_scalar('Logits/mean', test_metrics['logit_mean'], epoch)
                writer.add_scalar('Logits/std', test_metrics['logit_std'], epoch)
                writer.add_scalar('Logits/max_abs', test_metrics['logit_max_abs'], epoch)
                writer.add_scalar('Logits/positive_class_mean', test_metrics['logit_pos_mean'], epoch)
                writer.add_scalar('Logits/negative_class_mean', test_metrics['logit_neg_mean'], epoch)

                # Wrong predictions
                writer.add_scalar('Errors/total_wrong', test_metrics['total_wrong'], epoch)
                writer.add_scalar('Errors/false_positives', test_metrics['false_positives'], epoch)
                writer.add_scalar('Errors/false_negatives', test_metrics['false_negatives'], epoch)
                writer.add_scalar('Errors/extreme_wrong_count', test_metrics['wrong_extreme_count'], epoch)
                writer.add_scalar('Errors/wrong_mean_logit', test_metrics['wrong_mean_logit'], epoch)

                # Gradient statistics (batch-level from training)
                writer.add_scalar('Gradients/batch_mean', train_metrics['grad_mean'], epoch)
                writer.add_scalar('Gradients/batch_std', train_metrics['grad_std'], epoch)
                writer.add_scalar('Gradients/batch_max', train_metrics['grad_max'], epoch)

                # Per-class gradients (when computed)
                if epoch % compute_grad_every == 0 or epoch == epochs - 1:
                    writer.add_scalar('Gradients/positive_class_mean', grad_stats['pos_mean'], epoch)
                    writer.add_scalar('Gradients/negative_class_mean', grad_stats['neg_mean'], epoch)
                    writer.add_scalar('Gradients/positive_class_zero_frac', grad_stats['pos_zero_frac'], epoch)
                    writer.add_scalar('Gradients/negative_class_zero_frac', grad_stats['neg_zero_frac'], epoch)

                # Logit histograms (periodically - more expensive)
                if epoch % store_logits_every == 0 or epoch == epochs - 1:
                    logits = test_metrics['logits']
                    targets = test_metrics['targets']
                    probs = test_metrics['probs']

                    writer.add_histogram('Logits/all', logits, epoch)
                    writer.add_histogram('Logits/positive_class', logits[targets == 1], epoch)
                    writer.add_histogram('Logits/negative_class', logits[targets == 0], epoch)
                    writer.add_histogram('Probabilities/all', probs, epoch)

                    # Confidence distribution for wrong predictions
                    preds = (probs > 0.5).astype(float)
                    wrong_mask = preds != targets
                    if wrong_mask.sum() > 0:
                        writer.add_histogram('Logits/wrong_predictions', logits[wrong_mask], epoch)

        if writer:
            # Log hyperparameters and final metrics
            hparams = {
                'dataset': dataset,
                'n_positive_classes': n_positive_classes,
                'weight_decay': weight_decay,
                'seed': seed,
                'loss_function': loss_name,
                'lr': lr,
                'epochs': epochs,
            }
            final_metrics = {
                'hparam/accuracy': history['test_accuracy'][-1],
                'hparam/acc_positive': history['test_acc_pos'][-1],
                'hparam/acc_negative': history['test_acc_neg'][-1],
                'hparam/brier_score': history['test_brier_score'][-1],
                'hparam/extreme_wrong': history['test_wrong_extreme_count'][-1],
            }
            writer.add_hparams(hparams, final_metrics)
            writer.close()

        results[loss_name] = {
            'history': dict(history),
            'final': {k: v[-1] for k, v in history.items() if len(v) > 0},
            'logit_snapshots': logit_snapshots,
            'actual_ratio': actual_ratio,
        }

    return results


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistical_tests(all_results: Dict, metric: str = 'test_accuracy') -> Dict:
    """
    Compute statistical significance tests for differences between methods.
    Uses paired t-test since same seeds are used.
    """
    stats_results = {}

    for key in all_results:
        bcewl_values = [r['BCEWithLogits']['final'][metric]
                       for r in all_results[key]]
        bce_values = [r['BCE']['final'][metric]
                     for r in all_results[key]]

        # Paired t-test (same seeds = paired samples)
        t_stat, p_value = stats.ttest_rel(bcewl_values, bce_values)

        # Effect size (Cohen's d for paired samples)
        diff = np.array(bcewl_values) - np.array(bce_values)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        stats_results[key] = {
            'bcewl_mean': np.mean(bcewl_values),
            'bcewl_std': np.std(bcewl_values),
            'bce_mean': np.mean(bce_values),
            'bce_std': np.std(bce_values),
            'diff_mean': np.mean(diff),
            'diff_std': np.std(diff),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
        }

    return stats_results


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def _run_experiment_wrapper(args):
    """Wrapper for parallel execution."""
    dataset, n_positive, weight_decay, seed, epochs, log_dir, optimizer_type, lr = args
    result = run_single_experiment(
        dataset=dataset,
        n_positive_classes=n_positive,
        weight_decay=weight_decay,
        seed=seed,
        epochs=epochs,
        lr=lr,
        log_dir=log_dir,
        optimizer_type=optimizer_type,
    )
    return (n_positive, weight_decay, seed, result)


def run_full_experiment(
    dataset: str,
    seeds: List[int],
    n_positive_list: List[int],
    weight_decays: List[float],
    epochs: int = 50,
    lr: float = 1e-3,
    n_workers: int = None,
    log_dir: str = 'tensorboard/runs_v2',
    optimizer_type: str = 'adamw'
) -> Dict:
    """Run full grid experiment with parallel execution."""

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    # Create log directory
    if log_dir:
        Path(log_dir).mkdir(exist_ok=True)

    experiments = [
        (dataset, n_positive, wd, seed, epochs, log_dir, optimizer_type, lr)
        for n_positive in n_positive_list
        for wd in weight_decays
        for seed in seeds
    ]

    all_results = {}
    for n_positive in n_positive_list:
        for wd in weight_decays:
            key = f"pos_{n_positive}_wd_{wd}"
            all_results[key] = []

    print(f"Running {len(experiments)} experiments with {n_workers} workers...")

    if n_workers == 1:
        for args in tqdm(experiments, desc="Experiments"):
            n_positive, wd, seed, result = _run_experiment_wrapper(args)
            key = f"pos_{n_positive}_wd_{wd}"
            all_results[key].append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_run_experiment_wrapper, experiments),
                total=len(experiments),
                desc="Experiments"
            ))

        for n_positive, wd, seed, result in results:
            key = f"pos_{n_positive}_wd_{wd}"
            all_results[key].append(result)

    return all_results


# =============================================================================
# AGGREGATION AND REPORTING
# =============================================================================

def aggregate_results(all_results: Dict) -> Dict:
    """Aggregate results across seeds with statistics."""
    aggregated = {}

    for key, seed_results in all_results.items():
        aggregated[key] = {}

        for loss_name in ['BCEWithLogits', 'BCE']:
            finals = [r[loss_name]['final'] for r in seed_results]

            agg = {}
            for metric in finals[0].keys():
                values = [f[metric] for f in finals]
                agg[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': [float(v) for v in values],
                }

            aggregated[key][loss_name] = agg

    return aggregated


def print_results_with_significance(aggregated: Dict, stat_tests: Dict):
    """Print results with statistical significance markers."""

    print("\n" + "=" * 110)
    print("RESULTS WITH STATISTICAL SIGNIFICANCE")
    print("=" * 110)
    print(f"{'Config':<20} | {'BCEWithLogits':^20} | {'BCE':^20} | {'Diff':^12} | {'p-value':^10} | {'Sig?':^6}")
    print("-" * 110)

    for key in sorted(aggregated.keys()):
        st = stat_tests.get(key, {})
        bcewl_acc = aggregated[key]['BCEWithLogits']['test_accuracy']
        bce_acc = aggregated[key]['BCE']['test_accuracy']

        diff = st.get('diff_mean', bcewl_acc['mean'] - bce_acc['mean'])
        p_val = st.get('p_value', 1.0)
        sig = '*' if st.get('significant_005', False) else ''
        sig = '**' if st.get('significant_001', False) else sig

        print(f"{key:<20} | "
              f"{bcewl_acc['mean']:.4f} ± {bcewl_acc['std']:.4f}   | "
              f"{bce_acc['mean']:.4f} ± {bce_acc['std']:.4f}   | "
              f"{diff:>+.4f}     | "
              f"{p_val:^10.4f} | "
              f"{sig:^6}")

    print("-" * 110)
    print("* p < 0.05, ** p < 0.01")


def print_detailed_comparison(aggregated: Dict, stat_tests: Dict):
    """Print detailed comparison across multiple metrics."""

    metrics_to_compare = [
        ('test_accuracy', 'Accuracy'),
        ('test_acc_pos', 'Minority Acc'),
        ('test_acc_neg', 'Majority Acc'),
        ('test_brier_score', 'Brier Score'),
        ('test_wrong_extreme_count', 'Extreme Wrong'),
        ('test_false_negatives', 'False Negatives'),
    ]

    for metric_key, metric_name in metrics_to_compare:
        print(f"\n{'='*80}")
        print(f"{metric_name.upper()}")
        print(f"{'='*80}")

        # Compute stats for this metric
        metric_stats = compute_statistical_tests(
            {k: [{'BCEWithLogits': {'final': {metric_key: v['BCEWithLogits'][metric_key]['values'][i]}},
                  'BCE': {'final': {metric_key: v['BCE'][metric_key]['values'][i]}}}
                 for i in range(len(v['BCEWithLogits'][metric_key]['values']))]
             for k, v in aggregated.items()},
            metric=metric_key
        )

        print(f"{'Config':<20} | {'BCEWithLogits':^18} | {'BCE':^18} | {'p-value':^10}")
        print("-" * 80)

        for key in sorted(aggregated.keys()):
            bcewl = aggregated[key]['BCEWithLogits'][metric_key]
            bce = aggregated[key]['BCE'][metric_key]
            p_val = metric_stats.get(key, {}).get('p_value', 1.0)

            print(f"{key:<20} | "
                  f"{bcewl['mean']:>8.4f} ± {bcewl['std']:<6.4f} | "
                  f"{bce['mean']:>8.4f} ± {bce['std']:<6.4f} | "
                  f"{p_val:^10.4f}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_results: Dict, aggregated: Dict, save_dir: str = 'results/plots_v2'):
    """Create comprehensive visualizations."""
    Path(save_dir).mkdir(exist_ok=True)

    n_positive_list = sorted(set(int(k.split('_')[1]) for k in all_results.keys()))
    wds = sorted(set(float(k.split('_')[3]) for k in all_results.keys()))

    # 1. Accuracy curves with error bands
    fig, axes = plt.subplots(len(n_positive_list), len(wds),
                             figsize=(5*len(wds), 4*len(n_positive_list)))
    if len(n_positive_list) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, n_pos in enumerate(n_positive_list):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"pos_{n_pos}_wd_{wd}"

            for loss_name, color in [('BCEWithLogits', 'blue'), ('BCE', 'orange')]:
                all_accs = [r[loss_name]['history']['test_accuracy']
                           for r in all_results[key]]
                mean_acc = np.mean(all_accs, axis=0)
                std_acc = np.std(all_accs, axis=0)
                epochs = range(1, len(mean_acc) + 1)

                ax.plot(epochs, mean_acc, color=color, linewidth=2, label=loss_name)
                ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                               color=color, alpha=0.2)

            ratio = n_pos * 10
            ax.set_title(f'{ratio}% Positive, WD={wd}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Test Accuracy')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_curves.png', dpi=150)
    plt.close()

    # 2. Per-class gradient evolution
    fig, axes = plt.subplots(len(n_positive_list), len(wds),
                             figsize=(5*len(wds), 4*len(n_positive_list)))
    if len(n_positive_list) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, n_pos in enumerate(n_positive_list):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"pos_{n_pos}_wd_{wd}"

            for loss_name, color, marker in [('BCEWithLogits', 'blue', 'o'),
                                              ('BCE', 'orange', 's')]:
                all_pos_grads = [r[loss_name]['history']['grad_pos_mean']
                                for r in all_results[key]]
                all_neg_grads = [r[loss_name]['history']['grad_neg_mean']
                                for r in all_results[key]]
                epochs = all_results[key][0][loss_name]['history']['grad_epoch']

                mean_pos = np.mean(all_pos_grads, axis=0)
                mean_neg = np.mean(all_neg_grads, axis=0)

                ax.plot(epochs, mean_pos, marker=marker, color=color, linestyle='-',
                       label=f'{loss_name} (pos)', markersize=4)
                ax.plot(epochs, mean_neg, marker=marker, color=color, linestyle='--',
                       label=f'{loss_name} (neg)', markersize=4, alpha=0.6)

            ratio = n_pos * 10
            ax.set_title(f'{ratio}% Positive, WD={wd}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Magnitude')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/gradient_by_class.png', dpi=150)
    plt.close()

    # 3. Logit distributions
    fig, axes = plt.subplots(len(n_positive_list), len(wds),
                             figsize=(5*len(wds), 4*len(n_positive_list)))
    if len(n_positive_list) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, n_pos in enumerate(n_positive_list):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"pos_{n_pos}_wd_{wd}"

            result = all_results[key][0]  # First seed
            final_epoch = max(result['BCEWithLogits']['logit_snapshots'].keys())

            for loss_name, color in [('BCEWithLogits', 'blue'), ('BCE', 'orange')]:
                snapshot = result[loss_name]['logit_snapshots'][final_epoch]
                logits = snapshot['logits']
                ax.hist(logits, bins=50, alpha=0.5, label=loss_name,
                       color=color, density=True)

            ax.axvline(x=17, color='red', linestyle='--', alpha=0.5, label='|logit|=17')
            ax.axvline(x=-17, color='red', linestyle='--', alpha=0.5)

            ratio = n_pos * 10
            ax.set_title(f'{ratio}% Positive, WD={wd} (final)')
            ax.set_xlabel('Logit')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.set_xlim(-60, 60)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/logit_distributions.png', dpi=150)
    plt.close()

    # 4. Statistical significance summary
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['test_accuracy', 'test_acc_pos', 'test_brier_score']
    x = np.arange(len(all_results))
    width = 0.25

    for idx, metric in enumerate(metrics):
        diffs = []
        for key in sorted(all_results.keys()):
            bcewl = aggregated[key]['BCEWithLogits'][metric]['mean']
            bce = aggregated[key]['BCE'][metric]['mean']
            diffs.append(bcewl - bce)

        ax.bar(x + idx * width, diffs, width, label=metric.replace('test_', ''))

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('BCEWithLogits - BCE')
    ax.set_title('Metric Differences (positive = BCEWithLogits better)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace('_', '\n') for k in sorted(all_results.keys())],
                       fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/metric_differences.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {save_dir}/")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Configuration
    DATASET = 'cifar10'  # 'mnist' or 'cifar10'
    OPTIMIZER = 'sgd'    # 'adamw' or 'sgd' - use 'sgd' to test without adaptive LR
    LR = 0.01 if OPTIMIZER == 'sgd' else 1e-3  # SGD typically needs higher LR

    # 15 seeds for 95% statistical power (180 paired comparisons)
    SEEDS = [
        42, 123, 456, 789, 1337,        # original 5
        2024, 999, 314, 271, 161,       # +5 for 80% power
        577, 1618, 2718, 1414, 1732,    # +5 for 95% power
    ]
    N_POSITIVE_LIST = [1, 2, 3, 5]  # 10%, 20%, 30%, 50%
    WEIGHT_DECAYS = [0.0, 0.01, 0.1]
    EPOCHS = 50

    # Add suffix for non-default optimizer to avoid overwriting AdamW results
    SUFFIX = f'_{OPTIMIZER}' if OPTIMIZER != 'adamw' else ''

    print(f"\nExperiment configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Optimizer: {OPTIMIZER} (lr={LR})")
    print(f"  Seeds: {SEEDS}")
    print(f"  Positive class counts: {N_POSITIVE_LIST} ({[n*10 for n in N_POSITIVE_LIST]}%)")
    print(f"  Weight decays: {WEIGHT_DECAYS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total runs: {len(SEEDS) * len(N_POSITIVE_LIST) * len(WEIGHT_DECAYS) * 2}")

    LOG_DIR = f'tensorboard/runs_v2{SUFFIX}'

    # Run experiment
    all_results = run_full_experiment(
        dataset=DATASET,
        seeds=SEEDS,
        n_positive_list=N_POSITIVE_LIST,
        weight_decays=WEIGHT_DECAYS,
        epochs=EPOCHS,
        lr=LR,
        log_dir=LOG_DIR,
        optimizer_type=OPTIMIZER,
    )

    # Aggregate
    aggregated = aggregate_results(all_results)

    # Statistical tests
    stat_tests = compute_statistical_tests(all_results, metric='test_accuracy')

    # Print results
    print_results_with_significance(aggregated, stat_tests)
    print_detailed_comparison(aggregated, stat_tests)

    # Plot
    plot_results(all_results, aggregated, save_dir=f'results/plots_v2{SUFFIX}')

    # Save results
    output = {
        'all_results': all_results,
        'aggregated': aggregated,
        'statistical_tests': stat_tests,
        'config': {
            'dataset': DATASET,
            'seeds': SEEDS,
            'n_positive_list': N_POSITIVE_LIST,
            'weight_decays': WEIGHT_DECAYS,
            'epochs': EPOCHS,
            'optimizer': OPTIMIZER,
            'lr': LR,
        }
    }

    results_file = f'results/experiment_results_v2{SUFFIX}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(output, f)

    print("\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - results/plots_v2{SUFFIX}/")
    print(f"  - {LOG_DIR}/ (TensorBoard logs)")
    print(f"\nTo view TensorBoard: tensorboard --logdir={LOG_DIR}")
