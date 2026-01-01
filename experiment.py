"""
BCE vs BCEWithLogitsLoss: Multi-seed experiment
Investigating the effect of class imbalance and weight decay on numerical stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# def get_device():
#     if torch.cuda.is_available():
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         return torch.device("mps")
#     else:
#         return torch.device("cpu")

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist_binary(minority_ratio=0.1, batch_size=128, seed=42):
    """
    Load MNIST as binary classification with controlled class imbalance.

    minority_ratio: fraction of positive class (0.1 = 10% positive, 90% negative)
    Returns: train_loader, test_loader, actual_ratio
    """
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)

    # Binary: digits 0-4 = positive (1), digits 5-9 = negative (0)
    X_train = train_data.data.float().view(-1, 28*28) / 255.0
    y_train_raw = train_data.targets
    y_train = torch.tensor([1 if y < 5 else 0 for y in y_train_raw])

    X_test = test_data.data.float().view(-1, 28*28) / 255.0
    y_test_raw = test_data.targets
    y_test = torch.tensor([1 if y < 5 else 0 for y in y_test_raw])

    # Normalize
    X_train = (X_train - 0.1307) / 0.3081
    X_test = (X_test - 0.1307) / 0.3081

    # Subsample to achieve desired minority ratio
    pos_idx = (y_train == 1).nonzero(as_tuple=True)[0].numpy()
    neg_idx = (y_train == 0).nonzero(as_tuple=True)[0].numpy()

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    if minority_ratio <= 0.5:
        # Positive class is minority
        n_neg = len(neg_idx)
        n_pos = int(n_neg * minority_ratio / (1 - minority_ratio))
        n_pos = min(n_pos, len(pos_idx))
        if n_pos < int(n_neg * minority_ratio / (1 - minority_ratio)):
            n_neg = int(n_pos * (1 - minority_ratio) / minority_ratio)
    else:
        # Negative class is minority
        n_pos = len(pos_idx)
        n_neg = int(n_pos * (1 - minority_ratio) / minority_ratio)
        n_neg = min(n_neg, len(neg_idx))
        if n_neg < int(n_pos * (1 - minority_ratio) / minority_ratio):
            n_pos = int(n_neg * minority_ratio / (1 - minority_ratio))

    selected_pos = pos_idx[:n_pos]
    selected_neg = neg_idx[:n_neg]
    selected_idx = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(selected_idx)

    X_train_sub = X_train[selected_idx]
    y_train_sub = y_train[selected_idx]

    actual_ratio = y_train_sub.float().mean().item()

    train_loader = DataLoader(
        list(zip(X_train_sub, y_train_sub)),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        list(zip(X_test, y_test)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader, actual_ratio


class MLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

    def get_final_layer(self):
        return self.net[-1]


def compute_per_class_gradients(model, X, y, criterion, use_bce_with_logits, device):
    """Compute gradient magnitudes separately for each class."""
    model.train()

    grad_pos = []
    grad_neg = []

    # Process samples individually to get per-sample gradients
    for i in range(len(X)):
        xi = X[i:i+1].to(device)
        yi = y[i:i+1].float().to(device)

        model.zero_grad()
        logit = model(xi).squeeze()

        if use_bce_with_logits:
            loss = criterion(logit.unsqueeze(0), yi).mean()
        else:
            prob = torch.sigmoid(logit)
            loss = criterion(prob.unsqueeze(0), yi).mean()

        loss.backward()

        grad_mag = model.get_final_layer().weight.grad.abs().mean().item()

        if y[i].item() == 1:
            grad_pos.append(grad_mag)
        else:
            grad_neg.append(grad_mag)

    return {
        'pos_mean': np.mean(grad_pos) if grad_pos else 0,
        'pos_std': np.std(grad_pos) if grad_pos else 0,
        'neg_mean': np.mean(grad_neg) if grad_neg else 0,
        'neg_std': np.std(grad_neg) if grad_neg else 0,
        'pos_zero_frac': np.mean([g < 1e-7 for g in grad_pos]) if grad_pos else 0,
        'neg_zero_frac': np.mean([g < 1e-7 for g in grad_neg]) if grad_neg else 0,
    }


def analyze_wrong_predictions(logits, targets, preds):
    """Detailed analysis of wrong predictions."""
    wrong = preds != targets

    if wrong.sum() == 0:
        return {
            'total_wrong': 0,
            'false_pos': 0, 'false_neg': 0,
            'wrong_mean_logit': 0, 'wrong_max_logit': 0,
            'wrong_extreme_count': 0,
            'fp_mean_logit': 0, 'fn_mean_logit': 0,
            'fp_extreme': 0, 'fn_extreme': 0,
        }

    wrong_logits = logits[wrong]

    # False positives: target=0, pred=1 (positive logit when should be negative)
    false_pos = wrong & (targets == 0)
    # False negatives: target=1, pred=0 (negative logit when should be positive)
    false_neg = wrong & (targets == 1)

    fp_logits = logits[false_pos] if false_pos.sum() > 0 else torch.tensor([0.])
    fn_logits = logits[false_neg] if false_neg.sum() > 0 else torch.tensor([0.])

    return {
        'total_wrong': wrong.sum().item(),
        'false_pos': false_pos.sum().item(),
        'false_neg': false_neg.sum().item(),
        'wrong_mean_logit': wrong_logits.abs().mean().item(),
        'wrong_max_logit': wrong_logits.abs().max().item(),
        'wrong_extreme_count': (wrong_logits.abs() > 15).sum().item(),
        'fp_mean_logit': fp_logits.abs().mean().item(),
        'fn_mean_logit': fn_logits.abs().mean().item(),
        'fp_extreme': (fp_logits.abs() > 15).sum().item(),
        'fn_extreme': (fn_logits.abs() > 15).sum().item(),
    }


def get_logit_distribution_stats(logits, targets):
    """Get statistics about logit distribution by class."""
    pos_logits = logits[targets == 1]
    neg_logits = logits[targets == 0]

    return {
        # Positive class (should have positive logits)
        'pos_mean': pos_logits.mean().item(),
        'pos_std': pos_logits.std().item(),
        'pos_min': pos_logits.min().item(),
        'pos_max': pos_logits.max().item(),
        'pos_median': pos_logits.median().item(),
        'pos_extreme_neg': (pos_logits < -15).sum().item(),  # Very wrong
        'pos_extreme_pos': (pos_logits > 15).sum().item(),   # Very confident correct

        # Negative class (should have negative logits)
        'neg_mean': neg_logits.mean().item(),
        'neg_std': neg_logits.std().item(),
        'neg_min': neg_logits.min().item(),
        'neg_max': neg_logits.max().item(),
        'neg_median': neg_logits.median().item(),
        'neg_extreme_neg': (neg_logits < -15).sum().item(),  # Very confident correct
        'neg_extreme_pos': (neg_logits > 15).sum().item(),   # Very wrong

        # Overall
        'all_mean': logits.mean().item(),
        'all_std': logits.std().item(),
        'all_max_abs': logits.abs().max().item(),
    }


def train_epoch(model, train_loader, optimizer, criterion, use_bce_with_logits, device):
    """Train for one epoch, return metrics."""
    model.train()
    total_loss = 0
    grad_magnitudes = []
    all_logits = []
    all_targets = []

    for X, y in train_loader:
        X, y = X.to(device), y.float().to(device)

        optimizer.zero_grad()
        logits = model(X).squeeze()

        if use_bce_with_logits:
            loss = criterion(logits, y).mean()
        else:
            probs = torch.sigmoid(logits)
            loss = criterion(probs, y).mean()

        loss.backward()

        final_layer = model.get_final_layer()
        if final_layer.weight.grad is not None:
            grad_magnitudes.append(final_layer.weight.grad.abs().mean().item())

        optimizer.step()
        total_loss += loss.item()

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return {
        'loss': total_loss / len(train_loader),
        'grad_mean': np.mean(grad_magnitudes) if grad_magnitudes else 0,
        'grad_std': np.std(grad_magnitudes) if grad_magnitudes else 0,
        'logits': all_logits,
        'targets': all_targets,
    }


def evaluate(model, test_loader, criterion, use_bce_with_logits, device):
    """Evaluate model, return detailed metrics."""
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.float().to(device)
            logits = model(X).squeeze()

            if use_bce_with_logits:
                total_loss += criterion(logits, y).mean().item()
            else:
                probs = torch.sigmoid(logits)
                total_loss += criterion(probs, y).mean().item()

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_preds = (torch.sigmoid(all_logits) > 0.5).float()

    # Basic metrics
    correct = (all_preds == all_targets).float()
    pos_mask = all_targets == 1
    neg_mask = all_targets == 0

    # Logit distribution
    logit_stats = get_logit_distribution_stats(all_logits, all_targets)

    # Wrong prediction analysis
    wrong_analysis = analyze_wrong_predictions(all_logits, all_targets, all_preds)

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': correct.mean().item(),
        'acc_pos': correct[pos_mask].mean().item() if pos_mask.sum() > 0 else 0,
        'acc_neg': correct[neg_mask].mean().item() if neg_mask.sum() > 0 else 0,
        'logits': all_logits,
        'targets': all_targets,
        'preds': all_preds,
        **logit_stats,
        **wrong_analysis,
    }


def run_single_experiment(minority_ratio, weight_decay, seed, epochs=50, lr=1e-3,
                          log_dir=None, compute_per_class_grad_every=10):
    """
    Run a single experiment with given parameters.
    Returns results for both BCE and BCEWithLogits.
    """
    set_seed(seed)

    train_loader, test_loader, actual_ratio = load_mnist_binary(
        minority_ratio=minority_ratio, seed=seed
    )

    results = {}

    for use_bcewl in [True, False]:
        loss_name = 'BCEWithLogits' if use_bcewl else 'BCE'

        # TensorBoard writer
        writer = None
        if log_dir:
            run_name = f"ratio{minority_ratio}_wd{weight_decay}_seed{seed}_{loss_name}"
            writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

        set_seed(seed)
        model = MLP().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if use_bcewl:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            criterion = nn.BCELoss(reduction='none')

        history = defaultdict(list)

        # Store logit distributions at key epochs
        logit_snapshots = {}

        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, use_bcewl, DEVICE
            )
            eval_metrics = evaluate(
                model, test_loader, criterion, use_bcewl, DEVICE
            )

            # Store basic metrics
            history['train_loss'].append(train_metrics['loss'])
            history['grad_mean'].append(train_metrics['grad_mean'])
            history['grad_std'].append(train_metrics['grad_std'])

            # Store eval metrics (excluding tensors)
            for k, v in eval_metrics.items():
                if not isinstance(v, torch.Tensor):
                    history[f'test_{k}'].append(v)

            # Store logit snapshots at key epochs
            if epoch in [0, epochs//4, epochs//2, 3*epochs//4, epochs-1]:
                logit_snapshots[epoch] = {
                    'logits': eval_metrics['logits'].numpy(),
                    'targets': eval_metrics['targets'].numpy(),
                    'preds': eval_metrics['preds'].numpy(),
                }

            # Compute per-class gradients periodically (expensive)
            if epoch % compute_per_class_grad_every == 0:
                # Sample a batch for gradient analysis
                sample_X, sample_y = next(iter(test_loader))
                sample_X, sample_y = sample_X[:64], sample_y[:64]  # Limit for speed

                grad_stats = compute_per_class_gradients(
                    model, sample_X, sample_y, criterion, use_bcewl, DEVICE
                )
                for k, v in grad_stats.items():
                    history[f'grad_{k}'].append(v)
                history['grad_epoch'].append(epoch)

            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                writer.add_scalar('Loss/test', eval_metrics['loss'], epoch)
                writer.add_scalar('Accuracy/test', eval_metrics['accuracy'], epoch)
                writer.add_scalar('Accuracy/test_pos', eval_metrics['acc_pos'], epoch)
                writer.add_scalar('Accuracy/test_neg', eval_metrics['acc_neg'], epoch)

                # Logit stats
                writer.add_scalar('Logits/pos_mean', eval_metrics['pos_mean'], epoch)
                writer.add_scalar('Logits/neg_mean', eval_metrics['neg_mean'], epoch)
                writer.add_scalar('Logits/all_max_abs', eval_metrics['all_max_abs'], epoch)

                # Wrong predictions
                writer.add_scalar('Wrong/total', eval_metrics['total_wrong'], epoch)
                writer.add_scalar('Wrong/false_pos', eval_metrics['false_pos'], epoch)
                writer.add_scalar('Wrong/false_neg', eval_metrics['false_neg'], epoch)
                writer.add_scalar('Wrong/extreme_count', eval_metrics['wrong_extreme_count'], epoch)

                # Gradient stats
                writer.add_scalar('Gradient/mean', train_metrics['grad_mean'], epoch)

                # Log logit histograms periodically
                if epoch % 10 == 0:
                    writer.add_histogram('Logits/positive_class',
                                        eval_metrics['logits'][eval_metrics['targets'] == 1], epoch)
                    writer.add_histogram('Logits/negative_class',
                                        eval_metrics['logits'][eval_metrics['targets'] == 0], epoch)

        if writer:
            writer.close()

        results[loss_name] = {
            'history': dict(history),
            'final': {k: v[-1] for k, v in history.items() if len(v) > 0},
            'logit_snapshots': logit_snapshots,
            'actual_ratio': actual_ratio,
        }

    return results


def _run_experiment_wrapper(args):
    """Wrapper for parallel execution."""
    minority_ratio, weight_decay, seed, epochs, log_dir = args
    result = run_single_experiment(
        minority_ratio=minority_ratio,
        weight_decay=weight_decay,
        seed=seed,
        epochs=epochs,
        log_dir=log_dir,
    )
    return (minority_ratio, weight_decay, seed, result)


def run_full_experiment(seeds, minority_ratios, weight_decays, epochs=50, log_dir='runs',
                        n_workers=None):
    """Run full grid experiment with parallel execution."""
    all_results = {}

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues

    Path(log_dir).mkdir(exist_ok=True)

    # Build list of all experiments
    experiments = [
        (minority_ratio, weight_decay, seed, epochs, log_dir)
        for minority_ratio in minority_ratios
        for weight_decay in weight_decays
        for seed in seeds
    ]

    # Initialize result structure
    for minority_ratio in minority_ratios:
        for weight_decay in weight_decays:
            key = f"ratio_{minority_ratio}_wd_{weight_decay}"
            all_results[key] = []

    if n_workers == 1:
        # Sequential execution (for debugging)
        for args in tqdm(experiments, desc="Experiments"):
            _, _, _, result = _run_experiment_wrapper(args)
            minority_ratio, weight_decay, seed = args[0], args[1], args[2]
            key = f"ratio_{minority_ratio}_wd_{weight_decay}"
            all_results[key].append(result)
    else:
        # Parallel execution
        print(f"Running {len(experiments)} experiments with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_run_experiment_wrapper, experiments),
                total=len(experiments),
                desc="Experiments"
            ))

        # Organize results
        for minority_ratio, weight_decay, seed, result in results:
            key = f"ratio_{minority_ratio}_wd_{weight_decay}"
            all_results[key].append(result)

    return all_results


def aggregate_results(all_results):
    """Aggregate results across seeds."""
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


def print_summary_table(aggregated):
    """Print summary tables."""
    ratios = sorted(set(float(k.split('_')[1]) for k in aggregated.keys()))
    wds = sorted(set(float(k.split('_')[3]) for k in aggregated.keys()))

    # Accuracy table
    print("\n" + "=" * 100)
    print("FINAL TEST ACCURACY (mean ± std)")
    print("=" * 100)
    print(f"{'Ratio':<8} | {'WD':<8} | {'BCEWithLogits':^20} | {'BCE':^20} | {'Diff':^10}")
    print("-" * 100)

    for ratio in ratios:
        for wd in wds:
            key = f"ratio_{ratio}_wd_{wd}"
            bcewl = aggregated[key]['BCEWithLogits']['test_accuracy']
            bce = aggregated[key]['BCE']['test_accuracy']
            diff = bcewl['mean'] - bce['mean']
            print(f"{ratio:<8.2f} | {wd:<8.4f} | "
                  f"{bcewl['mean']:.4f} ± {bcewl['std']:.4f}   | "
                  f"{bce['mean']:.4f} ± {bce['std']:.4f}   | "
                  f"{diff:+.4f}")
        print("-" * 100)

    # Wrong predictions with extreme logits
    print("\n" + "=" * 100)
    print("WRONG PREDICTIONS WITH |logit| > 15 (mean ± std)")
    print("=" * 100)
    print(f"{'Ratio':<8} | {'WD':<8} | {'BCEWithLogits':^20} | {'BCE':^20} | {'Diff':^10}")
    print("-" * 100)

    for ratio in ratios:
        for wd in wds:
            key = f"ratio_{ratio}_wd_{wd}"
            bcewl = aggregated[key]['BCEWithLogits']['test_wrong_extreme_count']
            bce = aggregated[key]['BCE']['test_wrong_extreme_count']
            diff = bcewl['mean'] - bce['mean']
            print(f"{ratio:<8.2f} | {wd:<8.4f} | "
                  f"{bcewl['mean']:>6.1f} ± {bcewl['std']:<6.1f}   | "
                  f"{bce['mean']:>6.1f} ± {bce['std']:<6.1f}   | "
                  f"{diff:>+7.1f}")
        print("-" * 100)

    # Max absolute logit
    print("\n" + "=" * 100)
    print("MAX |LOGIT| (mean ± std)")
    print("=" * 100)
    print(f"{'Ratio':<8} | {'WD':<8} | {'BCEWithLogits':^20} | {'BCE':^20} | {'Diff':^10}")
    print("-" * 100)

    for ratio in ratios:
        for wd in wds:
            key = f"ratio_{ratio}_wd_{wd}"
            bcewl = aggregated[key]['BCEWithLogits']['test_all_max_abs']
            bce = aggregated[key]['BCE']['test_all_max_abs']
            diff = bcewl['mean'] - bce['mean']
            print(f"{ratio:<8.2f} | {wd:<8.4f} | "
                  f"{bcewl['mean']:>6.1f} ± {bcewl['std']:<6.1f}   | "
                  f"{bce['mean']:>6.1f} ± {bce['std']:<6.1f}   | "
                  f"{diff:>+7.1f}")
        print("-" * 100)

    # False negatives (minority class errors when minority_ratio < 0.5)
    print("\n" + "=" * 100)
    print("FALSE NEGATIVES - Minority class misclassified (mean ± std)")
    print("=" * 100)
    print(f"{'Ratio':<8} | {'WD':<8} | {'BCEWithLogits':^20} | {'BCE':^20} | {'Diff':^10}")
    print("-" * 100)

    for ratio in ratios:
        for wd in wds:
            key = f"ratio_{ratio}_wd_{wd}"
            bcewl = aggregated[key]['BCEWithLogits']['test_false_neg']
            bce = aggregated[key]['BCE']['test_false_neg']
            diff = bcewl['mean'] - bce['mean']
            print(f"{ratio:<8.2f} | {wd:<8.4f} | "
                  f"{bcewl['mean']:>6.1f} ± {bcewl['std']:<6.1f}   | "
                  f"{bce['mean']:>6.1f} ± {bce['std']:<6.1f}   | "
                  f"{diff:>+7.1f}")
        print("-" * 100)


def plot_results(all_results, save_dir='plots'):
    """Create comprehensive visualizations."""
    Path(save_dir).mkdir(exist_ok=True)

    ratios = sorted(set(float(k.split('_')[1]) for k in all_results.keys()))
    wds = sorted(set(float(k.split('_')[3]) for k in all_results.keys()))

    # 1. Accuracy curves
    fig, axes = plt.subplots(len(ratios), len(wds), figsize=(5*len(wds), 4*len(ratios)))
    if len(ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, ratio in enumerate(ratios):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"ratio_{ratio}_wd_{wd}"

            for loss_name, color in [('BCEWithLogits', 'blue'), ('BCE', 'orange')]:
                all_accs = [r[loss_name]['history']['test_accuracy']
                           for r in all_results[key]]
                for acc in all_accs:
                    ax.plot(acc, color=color, alpha=0.2, linewidth=0.5)
                mean_acc = np.mean(all_accs, axis=0)
                ax.plot(mean_acc, color=color, linewidth=2, label=loss_name)

            ax.set_title(f'Ratio={ratio}, WD={wd}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Test Accuracy')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.85, 1.0)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_curves.png', dpi=150)
    plt.close()

    # 2. Logit distribution evolution (for one seed)
    fig, axes = plt.subplots(len(ratios), len(wds), figsize=(5*len(wds), 4*len(ratios)))
    if len(ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, ratio in enumerate(ratios):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"ratio_{ratio}_wd_{wd}"

            # Use first seed
            result = all_results[key][0]

            # Get final epoch snapshot
            final_epoch = max(result['BCEWithLogits']['logit_snapshots'].keys())

            for loss_name, color in [('BCEWithLogits', 'blue'), ('BCE', 'orange')]:
                snapshot = result[loss_name]['logit_snapshots'][final_epoch]
                logits = snapshot['logits']
                ax.hist(logits, bins=50, alpha=0.5, label=loss_name, color=color, density=True)

            ax.axvline(x=15, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-15, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'Ratio={ratio}, WD={wd} (final)')
            ax.set_xlabel('Logit')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.set_xlim(-50, 50)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/logit_distributions.png', dpi=150)
    plt.close()

    # 3. Wrong predictions breakdown
    fig, axes = plt.subplots(2, len(ratios), figsize=(5*len(ratios), 8))
    if len(ratios) == 1:
        axes = axes.reshape(-1, 1)

    for i, ratio in enumerate(ratios):
        # False positives
        ax = axes[0, i]
        x = np.arange(len(wds))
        width = 0.35

        bcewl_fp = [aggregated[f"ratio_{ratio}_wd_{wd}"]['BCEWithLogits']['test_false_pos']['mean']
                    for wd in wds]
        bce_fp = [aggregated[f"ratio_{ratio}_wd_{wd}"]['BCE']['test_false_pos']['mean']
                  for wd in wds]

        ax.bar(x - width/2, bcewl_fp, width, label='BCEWithLogits', color='blue', alpha=0.7)
        ax.bar(x + width/2, bce_fp, width, label='BCE', color='orange', alpha=0.7)
        ax.set_title(f'False Positives (Ratio={ratio})')
        ax.set_xticks(x)
        ax.set_xticklabels([str(wd) for wd in wds])
        ax.set_xlabel('Weight Decay')
        ax.legend()

        # False negatives
        ax = axes[1, i]
        bcewl_fn = [aggregated[f"ratio_{ratio}_wd_{wd}"]['BCEWithLogits']['test_false_neg']['mean']
                    for wd in wds]
        bce_fn = [aggregated[f"ratio_{ratio}_wd_{wd}"]['BCE']['test_false_neg']['mean']
                  for wd in wds]

        ax.bar(x - width/2, bcewl_fn, width, label='BCEWithLogits', color='blue', alpha=0.7)
        ax.bar(x + width/2, bce_fn, width, label='BCE', color='orange', alpha=0.7)
        ax.set_title(f'False Negatives (Ratio={ratio})')
        ax.set_xticks(x)
        ax.set_xticklabels([str(wd) for wd in wds])
        ax.set_xlabel('Weight Decay')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/wrong_predictions.png', dpi=150)
    plt.close()

    # 4. Per-class gradient analysis
    fig, axes = plt.subplots(len(ratios), len(wds), figsize=(5*len(wds), 4*len(ratios)))
    if len(ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(wds) == 1:
        axes = axes.reshape(-1, 1)

    for i, ratio in enumerate(ratios):
        for j, wd in enumerate(wds):
            ax = axes[i, j]
            key = f"ratio_{ratio}_wd_{wd}"

            for loss_name, color, marker in [('BCEWithLogits', 'blue', 'o'), ('BCE', 'orange', 's')]:
                # Average across seeds
                all_pos_grads = [r[loss_name]['history']['grad_pos_mean'] for r in all_results[key]]
                all_neg_grads = [r[loss_name]['history']['grad_neg_mean'] for r in all_results[key]]
                epochs = all_results[key][0][loss_name]['history']['grad_epoch']

                mean_pos = np.mean(all_pos_grads, axis=0)
                mean_neg = np.mean(all_neg_grads, axis=0)

                ax.plot(epochs, mean_pos, marker=marker, color=color, linestyle='-',
                       label=f'{loss_name} (pos)', markersize=4)
                ax.plot(epochs, mean_neg, marker=marker, color=color, linestyle='--',
                       label=f'{loss_name} (neg)', markersize=4, alpha=0.6)

            ax.set_title(f'Ratio={ratio}, WD={wd}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Magnitude')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/gradient_by_class.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {save_dir}/")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # Experiment parameters
    SEEDS = [42, 123, 456, 789, 1337]
    MINORITY_RATIOS = [0.1, 0.3, 0.5]
    WEIGHT_DECAYS = [0.0, 0.001, 0.01, 0.1]
    EPOCHS = 50

    print(f"\nExperiment configuration:")
    print(f"  Seeds: {SEEDS}")
    print(f"  Minority ratios: {MINORITY_RATIOS}")
    print(f"  Weight decays: {WEIGHT_DECAYS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total runs: {len(SEEDS) * len(MINORITY_RATIOS) * len(WEIGHT_DECAYS) * 2}")

    # Run experiment
    all_results = run_full_experiment(
        seeds=SEEDS,
        minority_ratios=MINORITY_RATIOS,
        weight_decays=WEIGHT_DECAYS,
        epochs=EPOCHS,
        log_dir='runs'
    )

    # Aggregate and print
    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)

    # Plot
    plot_results(all_results, save_dir='plots')

    # Save results with pickle (preserves numpy arrays properly)
    with open('experiment_results.pkl', 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'aggregated': aggregated,
            'config': {
                'seeds': SEEDS,
                'minority_ratios': MINORITY_RATIOS,
                'weight_decays': WEIGHT_DECAYS,
                'epochs': EPOCHS,
            }
        }, f)

    print("\nResults saved to:")
    print("  - experiment_results.pkl (full data)")
    print("  - runs/ (TensorBoard logs)")
    print("  - plots/ (static plots)")
    print("\nTo view TensorBoard: tensorboard --logdir=runs")
