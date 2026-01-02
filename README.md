# BCE vs BCEWithLogitsLoss: A Deep Dive into Numerical Stability

An empirical investigation into when and why `BCEWithLogitsLoss` outperforms `Sigmoid + BCELoss` in PyTorch.

## TL;DR

| Network Type | BCEWithLogitsLoss | BCELoss | Verdict |
|--------------|-------------------|---------|---------|
| Shallow (MLP) | 97% | 97% | **No difference** |
| Deep (VGG16) | 92% | 50% | **BCEWithLogitsLoss wins** |
| Deep (VGG16) + proper config | 92% | 92% | **No difference** |

**Bottom line:** BCEWithLogitsLoss is strictly better because it gracefully handles edge cases where BCELoss catastrophically fails. In normal operation, they're equivalent.

## Key Findings

### Shallow Networks: No Difference

On MLPs with MNIST/CIFAR-10, after 180 paired comparisons across 15 seeds:
- BCEWithLogitsLoss win rate: **50.6%** (a coin flip)
- p-value: **0.47** (not significant)
- Effect size: **d = 0.01** (negligible)

An initial "significant" result (p=0.026) with 5 seeds was revealed to be a statistical fluke when we tripled the sample size.

### Deep Networks: BCELoss Fails Catastrophically

On VGG16 with CelebA, BCELoss achieves only ~50% accuracy (random guessing) while BCEWithLogitsLoss reaches ~92%.

**Root cause:** Deep networks without BatchNorm can cause logits to explode after the first optimizer step. When sigmoid saturates to exactly 0 or 1:

| Loss Function | Gradient at saturation |
|---------------|------------------------|
| BCEWithLogitsLoss | **1.0** (bounded, recovers) |
| BCELoss | **0.0** (dead, cannot recover) |

### The Fix

BCELoss works fine on deep networks if you prevent saturation:
- Use proper initialization (Kaiming + Xavier)
- Use lower learning rate (0.0001 instead of 0.001)
- Or just use BCEWithLogitsLoss and don't worry about it

## Repository Structure

```
.
├── 01_shallow_networks/          # Phase 1: MLP experiments
│   ├── experiment.py             # Original experiment
│   ├── experiment_v2.py          # Improved with statistical tests
│   ├── OBSERVATIONS_V1.md        # Initial findings
│   ├── OBSERVATIONS_V2.md        # Final analysis (15 seeds)
│   ├── results/                  # Plots and pickled results
│   └── tensorboard/              # Training logs
│
├── 02_deep_networks/             # Phase 2: VGG16 investigation
│   ├── vgg16_bce_investigation.py    # Main investigation script
│   ├── proper_init_comparison.py     # Fair comparison when configured properly
│   ├── raschka_reproduction.ipynb    # Reproducing Raschka's results
│   ├── OBSERVATIONS.md               # Deep network analysis
│   ├── results/                      # JSON results and images
│   └── tensorboard/                  # Training logs
│
├── data/celeba/                  # CelebA dataset (not tracked)
└── docs/                         # PyTorch loss function documentation
```

## Running the Experiments

### Prerequisites

```bash
# Install dependencies
uv sync

# Or with pip
pip install torch torchvision numpy matplotlib scipy tqdm tensorboard
```

### Phase 1: Shallow Networks (MLP on CIFAR-10)

```bash
cd 01_shallow_networks

# Run full experiment (15 seeds × 12 configs × 2 losses = 360 runs)
# Takes ~2-3 hours on CPU
python experiment_v2.py

# View results
tensorboard --logdir=tensorboard/runs_v2
```

**What it tests:** Whether BCE and BCEWithLogitsLoss differ on a 3-layer MLP with various class imbalance ratios and weight decay values.

**Expected result:** No significant difference between loss functions.

### Phase 2: Deep Networks (VGG16 on CelebA)

```bash
cd 02_deep_networks

# Download CelebA dataset first (will auto-download on first run)
# ~1.4GB download, extracts to ~1.5GB

# Run all 4 combinations: (raschka/proper init) × (BCE/BCEWithLogits)
python vgg16_bce_investigation.py --mode all --epochs 4

# Run fair comparison with proper configuration
python proper_init_comparison.py --epochs 10

# View results
tensorboard --logdir=tensorboard/runs_vgg16_investigation
```

**What it tests:** Why BCELoss fails on deep networks and what conditions cause the failure.

**Expected result:**
- Raschka init or high LR → BCELoss fails (~50%), BCEWithLogitsLoss works (~92%)
- Proper init + low LR → Both work (~92%)

## The Mathematics

### Why BCEWithLogitsLoss is More Stable

**BCELoss gradient:**
```
∂L/∂x = (1/p - 1) · sigmoid'(x)    when y=1
      = (1/(1-p)) · sigmoid'(x)    when y=0
```

When `sigmoid(x)` saturates to exactly 0 or 1 (which happens at |x| > ~17 in float32), `sigmoid'(x) = 0`, so the gradient is **exactly zero**.

**BCEWithLogitsLoss gradient:**
```
∂L/∂x = sigmoid(x) - y
```

This is bounded between -1 and 1, and is **never zero** for incorrect predictions.

### When Does Saturation Happen?

| Scenario | Max Logit Magnitude | Saturation Risk |
|----------|---------------------|-----------------|
| MLP + AdamW | ~100 | Low (gradients still flow) |
| MLP + SGD + WD | ~12 | None |
| VGG16 + lr=0.001 | ~1300 (explodes) | **Critical** |
| VGG16 + lr=0.0001 | ~15 | None |

## Lessons Learned

1. **p < 0.05 is not proof** - Our initial "significant" result was a statistical fluke
2. **Deep networks are unstable** - Without BatchNorm, a single step can cause logit explosion
3. **Numerical stability matters at the edges** - BCEWithLogitsLoss handles edge cases gracefully
4. **Use BCEWithLogitsLoss** - It's never worse and sometimes much better

## References

- Sebastian Raschka's investigation into BCE vs BCEWithLogitsLoss:
  - [Blog post: Losses Learned Part 1](https://sebastianraschka.com/blog/2022/losses-learned-part1.html)
  - [Code: VGG16 smile classifier](https://github.com/rasbt/machine-learning-notes/tree/main/losses/pytorch-loss-functions/vgg16-smile-classifier)
- [PyTorch BCEWithLogitsLoss documentation](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

## Citation

If you find this investigation useful, feel free to reference it in your work.
