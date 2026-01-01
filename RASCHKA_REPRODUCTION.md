# Reproducing Raschka's BCE vs BCEWithLogitsLoss Experiment

## Background

Sebastian Raschka's blog post ["Losses Learned"](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) demonstrated a dramatic difference between BCELoss and BCEWithLogitsLoss:

| Method | Accuracy |
|--------|----------|
| BCEWithLogitsLoss | 92.18% |
| Sigmoid + BCELoss | 50.03% (random guessing) |

This appeared to contradict our findings (see `OBSERVATIONS_V2.md`) where both loss functions performed identically across 180 paired comparisons.

---

## Our Reproduction

### Setup (Matching Raschka Exactly)

| Parameter | Value |
|-----------|-------|
| Model | VGG16 (65.1M parameters) |
| Dataset | CelebA (smile classification) |
| Optimizer | Adam (lr=0.001, **no weight decay**) |
| Batch Size | 256 |
| Epochs | 4 |
| Initialization | `normal(0, 0.05)` for all weights |

### Results: Reproduction Successful

| Method | Our Result | Raschka's Result |
|--------|------------|------------------|
| BCEWithLogitsLoss | 91.48% | 92.18% |
| Sigmoid + BCELoss | 50.03% | 50.03% |
| **Difference** | **41.45pp** | **42.15pp** |

We successfully reproduced the catastrophic failure of BCELoss.

---

## Investigation: Why Does BCELoss Fail?

### Hypothesis 1: Extreme Logits After Training

We analyzed the logits from both trained models on a test batch:

| Statistic | BCEWithLogitsLoss | BCELoss (pre-sigmoid) |
|-----------|------------------:|----------------------:|
| Min | -6.78 | **3,265,233,408** |
| Max | 10.00 | **6,722,117,632** |
| Mean | 1.19 | **5,019,848,704** |
| \|logit\| > 10 | 0 | 256 (100%) |
| \|logit\| > 100 | 0 | 256 (100%) |

**Finding**: The BCELoss model's logits exploded to **billions**. At these values:
- `sigmoid(3 billion) = 1.0` (saturated)
- Gradient = 0 (dead)

But this raised the question: why did BCEWithLogitsLoss survive the same conditions?

### Hypothesis 2: Bad Initialization

We tested different initializations on **untrained** models:

| Initialization | Initial Sigmoid Output |
|----------------|------------------------|
| Raschka `N(0, 0.05)` | 0.000000 - 1.000000 (saturated!) |
| PyTorch Default | 0.501737 - 0.501737 (perfect!) |
| Kaiming | 0.049432 - 0.999987 (biased) |

**Finding**: Raschka's initialization produces saturated sigmoid outputs even at initialization!

### Hypothesis 3: Initialization is the Root Cause

We trained BCELoss with different initializations:

| Experiment | Initialization | Initial Sigmoid | Final Accuracy |
|------------|----------------|-----------------|----------------|
| 1 | Raschka `N(0, 0.05)` | 0.0 - 1.0 | 50.03% |
| 2 | Kaiming | 0.87 - 1.0 | 49.97% |
| 3 | **PyTorch Default** | **0.499 (perfect!)** | **49.97%** |

**Shocking result**: Even with perfect initialization (sigmoid outputs at exactly 0.5), BCELoss still fails to learn!

---

## The Mystery Deepens

### What We Expected

With PyTorch default initialization:
- Initial logits: -0.0045 (all identical)
- Initial sigmoid: 0.498883 (perfect, near 0.5)
- Initial loss: 0.6925 (correct for random predictions, ln(2) ≈ 0.693)

This is the **ideal** starting point for BCELoss.

### What Actually Happened

```
Epoch 1/4 | Loss: 0.6925 | Test Acc: 49.97%
Epoch 2/4 | Loss: 0.6924 | Test Acc: 49.97%
Epoch 3/4 | Loss: 0.6924 | Test Acc: 49.97%
Epoch 4/4 | Loss: 0.6923 | Test Acc: 49.97%
```

The loss barely moves (0.6925 → 0.6923) and the model learns **nothing**.

---

## Summary of All Experiments

| # | Loss Function | Initialization | Initial Sigmoid | Final Accuracy |
|---|---------------|----------------|-----------------|----------------|
| 1 | BCEWithLogitsLoss | Raschka | N/A | **91.48%** |
| 2 | BCELoss | Raschka | 0.0 - 1.0 | 50.03% |
| 3 | BCELoss | Kaiming | 0.87 - 1.0 | 49.97% |
| 4 | BCELoss | PyTorch Default | **0.499** | **49.97%** |

---

## Open Questions

### The Fundamental Question

**Why can't `Sigmoid + BCELoss` train VGG16, even with perfect initialization, when `BCEWithLogitsLoss` can?**

### Possible Explanations

1. **Gradient computation differences**:
   - BCEWithLogitsLoss uses the log-sum-exp trick for numerically stable gradients
   - Sigmoid + BCELoss computes `sigmoid'(x) * BCE'(sigmoid(x))` which may have numerical issues in deep networks

2. **Backward pass instability**:
   - Even when forward pass produces correct values, the backward pass through separated Sigmoid → BCELoss may lose precision

3. **Network depth interaction**:
   - Our MLP experiments (260K params) worked fine with BCELoss
   - VGG16 (65M params) fails completely
   - Something about very deep networks exposes this issue

4. **Floating point accumulation**:
   - 65M parameters means many more floating point operations
   - Small numerical errors may compound differently in the two formulations

### When Does This Effect Show Up?

Based on our experiments:

| Architecture | BCELoss Works? |
|--------------|----------------|
| 2-layer MLP (260K params) | Yes |
| VGG16 (65M params) | **No** |

The boundary conditions (model depth, width, specific architectures) remain unexplored.

---

## Reconciling With Our Original Findings

### Why Our Experiments Showed No Difference

| Factor | Our Experiments | Raschka's Setup |
|--------|-----------------|-----------------|
| Model | MLP (260K params) | VGG16 (65M params) |
| Depth | 2 hidden layers | 16 layers |
| Dataset | MNIST/CIFAR-10 | CelebA |
| Initialization | PyTorch defaults | Custom `N(0, 0.05)` |

Our simpler architecture never triggered the failure mode that VGG16 exposes.

### The Complete Picture

1. **For shallow networks**: BCE and BCEWithLogitsLoss are equivalent (our finding)
2. **For deep networks (VGG16+)**: BCEWithLogitsLoss is **required** - BCELoss fundamentally cannot train the network
3. **The reason**: Unknown - not initialization, not weight decay, something deeper about gradient flow

---

## Practical Recommendations

| Recommendation | Reason |
|----------------|--------|
| **Always use BCEWithLogitsLoss** | It works universally; BCELoss fails on deep networks |
| Don't expect performance gains on shallow networks | They're equivalent there |
| The "numerical stability" explanation is incomplete | Perfect initialization still fails |

---

## Files

- `raschka_reproduction.ipynb` - Full reproduction notebook with all experiments
- `rasbt_results/` - Original Raschka code (PyTorch Lightning version)

---

## Future Work

1. Identify the exact network depth/width threshold where BCELoss starts failing
2. Analyze gradient magnitudes through each layer during training
3. Compare the actual floating point operations in both loss computations
4. Test on other deep architectures (ResNet, Transformer, etc.)
