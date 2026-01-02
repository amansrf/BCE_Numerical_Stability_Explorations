# Observations V3: Deep Network BCE Failure Analysis

## Summary

This investigation explains why `Sigmoid + BCELoss` fails catastrophically on VGG16 (~50% accuracy) while `BCEWithLogitsLoss` succeeds (~92% accuracy), even when both start with identical weights and perfect initialization.

**Root Cause:** Deep networks without BatchNorm are unstable. A single optimizer step can cause logits to explode. BCEWithLogitsLoss maintains gradients at saturation and recovers. BCELoss produces zero gradients at saturation and cannot recover.

## Experimental Setup

- **Model:** VGG16 (65M parameters, 16 layers, no BatchNorm)
- **Dataset:** CelebA smile classification (162K train, 20K test)
- **Optimizer:** Adam, lr=0.001
- **Initialization:** "Proper" init (Kaiming for ReLU layers, Xavier for final layer)
- **Initial state:** logit_mean=-0.008, logit_std=0.021, sigmoid≈0.5, 0% saturated

## Key Experiment: Lock-Step Training

Both models trained on identical batches, comparing behavior step by step.

### BCEWithLogitsLoss Behavior

```
Batch | loss       | logit_mean | logit_std | saturated% | fc3_grad_norm
---------------------------------------------------------------------------
    0 |       0.69 |      -0.01 |      0.02 |      0.0% | 0.311588
    1 |     630.95 |   -1315.54 |    203.83 |    100.0% | 990.165039   <-- EXPLOSION
    2 |       0.70 |       0.03 |      0.00 |      0.0% | 0.054141     <-- RECOVERED
    3 |       0.69 |       0.01 |      0.00 |      0.0% | 0.021851
   ...continues normally...
```

### BCELoss Behavior

```
Batch | loss       | logit_mean | logit_std | saturated% | fc3_grad_norm
---------------------------------------------------------------------------
    0 |       0.69 |      -0.01 |      0.02 |      0.0% | 0.311588
    1 |      48.83 |   -1315.46 |    203.81 |    100.0% | 0.000000     <-- EXPLOSION + ZERO GRAD
    2 |      42.97 |  -34910.15 |   5106.71 |    100.0% | 0.000000     <-- DRIFTING
    3 |      45.70 | -294127.69 |  41906.13 |    100.0% | 0.000000
   ...
   14 |      48.05 | -531271616.00 | 83038472.00 |    100.0% | 0.000000  <-- EXPLODED TO -531M
```

## The Mechanism

### Step 1: Initial Explosion (Batch 0 → 1)

The first optimizer step causes logits to explode from std=0.02 to std=204:

```
INITIAL STATE:
  logit mean: -0.007816
  logit std:  0.021309
  sigmoid mean: 0.498046
  |logit| > 17: 0

AFTER FIRST UPDATE:
  logit mean: -1334.466919
  logit std:  198.889832
  sigmoid mean: 0.000000
  |logit| > 17: 256 (ALL samples)
```

This happens because VGG16 is deep (16 layers) and has no BatchNorm. A tiny weight change (norm=0.0003) gets amplified through the layers into massive logit changes.

### Step 2: Gradient Divergence at Saturation

When sigmoid saturates to exactly 0 or 1 (float32):

| Metric | BCEWithLogitsLoss | BCELoss |
|--------|-------------------|---------|
| Loss at logit=1000 | 1000.0 | 100.0 (clamped) |
| Gradient at logit=1000 | **1.0** | **0.0** |

BCEWithLogitsLoss gradient: `sigmoid(x) - y` → bounded, always non-zero for wrong predictions

BCELoss gradient: `(1/p or 1/(1-p)) * sigmoid'(x)` → when sigmoid'(x)=0 at saturation, gradient=0

### Step 3: Recovery vs Explosion

- **BCEWithLogitsLoss:** Gradient=990 at batch 1 → massive correction → recovered by batch 2
- **BCELoss:** Gradient=0 at batch 1 → no correction → Adam momentum drifts → logits explode to -531M

## Gradient Behavior at Extreme Logits

```
logit=         0 | sigmoid=0.5000 | grad_BCEWL=0.5000 | grad_BCE=0.5000
logit=        10 | sigmoid=1.0000 | grad_BCEWL=1.0000 | grad_BCE=1.0000
logit=        50 | sigmoid=1.0000 | grad_BCEWL=1.0000 | grad_BCE=0.0000  <-- DIVERGENCE
logit=       100 | sigmoid=1.0000 | grad_BCEWL=1.0000 | grad_BCE=0.0000
logit=      1000 | sigmoid=1.0000 | grad_BCEWL=1.0000 | grad_BCE=0.0000
logit=   1000000 | sigmoid=1.0000 | grad_BCEWL=1.0000 | grad_BCE=0.0000
```

At logit ≥ 50, sigmoid saturates to exactly 1.0 in float32. BCEWithLogitsLoss maintains gradient=1.0. BCELoss gradient becomes exactly 0.0.

## Attempted Fixes

### Weight Decay (Does NOT Work)

```
weight_decay=0.01:  batch 14 logit_mean = -530 million
weight_decay=0.1:   batch 14 logit_mean = -522 million
weight_decay=1.0:   batch 14 logit_mean = -449 million
```

Weight decay shrinks weights uniformly but provides no directional correction. Gradient is still 0.

### Gradient Clipping (Does NOT Work)

Clipping happens after backward pass. By then, gradients are already 0 from saturation.

### Smaller Learning Rate (WORKS)

```
BCELoss + Adam lr=0.0001 (10x smaller)
Batch | loss       | logit_mean | logit_std | saturated% | fc3_grad_norm
    0 |       0.69 |      -0.01 |      0.02 |      0.0% | 0.311588
    1 |       1.49 |      -3.00 |      0.49 |      0.0% | 3.900444
    2 |       0.68 |      -0.17 |      0.03 |      0.0% | 0.115023
   ...continues normally, no explosion...
```

Smaller LR prevents the initial explosion, so saturation is never reached.

### BatchNorm Only (Does NOT Work)

```
BCELoss + VGG16_BN + Adam lr=0.001
Batch | loss       | logit_mean | logit_std | saturated%
    0 |     0.7136 |     0.2352 |    0.2186 |      0.0%
    1 |    32.1556 |   -72.1302 |   37.9366 |    100.0%  <-- Still saturates
   ...still explodes, just slower...
```

BatchNorm slows the explosion but doesn't prevent saturation at lr=0.001.

### BatchNorm + Smaller LR (WORKS)

```
BCELoss + VGG16_BN + Adam lr=0.0001
Batch | loss       | logit_mean | logit_std | saturated% | grad_norm
    0 |     0.7136 |     0.2352 |    0.2186 |      0.0% | 1.563861
    1 |     1.6724 |    -3.3622 |    0.9297 |      7.4% | 8.790253
    2 |     0.6877 |     0.0907 |    0.2644 |      0.0% | 1.596152
   ...continues normally...
```

Combination of BatchNorm + smaller LR keeps saturation minimal (0-7%) and gradients non-zero.

## Summary Table: Fixes for BCE on Deep Networks

| Fix | Works? | Reason |
|-----|--------|--------|
| Smaller LR (0.0001) | ✓ | Prevents initial explosion that causes saturation |
| Weight decay (any) | ✗ | No directional correction, gradient still 0 |
| Gradient clipping | ✗ | Too late, gradient already 0 after saturation |
| BatchNorm + lr=0.001 | ✗ | Slows explosion but still saturates |
| BatchNorm + lr=0.0001 | ✓ | Stabilizes activations AND prevents large updates |

## Conclusions

1. **The failure is architectural, not loss-function-inherent:** VGG16 without BatchNorm is unstable at lr=0.001. The first optimizer step causes logit explosion.

2. **BCEWithLogitsLoss is robust to instability:** It maintains gradients at saturation (gradient = sigmoid(x) - y, bounded between -1 and 1), allowing recovery from explosions.

3. **BCELoss is fragile:** When sigmoid saturates to exactly 0 or 1, the gradient becomes exactly 0 due to sigmoid'(x) = 0. The network cannot recover.

4. **Why shallow networks work:** MLPs don't amplify weight changes as severely, so logits stay in non-saturated regions. Both losses work fine.

5. **Practical recommendation:** Always use BCEWithLogitsLoss for robustness. If you must use BCELoss on deep networks, ensure:
   - BatchNorm or LayerNorm is present
   - Learning rate is small enough to prevent saturation
   - Or use a more stable architecture (ResNet, etc.)

## Full Training Comparison: Lower LR with Raschka Init

Testing whether lower LR can save BCELoss when using Raschka's original initialization.

### Initial State (Raschka Init)

```
RASCHKA INIT - INITIAL STATE (before any training):
  logit mean: -124.39
  logit std:  337.93
  logit min:  -1038.59
  logit max:  691.30
  sigmoid mean: 0.405462
  SATURATED:    99.2%   <-- Already saturated before training!
```

### 10 Epoch Training (Adam lr=0.0001)

```
BCEWithLogitsLoss:
Epoch  Train Loss   Train Acc    Test Acc
1      17.5728      0.6357       0.7972
2      0.4540       0.7994       0.8599
...
10     0.2025       0.9150       0.9142   ✓ Learns fine

BCELoss:
Epoch  Train Loss   Train Acc    Test Acc
1      52.0238      0.4797       0.5003   <-- Stuck from epoch 1
2      52.0317      0.4797       0.5003
...
10     52.0315      0.4797       0.5003   ✗ Never learns
```

### Result

| Loss | Final Test Acc |
|------|----------------|
| BCEWithLogitsLoss | **91.42%** |
| BCELoss | **50.03%** (random) |

**Difference: 41.39 percentage points**

Lower LR does NOT help with Raschka init because the problem is the initial state, not training dynamics. With 99.2% initial saturation, BCE gradients are 0 before the first step ever happens.

### Two Failure Modes

| Init | Problem | Lower LR Helps? |
|------|---------|-----------------|
| Raschka | Initial saturation (99.2%) | No - damage before training |
| Proper | Explosion during training | Yes - prevents explosion |

## Fair Comparison: Proper Init + Lower LR

When saturation is avoided entirely, are the losses equivalent?

**Script:** `proper_init_comparison.py`

```
Init: Proper (Kaiming + Xavier)
LR: 0.0001
Epochs: 10

BCEWithLogitsLoss:
Epoch  Train Loss   Train Acc    Test Acc
1      0.2770       0.8691       0.9189
...
10     0.0738       0.9701       0.9199

BCELoss:
Epoch  Train Loss   Train Acc    Test Acc
1      0.2801       0.8675       0.9196
...
10     0.0711       0.9716       0.9198

FINAL:
BCEWithLogitsLoss         0.9701             0.9199
BCELoss                   0.9716             0.9198

Difference: +0.01 percentage points
```

**Conclusion: Both losses perform identically when saturation is avoided.**

## The Complete Picture

| Scenario | BCEWithLogitsLoss | BCELoss | Winner |
|----------|-------------------|---------|--------|
| Raschka init + lr=0.001 | 92% | 50% | BCEWithLogits |
| Raschka init + lr=0.0001 | 91% | 50% | BCEWithLogits |
| Proper init + lr=0.001 | 92% | 50% | BCEWithLogits |
| Proper init + lr=0.0001 | **92%** | **92%** | **Tie** |

BCEWithLogitsLoss is strictly better because it handles all scenarios. BCELoss only matches it when everything is perfectly configured to avoid saturation.

## Code Reference

The investigation script is in `vgg16_bce_investigation.py`. Key experiments:

```bash
# Run all 4 combinations (raschka/proper init × BCE/BCEWithLogits)
python vgg16_bce_investigation.py --mode all --epochs 4

# View detailed logs
tensorboard --logdir=runs_vgg16_investigation
```

Lock-step training analysis was done interactively (see conversation for exact code).
