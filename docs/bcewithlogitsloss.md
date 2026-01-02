# torch.nn.BCEWithLogitsLoss

Combines sigmoid activation with binary cross-entropy loss in a single, **numerically stable** module. It leverages the **log-sum-exp trick** to prevent overflow/underflow issues.

## Formula

The unreduced loss for single-label case:

$$\ell(x, y) = L = \{l_1, \ldots, l_N\}^T$$

$$l_n = -w_n \left[ y_n \cdot \log(\sigma(x_n)) + (1 - y_n) \cdot \log(1 - \sigma(x_n)) \right]$$

For multi-label case:

$$l_{n,c} = -w_{n,c} \left[ p_c \cdot y_{n,c} \cdot \log(\sigma(x_{n,c})) + (1 - y_{n,c}) \cdot \log(1 - \sigma(x_{n,c})) \right]$$

Where $p_c$ is the `pos_weight` for class $c$.

### Numerically Stable Implementation

Instead of computing sigmoid and log separately, the loss is computed as:

$$l_n = \max(x_n, 0) - x_n \cdot y_n + \log(1 + e^{-|x_n|})$$

This avoids:
- Computing `log(0)` when sigmoid saturates to 0 or 1
- Overflow in `exp()` for large positive values
- Underflow issues at extreme logit values

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor, optional | Manual rescaling weight given to the loss of each batch element (size: nbatch) |
| `size_average` | bool, optional | **Deprecated.** Averages losses over batch elements by default |
| `reduce` | bool, optional | **Deprecated.** When False, returns per-element loss |
| `reduction` | str | Specifies output reduction: `'none'`, `'mean'` (default), or `'sum'` |
| `pos_weight` | Tensor, optional | Weight of positive examples to be broadcasted with target |

## Shape

- **Input:** `(*)` any shape (raw logits, NOT probabilities)
- **Target:** `(*)` same shape as input
- **Output:** scalar (or same shape as input if `reduction='none'`)

## Handling Class Imbalance

The `pos_weight` parameter addresses class imbalance by weighting positive samples:

For a dataset with 100 positive and 300 negative examples:
```python
pos_weight = torch.tensor([3.0])  # 300/100
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

The loss becomes:

$$l_n = -w_n \left[ p \cdot y_n \cdot \log(\sigma(x_n)) + (1 - y_n) \cdot \log(1 - \sigma(x_n)) \right]$$

Where $p$ is the `pos_weight`.

## Example

```python
import torch
import torch.nn as nn

loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)  # raw logits
target = torch.empty(3).random_(2)
output = loss(input, target)
output.backward()
```

## Example with pos_weight

```python
import torch
import torch.nn as nn

# For imbalanced dataset: 100 positive, 300 negative
pos_weight = torch.tensor([3.0])
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
output = loss(input, target)
output.backward()
```

## Notes

- **Input should be raw logits** (before sigmoid), NOT probabilities
- **More numerically stable** than `nn.Sigmoid()` + `nn.BCELoss()`
- **Recommended** for binary classification tasks
- The sigmoid is applied internally using the log-sum-exp trick

## Why Use BCEWithLogitsLoss Instead of BCELoss?

1. **Numerical Stability:** Avoids `log(0)` and `exp()` overflow issues
2. **Single Operation:** Combines sigmoid + BCE, reducing computation
3. **Consistent Gradients:** Maintains proper gradient flow at extreme logit values
4. **Symmetric Behavior:** Loss and gradients behave symmetrically for both classes
