# torch.nn.BCELoss

Creates a criterion that measures the Binary Cross Entropy between the target and input probabilities. Commonly used for reconstruction errors in autoencoders.

## Formula

The unreduced loss is expressed as:

$$\ell(x, y) = L = \{l_1, \ldots, l_N\}^T$$

$$l_n = -w_n \left[ y_n \cdot \log(x_n) + (1 - y_n) \cdot \log(1 - x_n) \right]$$

Where:
- $x_n$ is the input (predicted probability)
- $y_n$ is the target (0 or 1)
- $w_n$ is the optional weight

When reduction is applied:
- `'mean'`: $\text{mean}(L)$
- `'sum'`: $\text{sum}(L)$

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight` | Tensor, optional | Manual rescaling weight per batch element |
| `size_average` | bool, optional | **Deprecated.** Averages losses over batch elements by default |
| `reduce` | bool, optional | **Deprecated.** When False, returns per-element loss |
| `reduction` | str | Controls output reduction: `'none'`, `'mean'` (default), or `'sum'` |

## Shape

- **Input:** `(*)` any number of dimensions
- **Target:** `(*)` same shape as input
- **Output:** scalar (or same shape as input if `reduction='none'`)

## Important Implementation Detail

> To prevent undefined logarithmic behavior when inputs are 0 or 1, PyTorch **clamps log outputs to ≥ -100**, ensuring finite loss values and linear gradient behavior.

This means:
- `log(0)` is clamped to `-100` instead of `-inf`
- Maximum loss per sample is clamped to `100`

## Example

```python
import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, 2, requires_grad=True)
target = torch.rand(3, 2, requires_grad=False)
output = loss(m(input), target)
output.backward()
```

## Notes

- **Target values should range between 0 and 1**
- **Input must be probabilities** (typically after applying sigmoid)
- For numerical stability, consider using `BCEWithLogitsLoss` instead, which combines sigmoid and BCE in a single operation
