# torch.nn.Sigmoid

Applies the element-wise sigmoid function.

## Formula

$$\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$

## Shape

- **Input:** `(*)` any shape
- **Output:** `(*)` same shape as input

## Common Use Cases

- **Binary classification:** Outputs probability-like values between 0 and 1
- **Gate mechanisms:** Used in architectures like LSTMs and GRUs
- **Activation function:** Standard choice in traditional neural networks

## Related Functions

- `torch.nn.LogSigmoid` - applies the logarithm of sigmoid
- `torch.nn.functional.sigmoid()` - functional variant

## Example

```python
import torch
import torch.nn as nn

m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)
```

## Notes

The sigmoid function squashes input values to a range between 0 and 1, making it useful for producing probability-like outputs.
