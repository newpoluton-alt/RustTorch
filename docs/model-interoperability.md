# Model interoperability

SafeTensors is the RustTorch 0.1 device-neutral contract for model state. A weight
file does not define an arbitrary architecture: the Rust and Python models must
be equivalent, or a supported Graph IR must represent the architecture.

## PyTorch to RustTorch

```python
from safetensors.torch import save_file
save_file(model.state_dict(), "weights.safetensors")
```

Build the equivalent RustTorch model, then call `load_weights`. Strict loading
is the default. Missing, unexpected, duplicate-mapped, shape-incompatible, and
unsafe dtype-incompatible entries produce structured failures.

## RustTorch to PyTorch

Call `save_weights`, construct the equivalent Python model, and load with:

```python
from safetensors.torch import load_file
model.load_state_dict(load_file("weights.safetensors"), strict=True)
```

The parity tools use deterministic assigned weights and inputs and compare both
directions. Where practical they also compare losses and gradients.

Current verified result: Python→Rust strict load matches Linear forward,
input/parameter gradients, cross-entropy, MSE, one SGD step, one Adam step, and
residual forward/input/parameter gradients on CPU. Rust→Python strict load and
forward also match. Rust-to-Rust SafeTensors transfer between CPU and MPS
passes on the current host. Cross-language MPS and CUDA interchange have not
been executed; CUDA was unavailable.

## State rules

- RustTorch 0.1 modules expose parameter state through `tch::nn::VarStore`;
  dedicated persistent-buffer registration is not implemented.
- Names follow PyTorch conventions where practical; differences require an
  explicit mapping.
- Loading targets the model's current CPU, CUDA, or MPS device.
- No fuzzy matching, silent transpose, reshape, or dtype coercion is allowed.
- SafeTensors stores values by name, not Rust alias relationships.
  Tied-parameter alias preservation is not implemented or tested in 0.1.

## Other formats

Pickle-based `.pt`/`.bin` state dictionaries are untrusted and are not accepted
by the RustTorch 0.1 API. Arbitrary `torch.save(model)` objects require Python
classes and code and are not portable. RustTorch does not expose TorchScript or
PT2 loading in 0.1; callers needing opaque TorchScript inference can use
`tch::CModule` directly.

A future RustTorch package may contain `manifest.json`, `graph.json`,
`weights.safetensors`, and `metadata.json`. Its SafeTensors file stays directly
extractable for Python. The package format and stable graph serialization are
not implemented in 0.1.
