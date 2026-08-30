# Checkpoint compatibility

Weights, architecture, and resumable training state are separate contracts.

## Portable model checkpoint

```text
weights.safetensors
optional model or graph metadata
```

This is the required cross-language path. Equivalent PyTorch and RustTorch
models can load the weights when names or explicit mappings, shapes, and dtypes
agree.

## RustTorch training checkpoint

A RustTorch-local checkpoint may additionally contain optimizer and scheduler
state, epoch, step, and RNG metadata. If implemented, it is versioned and is
not automatically a PyTorch checkpoint.

## Cross-language full resume

Cross-language optimizer resume is not implemented in RustTorch 0.1. Adam and
SGD state can include per-parameter tensors, counters, parameter groups, and
runtime-specific metadata. A future implementation would require an explicit
mapping and parity tests; RustTorch currently makes no continuation promise for
a Python optimizer checkpoint.

Moving a model between devices can invalidate optimizer state. The safe default
is to move the model and rebuild the optimizer. State movement is allowed only
when the implementation verifies every optimizer tensor and association.

Never load untrusted Python pickle. SafeTensors avoids Python pickle execution
semantics for model state; a conversion script must be explicit and
user-invoked.
