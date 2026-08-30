# PyTorch compatibility

RustTorch 0.1.0 targets:

- `tch` 0.26.0
- PyTorch/LibTorch 2.13.0
- behavior from PyTorch tag v2.13.0, commit `cf30153`

The compatibility promise is scoped. Equivalent eager modules and supported
Graph IR operations follow PyTorch defaults, validation, parameter naming, and
train/eval behavior where Rust and `tch` can represent them. Numerical work and
autograd are delegated to LibTorch.

The canonical deterministic CPU model is verified against Python PyTorch
2.13.0 for strict bidirectional SafeTensors loading, forward values, input and
parameter gradients, cross-entropy, MSE, one Adam/SGD step, and residual
forward/backward. This establishes cross-language parity only on CPU.
Separately, Rust CPU-to-MPS backend tests passed on the current macOS arm64
development host for forward/backward, gradients, one Adam/SGD step, movement,
and SafeTensors transfer. CUDA was unavailable and was skipped, not passed.

## Interchange levels

1. Weight/state interchange is required through SafeTensors when architectures,
   names or mappings, shapes, and dtypes agree.
2. Architecture interchange is supported only for manually equivalent eager
   models and operations represented by RustTorch Graph IR.
3. Full training-checkpoint interchange is not promised. Optimizer state and
   Python runtime structures are not a stable cross-language contract.

Backend changes can introduce normal floating-point differences without
changing the logical state format. Tests use deterministic assigned weights
and documented tolerances rather than assuming identical RNG streams.

## Important differences

- Rust configuration types replace Python keyword arguments and dynamic values.
- Global hooks, decorators, arbitrary Python containers, full control flow,
  and Python class reconstruction are not supported.
- `eval()` changes module behavior but does not disable autograd.
- Explicit unavailable devices error instead of silently falling back.
- SafeTensors is the only model-state format accepted by RustTorch 0.1;
  pickle-based `.pt`/`.bin` files are not accepted.
- TorchScript is not exposed by RustTorch 0.1. Callers can use `tch::CModule`
  directly for opaque legacy inference, outside the RustTorch compatibility
  surface.
- `.pt2` import/export is not implemented or claimed until separately tested.

The compatibility manifest records each supported symbol, its source, support
level, classification, naming, and test coverage.
