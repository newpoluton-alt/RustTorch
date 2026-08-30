# PyTorch compatibility

RustTorch 0.1.0 targets:

- `tch` 0.26.0
- PyTorch/LibTorch 2.13.0
- behavior from PyTorch tag v2.13.0, commit `cf30153`

The compatibility promise is scoped. Equivalent eager modules and supported
Graph IR operations follow PyTorch defaults, validation, parameter naming, and
train/eval behavior where Rust and `tch` can represent them. Numerical work and
autograd are delegated to LibTorch.

## Compatibility ledger

[`compat/pytorch_api.toml`](../compat/pytorch_api.toml) is the canonical
machine-readable inventory. It pins PyTorch tag `v2.13.0` at commit `cf30153`
and `tch` 0.26.0, then records each capability's stable ID, PyTorch and
RustTorch symbols, implementation boundary, exact scope, upstream source,
evidence, and notes. [`api-coverage.md`](api-coverage.md) is generated from
that ledger and included on the crate's rustdoc landing page.

The statuses mean:

- `supported`: executable evidence covers the row's exact written scope;
- `partial`: a Rust surface exists, but the broader PyTorch area or some
  exposed behavior is not yet evidenced;
- `planned`: the capability is an intended milestone without a current
  implementation claim;
- `python_only`: the behavior depends on Python runtime semantics rather than
  representing a native RustTorch surface; and
- `not_supported`: RustTorch intentionally makes no support claim.

Rows are independently scoped and differ greatly in size. Their count is not a
meaningful denominator, so the project does not derive a compatibility
percentage from them. Coverage grows feature by feature with executable Rust,
Python, and backend evidence where each claim requires it.

After editing the canonical ledger, regenerate and verify the public page:

```sh
python3 scripts/check-compatibility.py --write
python3 scripts/check-compatibility.py --check
```

Do not edit the generated page by hand. The checker validates schema version,
sorted IDs, pinned metadata, source paths, exact evidence declarations, and
byte-for-byte generated output.

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

The canonical ledger and generated coverage page remain the authority when a
summary elsewhere differs.
