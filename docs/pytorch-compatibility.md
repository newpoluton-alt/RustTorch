# PyTorch compatibility

RustTorch targets:

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
that ledger for compatibility review. Filtered
[`rusttorch-core`](../crates/rusttorch-core/COMPATIBILITY.md) and
[`rusttorch-data`](../crates/rusttorch-data/COMPATIBILITY.md) pages are
generated from the same ledger. Crate landing pages teach RustTorch usage with
Rust examples and link to this evidence separately.

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
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Do not edit any generated compatibility page by hand. The checker validates
schema version, sorted IDs, pinned metadata, source paths, exact evidence
declarations, and all three byte-for-byte generated outputs. Package extraction
changes symbol ownership only: `rusttorch_core` and `rusttorch_data` are the
direct surfaces, while the existing `rusttorch` paths remain facade-compatible.
The data ledger separately records loader workers, prefetch, pinning,
distributed sampling, and RustTorch-native checkpoint/resume scope.

The canonical deterministic CPU model is verified against Python PyTorch
2.13.0 for strict bidirectional SafeTensors loading, forward values, input and
parameter gradients, cross-entropy, MSE, one Adam/SGD step, and residual
forward/backward. This establishes cross-language parity only on CPU.
Separately, Rust CPU-to-MPS backend tests passed on an earlier macOS arm64
development host for forward/backward, gradients, one Adam/SGD step, movement,
and SafeTensors transfer. CUDA was unavailable and was skipped, not passed. The latest run and the
new model/optimizer CPU scopes are recorded in [backend evidence](backend-parity.md).

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
- Rust worker threads replace Python multiprocessing contexts and pickling;
  typed collators replace the mutable Python collation registry.
- DataPipe classes, their functional registration decorators, runtime
  validation contexts, and dataframe tracing are separately scoped in the
  ledger rather than inferred from ordinary Rust iterators.
- Deprecated `pin_memory_device` is not reproduced; `.pin_memory_for(Device)`
  is the typed explicit-device replacement. Pinned classic DataLoader 2.13 has
  no public checkpoint API, so loader checkpoint rows are RustTorch extensions.
- `eval()` changes module behavior but does not disable autograd.
- Explicit unavailable devices error instead of silently falling back.
- SafeTensors is the only model-state format accepted by RustTorch;
  pickle-based `.pt`/`.bin` files are not accepted.
- TorchScript is not exposed by RustTorch. Callers can use `tch::CModule`
  directly for opaque legacy inference, outside the RustTorch compatibility
  surface.
- `.pt2` import/export is not implemented or claimed until separately tested.

The canonical ledger and generated coverage page remain the authority when a
summary elsewhere differs.
