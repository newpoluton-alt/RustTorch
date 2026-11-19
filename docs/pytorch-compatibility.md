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
declarations, and the generated coverage pages for the facade and domain crates. Package extraction
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

## Choose an interchange contract

| Use case | RustTorch entry point | Supported boundary |
|---|---|---|
| Transfer trained parameters | `interop::save_state_dict`, `load_state_dict_with_mapping` | SafeTensors weights and persistent buffers with matching shapes and dtypes. Rebuild the architecture and map renamed parameters. |
| Resume a Rust training job | `checkpoint::save`, `TrainingCheckpoint::restore` | Versioned archive containing weights, optimizer state and typed application progress. The caller supplies scheduler, scaler, loader or RNG state when needed. |
| Exchange a functional tensor program | `deployment::Model::{load_pt2,save_pt2}` | Versioned PT2 subset with supported operators, calling trees, named state and dimension guards. |
| Deploy an inference graph | `deployment::Model::{load_onnx,save_onnx}` | Default-domain ONNX opset 18, IR 10 and the documented operator subset. Initializers serve as inference constants. |
| Run a native traced graph | `Model::trace`, `TracedModel::{save,load}` | Branch-free, single-output RustTorch tensor programs. Keep the guards sidecar with the frozen TorchScript artifact. |

For runnable Rust examples, see [deployment](deployment.md),
[checkpoint and resume](checkpoint-compatibility.md) and [weight exchange](model-interoperability.md).
The reference tests use deterministic weights and per-case tolerances for
cross-runtime and cross-device comparisons.

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
- Weight and training-checkpoint readers accept their documented formats.
  Arbitrary Python model objects and pickle-based optimizer state require a
  separate trusted conversion. Native traced inference requires a trusted model producer.
- Portable RustTorch conditionals can execute a selected branch with
  gradients. PT2/ONNX interchange and native tracing each have a narrower
  operator contract. See the [deployment guide](deployment.md) for the
  version, supported operators and limits for each format.

The canonical ledger and generated coverage page remain the authority when a
summary elsewhere differs.
