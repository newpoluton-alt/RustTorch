# Architecture

RustTorch is a high-level Rust layer over `tch`; `tch` binds LibTorch, which
owns tensors, kernels, devices, and automatic differentiation.

```text
eager modules ───────────────┐
                            ├─> tch::Tensor -> LibTorch -> CPU/CUDA/MPS
Graph IR -> EagerExecutor ───┘                    └──────> autograd
```

## Data path

Map-style data flows from a borrowed `Dataset` through a caller-selected
sampler and `DataLoader`. Streaming data stays an ordinary fallible Rust
iterator and enters through `batches`. Both routes share the same
single-threaded batching core:

```text
Dataset + sampler -> DataLoader ─┐
fallible iterator -> batches ────┴─> Vec<Sample> -> optional collation -> model
```

Samples are moved into one pre-sized vector per batch. Custom collation owns
that vector and can stack tensors, pad sequences, or build structured batches
without an implicit tensor copy in the loader. A sampler-local RNG makes
seeded shuffling reproducible without consuming LibTorch's global random
state. Workers, prefetch, pinned memory, distributed sharding, and loader
checkpoint/resume are later milestones.

## Eager path

Eager execution is the primary API. Modules perform ordinary `tch::Tensor`
operations, so LibTorch records the runtime autograd graph directly. RustTorch
must not detach, copy through host memory, change device, or enter no-gradient
mode unless the caller explicitly requests it.

Models use `tch::nn::VarStore` for parameters. Module wrappers own high-level
configuration, validation, names, initialization behavior, and train/eval
state; numerical work remains in `tch`/LibTorch.

## Explicit graph path

Graph IR is optional for callers who need named connectivity, validation,
inspection, transformation, or a future compilation boundary. It is a
backend-independent DAG, not an autograd engine. `EagerExecutor` traverses a
validated topological order and dispatches the same tensor operations used by
eager modules. Branches and residual edges therefore remain visible to
LibTorch autograd.

## Cross-cutting ownership

- `device`: resolves requested backends and reports capabilities.
- `nn`: eager modules, composition, initialization, and functional operations.
- `optim`: ergonomic configuration over backend optimizers.
- `graph`: IR, validation, passes, inspection, and eager execution.
- `data`: fallible map and stream batching plus deterministic local sampling.
- `interop`: state naming, explicit mappings, SafeTensors, and format policy.
- `error`: structured failures at recoverable boundaries.

No second tensor store, parameter store, kernel layer, native bridge, or
differentiation system belongs in the MVP.
