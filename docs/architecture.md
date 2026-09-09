# Architecture

RustTorch is a high-level Rust layer over `tch`; `tch` binds LibTorch, which
owns tensors, kernels, devices, and automatic differentiation.

The workspace dependency direction is:

```text
rusttorch -> rusttorch-data -> rusttorch-core -> tch
rusttorch -------------------------------> tch
```

`rusttorch-core` owns shared runtime, device, tensor, and error contracts.
`rusttorch-data` owns datasets, samplers, batching, and loading. The
`rusttorch` facade re-exports both packages so existing facade imports remain
source-compatible; extracting package ownership does not expand the supported
API scope.

```text
eager modules ───────────────┐
                            ├─> tch::Tensor -> LibTorch -> CPU/CUDA/MPS
Graph IR -> EagerExecutor ───┘                    └──────> autograd
```

## Data path

Map-style data flows from a `Dataset` through a sampler and either the borrowed
debugging loader or the owned loader. Streaming data uses an ordinary fallible
iterator at zero workers and an explicit `WorkerSourceFactory` for sharded
positive-worker loading:

```text
borrowed Dataset + sampler -> DataLoader ──────────────┐
owned Dataset + sampler -> bounded worker lanes ───────┤
fallible iterator -> batches ──────────────────────────┼─> coordinator collation -> pinning -> model
WorkerSourceFactory -> bounded sharded worker lanes ───┘
```

Workers are bounded Rust threads rather than Python subprocesses. Map workers
share the dataset; stream workers own factory-created shards. Fetch and
deterministic transforms run in worker lanes, while collation runs once on the
coordinator and optional pinning runs immediately before yield. Sampler-local
and task-local RNGs avoid LibTorch's global random state. Item prefetch is
always bounded and an optional logical-payload byte budget adds backpressure.

Exact next-visible-batch checkpoints are a RustTorch extension. They cover
ordered replay-safe or transactional map configurations and explicitly
checkpointable ordered sharded streams; unsupported settings fail at build or
checkpoint time. Iterator drop cooperatively cancels and joins all workers, so
a native callback that ignores cancellation can delay teardown.

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
- `data`: fallible map and sharded-stream loading, bounded workers, collation,
  pinning, distributed sampling, and typed checkpoint state.
- `interop`: state naming, explicit mappings, SafeTensors, and format policy.
- `error`: structured failures at recoverable boundaries.

No second tensor store, parameter store, kernel layer, native bridge, or
differentiation system belongs in the MVP.
