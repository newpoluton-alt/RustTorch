# RustTorch roadmap

RustTorch's goal is to make PyTorch's framework functionality available through
ergonomic Rust APIs: tensor operations, automatic differentiation, model
building, training, data loading, and deployment. The implementation uses
LibTorch kernels through `tch` wherever available. Python-specific language
mechanisms need a documented Rust equivalent or artifact boundary.

This is the target, not a claim that all functionality is available today.
[API coverage](api-coverage.md) records exact implemented scopes and evidence.
A broad area stays partial until its remaining options and behaviors are
verified. A symbol reexport or a matching name alone does not establish parity.

## Available foundations

- Eager tensor operations and reverse-mode differentiation through LibTorch.
- Dense models, convolution in one to three spatial dimensions, layer
  normalization, embedding lookup, and common activation/dropout layers.
- Adam, AdamW, RMSprop, SGD, scalar loss updates, gradient accumulation,
  clipping, and learning-rate adjustment.
- Typed datasets, dataset adapters, local and distributed sampling, collation,
  bounded workers, deterministic transforms, sharded streams, and exact resume
  for the documented checkpoint configurations.
- Device selection, SafeTensors weight exchange, and an explicit inspectable
  graph executor.

See [models and training](training.md) and the
[data guide](../crates/rusttorch-data/README.md) for working examples.

## Implementation order

Core models and training take priority. Each row represents a separate
implementation milestone; completing one does not mark the entire framework
complete.

| Priority | Milestone | Completion requirements |
|---|---|---|
| 1 | Core model layers | Remaining normalization, pooling, convolution variants, activations, recurrent layers, attention, and transformers; forward/backward, defaults, train/eval, parameter/buffer state, and shape/error tests |
| 2 | Training controls | Configurable losses, optimizer families/options and parameter groups, schedulers, optimizer checkpoint state, mixed precision and gradient scaling; multi-step parity and save/resume evidence |
| 3 | Complete API census | Every pinned documented public symbol and canonical ATen schema mapped to exactly one ledger disposition; deterministic refresh and offline CI validation |
| 4 | Tensor and differentiation coverage | Indexing, views/aliasing, dtype/device conversion, reductions, linalg, FFT, special functions, random/distributions, sparse/quantized/nested layouts, forward AD, and custom gradients |
| 5 | Distributed training | Process groups, collectives, distributed model/optimizer state, DDP/FSDP equivalents, failure handling, and multi-process numerical evidence |
| 6 | Compilation and deployment | Versioned graph/operator schemas, guards, control flow, PT2/ONNX interoperability, supported artifact execution, and numerical round trips |
| 7 | Domain data packages | Vision, codecs, audio, text, and tabular pipelines following the approved shared data design; at least one real pipeline per advertised package |
| 8 | Framework tools | Profiling, testing utilities, reproducibility controls, checkpoint/export integration, and evidence on every advertised backend |

The census may proceed alongside core model work. Its committed synchronizer,
full inventory, and one-to-one mapping are still pending; the current ledger
is an area inventory, not an exhaustive list of all public symbols. Until the
census exists, no claim of complete symbol coverage is justified.

## Plans and evidence

| Plan | Status and next action |
|---|---|
| [Workspace foundation](superpowers/plans/2026-08-31-data-workspace-foundation.md) | Core/data packages and facade paths exist; preserve their source compatibility |
| [Complete DataLoader](superpowers/plans/2026-08-31-complete-data-loader.md) | Runtime tasks 1–13 have implementations and focused tests; task 14 has examples, repeated benchmarks, docs and CI lanes. Hardware-conditional evidence and documented unsupported checkpoint combinations remain scoped |
| [Core models and documentation](superpowers/plans/2026-09-12-core-models-and-documentation.md) | Defines the present contribution and its review/validation criteria |
| [API census](superpowers/plans/2026-08-31-pytorch-api-census.md) | Pending semantic inventory tooling, pinned source refresh, complete mappings, and offline enforcement |
| [Shared data ecosystem](superpowers/specs/2026-08-31-unified-data-ecosystem-design.md) | Loader foundation exists; optional modality packages and their integration remain pending |
| [Graph compilation](graph-compilation-roadmap.md) | Eager graph execution exists; compilation and lowering remain pending |
| [PT2 interoperability](torch-export-roadmap.md) | Archive import, schema validation, operator mapping, and round trips remain pending |

Older plan checkboxes describe the original implementation procedure. This
index and the executable compatibility ledger distinguish shipped source from
remaining work; unexecuted platform or release checks must remain unclaimed.

## Contribution contract

For each coherent feature family:

1. Open an issue and record the intended Rust API and upstream behavioral scope.
2. Reference the pinned source, reuse existing runtime capabilities, and add
   focused failing tests for values, gradients, configuration, and failures.
3. Implement the feature with a Rust example and public API documentation.
4. Run numerical comparison where compatibility is claimed and record backend
   evidence separately from hardware skips.
5. Update the canonical ledger, regenerate coverage, add a changelog entry,
   and pass the contribution checks before a DCO-signed commit and review.

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for the exact checks and
[porting policy](porting-policy.md) for provenance. Source attribution belongs
in those references; application guides should explain what a RustTorch user
can build and how to build it.
