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

- Tensor indexing, alias-aware views, normalization, least squares, signal
  filtering, sparse matrix workflows and inference quantization through LibTorch.
- Functional scalar gradients, vector products, Jacobians, Hessians and explicit
  surrogate gradients; Normal, Bernoulli and Categorical probability utilities.
- Dense, convolutional and transposed-convolution models; batch, instance, group
  and layer normalization; spatial pooling; embeddings and activation layers.
- RNN, LSTM, GRU, masked multi-head attention and Transformer encoder/decoder
  layers, stacks and complete sequence-to-sequence models.
- Configurable losses; Adam, AdamW, RMSprop, SGD, Adagrad, Adadelta and Adamax;
  parameter groups, five schedulers, exact optimizer state restoration,
  gradient accumulation/clipping, CUDA autocast and dense-gradient scaling.
- Typed datasets, dataset adapters, local and distributed sampling, collation,
  bounded workers, deterministic transforms, sharded streams, and exact resume
  for the documented checkpoint configurations.
- Device selection, SafeTensors weight exchange, and an explicit inspectable
  graph executor.

See [models and training](training.md) and the
[data guide](../crates/rusttorch-data/README.md) for working examples.

## Implementation order

The first delivery covers the core contracts for workstreams 1 and 2; the
0.3.0 delivery adds the API census and the scoped tensor/differentiation contracts
for workstreams 3 and 4. Together these are **4 of 8 workstreams (50% by scoped
workstream count)**. This is not 50% of APIs, effort, or complete framework parity.
Each delivery has a finite acceptance contract and a separate extension backlog:
[models/training plan](superpowers/plans/2026-09-12-quarter-roadmap.md) and
[census/tensor plan](superpowers/plans/2026-09-12-census-tensor-docs.md).
The canonical ledger records the exact numerical evidence and limitations.

| Priority | Milestone | Completion requirements |
|---|---|---|
| 1 | Core model layers — core delivered in source | All eight model contracts have implementations, Rust examples, forward/backward and configuration checks, CPU numerical comparisons and parameter/buffer persistence evidence |
| 2 | Training controls — core delivered in source | Configurable losses, seven optimizer families, parameter groups, five schedules, versioned state and exact resume; dense-gradient scaling and CUDA autocast API with hardware-conditional checks |
| 3 | Complete API census — core delivered in source | Every pinned documented public symbol and canonical ATen schema mapped to exactly one ledger disposition; deterministic refresh and offline CI validation |
| 4 | Tensor and differentiation — core delivered in source | Tested indexing, views/aliasing, conversion/reductions, linalg/FFT/special workflows, three distributions, validated sparse/quantized boundaries and functional higher derivatives; native forward AD, general custom callbacks and nested support remain extensions |
| 5 | Distributed training | Process groups, collectives, distributed model/optimizer state, DDP/FSDP equivalents, failure handling, and multi-process numerical evidence |
| 6 | Compilation and deployment | Versioned graph/operator schemas, guards, control flow, PT2/ONNX interoperability, supported artifact execution, and numerical round trips |
| 7 | Domain data packages | Vision, codecs, audio, text, and tabular pipelines following the approved shared data design; at least one real pipeline per advertised package |
| 8 | Framework tools | Profiling, testing utilities, reproducibility controls, checkpoint/export integration, and evidence on every advertised backend |

Extensions to the first two workstreams remain: packed/nested recurrent inputs,
recurrent cells, attention with different key/value feature widths, distributed
normalization, specialized padding/pooling/activation families, closure-based
and fused/capturable optimizers, additional schedules, and non-CUDA autocast dtype
policies. CUDA execution evidence requires CUDA hardware; its absence is recorded
as a skip. These extensions keep the corresponding broad ledger areas partial.

Tensor extensions remain: native dual-level forward AD and vectorized function
transforms; custom backward callbacks; complex/sparse higher derivatives; the
complete distribution catalog; compressed/nested layout operator coverage;
tracked COO coordinate construction and modern quantization/export workflows.
The current JVP uses reverse-over-reverse differentiation. Nested construction
is explicitly experimental and has no stable helper. The pinned native
quantized dtype operations are deprecated upstream; this is a scoped legacy
inference boundary, not a new quantization training framework.

The census maps documented identities and canonical ATen schemas separately
from implementation status. Every entry has a disposition; a native schema's
presence alone never promotes it to supported. See the
[maintenance guide](api-census.md), [generated coverage](api-coverage.md),
[tensor guide](tensor-workflows.md) and
[differentiation/probability guide](differentiation.md).

## Plans and evidence

| Plan | Status and next action |
|---|---|
| [Workspace foundation](superpowers/plans/2026-08-31-data-workspace-foundation.md) | Core/data packages and facade paths exist; preserve their source compatibility |
| [Complete DataLoader](superpowers/plans/2026-08-31-complete-data-loader.md) | Runtime tasks 1–13 have implementations and focused tests; task 14 has examples, repeated benchmarks, docs and CI lanes. Hardware-conditional evidence and documented unsupported checkpoint combinations remain scoped |
| [Core models and documentation](superpowers/plans/2026-09-12-core-models-and-documentation.md) | Initial model and documentation foundation implemented |
| [Quarter-roadmap delivery](superpowers/plans/2026-09-12-quarter-roadmap.md) | Defines the first two workstream contracts, evidence and remaining specialized variants |
| [API census](superpowers/plans/2026-08-31-pytorch-api-census.md) | Semantic inventory, canonical schemas, complete dispositions, deterministic refresh and offline enforcement delivered; refresh the complete snapshot when changing the pin |
| [Census, tensors and documentation](superpowers/plans/2026-09-12-census-tensor-docs.md) | Defines second-delivery contracts, executable examples, numerical evidence and remaining extensions |
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
