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

The 0.4.0 release completes the remaining four scoped workstreams,
following the model/training delivery and 0.3.0 census/tensor delivery. All
**8 of 8 core workstream contracts now have implementations and executable
acceptance evidence**. This measures the finite delivery contracts, not full
PyTorch API parity, total engineering effort or every extension. All nine
packages are published, with [RustTorch tutorials](https://docs.rs/rusttorch/0.4.0/rusttorch/tutorials/)
and [DataLoader examples](https://docs.rs/rusttorch-data/0.4.0/rusttorch_data/) live
on docs.rs. Release evidence is recorded in the
[remaining-workstreams plan](superpowers/plans/2026-09-13-remaining-roadmap.md).
The canonical ledger continues to mark broader incomplete families as partial.

| Priority | Milestone | Completion requirements |
|---|---|---|
| 1 | Core model layers — core released | All eight model contracts have implementations, Rust examples, forward/backward and configuration checks, CPU numerical comparisons and parameter/buffer persistence evidence |
| 2 | Training controls — core released | Configurable losses, seven optimizer families, parameter groups, five schedules, versioned state and exact resume; dense-gradient scaling and CUDA autocast API with hardware-conditional checks |
| 3 | Complete API census — core released | Every pinned documented public symbol and canonical ATen schema mapped to exactly one ledger disposition; deterministic refresh and offline CI validation |
| 4 | Tensor and differentiation — core released | Tested indexing, views/aliasing, conversion/reductions, linalg/FFT/special workflows, three distributions, validated sparse/quantized boundaries and functional higher derivatives; native forward AD, general custom callbacks and nested support remain extensions |
| 5 | Distributed training — core released | Process groups, collectives, distributed model/optimizer state, DDP/FSDP equivalents, failure handling, and multi-process numerical evidence |
| 6 | Compilation and deployment — core released | Versioned graph/operator schemas, guards, control flow, PT2/ONNX interoperability, supported artifact execution, and numerical round trips |
| 7 | Domain data packages — core released | Vision, codecs, audio, text, and tabular pipelines following the approved shared data design; at least one real pipeline per advertised package |
| 8 | Framework tools — core released | Profiling, testing utilities, reproducibility controls, checkpoint/export integration, and evidence on every advertised backend |

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

Distributed extensions remain: accelerator collectives, native NCCL/Gloo bindings,
automatic backward hooks, communication overlap, multi-group/layer-wise sharding,
mesh APIs and Python distributed-checkpoint formats. CPU TCP groups and explicit
functional sharding have real multiprocess/resharding evidence; no scaling or
complete DDP/FSDP API claim follows from that core delivery.

Deployment extensions remain: arbitrary capture, AOTInductor, complete ONNX/ATen
operator coverage, custom operators and unsupported PT2/tree/schema versions.
TorchScript compilation runs real artifacts; the portable executor and selected
conditional branch gradient behavior have their own precise contracts.

Domain extensions remain format-specific: more annotations/transforms/codecs,
compressed Parquet, database adapters, additional tokenizer workflows and native
platform profiles. Profiling currently times explicit host regions; automatic
kernel/memory capture and full framework testing catalogs remain extensions.
Read [distributed training](distributed-training.md), [deployment](deployment.md),
[domain data](domain-data.md) and [framework tools](framework-tools.md) for working
Rust recipes and exact boundaries.

## Plans and evidence

| Plan | Status and next action |
|---|---|
| [Workspace foundation](superpowers/plans/2026-08-31-data-workspace-foundation.md) | Core/data packages and facade paths exist; preserve their source compatibility |
| [Complete DataLoader](superpowers/plans/2026-08-31-complete-data-loader.md) | Runtime tasks 1–13 have implementations and focused tests; task 14 has examples, repeated benchmarks, docs and CI lanes. Hardware-conditional evidence and documented unsupported checkpoint combinations remain scoped |
| [Core models and documentation](superpowers/plans/2026-09-12-core-models-and-documentation.md) | Initial model and documentation foundation implemented |
| [Quarter-roadmap delivery](superpowers/plans/2026-09-12-quarter-roadmap.md) | Defines the first two workstream contracts, evidence and remaining specialized variants |
| [API census](superpowers/plans/2026-08-31-pytorch-api-census.md) | Semantic inventory, canonical schemas, complete dispositions, deterministic refresh and offline enforcement delivered; refresh the complete snapshot when changing the pin |
| [Census, tensors and documentation](superpowers/plans/2026-09-12-census-tensor-docs.md) | Defines second-delivery contracts, executable examples, numerical evidence and remaining extensions |
| [Remaining four workstreams](superpowers/plans/2026-09-13-remaining-roadmap.md) | Distributed, deployment, five domain packages and framework tools released in 0.4.0 with passing Linux/macOS/Windows CI and live GitHub/docs.rs guides |
| [Shared data ecosystem](superpowers/specs/2026-08-31-unified-data-ecosystem-design.md) | Five optional domain packages use shared loader/collation/resource contracts; native FFmpeg and portable feature tests cover the documented pipeline profiles |
| [Graph compilation](graph-compilation-roadmap.md) | Evaluation graph lowering, guarded portable execution and actual TorchScript compilation exist; general program capture and AOTInductor remain extensions |
| [PT2 interoperability](torch-export-roadmap.md) | Pinned archive/schema validation, bounded state and operator mapping, control flow and forward/gradient round trips delivered for the documented subset |

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
