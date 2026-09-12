# Changelog

All notable changes are recorded here. The 0.x series does not yet promise API
or graph-format stability.

## Unreleased

## 0.3.0 - 2026-09-12

- Add tensor workflows for indexing, alias-aware reshaping, standardization,
  least squares, spectral filtering, sparse matrices and inference quantization;
  document native linear algebra, special functions and layout boundaries.
- Add scalar functional gradients, fixed-seed vector products, Jacobians,
  Hessians and algebraic surrogate gradients. Add Normal, Bernoulli and
  Categorical sampling, scores and entropy with CPU numerical parity.
- Publish full training, sequence, tensor and differentiation tutorials as
  navigable Rustdoc pages. Expand DataLoader, builder, stream and checkpoint
  documentation with complete task examples and inline facade documentation.
- Fix the Windows stream test's false end-of-stream assertion under a 1 ns
  deadline; preserve the queued-record behavior and deterministic terminal tests.
- Add a pinned semantic API/schema census, explicit per-symbol dispositions,
  offline validation and reproducible maintainer refresh tooling.
- Record branch naming rules: `b/` features, `f/` fixes and descriptive prefixes
  for other contributions; do not create branches with `codex/`.


- Add BatchNorm/InstanceNorm1d/2d/3d, GroupNorm, transposed convolution,
  max/average/adaptive pools, seven activation layers, and generic registered
  `SequentialBuilder::layer` composition. Persist normalization buffers.
- Add RNN, LSTM with projections, GRU, masked multi-head attention, Transformer
  encoder/decoder layers, stacks and full sequence-to-sequence models.
- Add configurable regression and classification losses; optimizer parameter
  groups and versioned named moment checkpoints; Adagrad, Adadelta and Adamax;
  five learning-rate schedulers with serializable state.
- Add CUDA autocast with panic restoration and a dense-gradient scaler with
  nonfinite update skipping, accumulation, clipping and checkpoint support.
- Add executable image/sequence training guides and a composite checkpoint
  example that verifies the next update exactly after restoring all state.
- Fix Linear initialization to draw weights before bias from the fan-in uniform
  distribution. **Migration:** freshly initialized models now produce different
  seeded parameters from 0.2.0. Existing saved weights keep their names/shapes
  and load unchanged. See issue #10 and `docs/training.md` for training-state
  format and supported optimizer options.
- Use fallible tensor updates with explicit optimizer moments. Existing `step`
  and `zero_grad` methods remain; use `try_step` and `try_zero_grad` for error
  propagation. Move models to their final dtype/device before constructing an
  optimizer. Checkpoints reject inconsistent live parameter/moment metadata.

- Add fallible 1D/2D/3D convolution, layer normalization, and embedding layers
  with validated configuration, named parameters, and CPU forward/gradient tests.
- Expose the existing parameter store and parameter paths through `rusttorch::nn`
  for custom models that import only RustTorch APIs.
- Return typed errors for undefined model inputs and non-differentiable
  parameter dtypes instead of panicking or poisoning the parameter store.
- Add AdamW and RMSprop builders, learning-rate adjustment, and fallible gradient
  clipping, with multi-step optimizer parity coverage.
- Replace docs.rs implementation inventories with RustTorch tutorials and use
  cases; add a models/training guide and an explicit framework coverage roadmap.
- Complete loader examples for distributed epochs and exact prefetched resume,
  correct checkpoint/lifecycle documentation, and report repeatable benchmark
  samples with warmups and variance. Expand CI to execute doctests and loader
  integration tests and examples across platform lanes.

## 0.2.0 - 2026-09-09

- Add the shared `rusttorch-core` and `rusttorch-data` workspace crates.
- Add typed datasets, samplers, collation, threaded map and stream loaders,
  bounded prefetch, distributed sharding, memory budgets, and pinning.
- Add exact checkpoint/resume for the documented map and sharded-stream modes.
- Expand API documentation, compatibility tracking, examples, and benchmarks.
- Validate Linux, macOS, and Windows loader builds with locked Python environments.
- Add contributor rules, required CI checks, and release provenance workflows.
- Add default acquisition of the pinned official LibTorch runtime through
  `torch-sys` and keep docs.rs builds network-free.
- Add the `rusttorch setup` project bootstrap with automatic, CPU, and CUDA
  12.6 backend selection and isolated managed target directories.
- Document Python, system LibTorch, offline, CUDA-driver, and contributor
  workflows.

## 0.1.0 - 2026-08-30

- Establish the `rusttorch` MVP package and `rusttorch` library crate.
- Target `tch` 0.26.0 and PyTorch/LibTorch 2.13.0.
- Define eager modules, device selection, SafeTensors interchange, and optional
  explicit Graph IR surfaces.
- Add project-local environment, backend inspection, and Python parity tools.
- Add architecture, compatibility, backend, interop, and roadmap documentation.
- Publish complete public API documentation and docs.rs build configuration.
- Verify strict bidirectional SafeTensors loading and deterministic CPU parity
  for forward, gradients, losses, Adam, SGD, and a residual model.
- Verify Rust MPS eager/graph execution, losses, gradients, optimizer steps,
  SafeTensors transfer, and CPU↔MPS movement on the development Mac.
