# Changelog

All notable changes are recorded here. The 0.x series does not yet promise API
or graph-format stability.

## Unreleased

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
