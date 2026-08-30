# PyTorch Compatibility Program

**Status:** Approved on 2026-08-30

## Product credo

RustTorch is **easy to use and easy to implement, but crazy fast**.

That sentence is an engineering constraint, not an unqualified benchmark
claim:

- Common model, optimizer, data-loading, and device-selection code should be
  no more ceremonious than the equivalent PyTorch code.
- RustTorch adds static checking, explicit recoverable errors, predictable
  ownership, safe concurrency, and deployment without a Python runtime.
- LibTorch owns tensor kernels and autograd performance. RustTorch must avoid
  adding frontend overhead, copies, allocations, synchronization, or dynamic
  dispatch that are not required by the operation.
- Performance claims require a reproducible benchmark and named comparison.
  Rust alone is not evidence that a tensor operation is faster than PyTorch.

## Compatibility target

RustTorch targets PyTorch/LibTorch 2.13.0 through `tch` 0.26.0. Work proceeds
feature by feature against the pinned PyTorch tag and source reference.

The long-term objective is complete practical coverage of PyTorch's usable
capabilities from Rust. Coverage is recorded explicitly rather than expressed
as an unsupported percentage. Each compatibility item has:

- a stable identifier and PyTorch symbols;
- the RustTorch symbols, when any;
- an explicit supported scope;
- one of `supported`, `partial`, `python_only`, `planned`, or
  `not_supported`;
- one of `libtorch`, `rusttorch`, `mixed`, or `none` as its implementation;
- executable evidence for supported behavior; and
- documented differences caused by Rust or unavailable Python semantics.

An item is `supported` only inside its written scope. Python-specific behavior
such as CPython bytecode capture, arbitrary pickle objects, and dynamic class
reconstruction remains visible in the ledger even when the correct Rust
status is `python_only` or `not_supported`.

## Architecture

RustTorch remains a Rust frontend over LibTorch:

1. ATen operators, storage, dispatch, autograd, and CPU/accelerator kernels
   stay delegated to LibTorch.
2. RustTorch supplies the ergonomic, fallible Rust model, optimizer, data,
   serialization, device, and graph APIs.
3. Bindings or repetitive operator surfaces should be generated from pinned
   upstream schemas when generation is safer than handwritten wrappers.
4. Python-dependent compiler and distributed orchestration features are
   implemented through compatible artifacts or purpose-built Rust APIs; their
   Python runtime details are not falsely described as native Rust parity.

The program is divided into independently releasable subsystems:

1. runtime acquisition and packaging;
2. compatibility inventory and documentation enforcement;
3. generic data loading and deterministic sampling;
4. generated tensor/operator exposure;
5. neural-network modules and functional operations;
6. optimizers, schedulers, AMP, and training utilities;
7. serialization and model interchange;
8. sparse, quantized, nested, complex, and device-specific behavior;
9. distributed training and checkpointing;
10. export, compilation, and deployment interoperability; and
11. optional vision, audio, video, text, tabular, graph, and specialized data
    packages.

## First milestone: runtime, ledger, and data foundation

### Runtime acquisition

The `rusttorch` crate exposes a default `download-libtorch` feature that
forwards to `tch/download-libtorch`. Existing `LIBTORCH_USE_PYTORCH`,
`LIBTORCH`, and supported system installations remain higher-priority escape
hatches. docs.rs disables default features and enables `tch/doc-only` so its
network-free builds remain valid.

A companion Cargo package named `rusttorch-cli` installs a binary named
`rusttorch` without depending on `tch`, so it can run before LibTorch exists.
It supports exactly:

```text
rusttorch setup --backend auto
rusttorch setup --backend cpu
rusttorch setup --backend cuda-12.6
```

Setup runs in a Cargo project, safely updates project-local Cargo
configuration, selects a backend-specific target directory so CPU and CUDA
artifacts cannot be confused, and launches a Cargo check that triggers the
official upstream downloader.

- `auto` reuses an explicitly configured LibTorch/Python installation. With
  no explicit installation, macOS selects the CPU/MPS distribution; Linux and
  Windows select CUDA 12.6 only when `nvidia-smi` reports a driver compatible
  with CUDA 12.x, otherwise CPU.
- `cpu` selects the CPU/MPS distribution and refuses conflicting active
  LibTorch-selection environment variables.
- `cuda-12.6` maps to `TORCH_CUDA_VERSION=cu126`, is accepted only on Linux or
  Windows, validates a compatible NVIDIA driver, and never installs or
  modifies the system driver.

Setup preserves unrelated Cargo configuration. A failed download or Cargo
check returns a non-zero status and a direct recovery message. Downloaded
native artifacts are never committed or included in a crates.io archive.

### Compatibility ledger

`compat/pytorch_api.toml` is the single canonical ledger. A standard-library
Python checker validates its schema, stable sorted identifiers, enum values,
source/evidence paths, pinned package versions, and generated documentation.
`docs/api-coverage.md` is generated from that ledger and included on the
docs.rs crate landing page. Rust documentation compilation denies missing
public documentation.

The ledger must include both shipped capabilities and the major unimplemented
PyTorch areas. It must never calculate or advertise a percentage from rows
with unequal scope.

### Data foundation

The initial `rusttorch::data` API is format-agnostic and allocation-conscious:

- `Dataset` is a fallible map-style dataset with `len`, `is_empty`, `get`, and
  a `samples` adapter.
- `SequentialSampler` and `RandomSampler` are ordinary monomorphized Rust
  iterators. Random sampling uses a local seeded RNG and does not mutate
  LibTorch's global RNG.
- `DataLoader` batches any ordinary fallible iterator. This covers map
  datasets through `Dataset::samples` and streaming sources through their
  existing Rust iterators without an `IterableDataset` wrapper.
- Default collation moves samples into a pre-sized `Vec`. A closure can replace
  collation for tensor stacking, padding, or structured batches.
- `batch_size` must be non-zero, `drop_last` is explicit, source and collation
  failures stop iteration, and no background worker is leaked.

Phase one is single-threaded and deterministic. It deliberately has no trait
objects, `Arc`, channels, worker pool, process abstraction, or tensor copy.
Threaded bounded prefetch, pinning, distributed sharding, and exact loader
checkpoint/resume are later data milestones and require their own benchmarks
and parity tests.

Format support belongs in optional packages. Image, audio/video, text, and
tabular decoders transform bytes or records into typed samples consumed by the
same loader; the core crate does not claim to decode every format itself.

## Quality and release gates

Every public compatibility item must have:

- complete rustdoc and an ergonomic example;
- a narrow Rust test that fails without the behavior;
- Python parity evidence when matching a Python-visible default or numerical
  result;
- backend evidence for every backend claimed in its scope;
- compatibility-ledger and generated-document updates; and
- no measurable regression in a relevant existing benchmark.

The workspace gates are formatting, compatibility-ledger validation, checks,
Clippy with warnings denied, all tests, rustdoc with warnings denied, package
inspection, and Python/backend parity where applicable. Releases are versioned
milestones; GitHub and crates.io descriptions state exactly what is supported
and what remains planned.

