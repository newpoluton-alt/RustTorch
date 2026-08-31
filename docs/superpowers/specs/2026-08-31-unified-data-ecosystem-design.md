# Unified RustTorch Data Ecosystem

**Status:** Approved on 2026-08-31

## Purpose

RustTorch will provide one coherent, typed data pipeline for training and
inference while reusing mature Rust and native libraries instead of rebuilding
codecs, parsers, tokenizers, or columnar formats. The ecosystem will live in
the existing RustTorch Git repository as a synchronized Cargo workspace.

This design establishes the workspace, dependency direction, public API,
loader contracts, modality boundaries, error model, release policy, and
quality gates. It decomposes implementation into independently reviewable
milestones. It does not claim that all of PyTorch or every format is already
implemented.

## Product constraints

- Common data-loading code must remain ordinary ergonomic Rust and should be
  no more ceremonious than the equivalent PyTorch workflow.
- LibTorch continues to own tensor storage, pinning, device transfer, kernels,
  and autograd. RustTorch must not introduce avoidable copies or a second
  tensor representation.
- Existing libraries are preferred over new RustTorch implementations when
  they satisfy correctness, maintenance, licensing, platform, and MSRV needs.
- Safe Rust bindings to C, C++, FFmpeg, CUDA, and other native libraries are
  allowed when the capability lives there. Unsafe code is isolated behind a
  safe crate boundary and audited separately.
- Expensive or native dependencies are opt-in. The default `rusttorch` build
  must not compile FFmpeg, Arrow, Parquet, image codecs, or tokenizers.
- Supported behavior requires executable evidence and exact compatibility
  scope. A crate name, type stub, or dependency declaration is not a feature.

## Chosen repository structure

RustTorch uses one Git repository with multiple publishable Cargo packages:

```text
RustTorch/
├── Cargo.toml                       # package plus workspace facade
├── crates/
│   ├── rusttorch-core/              # shared tensor/device/error contracts
│   ├── rusttorch-data/              # generic loading engine
│   ├── rusttorch-cli/               # runtime setup
│   ├── rusttorch-vision/            # image and vision samples
│   ├── rusttorch-codec/             # FFmpeg audio/video codec layer
│   ├── rusttorch-audio/             # waveform and feature transforms
│   ├── rusttorch-text/              # tokenization and sequence batching
│   └── rusttorch-tabular/           # row and columnar data
├── compat/                          # canonical compatibility ledger
└── docs/                            # workspace and package guides
```

The root package remains named `rusttorch`. Existing tensor, neural-network,
optimizer, graph, interop, device, and error paths remain available through
facade re-exports. `rusttorch-core` contains only contracts needed below the
facade, including the current `RustTorchError` and `Result` definitions.
`rusttorch-data` therefore keeps the exact existing construction-error type
instead of substituting a new crate-local error. This prevents domain crates
from depending on `rusttorch`, preserves callers that match current error
variants, and avoids a dependency cycle.

Dependency direction is acyclic. In the diagram below, `A --> B` means package
`A` depends on package `B`:

```text
rusttorch facade --> rusttorch-core
       │
       ├-----------> rusttorch-data --> rusttorch-core
       │
       └--optional-> domain crate --> rusttorch-data --> rusttorch-core

rusttorch-audio --optional--> rusttorch-codec
```

No domain crate may depend on the `rusttorch` facade. Cargo workspace
dependencies pin a single `tch` version so every package uses the same
`Tensor` type and LibTorch ABI.

## Public package experience

Every domain is usable directly:

```rust
use rusttorch_vision::{ImageFolder, VisionSample};
```

The root facade also exposes optional modules:

```toml
[dependencies]
rusttorch = { version = "0.x", features = ["vision", "audio"] }
```

```rust
use rusttorch::{audio, data, vision};
```

The root feature mapping is:

- `default` -> `download-libtorch`, preserving the existing automatic
  acquisition behavior;
- `download-libtorch` -> `rusttorch-core/download-libtorch` ->
  `tch/download-libtorch`;
- `doc-only` -> `rusttorch-core/doc-only` -> `tch/doc-only`, preserving the
  network-free docs.rs build;
- `vision` -> `rusttorch-vision`;
- `codec` -> `rusttorch-codec`;
- `audio` -> `rusttorch-audio`;
- `text` -> `rusttorch-text`;
- `tabular` -> `rusttorch-tabular`; and
- `full` -> every domain feature.

`full` is a convenience feature and is never a default. `rusttorch-data` is a
normal root dependency and remains available as `rusttorch::data`. Current
`Dataset`, `DatasetSamples`, samplers, batching functions, and `DataLoader`
paths remain source-compatible during extraction.

Lower workspace packages depend on `rusttorch-core` with default features
disabled so they cannot accidentally select a LibTorch acquisition strategy.
Tensor-using domain packages forward `download-libtorch` and `doc-only` for
direct-package consumers, while the root facade remains the seamless default
installation path. CI continues to exercise the root default and the explicit
`--no-default-features --features doc-only` documentation path.

## Dependency reuse policy

The initial adapters use the following established projects:

| Need | Upstream implementation | RustTorch responsibility |
|---|---|---|
| Tensor runtime and pinned memory | `tch` / LibTorch | Typed policy, errors, parity tests |
| Bounded worker queues | `crossbeam-channel` | Ordering, cancellation, seeding, loader lifecycle |
| Images | `image` and focused `imageproc` operations | Tensor conversion, target-aware transforms, datasets |
| FFmpeg | `rsmpeg` and `rusty_ffmpeg` | Safe decoding API, timestamps, seeking, streaming, hardware selection |
| Rust-native audio decoding | `symphonia` | Waveform samples and metadata normalization |
| Resampling | `rubato` | PyTorch-compatible configuration and tensor conversion |
| Spectral primitives | LibTorch tensor operations, with `realfft` only where needed | Spectrogram, Mel, MFCC, parity and gradients where applicable |
| Tokenization | Hugging Face `tokenizers` | Adapter trait, masks, batch policy, model-ready tensors |
| CSV | `csv` | Typed schema and preprocessing pipeline |
| JSONL | `serde_json` streaming deserialization | Typed records and source diagnostics |
| Arrow and Parquet | Apache `arrow-rs` and `parquet` | Batch-to-tensor conversion and schema policy |

Exact dependency versions are chosen in each implementation milestone and
committed in `Cargo.lock`. A candidate is accepted only when its license is
compatible, it supports RustTorch's MSRV, its required platforms pass CI, and
its relevant feature set is maintained. `THIRD_PARTY_NOTICES.md` records every
accepted native and Rust dependency.

RustTorch does not fork upstream code merely to change ergonomics. A thin
adapter, upstream contribution, or explicitly scoped compatibility difference
is preferred.

## Core data contracts

### Typed sources

`rusttorch-data` keeps two primary source shapes:

1. `Dataset` is a finite map-style source with an owned `Sample`, an associated
   error, `len`, `is_empty`, `get`, and lazy sequential iteration.
2. Any `Iterator<Item = Result<Sample, Error>>` is a streaming source and can
   be batched without an `IterableDataset` wrapper.

The existing borrowed `DataLoader::new(&dataset, ...)` remains the zero-worker
compatibility path. A threaded loader owns its dataset or accepts `Arc<D>` and
requires the dataset, samples, and errors crossing worker boundaries to meet
the appropriate `Send`, `Sync`, and `'static` bounds. RustTorch never extends a
borrowed dataset lifetime with unsafe code.

An ordinary streaming iterator also remains valid for zero-worker loading.
Threaded streaming requires a `WorkerSourceFactory` that creates an explicitly
sharded source for each worker; RustTorch does not place one iterator behind a
global lock and describe that as parallel input loading. An ordered threaded
stream must attach a unique, monotonically comparable `SequenceId` from one
global logical order to every record. The loader merges on that identifier.
A source that cannot provide global sequence identifiers is available only in
explicit unordered mode and cannot claim exact checkpoint/resume.

Domain packages define their own sample structures. A custom format joins the
pipeline by implementing `Dataset` or producing the ordinary fallible
iterator. No global registry, dynamic `Any` container, file-extension switch,
or plugin loader is required.

Random or stateful preprocessing implements a small fallible `Transform`
contract receiving a task-scoped RNG context derived from the loader seed,
epoch, rank, logical sample identifier, and transform stage. Random output is
therefore independent of which worker happens to execute the task. Pure
closures remain accepted for simple transformations, but an opaque closure is
not checkpointable unless the caller explicitly wraps it in a stateless or
checkpointable component. A stateful transform may participate in exact
checkpointing with worker prefetch only when it additionally provides
transactional snapshots that restore each worker to the committed logical
boundary; otherwise exact checkpointing requires the zero-worker path.

### Sampling and batching

The sampler layer contains:

- sequential, random, subset, weighted, and replacement sampling;
- a `BatchSampler` contract for explicit groups of indices;
- distributed sharding with rank, replica count, epoch, seed, padding, and
  drop policy;
- deterministic epoch updates; and
- serializable state for checkpointable samplers.

The loader rejects mutually exclusive `sampler`, `shuffle`, and
`batch_sampler` combinations at construction. Batch size is nonzero,
`drop_last` is explicit, and token-budget or modality-aware batching is
implemented as a batch sampler rather than a second loader.

### Collation

Default collation moves owned samples into a pre-sized `Vec`. A typed
`Collate<Sample>` contract and ordinary fallible closure can instead produce
tuples, structures, padded sequences, masks, or stacked tensors.

RustTorch provides focused implementations for tensors, scalar primitives,
options, vectors, tuples, and domain sample types. It does not reproduce
Python's runtime object registry or silently coerce unrelated Rust types.
Stateful collation must implement the checkpointable-component contract before
exact loader checkpointing is enabled. Collation runs on the coordinator only
for the next batch being made visible, so its committed snapshot is taken
immediately after the last yielded batch and is never advanced by worker
prefetch.

### Pinning

`PinMemory` is a fallible recursive trait implemented for `Tensor`, supported
containers, tuples, and domain batch structures. Tensor pinning delegates to
LibTorch. Pinning runs after collation and before a batch becomes visible to
the consumer. Unsupported devices and dtypes return contextual errors rather
than silently copying through ordinary pageable memory.

## Loader execution

The common data path is:

```text
map dataset -> sampler -> distributed shard -> batch indices ----┐
                                                                 ├-> bounded workers
stream source -> worker-aware shard -> typed records ------------┘
       -> decode/parse -> transform -> ordered results
       -> collate -> optional pin -> yielded batch
```

`DataLoader::builder` exposes the PyTorch-relevant controls with Rust defaults:

```rust
let loader = DataLoader::builder(dataset)
    .sampler(RandomSampler::seeded(42))
    .batch_size(64)
    .workers(4)
    .prefetch_factor(2)
    .pin_memory(true)
    .build()?;
```

- `workers(0)` uses the existing single-threaded path and allocates no worker
  pool or channels.
- Positive worker counts use owned, joinable threads and bounded
  multi-producer, multi-consumer queues. The dataset is shared through `Arc`
  only on this path.
- Queue capacity is derived from worker count and prefetch factor and is never
  unbounded.
- Results are yielded in sampler order by default. Unordered completion is an
  explicit builder option and is recorded in compatibility scope.
- Worker initialization has a documented worker seed for worker-aware source
  setup, while sample transformations use schedule-independent task RNGs. No
  loader path mutates LibTorch's global RNG.
- Early loader drop closes work submission, requests cooperative cancellation,
  wakes workers blocked on RustTorch queues, joins every worker that returns,
  and discards unpublished batches.
- Source, transform, collation, timeout, pinning, channel, and worker-panic
  failures are yielded once with context; iteration then terminates.
- Persistent workers are opt-in and cannot outlive the loader owner.

Rust cannot force-cancel an arbitrary blocking `Dataset::get`, system call, or
native decoder. Worker contexts therefore include a cooperative cancellation
token and deadline that built-in sources must honor between bounded operations.
Loader timeout covers queue waits and cooperating sources; it is not described
as a kill signal for foreign code. Dropping a loader waits for an active
non-cooperative call to return before its thread can be joined. This limitation
is part of the public contract, and built-in network/native sources must use
their own finite I/O deadlines.

The core loader remains a blocking Rust iterator. Async file or network
sources may use adapters in domain crates, but `tokio` is not a mandatory
dependency of `rusttorch-data`.

## Checkpoint and resume

`LoaderState` is a versioned, serializable state object representing the next
batch visible to the consumer. A checkpoint is taken only at a yielded-batch
boundary.

Exact checkpointing is enabled only for ordered loading when the source,
sampler, transforms, and stateful collator all implement their checkpoint
contracts. Opaque stateful closures, unordered delivery, non-deterministic map
access, or a stream without global sequence identifiers make exact
checkpointing unavailable and return a capability error at builder validation.
With worker prefetch, every stateful source or transform must also support a
transactional snapshot at the committed logical boundary. A merely
serializable current state is insufficient because workers may already have
processed unconsumed records.

For a map dataset it records:

- schema version and compatibility identity;
- epoch, next committed batch, and sampler state;
- loader and component states plus task-RNG derivation version;
- distributed rank/replica and padding policy; and
- relevant batching and ordering configuration.

The checkpoint barrier stops new work, accounts for ordered in-flight work,
and records the next unconsumed logical batch. It restores each prefetched
stateful component to a snapshot captured at that committed boundary before
saving state. The coordinator saves its collator state from the same boundary.
Arbitrary decoded sample values are not serialized by the generic loader, so a
component that cannot roll back and replay is rejected from exact prefetched
checkpointing rather than saved at its advanced current state.

For a map dataset, prefetched but unconsumed samples may be fetched and
transformed again after resume only when `get(index)` is deterministic for the
fixed dataset identity and every transform is stateless/task-deterministic or
transactionally restored. Task-scoped RNG derivation then makes replay
identical and the logical batch is never yielded twice.

For a stream, `CheckpointableSource` must capture the position and decoder
state before each dispatched logical record and retain boundary snapshots for
at least the configured prefetch window. At a checkpoint, each worker restores
the snapshot preceding its first unconsumed record. Sources unable to provide
that transactional boundary contract remain batchable but cannot enable exact
checkpointing with prefetch.

An ordinary Rust iterator is not falsely described as resumable. Streaming
resume is available only when the source implements a `CheckpointableSource`
contract that can save and restore its own offset and decoder state. Ordered
multi-worker resume additionally requires globally unique sequence identifiers
and checkpointable shard state. Attempting to request a checkpoint from an
unsupported stream returns an explicit error.

State deserialization validates version, dataset identity, world size, rank,
batch configuration, and sampler kind before any worker starts. Incompatible
state never falls back to a fresh epoch silently.

## Domain packages

### `rusttorch-vision`

The vision package owns typed images, bounding boxes, masks, keypoints, and
target-aware transforms. Geometric transforms update every associated target
or fail when a transformation cannot preserve its contract. Initial dataset
families are image folders, MNIST, CIFAR, and COCO-style detection data;
additional torchvision-compatible datasets are added as separate ledger rows.

Image decoding and basic operations delegate to `image`/`imageproc`. Tensor
conversion specifies channel order, dtype, range, color space, and layout.

### `rusttorch-codec`

The codec package owns media time bases, timestamps, packets, streams, decoded
audio/video frames, seeking, frame iteration, and decoder capability queries.
FFmpeg access uses `rsmpeg`; RustTorch wraps only the subset needed for safe,
typed data pipelines.

CPU decoding is the required baseline. Hardware decoding is optional and
selected only after FFmpeg reports a compatible device and decoder. CUDA,
VideoToolbox, VA-API, and other backends receive separate features and tests;
no unavailable backend silently falls back when explicitly requested.

The first codec milestone supports explicit system or `vcpkg` FFmpeg linking.
Because FFmpeg does not publish one official cross-platform binary bundle,
RustTorch will not download an unaudited third-party build. A future managed
FFmpeg mode requires pinned source/binary provenance, checksum verification,
license review, cache isolation, and platform CI before becoming a setup-CLI
option.

Official CI and release evidence uses a dynamically linked, source-built LGPL
FFmpeg profile with explicit configuration flags that disable GPL and nonfree
components. The build records FFmpeg's reported configuration, version,
enabled libraries, and link mode. A GPL or nonfree configuration cannot
produce an official RustTorch binary artifact. User-selected system and
`vcpkg` installations may differ, so diagnostics expose their configuration
and documentation makes clear that obligations follow the actual FFmpeg build
and linkage. RustTorch never labels an arbitrary system installation as
license-approved.

### `rusttorch-audio`

The audio package owns typed waveforms with sample rate, channel layout, time
origin, and sample format. It provides decoding adapters, resampling,
spectrograms, Mel filter banks, MFCC, frequency/time masking, gain, noise,
time-shift, and waveform composition.

Symphonia is the Rust-native decoding path; `rusttorch-codec` is an optional
FFmpeg path. Rubato performs sample-rate conversion. Numerical transforms use
LibTorch tensor operations where that improves PyTorch parity and device
execution.

### `rusttorch-text`

The text package defines a tokenizer adapter, vocabulary metadata, encoded
sequences, special-token policy, padding, truncation, attention masks, token
type IDs, and token-budget batch sampling. Hugging Face `tokenizers` is the
first adapter and remains accessible for advanced configuration.

Padding and truncation are explicit deterministic policies. Token-budget
batching is a `BatchSampler` over measured sequence lengths and integrates
with distributed sharding and checkpoint state.

### `rusttorch-tabular`

The tabular package provides typed CSV and JSONL streams plus optional Arrow
and Parquet readers. It owns schema mapping, categorical vocabularies, missing
value policy, numerical normalization, and conversion to model-ready tensors.

Fitted preprocessing state is immutable during iteration, serializable, and
separate from fitting. Training statistics cannot be recomputed implicitly on
validation or test data. Arrow/Parquet support is feature-gated so CSV users do
not compile the columnar stack.

## Dataset and native-asset safety

Common dataset adapters accept local roots and may offer an opt-in download
helper. Every built-in download has a pinned source identity, expected size,
cryptographic checksum, license/terms link, bounded retry policy, and atomic
cache publication. Archive extraction rejects absolute paths, parent traversal,
links escaping the destination, duplicate unsafe entries, and configured size
limits. Ordinary tests never download datasets.

FFmpeg and image decoders treat all media as untrusted input. The adapters
enforce checked size, duration, stream-count, allocation, and timestamp limits
before converting data into tensors. Native crashes cannot always be converted
into Rust errors, so supported codec/library versions are pinned and exercised
with fuzz and malformed-input corpora.

Built-in sources use a shared `ResourceLimits` policy with finite documented
defaults. It covers encoded and decoded bytes, image dimensions, tensor
elements, string and field lengths, records and columns, token counts, Arrow
batch and row-group bytes, media duration, and stream count. Limits are checked
before allocation whenever the format exposes the required metadata. Unlimited
operation is an explicit opt-in.

The generic loader always bounds queued item count. It additionally enforces a
byte budget when samples implement `MemoryFootprint`; all built-in domain
sample types implement that contract. Arbitrary user sample types remain
item-bounded unless the user supplies a `MemoryFootprint` implementation, and
the API and diagnostics state that distinction directly.

## Error model

`rusttorch-core` owns the existing non-exhaustive `RustTorchError` and `Result`
types. Existing data constructors continue returning that exact `Result`, and
existing single-threaded iteration continues yielding the dataset or collation
error type directly. Moving the implementation into `rusttorch-data` must not
change pattern matches such as `RustTorchError::InvalidConfiguration`.

The new threaded path yields `LoaderError<E>`, where `E` is the typed source or
collation error. `LoaderError<E>` preserves `E` and adds loader-lifecycle
variants for:

- dataset/source, transform, and collation failures through `E`;
- sampler and distributed configuration failures;
- pinning failures;
- worker panic, cancellation, and timeout;
- channel shutdown; and
- incompatible, corrupt, or unsupported checkpoint state.

Invalid builder configuration is rejected before iteration with the shared
`RustTorchError`. Each modality crate owns a separate non-exhaustive typed
error enum and preserves its native source chain. The root facade re-exports
those domain errors; it does not erase them into strings or pretend that every
third-party failure is an existing core variant.

Errors include available index, batch, worker, rank, path, stream, and timestamp
context. Recoverable user or data failures never panic. Internal invariants may
use assertions in tests, but unsafe/native failures must cross a checked Rust
boundary.

## Performance and memory rules

- Every queue and prefetch buffer is item-bounded; built-in sample types also
  share a configurable byte budget through `MemoryFootprint`.
- Samples move through the pipeline; cloning requires an explicit sample or
  transform contract.
- Collation preallocates known capacities and avoids an intermediate tensor
  representation.
- Domain decoding streams frames or records instead of reading whole datasets
  unless the format requires it.
- Ordered loading may buffer completed out-of-order work only up to the
  configured bounded capacity.
- Arbitrary sample types without `MemoryFootprint` are never described as
  byte-bounded.
- Benchmarks report dataset, format, hardware, worker count, queue capacity,
  batch size, and comparison baseline. The project credo alone is not a
  performance result.

## Testing and compatibility evidence

Every public behavior requires complete rustdoc, a compiling example, a
focused Rust test, and a compatibility-ledger entry. Numerical PyTorch-visible
defaults require parity fixtures generated from the pinned Python environment.

The workspace CI matrix includes:

- formatting, checks, Clippy with warnings denied, tests, and rustdoc for the
  facade and every workspace package;
- default, no-default-feature, and supported feature combinations;
- RustTorch's MSRV and stable Rust;
- Linux, macOS, and Windows for portable crates;
- exact-resume tests across epochs, partial batches, worker counts, errors,
  distributed ranks, deterministic random transforms, stateful collation, and
  early loader drop, plus rejection tests for non-checkpointable modes;
- bounded-memory and clean-shutdown stress tests, including rejection at
  record, token, tensor, Arrow row-group, media, and aggregate byte limits;
- committed tiny fixtures for images, audio, video, text, CSV, JSONL, Arrow,
  and Parquet, with no network dependency in ordinary tests;
- FFmpeg system-link lanes for supported FFmpeg majors, plus an official
  artifact lane that records configuration flags and dynamic linkage and
  rejects GPL or nonfree builds;
- conditional hardware-decoder and CUDA tests on compatible runners; and
- package inspection preventing native runtimes, generated datasets, Python
  environments, and downloaded models from entering crate archives.

Unavailable hardware is reported as skipped and is never counted as passed.
Fuzzing targets decoders' Rust boundary, collation state, and checkpoint
deserialization. Benchmarks cover single-thread overhead, scaling, bounded
prefetch, decoding, transforms, and collation.

## Documentation

Every crate denies missing public documentation and has:

- a package README with installation, features, one end-to-end example, and
  native prerequisites;
- rustdoc examples for primary source, transform, and loader paths;
- format/backend support tables tied to tests;
- a generated compatibility section sourced from the canonical ledger; and
- migration notes for every changed facade path.

Documentation covers shipped behavior, not roadmap intent. A package is not
marked supported until its documented example and relevant CI lane pass.

## Versioning and release order

All RustTorch packages use synchronized versions during the 0.x series.
Workspace dependencies include both an exact compatible version and a local
path. Release automation verifies version agreement and publishes in dependency
order:

1. `rusttorch-core`;
2. `rusttorch-data`;
3. domain crates;
4. `rusttorch-cli`; and
5. the `rusttorch` facade.

Each release packages exact archives once, generates SLSA provenance, and
publishes only after the complete workspace gate passes. Native FFmpeg or
LibTorch binaries are not embedded in crates.io archives.

## Delivery decomposition

This program is too large for one safe implementation change. Work proceeds
through these independently specified and merged milestones:

1. **Workspace foundation:** introduce `rusttorch-core` and `rusttorch-data`,
   preserve existing public paths, centralize package metadata and versions,
   and extend CI/package checks.
2. **Complete loader:** samplers, batch samplers, workers, bounded prefetch,
   pinning, distributed sharding, checkpoint/resume, docs, stress tests, and
   benchmarks.
3. **Vision:** typed targets, decoding, transforms, and initial datasets.
4. **Codec:** CPU FFmpeg decoding, time model, seeking, and streaming, followed
   by separately evidenced hardware decoding.
5. **Audio:** waveforms, resampling, spectral features, masking, and waveform
   augmentation.
6. **Text:** tokenizer adapters, vocabulary and sequence policies, masks, and
   token-budget batching.
7. **Tabular:** CSV/JSONL, Arrow/Parquet, preprocessing, and fitted-state
   serialization.
8. **Cross-modality integration:** mixed typed samples, distributed resume,
   end-to-end examples, performance baselines, and release documentation.

Full PyTorch framework coverage remains the broader compatibility program.
Tensor operators, neural-network modules, optimizers, distributed training,
compilation/export, sparse, quantization, and other PyTorch areas receive their
own scoped specifications and executable evidence. This data design neither
blocks those areas nor falsely declares them complete.

## Acceptance criteria for this design

The architecture is realized when:

- all named packages exist in this repository with the stated acyclic
  dependency direction;
- the root facade supports direct and feature-gated import styles;
- the existing data API remains source-compatible through extraction;
- the loader passes deterministic worker, bounded-prefetch, pinning,
  distributed, failure, cancellation, and resume tests;
- each domain crate has at least one real end-to-end typed pipeline before it
  is advertised as supported;
- every public symbol is documented and every support statement links to
  executable evidence;
- platform and native prerequisites are explicit; and
- CI, package inspection, provenance, and protected-main gates are green.
