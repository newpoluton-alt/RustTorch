# rusttorch-data

Typed datasets, samplers, batching, and loading for RustTorch.

## Installation

```toml
[dependencies]
rusttorch-data = { version = "0.1", features = ["download-libtorch"] }
```

## Features

This package has no default features. `download-libtorch` forwards to `tch` to
acquire its compatible LibTorch 2.13.0 runtime. `doc-only` is for checks and
rustdoc; it intentionally does not provide a runtime for an executable.

## Native runtime

An executable needs LibTorch/PyTorch 2.13.0, matching `tch` 0.26.0. Use
`download-libtorch`, or disable default features and build with either
`LIBTORCH_USE_PYTORCH=1` against an installed Python `torch` 2.13.0 or
`LIBTORCH=/absolute/path/to/libtorch`. The platform dynamic loader must find
the selected runtime's shared libraries when the executable runs.

## Example

Use this package directly when an application wants the data layer separately:

```rust
use std::convert::Infallible;

use rusttorch_data::{DataLoader, Dataset, SequentialSampler};

struct Rows([i64; 3]);

impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(self.0[index])
    }
}

fn main() {
    let rows = Rows([2, 3, 5]);
    let batches = DataLoader::new(&rows, SequentialSampler::new(rows.len()), 2, false)
        .expect("batch size is nonzero")
        .collect::<Result<Vec<_>, _>>()
        .expect("rows are infallible");

    assert_eq!(batches, vec![vec![2, 3], vec![5]]);
}
```

The `rusttorch::data` facade re-exports the same API and is the seamless
default for applications already using `rusttorch`.

Owned map loaders can select bounded Rust workers with `.workers(count)` and
`.prefetch_factor(batches_per_worker)`. Map datasets are shared through
`Arc`; workers fetch and transform in deterministic lanes, while collation
runs on the coordinator. Results preserve sampler order unless
`.in_order(false)` is selected. Build rejects a conservative aggregate of
concrete task/control/completion slot buffers, worker/vector bookkeeping, and
bounded channel control-block allowances above 64 MiB before sampler or
batch-source callbacks.

Positive-worker `.timeout(duration)` must fit the platform monotonic clock
range and applies a fresh deadline to each blocking `next()` call. Dataset and
transform contexts can observe that deadline or cooperative cancellation
without polling. Persistent pools reuse
their threads, initial worker seeds, initializer calls, and transforms across
iterator generations, but remain owned and joined by the loader. Iterator drop
cancels and quiesces its generation; owner drop shuts down the pool. Rust
cannot force-cancel an arbitrary blocking `Dataset::get`, system call, or
native decoder, so drop waits until non-cooperative work returns.

`StreamDataLoaderBuilder` is the positive-worker path for streaming data. A
`WorkerSourceFactory` creates one independently owned shard iterator per worker
and generation. Ordered records carry one global, zero-based contiguous
`SequenceId`; unordered records may omit it but still carry a stable
`LogicalSampleId` for deterministic task randomness. Per-worker credits and a
bounded global result/reassembly budget prevent a fast shard from starving the
worker holding the next ordered record. The coordinator merges records before
collation and drops at most one global tail. Persistent stream pools recreate
sources with fresh generation cancellation and seeds while retaining their
threads, initializer calls, and transform state.

Ordered reassembly uses one loader-owned flat slot vector whose actual retained
capacity is aggregate-checked before source callbacks and reused across
generations. Lookup is linear in the deliberately bounded window, so larger
prefetch factors trade wider disorder tolerance for memory and scan cost.

Ordered reassembly supports delayed IDs while some worker credit remains
outside the reassembly window. If all `workers * prefetch_factor` credits are
held by higher IDs and the next global ID is absent, the loader reports a typed
protocol error: no shard can advance to discover another record or its end.
Increase the factor or ensure the missing lower ID is produced by a shard with
an independently available credit when a source intentionally emits a wider
out-of-order window. Ready records, failures, and end markers take precedence
over an expiring per-`next` deadline.

Map and stream workers can additionally select
`.prefetch_bytes(NonZeroUsize)`. This strict generation-scoped budget measures
the final transformed values held in result queues and ordered reassembly;
the coordinator releases their permits when it moves those values into its
one active item-bounded collation batch. `MemoryFootprint` reports conservative
logical payload rather than process RSS, so tensor views and allocator
metadata are deliberately outside the estimate. Ordered byte-bounded streams
also require each shard's sequence IDs to increase strictly. One bounded front
waiter per shard lets a missing global ID fail as a typed protocol error rather
than deadlocking the byte budget.

`.pin_memory()` recursively pins each successfully collated batch for CUDA
device zero when CUDA is available. CPU-only and MPS-only runtimes record
`PinMemoryStatus::DisabledNoAccelerator` and preserve the exact batch type as a
no-op. `.pin_memory_for(Device)` accepts only an available in-range CUDA device
and rejects unsupported devices before source, transform, collator, or worker
callbacks. `PinMemory` supports tensors and the built-in recursive container
shapes; backend rejection remains a typed iteration error with its source.

## Exact serial checkpoint and resume

Owned map loaders can save a versioned, fully typed `LoaderState` at the next
consumer-visible batch boundary. Exact mode is opt-in: wrap immutable data in
`ReplaySafeMap` (or use `TensorDataset::into_replay_safe`) or wrap a stateful
dataset in `TransactionalMap`, then assign an exact caller-owned dataset
identity. The default identity transform and built-in collators are already
checkpointable.

```rust
use std::convert::Infallible;

use rusttorch_data::{
    DataLoader, Dataset, LoaderState, ReplaySafeDataset, ReplaySafeMap,
    VecCollate,
};

#[derive(Clone)]
struct Rows(Vec<i64>);

impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize { self.0.len() }
    fn get(&self, index: usize) -> Result<i64, Infallible> { Ok(self.0[index]) }
}

impl ReplaySafeDataset for Rows {}

let rows = || ReplaySafeMap::new(Rows(vec![2, 3, 5]));
let mut loader = DataLoader::builder(rows())
    .batch_size(2)
    .collate(VecCollate)
    .dataset_identity("training-rows-v1".to_owned())
    .build()?;
let mut iteration = loader.iter();
assert_eq!(iteration.next().unwrap()?, [2, 3]);
let state = iteration.checkpoint()?;

// Storage is caller-selected; `LoaderState` implements serde traits.
let json = serde_json::to_string(&state)?;
let restored: LoaderState<_, _, _, _> = serde_json::from_str(&json)?;
let mut resumed = DataLoader::builder(rows())
    .batch_size(2)
    .collate(VecCollate)
    .dataset_identity("training-rows-v1".to_owned())
    .resume_from(restored)
    .build()?;
assert_eq!(resumed.iter().next().unwrap()?, [5]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The example chooses `serde_json`, which applications add as their own storage
dependency; `rusttorch-data` intentionally depends only on serde's format-neutral
traits in production.

`checkpoint()` is valid initially and after `next()` returns a successful,
fully pinned visible batch. It is deliberately unavailable after an error,
end-of-input, or a hidden `drop_last` tail. Resume rejects schema, identity,
seed, epoch, sampler, batching, ordering, rank/world, pinning, derivation
version, or cursor drift before applying component state. Stateful transforms
opt in with `.checkpoint_transactional()`; explicitly stateless transforms use
`.checkpoint_stateless()`. Stateful collators/converters implement
`Checkpointable`.

Ordered prefetched map workers also support exact replay for `ReplaySafeMap`
datasets and explicit checkpointable worker transforms. Exact workers require
`MemoryDisabled`, `NoWorkerInit`, nonpersistent execution, and no timeout.
The checkpoint storage format, atomic file replacement, retention, and encryption policy are
owned by the application; malformed storage errors are never treated as a
request to start fresh.

## Exact sharded stream checkpoint and resume

`StreamDataLoaderBuilder::checkpointable("source-contents-v1")` enables exact
stream replay. The factory implements `CheckpointSourceFactory` with a stable
`CHECKPOINT_KIND` including its format version. Each source implements
`CheckpointableSource`: owned serde state, `snapshot`, read-only
`validate_snapshot`, and infallible `restore_validated`. Its required read-only
`error_sequence(&error)` identifies each failed read's stable global position
after the failed attempt; that position must repeat after rollback and obey
the shard's increasing sequence contract. Infallible sources implement this
method by matching the uninhabited error. Transforms implement
`WorkerCheckpoint` (or use the explicit `StatelessWorker` or
`TransactionalWorker` adapters), and the coordinator implements `Checkpointable`.
The identity transform and built-in collators already supply these contracts.

Build the loader, call `let mut iteration = loader.iter()`, and capture
`let state = iteration.checkpoint()?` at a visible batch boundary. A new matching
builder uses `.resume("source-contents-v1", state).build()?`. The dedicated
`StreamLoaderState` records every shard's paired source/transform state, the
coordinator state, next batch/sequence, source length, settings, and versioned
source/transport identities. No decoded samples are serialized. Storage uses
the application's chosen serde format.

Exact streams require ordered, strictly increasing per-shard global sequence
IDs, no byte budget, no custom worker initializer, no persistent workers, and
no timeout. Ordinary iterator batching and opaque factories cannot checkpoint.
Pinning still runs after coordinator collation. Unequal and empty shards merge
before the one global tail policy is applied.

A checkpoint cancels current reads, restores paired snapshots preceding each
shard's first unconsumed attempt, drains unpublished records, and advances the
transport generation. The original iterator can continue and checkpoint again.
Source/transform errors share bounded ordered reassembly with records, so a
faster later failure cannot suppress earlier valid batches. They replay at the
same global position from the previous successful visible boundary;
coordinator/protocol failures reject checkpointing. Pre-End snapshots are
retained even for non-fused sources. Sources that retain `WorkerContext` must
override `set_run_context` to replace cancellation/deadline context before
resumed reads; logical worker seed identity remains stable.

Each shard's journal holds at most `prefetch_factor + batch_size + 1` paired
states, including the assembling batch. Typed journal/channel storage participates
in the existing 64 MiB aggregate preflight. Arbitrary heap allocations within
user states are bounded by entry count, not by total resident bytes. Cancellation
and drop still wait for non-cooperative native callbacks to return.
