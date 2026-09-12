# rusttorch-data

Build training and inference input pipelines from Rust datasets and iterators.
Select rows, shuffle each epoch, transform samples, assemble typed batches,
and prefetch with bounded worker threads. Long-running jobs can save a loader
checkpoint and resume from the next unconsumed batch.

Applications using `rusttorch` can import the same API from `rusttorch::data`.
Use this package directly when you only need the data layer.

## Installation

```toml
[dependencies]
rusttorch-data = { git = "https://github.com/newpoluton-alt/RustTorch", features = ["download-libtorch"] }
rusttorch-core = { git = "https://github.com/newpoluton-alt/RustTorch" }
```

`rusttorch-core` supplies `Tensor`, `Kind` and `Device` for tensor pipelines.
The workspace packages currently install from Git; keep both dependencies on
the same repository revision when pinning them.

## Features

| Feature | Use it for |
|---|---|
| No default features | Link to a runtime you provide. |
| `download-libtorch` | Download the compatible LibTorch 2.13.0 runtime through `tch`. |
| `doc-only` | Build API documentation and type-check without a native runtime. Executables need a real runtime. |

## Native runtime

RustTorch tensors use LibTorch through `tch` 0.26.0. An executable needs
LibTorch 2.13.0 and its shared libraries on the platform's library search path.
Instead of downloading it, set `LIBTORCH=/absolute/path/to/libtorch`, or use
`LIBTORCH_USE_PYTORCH=1` with an installed Python `torch` 2.13.0. The Python
installation is an optional way to provide these native libraries; the loader
and application code are Rust.

## Example: batch features and labels

`TensorDataset` keeps matching feature and label tensors aligned. Its first
axis indexes samples; `DefaultCollator` stacks the selected rows into a batch.
Use this path for small datasets already held in memory.

```rust
use rusttorch_core::Tensor;
use rusttorch_data::{DataLoader, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = Tensor::from_slice(&[1_f32, 2., 3., 4., 5., 6.]).reshape([3, 2]);
    let labels = Tensor::from_slice(&[0_i64, 1, 0]);
    let dataset = TensorDataset::new(vec![features, labels])?;
    let mut loader = DataLoader::builder(dataset)
        .shuffle(42)?
        .batch_size(2)
        .build()?;

    assert_eq!(loader.len(), Some(2));
    for batch in loader.iter() {
        let batch = batch?;
        let inputs = &batch[0];
        let targets = &batch[1];
        assert_eq!(inputs.size()[0], targets.size()[0]);
        assert_eq!(inputs.size()[1], 2);
        // Pass inputs to your model and targets to its loss function.
    }
    Ok(())
}
```

The last batch contains one sample. Set `.drop_last(true)` when every visible
batch must have the configured size. Use `.collate(VecCollate)` to keep Rust
values as a `Vec`, or `.collate(FnCollate::new(...))` for your own padding or
batch representation. `.without_batching()` converts one sample at a time.

## Choose a loading path

| Input or use case | API |
|---|---|
| Inspect a dataset on the calling thread | `DataLoader::new(&dataset, sampler, batch_size, drop_last)` |
| Reuse a map dataset over multiple epochs | `DataLoader::builder(dataset)` and `loader.iter()` |
| Decode indexed files concurrently | Implement `Dataset`, then select `.workers(count)` |
| Batch a serial iterator | `batches(iterator, batch_size, drop_last)` |
| Read independently owned stream shards concurrently | Implement `WorkerSourceFactory`; use `StreamDataLoaderBuilder` |
| Select a subset or training/validation split | `Subset` or `random_split` |
| Combine map datasets or iterator streams | `ConcatDataset`, `StackDataset`, or `chain_datasets` |
| Address class imbalance | `WeightedRandomSampler` |
| Partition one dataset across training ranks | `DistributedSampler` |

The borrowed `DataLoader::new` returns an iterator directly: iterate over it
without `build()` or `iter()`. It keeps your dataset available to inspect after
loading and runs each fetch on the calling thread, which is useful for debugging.

For a custom `Dataset`, implement `len()` and `get(index)`. Samples and errors
keep their concrete Rust types. `get_batch(indices)` is optional: override it
when one database query or decode operation can fetch an entire index batch.

[`examples/loader.rs`](examples/loader.rs) is a complete, runnable example of:

- shuffled tensor batches with four workers and automatic pinning;
- distributed sampling with `set_epoch(epoch)` before each epoch; and
- lazy stream shards merged in global order with one final short batch.

In a repository checkout with the native runtime configured, run:

```sh
cargo run -p rusttorch-data --example loader --locked
```

Distributed ranks must agree on dataset length, replica count, seed and epoch.
Set `.rank(rank)` on the loader as well as selecting the sampler's rank.
The sampler partitions input indices; gradient synchronization belongs to
training. Sampler `drop_last` controls division across ranks; loader
`drop_last` separately controls the final local batch.

## Prefetch, transform, and transfer

Owned map loaders default to the calling thread. `.workers(4)` shares the map
dataset through `Arc`, so the dataset must be `Send + Sync`; samples must be
`Send`. A dataset can return newly decoded tensors without sharing tensor
storage between threads. Serial loading accepts datasets that cannot meet
those bounds, including `TensorDataset`.

Use `.transform(...)` to transform each sample before collation, or
`.transform_factory(...)` to create a separate decoder or transform for each
worker. `TaskContext` provides deterministic randomness derived from the
loader seed, epoch, rank and logical sample identity. `.worker_init(...)`
initializes per-worker resources once per pool.

`.prefetch_factor(2)` permits at most `workers * 2` outstanding map batches or
stream records. Results preserve sampler/global sequence order by default;
`.ordered(false)` delivers completion order. For streams, each worker owns a
separate iterator. Ordered records need unique, contiguous global `SequenceId`
values starting at zero. Each record also carries a stable `LogicalSampleId`
for task randomness. Stream collation and `drop_last` apply globally after
merging, so uneven shards do not each lose their own tail.

For a stream that emits records out of order, the next global ID must remain
reachable within the prefetch window. Exhausting the window with higher IDs
reports a protocol error. Increase the factor or change the sharding strategy
when a legitimate source needs a wider window.

`.prefetch_bytes(NonZeroUsize)` additionally limits the logical payload held in
prefetched transformed results. Map budgets must fit at least one complete
transformed batch; stream budgets must fit one record. The final transformed
type implements `MemoryFootprint`. Byte-bounded ordered streams require
strictly increasing IDs within each shard. This budget excludes the current
collation batch, callback allocations, tensor aliasing and allocator metadata;
it is not a process memory limit. Build also rejects configurations whose
checked aggregate queue/bookkeeping allocation exceeds 64 MiB.

`.pin_memory()` pins each collated batch on available CUDA device zero.
CPU-only and MPS-only runtimes preserve the batch and report
`PinMemoryStatus::DisabledNoAccelerator`. `.pin_memory_for(Device::Cuda(index))`
selects a specific available CUDA device; unsupported selections fail before
callbacks run. Pinning prepares host memory for transfers; it does not move
batches to the model's device.

`.timeout(duration)` limits each blocking `next()` call with positive workers.
Callbacks can observe deadlines and cancellation through `WorkerContext`.
Dropping an iterator cancels its generation and waits for active work to stop.
With `.persistent_workers(true)`, threads and transforms remain alive for the
next iterator; dropping the loader shuts down and joins the pool. Rust cannot
interrupt a blocking decoder or foreign call that ignores cancellation, so
cleanup waits for that call to return.

## Checkpoint and resume a training input pipeline

[`examples/checkpoint.rs`](examples/checkpoint.rs) saves an ordered, prefetched
map loader after one batch, round-trips its state through JSON, and verifies
that a recreated loader returns exactly the same remaining samples:

```sh
cargo run -p rusttorch-data --example checkpoint --locked
```

Applications choose the storage format; add `serde_json` if using that example.
The library depends only on serde's format-neutral traits in production.

For an immutable map dataset, implement `ReplaySafeDataset`, wrap it in
`ReplaySafeMap`, and set `.dataset_identity("contents-version".to_owned())`.
Use a content/version identity that changes when the dataset changes.
`TensorDataset::into_replay_safe()` creates isolated copies for serial replay.
Stateful serial datasets use `TransactionalMap` and `TransactionalCheckpoint`.
The identity transform and built-in collators already support checkpoints;
custom components must supply the corresponding state contracts.

Call `iteration.checkpoint()?` initially or after a successful visible batch.
Save the returned `LoaderState` and resume with a matching builder's
`.resume_from(state).build()?`. For a complete training restart, save model
weights, optimizer state and training counters at the same logical step.
The loader checkpoint covers the input pipeline only.

| Loading mode | Exact checkpoint support | Component requirements |
|---|---|---|
| Borrowed map loader or `batches` | Unavailable | Use an owned loader for checkpointing. |
| Serial owned map | Available | Replay-safe or transactional dataset; checkpointable sampler, transform and collator/converter. |
| Ordered map workers | Available | Replay-safe dataset; checkpointable sampler, worker transforms and collator/converter. |
| Ordered stream workers | Available | `CheckpointSourceFactory`, `CheckpointableSource`, checkpointable worker transforms and collator. |
| Completion-order workers | Unavailable | Select ordered delivery for exact replay. |

Exact worker modes additionally require no byte budget, no custom worker
initializer, no persistent workers and no timeout. These configurations are
rejected instead of promising inexact replay. Stateful transforms can opt in
with `.checkpoint_transactional()`; explicitly stateless map transforms use
`.checkpoint_stateless()`. Stream transforms use `WorkerCheckpoint`,
`StatelessWorker`, or `TransactionalWorker`.

For streams, enable `.checkpointable("source-contents-v1")`, call
`iteration.checkpoint()`, then recreate with `.resume("source-contents-v1", state)`.
A source stores an owned serde cursor, validates it without mutation, and
restores it through `CheckpointableSource`. The factory supplies a versioned
`CHECKPOINT_KIND`. Ordered IDs must increase strictly within each shard;
`error_sequence` identifies a failed read's reproducible global position.
Sources retaining `WorkerContext` must replace it in `set_run_context` before
resumed reads. See the [source contract](https://docs.rs/rusttorch-data/latest/rusttorch_data/trait.CheckpointableSource.html).

Checkpoints save component state, not decoded samples. Prefetched work is
rolled back to the next consumer-visible boundary and replayed; the original
iterator can continue after checkpointing. Do not checkpoint after an
iteration error, end-of-input or a hidden dropped tail. Resume checks the
schema, identity and configuration before restoring component state. Propagate
storage and validation errors rather than silently restarting at the beginning.
For crash recovery, the application owns atomic storage and retention.

## Measure your workload

The dependency-free benchmark checks output checksums and reports two warm-up
runs, seven raw timings, a mean and sample standard deviation for each case.
It covers borrowed/owned serial overhead, worker counts, ordering, prefetch
factors, byte bounds, transforms, collation, automatic pinning, and checkpoint
barriers paired with equivalent work without a barrier.

```sh
rustc --version
RUSTTORCH_BENCH_HARDWARE="your CPU model; your GPU model or CPU-only" \
  cargo bench -p rusttorch-data --bench data_loader --locked
```

Preserve the output with the machine's OS version and power configuration when
sharing results. Each row reports the dataset, batch/queue settings and named
baseline; the header records architecture, compiler and runtime versions.
This generated in-memory workload measures loader overhead, not image decoding
or training throughput. Automatic pinning is a no-op when CUDA is unavailable;
that result does not measure CUDA transfer performance. No timing threshold or
speedup claim is built into the benchmark.

## API and compatibility

Read the [Rust API reference](https://docs.rs/rusttorch-data) for complete item
contracts and examples. The [compatibility record](COMPATIBILITY.md) separately
tracks tested behavior against the project's pinned upstream reference. It is
an engineering audit, not a prerequisite for using RustTorch's data API.
