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
rusttorch-data = { version = "0.3", features = ["download-libtorch"] }
rusttorch-core = "0.3"
```

`rusttorch-core` supplies `Tensor`, `Kind` and `Device` for tensor pipelines.
Use the same release series for both packages.

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

A complete borrowed pass keeps the dataset available afterward. It does not
stack samples automatically, so the output here is `Vec<i64>`:

```rust
use rusttorch_data::{DataLoader, Dataset};
use std::convert::Infallible;
struct Rows([i64; 3]);
impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, i: usize) -> Result<i64, Infallible> {
        Ok(self.0[i])
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Rows([5, 8, 13]);
    let loader = DataLoader::new(&dataset, 0..dataset.len(), 2, false)?;
    assert_eq!(
        loader.collect::<Result<Vec<_>, _>>()?,
        [vec![5, 8], vec![13]]
    );
    assert_eq!(dataset.get(0)?, 5); // The dataset is still available.
    Ok(())
}
```

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

## Train with typed feature/label batches

When building a complete application, use the facade dependency instead of
importing the data and core crates separately:

```toml
[dependencies]
rusttorch = { version = "0.3", features = ["download-libtorch"] }
```

The default collator turns `(Tensor, i64)` samples into `(Tensor, Tensor)`
batches. Feature rows are stacked, and labels become an `Int64` tensor suitable
for classification loss. The dataset below stores ordinary Rust values and
creates each tensor when fetched, so workers can share it safely.

```rust
use rusttorch::{
    DeviceSpec, Tensor,
    data::{DataLoader, Dataset},
    nn::{Sequential, functional},
    optim::Adam,
};
use std::convert::Infallible;

struct Examples(Vec<([f32; 2], i64)>);
impl Dataset for Examples {
    type Sample = (Tensor, i64);
    type Error = Infallible;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Result<Self::Sample, Infallible> {
        let (features, label) = &self.0[index];
        Ok((Tensor::from_slice(features), *label))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let examples = Examples(vec![
        ([-1., 0.], 0),
        ([1., 0.], 1),
        ([-2., 1.], 0),
        ([2., 1.], 1),
    ]);
    let mut loader = DataLoader::builder(examples)
        .batch_size(2)
        .shuffle(42)?
        .workers(2)
        .prefetch_factor(2)
        .build()?;
    let model = Sequential::builder().linear(2, 2).build(DeviceSpec::Cpu)?;
    let mut optimizer = Adam::builder()
        .learning_rate(0.01)
        .build(model.var_store())?;

    for epoch in 0..3 {
        loader.set_epoch(epoch);
        for batch in loader.iter() {
            let (features, labels) = batch?;
            assert_eq!(features.size(), [2, 2]);
            assert_eq!(labels.size(), [2]);
            let logits = model.forward_t(&features, true)?;
            let loss = functional::cross_entropy(&logits, &labels)?;
            optimizer.backward_step(&loss)?;
        }
    }
    Ok(())
}
```

Calling `iter()` again repeats the selected epoch. Call `set_epoch(epoch)` to
select that epoch's deterministic shuffle before its first batch. `.shuffle(42)`
sets the sampler seed; `.seed(...)` separately controls task/worker randomness.
The example keeps the model and inputs on CPU. When training on an accelerator,
move both feature and target tensors to the model's device before the loss.

## Pad sequences or return individual samples

Custom collation runs after per-sample transforms. Use it to pad variable-length
sequences and retain the lengths needed by the model. The following program
returns token IDs shaped `[batch, longest_sequence]` and an `Int64` length vector.
Zero is the chosen padding token; use a different value if your vocabulary
assigns zero to an ordinary token.

```rust
use rusttorch_core::Tensor;
use rusttorch_data::{DataLoader, Dataset, FnCollate};
use std::convert::Infallible;
struct Sentences(Vec<Vec<i64>>);
impl Dataset for Sentences {
    type Sample = Vec<i64>;
    type Error = Infallible;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Result<Self::Sample, Infallible> {
        Ok(self.0[index].clone())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let padding = FnCollate::new(|rows: Vec<Vec<i64>>| {
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        let lengths: Vec<_> = rows.iter().map(|row| row.len() as i64).collect();
        let count = rows.len();
        let mut values = Vec::new();
        for mut row in rows {
            row.resize(width, 0);
            values.extend(row);
        }
        Ok::<_, rusttorch_core::RustTorchError>((
            Tensor::f_from_slice(&values)?.f_reshape([count as i64, width as i64])?,
            Tensor::f_from_slice(&lengths)?,
        ))
    });
    let mut loader = DataLoader::builder(Sentences(vec![vec![1, 2], vec![3, 4, 5]]))
        .batch_size(2)
        .collate(padding)
        .build()?;
    let (tokens, lengths) = loader.iter().next().unwrap()?;
    assert_eq!(
        Vec::<Vec<i64>>::try_from(&tokens)?,
        [vec![1, 2, 0], vec![3, 4, 5]]
    );
    assert_eq!(Vec::<i64>::try_from(&lengths)?, [2, 3]);
    Ok(())
}
```

For inference on one record at a time, or when each record is already a batch,
use `without_batching()`. It converts samples without inserting a batch axis;
do not combine it with `batch_size` or `drop_last`.

```rust
use rusttorch_core::Tensor;
use rusttorch_data::{DataLoader, TensorDataset};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let features = Tensor::from_slice(&[1_f32, 2., 3., 4.]).reshape([2, 2]);
    let mut loader = DataLoader::builder(TensorDataset::new(vec![features])?)
        .without_batching()
        .build()?;
    for sample in loader.iter() {
        assert_eq!(sample?[0].size(), [2]);
    }
    Ok(())
}
```

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

## Read lazy stream shards

A worker source factory opens disjoint records for each worker. The loader does
not partition your data automatically: returning the full input from every
worker duplicates records. Use the worker ID and worker count to select a shard.
This program keeps only a lazy range cursor per shard and merges five records
into `[0, 1]`, `[2, 3]`, and one global tail `[4]`.

```rust
use rusttorch_data::{
    LogicalSampleId, SequenceId, StreamDataLoaderBuilder, VecCollate, WorkerContext, WorkerRecord,
    WorkerSourceFactory,
};
use std::{convert::Infallible, iter::StepBy, ops::Range};
struct Rows(usize);
struct Shard(StepBy<Range<usize>>);
impl Iterator for Shard {
    type Item = Result<WorkerRecord<usize>, Infallible>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|row| {
            Ok(WorkerRecord {
                sequence: Some(SequenceId::new(row as u64)),
                logical_id: LogicalSampleId::new(row as u64),
                sample: row,
            })
        })
    }
}
impl WorkerSourceFactory for Rows {
    type Sample = usize;
    type Error = Infallible;
    type Source = Shard;
    fn create(&self, worker: WorkerContext) -> Result<Shard, Infallible> {
        Ok(Shard(
            (worker.info.id..self.0).step_by(worker.info.num_workers),
        ))
    }
    fn exact_len(&self) -> Option<usize> {
        Some(self.0)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut loader = StreamDataLoaderBuilder::new(Rows(5))
        .workers(2)
        .prefetch_factor(2)
        .batch_size(2)
        .collate(VecCollate)
        .build()?;
    let batches = loader.iter().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(batches, [vec![0, 1], vec![2, 3], vec![4]]);
    Ok(())
}
```

For an existing serial iterator, `batches(iterator, size, drop_last)` is the
smaller entry point. Use worker shards when a source can be opened independently
and preparation benefits from concurrency. A source without an exact length
can leave `exact_len()` at its default `None`.

## Checkpoint and resume a training input pipeline

[`examples/checkpoint.rs`](examples/checkpoint.rs) saves an ordered, prefetched
map loader after one batch, round-trips its state through JSON, and verifies
that a recreated loader returns exactly the same remaining samples:

```sh
cargo run -p rusttorch-data --example checkpoint --locked
```

Applications choose the storage format. The complete example below uses JSON,
so add `serde_json = "1"` alongside the data dependencies. The library uses
serde's format-neutral traits in production.

```rust
use rusttorch_data::{DataLoader, Dataset, ReplaySafeDataset, ReplaySafeMap, VecCollate};
use std::convert::Infallible;
struct Rows([i64; 5]);
impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Result<i64, Infallible> {
        Ok(self.0[index])
    }
}
impl ReplaySafeDataset for Rows {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let build = || {
        DataLoader::builder(ReplaySafeMap::new(Rows([2, 3, 5, 7, 11])))
            .batch_size(2)
            .workers(2)
            .prefetch_factor(2)
            .collate(VecCollate)
            .dataset_identity("prime-rows-v1".to_owned())
    };
    let mut loader = build().build()?;
    let mut iteration = loader.iter();
    assert_eq!(iteration.next().unwrap()?, [2, 3]);
    let saved = serde_json::to_string(&iteration.checkpoint()?)?;
    let uninterrupted = iteration.collect::<Result<Vec<_>, _>>()?;

    let mut restored = build().resume_from(serde_json::from_str(&saved)?).build()?;
    let remaining = restored.iter().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(remaining, [vec![5, 7], vec![11]]);
    assert_eq!(remaining, uninterrupted);
    Ok(())
}
```

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

### Complete stream cursor and JSON restore

A stream reader must save every piece of state affecting its next record.
This example stores each shard's local cursor, validates it without mutation,
and replaces its cancellation context when reads resume. The factory kind
versions the cursor format; the content identity versions the actual rows.
Use the data/core dependencies above plus `serde_json = "1"`.

```rust
use rusttorch_data::{
    CheckpointSourceFactory, CheckpointableSource, LogicalSampleId, SequenceId,
    StreamDataLoaderBuilder, StreamLoaderState, VecCollate, WorkerContext, WorkerRecord,
    WorkerSourceFactory,
};
use std::io;

const ROWS: usize = 11;
struct Rows;
struct Shard {
    cursor: usize,
    context: WorkerContext,
}
impl Shard {
    fn len(&self) -> usize {
        ROWS.saturating_sub(self.context.info.id)
            .div_ceil(self.context.info.num_workers)
    }
}
impl Iterator for Shard {
    type Item = Result<WorkerRecord<usize>, io::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.context.cancellation.is_cancelled() || self.cursor == self.len() {
            return None;
        }
        let row = self.context.info.id + self.cursor * self.context.info.num_workers;
        self.cursor += 1;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(row as u64)),
            logical_id: LogicalSampleId::new(row as u64),
            sample: row,
        }))
    }
}
impl WorkerSourceFactory for Rows {
    type Sample = usize;
    type Error = io::Error;
    type Source = Shard;
    fn create(&self, context: WorkerContext) -> Result<Shard, io::Error> {
        Ok(Shard { cursor: 0, context })
    }
    fn exact_len(&self) -> Option<usize> {
        Some(ROWS)
    }
}
impl CheckpointSourceFactory for Rows {
    const CHECKPOINT_KIND: &'static str = "example.modulo-rows.v1";
}
impl CheckpointableSource for Shard {
    type Sample = usize;
    type Error = io::Error;
    type State = usize;
    fn snapshot(&self) -> usize {
        self.cursor
    }
    fn validate_snapshot(&self, cursor: &usize) -> Result<(), io::Error> {
        if *cursor > self.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid shard cursor",
            ));
        }
        Ok(())
    }
    fn restore_validated(&mut self, cursor: &usize) {
        self.cursor = *cursor;
    }
    fn set_run_context(&mut self, context: WorkerContext) {
        self.context = context;
    }
    fn error_sequence(&self, _: &io::Error) -> SequenceId {
        unreachable!("this in-memory source never returns a read error")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let build = || {
        StreamDataLoaderBuilder::new(Rows)
            .workers(3)
            .prefetch_factor(2)
            .batch_size(4)
            .collate(VecCollate)
    };
    let mut loader = build().checkpointable("rows-0-through-10.v1").build()?;
    let mut iteration = loader.iter();
    assert_eq!(iteration.next().unwrap()?, [0, 1, 2, 3]);
    let saved = serde_json::to_string(&iteration.checkpoint()?)?;
    let uninterrupted = iteration.collect::<Result<Vec<_>, _>>()?;

    let state: StreamLoaderState<usize> = serde_json::from_str(&saved)?;
    let mut restored = build().resume("rows-0-through-10.v1", state).build()?;
    let remaining = restored.iter().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(remaining, [vec![4, 5, 6, 7], vec![8, 9, 10]]);
    assert_eq!(remaining, uninterrupted);
    Ok(())
}
```

A real file decoder may also need a byte offset, decoder buffer state and
version information. It must report a stable sequence position for read
failures through `error_sequence`. The example only produces successful reads;
its `io::Error` type is used to reject invalid cursor state.

Checkpoints save component state, not decoded samples. Prefetched work is
rolled back to the next consumer-visible boundary and replayed; the original
iterator can continue after checkpointing. Map checkpoints require an initial
or successful batch boundary before any iteration error or observed exhaustion.
Exact streams can also replay a source/transform error from their last successful
boundary; collation/pinning errors, observed exhaustion and hidden dropped tails
cannot be checkpointed. Resume checks the
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
