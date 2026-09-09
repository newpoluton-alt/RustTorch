# Complete DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the classic PyTorch 2.13 `torch.utils.data` dataset, sampler, collation, worker, pinning, distributed-sharding, and bounded-prefetch surface as an ergonomic, safe Rust data engine, then add exact resume as an explicitly RustTorch-native extension.

**Architecture:** `rusttorch-data` keeps the existing borrowed single-threaded API and adds an owned builder path. Map workers are bounded Rust threads sharing an `Arc<Dataset>`; streaming workers are created by an explicit shard factory. Workers fetch and transform typed samples, while the coordinator owns ordering, collation, pinning, byte-budget permits, lifecycle, and committed checkpoint boundaries. These are documented Rust-native differences from PyTorch's process-local dataset copies and worker-side collation; task RNG state is derived from logical identity rather than scheduling.

**Tech Stack:** Rust 2024, Rust 1.88 MSRV, `crossbeam-channel`, `rand`/`rand_chacha`, `serde`, `tch`/LibTorch 2.13, standard-library threads/synchronization, UV-managed Python parity fixtures.

**Spec:** `docs/superpowers/specs/2026-08-31-unified-data-ecosystem-design.md`

**Prerequisite:** `docs/superpowers/plans/2026-08-31-data-workspace-foundation.md`

## Global Constraints

- Compatibility is pinned to PyTorch `2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`, and `tch` `0.26.0`.
- Preserve `rusttorch::data::*`, `DataLoader::new`, `DataLoader::with_collate`, `batches`, `batches_with_collate`, `SequentialSampler::new`, and `RandomSampler::new(length, seed)` exactly.
- The existing borrowed constructors remain synchronous and return the existing dataset/collation error type directly.
- The new owned path uses `LoaderError<E>` and never erases `E` into a string.
- Positive worker counts own or `Arc`-share the dataset and require the values crossing threads to be `Send + 'static`; shared map datasets also require `Sync`.
- Ordinary streaming iterators remain the zero-worker API. Positive stream workers require `WorkerSourceFactory`; no iterator is hidden behind a global lock.
- Rust threads replace Python subprocesses because Rust has no GIL. Preserve observable ordering, seeding, prefetch, timeout, initialization, teardown, and persistent-worker behavior, not Python pickling/process internals.
- Queue item count is always bounded. Byte bounds apply when `MemoryFootprint` is implemented; every built-in sample type must implement it.
- Collation runs on the coordinator for the next visible batch. Pinning runs after collation and before yield.
- Task randomness depends on loader seed, epoch, rank, logical sample identifier, and transform stage; it never depends on worker assignment or LibTorch's global RNG.
- Exact checkpointing is available only for ordered, deterministic, checkpointable configurations and always means the next visible batch.
- No unsafe lifetime extension, detached worker, forced thread cancellation, unbounded queue, network-dependent ordinary test, paid service, or native binary in a crate archive.
- Before running task commands, source `. scripts/dev-env.sh`; it selects the locked UV environment, sets `LIBTORCH_USE_PYTORCH=1`, and configures the platform library path. `doc-only` is reserved for `cargo check`, Clippy, and rustdoc because it intentionally does not link a runtime.
- Every task adds focused tests, complete rustdoc, compatibility-ledger rows/evidence, regenerated `docs/api-coverage.md`, and a DCO-signed commit.

## Pinned upstream files

- `torch/utils/data/__init__.py`
- `torch/utils/data/dataset.py`
- `torch/utils/data/sampler.py`
- `torch/utils/data/distributed.py`
- `torch/utils/data/dataloader.py`
- `torch/utils/data/_utils/collate.py`
- `torch/utils/data/_utils/fetch.py`
- `torch/utils/data/_utils/worker.py`
- `torch/utils/data/_utils/pin_memory.py`

Private Python queue, pickle, daemon, signal, file-descriptor, and garbage-collection machinery is reference material only. DataPipes are a separate subsystem; `DataLoader2` is not present in the pinned core package.

---

### Task 1: Add batched fetch and PyTorch dataset adapters

**Files:**
- Create: `crates/rusttorch-data/src/dataset.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/datasets.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Preserves: `Dataset::{len, is_empty, get, samples}`.
- Adds: `Dataset::get_batch(&self, indices: &[usize]) -> Result<Vec<Sample>, Error>` with a per-index default.
- Produces: `TensorDataset`, `StackDataset`, `ConcatDataset`, `Subset`, `SplitLength`, `random_split`, and `chain_datasets`.
- Produces: blanket `Dataset` delegation for `Arc<D>` so subsets can share one source safely.

- [ ] **Step 1: Write failing adapter and batch-fast-path tests**

Add tests proving:

```rust
#[derive(Clone)]
struct IntDataset(Vec<i64>);

impl IntDataset {
    fn new(values: Vec<i64>) -> Self { Self(values) }
}

impl Dataset for IntDataset {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize { self.0.len() }
    fn get(&self, index: usize) -> std::result::Result<i64, Infallible> {
        Ok(self.0[index])
    }
}

let tensors = TensorDataset::new(vec![
    Tensor::from_slice(&[1_i64, 2, 3]),
    Tensor::from_slice(&[10_i64, 20, 30]),
])?;
assert_eq!(tensors.len(), 3);
assert_eq!(tensors.get(1)?[0].int64_value(&[]), 2);

let rows = Arc::new(IntDataset::new(vec![10, 20, 30, 40]));
let subset = Subset::new(Arc::clone(&rows), vec![3, 1])?;
assert_eq!(subset.get(0)?, 40);

let left = IntDataset::new(vec![1, 2]);
let right = IntDataset::new(vec![3]);
let concat = ConcatDataset::new(vec![left, right])?;
assert_eq!(concat.len(), 3);
```

Use an atomic counter dataset whose overridden `get_batch` increments once and whose `get` panics; assert the loader fast path calls `get_batch` once for one index batch and rejects a returned vector with the wrong cardinality.

Cover `StackDataset<(A, B)>` length equality and tuple output, empty `ConcatDataset`, out-of-range subset indices, integer split sums, fractional floor-plus-round-robin remainder, deterministic split permutations, zero-length splits, and mismatched Tensor first dimensions. Prove `TensorDataset::get` and `get_batch` preserve PyTorch's storage-sharing row-view semantics: mutate a fetched row in place and assert a later fetch sees that mutation.

- [ ] **Step 2: Run focused tests and verify missing symbols**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test datasets
```

Expected: compilation fails on `TensorDataset`, `StackDataset`, `ConcatDataset`, `Subset`, and `random_split`.

- [ ] **Step 3: Extend the dataset contract and add adapters**

Add this source-compatible default method:

```rust
fn get_batch(
    &self,
    indices: &[usize],
) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
    indices.iter().map(|&index| self.get(index)).collect()
}
```

Use these public shapes:

```rust
pub struct TensorDataset { tensors: Vec<Tensor>, len: usize }
pub struct StackDataset<T> { datasets: T, len: usize }
pub struct ConcatDataset<D> { datasets: Vec<D>, cumulative_sizes: Vec<usize> }
pub struct Subset<D> { dataset: D, indices: Vec<usize> }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitLength { Count(usize), Fraction(f64) }

pub fn random_split<D>(
    dataset: Arc<D>,
    lengths: &[SplitLength],
    seed: u64,
) -> rusttorch_core::Result<Vec<Subset<Arc<D>>>>
where D: Dataset;
```

Implement `StackDataset` tuple forms for arities two through eight with one private macro and a shared child error type. `ConcatDataset` uses cumulative sizes and binary search. Rust indices are `usize`, so do not reproduce Python negative indexing. `chain_datasets` delegates to `IntoIterator::into_iter().flatten()` and adds no custom buffering.

`TensorDataset::get`/`get_batch` use ordinary LibTorch indexing views and avoid
implicit copies, matching PyTorch. Because tensor samples have interior mutable
shared storage, Task 11 does not automatically mark this adapter replay-safe.

- [ ] **Step 4: Validate adapters and publish exact scopes**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test datasets
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Add separate ledger rows for batched fetch, TensorDataset, StackDataset tuple scope, ConcatDataset, Subset, random split, and the standard-iterator ChainDataset equivalent. Notes must identify Rust's `usize` indices and typed tuple/struct replacement for Python dictionaries.

- [ ] **Step 5: Commit**

```sh
git add crates/rusttorch-data/src crates/rusttorch-data/tests/datasets.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): add dataset adapters and batched fetch"
```

### Task 2: Complete local sampler behavior

**Files:**
- Create: `crates/rusttorch-data/src/sampler.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/samplers.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Moves without behavior change: `SequentialSampler` and current `RandomSampler::new(length, seed)`.
- Adds: `RandomSampler::without_replacement`, `RandomSampler::with_replacement`, `SubsetRandomSampler`, and `WeightedRandomSampler`.
- Adds: re-iterable `Sampler` and `BatchSource` contracts that create a fresh finite iterator for every epoch without requiring an exact length; concrete built-ins remain directly iterable for source compatibility.
- Adds: `FnSampler` and `FnBatchSource` adapters for custom epoch-aware factories. One-shot plain iterators remain accepted only by the existing borrowed/single-pass loader APIs.

- [ ] **Step 1: Write sampler parity and validation tests**

Assert these public calls:

```rust
let replacement = RandomSampler::with_replacement(4, 12, 7)?.collect::<Vec<_>>();
assert_eq!(replacement.len(), 12);
assert!(replacement.iter().all(|&index| index < 4));

let repeated = RandomSampler::without_replacement(3, 8, 7)?.collect::<Vec<_>>();
assert_eq!(repeated.len(), 8);
for permutation in repeated[..6].chunks_exact(3) {
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2]);
}

let mut subset = SubsetRandomSampler::new(vec![9, 4, 7], 3)?.collect::<Vec<_>>();
subset.sort_unstable();
assert_eq!(subset, vec![4, 7, 9]);
```

For weighted sampling, prove deterministic output, positive sample count, nonempty one-dimensional weights, finite nonnegative weights, positive total weight, replacement cardinality, and unique indices without replacement. Without replacement permits up to `weights.len()` samples even when fewer entries have positive weight; positive-weight entries are selected first and zero-weight indices may fill the remainder, matching pinned `torch.multinomial` validation. Retain the existing exact `RandomSampler::new(0, seed)` error contract.

Build one owned loader, iterate it for two epochs, and assert that sequential sampling restarts while random sampling produces the deterministic permutation for each epoch. Add custom sized and unsized `FnSampler` tests that record calls and prove their factories are invoked once per epoch. The unsized sampler must iterate normally while `loader.len()` returns `None`. These tests prevent a consumed iterator from silently producing an empty second epoch or optional PyTorch sampler length from becoming a loading requirement.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test samplers
```

Expected: compilation fails on the new sampler constructors/types.

- [ ] **Step 3: Implement samplers with local RNG state**

Use one owned index vector and cursor for finite samplers. Without-replacement `num_samples > length` concatenates complete independently shuffled permutations plus a prefix, matching PyTorch's observable behavior. Replacement uses `Rng::gen_range`. Weighted replacement uses `rand::distributions::WeightedIndex`; weighted non-replacement selects unique positive-weight indices with weighted keys, then deterministically shuffles zero-weight indices to fill any remaining requested slots. It rejects only a requested count larger than the weight vector length, not one larger than the positive-weight count. The local sequence remains intentionally different from PyTorch Philox.

Expose a fresh-iterator contract distinct from Rust's one-shot `Iterator`:

```rust
pub trait Sampler {
    type Iter: Iterator<Item = usize>;

    fn iter(&self) -> Self::Iter;
    fn exact_len(&self) -> Option<usize> { None }
    fn epoch(&self) -> u64;
    fn set_epoch(&mut self, epoch: u64);
}

pub struct FnSampler<F> {
    exact_len: Option<usize>,
    epoch: u64,
    make_iter: F,
}
```

`FnSampler<F>` calls `F: Fn(u64) -> I` with the current epoch. Concrete iterator values retain their own `position()` for checkpointing, while the reusable sampler stores configuration and epoch only. The owned loader requests a new iterator at the start of every epoch; it never stores and reuses a consumed iterator. The current seeded permutation remains local ChaCha12 and does not claim PyTorch's Philox sequence.

- [ ] **Step 4: Run tests and update evidence**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test samplers
cargo test -p rusttorch --test data_libtorch
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Expected: sampler tests and global-LibTorch-RNG isolation pass.

- [ ] **Step 5: Commit**

```sh
git add crates/rusttorch-data/src crates/rusttorch-data/tests/samplers.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): complete local samplers"
```

### Task 3: Add batch and distributed samplers

**Files:**
- Modify: `crates/rusttorch-data/src/sampler.rs`
- Create: `crates/rusttorch-data/tests/batch_sampler.rs`
- Create: `crates/rusttorch-data/tests/distributed_sampler.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: `BatchSampler<S>::new(sampler, batch_size, drop_last)` yielding `Vec<usize>`.
- Produces: `BatchSource { type Iter: Iterator<Item = Vec<usize>>; fn iter(&self) -> Self::Iter; fn exact_len(&self) -> Option<usize> { None }; fn epoch(&self) -> u64; fn set_epoch(&mut self, u64) }` plus sized/unsized `FnBatchSource`, so custom batch plans are also recreated per epoch and receive loader epoch updates without making `__len__` mandatory.
- Produces: `DistributedSampler::new(length, replicas, rank, shuffle, seed, drop_last)`.
- Produces: `DistributedSampler::{set_epoch, epoch, len, position}`.

- [ ] **Step 1: Write batch-sampler tests**

Assert size validation, ceiling/floor length, exact keep/drop tails, and nonuniform custom batch iterators:

```rust
let batches = BatchSampler::new(0..5, 2, false)?.collect::<Vec<_>>();
assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
assert_eq!(BatchSampler::new(0..5, 2, true)?.exact_len(), Some(2));
```

- [ ] **Step 2: Write distributed-sampler tests**

For dataset length 10 and three ranks, assert equal rank lengths, cyclic padding when `drop_last=false`, truncation when true, rank-strided disjoint indices before padding, invalid replica/rank errors, identical order for equal seed/epoch, and changed order after `set_epoch(1)`. Iterate sized and unsized `FnBatchSource` values at epoch zero, call `set_epoch(1)`, iterate again, and prove the closure receives both epochs and recreates both batch iterators; only the sized source reports `Some(len)`.

- [ ] **Step 3: Run tests and verify failure**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test batch_sampler --test distributed_sampler
```

Expected: compilation fails because both sampler types are absent.

- [ ] **Step 4: Implement exact index arithmetic**

Use ceiling division for padded rank length. When dropping, truncate the shuffled/global sequence to a multiple of replicas. Otherwise cyclically repeat from the beginning until divisible. Select a rank by `indices.into_iter().skip(rank).step_by(replicas)`. Rebuild deterministic shuffle from `seed` and `epoch` without changing LibTorch RNG. `BatchSampler<S>` implements `BatchSource` when `S: Sampler`; `set_epoch` forwards to the sampler, `iter()` requests a fresh sampler iterator, and `exact_len()` returns `None` when the sampler is unsized. `FnBatchSource` stores the current epoch and optional exact length and passes the epoch to its closure for every fresh iterator. `OwnedDataLoader::set_epoch` forwards through `AutoBatch`, `ExplicitBatches`, and `NoBatch` index plans.

- [ ] **Step 5: Run evidence and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test batch_sampler --test distributed_sampler
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src/sampler.rs crates/rusttorch-data/tests crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): add batch and distributed samplers"
```

### Task 4: Add typed default collation

**Files:**
- Create: `crates/rusttorch-data/src/collate.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/collate.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: `Collate<Sample> { type Batch; type Error; fn collate(&mut self, Vec<Sample>) }`.
- Produces: `VecCollate`, `FnCollate<F>`, `DefaultCollator`, `DefaultCollate`, `DefaultConverter`, `DefaultConvert`, and a `Bytes(Vec<u8>)` newtype.
- Implements typed collation for tensors, Rust numeric primitives, strings/`Bytes`, options, equal-length vectors, tuples of arity two through eight, and `BTreeMap` values with identical key sets.
- Keeps existing closure entry points unchanged.

- [ ] **Step 1: Write structured collation tests**

Prove Tensor inputs stack on dimension zero, integer/float scalars produce tensors of documented kinds, strings remain `Vec<String>`, `Bytes` values remain byte records, tuple fields collate independently, equal-length vectors transpose/collate, mismatched vector lengths error, option presence must match, map key sets must match, and sparse/nested Tensor stacking returns the LibTorch error rather than panicking. Prove `DefaultConverter` leaves an already typed Tensor/scalar/string/`Bytes` sample unbatched and recursively converts options, vectors, tuples, and maps without the sequence transposition performed by collation.

Use the custom escape hatch:

```rust
let mut collate = FnCollate::new(|samples: Vec<Vec<i64>>| {
    Ok::<_, Infallible>(samples.into_iter().flatten().collect::<Vec<_>>())
});
assert_eq!(collate.collate(vec![vec![1, 2], vec![3]])?, vec![1, 2, 3]);
```

- [ ] **Step 2: Run the test and verify missing contracts**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test collate
```

Expected: compilation fails on `Collate`, `FnCollate`, and `DefaultCollator`.

- [ ] **Step 3: Implement typed recursion without a runtime registry**

Use associated batch/error types and private tuple macros. Tensor collation delegates to `Tensor::f_stack(&samples, 0)`. Numeric conversion uses `Tensor::from_slice`. Reserve `Vec<T>` for recursive equal-length sequence collation; `Bytes` is the coherence-safe Rust replacement for Python byte strings and avoids an overlapping `Vec<u8>` implementation. `DefaultConvert` is the typed no-auto-batching counterpart of PyTorch `default_convert`: it preserves Rust-native typed samples and recursively applies associated output conversions without adding a dynamic registry. Unequal structures return a non-exhaustive `CollateError`; do not add `Any`, a global mutable function map, NumPy handling, or implicit conversions between unrelated Rust types.

- [ ] **Step 4: Run tests, rustdoc, ledger generation, and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test collate
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch-data --no-deps --no-default-features --features doc-only
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests/collate.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): add typed default collation"
```

### Task 5: Add the owned serial builder and PyTorch argument semantics

**Files:**
- Create: `crates/rusttorch-data/src/error.rs`
- Create: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/loader_builder.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Preserves the existing generic borrowed `DataLoader` and its constructors.
- Adds default generic marker parameters so `DataLoader::builder(dataset)` resolves without changing explicit existing type annotations.
- Produces: `DataLoaderBuilder<D, P, C>`, `OwnedDataLoader<D, P, C>`, `AutoBatch<S>`, `ExplicitBatches<B>`, `NoBatch<S, V = DefaultConverter>`, and `LoaderError<E>`.
- Builder defaults: batch size 1, sequential sampling, workers 0, ordered output, no pinning, no timeout, no persistence.
- Effective prefetch is `None` with zero workers and `2` batches per worker when workers are positive unless the caller supplies another nonzero factor, matching pinned DataLoader defaults.

- [ ] **Step 1: Write builder-default and conflict tests**

Assert:

```rust
let mut loader = DataLoader::builder(rows).build()?;
let batches = loader.iter().collect::<Result<Vec<Tensor>, _>>()?;
assert_eq!(batches.len(), 3);
assert_eq!(batches[0].int64_value(&[0]), 0);
assert_eq!(batches[2].int64_value(&[0]), 2);

let mut loader = DataLoader::builder(rows)
    .batch_size(2)
    .drop_last(true)
    .build()?;
assert_eq!(loader.len(), Some(1));
```

Test explicit sampler, seeded shuffle, custom batch sampler, no-auto-batching mode with `DefaultConverter`, custom converter, explicit `VecCollate`, custom collator, and all PyTorch argument exclusions: sampler plus shuffle, batch sampler plus batch size/shuffle/sampler/drop-last, no batching plus drop-last, zero batch size, positive timeout with zero workers, prefetch with zero workers, zero prefetch factor, and persistent workers with zero workers. Assert the default `DefaultCollator` stacks Tensor samples and converts numeric scalars to Tensor batches, while explicit `VecCollate` preserves the former vector-batch behavior. Assert unsized samplers and batch sources iterate but make `OwnedDataLoader::len()` return `None`. Assert positive workers without an explicit factor normalize to two batches per worker and that an explicit factor overrides it. `Duration::ZERO` disables timeout; Rust's `Duration` makes a negative timeout unrepresentable.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test loader_builder
```

Expected: `DataLoader::builder` and `OwnedDataLoader` are absent.

- [ ] **Step 3: Implement the builder over the existing serial core**

Use a public doc-hidden dataset marker as the default `D` parameter on the existing `DataLoader` so this compiles:

```rust
/// Type-level marker used only to make `DataLoader::builder` inferable.
#[doc(hidden)]
pub struct BuilderDatasetMarker {
    private: (),
}

impl Dataset for BuilderDatasetMarker {
    type Sample = ();
    type Error = Infallible;

    fn len(&self) -> usize { 0 }

    fn get(&self, _index: usize) -> std::result::Result<(), Infallible> {
        unreachable!("the DataLoader builder marker never loads samples")
    }
}

pub struct DataLoader<
    'a,
    D = BuilderDatasetMarker,
    S = std::iter::Empty<usize>,
    C = (),
    B = (),
    E = Infallible,
>
where
    D: Dataset,
{
    batches: BatchIterator<DatasetSource<'a, D, S, E>, C, D::Sample, B, E>,
}

impl DataLoader<'static, BuilderDatasetMarker, std::iter::Empty<usize>, (), (), Infallible> {
    pub fn builder<D>(
        dataset: D,
    ) -> DataLoaderBuilder<D, AutoBatch<SequentialSampler>, DefaultCollator>
    where
        D: Dataset,
    {
        DataLoaderBuilder::new(dataset)
    }
}
```

Represent index plans explicitly:

```rust
pub struct AutoBatch<S> {
    sampler: S,
    batch_size: NonZeroUsize,
    drop_last: bool,
}

pub struct ExplicitBatches<B> { batches: B }
pub struct NoBatch<S, V = DefaultConverter> { sampler: S, converter: V }
```

Each plan creates a fresh iterator of `Vec<usize>` for the loader's current
epoch. `AutoBatch` wraps a reusable `Sampler` in `BatchSampler`,
`ExplicitBatches` stores a reusable `BatchSource`, and `NoBatch` yields
one-index groups passed through `DefaultConverter` rather than a collator.
`OwnedDataLoader::len() -> Option<usize>` performs batch/drop-tail arithmetic
only when the active sampler or batch source reports an exact length; iteration
never depends on length availability.
Expose typed builder transformations for `.sampler`,
`.shuffle(seed)`, `.batch_sampler`, `.without_batching`, and
`.collate`; `.convert` customizes only the no-auto-batching path. Add ordinary
validated setters for size, workers, persistence,
timeout, ordering, and drop policy. `.prefetch_factor` accepts ergonomic
`usize`, rejects zero, stores `NonZeroUsize` internally,
and remains invalid with zero workers; build normalizes an unset factor to
`None` for serial loading or `NonZeroUsize::new(2)` for positive workers.
`.sampler` requires `Sampler` and
`.batch_sampler` requires `BatchSource`; custom one-shot iterators continue to
work through the preserved borrowed APIs. `OwnedDataLoader::iter(&mut self)`
returns a named `LoaderIter<'_, D, P, C>` that creates fresh sampler/batch
iterators for the configured epoch; `set_epoch` changes that epoch explicitly.
The existing borrowed `DataLoader` remains an `Iterator` itself.

Define loader lifecycle errors without changing existing construction errors:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoaderError<E> {
    #[error("data pipeline failed (batch {batch:?}, worker {worker:?}): {source}")]
    Pipeline {
        batch: Option<u64>,
        worker: Option<usize>,
        #[source]
        source: E,
    },
    #[error("worker {worker} panicked while loading batch {batch:?}")]
    WorkerPanic { worker: usize, batch: Option<u64> },
    #[error("timed out waiting for batch {batch}")]
    Timeout { batch: u64 },
    #[error("loader was cancelled")]
    Cancelled,
    #[error("loader channel closed before batch {batch}")]
    ChannelClosed { batch: u64 },
    #[error("batch {batch} on worker {worker:?} requested {expected} samples but the dataset returned {actual}")]
    InvalidBatchCardinality {
        batch: u64,
        worker: Option<usize>,
        expected: usize,
        actual: usize,
    },
    #[error("stream protocol failed at sequence {sequence:?}: {reason}")]
    StreamProtocol { sequence: Option<u64>, reason: String },
    #[error("prefetch item exceeded the {limit} byte budget with {actual} bytes (batch {batch:?}, worker {worker:?}, sequence {sequence:?}, logical sample {logical_id:?})")]
    MemoryLimit {
        batch: Option<u64>,
        worker: Option<usize>,
        sequence: Option<u64>,
        logical_id: Option<u64>,
        limit: usize,
        actual: usize,
    },
    #[error("pinning batch {batch} failed: {source}")]
    PinMemory {
        batch: u64,
        #[source]
        source: rusttorch_core::RustTorchError,
    },
    #[error("checkpoint is incompatible or unavailable: {reason}")]
    Checkpoint { reason: String },
    #[error(transparent)]
    Configuration(#[from] rusttorch_core::RustTorchError),
}
```

Serial fetch/transform errors use `worker = None`; worker fetch/transform errors
carry `Some(id)` and `Some(batch)`; worker initialization errors carry
`Some(id)` and `batch = None`; coordinator collation errors carry
`worker = None`. Cardinality failures likewise identify the worker when the
batched fetch ran in a worker.

- [ ] **Step 4: Prove serial parity and source compatibility**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test loader_builder
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch --test data
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

Expected: builder behavior passes and all existing facade data tests remain unchanged.

- [ ] **Step 5: Commit**

```sh
git add crates/rusttorch-data/src crates/rusttorch-data/tests/loader_builder.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): add the owned loader builder"
```

### Task 6: Add deterministic task transforms and worker context

**Files:**
- Create: `crates/rusttorch-data/src/transform.rs`
- Create: `crates/rusttorch-data/src/worker_context.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/transform.rs`
- Create: `crates/rusttorch-data/tests/worker_context.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: `TaskContext`, `Transform<Input>`, `FnTransform<F>`, `TransformFactory<Input>`, `WorkerInit`, and `WorkerInfo`.
- Produces: `get_worker_info() -> Option<WorkerInfo>` through thread-local read-only context.
- Evolves the owned types to `DataLoaderBuilder<D, P, C, F = IdentityTransformFactory, I = NoWorkerInit>` and `OwnedDataLoader<D, P, C, F, I>` so transform/factory and initialization errors remain statically represented.
- Adds builder `.transform`, `.transform_factory`, `.worker_init`, `.seed`, `.epoch`, and `.rank`.

- [ ] **Step 1: Write schedule-independent RNG tests**

Construct contexts with the same seed/epoch/rank/sample/stage and different worker IDs. Assert their first 32 ChaCha12 outputs are identical. Change each logical identity field separately and assert the sequence changes. Assert the transform never changes a seeded LibTorch random sequence.

- [ ] **Step 2: Write worker-info lifecycle tests**

Assert `get_worker_info()` is `None` on the coordinator. In a simulated worker scope assert `id`, `num_workers`, and `seed`, then verify the thread-local value is cleared after scope exit even when the worker callback panics.

For one loader seed/rank/iterator generation, assert worker seeds are consecutive
by worker ID; changing rank or iterator generation changes the base. Lock
`(loader_seed=42, rank=1, iterator_generation=2, worker_id=3)` to
`0xdcae_5da8_9952_36e4` and assert LibTorch's global RNG remains untouched.

Use distinct error types for dataset fetch, transform, collation, transform construction, and worker initialization. Assert each arrives in the matching `PipelineError` variant without string conversion. Prove `.transform_factory` creates one transform per worker, while serial iteration creates one transform for the iterator and reuses it across that epoch.

- [ ] **Step 3: Implement stable seed derivation and transform contracts**

Expose:

```rust
pub struct TaskContext {
    pub loader_seed: u64,
    pub epoch: u64,
    pub rank: usize,
    pub logical_sample: u64,
    pub stage: u32,
}

pub trait Transform<Input> {
    type Output;
    type Error;
    fn transform(
        &mut self,
        input: Input,
        context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error>;
}

pub trait TransformFactory<Input>: Send + Sync {
    type Transform: Transform<Input>;
    type Error;

    fn create(
        &self,
        worker: Option<&WorkerInfo>,
    ) -> std::result::Result<Self::Transform, Self::Error>;
}

pub trait WorkerInit: Send + Sync {
    type Error;
    fn initialize(&self, worker: &WorkerInfo) -> std::result::Result<(), Self::Error>;
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError<DE, TE, CE, FE, IE> {
    #[error("dataset fetch failed: {0}")]
    Dataset(DE),
    #[error("transform failed: {0}")]
    Transform(TE),
    #[error("collation failed: {0}")]
    Collate(CE),
    #[error("transform initialization failed: {0}")]
    TransformInit(FE),
    #[error("worker initialization failed: {0}")]
    WorkerInit(IE),
}
```

`IdentityTransformFactory` and `NoWorkerInit` use `Infallible`. `.transform(t)`
wraps `t` in a clone-based factory; `.transform_factory(f)` accepts explicit
per-worker construction when cloning is not appropriate. A positive worker
count requires the factory and initializer to be `Send + Sync + 'static`, each
produced transform to be `Send + 'static`, and checkpointable worker transforms
to use either the `Stateless` or `TransactionalCheckpoint` adapter introduced in
Task 11. Stateful transforms are supported in serial and worker modes; Task 7's
deterministic task assignment gives every worker an unambiguous state history.

Use a documented fixed SplitMix64-style mixer over the five fields, then seed `ChaCha12Rng`; do not use randomized hash state. Worker ID is intentionally absent. `WorkerInfo` contains only stable Rust guarantees (`id`, `num_workers`, `seed`, `rank`). Unlike PyTorch, it does not expose an erased worker-local dataset because map workers share a typed `Arc<D>` and streaming factories receive their source explicitly. Record this, worker-side collation, and process-local dataset copies as three precise Rust-native compatibility differences rather than parity claims.

Use derivation version 1 exactly:

```rust
pub const TASK_RNG_DERIVATION_VERSION: u32 = 1;
pub const WORKER_SEED_DERIVATION_VERSION: u32 = 1;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_task_seed(context: &TaskContext) -> u64 {
    let fields = [
        context.loader_seed,
        context.epoch,
        context.rank as u64,
        context.logical_sample,
        u64::from(context.stage),
    ];
    fields.into_iter().enumerate().fold(
        0x5255_5354_544f_5243,
        |state, (index, value)| {
            let domain = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            splitmix64(state ^ value.wrapping_add(domain))
        },
    )
}

fn derive_worker_seed(
    loader_seed: u64,
    rank: usize,
    iterator_generation: u64,
    worker_id: usize,
) -> u64 {
    let rank_domain = (rank as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    let iterator_domain = iterator_generation.wrapping_mul(0xa076_1d64_78bd_642f);
    splitmix64(loader_seed ^ rank_domain ^ iterator_domain)
        .wrapping_add(worker_id as u64)
}
```

Lock the test vector `(42, 3, 1, 99, 7)` to seed
`0x1d7d_73dc_f6e9_4f2d` so checkpoint compatibility cannot drift silently.
`OwnedDataLoader` increments `iterator_generation` whenever it creates a new
non-persistent worker pool, matching PyTorch's observable new-base-seed-per-
iterator behavior. A persistent pool retains the seeds from its initial
generation; epoch remains part of `TaskContext` instead. The mixer is a
documented Rust sequence and does not claim PyTorch generator/Philox equality.

- [ ] **Step 4: Run tests, update evidence, and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test transform --test worker_context
cargo test -p rusttorch --test data_libtorch
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): add deterministic task transforms"
```

### Task 7: Add bounded ordered map workers

**Files:**
- Create: `crates/rusttorch-data/src/worker.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Modify: `Cargo.toml`
- Modify: `crates/rusttorch-data/Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `THIRD_PARTY_NOTICES.md`
- Create: `crates/rusttorch-data/tests/map_workers.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Adds `crossbeam-channel` as the bounded transport.
- Adds positive `.workers(count)` and `.prefetch_factor(batches_per_worker)` execution.
- Yields sampler order by default; `.in_order(false)` yields completion order.
- Fetches map batches through `Dataset::get_batch` and validates result cardinality.

- [ ] **Step 1: Write real parallel/order/backpressure tests**

Use a dataset with a barrier and per-index delays to prove at least two worker thread IDs fetch concurrently. Reverse completion timing and assert ordered mode still yields sampler order. Assert unordered mode yields the controlled completion order. Prove batch sequence `s` is always assigned to worker `s % workers` across repeated runs. With no explicit prefetch factor, prove total outstanding work never exceeds `workers * 2`; then override the factor and prove neither per-worker task queues nor total result/reassembly state exceeds the configured bounds.

Add source-error-once, transform-error-once, wrong batched-fetch cardinality, and worker-panic tests with exact batch/worker context.

- [ ] **Step 2: Run the worker test and verify the serial implementation fails**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test map_workers
```

Expected: positive worker construction or concurrency assertions fail.

- [ ] **Step 3: Implement bounded tasks, results, and ordered merge**

Use this private completed-work envelope:

```rust
struct WorkerBatch<T> {
    generation: u64,
    batch_sequence: u64,
    samples: Vec<T>,
}
```

Define private messages carrying monotonic loader `generation`, `batch_sequence`, logical sample IDs, and index vectors. Create one task channel per worker with capacity `prefetch_factor` and assign sequence `s` to worker `s % workers`; create one result channel with capacity `workers * prefetch_factor`. This deterministic routing is required for reproducible stateful transforms and exact worker checkpoints. Workers share `Arc<D>`, enter a `WorkerInfo` scope, initialize their own transform, fetch/transform, and send one terminal `Result<WorkerBatch<T>, LoaderError<PipelineError<DE, TE, CE, FE, IE>>>`. The coordinator accepts only its active generation, collates completed samples, and uses a `BTreeMap` only for out-of-order completions; the result channel plus reassembly map never hold more than `workers * prefetch_factor` batches. Document shared map datasets and coordinator-side collation as Rust-native thread-model differences from PyTorch multiprocessing, while preserving ordering, deterministic randomness, backpressure, and error behavior.

Catch Rust panics at the worker boundary with `catch_unwind`, convert them to `LoaderError::WorkerPanic`, close submission after the first error, yield that error once, and join workers during iterator teardown.

- [ ] **Step 4: Run worker, serial, Clippy, and evidence checks**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test map_workers
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test loader_builder
cargo clippy -p rusttorch-data --all-targets --no-default-features --features doc-only -- -D warnings
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
```

- [ ] **Step 5: Commit**

```sh
git add crates/rusttorch-data Cargo.toml Cargo.lock THIRD_PARTY_NOTICES.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): load map datasets with bounded workers"
```

### Task 8: Complete worker timeout, cancellation, and persistence

**Files:**
- Modify: `crates/rusttorch-data/src/worker.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/dataset.rs`
- Modify: `crates/rusttorch-data/src/transform.rs`
- Modify: `crates/rusttorch-data/src/worker_context.rs`
- Create: `crates/rusttorch-data/tests/worker_lifecycle.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: cooperative `CancellationToken`, `Deadline`, `WaitOutcome`, `LoaderCancelled`, and `WorkerContext { info, cancellation, deadline }`.
- Adds context-aware dataset fetch and evolves transform/factory/initializer contexts so cooperating user code can observe cancellation and timeout without breaking legacy dataset methods.
- Implements `.timeout(Duration)`, `.persistent_workers(true)`, early-drop join, and epoch reset.
- Documents that arbitrary blocking/native calls cannot be force-cancelled.

- [ ] **Step 1: Write lifecycle tests with event synchronization**

Use channels/barriers rather than sleeps to test: dropping with workers blocked on a full result queue; dropping while workers wait for tasks; first-error cancellation; timeout yielded once; no result after timeout; worker-init error/panic; and every joinable thread incrementing an exit counter exactly once. Add a context-aware dataset and transform that block until their `CancellationToken` changes, then prove early drop and deadline wake them without external release. With persistent workers, consume one batch, drop the iterator while later batches are in flight, begin the next generation, and prove stale tagged results are discarded, sequence zero is not confused across generations, a fresh per-generation token is live, and the same worker threads safely serve the next epoch.

Run a serial transform factory and assert it receives `None`, while worker
factories receive `Some(&WorkerContext)` and worker initialization always
receives `&WorkerContext`.

For non-persistent workers, iterate twice and assert worker base seeds change while remaining consecutive by ID. For persistence, iterate two epochs through `&mut OwnedDataLoader`, assert the same worker thread IDs and initial seeds are reused, worker init runs once per worker, epoch-visible task context changes, and dropping the owner joins the persistent pool.

- [ ] **Step 2: Run the test and verify missing lifecycle behavior**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test worker_lifecycle
```

Expected: lifecycle assertions fail or APIs are absent.

- [ ] **Step 3: Implement cooperative shutdown**

Store cancellation in one shared state:

```rust
struct CancellationState {
    cancelled: AtomicBool,
    wait_lock: Mutex<()>,
    wake: Condvar,
}
```

Expose the execution context and source-compatible fetch hook:

```rust
#[derive(Clone)]
pub struct WorkerContext {
    pub info: WorkerInfo,
    pub cancellation: CancellationToken,
    pub deadline: Deadline,
}

impl CancellationToken {
    pub fn is_cancelled(&self) -> bool;
    pub fn wait_cancelled(&self);
    pub fn wait_cancelled_timeout(&self, timeout: Duration) -> bool;
}

impl Deadline {
    pub fn is_expired(&self) -> bool;
    pub fn remaining(&self) -> Option<Duration>;
}

impl WorkerContext {
    pub fn check(&self) -> std::result::Result<(), LoaderCancelled>;
    pub fn wait_cancelled_or_deadline(&self) -> WaitOutcome;
}

fn get_batch_with_context(
    &self,
    indices: &[usize],
    context: &WorkerContext,
) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
    let _ = context;
    self.get_batch(indices)
}
```

Task 8 evolves `TaskContext` to expose the same token/deadline and evolves
`TransformFactory::create` to receive `Option<&WorkerContext>` (`None` on the
serial coordinator) plus `WorkerInit::initialize` to receive `&WorkerContext`.
Legacy `Dataset::get_batch` remains
the default implementation behind `get_batch_with_context`, so existing
datasets compile unchanged. Use crossbeam select/timeout for queue waits.
Built-in worker loops check cancellation before and after every bounded
fetch/transform/send operation. On drop: stop submission, set cancellation,
disconnect channels, notify the condition variable, and join every worker that
returns. A documented test fixture proves drop waits for a deliberately
non-cooperative native call until that call is released.

`wait_cancelled` uses the shared condition variable and does not poll.
`wait_cancelled_or_deadline` returns `WaitOutcome::Cancelled` or
`WaitOutcome::DeadlineExpired`; an absent deadline waits only for cancellation.
All public methods receive rustdoc and have event-driven tests, making the
context usable by custom datasets, transforms, decoders, and source factories.

Persistent pools have a pool-lifetime shutdown token plus a distinct
per-generation cancellation token. Every task, result, error, and control/ack
message carries the monotonic generation. Dropping an iterator stops submission,
cancels that generation, waits for one quiesced acknowledgement per worker, and
drains/discards all results for that generation before installing a fresh token
and allowing the next iterator to submit. Pool shutdown then cancels the active
generation and joins all threads. Persistent pools are owned by
`OwnedDataLoader`, never by a detached iterator.

- [ ] **Step 4: Run lifecycle, stress, evidence, and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test worker_lifecycle
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test map_workers
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests/worker_lifecycle.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): make loader workers lifecycle-safe"
```

### Task 9: Add explicitly sharded streaming workers

**Files:**
- Create: `crates/rusttorch-data/src/stream.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/stream_workers.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Preserves ordinary `batches`/`batches_with_collate` for zero-worker streams.
- Produces: `WorkerSourceFactory`, `WorkerRecord<T>`, `LogicalSampleId`, `SequenceId`, and `StreamDataLoaderBuilder`.
- Produces optional exact global-length reporting and `StreamDataLoader::len() -> Option<usize>` with batch/drop-tail arithmetic.
- Requires globally unique, zero-based contiguous sequence IDs for ordered threaded streams; permits unsequenced records only in explicit unordered mode.

- [ ] **Step 1: Write shard/order/drop-last tests**

Create a factory that returns disjoint modulo shards. Assert each logical record appears exactly once across four workers. Ordered mode requires `SequenceId` values to form the contiguous epoch range `0..N`; delay lower IDs and prove the coordinator waits for the exact next value. Assert duplicate IDs, an internal/final gap, a first ID other than zero, and ordered sources without IDs are errors. Prove unordered unsequenced mode works when every record still supplies a stable `LogicalSampleId` for deterministic transforms. For factories with and without an exact global length, verify `len()` returns the correct `Some(ceil/floor batches)` or `None`. Verify the coordinator merges ordered records before batching, so `drop_last` applies once to the global logical stream. Record PyTorch's per-process iterable-tail behavior as an intentional Rust difference that avoids silently dropping several shard tails.

- [ ] **Step 2: Run tests and verify missing stream worker types**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test stream_workers
```

Expected: compilation fails on `WorkerSourceFactory` and `SequenceId`.

- [ ] **Step 3: Implement factory-owned shard iterators**

Expose:

```rust
pub trait WorkerSourceFactory: Send + Sync + 'static {
    type Sample: Send + 'static;
    type Error: Send + 'static;
    type Source: Iterator<
            Item = std::result::Result<WorkerRecord<Self::Sample>, Self::Error>,
        > + Send
        + 'static;

    fn create(
        &self,
        worker: WorkerContext,
    ) -> std::result::Result<Self::Source, Self::Error>;

    fn exact_len(&self) -> Option<usize> { None }
}

pub struct WorkerRecord<T> {
    pub sequence: Option<SequenceId>,
    pub logical_id: LogicalSampleId,
    pub sample: T,
}
```

`SequenceId` is a checked `u64` newtype. In ordered mode, every epoch starts at
zero and IDs are contiguous; the coordinator publishes only its `next_sequence`
and reports a protocol gap after all workers end if a larger buffered ID remains.
`LogicalSampleId` is always required and is the RNG/checkpoint identity even for
unordered records without a sequence. Each worker owns its iterator and receives
cancellation/deadline through `WorkerContext`; sources may cooperate, while
arbitrary blocking decoder/native calls retain the documented no-force-cancel
limitation. Reuse the bounded result/lifecycle engine without putting a source
behind a lock.

- [ ] **Step 4: Run stream/worker/evidence checks and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test stream_workers --test worker_lifecycle
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests/stream_workers.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): load explicitly sharded streams"
```

### Task 10: Enforce byte budgets and recursive pinning

**Files:**
- Create: `crates/rusttorch-data/src/memory.rs`
- Create: `crates/rusttorch-data/src/pin_memory.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/worker.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Create: `crates/rusttorch-data/tests/memory_budget.rs`
- Create: `crates/rusttorch-data/tests/pin_memory.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: `MemoryFootprint::resident_bytes(&self) -> usize`.
- Adds builder `.prefetch_bytes(NonZeroUsize)`; item capacity remains mandatory.
- Produces: same-shape `PinMemory: Sized { fn pin_memory(self, device: Device) -> Result<Self> }` for Tensor, Vec, Option, tuples, `BTreeMap`, domain batches, and identity scalar/string/`Bytes` leaves.
- Adds type-state builder `.pin_memory()` and `.pin_memory_for(device)` enable methods only when the collated batch implements `PinMemory`; the default disabled state has no such bound.

- [ ] **Step 1: Write byte-budget tests**

Use records with controlled byte sizes and assert outstanding permits never exceed the configured budget, oversized single records fail before publication, permits are released after consumption/error/drop, ordered reassembly shares the same budget, and a custom type without `MemoryFootprint` remains item-bounded but is not reported as byte-bounded. Cover an unordered, unsequenced streaming record and assert its `MemoryLimit` carries `worker` plus `logical_id`, while `batch` and `sequence` remain `None`.

- [ ] **Step 2: Write pinning tests**

On a compatible CUDA runner, assert Tensor and nested supported containers return `f_is_pinned(device) == true`. On CPU-only environments, assert automatic `.pin_memory()` degrades to a documented disabled/no-op status and yields the same `Batch` type unpinned, matching pinned DataLoader behavior rather than failing iteration. An explicit unsupported `.pin_memory_for(device)` remains a configuration error. Prove a custom batch without `PinMemory` still builds in the default disabled state, while enabling pinning fails at compile time. Prove pinning occurs after collation and before the batch is observable. Test `Vec`, `Option`, tuple, and `BTreeMap` recursion; scalar, `String`, and `Bytes` leaves remain unchanged, so every built-in `DefaultCollator` output type-checks with pinning enabled.

- [ ] **Step 3: Implement permits and LibTorch delegation**

Implement the byte budget with a standard-library `Mutex<usize>`/`Condvar` permit object that releases on `Drop` and wakes on cancellation. Tensor footprint uses checked `numel * element_size`; aggregate containers use checked sums. `Tensor` pinning delegates to `Tensor::f_pin_memory(device)` and preserves `TchError` as the source. `BTreeMap<K, V>` preserves keys and recursively pins values; non-Tensor leaves return themselves without cloning. The builder starts in `PinDisabled`, which never requires `Batch: PinMemory`. `.pin_memory()` transitions to `PinEnabled<Auto>` and uses a supported current accelerator when available, otherwise recording `PinMemoryStatus::DisabledNoAccelerator` and becoming a no-op. `.pin_memory_for(Device)` transitions to `PinEnabled<Explicit>` and errors when that device cannot back pinned host memory. This method split is the Rust type-safe replacement for Python's runtime boolean.

- [ ] **Step 4: Run local/conditional tests, evidence, and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test memory_budget
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test pin_memory
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): bound loader memory and pin batches"
```

### Task 11: Add versioned sampler/component checkpoints and exact serial resume

**Files:**
- Create: `crates/rusttorch-data/src/checkpoint.rs`
- Modify: `crates/rusttorch-data/src/dataset.rs`
- Modify: `crates/rusttorch-data/src/sampler.rs`
- Modify: `crates/rusttorch-data/src/collate.rs`
- Modify: `crates/rusttorch-data/src/transform.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Modify: `crates/rusttorch-data/src/error.rs`
- Modify: `crates/rusttorch-data/src/lib.rs`
- Modify: `crates/rusttorch-data/Cargo.toml`
- Modify: `Cargo.lock`
- Create: `crates/rusttorch-data/tests/checkpoint_map.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Adds workspace `serde` derive support without choosing a storage format and member-test-only `serde_json` for round-trip evidence.
- Produces: generic `LoaderState<DatasetState, SamplerState, TransformState = (), CollateState = ()>`.
- Produces: `Checkpointable`, `ReplaySafeDataset`, `DatasetCheckpoint`, `Stateless`, `TransactionalCheckpoint`, and `WorkerCheckpoint` contracts plus explicit dataset/worker-transform checkpoint adapters and `ReplaySafeTensorDataset`.
- Adds `.dataset_identity(String)`, `.resume_from(state)`, and `LoaderIter::checkpoint(&mut self)` on the active iterator that owns the exact next-visible-batch boundary.

- [ ] **Step 1: Write serial next-visible-batch tests**

Create `let mut iteration = loader.iter()`, consume two batches, call `iteration.checkpoint()`, serialize to JSON in the test, drop the iterator, reconstruct the same loader, resume, and assert the remaining batches exactly match uninterrupted execution with no duplicate. Cover short tails, drop-last, random/replacement/weighted/distributed sampler positions, epoch, task/worker RNG derivation versions, a replay-safe deterministic dataset, a transactional serial dataset, stateful serial transforms, stateful coordinator collation, and stateless transforms.

Reject wrong schema version, dataset identity, batch settings, sampler kind, world size, rank, distributed shuffle/seed/padding policy, corrupt state, unordered mode, and opaque stateful closures. Add compile-fail tests proving an arbitrary unmarked `Dataset` and ordinary storage-sharing `TensorDataset` cannot expose exact checkpointing. A custom deterministic dataset opts into `ReplaySafeDataset`; a stateful/nondeterministic dataset must use the transactional dataset adapter and zero workers. Use a component whose `load_validated` increments a counter and prove a different invalid component prevents every apply call, establishing validate-before-mutate behavior. Construct `ReplaySafeTensorDataset` from aliased input tensors, mutate both the original alias and repeatedly fetched replacement-sampled rows, and prove its private backing storage, later fetches, uninterrupted output, and resumed output remain identical.

- [ ] **Step 2: Run the checkpoint test and verify missing APIs**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_map
```

Expected: compilation fails on `LoaderState`, `LoaderIter::checkpoint`, and `resume_from`.

- [ ] **Step 3: Implement typed versioned state**

Add `serde = { version = "1.0", features = ["derive"] }` and
`serde_json = "1.0"` to `[workspace.dependencies]`. Use `serde.workspace = true`
in `rusttorch-data` dependencies and `serde_json.workspace = true` in that
member's dev-dependencies; production code never depends on a storage format.

Expose the stable envelope:

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LoaderConfiguration {
    pub batch_size: Option<usize>,
    pub drop_last: bool,
    pub workers: usize,
    pub prefetch_factor: Option<usize>,
    pub in_order: bool,
    pub rank: usize,
    pub replicas: usize,
    pub distributed_shuffle: Option<bool>,
    pub distributed_seed: Option<u64>,
    pub distributed_drop_last: Option<bool>,
    pub sampler_kind: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LoaderState<D, S, T = (), C = ()> {
    pub schema_version: u32,
    pub dataset_identity: String,
    pub epoch: u64,
    pub iterator_generation: u64,
    pub next_batch: u64,
    pub dataset: D,
    pub sampler: S,
    pub transform: T,
    pub collate: C,
    pub rng_derivation_version: u32,
    pub worker_seed_derivation_version: u32,
    pub configuration: LoaderConfiguration,
}

pub trait Checkpointable {
    type State: Clone + serde::Serialize + serde::de::DeserializeOwned;
    fn save_state(&self) -> Self::State;
    fn validate_state(&self, state: &Self::State) -> rusttorch_core::Result<()>;
    fn load_validated(&mut self, state: &Self::State);
}

pub trait ReplaySafeDataset: Dataset {}

pub trait DatasetCheckpoint: Dataset {
    type State: Clone + serde::Serialize + serde::de::DeserializeOwned;
    fn snapshot_dataset(&self) -> Self::State;
    fn validate_dataset_state(&self, state: &Self::State)
        -> rusttorch_core::Result<()>;
    fn restore_dataset_validated(&mut self, state: &Self::State);
}

pub trait Stateless {}

pub trait TransactionalCheckpoint {
    type State: Clone + serde::Serialize + serde::de::DeserializeOwned;
    fn snapshot(&self) -> Self::State;
    fn validate_snapshot(&self, state: &Self::State) -> rusttorch_core::Result<()>;
    fn restore_validated(&mut self, state: &Self::State);
}

pub trait WorkerCheckpoint {
    type State: Clone + serde::Serialize + serde::de::DeserializeOwned;
    fn snapshot(&self) -> Self::State;
    fn validate_snapshot(&self, state: &Self::State) -> rusttorch_core::Result<()>;
    fn restore_validated(&mut self, state: &Self::State);
}
```

`OwnedDataLoader::iter(&mut self)` holds the mutable owner borrow, so checkpointing is deliberately a method on that named active iterator rather than on the owner. Validation methods are documented as read-only contracts. Resume validates every envelope/configuration/component state before calling any apply method; `load_validated`/`restore_validated` are deliberately infallible after successful validation. This two-phase contract prevents partial live mutation on any recoverable error without requiring every stateful collator or native component to be cloneable. Tests instrument all apply calls and include validation failure in the final component. Serial transform and coordinator collation state are saved immediately after the last visible batch.

Provide `StatelessWorker<T>` and `TransactionalWorker<T>` wrappers that forward
`Transform` and implement `WorkerCheckpoint`: the stateless wrapper stores `()`,
and the transactional wrapper delegates to `TransactionalCheckpoint`. Builder
methods `.checkpoint_stateless()` and `.checkpoint_transactional()` select the
wrapper explicitly, so Rust never needs specialization or an ambiguous
“`Stateless` or transactional” trait bound.

Provide `ReplaySafeMap<D>` and `TransactionalMap<D>` wrappers that forward
`Dataset` and implement `DatasetCheckpoint`: the replay-safe wrapper requires
`D: ReplaySafeDataset` and stores `()`, while the transactional wrapper delegates
to `Checkpointable`. Ordinary `TensorDataset` deliberately does not implement
`ReplaySafeDataset` because its PyTorch-compatible row views share storage.
`TensorDataset::into_replay_safe()` performs one explicit deep copy of every
backing tensor and returns `ReplaySafeTensorDataset`; that wrapper deep-copies
every fetched row through the fallible LibTorch clone operation and implements
`ReplaySafeDataset`. The other built-in immutable dataset adapters implement it
when their children do. Checkpoint-capable builder
type state requires one of these wrappers, so `.dataset_identity(...)` alone is
never evidence that fetches can be replayed. Positive-worker exact checkpointing
requires `ReplaySafeMap`; `TransactionalMap` is serial-only because concurrent
stateful fetch ordering is not a stable replay contract.

- [ ] **Step 4: Run checkpoint/evidence checks and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_map
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add Cargo.toml crates/rusttorch-data Cargo.lock compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): resume serial loaders exactly"
```

### Task 12: Make prefetched map checkpoints replay-safe

**Files:**
- Modify: `crates/rusttorch-data/src/checkpoint.rs`
- Modify: `crates/rusttorch-data/src/worker.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Extend: `crates/rusttorch-data/tests/checkpoint_map.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Adds a checkpoint barrier at a yielded-batch boundary.
- Replays deterministic map fetch/transform work after resume rather than serializing arbitrary samples.
- Supports stateless and transactional stateful positive-worker transforms through `WorkerCheckpoint`, using deterministic round-robin task assignment and per-worker boundary snapshots.
- Requires `ReplaySafeMap<D>` for positive-worker replay; transactional/nondeterministic datasets remain a serial checkpoint capability.

- [ ] **Step 1: Add in-flight and rollback tests**

At every batch boundary of a controlled delayed dataset, call checkpoint on the active iterator for worker counts 1, 2, and 4 and prefetch factors 1 and 3. Resume and compare with uninterrupted output. Prove prefetched samples are never yielded twice, stateless deterministic transforms replay identically, transactional stateful transforms restore each worker's snapshot preceding its first unconsumed assigned task, and errors after the boundary recur at the same logical point.

Assert an opaque stateful transform without an explicit checkpoint adapter cannot expose `checkpoint()` at compile time. Prove both explicit adapters work with positive workers, and prove worker-count/configuration drift is rejected before restore.

- [ ] **Step 2: Run the extended test and verify advanced-state failure**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_map prefetched
```

Expected: resumed output or transform state diverges until the barrier exists.

- [ ] **Step 3: Implement the committed-boundary barrier**

Before each worker executes an assigned task, retain its transform snapshot keyed by batch sequence in a ring sized to `prefetch_factor + 1`. On checkpoint request, stop new submissions, receive/account for every dispatched task, and identify the next visible sequence. For each deterministic worker lane, restore the snapshot immediately preceding that lane's first assigned unconsumed sequence; discard unpublished results; rewind the checkpointable sampler cursor; and save the per-worker transform states plus coordinator state at the same boundary. Resume recreates the same lanes and replays from that boundary. Do not serialize decoded samples. Reject custom samplers or transforms that cannot restore the required logical boundary before workers start.

- [ ] **Step 4: Run map worker/checkpoint stress, evidence, and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_map --test map_workers --test worker_lifecycle
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests/checkpoint_map.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): checkpoint prefetched map loaders"
```

### Task 13: Add transactional streaming checkpoint/resume

**Files:**
- Modify: `crates/rusttorch-data/src/checkpoint.rs`
- Modify: `crates/rusttorch-data/src/stream.rs`
- Modify: `crates/rusttorch-data/src/loader.rs`
- Create: `crates/rusttorch-data/tests/checkpoint_stream.rs`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces: `CheckpointableSource` with typed state and boundary snapshot/restore.
- Stores one source state per worker shard plus the next global `SequenceId`.
- Rejects exact resume for ordinary iterators, unsequenced streams, unordered delivery, or factories that cannot retain the configured prefetch-window boundary states.

- [ ] **Step 1: Write stream state/resume tests**

Use a finite source state containing offset and decoder counter. Checkpoint at every batch boundary with one and three workers; serialize, recreate, and resume. Assert exact global ordering, no duplicates/gaps, restored per-shard offsets, and deterministic decode/transform counters. Checkpoint with in-flight prefetched records and prove each worker restores the snapshot preceding its first unconsumed record.

Add explicit rejection tests for ordinary `batches`, missing sequence IDs, duplicate sequence IDs, unordered mode, wrong worker count, and insufficient snapshot-window retention.

- [ ] **Step 2: Run tests and verify missing source checkpoint contract**

Run:

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_stream
```

Expected: compilation fails on `CheckpointableSource` or resume diverges.

- [ ] **Step 3: Implement typed per-worker boundary state**

Expose:

```rust
pub trait CheckpointableSource:
    Iterator<
        Item = std::result::Result<WorkerRecord<Self::Sample>, Self::Error>,
    >
{
    type Sample;
    type State: Clone + serde::Serialize + serde::de::DeserializeOwned;
    type Error;

    fn snapshot(&self) -> Self::State;
    fn restore(
        &mut self,
        state: &Self::State,
    ) -> std::result::Result<(), Self::Error>;
}
```

The stream worker retains snapshots for at least its configured outstanding record window. The barrier chooses the snapshot before the first unconsumed global record for each shard and validates worker/factory identity before restore.

- [ ] **Step 4: Run checkpoint/evidence checks and commit**

```sh
LIBTORCH_USE_PYTORCH=1 cargo test -p rusttorch-data --test checkpoint_stream --test stream_workers
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
git add crates/rusttorch-data/src crates/rusttorch-data/tests/checkpoint_stream.rs crates/rusttorch-data/COMPATIBILITY.md compat/pytorch_api.toml docs/api-coverage.md
git commit -s -m "feat(data): checkpoint sharded streams exactly"
```

### Task 14: Finish compatibility evidence, documentation, benchmarks, and CI

**Files:**
- Create: `crates/rusttorch-data/examples/loader.rs`
- Create: `crates/rusttorch-data/benches/data_loader.rs`
- Remove after replacement: `benches/data_loader.rs`
- Modify: `Cargo.toml`
- Modify: `crates/rusttorch-data/Cargo.toml`
- Modify: `README.md`
- Modify: `crates/rusttorch-data/README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/pytorch-compatibility.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `compat/pytorch_api.toml`
- Regenerate: `docs/api-coverage.md`
- Regenerate: `crates/rusttorch-data/COMPATIBILITY.md`

**Interfaces:**
- Produces one end-to-end map example and one sharded stream example.
- Produces reproducible benchmarks for serial overhead, worker scaling, ordered/unordered completion, prefetch bounds, transform, collation, pinning, and checkpoint barriers.
- Gives every classic `torch.utils.data` public symbol an explicit ledger row and status.

- [ ] **Step 1: Close the pinned classic-data inventory**

Audit every public symbol exported by pinned `torch.utils.data.__all__` and the pinned dataset/sampler/distributed/DataLoader pages. Ledger rows must cover Dataset, IterableDataset's Rust iterator/factory equivalent, all dataset adapters, random split, Sampler, all sampler classes, DataLoader arguments/length, default collate/convert, worker info, pinning, and distributed sampling.

Mark Python multiprocessing context/pickling, mutable dynamic collate registry, deprecated `pin_memory_device`, DataPipe decorators, and DataPipes with their precise Rust replacement or separately scoped status. Do not claim DataLoader2 because it is absent from the pinned core release.

Keep checkpoint/resume in a RustTorch-extension row with no invented PyTorch
symbol mapping; pinned classic DataLoader 2.13 has no public checkpoint API.

- [ ] **Step 2: Add compiling ergonomic examples and support tables**

The primary example must fit this shape:

```rust
let mut loader = DataLoader::builder(dataset)
    .shuffle(42)?
    .batch_size(64)
    .workers(4)
    .prefetch_factor(2)
    .collate(DefaultCollator)
    .pin_memory()
    .build()?;

for batch in loader.iter() {
    let batch = batch?;
    assert_eq!(batch.size()[0], 64);
}
```

Document the borrowed debugging path, owned threaded path, stream factory, distributed epoch update, resource limits, cancellation limitation, and checkpoint capability matrix.

- [ ] **Step 3: Replace the benchmark and add CI lanes**

Move the benchmark target to `rusttorch-data`. Report dataset size, hardware, worker count, queue capacity, byte budget, batch size, ordering, prefetch factor, transform/collate configuration, and comparison baseline. Do not assert machine-dependent timing thresholds.

Add stable and MSRV checks for `rusttorch-data`, no-default/doc-only rustdoc, feature combinations, package inspection, platform tests on Linux/macOS/Windows, stress tests, and conditional CUDA pinning. Ordinary CI uses committed tiny data only and no network fetch beyond locked build dependencies.

- [ ] **Step 4: Run the complete workspace gate**

Run with the UV-managed environment and platform LibTorch library path:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python scripts/check-compatibility.py --write
.venv/bin/python scripts/check-compatibility.py --check
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
scripts/run-python-parity.sh
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo bench -p rusttorch-data --bench data_loader
cargo package -p rusttorch-data --locked --list
```

Expected: all commands pass, no test leaks a thread, documentation has no missing public items, package contents are clean, and the compatibility page contains no unsupported blanket percentage claim.

- [ ] **Step 5: Commit**

```sh
git add Cargo.toml Cargo.lock crates/rusttorch-data benches README.md docs THIRD_PARTY_NOTICES.md .github/workflows/ci.yml compat/pytorch_api.toml
git commit -s -m "docs(data): complete the DataLoader compatibility milestone"
```
