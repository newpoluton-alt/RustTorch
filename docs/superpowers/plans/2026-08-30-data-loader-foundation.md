# DataLoader Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, fallible, zero-copy-by-default map and streaming DataLoader foundation with sequential and seeded random sampling.

**Architecture:** Map datasets expose owned samples through a small `Dataset` trait. Samplers are ordinary iterators, and one monomorphized batching core consumes either dataset samples or an existing fallible stream and moves each pre-sized sample vector directly into a user collation closure.

**Tech Stack:** Rust iterators/generics, `rand` 0.8, `rand_chacha` 0.3, LibTorch tensors only in examples/tests.

**Spec:** `docs/superpowers/specs/2026-08-30-pytorch-compatibility-program.md`

## Global Constraints

- Product credo: “easy to use and easy to implement, but crazy fast.”
- Phase one is deterministic and single-threaded.
- No trait objects, boxing, `Arc`, channels, locks, process abstraction, or implicit tensor copy.
- Sequential sampling allocates nothing; random sampling allocates one index vector.
- A batch allocates one pre-sized `Vec<Sample>` and moves it into collation.
- Sampling randomness is local and does not mutate LibTorch's global RNG.
- Source or collation failure terminates the loader after yielding the error once.
- `batch_size == 0` is a structured `RustTorchError::InvalidConfiguration`.
- `DataLoader` is the map-dataset-plus-sampler adapter. `batches` is the
  ordinary fallible-stream adapter. Both are public DataLoader surfaces and
  share one private monomorphized batching core.
- `new` and `batches` use identity `Vec<Sample>` collation;
  `with_collate` and `batches_with_collate` accept custom `FnMut(Vec<Sample>)`
  collation.
- Constructors use `crate::Result` only for configuration errors. Iterator,
  source, dataset, and collation errors use `std::result::Result<_, E>`.

---

### Task 1: Add map datasets and deterministic samplers

**Files:**
- Create: `src/data.rs`
- Modify: `src/lib.rs`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Create: `tests/data.rs`

**Interfaces:**
- Produces: `data::Dataset` with associated `Sample` and `Error`, `len`, `is_empty`, and `get`.
- Produces: `Dataset::samples() -> DatasetSamples<'_, Self>`, a borrowing,
  sequential fallible iterator over `get` with no dataset clone or allocation.
- Produces: `SequentialSampler::new(length) -> SequentialSampler`.
- Produces: `RandomSampler::new(length, seed) -> rusttorch::Result<RandomSampler>`.

- [ ] **Step 1: Write sampler and dataset tests**

Define a small dataset and assert:

```rust
assert_eq!(SequentialSampler::new(4).collect::<Vec<_>>(), [0, 1, 2, 3]);

let first = RandomSampler::new(8, 42).unwrap().collect::<Vec<_>>();
let second = RandomSampler::new(8, 42).unwrap().collect::<Vec<_>>();
assert_eq!(first, second);
let mut sorted = first.clone();
sorted.sort_unstable();
assert_eq!(sorted, (0..8).collect::<Vec<_>>());
assert!(RandomSampler::new(0, 42).is_err());
```

Also assert the default `is_empty` follows `len`, `samples` yields every item
in order without cloning the dataset, and constructing/consuming a seeded
random sampler does not change LibTorch's seeded global random sequence.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```sh
cargo test -p rusttorch --test data sampler --features tch/doc-only
```

Expected: FAIL because `rusttorch::data` does not exist.

- [ ] **Step 3: Implement the minimal API**

Use these public signatures:

```rust
pub trait Dataset {
    type Sample;
    type Error;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error>;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn samples(&self) -> DatasetSamples<'_, Self>
    where
        Self: Sized;
}

pub struct SequentialSampler { /* Range<usize> */ }
pub struct RandomSampler { /* vec::IntoIter<usize> */ }
```

Declare `rand = "0.8"` and `rand_chacha = "0.3"` directly. Shuffle with a
local `ChaCha12Rng::seed_from_u64(seed)` and `SliceRandom`. `RandomSampler::new`
rejects length zero with `RustTorchError::InvalidConfiguration` naming the
`length` field; it does not claim PyTorch's exact RNG sequence. Add
`pub mod data;` without root reexports.

- [ ] **Step 4: Run tests and documentation checks**

Run:

```sh
cargo test -p rusttorch --test data --no-default-features --features tch/doc-only
cargo check -p rusttorch --all-targets --no-default-features --features tch/doc-only
```

Expected: sampler and dataset tests pass with no missing public docs.

- [ ] **Step 5: Commit**

```sh
git add src/data.rs src/lib.rs tests/data.rs Cargo.toml Cargo.lock
git commit -m "feat: add datasets and deterministic samplers"
```

### Task 2: Batch map-style datasets with fallible collation

**Files:**
- Modify: `src/data.rs`
- Modify: `tests/data.rs`

**Interfaces:**
- Consumes: `Dataset` and any `Iterator<Item = usize>` sampler.
- Produces: `DataLoader::new(dataset, sampler, batch_size, drop_last) -> crate::Result<DataLoader<...>>` with identity `Vec<Sample>` collation.
- Produces: `DataLoader::with_collate(dataset, sampler, batch_size, drop_last, collate) -> crate::Result<DataLoader<...>>` for fallible custom collation.
- Produces: `Iterator<Item = std::result::Result<Batch, E>>` with
  `E: From<Dataset::Error>` for custom collation.

- [ ] **Step 1: Write batching behavior tests**

Cover both tail modes:

```rust
let kept = DataLoader::new(&dataset, SequentialSampler::new(5), 2, false)
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
assert_eq!(kept, vec![vec![0, 1], vec![2, 3], vec![4]]);

let dropped = DataLoader::new(&dataset, SequentialSampler::new(5), 2, true)
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
assert_eq!(dropped, vec![vec![0, 1], vec![2, 3]]);
```

Add `batch_size == 0`, `with_collate`, non-`Clone` samples, an empty dataset,
dataset failure before and after a partial batch, a partial `drop_last` failure,
and “error is yielded once, then iterator is exhausted” tests.

- [ ] **Step 2: Run batching tests and verify failure**

Run:

```sh
cargo test -p rusttorch --test data loader --no-default-features --features tch/doc-only
```

Expected: FAIL because `DataLoader` is absent.

- [ ] **Step 3: Implement one allocation-conscious batching loop**

Use this public constructor shape:

```rust
pub fn new(
    dataset: &'a D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
) -> crate::Result<Self>;

pub fn with_collate(
    dataset: &'a D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
    collate: C,
) -> crate::Result<Self>;
```

In `next`, obtain the first successful sample before allocating, then create
exactly one `Vec::with_capacity(batch_size)`, move samples into it, and move the
vector into identity or custom collation. An empty loader allocates no batch
vector. Set `exhausted` before returning any dataset or collation error,
including an error after a partial `drop_last` batch. A short successful tail
returns only when `drop_last == false`.

- [ ] **Step 4: Run focused and full data tests**

Run:

```sh
cargo test -p rusttorch --test data --no-default-features --features tch/doc-only
```

Expected: all map-style batching, ownership, error, and tail tests pass.

- [ ] **Step 5: Commit**

```sh
git add src/data.rs tests/data.rs
git commit -m "feat: batch map datasets with fallible collation"
```

### Task 3: Batch ordinary streaming iterators

**Files:**
- Modify: `src/data.rs`
- Modify: `tests/data.rs`

**Interfaces:**
- Produces: `batches(source, batch_size, drop_last) -> crate::Result<impl Iterator<Item = std::result::Result<Vec<Sample>, E>>>`.
- Produces: `batches_with_collate(source, batch_size, drop_last, collate) -> crate::Result<impl Iterator<Item = std::result::Result<Batch, E>>>`.
- Reuses: the same private batching core and error/exhaustion semantics as Task 2.

- [ ] **Step 1: Write stream parity tests**

Use an ordinary iterator, not a wrapper trait:

```rust
let source = (0..5).map(Ok::<_, TestError>);
let batches = batches(source, 2, false)
    .unwrap()
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
```

Add `batches_with_collate`, empty-stream allocation, drop-last, source error
after a partial dropped batch, collate error, non-`Clone` sample, and exact
error-once tests.

- [ ] **Step 2: Run stream tests and verify failure**

Run:

```sh
cargo test -p rusttorch --test data stream --no-default-features --features tch/doc-only
```

Expected: FAIL because `batches` does not exist.

- [ ] **Step 3: Extract and reuse the private batching core**

Both map and stream entry points must route through the same code that fills a
pre-sized vector and terminates after an error. Obtain a first source item
before allocating. Set `exhausted` before returning any source or collation
error. Keep all concrete iterator and closure types monomorphized; do not box
the returned iterator.

- [ ] **Step 4: Run full data tests**

Run:

```sh
cargo test -p rusttorch --test data --no-default-features --features tch/doc-only
```

Expected: map and stream paths have identical batching/error behavior.

- [ ] **Step 5: Commit**

```sh
git add src/data.rs tests/data.rs
git commit -m "feat: batch fallible data streams"
```

### Task 4: Document, benchmark, and declare data compatibility

**Files:**
- Create: `benches/data_loader.rs`
- Modify: `src/data.rs`
- Modify: `Cargo.toml`
- Modify: `compat/pytorch_api.toml`
- Modify: `docs/architecture.md`
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: public data APIs from Tasks 1-3.
- Produces: map and streaming examples, a dependency-free `harness = false` benchmark, and compatibility evidence.

**Prerequisite:** Compatibility-ledger Tasks 1-2 are complete. Update schema-v2
rows and regenerate `docs/api-coverage.md`; never hand-edit generated coverage.

- [ ] **Step 1: Add ergonomic rustdoc examples**

Show map-style use with `RandomSampler` and tensor collation, plus streaming use
with an ordinary fallible iterator. Examples must compile without spelling
internal adapter types.

- [ ] **Step 2: Add the microbenchmark target**

Configure:

```toml
[[bench]]
name = "data_loader"
harness = false
```

The benchmark uses `Instant` and `std::hint::black_box`, reports sequential
loader nanoseconds per sample separately from random-shuffle construction, and
does not assert a machine-dependent timing threshold.

- [ ] **Step 3: Update compatibility and attribution**

Add exact schema-v2 rows `data.batches`, `data.dataset`, `data.loader`, and
`data.sampler`, with `tests/data.rs::exact_test_name` evidence,
`implementation = "rusttorch"`, and precise local-RNG scope. Mark workers,
prefetch, pinning, distributed sampling, and checkpoint/resume as planned.
Regenerate `docs/api-coverage.md` from the ledger.
Attribute substantially adapted behavior to PyTorch 2.13 data source paths in
`THIRD_PARTY_NOTICES.md`.

- [ ] **Step 4: Run quality and performance gates**

Run:

```sh
cargo fmt --all -- --check
cargo test -p rusttorch --test data
cargo clippy -p rusttorch --all-targets --no-default-features --features tch/doc-only -- -D warnings
cargo bench -p rusttorch --bench data_loader
RUSTDOCFLAGS="-D warnings" cargo doc -p rusttorch --no-deps \
  --no-default-features --features tch/doc-only
```

Expected: tests/docs pass and the benchmark prints separate sequential and shuffle measurements.

- [ ] **Step 5: Commit**

```sh
git add benches/data_loader.rs src/data.rs Cargo.toml compat/pytorch_api.toml docs/api-coverage.md docs/architecture.md README.md THIRD_PARTY_NOTICES.md
git commit -m "docs: publish DataLoader coverage and benchmark"
```
