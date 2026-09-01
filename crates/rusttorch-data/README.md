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

Pinning and checkpoint/resume are not implemented in this scope.
