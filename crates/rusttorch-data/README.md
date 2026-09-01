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
concrete task/completion slot buffers, worker/vector bookkeeping, and bounded
channel control-block allowances above 64 MiB before sampler or batch-source
callbacks. Positive-worker timeout and persistence requests also reject at
build; pinning, streaming workers, and checkpointing are not implemented in
this scope.
