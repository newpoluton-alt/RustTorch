# rusttorch-data

Typed datasets, samplers, batching, and loading for RustTorch.

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
