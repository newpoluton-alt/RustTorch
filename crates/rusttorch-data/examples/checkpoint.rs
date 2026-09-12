use std::convert::Infallible;

use rusttorch_data::{DataLoader, Dataset, ReplaySafeDataset, ReplaySafeMap, VecCollate};

// Immutable input plus stable indexing makes re-reading prefetched rows safe.
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

    // Save after a successfully consumed batch, together with the matching
    // model and optimizer state in a training application.
    let saved = serde_json::to_string(&iteration.checkpoint()?)?;
    let uninterrupted = iteration.collect::<Result<Vec<_>, _>>()?;

    // A new process recreates the same dataset and configuration, then loads
    // its saved state. Never turn a decoding or identity error into a restart.
    let state = serde_json::from_str(&saved)?;
    let mut resumed = build().resume_from(state).build()?;
    let remaining = resumed.iter().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(remaining, [vec![5, 7], vec![11]]);
    assert_eq!(remaining, uninterrupted);
    Ok(())
}
