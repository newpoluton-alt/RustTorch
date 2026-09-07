use std::convert::Infallible;

use rusttorch_core::Tensor;
use rusttorch_data::{
    DataLoader, Dataset, DefaultCollator, LogicalSampleId, SequenceId, StreamDataLoaderBuilder,
    VecCollate, WorkerContext, WorkerRecord, WorkerSourceFactory,
};

struct Rows(usize);

impl Dataset for Rows {
    type Sample = Tensor;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(Tensor::from_slice(&[index as f32]))
    }
}

#[derive(Clone)]
struct Shards(usize);

impl WorkerSourceFactory for Shards {
    type Sample = usize;
    type Error = Infallible;
    type Source = std::vec::IntoIter<Result<WorkerRecord<usize>, Infallible>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok((worker.info.id..self.0)
            .step_by(worker.info.num_workers)
            .map(|index| {
                Ok(WorkerRecord {
                    sequence: Some(SequenceId::new(index as u64)),
                    logical_id: LogicalSampleId::new(index as u64),
                    sample: index,
                })
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut loader = DataLoader::builder(Rows(128))
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

    let mut stream = StreamDataLoaderBuilder::new(Shards(16))
        .workers(4)
        .batch_size(4)
        .collate(VecCollate)
        .build()?;
    let values = stream
        .iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(values, (0..16).collect::<Vec<_>>());
    Ok(())
}
