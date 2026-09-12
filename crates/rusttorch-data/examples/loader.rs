use std::convert::Infallible;

use rusttorch_core::Tensor;
use rusttorch_data::{
    DataLoader, Dataset, DefaultCollator, DistributedSampler, LogicalSampleId, SequenceId,
    StreamDataLoaderBuilder, VecCollate, WorkerContext, WorkerRecord, WorkerSourceFactory,
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

struct Shards(usize);

struct RowShard {
    next: usize,
    end: usize,
    stride: usize,
}

impl Iterator for RowShard {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.end {
            return None;
        }
        let index = self.next;
        self.next += self.stride;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(index as u64)),
            logical_id: LogicalSampleId::new(index as u64),
            sample: index,
        }))
    }
}

impl WorkerSourceFactory for Shards {
    type Sample = usize;
    type Error = Infallible;
    type Source = RowShard;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        // Each worker owns a disjoint shard and reads one record at a time.
        Ok(RowShard {
            next: worker.info.id,
            end: self.0,
            stride: worker.info.num_workers,
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.0)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use batched tensors as model inputs. Workers load the rows, and the
    // coordinator stacks them into [batch, feature] tensors.
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

    // Every process selects its own rank; use identical length, seed and
    // epoch on all ranks. Sampling partitions inputs, not model gradients.
    let rank = 0;
    let replicas = 2;
    let sampler = DistributedSampler::new(128, replicas, rank, true, 42, false)?;
    let mut distributed = DataLoader::builder(Rows(128))
        .sampler(sampler)
        .rank(rank)
        .batch_size(32)
        .build()?;
    for epoch in 0..2 {
        distributed.set_epoch(epoch);
        for batch in distributed.iter() {
            assert_eq!(batch?.size(), [32, 1]);
        }
    }

    // Merge independently read shards into one ordered stream, then batch
    // globally. A non-divisible input retains exactly one short final batch.
    let mut stream = StreamDataLoaderBuilder::new(Shards(17))
        .workers(4)
        .batch_size(4)
        .collate(VecCollate)
        .build()?;
    let batches = stream.iter().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(batches.last().unwrap(), &[16]);
    let values = batches.into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(values, (0..17).collect::<Vec<_>>());
    Ok(())
}
