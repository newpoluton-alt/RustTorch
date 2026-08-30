use std::{convert::Infallible, hint::black_box, time::Instant};

use rusttorch::data::{DataLoader, Dataset, RandomSampler, SequentialSampler};

const SAMPLES: usize = 100_000;
const ROUNDS: usize = 20;
const BATCH_SIZE: usize = 256;

struct RangeDataset(usize);

impl Dataset for RangeDataset {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(index)
    }
}

fn main() {
    let dataset = RangeDataset(SAMPLES);
    let started = Instant::now();
    let mut checksum = 0_usize;
    for _ in 0..ROUNDS {
        let loader = DataLoader::new(&dataset, SequentialSampler::new(SAMPLES), BATCH_SIZE, false)
            .expect("batch size is nonzero");
        for batch in loader {
            for sample in batch.expect("range dataset is infallible") {
                checksum = checksum.wrapping_add(black_box(sample));
            }
        }
    }
    let sequential_ns = started.elapsed().as_nanos() as f64 / (SAMPLES * ROUNDS) as f64;
    black_box(checksum);

    let started = Instant::now();
    for seed in 0..ROUNDS as u64 {
        black_box(RandomSampler::new(SAMPLES, seed).expect("sample count is nonzero"));
    }
    let shuffle_elapsed = started.elapsed().as_nanos() as f64;

    println!("sequential loader: {sequential_ns:.2} ns/sample");
    println!(
        "random-shuffle construction: {:.0} ns/construction ({:.2} ns/index)",
        shuffle_elapsed / ROUNDS as f64,
        shuffle_elapsed / (SAMPLES * ROUNDS) as f64,
    );
}
