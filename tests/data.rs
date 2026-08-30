use std::{cell::Cell, convert::Infallible};

use rusttorch::{
    RustTorchError,
    data::{Dataset, RandomSampler, SequentialSampler},
};

struct CountingDataset {
    values: Vec<i32>,
    gets: Cell<usize>,
}

impl CountingDataset {
    fn new(values: Vec<i32>) -> Self {
        Self {
            values,
            gets: Cell::new(0),
        }
    }
}

impl Dataset for CountingDataset {
    type Sample = i32;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        self.gets.set(self.gets.get() + 1);
        Ok(self.values[index])
    }
}

#[test]
fn dataset_default_is_empty_follows_len() {
    assert!(CountingDataset::new(vec![]).is_empty());
    assert!(!CountingDataset::new(vec![7]).is_empty());
}

#[test]
fn dataset_samples_borrow_and_fetch_lazily_in_order() {
    let dataset = CountingDataset::new(vec![3, 5, 8]);
    let mut samples = dataset.samples();

    assert_eq!(dataset.gets.get(), 0);
    assert_eq!(samples.next(), Some(Ok(3)));
    assert_eq!(dataset.gets.get(), 1);
    assert_eq!(samples.collect::<Result<Vec<_>, _>>(), Ok(vec![5, 8]));
    assert_eq!(dataset.gets.get(), 3);
}

#[test]
fn sequential_sampler_yields_every_index_in_order() {
    assert_eq!(
        SequentialSampler::new(4).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn random_sampler_is_seeded_and_yields_a_permutation() {
    let first = RandomSampler::new(8, 42)
        .expect("positive length must be valid")
        .collect::<Vec<_>>();
    let second = RandomSampler::new(8, 42)
        .expect("positive length must be valid")
        .collect::<Vec<_>>();

    assert_eq!(first, second);
    let mut sorted = first;
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn random_sampler_rejects_zero_length_with_a_structured_error() {
    assert!(matches!(
        RandomSampler::new(0, 42),
        Err(RustTorchError::InvalidConfiguration {
            field: "length",
            ..
        })
    ));
}
