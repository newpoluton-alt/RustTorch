use std::{cell::Cell, convert::Infallible};

use rusttorch::{
    RustTorchError,
    data::{DataLoader, Dataset, RandomSampler, SequentialSampler, batches, batches_with_collate},
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

#[test]
fn loader_keeps_a_short_tail() {
    let dataset = CountingDataset::new(vec![0, 1, 2, 3, 4]);

    let batches = DataLoader::new(&dataset, SequentialSampler::new(5), 2, false)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the dataset is infallible");

    assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
}

#[test]
fn loader_drops_a_short_tail() {
    let dataset = CountingDataset::new(vec![0, 1, 2, 3, 4]);

    let batches = DataLoader::new(&dataset, SequentialSampler::new(5), 2, true)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the dataset is infallible");

    assert_eq!(batches, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn loader_rejects_zero_batch_size_with_a_structured_error() {
    let dataset = CountingDataset::new(vec![0]);

    assert!(matches!(
        DataLoader::new(&dataset, SequentialSampler::new(1), 0, false),
        Err(RustTorchError::InvalidConfiguration {
            field: "batch_size",
            ..
        })
    ));
}

#[test]
fn loader_applies_fallible_collation() {
    let dataset = CountingDataset::new(vec![1, 2, 3, 4]);

    let batches =
        DataLoader::with_collate(&dataset, SequentialSampler::new(4), 2, false, |samples| {
            Ok::<_, Infallible>(samples.into_iter().sum::<i32>())
        })
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the dataset and collation are infallible");

    assert_eq!(batches, vec![3, 7]);
}

#[derive(Debug, PartialEq, Eq)]
struct NonCloneSample(i32);

struct NonCloneDataset(Vec<i32>);

impl Dataset for NonCloneDataset {
    type Sample = NonCloneSample;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(NonCloneSample(self.0[index]))
    }
}

#[test]
fn loader_moves_non_clone_samples_into_batches() {
    let dataset = NonCloneDataset(vec![1, 2, 3]);

    let batches = DataLoader::new(&dataset, SequentialSampler::new(3), 2, false)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the dataset is infallible");

    assert_eq!(
        batches,
        vec![
            vec![NonCloneSample(1), NonCloneSample(2)],
            vec![NonCloneSample(3)]
        ]
    );
}

#[test]
fn loader_does_not_fetch_or_collate_an_empty_dataset() {
    let dataset = CountingDataset::new(vec![]);
    let collations = Cell::new(0);

    let mut loader =
        DataLoader::with_collate(&dataset, SequentialSampler::new(0), 2, false, |samples| {
            collations.set(collations.get() + 1);
            Ok::<_, Infallible>(samples)
        })
        .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), None);
    assert_eq!(dataset.gets.get(), 0);
    assert_eq!(collations.get(), 0);
}

#[derive(Debug, PartialEq, Eq)]
struct DatasetFailure(usize);

struct FailingDataset {
    length: usize,
    fail_at: usize,
}

impl Dataset for FailingDataset {
    type Sample = usize;
    type Error = DatasetFailure;

    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        if index == self.fail_at {
            Err(DatasetFailure(index))
        } else {
            Ok(index)
        }
    }
}

#[test]
fn loader_yields_a_first_sample_failure_once_then_exhausts() {
    let dataset = FailingDataset {
        length: 3,
        fail_at: 0,
    };
    let mut loader = DataLoader::new(&dataset, SequentialSampler::new(3), 2, false)
        .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Err(DatasetFailure(0))));
    assert_eq!(loader.next(), None);
}

#[test]
fn loader_discards_a_partial_batch_on_dataset_failure_then_exhausts() {
    let dataset = FailingDataset {
        length: 4,
        fail_at: 3,
    };
    let mut loader = DataLoader::new(&dataset, SequentialSampler::new(4), 2, false)
        .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Ok(vec![0, 1])));
    assert_eq!(loader.next(), Some(Err(DatasetFailure(3))));
    assert_eq!(loader.next(), None);
}

#[test]
fn loader_reports_a_partial_drop_last_failure_then_exhausts() {
    let dataset = FailingDataset {
        length: 4,
        fail_at: 3,
    };
    let mut loader = DataLoader::new(&dataset, SequentialSampler::new(4), 2, true)
        .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Ok(vec![0, 1])));
    assert_eq!(loader.next(), Some(Err(DatasetFailure(3))));
    assert_eq!(loader.next(), None);
}

#[derive(Debug, PartialEq, Eq)]
enum LoaderFailure {
    Dataset(DatasetFailure),
    Collate,
}

impl From<DatasetFailure> for LoaderFailure {
    fn from(error: DatasetFailure) -> Self {
        Self::Dataset(error)
    }
}

#[test]
fn loader_yields_a_collation_failure_once_then_exhausts() {
    let dataset = FailingDataset {
        length: 4,
        fail_at: usize::MAX,
    };
    let collations = Cell::new(0);
    let mut loader =
        DataLoader::with_collate(&dataset, SequentialSampler::new(4), 2, false, |_| {
            collations.set(collations.get() + 1);
            Err::<Vec<usize>, _>(LoaderFailure::Collate)
        })
        .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Err(LoaderFailure::Collate)));
    assert_eq!(loader.next(), None);
    assert_eq!(collations.get(), 1);
}

#[test]
fn stream_batches_keep_a_short_tail() {
    let source = (0..5).map(Ok::<_, Infallible>);

    let batches = batches(source, 2, false)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the source is infallible");

    assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
}

#[test]
fn stream_batches_drop_a_short_tail() {
    let source = (0..5).map(Ok::<_, Infallible>);

    let batches = batches(source, 2, true)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the source is infallible");

    assert_eq!(batches, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn stream_batches_reject_zero_batch_size_with_a_structured_error() {
    let source = std::iter::empty::<Result<i32, Infallible>>();

    assert!(matches!(
        batches(source, 0, false),
        Err(RustTorchError::InvalidConfiguration {
            field: "batch_size",
            ..
        })
    ));
}

#[test]
fn stream_batches_apply_fallible_collation() {
    let source = (1..=4).map(Ok::<_, Infallible>);

    let batches = batches_with_collate(source, 2, false, |samples| {
        Ok::<_, Infallible>(samples.into_iter().sum::<i32>())
    })
    .expect("a positive batch size must be valid")
    .collect::<Result<Vec<_>, _>>()
    .expect("the source and collation are infallible");

    assert_eq!(batches, vec![3, 7]);
}

#[test]
fn stream_batches_move_non_clone_samples() {
    let source = [1, 2, 3]
        .into_iter()
        .map(|value| Ok::<_, Infallible>(NonCloneSample(value)));

    let batches = batches(source, 2, false)
        .expect("a positive batch size must be valid")
        .collect::<Result<Vec<_>, _>>()
        .expect("the source is infallible");

    assert_eq!(
        batches,
        vec![
            vec![NonCloneSample(1), NonCloneSample(2)],
            vec![NonCloneSample(3)]
        ]
    );
}

#[test]
fn stream_batches_do_not_collate_an_empty_source() {
    let source_polls = Cell::new(0);
    let collations = Cell::new(0);
    let source = std::iter::from_fn(|| {
        source_polls.set(source_polls.get() + 1);
        None::<Result<i32, Infallible>>
    });
    let mut loader = batches_with_collate(source, 2, false, |samples| {
        collations.set(collations.get() + 1);
        Ok::<_, Infallible>(samples)
    })
    .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), None);
    assert_eq!(loader.next(), None);
    assert_eq!(source_polls.get(), 1);
    assert_eq!(collations.get(), 0);
}

#[derive(Debug, PartialEq, Eq)]
enum StreamFailure {
    Source,
    Collate,
}

#[test]
fn stream_batches_report_a_partial_drop_last_failure_once_then_exhaust() {
    let source = [Ok(0), Err(StreamFailure::Source), Ok(2)].into_iter();
    let mut loader = batches(source, 2, true).expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Err(StreamFailure::Source)));
    assert_eq!(loader.next(), None);
}

#[test]
fn stream_batches_report_a_collation_failure_once_then_exhaust() {
    let source = (0..4).map(Ok::<_, StreamFailure>);
    let collations = Cell::new(0);
    let mut loader = batches_with_collate(source, 2, false, |_| {
        collations.set(collations.get() + 1);
        Err::<Vec<i32>, _>(StreamFailure::Collate)
    })
    .expect("a positive batch size must be valid");

    assert_eq!(loader.next(), Some(Err(StreamFailure::Collate)));
    assert_eq!(loader.next(), None);
    assert_eq!(collations.get(), 1);
}
