use std::{
    convert::Infallible,
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Duration,
};

use rusttorch_core::{RustTorchError, Tensor};
use rusttorch_data::{
    CollateError, DataLoader, Dataset, FnBatchSource, FnCollate, FnSampler, LoaderError,
    PipelineError, RandomSampler, Sampler, VecCollate,
};

type DefaultLoaderError =
    LoaderError<PipelineError<Infallible, Infallible, CollateError, Infallible, Infallible>>;

struct Rows(Vec<i64>);

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

struct CustomSample;

struct CustomRows;

impl Dataset for CustomRows {
    type Sample = CustomSample;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(CustomSample)
    }
}

#[test]
fn defaults_batch_scalars_and_fresh_iterations_restart() -> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(Rows(vec![0, 1, 2])).build()?;

    assert_eq!(loader.len(), Some(3));
    assert_eq!(loader.workers(), 0);
    assert_eq!(loader.effective_prefetch_factor(), None);
    assert!(loader.is_ordered());
    assert!(!loader.pin_memory_enabled());
    assert_eq!(loader.timeout(), None);
    assert!(!loader.persistent_workers_enabled());

    let first = loader
        .iter()
        .collect::<Result<Vec<Tensor>, DefaultLoaderError>>()
        .expect("default scalar collation succeeds");
    let second = loader
        .iter()
        .collect::<Result<Vec<Tensor>, DefaultLoaderError>>()
        .expect("a fresh sampler iterator is created");
    assert_eq!(first.len(), 3);
    assert_eq!(first[0].int64_value(&[0]), 0);
    assert_eq!(first[2].int64_value(&[0]), 2);
    assert_eq!(second[0].int64_value(&[0]), 0);
    assert_eq!(second[2].int64_value(&[0]), 2);
    Ok(())
}

struct TensorRows;

impl Dataset for TensorRows {
    type Sample = Tensor;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(Tensor::from_slice(&[index as i64, index as i64 + 10]))
    }
}

#[test]
fn default_collator_stacks_tensor_samples() -> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(TensorRows).batch_size(2).build()?;
    let batch = loader
        .iter()
        .next()
        .expect("one batch")
        .expect("tensor stacking succeeds");
    assert_eq!(batch.size(), [2, 2]);
    assert_eq!(batch.int64_value(&[1, 0]), 1);
    Ok(())
}

#[test]
fn batching_controls_custom_sources_and_collators() -> Result<(), RustTorchError> {
    let dropped = DataLoader::builder(Rows(vec![0, 1, 2]))
        .batch_size(2)
        .drop_last(true)
        .build()?;
    assert_eq!(dropped.len(), Some(1));

    let mut sampled = DataLoader::builder(Rows(vec![10, 20, 30]))
        .sampler(FnSampler::new(Some(2), |_| [2, 0].into_iter()))
        .collate(VecCollate)
        .build()?;
    assert_eq!(
        sampled.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        vec![vec![30], vec![10]]
    );

    let mut shuffled = DataLoader::builder(Rows((0..8).collect()))
        .shuffle(7)?
        .collate(VecCollate)
        .build()?;
    let first = shuffled.iter().collect::<Result<Vec<_>, _>>().unwrap();
    let second = shuffled.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(first, second);

    let batches = FnBatchSource::new(Some(2), |_| vec![vec![2, 0], vec![1]].into_iter());
    let mut explicit = DataLoader::builder(Rows(vec![10, 20, 30]))
        .batch_sampler(batches)
        .collate(FnCollate::new(|values: Vec<i64>| {
            Ok::<_, Infallible>(values.into_iter().sum::<i64>())
        }))
        .build()?;
    assert_eq!(explicit.len(), Some(2));
    let expected = [40, 20];
    assert_eq!(
        explicit.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        expected
    );
    assert_eq!(
        explicit.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        expected
    );
    Ok(())
}

#[test]
fn no_batching_uses_default_and_custom_conversion() -> Result<(), RustTorchError> {
    let mut default = DataLoader::builder(Rows(vec![4, 5]))
        .without_batching()
        .build()?;
    assert_eq!(
        default.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [4, 5]
    );

    let mut custom = DataLoader::builder(Rows(vec![4, 5]))
        .without_batching()
        .convert(FnCollate::new(|values: Vec<i64>| {
            Ok::<_, Infallible>(values[0] * 10)
        }))
        .build()?;
    assert_eq!(
        custom.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [40, 50]
    );
    Ok(())
}

#[test]
fn optional_lengths_never_gate_iteration() -> Result<(), RustTorchError> {
    let mut sized = DataLoader::builder(Rows(vec![3, 4, 5]))
        .sampler(FnSampler::new(Some(2), |_| [2, 0].into_iter()))
        .collate(VecCollate)
        .build()?;
    assert_eq!(sized.len(), Some(2));
    let sized_expected = [vec![5], vec![3]];
    assert_eq!(
        sized.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        sized_expected
    );
    assert_eq!(
        sized.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        sized_expected
    );

    let mut unknown_len = DataLoader::builder(Rows(vec![3, 4, 5]))
        .sampler(FnSampler::new(None, |_| [1, 2].into_iter()))
        .collate(VecCollate)
        .build()?;
    assert_eq!(unknown_len.len(), None);
    let unknown_expected = [vec![4], vec![5]];
    assert_eq!(
        unknown_len.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        unknown_expected
    );
    assert_eq!(
        unknown_len.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        unknown_expected
    );

    let mut unsized_batches = DataLoader::builder(Rows(vec![3, 4, 5]))
        .batch_sampler(FnBatchSource::new(None, |_| vec![vec![0, 2]].into_iter()))
        .collate(VecCollate)
        .build()?;
    assert_eq!(unsized_batches.len(), None);
    let batch_expected = [vec![3, 5]];
    assert_eq!(
        unsized_batches
            .iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        batch_expected
    );
    assert_eq!(
        unsized_batches
            .iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        batch_expected
    );
    Ok(())
}

#[test]
fn set_epoch_forwards_through_every_plan() -> Result<(), RustTorchError> {
    let auto_epochs = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&auto_epochs);
    let auto_sampler = FnSampler::new(Some(1), move |epoch| {
        observed.lock().unwrap().push(epoch);
        [0].into_iter()
    });
    let mut auto = DataLoader::builder(Rows(vec![1]))
        .sampler(auto_sampler)
        .collate(VecCollate)
        .build()?;
    auto.set_epoch(7);
    auto.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(*auto_epochs.lock().unwrap(), [7]);

    let batch_epochs = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&batch_epochs);
    let batches = FnBatchSource::new(Some(1), move |epoch| {
        observed.lock().unwrap().push(epoch);
        vec![vec![0]].into_iter()
    });
    let mut explicit = DataLoader::builder(Rows(vec![1]))
        .batch_sampler(batches)
        .collate(VecCollate)
        .build()?;
    explicit.set_epoch(8);
    explicit.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(*batch_epochs.lock().unwrap(), [8]);

    let no_batch_epochs = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&no_batch_epochs);
    let sampler = FnSampler::new(Some(1), move |epoch| {
        observed.lock().unwrap().push(epoch);
        [0].into_iter()
    });
    let mut no_batch = DataLoader::builder(Rows(vec![1]))
        .sampler(sampler)
        .without_batching()
        .build()?;
    no_batch.set_epoch(9);
    no_batch.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(*no_batch_epochs.lock().unwrap(), [9]);
    Ok(())
}

#[test]
fn random_sampler_epoch_is_forwarded_and_each_iteration_is_fresh() -> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(Rows((0..8).collect()))
        .sampler(RandomSampler::new(8, 42)?)
        .collate(VecCollate)
        .build()?;
    let epoch_zero = loader.iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(
        epoch_zero,
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap()
    );
    loader.set_epoch(1);
    let epoch_one = loader.iter().collect::<Result<Vec<_>, _>>().unwrap();

    let mut expected = RandomSampler::new(8, 42)?;
    Sampler::set_epoch(&mut expected, 1);
    assert_eq!(
        epoch_one.into_iter().flatten().collect::<Vec<_>>(),
        Sampler::iter(&expected)
            .map(|index| index as i64)
            .collect::<Vec<_>>()
    );
    Ok(())
}

struct ShortBatch;

impl Dataset for ShortBatch {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("get_batch is overridden")
    }

    fn get_batch(&self, _indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        Ok(vec![1])
    }
}

#[test]
fn wrong_batch_cardinality_is_typed_and_precedes_collation() -> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(ShortBatch)
        .batch_size(2)
        .collate(FnCollate::new(
            |_: Vec<i64>| -> Result<Vec<i64>, Infallible> { panic!("collation must not run") },
        ))
        .build()?;
    assert!(matches!(
        loader.iter().next(),
        Some(Err(LoaderError::InvalidBatchCardinality {
            batch: 0,
            worker: None,
            expected: 2,
            actual: 1,
        }))
    ));
    assert!(
        loader.iter().next().is_some(),
        "a new loader iterator starts fresh"
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum PipelineFailure {
    Dataset,
    Collate,
}

impl fmt::Display for PipelineFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for PipelineFailure {}

struct Fails(bool);

impl Dataset for Fails {
    type Sample = i64;
    type Error = PipelineFailure;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        if self.0 && index == 0 {
            Err(PipelineFailure::Dataset)
        } else {
            Ok(index as i64)
        }
    }
}

#[test]
fn pipeline_failures_remain_typed_and_terminate_iteration() -> Result<(), RustTorchError> {
    let mut dataset_failure = DataLoader::builder(Fails(true))
        .collate(FnCollate::new(|values: Vec<i64>| {
            Ok::<_, PipelineFailure>(values)
        }))
        .build()?;
    let mut iter = dataset_failure.iter();
    assert!(matches!(
        iter.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Dataset(PipelineFailure::Dataset),
        }))
    ));
    assert!(iter.next().is_none());

    let mut collate_failure = DataLoader::builder(Fails(false))
        .collate(FnCollate::new(|_: Vec<i64>| {
            Err::<Vec<i64>, _>(PipelineFailure::Collate)
        }))
        .build()?;
    let mut iter = collate_failure.iter();
    assert!(matches!(
        iter.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Collate(PipelineFailure::Collate),
        }))
    ));
    assert!(iter.next().is_none());
    Ok(())
}

fn assert_invalid(result: Result<impl Sized, RustTorchError>, field: &'static str) {
    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration { field: actual, .. }) if actual == field
    ));
}

#[test]
fn conflicts_are_order_independent() {
    let explicit_sampler = || FnSampler::new(Some(1), |_| [0].into_iter());
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .sampler(explicit_sampler())
            .shuffle(1)
            .unwrap()
            .build(),
        "sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .shuffle(1)
            .unwrap()
            .sampler(explicit_sampler())
            .build(),
        "sampler",
    );

    let batches = || FnBatchSource::new(Some(1), |_| vec![vec![0]].into_iter());
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_size(2)
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_sampler(batches())
            .batch_size(2)
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .shuffle(1)
            .unwrap()
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_sampler(batches())
            .shuffle(1)
            .unwrap()
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .sampler(explicit_sampler())
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_sampler(batches())
            .sampler(explicit_sampler())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .drop_last(true)
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_sampler(batches())
            .drop_last(true)
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .without_batching()
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .batch_sampler(batches())
            .without_batching()
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .drop_last(true)
            .without_batching()
            .build(),
        "drop_last",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .without_batching()
            .drop_last(true)
            .build(),
        "drop_last",
    );
}

#[test]
fn batch_sampler_conflicts_do_not_require_discarded_operational_bounds() {
    let batches = || FnBatchSource::new(Some(1), |_| vec![vec![0]].into_iter());
    let sampler = || FnSampler::new(Some(1), |_| [0].into_iter());

    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_size(2)
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_sampler(batches())
            .batch_size(2)
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .sampler(sampler())
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_sampler(batches())
            .sampler(sampler())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .shuffle(1)
            .unwrap()
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_sampler(batches())
            .shuffle(1)
            .unwrap()
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .drop_last(true)
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_sampler(batches())
            .drop_last(true)
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .without_batching()
            .convert(VecCollate)
            .batch_sampler(batches())
            .build(),
        "batch_sampler",
    );
    assert_invalid(
        DataLoader::builder(CustomRows)
            .batch_sampler(batches())
            .without_batching()
            .convert(VecCollate)
            .build(),
        "batch_sampler",
    );
}

#[test]
fn scalar_configuration_is_validated_and_prefetch_is_normalized() -> Result<(), RustTorchError> {
    assert_invalid(
        DataLoader::builder(Rows(vec![1])).batch_size(0).build(),
        "batch_size",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .timeout(Duration::from_millis(1))
            .build(),
        "timeout",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .prefetch_factor(2)
            .build(),
        "prefetch_factor",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .workers(1)
            .prefetch_factor(0)
            .build(),
        "prefetch_factor",
    );
    assert_invalid(
        DataLoader::builder(Rows(vec![1]))
            .persistent_workers(true)
            .build(),
        "persistent_workers",
    );

    let zero_timeout = DataLoader::builder(Rows(vec![1]))
        .timeout(Duration::ZERO)
        .build()?;
    assert_eq!(zero_timeout.timeout(), None);

    let defaults = DataLoader::builder(Rows(vec![1])).workers(2).build()?;
    assert_eq!(
        defaults
            .effective_prefetch_factor()
            .map(|value| value.get()),
        Some(2)
    );
    let explicit = DataLoader::builder(Rows(vec![1]))
        .workers(2)
        .prefetch_factor(5)
        .ordered(false)
        .pin_memory()
        .build()?;
    assert_eq!(
        explicit
            .effective_prefetch_factor()
            .map(|value| value.get()),
        Some(5)
    );
    assert!(!explicit.is_ordered());
    assert!(explicit.pin_memory_enabled());
    assert!(!explicit.persistent_workers_enabled());
    let timed = DataLoader::builder(Rows(vec![1]))
        .workers(2)
        .timeout(Duration::from_millis(1))
        .build()?;
    assert_eq!(timed.timeout(), Some(Duration::from_millis(1)));
    let persistent = DataLoader::builder(Rows(vec![1]))
        .workers(2)
        .persistent_workers(true)
        .build()?;
    assert!(persistent.persistent_workers_enabled());
    Ok(())
}

#[test]
fn unallocatable_batch_size_is_a_typed_build_error_without_panicking() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        DataLoader::builder(Rows(vec![]))
            .batch_size(usize::MAX)
            .collate(VecCollate)
            .build()
    }));
    assert!(result.is_ok(), "caller-controlled capacity must not panic");
    assert_invalid(result.unwrap(), "batch_size");
}
