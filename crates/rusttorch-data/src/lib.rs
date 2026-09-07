//! Typed datasets, samplers, batching, and loading for RustTorch.

#![deny(missing_docs)]
#![doc = include_str!("../COMPATIBILITY.md")]

use std::{convert::Infallible, marker::PhantomData};

use rusttorch_core::{Result, RustTorchError};

mod checkpoint;
mod collate;
mod dataset;
mod error;
mod loader;
mod memory;
mod pin_memory;
mod sampler;
mod stream;
mod transform;
mod worker;
mod worker_checkpoint;
mod worker_context;

pub use checkpoint::{
    CheckpointActive, CheckpointBuildError, CheckpointDisabled, CheckpointFresh,
    CheckpointIteration, CheckpointPinRequest, CheckpointPinStatus, CheckpointResume,
    Checkpointable, DatasetCheckpoint, LOADER_STATE_SCHEMA_VERSION, LoaderConfiguration,
    LoaderState, ReplaySafeDataset, Stateless, StatelessTransformFactory, StatelessWorker,
    TransactionalCheckpoint, TransactionalTransformFactory, TransactionalWorker, WorkerCheckpoint,
    WorkerCheckpointActive, WorkerCheckpointIteration, WorkerLaneState, WorkerTransformLanes,
    WorkerTransformState,
};
pub use collate::{
    Bytes, Collate, CollateError, DefaultCollate, DefaultCollator, DefaultConvert,
    DefaultConverter, FnCollate, VecCollate,
};
pub use dataset::{
    ConcatDataset, ReplaySafeMap, ReplaySafeTensorDataset, SplitLength, StackDataset, StackTuple,
    Subset, TensorDataset, TransactionalMap, chain_datasets, random_split,
};
pub use error::{LoaderError, PipelineError};
pub use loader::{
    AutoBatch, BuilderDatasetMarker, CheckpointPlan, DataLoaderBuilder, ExplicitBatches,
    LoaderIter, LoaderPlan, LoaderPlanConfiguration, NoBatch, OwnedDataLoader,
    PlanCheckpointIdentity, SerialExecution, WorkerExecution, WorkerLoaderIter,
};
pub use memory::{MemoryDisabled, MemoryEnabled, MemoryFootprint};
pub use pin_memory::{Auto, Explicit, PinDisabled, PinEnabled, PinMemory, PinMemoryStatus};
pub use sampler::{
    BatchSampler, BatchSamplerState, BatchSource, BatchSourceCheckpoint, DistributedConfiguration,
    DistributedSampler, DistributedSamplerState, FnBatchSource, FnSampler, RandomReplacement,
    RandomSampler, RandomSamplerState, Sampler, SamplerCheckpoint, SequentialSampler,
    SequentialSamplerState, SubsetRandomSampler, SubsetRandomSamplerState, WeightedRandomSampler,
    WeightedRandomSamplerState,
};
pub use stream::{
    LogicalSampleId, SequenceId, StreamDataLoader, StreamDataLoaderBuilder, StreamLoaderIter,
    WorkerRecord, WorkerSourceFactory,
};
pub use transform::{
    CloneTransformFactory, FnTransform, FnTransformFactory, IdentityTransform,
    IdentityTransformFactory, TASK_RNG_DERIVATION_VERSION, TaskContext, Transform,
    TransformFactory,
};
pub use worker_context::{
    CancellationToken, Deadline, FnWorkerInit, LoaderCancelled, NoWorkerInit,
    WORKER_SEED_DERIVATION_VERSION, WaitOutcome, WorkerContext, WorkerInfo, WorkerInit,
    get_worker_info, with_worker_info,
};

/// A finite, indexable collection of samples.
///
/// Implementations return owned samples so loaders can move them into batches
/// without cloning. Dataset-specific failures remain in [`Dataset::Error`].
pub trait Dataset {
    /// One owned item produced by the dataset.
    type Sample;

    /// An error returned while loading a sample.
    type Error;

    /// Returns the number of addressable samples.
    fn len(&self) -> usize;

    /// Loads the sample at `index`.
    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error>;

    /// Loads samples for one index batch.
    ///
    /// The default calls [`Dataset::get`] once per index. Overrides must
    /// return exactly one sample per supplied index and preserve index order.
    fn get_batch(&self, indices: &[usize]) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        indices.iter().map(|&index| self.get(index)).collect()
    }

    /// Loads one index batch with cooperative worker lifecycle context.
    ///
    /// The source-compatible default ignores `context` and delegates to
    /// [`Dataset::get_batch`] exactly once. Context-aware datasets may
    /// override this hook to honor cancellation and deadlines between bounded
    /// operations. Rust cannot force-cancel an arbitrary blocking native call.
    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> std::result::Result<Vec<Self::Sample>, Self::Error> {
        let _ = context;
        self.get_batch(indices)
    }

    /// Returns `true` when the dataset has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over every sample in index order.
    ///
    /// The iterator borrows the dataset and calls [`Dataset::get`] lazily.
    fn samples(&self) -> DatasetSamples<'_, Self>
    where
        Self: Sized,
    {
        DatasetSamples {
            dataset: self,
            next_index: 0,
        }
    }
}

/// A borrowing, sequential iterator over a [`Dataset`].
pub struct DatasetSamples<'a, D> {
    dataset: &'a D,
    next_index: usize,
}

impl<D> Iterator for DatasetSamples<'_, D>
where
    D: Dataset,
{
    type Item = std::result::Result<D::Sample, D::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.dataset.len() {
            return None;
        }

        let index = self.next_index;
        self.next_index += 1;
        Some(self.dataset.get(index))
    }
}

type IdentityCollate<T, E> = fn(Vec<T>) -> std::result::Result<Vec<T>, E>;

fn identity_batch<T, E>(samples: Vec<T>) -> std::result::Result<Vec<T>, E> {
    Ok(samples)
}

struct DatasetBatchIterator<'a, D, S, C, B, E> {
    dataset: &'a D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
    collate: C,
    exhausted: bool,
    output: PhantomData<fn() -> (B, E)>,
}

impl<D, S, C, B, E> Iterator for DatasetBatchIterator<'_, D, S, C, B, E>
where
    D: Dataset,
    S: Iterator<Item = usize>,
    C: FnMut(Vec<D::Sample>) -> std::result::Result<B, E>,
    E: From<D::Error>,
{
    type Item = std::result::Result<B, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let mut indices = Vec::with_capacity(self.batch_size);
        while indices.len() < self.batch_size {
            match self.sampler.next() {
                Some(index) => indices.push(index),
                None => {
                    self.exhausted = true;
                    if indices.is_empty() {
                        return None;
                    }
                    break;
                }
            }
        }

        let samples = match self.dataset.get_batch(&indices) {
            Ok(samples) => samples,
            Err(error) => {
                self.exhausted = true;
                return Some(Err(E::from(error)));
            }
        };
        assert_eq!(
            samples.len(),
            indices.len(),
            "Dataset::get_batch returned {} samples for {} indices",
            samples.len(),
            indices.len()
        );
        if self.drop_last && indices.len() < self.batch_size {
            return None;
        }
        let batch = (self.collate)(samples);
        if batch.is_err() {
            self.exhausted = true;
        }
        Some(batch)
    }
}

struct BatchIterator<I, C, T, B, E> {
    source: I,
    batch_size: usize,
    drop_last: bool,
    collate: C,
    exhausted: bool,
    output: PhantomData<fn(T) -> (B, E)>,
}

impl<I, C, T, B, E> BatchIterator<I, C, T, B, E> {
    fn new(source: I, batch_size: usize, drop_last: bool, collate: C) -> Result<Self> {
        if batch_size == 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "batch_size",
                reason: "must be greater than zero".to_owned(),
            });
        }

        Ok(Self {
            source,
            batch_size,
            drop_last,
            collate,
            exhausted: false,
            output: PhantomData,
        })
    }
}

impl<I, C, T, B, E> Iterator for BatchIterator<I, C, T, B, E>
where
    I: Iterator<Item = std::result::Result<T, E>>,
    C: FnMut(Vec<T>) -> std::result::Result<B, E>,
{
    type Item = std::result::Result<B, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let first = match self.source.next() {
            Some(Ok(sample)) => sample,
            Some(Err(error)) => {
                self.exhausted = true;
                return Some(Err(error));
            }
            None => {
                self.exhausted = true;
                return None;
            }
        };

        let mut samples = Vec::with_capacity(self.batch_size);
        samples.push(first);
        while samples.len() < self.batch_size {
            match self.source.next() {
                Some(Ok(sample)) => samples.push(sample),
                Some(Err(error)) => {
                    self.exhausted = true;
                    return Some(Err(error));
                }
                None => {
                    self.exhausted = true;
                    if self.drop_last {
                        return None;
                    }
                    break;
                }
            }
        }

        let batch = (self.collate)(samples);
        if batch.is_err() {
            self.exhausted = true;
        }
        Some(batch)
    }
}

/// Batches an ordinary fallible sample iterator into vectors.
///
/// Samples are moved into one pre-sized vector per batch. A short final batch
/// is omitted when `drop_last` is `true`. Source errors are yielded once and
/// then terminate the returned iterator.
///
/// # Errors
///
/// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is zero.
///
/// # Examples
///
/// An ordinary fallible iterator is enough for streaming data:
///
/// ```
/// use rusttorch_data::batches;
///
/// let records = ["10", "20", "30"].into_iter().map(str::parse::<i64>);
/// let loader = batches(records, 2, false).expect("batch size is nonzero");
/// let batches = loader
///     .collect::<Result<Vec<_>, _>>()
///     .expect("every record is an integer");
///
/// assert_eq!(batches, vec![vec![10, 20], vec![30]]);
/// ```
pub fn batches<I, T, E>(
    source: I,
    batch_size: usize,
    drop_last: bool,
) -> Result<impl Iterator<Item = std::result::Result<Vec<T>, E>>>
where
    I: Iterator<Item = std::result::Result<T, E>>,
{
    batches_with_collate(source, batch_size, drop_last, identity_batch::<T, E>)
}

/// Batches an ordinary fallible sample iterator with custom collation.
///
/// The closure receives ownership of exactly one vector of samples and may
/// return any batch type. A short final batch is omitted when `drop_last` is
/// `true`. Source and collation errors are yielded once and then terminate the
/// returned iterator.
///
/// # Errors
///
/// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is zero.
pub fn batches_with_collate<I, C, T, B, E>(
    source: I,
    batch_size: usize,
    drop_last: bool,
    collate: C,
) -> Result<impl Iterator<Item = std::result::Result<B, E>>>
where
    I: Iterator<Item = std::result::Result<T, E>>,
    C: FnMut(Vec<T>) -> std::result::Result<B, E>,
{
    BatchIterator::new(source, batch_size, drop_last, collate)
}

/// A single-threaded iterator over batches from a map-style [`Dataset`].
///
/// The loader borrows its dataset and owns both its sampler and collation
/// closure. Samples are moved into one pre-sized vector per batch without a
/// `Clone` requirement. Dataset and collation errors are yielded once and
/// then terminate the iterator.
///
/// # Examples
///
/// A seeded sampler and a collation closure can produce tensor batches:
///
/// ```no_run
/// use std::convert::Infallible;
///
/// use rusttorch_core::{Result, Tensor};
/// use rusttorch_data::{DataLoader, Dataset, RandomSampler};
///
/// struct Rows(usize);
///
/// impl Dataset for Rows {
///     type Sample = Tensor;
///     type Error = Infallible;
///
///     fn len(&self) -> usize {
///         self.0
///     }
///
///     fn get(&self, index: usize) -> std::result::Result<Tensor, Infallible> {
///         Ok(Tensor::from_slice(&[index as f32]))
///     }
/// }
///
/// fn main() -> Result<()> {
///     let dataset = Rows(8);
///     let sampler = RandomSampler::new(dataset.len(), 42)?;
///     let loader = DataLoader::with_collate(
///         &dataset,
///         sampler,
///         4,
///         false,
///         |samples| Ok::<_, Infallible>(Tensor::stack(&samples, 0)),
///     )?;
///
///     for batch in loader {
///         let tensor = batch.expect("dataset and collation are infallible");
///         assert_eq!(tensor.size(), [4, 1]);
///     }
///     Ok(())
/// }
/// ```
pub struct DataLoader<
    'a,
    D = BuilderDatasetMarker,
    S = std::iter::Empty<usize>,
    C = (),
    B = (),
    E = Infallible,
> where
    D: Dataset,
{
    batches: DatasetBatchIterator<'a, D, S, C, B, E>,
}

impl DataLoader<'static, BuilderDatasetMarker, std::iter::Empty<usize>, (), (), Infallible> {
    /// Starts an owned, re-iterable loader builder for `dataset`.
    ///
    /// ```no_run
    /// use std::convert::Infallible;
    /// use rusttorch_core::Result;
    /// use rusttorch_data::{DataLoader, Dataset, VecCollate};
    ///
    /// struct Rows(Vec<i64>);
    ///
    /// impl Dataset for Rows {
    ///     type Sample = i64;
    ///     type Error = Infallible;
    ///
    ///     fn len(&self) -> usize { self.0.len() }
    ///     fn get(&self, index: usize) -> std::result::Result<i64, Infallible> {
    ///         Ok(self.0[index])
    ///     }
    /// }
    ///
    /// fn main() -> Result<()> {
    ///     let mut loader = DataLoader::builder(Rows(vec![1, 2, 3]))
    ///         .batch_size(2)
    ///         .collate(VecCollate)
    ///         .build()?;
    ///     let batches = loader.iter().collect::<std::result::Result<Vec<_>, _>>()
    ///         .expect("the dataset and VecCollate are infallible");
    ///     assert_eq!(batches, [vec![1, 2], vec![3]]);
    ///     Ok(())
    /// }
    /// ```
    pub fn builder<D>(
        dataset: D,
    ) -> DataLoaderBuilder<D, AutoBatch<SequentialSampler>, DefaultCollator>
    where
        D: Dataset,
    {
        DataLoaderBuilder::new(dataset)
    }
}

impl<'a, D, S> DataLoader<'a, D, S, IdentityCollate<D::Sample, D::Error>, Vec<D::Sample>, D::Error>
where
    D: Dataset,
    S: Iterator<Item = usize>,
{
    /// Creates a loader whose batches are vectors of owned samples.
    ///
    /// `dataset` is borrowed, while `sampler` is consumed by the loader.
    /// A short final batch is omitted when `drop_last` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is
    /// zero.
    pub fn new(dataset: &'a D, sampler: S, batch_size: usize, drop_last: bool) -> Result<Self> {
        Self::with_collate(
            dataset,
            sampler,
            batch_size,
            drop_last,
            identity_batch::<D::Sample, D::Error>,
        )
    }
}

impl<'a, D, S, C, B, E> DataLoader<'a, D, S, C, B, E>
where
    D: Dataset,
    S: Iterator<Item = usize>,
    C: FnMut(Vec<D::Sample>) -> std::result::Result<B, E>,
    E: From<D::Error>,
{
    /// Creates a loader with fallible custom collation.
    ///
    /// The closure receives ownership of exactly one vector of samples and
    /// may return any batch type. Dataset failures are converted into the
    /// closure's error type through [`From`]. A short final batch is omitted
    /// when `drop_last` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `batch_size` is
    /// zero. Dataset and collation failures are returned by iteration.
    pub fn with_collate(
        dataset: &'a D,
        sampler: S,
        batch_size: usize,
        drop_last: bool,
        collate: C,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "batch_size",
                reason: "must be greater than zero".to_owned(),
            });
        }
        Ok(Self {
            batches: DatasetBatchIterator {
                dataset,
                sampler,
                batch_size,
                drop_last,
                collate,
                exhausted: false,
                output: PhantomData,
            },
        })
    }
}

impl<D, S, C, B, E> Iterator for DataLoader<'_, D, S, C, B, E>
where
    D: Dataset,
    S: Iterator<Item = usize>,
    C: FnMut(Vec<D::Sample>) -> std::result::Result<B, E>,
    E: From<D::Error>,
{
    type Item = std::result::Result<B, E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next()
    }
}
