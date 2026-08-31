use std::{convert::Infallible, marker::PhantomData, num::NonZeroUsize, time::Duration};

use rusttorch_core::{Result, RustTorchError};

use crate::sampler::validate_batch_size;
use crate::{
    BatchSampler, BatchSource, CloneTransformFactory, Collate, Dataset, DefaultCollator,
    DefaultConverter, IdentityTransformFactory, LoaderError, NoWorkerInit, PipelineError,
    RandomSampler, Sampler, SequentialSampler, TaskContext, Transform, TransformFactory,
    WorkerInit,
};

/// Type-level marker used only to make [`crate::DataLoader::builder`] inferable.
#[doc(hidden)]
pub struct BuilderDatasetMarker {
    _private: (),
}

impl Dataset for BuilderDatasetMarker {
    type Sample = ();
    type Error = Infallible;

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        unreachable!("the DataLoader builder marker never loads samples")
    }
}

/// Automatic fixed-size batching over a reusable sampler.
pub struct AutoBatch<S> {
    sampler: S,
    batch_size: NonZeroUsize,
    drop_last: bool,
}

/// Explicit reusable batches of indices.
pub struct ExplicitBatches<B> {
    batches: B,
}

/// One-sample loading without an automatic batch dimension.
pub struct NoBatch<S, V = DefaultConverter> {
    sampler: S,
    converter: V,
}

/// Internal plan behavior used by the named owned-loader iterator.
#[doc(hidden)]
pub trait LoaderPlan<Sample, C> {
    /// Fresh iterator of index groups.
    type Iter: Iterator<Item = Vec<usize>>;
    /// Item yielded by the loader.
    type Batch;
    /// Typed dataset/collation pipeline error.
    type Error;

    /// Creates fresh index groups for the current epoch.
    fn iter(&self) -> Self::Iter;
    /// Returns the exact output count when known.
    fn exact_len(&self) -> Option<usize>;
    /// Returns the configured epoch.
    fn epoch(&self) -> u64;
    /// Forwards a new epoch to the reusable source.
    fn set_epoch(&mut self, epoch: u64);
    /// Converts fetched samples into one visible item.
    fn finish(
        &mut self,
        collator: &mut C,
        samples: Vec<Sample>,
    ) -> std::result::Result<Self::Batch, Self::Error>;
}

/// Configuration-only behavior shared by loader index plans.
#[doc(hidden)]
pub trait LoaderPlanConfiguration {
    /// Applies builder-level automatic batch controls when relevant.
    fn apply_batch_options(&mut self, batch_size: NonZeroUsize, drop_last: bool);
    /// Applies the initial epoch selected by the builder.
    fn apply_epoch(&mut self, epoch: u64);
}

impl<Sample, S, C> LoaderPlan<Sample, C> for AutoBatch<S>
where
    S: Sampler,
    C: Collate<Sample>,
{
    type Iter = BatchSampler<S::Iter>;
    type Batch = C::Batch;
    type Error = C::Error;

    fn iter(&self) -> Self::Iter {
        BatchSampler::from_validated(self.sampler.iter(), self.batch_size, self.drop_last)
    }

    fn exact_len(&self) -> Option<usize> {
        self.sampler
            .exact_len()
            .map(|length| batch_count(length, self.batch_size, self.drop_last))
    }

    fn epoch(&self) -> u64 {
        self.sampler.epoch()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }

    fn finish(
        &mut self,
        collator: &mut C,
        samples: Vec<Sample>,
    ) -> std::result::Result<Self::Batch, Self::Error> {
        collator.collate(samples)
    }
}

impl<S> LoaderPlanConfiguration for AutoBatch<S>
where
    S: Sampler,
{
    fn apply_batch_options(&mut self, batch_size: NonZeroUsize, drop_last: bool) {
        self.batch_size = batch_size;
        self.drop_last = drop_last;
    }

    fn apply_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }
}

impl<Sample, B, C> LoaderPlan<Sample, C> for ExplicitBatches<B>
where
    B: BatchSource,
    C: Collate<Sample>,
{
    type Iter = B::Iter;
    type Batch = C::Batch;
    type Error = C::Error;

    fn iter(&self) -> Self::Iter {
        self.batches.iter()
    }

    fn exact_len(&self) -> Option<usize> {
        self.batches.exact_len()
    }

    fn epoch(&self) -> u64 {
        self.batches.epoch()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.batches.set_epoch(epoch);
    }

    fn finish(
        &mut self,
        collator: &mut C,
        samples: Vec<Sample>,
    ) -> std::result::Result<Self::Batch, Self::Error> {
        collator.collate(samples)
    }
}

impl<B> LoaderPlanConfiguration for ExplicitBatches<B>
where
    B: BatchSource,
{
    fn apply_batch_options(&mut self, _batch_size: NonZeroUsize, _drop_last: bool) {}

    fn apply_epoch(&mut self, epoch: u64) {
        self.batches.set_epoch(epoch);
    }
}

fn singleton(index: usize) -> Vec<usize> {
    vec![index]
}

impl<Sample, S, V, C> LoaderPlan<Sample, C> for NoBatch<S, V>
where
    S: Sampler,
    V: Collate<Sample>,
{
    type Iter = std::iter::Map<S::Iter, fn(usize) -> Vec<usize>>;
    type Batch = V::Batch;
    type Error = V::Error;

    fn iter(&self) -> Self::Iter {
        self.sampler.iter().map(singleton)
    }

    fn exact_len(&self) -> Option<usize> {
        self.sampler.exact_len()
    }

    fn epoch(&self) -> u64 {
        self.sampler.epoch()
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }

    fn finish(
        &mut self,
        _collator: &mut C,
        samples: Vec<Sample>,
    ) -> std::result::Result<Self::Batch, Self::Error> {
        self.converter.collate(samples)
    }
}

impl<S, V> LoaderPlanConfiguration for NoBatch<S, V>
where
    S: Sampler,
{
    fn apply_batch_options(&mut self, _batch_size: NonZeroUsize, _drop_last: bool) {}

    fn apply_epoch(&mut self, epoch: u64) {
        self.sampler.set_epoch(epoch);
    }
}

#[derive(Clone, Copy, Default)]
struct ExplicitArguments {
    batch_size: bool,
    drop_last: bool,
    sampler: bool,
    shuffle: bool,
    batch_sampler: bool,
    without_batching: bool,
    prefetch_factor: bool,
}

#[derive(Clone, Copy)]
struct LoaderConfiguration {
    batch_size: usize,
    drop_last: bool,
    workers: usize,
    persistent_workers: bool,
    timeout: Option<Duration>,
    ordered: bool,
    pin_memory: bool,
    prefetch_factor: Option<usize>,
    loader_seed: u64,
    epoch: u64,
    rank: usize,
}

impl Default for LoaderConfiguration {
    fn default() -> Self {
        Self {
            batch_size: 1,
            drop_last: false,
            workers: 0,
            persistent_workers: false,
            timeout: None,
            ordered: true,
            pin_memory: false,
            prefetch_factor: None,
            loader_seed: 0,
            epoch: 0,
            rank: 0,
        }
    }
}

/// Builder for an owned, re-iterable map-style data loader.
pub struct DataLoaderBuilder<D, P, C, F = IdentityTransformFactory, I = NoWorkerInit> {
    dataset: D,
    plan: P,
    collator: C,
    transform_factory: F,
    worker_init: I,
    configuration: LoaderConfiguration,
    explicit: ExplicitArguments,
}

impl<D> DataLoaderBuilder<D, AutoBatch<SequentialSampler>, DefaultCollator>
where
    D: Dataset,
{
    /// Creates the default serial builder for `dataset`.
    pub fn new(dataset: D) -> Self {
        Self {
            plan: AutoBatch {
                sampler: SequentialSampler::new(dataset.len()),
                batch_size: NonZeroUsize::MIN,
                drop_last: false,
            },
            dataset,
            collator: DefaultCollator,
            transform_factory: IdentityTransformFactory,
            worker_init: NoWorkerInit,
            configuration: LoaderConfiguration::default(),
            explicit: ExplicitArguments::default(),
        }
    }
}

impl<D, P, C, F, I> DataLoaderBuilder<D, P, C, F, I> {
    fn map<P2, C2>(
        self,
        transform: impl FnOnce(P, C) -> (P2, C2),
    ) -> DataLoaderBuilder<D, P2, C2, F, I> {
        let (plan, collator) = transform(self.plan, self.collator);
        DataLoaderBuilder {
            dataset: self.dataset,
            plan,
            collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
        }
    }

    /// Sets the automatic batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.configuration.batch_size = batch_size;
        self.explicit.batch_size = true;
        self
    }

    /// Selects whether a short final automatic batch is omitted.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.configuration.drop_last = drop_last;
        self.explicit.drop_last = true;
        self
    }

    /// Stores the requested worker count.
    pub fn workers(mut self, workers: usize) -> Self {
        self.configuration.workers = workers;
        self
    }

    /// Sets the seed used by deterministic task and worker contexts.
    pub fn seed(mut self, seed: u64) -> Self {
        self.configuration.loader_seed = seed;
        self
    }

    /// Sets the initial epoch used by sampling and task contexts.
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.configuration.epoch = epoch;
        self
    }

    /// Sets the distributed rank included in deterministic contexts.
    pub fn rank(mut self, rank: usize) -> Self {
        self.configuration.rank = rank;
        self
    }

    /// Clones `transform` once for each serial iterator or worker lifecycle.
    pub fn transform<T>(
        self,
        transform: T,
    ) -> DataLoaderBuilder<D, P, C, CloneTransformFactory<T>, I> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: CloneTransformFactory::new(transform),
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
        }
    }

    /// Replaces transform construction with an explicit typed factory.
    pub fn transform_factory<F2>(self, factory: F2) -> DataLoaderBuilder<D, P, C, F2, I> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
        }
    }

    /// Stores the initializer used by future positive-worker execution.
    pub fn worker_init<I2>(self, worker_init: I2) -> DataLoaderBuilder<D, P, C, F, I2> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
        }
    }

    /// Selects persistent workers for positive-worker execution.
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.configuration.persistent_workers = persistent;
        self
    }

    /// Sets a worker wait timeout; [`Duration::ZERO`] disables it.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.configuration.timeout = (!timeout.is_zero()).then_some(timeout);
        self
    }

    /// Selects ordered or completion-order delivery.
    pub fn ordered(mut self, ordered: bool) -> Self {
        self.configuration.ordered = ordered;
        self
    }

    /// Requests recursive batch pinning when supported.
    pub fn pin_memory(mut self) -> Self {
        self.configuration.pin_memory = true;
        self
    }

    /// Sets batches prefetched per worker.
    pub fn prefetch_factor(mut self, factor: usize) -> Self {
        self.configuration.prefetch_factor = Some(factor);
        self.explicit.prefetch_factor = true;
        self
    }

    /// Replaces automatic batching with explicit reusable index batches.
    pub fn batch_sampler<B>(
        mut self,
        batches: B,
    ) -> DataLoaderBuilder<D, ExplicitBatches<B>, C, F, I> {
        self.explicit.batch_sampler = true;
        self.map(|_, collator| (ExplicitBatches { batches }, collator))
    }

    /// Validates configuration and constructs the owned loader.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] for incompatible
    /// PyTorch-style arguments or zero-valued size controls.
    pub fn build(mut self) -> Result<OwnedDataLoader<D, P, C, F, I>>
    where
        D: Dataset,
        P: LoaderPlanConfiguration,
    {
        validate_configuration(&self.configuration, self.explicit)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        self.plan
            .apply_batch_options(batch_size, self.configuration.drop_last);
        self.plan.apply_epoch(self.configuration.epoch);
        let effective_prefetch = match (
            self.configuration.workers,
            self.configuration.prefetch_factor,
        ) {
            (0, _) => None,
            (_, Some(factor)) => NonZeroUsize::new(factor),
            (_, None) => NonZeroUsize::new(2),
        };
        Ok(OwnedDataLoader {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            _worker_init: self.worker_init,
            workers: self.configuration.workers,
            persistent_workers: self.configuration.persistent_workers,
            timeout: self.configuration.timeout,
            ordered: self.configuration.ordered,
            pin_memory: self.configuration.pin_memory,
            effective_prefetch,
            loader_seed: self.configuration.loader_seed,
            rank: self.configuration.rank,
        })
    }
}

impl<D, S, C, F, I> DataLoaderBuilder<D, AutoBatch<S>, C, F, I>
where
    D: Dataset,
{
    /// Replaces the current sampler.
    pub fn sampler<S2>(mut self, sampler: S2) -> DataLoaderBuilder<D, AutoBatch<S2>, C, F, I> {
        self.explicit.sampler = true;
        self.map(|plan, collator| {
            (
                AutoBatch {
                    sampler,
                    batch_size: plan.batch_size,
                    drop_last: plan.drop_last,
                },
                collator,
            )
        })
    }

    /// Replaces the current sampler with a seeded random sampler.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for an empty dataset or an
    /// unallocatable permutation.
    pub fn shuffle(
        mut self,
        seed: u64,
    ) -> Result<DataLoaderBuilder<D, AutoBatch<RandomSampler>, C, F, I>> {
        self.explicit.shuffle = true;
        let sampler = RandomSampler::new(self.dataset.len(), seed)?;
        Ok(self.map(|plan, collator| {
            (
                AutoBatch {
                    sampler,
                    batch_size: plan.batch_size,
                    drop_last: plan.drop_last,
                },
                collator,
            )
        }))
    }

    /// Disables automatic batching and selects default conversion.
    pub fn without_batching(
        mut self,
    ) -> DataLoaderBuilder<D, NoBatch<S, DefaultConverter>, C, F, I> {
        self.explicit.without_batching = true;
        self.map(|plan, collator| {
            (
                NoBatch {
                    sampler: plan.sampler,
                    converter: DefaultConverter,
                },
                collator,
            )
        })
    }

    /// Replaces the automatic-batch collator.
    pub fn collate<C2>(self, collator: C2) -> DataLoaderBuilder<D, AutoBatch<S>, C2, F, I> {
        self.map(|plan, _| (plan, collator))
    }
}

impl<D, B, C, F, I> DataLoaderBuilder<D, ExplicitBatches<B>, C, F, I>
where
    D: Dataset,
{
    /// Selects a sampler; build rejects its conflict with the earlier batch sampler.
    pub fn sampler<S>(mut self, sampler: S) -> DataLoaderBuilder<D, AutoBatch<S>, C, F, I> {
        self.explicit.sampler = true;
        let batch_size =
            NonZeroUsize::new(self.configuration.batch_size).unwrap_or(NonZeroUsize::MIN);
        let drop_last = self.configuration.drop_last;
        self.map(|_, collator| {
            (
                AutoBatch {
                    sampler,
                    batch_size,
                    drop_last,
                },
                collator,
            )
        })
    }

    /// Selects shuffle; build rejects its conflict with the earlier batch sampler.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for an empty dataset or an
    /// unallocatable permutation.
    pub fn shuffle(
        mut self,
        seed: u64,
    ) -> Result<DataLoaderBuilder<D, AutoBatch<RandomSampler>, C, F, I>> {
        self.explicit.shuffle = true;
        let sampler = RandomSampler::new(self.dataset.len(), seed)?;
        let batch_size =
            NonZeroUsize::new(self.configuration.batch_size).unwrap_or(NonZeroUsize::MIN);
        let drop_last = self.configuration.drop_last;
        Ok(self.map(|_, collator| {
            (
                AutoBatch {
                    sampler,
                    batch_size,
                    drop_last,
                },
                collator,
            )
        }))
    }

    /// Disables batching; build rejects its conflict with the earlier batch sampler.
    pub fn without_batching(
        mut self,
    ) -> DataLoaderBuilder<D, NoBatch<SequentialSampler, DefaultConverter>, C, F, I> {
        self.explicit.without_batching = true;
        let length = self.dataset.len();
        self.map(|_, collator| {
            (
                NoBatch {
                    sampler: SequentialSampler::new(length),
                    converter: DefaultConverter,
                },
                collator,
            )
        })
    }

    /// Replaces the explicit-batch collator.
    pub fn collate<C2>(self, collator: C2) -> DataLoaderBuilder<D, ExplicitBatches<B>, C2, F, I> {
        self.map(|plan, _| (plan, collator))
    }
}

impl<D, S, V, C, F, I> DataLoaderBuilder<D, NoBatch<S, V>, C, F, I>
where
    D: Dataset,
{
    /// Replaces the no-batching sampler.
    pub fn sampler<S2>(mut self, sampler: S2) -> DataLoaderBuilder<D, NoBatch<S2, V>, C, F, I> {
        self.explicit.sampler = true;
        self.map(|plan, collator| {
            (
                NoBatch {
                    sampler,
                    converter: plan.converter,
                },
                collator,
            )
        })
    }

    /// Replaces the no-batching sampler with a seeded random sampler.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for an empty dataset or an
    /// unallocatable permutation.
    #[allow(clippy::type_complexity)]
    pub fn shuffle(
        mut self,
        seed: u64,
    ) -> Result<DataLoaderBuilder<D, NoBatch<RandomSampler, V>, C, F, I>> {
        self.explicit.shuffle = true;
        let sampler = RandomSampler::new(self.dataset.len(), seed)?;
        Ok(self.map(|plan, collator| {
            (
                NoBatch {
                    sampler,
                    converter: plan.converter,
                },
                collator,
            )
        }))
    }

    /// Replaces the no-batching converter.
    pub fn convert<V2>(self, converter: V2) -> DataLoaderBuilder<D, NoBatch<S, V2>, C, F, I> {
        self.map(|plan, collator| {
            (
                NoBatch {
                    sampler: plan.sampler,
                    converter,
                },
                collator,
            )
        })
    }
}

/// An owned, re-iterable map-style data loader.
pub struct OwnedDataLoader<D, P, C, F = IdentityTransformFactory, I = NoWorkerInit> {
    dataset: D,
    plan: P,
    collator: C,
    transform_factory: F,
    _worker_init: I,
    workers: usize,
    persistent_workers: bool,
    timeout: Option<Duration>,
    ordered: bool,
    pin_memory: bool,
    effective_prefetch: Option<NonZeroUsize>,
    loader_seed: u64,
    rank: usize,
}

impl<D, P, C, F, I> OwnedDataLoader<D, P, C, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
{
    /// Starts a fresh finite iteration for the configured epoch.
    pub fn iter(&mut self) -> LoaderIter<'_, D, P, C, F, I> {
        let workers_unsupported = self.workers > 0;
        let (transform, transform_error) = if workers_unsupported {
            (None, None)
        } else {
            match self.transform_factory.create(None) {
                Ok(transform) => (Some(transform), None),
                Err(error) => (None, Some(error)),
            }
        };
        let batches = transform.as_ref().map(|_| self.plan.iter());
        let epoch = self.plan.epoch();
        LoaderIter {
            dataset: &self.dataset,
            plan: &mut self.plan,
            collator: &mut self.collator,
            batches,
            transform,
            transform_error,
            next_batch: 0,
            next_logical_sample: 0,
            exhausted: false,
            workers_unsupported,
            loader_seed: self.loader_seed,
            epoch,
            rank: self.rank,
            output: PhantomData,
        }
    }

    /// Returns the exact number of yielded items when the source is sized.
    pub fn len(&self) -> Option<usize> {
        self.plan.exact_len()
    }

    /// Returns whether iteration is empty when the source is sized.
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|length| length == 0)
    }

    /// Returns the current epoch.
    pub fn epoch(&self) -> u64 {
        self.plan.epoch()
    }

    /// Selects the epoch used by future iterations.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.plan.set_epoch(epoch);
    }

    /// Returns the configured worker count.
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Returns the effective batches-prefetched-per-worker value.
    pub fn effective_prefetch_factor(&self) -> Option<NonZeroUsize> {
        self.effective_prefetch
    }

    /// Returns whether results are configured for sampler order.
    pub fn is_ordered(&self) -> bool {
        self.ordered
    }

    /// Returns whether pinning was requested.
    pub fn pin_memory_enabled(&self) -> bool {
        self.pin_memory
    }

    /// Returns the active timeout, if any.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Returns whether persistent workers were requested.
    pub fn persistent_workers_enabled(&self) -> bool {
        self.persistent_workers
    }
}

/// One fresh iteration borrowed from an [`OwnedDataLoader`].
pub struct LoaderIter<'a, D, P, C, F = IdentityTransformFactory, I = NoWorkerInit>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
{
    dataset: &'a D,
    plan: &'a mut P,
    collator: &'a mut C,
    batches: Option<P::Iter>,
    transform: Option<F::Transform>,
    transform_error: Option<F::Error>,
    next_batch: u64,
    next_logical_sample: u64,
    exhausted: bool,
    workers_unsupported: bool,
    loader_seed: u64,
    epoch: u64,
    rank: usize,
    output: PhantomData<fn() -> I>,
}

impl<D, P, C, F, I> Iterator for LoaderIter<'_, D, P, C, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
{
    type Item = std::result::Result<
        P::Batch,
        LoaderError<
            PipelineError<
                D::Error,
                <F::Transform as Transform<D::Sample>>::Error,
                P::Error,
                F::Error,
                I::Error,
            >,
        >,
    >;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        if self.workers_unsupported {
            self.exhausted = true;
            return Some(Err(LoaderError::Configuration(invalid_configuration(
                "workers",
                "positive-worker execution is scheduled for DataLoader Task 7",
            ))));
        }
        if let Some(source) = self.transform_error.take() {
            self.exhausted = true;
            return Some(Err(LoaderError::Pipeline {
                batch: None,
                worker: None,
                source: PipelineError::TransformInit(source),
            }));
        }

        let indices = match self.batches.as_mut().and_then(Iterator::next) {
            Some(indices) => indices,
            None => {
                self.exhausted = true;
                return None;
            }
        };
        let batch = self.next_batch;
        let samples = match self.dataset.get_batch(&indices) {
            Ok(samples) => samples,
            Err(source) => {
                self.exhausted = true;
                return Some(Err(LoaderError::Pipeline {
                    batch: Some(batch),
                    worker: None,
                    source: PipelineError::Dataset(source),
                }));
            }
        };
        if samples.len() != indices.len() {
            self.exhausted = true;
            return Some(Err(LoaderError::InvalidBatchCardinality {
                batch,
                worker: None,
                expected: indices.len(),
                actual: samples.len(),
            }));
        }
        let mut transformed = Vec::with_capacity(samples.len());
        for sample in samples {
            let context = TaskContext {
                loader_seed: self.loader_seed,
                epoch: self.epoch,
                rank: self.rank,
                logical_sample: self.next_logical_sample,
                stage: 0,
            };
            let Some(transform) = self.transform.as_mut() else {
                self.exhausted = true;
                return None;
            };
            match transform.transform(sample, &context) {
                Ok(sample) => transformed.push(sample),
                Err(source) => {
                    self.exhausted = true;
                    return Some(Err(LoaderError::Pipeline {
                        batch: Some(batch),
                        worker: None,
                        source: PipelineError::Transform(source),
                    }));
                }
            }
            match self.next_logical_sample.checked_add(1) {
                Some(next) => self.next_logical_sample = next,
                None => self.exhausted = true,
            }
        }
        let result = self
            .plan
            .finish(self.collator, transformed)
            .map_err(|source| LoaderError::Pipeline {
                batch: Some(batch),
                worker: None,
                source: PipelineError::Collate(source),
            });
        if result.is_err() {
            self.exhausted = true;
        } else if let Some(next) = self.next_batch.checked_add(1) {
            self.next_batch = next;
        } else {
            self.exhausted = true;
        }
        Some(result)
    }
}

fn validate_configuration(
    configuration: &LoaderConfiguration,
    explicit: ExplicitArguments,
) -> Result<()> {
    if configuration.batch_size == 0 {
        return Err(invalid_configuration(
            "batch_size",
            "must be greater than zero",
        ));
    }
    if explicit.sampler && explicit.shuffle {
        return Err(invalid_configuration(
            "sampler",
            "cannot be combined with shuffle",
        ));
    }
    if explicit.batch_sampler
        && (explicit.batch_size
            || explicit.shuffle
            || explicit.sampler
            || explicit.drop_last
            || explicit.without_batching)
    {
        return Err(invalid_configuration(
            "batch_sampler",
            "cannot be combined with batch_size, shuffle, sampler, drop_last, or no batching",
        ));
    }
    if explicit.without_batching && explicit.drop_last && configuration.drop_last {
        return Err(invalid_configuration(
            "drop_last",
            "cannot be enabled when automatic batching is disabled",
        ));
    }
    if configuration.workers == 0 && configuration.timeout.is_some() {
        return Err(invalid_configuration(
            "timeout",
            "requires a positive worker count",
        ));
    }
    if explicit.prefetch_factor && configuration.workers == 0 {
        return Err(invalid_configuration(
            "prefetch_factor",
            "requires a positive worker count",
        ));
    }
    if configuration.prefetch_factor == Some(0) {
        return Err(invalid_configuration(
            "prefetch_factor",
            "must be greater than zero",
        ));
    }
    if configuration.persistent_workers && configuration.workers == 0 {
        return Err(invalid_configuration(
            "persistent_workers",
            "requires a positive worker count",
        ));
    }
    Ok(())
}

fn batch_count(length: usize, batch_size: NonZeroUsize, drop_last: bool) -> usize {
    let batch_size = batch_size.get();
    let complete = length / batch_size;
    complete + usize::from(!drop_last && !length.is_multiple_of(batch_size))
}

fn invalid_configuration(field: &'static str, reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
