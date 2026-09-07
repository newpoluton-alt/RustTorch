use std::{
    collections::BTreeMap,
    convert::Infallible,
    marker::PhantomData,
    num::NonZeroUsize,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use rusttorch_core::{Device, Result, RustTorchError, available_devices};

#[path = "loader_checkpoint.rs"]
mod checkpoint_workers;

use crate::checkpoint::{StatelessTransformFactory, TransactionalTransformFactory};
use crate::checkpoint::{
    WorkerCheckpointActive, WorkerCheckpointIteration, WorkerLaneState, WorkerTransformLanes,
    WorkerTransformState,
};
use crate::memory::{ByteBudget, MemoryDisabled, MemoryEnabled, MemoryPolicy};
use crate::sampler::{
    BatchSourceCheckpoint, DistributedConfiguration, SamplerCheckpoint, validate_batch_size,
};
use crate::worker::{
    WorkerBatch, WorkerFailure, WorkerMessage, WorkerPool, WorkerPoolConfiguration, WorkerReceive,
    WorkerRunContext, WorkerSubmit, WorkerTask, validate_worker_pool_capacity,
};
use crate::{
    Auto, BatchSampler, BatchSource, CheckpointActive, CheckpointBuildError, CheckpointDisabled,
    CheckpointFresh, CheckpointIteration, CheckpointPinRequest, CheckpointPinStatus,
    CheckpointResume, Checkpointable, CloneTransformFactory, Collate, Dataset, DatasetCheckpoint,
    Deadline, DefaultCollator, DefaultConverter, Explicit, IdentityTransformFactory,
    LOADER_STATE_SCHEMA_VERSION, LoaderError, LoaderState, MemoryFootprint, NoWorkerInit,
    PinDisabled, PinEnabled, PinMemory, PinMemoryStatus, PipelineError, RandomSampler, Sampler,
    SequentialSampler, TASK_RNG_DERIVATION_VERSION, TaskContext, Transform, TransformFactory,
    WORKER_SEED_DERIVATION_VERSION, WorkerCheckpoint, WorkerInit,
};

/// Type-level marker used only to make [`crate::DataLoader::builder`] inferable.
#[doc(hidden)]
pub struct BuilderDatasetMarker {
    _private: (),
}

/// Default execution capability preserving fully local serial data types.
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialExecution;

/// Execution capability selected by [`DataLoaderBuilder::workers`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerExecution;

trait SerialBoundaryPolicy {
    fn begin_next(&mut self);
    fn commit_visible(&mut self);
}

trait WorkerBoundaryPolicy<E> {
    fn begin_next(&mut self) {}
    fn submitted(&mut self, _sequence: u64, _logical_end: u64) {}
    fn committed(&mut self, _sequence: u64, _serial_logical_end: u64) {}
    fn defer_error(&mut self, _sequence: u64, error: E) -> Option<E> {
        Some(error)
    }
    fn take_error(&mut self, _sequence: u64) -> Option<E> {
        None
    }
}
impl<E> WorkerBoundaryPolicy<E> for CheckpointDisabled {}
impl<S, E> WorkerBoundaryPolicy<E> for WorkerCheckpointIteration<S, E> {
    fn begin_next(&mut self) {
        self.boundary_valid = false;
    }
    fn submitted(&mut self, sequence: u64, logical_end: u64) {
        self.logical_ends.insert(sequence, logical_end);
    }
    fn committed(&mut self, sequence: u64, serial_logical_end: u64) {
        self.next_logical_sample = self
            .logical_ends
            .remove(&sequence)
            .unwrap_or(serial_logical_end);
        self.boundary_valid = true;
        if let Some(barrier) = &self.barrier {
            barrier
                .committed
                .store(sequence + 1, std::sync::atomic::Ordering::Release);
        }
    }
    fn defer_error(&mut self, sequence: u64, error: E) -> Option<E> {
        self.errors.insert(sequence, error);
        None
    }
    fn take_error(&mut self, sequence: u64) -> Option<E> {
        self.errors.remove(&sequence)
    }
}

impl SerialBoundaryPolicy for CheckpointDisabled {
    fn begin_next(&mut self) {}
    fn commit_visible(&mut self) {}
}

impl SerialBoundaryPolicy for CheckpointIteration {
    fn begin_next(&mut self) {
        self.boundary_valid = false;
    }

    fn commit_visible(&mut self) {
        self.boundary_valid = true;
    }
}

type TransformOutput<D, F> =
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Output;
type TransformFailure<D, F> =
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Error;
type PlanFailure<D, P, C, F> = <P as LoaderPlan<TransformOutput<D, F>, C>>::Error;
type IterPipelineError<D, P, C, F, I> = PipelineError<
    <D as Dataset>::Error,
    TransformFailure<D, F>,
    PlanFailure<D, P, C, F>,
    <F as TransformFactory<<D as Dataset>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;
type IterError<D, P, C, F, I> = LoaderError<IterPipelineError<D, P, C, F, I>>;
type IterResult<D, P, C, F, I> = std::result::Result<
    <P as LoaderPlan<TransformOutput<D, F>, C>>::Batch,
    IterError<D, P, C, F, I>,
>;
type CompletedBatch<D, F, M> = (
    usize,
    WorkerBatch<TransformOutput<D, F>, <M as MemoryPolicy>::Permit>,
);
type FootprintFn<D, F> = fn(&[TransformOutput<D, F>]) -> usize;
type WorkerStageFailure<D, F, I> = WorkerFailure<
    <D as Dataset>::Error,
    TransformFailure<D, F>,
    <F as TransformFactory<<D as Dataset>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;

#[doc(hidden)]
pub trait MapPinPolicy<D, P, C, F>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<TransformOutput<D, F>, C>,
{
    fn pin(batch: P::Batch, device: Device) -> Result<P::Batch>;
}

impl<D, P, C, F> MapPinPolicy<D, P, C, F> for PinDisabled
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<TransformOutput<D, F>, C>,
{
    fn pin(batch: P::Batch, _device: Device) -> Result<P::Batch> {
        Ok(batch)
    }
}

impl<D, P, C, F, Q> MapPinPolicy<D, P, C, F> for PinEnabled<Q>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<TransformOutput<D, F>, C>,
    P::Batch: PinMemory,
{
    fn pin(batch: P::Batch, device: Device) -> Result<P::Batch> {
        batch.pin_memory(device)
    }
}

struct PendingSubmission {
    worker: usize,
    task: WorkerTask,
    next_submission: u64,
    logical_end: u64,
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

    /// Panic stage used while reading the plan epoch.
    const EPOCH_PANIC_STAGE: &'static str = "sampler epoch";
    /// Panic stage used while creating a fresh plan iterator.
    const CREATION_PANIC_STAGE: &'static str = "sampler creation";
    /// Panic stage used while refilling from the plan iterator.
    const REFILL_PANIC_STAGE: &'static str = "sampler refill";
    /// Panic stage used while producing the visible item.
    const FINISH_PANIC_STAGE: &'static str = "collation";

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

#[doc(hidden)]
#[derive(Clone)]
pub struct PlanCheckpointIdentity {
    batch_size: Option<usize>,
    drop_last: bool,
    sampler_kind: String,
    distributed: Option<DistributedConfiguration>,
}

#[doc(hidden)]
pub trait CheckpointPlan<Sample, C>: LoaderPlan<Sample, C> {
    type State;
    type CoordinatorState;

    const AUTOMATIC_BATCHING: bool = false;

    fn checkpoint_identity(&self) -> PlanCheckpointIdentity;
    /// Derives resume identity without observing a live plan or component.
    fn checkpoint_identity_from_state(
        state: &Self::State,
        batch_size: NonZeroUsize,
        drop_last: bool,
    ) -> Result<PlanCheckpointIdentity>;
    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State>;
    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()>;
    fn validate_checkpoint_state_with_batch_options(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
        _batch_size: NonZeroUsize,
        _drop_last: bool,
    ) -> Result<()> {
        self.validate_checkpoint_state(state, epoch, next_batch, next_logical_sample)
    }
    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter;
    fn save_coordinator(&self, collator: &C) -> Self::CoordinatorState;
    fn validate_coordinator(&self, collator: &C, state: &Self::CoordinatorState) -> Result<()>;
    fn restore_coordinator_validated(&mut self, collator: &mut C, state: &Self::CoordinatorState);
}

impl<Sample, S, C> LoaderPlan<Sample, C> for AutoBatch<S>
where
    S: Sampler,
    C: Collate<Sample>,
{
    type Iter = BatchSampler<S::Iter>;
    type Batch = C::Batch;
    type Error = C::Error;

    const EPOCH_PANIC_STAGE: &'static str = "sampler epoch";
    const CREATION_PANIC_STAGE: &'static str = "sampler creation";
    const REFILL_PANIC_STAGE: &'static str = "sampler refill";
    const FINISH_PANIC_STAGE: &'static str = "collation";

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

impl<Sample, S, C> CheckpointPlan<Sample, C> for AutoBatch<S>
where
    S: SamplerCheckpoint,
    C: Collate<Sample> + Checkpointable,
{
    type State = S::State;
    type CoordinatorState = C::State;

    const AUTOMATIC_BATCHING: bool = true;

    fn checkpoint_identity(&self) -> PlanCheckpointIdentity {
        PlanCheckpointIdentity {
            batch_size: Some(self.batch_size.get()),
            drop_last: self.drop_last,
            sampler_kind: self.sampler.kind().to_owned(),
            distributed: self.sampler.distributed_configuration(),
        }
    }

    fn checkpoint_identity_from_state(
        state: &Self::State,
        batch_size: NonZeroUsize,
        drop_last: bool,
    ) -> Result<PlanCheckpointIdentity> {
        Ok(PlanCheckpointIdentity {
            batch_size: Some(batch_size.get()),
            drop_last,
            sampler_kind: S::checkpoint_kind_from_state(state),
            distributed: S::checkpoint_distributed_configuration_from_state(state)?,
        })
    }

    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State> {
        validate_auto_batch_boundary(
            self.sampler.exact_len(),
            self.batch_size,
            self.drop_last,
            next_batch,
            next_logical_sample,
        )?;
        self.sampler.checkpoint_state(next_logical_sample)
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()> {
        <Self as CheckpointPlan<Sample, C>>::validate_checkpoint_state_with_batch_options(
            self,
            state,
            epoch,
            next_batch,
            next_logical_sample,
            self.batch_size,
            self.drop_last,
        )
    }

    fn validate_checkpoint_state_with_batch_options(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
        batch_size: NonZeroUsize,
        drop_last: bool,
    ) -> Result<()> {
        validate_auto_batch_boundary(
            self.sampler.exact_len(),
            batch_size,
            drop_last,
            next_batch,
            next_logical_sample,
        )?;
        self.sampler
            .validate_checkpoint_state(state, epoch, next_logical_sample)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        let iterator = self.sampler.restore_checkpoint_iter_validated(state);
        BatchSampler::from_validated(iterator, self.batch_size, self.drop_last)
    }

    fn save_coordinator(&self, collator: &C) -> Self::CoordinatorState {
        collator.save_state()
    }

    fn validate_coordinator(&self, collator: &C, state: &Self::CoordinatorState) -> Result<()> {
        collator.validate_state(state)
    }

    fn restore_coordinator_validated(&mut self, collator: &mut C, state: &Self::CoordinatorState) {
        collator.load_validated(state);
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

    const EPOCH_PANIC_STAGE: &'static str = "batch source epoch";
    const CREATION_PANIC_STAGE: &'static str = "batch source creation";
    const REFILL_PANIC_STAGE: &'static str = "batch source refill";
    const FINISH_PANIC_STAGE: &'static str = "collation";

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

impl<Sample, B, C> CheckpointPlan<Sample, C> for ExplicitBatches<B>
where
    B: BatchSourceCheckpoint,
    C: Collate<Sample> + Checkpointable,
{
    type State = B::State;
    type CoordinatorState = C::State;

    fn checkpoint_identity(&self) -> PlanCheckpointIdentity {
        PlanCheckpointIdentity {
            batch_size: None,
            drop_last: false,
            sampler_kind: self.batches.kind(),
            distributed: self.batches.distributed_configuration(),
        }
    }

    fn checkpoint_identity_from_state(
        state: &Self::State,
        _batch_size: NonZeroUsize,
        _drop_last: bool,
    ) -> Result<PlanCheckpointIdentity> {
        Ok(PlanCheckpointIdentity {
            batch_size: None,
            drop_last: false,
            sampler_kind: B::checkpoint_kind_from_state(state),
            distributed: B::checkpoint_distributed_configuration_from_state(state)?,
        })
    }

    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State> {
        self.batches
            .checkpoint_state(next_batch, next_logical_sample)
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()> {
        self.batches
            .validate_checkpoint_state(state, epoch, next_batch, next_logical_sample)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.batches.restore_checkpoint_iter_validated(state)
    }

    fn save_coordinator(&self, collator: &C) -> Self::CoordinatorState {
        collator.save_state()
    }

    fn validate_coordinator(&self, collator: &C, state: &Self::CoordinatorState) -> Result<()> {
        collator.validate_state(state)
    }

    fn restore_coordinator_validated(&mut self, collator: &mut C, state: &Self::CoordinatorState) {
        collator.load_validated(state);
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

    const EPOCH_PANIC_STAGE: &'static str = "sampler epoch";
    const CREATION_PANIC_STAGE: &'static str = "sampler creation";
    const REFILL_PANIC_STAGE: &'static str = "sampler refill";
    const FINISH_PANIC_STAGE: &'static str = "conversion";

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

impl<Sample, S, V, C> CheckpointPlan<Sample, C> for NoBatch<S, V>
where
    S: SamplerCheckpoint,
    V: Collate<Sample> + Checkpointable,
{
    type State = S::State;
    type CoordinatorState = V::State;

    fn checkpoint_identity(&self) -> PlanCheckpointIdentity {
        PlanCheckpointIdentity {
            batch_size: None,
            drop_last: false,
            sampler_kind: self.sampler.kind().to_owned(),
            distributed: self.sampler.distributed_configuration(),
        }
    }

    fn checkpoint_identity_from_state(
        state: &Self::State,
        _batch_size: NonZeroUsize,
        _drop_last: bool,
    ) -> Result<PlanCheckpointIdentity> {
        Ok(PlanCheckpointIdentity {
            batch_size: None,
            drop_last: false,
            sampler_kind: S::checkpoint_kind_from_state(state),
            distributed: S::checkpoint_distributed_configuration_from_state(state)?,
        })
    }

    fn checkpoint_state(&self, next_batch: u64, next_logical_sample: u64) -> Result<Self::State> {
        if next_batch != next_logical_sample {
            return Err(invalid_configuration(
                "checkpoint",
                "no-batch cursor must advance one batch per logical sample",
            ));
        }
        self.sampler.checkpoint_state(next_logical_sample)
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        next_batch: u64,
        next_logical_sample: u64,
    ) -> Result<()> {
        if next_batch != next_logical_sample {
            return Err(invalid_configuration(
                "checkpoint",
                "no-batch cursor must advance one batch per logical sample",
            ));
        }
        self.sampler
            .validate_checkpoint_state(state, epoch, next_logical_sample)
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.sampler
            .restore_checkpoint_iter_validated(state)
            .map(singleton as fn(usize) -> Vec<usize>)
    }

    fn save_coordinator(&self, _collator: &C) -> Self::CoordinatorState {
        self.converter.save_state()
    }

    fn validate_coordinator(&self, _collator: &C, state: &Self::CoordinatorState) -> Result<()> {
        self.converter.validate_state(state)
    }

    fn restore_coordinator_validated(&mut self, _collator: &mut C, state: &Self::CoordinatorState) {
        self.converter.load_validated(state);
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
    prefetch_bytes: bool,
}

#[derive(Clone, Copy)]
enum PinRequest {
    Disabled,
    Auto,
    Explicit(Device),
}

#[derive(Clone, Copy)]
struct BuilderConfiguration {
    batch_size: usize,
    drop_last: bool,
    workers: usize,
    persistent_workers: bool,
    timeout: Option<Duration>,
    ordered: bool,
    pin_memory: PinRequest,
    prefetch_factor: Option<usize>,
    prefetch_bytes: Option<NonZeroUsize>,
    loader_seed: u64,
    epoch: u64,
    rank: usize,
}

impl Default for BuilderConfiguration {
    fn default() -> Self {
        Self {
            batch_size: 1,
            drop_last: false,
            workers: 0,
            persistent_workers: false,
            timeout: None,
            ordered: true,
            pin_memory: PinRequest::Disabled,
            prefetch_factor: None,
            prefetch_bytes: None,
            loader_seed: 0,
            epoch: 0,
            rank: 0,
        }
    }
}

/// Builder for an owned, re-iterable map-style data loader.
pub struct DataLoaderBuilder<
    D,
    P,
    C,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    X = SerialExecution,
    M = MemoryDisabled,
    N = PinDisabled,
    K = CheckpointDisabled,
> {
    dataset: D,
    plan: P,
    collator: C,
    transform_factory: F,
    worker_init: I,
    configuration: BuilderConfiguration,
    explicit: ExplicitArguments,
    checkpoint: K,
    states: PhantomData<(X, M, N)>,
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
            configuration: BuilderConfiguration::default(),
            explicit: ExplicitArguments::default(),
            checkpoint: CheckpointDisabled,
            states: PhantomData,
        }
    }
}

impl<D, P, C, F, I, X, M, N, K> DataLoaderBuilder<D, P, C, F, I, X, M, N, K> {
    fn map<P2, C2>(
        self,
        transform: impl FnOnce(P, C) -> (P2, C2),
    ) -> DataLoaderBuilder<D, P2, C2, F, I, X, M, N, K> {
        let (plan, collator) = transform(self.plan, self.collator);
        DataLoaderBuilder {
            dataset: self.dataset,
            plan,
            collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
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

    /// Selects bounded map-worker execution and its thread-safe capability.
    ///
    /// Positive counts share the map dataset through [`Arc`], fetch and
    /// transform in deterministic worker lanes, and collate on the calling
    /// thread. Passing zero keeps serial execution, but selecting this method
    /// still requires worker-safe types at compile time.
    pub fn workers(
        mut self,
        workers: usize,
    ) -> DataLoaderBuilder<D, P, C, F, I, WorkerExecution, M, N, K> {
        self.configuration.workers = workers;
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
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
    ) -> DataLoaderBuilder<D, P, C, CloneTransformFactory<T>, I, X, M, N, K> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: CloneTransformFactory::new(transform),
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Replaces transform construction with an explicit typed factory.
    pub fn transform_factory<F2>(
        self,
        factory: F2,
    ) -> DataLoaderBuilder<D, P, C, F2, I, X, M, N, K> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Marks the current factory's transform as explicitly stateless for
    /// exact checkpointing.
    pub fn checkpoint_stateless(
        self,
    ) -> DataLoaderBuilder<D, P, C, StatelessTransformFactory<F>, I, X, M, N, K> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: StatelessTransformFactory(self.transform_factory),
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Marks the current factory's transform as transactional for exact
    /// checkpointing.
    pub fn checkpoint_transactional(
        self,
    ) -> DataLoaderBuilder<D, P, C, TransactionalTransformFactory<F>, I, X, M, N, K> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: TransactionalTransformFactory(self.transform_factory),
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Stores the initializer used by future positive-worker execution.
    pub fn worker_init<I2>(self, worker_init: I2) -> DataLoaderBuilder<D, P, C, F, I2, X, M, N, K> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Selects whether positive-worker iterators reuse loader-owned threads.
    ///
    /// Persistent workers keep their initial [`crate::WorkerInfo`] seeds and
    /// factory-created state across epochs. Each iterator still receives a
    /// fresh generation cancellation token and deadline, and dropping either
    /// the iterator or loader cooperatively wakes and joins the affected work.
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.configuration.persistent_workers = persistent;
        self
    }

    /// Sets the cooperative timeout for each blocking [`Iterator::next`] call.
    ///
    /// [`Duration::ZERO`] disables the timeout. Expiry cancels the current
    /// generation, drains it to quiescence, yields one typed timeout error, and
    /// then ends the iterator. User dataset and transform code must observe its
    /// [`crate::WorkerContext`] or [`crate::TaskContext`] to stop promptly;
    /// Rust threads are never force-cancelled. A nonzero timeout that exceeds
    /// the platform monotonic clock range is rejected by [`Self::build`].
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.configuration.timeout = (!timeout.is_zero()).then_some(timeout);
        self
    }

    /// Selects ordered or completion-order delivery.
    pub fn ordered(mut self, ordered: bool) -> Self {
        self.configuration.ordered = ordered;
        self
    }

    /// Selects sampler-order or completion-order delivery.
    pub fn in_order(self, in_order: bool) -> Self {
        self.ordered(in_order)
    }

    /// Enables automatic recursive pinning for CUDA device zero when available.
    ///
    /// Enabling pinning requires the final batch type to implement
    /// [`PinMemory`] at build time:
    ///
    /// ```compile_fail
    /// use std::convert::Infallible;
    /// use rusttorch_data::{DataLoader, Dataset, FnCollate};
    ///
    /// struct Rows;
    /// struct Batch;
    /// impl Dataset for Rows {
    ///     type Sample = u8;
    ///     type Error = Infallible;
    ///     fn len(&self) -> usize { 1 }
    ///     fn get(&self, _: usize) -> Result<u8, Infallible> { Ok(1) }
    /// }
    ///
    /// let _ = DataLoader::builder(Rows)
    ///     .collate(FnCollate::new(|_: Vec<u8>| Ok::<_, Infallible>(Batch)))
    ///     .pin_memory()
    ///     .build();
    /// ```
    pub fn pin_memory(mut self) -> DataLoaderBuilder<D, P, C, F, I, X, M, PinEnabled<Auto>, K> {
        self.configuration.pin_memory = PinRequest::Auto;
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Enables recursive pinning for one explicit, available CUDA device.
    pub fn pin_memory_for(
        mut self,
        device: Device,
    ) -> DataLoaderBuilder<D, P, C, F, I, X, M, PinEnabled<Explicit>, K> {
        self.configuration.pin_memory = PinRequest::Explicit(device);
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Sets bounded batches prefetched per worker.
    ///
    /// The worker count multiplied by this factor is checked before any
    /// worker starts and is also the global outstanding-work credit limit.
    /// Capacities that overflow crossbeam's ring arithmetic or whose
    /// conservative aggregate of concrete task/control/completion slots, worker
    /// bookkeeping, and channel control-block allowances exceeds the checked
    /// 64 MiB queue-allocation ceiling are rejected during build.
    pub fn prefetch_factor(mut self, factor: usize) -> Self {
        self.configuration.prefetch_factor = Some(factor);
        self.explicit.prefetch_factor = true;
        self
    }

    /// Enables a nonzero byte budget for final post-transform prefetched data.
    ///
    /// The coordinator's active collation batch is outside this queue budget
    /// and remains bounded by the configured batch size.
    ///
    /// Enabling the budget requires the final transformed type to implement
    /// [`MemoryFootprint`] at build time:
    ///
    /// ```compile_fail
    /// use std::{convert::Infallible, num::NonZeroUsize};
    /// use rusttorch_data::{DataLoader, Dataset, VecCollate};
    ///
    /// struct Sample;
    /// struct Rows;
    /// impl Dataset for Rows {
    ///     type Sample = Sample;
    ///     type Error = Infallible;
    ///     fn len(&self) -> usize { 1 }
    ///     fn get(&self, _: usize) -> Result<Sample, Infallible> { Ok(Sample) }
    /// }
    ///
    /// let _ = DataLoader::builder(Rows)
    ///     .workers(1)
    ///     .collate(VecCollate)
    ///     .prefetch_bytes(NonZeroUsize::MIN)
    ///     .build();
    /// ```
    pub fn prefetch_bytes(
        mut self,
        limit: NonZeroUsize,
    ) -> DataLoaderBuilder<D, P, C, F, I, X, MemoryEnabled, N, K> {
        self.configuration.prefetch_bytes = Some(limit);
        self.explicit.prefetch_bytes = true;
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: self.checkpoint,
            states: PhantomData,
        }
    }

    /// Replaces automatic batching with explicit reusable index batches.
    pub fn batch_sampler<B>(
        mut self,
        batches: B,
    ) -> DataLoaderBuilder<D, ExplicitBatches<B>, C, F, I, X, M, N, K> {
        self.explicit.batch_sampler = true;
        self.map(|_, collator| (ExplicitBatches { batches }, collator))
    }

    /// Validates configuration and constructs the owned loader.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] before plan callbacks
    /// for incompatible zero-worker options, zero-valued controls, or
    /// unallocatable worker queue sizes.
    #[allow(clippy::type_complexity)]
    fn build_inner(
        mut self,
        footprint: Option<FootprintFn<D, F>>,
    ) -> Result<OwnedDataLoader<D, P, C, F, I, X, M, N, K>>
    where
        D: Dataset,
        F: TransformFactory<D::Sample>,
        P: LoaderPlanConfiguration,
        I: WorkerInit,
        M: MemoryPolicy,
    {
        validate_configuration(&self.configuration, self.explicit)?;
        let pin_memory_status = resolve_pin_request(self.configuration.pin_memory)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        let effective_prefetch = match (
            self.configuration.workers,
            self.configuration.prefetch_factor,
        ) {
            (0, _) => None,
            (_, Some(factor)) => NonZeroUsize::new(factor),
            (_, None) => NonZeroUsize::new(2),
        };
        let outstanding_capacity = match (self.configuration.workers, effective_prefetch) {
            (0, _) => None,
            (workers, Some(factor)) => {
                Some(workers.checked_mul(factor.get()).ok_or_else(|| {
                    invalid_configuration(
                        "prefetch_factor",
                        "workers multiplied by prefetch_factor exceeds usize",
                    )
                })?)
            }
            (_, None) => unreachable!("positive workers always have an effective prefetch factor"),
        };
        if let Some(capacity) = outstanding_capacity {
            validate_worker_pool_capacity::<D, F, I, M>(
                self.configuration.workers,
                effective_prefetch
                    .expect("positive workers have a prefetch factor")
                    .get(),
                capacity,
            )?;
        }
        self.plan
            .apply_batch_options(batch_size, self.configuration.drop_last);
        self.plan.apply_epoch(self.configuration.epoch);
        Ok(OwnedDataLoader {
            dataset: Arc::new(self.dataset),
            plan: self.plan,
            collator: self.collator,
            transform_factory: Arc::new(self.transform_factory),
            worker_init: Arc::new(self.worker_init),
            workers: self.configuration.workers,
            persistent_workers: self.configuration.persistent_workers,
            timeout: self.configuration.timeout,
            ordered: self.configuration.ordered,
            pin_memory_status,
            footprint,
            effective_prefetch_bytes: self.configuration.prefetch_bytes,
            effective_prefetch,
            outstanding_capacity,
            loader_seed: self.configuration.loader_seed,
            rank: self.configuration.rank,
            next_generation: 0,
            persistent_pool: None,
            checkpoint: self.checkpoint,
            policies: PhantomData,
        })
    }
}

impl<D, P, C, F, I, X, M, N> DataLoaderBuilder<D, P, C, F, I, X, M, N, CheckpointDisabled> {
    /// Enables exact checkpointing with a caller-defined dataset identity.
    ///
    /// The identity must describe the exact dataset contents, not merely its
    /// Rust type or path. Empty identities are rejected by [`Self::build`].
    pub fn dataset_identity(
        self,
        identity: String,
    ) -> DataLoaderBuilder<D, P, C, F, I, X, M, N, CheckpointFresh> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: CheckpointFresh { identity },
            states: PhantomData,
        }
    }
}

impl<D, P, C, F, I, X, M, N> DataLoaderBuilder<D, P, C, F, I, X, M, N, CheckpointFresh> {
    /// Selects a typed serial loader checkpoint for validated resume.
    pub fn resume_from<S>(
        self,
        state: S,
    ) -> DataLoaderBuilder<D, P, C, F, I, X, M, N, CheckpointResume<S>> {
        DataLoaderBuilder {
            dataset: self.dataset,
            plan: self.plan,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            explicit: self.explicit,
            checkpoint: CheckpointResume {
                identity: self.checkpoint.identity,
                state,
            },
            states: PhantomData,
        }
    }
}

fn resident_bytes<T: MemoryFootprint>(values: &[T]) -> usize {
    let mut total = 0usize;
    for value in values {
        let Some(next) = total.checked_add(value.resident_bytes()) else {
            return usize::MAX;
        };
        total = next;
    }
    total
}

impl<D, P, C, F, I, X>
    DataLoaderBuilder<D, P, C, F, I, X, MemoryDisabled, PinDisabled, CheckpointDisabled>
{
    /// Validates configuration and builds an item-bounded, unpinned loader.
    #[allow(clippy::type_complexity)]
    pub fn build(
        self,
    ) -> Result<OwnedDataLoader<D, P, C, F, I, X, MemoryDisabled, PinDisabled, CheckpointDisabled>>
    where
        D: Dataset,
        F: TransformFactory<D::Sample>,
        P: LoaderPlanConfiguration,
        I: WorkerInit,
    {
        self.build_inner(None)
    }
}

impl<D, P, C, F, I, X>
    DataLoaderBuilder<D, P, C, F, I, X, MemoryEnabled, PinDisabled, CheckpointDisabled>
{
    /// Validates configuration and builds a byte-bounded, unpinned loader.
    #[allow(clippy::type_complexity)]
    pub fn build(
        self,
    ) -> Result<OwnedDataLoader<D, P, C, F, I, X, MemoryEnabled, PinDisabled, CheckpointDisabled>>
    where
        D: Dataset,
        F: TransformFactory<D::Sample>,
        TransformOutput<D, F>: MemoryFootprint,
        P: LoaderPlanConfiguration,
        I: WorkerInit,
    {
        self.build_inner(Some(resident_bytes::<TransformOutput<D, F>>))
    }
}

impl<D, P, C, F, I, X, Q>
    DataLoaderBuilder<D, P, C, F, I, X, MemoryDisabled, PinEnabled<Q>, CheckpointDisabled>
{
    /// Validates configuration and builds an item-bounded pinned loader.
    #[allow(clippy::type_complexity)]
    pub fn build(
        self,
    ) -> Result<OwnedDataLoader<D, P, C, F, I, X, MemoryDisabled, PinEnabled<Q>, CheckpointDisabled>>
    where
        D: Dataset,
        F: TransformFactory<D::Sample>,
        P: LoaderPlanConfiguration + LoaderPlan<TransformOutput<D, F>, C>,
        P::Batch: PinMemory,
        I: WorkerInit,
    {
        self.build_inner(None)
    }
}

impl<D, P, C, F, I, X, Q>
    DataLoaderBuilder<D, P, C, F, I, X, MemoryEnabled, PinEnabled<Q>, CheckpointDisabled>
{
    /// Validates configuration and builds a byte-bounded pinned loader.
    #[allow(clippy::type_complexity)]
    pub fn build(
        self,
    ) -> Result<OwnedDataLoader<D, P, C, F, I, X, MemoryEnabled, PinEnabled<Q>, CheckpointDisabled>>
    where
        D: Dataset,
        F: TransformFactory<D::Sample>,
        TransformOutput<D, F>: MemoryFootprint,
        P: LoaderPlanConfiguration + LoaderPlan<TransformOutput<D, F>, C>,
        P::Batch: PinMemory,
        I: WorkerInit,
    {
        self.build_inner(Some(resident_bytes::<TransformOutput<D, F>>))
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, N>
    DataLoaderBuilder<D, P, C, F, I, SerialExecution, MemoryDisabled, N, CheckpointFresh>
where
    D: DatasetCheckpoint,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint,
    P: LoaderPlanConfiguration
        + LoaderPlan<TransformOutput<D, F>, C>
        + CheckpointPlan<TransformOutput<D, F>, C>,
    I: WorkerInit,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Builds a fresh exact serial loader without constructing its transform.
    #[allow(clippy::type_complexity)]
    pub fn build(
        mut self,
    ) -> Result<
        OwnedDataLoader<
            D,
            P,
            C,
            F,
            I,
            SerialExecution,
            MemoryDisabled,
            N,
            CheckpointActive<P::Iter, F::Transform>,
        >,
    > {
        validate_exact_builder(
            &self.configuration,
            self.explicit,
            &self.checkpoint.identity,
        )?;
        let pin_memory_status = resolve_pin_request(self.configuration.pin_memory)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        self.plan
            .apply_batch_options(batch_size, self.configuration.drop_last);
        self.plan.apply_epoch(self.configuration.epoch);
        let identity = self.plan.checkpoint_identity();
        validate_distributed_rank(&identity, self.configuration.rank)?;
        let checkpoint_configuration =
            checkpoint_configuration(&identity, &self.configuration, pin_memory_status)?;

        Ok(OwnedDataLoader {
            dataset: Arc::new(self.dataset),
            plan: self.plan,
            collator: self.collator,
            transform_factory: Arc::new(self.transform_factory),
            worker_init: Arc::new(self.worker_init),
            workers: 0,
            persistent_workers: false,
            timeout: None,
            ordered: true,
            pin_memory_status,
            footprint: None,
            effective_prefetch_bytes: None,
            effective_prefetch: None,
            outstanding_capacity: None,
            loader_seed: self.configuration.loader_seed,
            rank: self.configuration.rank,
            next_generation: 0,
            persistent_pool: None,
            checkpoint: CheckpointActive {
                identity: self.checkpoint.identity,
                configuration: checkpoint_configuration,
                pending: None,
            },
            policies: PhantomData,
        })
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, N, DS, SS, TS, CS>
    DataLoaderBuilder<
        D,
        P,
        C,
        F,
        I,
        SerialExecution,
        MemoryDisabled,
        N,
        CheckpointResume<LoaderState<DS, SS, TS, CS>>,
    >
where
    D: DatasetCheckpoint<State = DS>,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint<State = TS>,
    P: LoaderPlanConfiguration
        + LoaderPlan<TransformOutput<D, F>, C>
        + CheckpointPlan<TransformOutput<D, F>, C, State = SS, CoordinatorState = CS>,
    I: WorkerInit,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Validates and restores an exact serial loader transactionally.
    #[allow(clippy::type_complexity)]
    pub fn build(
        mut self,
    ) -> std::result::Result<
        OwnedDataLoader<
            D,
            P,
            C,
            F,
            I,
            SerialExecution,
            MemoryDisabled,
            N,
            CheckpointActive<P::Iter, F::Transform>,
        >,
        CheckpointBuildError<F::Error>,
    > {
        let CheckpointResume {
            identity: requested_identity,
            state,
        } = self.checkpoint;
        validate_exact_builder(&self.configuration, self.explicit, &requested_identity)?;
        let pin_memory_status = resolve_pin_request(self.configuration.pin_memory)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        validate_static_loader_state_envelope(
            &state,
            &requested_identity,
            &self.configuration,
            pin_memory_status,
            <P as CheckpointPlan<TransformOutput<D, F>, C>>::AUTOMATIC_BATCHING,
            0,
        )?;

        let plan_identity =
            <P as CheckpointPlan<TransformOutput<D, F>, C>>::checkpoint_identity_from_state(
                &state.sampler,
                batch_size,
                self.configuration.drop_last,
            )?;
        validate_distributed_rank(&plan_identity, self.configuration.rank)?;
        let expected_configuration =
            checkpoint_configuration(&plan_identity, &self.configuration, pin_memory_status)?;
        validate_plan_loader_state_envelope(&state, &expected_configuration)?;

        self.dataset.validate_dataset_state(&state.dataset)?;
        self.plan.validate_checkpoint_state_with_batch_options(
            &state.sampler,
            state.epoch,
            state.next_batch,
            state.next_logical_sample,
            batch_size,
            self.configuration.drop_last,
        )?;
        let mut transform = self
            .transform_factory
            .create(None)
            .map_err(CheckpointBuildError::TransformFactory)?;
        transform.validate_snapshot(&state.transform)?;
        self.plan
            .validate_coordinator(&self.collator, &state.collate)?;

        self.dataset.restore_dataset_validated(&state.dataset);
        self.plan
            .apply_batch_options(batch_size, self.configuration.drop_last);
        let batches = self.plan.restore_checkpoint_iter_validated(&state.sampler);
        transform.restore_validated(&state.transform);
        self.plan
            .restore_coordinator_validated(&mut self.collator, &state.collate);

        Ok(OwnedDataLoader {
            dataset: Arc::new(self.dataset),
            plan: self.plan,
            collator: self.collator,
            transform_factory: Arc::new(self.transform_factory),
            worker_init: Arc::new(self.worker_init),
            workers: 0,
            persistent_workers: false,
            timeout: None,
            ordered: true,
            pin_memory_status,
            footprint: None,
            effective_prefetch_bytes: None,
            effective_prefetch: None,
            outstanding_capacity: None,
            loader_seed: self.configuration.loader_seed,
            rank: self.configuration.rank,
            next_generation: 0,
            persistent_pool: None,
            checkpoint: CheckpointActive {
                identity: requested_identity,
                configuration: expected_configuration,
                pending: Some(crate::checkpoint::PendingSerial {
                    batches,
                    transform,
                    next_batch: state.next_batch,
                    next_logical_sample: state.next_logical_sample,
                }),
            },
            policies: PhantomData,
        })
    }
}

impl<D, S, C, F, I, X, M, N, K> DataLoaderBuilder<D, AutoBatch<S>, C, F, I, X, M, N, K>
where
    D: Dataset,
{
    /// Replaces the current sampler.
    pub fn sampler<S2>(
        mut self,
        sampler: S2,
    ) -> DataLoaderBuilder<D, AutoBatch<S2>, C, F, I, X, M, N, K> {
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
    #[allow(clippy::type_complexity)]
    pub fn shuffle(
        mut self,
        seed: u64,
    ) -> Result<DataLoaderBuilder<D, AutoBatch<RandomSampler>, C, F, I, X, M, N, K>> {
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
    ) -> DataLoaderBuilder<D, NoBatch<S, DefaultConverter>, C, F, I, X, M, N, K> {
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
    pub fn collate<C2>(
        self,
        collator: C2,
    ) -> DataLoaderBuilder<D, AutoBatch<S>, C2, F, I, X, M, N, K> {
        self.map(|plan, _| (plan, collator))
    }
}

impl<D, B, C, F, I, X, M, N, K> DataLoaderBuilder<D, ExplicitBatches<B>, C, F, I, X, M, N, K>
where
    D: Dataset,
{
    /// Selects a sampler; build rejects its conflict with the earlier batch sampler.
    pub fn sampler<S>(
        mut self,
        sampler: S,
    ) -> DataLoaderBuilder<D, AutoBatch<S>, C, F, I, X, M, N, K> {
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
    #[allow(clippy::type_complexity)]
    pub fn shuffle(
        mut self,
        seed: u64,
    ) -> Result<DataLoaderBuilder<D, AutoBatch<RandomSampler>, C, F, I, X, M, N, K>> {
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
    ) -> DataLoaderBuilder<D, NoBatch<SequentialSampler, DefaultConverter>, C, F, I, X, M, N, K>
    {
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
    pub fn collate<C2>(
        self,
        collator: C2,
    ) -> DataLoaderBuilder<D, ExplicitBatches<B>, C2, F, I, X, M, N, K> {
        self.map(|plan, _| (plan, collator))
    }
}

impl<D, S, V, C, F, I, X, M, N, K> DataLoaderBuilder<D, NoBatch<S, V>, C, F, I, X, M, N, K>
where
    D: Dataset,
{
    /// Replaces the no-batching sampler.
    pub fn sampler<S2>(
        mut self,
        sampler: S2,
    ) -> DataLoaderBuilder<D, NoBatch<S2, V>, C, F, I, X, M, N, K> {
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
    ) -> Result<DataLoaderBuilder<D, NoBatch<RandomSampler, V>, C, F, I, X, M, N, K>> {
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
    pub fn convert<V2>(
        self,
        converter: V2,
    ) -> DataLoaderBuilder<D, NoBatch<S, V2>, C, F, I, X, M, N, K> {
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
///
/// The default [`SerialExecution`] capability accepts local non-`Send` data.
/// Calling [`DataLoaderBuilder::workers`] selects [`WorkerExecution`], where
/// positive counts share the dataset through [`Arc`]. Each worker owns a
/// bounded task lane and one transform instance; the coordinator alone owns
/// ordering and collation. This differs from PyTorch's process-local dataset
/// copies and worker-side collation while preserving bounded prefetch,
/// deterministic routing, task-local randomness, and ordered delivery.
#[allow(private_bounds)]
pub struct OwnedDataLoader<
    D,
    P,
    C,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    X = SerialExecution,
    M = MemoryDisabled,
    N = PinDisabled,
    K = CheckpointDisabled,
> where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    dataset: Arc<D>,
    plan: P,
    collator: C,
    transform_factory: Arc<F>,
    worker_init: Arc<I>,
    workers: usize,
    persistent_workers: bool,
    timeout: Option<Duration>,
    ordered: bool,
    pin_memory_status: PinMemoryStatus,
    footprint: Option<FootprintFn<D, F>>,
    effective_prefetch_bytes: Option<NonZeroUsize>,
    effective_prefetch: Option<NonZeroUsize>,
    outstanding_capacity: Option<usize>,
    loader_seed: u64,
    rank: usize,
    next_generation: u64,
    persistent_pool: Option<WorkerPool<D, F, I, M>>,
    checkpoint: K,
    policies: PhantomData<(X, M, N)>,
}

#[allow(private_bounds)]
impl<D, P, C, F, I, M, N> OwnedDataLoader<D, P, C, F, I, SerialExecution, M, N, CheckpointDisabled>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
    M: MemoryPolicy,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Starts a fresh finite iteration for the configured epoch.
    pub fn iter(&mut self) -> LoaderIter<'_, D, P, C, F, I, M, N, CheckpointDisabled> {
        let (transform, transform_error) = match self.transform_factory.create(None) {
            Ok(transform) => (Some(transform), None),
            Err(error) => (None, Some(error)),
        };
        let batches = transform.as_ref().map(|_| self.plan.iter());
        let epoch = self.plan.epoch();
        let run_context = WorkerRunContext::new(0, self.loader_seed, epoch);
        LoaderIter {
            dataset: self.dataset.as_ref(),
            plan: &mut self.plan,
            collator: &mut self.collator,
            batches,
            transform,
            transform_error,
            next_batch: 0,
            next_logical_sample: 0,
            exhausted: false,
            loader_seed: self.loader_seed,
            epoch,
            rank: self.rank,
            run_context,
            pin_memory_status: self.pin_memory_status,
            checkpoint: CheckpointDisabled,
            output: PhantomData,
            policies: PhantomData,
        }
    }
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, I, N>
    OwnedDataLoader<
        D,
        P,
        C,
        F,
        I,
        SerialExecution,
        MemoryDisabled,
        N,
        CheckpointActive<P::Iter, F::Transform>,
    >
where
    D: DatasetCheckpoint,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint,
    P: LoaderPlan<TransformOutput<D, F>, C> + CheckpointPlan<TransformOutput<D, F>, C>,
    I: WorkerInit,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Starts the retained resumed cursor once, then fresh exact iterations.
    pub fn iter(
        &mut self,
    ) -> LoaderIter<'_, D, P, C, F, I, MemoryDisabled, N, CheckpointIteration> {
        let pending = self.checkpoint.pending.take();
        let (batches, transform, transform_error, next_batch, next_logical_sample) = match pending {
            Some(pending) => (
                Some(pending.batches),
                Some(pending.transform),
                None,
                pending.next_batch,
                pending.next_logical_sample,
            ),
            None => match self.transform_factory.create(None) {
                Ok(transform) => (Some(self.plan.iter()), Some(transform), None, 0, 0),
                Err(error) => (None, None, Some(error), 0, 0),
            },
        };
        let epoch = self.plan.epoch();
        let run_context = WorkerRunContext::new(0, self.loader_seed, epoch);
        LoaderIter {
            dataset: self.dataset.as_ref(),
            plan: &mut self.plan,
            collator: &mut self.collator,
            batches,
            transform,
            transform_error,
            next_batch,
            next_logical_sample,
            exhausted: false,
            loader_seed: self.loader_seed,
            epoch,
            rank: self.rank,
            run_context,
            pin_memory_status: self.pin_memory_status,
            checkpoint: CheckpointIteration {
                identity: self.checkpoint.identity.clone(),
                configuration: self.checkpoint.configuration.clone(),
                boundary_valid: true,
            },
            output: PhantomData,
            policies: PhantomData,
        }
    }

    /// Selects the epoch used by future iterations.
    ///
    /// Calling this before the retained resumed cursor is consumed deliberately
    /// discards that cursor and its restored transform, so the selected epoch
    /// starts as an ordinary fresh iteration instead of mixing two epochs.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.plan.set_epoch(epoch);
        self.checkpoint.pending = None;
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, X, M, N> OwnedDataLoader<D, P, C, F, I, X, M, N, CheckpointDisabled>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    /// Selects the epoch used by future iterations.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.plan.set_epoch(epoch);
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, X, M, N, K> OwnedDataLoader<D, P, C, F, I, X, M, N, K>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
    M: MemoryPolicy,
{
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
        !matches!(self.pin_memory_status, PinMemoryStatus::Disabled)
    }

    /// Returns the effective recursive pinning behavior.
    pub fn pin_memory_status(&self) -> PinMemoryStatus {
        self.pin_memory_status
    }

    /// Returns the effective post-transform prefetch byte budget.
    pub fn effective_prefetch_bytes(&self) -> Option<NonZeroUsize> {
        self.effective_prefetch_bytes
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
pub struct LoaderIter<
    'a,
    D,
    P,
    C,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    M = MemoryDisabled,
    N = PinDisabled,
    K = CheckpointDisabled,
> where
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
    loader_seed: u64,
    epoch: u64,
    rank: usize,
    run_context: WorkerRunContext,
    pin_memory_status: PinMemoryStatus,
    checkpoint: K,
    output: PhantomData<fn() -> I>,
    policies: PhantomData<(M, N)>,
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, I, N> LoaderIter<'_, D, P, C, F, I, MemoryDisabled, N, CheckpointIteration>
where
    D: DatasetCheckpoint,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint,
    P: LoaderPlan<TransformOutput<D, F>, C> + CheckpointPlan<TransformOutput<D, F>, C>,
    I: WorkerInit,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Captures the exact next-visible-batch boundary.
    ///
    /// # Errors
    ///
    /// Returns [`LoaderError::Checkpoint`] after an error, end-of-input, or
    /// while a `next` call has advanced work that did not become visible.
    #[allow(clippy::type_complexity)]
    pub fn checkpoint(
        &mut self,
    ) -> std::result::Result<
        LoaderState<
            D::State,
            P::State,
            <F::Transform as WorkerCheckpoint>::State,
            P::CoordinatorState,
        >,
        IterError<D, P, C, F, I>,
    > {
        if !self.checkpoint.boundary_valid {
            return Err(LoaderError::Checkpoint {
                reason: "the active iterator is not at a consumer-visible batch boundary"
                    .to_owned(),
            });
        }
        let transform = self
            .transform
            .as_ref()
            .ok_or_else(|| LoaderError::Checkpoint {
                reason: "the serial transform was not constructed".to_owned(),
            })?;
        let sampler = self
            .plan
            .checkpoint_state(self.next_batch, self.next_logical_sample)
            .map_err(LoaderError::Configuration)?;
        Ok(LoaderState {
            schema_version: LOADER_STATE_SCHEMA_VERSION,
            dataset_identity: self.checkpoint.identity.clone(),
            epoch: self.epoch,
            iterator_generation: 0,
            next_batch: self.next_batch,
            next_logical_sample: self.next_logical_sample,
            dataset: self.dataset.snapshot_dataset(),
            sampler,
            transform: transform.snapshot(),
            collate: self.plan.save_coordinator(self.collator),
            rng_derivation_version: TASK_RNG_DERIVATION_VERSION,
            worker_seed_derivation_version: WORKER_SEED_DERIVATION_VERSION,
            configuration: self.checkpoint.configuration.clone(),
        })
    }
}

impl<D, P, C, F, I, M, N, K> Iterator for LoaderIter<'_, D, P, C, F, I, M, N, K>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
    N: MapPinPolicy<D, P, C, F>,
    K: SerialBoundaryPolicy,
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
        self.checkpoint.begin_next();
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
                cancellation: self.run_context.cancellation.clone(),
                deadline: self.run_context.deadline.clone(),
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
        let result =
            self.plan
                .finish(self.collator, transformed)
                .map_err(|source| LoaderError::Pipeline {
                    batch: Some(batch),
                    worker: None,
                    source: PipelineError::Collate(source),
                })
                .and_then(|value| match self.pin_memory_status {
                    PinMemoryStatus::Enabled(device) => N::pin(value, device)
                        .map_err(|source| LoaderError::PinMemory { batch, source }),
                    PinMemoryStatus::Disabled | PinMemoryStatus::DisabledNoAccelerator => Ok(value),
                });
        if result.is_err() {
            self.exhausted = true;
        } else if let Some(next) = self.next_batch.checked_add(1) {
            self.next_batch = next;
            self.checkpoint.commit_visible();
        } else {
            self.exhausted = true;
        }
        Some(result)
    }
}

/// One positive-worker iteration borrowed from an [`OwnedDataLoader`].
#[allow(private_bounds)]
pub struct WorkerLoaderIter<
    'a,
    D,
    P,
    C,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    M = MemoryDisabled,
    N = PinDisabled,
    K = CheckpointDisabled,
> where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    dataset: Arc<D>,
    plan: &'a mut P,
    collator: &'a mut C,
    batches: Option<P::Iter>,
    serial_transform: Option<F::Transform>,
    serial_transform_error: Option<F::Error>,
    pool: Option<IteratorPool<'a, D, F, I, M>>,
    completed: BTreeMap<u64, CompletedBatch<D, F, M>>,
    pending_error: Option<IterError<D, P, C, F, I>>,
    generation: u64,
    run_context: Option<WorkerRunContext>,
    workers: usize,
    capacity: usize,
    ordered: bool,
    timeout: Option<Duration>,
    loader_seed: u64,
    epoch: u64,
    rank: usize,
    next_submission: u64,
    next_visible: u64,
    next_logical_sample: u64,
    pending_submission: Option<PendingSubmission>,
    outstanding: usize,
    source_exhausted: bool,
    submission_closed: bool,
    exhausted: bool,
    pin_memory_status: PinMemoryStatus,
    _checkpoint: K,
    policies: PhantomData<(M, N)>,
}

enum IteratorPool<'a, D, F, I, M>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    Owned(WorkerPool<D, F, I, M>),
    Persistent {
        pool: &'a mut WorkerPool<D, F, I, M>,
        generation: u64,
    },
}

impl<D, F, I, M> IteratorPool<'_, D, F, I, M>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    fn pool(&self) -> &WorkerPool<D, F, I, M> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent { pool, .. } => pool,
        }
    }

    fn pool_mut(&mut self) -> &mut WorkerPool<D, F, I, M> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent { pool, .. } => pool,
        }
    }

    fn is_persistent(&self) -> bool {
        matches!(self, Self::Persistent { .. })
    }
}

impl<D, F, I, M> Drop for IteratorPool<'_, D, F, I, M>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
    M: MemoryPolicy,
{
    fn drop(&mut self) {
        if let Self::Persistent { pool, generation } = self
            && pool.quiesce(*generation).is_err()
        {
            pool.poison();
        }
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, M, N> OwnedDataLoader<D, P, C, F, I, WorkerExecution, M, N, CheckpointDisabled>
where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: MemoryPolicy + Send + 'static,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Starts a fresh generation on a bounded worker pool, or the explicit
    /// zero-worker path.
    pub fn iter(&mut self) -> WorkerLoaderIter<'_, D, P, C, F, I, M, N, CheckpointDisabled> {
        let (epoch, epoch_panicked) = if self.workers == 0 {
            (self.plan.epoch(), false)
        } else {
            match catch_unwind(AssertUnwindSafe(|| self.plan.epoch())) {
                Ok(epoch) => (epoch, false),
                Err(_) => (0, true),
            }
        };
        let mut iterator = WorkerLoaderIter {
            dataset: Arc::clone(&self.dataset),
            plan: &mut self.plan,
            collator: &mut self.collator,
            batches: None,
            serial_transform: None,
            serial_transform_error: None,
            pool: None,
            completed: BTreeMap::new(),
            pending_error: None,
            generation: self.next_generation,
            run_context: Some(WorkerRunContext::new(
                self.next_generation,
                self.loader_seed,
                epoch,
            )),
            workers: self.workers,
            capacity: self.outstanding_capacity.unwrap_or(0),
            ordered: self.ordered,
            timeout: self.timeout,
            loader_seed: self.loader_seed,
            epoch,
            rank: self.rank,
            next_submission: 0,
            next_visible: 0,
            next_logical_sample: 0,
            pending_submission: None,
            outstanding: 0,
            source_exhausted: false,
            submission_closed: false,
            exhausted: false,
            pin_memory_status: self.pin_memory_status,
            _checkpoint: CheckpointDisabled,
            policies: PhantomData,
        };

        if epoch_panicked {
            iterator.pending_error = Some(LoaderError::CoordinatorPanic {
                stage: P::EPOCH_PANIC_STAGE,
                batch: None,
            });
            return iterator;
        }
        if self.workers == 0 {
            match self.transform_factory.create(None) {
                Ok(transform) => {
                    iterator.serial_transform = Some(transform);
                    iterator.batches = Some(iterator.plan.iter());
                }
                Err(error) => iterator.serial_transform_error = Some(error),
            }
            return iterator;
        }
        let Some(next_generation) = self.next_generation.checked_add(1) else {
            iterator.pending_error = Some(LoaderError::Configuration(invalid_configuration(
                "workers",
                "iterator generation overflowed",
            )));
            return iterator;
        };
        iterator.batches = match catch_unwind(AssertUnwindSafe(|| iterator.plan.iter())) {
            Ok(batches) => Some(batches),
            Err(_) => {
                iterator.pending_error = Some(LoaderError::CoordinatorPanic {
                    stage: P::CREATION_PANIC_STAGE,
                    batch: None,
                });
                return iterator;
            }
        };
        self.next_generation = next_generation;
        let configuration = WorkerPoolConfiguration {
            workers: self.workers,
            prefetch_factor: self
                .effective_prefetch
                .expect("positive workers have a validated prefetch factor")
                .get(),
            result_capacity: iterator.capacity,
            seed_generation: iterator.generation,
            loader_seed: self.loader_seed,
            rank: self.rank,
        };
        let run_context =
            WorkerRunContext::new(iterator.generation, self.loader_seed, iterator.epoch)
                .with_byte_budget(
                    self.effective_prefetch_bytes
                        .map(|limit| ByteBudget::new(limit.get())),
                );
        iterator.run_context = Some(run_context.clone());

        if self.persistent_workers {
            if self.persistent_pool.is_none() {
                match WorkerPool::new(
                    Arc::clone(&self.dataset),
                    Arc::clone(&self.transform_factory),
                    Arc::clone(&self.worker_init),
                    configuration,
                    self.footprint,
                    self.ordered,
                ) {
                    Ok(pool) => self.persistent_pool = Some(pool),
                    Err(error) => {
                        iterator.pending_error = Some(LoaderError::Configuration(error));
                        return iterator;
                    }
                }
            }
            let pool = self
                .persistent_pool
                .as_mut()
                .expect("persistent pool was created");
            if pool.is_poisoned() || pool.start_generation(run_context).is_err() {
                iterator.pending_error = Some(LoaderError::ChannelClosed { batch: 0 });
                return iterator;
            }
            iterator.pool = Some(IteratorPool::Persistent {
                pool,
                generation: iterator.generation,
            });
        } else {
            match WorkerPool::new(
                Arc::clone(&self.dataset),
                Arc::clone(&self.transform_factory),
                Arc::clone(&self.worker_init),
                configuration,
                self.footprint,
                self.ordered,
            ) {
                Ok(mut pool) => {
                    if pool.start_generation(run_context).is_err() {
                        iterator.pending_error = Some(LoaderError::ChannelClosed { batch: 0 });
                    } else {
                        iterator.pool = Some(IteratorPool::Owned(pool));
                    }
                }
                Err(error) => iterator.pending_error = Some(LoaderError::Configuration(error)),
            }
        }
        if iterator.pending_error.is_none() {
            iterator.fill_available();
        }
        iterator
    }
}

#[allow(private_bounds)]
impl<D, P, C, F, I, M, N, K> WorkerLoaderIter<'_, D, P, C, F, I, M, N, K>
where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: MemoryPolicy + Send + 'static,
    N: MapPinPolicy<D, P, C, F>,
    K: WorkerBoundaryPolicy<IterError<D, P, C, F, I>>,
{
    fn fill_available(&mut self) {
        while self.outstanding < self.capacity && !self.source_exhausted && !self.submission_closed
        {
            if !self.submit_one() {
                break;
            }
        }
    }

    fn submit_one(&mut self) -> bool {
        if let Some(pending) = self.pending_submission.take() {
            return self.try_submit(pending);
        }
        let next_batch = catch_unwind(AssertUnwindSafe(|| {
            self.batches.as_mut().and_then(Iterator::next)
        }));
        let Some(indices) = (match next_batch {
            Ok(indices) => indices,
            Err(_) => {
                self.pending_error = Some(LoaderError::CoordinatorPanic {
                    stage: P::REFILL_PANIC_STAGE,
                    batch: Some(self.next_submission),
                });
                self.submission_closed = true;
                return false;
            }
        }) else {
            self.source_exhausted = true;
            return false;
        };
        let batch_sequence = self.next_submission;
        let Some(next_submission) = batch_sequence.checked_add(1) else {
            self.pending_error = Some(LoaderError::Configuration(invalid_configuration(
                "sampler",
                "batch sequence overflowed",
            )));
            self.submission_closed = true;
            return false;
        };
        let logical_start = self.next_logical_sample;
        let Some(logical_end) = logical_start.checked_add(indices.len() as u64) else {
            self.pending_error = Some(LoaderError::Configuration(invalid_configuration(
                "sampler",
                "logical sample identifier overflowed",
            )));
            self.submission_closed = true;
            return false;
        };
        let logical_samples = (logical_start..logical_end).collect();
        let worker = batch_sequence as usize % self.workers;
        self.try_submit(PendingSubmission {
            worker,
            task: WorkerTask {
                generation: self.generation,
                batch_sequence,
                logical_samples,
                indices,
            },
            next_submission,
            logical_end,
        })
    }

    fn try_submit(&mut self, pending: PendingSubmission) -> bool {
        let PendingSubmission {
            worker,
            task,
            next_submission,
            logical_end,
        } = pending;
        match self
            .pool
            .as_ref()
            .expect("submissions require an active worker pool")
            .pool()
            .submit(worker, task)
        {
            WorkerSubmit::Submitted => {
                self._checkpoint.submitted(next_submission - 1, logical_end);
                self.next_submission = next_submission;
                self.next_logical_sample = logical_end;
                self.outstanding += 1;
                true
            }
            WorkerSubmit::Full(task) => {
                self.pending_submission = Some(PendingSubmission {
                    worker,
                    task,
                    next_submission,
                    logical_end,
                });
                false
            }
            WorkerSubmit::Closed => {
                self.submission_closed = true;
                false
            }
        }
    }

    fn stop(&mut self) {
        self.stop_inner(false);
    }

    fn stop_poisoned(&mut self) {
        self.stop_inner(true);
    }

    fn stop_inner(&mut self, poisoned: bool) {
        self.exhausted = true;
        if let Some(run_context) = &self.run_context {
            run_context.cancel();
        }
        if let Some(mut pool) = self.pool.take() {
            let should_poison = poisoned
                || (pool.is_persistent() && pool.pool_mut().quiesce(self.generation).is_err());
            if should_poison {
                pool.pool_mut().poison();
            }
        }
        self.completed.clear();
        self.pending_submission = None;
    }

    fn next_serial(&mut self) -> Option<IterResult<D, P, C, F, I>> {
        if let Some(source) = self.serial_transform_error.take() {
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
        let batch = self.next_visible;
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
                cancellation: self
                    .run_context
                    .as_ref()
                    .expect("serial generation context")
                    .cancellation
                    .clone(),
                deadline: self
                    .run_context
                    .as_ref()
                    .expect("serial generation context")
                    .deadline
                    .clone(),
            };
            let transform = self
                .serial_transform
                .as_mut()
                .expect("serial transform exists after successful construction");
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
            self.next_logical_sample += 1;
        }
        let result =
            self.plan
                .finish(self.collator, transformed)
                .map_err(|source| LoaderError::Pipeline {
                    batch: Some(batch),
                    worker: None,
                    source: PipelineError::Collate(source),
                })
                .and_then(|value| match self.pin_memory_status {
                    PinMemoryStatus::Enabled(device) => N::pin(value, device)
                        .map_err(|source| LoaderError::PinMemory { batch, source }),
                    PinMemoryStatus::Disabled | PinMemoryStatus::DisabledNoAccelerator => Ok(value),
                });
        if result.is_err() {
            self.exhausted = true;
        } else {
            self._checkpoint
                .committed(self.next_visible, self.next_logical_sample);
            self.next_visible += 1;
        }
        Some(result)
    }

    fn publish(
        &mut self,
        worker: usize,
        batch: WorkerBatch<
            <F::Transform as Transform<D::Sample>>::Output,
            <M as MemoryPolicy>::Permit,
        >,
    ) -> IterResult<D, P, C, F, I> {
        debug_assert_eq!(batch.generation, self.generation);
        self.outstanding -= 1;
        let sequence = batch.batch_sequence;
        let WorkerBatch {
            batch_sequence,
            samples,
            permit,
            ..
        } = batch;
        drop(permit);
        let result = match catch_unwind(AssertUnwindSafe(|| {
            self.plan.finish(self.collator, samples)
        })) {
            Ok(result) => result.map_err(|source| LoaderError::Pipeline {
                batch: Some(sequence),
                worker: None,
                source: PipelineError::Collate(source),
            }),
            Err(_) => Err(LoaderError::CoordinatorPanic {
                stage: P::FINISH_PANIC_STAGE,
                batch: Some(sequence),
            }),
        };
        let result = result.and_then(|value| match self.pin_memory_status {
            PinMemoryStatus::Enabled(device) => {
                N::pin(value, device).map_err(|source| LoaderError::PinMemory {
                    batch: batch_sequence,
                    source,
                })
            }
            PinMemoryStatus::Disabled | PinMemoryStatus::DisabledNoAccelerator => Ok(value),
        });
        if result.is_ok() {
            self._checkpoint
                .committed(sequence, self.next_logical_sample);
            self.next_visible = self.next_visible.saturating_add(1);
            self.fill_available();
        } else {
            self.stop();
        }
        let _ = worker;
        result
    }

    fn map_failure(
        &mut self,
        worker: usize,
        batch: Option<u64>,
        failure: WorkerStageFailure<D, F, I>,
    ) -> IterError<D, P, C, F, I> {
        match failure {
            WorkerFailure::Dataset(source) => LoaderError::Pipeline {
                batch,
                worker: Some(worker),
                source: PipelineError::Dataset(source),
            },
            WorkerFailure::Transform(source) => LoaderError::Pipeline {
                batch,
                worker: Some(worker),
                source: PipelineError::Transform(source),
            },
            WorkerFailure::TransformInit(source) => LoaderError::Pipeline {
                batch,
                worker: Some(worker),
                source: PipelineError::TransformInit(source),
            },
            WorkerFailure::WorkerInit(source) => LoaderError::Pipeline {
                batch,
                worker: Some(worker),
                source: PipelineError::WorkerInit(source),
            },
            WorkerFailure::InvalidBatchCardinality { expected, actual } => {
                LoaderError::InvalidBatchCardinality {
                    batch: batch.unwrap_or(self.next_visible),
                    worker: Some(worker),
                    expected,
                    actual,
                }
            }
            WorkerFailure::MemoryLimit { limit, actual } => LoaderError::MemoryLimit {
                batch,
                worker: Some(worker),
                sequence: None,
                logical_id: None,
                limit,
                actual,
            },
            WorkerFailure::Panic => LoaderError::WorkerPanic { worker, batch },
        }
    }
}

impl<D, P, C, F, I, M, N, K> Iterator for WorkerLoaderIter<'_, D, P, C, F, I, M, N, K>
where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    P: LoaderPlan<<F::Transform as Transform<D::Sample>>::Output, C>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: MemoryPolicy + Send + 'static,
    N: MapPinPolicy<D, P, C, F>,
    K: WorkerBoundaryPolicy<IterError<D, P, C, F, I>>,
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
        self._checkpoint.begin_next();
        if self.exhausted {
            return None;
        }
        if let Some(error) = self.pending_error.take() {
            self.stop();
            return Some(Err(error));
        }
        if self.workers == 0 {
            return self.next_serial();
        }
        let mut deadline_armed = false;
        loop {
            if let Some(error) = self._checkpoint.take_error(self.next_visible) {
                self.stop();
                return Some(Err(error));
            }
            if self.ordered
                && let Some((worker, batch)) = self.completed.remove(&self.next_visible)
            {
                if deadline_armed {
                    self.run_context
                        .as_ref()
                        .expect("worker context")
                        .deadline
                        .disarm();
                }
                return Some(self.publish(worker, batch));
            }
            if self.outstanding == 0 && self.source_exhausted {
                if deadline_armed {
                    self.run_context
                        .as_ref()
                        .expect("worker context")
                        .deadline
                        .disarm();
                }
                self.stop();
                return None;
            }
            let run_context = self
                .run_context
                .as_ref()
                .expect("worker iteration has a generation context")
                .clone();
            if !deadline_armed && let Some(timeout) = self.timeout {
                run_context.deadline.arm(timeout);
                deadline_armed = true;
            }
            let remaining = self.timeout.and_then(|_| run_context.deadline.remaining());
            let received = self
                .pool
                .as_ref()
                .expect("worker iteration has an active pool")
                .pool()
                .receive(&run_context, remaining);
            let completion = match received {
                WorkerReceive::Completion(completion) => completion,
                WorkerReceive::Timeout => {
                    let batch = self.next_visible;
                    run_context.cancel();
                    run_context.deadline.disarm();
                    self.stop();
                    return Some(Err(LoaderError::Timeout { batch }));
                }
                WorkerReceive::Cancelled => {
                    run_context.deadline.disarm();
                    self.stop();
                    return Some(Err(LoaderError::Cancelled));
                }
                WorkerReceive::Closed => {
                    let batch = self.next_visible;
                    run_context.deadline.disarm();
                    self.stop_poisoned();
                    return Some(Err(LoaderError::ChannelClosed { batch }));
                }
            };
            if completion.generation != self.generation {
                continue;
            }
            match completion.result {
                Err(failure) => {
                    let poisons_pool = matches!(
                        &failure,
                        WorkerFailure::TransformInit(_)
                            | WorkerFailure::WorkerInit(_)
                            | WorkerFailure::Panic
                    );
                    let error =
                        self.map_failure(completion.worker, completion.batch_sequence, failure);
                    if !poisons_pool && let Some(sequence) = completion.batch_sequence {
                        match self._checkpoint.defer_error(sequence, error) {
                            None => continue,
                            Some(error) => {
                                run_context.deadline.disarm();
                                self.stop();
                                return Some(Err(error));
                            }
                        }
                    }
                    run_context.deadline.disarm();
                    if poisons_pool {
                        self.stop_poisoned();
                    } else {
                        self.stop();
                    }
                    return Some(Err(error));
                }
                Ok(WorkerMessage::Batch(batch))
                    if self.ordered && batch.batch_sequence != self.next_visible =>
                {
                    self.completed
                        .insert(batch.batch_sequence, (completion.worker, batch));
                }
                Ok(WorkerMessage::Batch(batch)) => {
                    run_context.deadline.disarm();
                    return Some(self.publish(completion.worker, batch));
                }
                Ok(WorkerMessage::Quiesced) => continue,
            }
        }
    }
}

fn validate_configuration(
    configuration: &BuilderConfiguration,
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
    if configuration
        .timeout
        .is_some_and(|timeout| !Deadline::can_represent(timeout))
    {
        return Err(invalid_configuration(
            "timeout",
            "exceeds the platform monotonic clock range",
        ));
    }
    if explicit.prefetch_factor && configuration.workers == 0 {
        return Err(invalid_configuration(
            "prefetch_factor",
            "requires a positive worker count",
        ));
    }
    if explicit.prefetch_bytes && configuration.workers == 0 {
        return Err(invalid_configuration(
            "prefetch_bytes",
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

fn validate_exact_builder(
    configuration: &BuilderConfiguration,
    explicit: ExplicitArguments,
    identity: &str,
) -> Result<()> {
    validate_configuration(configuration, explicit)?;
    if identity.is_empty() {
        return Err(invalid_configuration(
            "dataset_identity",
            "must not be empty",
        ));
    }
    if configuration.workers != 0 {
        return Err(invalid_configuration(
            "workers",
            "Task 11 exact checkpoints require serial execution",
        ));
    }
    if !configuration.ordered {
        return Err(invalid_configuration(
            "in_order",
            "exact checkpoints require ordered delivery",
        ));
    }
    Ok(())
}

fn validate_distributed_rank(identity: &PlanCheckpointIdentity, rank: usize) -> Result<()> {
    if let Some(distributed) = identity.distributed
        && distributed.rank != rank
    {
        return Err(invalid_configuration(
            "rank",
            "loader task rank must equal the distributed sampler rank",
        ));
    }
    Ok(())
}

fn checkpoint_configuration(
    identity: &PlanCheckpointIdentity,
    builder: &BuilderConfiguration,
    pin_status: PinMemoryStatus,
) -> Result<crate::LoaderConfiguration> {
    let (replicas, distributed_shuffle, distributed_seed, distributed_drop_last) =
        match identity.distributed {
            Some(distributed) => (
                distributed.replicas,
                Some(distributed.shuffle),
                Some(distributed.seed),
                Some(distributed.drop_last),
            ),
            None => (1, None, None, None),
        };
    let pin_request = checkpoint_pin_request(builder.pin_memory)?;
    let pin_status = checkpoint_pin_status(pin_status)?;
    Ok(crate::LoaderConfiguration {
        batch_size: identity.batch_size,
        drop_last: identity.drop_last,
        workers: builder.workers,
        prefetch_factor: builder.prefetch_factor,
        in_order: true,
        loader_seed: builder.loader_seed,
        rank: builder.rank,
        replicas,
        distributed_shuffle,
        distributed_seed,
        distributed_drop_last,
        sampler_kind: identity.sampler_kind.clone(),
        pin_request,
        pin_status,
    })
}

fn checkpoint_pin_request(request: PinRequest) -> Result<CheckpointPinRequest> {
    Ok(match request {
        PinRequest::Disabled => CheckpointPinRequest::Disabled,
        PinRequest::Auto => CheckpointPinRequest::Auto,
        PinRequest::Explicit(Device::Cuda(index)) => CheckpointPinRequest::ExplicitCuda { index },
        PinRequest::Explicit(device) => {
            return Err(invalid_configuration(
                "pin_memory",
                format!("unsupported checkpoint pin device {device:?}"),
            ));
        }
    })
}

fn checkpoint_pin_status(status: PinMemoryStatus) -> Result<CheckpointPinStatus> {
    Ok(match status {
        PinMemoryStatus::Disabled => CheckpointPinStatus::Disabled,
        PinMemoryStatus::DisabledNoAccelerator => CheckpointPinStatus::DisabledNoAccelerator,
        PinMemoryStatus::Enabled(Device::Cuda(index)) => CheckpointPinStatus::Cuda { index },
        PinMemoryStatus::Enabled(device) => {
            return Err(invalid_configuration(
                "pin_memory",
                format!("unsupported effective checkpoint pin device {device:?}"),
            ));
        }
    })
}

fn validate_static_loader_state_envelope<D, S, T, C>(
    state: &LoaderState<D, S, T, C>,
    identity: &str,
    builder: &BuilderConfiguration,
    pin_status: PinMemoryStatus,
    automatic_batching: bool,
    expected_generation: u64,
) -> Result<()> {
    let expected_batch_size = automatic_batching.then_some(builder.batch_size);
    let expected_drop_last = automatic_batching && builder.drop_last;
    let expected_pin_request = checkpoint_pin_request(builder.pin_memory)?;
    let expected_pin_status = checkpoint_pin_status(pin_status)?;

    require_checkpoint_equal(
        "schema_version",
        state.schema_version,
        LOADER_STATE_SCHEMA_VERSION,
    )?;
    require_checkpoint_equal(
        "dataset_identity",
        state.dataset_identity.as_str(),
        identity,
    )?;
    require_checkpoint_equal(
        "iterator_generation",
        state.iterator_generation,
        expected_generation,
    )?;
    require_checkpoint_equal(
        "rng_derivation_version",
        state.rng_derivation_version,
        TASK_RNG_DERIVATION_VERSION,
    )?;
    require_checkpoint_equal(
        "worker_seed_derivation_version",
        state.worker_seed_derivation_version,
        WORKER_SEED_DERIVATION_VERSION,
    )?;
    require_checkpoint_equal(
        "batch_size",
        state.configuration.batch_size,
        expected_batch_size,
    )?;
    require_checkpoint_equal(
        "drop_last",
        state.configuration.drop_last,
        expected_drop_last,
    )?;
    require_checkpoint_equal("workers", state.configuration.workers, builder.workers)?;
    require_checkpoint_equal(
        "prefetch_factor",
        state.configuration.prefetch_factor,
        builder.prefetch_factor,
    )?;
    require_checkpoint_equal("in_order", state.configuration.in_order, builder.ordered)?;
    require_checkpoint_equal(
        "loader_seed",
        state.configuration.loader_seed,
        builder.loader_seed,
    )?;
    require_checkpoint_equal("rank", state.configuration.rank, builder.rank)?;
    require_checkpoint_equal(
        "pin_request",
        &state.configuration.pin_request,
        &expected_pin_request,
    )?;
    require_checkpoint_equal(
        "pin_status",
        &state.configuration.pin_status,
        &expected_pin_status,
    )
}

fn validate_plan_loader_state_envelope<D, S, T, C>(
    state: &LoaderState<D, S, T, C>,
    expected: &crate::LoaderConfiguration,
) -> Result<()> {
    require_checkpoint_equal("replicas", state.configuration.replicas, expected.replicas)?;
    require_checkpoint_equal(
        "distributed_shuffle",
        state.configuration.distributed_shuffle,
        expected.distributed_shuffle,
    )?;
    require_checkpoint_equal(
        "distributed_seed",
        state.configuration.distributed_seed,
        expected.distributed_seed,
    )?;
    require_checkpoint_equal(
        "distributed_drop_last",
        state.configuration.distributed_drop_last,
        expected.distributed_drop_last,
    )?;
    require_checkpoint_equal(
        "sampler_kind",
        state.configuration.sampler_kind.as_str(),
        expected.sampler_kind.as_str(),
    )
}

fn require_checkpoint_equal<T>(field: &'static str, actual: T, expected: T) -> Result<()>
where
    T: PartialEq,
{
    if actual != expected {
        return Err(invalid_configuration(
            field,
            "checkpoint value does not match the loader",
        ));
    }
    Ok(())
}

fn resolve_pin_request(request: PinRequest) -> Result<PinMemoryStatus> {
    match request {
        PinRequest::Disabled => Ok(PinMemoryStatus::Disabled),
        PinRequest::Auto => {
            let capabilities = available_devices();
            if capabilities.cuda && capabilities.cuda_device_count > 0 {
                Ok(PinMemoryStatus::Enabled(Device::Cuda(0)))
            } else {
                Ok(PinMemoryStatus::DisabledNoAccelerator)
            }
        }
        PinRequest::Explicit(Device::Cuda(index)) => {
            let capabilities = available_devices();
            if capabilities.cuda && index < capabilities.cuda_device_count {
                Ok(PinMemoryStatus::Enabled(Device::Cuda(index)))
            } else {
                Err(invalid_configuration(
                    "pin_memory",
                    format!(
                        "CUDA device {index} was requested, but the linked runtime exposes {} available CUDA device(s)",
                        capabilities.cuda_device_count
                    ),
                ))
            }
        }
        PinRequest::Explicit(device) => Err(invalid_configuration(
            "pin_memory",
            format!("only an available CUDA device can back pinned host memory, got {device:?}"),
        )),
    }
}

fn validate_auto_batch_boundary(
    exact_len: Option<usize>,
    batch_size: NonZeroUsize,
    drop_last: bool,
    next_batch: u64,
    next_logical_sample: u64,
) -> Result<()> {
    let batch_size = u64::try_from(batch_size.get()).map_err(|_| {
        invalid_configuration(
            "checkpoint",
            "batch size does not fit the checkpoint schema",
        )
    })?;
    let complete = next_logical_sample / batch_size;
    let aligned = next_logical_sample.is_multiple_of(batch_size);
    let final_short = match exact_len {
        Some(length) => {
            let length = u64::try_from(length).map_err(|_| {
                invalid_configuration(
                    "checkpoint",
                    "sampler length does not fit the checkpoint schema",
                )
            })?;
            !drop_last
                && next_logical_sample == length
                && !aligned
                && next_batch == complete.saturating_add(1)
        }
        None => false,
    };
    if (!aligned || next_batch != complete) && !final_short {
        return Err(invalid_configuration(
            "checkpoint",
            "batch and logical sample cursors are not on a visible batch boundary",
        ));
    }
    if let Some(length) = exact_len {
        let length = u64::try_from(length).map_err(|_| {
            invalid_configuration(
                "checkpoint",
                "sampler length does not fit the checkpoint schema",
            )
        })?;
        if next_logical_sample > length {
            return Err(invalid_configuration(
                "checkpoint",
                "logical sample cursor exceeds the sampler length",
            ));
        }
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
