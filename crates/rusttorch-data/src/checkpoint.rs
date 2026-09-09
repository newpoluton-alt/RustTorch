use std::{error::Error, fmt};

use rusttorch_core::{Result, RustTorchError};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{TaskContext, Transform, TransformFactory, WorkerContext};

/// Replay-safe source with self-contained state captured before each read.
///
/// Validation is read-only; applying accepted state must not fail or panic.
/// Records must have strictly increasing per-shard global sequence IDs.
pub trait CheckpointableSource:
    Iterator<Item = std::result::Result<crate::WorkerRecord<Self::Sample>, Self::Error>>
{
    /// Owned decoded sample.
    type Sample;
    /// Owned state, independent of future decoder progress.
    type State: Clone + Serialize + DeserializeOwned;
    /// Preserved source error.
    type Error;
    /// Returns the failed read's stable global position without changing state.
    ///
    /// Called after `next()` returns an error. This position must obey the
    /// shard's strictly increasing sequence contract and repeat after rollback.
    fn error_sequence(&self, error: &Self::Error) -> crate::SequenceId;
    /// Captures the boundary before the next read attempt.
    fn snapshot(&self) -> Self::State;
    /// Validates without changing the source.
    fn validate_snapshot(&self, state: &Self::State) -> std::result::Result<(), Self::Error>;
    /// Applies state previously accepted by validation.
    fn restore_validated(&mut self, state: &Self::State);
    /// Replaces the run cancellation/deadline context before resumed reads.
    ///
    /// Sources retaining a context must override this method and replace it.
    /// Logical worker generation and seed remain unchanged. This must not fail.
    fn set_run_context(&mut self, _context: WorkerContext) {}
}

/// Stable factory identity for explicitly checkpointable stream sources.
///
/// An ordinary factory cannot acquire exact replay just by setting an identity:
///
/// ```compile_fail
/// use rusttorch_data::*;
/// use std::convert::Infallible;
/// struct Factory;
/// impl WorkerSourceFactory for Factory {
///     type Sample = usize;
///     type Error = Infallible;
///     type Source = std::iter::Empty<Result<WorkerRecord<usize>, Infallible>>;
///     fn create(&self, _: WorkerContext) -> Result<Self::Source, Infallible> { Ok(std::iter::empty()) }
/// }
/// StreamDataLoaderBuilder::new(Factory).collate(VecCollate)
///     .checkpointable("ordinary").build().unwrap();
/// ```
///
/// Opaque transforms also need an explicit `WorkerCheckpoint` implementation or
/// an explicit `StatelessWorker`/`TransactionalWorker` adapter:
///
/// ```compile_fail
/// use rusttorch_data::*;
/// use std::convert::Infallible;
/// struct Source;
/// impl Iterator for Source {
///     type Item = Result<WorkerRecord<usize>, Infallible>;
///     fn next(&mut self) -> Option<Self::Item> { None }
/// }
/// impl CheckpointableSource for Source {
///     type Sample = usize;
///     type Error = Infallible;
///     type State = ();
///     fn error_sequence(&self, error: &Infallible) -> SequenceId { match *error {} }
///     fn snapshot(&self) {}
///     fn validate_snapshot(&self, _: &()) -> Result<(), Infallible> { Ok(()) }
///     fn restore_validated(&mut self, _: &()) {}
/// }
/// struct Factory;
/// impl WorkerSourceFactory for Factory {
///     type Sample = usize;
///     type Error = Infallible;
///     type Source = Source;
///     fn create(&self, _: WorkerContext) -> Result<Source, Infallible> { Ok(Source) }
/// }
/// impl CheckpointSourceFactory for Factory { const CHECKPOINT_KIND: &'static str = "empty.v1"; }
/// let transform = FnTransform::new(|sample: usize, _: &TaskContext| Ok::<_, Infallible>(sample));
/// StreamDataLoaderBuilder::new(Factory).transform(transform).collate(VecCollate)
///     .checkpointable("opaque").build().unwrap();
/// ```
pub trait CheckpointSourceFactory: crate::WorkerSourceFactory {
    /// Stable source kind and state-format version, for example `records.v1`.
    const CHECKPOINT_KIND: &'static str;
}

/// Exact streaming configuration, with no sampler state.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamCheckpointConfiguration {
    /// Positive shard count.
    pub workers: usize,
    /// Maximum unpublished records per shard.
    pub prefetch_factor: usize,
    /// Coordinator batch size.
    pub batch_size: usize,
    /// Global short-tail policy.
    pub drop_last: bool,
    /// Task randomness seed.
    pub loader_seed: u64,
    /// Distributed task rank.
    pub rank: usize,
    /// Requested pinning behavior.
    pub pin_request: CheckpointPinRequest,
    /// Effective pinning behavior.
    pub pin_status: CheckpointPinStatus,
}

/// Paired source and transform boundary for one worker shard.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreamLaneState<S, T> {
    /// Unique worker index in canonical order.
    pub id: usize,
    /// Source state preceding its first unconsumed attempt.
    pub source: S,
    /// Transform state preceding the same attempt.
    pub transform: T,
}

/// Versioned exact stream state at the next consumer-visible batch.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreamLoaderState<S, T = (), C = ()> {
    /// Stream schema version; currently one.
    pub schema_version: u32,
    /// Caller identity for exact source contents.
    pub source_identity: String,
    /// Stable factory kind/version.
    pub factory_kind: String,
    /// Task epoch.
    pub epoch: u64,
    /// Logical source and transform initialization generation.
    pub source_generation: u64,
    /// Transport generation, advanced by each checkpoint barrier.
    pub transport_generation: u64,
    /// Next visible batch index.
    pub next_batch: u64,
    /// Next expected global sequence ID.
    pub next_sequence: u64,
    /// Whether the last visible batch was the final incomplete global tail.
    pub short_tail: bool,
    /// Exact global source length when supplied by the factory.
    pub source_length: Option<usize>,
    /// Task RNG derivation version.
    pub rng_derivation_version: u32,
    /// Worker seed derivation version.
    pub worker_seed_derivation_version: u32,
    /// Exact runtime settings.
    pub configuration: StreamCheckpointConfiguration,
    /// Canonically ordered worker states; never contains decoded samples.
    pub lanes: Vec<StreamLaneState<S, T>>,
    /// Coordinator collation state.
    pub coordinator: C,
}

/// Typed stream construction or transactional resume failure.
#[derive(Debug, thiserror::Error)]
pub enum StreamCheckpointBuildError<S, F> {
    /// Static configuration rejection.
    #[error(transparent)]
    Configuration(#[from] RustTorchError),
    /// Source creation or read-only state validation failed.
    #[error("stream source on worker {worker} failed: {source}")]
    Source {
        /// Worker lane.
        worker: usize,
        /// Original source error.
        #[source]
        source: S,
    },
    /// Transform creation failed.
    #[error("stream transform initialization on worker {worker} failed: {source}")]
    TransformFactory {
        /// Worker lane.
        worker: usize,
        /// Original factory error.
        #[source]
        source: F,
    },
    /// Transform state validation failed.
    #[error("stream transform validation on worker {worker} failed: {source}")]
    TransformState {
        /// Worker lane.
        worker: usize,
        /// Original validation error.
        #[source]
        source: RustTorchError,
    },
    /// Initialization or checkpoint callback panicked.
    #[error("stream checkpoint worker {worker} panicked")]
    WorkerPanic {
        /// Worker lane.
        worker: usize,
    },
}

/// Schema version emitted by serial map-loader checkpoints.
pub const LOADER_STATE_SCHEMA_VERSION: u32 = 1;

/// Pinning request recorded in a loader checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CheckpointPinRequest {
    /// Pinning was not requested.
    Disabled,
    /// Pinning was requested only when a CUDA accelerator is available.
    Auto,
    /// Pinning was requested for one CUDA device.
    ExplicitCuda {
        /// CUDA device index.
        index: usize,
    },
}

/// Effective pinning behavior recorded in a loader checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CheckpointPinStatus {
    /// Pinning was not requested.
    Disabled,
    /// Automatic pinning found no supported accelerator.
    DisabledNoAccelerator,
    /// Batches are pinned for one CUDA device.
    Cuda {
        /// CUDA device index.
        index: usize,
    },
}

/// Loader settings that must match when a serial checkpoint is resumed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LoaderConfiguration {
    /// Automatic batch size, or `None` for explicit/no batching.
    pub batch_size: Option<usize>,
    /// Whether an incomplete automatic batch is omitted.
    pub drop_last: bool,
    /// Exact worker count; zero for serial execution.
    pub workers: usize,
    /// Effective worker prefetch factor, or `None` for serial execution.
    pub prefetch_factor: Option<usize>,
    /// Whether results follow sampler order.
    pub in_order: bool,
    /// Loader seed used by task-local random transforms.
    pub loader_seed: u64,
    /// Distributed task rank.
    pub rank: usize,
    /// Distributed replica count, or one for an unsharded sampler.
    pub replicas: usize,
    /// Distributed shuffle policy when applicable.
    pub distributed_shuffle: Option<bool>,
    /// Distributed sampler seed when applicable.
    pub distributed_seed: Option<u64>,
    /// Distributed tail policy when applicable.
    pub distributed_drop_last: Option<bool>,
    /// Stable RustTorch sampler/plan kind.
    pub sampler_kind: String,
    /// Requested pinning policy.
    pub pin_request: CheckpointPinRequest,
    /// Effective pinning status visible to consumers.
    pub pin_status: CheckpointPinStatus,
}

/// Versioned state for the next batch visible from a map loader.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoaderState<D, S, T = (), C = ()> {
    /// State schema version.
    pub schema_version: u32,
    /// Caller-defined identity for the exact dataset contents.
    pub dataset_identity: String,
    /// Active sampler epoch.
    pub epoch: u64,
    /// Transport generation. Serial state is always zero.
    pub iterator_generation: u64,
    /// Next consumer-visible batch number.
    pub next_batch: u64,
    /// Number of logical sample occurrences already consumed.
    pub next_logical_sample: u64,
    /// Dataset component state.
    pub dataset: D,
    /// Active sampler or batch-source cursor state.
    pub sampler: S,
    /// Serial transform state or a versioned worker transform envelope.
    pub transform: T,
    /// Effective coordinator collator/converter state.
    pub collate: C,
    /// Task-RNG derivation version.
    pub rng_derivation_version: u32,
    /// Worker-seed derivation version.
    pub worker_seed_derivation_version: u32,
    /// Exact loader configuration identity.
    pub configuration: LoaderConfiguration,
}

/// A component whose state can be validated before it is restored.
///
/// Validation must be read-only. `load_validated` must not fail or panic after
/// the same component accepted the supplied state; violating either rule
/// breaks the resume transaction contract.
pub trait Checkpointable {
    /// Owned serializable component state.
    type State: Clone + Serialize + DeserializeOwned;

    /// Captures the current component state.
    fn save_state(&self) -> Self::State;

    /// Checks state compatibility without mutating the component.
    fn validate_state(&self, state: &Self::State) -> Result<()>;

    /// Restores state previously accepted by [`Self::validate_state`].
    fn load_validated(&mut self, state: &Self::State);
}

/// Marker for a map dataset whose fixed-identity fetches are replay-safe.
pub trait ReplaySafeDataset: crate::Dataset {}

/// Dataset state used by exact serial loader checkpoints.
///
/// Validation must be read-only, and restore must be infallible after matching
/// validation succeeds.
pub trait DatasetCheckpoint: crate::Dataset {
    /// Owned serializable dataset state.
    type State: Clone + Serialize + DeserializeOwned;

    /// Captures dataset state at the visible boundary.
    fn snapshot_dataset(&self) -> Self::State;

    /// Checks state compatibility without mutation.
    fn validate_dataset_state(&self, state: &Self::State) -> Result<()>;

    /// Restores state accepted by [`Self::validate_dataset_state`].
    fn restore_dataset_validated(&mut self, state: &Self::State);
}

/// Marker for a component with no mutable checkpoint state.
pub trait Stateless {}

/// Transactional state contract for a serial component.
///
/// Validation must be read-only, and restore must be infallible after matching
/// validation succeeds.
pub trait TransactionalCheckpoint {
    /// Owned serializable component state.
    type State: Clone + Serialize + DeserializeOwned;

    /// Captures current component state.
    fn snapshot(&self) -> Self::State;

    /// Checks state compatibility without mutation.
    fn validate_snapshot(&self, state: &Self::State) -> Result<()>;

    /// Restores state accepted by [`Self::validate_snapshot`].
    fn restore_validated(&mut self, state: &Self::State);
}

/// Checkpoint contract for one transform instance.
///
/// Serial checkpoints capture one instance; worker barriers restore each
/// deterministic lane to its first unconsumed task.
pub trait WorkerCheckpoint {
    /// Owned serializable transform state.
    type State: Clone + Serialize + DeserializeOwned;

    /// Captures current transform state.
    fn snapshot(&self) -> Self::State;

    /// Checks state compatibility without mutation.
    fn validate_snapshot(&self, state: &Self::State) -> Result<()>;

    /// Restores state accepted by [`Self::validate_snapshot`].
    fn restore_validated(&mut self, state: &Self::State);
}

/// Explicit unit-state adapter for a transform the caller asserts is stateless.
///
/// Construction is the opt-in assertion: mutable behavior that affects future
/// outputs violates this adapter's exact-resume contract. This makes opaque
/// closure transforms usable without requiring downstream orphan impls.
#[derive(Clone, Copy, Debug, Default)]
pub struct StatelessWorker<T> {
    inner: T,
}

impl<T> StatelessWorker<T> {
    /// Explicitly asserts that a transform has no state affecting replay.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Returns the wrapped transform.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<Input, T> Transform<Input> for StatelessWorker<T>
where
    T: Transform<Input>,
{
    type Output = T::Output;
    type Error = T::Error;

    fn transform(
        &mut self,
        input: Input,
        context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error> {
        self.inner.transform(input, context)
    }
}

impl<T> WorkerCheckpoint for StatelessWorker<T> {
    type State = ();

    fn snapshot(&self) -> Self::State {}

    fn validate_snapshot(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }

    fn restore_validated(&mut self, _state: &Self::State) {}
}

/// Explicit transactional adapter for a stateful transform.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransactionalWorker<T> {
    inner: T,
}

impl<T> TransactionalWorker<T> {
    /// Wraps a transform with transactional checkpoint support.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Returns the wrapped transform.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<Input, T> Transform<Input> for TransactionalWorker<T>
where
    T: Transform<Input> + TransactionalCheckpoint,
{
    type Output = T::Output;
    type Error = T::Error;

    fn transform(
        &mut self,
        input: Input,
        context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error> {
        self.inner.transform(input, context)
    }
}

impl<T> WorkerCheckpoint for TransactionalWorker<T>
where
    T: TransactionalCheckpoint,
{
    type State = T::State;

    fn snapshot(&self) -> Self::State {
        self.inner.snapshot()
    }

    fn validate_snapshot(&self, state: &Self::State) -> Result<()> {
        self.inner.validate_snapshot(state)
    }

    fn restore_validated(&mut self, state: &Self::State) {
        self.inner.restore_validated(state);
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct StatelessTransformFactory<F>(pub(crate) F);

impl<Input, F> TransformFactory<Input> for StatelessTransformFactory<F>
where
    F: TransformFactory<Input>,
{
    type Transform = StatelessWorker<F::Transform>;
    type Error = F::Error;

    fn create(
        &self,
        worker: Option<&WorkerContext>,
    ) -> std::result::Result<Self::Transform, Self::Error> {
        self.0.create(worker).map(StatelessWorker::new)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct TransactionalTransformFactory<F>(pub(crate) F);

impl<Input, F> TransformFactory<Input> for TransactionalTransformFactory<F>
where
    F: TransformFactory<Input>,
    F::Transform: TransactionalCheckpoint,
{
    type Transform = TransactionalWorker<F::Transform>;
    type Error = F::Error;

    fn create(
        &self,
        worker: Option<&WorkerContext>,
    ) -> std::result::Result<Self::Transform, Self::Error> {
        self.0.create(worker).map(TransactionalWorker::new)
    }
}

/// Resume-build error preserving either a RustTorch validation error or the
/// transform factory's original error value.
#[derive(Debug)]
pub enum CheckpointBuildError<F> {
    /// Loader configuration or component validation failed.
    Configuration(RustTorchError),
    /// Constructing the one retained serial transform failed.
    TransformFactory(F),
}

impl<F> From<RustTorchError> for CheckpointBuildError<F> {
    fn from(error: RustTorchError) -> Self {
        Self::Configuration(error)
    }
}

impl<F> fmt::Display for CheckpointBuildError<F>
where
    F: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => error.fmt(formatter),
            Self::TransformFactory(error) => {
                write!(
                    formatter,
                    "transform initialization failed while resuming: {error}"
                )
            }
        }
    }
}

impl<F> Error for CheckpointBuildError<F>
where
    F: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::TransformFactory(error) => Some(error),
        }
    }
}

/// Default builder/loader state with exact checkpointing disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct CheckpointDisabled;

/// Builder state after a dataset identity enables a fresh exact checkpoint.
///
/// Exact checkpointing requires explicit replay evidence; an arbitrary
/// dataset is not enabled merely by assigning an identity:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{DataLoader, Dataset, VecCollate};
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// let _ = DataLoader::builder(Rows)
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build();
/// ```
///
/// Ordinary [`crate::TensorDataset`] rows share storage and are therefore not
/// exact-checkpoint capable until [`crate::TensorDataset::into_replay_safe`]
/// succeeds:
///
/// ```compile_fail
/// use rusttorch_core::Tensor;
/// use rusttorch_data::{DataLoader, TensorDataset, VecCollate};
/// let dataset = TensorDataset::new(vec![Tensor::from_slice(&[1_i64])]).unwrap();
/// let _ = DataLoader::builder(dataset)
///     .collate(VecCollate)
///     .dataset_identity("tensor".to_owned())
///     .build();
/// ```
///
/// An opaque closure transform has no restorable state contract unless the
/// caller explicitly selects a stateless or transactional adapter:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{
///     DataLoader, Dataset, FnTransform, ReplaySafeDataset, ReplaySafeMap,
///     TaskContext, VecCollate,
/// };
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let transform = FnTransform::new(|value, _: &TaskContext| Ok::<_, Infallible>(value));
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .transform(transform)
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build();
/// ```
///
/// Closure-backed collation is likewise opaque without an explicit
/// [`Checkpointable`] implementation:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{
///     DataLoader, Dataset, FnCollate, ReplaySafeDataset, ReplaySafeMap,
/// };
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .collate(FnCollate::new(|values| Ok::<_, Infallible>(values)))
///     .dataset_identity("rows".to_owned())
///     .build();
/// ```
///
/// Closure-backed sampler and explicit-batch sources do not expose stable
/// configuration/cursor state:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{
///     DataLoader, Dataset, FnSampler, ReplaySafeDataset, ReplaySafeMap,
///     VecCollate,
/// };
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .sampler(FnSampler::new(Some(1), |_| [0].into_iter()))
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build();
/// ```
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{
///     DataLoader, Dataset, FnBatchSource, ReplaySafeDataset, ReplaySafeMap,
///     VecCollate,
/// };
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .batch_sampler(FnBatchSource::new(Some(1), |_| vec![vec![0]].into_iter()))
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build();
/// ```
///
/// Worker replay supports explicit immutable datasets and checkpointable
/// transforms. Zero workers use the same tagged envelope:
///
/// ```
/// use std::convert::Infallible;
/// use rusttorch_data::{
///     DataLoader, Dataset, ReplaySafeDataset, ReplaySafeMap, VecCollate,
/// };
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let mut loader = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .workers(0)
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build().unwrap();
/// let state = loader.iter().checkpoint().unwrap();
/// let mut resumed = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .workers(0).collate(VecCollate).dataset_identity("rows".to_owned())
///     .resume_from(state).build().unwrap();
/// assert_eq!(resumed.iter().next().unwrap().unwrap(), vec![1]);
/// ```
#[derive(Clone, Debug)]
pub struct CheckpointFresh {
    pub(crate) identity: String,
}

/// Builder state carrying a typed checkpoint selected for resume.
#[derive(Clone, Debug)]
pub struct CheckpointResume<S> {
    pub(crate) identity: String,
    pub(crate) state: S,
}

/// Retained first serial cursor and transform for a resumed loader.
#[doc(hidden)]
pub struct PendingSerial<I, T> {
    pub(crate) batches: I,
    pub(crate) transform: T,
    pub(crate) next_batch: u64,
    pub(crate) next_logical_sample: u64,
}

/// Built-loader state enabling exact serial checkpoints.
#[doc(hidden)]
pub struct CheckpointActive<I, T> {
    pub(crate) identity: String,
    pub(crate) configuration: LoaderConfiguration,
    pub(crate) pending: Option<PendingSerial<I, T>>,
}

/// Active-iterator state at an exact serial checkpoint boundary.
#[doc(hidden)]
pub struct CheckpointIteration {
    pub(crate) identity: String,
    pub(crate) configuration: LoaderConfiguration,
    pub(crate) boundary_valid: bool,
}

/// One deterministic worker lane's committed transform state.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkerLaneState<S> {
    /// Zero-based lane identity.
    pub id: usize,
    /// Transform state before this lane's next unconsumed task.
    pub state: S,
}

/// Explicit zero-worker or positive-worker transform state.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum WorkerTransformLanes<S> {
    /// One transform running on the calling thread.
    Serial(S),
    /// Transform states in deterministic lane order.
    Workers(Vec<WorkerLaneState<S>>),
}

/// Versioned worker replay envelope within schema-v1 loader state.
///
/// An opaque transform cannot enable worker checkpointing without an explicit
/// adapter, even when its closure appears deterministic:
///
/// ```compile_fail
/// use std::convert::Infallible;
/// use rusttorch_data::{DataLoader, Dataset, FnTransform, ReplaySafeDataset, ReplaySafeMap, TaskContext, VecCollate};
/// struct Rows;
/// impl Dataset for Rows {
///     type Sample = i64;
///     type Error = Infallible;
///     fn len(&self) -> usize { 1 }
///     fn get(&self, _: usize) -> Result<i64, Infallible> { Ok(1) }
/// }
/// impl ReplaySafeDataset for Rows {}
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .workers(2).collate(VecCollate)
///     .transform(FnTransform::new(|x: i64, _: &TaskContext| Ok::<_, Infallible>(x)))
///     .dataset_identity("rows".to_owned()).build();
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkerTransformState<S> {
    /// Worker envelope version, currently one.
    pub schema_version: u32,
    /// Exact worker count.
    pub workers: usize,
    /// Effective prefetch factor; absent for zero workers.
    pub prefetch_factor: Option<usize>,
    /// Original transform factory initialization identity.
    pub factory_generation: u64,
    /// Current transport generation used to reject stale messages.
    pub run_generation: u64,
    /// Explicit lane identities and states.
    pub lanes: WorkerTransformLanes<S>,
}

/// Retained worker checkpoint components for the next exact iteration.
#[doc(hidden)]
pub struct WorkerCheckpointActive<I, T, S> {
    pub(crate) identity: String,
    pub(crate) configuration: LoaderConfiguration,
    pub(crate) batches: Option<I>,
    pub(crate) serial: Option<T>,
    pub(crate) barrier: Option<crate::worker_checkpoint::Barrier<S>>,
    pub(crate) next_batch: u64,
    pub(crate) next_logical_sample: u64,
    pub(crate) factory_generation: u64,
    pub(crate) run_generation: u64,
}

/// Committed coordinator and worker boundary for one exact iteration.
#[doc(hidden)]
pub struct WorkerCheckpointIteration<S, E> {
    pub(crate) identity: String,
    pub(crate) configuration: LoaderConfiguration,
    pub(crate) barrier: Option<crate::worker_checkpoint::Barrier<S>>,
    pub(crate) logical_ends: std::collections::BTreeMap<u64, u64>,
    pub(crate) errors: std::collections::BTreeMap<u64, E>,
    pub(crate) next_logical_sample: u64,
    pub(crate) factory_generation: u64,
    pub(crate) boundary_valid: bool,
}

impl Checkpointable for () {
    type State = ();

    fn save_state(&self) -> Self::State {}

    fn validate_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }

    fn load_validated(&mut self, _state: &Self::State) {}
}
