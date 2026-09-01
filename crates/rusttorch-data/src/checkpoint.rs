use std::{error::Error, fmt};

use rusttorch_core::{Result, RustTorchError};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{TaskContext, Transform, TransformFactory, WorkerContext};

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
    /// Worker count. Task 11 exact checkpoints require zero.
    pub workers: usize,
    /// Worker prefetch factor. Task 11 exact checkpoints require `None`.
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

/// Versioned state for the next batch visible from a serial map loader.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoaderState<D, S, T = (), C = ()> {
    /// State schema version.
    pub schema_version: u32,
    /// Caller-defined identity for the exact dataset contents.
    pub dataset_identity: String,
    /// Active sampler epoch.
    pub epoch: u64,
    /// Iterator generation. Serial Task 11 state is always zero.
    pub iterator_generation: u64,
    /// Next consumer-visible batch number.
    pub next_batch: u64,
    /// Number of logical sample occurrences already consumed.
    pub next_logical_sample: u64,
    /// Dataset component state.
    pub dataset: D,
    /// Active sampler or batch-source cursor state.
    pub sampler: S,
    /// Serial transform state.
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
/// Task 11 uses this contract only on the serial transform. Later worker
/// checkpoint barriers can reuse the same explicit state boundary.
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
/// Selecting the worker execution type, even with zero workers, deliberately
/// defers exact resume to the prefetched checkpoint barrier:
///
/// ```compile_fail
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
/// let _ = DataLoader::builder(ReplaySafeMap::new(Rows))
///     .workers(0)
///     .collate(VecCollate)
///     .dataset_identity("rows".to_owned())
///     .build();
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

impl Checkpointable for () {
    type State = ();

    fn save_state(&self) -> Self::State {}

    fn validate_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }

    fn load_validated(&mut self, _state: &Self::State) {}
}
