use std::convert::Infallible;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::WorkerInfo;

/// Version of RustTorch's deterministic task-RNG seed derivation.
pub const TASK_RNG_DERIVATION_VERSION: u32 = 1;

/// Logical identity used to seed one transformation occurrence.
///
/// Worker assignment is deliberately absent, so random transforms are stable
/// when scheduling or worker counts change. The resulting ChaCha12 sequence is
/// a RustTorch contract and does not claim PyTorch Philox sequence identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskContext {
    /// Loader-level seed.
    pub loader_seed: u64,
    /// Current data epoch.
    pub epoch: u64,
    /// Distributed rank.
    pub rank: usize,
    /// Monotonic logical sample occurrence in this iterator.
    pub logical_sample: u64,
    /// Transform stage, starting at zero.
    pub stage: u32,
}

impl TaskContext {
    /// Returns the stable version-1 seed derived from this logical identity.
    pub fn deterministic_seed(&self) -> u64 {
        derive_task_seed(self)
    }

    /// Creates task-local ChaCha12 state without changing LibTorch's RNG.
    pub fn rng(&self) -> ChaCha12Rng {
        ChaCha12Rng::seed_from_u64(self.deterministic_seed())
    }
}

/// A fallible sample transformation with task-local deterministic context.
pub trait Transform<Input> {
    /// Output passed to collation.
    type Output;
    /// Transformation failure.
    type Error;

    /// Transforms one logical sample occurrence.
    fn transform(
        &mut self,
        input: Input,
        context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error>;
}

/// Adapts a fallible closure into a [`Transform`].
///
/// ```
/// use std::convert::Infallible;
/// use rusttorch_data::{FnTransform, TaskContext, Transform};
///
/// let context = TaskContext {
///     loader_seed: 42,
///     epoch: 3,
///     rank: 1,
///     logical_sample: 99,
///     stage: 7,
/// };
/// let mut double = FnTransform::new(|value: i64, _: &TaskContext| {
///     Ok::<_, Infallible>(value * 2)
/// });
/// assert_eq!(double.transform(3, &context), Ok(6));
/// assert_eq!(context.deterministic_seed(), 0x1d7d_73dc_f6e9_4f2d);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FnTransform<F> {
    transform: F,
}

impl<F> FnTransform<F> {
    /// Creates a closure-backed transform.
    pub fn new(transform: F) -> Self {
        Self { transform }
    }
}

impl<Input, Output, E, F> Transform<Input> for FnTransform<F>
where
    F: FnMut(Input, &TaskContext) -> std::result::Result<Output, E>,
{
    type Output = Output;
    type Error = E;

    fn transform(
        &mut self,
        input: Input,
        context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error> {
        (self.transform)(input, context)
    }
}

/// Constructs one transform for a serial iterator or worker.
pub trait TransformFactory<Input>: Send + Sync {
    /// Transform instance created by this factory.
    type Transform: Transform<Input>;
    /// Construction failure.
    type Error;

    /// Creates a transform for `worker`, or for the serial path when `None`.
    fn create(
        &self,
        worker: Option<&WorkerInfo>,
    ) -> std::result::Result<Self::Transform, Self::Error>;
}

/// Adapts a fallible closure into a [`TransformFactory`].
#[derive(Clone, Copy, Debug)]
pub struct FnTransformFactory<F> {
    create: F,
}

impl<F> FnTransformFactory<F> {
    /// Creates a closure-backed factory.
    pub fn new(create: F) -> Self {
        Self { create }
    }
}

impl<Input, T, E, F> TransformFactory<Input> for FnTransformFactory<F>
where
    T: Transform<Input>,
    F: Fn(Option<&WorkerInfo>) -> std::result::Result<T, E> + Send + Sync,
{
    type Transform = T;
    type Error = E;

    fn create(
        &self,
        worker: Option<&WorkerInfo>,
    ) -> std::result::Result<Self::Transform, Self::Error> {
        (self.create)(worker)
    }
}

/// Factory used by [`crate::DataLoaderBuilder::transform`] to clone a template.
#[derive(Clone, Copy, Debug)]
pub struct CloneTransformFactory<T> {
    transform: T,
}

impl<T> CloneTransformFactory<T> {
    pub(crate) fn new(transform: T) -> Self {
        Self { transform }
    }
}

impl<Input, T> TransformFactory<Input> for CloneTransformFactory<T>
where
    T: Transform<Input> + Clone + Send + Sync,
{
    type Transform = T;
    type Error = Infallible;

    fn create(
        &self,
        _worker: Option<&WorkerInfo>,
    ) -> std::result::Result<Self::Transform, Self::Error> {
        Ok(self.transform.clone())
    }
}

/// Identity sample transform used by the default factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityTransform;

impl<Input> Transform<Input> for IdentityTransform {
    type Output = Input;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: Input,
        _context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error> {
        Ok(input)
    }
}

/// Default factory that creates infallible identity transforms.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityTransformFactory;

impl<Input> TransformFactory<Input> for IdentityTransformFactory {
    type Transform = IdentityTransform;
    type Error = Infallible;

    fn create(
        &self,
        _worker: Option<&WorkerInfo>,
    ) -> std::result::Result<Self::Transform, Self::Error> {
        Ok(IdentityTransform)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_task_seed(context: &TaskContext) -> u64 {
    let fields = [
        context.loader_seed,
        context.epoch,
        context.rank as u64,
        context.logical_sample,
        u64::from(context.stage),
    ];
    fields
        .into_iter()
        .enumerate()
        .fold(0x5255_5354_544f_5243, |state, (index, value)| {
            let domain = (index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            splitmix64(state ^ value.wrapping_add(domain))
        })
}
