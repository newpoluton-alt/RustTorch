use std::{cell::RefCell, convert::Infallible};

use rusttorch_core::{Result, RustTorchError};

/// Version of RustTorch's deterministic worker-seed derivation.
pub const WORKER_SEED_DERIVATION_VERSION: u32 = 1;

/// Stable information visible while a loader worker callback is active.
///
/// Unlike PyTorch's process workers, RustTorch map workers share a typed
/// dataset and therefore do not expose an erased dataset through this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerInfo {
    /// Zero-based worker identifier.
    pub id: usize,
    /// Total workers in the pool.
    pub num_workers: usize,
    /// Worker-local initialization seed.
    pub seed: u64,
    /// Distributed rank configured on the loader.
    pub rank: usize,
}

impl WorkerInfo {
    /// Creates validated worker information from an explicit seed.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when `num_workers` is
    /// zero or `id` is outside the worker pool.
    pub fn new(id: usize, num_workers: usize, seed: u64, rank: usize) -> Result<Self> {
        validate_worker(id, num_workers)?;
        Ok(Self {
            id,
            num_workers,
            seed,
            rank,
        })
    }

    /// Creates validated worker information using version-1 seed derivation.
    ///
    /// Worker seeds are consecutive by identifier for one loader seed, rank,
    /// and iterator generation. This RustTorch sequence does not claim
    /// PyTorch generator or Philox identity.
    ///
    /// # Errors
    ///
    /// Returns [`RustTorchError::InvalidConfiguration`] when the worker
    /// identity is outside the configured pool.
    pub fn from_loader_seed(
        id: usize,
        num_workers: usize,
        loader_seed: u64,
        rank: usize,
        iterator_generation: u64,
    ) -> Result<Self> {
        Self::new(
            id,
            num_workers,
            derive_worker_seed(loader_seed, rank, iterator_generation, id),
            rank,
        )
    }
}

/// Initializes one worker after its context becomes active.
pub trait WorkerInit: Send + Sync {
    /// Initialization failure.
    type Error;

    /// Initializes `worker` once for its worker lifecycle.
    fn initialize(&self, worker: &WorkerInfo) -> std::result::Result<(), Self::Error>;
}

/// Adapts a fallible closure into a [`WorkerInit`].
#[derive(Clone, Copy, Debug)]
pub struct FnWorkerInit<F> {
    initialize: F,
}

impl<F> FnWorkerInit<F> {
    /// Creates a closure-backed worker initializer.
    pub fn new(initialize: F) -> Self {
        Self { initialize }
    }
}

impl<E, F> WorkerInit for FnWorkerInit<F>
where
    F: Fn(&WorkerInfo) -> std::result::Result<(), E> + Send + Sync,
{
    type Error = E;

    fn initialize(&self, worker: &WorkerInfo) -> std::result::Result<(), Self::Error> {
        (self.initialize)(worker)
    }
}

/// Default worker initializer that performs no work.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoWorkerInit;

impl WorkerInit for NoWorkerInit {
    type Error = Infallible;

    fn initialize(&self, _worker: &WorkerInfo) -> std::result::Result<(), Self::Error> {
        Ok(())
    }
}

thread_local! {
    static WORKER_INFO: RefCell<Option<WorkerInfo>> = const { RefCell::new(None) };
}

/// Returns worker information for the active callback, if any.
///
/// ```
/// use rusttorch_data::get_worker_info;
///
/// assert_eq!(get_worker_info(), None);
/// ```
pub fn get_worker_info() -> Option<WorkerInfo> {
    WORKER_INFO.with(|current| *current.borrow())
}

/// Runs a callback in a simulated worker scope.
///
/// This hook supports worker-contract tests and the bounded worker engine. Its
/// RAII guard restores nested context during normal return and unwinding.
#[doc(hidden)]
pub fn with_worker_info<T>(worker: WorkerInfo, callback: impl FnOnce() -> T) -> T {
    struct Restore(Option<WorkerInfo>);

    impl Drop for Restore {
        fn drop(&mut self) {
            WORKER_INFO.with(|current| *current.borrow_mut() = self.0);
        }
    }

    let previous = WORKER_INFO.with(|current| current.borrow_mut().replace(worker));
    let _restore = Restore(previous);
    callback()
}

fn validate_worker(id: usize, num_workers: usize) -> Result<()> {
    if num_workers == 0 {
        return Err(invalid_configuration(
            "num_workers",
            "must be greater than zero",
        ));
    }
    if id >= num_workers {
        return Err(invalid_configuration(
            "worker_id",
            "must be less than num_workers",
        ));
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn derive_worker_seed(
    loader_seed: u64,
    rank: usize,
    iterator_generation: u64,
    worker_id: usize,
) -> u64 {
    let rank_domain = (rank as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    let iterator_domain = iterator_generation.wrapping_mul(0xa076_1d64_78bd_642f);
    splitmix64(loader_seed ^ rank_domain ^ iterator_domain).wrapping_add(worker_id as u64)
}

fn invalid_configuration(field: &'static str, reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
