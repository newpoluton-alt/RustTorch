use std::{
    cell::RefCell,
    convert::Infallible,
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, bounded};
use rusttorch_core::{Result, RustTorchError};

/// Version of RustTorch's deterministic worker-seed derivation.
pub const WORKER_SEED_DERIVATION_VERSION: u32 = 1;

struct CancellationState {
    cancelled: AtomicBool,
    wait_lock: Mutex<()>,
    deadline: Mutex<Option<Instant>>,
    wake: Condvar,
    signal: Receiver<()>,
    signal_sender: Mutex<Option<Sender<()>>>,
}

impl CancellationState {
    fn new(deadline: Option<Instant>) -> Arc<Self> {
        let (signal_sender, signal) = bounded(0);
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            wait_lock: Mutex::new(()),
            deadline: Mutex::new(deadline),
            wake: Condvar::new(),
            signal,
            signal_sender: Mutex::new(Some(signal_sender)),
        })
    }
}

/// Cloneable cooperative cancellation notification.
///
/// Cancellation is sticky. Waiting uses a condition variable and queue waits
/// use the same token's disconnection signal, so neither path polls.
#[derive(Clone)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Creates a live token with no deadline.
    pub fn new() -> Self {
        Self {
            state: CancellationState::new(None),
        }
    }

    /// Requests cooperative cancellation and wakes every waiter.
    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            self.state.signal_sender.lock().unwrap().take();
            self.state.wake.notify_all();
        }
    }

    /// Returns whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Blocks until cancellation is requested.
    pub fn wait_cancelled(&self) {
        let mut guard = self.state.wait_lock.lock().unwrap();
        while !self.is_cancelled() {
            guard = self.state.wake.wait(guard).unwrap();
        }
    }

    /// Waits up to `timeout`, returning `true` when cancellation won.
    pub fn wait_cancelled_timeout(&self, timeout: Duration) -> bool {
        if self.is_cancelled() {
            return true;
        }
        let guard = self.state.wait_lock.lock().unwrap();
        let _ = self
            .state
            .wake
            .wait_timeout_while(guard, timeout, |_| !self.is_cancelled())
            .unwrap();
        self.is_cancelled()
    }

    pub(crate) fn signal(&self) -> &Receiver<()> {
        &self.state.signal
    }

    pub(crate) fn paired_deadline(&self) -> Deadline {
        Deadline {
            state: Arc::clone(&self.state),
        }
    }

    pub(crate) fn wait_cancelled_or_deadline(&self, deadline: &Deadline) -> WaitOutcome {
        let mut guard = self.state.wait_lock.lock().unwrap();
        loop {
            if self.is_cancelled() {
                return WaitOutcome::Cancelled;
            }
            match deadline.remaining() {
                Some(remaining) if remaining.is_zero() => return WaitOutcome::DeadlineExpired,
                Some(remaining) => {
                    let (next, _) = self.state.wake.wait_timeout(guard, remaining).unwrap();
                    guard = next;
                }
                None => guard = self.state.wake.wait(guard).unwrap(),
            }
        }
    }
}

/// Cloneable monotonic deadline visible to cooperative worker code.
///
/// Loader-owned deadlines are armed for each blocking `Iterator::next` call
/// and disarmed afterwards. [`Deadline::none`] never expires.
#[derive(Clone)]
pub struct Deadline {
    state: Arc<CancellationState>,
}

impl fmt::Debug for Deadline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Deadline")
            .field("remaining", &self.remaining())
            .finish_non_exhaustive()
    }
}

impl Default for Deadline {
    fn default() -> Self {
        Self::none()
    }
}

impl Deadline {
    /// Creates an unarmed deadline.
    pub fn none() -> Self {
        Self {
            state: CancellationState::new(None),
        }
    }

    /// Creates a deadline expiring after `timeout`.
    pub fn after(timeout: Duration) -> Self {
        Self {
            state: CancellationState::new(Instant::now().checked_add(timeout)),
        }
    }

    /// Returns whether the armed deadline has expired.
    pub fn is_expired(&self) -> bool {
        self.remaining()
            .is_some_and(|remaining| remaining.is_zero())
    }

    /// Returns the remaining duration, or `None` when unarmed.
    pub fn remaining(&self) -> Option<Duration> {
        self.state
            .deadline
            .lock()
            .unwrap()
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    pub(crate) fn arm(&self, timeout: Duration) {
        *self.state.deadline.lock().unwrap() = Instant::now().checked_add(timeout);
        self.state.wake.notify_all();
    }

    pub(crate) fn disarm(&self) {
        *self.state.deadline.lock().unwrap() = None;
        self.state.wake.notify_all();
    }
}

/// Outcome of waiting for cooperative cancellation or a deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitOutcome {
    /// Cancellation was requested.
    Cancelled,
    /// The monotonic deadline expired.
    DeadlineExpired,
}

/// Error returned by [`WorkerContext::check`] after cancellation or expiry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("loader work was cancelled or its deadline expired")]
pub struct LoaderCancelled;

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

/// Cancellation and deadline state passed to worker lifecycle callbacks.
///
/// For persistent pools this context belongs to the pool lifetime: its token
/// is cancelled only when the owner shuts the pool down. Dataset fetches and
/// task transforms receive a separate, fresh context for each iterator
/// generation.
#[derive(Clone, Debug)]
pub struct WorkerContext {
    /// Stable worker identity and initialization seed.
    pub info: WorkerInfo,
    /// Cooperative cancellation token.
    pub cancellation: CancellationToken,
    /// Dynamic monotonic deadline.
    pub deadline: Deadline,
}

impl WorkerContext {
    /// Creates a context from explicit worker information and controls.
    pub fn new(info: WorkerInfo, cancellation: CancellationToken, deadline: Deadline) -> Self {
        Self {
            info,
            cancellation,
            deadline,
        }
    }

    /// Returns an error when cancelled or expired.
    pub fn check(&self) -> std::result::Result<(), LoaderCancelled> {
        if self.cancellation.is_cancelled() || self.deadline.is_expired() {
            Err(LoaderCancelled)
        } else {
            Ok(())
        }
    }

    /// Blocks until cancellation or deadline expiry.
    pub fn wait_cancelled_or_deadline(&self) -> WaitOutcome {
        self.cancellation.wait_cancelled_or_deadline(&self.deadline)
    }
}

/// Initializes one worker after its context becomes active.
///
/// The default serial loader imposes no thread-safety bound; positive worker
/// execution requires a shared `Send + Sync` initializer.
pub trait WorkerInit {
    /// Initialization failure.
    type Error;

    /// Initializes `worker` once for its worker lifecycle.
    fn initialize(&self, worker: &WorkerContext) -> std::result::Result<(), Self::Error>;
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
    F: Fn(&WorkerContext) -> std::result::Result<(), E>,
{
    type Error = E;

    fn initialize(&self, worker: &WorkerContext) -> std::result::Result<(), Self::Error> {
        (self.initialize)(worker)
    }
}

/// Default worker initializer that performs no work.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoWorkerInit;

impl WorkerInit for NoWorkerInit {
    type Error = Infallible;

    fn initialize(&self, _worker: &WorkerContext) -> std::result::Result<(), Self::Error> {
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
