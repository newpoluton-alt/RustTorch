//! Explicitly sharded positive-worker stream loading.
//!
//! Each worker owns the iterator returned by [`WorkerSourceFactory::create`];
//! no shared iterator lock is involved. Ordered sources assign one contiguous
//! global [`SequenceId`] range starting at zero. Unordered sources may omit
//! sequence IDs, but every [`LogicalSampleId`] must remain stable and unique
//! within the caller's generation so task randomness has a scheduling-neutral
//! identity. Records are merged before coordinator batching, so `drop_last`
//! drops at most one global tail rather than one tail per shard.
//!
//! Ordered reassembly retains at most `workers * prefetch_factor` records. If
//! that entire validated window contains higher IDs while the next ID is
//! absent, no worker credit remains with which a shard could advance, so the
//! iterator reports a protocol error instead of waiting for an unreachable end
//! marker. A source that emits a lower ID after more than this global window of
//! higher IDs must increase the factor or shard the lower ID onto a worker that
//! keeps an independent credit available. A worker already producing the
//! missing low ID holds its credit outside reassembly and is allowed to finish.
//! The ordered window is one loader-owned flat slot vector reserved and
//! aggregate-checked before source callbacks, then reused across generations.
//! Lookup is linear in the deliberately bounded window; increase the factor
//! only when wider source disorder justifies its memory and scan cost.
//!
//! With `prefetch_bytes` enabled, workers measure final post-transform records
//! and carry cancellation-safe byte permits through the result queue and this
//! reassembly window. Each ordered shard must then emit strictly increasing
//! sequence IDs. One bounded front-waiter slot per shard distinguishes a slow
//! expected record from a globally missing ID without weakening the byte cap.
//! The coordinator's active item-bounded collation batch is outside the byte
//! budget. Recursive pinning happens only after successful collation.
//!
//! ```
//! use std::convert::Infallible;
//! use rusttorch_data::{
//!     LogicalSampleId, SequenceId, StreamDataLoaderBuilder, VecCollate,
//!     WorkerContext, WorkerRecord, WorkerSourceFactory,
//! };
//!
//! struct ModuloShards(usize);
//!
//! impl WorkerSourceFactory for ModuloShards {
//!     type Sample = usize;
//!     type Error = Infallible;
//!     type Source = std::vec::IntoIter<Result<WorkerRecord<usize>, Infallible>>;
//!
//!     fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
//!         Ok((worker.info.id..self.0)
//!             .step_by(worker.info.num_workers)
//!             .map(|value| Ok(WorkerRecord {
//!                 sequence: Some(SequenceId::new(value as u64)),
//!                 logical_id: LogicalSampleId::new(value as u64),
//!                 sample: value,
//!             }))
//!             .collect::<Vec<_>>()
//!             .into_iter())
//!     }
//!
//!     fn exact_len(&self) -> Option<usize> { Some(self.0) }
//! }
//!
//! let mut loader = StreamDataLoaderBuilder::new(ModuloShards(5))
//!     .workers(2)
//!     .batch_size(2)
//!     .collate(VecCollate)
//!     .build()?;
//! let batches = loader.iter().collect::<Result<Vec<_>, _>>().unwrap();
//! assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
//! # Ok::<(), rusttorch_core::RustTorchError>(())
//! ```
//!
//! Cancellation and deadlines are cooperative. Iterator and owner drop join
//! their workers, and therefore wait for a source blocked in non-cooperative
//! native code until that call returns.

use std::{
    marker::PhantomData,
    mem::{MaybeUninit, size_of},
    num::NonZeroUsize,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, atomic::AtomicUsize},
    thread::{self, JoinHandle},
    time::Duration,
};

use crossbeam_channel::{Receiver, RecvError, Sender, TryRecvError, after, bounded, select};

use rusttorch_core::{Device, Result, RustTorchError, available_devices};

use crate::memory::{BudgetError, ByteBudget, MemoryDisabled, MemoryEnabled, MemoryPolicy};
use crate::worker::WorkerRunContext;
use crate::{
    Auto, CancellationToken, CloneTransformFactory, Collate, Deadline, DefaultCollator, Explicit,
    IdentityTransformFactory, LoaderError, MemoryFootprint, NoWorkerInit, PinDisabled, PinEnabled,
    PinMemory, PinMemoryStatus, PipelineError, TaskContext, Transform, TransformFactory,
    WorkerContext, WorkerInfo, WorkerInit, with_worker_info,
};

/// A record's position in one ordered global stream generation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Creates an identifier from its checked integer representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the integer representation.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next identifier, or `None` at the integer boundary.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// Stable identity used for task randomness and future stream checkpoints.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LogicalSampleId(u64);

impl LogicalSampleId {
    /// Creates an identifier from its checked integer representation.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the integer representation.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next identifier, or `None` at the integer boundary.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// One typed record produced by an explicitly sharded worker source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerRecord<T> {
    /// Position in the global ordered generation, when supplied.
    pub sequence: Option<SequenceId>,
    /// Stable task-randomness and checkpoint identity.
    pub logical_id: LogicalSampleId,
    /// Owned source sample.
    pub sample: T,
}

/// Creates one independently owned stream shard per worker and generation.
pub trait WorkerSourceFactory: Send + Sync + 'static {
    /// Sample produced by every shard.
    type Sample: Send + 'static;
    /// Typed source creation or iteration failure.
    type Error: Send + 'static;
    /// Independently owned shard iterator.
    type Source: Iterator<Item = std::result::Result<WorkerRecord<Self::Sample>, Self::Error>>
        + Send
        + 'static;

    /// Creates this worker's shard for the supplied generation context.
    fn create(&self, worker: WorkerContext) -> std::result::Result<Self::Source, Self::Error>;

    /// Returns the exact global record count when known.
    fn exact_len(&self) -> Option<usize> {
        None
    }
}

type TransformOutput<S, F> =
    <<F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Transform as Transform<
        <S as WorkerSourceFactory>::Sample,
    >>::Output;
type TransformFailure<S, F> =
    <<F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Transform as Transform<
        <S as WorkerSourceFactory>::Sample,
    >>::Error;
type StreamPipelineError<S, C, F, I> = PipelineError<
    <S as WorkerSourceFactory>::Error,
    TransformFailure<S, F>,
    <C as Collate<TransformOutput<S, F>>>::Error,
    <F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;
type StreamLoaderError<S, C, F, I> = LoaderError<StreamPipelineError<S, C, F, I>>;
type ReassemblyEntry<T, P> = (u64, BufferedRecord<T, P>);
type RetainedReassembly<T, P> = Mutex<Vec<ReassemblyEntry<T, P>>>;
type StreamFootprint<S, F> = fn(&TransformOutput<S, F>) -> usize;

trait StreamMemoryPolicy: MemoryPolicy {
    type Waiters;

    fn waiter_bytes(workers: usize) -> Result<usize>;
    fn preflight_waiters(workers: usize) -> Result<()>;
    fn new_waiters(workers: usize) -> Result<Self::Waiters>;
    fn waiters(waiters: &Self::Waiters) -> Option<&[Option<u64>]>;
    fn waiters_mut(waiters: &mut Self::Waiters) -> Option<&mut [Option<u64>]>;
}

impl StreamMemoryPolicy for MemoryDisabled {
    type Waiters = ();

    fn waiter_bytes(_workers: usize) -> Result<usize> {
        Ok(0)
    }

    fn preflight_waiters(_workers: usize) -> Result<()> {
        Ok(())
    }

    fn new_waiters(_workers: usize) -> Result<Self::Waiters> {
        Ok(())
    }

    fn waiters(_waiters: &Self::Waiters) -> Option<&[Option<u64>]> {
        None
    }

    fn waiters_mut(_waiters: &mut Self::Waiters) -> Option<&mut [Option<u64>]> {
        None
    }
}

impl StreamMemoryPolicy for MemoryEnabled {
    type Waiters = Vec<Option<u64>>;

    fn waiter_bytes(workers: usize) -> Result<usize> {
        workers
            .checked_mul(size_of::<Option<u64>>())
            .ok_or_else(|| capacity_error("stream byte waiter storage exceeds usize"))
    }

    fn preflight_waiters(workers: usize) -> Result<()> {
        preflight_vec::<Option<u64>>(workers, "stream byte waiter state")
    }

    fn new_waiters(workers: usize) -> Result<Self::Waiters> {
        let mut waiters = Vec::new();
        reserve_exact(&mut waiters, workers, "stream byte waiter state")?;
        waiters.resize(workers, None);
        Ok(waiters)
    }

    fn waiters(waiters: &Self::Waiters) -> Option<&[Option<u64>]> {
        Some(waiters)
    }

    fn waiters_mut(waiters: &mut Self::Waiters) -> Option<&mut [Option<u64>]> {
        Some(waiters)
    }
}

#[doc(hidden)]
pub trait StreamPinPolicy<S, C, F>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
{
    fn pin(batch: C::Batch, device: Device) -> Result<C::Batch>;
}

impl<S, C, F> StreamPinPolicy<S, C, F> for PinDisabled
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
{
    fn pin(batch: C::Batch, _device: Device) -> Result<C::Batch> {
        Ok(batch)
    }
}

impl<S, C, F, Q> StreamPinPolicy<S, C, F> for PinEnabled<Q>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    C::Batch: PinMemory,
{
    fn pin(batch: C::Batch, device: Device) -> Result<C::Batch> {
        batch.pin_memory(device)
    }
}

type StreamCompletionFor<S, F, I, M> = StreamCompletion<
    TransformOutput<S, F>,
    <M as MemoryPolicy>::Permit,
    <S as WorkerSourceFactory>::Error,
    TransformFailure<S, F>,
    <F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;

enum StreamFailure<SE, TE, FE, IE> {
    Source(SE),
    Transform(TE),
    TransformInit(FE),
    WorkerInit(IE),
    MemoryLimit {
        limit: usize,
        actual: usize,
    },
    Protocol {
        sequence: Option<u64>,
        reason: String,
    },
    Panic,
}

enum StreamMessage<T, P> {
    Record(WorkerRecord<T>, P),
    Waiting { sequence: u64 },
    End,
}

struct StreamCompletion<T, P, SE, TE, FE, IE> {
    worker: usize,
    generation: u64,
    sequence: Option<u64>,
    logical_id: Option<u64>,
    holds_credit: bool,
    result: std::result::Result<StreamMessage<T, P>, StreamFailure<SE, TE, FE, IE>>,
}

enum StreamControl {
    Begin(WorkerRunContext),
}

enum StreamReceive<S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    Completion(StreamCompletionFor<S, F, I, M>),
    Timeout,
    Cancelled,
    Closed,
}

enum ReadyFirst<T> {
    Completion(T),
    Timeout,
    Cancelled,
    Closed,
}

const MAX_STREAM_QUEUE_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;
const CHANNEL_CONTROL_BLOCK_ALLOWANCE_BYTES: usize = 2 * 1024;

#[allow(dead_code)]
#[repr(C)]
struct CapacitySlot<T> {
    stamp: AtomicUsize,
    message: MaybeUninit<T>,
}

fn validate_stream_capacity<S, F, I, M>(
    workers: usize,
    prefetch_factor: usize,
    outstanding: usize,
    batch_size: usize,
    reassembly_slots: usize,
) -> Result<()>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    let expected = workers
        .checked_mul(prefetch_factor)
        .ok_or_else(|| capacity_error("workers multiplied by prefetch_factor exceeds usize"))?;
    if expected != outstanding {
        return Err(capacity_error(
            "outstanding capacity does not match workers multiplied by prefetch_factor",
        ));
    }
    for capacity in [1, prefetch_factor, outstanding] {
        capacity
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .and_then(|mark_bit| mark_bit.checked_mul(2))
            .ok_or_else(|| capacity_error("bounded channel ring arithmetic overflowed"))?;
    }

    let credit_bytes = outstanding
        .checked_mul(size_of::<CapacitySlot<()>>())
        .ok_or_else(|| capacity_error("stream credit storage exceeds usize"))?;
    let result_bytes = outstanding
        .checked_mul(size_of::<CapacitySlot<StreamCompletionFor<S, F, I, M>>>())
        .ok_or_else(|| capacity_error("stream result storage exceeds usize"))?;
    let control_bytes = workers
        .checked_mul(size_of::<CapacitySlot<StreamControl>>())
        .ok_or_else(|| capacity_error("stream control storage exceeds usize"))?;
    let batch_bytes = batch_size
        .checked_mul(size_of::<TransformOutput<S, F>>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| capacity_error("stream batch storage exceeds usize"))?;
    let reassembly_bytes = reassembly_slots
        .checked_mul(size_of::<
            ReassemblyEntry<TransformOutput<S, F>, <M as MemoryPolicy>::Permit>,
        >())
        .ok_or_else(|| capacity_error("ordered stream reassembly exceeds usize"))?;
    let per_worker = size_of::<WorkerInfo>()
        .checked_add(size_of::<(Sender<()>, Receiver<()>)>())
        .and_then(|bytes| {
            bytes.checked_add(size_of::<(Sender<StreamControl>, Receiver<StreamControl>)>())
        })
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<()>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<StreamControl>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<JoinHandle<()>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<bool>()))
        .ok_or_else(|| capacity_error("stream worker bookkeeping exceeds usize"))?;
    let waiter_bytes = M::waiter_bytes(workers)?;
    let bookkeeping = workers
        .checked_mul(per_worker)
        .and_then(|bytes| bytes.checked_add(waiter_bytes))
        .ok_or_else(|| capacity_error("stream worker bookkeeping exceeds usize"))?;
    let channel_count = workers
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| capacity_error("stream channel count exceeds usize"))?;
    let channel_bytes = channel_count
        .checked_mul(CHANNEL_CONTROL_BLOCK_ALLOWANCE_BYTES)
        .ok_or_else(|| capacity_error("stream channel controls exceed usize"))?;
    let aggregate = credit_bytes
        .checked_add(result_bytes)
        .and_then(|bytes| bytes.checked_add(control_bytes))
        .and_then(|bytes| bytes.checked_add(batch_bytes))
        .and_then(|bytes| bytes.checked_add(reassembly_bytes))
        .and_then(|bytes| bytes.checked_add(bookkeeping))
        .and_then(|bytes| bytes.checked_add(channel_bytes))
        .ok_or_else(|| capacity_error("aggregate stream queue storage exceeds usize"))?;
    if aggregate > MAX_STREAM_QUEUE_ALLOCATION_BYTES {
        return Err(capacity_error(format!(
            "aggregate stream queue storage requires {aggregate} bytes, above the {MAX_STREAM_QUEUE_ALLOCATION_BYTES}-byte safety ceiling"
        )));
    }
    preflight_channel_storage::<()>(outstanding)?;
    preflight_channel_storage::<StreamControl>(workers)?;
    preflight_channel_storage::<StreamCompletionFor<S, F, I, M>>(outstanding)?;
    preflight_vec::<TransformOutput<S, F>>(batch_size, "stream batch")?;
    M::preflight_waiters(workers)?;
    Ok(())
}

fn preflight_channel_storage<T>(capacity: usize) -> Result<()> {
    let mut reservation: Vec<MaybeUninit<CapacitySlot<T>>> = Vec::new();
    reservation
        .try_reserve_exact(capacity)
        .map_err(|error| capacity_error(format!("bounded stream storage is unavailable: {error}")))
}

fn preflight_vec<T>(capacity: usize, label: &'static str) -> Result<()> {
    let mut reservation: Vec<T> = Vec::new();
    reservation
        .try_reserve_exact(capacity)
        .map_err(|error| capacity_error(format!("{label} capacity is unavailable: {error}")))
}

fn capacity_error(reason: impl Into<String>) -> RustTorchError {
    invalid_configuration("prefetch_factor", reason)
}

struct StreamWorkerPool<S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    controls: Vec<Sender<StreamControl>>,
    credits: Vec<Sender<()>>,
    results: Option<Receiver<StreamCompletionFor<S, F, I, M>>>,
    handles: Vec<JoinHandle<()>>,
    shutdown: CancellationToken,
    active: Option<WorkerRunContext>,
    terminal: Vec<bool>,
    terminal_count: usize,
    fatal_exit: bool,
    poisoned: bool,
}

impl<S, F, I, M> StreamWorkerPool<S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy + Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_factory: Arc<S>,
        transform_factory: Arc<F>,
        initializer: Arc<I>,
        configuration: StreamConfiguration,
        generation: u64,
        outstanding: usize,
        reassembly_slots: usize,
        footprint: Option<StreamFootprint<S, F>>,
    ) -> Result<Self> {
        validate_stream_capacity::<S, F, I, M>(
            configuration.workers,
            configuration.prefetch_factor,
            outstanding,
            configuration.batch_size,
            reassembly_slots,
        )?;

        let (result_sender, result_receiver) =
            bounded_checked(outstanding, "stream result channel")?;
        let mut credit_lanes = Vec::new();
        let mut control_lanes = Vec::new();
        reserve_exact(
            &mut credit_lanes,
            configuration.workers,
            "stream credit channels",
        )?;
        reserve_exact(
            &mut control_lanes,
            configuration.workers,
            "stream control channels",
        )?;
        for _ in 0..configuration.workers {
            let (credit_sender, credit_receiver) =
                bounded_checked(configuration.prefetch_factor, "stream credit channel")?;
            for _ in 0..configuration.prefetch_factor {
                credit_sender.try_send(()).map_err(|_| {
                    capacity_error("validated stream credit channel rejected a token")
                })?;
            }
            credit_lanes.push((credit_sender, credit_receiver));
            control_lanes.push(bounded_checked(1, "stream control channel")?);
        }

        let shutdown = CancellationToken::new();
        let mut pool = Self {
            controls: Vec::new(),
            credits: Vec::new(),
            results: Some(result_receiver),
            handles: Vec::new(),
            shutdown: shutdown.clone(),
            active: None,
            terminal: Vec::new(),
            terminal_count: 0,
            fatal_exit: false,
            poisoned: false,
        };
        reserve_exact(&mut pool.controls, configuration.workers, "stream controls")?;
        reserve_exact(&mut pool.credits, configuration.workers, "stream credits")?;
        reserve_exact(&mut pool.handles, configuration.workers, "stream workers")?;
        reserve_exact(
            &mut pool.terminal,
            configuration.workers,
            "stream terminal state",
        )?;
        pool.terminal.resize(configuration.workers, false);

        for (id, ((credit_sender, credit_receiver), (control_sender, control_receiver))) in
            credit_lanes.into_iter().zip(control_lanes).enumerate()
        {
            let info = WorkerInfo::from_loader_seed(
                id,
                configuration.workers,
                configuration.loader_seed,
                configuration.rank,
                generation,
            )?;
            pool.credits.push(credit_sender);
            pool.controls.push(control_sender);
            let source_factory = Arc::clone(&source_factory);
            let transform_factory = Arc::clone(&transform_factory);
            let initializer = Arc::clone(&initializer);
            let results = result_sender.clone();
            let shutdown = shutdown.clone();
            let handle = thread::Builder::new()
                .name(format!("rusttorch-stream-worker-{id}"))
                .spawn(move || {
                    run_stream_worker::<S, F, I, M>(
                        info,
                        generation,
                        source_factory,
                        transform_factory,
                        initializer,
                        control_receiver,
                        credit_receiver,
                        results,
                        shutdown,
                        footprint,
                        configuration.ordered,
                    );
                })
                .map_err(|error| RustTorchError::BackendUnavailable {
                    backend: "data loader stream worker threads",
                    reason: error.to_string(),
                })?;
            pool.handles.push(handle);
        }
        drop(result_sender);
        Ok(pool)
    }
}

impl<S, F, I, M> StreamWorkerPool<S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    fn start_generation(&mut self, context: WorkerRunContext) -> std::result::Result<(), ()> {
        if self.poisoned || self.active.is_some() || self.shutdown.is_cancelled() {
            return Err(());
        }
        self.terminal.fill(false);
        self.terminal_count = 0;
        self.fatal_exit = false;
        self.active = Some(context.clone());
        for control in &self.controls {
            if control.send(StreamControl::Begin(context.clone())).is_err() {
                self.poisoned = true;
                self.shutdown();
                return Err(());
            }
        }
        Ok(())
    }

    fn receive(
        &self,
        context: &WorkerRunContext,
        timeout: Option<Duration>,
    ) -> StreamReceive<S, F, I, M> {
        let results = self
            .results
            .as_ref()
            .expect("stream result receiver exists before shutdown");
        match receive_ready_first(results, context, &self.shutdown, timeout) {
            ReadyFirst::Completion(completion) => StreamReceive::Completion(completion),
            ReadyFirst::Timeout => StreamReceive::Timeout,
            ReadyFirst::Cancelled => StreamReceive::Cancelled,
            ReadyFirst::Closed => StreamReceive::Closed,
        }
    }

    fn account(
        &mut self,
        completion: &StreamCompletionFor<S, F, I, M>,
    ) -> std::result::Result<(), ()> {
        let fatal = matches!(
            &completion.result,
            Err(StreamFailure::TransformInit(_)
                | StreamFailure::WorkerInit(_)
                | StreamFailure::Panic)
        );
        let terminal = fatal || matches!(&completion.result, Ok(StreamMessage::End));
        self.account_parts(completion.worker, completion.holds_credit, terminal, fatal)
    }

    fn account_parts(
        &mut self,
        worker: usize,
        holds_credit: bool,
        terminal: bool,
        fatal: bool,
    ) -> std::result::Result<(), ()> {
        if holds_credit {
            self.return_credit(worker)?;
        }
        if fatal {
            self.fatal_exit = true;
        }
        if terminal {
            let Some(seen) = self.terminal.get_mut(worker) else {
                return Err(());
            };
            if !*seen {
                *seen = true;
                self.terminal_count += 1;
            }
        }
        Ok(())
    }

    fn return_credit(&self, worker: usize) -> std::result::Result<(), ()> {
        let Some(credit) = self.credits.get(worker) else {
            return Err(());
        };
        credit.try_send(()).map_err(|_| ())
    }

    fn all_terminal(&self) -> bool {
        self.terminal_count == self.terminal.len()
    }

    fn quiesce(&mut self, generation: u64) -> std::result::Result<(), ()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        active.cancel();
        while !self.all_terminal() {
            let completion = self.results.as_ref().ok_or(())?.recv().map_err(|_| ())?;
            if completion.generation == generation {
                self.account(&completion)?;
            } else if completion.holds_credit {
                self.return_credit(completion.worker)?;
            }
        }
        while let Some(results) = self.results.as_ref() {
            match results.try_recv() {
                Ok(completion) => self.account(&completion)?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Err(()),
            }
        }
        if self.fatal_exit {
            self.poisoned = true;
            self.shutdown();
            return Err(());
        }
        Ok(())
    }

    fn poison(&mut self) {
        self.poisoned = true;
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(active) = self.active.take() {
            active.cancel();
        }
        self.shutdown.cancel();
        self.controls.clear();
        self.credits.clear();
        self.results.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl<S, F, I, M> Drop for StreamWorkerPool<S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn receive_ready_first<T>(
    results: &Receiver<T>,
    context: &WorkerRunContext,
    shutdown: &CancellationToken,
    timeout: Option<Duration>,
) -> ReadyFirst<T> {
    match results.try_recv() {
        Ok(completion) => return ReadyFirst::Completion(completion),
        Err(TryRecvError::Disconnected) => return ReadyFirst::Closed,
        Err(TryRecvError::Empty) => {}
    }
    if let Some(timeout) = timeout {
        let timer = after(timeout);
        select! {
            recv(results) -> result => map_ready_receive(result),
            recv(context.cancellation.signal()) -> _ => match results.try_recv() {
                Ok(completion) => ReadyFirst::Completion(completion),
                Err(TryRecvError::Empty) => ReadyFirst::Cancelled,
                Err(TryRecvError::Disconnected) => ReadyFirst::Closed,
            },
            recv(shutdown.signal()) -> _ => ReadyFirst::Closed,
            recv(timer) -> _ => match results.try_recv() {
                Ok(completion) => ReadyFirst::Completion(completion),
                Err(TryRecvError::Empty) => ReadyFirst::Timeout,
                Err(TryRecvError::Disconnected) => ReadyFirst::Closed,
            },
        }
    } else {
        select! {
            recv(results) -> result => map_ready_receive(result),
            recv(context.cancellation.signal()) -> _ => match results.try_recv() {
                Ok(completion) => ReadyFirst::Completion(completion),
                Err(TryRecvError::Empty) => ReadyFirst::Cancelled,
                Err(TryRecvError::Disconnected) => ReadyFirst::Closed,
            },
            recv(shutdown.signal()) -> _ => ReadyFirst::Closed,
        }
    }
}

fn map_ready_receive<T>(result: std::result::Result<T, RecvError>) -> ReadyFirst<T> {
    match result {
        Ok(completion) => ReadyFirst::Completion(completion),
        Err(_) => ReadyFirst::Closed,
    }
}

fn reserve_exact<T>(values: &mut Vec<T>, capacity: usize, label: &'static str) -> Result<()> {
    values
        .try_reserve_exact(capacity)
        .map_err(|error| capacity_error(format!("{label} capacity is unavailable: {error}")))
}

fn bounded_checked<T>(capacity: usize, label: &'static str) -> Result<(Sender<T>, Receiver<T>)> {
    catch_unwind(AssertUnwindSafe(|| bounded(capacity))).map_err(|_| {
        capacity_error(format!(
            "{label} rejected the validated capacity {capacity}"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn run_stream_worker<S, F, I, M>(
    info: WorkerInfo,
    initialization_generation: u64,
    source_factory: Arc<S>,
    transform_factory: Arc<F>,
    initializer: Arc<I>,
    controls: Receiver<StreamControl>,
    credits: Receiver<()>,
    results: Sender<StreamCompletionFor<S, F, I, M>>,
    shutdown: CancellationToken,
    footprint: Option<StreamFootprint<S, F>>,
    ordered: bool,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy + Send + 'static,
{
    let lifecycle = WorkerContext::new(info, shutdown.clone(), shutdown.paired_deadline());
    let mut active: Option<(WorkerRunContext, Option<u64>, Option<u64>, bool)> = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        with_worker_info(info, || {
            if let Err(error) = initializer.initialize(&lifecycle) {
                send_stream_failure(
                    &results,
                    &shutdown,
                    info.id,
                    initialization_generation,
                    None,
                    None,
                    false,
                    StreamFailure::WorkerInit(error),
                );
                return true;
            }
            let mut transform = match transform_factory.create(Some(&lifecycle)) {
                Ok(transform) => transform,
                Err(error) => {
                    send_stream_failure(
                        &results,
                        &shutdown,
                        info.id,
                        initialization_generation,
                        None,
                        None,
                        false,
                        StreamFailure::TransformInit(error),
                    );
                    return true;
                }
            };

            loop {
                let run = select! {
                    recv(controls) -> control => match control {
                        Ok(StreamControl::Begin(run)) => run,
                        Err(_) => return false,
                    },
                    recv(shutdown.signal()) -> _ => return false,
                };
                active = Some((run.clone(), None, None, false));
                run_stream_generation::<S, F, I, M>(
                    info,
                    &source_factory,
                    &mut transform,
                    &credits,
                    &results,
                    &shutdown,
                    &run,
                    &mut active,
                    footprint,
                    ordered,
                );
                active = None;
                if shutdown.is_cancelled() {
                    return false;
                }
            }
        })
    }));
    let fatal = match outcome {
        Ok(fatal) => fatal,
        Err(_) => {
            let (generation, sequence, logical_id, holds_credit) = active
                .as_ref()
                .map(|(run, sequence, logical_id, held)| {
                    (run.generation, *sequence, *logical_id, *held)
                })
                .unwrap_or((initialization_generation, None, None, false));
            send_stream_failure(
                &results,
                &shutdown,
                info.id,
                generation,
                sequence,
                logical_id,
                holds_credit,
                StreamFailure::Panic,
            );
            true
        }
    };
    if fatal {
        shutdown.wait_cancelled();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_stream_generation<S, F, I, M>(
    info: WorkerInfo,
    source_factory: &S,
    transform: &mut F::Transform,
    credits: &Receiver<()>,
    results: &Sender<StreamCompletionFor<S, F, I, M>>,
    shutdown: &CancellationToken,
    run: &WorkerRunContext,
    active: &mut Option<(WorkerRunContext, Option<u64>, Option<u64>, bool)>,
    footprint: Option<StreamFootprint<S, F>>,
    ordered: bool,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy,
{
    let generation_info = WorkerInfo::from_loader_seed(
        info.id,
        info.num_workers,
        run.loader_seed,
        info.rank,
        run.generation,
    )
    .expect("validated stream worker identity remains valid for every generation");
    let context = WorkerContext::new(
        generation_info,
        run.cancellation.clone(),
        run.deadline.clone(),
    );
    let mut source = match source_factory.create(context.clone()) {
        Ok(source) if context.check().is_ok() => source,
        Ok(source) => {
            drop(source);
            send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, false);
            return;
        }
        Err(error) => {
            send_stream_failure(
                results,
                shutdown,
                info.id,
                run.generation,
                None,
                None,
                false,
                StreamFailure::Source(error),
            );
            run.cancel();
            send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, false);
            return;
        }
    };

    let mut last_sequence = None;
    loop {
        let acquired = select! {
            recv(credits) -> credit => credit.is_ok(),
            recv(run.cancellation.signal()) -> _ => false,
            recv(shutdown.signal()) -> _ => return,
        };
        if !acquired {
            drop(source);
            send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, false);
            return;
        }
        active.as_mut().expect("active stream generation").3 = true;
        if context.check().is_err() {
            drop(source);
            send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
            active.as_mut().expect("active stream generation").3 = false;
            return;
        }

        let record = match source.next() {
            Some(Ok(record)) => {
                if context.check().is_err() {
                    drop(source);
                    send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
                    active.as_mut().expect("active stream generation").3 = false;
                    return;
                }
                record
            }
            None => {
                drop(source);
                send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
                active.as_mut().expect("active stream generation").3 = false;
                return;
            }
            Some(Err(error)) => {
                drop(source);
                send_stream_failure(
                    results,
                    shutdown,
                    info.id,
                    run.generation,
                    None,
                    None,
                    true,
                    StreamFailure::Source(error),
                );
                active.as_mut().expect("active stream generation").3 = false;
                run.cancel();
                send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, false);
                return;
            }
        };
        let sequence = record.sequence.map(SequenceId::get);
        let logical_id = record.logical_id.get();
        {
            let active = active.as_mut().expect("active stream generation");
            active.1 = sequence;
            active.2 = Some(logical_id);
        }
        let task_context = TaskContext {
            loader_seed: run.loader_seed,
            epoch: run.epoch,
            rank: info.rank,
            logical_sample: logical_id,
            stage: 0,
            cancellation: run.cancellation.clone(),
            deadline: run.deadline.clone(),
        };
        let sample = match transform.transform(record.sample, &task_context) {
            Ok(sample) if context.check().is_ok() => sample,
            Ok(_) => {
                drop(source);
                send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
                active.as_mut().expect("active stream generation").3 = false;
                return;
            }
            Err(error) => {
                drop(source);
                send_stream_failure(
                    results,
                    shutdown,
                    info.id,
                    run.generation,
                    sequence,
                    Some(logical_id),
                    true,
                    StreamFailure::Transform(error),
                );
                active.as_mut().expect("active stream generation").3 = false;
                run.cancel();
                send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, false);
                return;
            }
        };
        let permit = if let Some(footprint) = footprint {
            let actual = footprint(&sample);
            let budget = run
                .byte_budget
                .as_ref()
                .expect("enabled stream byte accounting has a generation budget");
            let acquired = if ordered {
                let Some(sequence) = sequence else {
                    drop(source);
                    send_stream_failure(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        None,
                        Some(logical_id),
                        true,
                        StreamFailure::Protocol {
                            sequence: None,
                            reason: "ordered byte-bounded stream record did not provide a sequence identifier".to_owned(),
                        },
                    );
                    active.as_mut().expect("active stream generation").3 = false;
                    run.cancel();
                    send_stream_end::<S, F, I, M>(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        false,
                    );
                    return;
                };
                if last_sequence.is_some_and(|previous| sequence <= previous) {
                    drop(source);
                    send_stream_failure(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        Some(sequence),
                        Some(logical_id),
                        true,
                        StreamFailure::Protocol {
                            sequence: Some(sequence),
                            reason: format!(
                                "worker {} emitted non-increasing sequence {sequence}",
                                info.id
                            ),
                        },
                    );
                    active.as_mut().expect("active stream generation").3 = false;
                    run.cancel();
                    send_stream_end::<S, F, I, M>(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        false,
                    );
                    return;
                }
                last_sequence = Some(sequence);
                if !send_stream_completion(
                    results,
                    shutdown,
                    Some(&run.cancellation),
                    StreamCompletion {
                        worker: info.id,
                        generation: run.generation,
                        sequence: Some(sequence),
                        logical_id: Some(logical_id),
                        holds_credit: false,
                        result: Ok(StreamMessage::Waiting { sequence }),
                    },
                ) {
                    drop(source);
                    send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
                    active.as_mut().expect("active stream generation").3 = false;
                    return;
                }
                budget.acquire_ordered(sequence, actual)
            } else {
                budget.acquire(actual)
            };
            match acquired {
                Ok(permit) => Some(permit),
                Err(BudgetError::Cancelled) => {
                    drop(source);
                    send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
                    active.as_mut().expect("active stream generation").3 = false;
                    return;
                }
                Err(BudgetError::Oversize { limit, actual }) => {
                    drop(source);
                    send_stream_failure(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        sequence,
                        Some(logical_id),
                        true,
                        StreamFailure::MemoryLimit { limit, actual },
                    );
                    active.as_mut().expect("active stream generation").3 = false;
                    run.cancel();
                    send_stream_end::<S, F, I, M>(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        false,
                    );
                    return;
                }
                Err(BudgetError::SequenceAlreadyAdmitted) => {
                    drop(source);
                    send_stream_failure(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        sequence,
                        Some(logical_id),
                        true,
                        StreamFailure::Protocol {
                            sequence,
                            reason: "sequence was already admitted to the byte budget".to_owned(),
                        },
                    );
                    active.as_mut().expect("active stream generation").3 = false;
                    run.cancel();
                    send_stream_end::<S, F, I, M>(
                        results,
                        shutdown,
                        info.id,
                        run.generation,
                        false,
                    );
                    return;
                }
            }
        } else {
            None
        };
        let sent = send_stream_completion(
            results,
            shutdown,
            Some(&run.cancellation),
            StreamCompletion {
                worker: info.id,
                generation: run.generation,
                sequence,
                logical_id: Some(logical_id),
                holds_credit: true,
                result: Ok(StreamMessage::Record(
                    WorkerRecord {
                        sequence: record.sequence,
                        logical_id: record.logical_id,
                        sample,
                    },
                    M::permit(permit),
                )),
            },
        );
        if sent {
            let active = active.as_mut().expect("active stream generation");
            active.1 = None;
            active.2 = None;
            active.3 = false;
        } else {
            drop(source);
            send_stream_end::<S, F, I, M>(results, shutdown, info.id, run.generation, true);
            active.as_mut().expect("active stream generation").3 = false;
            return;
        }
    }
}

fn send_stream_end<S, F, I, M>(
    results: &Sender<StreamCompletionFor<S, F, I, M>>,
    shutdown: &CancellationToken,
    worker: usize,
    generation: u64,
    holds_credit: bool,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    let _ = send_stream_completion(
        results,
        shutdown,
        None,
        StreamCompletion {
            worker,
            generation,
            sequence: None,
            logical_id: None,
            holds_credit,
            result: Ok(StreamMessage::End),
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn send_stream_failure<T, P, SE, TE, FE, IE>(
    results: &Sender<StreamCompletion<T, P, SE, TE, FE, IE>>,
    shutdown: &CancellationToken,
    worker: usize,
    generation: u64,
    sequence: Option<u64>,
    logical_id: Option<u64>,
    holds_credit: bool,
    failure: StreamFailure<SE, TE, FE, IE>,
) {
    let _ = send_stream_completion(
        results,
        shutdown,
        None,
        StreamCompletion {
            worker,
            generation,
            sequence,
            logical_id,
            holds_credit,
            result: Err(failure),
        },
    );
}

fn send_stream_completion<T, P, SE, TE, FE, IE>(
    results: &Sender<StreamCompletion<T, P, SE, TE, FE, IE>>,
    shutdown: &CancellationToken,
    cancellation: Option<&CancellationToken>,
    completion: StreamCompletion<T, P, SE, TE, FE, IE>,
) -> bool {
    if let Some(cancellation) = cancellation {
        select! {
            send(results, completion) -> result => result.is_ok(),
            recv(cancellation.signal()) -> _ => false,
            recv(shutdown.signal()) -> _ => false,
        }
    } else {
        select! {
            send(results, completion) -> result => result.is_ok(),
            recv(shutdown.signal()) -> _ => false,
        }
    }
}

#[derive(Clone, Copy)]
struct StreamConfiguration {
    workers: usize,
    batch_size: usize,
    drop_last: bool,
    prefetch_factor: usize,
    ordered: bool,
    persistent_workers: bool,
    timeout: Option<Duration>,
    loader_seed: u64,
    epoch: u64,
    rank: usize,
    prefetch_bytes: Option<NonZeroUsize>,
    pin_memory: StreamPinRequest,
}

#[derive(Clone, Copy)]
enum StreamPinRequest {
    Disabled,
    Auto,
    Explicit(Device),
}

impl Default for StreamConfiguration {
    fn default() -> Self {
        Self {
            workers: 1,
            batch_size: 1,
            drop_last: false,
            prefetch_factor: 2,
            ordered: true,
            persistent_workers: false,
            timeout: None,
            loader_seed: 0,
            epoch: 0,
            rank: 0,
            prefetch_bytes: None,
            pin_memory: StreamPinRequest::Disabled,
        }
    }
}

/// Builder for an explicitly sharded positive-worker stream loader.
pub struct StreamDataLoaderBuilder<
    S,
    C = DefaultCollator,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    M = MemoryDisabled,
    N = PinDisabled,
> {
    factory: S,
    collator: C,
    transform_factory: F,
    worker_init: I,
    configuration: StreamConfiguration,
    states: PhantomData<(M, N)>,
}

impl<S> StreamDataLoaderBuilder<S>
where
    S: WorkerSourceFactory,
{
    /// Starts a builder with one worker, ordered delivery, batch size one,
    /// prefetch factor two, no timeout, and no persistence.
    pub fn new(factory: S) -> Self {
        Self {
            factory,
            collator: DefaultCollator,
            transform_factory: IdentityTransformFactory,
            worker_init: NoWorkerInit,
            configuration: StreamConfiguration::default(),
            states: PhantomData,
        }
    }
}

impl<S, C, F, I, M, N> StreamDataLoaderBuilder<S, C, F, I, M, N> {
    /// Sets the positive stream worker count.
    pub fn workers(mut self, workers: usize) -> Self {
        self.configuration.workers = workers;
        self
    }

    /// Sets the global coordinator batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.configuration.batch_size = batch_size;
        self
    }

    /// Selects whether the one final global short batch is omitted.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.configuration.drop_last = drop_last;
        self
    }

    /// Sets bounded unpublished records permitted per worker.
    ///
    /// In ordered mode, filling the complete global reassembly window with
    /// higher IDs before the next ID is observable is a stream protocol error.
    pub fn prefetch_factor(mut self, factor: usize) -> Self {
        self.configuration.prefetch_factor = factor;
        self
    }

    /// Enables a nonzero byte budget for final post-transform records.
    ///
    /// Ordered byte-bounded shards must emit strictly increasing sequence IDs.
    /// The coordinator tracks one bounded front waiter per shard and reports a
    /// protocol error when every active shard has advanced beyond a missing ID.
    pub fn prefetch_bytes(
        mut self,
        limit: NonZeroUsize,
    ) -> StreamDataLoaderBuilder<S, C, F, I, MemoryEnabled, N> {
        self.configuration.prefetch_bytes = Some(limit);
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Enables automatic recursive pinning for CUDA device zero when available.
    pub fn pin_memory(mut self) -> StreamDataLoaderBuilder<S, C, F, I, M, PinEnabled<Auto>> {
        self.configuration.pin_memory = StreamPinRequest::Auto;
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Enables recursive pinning for one explicit, available CUDA device.
    pub fn pin_memory_for(
        mut self,
        device: Device,
    ) -> StreamDataLoaderBuilder<S, C, F, I, M, PinEnabled<Explicit>> {
        self.configuration.pin_memory = StreamPinRequest::Explicit(device);
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Selects globally sequenced or completion-order delivery.
    pub fn ordered(mut self, ordered: bool) -> Self {
        self.configuration.ordered = ordered;
        self
    }

    /// Selects globally sequenced or completion-order delivery.
    pub fn in_order(self, in_order: bool) -> Self {
        self.ordered(in_order)
    }

    /// Sets the deterministic loader seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.configuration.loader_seed = seed;
        self
    }

    /// Sets the initial stream epoch.
    pub fn epoch(mut self, epoch: u64) -> Self {
        self.configuration.epoch = epoch;
        self
    }

    /// Sets the distributed rank included in worker and task contexts.
    pub fn rank(mut self, rank: usize) -> Self {
        self.configuration.rank = rank;
        self
    }

    /// Sets the cooperative timeout for each blocking iterator call.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.configuration.timeout = (!timeout.is_zero()).then_some(timeout);
        self
    }

    /// Selects loader-owned persistent worker threads.
    pub fn persistent_workers(mut self, persistent: bool) -> Self {
        self.configuration.persistent_workers = persistent;
        self
    }

    /// Replaces coordinator collation.
    pub fn collate<C2>(self, collator: C2) -> StreamDataLoaderBuilder<S, C2, F, I, M, N> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Clones one transform template for each worker lifecycle.
    pub fn transform<T>(
        self,
        transform: T,
    ) -> StreamDataLoaderBuilder<S, C, CloneTransformFactory<T>, I, M, N> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: CloneTransformFactory::new(transform),
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Replaces worker transform construction.
    pub fn transform_factory<F2>(self, factory: F2) -> StreamDataLoaderBuilder<S, C, F2, I, M, N> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Replaces worker initialization.
    pub fn worker_init<I2>(self, worker_init: I2) -> StreamDataLoaderBuilder<S, C, F, I2, M, N> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init,
            configuration: self.configuration,
            states: PhantomData,
        }
    }

    /// Validates configuration and constructs the stream owner.
    fn build_inner(
        self,
        footprint: Option<StreamFootprint<S, F>>,
    ) -> Result<StreamDataLoader<S, C, F, I, M, N>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        C: Collate<TransformOutput<S, F>>,
        I: WorkerInit,
        M: StreamMemoryPolicy,
    {
        let pin_memory_status = resolve_stream_pin(self.configuration.pin_memory)?;
        if self.configuration.workers == 0 {
            return Err(invalid_configuration(
                "workers",
                "stream workers must be positive; use batches or batches_with_collate for an ordinary iterator",
            ));
        }
        if self.configuration.batch_size == 0 {
            return Err(invalid_configuration(
                "batch_size",
                "must be greater than zero",
            ));
        }
        if self.configuration.prefetch_factor == 0 {
            return Err(invalid_configuration(
                "prefetch_factor",
                "must be greater than zero",
            ));
        }
        if self
            .configuration
            .timeout
            .is_some_and(|timeout| !Deadline::can_represent(timeout))
        {
            return Err(invalid_configuration(
                "timeout",
                "exceeds the platform monotonic clock range",
            ));
        }
        let outstanding_capacity = self
            .configuration
            .workers
            .checked_mul(self.configuration.prefetch_factor)
            .ok_or_else(|| {
                invalid_configuration(
                    "prefetch_factor",
                    "workers multiplied by prefetch_factor exceeds usize",
                )
            })?;
        validate_stream_capacity::<S, F, I, M>(
            self.configuration.workers,
            self.configuration.prefetch_factor,
            outstanding_capacity,
            self.configuration.batch_size,
            usize::from(self.configuration.ordered) * outstanding_capacity,
        )?;
        let mut ordered_reassembly = Vec::new();
        if self.configuration.ordered {
            ordered_reassembly
                .try_reserve_exact(outstanding_capacity)
                .map_err(|error| {
                    capacity_error(format!(
                        "ordered stream reassembly capacity is unavailable: {error}"
                    ))
                })?;
        }
        validate_stream_capacity::<S, F, I, M>(
            self.configuration.workers,
            self.configuration.prefetch_factor,
            outstanding_capacity,
            self.configuration.batch_size,
            ordered_reassembly.capacity(),
        )?;
        let exact_len = self.factory.exact_len();
        Ok(StreamDataLoader {
            factory: Arc::new(self.factory),
            collator: self.collator,
            transform_factory: Arc::new(self.transform_factory),
            worker_init: Arc::new(self.worker_init),
            configuration: self.configuration,
            outstanding_capacity,
            exact_len,
            next_generation: 0,
            persistent_pool: None,
            ordered_reassembly: Mutex::new(ordered_reassembly),
            footprint,
            pin_memory_status,
            policies: PhantomData,
        })
    }
}

fn stream_resident_bytes<T: MemoryFootprint>(value: &T) -> usize {
    value.resident_bytes()
}

fn resolve_stream_pin(request: StreamPinRequest) -> Result<PinMemoryStatus> {
    match request {
        StreamPinRequest::Disabled => Ok(PinMemoryStatus::Disabled),
        StreamPinRequest::Auto => {
            let capabilities = available_devices();
            if capabilities.cuda && capabilities.cuda_device_count > 0 {
                Ok(PinMemoryStatus::Enabled(Device::Cuda(0)))
            } else {
                Ok(PinMemoryStatus::DisabledNoAccelerator)
            }
        }
        StreamPinRequest::Explicit(Device::Cuda(index)) => {
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
        StreamPinRequest::Explicit(device) => Err(invalid_configuration(
            "pin_memory",
            format!("only an available CUDA device can back pinned host memory, got {device:?}"),
        )),
    }
}

impl<S, C, F, I> StreamDataLoaderBuilder<S, C, F, I, MemoryDisabled, PinDisabled> {
    /// Validates configuration and builds an item-bounded, unpinned stream.
    pub fn build(self) -> Result<StreamDataLoader<S, C, F, I, MemoryDisabled, PinDisabled>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        C: Collate<TransformOutput<S, F>>,
        I: WorkerInit,
    {
        self.build_inner(None)
    }
}

impl<S, C, F, I> StreamDataLoaderBuilder<S, C, F, I, MemoryEnabled, PinDisabled> {
    /// Validates configuration and builds a byte-bounded, unpinned stream.
    pub fn build(self) -> Result<StreamDataLoader<S, C, F, I, MemoryEnabled, PinDisabled>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        TransformOutput<S, F>: MemoryFootprint,
        C: Collate<TransformOutput<S, F>>,
        I: WorkerInit,
    {
        self.build_inner(Some(stream_resident_bytes::<TransformOutput<S, F>>))
    }
}

impl<S, C, F, I, Q> StreamDataLoaderBuilder<S, C, F, I, MemoryDisabled, PinEnabled<Q>> {
    /// Validates configuration and builds an item-bounded pinned stream.
    pub fn build(self) -> Result<StreamDataLoader<S, C, F, I, MemoryDisabled, PinEnabled<Q>>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        C: Collate<TransformOutput<S, F>>,
        C::Batch: PinMemory,
        I: WorkerInit,
    {
        self.build_inner(None)
    }
}

impl<S, C, F, I, Q> StreamDataLoaderBuilder<S, C, F, I, MemoryEnabled, PinEnabled<Q>> {
    /// Validates configuration and builds a byte-bounded pinned stream.
    pub fn build(self) -> Result<StreamDataLoader<S, C, F, I, MemoryEnabled, PinEnabled<Q>>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        TransformOutput<S, F>: MemoryFootprint,
        C: Collate<TransformOutput<S, F>>,
        C::Batch: PinMemory,
        I: WorkerInit,
    {
        self.build_inner(Some(stream_resident_bytes::<TransformOutput<S, F>>))
    }
}

/// Owned, re-iterable explicitly sharded stream loader.
#[allow(private_bounds)]
pub struct StreamDataLoader<
    S,
    C,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
    M = MemoryDisabled,
    N = PinDisabled,
> where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    factory: Arc<S>,
    collator: C,
    transform_factory: Arc<F>,
    worker_init: Arc<I>,
    configuration: StreamConfiguration,
    outstanding_capacity: usize,
    exact_len: Option<usize>,
    next_generation: u64,
    persistent_pool: Option<StreamWorkerPool<S, F, I, M>>,
    // `iter` has exclusive loader access; this wrapper preserves `Sync` for
    // output that is `Send` but not `Sync` without runtime contention.
    ordered_reassembly: RetainedReassembly<TransformOutput<S, F>, M::Permit>,
    footprint: Option<StreamFootprint<S, F>>,
    pin_memory_status: PinMemoryStatus,
    policies: PhantomData<(M, N)>,
}

#[allow(private_bounds)]
impl<S, C, F, I, M, N> StreamDataLoader<S, C, F, I, M, N>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    /// Returns the exact global batch count when the factory is sized.
    pub fn len(&self) -> Option<usize> {
        self.exact_len.map(|length| {
            let complete = length / self.configuration.batch_size;
            complete
                + usize::from(
                    !self.configuration.drop_last
                        && !length.is_multiple_of(self.configuration.batch_size),
                )
        })
    }

    /// Returns whether the sized stream yields no batches.
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|length| length == 0)
    }

    /// Returns the configured worker count.
    pub fn workers(&self) -> usize {
        self.configuration.workers
    }

    /// Returns the effective records-prefetched-per-worker factor.
    pub fn effective_prefetch_factor(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.configuration.prefetch_factor)
            .expect("validated prefetch factor is nonzero")
    }

    /// Returns the effective post-transform record byte budget.
    pub fn effective_prefetch_bytes(&self) -> Option<NonZeroUsize> {
        self.configuration.prefetch_bytes
    }

    /// Returns whether recursive batch pinning was requested.
    pub fn pin_memory_enabled(&self) -> bool {
        !matches!(self.pin_memory_status, PinMemoryStatus::Disabled)
    }

    /// Returns the effective recursive pinning behavior.
    pub fn pin_memory_status(&self) -> PinMemoryStatus {
        self.pin_memory_status
    }

    /// Returns whether ordered delivery is enabled.
    pub fn is_ordered(&self) -> bool {
        self.configuration.ordered
    }

    /// Returns the configured epoch.
    pub fn epoch(&self) -> u64 {
        self.configuration.epoch
    }

    /// Selects the epoch used by future iterator generations.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.configuration.epoch = epoch;
    }
}

struct BufferedRecord<T, P> {
    worker: usize,
    record: WorkerRecord<T>,
    permit: P,
}

enum StreamIteratorPool<'a, S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    Owned(StreamWorkerPool<S, F, I, M>),
    Persistent(&'a mut StreamWorkerPool<S, F, I, M>),
}

impl<S, F, I, M> StreamIteratorPool<'_, S, F, I, M>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    fn pool(&self) -> &StreamWorkerPool<S, F, I, M> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent(pool) => pool,
        }
    }

    fn pool_mut(&mut self) -> &mut StreamWorkerPool<S, F, I, M> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent(pool) => pool,
        }
    }

    fn is_persistent(&self) -> bool {
        matches!(self, Self::Persistent(_))
    }
}

#[allow(private_bounds)]
impl<S, C, F, I, M, N> StreamDataLoader<S, C, F, I, M, N>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy + Send + 'static,
    N: StreamPinPolicy<S, C, F>,
{
    /// Starts a fresh explicitly sharded worker generation.
    pub fn iter(&mut self) -> StreamLoaderIter<'_, S, C, F, I, M, N> {
        let ordered_reassembly = match self.ordered_reassembly.get_mut() {
            Ok(reassembly) => reassembly,
            Err(poisoned) => poisoned.into_inner(),
        };
        ordered_reassembly.clear();
        let reassembly_allocation_slots = ordered_reassembly.capacity();
        let generation = self.next_generation;
        let run_context = WorkerRunContext::new(
            generation,
            self.configuration.loader_seed,
            self.configuration.epoch,
        )
        .with_byte_budget(
            self.configuration
                .prefetch_bytes
                .map(|limit| ByteBudget::new(limit.get())),
        );
        let mut pending_error = None;
        if let Some(next) = generation.checked_add(1) {
            self.next_generation = next;
        } else {
            pending_error = Some(LoaderError::Configuration(invalid_configuration(
                "workers",
                "stream iterator generation overflowed",
            )));
        }

        let mut partial = Vec::new();
        if partial
            .try_reserve_exact(self.configuration.batch_size)
            .is_err()
        {
            pending_error = Some(LoaderError::Configuration(invalid_configuration(
                "batch_size",
                "stream batch allocation is unavailable",
            )));
        }

        let front_waiters = match M::new_waiters(self.configuration.workers) {
            Ok(waiters) => waiters,
            Err(error) => {
                pending_error = Some(LoaderError::Configuration(error));
                M::new_waiters(0).expect("zero waiter state is always allocatable")
            }
        };

        let mut pool = None;
        if pending_error.is_none() {
            if self.configuration.persistent_workers {
                if self.persistent_pool.is_none() {
                    match StreamWorkerPool::new(
                        Arc::clone(&self.factory),
                        Arc::clone(&self.transform_factory),
                        Arc::clone(&self.worker_init),
                        self.configuration,
                        generation,
                        self.outstanding_capacity,
                        reassembly_allocation_slots,
                        self.footprint,
                    ) {
                        Ok(created) => self.persistent_pool = Some(created),
                        Err(error) => pending_error = Some(LoaderError::Configuration(error)),
                    }
                }
                if pending_error.is_none() {
                    let persistent = self
                        .persistent_pool
                        .as_mut()
                        .expect("persistent stream pool was created");
                    if persistent.start_generation(run_context.clone()).is_err() {
                        pending_error = Some(LoaderError::ChannelClosed { batch: 0 });
                    } else {
                        pool = Some(StreamIteratorPool::Persistent(persistent));
                    }
                }
            } else {
                match StreamWorkerPool::new(
                    Arc::clone(&self.factory),
                    Arc::clone(&self.transform_factory),
                    Arc::clone(&self.worker_init),
                    self.configuration,
                    generation,
                    self.outstanding_capacity,
                    reassembly_allocation_slots,
                    self.footprint,
                ) {
                    Ok(mut created) => {
                        if created.start_generation(run_context.clone()).is_err() {
                            pending_error = Some(LoaderError::ChannelClosed { batch: 0 });
                        } else {
                            pool = Some(StreamIteratorPool::Owned(created));
                        }
                    }
                    Err(error) => pending_error = Some(LoaderError::Configuration(error)),
                }
            }
        }

        StreamLoaderIter {
            collator: &mut self.collator,
            pool,
            run_context,
            generation,
            ordered: self.configuration.ordered,
            timeout: self.configuration.timeout,
            batch_size: self.configuration.batch_size,
            drop_last: self.configuration.drop_last,
            expected_records: self.exact_len,
            reassembly_capacity: self.outstanding_capacity,
            completed: ordered_reassembly,
            partial,
            next_sequence: 0,
            next_batch: 0,
            pending_error,
            source_complete: false,
            exhausted: false,
            front_waiters,
            pin_memory_status: self.pin_memory_status,
            marker: PhantomData,
        }
    }
}

/// One borrowing iterator generation from a [`StreamDataLoader`].
#[allow(private_bounds)]
pub struct StreamLoaderIter<'a, S, C, F, I, M = MemoryDisabled, N = PinDisabled>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    collator: &'a mut C,
    pool: Option<StreamIteratorPool<'a, S, F, I, M>>,
    run_context: WorkerRunContext,
    generation: u64,
    ordered: bool,
    timeout: Option<Duration>,
    batch_size: usize,
    drop_last: bool,
    expected_records: Option<usize>,
    reassembly_capacity: usize,
    completed: &'a mut Vec<ReassemblyEntry<TransformOutput<S, F>, M::Permit>>,
    partial: Vec<TransformOutput<S, F>>,
    next_sequence: u64,
    next_batch: u64,
    pending_error: Option<StreamLoaderError<S, C, F, I>>,
    source_complete: bool,
    exhausted: bool,
    front_waiters: M::Waiters,
    pin_memory_status: PinMemoryStatus,
    marker: PhantomData<&'a mut (S, C, F, I, M, N)>,
}

#[allow(private_bounds)]
impl<S, C, F, I, M, N> StreamLoaderIter<'_, S, C, F, I, M, N>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy + Send + 'static,
    N: StreamPinPolicy<S, C, F>,
{
    fn return_credit(&mut self, worker: usize) -> std::result::Result<(), ()> {
        self.pool.as_ref().ok_or(())?.pool().return_credit(worker)
    }

    fn disarm_deadline(&self) {
        self.run_context.deadline.disarm();
    }

    fn set_front_waiter(&mut self, worker: usize, sequence: Option<u64>) {
        if let Some(waiters) = M::waiters_mut(&mut self.front_waiters) {
            waiters[worker] = sequence;
        }
    }

    fn close(&mut self, poisoned: bool) {
        self.exhausted = true;
        self.run_context.cancel();
        let mut credit_failed = false;
        while let Some((_, buffered)) = self.completed.pop() {
            credit_failed |= self.return_credit(buffered.worker).is_err();
        }
        if let Some(mut pool) = self.pool.take() {
            let persistent = pool.is_persistent();
            let quiesce_failed = persistent && pool.pool_mut().quiesce(self.generation).is_err();
            if poisoned || credit_failed || quiesce_failed {
                pool.pool_mut().poison();
            }
        }
        self.partial.clear();
    }

    fn protocol_error(
        &mut self,
        sequence: Option<u64>,
        reason: impl Into<String>,
    ) -> StreamLoaderError<S, C, F, I> {
        self.disarm_deadline();
        self.close(false);
        LoaderError::StreamProtocol {
            sequence,
            reason: reason.into(),
        }
    }

    fn consume(
        &mut self,
        buffered: BufferedRecord<TransformOutput<S, F>, M::Permit>,
    ) -> std::result::Result<bool, StreamLoaderError<S, C, F, I>> {
        let BufferedRecord {
            worker,
            record,
            permit,
        } = buffered;
        let next_sequence = if self.ordered {
            let Some(next) = record
                .sequence
                .expect("ordered records are validated")
                .checked_next()
            else {
                self.run_context.cancel();
                if self.return_credit(worker).is_err() {
                    self.close(true);
                    return Err(LoaderError::ChannelClosed {
                        batch: self.next_batch,
                    });
                }
                return Err(self.protocol_error(
                    Some(self.next_sequence),
                    "stream sequence identifier overflowed",
                ));
            };
            Some(next.get())
        } else {
            None
        };
        if self.return_credit(worker).is_err() {
            self.close(true);
            return Err(LoaderError::ChannelClosed {
                batch: self.next_batch,
            });
        }
        if let Some(next_sequence) = next_sequence {
            self.next_sequence = next_sequence;
        }
        drop(permit);
        self.partial.push(record.sample);
        Ok(self.partial.len() == self.batch_size)
    }

    fn finish_batch(&mut self) -> std::result::Result<C::Batch, StreamLoaderError<S, C, F, I>> {
        let mut replacement = Vec::new();
        if replacement.try_reserve_exact(self.batch_size).is_err() {
            self.close(false);
            return Err(LoaderError::Configuration(invalid_configuration(
                "batch_size",
                "stream batch allocation is unavailable",
            )));
        }
        let samples = std::mem::replace(&mut self.partial, replacement);
        let batch = self.next_batch;
        let result = match catch_unwind(AssertUnwindSafe(|| self.collator.collate(samples))) {
            Ok(result) => result.map_err(|source| LoaderError::Pipeline {
                batch: Some(batch),
                worker: None,
                source: PipelineError::Collate(source),
            }),
            Err(_) => Err(LoaderError::CoordinatorPanic {
                stage: "stream collation",
                batch: Some(batch),
            }),
        }
        .and_then(|value| match self.pin_memory_status {
            PinMemoryStatus::Enabled(device) => {
                N::pin(value, device).map_err(|source| LoaderError::PinMemory { batch, source })
            }
            PinMemoryStatus::Disabled | PinMemoryStatus::DisabledNoAccelerator => Ok(value),
        });
        match result {
            Ok(batch) => match self.next_batch.checked_add(1) {
                Some(next) => {
                    self.next_batch = next;
                    self.disarm_deadline();
                    Ok(batch)
                }
                None => {
                    self.close(false);
                    Err(LoaderError::Configuration(invalid_configuration(
                        "batch_size",
                        "stream batch sequence overflowed",
                    )))
                }
            },
            Err(error) => {
                self.disarm_deadline();
                self.close(false);
                Err(error)
            }
        }
    }

    fn map_failure(
        &mut self,
        worker: usize,
        sequence: Option<u64>,
        logical_id: Option<u64>,
        failure: StreamFailure<S::Error, TransformFailure<S, F>, F::Error, I::Error>,
    ) -> StreamLoaderError<S, C, F, I> {
        let batch = Some(self.next_batch);
        match failure {
            StreamFailure::Source(source) => LoaderError::StreamPipeline {
                batch,
                worker,
                sequence,
                logical_id,
                source: PipelineError::Source(source),
            },
            StreamFailure::Transform(source) => LoaderError::StreamPipeline {
                batch,
                worker,
                sequence,
                logical_id,
                source: PipelineError::Transform(source),
            },
            StreamFailure::TransformInit(source) => LoaderError::StreamPipeline {
                batch: None,
                worker,
                sequence: None,
                logical_id: None,
                source: PipelineError::TransformInit(source),
            },
            StreamFailure::WorkerInit(source) => LoaderError::StreamPipeline {
                batch: None,
                worker,
                sequence: None,
                logical_id: None,
                source: PipelineError::WorkerInit(source),
            },
            StreamFailure::MemoryLimit { limit, actual } => LoaderError::MemoryLimit {
                batch: None,
                worker: Some(worker),
                sequence,
                logical_id,
                limit,
                actual,
            },
            StreamFailure::Protocol { sequence, reason } => {
                LoaderError::StreamProtocol { sequence, reason }
            }
            StreamFailure::Panic => LoaderError::StreamWorkerPanic {
                worker,
                batch,
                sequence,
                logical_id,
            },
        }
    }
}

impl<S, C, F, I, M, N> Iterator for StreamLoaderIter<'_, S, C, F, I, M, N>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
    M: StreamMemoryPolicy + Send + 'static,
    N: StreamPinPolicy<S, C, F>,
{
    type Item = std::result::Result<C::Batch, StreamLoaderError<S, C, F, I>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        if let Some(error) = self.pending_error.take() {
            self.close(false);
            return Some(Err(error));
        }
        if self.source_complete {
            self.close(false);
            return None;
        }
        let mut deadline_armed = false;
        loop {
            if self.ordered
                && let Some(index) = self
                    .completed
                    .iter()
                    .position(|(sequence, _)| *sequence == self.next_sequence)
            {
                let (_, buffered) = self.completed.swap_remove(index);
                match self.consume(buffered) {
                    Ok(true) => return Some(self.finish_batch()),
                    Ok(false) => continue,
                    Err(error) => return Some(Err(error)),
                }
            }

            if self.ordered && self.completed.len() == self.reassembly_capacity {
                let expected = self.next_sequence;
                let found = self
                    .completed
                    .iter()
                    .map(|(sequence, _)| *sequence)
                    .min()
                    .expect("a full ordered reassembly window is nonempty");
                return Some(Err(self.protocol_error(
                    Some(expected),
                    format!(
                        "missing sequence {expected}; bounded reassembly window is full before buffered sequence {found}"
                    ),
                )));
            }

            if self.ordered
                && self.run_context.byte_budget.is_some()
                && let Some(pool) = self.pool.as_ref()
                && let Some(front_waiters) = M::waiters(&self.front_waiters)
                && front_waiters
                    .iter()
                    .enumerate()
                    .any(|(worker, _)| !pool.pool().terminal.get(worker).copied().unwrap_or(false))
                && front_waiters.iter().enumerate().all(|(worker, sequence)| {
                    pool.pool().terminal.get(worker).copied().unwrap_or(false)
                        || sequence.is_some_and(|sequence| sequence > self.next_sequence)
                })
            {
                let expected = self.next_sequence;
                let found = M::waiters(&self.front_waiters)
                    .expect("enabled byte accounting has waiter state")
                    .iter()
                    .flatten()
                    .copied()
                    .min()
                    .expect("a nonterminal byte-budget worker has a front waiter");
                return Some(Err(self.protocol_error(
                    Some(expected),
                    format!(
                        "missing sequence {expected}; all active byte-budget workers are waiting at sequence {found} or later"
                    ),
                )));
            }

            if self
                .pool
                .as_ref()
                .is_some_and(|pool| pool.pool().all_terminal())
            {
                if self.ordered
                    && let Some(found) = self.completed.iter().map(|(sequence, _)| *sequence).min()
                {
                    let expected = self.next_sequence;
                    return Some(Err(self.protocol_error(
                        Some(expected),
                        format!("missing sequence {expected} before buffered sequence {found}"),
                    )));
                }
                if self.ordered
                    && let Some(expected) = self.expected_records
                    && self.next_sequence != expected as u64
                {
                    let missing = self.next_sequence.min(expected as u64);
                    return Some(Err(self.protocol_error(
                        Some(missing),
                        format!(
                            "stream ended after {} ordered records, expected {expected}",
                            self.next_sequence
                        ),
                    )));
                }
                self.source_complete = true;
                if self.partial.is_empty() || self.drop_last {
                    self.close(false);
                    return None;
                }
                return Some(self.finish_batch());
            }

            if !deadline_armed && let Some(timeout) = self.timeout {
                self.run_context.deadline.arm(timeout);
                deadline_armed = true;
            }
            let remaining = self
                .timeout
                .and_then(|_| self.run_context.deadline.remaining());
            let received = self
                .pool
                .as_ref()
                .expect("active stream iterator has a pool")
                .pool()
                .receive(&self.run_context, remaining);
            let completion = match received {
                StreamReceive::Completion(completion) => completion,
                StreamReceive::Timeout => {
                    let batch = self.next_batch;
                    self.run_context.cancel();
                    self.disarm_deadline();
                    self.close(false);
                    return Some(Err(LoaderError::Timeout { batch }));
                }
                StreamReceive::Cancelled => {
                    self.disarm_deadline();
                    self.close(false);
                    return Some(Err(LoaderError::Cancelled));
                }
                StreamReceive::Closed => {
                    let batch = self.next_batch;
                    self.disarm_deadline();
                    self.close(true);
                    return Some(Err(LoaderError::ChannelClosed { batch }));
                }
            };
            if completion.generation != self.generation {
                if completion.holds_credit && self.return_credit(completion.worker).is_err() {
                    self.close(true);
                    return Some(Err(LoaderError::ChannelClosed {
                        batch: self.next_batch,
                    }));
                }
                continue;
            }

            match completion.result {
                Err(failure) => {
                    self.set_front_waiter(completion.worker, None);
                    self.run_context.cancel();
                    let fatal = matches!(
                        &failure,
                        StreamFailure::TransformInit(_)
                            | StreamFailure::WorkerInit(_)
                            | StreamFailure::Panic
                    );
                    if self
                        .pool
                        .as_mut()
                        .expect("stream pool")
                        .pool_mut()
                        .account_parts(completion.worker, completion.holds_credit, fatal, fatal)
                        .is_err()
                    {
                        self.close(true);
                        return Some(Err(LoaderError::ChannelClosed {
                            batch: self.next_batch,
                        }));
                    }
                    let error = self.map_failure(
                        completion.worker,
                        completion.sequence,
                        completion.logical_id,
                        failure,
                    );
                    self.disarm_deadline();
                    self.close(fatal);
                    return Some(Err(error));
                }
                Ok(StreamMessage::End) => {
                    self.set_front_waiter(completion.worker, None);
                    if self
                        .pool
                        .as_mut()
                        .expect("stream pool")
                        .pool_mut()
                        .account_parts(completion.worker, completion.holds_credit, true, false)
                        .is_err()
                    {
                        self.close(true);
                        return Some(Err(LoaderError::ChannelClosed {
                            batch: self.next_batch,
                        }));
                    }
                }
                Ok(StreamMessage::Waiting { sequence }) => {
                    self.set_front_waiter(completion.worker, Some(sequence));
                }
                Ok(StreamMessage::Record(record, permit)) => {
                    self.set_front_waiter(completion.worker, None);
                    let buffered = BufferedRecord {
                        worker: completion.worker,
                        record,
                        permit,
                    };
                    if !self.ordered {
                        match self.consume(buffered) {
                            Ok(true) => return Some(self.finish_batch()),
                            Ok(false) => {}
                            Err(error) => return Some(Err(error)),
                        }
                        continue;
                    }
                    let Some(sequence) = buffered.record.sequence else {
                        self.run_context.cancel();
                        if self.return_credit(buffered.worker).is_err() {
                            self.close(true);
                            return Some(Err(LoaderError::ChannelClosed {
                                batch: self.next_batch,
                            }));
                        }
                        return Some(Err(self.protocol_error(
                            None,
                            "ordered stream record did not provide a sequence identifier",
                        )));
                    };
                    let sequence = sequence.get();
                    if sequence < self.next_sequence {
                        self.run_context.cancel();
                        if self.return_credit(buffered.worker).is_err() {
                            self.close(true);
                            return Some(Err(LoaderError::ChannelClosed {
                                batch: self.next_batch,
                            }));
                        }
                        return Some(Err(self.protocol_error(
                            Some(sequence),
                            format!(
                                "duplicate or past sequence {sequence}; next expected is {}",
                                self.next_sequence
                            ),
                        )));
                    }
                    if self
                        .completed
                        .iter()
                        .any(|(buffered_sequence, _)| *buffered_sequence == sequence)
                    {
                        self.run_context.cancel();
                        if self.return_credit(buffered.worker).is_err() {
                            self.close(true);
                            return Some(Err(LoaderError::ChannelClosed {
                                batch: self.next_batch,
                            }));
                        }
                        return Some(Err(self.protocol_error(
                            Some(sequence),
                            format!("duplicate buffered sequence {sequence}"),
                        )));
                    }
                    self.completed.push((sequence, buffered));
                }
            }
        }
    }
}

impl<S, C, F, I, M, N> Drop for StreamLoaderIter<'_, S, C, F, I, M, N>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
    M: StreamMemoryPolicy,
{
    fn drop(&mut self) {
        self.run_context.cancel();
        let mut failed = false;
        while let Some((_, buffered)) = self.completed.pop() {
            failed |= self
                .pool
                .as_ref()
                .is_none_or(|pool| pool.pool().return_credit(buffered.worker).is_err());
        }
        if let Some(mut pool) = self.pool.take()
            && pool.is_persistent()
            && (failed || pool.pool_mut().quiesce(self.generation).is_err())
        {
            pool.pool_mut().poison();
        }
    }
}

fn invalid_configuration(field: &'static str, reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    type TestCompletion =
        StreamCompletion<usize, (), Infallible, Infallible, Infallible, Infallible>;

    fn ready_completion(
        result: std::result::Result<
            StreamMessage<usize, ()>,
            StreamFailure<Infallible, Infallible, Infallible, Infallible>,
        >,
    ) -> TestCompletion {
        StreamCompletion {
            worker: 0,
            generation: 0,
            sequence: None,
            logical_id: None,
            holds_credit: false,
            result,
        }
    }

    fn receive_after_expiry(completion: TestCompletion) -> TestCompletion {
        let (sender, receiver) = bounded(1);
        sender.send(completion).unwrap();
        let context = WorkerRunContext::new(0, 0, 0);
        context.deadline.arm(Duration::ZERO);
        let shutdown = CancellationToken::new();
        match receive_ready_first(&receiver, &context, &shutdown, Some(Duration::ZERO)) {
            ReadyFirst::Completion(completion) => completion,
            ReadyFirst::Timeout => panic!("ready completion lost to expired deadline"),
            ReadyFirst::Cancelled => panic!("ready completion lost to cancellation"),
            ReadyFirst::Closed => panic!("ready completion lost to channel close"),
        }
    }

    #[test]
    fn queued_error_and_end_beat_an_expired_receive_deadline() {
        let error = receive_after_expiry(ready_completion(Err(StreamFailure::Panic)));
        assert!(matches!(error.result, Err(StreamFailure::Panic)));

        let end = receive_after_expiry(ready_completion(Ok(StreamMessage::End)));
        assert!(matches!(end.result, Ok(StreamMessage::End)));
    }
}
