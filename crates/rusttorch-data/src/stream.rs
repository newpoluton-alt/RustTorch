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
    collections::BTreeMap,
    marker::PhantomData,
    mem::{MaybeUninit, size_of},
    num::NonZeroUsize,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, atomic::AtomicUsize},
    thread::{self, JoinHandle},
    time::Duration,
};

use crossbeam_channel::{Receiver, RecvError, Sender, TryRecvError, after, bounded, select};

use rusttorch_core::{Result, RustTorchError};

use crate::worker::WorkerRunContext;
use crate::{
    CancellationToken, CloneTransformFactory, Collate, Deadline, DefaultCollator,
    IdentityTransformFactory, LoaderError, NoWorkerInit, PipelineError, TaskContext, Transform,
    TransformFactory, WorkerContext, WorkerInfo, WorkerInit, with_worker_info,
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

type StreamCompletionFor<S, F, I> = StreamCompletion<
    TransformOutput<S, F>,
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
    Panic,
}

enum StreamMessage<T> {
    Record(WorkerRecord<T>),
    End,
}

struct StreamCompletion<T, SE, TE, FE, IE> {
    worker: usize,
    generation: u64,
    sequence: Option<u64>,
    logical_id: Option<u64>,
    holds_credit: bool,
    result: std::result::Result<StreamMessage<T>, StreamFailure<SE, TE, FE, IE>>,
}

enum StreamControl {
    Begin(WorkerRunContext),
}

enum StreamReceive<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    Completion(StreamCompletionFor<S, F, I>),
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

fn validate_stream_capacity<S, F, I>(
    workers: usize,
    prefetch_factor: usize,
    outstanding: usize,
    batch_size: usize,
) -> Result<()>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
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
        .checked_mul(size_of::<CapacitySlot<StreamCompletionFor<S, F, I>>>())
        .ok_or_else(|| capacity_error("stream result storage exceeds usize"))?;
    let control_bytes = workers
        .checked_mul(size_of::<CapacitySlot<StreamControl>>())
        .ok_or_else(|| capacity_error("stream control storage exceeds usize"))?;
    let batch_bytes = batch_size
        .checked_mul(size_of::<TransformOutput<S, F>>())
        .ok_or_else(|| capacity_error("stream batch storage exceeds usize"))?;
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
    let bookkeeping = workers
        .checked_mul(per_worker)
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
    preflight_channel_storage::<StreamCompletionFor<S, F, I>>(outstanding)?;
    preflight_vec::<TransformOutput<S, F>>(batch_size, "stream batch")?;
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

struct StreamWorkerPool<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    controls: Vec<Sender<StreamControl>>,
    credits: Vec<Sender<()>>,
    results: Option<Receiver<StreamCompletionFor<S, F, I>>>,
    handles: Vec<JoinHandle<()>>,
    shutdown: CancellationToken,
    active: Option<WorkerRunContext>,
    terminal: Vec<bool>,
    terminal_count: usize,
    fatal_exit: bool,
    poisoned: bool,
}

impl<S, F, I> StreamWorkerPool<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
{
    fn new(
        source_factory: Arc<S>,
        transform_factory: Arc<F>,
        initializer: Arc<I>,
        configuration: StreamConfiguration,
        generation: u64,
        outstanding: usize,
    ) -> Result<Self> {
        validate_stream_capacity::<S, F, I>(
            configuration.workers,
            configuration.prefetch_factor,
            outstanding,
            configuration.batch_size,
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
                    run_stream_worker::<S, F, I>(
                        info,
                        generation,
                        source_factory,
                        transform_factory,
                        initializer,
                        control_receiver,
                        credit_receiver,
                        results,
                        shutdown,
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

impl<S, F, I> StreamWorkerPool<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
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
    ) -> StreamReceive<S, F, I> {
        let results = self
            .results
            .as_ref()
            .expect("stream result receiver exists before shutdown");
        match results.try_recv() {
            Ok(completion) => return StreamReceive::Completion(completion),
            Err(TryRecvError::Disconnected) => return StreamReceive::Closed,
            Err(TryRecvError::Empty) => {}
        }
        if let Some(timeout) = timeout {
            let timer = after(timeout);
            select! {
                recv(results) -> result => map_stream_receive(result),
                recv(context.cancellation.signal()) -> _ => match results.try_recv() {
                    Ok(completion) => StreamReceive::Completion(completion),
                    Err(TryRecvError::Empty) => StreamReceive::Cancelled,
                    Err(TryRecvError::Disconnected) => StreamReceive::Closed,
                },
                recv(self.shutdown.signal()) -> _ => StreamReceive::Closed,
                recv(timer) -> _ => match results.try_recv() {
                    Ok(completion) => StreamReceive::Completion(completion),
                    Err(TryRecvError::Empty) => StreamReceive::Timeout,
                    Err(TryRecvError::Disconnected) => StreamReceive::Closed,
                },
            }
        } else {
            select! {
                recv(results) -> result => map_stream_receive(result),
                recv(context.cancellation.signal()) -> _ => match results.try_recv() {
                    Ok(completion) => StreamReceive::Completion(completion),
                    Err(TryRecvError::Empty) => StreamReceive::Cancelled,
                    Err(TryRecvError::Disconnected) => StreamReceive::Closed,
                },
                recv(self.shutdown.signal()) -> _ => StreamReceive::Closed,
            }
        }
    }

    fn account(
        &mut self,
        completion: &StreamCompletionFor<S, F, I>,
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
        active.cancellation.cancel();
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
            active.cancellation.cancel();
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

impl<S, F, I> Drop for StreamWorkerPool<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn map_stream_receive<S, F, I>(
    result: std::result::Result<StreamCompletionFor<S, F, I>, RecvError>,
) -> StreamReceive<S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    match result {
        Ok(completion) => StreamReceive::Completion(completion),
        Err(_) => StreamReceive::Closed,
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
fn run_stream_worker<S, F, I>(
    info: WorkerInfo,
    initialization_generation: u64,
    source_factory: Arc<S>,
    transform_factory: Arc<F>,
    initializer: Arc<I>,
    controls: Receiver<StreamControl>,
    credits: Receiver<()>,
    results: Sender<StreamCompletionFor<S, F, I>>,
    shutdown: CancellationToken,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
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
                run_stream_generation::<S, F, I>(
                    info,
                    &source_factory,
                    &mut transform,
                    &credits,
                    &results,
                    &shutdown,
                    &run,
                    &mut active,
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
fn run_stream_generation<S, F, I>(
    info: WorkerInfo,
    source_factory: &S,
    transform: &mut F::Transform,
    credits: &Receiver<()>,
    results: &Sender<StreamCompletionFor<S, F, I>>,
    shutdown: &CancellationToken,
    run: &WorkerRunContext,
    active: &mut Option<(WorkerRunContext, Option<u64>, Option<u64>, bool)>,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
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
        Ok(_) => {
            send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, false);
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
            run.cancellation.cancel();
            send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, false);
            return;
        }
    };

    loop {
        let acquired = select! {
            recv(credits) -> credit => credit.is_ok(),
            recv(run.cancellation.signal()) -> _ => false,
            recv(shutdown.signal()) -> _ => return,
        };
        if !acquired {
            send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, false);
            return;
        }
        active.as_mut().expect("active stream generation").3 = true;
        if context.check().is_err() {
            send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, true);
            active.as_mut().expect("active stream generation").3 = false;
            return;
        }

        let record = match source.next() {
            Some(Ok(record)) => {
                if context.check().is_err() {
                    send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, true);
                    active.as_mut().expect("active stream generation").3 = false;
                    return;
                }
                record
            }
            None => {
                send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, true);
                active.as_mut().expect("active stream generation").3 = false;
                return;
            }
            Some(Err(error)) => {
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
                run.cancellation.cancel();
                send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, false);
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
                send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, true);
                active.as_mut().expect("active stream generation").3 = false;
                return;
            }
            Err(error) => {
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
                run.cancellation.cancel();
                send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, false);
                return;
            }
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
                result: Ok(StreamMessage::Record(WorkerRecord {
                    sequence: record.sequence,
                    logical_id: record.logical_id,
                    sample,
                })),
            },
        );
        if sent {
            let active = active.as_mut().expect("active stream generation");
            active.1 = None;
            active.2 = None;
            active.3 = false;
        } else {
            send_stream_end::<S, F, I>(results, shutdown, info.id, run.generation, true);
            active.as_mut().expect("active stream generation").3 = false;
            return;
        }
    }
}

fn send_stream_end<S, F, I>(
    results: &Sender<StreamCompletionFor<S, F, I>>,
    shutdown: &CancellationToken,
    worker: usize,
    generation: u64,
    holds_credit: bool,
) where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
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
fn send_stream_failure<T, SE, TE, FE, IE>(
    results: &Sender<StreamCompletion<T, SE, TE, FE, IE>>,
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

fn send_stream_completion<T, SE, TE, FE, IE>(
    results: &Sender<StreamCompletion<T, SE, TE, FE, IE>>,
    shutdown: &CancellationToken,
    cancellation: Option<&CancellationToken>,
    completion: StreamCompletion<T, SE, TE, FE, IE>,
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
        }
    }
}

/// Builder for an explicitly sharded positive-worker stream loader.
pub struct StreamDataLoaderBuilder<
    S,
    C = DefaultCollator,
    F = IdentityTransformFactory,
    I = NoWorkerInit,
> {
    factory: S,
    collator: C,
    transform_factory: F,
    worker_init: I,
    configuration: StreamConfiguration,
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
        }
    }
}

impl<S, C, F, I> StreamDataLoaderBuilder<S, C, F, I> {
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
    pub fn prefetch_factor(mut self, factor: usize) -> Self {
        self.configuration.prefetch_factor = factor;
        self
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
    pub fn collate<C2>(self, collator: C2) -> StreamDataLoaderBuilder<S, C2, F, I> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
        }
    }

    /// Clones one transform template for each worker lifecycle.
    pub fn transform<T>(
        self,
        transform: T,
    ) -> StreamDataLoaderBuilder<S, C, CloneTransformFactory<T>, I> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: CloneTransformFactory::new(transform),
            worker_init: self.worker_init,
            configuration: self.configuration,
        }
    }

    /// Replaces worker transform construction.
    pub fn transform_factory<F2>(self, factory: F2) -> StreamDataLoaderBuilder<S, C, F2, I> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
        }
    }

    /// Replaces worker initialization.
    pub fn worker_init<I2>(self, worker_init: I2) -> StreamDataLoaderBuilder<S, C, F, I2> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init,
            configuration: self.configuration,
        }
    }

    /// Validates configuration and constructs the stream owner.
    pub fn build(self) -> Result<StreamDataLoader<S, C, F, I>>
    where
        S: WorkerSourceFactory,
        F: TransformFactory<S::Sample>,
        C: Collate<TransformOutput<S, F>>,
        I: WorkerInit,
    {
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
        validate_stream_capacity::<S, F, I>(
            self.configuration.workers,
            self.configuration.prefetch_factor,
            outstanding_capacity,
            self.configuration.batch_size,
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
        })
    }
}

/// Owned, re-iterable explicitly sharded stream loader.
pub struct StreamDataLoader<S, C, F = IdentityTransformFactory, I = NoWorkerInit>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
{
    factory: Arc<S>,
    collator: C,
    transform_factory: Arc<F>,
    worker_init: Arc<I>,
    configuration: StreamConfiguration,
    outstanding_capacity: usize,
    exact_len: Option<usize>,
    next_generation: u64,
    persistent_pool: Option<StreamWorkerPool<S, F, I>>,
}

impl<S, C, F, I> StreamDataLoader<S, C, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
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

struct BufferedRecord<T> {
    worker: usize,
    record: WorkerRecord<T>,
}

enum StreamIteratorPool<'a, S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    Owned(StreamWorkerPool<S, F, I>),
    Persistent(&'a mut StreamWorkerPool<S, F, I>),
}

impl<S, F, I> StreamIteratorPool<'_, S, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    I: WorkerInit,
{
    fn pool(&self) -> &StreamWorkerPool<S, F, I> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent(pool) => pool,
        }
    }

    fn pool_mut(&mut self) -> &mut StreamWorkerPool<S, F, I> {
        match self {
            Self::Owned(pool) => pool,
            Self::Persistent(pool) => pool,
        }
    }

    fn is_persistent(&self) -> bool {
        matches!(self, Self::Persistent(_))
    }
}

impl<S, C, F, I> StreamDataLoader<S, C, F, I>
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
{
    /// Starts a fresh explicitly sharded worker generation.
    pub fn iter(&mut self) -> StreamLoaderIter<'_, S, C, F, I> {
        let generation = self.next_generation;
        let run_context = WorkerRunContext::new(
            generation,
            self.configuration.loader_seed,
            self.configuration.epoch,
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
            completed: BTreeMap::new(),
            partial,
            next_sequence: 0,
            next_batch: 0,
            pending_error,
            source_complete: false,
            exhausted: false,
            marker: PhantomData,
        }
    }
}

/// One borrowing iterator generation from a [`StreamDataLoader`].
pub struct StreamLoaderIter<'a, S, C, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
{
    collator: &'a mut C,
    pool: Option<StreamIteratorPool<'a, S, F, I>>,
    run_context: WorkerRunContext,
    generation: u64,
    ordered: bool,
    timeout: Option<Duration>,
    batch_size: usize,
    drop_last: bool,
    expected_records: Option<usize>,
    completed: BTreeMap<u64, BufferedRecord<TransformOutput<S, F>>>,
    partial: Vec<TransformOutput<S, F>>,
    next_sequence: u64,
    next_batch: u64,
    pending_error: Option<StreamLoaderError<S, C, F, I>>,
    source_complete: bool,
    exhausted: bool,
    marker: PhantomData<&'a mut (S, C, F, I)>,
}

impl<S, C, F, I> StreamLoaderIter<'_, S, C, F, I>
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
{
    fn return_credit(&mut self, worker: usize) -> std::result::Result<(), ()> {
        self.pool.as_ref().ok_or(())?.pool().return_credit(worker)
    }

    fn disarm_deadline(&self) {
        self.run_context.deadline.disarm();
    }

    fn close(&mut self, poisoned: bool) {
        self.exhausted = true;
        let mut credit_failed = false;
        let workers = self
            .completed
            .values()
            .map(|buffered| buffered.worker)
            .collect::<Vec<_>>();
        self.completed.clear();
        for worker in workers {
            credit_failed |= self.return_credit(worker).is_err();
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
        buffered: BufferedRecord<TransformOutput<S, F>>,
    ) -> std::result::Result<bool, StreamLoaderError<S, C, F, I>> {
        if self.return_credit(buffered.worker).is_err() {
            self.close(true);
            return Err(LoaderError::ChannelClosed {
                batch: self.next_batch,
            });
        }
        if self.ordered {
            let Some(next) = buffered
                .record
                .sequence
                .expect("ordered records are validated")
                .checked_next()
            else {
                return Err(self.protocol_error(
                    Some(self.next_sequence),
                    "stream sequence identifier overflowed",
                ));
            };
            self.next_sequence = next.get();
        }
        self.partial.push(buffered.record.sample);
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
        };
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
            StreamFailure::Panic => LoaderError::StreamWorkerPanic {
                worker,
                batch,
                sequence,
                logical_id,
            },
        }
    }
}

impl<S, C, F, I> Iterator for StreamLoaderIter<'_, S, C, F, I>
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
                && let Some(buffered) = self.completed.remove(&self.next_sequence)
            {
                match self.consume(buffered) {
                    Ok(true) => return Some(self.finish_batch()),
                    Ok(false) => continue,
                    Err(error) => return Some(Err(error)),
                }
            }

            // A worker can observe the generation deadline and publish its end
            // marker before the coordinator's timed receive wakes. A completed
            // record still wins above, but an end marker must not turn an
            // expired generation into ordinary exhaustion.
            if deadline_armed && self.run_context.deadline.is_expired() {
                let batch = self.next_batch;
                self.run_context.cancellation.cancel();
                self.disarm_deadline();
                self.close(false);
                return Some(Err(LoaderError::Timeout { batch }));
            }

            if self
                .pool
                .as_ref()
                .is_some_and(|pool| pool.pool().all_terminal())
            {
                if self.ordered
                    && let Some((&found, _)) = self.completed.first_key_value()
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
                    self.run_context.cancellation.cancel();
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
                Ok(StreamMessage::Record(record)) => {
                    let buffered = BufferedRecord {
                        worker: completion.worker,
                        record,
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
                    if self.completed.contains_key(&sequence) {
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
                    self.completed.insert(sequence, buffered);
                }
            }
        }
    }
}

impl<S, C, F, I> Drop for StreamLoaderIter<'_, S, C, F, I>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
    I: WorkerInit,
{
    fn drop(&mut self) {
        let workers = self
            .completed
            .values()
            .map(|buffered| buffered.worker)
            .collect::<Vec<_>>();
        self.completed.clear();
        let mut failed = false;
        if let Some(pool) = self.pool.as_ref() {
            for worker in workers {
                failed |= pool.pool().return_credit(worker).is_err();
            }
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
