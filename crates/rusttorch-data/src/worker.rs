use std::{
    cell::UnsafeCell,
    mem::{MaybeUninit, size_of},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, atomic::AtomicUsize},
    thread::{self, JoinHandle},
    time::Duration,
};

use crossbeam_channel::{
    Receiver, RecvError, Sender, TryRecvError, TrySendError, after, bounded, select,
};
use rusttorch_core::{Result, RustTorchError};

use crate::memory::{BudgetError, ByteBudget, BytePermit};
use crate::{
    CancellationToken, Dataset, Deadline, TaskContext, Transform, TransformFactory, WorkerContext,
    WorkerInfo, WorkerInit, with_worker_info,
};

pub(crate) struct WorkerTask {
    pub(crate) generation: u64,
    pub(crate) batch_sequence: u64,
    pub(crate) logical_samples: Vec<u64>,
    pub(crate) indices: Vec<usize>,
}

pub(crate) enum WorkerSubmit {
    Submitted,
    Full(WorkerTask),
    Closed,
}

pub(crate) struct WorkerBatch<T> {
    pub(crate) generation: u64,
    pub(crate) batch_sequence: u64,
    pub(crate) samples: Vec<T>,
    pub(crate) permit: Option<BytePermit>,
}

pub(crate) enum WorkerFailure<DE, TE, FE, IE> {
    Dataset(DE),
    Transform(TE),
    TransformInit(FE),
    WorkerInit(IE),
    InvalidBatchCardinality { expected: usize, actual: usize },
    MemoryLimit { limit: usize, actual: usize },
    Panic,
}

pub(crate) enum WorkerMessage<T> {
    Batch(WorkerBatch<T>),
    Quiesced,
}

pub(crate) struct WorkerCompletion<T, DE, TE, FE, IE> {
    pub(crate) worker: usize,
    pub(crate) generation: u64,
    pub(crate) batch_sequence: Option<u64>,
    pub(crate) result: std::result::Result<WorkerMessage<T>, WorkerFailure<DE, TE, FE, IE>>,
}

type Completion<D, F, I> = WorkerCompletion<
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Output,
    <D as Dataset>::Error,
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Error,
    <F as TransformFactory<<D as Dataset>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;

type Failure<D, F, I> = WorkerFailure<
    <D as Dataset>::Error,
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Error,
    <F as TransformFactory<<D as Dataset>::Sample>>::Error,
    <I as WorkerInit>::Error,
>;

type WorkerFootprint<D, F> = fn(
    &[<<F as TransformFactory<<D as Dataset>::Sample>>::Transform as Transform<
        <D as Dataset>::Sample,
    >>::Output],
) -> usize;

const MAX_WORKER_QUEUE_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;
const CROSSBEAM_CHANNEL_CONTROL_BLOCK_ALLOWANCE_BYTES: usize = 2 * 1024;

#[allow(dead_code)]
#[repr(C)]
struct ChannelSlot<T> {
    stamp: AtomicUsize,
    message: UnsafeCell<MaybeUninit<T>>,
}

#[derive(Clone)]
pub(crate) struct WorkerRunContext {
    pub(crate) generation: u64,
    pub(crate) loader_seed: u64,
    pub(crate) epoch: u64,
    pub(crate) cancellation: CancellationToken,
    pub(crate) deadline: Deadline,
    pub(crate) byte_budget: Option<Arc<ByteBudget>>,
}

impl WorkerRunContext {
    pub(crate) fn new(generation: u64, loader_seed: u64, epoch: u64) -> Self {
        let cancellation = CancellationToken::new();
        let deadline = cancellation.paired_deadline();
        Self {
            generation,
            loader_seed,
            epoch,
            cancellation,
            deadline,
            byte_budget: None,
        }
    }

    pub(crate) fn with_byte_budget(mut self, byte_budget: Option<Arc<ByteBudget>>) -> Self {
        self.byte_budget = byte_budget;
        self
    }

    pub(crate) fn cancel(&self) {
        self.cancellation.cancel();
        if let Some(byte_budget) = &self.byte_budget {
            byte_budget.cancel();
        }
    }

    fn worker_context(&self, info: WorkerInfo) -> WorkerContext {
        WorkerContext::new(info, self.cancellation.clone(), self.deadline.clone())
    }
}

enum WorkerControl {
    Begin(WorkerRunContext),
}

pub(crate) enum WorkerReceive<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    Completion(Completion<D, F, I>),
    Timeout,
    Cancelled,
    Closed,
}

pub(crate) struct WorkerPool<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    controls: Vec<Sender<WorkerControl>>,
    tasks: Vec<Sender<WorkerTask>>,
    results: Option<Receiver<Completion<D, F, I>>>,
    handles: Vec<JoinHandle<()>>,
    shutdown: CancellationToken,
    active: Option<WorkerRunContext>,
    poisoned: bool,
}

pub(crate) struct WorkerPoolConfiguration {
    pub(crate) workers: usize,
    pub(crate) prefetch_factor: usize,
    pub(crate) result_capacity: usize,
    pub(crate) seed_generation: u64,
    pub(crate) loader_seed: u64,
    pub(crate) rank: usize,
}

impl<D, F, I> WorkerPool<D, F, I>
where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
{
    pub(crate) fn new(
        dataset: Arc<D>,
        factory: Arc<F>,
        initializer: Arc<I>,
        configuration: WorkerPoolConfiguration,
        footprint: Option<WorkerFootprint<D, F>>,
        ordered: bool,
    ) -> Result<Self> {
        validate_worker_pool_capacity::<D, F, I>(
            configuration.workers,
            configuration.prefetch_factor,
            configuration.result_capacity,
        )?;
        preflight_channel_storage::<WorkerTask>(configuration.result_capacity)?;
        preflight_channel_storage::<WorkerControl>(configuration.workers)?;
        preflight_channel_storage::<Completion<D, F, I>>(configuration.result_capacity)?;

        let mut infos = Vec::new();
        reserve_exact(&mut infos, configuration.workers, "workers")?;
        for id in 0..configuration.workers {
            infos.push(WorkerInfo::from_loader_seed(
                id,
                configuration.workers,
                configuration.loader_seed,
                configuration.rank,
                configuration.seed_generation,
            )?);
        }

        let (result_sender, result_receiver) =
            bounded_checked(configuration.result_capacity, "worker result channel")?;
        let mut task_lanes = Vec::new();
        let mut control_lanes = Vec::new();
        reserve_exact(
            &mut task_lanes,
            configuration.workers,
            "worker task channels",
        )?;
        reserve_exact(
            &mut control_lanes,
            configuration.workers,
            "worker control channels",
        )?;
        for _ in 0..configuration.workers {
            task_lanes.push(bounded_checked(
                configuration.prefetch_factor,
                "worker task channel",
            )?);
            control_lanes.push(bounded_checked(1, "worker control channel")?);
        }

        let shutdown = CancellationToken::new();
        let mut pool = Self {
            controls: Vec::new(),
            tasks: Vec::new(),
            results: Some(result_receiver),
            handles: Vec::new(),
            shutdown: shutdown.clone(),
            active: None,
            poisoned: false,
        };
        reserve_exact(&mut pool.controls, configuration.workers, "control senders")?;
        reserve_exact(&mut pool.tasks, configuration.workers, "task senders")?;
        reserve_exact(&mut pool.handles, configuration.workers, "worker handles")?;

        for ((info, (task_sender, task_receiver)), (control_sender, control_receiver)) in
            infos.into_iter().zip(task_lanes).zip(control_lanes)
        {
            pool.tasks.push(task_sender);
            pool.controls.push(control_sender);
            let dataset = Arc::clone(&dataset);
            let factory = Arc::clone(&factory);
            let initializer = Arc::clone(&initializer);
            let results = result_sender.clone();
            let shutdown = shutdown.clone();
            let initialization_generation = configuration.seed_generation;
            let handle = thread::Builder::new()
                .name(format!("rusttorch-data-worker-{}", info.id))
                .spawn(move || {
                    run_worker(
                        info,
                        initialization_generation,
                        dataset,
                        factory,
                        initializer,
                        control_receiver,
                        task_receiver,
                        results,
                        shutdown,
                        footprint,
                        ordered,
                    );
                })
                .map_err(|error| RustTorchError::BackendUnavailable {
                    backend: "data loader worker threads",
                    reason: error.to_string(),
                })?;
            pool.handles.push(handle);
        }
        drop(result_sender);
        Ok(pool)
    }
}

impl<D, F, I> WorkerPool<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    pub(crate) fn workers(&self) -> usize {
        self.tasks.len()
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub(crate) fn start_generation(
        &mut self,
        context: WorkerRunContext,
    ) -> std::result::Result<(), ()> {
        if self.poisoned || self.active.is_some() || self.shutdown.is_cancelled() {
            return Err(());
        }
        self.active = Some(context.clone());
        for control in &self.controls {
            if control.send(WorkerControl::Begin(context.clone())).is_err() {
                self.poisoned = true;
                self.shutdown();
                return Err(());
            }
        }
        Ok(())
    }

    pub(crate) fn submit(&self, worker: usize, task: WorkerTask) -> WorkerSubmit {
        let Some(active) = self.active.as_ref() else {
            return WorkerSubmit::Closed;
        };
        if active.cancellation.is_cancelled() || self.shutdown.is_cancelled() {
            return WorkerSubmit::Closed;
        }
        match self.tasks[worker].try_send(task) {
            Ok(()) => WorkerSubmit::Submitted,
            Err(TrySendError::Full(task)) => WorkerSubmit::Full(task),
            Err(TrySendError::Disconnected(_)) => WorkerSubmit::Closed,
        }
    }

    pub(crate) fn receive(
        &self,
        context: &WorkerRunContext,
        timeout: Option<Duration>,
    ) -> WorkerReceive<D, F, I> {
        let results = self
            .results
            .as_ref()
            .expect("worker result receiver exists before shutdown");
        match results.try_recv() {
            Ok(completion) => return WorkerReceive::Completion(completion),
            Err(TryRecvError::Disconnected) => return WorkerReceive::Closed,
            Err(TryRecvError::Empty) => {}
        }
        if let Some(timeout) = timeout {
            let timer = after(timeout);
            select! {
                recv(results) -> result => map_receive(result),
                recv(context.cancellation.signal()) -> _ => {
                    match results.try_recv() {
                        Ok(completion) => WorkerReceive::Completion(completion),
                        Err(TryRecvError::Empty) => WorkerReceive::Cancelled,
                        Err(TryRecvError::Disconnected) => WorkerReceive::Closed,
                    }
                },
                recv(self.shutdown.signal()) -> _ => WorkerReceive::Closed,
                recv(timer) -> _ => {
                    match results.try_recv() {
                        Ok(completion) => WorkerReceive::Completion(completion),
                        Err(TryRecvError::Empty) => WorkerReceive::Timeout,
                        Err(TryRecvError::Disconnected) => WorkerReceive::Closed,
                    }
                },
            }
        } else {
            select! {
                recv(results) -> result => map_receive(result),
                recv(context.cancellation.signal()) -> _ => {
                    match results.try_recv() {
                        Ok(completion) => WorkerReceive::Completion(completion),
                        Err(TryRecvError::Empty) => WorkerReceive::Cancelled,
                        Err(TryRecvError::Disconnected) => WorkerReceive::Closed,
                    }
                },
                recv(self.shutdown.signal()) -> _ => WorkerReceive::Closed,
            }
        }
    }

    pub(crate) fn quiesce(&mut self, generation: u64) -> std::result::Result<(), ()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        active.cancel();
        let worker_count = self.workers();
        let mut quiesced = Vec::new();
        if quiesced.try_reserve_exact(worker_count).is_err() {
            self.poisoned = true;
            self.shutdown();
            return Err(());
        }
        quiesced.resize(worker_count, false);
        let mut quiesced_count = 0;
        let mut fatal_exit = false;
        while quiesced_count < worker_count {
            let received = self.results.as_ref().ok_or(())?.recv();
            match received {
                Ok(completion) if completion.generation != generation => {}
                Ok(WorkerCompletion {
                    worker,
                    result: Ok(WorkerMessage::Quiesced),
                    ..
                }) => {
                    if worker >= worker_count {
                        self.poisoned = true;
                        self.shutdown();
                        return Err(());
                    }
                    if !quiesced[worker] {
                        quiesced[worker] = true;
                        quiesced_count += 1;
                    }
                }
                Ok(WorkerCompletion {
                    worker,
                    result:
                        Err(
                            WorkerFailure::TransformInit(_)
                            | WorkerFailure::WorkerInit(_)
                            | WorkerFailure::Panic,
                        ),
                    ..
                }) => {
                    if worker >= worker_count {
                        self.poisoned = true;
                        self.shutdown();
                        return Err(());
                    }
                    fatal_exit = true;
                    if !quiesced[worker] {
                        quiesced[worker] = true;
                        quiesced_count += 1;
                    }
                }
                Ok(_) => {}
                Err(_) => {
                    self.poisoned = true;
                    self.shutdown();
                    return Err(());
                }
            }
        }
        if fatal_exit {
            self.poisoned = true;
            self.shutdown();
            return Err(());
        }
        while self
            .results
            .as_ref()
            .is_some_and(|results| results.try_recv().is_ok())
        {}
        Ok(())
    }

    pub(crate) fn poison(&mut self) {
        self.poisoned = true;
        self.shutdown();
    }

    pub(crate) fn shutdown(&mut self) {
        if let Some(active) = self.active.take() {
            active.cancel();
        }
        self.shutdown.cancel();
        self.controls.clear();
        self.tasks.clear();
        self.results.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn map_receive<D, F, I>(
    result: std::result::Result<Completion<D, F, I>, RecvError>,
) -> WorkerReceive<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    match result {
        Ok(completion) => WorkerReceive::Completion(completion),
        Err(_) => WorkerReceive::Closed,
    }
}

pub(crate) fn validate_worker_pool_capacity<D, F, I>(
    workers: usize,
    prefetch_factor: usize,
    outstanding: usize,
) -> Result<()>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    let expected_outstanding = workers
        .checked_mul(prefetch_factor)
        .ok_or_else(|| capacity_error("workers multiplied by prefetch_factor exceeds usize"))?;
    if expected_outstanding != outstanding {
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

    let slots_per_credit = size_of::<ChannelSlot<WorkerTask>>()
        .checked_add(size_of::<ChannelSlot<Completion<D, F, I>>>())
        .ok_or_else(|| capacity_error("bounded channel slot sizes exceed usize"))?;
    let credit_bytes = outstanding
        .checked_mul(slots_per_credit)
        .ok_or_else(|| capacity_error("bounded channel storage exceeds Rust allocation limits"))?;
    let control_bytes = workers
        .checked_mul(size_of::<ChannelSlot<WorkerControl>>())
        .ok_or_else(|| capacity_error("worker control storage exceeds Rust allocation limits"))?;
    let per_worker_bookkeeping = size_of::<WorkerInfo>()
        .checked_add(size_of::<(Sender<WorkerTask>, Receiver<WorkerTask>)>())
        .and_then(|bytes| {
            bytes.checked_add(size_of::<(Sender<WorkerControl>, Receiver<WorkerControl>)>())
        })
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<WorkerTask>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<WorkerControl>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<JoinHandle<()>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<bool>()))
        .ok_or_else(|| capacity_error("worker bookkeeping size exceeds usize"))?;
    let fixed_bookkeeping = size_of::<Vec<()>>()
        .checked_mul(6)
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<Completion<D, F, I>>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<Receiver<Completion<D, F, I>>>()));
    let vector_bytes = workers
        .checked_mul(per_worker_bookkeeping)
        .and_then(|bytes| fixed_bookkeeping.and_then(|fixed| bytes.checked_add(fixed)))
        .ok_or_else(|| capacity_error("worker vector storage exceeds Rust allocation limits"))?;
    let channel_count = workers
        .checked_mul(2)
        .and_then(|channels| channels.checked_add(1))
        .ok_or_else(|| capacity_error("worker channel count exceeds usize"))?;
    let control_block_bytes = channel_count
        .checked_mul(CROSSBEAM_CHANNEL_CONTROL_BLOCK_ALLOWANCE_BYTES)
        .ok_or_else(|| {
            capacity_error("channel control-block storage exceeds Rust allocation limits")
        })?;
    let aggregate_bytes = credit_bytes
        .checked_add(control_bytes)
        .and_then(|bytes| bytes.checked_add(vector_bytes))
        .and_then(|bytes| bytes.checked_add(control_block_bytes))
        .ok_or_else(|| {
            capacity_error("aggregate worker queue storage exceeds Rust allocation limits")
        })?;
    if aggregate_bytes > MAX_WORKER_QUEUE_ALLOCATION_BYTES {
        return Err(capacity_error(format!(
            "aggregate worker queue storage requires {aggregate_bytes} bytes, above the {MAX_WORKER_QUEUE_ALLOCATION_BYTES}-byte safety ceiling"
        )));
    }
    Ok(())
}

fn preflight_channel_storage<T>(total_capacity: usize) -> Result<()> {
    let mut reservation: Vec<MaybeUninit<ChannelSlot<T>>> = Vec::new();
    reservation
        .try_reserve_exact(total_capacity)
        .map_err(|error| {
            capacity_error(format!("bounded channel storage is unavailable: {error}"))
        })?;
    Ok(())
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

fn capacity_error(reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "prefetch_factor",
        reason: reason.into(),
    }
}

impl<D, F, I> Drop for WorkerPool<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker<D, F, I>(
    info: WorkerInfo,
    initialization_generation: u64,
    dataset: Arc<D>,
    factory: Arc<F>,
    initializer: Arc<I>,
    controls: Receiver<WorkerControl>,
    tasks: Receiver<WorkerTask>,
    results: Sender<Completion<D, F, I>>,
    shutdown: CancellationToken,
    footprint: Option<WorkerFootprint<D, F>>,
    ordered: bool,
) where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
{
    let lifecycle = WorkerContext::new(info, shutdown.clone(), shutdown.paired_deadline());
    let mut active = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        with_worker_info(info, || {
            if let Err(error) = initializer.initialize(&lifecycle) {
                send_pool_failure(
                    &results,
                    &shutdown,
                    info.id,
                    initialization_generation,
                    WorkerFailure::WorkerInit(error),
                );
                return true;
            }
            let mut transform = match factory.create(Some(&lifecycle)) {
                Ok(transform) => transform,
                Err(error) => {
                    send_pool_failure(
                        &results,
                        &shutdown,
                        info.id,
                        initialization_generation,
                        WorkerFailure::TransformInit(error),
                    );
                    return true;
                }
            };

            loop {
                let run = select! {
                    recv(controls) -> control => match control {
                        Ok(WorkerControl::Begin(run)) => run,
                        Err(_) => return false,
                    },
                    recv(shutdown.signal()) -> _ => return false,
                };
                active = Some((run.clone(), None));
                run_generation::<D, F, I>(
                    info,
                    &dataset,
                    &mut transform,
                    &tasks,
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
    let fatal_exit = match outcome {
        Ok(fatal_exit) => fatal_exit,
        Err(_) => {
            let (generation, batch_sequence) = active
                .as_ref()
                .map(|(run, batch)| (run.generation, *batch))
                .unwrap_or((initialization_generation, None));
            let _ = send_completion(
                &results,
                &shutdown,
                None,
                WorkerCompletion {
                    worker: info.id,
                    generation,
                    batch_sequence,
                    result: Err(WorkerFailure::Panic),
                },
            );
            true
        }
    };
    if fatal_exit {
        shutdown.wait_cancelled();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generation<D, F, I>(
    info: WorkerInfo,
    dataset: &D,
    transform: &mut F::Transform,
    tasks: &Receiver<WorkerTask>,
    results: &Sender<Completion<D, F, I>>,
    shutdown: &CancellationToken,
    run: &WorkerRunContext,
    active: &mut Option<(WorkerRunContext, Option<u64>)>,
    footprint: Option<WorkerFootprint<D, F>>,
    ordered: bool,
) where
    D: Dataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Output: Send + 'static,
    <F::Transform as Transform<D::Sample>>::Error: Send + 'static,
    F::Error: Send + 'static,
    I: WorkerInit + Send + Sync + 'static,
    I::Error: Send + 'static,
{
    let context = run.worker_context(info);
    loop {
        let task = select! {
            recv(tasks) -> task => match task {
                Ok(task) => task,
                Err(_) => return,
            },
            recv(run.cancellation.signal()) -> _ => break,
            recv(shutdown.signal()) -> _ => return,
        };
        if context.check().is_err() {
            break;
        }
        if task.generation != run.generation {
            continue;
        }
        active.as_mut().expect("active generation").1 = Some(task.batch_sequence);
        let expected = task.indices.len();
        let samples = match dataset.get_batch_with_context(&task.indices, &context) {
            Ok(samples) if context.check().is_ok() => samples,
            Ok(_) => break,
            Err(error) => {
                if send_generation_failure::<D, F, I>(
                    results,
                    shutdown,
                    run,
                    info.id,
                    Some(task.batch_sequence),
                    WorkerFailure::Dataset(error),
                ) {
                    run.cancel();
                }
                break;
            }
        };
        if samples.len() != expected {
            if send_generation_failure::<D, F, I>(
                results,
                shutdown,
                run,
                info.id,
                Some(task.batch_sequence),
                WorkerFailure::InvalidBatchCardinality {
                    expected,
                    actual: samples.len(),
                },
            ) {
                run.cancel();
            }
            break;
        }
        let mut transformed = Vec::with_capacity(samples.len());
        for (sample, logical_sample) in samples.into_iter().zip(task.logical_samples) {
            if context.check().is_err() {
                break;
            }
            let task_context = TaskContext {
                loader_seed: run.loader_seed,
                epoch: run.epoch,
                rank: info.rank,
                logical_sample,
                stage: 0,
                cancellation: run.cancellation.clone(),
                deadline: run.deadline.clone(),
            };
            match transform.transform(sample, &task_context) {
                Ok(sample) if context.check().is_ok() => transformed.push(sample),
                Ok(_) => break,
                Err(error) => {
                    if send_generation_failure::<D, F, I>(
                        results,
                        shutdown,
                        run,
                        info.id,
                        Some(task.batch_sequence),
                        WorkerFailure::Transform(error),
                    ) {
                        run.cancel();
                    }
                    break;
                }
            }
        }
        if transformed.len() != expected || context.check().is_err() {
            break;
        }
        let permit = if let Some(footprint) = footprint {
            let actual = footprint(&transformed);
            let budget = run
                .byte_budget
                .as_ref()
                .expect("enabled byte accounting has a generation budget");
            let acquired = if ordered {
                budget.acquire_ordered(task.batch_sequence, actual)
            } else {
                budget.acquire(actual)
            };
            match acquired {
                Ok(permit) => Some(permit),
                Err(BudgetError::Oversize { limit, actual }) => {
                    if send_generation_failure::<D, F, I>(
                        results,
                        shutdown,
                        run,
                        info.id,
                        Some(task.batch_sequence),
                        WorkerFailure::MemoryLimit { limit, actual },
                    ) {
                        run.cancel();
                    }
                    break;
                }
                Err(BudgetError::Cancelled) => break,
                Err(BudgetError::SequenceAlreadyAdmitted) => {
                    if send_generation_failure::<D, F, I>(
                        results,
                        shutdown,
                        run,
                        info.id,
                        Some(task.batch_sequence),
                        WorkerFailure::Panic,
                    ) {
                        run.cancel();
                    }
                    break;
                }
            }
        } else {
            None
        };
        let batch = WorkerBatch {
            generation: task.generation,
            batch_sequence: task.batch_sequence,
            samples: transformed,
            permit,
        };
        if !send_completion(
            results,
            shutdown,
            Some(&run.cancellation),
            WorkerCompletion {
                worker: info.id,
                generation: task.generation,
                batch_sequence: Some(task.batch_sequence),
                result: Ok(WorkerMessage::Batch(batch)),
            },
        ) {
            break;
        }
        active.as_mut().expect("active generation").1 = None;
    }

    while let Ok(task) = tasks.try_recv() {
        if task.generation != run.generation {
            break;
        }
    }
    let _ = send_completion(
        results,
        shutdown,
        None,
        WorkerCompletion {
            worker: info.id,
            generation: run.generation,
            batch_sequence: None,
            result: Ok(WorkerMessage::Quiesced),
        },
    );
}

fn send_generation_failure<D, F, I>(
    results: &Sender<Completion<D, F, I>>,
    shutdown: &CancellationToken,
    run: &WorkerRunContext,
    worker: usize,
    batch_sequence: Option<u64>,
    failure: Failure<D, F, I>,
) -> bool
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    send_completion(
        results,
        shutdown,
        None,
        WorkerCompletion {
            worker,
            generation: run.generation,
            batch_sequence,
            result: Err(failure),
        },
    )
}

fn send_pool_failure<T, DE, TE, FE, IE>(
    results: &Sender<WorkerCompletion<T, DE, TE, FE, IE>>,
    shutdown: &CancellationToken,
    worker: usize,
    generation: u64,
    failure: WorkerFailure<DE, TE, FE, IE>,
) {
    let _ = send_completion(
        results,
        shutdown,
        None,
        WorkerCompletion {
            worker,
            generation,
            batch_sequence: None,
            result: Err(failure),
        },
    );
}

fn send_completion<T, DE, TE, FE, IE>(
    results: &Sender<WorkerCompletion<T, DE, TE, FE, IE>>,
    shutdown: &CancellationToken,
    cancellation: Option<&CancellationToken>,
    completion: WorkerCompletion<T, DE, TE, FE, IE>,
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
