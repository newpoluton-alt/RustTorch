use std::{
    cell::UnsafeCell,
    mem::{MaybeUninit, size_of},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, atomic::AtomicUsize},
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender, bounded};
use rusttorch_core::{Result, RustTorchError};

use crate::{
    Dataset, TaskContext, Transform, TransformFactory, WorkerInfo, WorkerInit, with_worker_info,
};

pub(crate) struct WorkerTask {
    pub(crate) generation: u64,
    pub(crate) batch_sequence: u64,
    pub(crate) logical_samples: Vec<u64>,
    pub(crate) indices: Vec<usize>,
}

pub(crate) struct WorkerBatch<T> {
    pub(crate) generation: u64,
    pub(crate) batch_sequence: u64,
    pub(crate) samples: Vec<T>,
}

pub(crate) enum WorkerFailure<DE, TE, FE, IE> {
    Dataset(DE),
    Transform(TE),
    TransformInit(FE),
    WorkerInit(IE),
    InvalidBatchCardinality { expected: usize, actual: usize },
    Panic,
}

pub(crate) struct WorkerCompletion<T, DE, TE, FE, IE> {
    pub(crate) worker: usize,
    pub(crate) generation: u64,
    pub(crate) batch_sequence: Option<u64>,
    pub(crate) result: std::result::Result<Option<WorkerBatch<T>>, WorkerFailure<DE, TE, FE, IE>>,
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

const MAX_WORKER_QUEUE_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;

#[allow(dead_code)]
#[repr(C)]
struct ChannelSlot<T> {
    stamp: AtomicUsize,
    message: UnsafeCell<MaybeUninit<T>>,
}

pub(crate) struct WorkerPool<D, F, I>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    I: WorkerInit,
{
    tasks: Vec<Sender<WorkerTask>>,
    results: Option<Receiver<Completion<D, F, I>>>,
    handles: Vec<JoinHandle<()>>,
}

pub(crate) struct WorkerPoolConfiguration {
    pub(crate) workers: usize,
    pub(crate) prefetch_factor: usize,
    pub(crate) result_capacity: usize,
    pub(crate) generation: u64,
    pub(crate) loader_seed: u64,
    pub(crate) epoch: u64,
    pub(crate) rank: usize,
}

#[derive(Clone, Copy)]
struct WorkerRunContext {
    generation: u64,
    loader_seed: u64,
    epoch: u64,
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
    ) -> Result<Self> {
        validate_worker_pool_capacity::<D, F, I>(
            configuration.workers,
            configuration.prefetch_factor,
            configuration.result_capacity,
        )?;
        preflight_channel_storage::<WorkerTask>(configuration.result_capacity)?;
        preflight_channel_storage::<Completion<D, F, I>>(configuration.result_capacity)?;

        let mut infos = Vec::new();
        reserve_exact(&mut infos, configuration.workers, "workers")?;
        for id in 0..configuration.workers {
            infos.push(WorkerInfo::from_loader_seed(
                id,
                configuration.workers,
                configuration.loader_seed,
                configuration.rank,
                configuration.generation,
            )?);
        }

        let (result_sender, result_receiver) =
            bounded_checked(configuration.result_capacity, "worker result channel")?;
        let mut lanes = Vec::new();
        reserve_exact(&mut lanes, configuration.workers, "worker task channels")?;
        for _ in 0..configuration.workers {
            lanes.push(bounded_checked(
                configuration.prefetch_factor,
                "worker task channel",
            )?);
        }
        let mut tasks = Vec::new();
        reserve_exact(&mut tasks, configuration.workers, "worker task senders")?;
        let mut handles = Vec::new();
        reserve_exact(&mut handles, configuration.workers, "worker thread handles")?;
        let mut pool = Self {
            tasks,
            results: Some(result_receiver),
            handles,
        };

        for (worker, (task_sender, task_receiver)) in infos.into_iter().zip(lanes) {
            pool.tasks.push(task_sender);
            let dataset = Arc::clone(&dataset);
            let factory = Arc::clone(&factory);
            let initializer = Arc::clone(&initializer);
            let results = result_sender.clone();
            let context = WorkerRunContext {
                generation: configuration.generation,
                loader_seed: configuration.loader_seed,
                epoch: configuration.epoch,
            };
            let handle = thread::Builder::new()
                .name(format!("rusttorch-data-worker-{}", worker.id))
                .spawn(move || {
                    run_worker(
                        worker,
                        context,
                        dataset,
                        factory,
                        initializer,
                        task_receiver,
                        results,
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

    pub(crate) fn submit(&self, worker: usize, task: WorkerTask) -> std::result::Result<(), ()> {
        self.tasks[worker].send(task).map_err(|_| ())
    }

    pub(crate) fn receive(&self) -> std::result::Result<Completion<D, F, I>, ()> {
        self.results
            .as_ref()
            .expect("worker result receiver is present while the pool is active")
            .recv()
            .map_err(|_| ())
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
    for capacity in [prefetch_factor, outstanding] {
        capacity
            .checked_add(1)
            .and_then(usize::checked_next_power_of_two)
            .and_then(|mark_bit| mark_bit.checked_mul(2))
            .ok_or_else(|| capacity_error("bounded channel ring arithmetic overflowed"))?;
    }

    let slots_per_credit = size_of::<ChannelSlot<WorkerTask>>()
        .checked_add(size_of::<ChannelSlot<Completion<D, F, I>>>())
        .ok_or_else(|| capacity_error("bounded channel slot sizes exceed usize"))?;
    let channel_bytes = outstanding
        .checked_mul(slots_per_credit)
        .ok_or_else(|| capacity_error("bounded channel storage exceeds Rust allocation limits"))?;
    let per_worker_bookkeeping = size_of::<WorkerInfo>()
        .checked_add(size_of::<(Sender<WorkerTask>, Receiver<WorkerTask>)>())
        .and_then(|bytes| bytes.checked_add(size_of::<Sender<WorkerTask>>()))
        .and_then(|bytes| bytes.checked_add(size_of::<JoinHandle<()>>()))
        .ok_or_else(|| capacity_error("worker bookkeeping size exceeds usize"))?;
    let fixed_bookkeeping = (4 * size_of::<Vec<()>>())
        .checked_add(size_of::<Sender<Completion<D, F, I>>>())
        .and_then(|bytes| bytes.checked_add(size_of::<Receiver<Completion<D, F, I>>>()));
    let vector_bytes = workers
        .checked_mul(per_worker_bookkeeping)
        .and_then(|bytes| fixed_bookkeeping.and_then(|fixed| bytes.checked_add(fixed)))
        .ok_or_else(|| capacity_error("worker vector storage exceeds Rust allocation limits"))?;
    let aggregate_bytes = channel_bytes.checked_add(vector_bytes).ok_or_else(|| {
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
        self.tasks.clear();
        self.results.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn run_worker<D, F, I>(
    worker: WorkerInfo,
    context: WorkerRunContext,
    dataset: Arc<D>,
    factory: Arc<F>,
    initializer: Arc<I>,
    tasks: Receiver<WorkerTask>,
    results: Sender<Completion<D, F, I>>,
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
    let mut active_batch = None;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        with_worker_info(worker, || {
            if let Err(error) = initializer.initialize(&worker) {
                let _ = results.send(WorkerCompletion {
                    worker: worker.id,
                    generation: context.generation,
                    batch_sequence: None,
                    result: Err(WorkerFailure::WorkerInit(error)),
                });
                return;
            }
            let mut transform = match factory.create(Some(&worker)) {
                Ok(transform) => transform,
                Err(error) => {
                    let _ = results.send(WorkerCompletion {
                        worker: worker.id,
                        generation: context.generation,
                        batch_sequence: None,
                        result: Err(WorkerFailure::TransformInit(error)),
                    });
                    return;
                }
            };
            while let Ok(task) = tasks.recv() {
                active_batch = Some((task.generation, task.batch_sequence));
                let expected = task.indices.len();
                let samples = match dataset.get_batch(&task.indices) {
                    Ok(samples) => samples,
                    Err(error) => {
                        let _ = results.send(WorkerCompletion {
                            worker: worker.id,
                            generation: task.generation,
                            batch_sequence: Some(task.batch_sequence),
                            result: Err(WorkerFailure::Dataset(error)),
                        });
                        return;
                    }
                };
                if samples.len() != expected {
                    let _ = results.send(WorkerCompletion {
                        worker: worker.id,
                        generation: task.generation,
                        batch_sequence: Some(task.batch_sequence),
                        result: Err(WorkerFailure::InvalidBatchCardinality {
                            expected,
                            actual: samples.len(),
                        }),
                    });
                    return;
                }
                let mut transformed = Vec::with_capacity(samples.len());
                for (sample, logical_sample) in samples.into_iter().zip(task.logical_samples) {
                    let context = TaskContext {
                        loader_seed: context.loader_seed,
                        epoch: context.epoch,
                        rank: worker.rank,
                        logical_sample,
                        stage: 0,
                    };
                    match transform.transform(sample, &context) {
                        Ok(sample) => transformed.push(sample),
                        Err(error) => {
                            let _ = results.send(WorkerCompletion {
                                worker: worker.id,
                                generation: task.generation,
                                batch_sequence: Some(task.batch_sequence),
                                result: Err(WorkerFailure::Transform(error)),
                            });
                            return;
                        }
                    }
                }
                let batch = WorkerBatch {
                    generation: task.generation,
                    batch_sequence: task.batch_sequence,
                    samples: transformed,
                };
                if results
                    .send(WorkerCompletion {
                        worker: worker.id,
                        generation: task.generation,
                        batch_sequence: Some(task.batch_sequence),
                        result: Ok(Some(batch)),
                    })
                    .is_err()
                {
                    return;
                }
                active_batch = None;
            }
        });
    }));
    if outcome.is_err() {
        let (generation, batch_sequence) = active_batch
            .map(|(generation, batch)| (generation, Some(batch)))
            .unwrap_or((context.generation, None));
        let _ = results.send(WorkerCompletion {
            worker: worker.id,
            generation,
            batch_sequence,
            result: Err(WorkerFailure::Panic),
        });
    }
}
