use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
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
        let infos = (0..configuration.workers)
            .map(|id| {
                WorkerInfo::from_loader_seed(
                    id,
                    configuration.workers,
                    configuration.loader_seed,
                    configuration.rank,
                    configuration.generation,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (result_sender, result_receiver) = bounded(configuration.result_capacity);
        let mut pool = Self {
            tasks: Vec::with_capacity(configuration.workers),
            results: Some(result_receiver),
            handles: Vec::with_capacity(configuration.workers),
        };

        for worker in infos {
            let (task_sender, task_receiver) = bounded(configuration.prefetch_factor);
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
