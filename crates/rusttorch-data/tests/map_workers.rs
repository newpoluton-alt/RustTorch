use std::{
    cell::Cell,
    collections::HashSet,
    convert::Infallible,
    error::Error,
    fmt,
    rc::Rc,
    sync::{
        Arc, Barrier, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread::ThreadId,
};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    DataLoader, Dataset, FnCollate, FnSampler, FnTransform, FnTransformFactory, FnWorkerInit,
    LoaderError, PipelineError, TaskContext, Transform, VecCollate, WorkerInfo, get_worker_info,
};

struct ConcurrentRows {
    entered: Arc<Barrier>,
    threads: Arc<Mutex<Vec<ThreadId>>>,
}

impl Dataset for ConcurrentRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("worker loading must use get_batch")
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        self.threads
            .lock()
            .unwrap()
            .push(std::thread::current().id());
        self.entered.wait();
        Ok(indices.to_vec())
    }
}

#[test]
fn positive_workers_fetch_batches_concurrently_on_distinct_threads() -> Result<(), RustTorchError> {
    let threads = Arc::new(Mutex::new(Vec::new()));
    let mut loader = DataLoader::builder(ConcurrentRows {
        entered: Arc::new(Barrier::new(2)),
        threads: Arc::clone(&threads),
    })
    .workers(2)
    .collate(VecCollate)
    .build()?;

    let batches = loader
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("positive workers must execute map batches");

    assert_eq!(batches, [vec![0], vec![1]]);
    assert_eq!(
        threads
            .lock()
            .unwrap()
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len(),
        2
    );
    Ok(())
}

struct CompletionRows {
    later_finished: Arc<(Mutex<bool>, Condvar)>,
}

impl Dataset for CompletionRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        4
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("worker loading must use get_batch")
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        let index = indices[0];
        let (finished, wake) = &*self.later_finished;
        if index == 0 {
            let mut finished = finished.lock().unwrap();
            while !*finished {
                finished = wake.wait(finished).unwrap();
            }
        } else if index == 3 {
            *finished.lock().unwrap() = true;
            wake.notify_all();
        }
        Ok(vec![index])
    }
}

fn controlled_ordered_loader() -> Result<Vec<Vec<usize>>, RustTorchError> {
    let mut loader = DataLoader::builder(CompletionRows {
        later_finished: Arc::new((Mutex::new(false), Condvar::new())),
    })
    .workers(2)
    .collate(VecCollate)
    .build()?;
    Ok(loader.iter().collect::<Result<Vec<_>, _>>().unwrap())
}

struct ManualCompletionRows {
    release_first: Arc<(Mutex<bool>, Condvar)>,
}

impl Dataset for ManualCompletionRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        if index == 0 {
            let (released, wake) = &*self.release_first;
            let mut released = released.lock().unwrap();
            while !*released {
                released = wake.wait(released).unwrap();
            }
        }
        Ok(index)
    }
}

#[test]
fn ordered_and_completion_modes_follow_the_selected_contract() -> Result<(), RustTorchError> {
    assert_eq!(
        controlled_ordered_loader()?,
        [vec![0], vec![1], vec![2], vec![3]]
    );

    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let mut loader = DataLoader::builder(ManualCompletionRows {
        release_first: Arc::clone(&release),
    })
    .workers(2)
    .in_order(false)
    .collate(VecCollate)
    .build()?;
    let mut iterator = loader.iter();
    assert_eq!(iterator.next().unwrap().unwrap(), vec![1]);
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
    assert!(iterator.next().is_none());
    Ok(())
}

struct RoutingRows {
    routes: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl Dataset for RoutingRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        9
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("worker loading must use get_batch")
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        let worker = get_worker_info().expect("a real worker scope");
        self.routes.lock().unwrap().push((indices[0], worker.id));
        Ok(indices.to_vec())
    }
}

#[test]
fn batch_sequence_routes_to_its_modulo_worker_across_iterations() -> Result<(), RustTorchError> {
    let routes = Arc::new(Mutex::new(Vec::new()));
    let mut loader = DataLoader::builder(RoutingRows {
        routes: Arc::clone(&routes),
    })
    .workers(3)
    .collate(VecCollate)
    .build()?;

    for _ in 0..2 {
        assert_eq!(loader.iter().count(), 9);
    }
    let routes = routes.lock().unwrap();
    assert_eq!(routes.len(), 18);
    assert!(
        routes
            .iter()
            .all(|(sequence, worker)| sequence % 3 == *worker)
    );
    Ok(())
}

struct CountingIndices {
    next: usize,
    end: usize,
    pulls: Arc<AtomicUsize>,
}

impl Iterator for CountingIndices {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        (self.next < self.end).then(|| {
            let index = self.next;
            self.next += 1;
            self.pulls.fetch_add(1, Ordering::SeqCst);
            index
        })
    }
}

struct GatedRows {
    gate: Arc<(Mutex<bool>, Condvar)>,
    length: usize,
}

impl Dataset for GatedRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        let (open, wake) = &*self.gate;
        let mut open = open.lock().unwrap();
        while !*open {
            open = wake.wait(open).unwrap();
        }
        Ok(index)
    }
}

fn assert_prefetch_bound(factor: Option<usize>, expected: usize) -> Result<(), RustTorchError> {
    let pulls = Arc::new(AtomicUsize::new(0));
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let sampler_pulls = Arc::clone(&pulls);
    let sampler = FnSampler::new(Some(12), move |_| CountingIndices {
        next: 0,
        end: 12,
        pulls: Arc::clone(&sampler_pulls),
    });
    let builder = DataLoader::builder(GatedRows {
        gate: Arc::clone(&gate),
        length: 12,
    })
    .sampler(sampler)
    .workers(2)
    .collate(VecCollate);
    let mut loader = match factor {
        Some(factor) => builder.prefetch_factor(factor).build()?,
        None => builder.build()?,
    };
    let iterator = loader.iter();
    assert_eq!(pulls.load(Ordering::SeqCst), expected);
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    assert_eq!(iterator.count(), 12);
    Ok(())
}

#[test]
fn global_outstanding_work_uses_default_and_explicit_checked_credits() -> Result<(), RustTorchError>
{
    assert_prefetch_bound(None, 4)?;
    assert_prefetch_bound(Some(3), 6)?;
    assert!(matches!(
        DataLoader::builder(GatedRows {
            gate: Arc::new((Mutex::new(true), Condvar::new())),
            length: 1,
        })
        .workers(usize::MAX)
        .prefetch_factor(2)
        .build(),
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));
    Ok(())
}

struct PlainRows(usize);

impl Dataset for PlainRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(index)
    }
}

#[test]
fn factory_and_initializer_run_once_per_worker_with_fresh_generations() -> Result<(), RustTorchError>
{
    let factories = Arc::new(Mutex::new(Vec::new()));
    let initializers = Arc::new(Mutex::new(Vec::new()));
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let factory = FnTransformFactory::new({
        let calls = Arc::clone(&factories);
        let contexts = Arc::clone(&contexts);
        move |worker: Option<&WorkerInfo>| {
            let worker = *worker.expect("worker factory receives real context");
            assert_eq!(get_worker_info(), Some(worker));
            calls.lock().unwrap().push(worker);
            let contexts = Arc::clone(&contexts);
            Ok::<_, TestError>(FnTransform::new(
                move |value: usize, context: &TaskContext| {
                    contexts.lock().unwrap().push(*context);
                    Ok::<_, TestError>(value)
                },
            ))
        }
    });
    let initializer = FnWorkerInit::new({
        let calls = Arc::clone(&initializers);
        move |worker: &WorkerInfo| {
            assert_eq!(get_worker_info(), Some(*worker));
            calls.lock().unwrap().push(*worker);
            Ok::<_, TestError>(())
        }
    });
    let mut loader = DataLoader::builder(PlainRows(4))
        .workers(2)
        .seed(42)
        .rank(3)
        .transform_factory(factory)
        .worker_init(initializer)
        .collate(VecCollate)
        .build()?;

    assert_eq!(loader.iter().count(), 4);
    assert_eq!(loader.iter().count(), 4);
    let mut factories = factories.lock().unwrap().clone();
    let mut initializers = initializers.lock().unwrap().clone();
    factories.sort_by_key(|worker| (worker.seed, worker.id));
    initializers.sort_by_key(|worker| (worker.seed, worker.id));
    assert_eq!(factories, initializers);
    assert_eq!(factories.len(), 4);
    for generation_index in 0..2 {
        let start = generation_index * 2;
        let generation = &factories[start..start + 2];
        assert_eq!(generation[0].id, 0);
        assert_eq!(generation[1].id, 1);
        assert_eq!(generation[1].seed, generation[0].seed + 1);
        assert!(
            generation
                .iter()
                .all(|worker| worker.num_workers == 2 && worker.rank == 3)
        );
    }
    assert_ne!(factories[0].seed, factories[2].seed);
    let contexts = contexts.lock().unwrap();
    assert!(contexts.iter().all(|context| {
        context.loader_seed == 42 && context.epoch == 0 && context.rank == 3 && context.stage == 0
    }));
    let mut logical = contexts
        .iter()
        .map(|context| context.logical_sample)
        .collect::<Vec<_>>();
    logical.sort_unstable();
    assert_eq!(logical, [0, 0, 1, 1, 2, 2, 3, 3]);
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum TestError {
    Dataset,
    Transform,
    Collate,
    Factory,
    Init,
}

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for TestError {}

struct ErrorRows;

impl Dataset for ErrorRows {
    type Sample = usize;
    type Error = TestError;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Err(TestError::Dataset)
    }
}

#[test]
fn worker_and_coordinator_errors_are_typed_visible_once_and_contextual()
-> Result<(), RustTorchError> {
    let mut dataset = DataLoader::builder(ErrorRows)
        .workers(1)
        .transform(FnTransform::new(|value: usize, _: &TaskContext| {
            Ok::<_, TestError>(value)
        }))
        .collate(FnCollate::new(|values: Vec<usize>| {
            Ok::<_, TestError>(values)
        }))
        .build()?;
    let mut iterator = dataset.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: Some(0),
            source: PipelineError::Dataset(TestError::Dataset),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut transform = DataLoader::builder(PlainRows(1))
        .workers(1)
        .transform(FnTransform::new(|_: usize, _: &TaskContext| {
            Err::<usize, _>(TestError::Transform)
        }))
        .collate(FnCollate::new(|values: Vec<usize>| {
            Ok::<_, TestError>(values)
        }))
        .build()?;
    let mut iterator = transform.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: Some(0),
            source: PipelineError::Transform(TestError::Transform),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut collate = DataLoader::builder(PlainRows(1))
        .workers(1)
        .transform(FnTransform::new(|value: usize, _: &TaskContext| {
            Ok::<_, TestError>(value)
        }))
        .collate(FnCollate::new(|_: Vec<usize>| {
            Err::<Vec<usize>, _>(TestError::Collate)
        }))
        .build()?;
    let mut iterator = collate.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: None,
            source: PipelineError::Collate(TestError::Collate),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut factory = DataLoader::builder(PlainRows(1))
        .workers(1)
        .transform_factory(FnTransformFactory::new(|_: Option<&WorkerInfo>| {
            Err::<rusttorch_data::IdentityTransform, _>(TestError::Factory)
        }))
        .collate(FnCollate::new(|values: Vec<usize>| {
            Ok::<_, TestError>(values)
        }))
        .build()?;
    let mut iterator = factory.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: None,
            worker: Some(0),
            source: PipelineError::TransformInit(TestError::Factory),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut initializer = DataLoader::builder(PlainRows(1))
        .workers(1)
        .worker_init(FnWorkerInit::new(|_: &WorkerInfo| {
            Err::<(), _>(TestError::Init)
        }))
        .collate(FnCollate::new(|values: Vec<usize>| {
            Ok::<_, TestError>(values)
        }))
        .build()?;
    let mut iterator = initializer.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: None,
            worker: Some(0),
            source: PipelineError::WorkerInit(TestError::Init),
        }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

struct ShortBatch;

impl Dataset for ShortBatch {
    type Sample = usize;
    type Error = TestError;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("get_batch is overridden")
    }

    fn get_batch(&self, _indices: &[usize]) -> Result<Vec<Self::Sample>, Self::Error> {
        Ok(vec![0])
    }
}

struct PanicRows;

impl Dataset for PanicRows {
    type Sample = usize;
    type Error = TestError;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        panic!("dataset panic")
    }
}

#[test]
fn worker_rejects_wrong_batch_cardinality_before_transform_and_collation()
-> Result<(), RustTorchError> {
    let mut loader = DataLoader::builder(ShortBatch)
        .workers(1)
        .batch_size(2)
        .transform(FnTransform::new(
            |_: usize, _: &TaskContext| -> Result<usize, TestError> {
                panic!("transform must not run")
            },
        ))
        .collate(FnCollate::new(
            |_: Vec<usize>| -> Result<Vec<usize>, TestError> { panic!("collation must not run") },
        ))
        .build()?;
    let mut iterator = loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::InvalidBatchCardinality {
            batch: 0,
            worker: Some(0),
            expected: 2,
            actual: 1,
        }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

#[test]
fn worker_panics_are_converted_with_worker_and_batch_context() -> Result<(), RustTorchError> {
    let mut dataset = DataLoader::builder(PanicRows)
        .workers(1)
        .collate(VecCollate)
        .build()?;
    let mut iterator = dataset.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::WorkerPanic {
            worker: 0,
            batch: Some(0),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut transform = DataLoader::builder(PlainRows(1))
        .workers(1)
        .transform(FnTransform::new(
            |_: usize, _: &TaskContext| -> Result<usize, TestError> { panic!("transform panic") },
        ))
        .collate(VecCollate)
        .build()?;
    let mut iterator = transform.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::WorkerPanic {
            worker: 0,
            batch: Some(0),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut factory = DataLoader::builder(PlainRows(1))
        .workers(1)
        .transform_factory(FnTransformFactory::new(|_: Option<&WorkerInfo>| {
            panic!("factory panic");
            #[allow(unreachable_code)]
            Ok::<_, TestError>(rusttorch_data::IdentityTransform)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = factory.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::WorkerPanic {
            worker: 0,
            batch: None,
        }))
    ));
    assert!(iterator.next().is_none());

    let mut initializer = DataLoader::builder(PlainRows(1))
        .workers(1)
        .worker_init(FnWorkerInit::new(
            |_: &WorkerInfo| -> Result<(), TestError> { panic!("initializer panic") },
        ))
        .collate(VecCollate)
        .build()?;
    let mut iterator = initializer.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::WorkerPanic {
            worker: 0,
            batch: None,
        }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

struct LocalRows(Rc<Cell<usize>>);

impl Dataset for LocalRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        Ok(self.0.get())
    }
}

#[derive(Clone)]
struct LocalTransform(Rc<Cell<usize>>);

impl Transform<usize> for LocalTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: usize,
        _context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(input + self.0.get())
    }
}

#[test]
fn default_serial_loader_keeps_non_send_dataset_and_transform_support() -> Result<(), RustTorchError>
{
    let state = Rc::new(Cell::new(4));
    let mut loader = DataLoader::builder(LocalRows(Rc::clone(&state)))
        .transform(LocalTransform(Rc::clone(&state)))
        .collate(VecCollate)
        .build()?;
    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![9]]
    );
    Ok(())
}

#[test]
fn timeout_and_persistence_reject_before_any_worker_side_effect() -> Result<(), RustTorchError> {
    use std::time::Duration;

    let creates = Arc::new(AtomicUsize::new(0));
    for persistent in [false, true] {
        let creates = Arc::clone(&creates);
        let mut builder = DataLoader::builder(PlainRows(1))
            .workers(1)
            .transform_factory(FnTransformFactory::new(move |_: Option<&WorkerInfo>| {
                creates.fetch_add(1, Ordering::SeqCst);
                Ok::<_, TestError>(rusttorch_data::IdentityTransform)
            }))
            .collate(VecCollate);
        builder = if persistent {
            builder.persistent_workers(true)
        } else {
            builder.timeout(Duration::from_millis(1))
        };
        let mut loader = builder.build()?;
        assert!(matches!(
            loader.iter().next(),
            Some(Err(LoaderError::Configuration(
                RustTorchError::InvalidConfiguration { .. }
            )))
        ));
    }
    assert_eq!(creates.load(Ordering::SeqCst), 0);
    Ok(())
}

#[derive(Clone)]
struct DropTransform(Arc<AtomicUsize>);

impl Drop for DropTransform {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Transform<usize> for DropTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: usize,
        _context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        Ok(input)
    }
}

struct DropRows {
    entered: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
}

impl Dataset for DropRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        20
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        self.entered.send(()).unwrap();
        let (released, wake) = &*self.release;
        let mut released = released.lock().unwrap();
        while !*released {
            released = wake.wait(released).unwrap();
        }
        Ok(index)
    }
}

#[test]
fn early_drop_disconnects_saturated_work_and_joins_every_worker() -> Result<(), RustTorchError> {
    let (entered, arrivals) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let mut loader = DataLoader::builder(DropRows {
        entered,
        release: Arc::clone(&release),
    })
    .workers(2)
    .prefetch_factor(1)
    .transform(DropTransform(Arc::clone(&dropped)))
    .collate(VecCollate)
    .build()?;
    let iterator = loader.iter();
    arrivals.recv().unwrap();
    arrivals.recv().unwrap();
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    drop(iterator);
    assert_eq!(dropped.load(Ordering::SeqCst), 2);
    Ok(())
}
