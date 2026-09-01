use std::{
    convert::Infallible,
    error::Error,
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    CancellationToken, DataLoader, Dataset, Deadline, FnCollate, FnTransform, FnTransformFactory,
    FnWorkerInit, LoaderError, PipelineError, TaskContext, Transform, VecCollate, WaitOutcome,
    WorkerContext,
};

struct Rows(usize);

impl Dataset for Rows {
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
fn public_cancellation_and_deadline_waits_are_notification_driven() {
    let cancellation = CancellationToken::new();
    let waiter = cancellation.clone();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        ready_tx.send(()).unwrap();
        waiter.wait_cancelled();
        done_tx.send(()).unwrap();
    });
    ready_rx.recv().unwrap();
    assert!(!cancellation.is_cancelled());
    assert!(done_rx.try_recv().is_err());
    cancellation.cancel();
    done_rx.recv().unwrap();
    handle.join().unwrap();
    assert!(cancellation.is_cancelled());
    assert!(cancellation.wait_cancelled_timeout(Duration::ZERO));

    let absent = Deadline::none();
    assert_eq!(absent.remaining(), None);
    assert!(!absent.is_expired());
    let context = WorkerContext::new(
        rusttorch_data::WorkerInfo::new(0, 1, 7, 0).unwrap(),
        CancellationToken::new(),
        Deadline::after(Duration::ZERO),
    );
    assert_eq!(
        context.wait_cancelled_or_deadline(),
        WaitOutcome::DeadlineExpired
    );
    assert!(context.check().is_err());
}

struct CooperativeRows {
    entered: mpsc::Sender<()>,
    exits: Arc<AtomicUsize>,
}

impl Dataset for CooperativeRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("positive workers use the context-aware batch hook")
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> Result<Vec<Self::Sample>, Self::Error> {
        self.entered.send(()).unwrap();
        let outcome = context.wait_cancelled_or_deadline();
        assert!(matches!(
            outcome,
            WaitOutcome::Cancelled | WaitOutcome::DeadlineExpired
        ));
        self.exits.fetch_add(1, Ordering::SeqCst);
        Ok(indices.to_vec())
    }
}

#[test]
fn early_drop_wakes_context_aware_fetch_and_joins() -> Result<(), RustTorchError> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let exits = Arc::new(AtomicUsize::new(0));
    let mut loader = DataLoader::builder(CooperativeRows {
        entered: entered_tx,
        exits: Arc::clone(&exits),
    })
    .workers(1)
    .collate(VecCollate)
    .build()?;
    let iterator = loader.iter();
    entered_rx.recv().unwrap();
    drop(iterator);
    assert_eq!(exits.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn timeout_is_per_next_yielded_once_and_cancels_late_results() -> Result<(), RustTorchError> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let exits = Arc::new(AtomicUsize::new(0));
    let mut loader = DataLoader::builder(CooperativeRows {
        entered: entered_tx,
        exits: Arc::clone(&exits),
    })
    .workers(1)
    .timeout(Duration::from_millis(50))
    .collate(VecCollate)
    .build()?;
    let mut iterator = loader.iter();
    entered_rx.recv().unwrap();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Timeout { batch: 0 }))
    ));
    assert!(iterator.next().is_none());
    drop(iterator);
    assert_eq!(exits.load(Ordering::SeqCst), 1);
    Ok(())
}

struct SecondBatchWaitRows {
    entered: mpsc::Sender<()>,
}

impl Dataset for SecondBatchWaitRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("positive workers use the context-aware batch hook")
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> Result<Vec<Self::Sample>, Self::Error> {
        if indices[0] == 1 {
            self.entered.send(()).unwrap();
            assert_eq!(
                context.wait_cancelled_or_deadline(),
                WaitOutcome::DeadlineExpired
            );
        }
        Ok(indices.to_vec())
    }
}

#[test]
fn timeout_deadline_is_fresh_for_each_blocking_next() -> Result<(), RustTorchError> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let mut loader = DataLoader::builder(SecondBatchWaitRows {
        entered: entered_tx,
    })
    .workers(1)
    .timeout(Duration::from_millis(50))
    .collate(VecCollate)
    .build()?;
    let mut iterator = loader.iter();

    assert!(matches!(iterator.next(), Some(Ok(batch)) if batch == vec![0]));
    entered_rx.recv().unwrap();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Timeout { batch: 1 }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

#[derive(Clone)]
struct WaitingTransform {
    entered: mpsc::Sender<()>,
    exits: Arc<AtomicUsize>,
}

impl Transform<usize> for WaitingTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: usize,
        context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        self.entered.send(()).unwrap();
        assert!(matches!(
            context.wait_cancelled_or_deadline(),
            WaitOutcome::Cancelled | WaitOutcome::DeadlineExpired
        ));
        self.exits.fetch_add(1, Ordering::SeqCst);
        Ok(input)
    }
}

#[test]
fn context_aware_transform_wakes_on_drop_and_deadline() -> Result<(), RustTorchError> {
    for timed in [false, true] {
        let (entered_tx, entered_rx) = mpsc::channel();
        let exits = Arc::new(AtomicUsize::new(0));
        let builder = DataLoader::builder(Rows(1))
            .workers(1)
            .transform(WaitingTransform {
                entered: entered_tx,
                exits: Arc::clone(&exits),
            })
            .collate(VecCollate);
        let mut loader = if timed {
            builder.timeout(Duration::from_millis(50)).build()?
        } else {
            builder.build()?
        };
        let mut iterator = loader.iter();
        entered_rx.recv().unwrap();
        if timed {
            assert!(matches!(
                iterator.next(),
                Some(Err(LoaderError::Timeout { batch: 0 }))
            ));
        }
        drop(iterator);
        assert_eq!(exits.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum TestError {
    Dataset,
    Init,
}

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for TestError {}

struct FirstErrorRows {
    second_entered: Arc<(Mutex<bool>, Condvar)>,
    cancelled: Arc<AtomicUsize>,
}

impl Dataset for FirstErrorRows {
    type Sample = usize;
    type Error = TestError;

    fn len(&self) -> usize {
        2
    }

    fn get(&self, _index: usize) -> Result<Self::Sample, Self::Error> {
        unreachable!("context-aware batch hook is required")
    }

    fn get_batch_with_context(
        &self,
        indices: &[usize],
        context: &WorkerContext,
    ) -> Result<Vec<Self::Sample>, Self::Error> {
        if indices[0] == 1 {
            *self.second_entered.0.lock().unwrap() = true;
            self.second_entered.1.notify_all();
            assert_eq!(context.wait_cancelled_or_deadline(), WaitOutcome::Cancelled);
            self.cancelled.fetch_add(1, Ordering::SeqCst);
            Ok(indices.to_vec())
        } else {
            let mut entered = self.second_entered.0.lock().unwrap();
            while !*entered {
                entered = self.second_entered.1.wait(entered).unwrap();
            }
            Err(TestError::Dataset)
        }
    }
}

#[derive(Clone)]
struct ExitTransform(Arc<AtomicUsize>);

impl Drop for ExitTransform {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Transform<usize> for ExitTransform {
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

#[test]
fn first_error_cancels_siblings_and_every_worker_exits_once() -> Result<(), RustTorchError> {
    let cancelled = Arc::new(AtomicUsize::new(0));
    let exits = Arc::new(AtomicUsize::new(0));
    let mut loader = DataLoader::builder(FirstErrorRows {
        second_entered: Arc::new((Mutex::new(false), Condvar::new())),
        cancelled: Arc::clone(&cancelled),
    })
    .workers(2)
    .prefetch_factor(1)
    .transform(ExitTransform(Arc::clone(&exits)))
    .collate(FnCollate::new(Ok::<_, Infallible>))
    .build()?;
    let mut iterator = loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: Some(0),
            worker: Some(0),
            source: PipelineError::Dataset(TestError::Dataset),
        }))
    ));
    assert!(iterator.next().is_none());
    drop(iterator);
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(exits.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn worker_init_error_and_panic_are_visible_once() -> Result<(), RustTorchError> {
    let mut error_loader = DataLoader::builder(Rows(1))
        .workers(1)
        .worker_init(FnWorkerInit::new(|_: &WorkerContext| {
            Err::<(), _>(TestError::Init)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = error_loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::Pipeline {
            batch: None,
            worker: Some(0),
            source: PipelineError::WorkerInit(TestError::Init),
        }))
    ));
    assert!(iterator.next().is_none());

    let mut panic_loader = DataLoader::builder(Rows(1))
        .workers(1)
        .worker_init(FnWorkerInit::new(
            |_: &WorkerContext| -> Result<(), TestError> { panic!("init panic") },
        ))
        .collate(VecCollate)
        .build()?;
    let mut iterator = panic_loader.iter();
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

#[test]
fn full_result_queue_and_idle_task_waits_wake_on_drop() -> Result<(), RustTorchError> {
    let exits = Arc::new(AtomicUsize::new(0));
    let (completed_tx, completed_rx) = mpsc::channel();
    let mut full = DataLoader::builder(Rows(2))
        .workers(2)
        .prefetch_factor(1)
        .transform(FnTransform::new(move |value, _: &TaskContext| {
            completed_tx.send(()).unwrap();
            Ok::<_, Infallible>(value)
        }))
        .collate(VecCollate)
        .build()?;
    let iterator = full.iter();
    completed_rx.recv().unwrap();
    completed_rx.recv().unwrap();
    drop(iterator);

    let mut idle = DataLoader::builder(Rows(0))
        .workers(2)
        .transform(ExitTransform(Arc::clone(&exits)))
        .collate(VecCollate)
        .build()?;
    let iterator = idle.iter();
    drop(iterator);
    assert_eq!(exits.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn nonpersistent_iterations_use_fresh_consecutive_seed_ranges() -> Result<(), RustTorchError> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::clone(&seen);
    let mut loader = DataLoader::builder(Rows(2))
        .workers(2)
        .seed(42)
        .transform_factory(FnTransformFactory::new(
            move |context: Option<&WorkerContext>| {
                calls.lock().unwrap().push(context.unwrap().info);
                Ok::<_, Infallible>(rusttorch_data::IdentityTransform)
            },
        ))
        .collate(VecCollate)
        .build()?;
    assert_eq!(loader.iter().count(), 2);
    assert_eq!(loader.iter().count(), 2);
    let mut seen = seen.lock().unwrap().clone();
    seen.sort_by_key(|worker| worker.seed);
    assert_eq!(seen.len(), 4);
    assert_eq!(seen[1].seed, seen[0].seed + 1);
    assert_eq!(seen[3].seed, seen[2].seed + 1);
    assert_ne!(seen[0].seed, seen[2].seed);
    Ok(())
}

#[test]
fn serial_factory_gets_none_and_worker_callbacks_get_lifecycle_context()
-> Result<(), RustTorchError> {
    let serial_calls = Arc::new(AtomicUsize::new(0));
    let serial_seen = Arc::clone(&serial_calls);
    let mut serial = DataLoader::builder(Rows(1))
        .transform_factory(FnTransformFactory::new(
            move |context: Option<&WorkerContext>| {
                assert!(context.is_none());
                serial_seen.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(rusttorch_data::IdentityTransform)
            },
        ))
        .collate(VecCollate)
        .build()?;
    assert_eq!(serial.iter().count(), 1);
    assert_eq!(serial_calls.load(Ordering::SeqCst), 1);

    let factory_calls = Arc::new(AtomicUsize::new(0));
    let init_calls = Arc::new(AtomicUsize::new(0));
    let factory_seen = Arc::clone(&factory_calls);
    let init_seen = Arc::clone(&init_calls);
    let mut workers = DataLoader::builder(Rows(2))
        .workers(2)
        .transform_factory(FnTransformFactory::new(
            move |context: Option<&WorkerContext>| {
                let context = context.expect("worker factory context");
                context.check().unwrap();
                factory_seen.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(rusttorch_data::IdentityTransform)
            },
        ))
        .worker_init(FnWorkerInit::new(move |context: &WorkerContext| {
            context.check().unwrap();
            init_seen.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        }))
        .collate(VecCollate)
        .build()?;
    assert_eq!(workers.iter().count(), 2);
    assert_eq!(factory_calls.load(Ordering::SeqCst), 2);
    assert_eq!(init_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

struct RecordTransform {
    contexts: Arc<Mutex<Vec<(std::thread::ThreadId, TaskContext)>>>,
    exits: Arc<AtomicUsize>,
}

impl Drop for RecordTransform {
    fn drop(&mut self) {
        self.exits.fetch_add(1, Ordering::SeqCst);
    }
}

impl Transform<usize> for RecordTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: usize,
        context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        self.contexts
            .lock()
            .unwrap()
            .push((std::thread::current().id(), context.clone()));
        Ok(input)
    }
}

#[test]
fn persistent_workers_reuse_threads_seeds_and_callbacks_across_epochs() -> Result<(), RustTorchError>
{
    let threads = Arc::new(Mutex::new(Vec::new()));
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let pool_tokens = Arc::new(Mutex::new(Vec::new()));
    let exits = Arc::new(AtomicUsize::new(0));
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let init_calls = Arc::new(AtomicUsize::new(0));
    let factory_threads = Arc::clone(&threads);
    let factory_contexts = Arc::clone(&contexts);
    let factory_tokens = Arc::clone(&pool_tokens);
    let factory_exits = Arc::clone(&exits);
    let factory_seen = Arc::clone(&factory_calls);
    let init_seen = Arc::clone(&init_calls);
    let mut loader = DataLoader::builder(Rows(4))
        .workers(2)
        .persistent_workers(true)
        .transform_factory(FnTransformFactory::new(
            move |context: Option<&WorkerContext>| {
                let context = context.unwrap();
                factory_threads
                    .lock()
                    .unwrap()
                    .push((context.info, std::thread::current().id()));
                factory_tokens
                    .lock()
                    .unwrap()
                    .push(context.cancellation.clone());
                factory_seen.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(RecordTransform {
                    contexts: Arc::clone(&factory_contexts),
                    exits: Arc::clone(&factory_exits),
                })
            },
        ))
        .worker_init(FnWorkerInit::new(move |_: &WorkerContext| {
            init_seen.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        }))
        .collate(VecCollate)
        .build()?;
    assert_eq!(loader.iter().count(), 4);
    let first_token = contexts.lock().unwrap()[0].1.cancellation.clone();
    assert!(first_token.is_cancelled());
    loader.set_epoch(1);
    let mut second = loader.iter();
    assert!(second.next().unwrap().is_ok());
    let second_token = contexts
        .lock()
        .unwrap()
        .iter()
        .find(|(_, context)| context.epoch == 1)
        .expect("second generation context")
        .1
        .cancellation
        .clone();
    assert!(!second_token.is_cancelled());
    assert_eq!(second.count(), 3);
    assert!(second_token.is_cancelled());
    assert_eq!(factory_calls.load(Ordering::SeqCst), 2);
    assert_eq!(init_calls.load(Ordering::SeqCst), 2);
    let threads = threads.lock().unwrap();
    assert_eq!(threads.len(), 2);
    assert_eq!(threads[1].0.seed, threads[0].0.seed + 1);
    let contexts = contexts.lock().unwrap();
    for epoch in [0, 1] {
        let mut logical = contexts
            .iter()
            .filter(|(_, context)| context.epoch == epoch)
            .map(|(_, context)| context.logical_sample)
            .collect::<Vec<_>>();
        logical.sort_unstable();
        assert_eq!(logical, [0, 1, 2, 3]);
    }
    drop(contexts);
    assert!(
        pool_tokens
            .lock()
            .unwrap()
            .iter()
            .all(|token| !token.is_cancelled())
    );
    assert_eq!(exits.load(Ordering::SeqCst), 0);
    drop(loader);
    assert!(
        pool_tokens
            .lock()
            .unwrap()
            .iter()
            .all(CancellationToken::is_cancelled)
    );
    assert_eq!(exits.load(Ordering::SeqCst), 2);
    Ok(())
}

#[derive(Clone)]
struct StaleTransform {
    later_entered: Arc<(Mutex<bool>, Condvar)>,
}

impl Transform<usize> for StaleTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: usize,
        context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        if context.epoch != 0 {
            return Ok(input);
        }
        if context.logical_sample == 0 {
            let mut entered = self.later_entered.0.lock().unwrap();
            while !*entered {
                entered = self.later_entered.1.wait(entered).unwrap();
            }
            Ok(input)
        } else {
            *self.later_entered.0.lock().unwrap() = true;
            self.later_entered.1.notify_all();
            let _ = context.wait_cancelled_or_deadline();
            Ok(999)
        }
    }
}

#[test]
fn persistent_early_drop_discards_stale_sequence_zero_in_ordered_and_unordered_modes()
-> Result<(), RustTorchError> {
    for ordered in [true, false] {
        let mut loader = DataLoader::builder(Rows(4))
            .workers(2)
            .prefetch_factor(1)
            .persistent_workers(true)
            .ordered(ordered)
            .transform(StaleTransform {
                later_entered: Arc::new((Mutex::new(false), Condvar::new())),
            })
            .collate(VecCollate)
            .build()?;
        let mut first = loader.iter();
        assert_eq!(first.next().unwrap().unwrap(), [0]);
        drop(first);

        loader.set_epoch(1);
        let mut next = loader
            .iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("fresh persistent generation succeeds")
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        next.sort_unstable();
        assert_eq!(next, [0, 1, 2, 3]);
    }
    Ok(())
}

struct NonCooperativeRows {
    entered: mpsc::Sender<()>,
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl Dataset for NonCooperativeRows {
    type Sample = usize;
    type Error = Infallible;

    fn len(&self) -> usize {
        1
    }

    fn get(&self, index: usize) -> Result<Self::Sample, Self::Error> {
        self.entered.send(()).unwrap();
        let (released, wake) = &*self.gate;
        let mut released = released.lock().unwrap();
        while !*released {
            released = wake.wait(released).unwrap();
        }
        Ok(index)
    }
}

#[test]
fn non_cooperative_fetch_makes_drop_wait_until_explicit_release() -> Result<(), RustTorchError> {
    let (entered_tx, entered_rx) = mpsc::channel();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let mut loader = DataLoader::builder(NonCooperativeRows {
        entered: entered_tx,
        gate: Arc::clone(&gate),
    })
    .workers(1)
    .collate(VecCollate)
    .build()?;
    let iterator = loader.iter();
    entered_rx.recv().unwrap();
    std::thread::scope(|scope| {
        let (drop_tx, drop_rx) = mpsc::channel();
        let dropper = scope.spawn(move || {
            drop(iterator);
            drop_tx.send(()).unwrap();
        });
        assert!(drop_rx.try_recv().is_err());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        drop_rx.recv().unwrap();
        dropper.join().unwrap();
    });
    Ok(())
}

fn _typed_error_is_preserved(
    _error: LoaderError<PipelineError<Infallible, Infallible, Infallible, Infallible, Infallible>>,
) {
}

fn _task_context_is_cloneable(context: &TaskContext) -> TaskContext {
    context.clone()
}
