use std::{
    convert::Infallible,
    error::Error,
    fmt,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use rusttorch_core::RustTorchError;
use rusttorch_data::{
    CancellationToken, FnCollate, FnTransform, FnTransformFactory, FnWorkerInit, IdentityTransform,
    LoaderError, LogicalSampleId, PipelineError, SequenceId, StreamDataLoaderBuilder, TaskContext,
    Transform, TransformFactory, VecCollate, WorkerContext, WorkerRecord, WorkerSourceFactory,
    batches, batches_with_collate,
};

#[derive(Clone)]
struct ModuloFactory {
    length: usize,
    exact: bool,
}

impl WorkerSourceFactory for ModuloFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = std::vec::IntoIter<Result<WorkerRecord<Self::Sample>, Self::Error>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok((worker.info.id..self.length)
            .step_by(worker.info.num_workers)
            .map(|value| {
                Ok(WorkerRecord {
                    sequence: Some(SequenceId::new(value as u64)),
                    logical_id: LogicalSampleId::new(value as u64),
                    sample: value,
                })
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact.then_some(self.length)
    }
}

#[test]
fn four_explicit_modulo_shards_produce_each_logical_record_once() -> Result<(), RustTorchError> {
    let mut loader = StreamDataLoaderBuilder::new(ModuloFactory {
        length: 17,
        exact: true,
    })
    .workers(4)
    .batch_size(3)
    .collate(VecCollate)
    .build()?;

    let records = loader
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("disjoint shards succeed")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    assert_eq!(records, (0..17).collect::<Vec<_>>());
    assert_eq!(loader.len(), Some(6));
    Ok(())
}

#[derive(Clone)]
struct DelayedLowFactory {
    high_started: Arc<(Mutex<bool>, Condvar)>,
}

struct DelayedLowSource {
    worker: usize,
    yielded: bool,
    high_started: Arc<(Mutex<bool>, Condvar)>,
}

impl Iterator for DelayedLowSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded {
            return None;
        }
        self.yielded = true;
        if self.worker == 0 {
            let mut started = self.high_started.0.lock().unwrap();
            while !*started {
                started = self.high_started.1.wait(started).unwrap();
            }
        } else {
            *self.high_started.0.lock().unwrap() = true;
            self.high_started.1.notify_all();
        }
        let value = self.worker;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(value as u64)),
            logical_id: LogicalSampleId::new(value as u64),
            sample: value,
        }))
    }
}

impl WorkerSourceFactory for DelayedLowFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = DelayedLowSource;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(DelayedLowSource {
            worker: worker.info.id,
            yielded: false,
            high_started: Arc::clone(&self.high_started),
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn delayed_low_sequence_still_yields_strict_global_order() -> Result<(), RustTorchError> {
    let mut loader = StreamDataLoaderBuilder::new(DelayedLowFactory {
        high_started: Arc::new((Mutex::new(false), Condvar::new())),
    })
    .workers(2)
    .batch_size(1)
    .collate(VecCollate)
    .build()?;

    assert_eq!(
        loader.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![0], vec![1]]
    );
    Ok(())
}

#[derive(Clone)]
struct RecordsFactory {
    shards: Arc<RecordShards>,
    exact: Option<usize>,
}

type RawRecord = (Option<u64>, u64, usize);
type RecordShards = Vec<Vec<RawRecord>>;

impl WorkerSourceFactory for RecordsFactory {
    type Sample = usize;
    type Error = TestError;
    type Source = std::vec::IntoIter<Result<WorkerRecord<Self::Sample>, Self::Error>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(self.shards[worker.info.id]
            .iter()
            .copied()
            .map(|(sequence, logical_id, sample)| {
                Ok(WorkerRecord {
                    sequence: sequence.map(SequenceId::new),
                    logical_id: LogicalSampleId::new(logical_id),
                    sample,
                })
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact
    }
}

fn records_factory(
    shards: Vec<Vec<(Option<u64>, u64, usize)>>,
    exact: Option<usize>,
) -> RecordsFactory {
    RecordsFactory {
        shards: Arc::new(shards),
        exact,
    }
}

fn assert_protocol_once(factory: RecordsFactory, expected_sequence: Option<u64>) {
    let workers = factory.shards.len();
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .workers(workers)
        .collate(VecCollate)
        .build()
        .unwrap();
    let mut iterator = loader.iter();
    loop {
        match iterator.next() {
            Some(Ok(_)) => {}
            Some(Err(LoaderError::StreamProtocol { sequence, .. })) => {
                assert_eq!(sequence, expected_sequence);
                assert!(iterator.next().is_none());
                return;
            }
            Some(Err(error)) => panic!("unexpected loader error: {error}"),
            None => panic!("expected one stream protocol error"),
        }
    }
}

#[test]
fn ordered_sequence_protocol_rejects_every_gap_and_duplicate_shape_once() {
    assert_protocol_once(
        records_factory(vec![vec![(Some(0), 0, 0), (Some(0), 1, 1)]], Some(2)),
        Some(0),
    );
    assert_protocol_once(
        records_factory(
            vec![vec![(Some(0), 0, 0), (Some(1), 1, 1), (Some(0), 2, 2)]],
            Some(3),
        ),
        Some(0),
    );
    assert_protocol_once(
        records_factory(vec![vec![(Some(1), 1, 1)]], Some(1)),
        Some(0),
    );
    assert_protocol_once(
        records_factory(vec![vec![(Some(0), 0, 0), (Some(2), 2, 2)]], Some(3)),
        Some(1),
    );
    assert_protocol_once(
        records_factory(vec![vec![(Some(0), 0, 0), (Some(1), 1, 1)]], Some(3)),
        Some(2),
    );
    assert_protocol_once(records_factory(vec![vec![(None, 0, 0)]], Some(1)), None);
}

#[test]
fn ordered_full_credit_window_reports_first_gap_instead_of_timing_out() {
    let mut loader = StreamDataLoaderBuilder::new(records_factory(
        vec![vec![(Some(1), 1, 1), (Some(2), 2, 2)]],
        Some(3),
    ))
    .prefetch_factor(2)
    .timeout(Duration::from_millis(50))
    .collate(VecCollate)
    .build()
    .unwrap();
    let mut iterator = loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamProtocol {
            sequence: Some(0),
            ..
        }))
    ));
    assert!(iterator.next().is_none());
}

#[test]
fn ordered_internal_gap_beyond_same_worker_quota_is_protocol_error() {
    let mut loader = StreamDataLoaderBuilder::new(records_factory(
        vec![vec![(Some(0), 0, 0), (Some(2), 2, 2), (Some(3), 3, 3)]],
        Some(4),
    ))
    .prefetch_factor(2)
    .timeout(Duration::from_millis(50))
    .collate(VecCollate)
    .build()
    .unwrap();
    let mut iterator = loader.iter();
    assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamProtocol {
            sequence: Some(1),
            ..
        }))
    ));
    assert!(iterator.next().is_none());
}

#[derive(Clone)]
struct TeardownWindowFactory {
    window: usize,
    state: Arc<(Mutex<FailureTeardownState>, Condvar)>,
}

struct TeardownWindowSource {
    next: usize,
    window: usize,
    state: Arc<(Mutex<FailureTeardownState>, Condvar)>,
}

#[derive(Default)]
struct FailureTeardownState {
    extra_call_entered: bool,
    failure_returned: bool,
    release_extra_call: bool,
}

impl Iterator for TeardownWindowSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next > self.window {
            let mut state = self.state.0.lock().unwrap();
            state.extra_call_entered = true;
            self.state.1.notify_all();
            while !state.release_extra_call {
                state = self.state.1.wait(state).unwrap();
            }
            return None;
        }
        let value = self.next;
        self.next += 1;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(value as u64)),
            logical_id: LogicalSampleId::new(value as u64),
            sample: value,
        }))
    }
}

impl WorkerSourceFactory for TeardownWindowFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = TeardownWindowSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(TeardownWindowSource {
            next: 1,
            window: self.window,
            state: Arc::clone(&self.state),
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.window)
    }
}

#[test]
fn first_protocol_failure_cancels_before_releasing_reassembly_credits() {
    const WINDOW: usize = 4_096;
    let state = Arc::new((Mutex::new(FailureTeardownState::default()), Condvar::new()));
    let worker_state = Arc::clone(&state);
    let handle = thread::spawn(move || {
        let mut loader = StreamDataLoaderBuilder::new(TeardownWindowFactory {
            window: WINDOW,
            state: Arc::clone(&worker_state),
        })
        .prefetch_factor(WINDOW)
        .timeout(Duration::from_secs(1))
        .collate(VecCollate)
        .build()
        .unwrap();
        let mut iterator = loader.iter();
        assert!(matches!(
            iterator.next(),
            Some(Err(LoaderError::StreamProtocol {
                sequence: Some(0),
                ..
            }))
        ));
        assert!(iterator.next().is_none());
        let mut state = worker_state.0.lock().unwrap();
        state.failure_returned = true;
        worker_state.1.notify_all();
    });

    let mut observed = state.0.lock().unwrap();
    while !observed.failure_returned && !observed.extra_call_entered {
        observed = state.1.wait(observed).unwrap();
    }
    let failure_won = observed.failure_returned && !observed.extra_call_entered;
    observed.release_extra_call = true;
    state.1.notify_all();
    drop(observed);
    handle.join().unwrap();
    assert!(
        failure_won,
        "worker entered another source call before protocol teardown cancelled"
    );
}

#[derive(Clone)]
struct DropTeardownFactory {
    window: usize,
    high_produced: Arc<(Mutex<usize>, Condvar)>,
    calls_after_window: Arc<AtomicUsize>,
}

struct DropTeardownSource {
    worker: usize,
    next: usize,
    window: usize,
    high_produced: Arc<(Mutex<usize>, Condvar)>,
    calls_after_window: Arc<AtomicUsize>,
}

impl Iterator for DropTeardownSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.worker == 1 {
            if self.next > 0 {
                return None;
            }
            let mut produced = self.high_produced.0.lock().unwrap();
            while *produced < self.window {
                produced = self.high_produced.1.wait(produced).unwrap();
            }
            self.next = 1;
            return Some(Ok(WorkerRecord {
                sequence: Some(SequenceId::new(0)),
                logical_id: LogicalSampleId::new(0),
                sample: 0,
            }));
        }
        if self.next > self.window {
            self.calls_after_window.fetch_add(1, Ordering::SeqCst);
            return None;
        }
        let value = self.next;
        self.next += 1;
        let mut produced = self.high_produced.0.lock().unwrap();
        *produced += 1;
        self.high_produced.1.notify_all();
        drop(produced);
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(value as u64)),
            logical_id: LogicalSampleId::new(value as u64),
            sample: value,
        }))
    }
}

impl WorkerSourceFactory for DropTeardownFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = DropTeardownSource;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(DropTeardownSource {
            worker: worker.info.id,
            next: usize::from(worker.info.id == 0),
            window: self.window,
            high_produced: Arc::clone(&self.high_produced),
            calls_after_window: Arc::clone(&self.calls_after_window),
        })
    }

    fn exact_len(&self) -> Option<usize> {
        self.window.checked_add(1)
    }
}

#[test]
fn iterator_drop_cancels_before_releasing_reassembly_credits() {
    const WINDOW: usize = 4_096;
    let calls_after_window = Arc::new(AtomicUsize::new(0));
    let mut loader = StreamDataLoaderBuilder::new(DropTeardownFactory {
        window: WINDOW,
        high_produced: Arc::new((Mutex::new(0), Condvar::new())),
        calls_after_window: Arc::clone(&calls_after_window),
    })
    .workers(2)
    .prefetch_factor(WINDOW)
    .collate(VecCollate)
    .build()
    .unwrap();
    let mut iterator = loader.iter();
    assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
    drop(iterator);
    assert_eq!(calls_after_window.load(Ordering::SeqCst), 0);
}

#[test]
fn unordered_records_may_be_unsequenced_and_task_rng_uses_logical_id() -> Result<(), RustTorchError>
{
    let factory = records_factory(
        vec![
            vec![(None, 8, 8), (None, 10, 10)],
            vec![(None, 9, 9), (None, 11, 11)],
        ],
        Some(4),
    );
    let run = |workers| {
        let mut loader = StreamDataLoaderBuilder::new(factory.clone())
            .workers(workers)
            .in_order(false)
            .seed(42)
            .epoch(3)
            .rank(2)
            .transform(FnTransform::new(|value, context: &TaskContext| {
                Ok::<_, Infallible>((value, context.logical_sample, context.deterministic_seed()))
            }))
            .collate(VecCollate)
            .build()
            .unwrap();
        let mut output = loader
            .iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        output.sort_unstable_by_key(|record| record.0);
        output
    };

    let one = StreamDataLoaderBuilder::new(records_factory(
        vec![vec![
            (None, 8, 8),
            (None, 9, 9),
            (None, 10, 10),
            (None, 11, 11),
        ]],
        Some(4),
    ))
    .workers(1)
    .in_order(false)
    .seed(42)
    .epoch(3)
    .rank(2)
    .transform(FnTransform::new(|value, context: &TaskContext| {
        Ok::<_, Infallible>((value, context.logical_sample, context.deterministic_seed()))
    }))
    .collate(VecCollate)
    .build()?;
    let mut one = one;
    let mut one_output = one
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    one_output.sort_unstable_by_key(|record| record.0);
    assert_eq!(run(2), one_output);
    assert!(
        one_output
            .iter()
            .all(|(value, logical, _)| *value as u64 == *logical)
    );
    Ok(())
}

#[test]
fn known_and_unknown_lengths_use_global_checked_batching() -> Result<(), RustTorchError> {
    let known = StreamDataLoaderBuilder::new(ModuloFactory {
        length: 10,
        exact: true,
    })
    .workers(2)
    .batch_size(4)
    .collate(VecCollate)
    .build()?;
    assert_eq!(known.len(), Some(3));

    let dropped = StreamDataLoaderBuilder::new(ModuloFactory {
        length: 10,
        exact: true,
    })
    .workers(2)
    .batch_size(4)
    .drop_last(true)
    .collate(VecCollate)
    .build()?;
    assert_eq!(dropped.len(), Some(2));

    let unknown = StreamDataLoaderBuilder::new(ModuloFactory {
        length: 10,
        exact: false,
    })
    .workers(2)
    .batch_size(4)
    .collate(VecCollate)
    .build()?;
    assert_eq!(unknown.len(), None);
    Ok(())
}

#[test]
fn coordinator_batches_globally_and_drops_only_one_global_tail() -> Result<(), RustTorchError> {
    let factory = records_factory(
        vec![
            vec![(Some(0), 0, 0), (Some(3), 3, 3), (Some(6), 6, 6)],
            vec![(Some(1), 1, 1)],
            vec![(Some(2), 2, 2), (Some(4), 4, 4), (Some(5), 5, 5)],
        ],
        Some(7),
    );
    let mut keep = StreamDataLoaderBuilder::new(factory.clone())
        .workers(3)
        .batch_size(3)
        .collate(VecCollate)
        .build()?;
    assert_eq!(
        keep.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![0, 1, 2], vec![3, 4, 5], vec![6]]
    );

    let mut drop = StreamDataLoaderBuilder::new(factory)
        .workers(3)
        .batch_size(3)
        .drop_last(true)
        .collate(VecCollate)
        .build()?;
    assert_eq!(
        drop.iter().collect::<Result<Vec<_>, _>>().unwrap(),
        [vec![0, 1, 2], vec![3, 4, 5]]
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestError {
    Source,
    Transform,
    Factory,
    Init,
    Collate,
}

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for TestError {}

fn _typed_stream_error(
    _error: LoaderError<PipelineError<TestError, Infallible, Infallible, Infallible, Infallible>>,
) {
}

enum ErrorSource {
    Value(Option<WorkerRecord<usize>>),
    Failure(bool),
}

impl Iterator for ErrorSource {
    type Item = Result<WorkerRecord<usize>, TestError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Value(record) => record.take().map(Ok),
            Self::Failure(pending) if *pending => {
                *pending = false;
                Some(Err(TestError::Source))
            }
            Self::Failure(_) => None,
        }
    }
}

#[derive(Clone, Copy)]
enum ErrorMode {
    Create,
    Iterate,
    Value,
    Panic,
}

#[derive(Clone, Copy)]
struct ErrorFactory(ErrorMode);

impl WorkerSourceFactory for ErrorFactory {
    type Sample = usize;
    type Error = TestError;
    type Source = ErrorSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        match self.0 {
            ErrorMode::Create => Err(TestError::Source),
            ErrorMode::Iterate => Ok(ErrorSource::Failure(true)),
            ErrorMode::Value => Ok(ErrorSource::Value(Some(WorkerRecord {
                sequence: Some(SequenceId::new(0)),
                logical_id: LogicalSampleId::new(7),
                sample: 1,
            }))),
            ErrorMode::Panic => panic!("source factory panic"),
        }
    }

    fn exact_len(&self) -> Option<usize> {
        Some(1)
    }
}

#[test]
fn every_stream_stage_failure_is_typed_contextual_and_visible_once() -> Result<(), RustTorchError> {
    for mode in [ErrorMode::Create, ErrorMode::Iterate] {
        let mut loader = StreamDataLoaderBuilder::new(ErrorFactory(mode))
            .collate(VecCollate)
            .build()?;
        let mut iterator = loader.iter();
        assert!(matches!(
            iterator.next(),
            Some(Err(LoaderError::StreamPipeline {
                batch: Some(0),
                worker: 0,
                sequence: None,
                logical_id: None,
                source: PipelineError::Source(TestError::Source),
            }))
        ));
        assert!(iterator.next().is_none());
    }

    let mut transform = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Value))
        .transform(FnTransform::new(|_: usize, _: &TaskContext| {
            Err::<usize, _>(TestError::Transform)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = transform.iter();
    let outcome = iterator.next();
    assert!(
        matches!(
            outcome,
            Some(Err(LoaderError::StreamPipeline {
                batch: Some(0),
                worker: 0,
                sequence: Some(0),
                logical_id: Some(7),
                source: PipelineError::Transform(TestError::Transform),
            }))
        ),
        "unexpected transform failure outcome: {outcome:?}"
    );
    assert!(iterator.next().is_none());

    let mut factory = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Value))
        .transform_factory(FnTransformFactory::new(|_: Option<&WorkerContext>| {
            Err::<IdentityTransform, _>(TestError::Factory)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = factory.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamPipeline {
            batch: None,
            worker: 0,
            source: PipelineError::TransformInit(TestError::Factory),
            ..
        }))
    ));
    assert!(iterator.next().is_none());

    let mut init = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Value))
        .worker_init(FnWorkerInit::new(|_: &WorkerContext| {
            Err::<(), _>(TestError::Init)
        }))
        .collate(VecCollate)
        .build()?;
    let mut iterator = init.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamPipeline {
            batch: None,
            worker: 0,
            source: PipelineError::WorkerInit(TestError::Init),
            ..
        }))
    ));
    assert!(iterator.next().is_none());

    let mut collate = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Value))
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
    Ok(())
}

#[test]
fn source_and_transform_panics_have_stream_metadata_and_join_once() -> Result<(), RustTorchError> {
    let mut loader = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Panic))
        .collate(VecCollate)
        .build()?;
    let mut iterator = loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamWorkerPanic {
            worker: 0,
            batch: Some(0),
            sequence: None,
            logical_id: None,
        }))
    ));
    assert!(iterator.next().is_none());
    drop(iterator);

    let mut loader = StreamDataLoaderBuilder::new(ErrorFactory(ErrorMode::Value))
        .transform(FnTransform::new(
            |_: usize, _: &TaskContext| -> Result<usize, Infallible> { panic!("transform panic") },
        ))
        .collate(VecCollate)
        .build()?;
    let mut iterator = loader.iter();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamWorkerPanic {
            worker: 0,
            batch: Some(0),
            sequence: Some(0),
            logical_id: Some(7),
        }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

#[derive(Clone)]
struct CooperativeFactory {
    entered: mpsc::Sender<()>,
    exits: Arc<AtomicUsize>,
}

struct CooperativeSource {
    context: WorkerContext,
    entered: mpsc::Sender<()>,
    exits: Arc<AtomicUsize>,
    done: bool,
}

impl Iterator for CooperativeSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        self.done = true;
        self.entered.send(()).unwrap();
        self.context.cancellation.wait_cancelled();
        self.exits.fetch_add(1, Ordering::SeqCst);
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(0)),
            logical_id: LogicalSampleId::new(0),
            sample: 0,
        }))
    }
}

impl WorkerSourceFactory for CooperativeFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = CooperativeSource;

    fn create(&self, context: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(CooperativeSource {
            context,
            entered: self.entered.clone(),
            exits: Arc::clone(&self.exits),
            done: false,
        })
    }
}

#[test]
fn cooperative_stream_source_wakes_on_timeout_and_early_drop() -> Result<(), RustTorchError> {
    for timed in [false, true] {
        let (entered_tx, entered_rx) = mpsc::channel();
        let exits = Arc::new(AtomicUsize::new(0));
        let builder = StreamDataLoaderBuilder::new(CooperativeFactory {
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
            assert!(iterator.next().is_none());
        }
        drop(iterator);
        assert_eq!(exits.load(Ordering::SeqCst), 1);
    }
    Ok(())
}

#[derive(Clone)]
struct QueuedLowFactory {
    queued: mpsc::Sender<()>,
}

struct QueuedLowSource {
    queued: mpsc::Sender<()>,
    step: usize,
}

impl Iterator for QueuedLowSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.step {
            0 => Some((1, 1)),
            1 => Some((0, 0)),
            2 => {
                self.queued.send(()).unwrap();
                None
            }
            _ => None,
        };
        self.step += 1;
        record.map(|(sequence, sample)| {
            Ok(WorkerRecord {
                sequence: Some(SequenceId::new(sequence)),
                logical_id: LogicalSampleId::new(sequence),
                sample,
            })
        })
    }
}

impl WorkerSourceFactory for QueuedLowFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = QueuedLowSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(QueuedLowSource {
            queued: self.queued.clone(),
            step: 0,
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(2)
    }
}

#[test]
fn queued_low_record_beats_an_expired_deadline_after_high_reassembly() -> Result<(), RustTorchError>
{
    let (queued_tx, queued_rx) = mpsc::channel();
    let mut loader = StreamDataLoaderBuilder::new(QueuedLowFactory { queued: queued_tx })
        .prefetch_factor(3)
        .timeout(Duration::from_nanos(1))
        .collate(VecCollate)
        .build()?;
    let mut iterator = loader.iter();
    queued_rx.recv().unwrap();
    assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
    assert_eq!(iterator.next().unwrap().unwrap(), vec![1]);
    assert!(iterator.next().is_none());
    Ok(())
}

#[derive(Clone)]
struct QueuedTerminalFactory {
    ready: mpsc::Sender<()>,
}

struct QueuedTerminalSource {
    ready: mpsc::Sender<()>,
    step: usize,
}

impl Iterator for QueuedTerminalSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.step {
            0 => Some(WorkerRecord {
                sequence: Some(SequenceId::new(1)),
                logical_id: LogicalSampleId::new(1),
                sample: 1,
            }),
            1 => Some(WorkerRecord {
                sequence: None,
                logical_id: LogicalSampleId::new(2),
                sample: 2,
            }),
            2 => {
                self.ready.send(()).unwrap();
                None
            }
            _ => None,
        };
        self.step += 1;
        record.map(Ok)
    }
}

impl WorkerSourceFactory for QueuedTerminalFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = QueuedTerminalSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(QueuedTerminalSource {
            ready: self.ready.clone(),
            step: 0,
        })
    }
}

#[test]
fn queued_protocol_error_beats_expiry_after_a_high_record() -> Result<(), RustTorchError> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let mut loader = StreamDataLoaderBuilder::new(QueuedTerminalFactory { ready: ready_tx })
        .prefetch_factor(3)
        .timeout(Duration::from_nanos(1))
        .collate(VecCollate)
        .build()?;
    let mut iterator = loader.iter();
    ready_rx.recv().unwrap();
    assert!(matches!(
        iterator.next(),
        Some(Err(LoaderError::StreamProtocol { sequence: None, .. }))
    ));
    assert!(iterator.next().is_none());
    Ok(())
}

#[derive(Clone)]
struct NonCooperativeFactory {
    entered: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
    dropped: Arc<AtomicUsize>,
}

struct NonCooperativeSource {
    entered: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
    dropped: Arc<AtomicUsize>,
}

impl Iterator for NonCooperativeSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entered.send(()).unwrap();
        let mut released = self.release.0.lock().unwrap();
        while !*released {
            released = self.release.1.wait(released).unwrap();
        }
        None
    }
}

impl Drop for NonCooperativeSource {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl WorkerSourceFactory for NonCooperativeFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = NonCooperativeSource;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(NonCooperativeSource {
            entered: self.entered.clone(),
            release: Arc::clone(&self.release),
            dropped: Arc::clone(&self.dropped),
        })
    }
}

#[test]
fn drop_waits_for_a_noncooperative_source_instead_of_detaching_it() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (drop_tx, drop_rx) = mpsc::channel();
    let (dropping_tx, dropping_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let factory = NonCooperativeFactory {
        entered: entered_tx,
        release: Arc::clone(&release),
        dropped: Arc::clone(&dropped),
    };

    let handle = thread::spawn(move || {
        let mut loader = StreamDataLoaderBuilder::new(factory)
            .collate(VecCollate)
            .build()
            .unwrap();
        let iterator = loader.iter();
        ready_tx.send(()).unwrap();
        drop_rx.recv().unwrap();
        dropping_tx.send(()).unwrap();
        drop(iterator);
        done_tx.send(()).unwrap();
    });

    ready_rx.recv().unwrap();
    entered_rx.recv().unwrap();
    drop_tx.send(()).unwrap();
    dropping_rx.recv().unwrap();
    assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    done_rx.recv().unwrap();
    handle.join().unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[derive(Clone)]
struct QueueFillFactory {
    advances: mpsc::Sender<()>,
}

struct QueueFillSource {
    worker: usize,
    advances: mpsc::Sender<()>,
    yielded: bool,
}

impl Iterator for QueueFillSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        self.advances.send(()).unwrap();
        if self.yielded {
            return None;
        }
        self.yielded = true;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(self.worker as u64)),
            logical_id: LogicalSampleId::new(self.worker as u64),
            sample: self.worker,
        }))
    }
}

impl WorkerSourceFactory for QueueFillFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = QueueFillSource;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(QueueFillSource {
            worker: worker.info.id,
            advances: self.advances.clone(),
            yielded: false,
        })
    }
}

#[derive(Clone)]
struct DropTransformFactory(Arc<AtomicUsize>);

struct DropTransform(Arc<AtomicUsize>);

impl TransformFactory<usize> for DropTransformFactory {
    type Transform = DropTransform;
    type Error = Infallible;

    fn create(&self, _worker: Option<&WorkerContext>) -> Result<Self::Transform, Self::Error> {
        Ok(DropTransform(Arc::clone(&self.0)))
    }
}

impl Transform<usize> for DropTransform {
    type Output = usize;
    type Error = Infallible;

    fn transform(&mut self, input: usize, _context: &TaskContext) -> Result<usize, Self::Error> {
        Ok(input)
    }
}

impl Drop for DropTransform {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn full_result_queue_drop_wakes_and_joins_every_worker() -> Result<(), RustTorchError> {
    let (advances_tx, advances_rx) = mpsc::channel();
    let transform_drops = Arc::new(AtomicUsize::new(0));
    let mut loader = StreamDataLoaderBuilder::new(QueueFillFactory {
        advances: advances_tx,
    })
    .workers(2)
    .prefetch_factor(2)
    .transform_factory(DropTransformFactory(Arc::clone(&transform_drops)))
    .collate(VecCollate)
    .build()?;
    let iterator = loader.iter();

    // Each worker advances once for its record and once for its terminal
    // marker, filling all four globally bounded result slots.
    for _ in 0..4 {
        advances_rx.recv().unwrap();
    }
    drop(iterator);
    assert_eq!(transform_drops.load(Ordering::SeqCst), 2);
    Ok(())
}

#[derive(Clone)]
struct CreditProbeFactory {
    records: usize,
    fast_progress: Arc<(Mutex<usize>, Condvar)>,
    observed_before_low: Arc<AtomicUsize>,
}

struct CreditProbeSource {
    worker: usize,
    workers: usize,
    next: usize,
    records: usize,
    fast_progress: Arc<(Mutex<usize>, Condvar)>,
    observed_before_low: Arc<AtomicUsize>,
}

impl Iterator for CreditProbeSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.records {
            return None;
        }
        if self.worker == 0 && self.next == 0 {
            let mut progress = self.fast_progress.0.lock().unwrap();
            while *progress < 4 {
                progress = self.fast_progress.1.wait(progress).unwrap();
            }
            self.observed_before_low.store(*progress, Ordering::SeqCst);
        } else if self.worker != 0 {
            let mut progress = self.fast_progress.0.lock().unwrap();
            *progress += 1;
            self.fast_progress.1.notify_all();
        }
        let value = self.next;
        self.next += self.workers;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(value as u64)),
            logical_id: LogicalSampleId::new(value as u64),
            sample: value,
        }))
    }
}

impl WorkerSourceFactory for CreditProbeFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = CreditProbeSource;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        Ok(CreditProbeSource {
            worker: worker.info.id,
            workers: worker.info.num_workers,
            next: worker.info.id,
            records: self.records,
            fast_progress: Arc::clone(&self.fast_progress),
            observed_before_low: Arc::clone(&self.observed_before_low),
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(self.records)
    }
}

#[test]
fn per_worker_credits_bound_fast_shards_without_blocking_their_progress()
-> Result<(), RustTorchError> {
    let fast_progress = Arc::new((Mutex::new(0), Condvar::new()));
    let observed_before_low = Arc::new(AtomicUsize::new(0));
    let mut loader = StreamDataLoaderBuilder::new(CreditProbeFactory {
        records: 6,
        fast_progress,
        observed_before_low: Arc::clone(&observed_before_low),
    })
    .workers(3)
    .prefetch_factor(2)
    .batch_size(6)
    .collate(VecCollate)
    .build()?;

    assert_eq!(
        loader.iter().next().transpose().expect("stream succeeds"),
        Some((0..6).collect())
    );
    // The two fast shards each made exactly their two credited records while
    // ordered sequence zero was blocked; neither could consume a third slot.
    assert_eq!(observed_before_low.load(Ordering::SeqCst), 4);
    Ok(())
}

#[derive(Clone)]
struct PersistentProbeFactory {
    creates: Arc<Vec<AtomicUsize>>,
    previous_tokens: Arc<Mutex<Vec<Option<CancellationToken>>>>,
    delay_first_generation_until_cancelled: bool,
}

impl WorkerSourceFactory for PersistentProbeFactory {
    type Sample = (usize, usize, usize, u64);
    type Error = Infallible;
    type Source = std::vec::IntoIter<Result<WorkerRecord<Self::Sample>, Self::Error>>;

    fn create(&self, worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        if self.delay_first_generation_until_cancelled
            && worker.info.id == 1
            && self.creates[1].load(Ordering::SeqCst) == 0
        {
            worker.cancellation.wait_cancelled();
        }
        // Early iterator drop can cancel a generation while its factory is
        // starting. The complete second generation below proves token renewal.
        let generation = self.creates[worker.info.id].fetch_add(1, Ordering::SeqCst);
        let mut tokens = self.previous_tokens.lock().unwrap();
        if let Some(previous) = tokens[worker.info.id].replace(worker.cancellation.clone()) {
            assert!(previous.is_cancelled());
        }
        drop(tokens);

        Ok((worker.info.id..4)
            .step_by(worker.info.num_workers)
            .map(|value| {
                Ok(WorkerRecord {
                    sequence: Some(SequenceId::new(value as u64)),
                    logical_id: LogicalSampleId::new(value as u64),
                    sample: (generation, worker.info.id, value, worker.info.seed),
                })
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn exact_len(&self) -> Option<usize> {
        Some(4)
    }
}

#[derive(Clone)]
struct PersistentTransformFactory {
    creates: Arc<AtomicUsize>,
}

struct PersistentTransform {
    calls: usize,
}

impl TransformFactory<(usize, usize, usize, u64)> for PersistentTransformFactory {
    type Transform = PersistentTransform;
    type Error = Infallible;

    fn create(&self, _worker: Option<&WorkerContext>) -> Result<Self::Transform, Self::Error> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        Ok(PersistentTransform { calls: 0 })
    }
}

impl Transform<(usize, usize, usize, u64)> for PersistentTransform {
    type Output = (usize, usize, usize, u64, usize);
    type Error = Infallible;

    fn transform(
        &mut self,
        (generation, worker, value, seed): (usize, usize, usize, u64),
        _context: &TaskContext,
    ) -> Result<Self::Output, Self::Error> {
        self.calls += 1;
        Ok((generation, worker, value, seed, self.calls))
    }
}

fn persistent_probe(ordered: bool, early_drop: bool) -> Result<(), RustTorchError> {
    let creates = Arc::new(vec![AtomicUsize::new(0), AtomicUsize::new(0)]);
    let previous_tokens = Arc::new(Mutex::new(vec![None, None]));
    let transform_creates = Arc::new(AtomicUsize::new(0));
    let init_calls = Arc::new(AtomicUsize::new(0));
    let mut loader = StreamDataLoaderBuilder::new(PersistentProbeFactory {
        creates: Arc::clone(&creates),
        previous_tokens: Arc::clone(&previous_tokens),
        delay_first_generation_until_cancelled: early_drop,
    })
    .workers(2)
    .prefetch_factor(2)
    .batch_size(if early_drop { 1 } else { 2 })
    .ordered(ordered)
    .persistent_workers(true)
    .transform_factory(PersistentTransformFactory {
        creates: Arc::clone(&transform_creates),
    })
    .worker_init(FnWorkerInit::new({
        let init_calls = Arc::clone(&init_calls);
        move |_: &WorkerContext| {
            init_calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        }
    }))
    .collate(VecCollate)
    .build()?;

    let first = if early_drop {
        let mut first = loader.iter();
        let first_batch = first
            .next()
            .expect("first batch")
            .expect("first generation");
        assert_eq!(first_batch.len(), 1);
        drop(first);
        Vec::new()
    } else {
        loader
            .iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("first generation")
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
    };
    let mut second = loader
        .iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("second generation")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    second.sort_by_key(|record| record.2);

    assert_eq!(
        creates
            .iter()
            .map(|count| count.load(Ordering::SeqCst))
            .sum::<usize>(),
        4
    );
    assert_eq!(init_calls.load(Ordering::SeqCst), 2);
    assert_eq!(transform_creates.load(Ordering::SeqCst), 2);
    assert!(second.iter().all(|record| record.0 == 1));
    assert_ne!(second[0].3, second[1].3, "workers receive distinct seeds");
    if !early_drop {
        let mut first = first;
        first.sort_by_key(|record| record.2);
        assert!(first.iter().all(|record| record.0 == 0));
        assert_ne!(first[0].3, second[0].3, "generation seed is refreshed");
        for worker in 0..2 {
            let first_calls = first
                .iter()
                .filter(|record| record.1 == worker)
                .map(|record| record.4)
                .collect::<Vec<_>>();
            let second_calls = second
                .iter()
                .filter(|record| record.1 == worker)
                .map(|record| record.4)
                .collect::<Vec<_>>();
            assert_eq!(first_calls, vec![1, 2]);
            assert_eq!(second_calls, vec![3, 4]);
        }
    }
    Ok(())
}

#[test]
fn persistent_ordered_and_unordered_workers_recreate_sources_but_keep_pool_state()
-> Result<(), RustTorchError> {
    persistent_probe(true, false)?;
    persistent_probe(false, false)
}

#[test]
fn persistent_early_drop_drains_stale_results_before_both_delivery_modes_restart()
-> Result<(), RustTorchError> {
    persistent_probe(true, true)?;
    persistent_probe(false, true)
}

#[derive(Clone)]
struct SourceDropFactory {
    entered: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
    creates: Arc<AtomicUsize>,
    panic_on_drop: bool,
}

struct SourceDropProbe {
    entered: mpsc::Sender<()>,
    release: Arc<(Mutex<bool>, Condvar)>,
    panic_on_drop: bool,
    yielded: bool,
}

impl Iterator for SourceDropProbe {
    type Item = Result<WorkerRecord<usize>, Infallible>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded {
            return None;
        }
        self.yielded = true;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(0)),
            logical_id: LogicalSampleId::new(0),
            sample: 0,
        }))
    }
}

impl Drop for SourceDropProbe {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        let mut released = self.release.0.lock().unwrap();
        while !*released {
            released = self.release.1.wait(released).unwrap();
        }
        if self.panic_on_drop {
            panic!("source destructor panic");
        }
    }
}

impl WorkerSourceFactory for SourceDropFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = SourceDropProbe;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        Ok(SourceDropProbe {
            entered: self.entered.clone(),
            release: Arc::clone(&self.release),
            panic_on_drop: self.panic_on_drop,
            yielded: false,
        })
    }

    fn exact_len(&self) -> Option<usize> {
        Some(1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DropOutcome {
    Exhausted,
    Panic,
    ChannelClosed,
    Timeout,
    Other,
}

#[test]
fn persistent_exhaustion_waits_until_the_generation_source_is_dropped() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let creates = Arc::new(AtomicUsize::new(0));
    let factory = SourceDropFactory {
        entered: entered_tx,
        release: Arc::clone(&release),
        creates,
        panic_on_drop: false,
    };
    let handle = thread::spawn(move || {
        let mut loader = StreamDataLoaderBuilder::new(factory)
            .persistent_workers(true)
            .collate(VecCollate)
            .build()
            .unwrap();
        let mut iterator = loader.iter();
        assert_eq!(iterator.next().unwrap().unwrap(), vec![0]);
        let outcome = match iterator.next() {
            None => DropOutcome::Exhausted,
            _ => DropOutcome::Other,
        };
        finished_tx.send(outcome).unwrap();
    });

    entered_rx.recv().unwrap();
    let premature = match finished_rx.recv_timeout(Duration::from_millis(50)) {
        Ok(outcome) => Some(outcome),
        Err(mpsc::RecvTimeoutError::Timeout) => None,
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("worker test disconnected"),
    };
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    let outcome = premature.unwrap_or_else(|| finished_rx.recv().unwrap());
    handle.join().unwrap();
    assert!(
        premature.is_none(),
        "exhaustion preceded source destruction"
    );
    assert_eq!(outcome, DropOutcome::Exhausted);
}

#[test]
fn persistent_source_destructor_panic_is_visible_and_poisons_the_pool() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (outcome_tx, outcome_rx) = mpsc::channel();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let creates = Arc::new(AtomicUsize::new(0));
    let factory = SourceDropFactory {
        entered: entered_tx,
        release: Arc::clone(&release),
        creates: Arc::clone(&creates),
        panic_on_drop: true,
    };
    let handle = thread::spawn(move || {
        let mut loader = StreamDataLoaderBuilder::new(factory)
            .persistent_workers(true)
            .timeout(Duration::from_secs(1))
            .collate(VecCollate)
            .build()
            .unwrap();
        let mut first = loader.iter();
        assert_eq!(first.next().unwrap().unwrap(), vec![0]);
        let first_outcome = match first.next() {
            Some(Err(LoaderError::StreamWorkerPanic {
                worker: 0,
                batch: Some(1),
                sequence: None,
                logical_id: None,
            })) => DropOutcome::Panic,
            None => DropOutcome::Exhausted,
            Some(Err(LoaderError::Timeout { .. })) => DropOutcome::Timeout,
            _ => DropOutcome::Other,
        };
        outcome_tx.send(first_outcome).unwrap();
        let second_outcome = if first_outcome == DropOutcome::Panic {
            assert!(first.next().is_none());
            drop(first);
            let mut second = loader.iter();
            match second.next() {
                Some(Err(LoaderError::ChannelClosed { .. })) => DropOutcome::ChannelClosed,
                Some(Err(LoaderError::Timeout { .. })) => DropOutcome::Timeout,
                _ => DropOutcome::Other,
            }
        } else {
            drop(first);
            DropOutcome::Other
        };
        outcome_tx.send(second_outcome).unwrap();
    });

    entered_rx.recv().unwrap();
    let premature = match outcome_rx.recv_timeout(Duration::from_millis(50)) {
        Ok(outcome) => Some(outcome),
        Err(mpsc::RecvTimeoutError::Timeout) => None,
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("worker test disconnected"),
    };
    *release.0.lock().unwrap() = true;
    release.1.notify_all();
    let first_outcome = premature.unwrap_or_else(|| outcome_rx.recv().unwrap());
    let second_outcome = outcome_rx.recv().unwrap();
    handle.join().unwrap();

    assert!(
        premature.is_none(),
        "terminal acknowledgement preceded Drop"
    );
    assert_eq!(first_outcome, DropOutcome::Panic);
    assert_eq!(second_outcome, DropOutcome::ChannelClosed);
    assert_eq!(creates.load(Ordering::SeqCst), 1);
}

#[derive(Clone)]
struct BuildSideEffectFactory(Arc<AtomicUsize>);

impl WorkerSourceFactory for BuildSideEffectFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<usize>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        panic!("invalid builder reached source creation")
    }

    fn exact_len(&self) -> Option<usize> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Some(0)
    }
}

#[test]
fn invalid_stream_builders_fail_before_factory_or_worker_side_effects() {
    let callbacks = Arc::new(AtomicUsize::new(0));
    let factory = || BuildSideEffectFactory(Arc::clone(&callbacks));

    let invalid = [
        StreamDataLoaderBuilder::new(factory())
            .workers(0)
            .collate(VecCollate)
            .build(),
        StreamDataLoaderBuilder::new(factory())
            .batch_size(0)
            .collate(VecCollate)
            .build(),
        StreamDataLoaderBuilder::new(factory())
            .prefetch_factor(0)
            .collate(VecCollate)
            .build(),
        StreamDataLoaderBuilder::new(factory())
            .workers(usize::MAX)
            .prefetch_factor(2)
            .collate(VecCollate)
            .build(),
        StreamDataLoaderBuilder::new(factory())
            .timeout(Duration::MAX)
            .collate(VecCollate)
            .build(),
        StreamDataLoaderBuilder::new(factory())
            .workers(1_000_000)
            .prefetch_factor(32)
            .collate(VecCollate)
            .build(),
    ];
    for error in invalid {
        assert!(matches!(
            error,
            Err(RustTorchError::InvalidConfiguration { .. })
        ));
    }
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);

    let zero_worker = StreamDataLoaderBuilder::new(factory())
        .workers(0)
        .collate(VecCollate)
        .build();
    let Err(RustTorchError::InvalidConfiguration { field, reason }) = zero_worker else {
        panic!("zero workers should be a typed configuration error")
    };
    assert_eq!(field, "workers");
    assert!(reason.contains("batches_with_collate"));
}

const LARGE_INLINE_BYTES: usize = 1_048_576;
type LargeInline = [u8; LARGE_INLINE_BYTES];

#[derive(Clone)]
struct LargeInlineFactory {
    exact_calls: Arc<AtomicUsize>,
    create_calls: Arc<AtomicUsize>,
}

impl WorkerSourceFactory for LargeInlineFactory {
    type Sample = LargeInline;
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<LargeInline>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        self.create_calls.fetch_add(1, Ordering::SeqCst);
        Ok(std::iter::empty())
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact_calls.fetch_add(1, Ordering::SeqCst);
        Some(0)
    }
}

#[test]
fn large_inline_batch_and_reassembly_storage_are_rejected_before_callbacks() {
    for (prefetch_factor, batch_size) in [(1, 32), (32, 1)] {
        let exact_calls = Arc::new(AtomicUsize::new(0));
        let create_calls = Arc::new(AtomicUsize::new(0));
        let init_calls = Arc::new(AtomicUsize::new(0));
        let collate_calls = Arc::new(AtomicUsize::new(0));
        let result = StreamDataLoaderBuilder::new(LargeInlineFactory {
            exact_calls: Arc::clone(&exact_calls),
            create_calls: Arc::clone(&create_calls),
        })
        .prefetch_factor(prefetch_factor)
        .batch_size(batch_size)
        .worker_init(FnWorkerInit::new({
            let init_calls = Arc::clone(&init_calls);
            move |_: &WorkerContext| {
                init_calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(())
            }
        }))
        .collate(FnCollate::new({
            let collate_calls = Arc::clone(&collate_calls);
            move |samples: Vec<LargeInline>| {
                collate_calls.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(samples)
            }
        }))
        .build();

        assert!(matches!(
            result,
            Err(RustTorchError::InvalidConfiguration {
                field: "prefetch_factor",
                ..
            })
        ));
        assert_eq!(exact_calls.load(Ordering::SeqCst), 0);
        assert_eq!(create_calls.load(Ordering::SeqCst), 0);
        assert_eq!(init_calls.load(Ordering::SeqCst), 0);
        assert_eq!(collate_calls.load(Ordering::SeqCst), 0);
    }
}

const SPARSE_INLINE_BYTES: usize = 6 * 1024 * 1024;
type SparseInline = [u8; SPARSE_INLINE_BYTES];
const OVERSIZED_INLINE_BYTES: usize = 17 * 1024 * 1024;

#[derive(Clone)]
struct InlineBoundaryFactory<const BYTES: usize> {
    exact_calls: Arc<AtomicUsize>,
    create_calls: Arc<AtomicUsize>,
}

impl<const BYTES: usize> WorkerSourceFactory for InlineBoundaryFactory<BYTES> {
    type Sample = [u8; BYTES];
    type Error = Infallible;
    type Source = std::iter::Empty<Result<WorkerRecord<[u8; BYTES]>, Infallible>>;

    fn create(&self, _worker: WorkerContext) -> Result<Self::Source, Self::Error> {
        self.create_calls.fetch_add(1, Ordering::SeqCst);
        Ok(std::iter::empty())
    }

    fn exact_len(&self) -> Option<usize> {
        self.exact_calls.fetch_add(1, Ordering::SeqCst);
        Some(0)
    }
}

#[test]
fn sparse_one_entry_reassembly_uses_one_concrete_vec_slot() {
    let exact_calls = Arc::new(AtomicUsize::new(0));
    let create_calls = Arc::new(AtomicUsize::new(0));
    let init_calls = Arc::new(AtomicUsize::new(0));
    let collate_calls = Arc::new(AtomicUsize::new(0));
    let result = StreamDataLoaderBuilder::new(InlineBoundaryFactory::<SPARSE_INLINE_BYTES> {
        exact_calls: Arc::clone(&exact_calls),
        create_calls: Arc::clone(&create_calls),
    })
    .workers(1)
    .prefetch_factor(1)
    .batch_size(1)
    .worker_init(FnWorkerInit::new({
        let init_calls = Arc::clone(&init_calls);
        move |_: &WorkerContext| {
            init_calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        }
    }))
    .collate(FnCollate::new({
        let collate_calls = Arc::clone(&collate_calls);
        move |samples: Vec<SparseInline>| {
            collate_calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(samples)
        }
    }))
    .build();

    assert!(result.is_ok());
    assert_eq!(exact_calls.load(Ordering::SeqCst), 1);
    assert_eq!(create_calls.load(Ordering::SeqCst), 0);
    assert_eq!(init_calls.load(Ordering::SeqCst), 0);
    assert_eq!(collate_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn oversized_one_entry_vec_aggregate_is_rejected_before_every_callback() {
    let exact_calls = Arc::new(AtomicUsize::new(0));
    let create_calls = Arc::new(AtomicUsize::new(0));
    let init_calls = Arc::new(AtomicUsize::new(0));
    let collate_calls = Arc::new(AtomicUsize::new(0));
    let result = StreamDataLoaderBuilder::new(InlineBoundaryFactory::<OVERSIZED_INLINE_BYTES> {
        exact_calls: Arc::clone(&exact_calls),
        create_calls: Arc::clone(&create_calls),
    })
    .workers(1)
    .prefetch_factor(1)
    .batch_size(1)
    .persistent_workers(true)
    .worker_init(FnWorkerInit::new({
        let init_calls = Arc::clone(&init_calls);
        move |_: &WorkerContext| {
            init_calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        }
    }))
    .collate(FnCollate::new({
        let collate_calls = Arc::clone(&collate_calls);
        move |samples: Vec<[u8; OVERSIZED_INLINE_BYTES]>| {
            collate_calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(samples)
        }
    }))
    .build();

    assert!(matches!(
        result,
        Err(RustTorchError::InvalidConfiguration {
            field: "prefetch_factor",
            ..
        })
    ));
    assert_eq!(exact_calls.load(Ordering::SeqCst), 0);
    assert_eq!(create_calls.load(Ordering::SeqCst), 0);
    assert_eq!(init_calls.load(Ordering::SeqCst), 0);
    assert_eq!(collate_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn ordinary_zero_worker_stream_helpers_remain_source_compatible() -> Result<(), RustTorchError> {
    let ordinary = batches([Ok::<_, Infallible>(1), Ok(2), Ok(3)].into_iter(), 2, false)?
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(ordinary, vec![vec![1, 2], vec![3]]);

    let collated = batches_with_collate(
        [Ok::<_, Infallible>(1), Ok(2)].into_iter(),
        2,
        false,
        |values| Ok::<_, Infallible>(values.into_iter().sum::<usize>()),
    )?
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
    assert_eq!(collated, vec![3]);
    Ok(())
}
