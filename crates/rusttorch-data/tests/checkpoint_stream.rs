use std::convert::Infallible;

use rusttorch_data::{
    CheckpointSourceFactory, CheckpointableSource, LogicalSampleId, SequenceId,
    StreamDataLoaderBuilder, VecCollate, WorkerContext, WorkerRecord, WorkerSourceFactory,
};
use rusttorch_data::{
    Checkpointable, Collate, LoaderError, StreamCheckpointBuildError, TaskContext, Transform,
    WorkerCheckpoint,
};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Cursor {
    offset: usize,
    decoded: usize,
}
struct Source {
    cursor: Cursor,
    length: usize,
    stride: usize,
}
impl Iterator for Source {
    type Item = Result<WorkerRecord<(usize, usize)>, Infallible>;
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.cursor.offset;
        if value >= self.length {
            return None;
        }
        self.cursor.offset += self.stride;
        self.cursor.decoded += 1;
        Some(Ok(WorkerRecord {
            sequence: Some(SequenceId::new(value as u64)),
            logical_id: LogicalSampleId::new(value as u64),
            sample: (value, self.cursor.decoded),
        }))
    }
}
impl CheckpointableSource for Source {
    type Sample = (usize, usize);
    type State = Cursor;
    type Error = Infallible;
    fn snapshot(&self) -> Cursor {
        self.cursor.clone()
    }
    fn validate_snapshot(&self, _: &Cursor) -> Result<(), Infallible> {
        Ok(())
    }
    fn restore_validated(&mut self, state: &Cursor) {
        self.cursor = state.clone();
    }
}
struct Factory(usize);
impl WorkerSourceFactory for Factory {
    type Sample = (usize, usize);
    type Error = Infallible;
    type Source = Source;
    fn create(&self, worker: WorkerContext) -> Result<Source, Infallible> {
        Ok(Source {
            cursor: Cursor {
                offset: worker.info.id,
                decoded: 0,
            },
            length: self.0,
            stride: worker.info.num_workers,
        })
    }
    fn exact_len(&self) -> Option<usize> {
        Some(self.0)
    }
}
impl CheckpointSourceFactory for Factory {
    const CHECKPOINT_KIND: &'static str = "test.modulo.v1";
}

#[test]
fn checkpoints_replay_every_boundary_and_original_continuation() {
    for length in [0, 1, 2, 11] {
        for drop_last in [false, true] {
            for workers in [1, 3] {
                for prefetch in [1, 3] {
                    let build = || {
                        StreamDataLoaderBuilder::new(Factory(length))
                            .workers(workers)
                            .prefetch_factor(prefetch)
                            .batch_size(4)
                            .drop_last(drop_last)
                            .transform(Counter(0))
                            .collate(VecCollate)
                    };
                    let expected = build()
                        .build()
                        .unwrap()
                        .iter()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    for boundary in 0..=expected.len() {
                        let mut original = build().checkpointable("numbers.v1").build().unwrap();
                        let mut iter = original.iter();
                        for batch in &expected[..boundary] {
                            assert_eq!(iter.next().unwrap().unwrap(), *batch);
                        }
                        let state = iter.checkpoint().unwrap();
                        let again = iter.checkpoint().unwrap();
                        assert_eq!(state.next_sequence, again.next_sequence);
                        for lane in &state.lanes {
                            assert_eq!(lane.source.decoded, lane.transform);
                            let expected_offset = (lane.id..length)
                                .step_by(workers)
                                .find(|value| *value >= state.next_sequence as usize)
                                .unwrap_or_else(|| {
                                    lane.id + (lane.id..length).step_by(workers).count() * workers
                                });
                            assert_eq!(lane.source.offset, expected_offset);
                        }
                        let state =
                            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
                        let mut resumed = build().resume("numbers.v1", state).build().unwrap();
                        assert_eq!(
                            iter.collect::<Result<Vec<_>, _>>().unwrap(),
                            expected[boundary..]
                        );
                        assert_eq!(
                            resumed.iter().collect::<Result<Vec<_>, _>>().unwrap(),
                            expected[boundary..]
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct Counter(usize);
impl Transform<(usize, usize)> for Counter {
    type Output = (usize, usize, usize);
    type Error = Infallible;
    fn transform(
        &mut self,
        input: (usize, usize),
        _: &TaskContext,
    ) -> Result<Self::Output, Infallible> {
        self.0 += 1;
        Ok((input.0, input.1, self.0))
    }
}
impl WorkerCheckpoint for Counter {
    type State = usize;
    fn snapshot(&self) -> usize {
        self.0
    }
    fn validate_snapshot(&self, _: &usize) -> rusttorch_core::Result<()> {
        Ok(())
    }
    fn restore_validated(&mut self, state: &usize) {
        self.0 = *state;
    }
}

#[derive(Default)]
struct Counts {
    creates: AtomicUsize,
    source_applies: AtomicUsize,
    transform_applies: AtomicUsize,
    drops: AtomicUsize,
    reject_lane: AtomicUsize,
}
struct ProbeFactory {
    empty_lanes: usize,
    counts: Arc<Counts>,
    sequence: Vec<Option<u64>>,
    fail_at: Option<usize>,
    end_at: Option<usize>,
    block: Option<mpsc::Sender<(bool, u64)>>,
}
struct Probe {
    counts: Arc<Counts>,
    sequence: Vec<Option<u64>>,
    cursor: usize,
    lane: usize,
    fail_at: Option<usize>,
    end_at: Option<usize>,
    context: WorkerContext,
    block: Option<mpsc::Sender<(bool, u64)>>,
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.counts.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl Iterator for Probe {
    type Item = Result<WorkerRecord<usize>, std::io::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor;
        self.cursor += 1;
        if self.fail_at == Some(usize::MAX) {
            panic!("source read panic");
        }
        if self.end_at == Some(cursor) {
            return None;
        }
        if cursor == 1
            && let Some(block) = &self.block
        {
            block
                .send((
                    self.context.cancellation.is_cancelled(),
                    self.context.info.seed,
                ))
                .unwrap();
            self.context.cancellation.wait_cancelled();
        }
        if self.fail_at == Some(cursor) {
            return Some(Err(std::io::Error::other("decoder advanced then failed")));
        }
        self.sequence.get(cursor).map(|sequence| {
            Ok(WorkerRecord {
                sequence: sequence.map(SequenceId::new),
                logical_id: LogicalSampleId::new(cursor as u64),
                sample: cursor,
            })
        })
    }
}
impl CheckpointableSource for Probe {
    type Sample = usize;
    type State = usize;
    type Error = std::io::Error;
    fn snapshot(&self) -> usize {
        self.cursor
    }
    fn validate_snapshot(&self, state: &usize) -> Result<(), Self::Error> {
        if *state > self.sequence.len() + 1
            || self.counts.reject_lane.load(Ordering::SeqCst) == self.lane + 1
        {
            Err(std::io::Error::other("invalid decoder state"))
        } else {
            Ok(())
        }
    }
    fn restore_validated(&mut self, state: &usize) {
        self.cursor = *state;
        self.counts.source_applies.fetch_add(1, Ordering::SeqCst);
    }
    fn set_run_context(&mut self, context: WorkerContext) {
        self.context = context;
    }
}
impl WorkerSourceFactory for ProbeFactory {
    type Sample = usize;
    type Error = std::io::Error;
    type Source = Probe;
    fn create(&self, context: WorkerContext) -> Result<Probe, Self::Error> {
        self.counts.creates.fetch_add(1, Ordering::SeqCst);
        Ok(Probe {
            counts: Arc::clone(&self.counts),
            sequence: if context.info.id < self.empty_lanes {
                vec![]
            } else {
                self.sequence.clone()
            },
            cursor: 0,
            lane: context.info.id,
            fail_at: self.fail_at,
            end_at: self.end_at,
            context,
            block: self.block.clone(),
        })
    }
}
impl CheckpointSourceFactory for ProbeFactory {
    const CHECKPOINT_KIND: &'static str = "probe.v1";
}
fn probe(counts: &Arc<Counts>, sequence: &[Option<u64>]) -> ProbeFactory {
    ProbeFactory {
        empty_lanes: 0,
        counts: Arc::clone(counts),
        sequence: sequence.to_vec(),
        fail_at: None,
        end_at: None,
        block: None,
    }
}

#[test]
fn malformed_protocol_errors_once_and_ordinary_modes_stay_available() {
    for sequence in [
        vec![None],
        vec![Some(0), Some(0)],
        vec![Some(1), Some(0)],
        vec![Some(0), Some(2)],
    ] {
        let counts = Arc::new(Counts::default());
        let mut loader = StreamDataLoaderBuilder::new(probe(&counts, &sequence))
            .prefetch_factor(3)
            .collate(VecCollate)
            .checkpointable("probe")
            .build()
            .unwrap();
        let mut iter = loader.iter();
        let mut errors = 0;
        for item in iter.by_ref() {
            if let Err(error) = item {
                assert!(matches!(error, LoaderError::StreamProtocol { .. }));
                errors += 1;
            }
        }
        assert_eq!(errors, 1);
        assert!(iter.checkpoint().is_err());
    }
    for workers in [1, 3] {
        let counts = Arc::new(Counts::default());
        let result = StreamDataLoaderBuilder::new(probe(&counts, &[]))
            .workers(workers)
            .ordered(false)
            .collate(VecCollate)
            .checkpointable("probe")
            .build();
        assert!(result.is_err());
        assert_eq!(counts.creates.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn source_error_rolls_back_pre_attempt_and_preserves_error_chain() {
    use std::error::Error;
    let counts = Arc::new(Counts::default());
    let mut factory = probe(&counts, &[Some(0), Some(1)]);
    factory.fail_at = Some(1);
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .collate(VecCollate)
        .checkpointable("error")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), vec![0]);
    let error = iter.next().unwrap().unwrap_err();
    assert!(
        error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .to_string()
            .contains("decoder advanced")
    );
    assert!(iter.next().is_none());
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.lanes[0].source, 1);
    assert!(iter.next().unwrap().is_err());
    assert_eq!(iter.checkpoint().unwrap().lanes[0].source, 1);
}

#[test]
fn retained_source_gets_fresh_context_and_cancels_at_each_barrier_and_drop() {
    let seed = rusttorch_data::WorkerInfo::from_loader_seed(0, 1, 0, 0, 0)
        .unwrap()
        .seed;
    let counts = Arc::new(Counts::default());
    let (tx, rx) = mpsc::channel();
    let mut factory = probe(&counts, &[Some(0), Some(1), Some(2)]);
    factory.block = Some(tx);
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .prefetch_factor(3)
        .collate(VecCollate)
        .checkpointable("blocked")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), vec![0]);
    assert_eq!(rx.recv().unwrap(), (false, seed));
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.lanes[0].source, 1);
    assert_eq!(rx.recv().unwrap(), (false, seed));
    let again = iter.checkpoint().unwrap();
    assert_eq!(again.source_generation, state.source_generation);
    assert!(again.transport_generation > state.transport_generation);
    assert_eq!(rx.recv().unwrap(), (false, seed));
    drop(iter);
    assert_eq!(counts.creates.load(Ordering::SeqCst), 1);
    assert_eq!(counts.drops.load(Ordering::SeqCst), 1);
}

#[derive(Clone)]
struct ApplyCounter {
    counts: Arc<Counts>,
    calls: usize,
    fail: bool,
}
impl Transform<usize> for ApplyCounter {
    type Output = usize;
    type Error = std::io::Error;
    fn transform(&mut self, input: usize, _: &TaskContext) -> Result<usize, Self::Error> {
        self.calls += 1;
        if self.fail {
            Err(std::io::Error::other("transform advanced then failed"))
        } else {
            Ok(input)
        }
    }
}
impl WorkerCheckpoint for ApplyCounter {
    type State = usize;
    fn snapshot(&self) -> usize {
        self.calls
    }
    fn validate_snapshot(&self, state: &usize) -> rusttorch_core::Result<()> {
        if *state == usize::MAX {
            Err(rusttorch_core::RustTorchError::InvalidConfiguration {
                field: "transform",
                reason: "bad state".to_owned(),
            })
        } else {
            Ok(())
        }
    }
    fn restore_validated(&mut self, state: &usize) {
        self.calls = *state;
        self.counts.transform_applies.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Default)]
struct Coordinator {
    reject: bool,
    applies: usize,
}
impl Collate<usize> for Coordinator {
    type Batch = Vec<usize>;
    type Error = Infallible;
    fn collate(&mut self, samples: Vec<usize>) -> Result<Vec<usize>, Infallible> {
        Ok(samples)
    }
}
impl Checkpointable for Coordinator {
    type State = ();
    fn save_state(&self) {}
    fn validate_state(&self, _: &()) -> rusttorch_core::Result<()> {
        if self.reject {
            Err(rusttorch_core::RustTorchError::InvalidConfiguration {
                field: "coordinator",
                reason: "rejected".to_owned(),
            })
        } else {
            Ok(())
        }
    }
    fn load_validated(&mut self, _: &()) {
        self.applies += 1;
    }
}

#[test]
fn resume_validation_is_atomic_and_static_rejection_precedes_callbacks() {
    let initial = Arc::new(Counts::default());
    let builder = |counts: &Arc<Counts>, reject| {
        StreamDataLoaderBuilder::new(probe(counts, &[]))
            .workers(3)
            .transform(ApplyCounter {
                counts: Arc::clone(counts),
                calls: 0,
                fail: false,
            })
            .collate(Coordinator { reject, applies: 0 })
    };
    let mut loader = builder(&initial, false)
        .checkpointable("transaction")
        .build()
        .unwrap();
    let state = loader.iter().checkpoint().unwrap();
    for variant in 0..8 {
        let counts = Arc::new(Counts::default());
        let mut state = state.clone();
        match variant {
            0 => counts.reject_lane.store(3, Ordering::SeqCst),
            1 => state.lanes[2].transform = usize::MAX,
            2 => {}
            3 => state.lanes[2].id = 0,
            4 => state.configuration.workers = 4,
            5 => state.schema_version = 999,
            6 => state.factory_kind = "wrong".to_owned(),
            _ => state.next_batch = u64::MAX,
        }
        let result = builder(&counts, variant == 2)
            .resume("transaction", state)
            .build();
        assert!(result.is_err());
        assert_eq!(counts.source_applies.load(Ordering::SeqCst), 0);
        assert_eq!(counts.transform_applies.load(Ordering::SeqCst), 0);
        assert_eq!(
            counts.creates.load(Ordering::SeqCst),
            counts.drops.load(Ordering::SeqCst)
        );
        if variant >= 3 {
            assert_eq!(counts.creates.load(Ordering::SeqCst), 0);
        }
        if variant == 0 {
            assert!(matches!(
                result,
                Err(StreamCheckpointBuildError::Source { worker: 2, .. })
            ));
        }
    }
}

#[test]
fn transform_failure_restores_paired_state_and_nonfused_end_replays() {
    let counts = Arc::new(Counts::default());
    let mut loader = StreamDataLoaderBuilder::new(probe(&counts, &[Some(0)]))
        .transform(ApplyCounter {
            counts: Arc::clone(&counts),
            calls: 0,
            fail: true,
        })
        .collate(VecCollate)
        .checkpointable("transform")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert!(iter.next().unwrap().is_err());
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.lanes[0].source, 0);
    assert_eq!(state.lanes[0].transform, 0);
    assert!(iter.next().unwrap().is_err());
    drop(iter);
    // None at zero followed by a possible record at one: do not resume after None.
    let mut factory = probe(&counts, &[Some(0), Some(1)]);
    factory.end_at = Some(0);
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .collate(VecCollate)
        .checkpointable("end")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert!(iter.next().is_none());
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.lanes[0].source, 0);
    assert!(iter.next().is_none());
    assert_eq!(iter.checkpoint().unwrap().lanes[0].source, 0);
}

#[derive(Clone)]
struct BlockingTransform {
    calls: usize,
    entered: mpsc::Sender<()>,
}
impl Transform<usize> for BlockingTransform {
    type Output = usize;
    type Error = Infallible;
    fn transform(&mut self, input: usize, context: &TaskContext) -> Result<usize, Infallible> {
        self.calls += 1;
        if input == 1 {
            self.entered.send(()).unwrap();
            context.cancellation.wait_cancelled();
        }
        Ok(input)
    }
}
impl WorkerCheckpoint for BlockingTransform {
    type State = usize;
    fn snapshot(&self) -> usize {
        self.calls
    }
    fn validate_snapshot(&self, _: &usize) -> rusttorch_core::Result<()> {
        Ok(())
    }
    fn restore_validated(&mut self, state: &usize) {
        self.calls = *state;
    }
}

#[test]
fn active_transform_rolls_back_to_paired_pre_source_boundary() {
    let counts = Arc::new(Counts::default());
    let (entered, events) = mpsc::channel();
    let mut loader = StreamDataLoaderBuilder::new(probe(&counts, &[Some(0), Some(1), Some(2)]))
        .transform(BlockingTransform { calls: 0, entered })
        .collate(VecCollate)
        .checkpointable("in-transform")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), vec![0]);
    events.recv().unwrap();
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.lanes[0].source, 1);
    assert_eq!(state.lanes[0].transform, 1);
    events.recv().unwrap();
    drop(iter);
    assert_eq!(counts.drops.load(Ordering::SeqCst), 1);
}

struct BrokenFactory(bool);
impl WorkerSourceFactory for BrokenFactory {
    type Sample = usize;
    type Error = std::io::Error;
    type Source = Probe;
    fn create(&self, _: WorkerContext) -> Result<Probe, std::io::Error> {
        assert!(!self.0, "factory panic");
        Err(std::io::Error::other("source init failed"))
    }
}
impl CheckpointSourceFactory for BrokenFactory {
    const CHECKPOINT_KIND: &'static str = "broken.v1";
}

#[test]
fn init_errors_and_panics_preserve_context_and_join() {
    use std::error::Error;
    let result = StreamDataLoaderBuilder::new(BrokenFactory(false))
        .workers(3)
        .collate(VecCollate)
        .checkpointable("init")
        .build();
    let error = result.err().unwrap();
    assert!(matches!(&error, StreamCheckpointBuildError::Source { worker, .. } if *worker < 3));
    assert_eq!(error.source().unwrap().to_string(), "source init failed");
    let result = StreamDataLoaderBuilder::new(BrokenFactory(true))
        .workers(3)
        .collate(VecCollate)
        .checkpointable("init")
        .build();
    assert!(matches!(
        result,
        Err(StreamCheckpointBuildError::WorkerPanic { .. })
    ));
    let counts = Arc::new(Counts::default());
    let result = StreamDataLoaderBuilder::new(probe(&counts, &[]))
        .transform_factory(rusttorch_data::FnTransformFactory::new(
            |_: Option<&WorkerContext>| {
                Err::<rusttorch_data::IdentityTransform, _>(std::io::Error::other(
                    "transform init failed",
                ))
            },
        ))
        .collate(VecCollate)
        .checkpointable("init")
        .build();
    let error = result.err().unwrap();
    assert!(matches!(
        &error,
        StreamCheckpointBuildError::TransformFactory { worker: 0, .. }
    ));
    assert_eq!(error.source().unwrap().to_string(), "transform init failed");
    assert_eq!(
        counts.creates.load(Ordering::SeqCst),
        counts.drops.load(Ordering::SeqCst)
    );
    let mut factory = probe(&counts, &[]);
    factory.fail_at = Some(usize::MAX);
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .collate(VecCollate)
        .checkpointable("panic")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert!(matches!(
        iter.next().unwrap(),
        Err(LoaderError::StreamWorkerPanic { .. })
    ));
    assert!(iter.checkpoint().is_err());
    assert!(iter.next().is_none());
    assert_eq!(
        counts.creates.load(Ordering::SeqCst),
        counts.drops.load(Ordering::SeqCst)
    );
}

#[test]
fn global_duplicate_and_state_configuration_drift_are_rejected() {
    let counts = Arc::new(Counts::default());
    let mut loader = StreamDataLoaderBuilder::new(probe(&counts, &[Some(0)]))
        .workers(3)
        .collate(VecCollate)
        .checkpointable("duplicates")
        .build()
        .unwrap();
    let errors = loader.iter().filter(Result::is_err).collect::<Vec<_>>();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0], Err(LoaderError::StreamProtocol { .. })));
    let builder = || {
        StreamDataLoaderBuilder::new(Factory(2))
            .workers(3)
            .collate(VecCollate)
    };
    let state = builder()
        .checkpointable("drift")
        .build()
        .unwrap()
        .iter()
        .checkpoint()
        .unwrap();
    for variant in 0..12 {
        let mut state = state.clone();
        match variant {
            0 => state.configuration.prefetch_factor += 1,
            1 => state.configuration.batch_size += 1,
            2 => state.configuration.drop_last = true,
            3 => state.configuration.loader_seed += 1,
            4 => state.configuration.rank += 1,
            5 => state.epoch += 1,
            6 => state.source_length = Some(3),
            7 => state.configuration.pin_request = rusttorch_data::CheckpointPinRequest::Auto,
            8 => state.rng_derivation_version += 1,
            9 => state.worker_seed_derivation_version += 1,
            10 => state.source_generation = u64::MAX,
            _ => state.transport_generation = u64::MAX,
        }
        assert!(builder().resume("drift", state).build().is_err());
    }
}

#[test]
fn terminal_shards_do_not_hide_a_full_active_lane_with_a_gap() {
    let counts = Arc::new(Counts::default());
    let mut factory = probe(&counts, &[Some(1), Some(2)]);
    factory.empty_lanes = 2;
    let mut loader = StreamDataLoaderBuilder::new(factory)
        .workers(3)
        .prefetch_factor(1)
        .collate(VecCollate)
        .checkpointable("terminal-gap")
        .build()
        .unwrap();
    let mut iter = loader.iter();
    assert!(matches!(
        iter.next().unwrap(),
        Err(LoaderError::StreamProtocol {
            sequence: Some(0),
            ..
        })
    ));
    assert!(iter.next().is_none());
    assert_eq!(counts.drops.load(Ordering::SeqCst), 3);
}

#[test]
fn capacity_overflow_and_unsupported_settings_reject_before_workers() {
    for variant in 0..5 {
        let counts = Arc::new(Counts::default());
        let builder = StreamDataLoaderBuilder::new(probe(&counts, &[]))
            .collate(VecCollate)
            .checkpointable("capacity");
        let builder = match variant {
            0 => builder.workers(usize::MAX),
            1 => builder.prefetch_factor(usize::MAX),
            2 => builder.batch_size(usize::MAX),
            3 => builder.persistent_workers(true),
            _ => builder.timeout(std::time::Duration::from_secs(1)),
        };
        assert!(builder.build().is_err());
        assert_eq!(counts.creates.load(Ordering::SeqCst), 0);
    }
}

struct LargeSource;
impl Iterator for LargeSource {
    type Item = Result<WorkerRecord<usize>, Infallible>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
impl CheckpointableSource for LargeSource {
    type Sample = usize;
    type Error = Infallible;
    type State = [[[[u8; 32]; 32]; 32]; 32];
    fn snapshot(&self) -> Self::State {
        panic!("capacity rejects before snapshots")
    }
    fn validate_snapshot(&self, _: &Self::State) -> Result<(), Infallible> {
        Ok(())
    }
    fn restore_validated(&mut self, _: &Self::State) {}
}
struct LargeFactory(Arc<AtomicUsize>);
impl WorkerSourceFactory for LargeFactory {
    type Sample = usize;
    type Error = Infallible;
    type Source = LargeSource;
    fn create(&self, _: WorkerContext) -> Result<LargeSource, Infallible> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(LargeSource)
    }
}
impl CheckpointSourceFactory for LargeFactory {
    const CHECKPOINT_KIND: &'static str = "large.v1";
}

#[test]
fn inline_snapshot_shapes_participate_in_aggregate_retention_limit() {
    let creates = Arc::new(AtomicUsize::new(0));
    let result = StreamDataLoaderBuilder::new(LargeFactory(Arc::clone(&creates)))
        .workers(3)
        .prefetch_factor(32)
        .collate(VecCollate)
        .checkpointable("large")
        .build();
    assert!(matches!(
        result,
        Err(StreamCheckpointBuildError::Configuration(_))
    ));
    assert_eq!(creates.load(Ordering::SeqCst), 0);
}

struct PinBatch(Vec<usize>, Arc<AtomicUsize>);
impl rusttorch_data::PinMemory for PinBatch {
    fn pin_memory(self, _: rusttorch_core::Device) -> rusttorch_core::Result<Self> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Ok(self)
    }
}
struct PinCollator {
    collates: Arc<AtomicUsize>,
    pins: Arc<AtomicUsize>,
    visible: usize,
}
impl Collate<usize> for PinCollator {
    type Batch = PinBatch;
    type Error = Infallible;
    fn collate(&mut self, samples: Vec<usize>) -> Result<PinBatch, Infallible> {
        self.collates.fetch_add(1, Ordering::SeqCst);
        self.visible += 1;
        Ok(PinBatch(samples, Arc::clone(&self.pins)))
    }
}
impl Checkpointable for PinCollator {
    type State = usize;
    fn save_state(&self) -> usize {
        self.visible
    }
    fn validate_state(&self, _: &usize) -> rusttorch_core::Result<()> {
        Ok(())
    }
    fn load_validated(&mut self, state: &usize) {
        self.visible = *state;
    }
}

#[test]
fn checkpoints_never_collate_or_pin_unpublished_records() {
    let counts = Arc::new(Counts::default());
    let collates = Arc::new(AtomicUsize::new(0));
    let pins = Arc::new(AtomicUsize::new(0));
    let build = || {
        StreamDataLoaderBuilder::new(probe(&counts, &[Some(0), Some(1), Some(2)])).collate(
            PinCollator {
                collates: Arc::clone(&collates),
                pins: Arc::clone(&pins),
                visible: 0,
            },
        )
    };
    // Pin setter after checkpoint selection must preserve exact capability.
    let mut loader = build().checkpointable("pin").pin_memory().build().unwrap();
    let pin_enabled = matches!(
        loader.pin_memory_status(),
        rusttorch_data::PinMemoryStatus::Enabled(_)
    );
    let mut iter = loader.iter();
    assert_eq!(iter.checkpoint().unwrap().coordinator, 0);
    assert_eq!(collates.load(Ordering::SeqCst), 0);
    assert_eq!(pins.load(Ordering::SeqCst), 0);
    assert_eq!(iter.next().unwrap().unwrap().0, vec![0]);
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.coordinator, 1);
    assert_eq!(collates.load(Ordering::SeqCst), 1);
    assert_eq!(pins.load(Ordering::SeqCst), usize::from(pin_enabled));
    drop(iter);
    let mut resumed = build().resume("pin", state).pin_memory().build().unwrap();
    let mut iter = resumed.iter();
    assert_eq!(iter.next().unwrap().unwrap().0, vec![1]);
    assert_eq!(iter.checkpoint().unwrap().coordinator, 2);
}
