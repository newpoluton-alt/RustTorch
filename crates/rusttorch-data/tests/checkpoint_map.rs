use std::{cell::Cell, convert::Infallible, error::Error, fmt, rc::Rc, sync::Arc};

use rusttorch_core::{Device, Result, RustTorchError, Tensor, available_devices};
use rusttorch_data::{
    BatchSampler, CheckpointBuildError, CheckpointPinRequest, CheckpointPinStatus, Checkpointable,
    Collate, ConcatDataset, DataLoader, Dataset, DistributedSampler, FnTransform,
    FnTransformFactory, IdentityTransform, LOADER_STATE_SCHEMA_VERSION, LoaderError, LoaderState,
    RandomSampler, ReplaySafeDataset, ReplaySafeMap, Sampler, SamplerCheckpoint, SequentialSampler,
    StackDataset, Stateless, Subset, SubsetRandomSampler, TASK_RNG_DERIVATION_VERSION, TaskContext,
    TransactionalCheckpoint, TransactionalMap, Transform, VecCollate,
    WORKER_SEED_DERIVATION_VERSION, WeightedRandomSampler,
};

fn resume_ok<T>(result: std::result::Result<T, CheckpointBuildError<Infallible>>) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(CheckpointBuildError::Configuration(error)) => Err(error),
        Err(CheckpointBuildError::TransformFactory(error)) => match error {},
    }
}

#[derive(Clone)]
struct Rows(Vec<i64>);

impl Dataset for Rows {
    type Sample = i64;
    type Error = Infallible;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Sample, Self::Error> {
        Ok(self.0[index])
    }
}

impl ReplaySafeDataset for Rows {}

fn replay_rows(values: &[i64]) -> ReplaySafeMap<Rows> {
    ReplaySafeMap::new(Rows(values.to_vec()))
}

#[test]
fn immutable_dataset_adapters_propagate_replay_safety() {
    fn accepts_replay_safe<T: ReplaySafeDataset>() {}
    fn accepts_send_sync<T: Send + Sync>() {}

    accepts_replay_safe::<Arc<Rows>>();
    accepts_replay_safe::<ConcatDataset<Rows>>();
    accepts_replay_safe::<Subset<Rows>>();
    accepts_replay_safe::<StackDataset<(Rows, Rows)>>();
    accepts_send_sync::<LoaderState<(), rusttorch_data::SequentialSamplerState>>();
}

#[test]
fn json_round_trip_resumes_every_serial_boundary_exactly() -> Result<()> {
    let expected = [vec![0, 1], vec![2, 3], vec![4]];

    for consumed in 0..=expected.len() {
        let mut loader = DataLoader::builder(replay_rows(&[0, 1, 2, 3, 4]))
            .batch_size(2)
            .collate(VecCollate)
            .dataset_identity("rows-v1".to_owned())
            .build()?;
        let mut iteration = loader.iter();
        let prefix = iteration
            .by_ref()
            .take(consumed)
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("prefix succeeds");
        let state = iteration
            .checkpoint()
            .expect("visible boundary checkpoints");
        let continued = iteration
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("original iterator continues");

        let json = serde_json::to_string(&state).expect("state serializes");
        let decoded: LoaderState<_, _, _, _> =
            serde_json::from_str(&json).expect("state deserializes");
        let mut resumed = resume_ok(
            DataLoader::builder(replay_rows(&[0, 1, 2, 3, 4]))
                .batch_size(2)
                .collate(VecCollate)
                .dataset_identity("rows-v1".to_owned())
                .resume_from(decoded)
                .build(),
        )?;
        let suffix = resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("resume succeeds");

        assert_eq!(prefix, expected[..consumed]);
        assert_eq!(continued, expected[consumed..]);
        assert_eq!(suffix, expected[consumed..]);
        assert_eq!(state.schema_version, LOADER_STATE_SCHEMA_VERSION);
        assert_eq!(state.next_batch, consumed as u64);
        assert_eq!(state.next_logical_sample, [0, 2, 4, 5][consumed]);
        assert_eq!(state.rng_derivation_version, TASK_RNG_DERIVATION_VERSION);
        assert_eq!(
            state.worker_seed_derivation_version,
            WORKER_SEED_DERIVATION_VERSION
        );
    }
    Ok(())
}

#[test]
fn short_tail_drop_last_and_invalid_boundaries_are_explicit() -> Result<()> {
    let mut dropped = DataLoader::builder(replay_rows(&[0, 1, 2, 3, 4]))
        .batch_size(2)
        .drop_last(true)
        .collate(VecCollate)
        .dataset_identity("drop".to_owned())
        .build()?;
    let mut iteration = dropped.iter();
    assert_eq!(iteration.next().unwrap().unwrap(), [0, 1]);
    assert_eq!(iteration.next().unwrap().unwrap(), [2, 3]);
    assert!(iteration.next().is_none());
    assert!(matches!(
        iteration.checkpoint(),
        Err(LoaderError::Checkpoint { .. })
    ));

    #[derive(Clone)]
    struct FailingRows;
    impl Dataset for FailingRows {
        type Sample = i64;
        type Error = &'static str;
        fn len(&self) -> usize {
            3
        }
        fn get(&self, index: usize) -> std::result::Result<i64, Self::Error> {
            (index != 2).then_some(index as i64).ok_or("boom")
        }
    }
    impl ReplaySafeDataset for FailingRows {}

    let mut failing = DataLoader::builder(ReplaySafeMap::new(FailingRows))
        .batch_size(1)
        .collate(VecCollate)
        .dataset_identity("failure".to_owned())
        .build()?;
    let mut before = failing.iter();
    assert_eq!(before.next().unwrap().unwrap(), [0]);
    let state = before.checkpoint().unwrap();
    assert_eq!(before.next().unwrap().unwrap(), [1]);
    assert!(before.next().unwrap().is_err());
    assert!(matches!(
        before.checkpoint(),
        Err(LoaderError::Checkpoint { .. })
    ));

    let mut resumed = resume_ok(
        DataLoader::builder(ReplaySafeMap::new(FailingRows))
            .batch_size(1)
            .collate(VecCollate)
            .dataset_identity("failure".to_owned())
            .resume_from(state)
            .build(),
    )?;
    let mut after = resumed.iter();
    assert_eq!(after.next().unwrap().unwrap(), [1]);
    assert!(after.next().unwrap().is_err());
    Ok(())
}

macro_rules! assert_resume_matches {
    ($sampler:expr, $identity:expr, $epoch:expr, $rank:expr $(,)?) => {{
        let values = (0..16).collect::<Vec<_>>();
        let mut uninterrupted = DataLoader::builder(replay_rows(&values))
            .sampler($sampler?)
            .epoch($epoch)
            .rank($rank)
            .batch_size(3)
            .collate(VecCollate)
            .dataset_identity($identity.to_owned())
            .build()?;
        let mut iter = uninterrupted.iter();
        let first = iter.next().unwrap().unwrap();
        let state = iter.checkpoint().unwrap();
        let rest = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();

        let mut resumed = resume_ok(
            DataLoader::builder(replay_rows(&values))
                .sampler($sampler?)
                .epoch($epoch)
                .rank($rank)
                .batch_size(3)
                .collate(VecCollate)
                .dataset_identity($identity.to_owned())
                .resume_from(state)
                .build(),
        )?;
        let resumed_rest = resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert!(!first.is_empty());
        assert_eq!(resumed_rest, rest);
    }};
}

#[test]
fn every_concrete_sampler_and_epoch_restores_its_active_cursor() -> Result<()> {
    assert_resume_matches!(
        Ok::<_, RustTorchError>(SequentialSampler::new(16)),
        "sequential",
        7,
        0
    );
    assert_resume_matches!(RandomSampler::new(16, 31), "random", 3, 0);
    assert_resume_matches!(
        RandomSampler::without_replacement(16, 23, 32),
        "random-long",
        4,
        0,
    );
    assert_resume_matches!(
        RandomSampler::with_replacement(16, 23, 33),
        "replacement",
        5,
        0,
    );
    assert_resume_matches!(
        SubsetRandomSampler::new(vec![1, 3, 5, 7, 9, 11], 34),
        "subset",
        6,
        0,
    );
    assert_resume_matches!(
        WeightedRandomSampler::new(vec![1.0; 16], 20, true, 35),
        "weighted-replacement",
        8,
        0,
    );
    assert_resume_matches!(
        WeightedRandomSampler::new(vec![1.0; 16], 12, false, 36),
        "weighted",
        9,
        0,
    );
    assert_resume_matches!(
        DistributedSampler::new(16, 3, 1, true, 37, false),
        "distributed-padded",
        10,
        1,
    );
    assert_resume_matches!(
        DistributedSampler::new(16, 3, 1, true, 38, true),
        "distributed-truncated",
        11,
        1,
    );
    Ok(())
}

#[test]
fn explicit_batch_sampler_and_no_batch_converter_resume() -> Result<()> {
    let batches = BatchSampler::new(SequentialSampler::new(5), 2, false)?;
    let mut explicit = DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
        .batch_sampler(batches)
        .collate(VecCollate)
        .dataset_identity("batches".to_owned())
        .build()?;
    let mut iter = explicit.iter();
    assert_eq!(iter.next().unwrap().unwrap(), [10, 20]);
    let state = iter.checkpoint().unwrap();
    let mut wrong = state.clone();
    wrong.sampler.batch_size += 1;
    assert!(
        DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(5), 2, false)?)
            .collate(VecCollate)
            .dataset_identity("batches".to_owned())
            .resume_from(wrong)
            .build()
            .is_err()
    );
    let mut wrong = state.clone();
    wrong.sampler.next_logical_sample += 1;
    assert!(
        DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(5), 2, false)?)
            .collate(VecCollate)
            .dataset_identity("batches".to_owned())
            .resume_from(wrong)
            .build()
            .is_err()
    );
    let mut wrong = state.clone();
    wrong.sampler.drop_last = true;
    assert!(
        DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(5), 2, false)?)
            .collate(VecCollate)
            .dataset_identity("batches".to_owned())
            .resume_from(wrong)
            .build()
            .is_err()
    );
    let mut wrong = state.clone();
    wrong.sampler.next_batch += 1;
    assert!(
        DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(5), 2, false)?)
            .collate(VecCollate)
            .dataset_identity("batches".to_owned())
            .resume_from(wrong)
            .build()
            .is_err()
    );
    let mut resumed = resume_ok(
        DataLoader::builder(replay_rows(&[10, 20, 30, 40, 50]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(5), 2, false)?)
            .collate(VecCollate)
            .dataset_identity("batches".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        [vec![30, 40], vec![50]]
    );

    let mut no_batch = DataLoader::builder(replay_rows(&[2, 3, 5]))
        .without_batching()
        .dataset_identity("no-batch".to_owned())
        .build()?;
    let mut iter = no_batch.iter();
    assert_eq!(iter.next().unwrap().unwrap(), 2);
    let state = iter.checkpoint().unwrap();
    let mut resumed = resume_ok(
        DataLoader::builder(replay_rows(&[2, 3, 5]))
            .without_batching()
            .dataset_identity("no-batch".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        [3, 5]
    );
    Ok(())
}

#[derive(Clone)]
struct StatefulTransform {
    calls: u64,
}

impl Transform<i64> for StatefulTransform {
    type Output = i64;
    type Error = Infallible;

    fn transform(
        &mut self,
        value: i64,
        _context: &TaskContext,
    ) -> std::result::Result<Self::Output, Self::Error> {
        let output = value + self.calls as i64;
        self.calls += 1;
        Ok(output)
    }
}

impl TransactionalCheckpoint for StatefulTransform {
    type State = u64;
    fn snapshot(&self) -> Self::State {
        self.calls
    }
    fn validate_snapshot(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }
    fn restore_validated(&mut self, state: &Self::State) {
        self.calls = *state;
    }
}

#[derive(Clone)]
struct StatelessDouble;
impl Stateless for StatelessDouble {}
impl Transform<i64> for StatelessDouble {
    type Output = i64;
    type Error = Infallible;
    fn transform(&mut self, value: i64, _: &TaskContext) -> std::result::Result<i64, Infallible> {
        Ok(value * 2)
    }
}

#[derive(Clone)]
struct DeterministicTransformFailure;

impl Stateless for DeterministicTransformFailure {}

impl Transform<i64> for DeterministicTransformFailure {
    type Output = i64;
    type Error = &'static str;

    fn transform(&mut self, value: i64, _: &TaskContext) -> std::result::Result<i64, &'static str> {
        (value != 2).then_some(value).ok_or("transform boom")
    }
}

#[test]
fn deterministic_transform_failure_recurs_at_the_same_logical_batch() -> Result<()> {
    let build = || {
        DataLoader::builder(replay_rows(&[0, 1, 2, 3]))
            .transform(DeterministicTransformFailure)
            .checkpoint_stateless()
            .collate(VecCollate)
            .dataset_identity("transform-failure".to_owned())
    };
    let mut loader = build().build()?;
    let mut iteration = loader.iter();
    assert_eq!(iteration.next().unwrap().unwrap(), [0]);
    let state = iteration.checkpoint().unwrap();
    assert_eq!(iteration.next().unwrap().unwrap(), [1]);
    assert!(matches!(iteration.next(), Some(Err(_))));

    let mut resumed = resume_ok(build().resume_from(state).build())?;
    let mut iteration = resumed.iter();
    assert_eq!(iteration.next().unwrap().unwrap(), [1]);
    assert!(matches!(iteration.next(), Some(Err(_))));
    Ok(())
}

#[derive(Clone)]
struct SumCollator {
    calls: u64,
}
impl Collate<i64> for SumCollator {
    type Batch = i64;
    type Error = Infallible;
    fn collate(&mut self, values: Vec<i64>) -> std::result::Result<i64, Infallible> {
        self.calls += 1;
        Ok(values.into_iter().sum::<i64>() + self.calls as i64)
    }
}
impl Checkpointable for SumCollator {
    type State = u64;
    fn save_state(&self) -> Self::State {
        self.calls
    }
    fn validate_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }
    fn load_validated(&mut self, state: &Self::State) {
        self.calls = *state;
    }
}

#[test]
fn stateful_transform_and_coordinator_restore_after_last_visible_batch() -> Result<()> {
    let mut loader = DataLoader::builder(replay_rows(&[1, 2, 3, 4]))
        .batch_size(2)
        .transform(StatefulTransform { calls: 0 })
        .checkpoint_transactional()
        .collate(SumCollator { calls: 0 })
        .dataset_identity("components".to_owned())
        .build()?;
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), 5);
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.transform, 2);
    assert_eq!(state.collate, 1);
    let expected = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();

    let mut resumed = resume_ok(
        DataLoader::builder(replay_rows(&[1, 2, 3, 4]))
            .batch_size(2)
            .transform(StatefulTransform { calls: 0 })
            .checkpoint_transactional()
            .collate(SumCollator { calls: 0 })
            .dataset_identity("components".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        expected
    );

    let mut stateless = DataLoader::builder(replay_rows(&[1, 2]))
        .transform(StatelessDouble)
        .checkpoint_stateless()
        .collate(VecCollate)
        .dataset_identity("stateless".to_owned())
        .build()?;
    assert_eq!(stateless.iter().next().unwrap().unwrap(), [2]);

    let opaque = || {
        DataLoader::builder(replay_rows(&[1, 2, 3]))
            .transform(FnTransform::new(|value, _: &TaskContext| {
                Ok::<_, Infallible>(value * 3)
            }))
            .checkpoint_stateless()
            .collate(VecCollate)
            .dataset_identity("opaque-stateless".to_owned())
    };
    let mut opaque_source = opaque().build()?;
    let mut iter = opaque_source.iter();
    assert_eq!(iter.next().unwrap().unwrap(), [3]);
    let state = iter.checkpoint().unwrap();
    let expected = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    let mut opaque_resumed = resume_ok(opaque().resume_from(state).build())?;
    assert_eq!(
        opaque_resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        expected
    );

    let mut explicit = DataLoader::builder(replay_rows(&[1, 2, 3, 4]))
        .batch_sampler(BatchSampler::new(SequentialSampler::new(4), 2, false)?)
        .collate(SumCollator { calls: 0 })
        .dataset_identity("explicit-collator".to_owned())
        .build()?;
    let mut iter = explicit.iter();
    assert_eq!(iter.next().unwrap().unwrap(), 4);
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.collate, 1);
    let expected = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    let mut resumed = resume_ok(
        DataLoader::builder(replay_rows(&[1, 2, 3, 4]))
            .batch_sampler(BatchSampler::new(SequentialSampler::new(4), 2, false)?)
            .collate(SumCollator { calls: 0 })
            .dataset_identity("explicit-collator".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        expected
    );
    Ok(())
}

#[test]
fn setting_epoch_before_first_resumed_iteration_discards_the_old_cursor() -> Result<()> {
    let values = (0..12).collect::<Vec<_>>();
    let build = |epoch| {
        DataLoader::builder(replay_rows(&values))
            .sampler(RandomSampler::new(values.len(), 73).expect("nonempty sampler"))
            .epoch(epoch)
            .batch_size(2)
            .transform(StatefulTransform { calls: 0 })
            .checkpoint_transactional()
            .collate(VecCollate)
            .dataset_identity("epoch-transition".to_owned())
    };

    let mut source = build(3).build()?;
    let mut iteration = source.iter();
    let _ = iteration.next().unwrap().unwrap();
    let state = iteration.checkpoint().unwrap();
    assert_eq!(state.epoch, 3);
    assert_eq!(state.transform, 2);

    let mut resumed = resume_ok(build(3).resume_from(state).build())?;
    resumed.set_epoch(4);
    let mut iteration = resumed.iter();
    let reset_state = iteration.checkpoint().unwrap();
    assert_eq!(reset_state.epoch, 4);
    assert_eq!(reset_state.sampler.epoch, 4);
    assert_eq!(reset_state.sampler.position, 0);
    assert_eq!(reset_state.next_batch, 0);
    assert_eq!(reset_state.next_logical_sample, 0);
    assert_eq!(reset_state.transform, 0);
    let actual = iteration
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    let mut fresh = build(4).build()?;
    let expected = fresh
        .iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(actual, expected);
    Ok(())
}

#[derive(Clone)]
struct TransactionalRows {
    values: Vec<i64>,
    offset: Cell<u64>,
}
impl Dataset for TransactionalRows {
    type Sample = i64;
    type Error = Infallible;
    fn len(&self) -> usize {
        self.values.len()
    }
    fn get(&self, index: usize) -> std::result::Result<i64, Infallible> {
        let offset = self.offset.get();
        self.offset.set(offset + 1);
        Ok(self.values[index] + offset as i64)
    }
}
impl Checkpointable for TransactionalRows {
    type State = u64;
    fn save_state(&self) -> Self::State {
        self.offset.get()
    }
    fn validate_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }
    fn load_validated(&mut self, state: &Self::State) {
        self.offset.set(*state);
    }
}

#[test]
fn transactional_dataset_restores_serial_fetch_state() -> Result<()> {
    let dataset = TransactionalMap::new(TransactionalRows {
        values: vec![10, 20, 30],
        offset: Cell::new(0),
    });
    let mut loader = DataLoader::builder(dataset)
        .collate(VecCollate)
        .dataset_identity("transactional".to_owned())
        .build()?;
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), [10]);
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.dataset, 1);

    let dataset = TransactionalMap::new(TransactionalRows {
        values: vec![10, 20, 30],
        offset: Cell::new(0),
    });
    let mut resumed = resume_ok(
        DataLoader::builder(dataset)
            .collate(VecCollate)
            .dataset_identity("transactional".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        [vec![21], vec![32]]
    );
    Ok(())
}

#[derive(Default)]
struct PlanObservations {
    kind_calls: Cell<usize>,
    distributed_calls: Cell<usize>,
    epoch_applies: Cell<usize>,
    cursor_applies: Cell<usize>,
}

impl PlanObservations {
    fn reset(&self) {
        self.kind_calls.set(0);
        self.distributed_calls.set(0);
        self.epoch_applies.set(0);
        self.cursor_applies.set(0);
    }

    fn assert_untouched(&self, case: &str) {
        assert_eq!(self.kind_calls.get(), 0, "{case}: sampler kind callback");
        assert_eq!(
            self.distributed_calls.get(),
            0,
            "{case}: sampler distributed callback"
        );
        assert_eq!(self.epoch_applies.get(), 0, "{case}: sampler epoch apply");
        assert_eq!(self.cursor_applies.get(), 0, "{case}: sampler cursor apply");
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct ObservableSamplerState {
    epoch: u64,
    position: u64,
}

#[derive(Clone)]
struct ObservableSampler {
    epoch: u64,
    observations: Rc<PlanObservations>,
}

impl Sampler for ObservableSampler {
    type Iter = std::vec::IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        vec![0, 1, 2].into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(3)
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.observations
            .epoch_applies
            .set(self.observations.epoch_applies.get() + 1);
        self.epoch = epoch;
    }
}

impl SamplerCheckpoint for ObservableSampler {
    type State = ObservableSamplerState;

    fn kind(&self) -> &'static str {
        self.observations
            .kind_calls
            .set(self.observations.kind_calls.get() + 1);
        "test-observable"
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        Ok(ObservableSamplerState {
            epoch: self.epoch,
            position,
        })
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        if state.epoch != epoch || state.position != position || position > 3 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "sampler",
                reason: "observable sampler state does not match its cursor".to_owned(),
            });
        }
        Ok(())
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.observations
            .cursor_applies
            .set(self.observations.cursor_applies.get() + 1);
        self.epoch = state.epoch;
        vec![0, 1, 2]
            .into_iter()
            .skip(usize::try_from(state.position).expect("validated test cursor"))
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn distributed_configuration(&self) -> Option<rusttorch_data::DistributedConfiguration> {
        self.observations
            .distributed_calls
            .set(self.observations.distributed_calls.get() + 1);
        None
    }
}

#[test]
fn static_envelope_rejects_before_public_plan_callbacks_or_mutation() -> Result<()> {
    type State = LoaderState<(), ObservableSamplerState>;
    type Corrupt = fn(&mut State);

    let observations = Rc::new(PlanObservations::default());
    let mut source = DataLoader::builder(replay_rows(&[1, 2, 3]))
        .sampler(ObservableSampler {
            epoch: 0,
            observations: Rc::clone(&observations),
        })
        .batch_size(2)
        .seed(17)
        .rank(0)
        .collate(VecCollate)
        .dataset_identity("observable-envelope".to_owned())
        .build()?;
    let state = source.iter().checkpoint().unwrap();

    let corruptions: [(&str, Corrupt); 14] = [
        ("schema", |state| state.schema_version += 1),
        ("identity", |state| {
            state.dataset_identity.push_str("-wrong")
        }),
        ("generation", |state| state.iterator_generation += 1),
        ("rng derivation", |state| state.rng_derivation_version += 1),
        ("worker seed derivation", |state| {
            state.worker_seed_derivation_version += 1;
        }),
        ("batch size", |state| {
            state.configuration.batch_size = Some(3)
        }),
        ("drop last", |state| state.configuration.drop_last = true),
        ("workers", |state| state.configuration.workers = 1),
        ("prefetch", |state| {
            state.configuration.prefetch_factor = Some(2);
        }),
        ("ordering", |state| state.configuration.in_order = false),
        ("loader seed", |state| state.configuration.loader_seed += 1),
        ("rank", |state| state.configuration.rank += 1),
        ("pin request", |state| {
            state.configuration.pin_request = CheckpointPinRequest::Auto;
        }),
        ("pin status", |state| {
            state.configuration.pin_status = CheckpointPinStatus::DisabledNoAccelerator;
        }),
    ];

    for (case, corrupt) in corruptions {
        let mut wrong = state.clone();
        corrupt(&mut wrong);
        let builder = DataLoader::builder(replay_rows(&[1, 2, 3]))
            .sampler(ObservableSampler {
                epoch: 0,
                observations: Rc::clone(&observations),
            })
            .batch_size(2)
            .seed(17)
            .rank(0)
            .collate(VecCollate)
            .dataset_identity("observable-envelope".to_owned())
            .resume_from(wrong);
        observations.reset();
        let result = builder.build();

        assert!(result.is_err(), "{case}: malformed state must reject");
        observations.assert_untouched(case);
    }

    Ok(())
}

#[derive(Clone)]
struct CountedRows {
    validates: Rc<Cell<usize>>,
    applies: Rc<Cell<usize>>,
}
impl Dataset for CountedRows {
    type Sample = i64;
    type Error = Infallible;
    fn len(&self) -> usize {
        1
    }
    fn get(&self, _: usize) -> std::result::Result<i64, Infallible> {
        Ok(1)
    }
}
impl Checkpointable for CountedRows {
    type State = u64;
    fn save_state(&self) -> Self::State {
        0
    }
    fn validate_state(&self, _: &Self::State) -> Result<()> {
        self.validates.set(self.validates.get() + 1);
        Ok(())
    }
    fn load_validated(&mut self, _: &Self::State) {
        self.applies.set(self.applies.get() + 1);
    }
}

#[derive(Clone)]
struct CountedSampler {
    epoch: u64,
    validates: Rc<Cell<usize>>,
    applies: Rc<Cell<usize>>,
}

impl Sampler for CountedSampler {
    type Iter = std::vec::IntoIter<usize>;

    fn iter(&self) -> Self::Iter {
        vec![0].into_iter()
    }

    fn exact_len(&self) -> Option<usize> {
        Some(1)
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }
}

impl SamplerCheckpoint for CountedSampler {
    type State = u64;

    fn kind(&self) -> &'static str {
        "test-counted"
    }

    fn checkpoint_state(&self, position: u64) -> Result<Self::State> {
        if position > 1 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "sampler.position",
                reason: "test sampler cursor exceeds its length".to_owned(),
            });
        }
        Ok(position)
    }

    fn validate_checkpoint_state(
        &self,
        state: &Self::State,
        epoch: u64,
        position: u64,
    ) -> Result<()> {
        self.validates.set(self.validates.get() + 1);
        if *state != position || position > 1 || epoch != self.epoch {
            return Err(RustTorchError::InvalidConfiguration {
                field: "sampler",
                reason: "test sampler state does not match its active cursor".to_owned(),
            });
        }
        Ok(())
    }

    fn restore_checkpoint_iter_validated(&mut self, state: &Self::State) -> Self::Iter {
        self.applies.set(self.applies.get() + 1);
        if *state == 0 {
            vec![0].into_iter()
        } else {
            Vec::new().into_iter()
        }
    }
}

#[derive(Clone)]
struct CountedTransform {
    validates: Rc<Cell<usize>>,
    applies: Rc<Cell<usize>>,
}

impl Transform<i64> for CountedTransform {
    type Output = i64;
    type Error = Infallible;

    fn transform(&mut self, value: i64, _: &TaskContext) -> std::result::Result<i64, Infallible> {
        Ok(value)
    }
}

impl TransactionalCheckpoint for CountedTransform {
    type State = ();

    fn snapshot(&self) -> Self::State {}

    fn validate_snapshot(&self, _: &Self::State) -> Result<()> {
        self.validates.set(self.validates.get() + 1);
        Ok(())
    }

    fn restore_validated(&mut self, _: &Self::State) {
        self.applies.set(self.applies.get() + 1);
    }
}

#[derive(Clone)]
struct RejectingCollator {
    reject: bool,
    validates: Rc<Cell<usize>>,
    applies: Rc<Cell<usize>>,
}
impl Collate<i64> for RejectingCollator {
    type Batch = Vec<i64>;
    type Error = Infallible;
    fn collate(&mut self, values: Vec<i64>) -> std::result::Result<Vec<i64>, Infallible> {
        Ok(values)
    }
}
impl Checkpointable for RejectingCollator {
    type State = ();
    fn save_state(&self) -> Self::State {}
    fn validate_state(&self, _: &Self::State) -> Result<()> {
        self.validates.set(self.validates.get() + 1);
        if self.reject {
            Err(RustTorchError::InvalidConfiguration {
                field: "collator",
                reason: "rejected for test".to_owned(),
            })
        } else {
            Ok(())
        }
    }
    fn load_validated(&mut self, _: &Self::State) {
        self.applies.set(self.applies.get() + 1);
    }
}

#[test]
fn resume_validates_every_component_before_any_apply() -> Result<()> {
    let validates = Rc::new(Cell::new(0));
    let dataset_applies = Rc::new(Cell::new(0));
    let sampler_validates = Rc::new(Cell::new(0));
    let sampler_applies = Rc::new(Cell::new(0));
    let transform_validates = Rc::new(Cell::new(0));
    let transform_applies = Rc::new(Cell::new(0));
    let collator_validates = Rc::new(Cell::new(0));
    let collator_applies = Rc::new(Cell::new(0));
    let mut source = DataLoader::builder(TransactionalMap::new(CountedRows {
        validates: Rc::clone(&validates),
        applies: Rc::clone(&dataset_applies),
    }))
    .sampler(CountedSampler {
        epoch: 0,
        validates: Rc::clone(&sampler_validates),
        applies: Rc::clone(&sampler_applies),
    })
    .transform(CountedTransform {
        validates: Rc::clone(&transform_validates),
        applies: Rc::clone(&transform_applies),
    })
    .checkpoint_transactional()
    .collate(RejectingCollator {
        reject: false,
        validates: Rc::clone(&collator_validates),
        applies: Rc::clone(&collator_applies),
    })
    .dataset_identity("transaction".to_owned())
    .build()?;
    let state = source.iter().checkpoint().unwrap();

    validates.set(0);
    dataset_applies.set(0);
    sampler_validates.set(0);
    sampler_applies.set(0);
    transform_validates.set(0);
    transform_applies.set(0);
    collator_validates.set(0);
    collator_applies.set(0);

    let mut wrong_envelope = state.clone();
    wrong_envelope.schema_version += 1;
    assert!(
        DataLoader::builder(TransactionalMap::new(CountedRows {
            validates: Rc::clone(&validates),
            applies: Rc::clone(&dataset_applies),
        }))
        .sampler(CountedSampler {
            epoch: 0,
            validates: Rc::clone(&sampler_validates),
            applies: Rc::clone(&sampler_applies),
        })
        .transform(CountedTransform {
            validates: Rc::clone(&transform_validates),
            applies: Rc::clone(&transform_applies),
        })
        .checkpoint_transactional()
        .collate(RejectingCollator {
            reject: false,
            validates: Rc::clone(&collator_validates),
            applies: Rc::clone(&collator_applies),
        })
        .dataset_identity("transaction".to_owned())
        .resume_from(wrong_envelope)
        .build()
        .is_err()
    );
    assert_eq!(validates.get(), 0);
    assert_eq!(dataset_applies.get(), 0);
    assert_eq!(sampler_validates.get(), 0);
    assert_eq!(sampler_applies.get(), 0);
    assert_eq!(transform_validates.get(), 0);
    assert_eq!(transform_applies.get(), 0);
    assert_eq!(collator_validates.get(), 0);
    assert_eq!(collator_applies.get(), 0);

    let error = match DataLoader::builder(TransactionalMap::new(CountedRows {
        validates: Rc::clone(&validates),
        applies: Rc::clone(&dataset_applies),
    }))
    .sampler(CountedSampler {
        epoch: 0,
        validates: Rc::clone(&sampler_validates),
        applies: Rc::clone(&sampler_applies),
    })
    .transform(CountedTransform {
        validates: Rc::clone(&transform_validates),
        applies: Rc::clone(&transform_applies),
    })
    .checkpoint_transactional()
    .collate(RejectingCollator {
        reject: true,
        validates: Rc::clone(&collator_validates),
        applies: Rc::clone(&collator_applies),
    })
    .dataset_identity("transaction".to_owned())
    .resume_from(state.clone())
    .build()
    {
        Ok(_) => panic!("final component must reject"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        CheckpointBuildError::Configuration(RustTorchError::InvalidConfiguration { .. })
    ));
    assert_eq!(validates.get(), 1);
    assert_eq!(sampler_validates.get(), 1);
    assert_eq!(transform_validates.get(), 1);
    assert_eq!(collator_validates.get(), 1);
    assert_eq!(dataset_applies.get(), 0);
    assert_eq!(sampler_applies.get(), 0);
    assert_eq!(transform_applies.get(), 0);
    assert_eq!(collator_applies.get(), 0);

    let mut resumed = resume_ok(
        DataLoader::builder(TransactionalMap::new(CountedRows {
            validates: Rc::clone(&validates),
            applies: Rc::clone(&dataset_applies),
        }))
        .sampler(CountedSampler {
            epoch: 0,
            validates: Rc::clone(&sampler_validates),
            applies: Rc::clone(&sampler_applies),
        })
        .transform(CountedTransform {
            validates: Rc::clone(&transform_validates),
            applies: Rc::clone(&transform_applies),
        })
        .checkpoint_transactional()
        .collate(RejectingCollator {
            reject: false,
            validates: Rc::clone(&collator_validates),
            applies: Rc::clone(&collator_applies),
        })
        .dataset_identity("transaction".to_owned())
        .resume_from(state)
        .build(),
    )?;
    assert_eq!(dataset_applies.get(), 1);
    assert_eq!(sampler_applies.get(), 1);
    assert_eq!(transform_applies.get(), 1);
    assert_eq!(collator_applies.get(), 1);
    assert_eq!(resumed.iter().next().unwrap().unwrap(), [1]);
    Ok(())
}

#[test]
fn envelope_and_configuration_drift_reject_before_iteration() -> Result<()> {
    let mut source = DataLoader::builder(replay_rows(&[1, 2, 3]))
        .batch_size(2)
        .seed(17)
        .rank(0)
        .collate(VecCollate)
        .dataset_identity("identity".to_owned())
        .build()?;
    let state = source.iter().checkpoint().unwrap();

    let reject = |state| {
        DataLoader::builder(replay_rows(&[1, 2, 3]))
            .batch_size(2)
            .seed(17)
            .rank(0)
            .collate(VecCollate)
            .dataset_identity("identity".to_owned())
            .resume_from(state)
            .build()
            .is_err()
    };

    let mut wrong = state.clone();
    wrong.schema_version += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.dataset_identity.push_str("-wrong");
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.loader_seed += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.sampler_kind.push_str("-wrong");
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.batch_size = Some(3);
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.drop_last = true;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.workers = 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.prefetch_factor = Some(2);
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.in_order = false;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.configuration.pin_status = CheckpointPinStatus::Cuda { index: 0 };
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.rng_derivation_version += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.worker_seed_derivation_version += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.iterator_generation += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.epoch += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.next_logical_sample = u64::MAX;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.sampler.position = u64::MAX;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.sampler.length += 1;
    assert!(reject(wrong));
    let mut wrong = state.clone();
    wrong.sampler.epoch += 1;
    assert!(reject(wrong));

    assert!(
        DataLoader::builder(replay_rows(&[1]))
            .dataset_identity(String::new())
            .build()
            .is_err()
    );
    assert!(
        DataLoader::builder(replay_rows(&[1]))
            .in_order(false)
            .dataset_identity("unordered".to_owned())
            .build()
            .is_err()
    );
    Ok(())
}

#[test]
fn replay_safe_tensor_dataset_owns_backing_and_each_fetched_row() -> Result<()> {
    let mut original = Tensor::from_slice(&[1_i64, 2, 3]);
    let dataset =
        rusttorch_data::TensorDataset::new(vec![original.shallow_clone()])?.into_replay_safe()?;
    let _ = original.fill_(99);
    assert_eq!(dataset.get(1)?[0].int64_value(&[]), 2);

    let mut fetched = dataset.get(1)?.remove(0);
    let _ = fetched.fill_(77);
    assert_eq!(dataset.get(1)?[0].int64_value(&[]), 2);

    let mut loader = DataLoader::builder(dataset)
        .sampler(RandomSampler::with_replacement(3, 5, 91)?)
        .collate(VecCollate)
        .dataset_identity("tensor-private".to_owned())
        .build()?;
    let mut iter = loader.iter();
    let _ = iter.next().unwrap().unwrap();
    let state = iter.checkpoint().unwrap();
    let continued = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();

    let dataset = rusttorch_data::TensorDataset::new(vec![Tensor::from_slice(&[1_i64, 2, 3])])?
        .into_replay_safe()?;
    let mut resumed = resume_ok(
        DataLoader::builder(dataset)
            .sampler(RandomSampler::with_replacement(3, 5, 91)?)
            .collate(VecCollate)
            .dataset_identity("tensor-private".to_owned())
            .resume_from(state)
            .build(),
    )?;
    let resumed = resumed
        .iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let values = |batches: Vec<Vec<Vec<Tensor>>>| {
        batches
            .into_iter()
            .map(|batch| batch[0][0].int64_value(&[]))
            .collect::<Vec<_>>()
    };
    assert_eq!(values(resumed), values(continued));
    Ok(())
}

#[derive(Clone)]
struct StatefulConverter {
    calls: u64,
}

impl Collate<i64> for StatefulConverter {
    type Batch = i64;
    type Error = Infallible;

    fn collate(&mut self, values: Vec<i64>) -> std::result::Result<i64, Infallible> {
        self.calls += 1;
        Ok(values[0] + self.calls as i64)
    }
}

impl Checkpointable for StatefulConverter {
    type State = u64;

    fn save_state(&self) -> Self::State {
        self.calls
    }
    fn validate_state(&self, _state: &Self::State) -> Result<()> {
        Ok(())
    }
    fn load_validated(&mut self, state: &Self::State) {
        self.calls = *state;
    }
}

#[test]
fn no_batch_state_targets_the_effective_converter() -> Result<()> {
    let mut loader = DataLoader::builder(replay_rows(&[10, 20, 30]))
        .without_batching()
        .convert(StatefulConverter { calls: 0 })
        .dataset_identity("converter".to_owned())
        .build()?;
    let mut iter = loader.iter();
    assert_eq!(iter.next().unwrap().unwrap(), 11);
    let state = iter.checkpoint().unwrap();
    assert_eq!(state.collate, 1);
    let expected = iter.collect::<std::result::Result<Vec<_>, _>>().unwrap();

    let mut resumed = resume_ok(
        DataLoader::builder(replay_rows(&[10, 20, 30]))
            .without_batching()
            .convert(StatefulConverter { calls: 0 })
            .dataset_identity("converter".to_owned())
            .resume_from(state)
            .build(),
    )?;
    assert_eq!(
        resumed
            .iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap(),
        expected
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct FactoryFailure;

impl fmt::Display for FactoryFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("factory failure")
    }
}

impl Error for FactoryFailure {}

#[test]
fn resume_build_error_preserves_configuration_and_factory_sources() -> Result<()> {
    let mut source = DataLoader::builder(replay_rows(&[1]))
        .collate(VecCollate)
        .dataset_identity("sources".to_owned())
        .build()?;
    let state = source.iter().checkpoint().unwrap();

    let configuration_error = match DataLoader::builder(replay_rows(&[1]))
        .collate(VecCollate)
        .dataset_identity("wrong".to_owned())
        .resume_from(state.clone())
        .build()
    {
        Ok(_) => panic!("identity drift must reject"),
        Err(error) => error,
    };
    assert!(matches!(
        configuration_error,
        CheckpointBuildError::Configuration(RustTorchError::InvalidConfiguration {
            field: "dataset_identity",
            ..
        })
    ));
    assert!(configuration_error.source().is_some());

    let factory_error = match DataLoader::builder(replay_rows(&[1]))
        .transform_factory(FnTransformFactory::new(
            |_: Option<&rusttorch_data::WorkerContext>| Err::<IdentityTransform, _>(FactoryFailure),
        ))
        .collate(VecCollate)
        .dataset_identity("sources".to_owned())
        .resume_from(state)
        .build()
    {
        Ok(_) => panic!("factory failure must be preserved"),
        Err(error) => error,
    };
    assert!(matches!(
        factory_error,
        CheckpointBuildError::TransformFactory(FactoryFailure)
    ));
    assert_eq!(
        factory_error.source().unwrap().to_string(),
        "factory failure"
    );
    Ok(())
}

#[test]
fn pin_request_and_effective_status_are_checkpoint_identity() -> Result<()> {
    let mut disabled = DataLoader::builder(replay_rows(&[1]))
        .collate(VecCollate)
        .dataset_identity("pin-disabled".to_owned())
        .build()?;
    let disabled_state = disabled.iter().checkpoint().unwrap();
    assert_eq!(
        disabled_state.configuration.pin_request,
        CheckpointPinRequest::Disabled
    );
    assert_eq!(
        disabled_state.configuration.pin_status,
        CheckpointPinStatus::Disabled
    );

    let mut automatic = DataLoader::builder(replay_rows(&[1]))
        .collate(VecCollate)
        .dataset_identity("pin-auto".to_owned())
        .pin_memory()
        .build()?;
    let automatic_state = automatic.iter().checkpoint().unwrap();
    assert_eq!(
        automatic_state.configuration.pin_request,
        CheckpointPinRequest::Auto
    );
    let expected = if available_devices().cuda {
        CheckpointPinStatus::Cuda { index: 0 }
    } else {
        CheckpointPinStatus::DisabledNoAccelerator
    };
    assert_eq!(automatic_state.configuration.pin_status, expected);

    let mut checkpoint_then_pin = DataLoader::builder(replay_rows(&[2]))
        .transform(StatelessDouble)
        .checkpoint_stateless()
        .pin_memory()
        .collate(VecCollate)
        .dataset_identity("checkpoint-then-pin".to_owned())
        .build()?;
    assert_eq!(
        checkpoint_then_pin
            .iter()
            .checkpoint()
            .unwrap()
            .configuration
            .pin_request,
        CheckpointPinRequest::Auto
    );

    let mut pin_then_checkpoint = DataLoader::builder(replay_rows(&[2]))
        .transform(StatelessDouble)
        .pin_memory()
        .checkpoint_stateless()
        .collate(VecCollate)
        .dataset_identity("pin-then-checkpoint".to_owned())
        .build()?;
    assert_eq!(
        pin_then_checkpoint
            .iter()
            .checkpoint()
            .unwrap()
            .configuration
            .pin_request,
        CheckpointPinRequest::Auto
    );

    let mut wrong = disabled_state;
    wrong.configuration.pin_request = CheckpointPinRequest::Auto;
    assert!(
        DataLoader::builder(replay_rows(&[1]))
            .collate(VecCollate)
            .dataset_identity("pin-disabled".to_owned())
            .resume_from(wrong)
            .build()
            .is_err()
    );

    let capabilities = available_devices();
    if capabilities.cuda && capabilities.cuda_device_count > 0 {
        let mut explicit = DataLoader::builder(replay_rows(&[1]))
            .collate(VecCollate)
            .pin_memory_for(Device::Cuda(0))
            .dataset_identity("pin-cuda".to_owned())
            .build()?;
        let state = explicit.iter().checkpoint().unwrap();
        assert_eq!(
            state.configuration.pin_request,
            CheckpointPinRequest::ExplicitCuda { index: 0 }
        );
        assert_eq!(
            state.configuration.pin_status,
            CheckpointPinStatus::Cuda { index: 0 }
        );
    }
    Ok(())
}

#[test]
fn sampler_and_distributed_semantic_corruption_rejects_without_panicking() -> Result<()> {
    type RandomState = LoaderState<(), rusttorch_data::RandomSamplerState, (), ()>;
    let mut random = DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
        .sampler(RandomSampler::with_replacement(4, 6, 7)?)
        .collate(VecCollate)
        .dataset_identity("random-config".to_owned())
        .build()?;
    let state = random.iter().checkpoint().unwrap();
    for mutate in [
        |state: &mut RandomState| state.sampler.length += 1,
        |state: &mut RandomState| state.sampler.num_samples += 1,
        |state: &mut RandomState| {
            state.sampler.replacement = rusttorch_data::RandomReplacement::Without
        },
        |state: &mut RandomState| state.sampler.seed += 1,
        |state: &mut RandomState| state.sampler.epoch += 1,
        |state: &mut RandomState| state.sampler.position = u64::MAX,
    ] {
        let mut wrong = state.clone();
        mutate(&mut wrong);
        assert!(
            DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
                .sampler(RandomSampler::with_replacement(4, 6, 7)?)
                .collate(VecCollate)
                .dataset_identity("random-config".to_owned())
                .resume_from(wrong)
                .build()
                .is_err()
        );
    }

    type SubsetState = LoaderState<(), rusttorch_data::SubsetRandomSamplerState, (), ()>;
    let mut subset = DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
        .sampler(SubsetRandomSampler::new(vec![0, 2, 3], 11)?)
        .collate(VecCollate)
        .dataset_identity("subset-config".to_owned())
        .build()?;
    let state = subset.iter().checkpoint().unwrap();
    for mutate in [
        |state: &mut SubsetState| state.sampler.indices[0] += 1,
        |state: &mut SubsetState| state.sampler.seed += 1,
        |state: &mut SubsetState| state.sampler.epoch += 1,
        |state: &mut SubsetState| state.sampler.position = u64::MAX,
    ] {
        let mut wrong = state.clone();
        mutate(&mut wrong);
        assert!(
            DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
                .sampler(SubsetRandomSampler::new(vec![0, 2, 3], 11)?)
                .collate(VecCollate)
                .dataset_identity("subset-config".to_owned())
                .resume_from(wrong)
                .build()
                .is_err()
        );
    }

    type WeightedState = LoaderState<(), rusttorch_data::WeightedRandomSamplerState, (), ()>;
    let mut weighted = DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
        .sampler(WeightedRandomSampler::new(vec![1.0; 4], 4, false, 8)?)
        .collate(VecCollate)
        .dataset_identity("weights".to_owned())
        .build()?;
    let state = weighted.iter().checkpoint().unwrap();
    for mutate in [
        |state: &mut WeightedState| state.sampler.weight_bits[0] ^= 1,
        |state: &mut WeightedState| state.sampler.num_samples += 1,
        |state: &mut WeightedState| state.sampler.replacement = true,
        |state: &mut WeightedState| state.sampler.seed += 1,
        |state: &mut WeightedState| state.sampler.epoch += 1,
        |state: &mut WeightedState| state.sampler.position = u64::MAX,
    ] {
        let mut wrong = state.clone();
        mutate(&mut wrong);
        assert!(
            DataLoader::builder(replay_rows(&(0..4).collect::<Vec<_>>()))
                .sampler(WeightedRandomSampler::new(vec![1.0; 4], 4, false, 8)?)
                .collate(VecCollate)
                .dataset_identity("weights".to_owned())
                .resume_from(wrong)
                .build()
                .is_err()
        );
    }

    let mut distributed = DataLoader::builder(replay_rows(&(0..8).collect::<Vec<_>>()))
        .sampler(DistributedSampler::new(8, 2, 1, true, 13, false)?)
        .rank(1)
        .collate(VecCollate)
        .dataset_identity("distributed".to_owned())
        .build()?;
    let state = distributed.iter().checkpoint().unwrap();
    for mutate in [
        |state: &mut LoaderState<(), _, (), ()>| state.configuration.replicas += 1,
        |state: &mut LoaderState<(), _, (), ()>| state.configuration.rank = 0,
        |state: &mut LoaderState<(), _, (), ()>| {
            state.configuration.distributed_shuffle = Some(false)
        },
        |state: &mut LoaderState<(), _, (), ()>| state.configuration.distributed_seed = Some(99),
        |state: &mut LoaderState<(), _, (), ()>| {
            state.configuration.distributed_drop_last = Some(true)
        },
    ] {
        let mut wrong = state.clone();
        mutate(&mut wrong);
        assert!(
            DataLoader::builder(replay_rows(&(0..8).collect::<Vec<_>>()))
                .sampler(DistributedSampler::new(8, 2, 1, true, 13, false)?)
                .rank(1)
                .collate(VecCollate)
                .dataset_identity("distributed".to_owned())
                .resume_from(wrong)
                .build()
                .is_err()
        );
    }
    for mutate in [
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.length += 1
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.replicas += 1
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.rank = 0
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.shuffle = false
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.seed += 1
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.drop_last = true
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.epoch += 1
        },
        |state: &mut LoaderState<(), rusttorch_data::DistributedSamplerState, (), ()>| {
            state.sampler.position = u64::MAX
        },
    ] {
        let mut wrong = state.clone();
        mutate(&mut wrong);
        assert!(
            DataLoader::builder(replay_rows(&(0..8).collect::<Vec<_>>()))
                .sampler(DistributedSampler::new(8, 2, 1, true, 13, false)?)
                .rank(1)
                .collate(VecCollate)
                .dataset_identity("distributed".to_owned())
                .resume_from(wrong)
                .build()
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn malformed_json_is_a_storage_error_not_a_loader_fallback() {
    let decoded = serde_json::from_str::<LoaderState<(), rusttorch_data::SequentialSamplerState>>(
        r#"{"schema_version":1}"#,
    );
    assert!(decoded.is_err());
}
