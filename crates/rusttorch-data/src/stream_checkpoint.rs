//! Retained-source lifecycle for exact streams. Ordinary streams have no journal.

use super::*;
use crate::{
    CheckpointDisabled, CheckpointFresh, CheckpointResume, CheckpointSourceFactory, Checkpointable,
    CheckpointableSource, StreamCheckpointBuildError, StreamCheckpointConfiguration,
    StreamLaneState, StreamLoaderState, TASK_RNG_DERIVATION_VERSION,
    WORKER_SEED_DERIVATION_VERSION, WorkerCheckpoint,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

type SourceState<S> = <<S as WorkerSourceFactory>::Source as CheckpointableSource>::State;
type TransformState<S, F> = <<F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Transform as WorkerCheckpoint>::State;
type LaneState<S, F> = StreamLaneState<SourceState<S>, TransformState<S, F>>;
type State<S, F, C> =
    StreamLoaderState<SourceState<S>, TransformState<S, F>, <C as Checkpointable>::State>;
type BuildError<S, F> = StreamCheckpointBuildError<
    <S as WorkerSourceFactory>::Error,
    <F as TransformFactory<<S as WorkerSourceFactory>::Sample>>::Error,
>;
type IterError<S, F, C> = StreamLoaderError<S, C, F, NoWorkerInit>;

fn invalid(reason: impl Into<String>) -> RustTorchError {
    invalid_configuration("stream_checkpoint", reason)
}

impl<S, C, F, I, M, N> StreamDataLoaderBuilder<S, C, F, I, M, N, CheckpointDisabled> {
    /// Enables exact checkpoints for an explicit replay-safe source identity.
    ///
    /// Building requires a checkpoint factory, checkpointable sources/transforms,
    /// ordered delivery, no byte budget, no custom initializer, no persistence,
    /// and no timeout. Ordinary iterator batching has no checkpoint method.
    ///
    /// ```compile_fail
    /// use rusttorch_data::batches;
    /// let mut iter = batches((0..4).map(Ok::<_, std::convert::Infallible>), 2, false).unwrap();
    /// iter.checkpoint().unwrap();
    /// ```
    pub fn checkpointable(
        self,
        identity: impl Into<String>,
    ) -> StreamDataLoaderBuilder<S, C, F, I, M, N, CheckpointFresh> {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
            checkpoint: CheckpointFresh {
                identity: identity.into(),
            },
        }
    }

    /// Selects a serialized next-visible-batch stream checkpoint for resume.
    #[allow(clippy::type_complexity)]
    pub fn resume<SS, TS, CS>(
        self,
        identity: impl Into<String>,
        state: StreamLoaderState<SS, TS, CS>,
    ) -> StreamDataLoaderBuilder<S, C, F, I, M, N, CheckpointResume<StreamLoaderState<SS, TS, CS>>>
    {
        StreamDataLoaderBuilder {
            factory: self.factory,
            collator: self.collator,
            transform_factory: self.transform_factory,
            worker_init: self.worker_init,
            configuration: self.configuration,
            states: PhantomData,
            checkpoint: CheckpointResume {
                identity: identity.into(),
                state,
            },
        }
    }
}

fn configuration(
    config: StreamConfiguration,
    pin: PinMemoryStatus,
) -> StreamCheckpointConfiguration {
    StreamCheckpointConfiguration {
        workers: config.workers,
        prefetch_factor: config.prefetch_factor,
        batch_size: config.batch_size,
        drop_last: config.drop_last,
        loader_seed: config.loader_seed,
        rank: config.rank,
        pin_request: match config.pin_memory {
            StreamPinRequest::Disabled => crate::CheckpointPinRequest::Disabled,
            StreamPinRequest::Auto => crate::CheckpointPinRequest::Auto,
            StreamPinRequest::Explicit(Device::Cuda(index)) => {
                crate::CheckpointPinRequest::ExplicitCuda { index }
            }
            _ => unreachable!("pin request validated"),
        },
        pin_status: match pin {
            PinMemoryStatus::Disabled => crate::CheckpointPinStatus::Disabled,
            PinMemoryStatus::DisabledNoAccelerator => {
                crate::CheckpointPinStatus::DisabledNoAccelerator
            }
            PinMemoryStatus::Enabled(Device::Cuda(index)) => {
                crate::CheckpointPinStatus::Cuda { index }
            }
            _ => unreachable!("pin status validated"),
        },
    }
}

enum Command {
    Apply,
    Begin(WorkerRunContext),
    Rollback { generation: u64, boundary: u64 },
}

enum Event<T, SE, TE> {
    Record(WorkerRecord<T>),
    End,
    Source(SE),
    Transform {
        sequence: u64,
        logical_id: u64,
        error: TE,
    },
    Protocol {
        sequence: Option<u64>,
        reason: String,
    },
    Panic,
}

enum AttemptOutcome {
    InFlight,
    Record,
    End,
    SourceError,
    TransformError,
    Protocol,
}

struct Attempt {
    sequence: Option<u64>,
    _logical_id: Option<u64>,
    _outcome: AttemptOutcome,
}

struct Completion<T, SE, TE> {
    worker: usize,
    generation: u64,
    event: Event<T, SE, TE>,
}
type CompletionFor<S, F> =
    Completion<TransformOutput<S, F>, <S as WorkerSourceFactory>::Error, TransformFailure<S, F>>;
type Ack<S, F> = (
    usize,
    u64,
    std::result::Result<LaneState<S, F>, BuildError<S, F>>,
);

struct Pool<S, F>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    F: TransformFactory<S::Sample>,
    F::Transform: WorkerCheckpoint,
{
    commands: Vec<Sender<Command>>,
    credits: Vec<Sender<()>>,
    results: Receiver<CompletionFor<S, F>>,
    #[allow(clippy::type_complexity)]
    initialized: Receiver<(usize, std::result::Result<(), BuildError<S, F>>)>,
    acknowledgments: Receiver<Ack<S, F>>,
    handles: Vec<JoinHandle<()>>,
    shutdown: CancellationToken,
    run: WorkerRunContext,
    committed: Arc<AtomicU64>,
}

impl<S, F> Drop for Pool<S, F>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    F: TransformFactory<S::Sample>,
    F::Transform: WorkerCheckpoint,
{
    fn drop(&mut self) {
        self.run.cancel();
        self.shutdown.cancel();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn validate_capacity<S, F>(config: StreamConfiguration) -> Result<usize>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    F: TransformFactory<S::Sample>,
    F::Transform: WorkerCheckpoint,
{
    if config.workers == 0 || config.prefetch_factor == 0 || config.batch_size == 0 {
        return Err(invalid(
            "workers, prefetch_factor, and batch_size must be positive",
        ));
    }
    if !config.ordered
        || config.persistent_workers
        || config.timeout.is_some()
        || config.prefetch_bytes.is_some()
    {
        return Err(invalid(
            "exact streams require ordered nonpersistent workers without timeout or byte accounting",
        ));
    }
    let outstanding = config
        .workers
        .checked_mul(config.prefetch_factor)
        .ok_or_else(|| invalid("outstanding overflow"))?;
    // Credits for an assembling batch are returned before that batch commits.
    // Its pre-read states therefore remain in addition to the prefetch window.
    let journal = config
        .prefetch_factor
        .checked_add(config.batch_size)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| invalid("journal window overflow"))?;
    let extra = config
        .workers
        .checked_mul(journal)
        .and_then(|n| n.checked_mul(size_of::<(Attempt, LaneState<S, F>)>()))
        .and_then(|n| {
            n.checked_add(
                config.workers.checked_mul(
                    size_of::<LaneState<S, F>>()
                        .checked_mul(8)?
                        .checked_add(8192)?
                        .checked_add(size_of::<CapacitySlot<Ack<S, F>>>())?
                        .checked_add(size_of::<
                            CapacitySlot<(usize, std::result::Result<(), BuildError<S, F>>)>,
                        >())?,
                )?,
            )
        })
        .and_then(|n| {
            n.checked_add(outstanding.checked_mul(size_of::<CapacitySlot<CompletionFor<S, F>>>())?)
        })
        .ok_or_else(|| invalid("snapshot storage overflow"))?;
    validate_stream_capacity_extra::<S, F, NoWorkerInit, MemoryDisabled>(
        config.workers,
        config.prefetch_factor,
        outstanding,
        config.batch_size,
        outstanding,
        extra,
    )?;
    preflight_vec::<(Attempt, LaneState<S, F>)>(journal, "stream checkpoint journal")?;
    Ok(journal)
}

impl<S, F> Pool<S, F>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
{
    fn new(
        factory: Arc<S>,
        transforms: Arc<F>,
        config: StreamConfiguration,
        source_generation: u64,
        run: WorkerRunContext,
        boundary: u64,
        states: Option<Vec<LaneState<S, F>>>,
    ) -> std::result::Result<Self, BuildError<S, F>> {
        let journal_capacity = validate_capacity::<S, F>(config)?;
        let (result_tx, results) = bounded_checked(
            config.workers * config.prefetch_factor,
            "exact stream results",
        )?;
        let (init_tx, initialized) =
            bounded_checked(config.workers, "exact stream initialization")?;
        let (ack_tx, acknowledgments) =
            bounded_checked(config.workers, "exact stream acknowledgments")?;
        let mut pool = Self {
            commands: Vec::new(),
            credits: Vec::new(),
            results,
            initialized,
            acknowledgments,
            handles: Vec::new(),
            shutdown: CancellationToken::new(),
            run,
            committed: Arc::new(AtomicU64::new(boundary)),
        };
        let mut pending = states.map(Vec::into_iter);
        for id in 0..config.workers {
            let (command_tx, commands) = bounded(1);
            let (credit_tx, credits) = bounded(config.prefetch_factor);
            for _ in 0..config.prefetch_factor {
                credit_tx.try_send(()).expect("fresh bounded credits");
            }
            pool.commands.push(command_tx);
            pool.credits.push(credit_tx);
            let source_factory = Arc::clone(&factory);
            let transform_factory = Arc::clone(&transforms);
            let run = pool.run.clone();
            let shutdown = pool.shutdown.clone();
            let committed = Arc::clone(&pool.committed);
            let initialized = init_tx.clone();
            let acknowledgments = ack_tx.clone();
            let results = result_tx.clone();
            let state = pending.as_mut().and_then(Iterator::next);
            let info = WorkerInfo::from_loader_seed(
                id,
                config.workers,
                config.loader_seed,
                config.rank,
                source_generation,
            )?;
            let mut journal = VecDeque::new();
            journal
                .try_reserve_exact(journal_capacity)
                .map_err(|e| invalid(format!("snapshot journal unavailable: {e}")))?;
            if journal.capacity() > journal_capacity {
                return Err(
                    invalid("snapshot journal exceeds validated retention capacity").into(),
                );
            }
            let handle = thread::Builder::new().name(format!("rusttorch-exact-stream-{id}")).spawn(move || {
                let mut did_initialize = false;
                let mut generation = run.generation;
                let outcome = catch_unwind(AssertUnwindSafe(|| with_worker_info(info, || {
                    let context = WorkerContext::new(info, run.cancellation.clone(), run.deadline.clone());
                    let mut source = source_factory.create(context.clone()).map_err(|source| StreamCheckpointBuildError::Source { worker: id, source })?;
                    let mut transform = transform_factory.create(Some(&context)).map_err(|source| StreamCheckpointBuildError::TransformFactory { worker: id, source })?;
                    if let Some(state) = &state {
                        source.validate_snapshot(&state.source).map_err(|source| StreamCheckpointBuildError::Source { worker: id, source })?;
                        transform.validate_snapshot(&state.transform).map_err(|source| StreamCheckpointBuildError::TransformState { worker: id, source })?;
                    }
                    if initialized.send((id, Ok(()))).is_err() { return Ok(()); }
                    did_initialize = true;
                    let mut pending = state;
                    loop {
                        let command = select! {
                            recv(commands) -> command => match command { Ok(command) => command, Err(_) => return Ok(()) },
                            recv(shutdown.signal()) -> _ => return Ok(()),
                        };
                        match command {
                            Command::Apply => {
                                if let Some(state) = pending.take() {
                                    source.restore_validated(&state.source);
                                    transform.restore_validated(&state.transform);
                                }
                            }
                            Command::Begin(run) => {
                                generation = run.generation;
                                source.set_run_context(WorkerContext::new(info, run.cancellation.clone(), run.deadline.clone()));
                                let mut previous = None;
                                loop {
                                    let acquired = select! {
                                        recv(credits) -> result => result.is_ok(),
                                        recv(run.cancellation.signal()) -> _ => false,
                                        recv(shutdown.signal()) -> _ => false,
                                    };
                                    if !acquired || run.cancellation.is_cancelled() || shutdown.is_cancelled() { break; }
                                    let boundary = committed.load(Ordering::Acquire);
                                    while journal.front().is_some_and(|(attempt, _): &(Attempt, LaneState<S, F>)| attempt.sequence.is_some_and(|sequence| sequence < boundary)) { journal.pop_front(); }
                                    assert!(journal.len() < journal_capacity, "validated stream snapshot window exhausted");
                                    journal.push_back((Attempt { sequence: None, _logical_id: None, _outcome: AttemptOutcome::InFlight }, StreamLaneState { id, source: source.snapshot(), transform: transform.snapshot() }));
                                    let event = match source.next() {
                                        None => Event::End,
                                        Some(Err(error)) => Event::Source(error),
                                        Some(Ok(record)) => {
                                            let sequence = record.sequence.map(SequenceId::get);
                                            match sequence {
                                                None => Event::Protocol { sequence, reason: "exact stream record is missing its sequence ID".to_owned() },
                                                Some(sequence) if sequence < boundary || previous.is_some_and(|previous| sequence <= previous) => Event::Protocol { sequence: Some(sequence), reason: "exact stream shard emitted a past or non-increasing sequence".to_owned() },
                                                Some(sequence) => {
                                                    previous = Some(sequence);
                                                    let logical_id = record.logical_id.get();
                                                    let task = TaskContext { loader_seed: run.loader_seed, epoch: run.epoch, rank: info.rank, logical_sample: logical_id, stage: 0, cancellation: run.cancellation.clone(), deadline: run.deadline.clone() };
                                                    match transform.transform(record.sample, &task) {
                                                        Ok(sample) => Event::Record(WorkerRecord { sequence: record.sequence, logical_id: record.logical_id, sample }),
                                                        Err(error) => Event::Transform { sequence, logical_id, error },
                                                    }
                                                }
                                            }
                                        }
                                    };
                                    let (sequence, logical_id, outcome) = match &event {
                                        Event::Record(record) => (record.sequence.map(SequenceId::get), Some(record.logical_id.get()), AttemptOutcome::Record),
                                        Event::End => (None, None, AttemptOutcome::End),
                                        Event::Source(_) => (None, None, AttemptOutcome::SourceError),
                                        Event::Transform { sequence, logical_id, .. } => (Some(*sequence), Some(*logical_id), AttemptOutcome::TransformError),
                                        Event::Protocol { sequence, .. } => (*sequence, None, AttemptOutcome::Protocol),
                                        Event::Panic => unreachable!("panic is handled outside production"),
                                    };
                                    journal.back_mut().expect("pre-attempt snapshot").0 = Attempt { sequence, _logical_id: logical_id, _outcome: outcome };
                                    let terminal = !matches!(event, Event::Record(_));
                                    let message = Completion { worker: id, generation, event };
                                    let sent = select! {
                                        send(results, message) -> result => result.is_ok(),
                                        recv(run.cancellation.signal()) -> _ => false,
                                        recv(shutdown.signal()) -> _ => false,
                                    };
                                    if terminal || !sent { break; }
                                }
                            }
                            Command::Rollback { generation: requested, boundary } => {
                                let state = if requested != generation { Err(invalid("stale rollback generation").into()) } else {
                                    let state = journal.iter().find(|(attempt, _)| attempt.sequence.is_none_or(|sequence| sequence >= boundary)).map(|(_, state)| state.clone())
                                        .unwrap_or_else(|| StreamLaneState { id, source: source.snapshot(), transform: transform.snapshot() });
                                    source.validate_snapshot(&state.source).map_err(|source| StreamCheckpointBuildError::Source { worker: id, source })
                                        .and_then(|()| transform.validate_snapshot(&state.transform).map_err(|source| StreamCheckpointBuildError::TransformState { worker: id, source }))
                                        .map(|()| { source.restore_validated(&state.source); transform.restore_validated(&state.transform); state })
                                };
                                journal.clear();
                                while credits.try_recv().is_ok() {}
                                if acknowledgments.send((id, generation, state)).is_err() { return Ok(()); }
                            }
                        }
                    }
                })));
                match outcome {
                    Ok(Err(error)) => { let _ = initialized.send((id, Err(error))); }
                    Err(_) if !did_initialize => { let _ = initialized.send((id, Err(StreamCheckpointBuildError::WorkerPanic { worker: id }))); }
                    Err(_) => {
                        let message = Completion { worker: id, generation, event: Event::Panic };
                        select! { send(results, message) -> _ => {}, recv(shutdown.signal()) -> _ => {} }
                    }
                    Ok(Ok(())) => {}
                }
            }).map_err(|e| invalid(format!("cannot start exact stream worker: {e}")))?;
            pool.handles.push(handle);
        }
        drop(init_tx);
        drop(ack_tx);
        drop(result_tx);
        let mut seen = vec![false; config.workers];
        for _ in 0..config.workers {
            let (id, result) = pool
                .initialized
                .recv()
                .map_err(|_| invalid("worker closed during validation"))?;
            if id >= seen.len() || std::mem::replace(&mut seen[id], true) {
                return Err(invalid("duplicate initialization lane").into());
            }
            result?;
        }
        Ok(pool)
    }

    fn begin(&self) -> Result<()> {
        for sender in &self.commands {
            sender
                .send(Command::Begin(self.run.clone()))
                .map_err(|_| invalid("worker closed before production"))?;
        }
        Ok(())
    }

    fn apply(&self) -> Result<()> {
        for sender in &self.commands {
            sender
                .send(Command::Apply)
                .map_err(|_| invalid("worker closed before validated apply"))?;
        }
        Ok(())
    }

    fn rollback(
        &mut self,
        boundary: u64,
        prefetch: usize,
    ) -> std::result::Result<Vec<LaneState<S, F>>, BuildError<S, F>> {
        let generation = self.run.generation;
        let next_generation = generation
            .checked_add(1)
            .filter(|g| *g < u64::MAX)
            .ok_or_else(|| invalid("transport generation overflow"))?;
        self.run.cancel();
        for command in &self.commands {
            command
                .send(Command::Rollback {
                    generation,
                    boundary,
                })
                .map_err(|_| invalid("worker failed before rollback"))?;
        }
        let states = (0..self.commands.len()).map(|_| None).collect::<Vec<_>>();
        self.collect_rollback(states, generation, next_generation, boundary, prefetch)
    }

    fn collect_rollback(
        &mut self,
        mut states: Vec<Option<LaneState<S, F>>>,
        generation: u64,
        next_generation: u64,
        boundary: u64,
        prefetch: usize,
    ) -> std::result::Result<Vec<LaneState<S, F>>, BuildError<S, F>> {
        while states.iter().any(Option::is_none) {
            select! {
                recv(self.acknowledgments) -> reply => {
                    let (id, received, state) = reply.map_err(|_| invalid("missing rollback acknowledgment"))?;
                    if id >= states.len() || received != generation || states[id].is_some() { return Err(invalid("stale or duplicate rollback acknowledgment").into()); }
                    states[id] = Some(state?);
                },
                recv(self.results) -> message => {
                    if matches!(message, Ok(Completion { event: Event::Panic, .. }) | Err(_)) { return Err(invalid("worker failed before rollback acknowledgment").into()); }
                },
            }
        }
        self.finish_rollback(states, next_generation, boundary, prefetch)
    }

    fn finish_rollback(
        &mut self,
        states: Vec<Option<LaneState<S, F>>>,
        generation: u64,
        boundary: u64,
        prefetch: usize,
    ) -> std::result::Result<Vec<LaneState<S, F>>, BuildError<S, F>> {
        if self.acknowledgments.try_recv().is_ok() {
            return Err(invalid("duplicate rollback acknowledgment").into());
        }
        while self.results.try_recv().is_ok() {}
        for credit in &self.credits {
            for _ in 0..prefetch {
                credit
                    .try_send(())
                    .map_err(|_| invalid("rollback credits were not drained"))?;
            }
        }
        self.committed.store(boundary, Ordering::Release);
        self.run = WorkerRunContext::new(generation, self.run.loader_seed, self.run.epoch);
        Ok(states.into_iter().map(Option::unwrap).collect())
    }
}

/// Exact stream owner with suspended, transactionally validated worker sources.
pub struct ExactStreamDataLoader<S, C, F = IdentityTransformFactory, N = PinDisabled>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    F: TransformFactory<S::Sample>,
    F::Transform: WorkerCheckpoint,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
{
    factory: Arc<S>,
    transform_factory: Arc<F>,
    collator: C,
    configuration: StreamConfiguration,
    identity: String,
    pin: PinMemoryStatus,
    length: Option<usize>,
    pending: Option<Pool<S, F>>,
    next_batch: u64,
    next_sequence: u64,
    short_tail: bool,
    source_generation: u64,
    next_generation: u64,
    policies: PhantomData<N>,
}

impl<S, C, F, N> StreamDataLoaderBuilder<S, C, F, NoWorkerInit, MemoryDisabled, N, CheckpointFresh>
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    /// Validates exact capabilities and constructs suspended worker instances.
    pub fn build(self) -> std::result::Result<ExactStreamDataLoader<S, C, F, N>, BuildError<S, F>> {
        let identity = self.checkpoint.identity.clone();
        build(self, identity, None)
    }
}

impl<S, C, F, N>
    StreamDataLoaderBuilder<
        S,
        C,
        F,
        NoWorkerInit,
        MemoryDisabled,
        N,
        CheckpointResume<State<S, F, C>>,
    >
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    /// Validates the envelope before callbacks, then all components before apply.
    pub fn build(self) -> std::result::Result<ExactStreamDataLoader<S, C, F, N>, BuildError<S, F>> {
        let StreamDataLoaderBuilder {
            factory,
            collator,
            transform_factory,
            worker_init,
            configuration,
            checkpoint,
            ..
        } = self;
        let builder = StreamDataLoaderBuilder {
            factory,
            collator,
            transform_factory,
            worker_init,
            configuration,
            checkpoint: CheckpointDisabled,
            states: PhantomData,
        };
        build(builder, checkpoint.identity, Some(checkpoint.state))
    }
}

fn build<S, C, F, N, K>(
    builder: StreamDataLoaderBuilder<S, C, F, NoWorkerInit, MemoryDisabled, N, K>,
    identity: String,
    state: Option<State<S, F, C>>,
) -> std::result::Result<ExactStreamDataLoader<S, C, F, N>, BuildError<S, F>>
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    let config = builder.configuration;
    validate_capacity::<S, F>(config)?;
    if identity.is_empty() || S::CHECKPOINT_KIND.is_empty() {
        return Err(invalid("source identity and factory kind must not be empty").into());
    }
    let pin = resolve_stream_pin(config.pin_memory)?;
    if let Some(state) = &state {
        let batch_size = u64::try_from(config.batch_size)
            .map_err(|_| invalid("batch size exceeds sequence range"))?;
        let full = state
            .next_batch
            .checked_mul(batch_size)
            .ok_or_else(|| invalid("batch cursor overflow"))?;
        let short_tail = !config.drop_last
            && state.next_batch > 0
            && state.next_sequence < full
            && state.next_sequence > full - batch_size;
        if state.schema_version != 1
            || state.source_identity != identity
            || state.factory_kind != S::CHECKPOINT_KIND
            || state.epoch != config.epoch
            || state.source_generation > state.transport_generation
            || state.transport_generation >= u64::MAX - 1
            || state.rng_derivation_version != TASK_RNG_DERIVATION_VERSION
            || state.worker_seed_derivation_version != WORKER_SEED_DERIVATION_VERSION
            || state.configuration != configuration(config, pin)
            || state.lanes.len() != config.workers
            || state
                .lanes
                .iter()
                .enumerate()
                .any(|(id, lane)| lane.id != id)
            || state.short_tail != short_tail
            || (state.next_sequence != full && !short_tail)
            || state.source_length.is_some_and(|length| {
                state.next_sequence > length as u64
                    || (short_tail && state.next_sequence != length as u64)
            })
        {
            return Err(invalid("incompatible or corrupt stream checkpoint envelope").into());
        }
    }
    let length = builder.factory.exact_len();
    if state
        .as_ref()
        .is_some_and(|state| state.source_length != length)
    {
        return Err(invalid("source length changed").into());
    }
    let mut loader = ExactStreamDataLoader {
        factory: Arc::new(builder.factory),
        transform_factory: Arc::new(builder.transform_factory),
        collator: builder.collator,
        configuration: config,
        identity,
        pin,
        length,
        pending: None,
        next_batch: state.as_ref().map_or(0, |s| s.next_batch),
        next_sequence: state.as_ref().map_or(0, |s| s.next_sequence),
        short_tail: state.as_ref().is_some_and(|s| s.short_tail),
        source_generation: state.as_ref().map_or(0, |s| s.source_generation),
        next_generation: state.as_ref().map_or(1, |s| s.transport_generation + 1),
        policies: PhantomData,
    };
    let generation = state.as_ref().map_or(0, |s| s.transport_generation);
    let pool = Pool::new(
        Arc::clone(&loader.factory),
        Arc::clone(&loader.transform_factory),
        config,
        loader.source_generation,
        WorkerRunContext::new(generation, config.loader_seed, config.epoch),
        loader.next_sequence,
        state.as_ref().map(|s| s.lanes.clone()),
    )?;
    if let Some(state) = &state {
        loader.collator.validate_state(&state.coordinator)?;
        pool.apply()?;
        loader.collator.load_validated(&state.coordinator);
    }
    loader.pending = Some(pool);
    Ok(loader)
}

/// Iterator whose checkpoint rolls unpublished reads back to the visible boundary.
pub struct ExactStreamLoaderIter<'a, S, C, F = IdentityTransformFactory, N = PinDisabled>
where
    S: WorkerSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    F: TransformFactory<S::Sample>,
    F::Transform: WorkerCheckpoint,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
{
    loader: &'a mut ExactStreamDataLoader<S, C, F, N>,
    pool: Option<Pool<S, F>>,
    completed: Vec<(usize, WorkerRecord<TransformOutput<S, F>>)>,
    ended: Vec<bool>,
    next_sequence: u64,
    visible_sequence: u64,
    short_tail: bool,
    next_batch: u64,
    stopped: bool,
    boundary_valid: bool,
    pending_error: Option<IterError<S, F, C>>,
}

impl<S, C, F, N> ExactStreamDataLoader<S, C, F, N>
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    /// Returns the exact global batch count, when known.
    pub fn len(&self) -> Option<usize> {
        self.length.map(|n| {
            if self.configuration.drop_last {
                n / self.configuration.batch_size
            } else {
                n.div_ceil(self.configuration.batch_size)
            }
        })
    }
    /// Returns whether the known batch count is zero.
    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }
    /// Returns the shard count.
    pub fn workers(&self) -> usize {
        self.configuration.workers
    }
    /// Returns the effective pinning policy.
    pub fn pin_memory_status(&self) -> PinMemoryStatus {
        self.pin
    }
    /// Selects a fresh epoch and discards a retained resume.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.pending = None;
        self.configuration.epoch = epoch;
    }
    /// Starts the retained resume once, then fresh logical generations.
    pub fn iter(&mut self) -> ExactStreamLoaderIter<'_, S, C, F, N> {
        let retained = self.pending.is_some();
        let mut pending_error = None;
        let pool = if retained {
            self.pending.take()
        } else {
            let generation = self.next_generation;
            if let Some(next) = generation.checked_add(1).filter(|g| *g < u64::MAX) {
                self.next_generation = next;
                self.source_generation = generation;
                match Pool::new(
                    Arc::clone(&self.factory),
                    Arc::clone(&self.transform_factory),
                    self.configuration,
                    generation,
                    WorkerRunContext::new(
                        generation,
                        self.configuration.loader_seed,
                        self.configuration.epoch,
                    ),
                    0,
                    None,
                ) {
                    Ok(pool) => Some(pool),
                    Err(error) => {
                        pending_error = Some(map_build_error::<S, F, C>(error));
                        None
                    }
                }
            } else {
                pending_error = Some(LoaderError::Configuration(invalid("generation overflow")));
                None
            }
        };
        if let Some(pool) = &pool
            && let Err(error) = pool.begin()
        {
            pending_error = Some(LoaderError::Configuration(error));
        }
        let next_batch = if retained { self.next_batch } else { 0 };
        let next_sequence = if retained { self.next_sequence } else { 0 };
        let short_tail = retained && self.short_tail;
        let ended = vec![false; self.configuration.workers];
        let completed =
            Vec::with_capacity(self.configuration.workers * self.configuration.prefetch_factor);
        ExactStreamLoaderIter {
            loader: self,
            pool,
            completed,
            ended,
            next_batch,
            next_sequence,
            visible_sequence: next_sequence,
            short_tail,
            stopped: false,
            boundary_valid: pending_error.is_none(),
            pending_error,
        }
    }
}

fn map_build_error<S, F, C>(error: BuildError<S, F>) -> IterError<S, F, C>
where
    S: WorkerSourceFactory,
    F: TransformFactory<S::Sample>,
    C: Collate<TransformOutput<S, F>>,
{
    match error {
        StreamCheckpointBuildError::Configuration(e) => LoaderError::Configuration(e),
        StreamCheckpointBuildError::Source { worker, source } => LoaderError::StreamPipeline {
            batch: None,
            worker,
            sequence: None,
            logical_id: None,
            source: PipelineError::Source(source),
        },
        StreamCheckpointBuildError::TransformFactory { worker, source } => {
            LoaderError::StreamPipeline {
                batch: None,
                worker,
                sequence: None,
                logical_id: None,
                source: PipelineError::TransformInit(source),
            }
        }
        StreamCheckpointBuildError::TransformState { source, .. } => {
            LoaderError::Configuration(source)
        }
        StreamCheckpointBuildError::WorkerPanic { worker } => LoaderError::WorkerPanic {
            worker,
            batch: None,
        },
    }
}

impl<S, C, F, N> ExactStreamLoaderIter<'_, S, C, F, N>
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    /// Cancels active callbacks and rolls paired snapshots back before resuming.
    ///
    /// The original iterator remains usable. Source/transform errors can be
    /// replayed from the last successful boundary; coordinator failures cannot.
    /// Blocking native callbacks must cooperate with cancellation to finish.
    pub fn checkpoint(&mut self) -> std::result::Result<State<S, F, C>, IterError<S, F, C>> {
        if !self.boundary_valid {
            return Err(LoaderError::Checkpoint {
                reason: "no recoverable visible stream boundary".to_owned(),
            });
        }
        let coordinator = self.loader.collator.save_state();
        let pool = self.pool.as_mut().ok_or_else(|| LoaderError::Checkpoint {
            reason: "missing worker pool".to_owned(),
        })?;
        let lanes = match pool.rollback(
            self.visible_sequence,
            self.loader.configuration.prefetch_factor,
        ) {
            Ok(lanes) => lanes,
            Err(error) => {
                self.boundary_valid = false;
                self.stopped = true;
                self.pool = None;
                return Err(map_build_error::<S, F, C>(error));
            }
        };
        self.completed.clear();
        self.ended.fill(false);
        self.next_sequence = self.visible_sequence;
        self.stopped = false;
        let state = StreamLoaderState {
            schema_version: 1,
            source_identity: self.loader.identity.clone(),
            factory_kind: S::CHECKPOINT_KIND.to_owned(),
            epoch: self.loader.configuration.epoch,
            source_generation: self.loader.source_generation,
            transport_generation: pool.run.generation,
            next_batch: self.next_batch,
            next_sequence: self.visible_sequence,
            short_tail: self.short_tail,
            source_length: self.loader.length,
            rng_derivation_version: TASK_RNG_DERIVATION_VERSION,
            worker_seed_derivation_version: WORKER_SEED_DERIVATION_VERSION,
            configuration: configuration(self.loader.configuration, self.loader.pin),
            lanes,
            coordinator,
        };
        self.loader.next_generation = self.loader.next_generation.max(pool.run.generation + 1);
        pool.begin().map_err(LoaderError::Configuration)?;
        Ok(state)
    }

    fn fail(
        &mut self,
        error: IterError<S, F, C>,
        recoverable: bool,
    ) -> Option<std::result::Result<C::Batch, IterError<S, F, C>>> {
        self.stopped = true;
        self.boundary_valid &= recoverable;
        if let Some(pool) = &self.pool {
            pool.run.cancel();
        }
        if !recoverable {
            self.pool = None;
        }
        Some(Err(error))
    }
}

impl<S, C, F, N> Iterator for ExactStreamLoaderIter<'_, S, C, F, N>
where
    S: CheckpointSourceFactory,
    S::Source: CheckpointableSource<Sample = S::Sample, Error = S::Error>,
    SourceState<S>: Send + 'static,
    F: TransformFactory<S::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<S, F>: Send + 'static,
    TransformOutput<S, F>: Send + 'static,
    TransformFailure<S, F>: Send + 'static,
    F::Error: Send + 'static,
    C: Collate<TransformOutput<S, F>> + Checkpointable,
    N: StreamPinPolicy<S, C, F>,
{
    type Item = std::result::Result<C::Batch, IterError<S, F, C>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.stopped {
            return None;
        }
        if let Some(error) = self.pending_error.take() {
            return self.fail(error, false);
        }
        let mut samples = Vec::with_capacity(self.loader.configuration.batch_size);
        loop {
            if let Some(index) = self.completed.iter().position(|(_, record)| {
                record.sequence.map(SequenceId::get) == Some(self.next_sequence)
            }) {
                let (worker, record) = self.completed.swap_remove(index);
                let Some(next) = self.next_sequence.checked_add(1) else {
                    return self.fail(
                        LoaderError::StreamProtocol {
                            sequence: Some(self.next_sequence),
                            reason: "sequence overflow".to_owned(),
                        },
                        false,
                    );
                };
                self.next_sequence = next;
                samples.push(record.sample);
                if self.pool.as_ref().expect("active pool").credits[worker]
                    .try_send(())
                    .is_err()
                {
                    return self.fail(
                        LoaderError::ChannelClosed {
                            batch: self.next_batch,
                        },
                        false,
                    );
                }
                if samples.len() == self.loader.configuration.batch_size {
                    break;
                }
                continue;
            }
            if self.ended.iter().all(|ended| *ended) {
                if !self.completed.is_empty()
                    || self
                        .loader
                        .length
                        .is_some_and(|length| self.next_sequence != length as u64)
                {
                    return self.fail(
                        LoaderError::StreamProtocol {
                            sequence: Some(self.next_sequence),
                            reason:
                                "stream ended with a missing global sequence or wrong exact length"
                                    .to_owned(),
                        },
                        false,
                    );
                }
                if samples.is_empty() || self.loader.configuration.drop_last {
                    self.stopped = true;
                    return None;
                }
                break;
            }
            // ponytail: bounded lane/window scan; retain lane counts if wide sharding matters.
            if self.ended.iter().enumerate().all(|(id, ended)| {
                *ended
                    || self
                        .completed
                        .iter()
                        .filter(|(worker, _)| *worker == id)
                        .count()
                        == self.loader.configuration.prefetch_factor
            }) {
                return self.fail(
                    LoaderError::StreamProtocol {
                        sequence: Some(self.next_sequence),
                        reason: "bounded stream window contains no expected sequence".to_owned(),
                    },
                    false,
                );
            }
            let pool = self.pool.as_ref().expect("active pool");
            let completion = match pool.results.recv() {
                Ok(completion) => completion,
                Err(_) => {
                    return self.fail(
                        LoaderError::ChannelClosed {
                            batch: self.next_batch,
                        },
                        false,
                    );
                }
            };
            if completion.generation != pool.run.generation {
                return self.fail(
                    LoaderError::StreamProtocol {
                        sequence: None,
                        reason: "stale transport message".to_owned(),
                    },
                    false,
                );
            }
            let worker = completion.worker;
            match completion.event {
                Event::Record(record) => {
                    let sequence = record.sequence.map(SequenceId::get);
                    if sequence.is_none_or(|sequence| sequence < self.next_sequence)
                        || self
                            .completed
                            .iter()
                            .any(|(_, previous)| previous.sequence == record.sequence)
                    {
                        return self.fail(
                            LoaderError::StreamProtocol {
                                sequence,
                                reason: "missing, duplicate, or past sequence ID".to_owned(),
                            },
                            false,
                        );
                    }
                    self.completed.push((worker, record));
                }
                Event::End => {
                    self.ended[worker] = true;
                }
                Event::Source(source) => {
                    return self.fail(
                        LoaderError::StreamPipeline {
                            batch: Some(self.next_batch),
                            worker,
                            sequence: None,
                            logical_id: None,
                            source: PipelineError::Source(source),
                        },
                        true,
                    );
                }
                Event::Transform {
                    sequence,
                    logical_id,
                    error,
                } => {
                    return self.fail(
                        LoaderError::StreamPipeline {
                            batch: Some(self.next_batch),
                            worker,
                            sequence: Some(sequence),
                            logical_id: Some(logical_id),
                            source: PipelineError::Transform(error),
                        },
                        true,
                    );
                }
                Event::Protocol { sequence, reason } => {
                    return self.fail(LoaderError::StreamProtocol { sequence, reason }, false);
                }
                Event::Panic => {
                    return self.fail(
                        LoaderError::StreamWorkerPanic {
                            batch: Some(self.next_batch),
                            worker,
                            sequence: None,
                            logical_id: None,
                        },
                        false,
                    );
                }
            }
        }
        let short_tail = samples.len() < self.loader.configuration.batch_size;
        let batch = match catch_unwind(AssertUnwindSafe(|| self.loader.collator.collate(samples))) {
            Ok(Ok(batch)) => batch,
            Ok(Err(source)) => {
                return self.fail(
                    LoaderError::Pipeline {
                        batch: Some(self.next_batch),
                        worker: None,
                        source: PipelineError::Collate(source),
                    },
                    false,
                );
            }
            Err(_) => {
                return self.fail(
                    LoaderError::CoordinatorPanic {
                        stage: "collation",
                        batch: Some(self.next_batch),
                    },
                    false,
                );
            }
        };
        let batch = if let PinMemoryStatus::Enabled(device) = self.loader.pin {
            match catch_unwind(AssertUnwindSafe(|| N::pin(batch, device))) {
                Ok(Ok(batch)) => batch,
                Ok(Err(source)) => {
                    return self.fail(
                        LoaderError::PinMemory {
                            batch: self.next_batch,
                            source,
                        },
                        false,
                    );
                }
                Err(_) => {
                    return self.fail(
                        LoaderError::CoordinatorPanic {
                            stage: "pinning",
                            batch: Some(self.next_batch),
                        },
                        false,
                    );
                }
            }
        } else {
            batch
        };
        let Some(next_batch) = self.next_batch.checked_add(1) else {
            return self.fail(
                LoaderError::Configuration(invalid("batch cursor overflow")),
                false,
            );
        };
        self.next_batch = next_batch;
        self.visible_sequence = self.next_sequence;
        self.short_tail = short_tail;
        self.pool
            .as_ref()
            .expect("active pool")
            .committed
            .store(self.visible_sequence, Ordering::Release);
        Some(Ok(batch))
    }
}
