use super::*;
use crate::worker_checkpoint::{Barrier, error, extra_capacity};

// Only the explicit immutable wrappers have a worker replay implementation.
trait WorkerReplayDataset: DatasetCheckpoint<State = ()> {}
impl<D: crate::ReplaySafeDataset> WorkerReplayDataset for crate::ReplaySafeMap<D> {}
impl WorkerReplayDataset for crate::ReplaySafeTensorDataset {}

type TransformState<D, F> =
    <<F as TransformFactory<<D as Dataset>::Sample>>::Transform as WorkerCheckpoint>::State;
type Active<D, P, C, F> = WorkerCheckpointActive<
    <P as LoaderPlan<TransformOutput<D, F>, C>>::Iter,
    <F as TransformFactory<<D as Dataset>::Sample>>::Transform,
    TransformState<D, F>,
>;
type ExactLoader<D, P, C, F, N> = OwnedDataLoader<
    D,
    P,
    C,
    F,
    NoWorkerInit,
    WorkerExecution,
    MemoryDisabled,
    N,
    Active<D, P, C, F>,
>;
type IterCheckpoint<D, P, C, F> =
    WorkerCheckpointIteration<TransformState<D, F>, IterError<D, P, C, F, NoWorkerInit>>;
type ExactState<D, P, C, F> = LoaderState<
    (),
    <P as CheckpointPlan<TransformOutput<D, F>, C>>::State,
    WorkerTransformState<TransformState<D, F>>,
    <P as CheckpointPlan<TransformOutput<D, F>, C>>::CoordinatorState,
>;

fn preflight<D, P, C, F>(
    configuration: &mut BuilderConfiguration,
    explicit: ExplicitArguments,
    identity: &str,
) -> Result<()>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint,
    P: LoaderPlan<TransformOutput<D, F>, C>,
{
    validate_configuration(configuration, explicit)?;
    if identity.is_empty() {
        return Err(error("dataset identity must not be empty"));
    }
    if !configuration.ordered || configuration.persistent_workers || configuration.timeout.is_some()
    {
        return Err(error(
            "exact worker replay requires ordered, nonpersistent workers without timeout",
        ));
    }
    if configuration.workers > 0 {
        let prefetch = configuration.prefetch_factor.get_or_insert(2);
        let outstanding = configuration
            .workers
            .checked_mul(*prefetch)
            .ok_or_else(|| error("outstanding capacity overflow"))?;
        crate::worker::validate_worker_pool_capacity_extra::<D, F, NoWorkerInit, MemoryDisabled>(
            configuration.workers,
            *prefetch,
            outstanding,
            extra_capacity::<TransformState<D, F>, IterError<D, P, C, F, NoWorkerInit>>(
                configuration.workers,
                *prefetch,
            )?,
        )?;
    }
    Ok(())
}

fn finish_builder<D, P, C, F, N, K>(
    builder: DataLoaderBuilder<D, P, C, F, NoWorkerInit, WorkerExecution, MemoryDisabled, N, K>,
    checkpoint: Active<D, P, C, F>,
    pin_memory_status: PinMemoryStatus,
) -> ExactLoader<D, P, C, F, N>
where
    D: Dataset,
    F: TransformFactory<D::Sample>,
    F::Transform: WorkerCheckpoint,
    P: LoaderPlan<TransformOutput<D, F>, C>,
{
    let effective_prefetch = builder
        .configuration
        .prefetch_factor
        .and_then(NonZeroUsize::new);
    let outstanding_capacity =
        effective_prefetch.map(|factor| builder.configuration.workers * factor.get());
    OwnedDataLoader {
        dataset: Arc::new(builder.dataset),
        plan: builder.plan,
        collator: builder.collator,
        transform_factory: Arc::new(builder.transform_factory),
        worker_init: Arc::new(builder.worker_init),
        workers: builder.configuration.workers,
        persistent_workers: false,
        timeout: None,
        ordered: true,
        pin_memory_status,
        footprint: None,
        effective_prefetch_bytes: None,
        effective_prefetch,
        outstanding_capacity,
        loader_seed: builder.configuration.loader_seed,
        rank: builder.configuration.rank,
        next_generation: 0,
        persistent_pool: None,
        checkpoint,
        policies: PhantomData,
    }
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, N>
    DataLoaderBuilder<D, P, C, F, NoWorkerInit, WorkerExecution, MemoryDisabled, N, CheckpointFresh>
where
    D: WorkerReplayDataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<D, F>: Send + 'static,
    TransformOutput<D, F>: Send + 'static,
    TransformFailure<D, F>: Send + 'static,
    F::Error: Send + 'static,
    P: LoaderPlanConfiguration + CheckpointPlan<TransformOutput<D, F>, C>,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Builds an exact replay loader with explicit immutable dataset evidence.
    ///
    /// Positive workers require ordered delivery, no timeout or persistence,
    /// disabled byte accounting, and no worker initializer. Snapshot heap
    /// contents are item-bounded, not measured as total resident bytes.
    pub fn build(mut self) -> Result<ExactLoader<D, P, C, F, N>> {
        preflight::<D, P, C, F>(
            &mut self.configuration,
            self.explicit,
            &self.checkpoint.identity,
        )?;
        let pin = resolve_pin_request(self.configuration.pin_memory)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        self.plan
            .apply_batch_options(batch_size, self.configuration.drop_last);
        self.plan.apply_epoch(self.configuration.epoch);
        let identity = self.plan.checkpoint_identity();
        validate_distributed_rank(&identity, self.configuration.rank)?;
        let checkpoint = WorkerCheckpointActive {
            identity: self.checkpoint.identity.clone(),
            configuration: checkpoint_configuration(&identity, &self.configuration, pin)?,
            batches: None,
            serial: None,
            barrier: None,
            next_batch: 0,
            next_logical_sample: 0,
            factory_generation: 0,
            run_generation: 0,
        };
        Ok(finish_builder(self, checkpoint, pin))
    }
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, N, SS, TS, CS>
    DataLoaderBuilder<
        D,
        P,
        C,
        F,
        NoWorkerInit,
        WorkerExecution,
        MemoryDisabled,
        N,
        CheckpointResume<LoaderState<(), SS, WorkerTransformState<TS>, CS>>,
    >
where
    D: WorkerReplayDataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint<State = TS> + Send + 'static,
    TS: Clone + Send + 'static,
    TransformOutput<D, F>: Send + 'static,
    TransformFailure<D, F>: Send + 'static,
    F::Error: Send + 'static,
    P: LoaderPlanConfiguration
        + CheckpointPlan<TransformOutput<D, F>, C, State = SS, CoordinatorState = CS>,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Validates every component before restoring retained worker instances.
    #[allow(clippy::type_complexity)]
    pub fn build(
        mut self,
    ) -> std::result::Result<ExactLoader<D, P, C, F, N>, CheckpointBuildError<F::Error>> {
        preflight::<D, P, C, F>(
            &mut self.configuration,
            self.explicit,
            &self.checkpoint.identity,
        )?;
        let pin = resolve_pin_request(self.configuration.pin_memory)?;
        let batch_size = validate_batch_size(self.configuration.batch_size)?;
        let state = &self.checkpoint.state;
        validate_static_loader_state_envelope(
            state,
            &self.checkpoint.identity,
            &self.configuration,
            pin,
            P::AUTOMATIC_BATCHING,
            state.transform.run_generation,
        )?;
        let transform = &state.transform;
        require_checkpoint_equal("worker_schema_version", transform.schema_version, 1)?;
        require_checkpoint_equal("workers", transform.workers, self.configuration.workers)?;
        require_checkpoint_equal(
            "prefetch_factor",
            transform.prefetch_factor,
            self.configuration.prefetch_factor,
        )?;
        if transform.factory_generation > transform.run_generation
            || transform.run_generation == u64::MAX
        {
            return Err(error("invalid or exhausted worker generation").into());
        }
        let states = match (&transform.lanes, transform.workers) {
            (WorkerTransformLanes::Serial(_), 0)
                if transform.factory_generation == 0 && transform.run_generation == 0 =>
            {
                None
            }
            (WorkerTransformLanes::Workers(lanes), workers)
                if workers > 0 && lanes.len() == workers =>
            {
                for (id, lane) in lanes.iter().enumerate() {
                    require_checkpoint_equal("worker_lane", lane.id, id)?;
                }
                Some(lanes.iter().map(|lane| lane.state.clone()).collect())
            }
            _ => return Err(error("worker lane state does not match execution mode").into()),
        };
        let identity = P::checkpoint_identity_from_state(
            &state.sampler,
            batch_size,
            self.configuration.drop_last,
        )?;
        validate_distributed_rank(&identity, self.configuration.rank)?;
        let configuration = checkpoint_configuration(&identity, &self.configuration, pin)?;
        validate_plan_loader_state_envelope(state, &configuration)?;
        self.dataset.validate_dataset_state(&state.dataset)?;
        self.plan.validate_checkpoint_state_with_batch_options(
            &state.sampler,
            state.epoch,
            state.next_batch,
            state.next_logical_sample,
            batch_size,
            self.configuration.drop_last,
        )?;

        let checkpoint = WorkerCheckpointActive {
            identity: self.checkpoint.identity.clone(),
            configuration,
            batches: None,
            serial: None,
            barrier: None,
            next_batch: state.next_batch,
            next_logical_sample: state.next_logical_sample,
            factory_generation: transform.factory_generation,
            run_generation: transform.run_generation,
        };
        // Move the state out without reconstructing or restoring any component.
        let DataLoaderBuilder {
            dataset,
            plan,
            collator,
            transform_factory,
            worker_init,
            configuration,
            explicit,
            checkpoint: CheckpointResume { state, identity },
            ..
        } = self;
        let builder = DataLoaderBuilder {
            dataset,
            plan,
            collator,
            transform_factory,
            worker_init,
            configuration,
            explicit,
            checkpoint: CheckpointFresh { identity },
            states: PhantomData,
        };
        let mut loader = finish_builder(builder, checkpoint, pin);
        if loader.workers == 0 {
            let mut transform = loader
                .transform_factory
                .create(None)
                .map_err(CheckpointBuildError::TransformFactory)?;
            let WorkerTransformLanes::Serial(snapshot) = &state.transform.lanes else {
                unreachable!()
            };
            transform.validate_snapshot(snapshot)?;
            loader
                .plan
                .validate_coordinator(&loader.collator, &state.collate)?;
            transform.restore_validated(snapshot);
            loader.checkpoint.serial = Some(transform);
        } else {
            let (mut pool, barrier) = loader.create_checkpoint_pool(
                states,
                state.next_batch,
                state.transform.factory_generation,
            )?;
            pool.validate_initializations(&barrier.initialized)?;
            loader
                .plan
                .validate_coordinator(&loader.collator, &state.collate)?;
            // Worker restore starts only after all read-only validations succeeded.
            barrier.apply()?;
            loader.persistent_pool = Some(pool);
            loader.checkpoint.barrier = Some(barrier);
        }
        loader
            .plan
            .apply_batch_options(batch_size, loader.checkpoint.configuration.drop_last);
        loader.checkpoint.batches = Some(
            loader
                .plan
                .restore_checkpoint_iter_validated(&state.sampler),
        );
        loader
            .plan
            .restore_coordinator_validated(&mut loader.collator, &state.collate);
        loader.next_generation = state
            .transform
            .run_generation
            .checked_add(1)
            .ok_or_else(|| error("generation overflow"))?;
        Ok(loader)
    }
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, N> ExactLoader<D, P, C, F, N>
where
    D: WorkerReplayDataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<D, F>: Send + 'static,
    TransformOutput<D, F>: Send + 'static,
    TransformFailure<D, F>: Send + 'static,
    F::Error: Send + 'static,
    P: CheckpointPlan<TransformOutput<D, F>, C>,
    N: MapPinPolicy<D, P, C, F>,
{
    #[allow(clippy::type_complexity)]
    fn create_checkpoint_pool(
        &self,
        states: Option<Vec<TransformState<D, F>>>,
        boundary: u64,
        factory_generation: u64,
    ) -> std::result::Result<
        (
            WorkerPool<D, F, NoWorkerInit, MemoryDisabled>,
            Barrier<TransformState<D, F>>,
        ),
        CheckpointBuildError<F::Error>,
    > {
        let prefetch = self.effective_prefetch.expect("positive workers").get();
        let (barrier, mut lanes) = Barrier::new(self.workers, prefetch, boundary, states)?;
        let configuration = WorkerPoolConfiguration {
            workers: self.workers,
            prefetch_factor: prefetch,
            result_capacity: self.outstanding_capacity.expect("positive workers"),
            seed_generation: factory_generation,
            loader_seed: self.loader_seed,
            rank: self.rank,
        };
        let pool = WorkerPool::new_with_hooks(
            Arc::clone(&self.dataset),
            Arc::clone(&self.transform_factory),
            Arc::clone(&self.worker_init),
            configuration,
            None,
            true,
            |id| lanes[id].take().expect("unique lane"),
        )?;
        Ok((pool, barrier))
    }

    /// Starts the retained resume once, then fresh deterministic worker generations.
    #[allow(clippy::type_complexity)]
    pub fn iter(
        &mut self,
    ) -> WorkerLoaderIter<'_, D, P, C, F, NoWorkerInit, MemoryDisabled, N, IterCheckpoint<D, P, C, F>>
    {
        let resumed = self.checkpoint.batches.is_some();
        let generation = if self.workers == 0 {
            0
        } else if resumed {
            self.checkpoint.run_generation
        } else {
            self.next_generation
        };
        let factory_generation = if resumed {
            self.checkpoint.factory_generation
        } else {
            generation
        };
        let next_batch = if resumed {
            self.checkpoint.next_batch
        } else {
            0
        };
        let next_logical_sample = if resumed {
            self.checkpoint.next_logical_sample
        } else {
            0
        };
        let mut pending_error = None;
        let mut pool = self.persistent_pool.take();
        let mut barrier = self.checkpoint.barrier.take();
        let mut serial = self.checkpoint.serial.take();
        let mut serial_error = None;
        if self.workers > 0 {
            match generation.checked_add(1) {
                Some(next) => self.next_generation = self.next_generation.max(next),
                None => {
                    pending_error = Some(LoaderError::Configuration(error("generation overflow")))
                }
            }
            if pool.is_none() && pending_error.is_none() {
                let created = self
                    .create_checkpoint_pool(None, next_batch, factory_generation)
                    .and_then(|(mut pool, barrier)| {
                        pool.validate_initializations(&barrier.initialized)?;
                        barrier.apply()?;
                        Ok((pool, barrier))
                    });
                match created {
                    Ok((created_pool, created_barrier)) => {
                        pool = Some(created_pool);
                        barrier = Some(created_barrier);
                    }
                    Err(CheckpointBuildError::Configuration(error)) => {
                        pending_error = Some(LoaderError::Configuration(error))
                    }
                    Err(CheckpointBuildError::TransformFactory(source)) => {
                        pending_error = Some(LoaderError::Pipeline {
                            batch: None,
                            worker: None,
                            source: PipelineError::TransformInit(source),
                        })
                    }
                }
            }
        } else if serial.is_none() {
            match self.transform_factory.create(None) {
                Ok(transform) => serial = Some(transform),
                Err(error) => serial_error = Some(error),
            }
        }
        let (epoch, batches) = if resumed {
            (self.plan.epoch(), self.checkpoint.batches.take())
        } else {
            match catch_unwind(AssertUnwindSafe(|| (self.plan.epoch(), self.plan.iter()))) {
                Ok((epoch, batches)) => (epoch, Some(batches)),
                Err(_) => {
                    pending_error = Some(LoaderError::CoordinatorPanic {
                        stage: P::CREATION_PANIC_STAGE,
                        batch: None,
                    });
                    (0, None)
                }
            }
        };
        let run = WorkerRunContext::new(generation, self.loader_seed, epoch);
        if let Some(pool) = &mut pool
            && pool.start_generation(run.clone()).is_err()
        {
            pending_error = Some(LoaderError::ChannelClosed { batch: next_batch });
        }
        let checkpoint = WorkerCheckpointIteration {
            identity: self.checkpoint.identity.clone(),
            configuration: self.checkpoint.configuration.clone(),
            barrier,
            logical_ends: BTreeMap::new(),
            errors: BTreeMap::new(),
            next_logical_sample,
            factory_generation,
            boundary_valid: pending_error.is_none() && serial_error.is_none(),
        };
        let mut iterator = WorkerLoaderIter {
            dataset: Arc::clone(&self.dataset),
            plan: &mut self.plan,
            collator: &mut self.collator,
            batches,
            serial_transform: serial,
            serial_transform_error: serial_error,
            pool: pool.map(IteratorPool::Owned),
            completed: BTreeMap::new(),
            pending_error,
            generation,
            run_context: Some(run),
            workers: self.workers,
            capacity: self.outstanding_capacity.unwrap_or(0),
            ordered: true,
            timeout: None,
            loader_seed: self.loader_seed,
            epoch,
            rank: self.rank,
            next_submission: next_batch,
            next_visible: next_batch,
            next_logical_sample,
            pending_submission: None,
            outstanding: 0,
            source_exhausted: false,
            submission_closed: false,
            exhausted: false,
            pin_memory_status: self.pin_memory_status,
            _checkpoint: checkpoint,
            policies: PhantomData,
        };
        if iterator.workers > 0 && iterator.pending_error.is_none() {
            iterator.fill_available();
        }
        iterator
    }

    /// Discards any retained resume and selects the epoch for fresh iteration.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.persistent_pool = None;
        self.checkpoint.barrier = None;
        self.checkpoint.batches = None;
        self.checkpoint.serial = None;
        self.plan.set_epoch(epoch);
    }
}

#[allow(private_bounds, private_interfaces)]
impl<D, P, C, F, N>
    WorkerLoaderIter<'_, D, P, C, F, NoWorkerInit, MemoryDisabled, N, IterCheckpoint<D, P, C, F>>
where
    D: WorkerReplayDataset + Send + Sync + 'static,
    D::Sample: Send + 'static,
    D::Error: Send + 'static,
    F: TransformFactory<D::Sample> + Send + Sync + 'static,
    F::Transform: WorkerCheckpoint + Send + 'static,
    TransformState<D, F>: Send + 'static,
    TransformOutput<D, F>: Send + 'static,
    TransformFailure<D, F>: Send + 'static,
    F::Error: Send + 'static,
    P: CheckpointPlan<TransformOutput<D, F>, C>,
    N: MapPinPolicy<D, P, C, F>,
{
    /// Rolls back unpublished worker work and captures the next visible boundary.
    ///
    /// The original iterator remains usable, including repeated checkpoints.
    /// Blocking callbacks still require cooperative cancellation to join.
    #[allow(clippy::type_complexity)]
    pub fn checkpoint(
        &mut self,
    ) -> std::result::Result<ExactState<D, P, C, F>, IterError<D, P, C, F, NoWorkerInit>> {
        if !self._checkpoint.boundary_valid || self.exhausted {
            return Err(LoaderError::Checkpoint {
                reason: "iterator is not at a successful visible boundary".to_owned(),
            });
        }
        let next_logical_sample = self._checkpoint.next_logical_sample;
        let sampler = self
            .plan
            .checkpoint_state(self.next_visible, next_logical_sample)
            .map_err(LoaderError::Configuration)?;
        self.plan
            .validate_checkpoint_state(&sampler, self.epoch, self.next_visible, next_logical_sample)
            .map_err(LoaderError::Configuration)?;
        let lanes = if self.workers == 0 {
            WorkerTransformLanes::Serial(
                self.serial_transform
                    .as_ref()
                    .expect("validated serial transform")
                    .snapshot(),
            )
        } else {
            let next_generation = self
                .generation
                .checked_add(1)
                .filter(|generation| *generation < u64::MAX)
                .ok_or_else(|| LoaderError::Configuration(error("generation overflow")))?;
            let rollback = (|| -> Result<Vec<TransformState<D, F>>> {
                let barrier = self
                    ._checkpoint
                    .barrier
                    .as_ref()
                    .ok_or_else(|| error("missing barrier"))?;
                barrier.request(self.generation, self.next_visible)?;
                let pool = self
                    .pool
                    .as_mut()
                    .ok_or_else(|| error("missing pool"))?
                    .pool_mut();
                pool.checkpoint_quiesce(self.generation)
                    .map_err(|()| error("worker failed before rollback acknowledgment"))?;
                barrier.collect(self.generation)
            })();
            let snapshots = match rollback {
                Ok(states) => states,
                Err(error) => {
                    self._checkpoint.boundary_valid = false;
                    self.stop_poisoned();
                    return Err(LoaderError::Configuration(error));
                }
            };
            self.completed.clear();
            self.pending_error = None;
            self.pending_submission = None;
            self._checkpoint.errors.clear();
            self._checkpoint.logical_ends.clear();
            self.outstanding = 0;
            self.next_submission = self.next_visible;
            self.next_logical_sample = next_logical_sample;
            self.batches = Some(self.plan.restore_checkpoint_iter_validated(&sampler));
            self.source_exhausted = false;
            self.submission_closed = false;
            self.generation = next_generation;
            let run = WorkerRunContext::new(next_generation, self.loader_seed, self.epoch);
            if self
                .pool
                .as_mut()
                .expect("live pool")
                .pool_mut()
                .start_generation(run.clone())
                .is_err()
            {
                self._checkpoint.boundary_valid = false;
                self.stop_poisoned();
                return Err(LoaderError::Configuration(error(
                    "worker failed to restart after rollback",
                )));
            }
            self.run_context = Some(run);
            WorkerTransformLanes::Workers(
                snapshots
                    .into_iter()
                    .enumerate()
                    .map(|(id, state)| WorkerLaneState { id, state })
                    .collect(),
            )
        };
        let state = LoaderState {
            schema_version: LOADER_STATE_SCHEMA_VERSION,
            dataset_identity: self._checkpoint.identity.clone(),
            epoch: self.epoch,
            iterator_generation: self.generation,
            next_batch: self.next_visible,
            next_logical_sample,
            dataset: (),
            sampler,
            transform: WorkerTransformState {
                schema_version: 1,
                workers: self.workers,
                prefetch_factor: self._checkpoint.configuration.prefetch_factor,
                factory_generation: self._checkpoint.factory_generation,
                run_generation: self.generation,
                lanes,
            },
            collate: self.plan.save_coordinator(self.collator),
            rng_derivation_version: TASK_RNG_DERIVATION_VERSION,
            worker_seed_derivation_version: WORKER_SEED_DERIVATION_VERSION,
            configuration: self._checkpoint.configuration.clone(),
        };
        if self.workers > 0 {
            self.fill_available();
        }
        Ok(state)
    }
}
