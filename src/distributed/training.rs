use super::{
    ProcessGroup, ReduceOp, decode, invalid, json,
    process::{Dtype, Payload},
};
use crate::{
    Device, Kind, Result, Tensor,
    nn::VarStore,
    no_grad,
    optim::{Optimizer, OptimizerState},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Explicit synchronization for an ordinary replicated CPU model.
///
/// Construct after registering every parameter and buffer. Construction checks
/// identical names/shapes/dtypes and broadcasts rank-zero state. Before a
/// training forward, call [`Self::sync_buffers`]; after backward, call
/// [`Self::sync_gradients`] before the optimizer update. This permits gradient
/// accumulation by delaying synchronization until the last microbatch.
///
/// ```
/// use rusttorch::{Device, Tensor, nn, distributed::{ProcessGroup, GroupOptions, DistributedDataParallel}};
/// let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
/// let mut group = ProcessGroup::accept(listener, 1, GroupOptions::new("replicated"))?;
/// let store = nn::VarStore::new(Device::Cpu);
/// let _linear = nn::LinearConfig::new(2, 1).build(&store.root())?;
/// let replica = DistributedDataParallel::new(&mut group, &store)?;
/// replica.sync_buffers(&mut group)?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
///
/// Model registration is fixed after construction. Undefined gradients are
/// skipped only when undefined on every rank; differing used-parameter sets,
/// sparse gradients and accelerator tensors fail coherently. Synchronization
/// is blocking and does not install automatic backward hooks.
#[derive(Debug)]
pub struct DistributedDataParallel {
    parameters: Vec<(String, Tensor)>,
    buffers: Vec<(String, Tensor)>,
    identity: GroupIdentity,
    variables: std::sync::Arc<std::sync::Mutex<tch::nn::Variables>>,
    specifications: BTreeMap<String, (Vec<i64>, Kind, bool)>,
}
impl DistributedDataParallel {
    /// Validates a fixed model layout and initializes every replica from rank zero.
    pub fn new(group: &mut ProcessGroup, store: &VarStore) -> Result<Self> {
        let variables: BTreeMap<_, _> = store.variables().into_iter().collect();
        let metadata = coordinated(
            group,
            variables
                .iter()
                .map(|(name, t)| {
                    let payload = Payload::capture(t, group.max_message_bytes())?;
                    Ok((
                        name.clone(),
                        payload.shape,
                        payload.dtype,
                        t.requires_grad(),
                    ))
                })
                .collect::<Result<Vec<_>>>(),
        )?;
        group.identical(&json(&metadata)?)?;
        let specifications = variables
            .iter()
            .map(|(name, t)| (name.clone(), (t.size(), t.kind(), t.requires_grad())))
            .collect();
        let values = variables
            .values()
            .map(|t| group.broadcast(t, 0))
            .collect::<Result<Vec<_>>>()?;
        no_grad(|| -> Result<()> {
            for (mut target, value) in variables.values().map(Tensor::shallow_clone).zip(values) {
                target.f_copy_(&value)?;
            }
            Ok(())
        })?;
        let (parameters, buffers) = variables.into_iter().partition(|(_, t)| t.requires_grad());
        Ok(Self {
            parameters,
            buffers,
            identity: GroupIdentity::new(group),
            variables: store.variables_.clone(),
            specifications,
        })
    }
    fn validate(&self, group: &mut ProcessGroup) -> Result<()> {
        self.identity.check(group)?;
        let status = (|| {
            let variables = self
                .variables
                .lock()
                .map_err(|_| invalid("replicated parameter store lock is poisoned"))?;
            if variables.named_variables.len() != self.specifications.len() {
                return Err(invalid("replicated model registration changed"));
            }
            for (name, tensor) in self.parameters.iter().chain(&self.buffers) {
                let current = variables
                    .named_variables
                    .get(name)
                    .ok_or_else(|| invalid("replicated model parameter was removed"))?;
                if current.device() != Device::Cpu
                    || !current.f_is_set_to(tensor)?
                    || self.specifications.get(name)
                        != Some(&(current.size(), current.kind(), current.requires_grad()))
                {
                    return Err(invalid(
                        "replicated model shape, dtype, identity or training flags changed",
                    ));
                }
            }
            Ok(())
        })();
        group.agree(status)
    }
    /// Copies rank-zero persistent buffers before the next training forward.
    ///
    /// This matches replicated running-statistic synchronization; it does not
    /// implement synchronized batch normalization across local minibatches.
    pub fn sync_buffers(&self, group: &mut ProcessGroup) -> Result<()> {
        self.validate(group)?;
        let values = self
            .buffers
            .iter()
            .map(|(_, t)| group.broadcast(t, 0))
            .collect::<Result<Vec<_>>>()?;
        let copied = no_grad(|| -> Result<()> {
            for ((_, target), value) in self.buffers.iter().zip(values) {
                target.shallow_clone().f_copy_(&value)?;
            }
            Ok(())
        });
        group.agree(copied)
    }
    /// Averages dense gradients, preserving accumulated local gradients until here.
    ///
    /// Each rank must contribute a loss with the same normalization: equal local
    /// batch sizes and mean losses match a global-batch mean. For uneven batches,
    /// scale each local sum by `world_size / global_sample_count` before backward.
    pub fn sync_gradients(&self, group: &mut ProcessGroup) -> Result<()> {
        self.validate(group)?;
        let gradients = coordinated(
            group,
            self.parameters
                .iter()
                .map(|(_, t)| Ok(t.f_grad()?))
                .collect::<Result<Vec<_>>>(),
        )?;
        let metadata = coordinated(
            group,
            gradients
                .iter()
                .zip(&self.parameters)
                .map(|(gradient, (name, p))| {
                    if gradient.defined() {
                        let payload = Payload::capture(gradient, group.max_message_bytes())?;
                        if gradient.size() != p.size() || gradient.kind() != p.kind() {
                            return Err(invalid("gradient differs from parameter shape or dtype"));
                        }
                        Ok((name.clone(), Some((payload.shape, payload.dtype))))
                    } else {
                        Ok((name.clone(), None))
                    }
                })
                .collect::<Result<Vec<_>>>(),
        )?;
        group.identical(&json(&metadata)?)?;
        let mut updates = Vec::new();
        for gradient in gradients.into_iter().filter(Tensor::defined) {
            let value = group.all_reduce(&gradient, ReduceOp::Mean)?;
            updates.push((gradient, value));
        }
        let copied = no_grad(|| -> Result<()> {
            for (mut gradient, value) in updates {
                gradient.f_copy_(&value)?;
            }
            Ok(())
        });
        group.agree(copied)
    }
    /// Clears gradients, differentiates a scalar local loss, averages, then updates.
    ///
    /// Build the loss after `sync_buffers` and use an optimizer attached to the
    /// same store. A failure after an update starts can leave ranks inconsistent;
    /// the group becomes unusable and all ranks must restore a checkpoint.
    pub fn backward_step(
        &self,
        group: &mut ProcessGroup,
        optimizer: &mut Optimizer,
        loss: &Tensor,
    ) -> Result<()> {
        self.validate(group)?;
        let attached = optimizer.named_parameters();
        let valid = if attached.len() != self.parameters.len() {
            Err(invalid("DDP optimizer parameter count differs"))
        } else {
            attached
                .iter()
                .zip(&self.parameters)
                .try_for_each(|((name, t), (expected, p))| {
                    if name != expected || !t.f_is_set_to(p)? {
                        Err(invalid("DDP optimizer is attached to a different model"))
                    } else {
                        Ok(())
                    }
                })
        };
        group.agree(valid)?;
        let settings = coordinated(group, optimizer.distributed_configuration())?;
        group.identical(&settings)?;
        group.agree(optimizer.try_zero_grad())?;
        group.agree(scalar(loss).and_then(|_| Ok(loss.f_backward()?)))?;
        self.sync_gradients(group)?;
        group.agree(optimizer.try_step())
    }
}

#[derive(Debug)]
struct GroupIdentity {
    rank: usize,
    world: usize,
    session: String,
}
impl GroupIdentity {
    fn new(group: &ProcessGroup) -> Self {
        Self {
            rank: group.rank(),
            world: group.world_size(),
            session: group.session().into(),
        }
    }
    fn check(&self, group: &mut ProcessGroup) -> Result<()> {
        group.agree(
            if self.rank == group.rank()
                && self.world == group.world_size()
                && self.session == group.session()
            {
                Ok(())
            } else {
                Err(invalid("trainer belongs to another process group"))
            },
        )
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterSpec {
    name: String,
    shape: Vec<i64>,
    dtype: Dtype,
}

/// One rank's versioned model and optimizer checkpoint, with no executable code.
///
/// Obtain this collectively from [`ShardedTrainer::checkpoint`], then persist it
/// with `serde_json::to_writer`. Save **all** ranks' records from the same call;
/// [`Self::consolidate`] checks rank coverage and matching optimizer metadata.
/// Restore one record per original rank with [`ShardedTrainer::restore`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardedCheckpoint {
    schema_version: u32,
    checkpoint_id: String,
    rank: usize,
    world_size: usize,
    parameters: Vec<ParameterSpec>,
    weights: BTreeMap<String, Payload>,
    optimizer: OptimizerState,
}
impl ShardedCheckpoint {
    /// Session/operation identity shared by records from the same capture.
    ///
    /// A complete set must have one identity, even when two jobs reached the
    /// same optimizer step. Use a fresh process-group session for each job.
    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }
    /// Rank whose model and optimizer shards this record contains.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Original number of ranks; all records are needed for consolidation.
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    /// Combines a complete checkpoint set into a portable full model/state value.
    ///
    /// Records may be supplied in any order. Missing/duplicate ranks, schemas,
    /// names, shapes, dtypes, optimizer families, settings or step mismatches fail.
    /// Consolidation allocates full weights and optimizer moments on the caller.
    pub fn consolidate(shards: &[Self]) -> Result<ConsolidatedCheckpoint> {
        let first = shards
            .first()
            .ok_or_else(|| invalid("empty checkpoint shard set"))?;
        if first.world_size == 0 || first.world_size > 1024 || shards.len() != first.world_size {
            return Err(invalid("checkpoint shard set is incomplete"));
        }
        let mut ordered: Vec<Option<&Self>> = vec![None; first.world_size];
        for shard in shards {
            shard.validate()?;
            if shard.world_size != first.world_size
                || shard.checkpoint_id != first.checkpoint_id
                || shard.parameters != first.parameters
                || ordered[shard.rank].replace(shard).is_some()
            {
                return Err(invalid(
                    "checkpoint shard identities differ or ranks repeat",
                ));
            }
        }
        let ordered: Vec<_> = ordered
            .into_iter()
            .map(|s| s.expect("every rank covered"))
            .collect();
        let mut weights = BTreeMap::new();
        for spec in &first.parameters {
            let count = elements(&spec.shape)?;
            let width = spec.dtype.kind().elt_size_in_bytes();
            let length = count
                .checked_mul(width)
                .ok_or_else(|| invalid("checkpoint byte count overflow"))?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(length)
                .map_err(|_| invalid("cannot allocate consolidated model"))?;
            for shard in &ordered {
                let source = &shard.weights[&spec.name].bytes;
                let remaining = length - bytes.len();
                bytes.extend_from_slice(&source[..remaining.min(source.len())]);
            }
            weights.insert(
                spec.name.clone(),
                Payload {
                    shape: spec.shape.clone(),
                    dtype: spec.dtype,
                    bytes,
                },
            );
        }
        let shapes = first
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone()))
            .collect();
        let optimizer = OptimizerState::consolidate_shards(
            &ordered
                .iter()
                .map(|s| s.optimizer.clone())
                .collect::<Vec<_>>(),
            &shapes,
        )?;
        Ok(ConsolidatedCheckpoint {
            schema_version: 1,
            parameters: first.parameters.clone(),
            weights,
            optimizer,
        })
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.checkpoint_id.is_empty()
            || self.checkpoint_id.len() > 300
            || self.world_size == 0
            || self.world_size > 1024
            || self.rank >= self.world_size
        {
            return Err(invalid("invalid checkpoint schema, rank or world size"));
        }
        validate_specs(&self.parameters)?;
        if self.weights.len() != self.parameters.len() {
            return Err(invalid("checkpoint weight count differs"));
        }
        let optimizer_specs = self.optimizer.parameter_specs()?;
        if optimizer_specs.len() != self.parameters.len() {
            return Err(invalid("optimizer and model shard counts differ"));
        }
        for spec in &self.parameters {
            let payload = self
                .weights
                .get(&spec.name)
                .ok_or_else(|| invalid("checkpoint weight name missing"))?;
            payload.validate(1024 * 1024 * 1024)?;
            if optimizer_specs.get(&spec.name)
                != Some(&(payload.shape.clone(), payload.dtype.kind()))
            {
                return Err(invalid("optimizer and model shard metadata differ"));
            }
            if payload.dtype != spec.dtype
                || payload.shape != [chunk_size(spec, self.world_size)? as i64]
            {
                return Err(invalid("checkpoint weight shard shape or dtype differs"));
            }
        }
        Ok(())
    }
}

/// Consolidated model and optimizer state, independent of the original rank count.
///
/// Use [`ShardedCheckpoint::consolidate`] or
/// [`ShardedTrainer::consolidate_checkpoint`], serialize with serde, and restore
/// with [`ShardedTrainer::restore_full`] on a new group. This is RustTorch's
/// versioned checkpoint schema, not a Python FSDP state-dictionary wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolidatedCheckpoint {
    schema_version: u32,
    parameters: Vec<ParameterSpec>,
    weights: BTreeMap<String, Payload>,
    optimizer: OptimizerState,
}
impl ConsolidatedCheckpoint {
    /// Returns independent named CPU tensors for ordinary model loading/inference.
    pub fn weights(&self) -> Result<BTreeMap<String, Tensor>> {
        self.validate()?;
        self.weights
            .iter()
            .map(|(name, value)| Ok((name.clone(), value.tensor()?)))
            .collect()
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(invalid("unsupported consolidated checkpoint schema"));
        }
        validate_specs(&self.parameters)?;
        if self.weights.len() != self.parameters.len() {
            return Err(invalid("consolidated weight count differs"));
        }
        for spec in &self.parameters {
            let payload = self
                .weights
                .get(&spec.name)
                .ok_or_else(|| invalid("consolidated weight name missing"))?;
            payload.validate(1024 * 1024 * 1024)?;
            if payload.shape != spec.shape || payload.dtype != spec.dtype {
                return Err(invalid("consolidated weight shape or dtype differs"));
            }
        }
        Ok(())
    }
    fn shard(&self, rank: usize, world: usize) -> Result<ShardedCheckpoint> {
        self.validate()?;
        if world == 0 || world > 1024 || rank >= world {
            return Err(invalid("invalid target checkpoint rank/world"));
        }
        let weights = self
            .weights
            .iter()
            .map(|(name, p)| {
                Ok((
                    name.clone(),
                    Payload::capture(
                        &shard_tensor(&p.tensor()?, rank, world)?,
                        1024 * 1024 * 1024,
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(ShardedCheckpoint {
            schema_version: 1,
            checkpoint_id: "consolidated".into(),
            rank,
            world_size: world,
            parameters: self.parameters.clone(),
            weights,
            optimizer: self.optimizer.shard(rank, world)?,
        })
    }
}

/// Functional CPU training with actual parameter, gradient and optimizer shards.
///
/// Each rank owns a padded flat slice of each parameter. A training call gathers
/// one full parameter group for the closure's forward/backward computation,
/// reduce-scatters averaged gradients, releases full tensors, and updates only
/// local slices using the existing optimizer. Moment tensors are local shards.
///
/// ```
/// use rusttorch::{Tensor, Kind, optim::Sgd,
///     distributed::{GroupOptions, ProcessGroup, ShardedTrainer}};
/// let listener=std::net::TcpListener::bind("127.0.0.1:0").unwrap();
/// let mut group=ProcessGroup::accept(listener,1,GroupOptions::new("sharded"))?;
/// let mut trainer=ShardedTrainer::new(&mut group,
///     vec![("weight".into(),Tensor::from_slice(&[0_f64,0.]))],
///     |store| Sgd::builder().learning_rate(0.1).build(store))?;
/// let loss=trainer.train_step(&mut group, |p| {
///     let prediction=Tensor::from_slice(&[1_f64,2.]).f_dot(&p["weight"])?;
///     Ok(prediction.f_sub_scalar(3.)?.f_square()?.f_mean(Kind::Double)?)
/// })?;
/// assert_eq!(loss,9.);
/// let checkpoint=trainer.checkpoint(&mut group)?;
/// assert_eq!(checkpoint.rank(),0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
///
/// This is a synchronous, single-group FSDP-equivalent training contract. Full
/// parameters coexist during forward/backward; there are no module hooks,
/// layer-wise resharding, mixed precision, overlap or accelerator support.
/// Closures must use every supplied parameter and must not retain full tensor
/// copies/graphs. Drop external initial-weight aliases to realize shard storage
/// savings. Checkpoint consolidation temporarily gathers full state on all ranks.
#[derive(Debug)]
pub struct ShardedTrainer {
    identity: GroupIdentity,
    parameters: Vec<ParameterSpec>,
    store: VarStore,
    optimizer: Optimizer,
}
impl ShardedTrainer {
    /// Consumes initial named CPU Float/Double tensors and builds a local optimizer.
    ///
    /// Names may contain dot-separated module paths. Parameters must be nonempty
    /// and have distinct canonical names. All ranks provide the same layout;
    /// initial values come from rank zero. The builder can use any supported
    /// optimizer family; all parameters are in group zero.
    pub fn new<F>(
        group: &mut ProcessGroup,
        parameters: Vec<(String, Tensor)>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(&VarStore) -> Result<Optimizer>,
    {
        let mut ordered = BTreeMap::new();
        let checked = (|| {
            for (name, tensor) in parameters {
                valid_name(&name)?;
                if ordered.insert(name, tensor).is_some() {
                    return Err(invalid("duplicate sharded parameter name"));
                }
            }
            let specs = ordered
                .iter()
                .map(|(name, t)| {
                    let p = Payload::capture(t, group.max_message_bytes())?;
                    Ok(ParameterSpec {
                        name: name.clone(),
                        shape: p.shape,
                        dtype: p.dtype,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            validate_specs(&specs)?;
            Ok(specs)
        })();
        let parameters = coordinated(group, checked)?;
        group.identical(&json(&parameters)?)?;
        let mut local = BTreeMap::new();
        for (name, value) in ordered {
            let full = group.broadcast(&value, 0)?;
            local.insert(name, shard_tensor(&full, group.rank(), group.world_size())?);
        }
        Self::from_local(group, parameters, local, build)
    }
    fn from_local<F>(
        group: &mut ProcessGroup,
        parameters: Vec<ParameterSpec>,
        weights: BTreeMap<String, Tensor>,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(&VarStore) -> Result<Optimizer>,
    {
        let built = (|| {
            let store = VarStore::new(Device::Cpu);
            for (name, tensor) in weights {
                valid_name(&name)?;
                let mut parts = name.split('.').collect::<Vec<_>>();
                let leaf = parts.pop().expect("nonempty validated name");
                let mut path = store.root();
                for part in parts {
                    path = path.sub(part);
                }
                let _ = path.add(leaf, tensor.f_set_requires_grad(true)?, true);
            }
            let optimizer = build(&store)?;
            Ok(Self {
                identity: GroupIdentity::new(group),
                parameters,
                store,
                optimizer,
            })
        })();
        let mut trainer = coordinated(group, built)?;
        let attached = trainer.optimizer.named_parameters();
        let local = trainer.store.variables();
        let registration = (|| {
            if attached.len() != local.len() {
                return Err(invalid(
                    "sharded optimizer is attached to another parameter store",
                ));
            }
            for (name, tensor) in attached {
                let value = local
                    .get(&name)
                    .ok_or_else(|| invalid("sharded optimizer parameter name differs"))?;
                if !tensor.f_is_set_to(value)? {
                    return Err(invalid("sharded optimizer parameter identity differs"));
                }
            }
            Ok(())
        })();
        group.agree(registration)?;
        let settings = coordinated(group, trainer.optimizer.distributed_configuration())?;
        group.identical(&settings)?;
        Ok(trainer)
    }
    /// Number of locally stored parameter elements, including padding.
    ///
    /// This is storage accounting, not a peak-memory or performance benchmark.
    /// Moment tensors have the same local shapes; full forward activations and
    /// temporary gathered weights are excluded.
    pub fn local_parameter_elements(&self) -> usize {
        self.store.variables().values().map(Tensor::numel).sum()
    }
    /// Accesses local optimizer controls, for example a scheduler's learning rate.
    ///
    /// Apply identical settings on every rank. Do not call its step/zero-grad
    /// methods outside a coordinated training operation.
    pub fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }
    /// Materializes independent full CPU weights on every rank for inference.
    ///
    /// Release this map after use to return to shard-sized persistent storage.
    pub fn full_parameters(&self, group: &mut ProcessGroup) -> Result<BTreeMap<String, Tensor>> {
        self.identity.check(group)?;
        let local = self.store.variables();
        let valid = (|| {
            if local.len() != self.parameters.len() {
                return Err(invalid("sharded parameter registration changed"));
            }
            for spec in &self.parameters {
                let tensor = local
                    .get(&spec.name)
                    .ok_or_else(|| invalid("sharded parameter name changed"))?;
                if tensor.device() != Device::Cpu
                    || tensor.kind() != spec.dtype.kind()
                    || tensor.size() != [chunk_size(spec, group.world_size())? as i64]
                    || !tensor.requires_grad()
                {
                    return Err(invalid(
                        "local parameter shard shape, dtype, device or training flags changed",
                    ));
                }
            }
            Ok(())
        })();
        group.agree(valid)?;
        let mut result = BTreeMap::new();
        for spec in &self.parameters {
            let shards = group.all_gather(&local[&spec.name])?;
            let full = Tensor::f_cat(&shards, 0)?
                .f_narrow(0, 0, elements(&spec.shape)? as i64)?
                .f_reshape(&spec.shape)?;
            result.insert(spec.name.clone(), full);
        }
        Ok(result)
    }
    /// Runs one local loss closure and returns the mean loss across ranks.
    ///
    /// Equal-size local batches with mean losses match one global-batch update.
    /// Every named parameter must participate in a scalar real loss. The closure
    /// supplies all data, buffers and stochastic behavior explicitly; preserve
    /// those separately when exact input/RNG replay is required.
    pub fn train_step<F>(&mut self, group: &mut ProcessGroup, loss: F) -> Result<f64>
    where
        F: FnOnce(&BTreeMap<String, Tensor>) -> Result<Tensor>,
    {
        let settings = coordinated(group, self.optimizer.distributed_configuration())?;
        group.identical(&settings)?;
        let mut full = self.full_parameters(group)?;
        for tensor in full.values_mut() {
            *tensor = tensor.f_set_requires_grad(true)?;
        }
        let objective = coordinated(group, loss(&full))?;
        group.agree(scalar(&objective))?;
        let local_loss = objective.f_double_value(&[])?;
        let gradients = coordinated(
            group,
            (|| {
                let inputs: Vec<_> = full.values().collect();
                let gradients = Tensor::f_run_backward(&[&objective], &inputs, false, false)?;
                if gradients.iter().any(|g| !g.defined()) {
                    return Err(invalid(
                        "every sharded parameter must participate in the loss",
                    ));
                }
                Ok(gradients)
            })(),
        )?;
        drop(objective);
        let mut reduced = Vec::new();
        for gradient in gradients {
            let chunks = (0..group.world_size())
                .map(|r| shard_tensor(&gradient, r, group.world_size()))
                .collect::<Result<Vec<_>>>()?;
            reduced.push(group.reduce_scatter(&chunks, ReduceOp::Mean)?);
        }
        drop(full);
        group.agree(self.optimizer.try_zero_grad())?;
        let local = self.store.variables();
        let seeded = (|| {
            for (spec, gradient) in self.parameters.iter().zip(reduced) {
                local[&spec.name]
                    .f_mul(&gradient.f_detach()?)?
                    .f_sum(spec.dtype.kind())?
                    .f_backward()?;
            }
            Ok(())
        })();
        group.agree(seeded)?;
        group.agree(self.optimizer.try_step())?;
        Ok(group
            .all_reduce(&Tensor::from(local_loss), ReduceOp::Mean)?
            .f_double_value(&[])?)
    }
    fn local_checkpoint(&mut self, group: &mut ProcessGroup) -> Result<ShardedCheckpoint> {
        self.identity.check(group)?;
        let result = (|| {
            let weights = self
                .store
                .variables()
                .into_iter()
                .map(|(name, t)| Ok((name, Payload::capture(&t, group.max_message_bytes())?)))
                .collect::<Result<_>>()?;
            Ok(ShardedCheckpoint {
                schema_version: 1,
                checkpoint_id: group.checkpoint_stamp(),
                rank: group.rank(),
                world_size: group.world_size(),
                parameters: self.parameters.clone(),
                weights,
                optimizer: self.optimizer.state_dict()?,
            })
        })();
        coordinated(group, result)
    }
    fn checkpoint_set(&mut self, group: &mut ProcessGroup) -> Result<Vec<ShardedCheckpoint>> {
        let local = self.local_checkpoint(group)?;
        let records = group.all_gather_bytes(&json(&local)?)?;
        coordinated(group, records.iter().map(|bytes| decode(bytes)).collect())
    }
    /// Captures and validates a completed model/optimizer checkpoint on every rank.
    ///
    /// This collective checks the complete shard set before returning local
    /// records. Persist each rank's result and publish your manifest only after
    /// all writes succeed. DataLoader/RNG state is application-owned.
    pub fn checkpoint(&mut self, group: &mut ProcessGroup) -> Result<ShardedCheckpoint> {
        let shards = self.checkpoint_set(group)?;
        group.agree(ShardedCheckpoint::consolidate(&shards).map(|_| ()))?;
        Ok(shards[group.rank()].clone())
    }
    /// Collectively captures full model and optimizer state on every rank.
    ///
    /// Save one returned record for restoring with a different world size.
    pub fn consolidate_checkpoint(
        &mut self,
        group: &mut ProcessGroup,
    ) -> Result<ConsolidatedCheckpoint> {
        let shards = self.checkpoint_set(group)?;
        coordinated(group, ShardedCheckpoint::consolidate(&shards))
    }
    /// Reconstructs a trainer from one checkpoint per original rank.
    ///
    /// The optimizer builder must select the saved algorithm family. Checkpoint
    /// settings replace its initial rates/moments. A new group/session is valid;
    /// the world size and each record's rank must match the original layout.
    pub fn restore<F>(
        group: &mut ProcessGroup,
        checkpoint: &ShardedCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(&VarStore) -> Result<Optimizer>,
    {
        group.agree(checkpoint.validate().and_then(|_| {
            if checkpoint.rank == group.rank() && checkpoint.world_size == group.world_size() {
                Ok(())
            } else {
                Err(invalid("checkpoint rank/world differs from process group"))
            }
        }))?;
        let records = group.all_gather_bytes(&json(checkpoint)?)?;
        let records = coordinated(
            group,
            records
                .iter()
                .map(|b| decode(b))
                .collect::<Result<Vec<ShardedCheckpoint>>>(),
        )?;
        group.agree(ShardedCheckpoint::consolidate(&records).map(|_| ()))?;
        let weights = coordinated(
            group,
            checkpoint
                .weights
                .iter()
                .map(|(name, p)| Ok((name.clone(), p.tensor()?)))
                .collect(),
        )?;
        let mut trainer = Self::from_local(group, checkpoint.parameters.clone(), weights, build)?;
        let prepared = coordinated(
            group,
            trainer.optimizer.prepare_state_dict(&checkpoint.optimizer),
        )?;
        trainer.optimizer.apply_prepared_state(prepared);
        Ok(trainer)
    }
    /// Restores a consolidated checkpoint with any valid new rank count.
    ///
    /// Every rank supplies the same full record. Parameter and moment bytes are
    /// repartitioned consistently, including padding and per-parameter steps.
    pub fn restore_full<F>(
        group: &mut ProcessGroup,
        checkpoint: &ConsolidatedCheckpoint,
        build: F,
    ) -> Result<Self>
    where
        F: FnOnce(&VarStore) -> Result<Optimizer>,
    {
        group.identical(&json(checkpoint)?)?;
        let shard = coordinated(group, checkpoint.shard(group.rank(), group.world_size()))?;
        Self::restore(group, &shard, build)
    }
}

fn coordinated<T>(group: &mut ProcessGroup, result: Result<T>) -> Result<T> {
    group.agree(
        result
            .as_ref()
            .map(|_| ())
            .map_err(|e| invalid(e.to_string())),
    )?;
    result
}
fn valid_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 4096
        || name.split('.').any(str::is_empty)
        || name.contains('\0')
    {
        Err(invalid(
            "parameter names require nonempty dot-separated components",
        ))
    } else {
        Ok(())
    }
}
fn validate_specs(values: &[ParameterSpec]) -> Result<()> {
    if values.is_empty() || values.windows(2).any(|p| p[0].name >= p[1].name) {
        return Err(invalid(
            "sharded parameter list must be nonempty with unique sorted names",
        ));
    }
    for spec in values {
        valid_name(&spec.name)?;
        if !matches!(spec.dtype, Dtype::Float | Dtype::Double) || elements(&spec.shape)? == 0 {
            return Err(invalid(
                "sharded parameters must be nonempty CPU Float/Double tensors",
            ));
        }
    }
    Ok(())
}
fn elements(shape: &[i64]) -> Result<usize> {
    if shape.len() > 64 {
        return Err(invalid("parameter rank exceeds 64"));
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| {
            usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
        })
        .filter(|n| *n <= i64::MAX as usize)
        .ok_or_else(|| invalid("parameter dimensions are negative or overflow"))
}
fn chunk_size(spec: &ParameterSpec, world: usize) -> Result<usize> {
    Ok(elements(&spec.shape)?.div_ceil(world))
}
fn shard_tensor(tensor: &Tensor, rank: usize, world: usize) -> Result<Tensor> {
    let full = tensor.f_detach()?.f_reshape([-1])?;
    let count = full.numel();
    let chunk = count.div_ceil(world);
    let start = rank
        .checked_mul(chunk)
        .ok_or_else(|| invalid("shard offset overflow"))?
        .min(count);
    let length = chunk.min(count - start);
    let part = full.f_narrow(0, start as i64, length as i64)?;
    let zeros = Tensor::f_zeros([(chunk - length) as i64], (tensor.kind(), Device::Cpu))?;
    Ok(Tensor::f_cat(&[part, zeros], 0)?)
}
fn scalar(loss: &Tensor) -> Result<()> {
    if !loss.defined()
        || loss.device() != Device::Cpu
        || loss.numel() != 1
        || !loss.requires_grad()
        || !matches!(loss.kind(), Kind::Float | Kind::Double)
    {
        Err(invalid(
            "training loss must be a tracked CPU Float/Double scalar",
        ))
    } else {
        Ok(())
    }
}
