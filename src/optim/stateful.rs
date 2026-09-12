use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Deserialize, Serialize};
use tch::{
    COptimizer, Device, Kind, Tensor,
    nn::{VarStore, Variables},
};

use super::{validate_beta, validate_loss, validate_non_negative};
use crate::{Result, RustTorchError, no_grad};

static NEXT_OPTIMIZER_ID: AtomicU64 = AtomicU64::new(1);
const SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) enum Algorithm {
    Adam {
        beta1: f64,
        beta2: f64,
        eps: f64,
        amsgrad: bool,
        decoupled: bool,
    },
    Sgd {
        momentum: f64,
        dampening: f64,
        nesterov: bool,
    },
    RmsProp {
        alpha: f64,
        eps: f64,
        momentum: f64,
        centered: bool,
    },
    Adagrad {
        lr_decay: f64,
        initial_accumulator: f64,
        eps: f64,
    },
    Adadelta {
        rho: f64,
        eps: f64,
    },
    Adamax {
        beta1: f64,
        beta2: f64,
        eps: f64,
    },
}

impl Algorithm {
    fn validate(&self) -> Result<()> {
        match *self {
            Self::Adam {
                beta1, beta2, eps, ..
            }
            | Self::Adamax { beta1, beta2, eps } => {
                validate_beta("beta1", beta1)?;
                validate_beta("beta2", beta2)?;
                validate_non_negative("eps", eps)
            }
            Self::Sgd {
                momentum,
                dampening,
                nesterov,
            } => {
                validate_non_negative("momentum", momentum)?;
                validate_non_negative("dampening", dampening)?;
                if nesterov && (momentum <= 0. || dampening != 0.) {
                    return Err(invalid(
                        "nesterov requires positive momentum and zero dampening",
                    ));
                }
                Ok(())
            }
            Self::RmsProp {
                alpha,
                eps,
                momentum,
                ..
            } => {
                validate_non_negative("alpha", alpha)?;
                validate_non_negative("eps", eps)?;
                validate_non_negative("momentum", momentum)
            }
            Self::Adagrad {
                lr_decay,
                initial_accumulator,
                eps,
            } => {
                validate_non_negative("lr_decay", lr_decay)?;
                validate_non_negative("initial_accumulator", initial_accumulator)?;
                validate_non_negative("eps", eps)
            }
            Self::Adadelta { rho, eps } => {
                if !rho.is_finite() || !(0. ..=1.).contains(&rho) {
                    return Err(invalid("rho must be finite and in [0, 1]"));
                }
                validate_non_negative("eps", eps)
            }
        }
    }

    fn slots(&self) -> &'static [&'static str] {
        match self {
            Self::Adam { amsgrad: true, .. } => &["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
            Self::Adam { .. } => &["exp_avg", "exp_avg_sq"],
            Self::Sgd { momentum, .. } if *momentum != 0. => &["momentum_buffer"],
            Self::Sgd { .. } => &[],
            Self::RmsProp {
                centered: true,
                momentum,
                ..
            } if *momentum != 0. => &["square_avg", "grad_avg", "momentum_buffer"],
            Self::RmsProp { centered: true, .. } => &["square_avg", "grad_avg"],
            Self::RmsProp { momentum, .. } if *momentum != 0. => &["square_avg", "momentum_buffer"],
            Self::RmsProp { .. } => &["square_avg"],
            Self::Adagrad { .. } => &["sum"],
            Self::Adadelta { .. } => &["square_avg", "acc_delta"],
            Self::Adamax { .. } => &["exp_avg", "exp_inf"],
        }
    }

    fn same_family(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && match (self, other) {
                (Self::Adam { decoupled: a, .. }, Self::Adam { decoupled: b, .. }) => a == b,
                _ => true,
            }
    }
}

/// Learning rate and weight decay for a named parameter group.
///
/// A group's identifier comes from `VarStore::root().set_group(id)`. Read these
/// settings with [`Optimizer::parameter_groups`] and change them through the
/// validated optimizer setters. The returned value is an independent snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterGroup {
    /// Identifier assigned by the parameter store's path.
    pub id: usize,
    /// Nonnegative finite learning rate.
    pub learning_rate: f64,
    /// Nonnegative finite weight decay.
    pub weight_decay: f64,
    /// Sorted names of parameters belonging to this group.
    pub parameters: Vec<String>,
}

#[derive(Debug)]
struct Parameter {
    tensor: Tensor,
    group: usize,
    step: u64,
    slots: BTreeMap<String, Tensor>,
}

/// Optimizer configuration, group settings and exact named moment tensors.
///
/// This versioned serde value contains no executable code or pickle. Tensor
/// payloads retain their dtype and little-endian bytes, including nonfinite
/// values. Model weights and accumulated gradients are **not** included: save
/// model weights alongside this state after a completed training step.
///
/// ```no_run
/// # fn example(optimizer: &mut rusttorch::optim::Optimizer) -> Result<(), Box<dyn std::error::Error>> {
/// let state = optimizer.state_dict()?;
/// let json = serde_json::to_vec(&state)?;
/// let decoded: rusttorch::optim::OptimizerState = serde_json::from_slice(&json)?;
/// optimizer.load_state_dict(&decoded)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizerState {
    schema_version: u32,
    algorithm: Algorithm,
    default_learning_rate: f64,
    default_weight_decay: f64,
    completed_steps: u64,
    groups: Vec<ParameterGroup>,
    parameters: BTreeMap<String, ParameterState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterState {
    shape: Vec<i64>,
    dtype: Dtype,
    group: usize,
    step: u64,
    slots: BTreeMap<String, TensorState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Dtype {
    Float32,
    Float64,
    Float16,
    BFloat16,
}

impl Dtype {
    fn from_kind(kind: Kind) -> Result<Self> {
        match kind {
            Kind::Float => Ok(Self::Float32),
            Kind::Double => Ok(Self::Float64),
            Kind::Half => Ok(Self::Float16),
            Kind::BFloat16 => Ok(Self::BFloat16),
            _ => Err(invalid(
                "optimizer parameters must use Float, Double, Half or BFloat16",
            )),
        }
    }
    fn kind(self) -> Kind {
        match self {
            Self::Float32 => Kind::Float,
            Self::Float64 => Kind::Double,
            Self::Float16 => Kind::Half,
            Self::BFloat16 => Kind::BFloat16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorState {
    shape: Vec<i64>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

impl TensorState {
    fn capture(tensor: &Tensor) -> Result<Self> {
        let dtype = Dtype::from_kind(tensor.kind())?;
        let size = tensor
            .numel()
            .checked_mul(dtype.kind().elt_size_in_bytes())
            .ok_or_else(|| invalid("tensor byte count overflow"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| invalid("cannot reserve checkpoint payload"))?;
        bytes.resize(size, 0);
        tensor
            .f_to_device(Device::Cpu)?
            .f_contiguous()?
            .f_copy_data_u8(&mut bytes, tensor.numel())?;
        if cfg!(target_endian = "big") {
            swap_elements(&mut bytes, dtype.kind().elt_size_in_bytes());
        }
        Ok(Self {
            shape: tensor.size(),
            dtype,
            bytes,
        })
    }
    fn validate(&self, shape: &[i64], dtype: Dtype) -> Result<()> {
        if self.shape != shape || self.dtype != dtype {
            return Err(invalid("moment shape or dtype differs from its parameter"));
        }
        let numel = self
            .shape
            .iter()
            .try_fold(1usize, |n, &dimension| {
                usize::try_from(dimension)
                    .ok()
                    .and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| invalid("invalid checkpoint dimensions"))?;
        if numel.checked_mul(dtype.kind().elt_size_in_bytes()) != Some(self.bytes.len()) {
            return Err(invalid(
                "checkpoint byte count does not match shape and dtype",
            ));
        }
        Ok(())
    }
    fn restore(&self, device: Device) -> Result<Tensor> {
        let mut bytes = self.bytes.clone();
        if cfg!(target_endian = "big") {
            swap_elements(&mut bytes, self.dtype.kind().elt_size_in_bytes());
        }
        Ok(
            Tensor::f_from_data_size(&bytes, &self.shape, self.dtype.kind())?
                .f_to_device(device)?,
        )
    }
}

fn swap_elements(bytes: &mut [u8], width: usize) {
    for element in bytes.chunks_exact_mut(width) {
        element.reverse();
    }
}

/// Updates named model parameters while retaining checkpointable training state.
///
/// Build after registering your initial parameters. Later registrations are
/// discovered before a step, gradient reset or checkpoint. Dense Float/Double/
/// Half/BFloat16 parameters are supported; plain SGD also accepts sparse COO
/// gradients. Advanced fused, differentiable and graph-captured updates are
/// outside this implementation. Use [`Self::try_step`] for recoverable errors.
#[derive(Debug)]
pub struct Optimizer {
    variables: Arc<Mutex<Variables>>,
    parameters: BTreeMap<String, Parameter>,
    groups: BTreeMap<usize, ParameterGroup>,
    algorithm: Algorithm,
    default_learning_rate: f64,
    default_weight_decay: f64,
    completed_steps: u64,
    // tch does not expose a safe gradient-to-None setter. A zero-state native
    // optimizer is used only for zero_grad, never for parameter updates.
    gradient_clear: COptimizer,
    id: u64,
}

impl Optimizer {
    pub(super) fn new(store: &VarStore, algorithm: Algorithm, lr: f64, wd: f64) -> Result<Self> {
        algorithm.validate()?;
        validate_non_negative("learning_rate", lr)?;
        validate_non_negative("weight_decay", wd)?;
        let mut optimizer = Self {
            variables: store.variables_.clone(),
            parameters: BTreeMap::new(),
            groups: BTreeMap::new(),
            algorithm,
            default_learning_rate: lr,
            default_weight_decay: wd,
            completed_steps: 0,
            gradient_clear: COptimizer::sgd(0., 0., 0., 0., false)?,
            id: NEXT_OPTIMIZER_ID.fetch_add(1, Ordering::Relaxed),
        };
        optimizer.sync_parameters()?;
        Ok(optimizer)
    }

    pub(crate) fn identity(&self) -> u64 {
        self.id
    }

    fn sync_parameters(&mut self) -> Result<()> {
        let variables = self
            .variables
            .lock()
            .map_err(|_| invalid("parameter store lock is poisoned"))?;
        if variables.trainable_variables.len() == self.parameters.len() {
            return Ok(());
        }
        if variables.trainable_variables.len() < self.parameters.len() {
            return Err(invalid("registered parameters were removed"));
        }
        // ponytail: O(parameters²) only when registration changes; index tensor
        // identity if model-construction profiling shows this lookup matters.
        let mut additions = Vec::new();
        for variable in &variables.trainable_variables {
            if !variable.tensor.defined() || variable.tensor.is_sparse() {
                return Err(invalid("parameters must be defined dense tensors"));
            }
            Dtype::from_kind(variable.tensor.kind())?;
            let mut names = variables
                .named_variables
                .iter()
                .filter_map(
                    |(name, tensor)| match tensor.f_is_set_to(&variable.tensor) {
                        Ok(true) => Some(Ok(name)),
                        Ok(false) => None,
                        Err(error) => Some(Err(error)),
                    },
                )
                .collect::<std::result::Result<Vec<_>, _>>()?;
            names.sort();
            let name = names
                .first()
                .ok_or_else(|| invalid("trainable parameter has no registered name"))?;
            if names.len() != 1 {
                return Err(invalid(
                    "aliased parameter registrations require one canonical name",
                ));
            }
            if let Some(current) = self.parameters.get(*name) {
                if current.group != variable.group
                    || !current.tensor.f_is_set_to(&variable.tensor)?
                {
                    return Err(invalid(
                        "parameter identity or group changed after registration",
                    ));
                }
            } else {
                if additions.iter().any(|(prior, _, _)| prior == *name) {
                    return Err(invalid("duplicate trainable parameter registration"));
                }
                additions.push((
                    (*name).clone(),
                    variable.group,
                    variable.tensor.shallow_clone(),
                ));
            }
        }
        for (name, group, tensor) in additions {
            self.gradient_clear.add_parameters(&tensor, 0)?;
            self.groups
                .entry(group)
                .or_insert_with(|| ParameterGroup {
                    id: group,
                    learning_rate: self.default_learning_rate,
                    weight_decay: self.default_weight_decay,
                    parameters: Vec::new(),
                })
                .parameters
                .push(name.clone());
            self.parameters.insert(
                name,
                Parameter {
                    tensor,
                    group,
                    step: 0,
                    slots: BTreeMap::new(),
                },
            );
        }
        for group in self.groups.values_mut() {
            group.parameters.sort();
        }
        Ok(())
    }

    /// Returns shallow handles to all currently registered trainable parameters.
    ///
    /// Inspect `parameter.grad()` to check or unscale gradients before stepping.
    /// Frozen parameters remain registered, matching the parameter store.
    pub fn trainable_variables(&self) -> Vec<Tensor> {
        self.variables
            .lock()
            .expect("parameter store lock is poisoned")
            .trainable_variables
            .iter()
            .map(|p| p.tensor.shallow_clone())
            .collect()
    }

    /// Returns tracked parameter names in stable lexical order and shallow tensors.
    ///
    /// Newly registered parameters become named here after the next step,
    /// gradient reset, or checkpoint synchronization.
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.parameters
            .iter()
            .map(|(name, p)| (name.clone(), p.tensor.shallow_clone()))
            .collect()
    }

    /// Returns independent snapshots of group settings, sorted by group identifier.
    pub fn parameter_groups(&self) -> Vec<ParameterGroup> {
        self.groups.values().cloned().collect()
    }

    pub(super) fn apply_group_rates(&mut self, rates: &[(usize, f64)]) -> Result<()> {
        self.sync_parameters()?;
        if rates.len() != self.groups.len()
            || rates.iter().map(|(id, _)| *id).collect::<Vec<_>>()
                != self.groups.keys().copied().collect::<Vec<_>>()
        {
            return Err(invalid("scheduler parameter groups changed"));
        }
        for (_, value) in rates {
            validate_non_negative("learning_rate", *value)?;
        }
        for (id, value) in rates {
            self.groups
                .get_mut(id)
                .expect("validated group")
                .learning_rate = *value;
        }
        Ok(())
    }

    /// Sets every group's learning rate, also selecting the default for future groups.
    pub fn set_learning_rate(&mut self, value: f64) -> Result<()> {
        validate_non_negative("learning_rate", value)?;
        self.sync_parameters()?;
        for group in self.groups.values_mut() {
            group.learning_rate = value;
        }
        self.default_learning_rate = value;
        Ok(())
    }

    /// Sets one registered group's learning rate without resetting moments.
    ///
    /// ```
    /// # fn example() -> rusttorch::Result<()> {
    /// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let _layer = rusttorch::nn::LinearConfig::new(2, 1).build(&store.root().set_group(3))?;
    /// let mut optimizer = rusttorch::optim::AdamW::builder().build(&store)?;
    /// optimizer.set_group_learning_rate(3, 0.0001)?;
    /// # Ok(()) }
    /// ```
    pub fn set_group_learning_rate(&mut self, id: usize, value: f64) -> Result<()> {
        validate_non_negative("learning_rate", value)?;
        self.sync_parameters()?;
        self.groups
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown parameter group"))?
            .learning_rate = value;
        Ok(())
    }

    /// Sets one registered group's weight decay; zero excludes it from regularization.
    pub fn set_group_weight_decay(&mut self, id: usize, value: f64) -> Result<()> {
        validate_non_negative("weight_decay", value)?;
        self.sync_parameters()?;
        self.groups
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown parameter group"))?
            .weight_decay = value;
        Ok(())
    }

    /// Clears gradients to undefined, so untouched parameters skip momentum and decay.
    pub fn try_zero_grad(&mut self) -> Result<()> {
        self.sync_parameters()?;
        self.gradient_clear.zero_grad()?;
        Ok(())
    }

    /// Clears gradients; panics on backend failure. Prefer [`Self::try_zero_grad`].
    pub fn zero_grad(&mut self) {
        self.try_zero_grad().expect("optimizer zero_grad failed");
    }

    /// Applies one update, panicking on failure. Prefer [`Self::try_step`] in applications.
    pub fn step(&mut self) {
        self.try_step().expect("optimizer step failed");
    }

    /// Validates every gradient and stages all updates before changing parameters.
    ///
    /// Undefined gradients are skipped. Shape, dtype, device, sparse-layout and
    /// counter errors return before any parameter or moment changes. Backend
    /// failures while staging also leave them untouched; a device failure during
    /// the final tensor copies may still leave a partially copied model.
    pub fn try_step(&mut self) -> Result<()> {
        self.sync_parameters()?;
        let next_step = self
            .completed_steps
            .checked_add(1)
            .filter(|step| *step < u64::MAX)
            .ok_or_else(|| invalid("optimizer step counter overflow"))?;
        for parameter in self.parameters.values() {
            let gradient = parameter.tensor.f_grad()?;
            if !gradient.defined() {
                continue;
            }
            validate_gradient(
                parameter,
                &gradient,
                &self.algorithm,
                &self.groups[&parameter.group],
            )?;
        }
        no_grad(|| -> Result<()> {
            let mut updates = Vec::new();
            for (name, parameter) in &self.parameters {
                let gradient = parameter.tensor.f_grad()?;
                if gradient.defined() {
                    let step = parameter
                        .step
                        .checked_add(1)
                        .filter(|step| *step < u64::MAX)
                        .ok_or_else(|| invalid("parameter step counter overflow"))?;
                    let (value, slots) = update(
                        &self.algorithm,
                        parameter,
                        gradient,
                        &self.groups[&parameter.group],
                        step,
                    )?;
                    updates.push((name.clone(), value, slots, step));
                }
            }
            for (name, value, slots, step) in updates {
                let parameter = self
                    .parameters
                    .get_mut(&name)
                    .expect("staged parameter exists");
                parameter.tensor.f_copy_(&value)?;
                parameter.slots = slots;
                parameter.step = step;
            }
            self.completed_steps = next_step;
            Ok(())
        })
    }

    /// Clears gradients, differentiates a defined scalar loss, and applies a fallible update.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        validate_loss(loss)?;
        self.try_zero_grad()?;
        loss.f_backward()?;
        self.try_step()
    }

    /// Clips the combined L2 norm of dense gradients before the next step.
    /// Returns an error for nonfinite or negative bounds and unsupported sparse gradients.
    pub fn clip_grad_norm(&self, max: f64) -> Result<()> {
        validate_non_negative("max_grad_norm", max)?;
        let gradients = self.dense_gradients()?;
        if gradients.is_empty() {
            return Ok(());
        }
        let norms = gradients
            .iter()
            .map(Tensor::f_norm)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let total = Tensor::f_stack(&norms, 0)?.f_norm()?.f_double_value(&[])?;
        let coefficient = max / (total + 1e-6);
        no_grad(|| -> Result<()> {
            if coefficient < 1. {
                for mut gradient in gradients {
                    let _ = gradient.f_mul_scalar_(coefficient)?;
                }
            }
            Ok(())
        })
    }

    /// Clamps dense gradient elements into `[-max, max]` before the next step.
    pub fn clip_grad_value(&self, max: f64) -> Result<()> {
        validate_non_negative("max_grad_value", max)?;
        let gradients = self.dense_gradients()?;
        no_grad(|| -> Result<()> {
            for mut gradient in gradients {
                let _ = gradient.f_clamp_(-max, max)?;
            }
            Ok(())
        })
    }

    fn dense_gradients(&self) -> Result<Vec<Tensor>> {
        let variables = self
            .variables
            .lock()
            .map_err(|_| invalid("parameter store lock is poisoned"))?;
        let mut gradients = Vec::new();
        for parameter in &variables.trainable_variables {
            let gradient = parameter.tensor.f_grad()?;
            if gradient.defined() {
                if gradient.is_sparse() {
                    return Err(invalid("gradient clipping requires dense gradients"));
                }
                gradients.push(gradient);
            }
        }
        Ok(gradients)
    }

    /// Captures exact named moments and validated configuration, without model weights.
    ///
    /// Save weights and this serde state at the same completed training step.
    /// Gradients are not captured; checkpoints belong between optimizer steps.
    pub fn state_dict(&mut self) -> Result<OptimizerState> {
        self.sync_parameters()?;
        for parameter in self.parameters.values() {
            validate_slots(parameter)?;
        }
        let mut parameters = BTreeMap::new();
        for (name, p) in &self.parameters {
            let mut slots = BTreeMap::new();
            for (slot, tensor) in &p.slots {
                slots.insert(slot.clone(), TensorState::capture(tensor)?);
            }
            parameters.insert(
                name.clone(),
                ParameterState {
                    shape: p.tensor.size(),
                    dtype: Dtype::from_kind(p.tensor.kind())?,
                    group: p.group,
                    step: p.step,
                    slots,
                },
            );
        }
        Ok(OptimizerState {
            schema_version: SCHEMA,
            algorithm: self.algorithm.clone(),
            default_learning_rate: self.default_learning_rate,
            default_weight_decay: self.default_weight_decay,
            completed_steps: self.completed_steps,
            groups: self.parameter_groups(),
            parameters,
        })
    }

    /// Validates the entire checkpoint, prepares all moment tensors, then restores state.
    ///
    /// Parameter names, groups, shapes and dtypes must match the current model.
    /// The algorithm family must match; saved hyperparameters and rates replace
    /// the builder settings. Device placement follows the current model.
    /// Validation or allocation failure changes no moments, settings or weights.
    pub fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        self.sync_parameters()?;
        if state.schema_version != SCHEMA || !self.algorithm.same_family(&state.algorithm) {
            return Err(invalid("optimizer checkpoint schema or algorithm mismatch"));
        }
        state.algorithm.validate()?;
        validate_non_negative("learning_rate", state.default_learning_rate)?;
        validate_non_negative("weight_decay", state.default_weight_decay)?;
        if state.completed_steps == u64::MAX {
            return Err(invalid("optimizer step counter cannot advance"));
        }
        let mut groups = BTreeMap::new();
        for group in &state.groups {
            validate_non_negative("learning_rate", group.learning_rate)?;
            validate_non_negative("weight_decay", group.weight_decay)?;
            let current = self
                .groups
                .get(&group.id)
                .ok_or_else(|| invalid("checkpoint has unknown parameter group"))?;
            if current.parameters != group.parameters
                || groups.insert(group.id, group.clone()).is_some()
            {
                return Err(invalid(
                    "checkpoint group membership differs or has duplicates",
                ));
            }
        }
        if groups.len() != self.groups.len() || state.parameters.len() != self.parameters.len() {
            return Err(invalid("checkpoint parameter/group count mismatch"));
        }
        for (name, parameter) in &self.parameters {
            let saved = state
                .parameters
                .get(name)
                .ok_or_else(|| invalid("checkpoint parameter name mismatch"))?;
            let dtype = Dtype::from_kind(parameter.tensor.kind())?;
            if saved.shape != parameter.tensor.size()
                || saved.dtype != dtype
                || saved.group != parameter.group
            {
                return Err(invalid(
                    "checkpoint parameter shape, dtype or group mismatch",
                ));
            }
            if saved.step > state.completed_steps || saved.step == u64::MAX {
                return Err(invalid("invalid parameter step counter"));
            }
            let expected: BTreeSet<_> = if saved.step == 0 {
                BTreeSet::new()
            } else {
                state.algorithm.slots().iter().copied().collect()
            };
            if saved
                .slots
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                != expected
            {
                return Err(invalid("checkpoint moment names or cardinality mismatch"));
            }
            for tensor in saved.slots.values() {
                tensor.validate(&saved.shape, dtype)?;
            }
        }
        let mut restored = BTreeMap::new();
        for (name, saved) in &state.parameters {
            let device = self.parameters[name].tensor.device();
            let mut slots = BTreeMap::new();
            for (slot, tensor) in &saved.slots {
                slots.insert(slot.clone(), tensor.restore(device)?);
            }
            restored.insert(name.clone(), slots);
        }
        for (name, slots) in restored {
            let parameter = self
                .parameters
                .get_mut(&name)
                .expect("validated checkpoint name");
            parameter.slots = slots;
            parameter.step = state.parameters[&name].step;
        }
        self.groups = groups;
        self.algorithm = state.algorithm.clone();
        self.default_learning_rate = state.default_learning_rate;
        self.default_weight_decay = state.default_weight_decay;
        self.completed_steps = state.completed_steps;
        Ok(())
    }
}

fn validate_gradient(
    p: &Parameter,
    gradient: &Tensor,
    algorithm: &Algorithm,
    group: &ParameterGroup,
) -> Result<()> {
    Dtype::from_kind(p.tensor.kind())?;
    if gradient.size() != p.tensor.size()
        || gradient.kind() != p.tensor.kind()
        || gradient.device() != p.tensor.device()
    {
        return Err(invalid(
            "gradient shape, dtype or device differs from its parameter",
        ));
    }
    if gradient.is_sparse()
        && !matches!(algorithm, Algorithm::Sgd { momentum: 0., .. } if group.weight_decay == 0.)
    {
        return Err(invalid(
            "sparse gradients require SGD without momentum or weight decay",
        ));
    }
    if !gradient.is_sparse() {
        // Compressed sparse layouts cannot report contiguous storage. Keep
        // their rejection in the validation pass, before staging any update.
        let _ = gradient.f_contiguous()?;
    }
    validate_slots(p)
}

fn validate_slots(p: &Parameter) -> Result<()> {
    if !p.tensor.defined() {
        return Err(invalid("parameter is undefined"));
    }
    Dtype::from_kind(p.tensor.kind())?;
    for slot in p.slots.values() {
        if slot.size() != p.tensor.size()
            || slot.kind() != p.tensor.kind()
            || slot.device() != p.tensor.device()
        {
            return Err(invalid(
                "parameter was moved or reshaped without matching optimizer state",
            ));
        }
    }
    Ok(())
}

fn initial(p: &Parameter, name: &str, value: f64) -> Result<Tensor> {
    Ok(match p.slots.get(name) {
        Some(tensor) => tensor.shallow_clone(),
        None => p.tensor.f_zeros_like()?.f_add_scalar(value)?,
    })
}

fn update(
    algorithm: &Algorithm,
    p: &Parameter,
    mut gradient: Tensor,
    group: &ParameterGroup,
    step: u64,
) -> Result<(Tensor, BTreeMap<String, Tensor>)> {
    // Recurrences follow PyTorch v2.13.0 (cf30153), torch/optim/{adam,
    // adamw,sgd,rmsprop,adagrad,adadelta,adamax}.py. Operations are expressed
    // independently using fallible tch tensors; see THIRD_PARTY_NOTICES.md.
    let mut value = p.tensor.shallow_clone();
    let mut slots = BTreeMap::new();
    let lr = group.learning_rate;
    let decay = group.weight_decay;
    if decay != 0. {
        if matches!(
            algorithm,
            Algorithm::Adam {
                decoupled: true,
                ..
            }
        ) {
            value = value.f_mul_scalar(1. - lr * decay)?;
        } else {
            gradient = gradient.f_add(&value.f_mul_scalar(decay)?)?;
        }
    }
    let direction = match *algorithm {
        Algorithm::Sgd {
            momentum,
            dampening,
            nesterov,
        } => {
            if momentum == 0. {
                gradient
            } else {
                let buffer = if p.step == 0 {
                    gradient.f_detach_copy()?
                } else {
                    initial(p, "momentum_buffer", 0.)?
                        .f_mul_scalar(momentum)?
                        .f_add(&gradient.f_mul_scalar(1. - dampening)?)?
                };
                let direction = if nesterov {
                    gradient.f_add(&buffer.f_mul_scalar(momentum)?)?
                } else {
                    buffer.shallow_clone()
                };
                slots.insert("momentum_buffer".into(), buffer);
                direction
            }
        }
        Algorithm::Adam {
            beta1,
            beta2,
            eps,
            amsgrad,
            ..
        } => {
            let m = initial(p, "exp_avg", 0.)?.f_lerp(&gradient, 1. - beta1)?;
            let v = initial(p, "exp_avg_sq", 0.)?
                .f_mul_scalar(beta2)?
                .f_add(&gradient.f_square()?.f_mul_scalar(1. - beta2)?)?;
            let mut denominator = v.shallow_clone();
            if amsgrad {
                denominator = initial(p, "max_exp_avg_sq", 0.)?.f_maximum(&v)?;
                slots.insert("max_exp_avg_sq".into(), denominator.shallow_clone());
            }
            denominator = denominator
                .f_sqrt()?
                .f_div_scalar((1. - beta2.powf(step as f64)).sqrt())?
                .f_add_scalar(eps)?;
            let direction = m
                .f_div(&denominator)?
                .f_div_scalar(1. - beta1.powf(step as f64))?;
            slots.insert("exp_avg".into(), m);
            slots.insert("exp_avg_sq".into(), v);
            direction
        }
        Algorithm::RmsProp {
            alpha,
            eps,
            momentum,
            centered,
        } => {
            let square = initial(p, "square_avg", 0.)?
                .f_mul_scalar(alpha)?
                .f_add(&gradient.f_square()?.f_mul_scalar(1. - alpha)?)?;
            let mut avg = square.shallow_clone();
            if centered {
                let mean = initial(p, "grad_avg", 0.)?.f_lerp(&gradient, 1. - alpha)?;
                avg = avg.f_sub(&mean.f_square()?)?;
                slots.insert("grad_avg".into(), mean);
            }
            let direction = gradient.f_div(&avg.f_sqrt()?.f_add_scalar(eps)?)?;
            slots.insert("square_avg".into(), square);
            if momentum != 0. {
                let buffer = initial(p, "momentum_buffer", 0.)?
                    .f_mul_scalar(momentum)?
                    .f_add(&direction)?;
                slots.insert("momentum_buffer".into(), buffer.shallow_clone());
                buffer
            } else {
                direction
            }
        }
        Algorithm::Adagrad {
            lr_decay,
            initial_accumulator,
            eps,
        } => {
            let sum = initial(p, "sum", initial_accumulator)?.f_add(&gradient.f_square()?)?;
            let direction = gradient
                .f_div(&sum.f_sqrt()?.f_add_scalar(eps)?)?
                .f_div_scalar(1. + (step - 1) as f64 * lr_decay)?;
            slots.insert("sum".into(), sum);
            direction
        }
        Algorithm::Adadelta { rho, eps } => {
            let square = initial(p, "square_avg", 0.)?
                .f_mul_scalar(rho)?
                .f_add(&gradient.f_square()?.f_mul_scalar(1. - rho)?)?;
            let acc = initial(p, "acc_delta", 0.)?;
            let delta = acc
                .f_add_scalar(eps)?
                .f_sqrt()?
                .f_div(&square.f_add_scalar(eps)?.f_sqrt()?)?
                .f_mul(&gradient)?;
            slots.insert(
                "acc_delta".into(),
                acc.f_mul_scalar(rho)?
                    .f_add(&delta.f_square()?.f_mul_scalar(1. - rho)?)?,
            );
            slots.insert("square_avg".into(), square);
            delta
        }
        Algorithm::Adamax { beta1, beta2, eps } => {
            let m = initial(p, "exp_avg", 0.)?.f_lerp(&gradient, 1. - beta1)?;
            let infinity = initial(p, "exp_inf", 0.)?
                .f_mul_scalar(beta2)?
                .f_maximum(&gradient.f_abs()?.f_add_scalar(eps)?)?;
            let direction = m
                .f_div(&infinity)?
                .f_div_scalar(1. - beta1.powf(step as f64))?;
            slots.insert("exp_avg".into(), m);
            slots.insert("exp_inf".into(), infinity);
            direction
        }
    };
    Ok((value.f_add(&direction.f_mul_scalar(-lr)?)?, slots))
}

pub(super) fn invalid(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "optimizer state",
        reason: reason.to_owned(),
    }
}
