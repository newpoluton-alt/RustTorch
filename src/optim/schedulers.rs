//! Learning-rate schedules with serde state for exact epoch-boundary replay.
// Numerical schedules follow PyTorch v2.13.0 (cf30153),
// torch/optim/lr_scheduler.py; see THIRD_PARTY_NOTICES.md.

use super::{Optimizer, stateful::invalid, validate_non_negative};
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleState {
    schema_version: u32,
    epoch: u64,
    group_ids: Vec<usize>,
    base_rates: Vec<f64>,
    last_rates: Vec<f64>,
}

impl ScheduleState {
    fn new(optimizer: &Optimizer) -> Result<Self> {
        let groups = optimizer.parameter_groups();
        if groups.is_empty() {
            return Err(invalid("a scheduler needs at least one parameter group"));
        }
        let rates: Vec<_> = groups.iter().map(|g| g.learning_rate).collect();
        Ok(Self {
            schema_version: 1,
            epoch: 0,
            group_ids: groups.iter().map(|g| g.id).collect(),
            base_rates: rates.clone(),
            last_rates: rates,
        })
    }
    fn current(&self, optimizer: &Optimizer) -> Result<Vec<f64>> {
        if self.schema_version != 1
            || self.epoch == u64::MAX
            || self.group_ids.is_empty()
            || self.group_ids.len() != self.base_rates.len()
            || self.group_ids.len() != self.last_rates.len()
            || self.group_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("invalid scheduler state or epoch counter"));
        }
        for &rate in self.base_rates.iter().chain(&self.last_rates) {
            validate_non_negative("learning_rate", rate)?;
        }
        let groups = optimizer.parameter_groups();
        if groups.iter().map(|g| g.id).collect::<Vec<_>>() != self.group_ids {
            return Err(invalid("scheduler parameter group identities changed"));
        }
        Ok(groups.into_iter().map(|g| g.learning_rate).collect())
    }
    fn apply(&mut self, optimizer: &mut Optimizer, rates: Vec<f64>, epoch: u64) -> Result<()> {
        let rates_by_group = self
            .group_ids
            .iter()
            .copied()
            .zip(rates.iter().copied())
            .collect::<Vec<_>>();
        optimizer.apply_group_rates(&rates_by_group)?;
        self.last_rates = rates;
        self.epoch = epoch;
        Ok(())
    }
}

macro_rules! accessors {
    () => {
        /// Returns the number of completed calls to `step`.
        pub const fn epoch(&self) -> u64 {
            self.state.epoch
        }
        /// Returns the most recently applied rates in ascending parameter-group order.
        pub fn last_learning_rates(&self) -> &[f64] {
            &self.state.last_rates
        }
    };
}

/// Multiplies every learning rate by `gamma` after each `step_size` epochs.
///
/// Construct before training and call `step` after the epoch's optimizer updates.
/// Serialize the scheduler itself with serde alongside [`super::OptimizerState`].
/// Restoring both values preserves the next epoch's rates. Changing the set of
/// parameter groups after construction is rejected.
///
/// ```
/// # fn example(mut optimizer: rusttorch::optim::Optimizer) -> rusttorch::Result<()> {
/// let mut schedule = rusttorch::optim::StepLr::new(&optimizer, 10, 0.1)?;
/// // After each epoch's optimizer updates:
/// schedule.step(&mut optimizer)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepLr {
    state: ScheduleState,
    step_size: u64,
    gamma: f64,
}
impl StepLr {
    /// Creates an epoch-zero schedule with a positive interval and nonnegative finite factor.
    pub fn new(optimizer: &Optimizer, step_size: u64, gamma: f64) -> Result<Self> {
        positive(step_size, "step_size")?;
        validate_non_negative("gamma", gamma)?;
        Ok(Self {
            state: ScheduleState::new(optimizer)?,
            step_size,
            gamma,
        })
    }
    /// Advances one epoch and updates all groups without changing optimizer moments.
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<()> {
        positive(self.step_size, "step_size")?;
        validate_non_negative("gamma", self.gamma)?;
        let mut rates = self.state.current(optimizer)?;
        let epoch = self.state.epoch + 1;
        if epoch.is_multiple_of(self.step_size) {
            for rate in &mut rates {
                *rate *= self.gamma;
            }
        }
        self.state.apply(optimizer, rates, epoch)
    }
    accessors!();
}

/// Multiplies each group's current learning rate by `gamma` after every epoch.
///
/// ```
/// # fn example(mut optimizer: rusttorch::optim::Optimizer) -> rusttorch::Result<()> {
/// let mut schedule = rusttorch::optim::ExponentialLr::new(&optimizer, 0.95)?;
/// schedule.step(&mut optimizer)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExponentialLr {
    state: ScheduleState,
    gamma: f64,
}
impl ExponentialLr {
    /// Creates an epoch-zero schedule; `gamma` must be finite and nonnegative.
    pub fn new(optimizer: &Optimizer, gamma: f64) -> Result<Self> {
        validate_non_negative("gamma", gamma)?;
        Ok(Self {
            state: ScheduleState::new(optimizer)?,
            gamma,
        })
    }
    /// Advances one epoch. Serialize this value to retain the epoch and last rates.
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<()> {
        validate_non_negative("gamma", self.gamma)?;
        let rates = self
            .state
            .current(optimizer)?
            .into_iter()
            .map(|rate| rate * self.gamma)
            .collect();
        self.state.apply(optimizer, rates, self.state.epoch + 1)
    }
    accessors!();
}

/// Decays learning rates at specified epochs; repeated milestones apply repeated decay.
///
/// Milestones are sorted at construction. A zero milestone applies immediately,
/// so this constructor takes a mutable optimizer. Later `step` calls correspond
/// to completed epochs one, two, and onward.
///
/// ```
/// # fn example(mut optimizer: rusttorch::optim::Optimizer) -> rusttorch::Result<()> {
/// let mut schedule = rusttorch::optim::MultiStepLr::new(&mut optimizer, vec![10, 20, 20], 0.1)?;
/// schedule.step(&mut optimizer)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiStepLr {
    state: ScheduleState,
    milestones: Vec<u64>,
    gamma: f64,
}
impl MultiStepLr {
    /// Creates a schedule with finite nonnegative decay; zero milestones apply now.
    pub fn new(optimizer: &mut Optimizer, mut milestones: Vec<u64>, gamma: f64) -> Result<Self> {
        validate_non_negative("gamma", gamma)?;
        milestones.sort_unstable();
        let mut state = ScheduleState::new(optimizer)?;
        let factor = gamma.powf(milestones.iter().filter(|&&m| m == 0).count() as f64);
        state.apply(
            optimizer,
            state.last_rates.iter().map(|rate| rate * factor).collect(),
            0,
        )?;
        Ok(Self {
            state,
            milestones,
            gamma,
        })
    }
    /// Advances one epoch, applying `gamma` once for each matching milestone.
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<()> {
        validate_non_negative("gamma", self.gamma)?;
        if self.milestones.windows(2).any(|p| p[0] > p[1]) {
            return Err(invalid("scheduler milestones must be sorted"));
        }
        let rates = self.state.current(optimizer)?;
        let epoch = self.state.epoch + 1;
        let count = self
            .milestones
            .iter()
            .filter(|&&milestone| milestone == epoch)
            .count();
        let rates = rates
            .into_iter()
            .map(|rate| rate * self.gamma.powf(count as f64))
            .collect();
        self.state.apply(optimizer, rates, epoch)
    }
    accessors!();
}

/// Anneals group learning rates along a cosine curve with minimum `eta_min`.
///
/// `t_max` counts scheduler steps to the first minimum. Continuing past that
/// point follows the cosine recurrence; no optimizer moments are reset.
///
/// ```
/// # fn example(mut optimizer: rusttorch::optim::Optimizer) -> rusttorch::Result<()> {
/// let mut schedule = rusttorch::optim::CosineAnnealingLr::new(&optimizer, 100, 0.00001)?;
/// schedule.step(&mut optimizer)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosineAnnealingLr {
    state: ScheduleState,
    t_max: u64,
    eta_min: f64,
}
impl CosineAnnealingLr {
    /// Creates an epoch-zero schedule with a positive cycle length and nonnegative minimum.
    pub fn new(optimizer: &Optimizer, t_max: u64, eta_min: f64) -> Result<Self> {
        positive(t_max, "t_max")?;
        validate_non_negative("eta_min", eta_min)?;
        if t_max > u64::MAX / 2 {
            return Err(invalid("t_max is too large"));
        }
        Ok(Self {
            state: ScheduleState::new(optimizer)?,
            t_max,
            eta_min,
        })
    }
    /// Advances one epoch using the current group rates and saved initial rates.
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<()> {
        positive(self.t_max, "t_max")?;
        validate_non_negative("eta_min", self.eta_min)?;
        if self.t_max > u64::MAX / 2 {
            return Err(invalid("t_max is too large"));
        }
        let current = self.state.current(optimizer)?;
        let epoch = self.state.epoch + 1;
        let pi = std::f64::consts::PI;
        let at_turn = epoch > self.t_max && (epoch - 1 - self.t_max).is_multiple_of(2 * self.t_max);
        let rates = current
            .iter()
            .zip(&self.state.base_rates)
            .map(|(&rate, &base)| {
                if at_turn {
                    rate + (base - self.eta_min) * (1. - (pi / self.t_max as f64).cos()) / 2.
                } else {
                    (1. + (pi * epoch as f64 / self.t_max as f64).cos())
                        / (1. + (pi * (epoch - 1) as f64 / self.t_max as f64).cos())
                        * (rate - self.eta_min)
                        + self.eta_min
                }
            })
            .collect();
        self.state.apply(optimizer, rates, epoch)
    }
    accessors!();
}

/// Direction in which a validation metric must improve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlateauMode {
    /// A smaller metric is better, as with validation loss.
    Min,
    /// A larger metric is better, as with validation accuracy.
    Max,
}

/// How the plateau scheduler measures a meaningful improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdMode {
    /// Compare with the best metric multiplied by `1 ± threshold`.
    Relative,
    /// Compare with the best metric plus or minus the threshold.
    Absolute,
}

/// Reduces learning rates when a finite validation metric stops improving.
///
/// Defaults: minimize the metric, factor `0.1`, patience `10`, relative threshold
/// `1e-4`, no cooldown, minimum rates zero, and update epsilon `1e-8`.
/// Call after validation. A reduction occurs after **more than** `patience`
/// bad epochs. Serialize this complete value to retain the best metric,
/// patience count, cooldown, per-group minima and current epoch.
///
/// ```
/// # fn example(mut optimizer: rusttorch::optim::Optimizer) -> rusttorch::Result<()> {
/// let mut schedule = rusttorch::optim::ReduceLrOnPlateau::new(&optimizer)?
///     .factor(0.5)?.patience(2);
/// schedule.step(&mut optimizer, 0.25)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReduceLrOnPlateau {
    state: ScheduleState,
    mode: PlateauMode,
    factor: f64,
    patience: u64,
    threshold: f64,
    threshold_mode: ThresholdMode,
    cooldown: u64,
    cooldown_remaining: u64,
    min_rates: Vec<f64>,
    eps: f64,
    best: Option<f64>,
    bad_epochs: u64,
}
impl ReduceLrOnPlateau {
    /// Creates a scheduler with the defaults described on this type.
    pub fn new(optimizer: &Optimizer) -> Result<Self> {
        let state = ScheduleState::new(optimizer)?;
        let min_rates = vec![0.; state.group_ids.len()];
        Ok(Self {
            state,
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Relative,
            cooldown: 0,
            cooldown_remaining: 0,
            min_rates,
            eps: 1e-8,
            best: None,
            bad_epochs: 0,
        })
    }
    /// Selects whether smaller or larger metrics count as improvement.
    #[must_use]
    pub fn mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self
    }
    /// Sets a finite reduction factor in `[0, 1)`.
    pub fn factor(mut self, factor: f64) -> Result<Self> {
        if !factor.is_finite() || !(0. ..1.).contains(&factor) {
            return Err(invalid("plateau factor must be finite and in [0, 1)"));
        }
        self.factor = factor;
        Ok(self)
    }
    /// Sets the number of tolerated bad epochs before reduction.
    #[must_use]
    pub fn patience(mut self, patience: u64) -> Self {
        self.patience = patience;
        self
    }
    /// Sets the nonnegative meaningful-improvement threshold and comparison mode.
    pub fn threshold(mut self, threshold: f64, mode: ThresholdMode) -> Result<Self> {
        validate_non_negative("threshold", threshold)?;
        self.threshold = threshold;
        self.threshold_mode = mode;
        Ok(self)
    }
    /// Sets how many epochs ignore bad metrics following a reduction.
    #[must_use]
    pub fn cooldown(mut self, cooldown: u64) -> Self {
        self.cooldown = cooldown;
        self
    }
    /// Sets per-group minimum rates in the same order as `parameter_groups()`.
    pub fn min_learning_rates(mut self, rates: Vec<f64>) -> Result<Self> {
        if rates.len() != self.state.group_ids.len() {
            return Err(invalid("minimum rate count differs from group count"));
        }
        for &rate in &rates {
            validate_non_negative("min_learning_rate", rate)?;
        }
        self.min_rates = rates;
        Ok(self)
    }
    /// Sets the nonnegative minimum rate change required to apply a reduction.
    pub fn eps(mut self, eps: f64) -> Result<Self> {
        validate_non_negative("eps", eps)?;
        self.eps = eps;
        Ok(self)
    }
    /// Records a finite metric, advances one epoch, and reduces rates when patience expires.
    /// Invalid serialized state, group drift or nonfinite metrics leave rates and state untouched.
    pub fn step(&mut self, optimizer: &mut Optimizer, metric: f64) -> Result<()> {
        let mut rates = self.state.current(optimizer)?;
        if !metric.is_finite()
            || !self.factor.is_finite()
            || !(0. ..1.).contains(&self.factor)
            || self.best.is_some_and(|best| !best.is_finite())
            || self.bad_epochs > self.state.epoch
            || self.cooldown_remaining > self.cooldown
            || self.min_rates.len() != rates.len()
        {
            return Err(invalid("invalid plateau metric or scheduler state"));
        }
        validate_non_negative("threshold", self.threshold)?;
        validate_non_negative("eps", self.eps)?;
        for &rate in &self.min_rates {
            validate_non_negative("min_learning_rate", rate)?;
        }
        let improved = self
            .best
            .is_none_or(|best| match (self.mode, self.threshold_mode) {
                (PlateauMode::Min, ThresholdMode::Relative) => {
                    metric < best * (1. - self.threshold)
                }
                (PlateauMode::Max, ThresholdMode::Relative) => {
                    metric > best * (1. + self.threshold)
                }
                (PlateauMode::Min, ThresholdMode::Absolute) => metric < best - self.threshold,
                (PlateauMode::Max, ThresholdMode::Absolute) => metric > best + self.threshold,
            });
        let best = if improved { Some(metric) } else { self.best };
        let mut bad = if improved { 0 } else { self.bad_epochs + 1 };
        let mut cooldown = self.cooldown_remaining;
        if cooldown > 0 {
            cooldown -= 1;
            bad = 0;
        }
        if bad > self.patience {
            for (rate, &minimum) in rates.iter_mut().zip(&self.min_rates) {
                let next = (*rate * self.factor).max(minimum);
                if *rate - next > self.eps {
                    *rate = next;
                }
            }
            cooldown = self.cooldown;
            bad = 0;
        }
        self.state.apply(optimizer, rates, self.state.epoch + 1)?;
        self.best = best;
        self.bad_epochs = bad;
        self.cooldown_remaining = cooldown;
        Ok(())
    }
    accessors!();
}

fn positive(value: u64, field: &'static str) -> Result<()> {
    if value == 0 {
        Err(crate::RustTorchError::InvalidConfiguration {
            field,
            reason: "must be positive".into(),
        })
    } else {
        Ok(())
    }
}
