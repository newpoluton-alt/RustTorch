//! Mixed-precision execution and safe scaled-gradient updates.
//!
//! CUDA autocast chooses lower-precision kernels while keeping model parameters
//! in their original dtype. A [`GradScaler`] scales the loss, unscales gradients,
//! skips updates containing infinities or NaNs, and adapts its scale. Gradient
//! scaling also works on CPU, which makes the update protocol easy to test.
//!
//! ```
//! use rusttorch::{DeviceSpec, Kind, Tensor, nn::{Sequential, functional}, optim::Sgd};
//! use rusttorch::amp::GradScaler;
//! let model = Sequential::builder().linear(2, 1).build(DeviceSpec::Cpu)?;
//! let mut optimizer = Sgd::builder().build(model.var_store())?;
//! let mut scaler = GradScaler::default();
//! optimizer.try_zero_grad()?;
//! let input = Tensor::ones([2, 2], (Kind::Float, model.device()));
//! let target = Tensor::zeros([2, 1], (Kind::Float, model.device()));
//! let loss = functional::mse_loss(&model.forward(&input)?, &target)?;
//! scaler.scale(&loss)?.f_backward()?;
//! scaler.unscale(&optimizer)?;
//! optimizer.clip_grad_norm(1.0)?;
//! let applied = scaler.step(&mut optimizer)?;
//! scaler.update()?;
//! assert!(applied);
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use serde::{Deserialize, Serialize};

use crate::{
    Device, DeviceSpec, Result, RustTorchError, Tensor, no_grad, optim::Optimizer, resolve_device,
};

/// Runs a closure with the native CUDA autocast policy enabled or disabled.
///
/// On a CPU-only runtime this is a no-op. This API exposes the CUDA policy
/// available through the linked `tch` runtime; it does not select CPU/BFloat16
/// or MPS autocast policies. Use [`autocast_for`] when enabling an unsupported
/// device must be an error. Nesting and previous state are restored even if
/// the closure unwinds.
///
/// ```
/// use rusttorch::{Tensor, amp::autocast};
/// let result = autocast(false, || Tensor::from_slice(&[2_f32]).f_square())?;
/// assert_eq!(result.double_value(&[0]), 4.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn autocast<T>(enabled: bool, operation: impl FnOnce() -> T) -> T {
    // Catch inside the native scope so tch can restore nesting/cache state
    // before the original panic resumes. No unsafe runtime bridge is needed.
    match tch::autocast(enabled, || catch_unwind(AssertUnwindSafe(operation))) {
        Ok(value) => value,
        Err(panic) => resume_unwind(panic),
    }
}

/// Runs a fallible closure with strict CUDA autocast device selection.
///
/// Enabling requires an available CUDA device. CPU and MPS requests return a
/// typed error before the closure runs. Disabled execution runs normally on
/// any requested device without changing its tensors.
///
/// ```
/// use rusttorch::{Device, amp::autocast_for};
/// assert_eq!(autocast_for(Device::Cpu, false, || Ok(42))?, 42);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn autocast_for<T>(
    device: Device,
    enabled: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if enabled {
        match device {
            Device::Cuda(index) => {
                resolve_device(DeviceSpec::Cuda(index))?;
            }
            _ => {
                return Err(invalid(
                    "autocast device",
                    "the linked safe API supports CUDA autocast only",
                ));
            }
        }
    }
    autocast(enabled, operation)
}

/// Scale policy for a [`GradScaler`].
///
/// Defaults are initial scale `65536`, growth factor `2`, backoff factor `0.5`,
/// growth interval `2000` successful steps, and enabled scaling.
///
/// ```
/// let scaler = rusttorch::amp::GradScalerConfig { growth_interval: 100,
///     ..Default::default() }.build()?;
/// assert_eq!(scaler.get_scale(), 65536.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradScalerConfig {
    /// Positive finite initial scale within normal `f32` magnitudes.
    pub initial_scale: f64,
    /// Finite multiplier greater than one applied after successful intervals.
    pub growth_factor: f64,
    /// Finite multiplier strictly between zero and one after a skipped update.
    pub backoff_factor: f64,
    /// Positive number of consecutive successful updates before growth.
    pub growth_interval: u64,
    /// Disable scaling while preserving the same call sequence.
    pub enabled: bool,
}

impl Default for GradScalerConfig {
    fn default() -> Self {
        Self {
            initial_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            enabled: true,
        }
    }
}

impl GradScalerConfig {
    /// Validates the policy and creates a scaler at a fresh step boundary.
    pub fn build(self) -> Result<GradScaler> {
        self.validate()?;
        Ok(GradScaler {
            config: self,
            scale: if self.enabled {
                (self.initial_scale as f32) as f64
            } else {
                1.0
            },
            growth_tracker: 0,
            phase: Phase::Ready,
        })
    }

    fn validate(self) -> Result<()> {
        validate_scale(self.initial_scale)?;
        if !self.growth_factor.is_finite() || self.growth_factor <= 1.0 {
            return Err(invalid(
                "growth_factor",
                "must be finite and greater than one",
            ));
        }
        if !self.backoff_factor.is_finite()
            || !(0.0..1.0).contains(&self.backoff_factor)
            || self.backoff_factor == 0.0
        {
            return Err(invalid(
                "backoff_factor",
                "must be finite and strictly between zero and one",
            ));
        }
        if self.growth_interval == 0 {
            return Err(invalid("growth_interval", "must be positive"));
        }
        Ok(())
    }
}

/// Versioned scaler state saved between completed updates.
///
/// Serialize this owned value with serde together with the model, optimizer,
/// scheduler and data-loader states required by your training job. State loaded
/// from storage is validated again by [`GradScaler::load_state_dict`].
///
/// ```
/// let scaler = rusttorch::amp::GradScaler::default();
/// let state = scaler.state_dict()?;
/// assert_eq!(state.schema_version, 1);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradScalerState {
    /// State format version; only version one is accepted.
    pub schema_version: u32,
    /// Policy controlling subsequent scale changes.
    pub config: GradScalerConfig,
    /// Current positive finite scale.
    pub scale: f64,
    /// Consecutive successful updates since the last growth or backoff.
    pub growth_tracker: u64,
}

#[derive(Debug)]
enum Phase {
    Ready,
    Unscaled { optimizer: u64, finite: bool },
    Stepped { applied: bool },
}

/// Coordinates one optimizer update at a time with scaled dense gradients.
///
/// Use `scale(loss) -> backward -> [unscale -> clip] -> step -> update`.
/// Multiple microbatch backward calls are allowed before unscale. Repeated
/// unscale/step calls, switching optimizers midway, and mid-update snapshots
/// return errors. Sparse gradients are rejected before any gradient is changed.
/// Model weights remain separate from scaler state.
#[derive(Debug)]
pub struct GradScaler {
    config: GradScalerConfig,
    scale: f64,
    growth_tracker: u64,
    phase: Phase,
}

impl Default for GradScaler {
    fn default() -> Self {
        GradScalerConfig::default()
            .build()
            .expect("valid default scaler policy")
    }
}

impl GradScaler {
    /// Returns the current loss multiplier, or one when scaling is disabled.
    pub const fn get_scale(&self) -> f64 {
        self.scale
    }

    /// Returns whether loss scaling and nonfinite update skipping are enabled.
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Multiplies a scalar loss without clearing or applying gradients.
    ///
    /// Call repeatedly for microbatch accumulation before unscaling. Undefined
    /// or non-scalar losses and calls during an unfinished update return errors.
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        self.require_ready()?;
        if !loss.defined() || !loss.size().is_empty() {
            return Err(invalid("loss", "scaling requires a defined scalar loss"));
        }
        loss.f_mul_scalar(self.scale).map_err(Into::into)
    }

    /// Unscales all dense gradients once, preparing them for optional clipping.
    ///
    /// A nonfinite gradient marks the update for skipping. Every gradient is
    /// validated before values are copied, so invalid layouts do not partly
    /// unscale the optimizer. At least one defined gradient is required.
    pub fn unscale(&mut self, optimizer: &Optimizer) -> Result<()> {
        self.require_ready()?;
        let gradients: Vec<_> = optimizer
            .trainable_variables()
            .iter()
            .map(Tensor::grad)
            .filter(Tensor::defined)
            .collect();
        if gradients.is_empty() {
            return Err(invalid(
                "gradients",
                "no gradients were produced for this optimizer",
            ));
        }
        if gradients.iter().any(Tensor::is_sparse) {
            return Err(invalid(
                "gradients",
                "gradient scaling currently requires dense gradients",
            ));
        }
        let mut finite = true;
        let mut staged = Vec::with_capacity(gradients.len());
        for gradient in gradients {
            let unscaled = gradient.f_div_scalar(self.scale)?;
            if self.config.enabled && unscaled.f_isfinite()?.f_all()?.f_int64_value(&[])? == 0 {
                finite = false;
            }
            staged.push((gradient, unscaled));
        }
        no_grad(|| -> Result<()> {
            for (mut gradient, unscaled) in staged {
                gradient.f_copy_(&unscaled)?;
            }
            Ok(())
        })?;
        self.phase = Phase::Unscaled {
            optimizer: optimizer.identity(),
            finite,
        };
        Ok(())
    }

    /// Applies the optimizer update if all scaled gradients were finite.
    ///
    /// Returns `true` when parameters were updated and `false` for a skipped
    /// nonfinite update. Automatically unscales when necessary. Always call
    /// [`Self::update`] next, including after a skipped step.
    pub fn step(&mut self, optimizer: &mut Optimizer) -> Result<bool> {
        if matches!(self.phase, Phase::Ready) {
            self.unscale(optimizer)?;
        }
        let Phase::Unscaled {
            optimizer: identity,
            finite,
        } = self.phase
        else {
            return Err(invalid(
                "scaler phase",
                "step already completed; call update before the next step",
            ));
        };
        if identity != optimizer.identity() {
            return Err(invalid(
                "optimizer",
                "must match the optimizer whose gradients were unscaled",
            ));
        }
        if finite {
            optimizer.try_step()?;
        }
        self.phase = Phase::Stepped { applied: finite };
        Ok(finite)
    }

    /// Applies scale growth or backoff and begins the next update cycle.
    ///
    /// Scale arithmetic rounds to `f32` and stays within positive finite normal magnitudes. This guard avoids
    /// storing an infinite or zero multiplier after repeated growth/backoff.
    pub fn update(&mut self) -> Result<()> {
        let Phase::Stepped { applied } = self.phase else {
            return Err(invalid("scaler phase", "update requires a completed step"));
        };
        if self.config.enabled {
            if applied {
                self.growth_tracker += 1;
                if self.growth_tracker == self.config.growth_interval {
                    let next = (self.scale * self.config.growth_factor) as f32;
                    if next.is_finite() {
                        self.scale = next as f64;
                    }
                    self.growth_tracker = 0;
                }
            } else {
                self.scale = ((self.scale * self.config.backoff_factor) as f32)
                    .max(f32::MIN_POSITIVE) as f64;
                self.growth_tracker = 0;
            }
        }
        self.phase = Phase::Ready;
        Ok(())
    }

    /// Returns a versioned snapshot after update, with no optimizer identity data.
    pub fn state_dict(&self) -> Result<GradScalerState> {
        self.require_ready()?;
        Ok(GradScalerState {
            schema_version: 1,
            config: self.config,
            scale: self.scale,
            growth_tracker: self.growth_tracker,
        })
    }

    /// Validates and restores a complete state without partial application.
    pub fn load_state_dict(&mut self, state: &GradScalerState) -> Result<()> {
        self.require_ready()?;
        if state.schema_version != 1 {
            return Err(invalid("scaler state", "unsupported schema version"));
        }
        state.config.validate()?;
        validate_scale(state.scale)?;
        if state.growth_tracker >= state.config.growth_interval
            || (!state.config.enabled && (state.scale != 1.0 || state.growth_tracker != 0))
        {
            return Err(invalid(
                "scaler state",
                "invalid growth tracker or disabled scale",
            ));
        }
        self.config = state.config;
        self.scale = (state.scale as f32) as f64;
        self.growth_tracker = state.growth_tracker;
        Ok(())
    }

    fn require_ready(&self) -> Result<()> {
        if matches!(self.phase, Phase::Ready) {
            Ok(())
        } else {
            Err(invalid(
                "scaler phase",
                "finish step and update before starting another cycle",
            ))
        }
    }
}

fn validate_scale(value: f64) -> Result<()> {
    if !value.is_finite() || value < f32::MIN_POSITIVE as f64 || value > f32::MAX as f64 {
        return Err(invalid(
            "scale",
            "must be within positive finite normal f32 magnitudes",
        ));
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
