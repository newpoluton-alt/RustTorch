//! Update model parameters from gradients during training.
//!
//! Build an optimizer from a model's parameter store, compute a scalar loss,
//! then call [`Optimizer::backward_step`] to clear old gradients, compute new
//! ones, and update the parameters. Keep the optimizer for the entire training
//! run so that momentum and running statistics carry across steps.
//!
//! # Fit continuous targets
//!
//! ```
//! use rusttorch::nn::{Sequential, functional::mse_loss};
//! use rusttorch::optim::Adam;
//! use rusttorch::{DeviceSpec, Kind, Result, Tensor};
//!
//! # fn main() -> Result<()> {
//! let model = Sequential::builder().linear(3, 1).build(DeviceSpec::Cpu)?;
//! let mut optimizer = Adam::builder()
//!     .learning_rate(0.01)
//!     .build(model.var_store())?;
//! let features = Tensor::f_ones([8, 3], (Kind::Float, model.device()))?;
//! let targets = Tensor::f_zeros([8, 1], (Kind::Float, model.device()))?;
//!
//! let loss = mse_loss(&model.forward(&features)?, &targets)?;
//! optimizer.backward_step(&loss)?;
//! # Ok(())
//! # }
//! ```
//!
//! For manual gradient accumulation, call [`Optimizer::zero_grad`] once before
//! the contributing backward passes, then [`Optimizer::step`] once afterwards.
//! Scale each loss to match the intended batch reduction. Calling
//! [`Optimizer::backward_step`] between microbatches would clear the accumulated
//! gradients.

//!
//! # Resume and control training
//!
//! Assign parameter groups with [`crate::nn::ParameterPath::set_group`], then
//! use [`Optimizer::set_group_learning_rate`] and [`Optimizer::set_group_weight_decay`]
//! for feature extractors and heads that need different update settings. Use
//! [`StepLr`], [`ExponentialLr`], [`MultiStepLr`], [`CosineAnnealingLr`] or
//! [`ReduceLrOnPlateau`] after optimizer updates to change rates over time.
//!
//! Save [`Optimizer::state_dict`] at a completed update boundary together with
//! model weights, the scheduler and [`crate::amp::GradScalerState`]. Rebuild the
//! same architecture and optimizer family before [`Optimizer::load_state_dict`].
//! The versioned state includes named moments and group configuration; it is
//! independent of model weight storage. For fallible training loops use
//! [`Optimizer::try_zero_grad`] and [`Optimizer::try_step`].

use tch::{Tensor, nn::VarStore};

use crate::{Result, RustTorchError};

mod schedulers;
mod stateful;
pub use schedulers::{
    CosineAnnealingLr, ExponentialLr, MultiStepLr, PlateauMode, ReduceLrOnPlateau, StepLr,
    ThresholdMode,
};
use stateful::Algorithm;
pub use stateful::{Optimizer, OptimizerState, ParameterGroup};

const DEFAULT_LEARNING_RATE: f64 = 1e-3;

/// Adaptive moment estimation for training with parameter-specific step sizes.
///
/// Adam tracks moving averages of gradients and squared gradients. Defaults
/// are a learning rate of `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, no
/// weight decay, and no AMSGrad. See the [module example](self) for a training
/// step and [`Adam::build`] to attach it to a model's parameter store.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    amsgrad: bool,
    maximize: bool,
    foreach: Option<bool>,
    capturable: bool,
    differentiable: bool,
    fused: Option<bool>,
    decoupled_weight_decay: bool,
}

impl Default for Adam {
    fn default() -> Self {
        // Defaults follow PyTorch v2.13.0 torch/optim/adam.py.
        // See THIRD_PARTY_NOTICES.md.
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            foreach: None,
            capturable: false,
            differentiable: false,
            fused: None,
            decoupled_weight_decay: false,
        }
    }
}

impl Adam {
    /// Creates an Adam builder with the defaults described on [`Adam`].
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }

    #[must_use]
    /// Sets the learning rate.
    pub const fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    #[must_use]
    /// Sets both first- and second-moment decay rates.
    pub const fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    #[must_use]
    /// Sets the first-moment decay rate.
    pub const fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    #[must_use]
    /// Sets the second-moment decay rate.
    pub const fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    #[must_use]
    /// Sets the denominator stability term.
    pub const fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    /// Sets coupled L2 weight decay.
    pub const fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    #[must_use]
    /// Enables or disables the AMSGrad variant.
    pub const fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Requests loss maximization; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Requests batched parameter updates; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn foreach(mut self, foreach: bool) -> Self {
        self.foreach = Some(foreach);
        self
    }

    /// Requests graph-capturable updates; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn capturable(mut self, capturable: bool) -> Self {
        self.capturable = capturable;
        self
    }

    /// Requests gradients through updates; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn differentiable(mut self, differentiable: bool) -> Self {
        self.differentiable = differentiable;
        self
    }

    /// Requests fused updates; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn fused(mut self, fused: bool) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Requests decoupled weight decay; enabling it makes [`Adam::build`] return an error.
    #[must_use]
    pub const fn decoupled_weight_decay(mut self, decoupled: bool) -> Self {
        self.decoupled_weight_decay = decoupled;
        self
    }

    /// Validates the configuration and builds an Adam optimizer.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        self.validate()?;
        Optimizer::new(
            var_store,
            Algorithm::Adam {
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                amsgrad: self.amsgrad,
                decoupled: false,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }

    fn validate(self) -> Result<()> {
        validate_non_negative("learning_rate", self.learning_rate)?;
        validate_beta("beta1", self.beta1)?;
        validate_beta("beta2", self.beta2)?;
        validate_non_negative("eps", self.eps)?;
        validate_non_negative("weight_decay", self.weight_decay)?;
        reject_unsupported("Adam", "maximize", self.maximize)?;
        reject_unsupported("Adam", "foreach", self.foreach == Some(true))?;
        reject_unsupported("Adam", "capturable", self.capturable)?;
        reject_unsupported("Adam", "differentiable", self.differentiable)?;
        reject_unsupported("Adam", "fused", self.fused == Some(true))?;
        reject_unsupported(
            "Adam",
            "decoupled_weight_decay",
            self.decoupled_weight_decay,
        )
    }
}

/// Stochastic gradient descent with optional momentum and weight decay.
///
/// Defaults are a learning rate of `0.001`, no momentum or weight decay, zero
/// dampening, and no Nesterov acceleration. Momentum retains information from
/// previous gradients; Nesterov acceleration requires positive momentum and
/// zero dampening.
///
/// ```
/// use rusttorch::nn::Sequential;
/// use rusttorch::optim::Sgd;
/// use rusttorch::{DeviceSpec, Result};
///
/// # fn main() -> Result<()> {
/// let model = Sequential::builder().linear(4, 2).build(DeviceSpec::Cpu)?;
/// let optimizer = Sgd::builder()
///     .learning_rate(0.01)
///     .momentum(0.9)
///     .nesterov(true)
///     .build(model.var_store())?;
/// # let _ = optimizer;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    learning_rate: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
    maximize: bool,
    foreach: Option<bool>,
    differentiable: bool,
    fused: Option<bool>,
}

impl Default for Sgd {
    fn default() -> Self {
        // Defaults follow PyTorch v2.13.0 torch/optim/sgd.py.
        // See THIRD_PARTY_NOTICES.md.
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            maximize: false,
            foreach: None,
            differentiable: false,
            fused: None,
        }
    }
}

impl Sgd {
    /// Creates an SGD builder with the defaults described on [`Sgd`].
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }

    #[must_use]
    /// Sets the learning rate.
    pub const fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    #[must_use]
    /// Sets the momentum factor.
    pub const fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    #[must_use]
    /// Sets momentum dampening.
    pub const fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    #[must_use]
    /// Sets L2 weight decay.
    pub const fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    #[must_use]
    /// Enables Nesterov momentum.
    ///
    /// Nesterov requires positive momentum and zero dampening.
    pub const fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Requests loss maximization; enabling it makes [`Sgd::build`] return an error.
    #[must_use]
    pub const fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Requests batched parameter updates; enabling it makes [`Sgd::build`] return an error.
    #[must_use]
    pub const fn foreach(mut self, foreach: bool) -> Self {
        self.foreach = Some(foreach);
        self
    }

    /// Requests gradients through updates; enabling it makes [`Sgd::build`] return an error.
    #[must_use]
    pub const fn differentiable(mut self, differentiable: bool) -> Self {
        self.differentiable = differentiable;
        self
    }

    /// Requests fused updates; enabling it makes [`Sgd::build`] return an error.
    #[must_use]
    pub const fn fused(mut self, fused: bool) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Validates the configuration and builds an SGD optimizer.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        self.validate()?;
        Optimizer::new(
            var_store,
            Algorithm::Sgd {
                momentum: self.momentum,
                dampening: self.dampening,
                nesterov: self.nesterov,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }

    fn validate(self) -> Result<()> {
        validate_non_negative("learning_rate", self.learning_rate)?;
        validate_non_negative("momentum", self.momentum)?;
        validate_non_negative("dampening", self.dampening)?;
        validate_non_negative("weight_decay", self.weight_decay)?;
        if self.nesterov && (self.momentum <= 0.0 || self.dampening != 0.0) {
            return Err(RustTorchError::InvalidConfiguration {
                field: "nesterov",
                reason: "requires momentum > 0 and dampening == 0".to_owned(),
            });
        }
        reject_unsupported("SGD", "maximize", self.maximize)?;
        reject_unsupported("SGD", "foreach", self.foreach == Some(true))?;
        reject_unsupported("SGD", "differentiable", self.differentiable)?;
        reject_unsupported("SGD", "fused", self.fused == Some(true))
    }
}

/// Adam with decoupled weight decay for regularized model training.
///
/// Defaults: learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`,
/// weight decay `0.01`, and AMSGrad disabled. Weight decay acts directly on
/// parameters rather than being included in the gradient moments.
///
/// ```no_run
/// use rusttorch::{DeviceSpec, Result, nn::Sequential, optim::AdamW};
/// # fn main() -> Result<()> {
/// let model = Sequential::builder().linear(8, 2).build(DeviceSpec::Cpu)?;
/// let optimizer = AdamW::builder().weight_decay(0.1).build(model.var_store())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamW {
    config: Adam,
}

impl Default for AdamW {
    fn default() -> Self {
        // PyTorch v2.13.0 torch/optim/adamw.py; see THIRD_PARTY_NOTICES.md.
        Self {
            config: Adam::default().weight_decay(0.01),
        }
    }
}

impl AdamW {
    /// Creates a builder with the defaults described by [`AdamW`].
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }

    /// Sets the non-negative learning rate.
    #[must_use]
    pub const fn learning_rate(mut self, value: f64) -> Self {
        self.config = self.config.learning_rate(value);
        self
    }

    /// Sets the first and second moment decay rates, each in `[0, 1)`.
    #[must_use]
    pub const fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.config = self.config.betas(beta1, beta2);
        self
    }

    /// Sets the non-negative denominator stability term.
    #[must_use]
    pub const fn eps(mut self, value: f64) -> Self {
        self.config = self.config.eps(value);
        self
    }

    /// Sets non-negative decoupled weight decay; zero disables regularization.
    #[must_use]
    pub const fn weight_decay(mut self, value: f64) -> Self {
        self.config = self.config.weight_decay(value);
        self
    }

    /// Enables or disables the AMSGrad maximum second-moment variant.
    #[must_use]
    pub const fn amsgrad(mut self, enabled: bool) -> Self {
        self.config = self.config.amsgrad(enabled);
        self
    }

    /// Attaches the optimizer to the store's trainable parameters.
    ///
    /// Returns an error for invalid or non-finite configuration, or if the
    /// parameters or optimizer cannot be initialized. This builder supports dense,
    /// ordinary updates; it does not expose fused or differentiable updates.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        self.config.validate()?;
        Optimizer::new(
            var_store,
            Algorithm::Adam {
                beta1: self.config.beta1,
                beta2: self.config.beta2,
                eps: self.config.eps,
                amsgrad: self.config.amsgrad,
                decoupled: true,
            },
            self.config.learning_rate,
            self.config.weight_decay,
        )
    }
}

/// RMSprop scales updates by a moving average of squared gradients.
///
/// Useful for noisy objectives and recurrent models. Defaults: learning rate
/// `0.01`, alpha `0.99`, epsilon `1e-8`, zero momentum and weight decay, and
/// uncentered updates. Centered mode estimates gradient variance by also
/// tracking the mean gradient.
///
/// ```no_run
/// use rusttorch::{DeviceSpec, Result, nn::Sequential, optim::RmsProp};
/// # fn main() -> Result<()> {
/// let model = Sequential::builder().linear(8, 2).build(DeviceSpec::Cpu)?;
/// let optimizer = RmsProp::builder().momentum(0.9).centered(true)
///     .build(model.var_store())?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsProp {
    learning_rate: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    centered: bool,
}

impl Default for RmsProp {
    fn default() -> Self {
        // PyTorch v2.13.0 torch/optim/rmsprop.py; see THIRD_PARTY_NOTICES.md.
        Self {
            learning_rate: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }
}

impl RmsProp {
    /// Creates a builder with the defaults described by [`RmsProp`].
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }

    /// Sets the non-negative learning rate.
    #[must_use]
    pub const fn learning_rate(mut self, value: f64) -> Self {
        self.learning_rate = value;
        self
    }

    /// Sets the squared-gradient smoothing coefficient (normally below one).
    ///
    /// Must be finite and non-negative; values above one can produce invalid
    /// variance estimates and are generally unsuitable for training.
    #[must_use]
    pub const fn alpha(mut self, value: f64) -> Self {
        self.alpha = value;
        self
    }

    /// Sets the non-negative denominator stability term.
    #[must_use]
    pub const fn eps(mut self, value: f64) -> Self {
        self.eps = value;
        self
    }

    /// Sets non-negative coupled L2 weight decay.
    #[must_use]
    pub const fn weight_decay(mut self, value: f64) -> Self {
        self.weight_decay = value;
        self
    }

    /// Sets the non-negative momentum factor; zero disables momentum.
    #[must_use]
    pub const fn momentum(mut self, value: f64) -> Self {
        self.momentum = value;
        self
    }

    /// Centers the squared-gradient average using the moving gradient mean.
    #[must_use]
    pub const fn centered(mut self, value: bool) -> Self {
        self.centered = value;
        self
    }

    /// Validates all settings and constructs a dense RMSprop optimizer.
    ///
    /// Returns an error for negative or non-finite numerical settings, or a
    /// parameter initialization failure. Fused, capturable, differentiable and
    /// sparse-gradient updates are outside this builder's contract.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        for (name, value) in [
            ("learning_rate", self.learning_rate),
            ("alpha", self.alpha),
            ("eps", self.eps),
            ("weight_decay", self.weight_decay),
            ("momentum", self.momentum),
        ] {
            validate_non_negative(name, value)?;
        }
        Optimizer::new(
            var_store,
            Algorithm::RmsProp {
                alpha: self.alpha,
                eps: self.eps,
                momentum: self.momentum,
                centered: self.centered,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }
}

/// Adagrad adapts each parameter's rate using accumulated squared gradients.
///
/// Defaults: learning rate `0.01`, no rate decay or weight decay, initial
/// accumulator zero, and epsilon `1e-10`. This builder supports dense gradients.
///
/// ```
/// # fn example(store: &rusttorch::nn::VarStore) -> rusttorch::Result<()> {
/// let optimizer = rusttorch::optim::Adagrad::builder().learning_rate(0.05).build(store)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adagrad {
    learning_rate: f64,
    lr_decay: f64,
    weight_decay: f64,
    initial_accumulator: f64,
    eps: f64,
}
impl Default for Adagrad {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            lr_decay: 0.,
            weight_decay: 0.,
            initial_accumulator: 0.,
            eps: 1e-10,
        }
    }
}
impl Adagrad {
    /// Creates a builder with the defaults described on this type.
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }
    /// Sets the nonnegative finite learning rate.
    #[must_use]
    pub const fn learning_rate(mut self, value: f64) -> Self {
        self.learning_rate = value;
        self
    }
    /// Sets learning-rate decay as `lr / (1 + (step - 1) * lr_decay)`.
    #[must_use]
    pub const fn lr_decay(mut self, value: f64) -> Self {
        self.lr_decay = value;
        self
    }
    /// Sets nonnegative coupled L2 weight decay.
    #[must_use]
    pub const fn weight_decay(mut self, value: f64) -> Self {
        self.weight_decay = value;
        self
    }
    /// Sets the nonnegative initial value of each squared-gradient accumulator.
    #[must_use]
    pub const fn initial_accumulator_value(mut self, value: f64) -> Self {
        self.initial_accumulator = value;
        self
    }
    /// Sets the nonnegative denominator stability term.
    #[must_use]
    pub const fn eps(mut self, value: f64) -> Self {
        self.eps = value;
        self
    }
    /// Validates configuration and attaches to the store's parameters.
    pub fn build(self, store: &VarStore) -> Result<Optimizer> {
        Optimizer::new(
            store,
            Algorithm::Adagrad {
                lr_decay: self.lr_decay,
                initial_accumulator: self.initial_accumulator,
                eps: self.eps,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }
}

/// Adadelta scales updates using moving averages of gradients and prior updates.
///
/// Defaults: learning rate `1.0`, rho `0.9`, epsilon `1e-6`, no weight decay.
/// All gradients must be dense.
///
/// ```
/// # fn example(store: &rusttorch::nn::VarStore) -> rusttorch::Result<()> {
/// let optimizer = rusttorch::optim::Adadelta::builder().rho(0.95).build(store)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adadelta {
    learning_rate: f64,
    rho: f64,
    eps: f64,
    weight_decay: f64,
}
impl Default for Adadelta {
    fn default() -> Self {
        Self {
            learning_rate: 1.,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.,
        }
    }
}
impl Adadelta {
    /// Creates a builder with the defaults described on this type.
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }
    /// Sets the nonnegative finite learning rate.
    #[must_use]
    pub const fn learning_rate(mut self, value: f64) -> Self {
        self.learning_rate = value;
        self
    }
    /// Sets the moving-average coefficient in `[0, 1]`.
    #[must_use]
    pub const fn rho(mut self, value: f64) -> Self {
        self.rho = value;
        self
    }
    /// Sets the nonnegative stability term.
    #[must_use]
    pub const fn eps(mut self, value: f64) -> Self {
        self.eps = value;
        self
    }
    /// Sets nonnegative coupled L2 weight decay.
    #[must_use]
    pub const fn weight_decay(mut self, value: f64) -> Self {
        self.weight_decay = value;
        self
    }
    /// Validates configuration and attaches to the store's parameters.
    pub fn build(self, store: &VarStore) -> Result<Optimizer> {
        Optimizer::new(
            store,
            Algorithm::Adadelta {
                rho: self.rho,
                eps: self.eps,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }
}

/// Adamax uses an exponentially weighted infinity norm to scale first moments.
///
/// Defaults: learning rate `0.002`, betas `(0.9, 0.999)`, epsilon `1e-8`,
/// and no weight decay. All gradients must be dense.
///
/// ```
/// # fn example(store: &rusttorch::nn::VarStore) -> rusttorch::Result<()> {
/// let optimizer = rusttorch::optim::Adamax::builder().betas(0.8, 0.99).build(store)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adamax {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}
impl Default for Adamax {
    fn default() -> Self {
        Self {
            learning_rate: 0.002,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.,
        }
    }
}
impl Adamax {
    /// Creates a builder with the defaults described on this type.
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }
    /// Sets the nonnegative finite learning rate.
    #[must_use]
    pub const fn learning_rate(mut self, value: f64) -> Self {
        self.learning_rate = value;
        self
    }
    /// Sets the two decay rates, each finite and in `[0, 1)`.
    #[must_use]
    pub const fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    /// Sets the nonnegative stability term.
    #[must_use]
    pub const fn eps(mut self, value: f64) -> Self {
        self.eps = value;
        self
    }
    /// Sets nonnegative coupled L2 weight decay.
    #[must_use]
    pub const fn weight_decay(mut self, value: f64) -> Self {
        self.weight_decay = value;
        self
    }
    /// Validates configuration and attaches to the store's parameters.
    pub fn build(self, store: &VarStore) -> Result<Optimizer> {
        Optimizer::new(
            store,
            Algorithm::Adamax {
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
            },
            self.learning_rate,
            self.weight_decay,
        )
    }
}

fn validate_loss(loss: &Tensor) -> Result<()> {
    if !loss.defined() {
        return Err(RustTorchError::InvalidConfiguration {
            field: "loss",
            reason: "must be a defined scalar tensor".to_owned(),
        });
    }
    let shape = loss.size();
    if !shape.is_empty() {
        return Err(RustTorchError::InvalidDimensions {
            context: "optimizer loss".to_owned(),
            expected: "scalar tensor with shape []".to_owned(),
            actual: format!("shape {shape:?}"),
        });
    }
    Ok(())
}

fn validate_non_negative(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(RustTorchError::InvalidConfiguration {
            field,
            reason: format!("must be finite and non-negative, got {value}"),
        })
    }
}

fn validate_beta(field: &'static str, value: f64) -> Result<()> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(RustTorchError::InvalidConfiguration {
            field,
            reason: format!("must be finite and in [0, 1), got {value}"),
        })
    }
}

fn reject_unsupported(component: &'static str, option: &'static str, enabled: bool) -> Result<()> {
    if enabled {
        Err(RustTorchError::UnsupportedOption { component, option })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn rejects_invalid_and_unsupported_configuration() {
        assert!(matches!(
            Adam::builder().betas(1.0, 0.999).validate(),
            Err(RustTorchError::InvalidConfiguration { field: "beta1", .. })
        ));
        assert!(matches!(
            Adam::builder().maximize(true).validate(),
            Err(RustTorchError::UnsupportedOption {
                option: "maximize",
                ..
            })
        ));
        assert!(matches!(
            Sgd::builder().nesterov(true).validate(),
            Err(RustTorchError::InvalidConfiguration {
                field: "nesterov",
                ..
            })
        ));
    }

    #[test]
    fn requires_a_defined_scalar_loss() {
        assert!(matches!(
            validate_loss(&Tensor::new()),
            Err(RustTorchError::InvalidConfiguration { field: "loss", .. })
        ));
        assert!(matches!(
            validate_loss(&Tensor::zeros([1], (Kind::Float, Device::Cpu))),
            Err(RustTorchError::InvalidDimensions { .. })
        ));
        assert!(validate_loss(&Tensor::zeros([], (Kind::Float, Device::Cpu))).is_ok());
    }
}
