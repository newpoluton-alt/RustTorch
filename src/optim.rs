//! Ergonomic optimizer builders backed by `tch`/LibTorch.

use tch::{Tensor, nn::VarStore};

use crate::{Result, RustTorchError};

const DEFAULT_LEARNING_RATE: f64 = 1e-3;

/// An optimizer whose updates are delegated to LibTorch.
#[derive(Debug)]
pub struct Optimizer {
    inner: tch::nn::Optimizer,
}

impl Optimizer {
    /// Clears gradients for all tracked parameters.
    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    /// Applies one optimizer step using the current gradients.
    pub fn step(&mut self) {
        self.inner.step();
    }

    /// Clears gradients, backpropagates a scalar loss, and applies one optimizer step.
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        validate_loss(loss)?;
        self.inner.backward_step(loss);
        Ok(())
    }
}

/// Builder for PyTorch-compatible Adam backed by `tch::nn::Adam`.
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
    /// Creates an Adam builder with PyTorch 2.13 defaults.
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

    /// Records PyTorch's `maximize` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Records PyTorch's `foreach` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn foreach(mut self, foreach: bool) -> Self {
        self.foreach = Some(foreach);
        self
    }

    /// Records PyTorch's `capturable` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn capturable(mut self, capturable: bool) -> Self {
        self.capturable = capturable;
        self
    }

    /// Records PyTorch's `differentiable` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn differentiable(mut self, differentiable: bool) -> Self {
        self.differentiable = differentiable;
        self
    }

    /// Records PyTorch's `fused` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn fused(mut self, fused: bool) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Records PyTorch's decoupled weight-decay option; `true` requires AdamW.
    #[must_use]
    pub const fn decoupled_weight_decay(mut self, decoupled: bool) -> Self {
        self.decoupled_weight_decay = decoupled;
        self
    }

    /// Validates the configuration and builds a LibTorch Adam optimizer.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        self.validate()?;
        let config = tch::nn::Adam {
            beta1: self.beta1,
            beta2: self.beta2,
            wd: self.weight_decay,
            eps: self.eps,
            amsgrad: self.amsgrad,
        };
        let inner = tch::nn::OptimizerConfig::build(config, var_store, self.learning_rate)?;
        Ok(Optimizer { inner })
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

/// Builder for PyTorch-compatible SGD backed by `tch::nn::Sgd`.
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
    /// Creates an SGD builder with PyTorch 2.13 defaults.
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

    /// Records PyTorch's `maximize` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Records PyTorch's `foreach` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn foreach(mut self, foreach: bool) -> Self {
        self.foreach = Some(foreach);
        self
    }

    /// Records PyTorch's `differentiable` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn differentiable(mut self, differentiable: bool) -> Self {
        self.differentiable = differentiable;
        self
    }

    /// Records PyTorch's `fused` option; `true` is unsupported by `tch`.
    #[must_use]
    pub const fn fused(mut self, fused: bool) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Validates the configuration and builds a LibTorch SGD optimizer.
    pub fn build(self, var_store: &VarStore) -> Result<Optimizer> {
        self.validate()?;
        let config = tch::nn::Sgd {
            momentum: self.momentum,
            dampening: self.dampening,
            wd: self.weight_decay,
            nesterov: self.nesterov,
        };
        let inner = tch::nn::OptimizerConfig::build(config, var_store, self.learning_rate)?;
        Ok(Optimizer { inner })
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
