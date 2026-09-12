//! Sample noise or discrete choices and evaluate their probabilities.
//!
//! Use [`Normal`] for continuous measurement noise and reparameterized latent
//! variables, [`Bernoulli`] for independent yes/no events, and [`Categorical`]
//! to choose one class. Shapes are `sample_shape + batch_shape`; the last
//! categorical parameter axis contains the classes. Sampling uses LibTorch's
//! generator, controlled by [`crate::manual_seed`]. It is shared with other
//! tensor operations, so concurrent draws change its stream order.
//!
//! ```
//! use rusttorch::{Tensor, distributions::Normal};
//! let noise = Normal::new(&Tensor::from(0_f32), &Tensor::from(0.1_f32))?;
//! let draws = noise.sample(&[8, 3])?;
//! assert_eq!(draws.size(), [8, 3]);
//! assert!(!draws.requires_grad());
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use crate::{Kind, Reduction, Result, RustTorchError, Tensor, no_grad};

/// Independent scalar Gaussian distributions with broadcastable location and scale.
///
/// Scale must be finite and positive. `sample` returns detached observations;
/// [`Normal::rsample`] keeps a pathwise derivative to location and scale, useful
/// for variational models. Parameters share the supplied tensors' storage/graphs;
/// rebuild the distribution if their shapes or values are modified in place.
///
/// ```
/// use rusttorch::{Tensor, distributions::Normal};
/// let mean = Tensor::from_slice(&[0_f64, 1.]).set_requires_grad(true);
/// let distribution = Normal::new(&mean, &Tensor::from(0.5_f64))?;
/// assert_eq!(distribution.rsample(&[4])?.size(), [4, 2]);
/// assert_eq!(distribution.mean().size(), [2]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug)]
pub struct Normal {
    location: Tensor,
    scale: Tensor,
}
impl Normal {
    /// Broadcasts finite real location/positive scale tensors on one dtype/device.
    pub fn new(location: &Tensor, scale: &Tensor) -> Result<Self> {
        compatible(location, scale)?;
        if !all(&scale.f_gt(0.)?)? {
            return Err(invalid("scale", "must be positive"));
        }
        let mut parameters = Tensor::f_broadcast_tensors(&[location, scale])?;
        let scale = parameters.pop().unwrap();
        let location = parameters.pop().unwrap();
        Ok(Self { location, scale })
    }
    /// Returns the broadcast parameter shape, excluding requested sample axes.
    pub fn batch_shape(&self) -> Vec<i64> {
        self.location.size()
    }
    /// Returns the location tensor; its storage and autograd graph are shared.
    pub fn mean(&self) -> Tensor {
        self.location.shallow_clone()
    }
    /// Returns the squared scale with its parameter gradient graph.
    pub fn variance(&self) -> Result<Tensor> {
        Ok(self.scale.f_square()?)
    }
    /// Draws detached samples with the requested leading sample dimensions.
    pub fn sample(&self, sample_shape: &[i64]) -> Result<Tensor> {
        no_grad(|| self.rsample(sample_shape))
    }
    /// Draws `location + scale * standard_normal` with pathwise gradients.
    ///
    /// Unlike `sample`, gradients can reach both parameters. Draws use the global
    /// native generator. Exact seed-stream parity is verified for CPU `rsample`.
    pub fn rsample(&self, sample_shape: &[i64]) -> Result<Tensor> {
        let shape = extended(sample_shape, &self.batch_shape())?;
        let noise = Tensor::f_randn(
            shape.as_slice(),
            (self.location.kind(), self.location.device()),
        )?;
        Ok(noise.f_mul(&self.scale)?.f_add(&self.location)?)
    }
    /// Returns elementwise log density; reduce over event axes explicitly.
    pub fn log_prob(&self, value: &Tensor) -> Result<Tensor> {
        compatible(value, &self.location)?;
        let standardized = value.f_sub(&self.location)?.f_div(&self.scale)?;
        Ok(standardized
            .f_square()?
            .f_mul_scalar(-0.5)?
            .f_sub(&self.scale.f_log()?)?
            .f_sub_scalar(0.5 * (2. * std::f64::consts::PI).ln())?)
    }
    /// Returns differential entropy for each broadcast parameter position.
    pub fn entropy(&self) -> Result<Tensor> {
        Ok(self
            .scale
            .f_log()?
            .f_add_scalar(0.5 * (2. * std::f64::consts::PI * std::f64::consts::E).ln())?)
    }
}

/// Independent binary events, with differentiable log probability and entropy.
///
/// Choose probabilities for known event rates or logits for a model's raw
/// outputs. Sampling is discrete and detached; optimize `log_prob` or another
/// differentiable objective instead of backpropagating through a draw.
///
/// ```
/// use rusttorch::{Tensor, distributions::Bernoulli};
/// let events = Bernoulli::from_probs(&Tensor::from_slice(&[0_f32, 1.]))?;
/// assert_eq!(Vec::<f32>::try_from(&events.sample(&[])?)?, [0., 1.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug)]
pub struct Bernoulli {
    probabilities: Tensor,
    logits: Tensor,
}
impl Bernoulli {
    /// Creates events with finite probabilities in `[0, 1]`.
    ///
    /// Logits use dtype-epsilon clamping at zero and one, matching the native
    /// distribution convention; log probabilities at those boundaries are finite.
    pub fn from_probs(probabilities: &Tensor) -> Result<Self> {
        real(probabilities)?;
        if !all(&probabilities
            .f_ge(0.)?
            .f_logical_and(&probabilities.f_le(1.)?)?)?
        {
            return Err(invalid("probabilities", "must be in [0, 1]"));
        }
        let epsilon = epsilon(probabilities.kind());
        let p = probabilities.f_clamp(epsilon, 1. - epsilon)?;
        let logits = p.f_log()?.f_sub(&p.f_neg()?.f_log1p()?)?;
        Ok(Self {
            probabilities: probabilities.shallow_clone(),
            logits,
        })
    }
    /// Creates events from finite, unnormalized binary scores.
    pub fn from_logits(logits: &Tensor) -> Result<Self> {
        real(logits)?;
        Ok(Self {
            probabilities: logits.f_sigmoid()?,
            logits: logits.shallow_clone(),
        })
    }
    /// Returns the event probability shape, excluding sample axes.
    pub fn batch_shape(&self) -> Vec<i64> {
        self.probabilities.size()
    }
    /// Returns the success probabilities with their autograd graph.
    pub fn probabilities(&self) -> &Tensor {
        &self.probabilities
    }
    /// Draws detached zeros or ones in the parameter dtype.
    pub fn sample(&self, sample_shape: &[i64]) -> Result<Tensor> {
        let shape = extended(sample_shape, &self.batch_shape())?;
        no_grad(|| {
            Ok(self
                .probabilities
                .f_expand(shape.as_slice(), false)?
                .f_bernoulli()?)
        })
    }
    /// Returns log probability of broadcastable zero/one observations.
    pub fn log_prob(&self, value: &Tensor) -> Result<Tensor> {
        compatible(value, &self.logits)?;
        if !all(&value.f_eq(0.)?.f_logical_or(&value.f_eq(1.)?)?)? {
            return Err(invalid("value", "binary observations must be zero or one"));
        }
        let tensors = Tensor::f_broadcast_tensors(&[&self.logits, value])?;
        Ok(tensors[0]
            .f_binary_cross_entropy_with_logits(
                &tensors[1],
                None::<&Tensor>,
                None::<&Tensor>,
                Reduction::None,
            )?
            .f_neg()?)
    }
    /// Returns entropy per binary event, preserving parameter derivatives.
    pub fn entropy(&self) -> Result<Tensor> {
        Ok(self.logits.f_binary_cross_entropy_with_logits(
            &self.probabilities,
            None::<&Tensor>,
            None::<&Tensor>,
            Reduction::None,
        )?)
    }
}

/// A choice among the classes on the final parameter axis.
///
/// Leading axes identify independent distributions. Samples are `Int64` class
/// IDs; `log_prob` accepts those IDs and differentiates the normalized scores.
///
/// ```
/// use rusttorch::{Tensor, distributions::Categorical};
/// let choices = Categorical::from_logits(&Tensor::from_slice(&[1_f32, 2., 3.]))?;
/// assert_eq!(choices.sample(&[5])?.size(), [5]);
/// assert!(choices.log_prob(&Tensor::from(2_i64))?.size().is_empty());
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug)]
pub struct Categorical {
    probabilities: Tensor,
    log_probabilities: Tensor,
}
impl Categorical {
    /// Normalizes finite class logits along the last dimension.
    pub fn from_logits(logits: &Tensor) -> Result<Self> {
        classes(logits)?;
        let log_probabilities = logits.f_log_softmax(-1, logits.kind())?;
        Ok(Self {
            probabilities: log_probabilities.f_exp()?,
            log_probabilities,
        })
    }
    /// Normalizes finite nonnegative weights with positive sum in every row.
    pub fn from_probs(probabilities: &Tensor) -> Result<Self> {
        classes(probabilities)?;
        let sums = probabilities.f_sum_dim_intlist([-1].as_slice(), true, probabilities.kind())?;
        if !all(&probabilities.f_ge(0.)?)?
            || !all(&sums.f_gt(0.)?.f_logical_and(&sums.f_isfinite()?)?)?
        {
            return Err(invalid(
                "probabilities",
                "each row needs nonnegative weights with a finite positive sum",
            ));
        }
        let probabilities = probabilities.f_div(&sums)?;
        let epsilon = epsilon(probabilities.kind());
        let log_probabilities = probabilities.f_clamp(epsilon, 1. - epsilon)?.f_log()?;
        Ok(Self {
            probabilities,
            log_probabilities,
        })
    }
    /// Returns parameter axes before the final class axis.
    pub fn batch_shape(&self) -> Vec<i64> {
        let mut shape = self.probabilities.size();
        shape.pop();
        shape
    }
    /// Returns normalized class probabilities with their autograd graph.
    pub fn probabilities(&self) -> &Tensor {
        &self.probabilities
    }
    /// Draws detached class IDs with shape `sample_shape + batch_shape`.
    pub fn sample(&self, sample_shape: &[i64]) -> Result<Tensor> {
        let shape = extended(sample_shape, &self.batch_shape())?;
        let count = sample_shape
            .iter()
            .try_fold(1_i64, |a, &b| a.checked_mul(b))
            .ok_or_else(|| invalid("sample_shape", "element count overflow"))?;
        if count == 0 {
            return Ok(Tensor::f_zeros(
                shape.as_slice(),
                (Kind::Int64, self.probabilities.device()),
            )?);
        }
        let classes = *self.probabilities.size().last().unwrap();
        no_grad(|| {
            Ok(self
                .probabilities
                .f_reshape([-1, classes])?
                .f_multinomial(count, true)?
                .f_transpose(0, 1)?
                .f_reshape(shape.as_slice())?)
        })
    }
    /// Returns log probability of broadcastable `Int64` class indices.
    pub fn log_prob(&self, value: &Tensor) -> Result<Tensor> {
        if !value.defined()
            || value.kind() != Kind::Int64
            || value.device() != self.probabilities.device()
        {
            return Err(invalid(
                "value",
                "expected defined Int64 indices on the parameter device",
            ));
        }
        let classes = *self.probabilities.size().last().unwrap();
        if !all(&value.f_ge(0)?.f_logical_and(&value.f_lt(classes)?)?)? {
            return Err(invalid("value", "class index out of range"));
        }
        let values = value.f_unsqueeze(-1)?;
        let tensors = Tensor::f_broadcast_tensors(&[&values, &self.log_probabilities])?;
        Ok(tensors[1]
            .f_gather(-1, &tensors[0].f_narrow(-1, 0, 1)?, false)?
            .f_squeeze_dim(-1)?)
    }
    /// Returns entropy across the class axis for each batch position.
    pub fn entropy(&self) -> Result<Tensor> {
        Ok(self
            .probabilities
            .f_mul(
                &self
                    .log_probabilities
                    .f_clamp_min(match self.probabilities.kind() {
                        Kind::Double => f64::MIN,
                        Kind::Half => -65504.,
                        Kind::BFloat16 => -3.3895313892515355e38,
                        _ => f32::MIN as f64,
                    })?,
            )?
            .f_sum_dim_intlist([-1].as_slice(), false, self.probabilities.kind())?
            .f_neg()?)
    }
}

fn classes(input: &Tensor) -> Result<()> {
    real(input)?;
    if input.size().last().is_none_or(|&n| n == 0) {
        return Err(invalid("classes", "expected a nonempty final class axis"));
    }
    Ok(())
}
fn compatible(left: &Tensor, right: &Tensor) -> Result<()> {
    real(left)?;
    real(right)?;
    if left.kind() != right.kind() || left.device() != right.device() {
        return Err(invalid("parameters", "dtype and device must match"));
    }
    Ok(())
}
fn real(input: &Tensor) -> Result<()> {
    if !input.defined() {
        return Err(invalid("tensor", "must be defined"));
    }
    if !matches!(
        input.kind(),
        Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16
    ) || input.is_sparse()
    {
        return Err(invalid(
            "tensor",
            "expected dense real floating-point values",
        ));
    }
    let _ = input.f_as_strided([0], [1], None)?;
    if !all(&input.f_isfinite()?)? {
        return Err(invalid("tensor", "values must be finite"));
    }
    Ok(())
}
fn all(input: &Tensor) -> Result<bool> {
    Ok(input.f_all()?.f_int64_value(&[])? != 0)
}
fn epsilon(kind: Kind) -> f64 {
    match kind {
        Kind::Double => f64::EPSILON,
        Kind::Half => 0.0009765625,
        Kind::BFloat16 => 0.0078125,
        _ => f32::EPSILON as f64,
    }
}
fn extended(sample: &[i64], batch: &[i64]) -> Result<Vec<i64>> {
    if sample.iter().any(|&n| n < 0) {
        return Err(invalid("sample_shape", "dimensions must be nonnegative"));
    }
    let mut shape = sample.to_vec();
    shape.extend_from_slice(batch);
    Ok(shape)
}
fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
