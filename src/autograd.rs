//! Differentiate calculations without changing parameter `.grad()` buffers.
//!
//! Use [`grad`] for a scalar objective, [`vjp`] for sensitivities to a weighted
//! output, [`jvp`] for a directional derivative, and [`jacobian`] or [`hessian`]
//! when you need every derivative entry. These APIs use safe native reverse-mode
//! differentiation. JVP uses reverse-over-reverse; it is not a native forward-AD
//! context. Inputs are dense real floating-point tensors.
//!
//! ```
//! use rusttorch::{Tensor, Kind, autograd::{grad, GradOptions}};
//! let x = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
//! let loss = x.f_square()?.f_sum(Kind::Double)?;
//! let derivative = grad(&loss, &[&x], GradOptions::default())?;
//! assert_eq!(Vec::<f64>::try_from(&derivative[0])?, [4., 6.]);
//! assert!(!x.grad().defined());
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use crate::{Kind, Result, RustTorchError, Tensor};

/// Controls graph lifetime for [`grad`] and [`vjp`].
///
/// Enable `create_graph` when differentiating the returned derivative again.
/// Enable `retain_graph` when another derivative will reuse the same forward
/// calculation. Creating a derivative graph also retains the forward graph.
///
/// ```
/// let options = rusttorch::autograd::GradOptions { create_graph: true,
///     ..Default::default() };
/// assert!(options.create_graph);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GradOptions {
    /// Retain the forward graph for another gradient query.
    pub retain_graph: bool,
    /// Record derivative operations for higher-order differentiation.
    pub create_graph: bool,
}

/// Returns the derivatives of a scalar loss with respect to each input.
///
/// Inputs must already require gradients and participate in the loss. Unused
/// inputs return an error; functional [`jacobian`] and [`hessian`] instead
/// materialize their mathematically zero entries. This does not populate or
/// clear input `.grad()` buffers. See the module example for a scalar objective.
pub fn grad(loss: &Tensor, inputs: &[&Tensor], options: GradOptions) -> Result<Vec<Tensor>> {
    scalar(loss)?;
    if inputs.is_empty() {
        return Err(invalid("inputs", "at least one input is required"));
    }
    for input in inputs {
        tracked(input)?;
    }
    if !loss.requires_grad() {
        return Err(invalid("loss", "the loss has no recorded gradient graph"));
    }
    let values = Tensor::f_run_backward(
        &[loss],
        inputs,
        options.retain_graph || options.create_graph,
        options.create_graph,
    )?;
    if values.iter().any(|value| !value.defined()) {
        return Err(invalid(
            "inputs",
            "an input does not participate in the loss",
        ));
    }
    Ok(values)
}

/// Computes a vector-Jacobian product for one tracked input.
///
/// The cotangent must match the output shape, dtype and device. It is treated
/// as a fixed seed: its own gradient graph is detached. Enable `create_graph`
/// to differentiate the result with respect to the input again.
///
/// ```
/// use rusttorch::{Tensor, autograd::{vjp, GradOptions}};
/// let x = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
/// let weights = Tensor::from_slice(&[1_f64, 2.]);
/// let sensitivity = vjp(&x.f_square()?, &x, &weights, GradOptions::default())?;
/// assert_eq!(Vec::<f64>::try_from(&sensitivity)?, [4., 12.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn vjp(
    output: &Tensor,
    input: &Tensor,
    cotangent: &Tensor,
    options: GradOptions,
) -> Result<Tensor> {
    real(output)?;
    same(output, cotangent, "cotangent")?;
    tracked(input)?;
    let weighted = output.f_mul(&cotangent.f_detach()?)?.f_sum(output.kind())?;
    Ok(grad(&weighted, &[input], options)?.remove(0))
}

/// Returns a function value and its directional derivative along `tangent`.
///
/// Evaluates the function once, then uses two reverse passes. Unlike a native
/// forward-mode implementation this requires double-backward support for the
/// chosen operators. Constant functions return a zero derivative. The tangent
/// is a fixed seed. With `create_graph`, an already tracked input stays connected
/// to results that depend on it; constant derivatives may have no graph.
/// Otherwise both outputs are detached.
///
/// ```
/// use rusttorch::{Tensor, autograd::jvp};
/// let x = Tensor::from_slice(&[2_f64, 3.]);
/// let direction = Tensor::from_slice(&[0.5_f64, -1.]);
/// let (value, derivative) = jvp(|x| Ok(x.f_square()?), &x, &direction, false)?;
/// assert_eq!(Vec::<f64>::try_from(&value)?, [4., 9.]);
/// assert_eq!(Vec::<f64>::try_from(&derivative)?, [2., -6.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn jvp(
    function: impl FnOnce(&Tensor) -> Result<Tensor>,
    input: &Tensor,
    tangent: &Tensor,
    create_graph: bool,
) -> Result<(Tensor, Tensor)> {
    same(input, tangent, "tangent")?;
    let x = prepare(input, create_graph)?;
    let output = function(&x)?;
    real(&output)?;
    let seed = output.f_zeros_like()?.f_set_requires_grad(true)?;
    let weighted = output.f_mul(&seed)?.f_sum(output.kind())?;
    let first = derivative_or_zero(&weighted, &x, true, true)?;
    let contracted = first.f_mul(&tangent.f_detach()?)?.f_sum(x.kind())?;
    let result = derivative_or_zero(&contracted, &seed, create_graph, create_graph)?;
    if create_graph {
        Ok((output, result))
    } else {
        Ok((output.f_detach()?, result.f_detach()?))
    }
}

/// Materializes the Jacobian with shape `output.shape + input.shape`.
///
/// The function is evaluated once. One reverse pass per output element gives
/// explicit derivatives for small functions, sensitivity analysis and tests.
/// For large outputs, use [`jvp`] or [`vjp`] to avoid allocating the full matrix.
/// Constant or unused input components have zero derivatives.
///
/// ```
/// use rusttorch::{Tensor, autograd::jacobian};
/// let x = Tensor::from_slice(&[2_f64, 3.]);
/// let matrix = jacobian(|x| Ok(x.f_square()?), &x, false)?;
/// assert_eq!(matrix.size(), [2, 2]);
/// assert_eq!(matrix.double_value(&[1, 1]), 6.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn jacobian(
    function: impl FnOnce(&Tensor) -> Result<Tensor>,
    input: &Tensor,
    create_graph: bool,
) -> Result<Tensor> {
    let x = prepare(input, create_graph)?;
    let output = function(&x)?;
    real(&output)?;
    let mut shape = output.size();
    shape.extend(x.size());
    if output.numel() == 0 {
        return Ok(Tensor::f_zeros(shape.as_slice(), (x.kind(), x.device()))?);
    }
    let flat = output.f_reshape([-1])?;
    // ponytail: one reverse pass per output; use JVP/VJP when a full Jacobian is too large.
    let rows = (0..output.numel())
        .map(|index| {
            let item = flat.f_select(0, index as i64)?;
            derivative_or_zero(
                &item,
                &x,
                index + 1 < output.numel() || create_graph,
                create_graph,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let result = Tensor::f_stack(&rows, 0)?.f_reshape(shape.as_slice())?;
    if create_graph {
        Ok(result)
    } else {
        Ok(result.f_detach()?)
    }
}

/// Computes the Hessian of a scalar function with shape `input.shape + input.shape`.
///
/// This differentiates a gradient using [`jacobian`], so the operators must
/// support double backward. Linear and constant functions yield zero matrices.
///
/// ```
/// use rusttorch::{Tensor, Kind, autograd::hessian};
/// let x = Tensor::from_slice(&[2_f64, 3.]);
/// let matrix = hessian(|x| Ok(x.f_pow_tensor_scalar(3)?.f_sum(Kind::Double)?), &x, false)?;
/// assert_eq!(matrix.double_value(&[0, 0]), 12.0);
/// assert_eq!(matrix.double_value(&[1, 1]), 18.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn hessian(
    function: impl FnOnce(&Tensor) -> Result<Tensor>,
    input: &Tensor,
    create_graph: bool,
) -> Result<Tensor> {
    jacobian(
        |x| {
            let loss = function(x)?;
            scalar(&loss)?;
            derivative_or_zero(&loss, x, true, true)
        },
        input,
        create_graph,
    )
}

/// Uses `value` in the forward pass and `surrogate` for all recorded derivatives.
///
/// This composes a custom derivative from ordinary tensor operations without
/// Python hooks. For a straight-through rounding estimator, pass `x.round()`
/// as the value and `x` as the surrogate. Both tensors must have matching dense
/// real metadata; the surrogate must be finite to make its zero-valued correction
/// exact. The value's own gradient graph is intentionally detached.
///
/// ```
/// use rusttorch::{Tensor, Kind, autograd::{with_surrogate_gradient, grad, GradOptions}};
/// let x = Tensor::from_slice(&[0.2_f32, 1.7]).set_requires_grad(true);
/// let rounded = with_surrogate_gradient(&x.f_round()?, &x)?;
/// assert_eq!(Vec::<f32>::try_from(&rounded)?, [0., 2.]);
/// let derivative = grad(&rounded.f_sum(Kind::Float)?, &[&x], GradOptions::default())?;
/// assert_eq!(Vec::<f32>::try_from(&derivative[0])?, [1., 1.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn with_surrogate_gradient(value: &Tensor, surrogate: &Tensor) -> Result<Tensor> {
    same(value, surrogate, "surrogate")?;
    if surrogate.f_isfinite()?.f_all()?.f_int64_value(&[])? == 0 {
        return Err(invalid(
            "surrogate",
            "must be finite to preserve the forward value",
        ));
    }
    Ok(value
        .f_detach()?
        .f_add(&surrogate.f_sub(&surrogate.f_detach()?)?)?)
}

fn derivative_or_zero(loss: &Tensor, input: &Tensor, keep: bool, create: bool) -> Result<Tensor> {
    if !loss.requires_grad() {
        return Ok(input.f_zeros_like()?);
    }
    let gradient = Tensor::f_run_backward(&[loss], &[input], keep, create)?.remove(0);
    if gradient.defined() {
        Ok(gradient)
    } else {
        Ok(input.f_zeros_like()?)
    }
}

fn prepare(input: &Tensor, create_graph: bool) -> Result<Tensor> {
    real(input)?;
    if create_graph && input.requires_grad() {
        Ok(input.shallow_clone())
    } else {
        Ok(input.f_detach()?.f_set_requires_grad(true)?)
    }
}

fn tracked(input: &Tensor) -> Result<()> {
    real(input)?;
    if !input.requires_grad() {
        return Err(invalid(
            "input",
            "must require gradients before the forward pass",
        ));
    }
    Ok(())
}
fn scalar(input: &Tensor) -> Result<()> {
    real(input)?;
    if !input.size().is_empty() {
        return Err(invalid("loss", "expected a scalar; reduce the loss first"));
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
            "expected a dense real floating-point tensor",
        ));
    }
    // A fallible strided operation rejects compressed sparse/other unsupported layouts.
    let _ = input.f_as_strided([0], [1], None)?;
    Ok(())
}
fn same(input: &Tensor, other: &Tensor, name: &'static str) -> Result<()> {
    real(input)?;
    real(other)?;
    if input.size() != other.size()
        || input.kind() != other.kind()
        || input.device() != other.device()
    {
        return Err(invalid(name, "shape, dtype and device must match"));
    }
    Ok(())
}
fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
