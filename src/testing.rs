//! Check tensor results and derivatives when developing models or custom operations.
//!
//! Comparisons validate metadata before checking values, so broadcasting cannot
//! hide a shape error. Gradient checks use small, deterministic Float64 inputs.
//!
//! ```
//! use rusttorch::{Tensor, testing::{assert_close, gradcheck, CloseOptions, GradcheckOptions}};
//! let x = Tensor::from_slice(&[0.2_f64, -0.4]);
//! assert_close(&x, &x, CloseOptions::default())?;
//! gradcheck(|x| Ok(x.f_tanh()?), &x, GradcheckOptions::default())?;
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use crate::{Device, Kind, Result, RustTorchError, Tensor, autograd};

/// Tolerances for [`compare`] and [`assert_close`].
///
/// Floating and complex values satisfy `abs(actual - expected) <= atol +
/// rtol * abs(expected)`. Integers and Boolean values compare exactly.
/// Defaults are explicit, dtype-independent `rtol=1e-5`, `atol=1e-8`;
/// NaNs are unequal. Shape and dtype must always match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloseOptions {
    /// Nonnegative finite relative tolerance.
    pub rtol: f64,
    /// Nonnegative finite absolute tolerance.
    pub atol: f64,
    /// Treat NaNs at matching locations as equal.
    pub equal_nan: bool,
    /// Require the same device; otherwise compare detached CPU copies.
    pub check_device: bool,
}

impl Default for CloseOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-8,
            equal_nan: false,
            check_device: true,
        }
    }
}

/// Value diagnostics from [`compare`], without retaining tensors or their graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// Number of scalar entries examined.
    pub elements: usize,
    /// Number of entries outside the requested tolerance.
    pub mismatches: usize,
    /// Coordinates of the first mismatching element; a scalar has empty coordinates.
    pub first_mismatch: Option<Vec<i64>>,
}

/// Compares two dense tensors and returns mismatch counts and the first location.
///
/// Undefined tensors, sparse layouts, mismatched shape/dtype/device and invalid
/// tolerances return errors. Empty tensors with matching metadata compare equal.
/// Operations run without recording gradients. Set `check_device=false` for
/// a deliberate CPU-versus-accelerator correctness comparison.
pub fn compare(actual: &Tensor, expected: &Tensor, options: CloseOptions) -> Result<Comparison> {
    validate_tolerances(options.rtol, options.atol)?;
    if !actual.defined() || !expected.defined() {
        return Err(invalid("comparison inputs must be defined"));
    }
    if actual.size() != expected.size() || actual.kind() != expected.kind() {
        return Err(invalid(&format!(
            "shape and dtype must match: {:?}/{:?} versus {:?}/{:?}",
            actual.size(),
            actual.kind(),
            expected.size(),
            expected.kind()
        )));
    }
    if options.check_device && actual.device() != expected.device() {
        return Err(invalid(
            "comparison devices differ; set check_device=false explicitly",
        ));
    }
    let kind = actual.kind();
    let floating = matches!(
        kind,
        Kind::Float
            | Kind::Double
            | Kind::Half
            | Kind::BFloat16
            | Kind::ComplexFloat
            | Kind::ComplexDouble
    );
    if !floating
        && !matches!(
            kind,
            Kind::Uint8 | Kind::Int8 | Kind::Int16 | Kind::Int | Kind::Int64 | Kind::Bool
        )
    {
        return Err(invalid(
            "comparison requires ordinary dense real, complex, integer or Boolean tensors",
        ));
    }
    crate::no_grad(|| {
        let prepare = |x: &Tensor| -> Result<Tensor> {
            let _ = x.f_as_strided([0], [1], None)?;
            let x = x.f_detach()?;
            Ok(if options.check_device {
                x
            } else {
                x.f_to_device(Device::Cpu)?
            })
        };
        let actual = prepare(actual)?;
        let expected = prepare(expected)?;
        let equal = if floating {
            actual.f_isclose(&expected, options.rtol, options.atol, options.equal_nan)?
        } else {
            actual.f_eq_tensor(&expected)?
        };
        let mismatch = equal.f_logical_not()?;
        let mismatches = usize::try_from(mismatch.f_sum(Kind::Int64)?.f_int64_value(&[])?)
            .map_err(|_| invalid("comparison count exceeds usize"))?;
        let first_mismatch = if mismatches == 0 {
            None
        } else {
            Some(Vec::<i64>::try_from(
                &mismatch
                    .f_nonzero()?
                    .f_select(0, 0)?
                    .f_to_device(Device::Cpu)?,
            )?)
        };
        Ok(Comparison {
            elements: actual.numel(),
            mismatches,
            first_mismatch,
        })
    })
}

/// Returns an error describing the first mismatch instead of panicking.
///
/// Use `assert_close(...)?` in a fallible integration test. The module example
/// shows an exact result comparison and a derivative check.
pub fn assert_close(actual: &Tensor, expected: &Tensor, options: CloseOptions) -> Result<()> {
    let report = compare(actual, expected, options)?;
    if report.mismatches != 0 {
        return Err(invalid(&format!(
            "{} of {} values differ; first mismatch at {:?}",
            report.mismatches,
            report.elements,
            report.first_mismatch.as_ref().unwrap()
        )));
    }
    Ok(())
}

/// Numerical settings for a small deterministic derivative check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradcheckOptions {
    /// Positive finite central-difference step, default `1e-6`.
    pub epsilon: f64,
    /// Relative derivative tolerance, default `1e-3`.
    pub rtol: f64,
    /// Absolute derivative tolerance, default `1e-5`.
    pub atol: f64,
    /// Maximum number of Jacobian entries, default 16,384.
    pub max_jacobian_elements: usize,
}

impl Default for GradcheckOptions {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            rtol: 1e-3,
            atol: 1e-5,
            max_jacobian_elements: 16_384,
        }
    }
}

/// Checks the full Jacobian of a deterministic function by central differences.
///
/// The input, output and derivatives must be finite, nonempty, dense CPU Float64 tensors.
/// The function must be pure: it can be evaluated repeatedly, must not mutate
/// inputs, and must return the same shape and values for a repeated input.
/// The input is copied and its existing gradients are untouched. Sparse,
/// complex, stochastic and nondifferentiable functions need separate checks.
/// Work is bounded by `max_jacobian_elements`; larger models should test a
/// representative small input or a directional derivative.
pub fn gradcheck<F>(function: F, input: &Tensor, options: GradcheckOptions) -> Result<()>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    validate_tolerances(options.rtol, options.atol)?;
    if !(options.epsilon * 2.).is_finite()
        || options.epsilon <= 0.
        || options.max_jacobian_elements == 0
    {
        return Err(invalid(
            "gradient-check epsilon and allocation limit must be positive and finite",
        ));
    }
    let validate = |x: &Tensor| -> Result<()> {
        if !x.defined() || x.kind() != Kind::Double || x.device() != Device::Cpu || x.numel() == 0 {
            return Err(invalid(
                "gradient checks require nonempty CPU Float64 tensors",
            ));
        }
        let _ = x.f_as_strided([0], [1], None)?;
        if x.f_isfinite()?.f_all()?.f_int64_value(&[])? == 0 {
            return Err(invalid("gradient-check inputs and outputs must be finite"));
        }
        Ok(())
    };
    validate(input)?;
    if input.numel() > options.max_jacobian_elements {
        return Err(invalid(
            "gradient-check input exceeds the Jacobian allocation limit",
        ));
    }
    let x = copied(&input.f_detach()?.f_contiguous()?)?;
    let output = function(&x)?;
    validate(&output)?;
    let entries = x
        .numel()
        .checked_mul(output.numel())
        .filter(|&n| n <= options.max_jacobian_elements)
        .ok_or_else(|| invalid("gradient-check Jacobian exceeds the allocation limit"))?;
    let exact = CloseOptions {
        rtol: 0.,
        atol: 0.,
        ..Default::default()
    };
    assert_close(&function(&x)?, &output, exact)?;
    let analytical = autograd::jacobian(|arg| function(arg), &x, false)?;
    validate(&analytical)?;
    let mut numerical = vec![0_f64; entries];
    let flat = x.f_reshape([-1])?;
    for column in 0..x.numel() {
        let center = flat.f_double_value(&[column as i64])?;
        if center + options.epsilon == center
            || center - options.epsilon == center
            || !(center + options.epsilon).is_finite()
            || !(center - options.epsilon).is_finite()
        {
            return Err(invalid(
                "finite-difference step is not representable at this input value",
            ));
        }
        let evaluate = |step: f64| -> Result<Tensor> {
            let perturbed = copied(&x)?;
            let _ = perturbed
                .f_reshape([-1])?
                .f_select(0, column as i64)?
                .f_fill_(center + step)?;
            let value = function(&perturbed)?;
            validate(&value)?;
            if value.size() != output.size() {
                return Err(invalid("gradient-check function changes its output shape"));
            }
            Ok(value.f_detach()?)
        };
        let derivative = evaluate(options.epsilon)?
            .f_sub(&evaluate(-options.epsilon)?)?
            .f_div_scalar(2. * options.epsilon)?
            .f_reshape([-1])?;
        validate(&derivative)?;
        for (row, value) in Vec::<f64>::try_from(&derivative)?.into_iter().enumerate() {
            numerical[row * x.numel() + column] = value;
        }
    }
    let numerical = Tensor::f_from_slice(&numerical)?.f_reshape(analytical.size())?;
    assert_close(
        &analytical,
        &numerical,
        CloseOptions {
            rtol: options.rtol,
            atol: options.atol,
            ..Default::default()
        },
    )
}

fn validate_tolerances(rtol: f64, atol: f64) -> Result<()> {
    if !rtol.is_finite() || !atol.is_finite() || rtol < 0. || atol < 0. {
        Err(invalid(
            "comparison tolerances must be finite and nonnegative",
        ))
    } else {
        Ok(())
    }
}

fn copied(input: &Tensor) -> Result<Tensor> {
    let mut copy = input.f_zeros_like()?;
    copy.f_copy_(input)?;
    Ok(copy)
}

fn invalid(reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field: "testing",
        reason: reason.into(),
    }
}
