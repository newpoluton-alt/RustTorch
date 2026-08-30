//! Fallible functional operations delegated to LibTorch.

use tch::{Reduction, Tensor};

use crate::{Result, RustTorchError, device::ensure_device};

use super::GeluApproximation;

/// Applies a linear transformation to the last input dimension.
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    ensure_device("linear weight", weight, input.device())?;
    if weight.size().len() != 2 {
        return Err(RustTorchError::InvalidDimensions {
            context: "linear weight".to_owned(),
            expected: "rank 2 [out_features, in_features]".to_owned(),
            actual: format!("shape {:?}", weight.size()),
        });
    }
    let input_shape = input.size();
    let Some(&input_features) = input_shape.last() else {
        return Err(RustTorchError::InvalidDimensions {
            context: "linear input".to_owned(),
            expected: "rank at least 1".to_owned(),
            actual: "scalar".to_owned(),
        });
    };
    if input_features != weight.size()[1] {
        return Err(RustTorchError::InvalidDimensions {
            context: "linear input".to_owned(),
            expected: format!("last dimension {}", weight.size()[1]),
            actual: input_features.to_string(),
        });
    }
    if let Some(bias) = bias {
        ensure_device("linear bias", bias, input.device())?;
        if bias.size() != [weight.size()[0]] {
            return Err(RustTorchError::ShapeMismatch {
                name: "linear bias".to_owned(),
                expected: vec![weight.size()[0]],
                actual: bias.size(),
            });
        }
    }
    input.f_linear(weight, bias).map_err(Into::into)
}

/// Applies an element-wise rectified linear unit.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    input.f_relu().map_err(Into::into)
}

/// Applies exact element-wise Gaussian error linear units.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    gelu_with_approximation(input, GeluApproximation::None)
}

/// Applies element-wise Gaussian error linear units with an approximation mode.
pub fn gelu_with_approximation(input: &Tensor, approximation: GeluApproximation) -> Result<Tensor> {
    input.f_gelu(approximation.as_str()).map_err(Into::into)
}

/// Applies dropout when `training` is true and validates `probability` in `[0, 1]`.
pub fn dropout(input: &Tensor, probability: f64, training: bool) -> Result<Tensor> {
    validate_dropout(probability)?;
    input.f_dropout(probability, training).map_err(Into::into)
}

pub(crate) fn validate_dropout(probability: f64) -> Result<()> {
    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
        Ok(())
    } else {
        Err(RustTorchError::InvalidConfiguration {
            field: "dropout probability",
            reason: format!("must be finite and in [0, 1], got {probability}"),
        })
    }
}

/// Flattens the inclusive range from `start_dim` through `end_dim`.
pub fn flatten(input: &Tensor, start_dim: i64, end_dim: i64) -> Result<Tensor> {
    input.f_flatten(start_dim, end_dim).map_err(Into::into)
}

/// Computes mean squared error with mean reduction.
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    ensure_device("MSE target", target, input.device())?;
    input
        .f_mse_loss(target, Reduction::Mean)
        .map_err(Into::into)
}

/// Computes cross-entropy loss with mean reduction and ignore index `-100`.
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    ensure_device("cross-entropy target", target, input.device())?;
    input
        .f_cross_entropy_loss::<&Tensor>(target, None, Reduction::Mean, -100, 0.0)
        .map_err(Into::into)
}
