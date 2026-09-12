//! Configurable objectives for regression and classification.

use super::functional::input_device;
use crate::{Reduction, Result, RustTorchError, Tensor, device::ensure_device};

/// Options for class-index or probability-target cross-entropy.
///
/// Class weights have shape `[classes]`. Class-index targets use integer labels
/// and may contain `ignore_index`; probability targets match the logits shape.
/// Defaults compute the mean loss with no weighting or smoothing.
///
/// ```
/// use rusttorch::{Tensor, nn::functional::{CrossEntropyOptions, cross_entropy_with_options}};
/// let logits = Tensor::from_slice(&[2_f32, -1., 0., 1.]).reshape([2, 2]);
/// let classes = Tensor::from_slice(&[0_i64, 1]);
/// let options = CrossEntropyOptions { label_smoothing: 0.1, ..Default::default() };
/// assert!(cross_entropy_with_options(&logits, &classes, options)?.double_value(&[]) > 0.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyOptions<'a> {
    /// Optional per-class importance weights on the logits device.
    pub weight: Option<&'a Tensor>,
    /// Elementwise, summed or averaged loss; defaults to [`Reduction::Mean`].
    pub reduction: Reduction,
    /// Class label excluded from loss and gradients; defaults to `-100`.
    pub ignore_index: i64,
    /// Probability mass mixed uniformly across classes, in `[0, 1]`.
    pub label_smoothing: f64,
}

impl Default for CrossEntropyOptions<'_> {
    fn default() -> Self {
        Self {
            weight: None,
            reduction: Reduction::Mean,
            ignore_index: -100,
            label_smoothing: 0.0,
        }
    }
}

/// Computes cross-entropy with weights, ignored labels and label smoothing.
///
/// Pass raw logits, not softmax probabilities. See [`CrossEntropyOptions`] for
/// an example. Invalid options, tensor devices/shapes and native errors return
/// `Err`; probability target normalization remains the caller's responsibility.
pub fn cross_entropy_with_options(
    input: &Tensor,
    target: &Tensor,
    options: CrossEntropyOptions<'_>,
) -> Result<Tensor> {
    validate(input, target, options.reduction)?;
    if !options.label_smoothing.is_finite() || !(0.0..=1.0).contains(&options.label_smoothing) {
        return Err(invalid("label_smoothing", "must be finite and in [0, 1]"));
    }
    if let Some(weight) = options.weight {
        ensure_device("class weights", weight, input_device(input)?)?;
    }
    input
        .f_cross_entropy_loss(
            target,
            options.weight,
            options.reduction,
            options.ignore_index,
            options.label_smoothing,
        )
        .map_err(Into::into)
}

/// Computes squared regression error with the selected reduction.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::mse_loss_with_reduction};
/// let input = Tensor::from_slice(&[1_f32, 3.]);
/// let target = Tensor::from_slice(&[0_f32, 1.]);
/// assert_eq!(mse_loss_with_reduction(&input, &target, Reduction::Sum)?.double_value(&[]), 5.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn mse_loss_with_reduction(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
) -> Result<Tensor> {
    validate(input, target, reduction)?;
    input.f_mse_loss(target, reduction).map_err(Into::into)
}

/// Computes absolute prediction error, useful when large errors should grow linearly.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::l1_loss};
/// let loss = l1_loss(&Tensor::from_slice(&[1_f32, 3.]), &Tensor::from_slice(&[0_f32, 1.]), Reduction::Mean)?;
/// assert_eq!(loss.double_value(&[]), 1.5);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    validate(input, target, reduction)?;
    input.f_l1_loss(target, reduction).map_err(Into::into)
}

/// Uses quadratic error below `beta` and linear error above it.
///
/// Useful for robust regression. `beta` must be finite and nonnegative; zero
/// gives absolute error. Broadcasting follows the native tensor operation.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::smooth_l1_loss};
/// let loss = smooth_l1_loss(&Tensor::from_slice(&[2_f32]), &Tensor::from_slice(&[0_f32]), 1.0, Reduction::Mean)?;
/// assert_eq!(loss.double_value(&[]), 1.5);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn smooth_l1_loss(
    input: &Tensor,
    target: &Tensor,
    beta: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    validate(input, target, reduction)?;
    if !beta.is_finite() || beta < 0.0 {
        return Err(invalid("beta", "must be finite and nonnegative"));
    }
    input
        .f_smooth_l1_loss(target, reduction, beta)
        .map_err(Into::into)
}

/// Computes Huber error with quadratic/linear transition at positive `delta`.
///
/// Unlike smooth L1, the linear region's slope is `delta`.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::huber_loss};
/// let loss = huber_loss(&Tensor::from_slice(&[3_f32]), &Tensor::from_slice(&[0_f32]), 2.0, Reduction::Mean)?;
/// assert_eq!(loss.double_value(&[]), 4.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn huber_loss(
    input: &Tensor,
    target: &Tensor,
    delta: f64,
    reduction: Reduction,
) -> Result<Tensor> {
    validate(input, target, reduction)?;
    if !delta.is_finite() || delta <= 0.0 {
        return Err(invalid("delta", "must be finite and positive"));
    }
    input
        .f_huber_loss(target, reduction, delta)
        .map_err(Into::into)
}

/// Computes binary cross-entropy from probabilities in `[0, 1]`.
///
/// Use [`binary_cross_entropy_with_logits`] when a model produces raw scores:
/// its combined operation is more numerically stable. Inputs and targets must
/// have identical shapes; an optional weight may broadcast over them.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::binary_cross_entropy};
/// let loss = binary_cross_entropy(&Tensor::from_slice(&[0.5_f32]), &Tensor::from_slice(&[1_f32]), None, Reduction::Mean)?;
/// assert!((loss.double_value(&[]) - 2_f64.ln()).abs() < 1e-6);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn binary_cross_entropy(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Result<Tensor> {
    validate_binary(input, target, weight, reduction)?;
    input
        .f_binary_cross_entropy(target, weight, reduction)
        .map_err(Into::into)
}

/// Computes stable binary classification loss directly from raw logits.
///
/// `pos_weight` weights positive examples and is useful for imbalanced labels;
/// for `[batch, classes]` targets a `[classes]` vector weights each class.
/// `weight` scales individual losses. Both may follow native broadcasting.
///
/// ```
/// use rusttorch::{Tensor, Reduction, nn::functional::binary_cross_entropy_with_logits};
/// let input = Tensor::from_slice(&[0_f32, 2.]);
/// let target = Tensor::from_slice(&[0_f32, 1.]);
/// let loss = binary_cross_entropy_with_logits(&input, &target, None, None, Reduction::Mean)?;
/// assert!(loss.double_value(&[]) < 0.5);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn binary_cross_entropy_with_logits(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    pos_weight: Option<&Tensor>,
    reduction: Reduction,
) -> Result<Tensor> {
    validate_binary(input, target, weight, reduction)?;
    if let Some(weight) = pos_weight {
        ensure_device("positive weights", weight, input_device(input)?)?;
    }
    input
        .f_binary_cross_entropy_with_logits(target, weight, pos_weight, reduction)
        .map_err(Into::into)
}

/// Computes negative log likelihood from log-probabilities and integer labels.
///
/// Supports unbatched, batched and spatial classification, optional class
/// weights, and an ignored label. For raw logits use cross-entropy instead.
///
/// ```
/// use rusttorch::{Tensor, Kind, Reduction, nn::functional::nll_loss};
/// let scores = Tensor::from_slice(&[2_f32, -1.]).reshape([1, 2]).log_softmax(-1, Kind::Float);
/// let classes = Tensor::from_slice(&[0_i64]);
/// assert!(nll_loss(&scores, &classes, None, -100, Reduction::Mean)?.double_value(&[]) < 0.1);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    ignore_index: i64,
    reduction: Reduction,
) -> Result<Tensor> {
    validate(input, target, reduction)?;
    if let Some(weight) = weight {
        ensure_device("class weights", weight, input_device(input)?)?;
    }
    input
        .f_nll_loss_nd(target, weight, reduction, ignore_index)
        .map_err(Into::into)
}

fn validate(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<()> {
    ensure_device("loss target", target, input_device(input)?)?;
    if !matches!(
        reduction,
        Reduction::None | Reduction::Mean | Reduction::Sum
    ) {
        return Err(invalid("reduction", "must be None, Mean or Sum"));
    }
    Ok(())
}

fn validate_binary(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Result<()> {
    validate(input, target, reduction)?;
    if input.size() != target.size() {
        return Err(RustTorchError::ShapeMismatch {
            name: "binary loss target".into(),
            expected: input.size(),
            actual: target.size(),
        });
    }
    if let Some(weight) = weight {
        ensure_device("loss weight", weight, input_device(input)?)?;
    }
    Ok(())
}

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}
