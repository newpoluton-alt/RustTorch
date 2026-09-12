//! Apply layers, activations, and losses directly to tensors.
//!
//! Functional operations are useful when a model has residual branches, shared
//! weights, or a custom training loop. They preserve autograd tracking and return
//! [`Result`] when an operation fails. They do not own or register parameters;
//! use the layer configurations in [`super`] when weights should belong to a model.
//!
//! ```
//! use rusttorch::{Device, Tensor, nn::{VarStore, linear, functional}};
//! let store = VarStore::new(Device::Cpu);
//! let head = linear(&store.root(), 3, 2)?;
//! let features = Tensor::from_slice(&[0.2_f32, 1.0, -0.3]).reshape([1, 3]);
//! let logits = head.forward(&features)?;
//! let labels = Tensor::from_slice(&[1_i64]);
//! let loss = functional::cross_entropy(&logits, &labels)?;
//! loss.f_backward()?;
//! assert!(head.weight().grad().defined());
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

use tch::{Reduction, Tensor};

use crate::{Result, RustTorchError, device::ensure_device};

use super::GeluApproximation;

/// Transforms the last input dimension with `input * weight.T + bias`.
///
/// Weights must be `[out_features, in_features]`, the optional bias must be
/// `[out_features]`, and input must end in `in_features`. Leading input
/// dimensions are preserved. Shape, device, and backend failures return errors.
///
/// ```
/// use rusttorch::{Tensor, nn::functional};
/// let input = Tensor::from_slice(&[1_f32, 2.]).reshape([1, 2]);
/// let weight = Tensor::from_slice(&[3_f32, 4.]).reshape([1, 2]);
/// assert_eq!(functional::linear(&input, &weight, None)?.double_value(&[0, 0]), 11.);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    ensure_device("linear weight", weight, input_device(input)?)?;
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
        ensure_device("linear bias", bias, input_device(input)?)?;
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

/// Replaces negative elements with zero and keeps the input shape.
///
/// Use ReLU between learned transformations to introduce a nonlinearity.
///
/// ```
/// let input = rusttorch::Tensor::from_slice(&[-2_f32, 0., 3.]);
/// let output = rusttorch::nn::functional::relu(&input)?;
/// assert_eq!(Vec::<f32>::try_from(&output)?, [0., 0., 3.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn relu(input: &Tensor) -> Result<Tensor> {
    input.f_relu().map_err(Into::into)
}

/// Applies the smooth activation `x * P(Z <= x)` for standard normal `Z`.
///
/// GELU is useful between projections in token and feature models. The output
/// keeps the input shape; use [`gelu_with_approximation`] for the tanh formula.
///
/// ```
/// let input = rusttorch::Tensor::from_slice(&[-1_f32, 0., 1.]);
/// let output = rusttorch::nn::functional::gelu(&input)?;
/// assert_eq!(output.size(), [3]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    gelu_with_approximation(input, GeluApproximation::None)
}

/// Applies GELU using an explicit exact or tanh approximation formula.
///
/// ```
/// use rusttorch::{Tensor, nn::{GeluApproximation, functional}};
/// let input = Tensor::from_slice(&[-1_f32, 0., 1.]);
/// let output = functional::gelu_with_approximation(&input, GeluApproximation::Tanh)?;
/// assert_eq!(output.size(), input.size());
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn gelu_with_approximation(input: &Tensor, approximation: GeluApproximation) -> Result<Tensor> {
    input.f_gelu(approximation.as_str()).map_err(Into::into)
}

/// Randomly zeroes elements during training and rescales the surviving elements.
///
/// Evaluation returns the input unchanged. A probability of zero disables
/// dropout; one drops every element during training. Non-finite probabilities
/// and values outside `[0, 1]` return errors.
///
/// ```
/// let input = rusttorch::Tensor::from_slice(&[1_f32, 2., 3.]);
/// let output = rusttorch::nn::functional::dropout(&input, 0.5, false)?;
/// assert_eq!(Vec::<f32>::try_from(&output)?, [1., 2., 3.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
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

/// Combines an inclusive range of dimensions into one dimension.
///
/// Negative dimensions count from the end. Use `(1, -1)` to preserve the batch
/// while preparing image or sequence features for a linear layer. Invalid
/// dimension ranges return errors.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::functional};
/// let image_features = Tensor::zeros([2, 8, 4, 4], (Kind::Float, Device::Cpu));
/// assert_eq!(functional::flatten(&image_features, 1, -1)?.size(), [2, 128]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn flatten(input: &Tensor, start_dim: i64, end_dim: i64) -> Result<Tensor> {
    input.f_flatten(start_dim, end_dim).map_err(Into::into)
}

/// Averages squared prediction errors for regression training.
///
/// The result is a scalar. Targets must share the input device and have a
/// broadcast-compatible shape. Prefer identical shapes to avoid unintentionally
/// comparing every prediction with several targets.
///
/// ```
/// use rusttorch::{Tensor, nn::functional};
/// let predictions = Tensor::from_slice(&[1_f32, 3.]);
/// let targets = Tensor::from_slice(&[0_f32, 1.]);
/// assert_eq!(functional::mse_loss(&predictions, &targets)?.double_value(&[]), 2.5);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    ensure_device("MSE target", target, input_device(input)?)?;
    input
        .f_mse_loss(target, Reduction::Mean)
        .map_err(Into::into)
}

/// Computes the mean classification loss from unnormalized class scores.
///
/// For batched classification, use scores `[batch, classes]` and integer class
/// indices `[batch]`. Do not apply softmax first. The default ignore index is
/// `-100`, label smoothing is zero, and no class weights are applied. Targets
/// must share the scores' device; invalid targets and shapes return errors.
///
/// ```
/// use rusttorch::{Tensor, nn::functional};
/// let scores = Tensor::from_slice(&[2_f32, -1.]).reshape([1, 2]);
/// let classes = Tensor::from_slice(&[0_i64]);
/// let loss = functional::cross_entropy(&scores, &classes)?;
/// assert!(loss.double_value(&[]) < 0.1);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    ensure_device("cross-entropy target", target, input_device(input)?)?;
    input
        .f_cross_entropy_loss::<&Tensor>(target, None, Reduction::Mean, -100, 0.0)
        .map_err(Into::into)
}

/// Applies a sequence convolution with weights `[out_channels, in_channels / groups, kernel]`.
///
/// Input may be `[batch, channels, length]` or `[channels, length]`. Stride and
/// dilation must be positive, padding nonnegative, and groups must divide both
/// channel counts. Invalid shapes, devices, and backend failures return errors.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::functional};
/// let input = Tensor::ones([1, 1, 5], (Kind::Float, Device::Cpu));
/// let weight = Tensor::ones([2, 1, 3], (Kind::Float, Device::Cpu));
/// assert_eq!(functional::conv1d(&input, &weight, None, [1], [0], [1], 1)?.size(), [1, 2, 3]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [i64; 1],
    padding: [i64; 1],
    dilation: [i64; 1],
    groups: i64,
) -> Result<Tensor> {
    convolution(input, weight, bias, &stride, &padding, &dilation, groups)
}

/// Applies an image convolution with weights `[out_channels, in_channels / groups, height, width]`.
///
/// Input may be `[batch, channels, height, width]` or omit the batch dimension.
/// Padding adds zeros symmetrically per axis. See [`conv1d`] for validation rules.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::functional};
/// let input = Tensor::ones([1, 3, 8, 8], (Kind::Float, Device::Cpu));
/// let weight = Tensor::ones([4, 3, 3, 3], (Kind::Float, Device::Cpu));
/// assert_eq!(functional::conv2d(&input, &weight, None, [1, 1], [1, 1], [1, 1], 1)?.size(), [1, 4, 8, 8]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
) -> Result<Tensor> {
    convolution(input, weight, bias, &stride, &padding, &dilation, groups)
}

/// Applies a volume convolution with weights `[out_channels, in_channels / groups, depth, height, width]`.
///
/// Input may be `[batch, channels, depth, height, width]` or omit the batch
/// dimension. See [`conv1d`] for validation rules.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::functional};
/// let input = Tensor::ones([1, 1, 4, 4, 4], (Kind::Float, Device::Cpu));
/// let weight = Tensor::ones([2, 1, 2, 2, 2], (Kind::Float, Device::Cpu));
/// assert_eq!(functional::conv3d(&input, &weight, None, [1; 3], [0; 3], [1; 3], 1)?.size(), [1, 2, 3, 3, 3]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: [i64; 3],
    padding: [i64; 3],
    dilation: [i64; 3],
    groups: i64,
) -> Result<Tensor> {
    convolution(input, weight, bias, &stride, &padding, &dilation, groups)
}

pub(crate) fn validate_conv_config(
    in_channels: i64,
    out_channels: i64,
    kernel: &[i64],
    stride: &[i64],
    padding: &[i64],
    dilation: &[i64],
    groups: i64,
) -> Result<()> {
    if !(1..=3).contains(&kernel.len()) {
        return Err(invalid(
            "convolution dimensions",
            "must be one, two, or three",
        ));
    }
    if in_channels <= 0 || out_channels <= 0 {
        return Err(invalid(
            "channels",
            "input and output channels must be positive",
        ));
    }
    if groups <= 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(invalid(
            "groups",
            "must be positive and divide both channel counts",
        ));
    }
    for (field, values) in [
        ("kernel_size", kernel),
        ("stride", stride),
        ("dilation", dilation),
    ] {
        if values.len() != kernel.len() || values.iter().any(|&value| value <= 0) {
            return Err(invalid(
                field,
                "must have one positive value per spatial dimension",
            ));
        }
    }
    if padding.len() != kernel.len() || padding.iter().any(|&value| value < 0) {
        return Err(invalid(
            "padding",
            "must have one nonnegative value per spatial dimension",
        ));
    }
    Ok(())
}

pub(crate) fn convolution(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: &[i64],
    padding: &[i64],
    dilation: &[i64],
    groups: i64,
) -> Result<Tensor> {
    ensure_device("convolution weight", weight, input_device(input)?)?;
    let shape = weight.size();
    let dimensions = stride.len();
    if shape.len() != dimensions + 2 {
        return Err(RustTorchError::InvalidDimensions {
            context: "convolution weight".to_owned(),
            expected: format!("rank {}", dimensions + 2),
            actual: format!("shape {shape:?}"),
        });
    }
    let in_channels = shape[1]
        .checked_mul(groups)
        .ok_or_else(|| invalid("groups", "input channels exceed i64"))?;
    validate_conv_config(
        in_channels,
        shape[0],
        &shape[2..],
        stride,
        padding,
        dilation,
        groups,
    )?;
    let input_shape = input.size();
    if ![dimensions + 1, dimensions + 2].contains(&input_shape.len()) {
        return Err(RustTorchError::InvalidDimensions {
            context: "convolution input".to_owned(),
            expected: format!("rank {} or {}", dimensions + 1, dimensions + 2),
            actual: format!("shape {input_shape:?}"),
        });
    }
    if input_shape[input_shape.len() - dimensions - 1] != in_channels {
        return Err(invalid(
            "input channels",
            "must match weight channels multiplied by groups",
        ));
    }
    if let Some(bias) = bias {
        ensure_device("convolution bias", bias, input_device(input)?)?;
        if bias.size() != [shape[0]] {
            return Err(RustTorchError::ShapeMismatch {
                name: "convolution bias".to_owned(),
                expected: vec![shape[0]],
                actual: bias.size(),
            });
        }
    }
    match dimensions {
        1 => input.f_conv1d(weight, bias, stride, padding, dilation, groups),
        2 => input.f_conv2d(weight, bias, stride, padding, dilation, groups),
        3 => input.f_conv3d(weight, bias, stride, padding, dilation, groups),
        _ => {
            return Err(invalid(
                "convolution dimensions",
                "must be one, two, or three",
            ));
        }
    }
    .map_err(Into::into)
}

/// Normalizes `normalized_shape` trailing dimensions using their biased variance.
///
/// Each trailing dimension must match the shape. Optional scale and bias must
/// have that exact shape and share the input's device. Epsilon must be finite
/// and nonnegative. The output keeps the input shape; invalid settings and
/// backend failures return errors.
///
/// ```
/// use rusttorch::{Tensor, nn::functional};
/// let input = Tensor::from_slice(&[1_f32, 3., 2., 4.]).reshape([2, 2]);
/// let output = functional::layer_norm(&input, &[2], None, None, 1e-5)?;
/// assert_eq!(output.size(), [2, 2]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[i64],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    validate_layer_norm(normalized_shape, eps)?;
    for (name, tensor) in [("layer norm weight", weight), ("layer norm bias", bias)] {
        if let Some(tensor) = tensor {
            ensure_device(name, tensor, input_device(input)?)?;
            if tensor.size() != normalized_shape {
                return Err(RustTorchError::ShapeMismatch {
                    name: name.to_owned(),
                    expected: normalized_shape.to_vec(),
                    actual: tensor.size(),
                });
            }
        }
    }
    input
        .f_layer_norm(normalized_shape, weight, bias, eps, true)
        .map_err(Into::into)
}

pub(crate) fn validate_layer_norm(normalized_shape: &[i64], eps: f64) -> Result<()> {
    if normalized_shape.is_empty() || normalized_shape.iter().any(|&dimension| dimension <= 0) {
        return Err(invalid(
            "normalized_shape",
            "must contain at least one positive dimension",
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(invalid("eps", "must be finite and nonnegative"));
    }
    Ok(())
}

/// Looks up integer token IDs in a weight matrix `[vocabulary, vector_width]`.
///
/// The output appends the vector width to the input shape. `padding_idx`
/// suppresses that row's gradient, but does not modify its stored values.
/// Negative padding indices count from the vocabulary's end. Frequency scaling
/// divides dense gradients by each token's input count; it cannot be combined
/// with sparse gradients. Invalid options, indices, dtypes, and devices return
/// errors. This function does not clip vector norms.
///
/// ```
/// use rusttorch::{Tensor, nn::functional};
/// let weights = Tensor::from_slice(&[1_f32, 2., 3., 4.]).reshape([2, 2]);
/// let ids = Tensor::from_slice(&[1_i64, 0]);
/// assert_eq!(functional::embedding(&ids, &weights, None, false, false)?.size(), [2, 2]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub fn embedding(
    input: &Tensor,
    weight: &Tensor,
    padding_idx: Option<i64>,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> Result<Tensor> {
    ensure_device("embedding weight", weight, input_device(input)?)?;
    let shape = weight.size();
    if shape.len() != 2 || shape[0] <= 0 || shape[1] <= 0 {
        return Err(invalid(
            "embedding weight",
            "must have shape [positive vocabulary size, positive vector width]",
        ));
    }
    let padding_idx =
        validate_embedding_options(shape[0], padding_idx, scale_grad_by_freq, sparse)?;
    Tensor::f_embedding(
        weight,
        input,
        padding_idx.unwrap_or(-1),
        scale_grad_by_freq,
        sparse,
    )
    .map_err(Into::into)
}

pub(crate) fn validate_embedding_options(
    num_embeddings: i64,
    padding_idx: Option<i64>,
    scale_grad_by_freq: bool,
    sparse: bool,
) -> Result<Option<i64>> {
    if sparse && scale_grad_by_freq {
        return Err(invalid(
            "scale_grad_by_freq",
            "is not supported with sparse embedding gradients",
        ));
    }
    padding_idx
        .map(|index| {
            if index < -num_embeddings || index >= num_embeddings {
                Err(invalid(
                    "padding_idx",
                    "must be within the vocabulary, allowing negative indexing",
                ))
            } else {
                Ok(if index < 0 {
                    index + num_embeddings
                } else {
                    index
                })
            }
        })
        .transpose()
}

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}

fn input_device(input: &Tensor) -> Result<tch::Device> {
    if input.defined() {
        Ok(input.device())
    } else {
        Err(RustTorchError::InvalidDimensions {
            context: "functional input".to_owned(),
            expected: "a defined tensor".to_owned(),
            actual: "undefined tensor".to_owned(),
        })
    }
}
