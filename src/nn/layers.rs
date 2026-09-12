//! Convolution, feature normalization, and token lookup layers.

use tch::{Tensor, nn::Init};

use crate::{Result, RustTorchError};

use super::{Module, ParameterPath, functional};

/// Configures a convolution over one, two, or three spatial dimensions.
///
/// Use convolutions to learn local patterns in sequences, images, or volumes.
/// The input has shape `[batch, channels, spatial...]`; an unbatched input is
/// also accepted. Kernels, strides, padding, and dilation use one value per
/// spatial axis. Padding adds zeros symmetrically on both sides of each axis.
/// Defaults are stride one, no padding, dilation one, one group, and bias.
///
/// Weights and biases start uniformly in `[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]`,
/// where `fan_in = in_channels / groups * product(kernel_size)`.
/// String padding modes, nonzero padding modes, and transposed convolutions are
/// not configured by this type.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{ConvConfig, Module}};
/// let store = rusttorch::nn::VarStore::new(Device::Cpu);
/// let layer = ConvConfig::new(3, 8, [3, 3]).padding([1, 1]).build(&store.root())?;
/// let image = Tensor::zeros([2, 3, 16, 16], (Kind::Float, Device::Cpu));
/// assert_eq!(layer.forward(&image)?.size(), [2, 8, 16, 16]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvConfig<const D: usize> {
    in_channels: i64,
    out_channels: i64,
    kernel_size: [i64; D],
    stride: [i64; D],
    padding: [i64; D],
    dilation: [i64; D],
    groups: i64,
    bias: bool,
}

impl<const D: usize> ConvConfig<D> {
    /// Sets input channels, output channels, and kernel size.
    ///
    /// ```
    /// let config = rusttorch::nn::ConvConfig::new(3, 16, [3, 3]);
    /// ```
    pub const fn new(in_channels: i64, out_channels: i64, kernel_size: [i64; D]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: [1; D],
            padding: [0; D],
            dilation: [1; D],
            groups: 1,
            bias: true,
        }
    }

    /// Sets the positive step between kernel positions on each spatial axis.
    ///
    /// ```
    /// let config = rusttorch::nn::ConvConfig::new(3, 16, [3, 3]).stride([2, 2]);
    /// ```
    #[must_use]
    pub const fn stride(mut self, stride: [i64; D]) -> Self {
        self.stride = stride;
        self
    }

    /// Sets nonnegative symmetric zero padding on each spatial axis.
    ///
    /// ```
    /// let config = rusttorch::nn::ConvConfig::new(3, 16, [3, 3]).padding([1, 1]);
    /// ```
    #[must_use]
    pub const fn padding(mut self, padding: [i64; D]) -> Self {
        self.padding = padding;
        self
    }

    /// Sets positive spacing between kernel elements on each spatial axis.
    ///
    /// ```
    /// let config = rusttorch::nn::ConvConfig::new(3, 16, [3, 3]).dilation([2, 2]);
    /// ```
    #[must_use]
    pub const fn dilation(mut self, dilation: [i64; D]) -> Self {
        self.dilation = dilation;
        self
    }

    /// Splits channels into independent groups; both channel counts must divide evenly.
    ///
    /// ```
    /// let depthwise = rusttorch::nn::ConvConfig::new(8, 8, [3, 3]).groups(8);
    /// ```
    #[must_use]
    pub const fn groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }

    /// Enables or disables the additive bias parameter.
    ///
    /// ```
    /// let config = rusttorch::nn::ConvConfig::new(3, 16, [3, 3]).bias(false);
    /// ```
    #[must_use]
    pub const fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Validates the configuration and registers `weight` and optional `bias`.
    ///
    /// Returns an error for dimensions outside `1..=3`, nonpositive channels,
    /// kernel sizes, strides, dilation or groups, negative padding, incompatible
    /// groups, overflowing fan-in, a non-differentiable parameter dtype, or a
    /// LibTorch allocation failure.
    ///
    /// ```
    /// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::ConvConfig::new(1, 4, [3]).build(&store.root())?;
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build(self, path: &ParameterPath<'_>) -> Result<Conv<D>> {
        validate_parameter_kind(path)?;
        functional::validate_conv_config(
            self.in_channels,
            self.out_channels,
            &self.kernel_size,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups,
        )?;
        let fan_in = self
            .kernel_size
            .iter()
            .try_fold(self.in_channels / self.groups, |size, &kernel| {
                size.checked_mul(kernel)
            })
            .ok_or_else(|| invalid("kernel_size", "fan-in exceeds i64"))?;
        // Adapted behavior: torch.nn.modules.conv._ConvNd.reset_parameters,
        // PyTorch v2.13.0 (cf30153), torch/nn/modules/conv.py.
        // See THIRD_PARTY_NOTICES.md. Kaiming uniform with a=sqrt(5) has this bound.
        let bound = 1.0 / (fan_in as f64).sqrt();
        let init = Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let mut shape = vec![self.out_channels, self.in_channels / self.groups];
        shape.extend(self.kernel_size);
        let weight = path.f_var("weight", &shape, init)?;
        let bias = self
            .bias
            .then(|| path.f_var("bias", &[self.out_channels], init))
            .transpose()?;
        Ok(Conv {
            weight,
            bias,
            config: self,
        })
    }
}

/// A trainable convolution, constructed with [`ConvConfig`].
///
/// Parameters belong to the supplied variable store, so they can be optimized
/// and saved with the rest of a model. See [`ConvConfig`] for an image example.
#[derive(Debug)]
pub struct Conv<const D: usize> {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig<D>,
}

/// A one-dimensional convolution for sequences with shape `[N, C, L]`.
///
/// ```
/// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
/// let layer: rusttorch::nn::Conv1d = rusttorch::nn::ConvConfig::new(1, 4, [3])
///     .build(&store.root())?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub type Conv1d = Conv<1>;

/// A two-dimensional convolution for images with shape `[N, C, H, W]`.
///
/// ```
/// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
/// let layer: rusttorch::nn::Conv2d = rusttorch::nn::ConvConfig::new(3, 8, [3, 3])
///     .build(&store.root())?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub type Conv2d = Conv<2>;

/// A three-dimensional convolution for volumes with shape `[N, C, D, H, W]`.
///
/// ```
/// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
/// let layer: rusttorch::nn::Conv3d = rusttorch::nn::ConvConfig::new(1, 4, [3, 3, 3])
///     .build(&store.root())?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub type Conv3d = Conv<3>;

impl<const D: usize> Conv<D> {
    /// Returns weights shaped `[out_channels, in_channels / groups, kernel...]`.
    ///
    /// ```
    /// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::ConvConfig::new(3, 8, [3, 3]).build(&store.root())?;
    /// assert_eq!(layer.weight().size(), [8, 3, 3, 3]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub const fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns the optional bias shaped `[out_channels]`.
    ///
    /// ```
    /// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::ConvConfig::new(1, 4, [3]).bias(false).build(&store.root())?;
    /// assert!(layer.bias().is_none());
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub const fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl<const D: usize> Module for Conv<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::convolution(
            input,
            &self.weight,
            self.bias.as_ref(),
            &self.config.stride,
            &self.config.padding,
            &self.config.dilation,
            self.config.groups,
        )
    }
}

/// Configures normalization over the trailing dimensions of each sample.
///
/// Use this layer to normalize token features or hidden activations independently
/// of the batch. Defaults are epsilon `1e-5`, learned scale initialized to one,
/// and learned bias initialized to zero. Training and evaluation behave alike.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{LayerNormConfig, Module}};
/// let store = rusttorch::nn::VarStore::new(Device::Cpu);
/// let layer = LayerNormConfig::new([8]).build(&store.root())?;
/// let tokens = Tensor::randn([2, 4, 8], (Kind::Float, Device::Cpu));
/// assert_eq!(layer.forward(&tokens)?.size(), [2, 4, 8]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormConfig {
    normalized_shape: Vec<i64>,
    eps: f64,
    elementwise_affine: bool,
    bias: bool,
}

impl LayerNormConfig {
    /// Selects the trailing dimensions whose mean and variance are normalized.
    ///
    /// ```
    /// let config = rusttorch::nn::LayerNormConfig::new([32]);
    /// ```
    pub fn new(normalized_shape: impl Into<Vec<i64>>) -> Self {
        Self {
            normalized_shape: normalized_shape.into(),
            eps: 1e-5,
            elementwise_affine: true,
            bias: true,
        }
    }

    /// Sets the nonnegative finite constant added to the variance before division.
    ///
    /// ```
    /// let config = rusttorch::nn::LayerNormConfig::new([32]).eps(1e-6);
    /// ```
    #[must_use]
    pub const fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enables learned element-wise scale and optional bias.
    ///
    /// Disabling this option removes both parameters, regardless of [`Self::bias`].
    ///
    /// ```
    /// let config = rusttorch::nn::LayerNormConfig::new([32]).elementwise_affine(false);
    /// ```
    #[must_use]
    pub const fn elementwise_affine(mut self, enabled: bool) -> Self {
        self.elementwise_affine = enabled;
        self
    }

    /// Enables additive bias when element-wise affine parameters are enabled.
    ///
    /// ```
    /// let config = rusttorch::nn::LayerNormConfig::new([32]).bias(false);
    /// ```
    #[must_use]
    pub const fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Registers the affine parameters and constructs the layer.
    ///
    /// Returns an error for an empty shape, nonpositive dimensions, invalid
    /// epsilon, a non-differentiable dtype for affine parameters, or a LibTorch
    /// allocation failure.
    ///
    /// ```
    /// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::LayerNormConfig::new([32]).build(&store.root())?;
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build(self, path: &ParameterPath<'_>) -> Result<LayerNorm> {
        functional::validate_layer_norm(&self.normalized_shape, self.eps)?;
        if self.elementwise_affine {
            validate_parameter_kind(path)?;
        }
        let weight = self
            .elementwise_affine
            .then(|| path.f_ones("weight", &self.normalized_shape))
            .transpose()?;
        let bias = (self.elementwise_affine && self.bias)
            .then(|| path.f_zeros("bias", &self.normalized_shape))
            .transpose()?;
        Ok(LayerNorm {
            weight,
            bias,
            config: self,
        })
    }
}

/// Normalizes trailing features while preserving the input shape.
///
/// Construct with [`LayerNormConfig`]; its example normalizes a batch of tokens.
#[derive(Debug)]
pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    config: LayerNormConfig,
}

impl LayerNorm {
    /// Returns the optional learned scale, shaped like the normalized dimensions.
    ///
    /// ```
    /// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::LayerNormConfig::new([4]).build(&store.root())?;
    /// assert_eq!(layer.weight().unwrap().size(), [4]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub const fn weight(&self) -> Option<&Tensor> {
        self.weight.as_ref()
    }

    /// Returns the optional learned additive bias.
    ///
    /// ```
    /// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::LayerNormConfig::new([4]).bias(false).build(&store.root())?;
    /// assert!(layer.bias().is_none());
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub const fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::layer_norm(
            input,
            &self.config.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.config.eps,
        )
    }
}

/// Configures a trainable lookup table for token IDs or categorical features.
///
/// Weights have shape `[num_embeddings, embedding_dim]` and start from a standard
/// normal distribution. A configured padding row starts at zero and receives no
/// gradient. Input indices must be integer tensors; the output appends the
/// embedding dimension to the input shape. Norm clipping is not implemented.
///
/// ```
/// use rusttorch::{Device, Tensor, nn::{EmbeddingConfig, Module}};
/// let store = rusttorch::nn::VarStore::new(Device::Cpu);
/// let embedding = EmbeddingConfig::new(100, 16).padding_idx(0).build(&store.root())?;
/// let token_ids = Tensor::from_slice(&[0_i64, 3, 7]).reshape([1, 3]);
/// assert_eq!(embedding.forward(&token_ids)?.size(), [1, 3, 16]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingConfig {
    num_embeddings: i64,
    embedding_dim: i64,
    padding_idx: Option<i64>,
    scale_grad_by_freq: bool,
    sparse: bool,
}

impl EmbeddingConfig {
    /// Sets the vocabulary size and width of each embedding vector.
    ///
    /// ```
    /// let config = rusttorch::nn::EmbeddingConfig::new(10_000, 128);
    /// ```
    pub const fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            padding_idx: None,
            scale_grad_by_freq: false,
            sparse: false,
        }
    }

    /// Selects a row that starts at zero and receives no gradient.
    ///
    /// Negative indices count from the vocabulary's end. Loading or explicitly
    /// editing a padding row changes its output; forward does not force it to zero.
    ///
    /// ```
    /// let config = rusttorch::nn::EmbeddingConfig::new(100, 16).padding_idx(0);
    /// ```
    #[must_use]
    pub const fn padding_idx(mut self, padding_idx: i64) -> Self {
        self.padding_idx = Some(padding_idx);
        self
    }

    /// Divides each row's gradient by that token's occurrence count in the input.
    ///
    /// This option requires dense gradients; combining it with [`Self::sparse`]
    /// returns an error when the layer is built.
    ///
    /// ```
    /// let config = rusttorch::nn::EmbeddingConfig::new(100, 16).scale_grad_by_freq(true);
    /// ```
    #[must_use]
    pub const fn scale_grad_by_freq(mut self, enabled: bool) -> Self {
        self.scale_grad_by_freq = enabled;
        self
    }

    /// Requests sparse weight gradients; the output remains dense.
    ///
    /// Use an optimizer that accepts sparse gradients, such as plain SGD.
    ///
    /// ```
    /// let config = rusttorch::nn::EmbeddingConfig::new(100, 16).sparse(true);
    /// ```
    #[must_use]
    pub const fn sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Validates the dimensions and padding index, then registers `weight`.
    ///
    /// Returns an error for nonpositive dimensions, an out-of-range padding row,
    /// sparse frequency scaling, a non-differentiable parameter dtype, or a
    /// LibTorch allocation failure.
    ///
    /// ```
    /// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::EmbeddingConfig::new(100, 16).build(&store.root())?;
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn build(mut self, path: &ParameterPath<'_>) -> Result<Embedding> {
        validate_parameter_kind(path)?;
        if self.num_embeddings <= 0 || self.embedding_dim <= 0 {
            return Err(invalid(
                "embedding dimensions",
                "vocabulary size and vector width must be positive",
            ));
        }
        self.padding_idx = functional::validate_embedding_options(
            self.num_embeddings,
            self.padding_idx,
            self.scale_grad_by_freq,
            self.sparse,
        )?;
        let weight = path.f_randn_standard("weight", &[self.num_embeddings, self.embedding_dim])?;
        // Adapted behavior: torch.nn.modules.sparse.Embedding.reset_parameters,
        // PyTorch v2.13.0 (cf30153), torch/nn/modules/sparse.py; see THIRD_PARTY_NOTICES.md.
        if let Some(index) = self.padding_idx {
            let _ = tch::no_grad(|| weight.f_get(index)?.f_zero_())?;
        }
        Ok(Embedding {
            weight,
            config: self,
        })
    }
}

/// Maps integer token IDs to trainable feature vectors.
///
/// See [`EmbeddingConfig`] for a padded token-batch example.
#[derive(Debug)]
pub struct Embedding {
    weight: Tensor,
    config: EmbeddingConfig,
}

impl Embedding {
    /// Returns the lookup table shaped `[num_embeddings, embedding_dim]`.
    ///
    /// ```
    /// # let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
    /// let layer = rusttorch::nn::EmbeddingConfig::new(100, 16).build(&store.root())?;
    /// assert_eq!(layer.weight().size(), [100, 16]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub const fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::embedding(
            input,
            &self.weight,
            self.config.padding_idx,
            self.config.scale_grad_by_freq,
            self.config.sparse,
        )
    }
}

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}

pub(super) fn validate_parameter_kind(path: &ParameterPath<'_>) -> Result<()> {
    use tch::Kind;
    if matches!(
        path.kind(),
        Kind::Uint8
            | Kind::Int8
            | Kind::Int16
            | Kind::Int
            | Kind::Int64
            | Kind::Bool
            | Kind::QInt8
            | Kind::QUInt8
            | Kind::QInt32
    ) {
        Err(invalid(
            "parameter dtype",
            "trainable parameters require a floating-point or complex dtype",
        ))
    } else {
        Ok(())
    }
}
