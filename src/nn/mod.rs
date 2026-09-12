//! Trainable layers and model composition for tensors.
//!
//! Build image feature extractors with [`ConvConfig`], token models with
//! [`EmbeddingConfig`] and [`LayerNormConfig`], and feed-forward networks with
//! [`Sequential`]. A [`VarStore`] holds parameters for custom models; [`Module`]
//! lets their fallible forward passes share the same interface.
//!
//! ```
//! use rusttorch::{DeviceSpec, Kind, Tensor, nn::{ConvConfig, Sequential}};
//! let model = Sequential::builder()
//!     .conv2d(ConvConfig::new(3, 8, [3, 3]).padding([1, 1]))
//!     .relu()
//!     .flatten(1, -1)
//!     .linear(8 * 8 * 8, 10)
//!     .build(DeviceSpec::Cpu)?;
//! let images = Tensor::zeros([2, 3, 8, 8], (Kind::Float, model.device()));
//! assert_eq!(model.forward(&images)?.size(), [2, 10]);
//! # Ok::<(), rusttorch::RustTorchError>(())
//! ```

pub mod functional;
mod layers;

pub use layers::{
    Conv, Conv1d, Conv2d, Conv3d, ConvConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig,
};

/// The parameter store shared by a model's layers and optimizer.
///
/// ```
/// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
/// let layer = rusttorch::nn::LinearConfig::new(4, 2).build(&store.root())?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub use tch::nn::VarStore;

/// A named location in a [`VarStore`] used to register layer parameters.
///
/// ```
/// let store = rusttorch::nn::VarStore::new(rusttorch::Device::Cpu);
/// let path: rusttorch::nn::ParameterPath<'_> = store.root() / "encoder";
/// let layer = rusttorch::nn::LinearConfig::new(4, 2).build(&path)?;
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub use tch::nn::Path as ParameterPath;

use std::{fmt, path::Path};

use tch::{Device, Tensor, no_grad};

use crate::{
    DeviceSpec, Result, RustTorchError,
    device::{ensure_device, resolve_device},
    interop::{
        LoadOptions, LoadReport, StateDictMapping, load_state_dict, load_state_dict_with_mapping,
        save_state_dict,
    },
};

/// A tensor transformation with a fallible forward pass.
///
/// Implement this trait to compose custom layers, residual branches, or shared
/// parameters. Override [`Module::forward_t`] when behavior depends on training
/// mode. Otherwise it calls [`Module::forward`] and ignores the mode flag.
///
/// ```
/// use rusttorch::{Result, Tensor, nn::Module};
/// struct ResidualRelu;
/// impl Module for ResidualRelu {
///     fn forward(&self, input: &Tensor) -> Result<Tensor> {
///         Ok(input.f_relu()?.f_add(input)?)
///     }
/// }
/// let output = ResidualRelu.forward(&Tensor::from_slice(&[-1_f32, 2.]))?;
/// assert_eq!(Vec::<f32>::try_from(&output)?, [-1., 4.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub trait Module: Send {
    /// Computes the module output using its default execution mode.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Computes the module output with an explicit training flag.
    ///
    /// Mode-independent modules use [`Module::forward`] by default.
    fn forward_t(&self, input: &Tensor, _training: bool) -> Result<Tensor> {
        self.forward(input)
    }
}

/// Configuration for a fully connected layer.
///
/// A linear layer converts each input feature vector into an output feature
/// vector, preserving all leading dimensions. Use it as a regression head,
/// classifier, or hidden projection. Bias is enabled by default.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{LinearConfig, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let head = LinearConfig::new(8, 3).build(&(store.root() / "head"))?;
/// let features = Tensor::ones([4, 8], (Kind::Float, Device::Cpu));
/// assert_eq!(head.forward(&features)?.size(), [4, 3]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearConfig {
    in_features: i64,
    out_features: i64,
    bias: bool,
}

impl LinearConfig {
    /// Creates a biased linear-layer configuration.
    pub const fn new(in_features: i64, out_features: i64) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    #[must_use]
    /// Enables or disables the additive bias parameter.
    pub const fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Registers the layer parameters under `path` and creates the layer.
    ///
    /// Returns an error for negative feature counts or a parameter-store dtype
    /// that cannot track gradients (integer, boolean, or quantized types).
    pub fn build(self, path: &ParameterPath<'_>) -> Result<Linear> {
        layers::validate_parameter_kind(path)?;
        if self.in_features < 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "in_features",
                reason: "must be non-negative".to_owned(),
            });
        }
        if self.out_features < 0 {
            return Err(RustTorchError::InvalidConfiguration {
                field: "out_features",
                reason: "must be non-negative".to_owned(),
            });
        }
        let inner = tch::nn::linear(
            path,
            self.in_features,
            self.out_features,
            tch::nn::LinearConfig {
                bias: self.bias,
                ..Default::default()
            },
        );
        // Adapted from PyTorch v2.13.0 torch/nn/modules/linear.py:
        // zero fan-in uses a zero bias bound. See THIRD_PARTY_NOTICES.md.
        if self.in_features == 0
            && let Some(bias) = &inner.bs
        {
            let mut bias = bias.shallow_clone();
            let _ = no_grad(|| bias.f_zero_())?;
        }
        Ok(Linear { inner })
    }
}

/// A trainable affine transformation of the last input dimension.
///
/// Construct with [`linear`] or [`LinearConfig`]. Its parameters are registered
/// as `weight` and optional `bias` beneath the chosen [`ParameterPath`].
/// [`LinearConfig`]'s example shows a classification head.
#[derive(Debug)]
pub struct Linear {
    inner: tch::nn::Linear,
}

impl Linear {
    /// Returns the weight parameter with shape `[out_features, in_features]`.
    pub fn weight(&self) -> &Tensor {
        &self.inner.ws
    }

    /// Returns the bias parameter, or `None` when bias was disabled.
    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bs.as_ref()
    }

    /// Applies the linear transformation to the last input dimension.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        <Self as Module>::forward(self, input)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::linear(input, &self.inner.ws, self.inner.bs.as_ref())
    }
}

/// Creates a biased linear layer and registers it under `path`.
pub fn linear(path: &ParameterPath<'_>, in_features: i64, out_features: i64) -> Result<Linear> {
    LinearConfig::new(in_features, out_features).build(path)
}

/// A module that returns a shallow clone of its input.
#[derive(Debug, Default)]
pub struct Identity;

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.shallow_clone())
    }
}

/// An element-wise rectified linear unit module.
#[derive(Debug, Default)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::relu(input)
    }
}

/// Approximation mode for Gaussian error linear units.
///
/// New backend-supported approximation modes may be added in future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GeluApproximation {
    /// Uses the exact formulation.
    #[default]
    None,
    /// Uses the tanh approximation.
    Tanh,
}

impl GeluApproximation {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Tanh => "tanh",
        }
    }
}

/// An element-wise Gaussian error linear unit module.
#[derive(Debug, Default)]
pub struct Gelu {
    approximation: GeluApproximation,
}

impl Gelu {
    /// Creates a GELU module with the requested approximation.
    pub const fn new(approximation: GeluApproximation) -> Self {
        Self { approximation }
    }
}

impl Module for Gelu {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::gelu_with_approximation(input, self.approximation)
    }
}

/// Randomly masks activations during training to regularize a model.
///
/// Standalone [`Module::forward`] uses evaluation behavior. Call
/// [`Module::forward_t`] with `true` to apply dropout, or put this layer in a
/// [`Sequential`] model whose training flag controls execution.
///
/// ```
/// use rusttorch::{Tensor, nn::{Dropout, Module}};
/// let dropout = Dropout::new(1.0)?;
/// let input = Tensor::from_slice(&[1_f32, 2.]);
/// assert_eq!(Vec::<f32>::try_from(&dropout.forward_t(&input, true)?)?, [0., 0.]);
/// assert_eq!(Vec::<f32>::try_from(&dropout.forward(&input)?)?, [1., 2.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug)]
pub struct Dropout {
    probability: f64,
}

impl Dropout {
    /// Creates dropout with a probability in the inclusive range `[0, 1]`.
    pub fn new(probability: f64) -> Result<Self> {
        functional::validate_dropout(probability)?;
        Ok(Self { probability })
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, false)
    }

    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        functional::dropout(input, self.probability, training)
    }
}

/// Flattens a contiguous range of tensor dimensions.
#[derive(Debug)]
pub struct Flatten {
    start_dim: i64,
    end_dim: i64,
}

impl Flatten {
    /// Creates a flatten module over the inclusive dimension range.
    pub const fn new(start_dim: i64, end_dim: i64) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new(1, -1)
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        functional::flatten(input, self.start_dim, self.end_dim)
    }
}

enum LayerSpec {
    Conv1d(ConvConfig<1>),
    Conv2d(ConvConfig<2>),
    Conv3d(ConvConfig<3>),
    LayerNorm(LayerNormConfig),
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Identity,
    ReLU,
    Gelu(GeluApproximation),
    Dropout(f64),
    Flatten(i64, i64),
}

/// Builder for an owned eager model and its `VarStore`.
#[derive(Default)]
pub struct SequentialBuilder {
    layers: Vec<LayerSpec>,
}

impl SequentialBuilder {
    /// Creates an empty sequential model builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a sequence convolution; validates the configuration when building.
    ///
    /// ```
    /// use rusttorch::nn::{Sequential, ConvConfig};
    /// let builder = Sequential::builder().conv1d(ConvConfig::new(1, 4, [3]));
    /// ```
    #[must_use]
    pub fn conv1d(mut self, config: ConvConfig<1>) -> Self {
        self.layers.push(LayerSpec::Conv1d(config));
        self
    }

    /// Appends an image convolution; validates the configuration when building.
    ///
    /// ```
    /// use rusttorch::nn::{Sequential, ConvConfig};
    /// let builder = Sequential::builder().conv2d(ConvConfig::new(3, 8, [3, 3]));
    /// ```
    #[must_use]
    pub fn conv2d(mut self, config: ConvConfig<2>) -> Self {
        self.layers.push(LayerSpec::Conv2d(config));
        self
    }

    /// Appends a volume convolution; validates the configuration when building.
    ///
    /// ```
    /// use rusttorch::nn::{Sequential, ConvConfig};
    /// let builder = Sequential::builder().conv3d(ConvConfig::new(1, 4, [3, 3, 3]));
    /// ```
    #[must_use]
    pub fn conv3d(mut self, config: ConvConfig<3>) -> Self {
        self.layers.push(LayerSpec::Conv3d(config));
        self
    }

    /// Appends normalization over trailing feature dimensions; validates the configuration when building.
    ///
    /// ```
    /// use rusttorch::nn::{Sequential, LayerNormConfig};
    /// let builder = Sequential::builder().layer_norm(LayerNormConfig::new([8]));
    /// ```
    #[must_use]
    pub fn layer_norm(mut self, config: LayerNormConfig) -> Self {
        self.layers.push(LayerSpec::LayerNorm(config));
        self
    }

    /// Appends a token lookup table; validates the configuration when building.
    ///
    /// ```
    /// use rusttorch::nn::{Sequential, EmbeddingConfig};
    /// let builder = Sequential::builder().embedding(EmbeddingConfig::new(100, 8));
    /// ```
    #[must_use]
    pub fn embedding(mut self, config: EmbeddingConfig) -> Self {
        self.layers.push(LayerSpec::Embedding(config));
        self
    }

    #[must_use]
    /// Appends a biased linear layer.
    pub fn linear(mut self, in_features: i64, out_features: i64) -> Self {
        self.layers.push(LayerSpec::Linear(LinearConfig::new(
            in_features,
            out_features,
        )));
        self
    }

    #[must_use]
    /// Appends a configured linear layer.
    pub fn linear_config(mut self, config: LinearConfig) -> Self {
        self.layers.push(LayerSpec::Linear(config));
        self
    }

    #[must_use]
    /// Appends an identity layer.
    pub fn identity(mut self) -> Self {
        self.layers.push(LayerSpec::Identity);
        self
    }

    #[must_use]
    /// Appends an element-wise ReLU layer.
    pub fn relu(mut self) -> Self {
        self.layers.push(LayerSpec::ReLU);
        self
    }

    #[must_use]
    /// Appends an exact GELU layer.
    pub fn gelu(mut self) -> Self {
        self.layers.push(LayerSpec::Gelu(GeluApproximation::None));
        self
    }

    #[must_use]
    /// Appends a GELU layer with an explicit approximation.
    pub fn gelu_approximate(mut self, approximation: GeluApproximation) -> Self {
        self.layers.push(LayerSpec::Gelu(approximation));
        self
    }

    #[must_use]
    /// Appends dropout with the given probability.
    ///
    /// Probability validation occurs when [`SequentialBuilder::build`] is called.
    pub fn dropout(mut self, probability: f64) -> Self {
        self.layers.push(LayerSpec::Dropout(probability));
        self
    }

    #[must_use]
    /// Appends a flatten layer over the inclusive dimension range.
    pub fn flatten(mut self, start_dim: i64, end_dim: i64) -> Self {
        self.layers.push(LayerSpec::Flatten(start_dim, end_dim));
        self
    }

    /// Builds the model and allocates all parameters on the resolved device.
    pub fn build(self, device: DeviceSpec) -> Result<Sequential> {
        let device = resolve_device(device)?;
        let var_store = VarStore::new(device);
        let mut layers: Vec<Box<dyn Module>> = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.into_iter().enumerate() {
            let path = var_store.root() / index.to_string();
            let layer: Box<dyn Module> = match layer {
                LayerSpec::Conv1d(config) => Box::new(config.build(&path)?),
                LayerSpec::Conv2d(config) => Box::new(config.build(&path)?),
                LayerSpec::Conv3d(config) => Box::new(config.build(&path)?),
                LayerSpec::LayerNorm(config) => Box::new(config.build(&path)?),
                LayerSpec::Embedding(config) => Box::new(config.build(&path)?),
                LayerSpec::Linear(config) => Box::new(config.build(&path)?),
                LayerSpec::Identity => Box::new(Identity),
                LayerSpec::ReLU => Box::new(ReLU),
                LayerSpec::Gelu(approximation) => Box::new(Gelu::new(approximation)),
                LayerSpec::Dropout(probability) => Box::new(Dropout::new(probability)?),
                LayerSpec::Flatten(start_dim, end_dim) => {
                    Box::new(Flatten::new(start_dim, end_dim))
                }
            };
            layers.push(layer);
        }
        Ok(Sequential {
            var_store,
            layers,
            training: true,
        })
    }
}

/// A sequence of layers with an owned parameter store.
///
/// Create a model with [`Sequential::builder`], pass its [`Self::var_store`] to
/// an optimizer, and use [`Self::save_weights`] to persist trained parameters.
/// Layers register parameters under their numeric position (for example,
/// `0.weight` and `2.bias`). New models start in training mode; call
/// [`Self::eval`] for inference. Evaluation changes dropout behavior but does
/// not disable gradient tracking; use [`crate::no_grad`] when needed.
///
/// ```
/// use rusttorch::{DeviceSpec, Kind, Tensor, nn::Sequential};
/// let mut model = Sequential::builder().linear(4, 8).relu().dropout(0.2)
///     .linear(8, 2).build(DeviceSpec::Cpu)?;
/// model.eval();
/// let features = Tensor::zeros([3, 4], (Kind::Float, model.device()));
/// let scores = rusttorch::no_grad(|| model.forward(&features))?;
/// assert_eq!(scores.size(), [3, 2]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
pub struct Sequential {
    var_store: VarStore,
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl fmt::Debug for Sequential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Sequential")
            .field("device", &self.device())
            .field("layers", &self.layers.len())
            .field("training", &self.training)
            .finish()
    }
}

impl Sequential {
    /// Creates an empty sequential model builder.
    pub fn builder() -> SequentialBuilder {
        SequentialBuilder::new()
    }

    /// Runs every layer using the model's current training state.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, self.training)
    }

    /// Runs every layer with an explicit training flag without changing model state.
    pub fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        ensure_device("Sequential input", input, self.device())?;
        self.layers
            .iter()
            .try_fold(input.shallow_clone(), |value, layer| {
                layer.forward_t(&value, training)
            })
    }

    /// Enables training behavior for subsequent [`Sequential::forward`] calls.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Enables evaluation behavior for subsequent [`Sequential::forward`] calls.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether default forward calls use training behavior.
    pub const fn is_training(&self) -> bool {
        self.training
    }

    /// Returns the device holding this model's parameters.
    pub fn device(&self) -> Device {
        self.var_store.device()
    }

    /// Returns the parameter store used by the model.
    pub const fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    /// Moves all model parameters to the resolved device.
    pub fn to_device(&mut self, device: DeviceSpec) -> Result<()> {
        self.var_store.set_device(resolve_device(device)?);
        Ok(())
    }

    /// Saves the model state as a device-neutral SafeTensors file.
    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        save_state_dict(path, &self.var_store)
    }

    /// Strictly loads model state from a SafeTensors file.
    pub fn load_weights(&self, path: impl AsRef<Path>) -> Result<LoadReport> {
        load_state_dict(path, &self.var_store)
    }

    /// Loads model state with explicit key mapping and strictness options.
    pub fn load_weights_with_mapping(
        &self,
        path: impl AsRef<Path>,
        mapping: &StateDictMapping,
        options: LoadOptions,
    ) -> Result<LoadReport> {
        load_state_dict_with_mapping(path, &self.var_store, mapping, options)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Sequential::forward(self, input)
    }

    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        Sequential::forward_t(self, input, training)
    }
}
