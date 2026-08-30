//! Eager neural-network modules.

pub mod functional;

use std::{fmt, path::Path};

use tch::{Device, Tensor, nn::VarStore, no_grad};

use crate::{
    DeviceSpec, Result, RustTorchError,
    device::{ensure_device, resolve_device},
    interop::{
        LoadOptions, LoadReport, StateDictMapping, load_state_dict, load_state_dict_with_mapping,
        save_state_dict,
    },
};

/// A fallible eager module. `forward_t` defaults to mode-independent execution.
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

/// Configuration for a PyTorch-compatible fully connected layer.
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
    pub fn build(self, path: &tch::nn::Path<'_>) -> Result<Linear> {
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

/// A fully connected transformation using LibTorch's linear operator.
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
pub fn linear(path: &tch::nn::Path<'_>, in_features: i64, out_features: i64) -> Result<Linear> {
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

/// A dropout module that is active only during training execution.
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

/// An owned eager sequence with stable PyTorch-style numeric parameter paths.
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
