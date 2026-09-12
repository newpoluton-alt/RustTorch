//! Channel normalization for image, sequence, and volume models.

use tch::{Kind, Tensor};

use super::{Module, ParameterPath, layers::validate_parameter_kind};
use crate::{Result, RustTorchError, device::ensure_device};

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Options {
    channels: i64,
    eps: f64,
    momentum: Option<f64>,
    affine: bool,
    bias: bool,
    track_running_stats: bool,
}

impl Options {
    const fn new(channels: i64, batch: bool) -> Self {
        Self {
            channels,
            eps: 1e-5,
            momentum: Some(0.1),
            affine: batch,
            bias: true,
            track_running_stats: batch,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.channels <= 0 {
            return Err(invalid("channels", "must be positive"));
        }
        if !self.eps.is_finite() || self.eps < 0.0 {
            return Err(invalid("eps", "must be finite and nonnegative"));
        }
        if self.momentum.is_some_and(|m| !m.is_finite()) {
            return Err(invalid("momentum", "must be finite or None"));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Parameters {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    running_mean: Option<Tensor>,
    running_var: Option<Tensor>,
    num_batches_tracked: Option<Tensor>,
}

impl Parameters {
    fn build(options: Options, path: &ParameterPath<'_>) -> Result<Self> {
        options.validate()?;
        if options.affine || options.track_running_stats {
            validate_parameter_kind(path)?;
        }
        let weight = options
            .affine
            .then(|| path.f_ones("weight", &[options.channels]))
            .transpose()?;
        let bias = (options.affine && options.bias)
            .then(|| path.f_zeros("bias", &[options.channels]))
            .transpose()?;
        // Registered non-trainable buffers must use the store dtype; tch's
        // f_zeros_no_train always allocates Float and cannot represent the counter.
        let running_mean = options
            .track_running_stats
            .then(|| {
                Ok::<_, RustTorchError>(path.add(
                    "running_mean",
                    Tensor::f_zeros([options.channels], (path.kind(), path.device()))?,
                    false,
                ))
            })
            .transpose()?;
        let running_var = options
            .track_running_stats
            .then(|| {
                Ok::<_, RustTorchError>(path.add(
                    "running_var",
                    Tensor::f_ones([options.channels], (path.kind(), path.device()))?,
                    false,
                ))
            })
            .transpose()?;
        let num_batches_tracked = options
            .track_running_stats
            .then(|| {
                Ok::<_, RustTorchError>(path.add(
                    "num_batches_tracked",
                    Tensor::f_zeros([], (Kind::Int64, path.device()))?,
                    false,
                ))
            })
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked,
        })
    }

    fn validate_input(&self, input: &Tensor) -> Result<()> {
        if !input.defined() {
            return Err(invalid("normalization input", "must be a defined tensor"));
        }
        for (name, parameter) in [
            ("normalization weight", &self.weight),
            ("normalization bias", &self.bias),
            ("running_mean", &self.running_mean),
            ("running_var", &self.running_var),
            ("num_batches_tracked", &self.num_batches_tracked),
        ] {
            if let Some(parameter) = parameter {
                ensure_device(name, parameter, input.device())?;
            }
        }
        Ok(())
    }
}

/// Configures channel normalization using batch statistics during training.
///
/// Use after convolutions to stabilize image, sequence, or volume training.
/// `D` is the number of spatial dimensions (`1..=3`). Defaults are learned
/// scale and bias, epsilon `1e-5`, momentum `Some(0.1)`, and tracked statistics.
/// Training uses biased variance; the stored variance uses the unbiased estimate.
/// Standalone `forward` uses evaluation; call `forward_t(input, true)` to train.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{BatchNormConfig, Module, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let norm = BatchNormConfig::<2>::new(8).momentum(None).build(&store.root())?;
/// let features = Tensor::randn([4, 8, 6, 6], (Kind::Float, Device::Cpu));
/// let normalized = norm.forward_t(&features, true)?;
/// assert_eq!(normalized.size(), features.size());
/// assert_eq!(norm.num_batches_tracked().unwrap().int64_value(&[]), 1);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormConfig<const D: usize>(Options);

/// Configures normalization independently for each sample and channel.
///
/// Useful for style transfer and image models whose batch sizes vary. `D` is
/// `1..=3`. Defaults are epsilon `1e-5`, momentum `Some(0.1)`, no affine
/// parameters, and no tracked statistics. Unbatched inputs are accepted.
/// Channels must match the configuration even when affine parameters are off.
/// Standalone `forward` evaluates; `forward_t(input, true)` updates running
/// statistics when tracking is enabled.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{InstanceNormConfig, Module, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let norm = InstanceNormConfig::<2>::new(3).affine(true).build(&store.root())?;
/// let image = Tensor::randn([3, 8, 8], (Kind::Float, Device::Cpu));
/// assert_eq!(norm.forward(&image)?.size(), [3, 8, 8]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceNormConfig<const D: usize>(Options);

macro_rules! norm_config {
    ($config:ident, $layer:ident, $batch:expr) => {
        impl<const D: usize> $config<D> {
            /// Selects the number of input channels. See the configuration example.
            pub const fn new(channels: i64) -> Self {
                Self(Options::new(channels, $batch))
            }
            /// Sets the finite, nonnegative variance stabilizer (default `1e-5`).
            #[must_use]
            pub const fn eps(mut self, eps: f64) -> Self {
                self.0.eps = eps;
                self
            }
            /// Sets the running-statistic update factor. Batch normalization uses
            /// a cumulative average with `None`; instance normalization then leaves
            /// running statistics unchanged. Defaults to `Some(0.1)`.
            #[must_use]
            pub const fn momentum(mut self, momentum: Option<f64>) -> Self {
                self.0.momentum = momentum;
                self
            }
            /// Enables learned channel scale and optional bias.
            #[must_use]
            pub const fn affine(mut self, affine: bool) -> Self {
                self.0.affine = affine;
                self
            }
            /// Enables additive bias when affine parameters are enabled.
            #[must_use]
            pub const fn bias(mut self, bias: bool) -> Self {
                self.0.bias = bias;
                self
            }
            /// Registers running mean, variance, and an integer batch counter.
            /// With tracking disabled, evaluation also uses input statistics.
            #[must_use]
            pub const fn track_running_stats(mut self, track: bool) -> Self {
                self.0.track_running_stats = track;
                self
            }
            /// Validates dimensions/options and registers parameters and buffers.
            /// Invalid channels, epsilon, momentum, dtype, and allocation failures
            /// return errors. See the configuration example for a complete model.
            pub fn build(self, path: &ParameterPath<'_>) -> Result<$layer<D>> {
                if !(1..=3).contains(&D) {
                    return Err(invalid(
                        "normalization dimensions",
                        "must be one, two, or three",
                    ));
                }
                Ok($layer {
                    parameters: Parameters::build(self.0, path)?,
                    options: self.0,
                })
            }
        }
    };
}
norm_config!(BatchNormConfig, BatchNorm, true);
norm_config!(InstanceNormConfig, InstanceNorm, false);

/// A channel-normalization layer with persistent batch statistics.
///
/// Build with [`BatchNormConfig`], whose example shows training with a cumulative
/// average. Parameters and buffers belong to the supplied variable store.
#[derive(Debug)]
pub struct BatchNorm<const D: usize> {
    parameters: Parameters,
    options: Options,
}

/// A per-sample channel-normalization layer, built with [`InstanceNormConfig`].
///
/// Its configuration example shows unbatched image normalization. Tracked
/// statistics are optional; the registered batch counter remains zero because
/// instance normalization does not use it to update running statistics.
#[derive(Debug)]
pub struct InstanceNorm<const D: usize> {
    parameters: Parameters,
    options: Options,
}

macro_rules! norm_accessors {
    ($layer:ident) => {
        impl<const D: usize> $layer<D> {
            /// Returns the optional learned channel scale, initialized to one.
            pub const fn weight(&self) -> Option<&Tensor> {
                self.parameters.weight.as_ref()
            }
            /// Returns the optional learned channel bias, initialized to zero.
            pub const fn bias(&self) -> Option<&Tensor> {
                self.parameters.bias.as_ref()
            }
            /// Returns the non-trainable running mean, or `None` without tracking.
            pub const fn running_mean(&self) -> Option<&Tensor> {
                self.parameters.running_mean.as_ref()
            }
            /// Returns the non-trainable running variance, initialized to one.
            pub const fn running_var(&self) -> Option<&Tensor> {
                self.parameters.running_var.as_ref()
            }
            /// Returns the scalar integer batch counter, or `None` without tracking.
            pub const fn num_batches_tracked(&self) -> Option<&Tensor> {
                self.parameters.num_batches_tracked.as_ref()
            }
            /// Resets tracked means/counters to zero and variances to one.
            /// Affine parameters are preserved; useful when recalibrating a model.
            pub fn reset_running_stats(&self) -> Result<()> {
                tch::no_grad(|| {
                    if let Some(mean) = self.running_mean() {
                        let _ = mean.shallow_clone().f_zero_()?;
                    }
                    if let Some(var) = self.running_var() {
                        let _ = var.shallow_clone().f_fill_(1.0)?;
                    }
                    if let Some(count) = self.num_batches_tracked() {
                        let _ = count.shallow_clone().f_zero_()?;
                    }
                    Ok(())
                })
            }
        }
    };
}
norm_accessors!(BatchNorm);
norm_accessors!(InstanceNorm);

impl<const D: usize> Module for BatchNorm<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, false)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.parameters.validate_input(input)?;
        let shape = input.size();
        if !(shape.len() == D + 2 || D == 1 && shape.len() == 2) {
            return Err(invalid(
                "batch normalization input",
                "rank must match the configured spatial dimensions (BatchNorm1d also accepts [N, C])",
            ));
        }
        if shape[1] != self.options.channels {
            return Err(invalid(
                "input channels",
                "must match the normalization configuration",
            ));
        }
        let use_input = training || !self.options.track_running_stats;
        if use_input
            && shape[2..]
                .iter()
                .try_fold(shape[0], |n, s| n.checked_mul(*s))
                == Some(1)
        {
            return Err(invalid(
                "batch normalization input",
                "training requires more than one value per channel",
            ));
        }
        // Adapted behavior: _BatchNorm.forward, PyTorch v2.13.0 (cf30153),
        // torch/nn/modules/batchnorm.py; see THIRD_PARTY_NOTICES.md.
        let mut momentum = self.options.momentum.unwrap_or(0.0);
        if training && let Some(count) = self.num_batches_tracked() {
            let _ = count.shallow_clone().f_add_scalar_(1_i64)?;
            if self.options.momentum.is_none() {
                momentum = 1.0 / count.f_int64_value(&[])? as f64;
            }
        }
        Ok(input.f_batch_norm(
            self.weight(),
            self.bias(),
            self.running_mean(),
            self.running_var(),
            use_input,
            momentum,
            self.options.eps,
            true,
        )?)
    }
}

impl<const D: usize> Module for InstanceNorm<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_t(input, false)
    }
    fn forward_t(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.parameters.validate_input(input)?;
        let shape = input.size();
        if ![D + 1, D + 2].contains(&shape.len()) {
            return Err(invalid(
                "instance normalization input",
                "rank must match the configured spatial dimensions, with an optional batch",
            ));
        }
        let unbatched = shape.len() == D + 1;
        let channel_axis = usize::from(!unbatched);
        if shape[channel_axis] != self.options.channels {
            return Err(invalid(
                "input channels",
                "must match the normalization configuration",
            ));
        }
        let use_input = training || !self.options.track_running_stats;
        if use_input && shape[channel_axis + 1..].iter().all(|s| *s == 1) {
            return Err(invalid(
                "instance normalization input",
                "input statistics require more than one spatial element",
            ));
        }
        let input = if unbatched {
            input.f_unsqueeze(0)?
        } else {
            input.shallow_clone()
        };
        let output = input.f_instance_norm(
            self.weight(),
            self.bias(),
            self.running_mean(),
            self.running_var(),
            use_input,
            self.options.momentum.unwrap_or(0.0),
            self.options.eps,
            true,
        )?;
        Ok(if unbatched {
            output.f_squeeze_dim(0)?
        } else {
            output
        })
    }
}

macro_rules! norm_aliases {
    ($layer:ident, $one:ident, $two:ident, $three:ident, $config:ident) => {
        #[doc = concat!("Sequence normalization; build with [`", stringify!($config), "::<1>`]. See its example.")]
        pub type $one = $layer<1>;
        #[doc = concat!("Image normalization; build with [`", stringify!($config), "::<2>`]. See its example.")]
        pub type $two = $layer<2>;
        #[doc = concat!("Volume normalization; build with [`", stringify!($config), "::<3>`]. See its example.")]
        pub type $three = $layer<3>;
    };
}
norm_aliases!(
    BatchNorm,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    BatchNormConfig
);
norm_aliases!(
    InstanceNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    InstanceNormConfig
);

/// Configures normalization within equal channel groups for every sample.
///
/// Useful for image models trained with small batches. Defaults are epsilon
/// `1e-5`, scale initialized to one, and bias initialized to zero. Training and
/// evaluation use the same input statistics; there are no running buffers.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{GroupNormConfig, Module, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let norm = GroupNormConfig::new(4, 8).build(&store.root())?;
/// let image = Tensor::randn([1, 8, 4, 4], (Kind::Float, Device::Cpu));
/// assert_eq!(norm.forward(&image)?.size(), [1, 8, 4, 4]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNormConfig {
    groups: i64,
    options: Options,
}

impl GroupNormConfig {
    /// Selects the number of equal groups and input channels; groups must divide channels.
    pub const fn new(groups: i64, channels: i64) -> Self {
        Self {
            groups,
            options: Options {
                track_running_stats: false,
                ..Options::new(channels, true)
            },
        }
    }
    /// Sets the finite, nonnegative variance stabilizer.
    #[must_use]
    pub const fn eps(mut self, eps: f64) -> Self {
        self.options.eps = eps;
        self
    }
    /// Enables learned channel scale and optional bias.
    #[must_use]
    pub const fn affine(mut self, affine: bool) -> Self {
        self.options.affine = affine;
        self
    }
    /// Enables additive bias when affine parameters are enabled.
    #[must_use]
    pub const fn bias(mut self, bias: bool) -> Self {
        self.options.bias = bias;
        self
    }
    /// Validates positive channels, divisibility, epsilon, and dtype, then registers parameters.
    pub fn build(self, path: &ParameterPath<'_>) -> Result<GroupNorm> {
        if self.groups <= 0 || self.options.channels % self.groups != 0 {
            return Err(invalid(
                "groups",
                "must be positive and divide the channel count",
            ));
        }
        Ok(GroupNorm {
            parameters: Parameters::build(self.options, path)?,
            config: self,
        })
    }
}

/// Normalizes each sample within channel groups; build with [`GroupNormConfig`].
///
/// The configuration example shows a single-image batch.
#[derive(Debug)]
pub struct GroupNorm {
    parameters: Parameters,
    config: GroupNormConfig,
}

impl GroupNorm {
    /// Returns the optional learned scale with one element per channel.
    pub const fn weight(&self) -> Option<&Tensor> {
        self.parameters.weight.as_ref()
    }
    /// Returns the optional learned additive bias.
    pub const fn bias(&self) -> Option<&Tensor> {
        self.parameters.bias.as_ref()
    }
}

impl Module for GroupNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.parameters.validate_input(input)?;
        let shape = input.size();
        if shape.len() < 2 || shape[1] != self.config.options.channels {
            return Err(invalid(
                "group normalization input",
                "must have shape [N, configured_channels, ...]",
            ));
        }
        if std::iter::once(shape[1] / self.config.groups)
            .chain(shape[2..].iter().copied())
            .try_fold(shape[0], i64::checked_mul)
            == Some(1)
        {
            return Err(invalid(
                "group normalization input",
                "requires more than one value per channel group across the batch",
            ));
        }
        Ok(input.f_group_norm(
            self.config.groups,
            self.weight(),
            self.bias(),
            self.config.options.eps,
            true,
        )?)
    }
}
