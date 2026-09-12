//! Spatial resizing with trainable transposed convolutions and pooling.

use tch::{Tensor, nn::Init};

use super::{Module, ParameterPath, functional, layers::validate_parameter_kind};
use crate::{Result, RustTorchError, device::ensure_device};

fn invalid(field: &'static str, reason: &str) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.to_owned(),
    }
}

fn spatial_input<const D: usize>(input: &Tensor) -> Result<Vec<i64>> {
    if !input.defined() {
        return Err(invalid("spatial input", "must be a defined tensor"));
    }
    let shape = input.size();
    if !(1..=3).contains(&D) || ![D + 1, D + 2].contains(&shape.len()) {
        return Err(invalid(
            "spatial input",
            "requires one, two, or three spatial axes, preceded by channels and an optional batch",
        ));
    }
    Ok(shape)
}

/// Configures a learned upsampling convolution for sequences, images, or volumes.
///
/// Use in image decoders and segmentation networks. `D` is `1..=3`; inputs use
/// `[N, C, spatial...]` or omit `N`. Defaults are stride/dilation one, zero
/// padding/output padding, one group, and bias. Only zero padding is supported.
/// Output length on each axis is `(input - 1) * stride - 2 * padding +
/// dilation * (kernel - 1) + output_padding + 1`.
/// Weights and biases start uniformly within `±1/sqrt(out_channels/groups *
/// product(kernel_size))`. Output padding selects a shape; it does not add zeros.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{ConvTransposeConfig, Module, VarStore}};
/// let store = VarStore::new(Device::Cpu);
/// let upsample = ConvTransposeConfig::new(8, 3, [4, 4])
///     .stride([2, 2]).padding([1, 1]).build(&store.root())?;
/// let features = Tensor::randn([2, 8, 8, 8], (Kind::Float, Device::Cpu));
/// assert_eq!(upsample.forward(&features)?.size(), [2, 3, 16, 16]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTransposeConfig<const D: usize> {
    in_channels: i64,
    out_channels: i64,
    kernel_size: [i64; D],
    stride: [i64; D],
    padding: [i64; D],
    output_padding: [i64; D],
    dilation: [i64; D],
    groups: i64,
    bias: bool,
}

impl<const D: usize> ConvTransposeConfig<D> {
    /// Selects input/output channels and a positive kernel size per axis.
    pub const fn new(in_channels: i64, out_channels: i64, kernel_size: [i64; D]) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: [1; D],
            padding: [0; D],
            output_padding: [0; D],
            dilation: [1; D],
            groups: 1,
            bias: true,
        }
    }
    /// Sets the positive upsampling stride per axis.
    #[must_use]
    pub const fn stride(mut self, stride: [i64; D]) -> Self {
        self.stride = stride;
        self
    }
    /// Sets the nonnegative padding adjustment per axis.
    #[must_use]
    pub const fn padding(mut self, padding: [i64; D]) -> Self {
        self.padding = padding;
        self
    }
    /// Selects additional output elements per axis; each must be nonnegative and
    /// smaller than the stride or dilation on that axis.
    #[must_use]
    pub const fn output_padding(mut self, output_padding: [i64; D]) -> Self {
        self.output_padding = output_padding;
        self
    }
    /// Sets positive kernel-element spacing per axis.
    #[must_use]
    pub const fn dilation(mut self, dilation: [i64; D]) -> Self {
        self.dilation = dilation;
        self
    }
    /// Splits channels into groups that divide both channel counts evenly.
    #[must_use]
    pub const fn groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }
    /// Enables the optional output-channel bias.
    #[must_use]
    pub const fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
    /// Validates options/dtype, then registers `weight` and optional `bias`.
    /// Invalid dimensions, output padding, overflowing fan-in, or backend
    /// allocation failures return errors. See the configuration example.
    pub fn build(self, path: &ParameterPath<'_>) -> Result<ConvTranspose<D>> {
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
        for axis in 0..D {
            if self.output_padding[axis] < 0
                || (self.output_padding[axis] >= self.stride[axis]
                    && self.output_padding[axis] >= self.dilation[axis])
            {
                return Err(invalid(
                    "output_padding",
                    "must be nonnegative and smaller than either stride or dilation per axis",
                ));
            }
        }
        let fan_in = self
            .kernel_size
            .iter()
            .try_fold(self.out_channels / self.groups, |n, k| n.checked_mul(*k))
            .ok_or_else(|| invalid("kernel_size", "fan-in exceeds i64"))?;
        // Adapted behavior: _ConvNd.reset_parameters, PyTorch v2.13.0
        // (cf30153), torch/nn/modules/conv.py; see THIRD_PARTY_NOTICES.md.
        let bound = 1.0 / (fan_in as f64).sqrt();
        let init = Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let mut shape = vec![self.in_channels, self.out_channels / self.groups];
        shape.extend(self.kernel_size);
        let weight = path.f_var("weight", &shape, init)?;
        let bias = self
            .bias
            .then(|| path.f_var("bias", &[self.out_channels], init))
            .transpose()?;
        Ok(ConvTranspose {
            weight,
            bias,
            config: self,
        })
    }
}

/// A learned spatial upsampler, built with [`ConvTransposeConfig`].
///
/// Its configuration example doubles image width and height. Trainable weights
/// and optional bias are registered in the supplied parameter store.
#[derive(Debug)]
pub struct ConvTranspose<const D: usize> {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTransposeConfig<D>,
}

impl<const D: usize> ConvTranspose<D> {
    /// Returns weights shaped `[in_channels, out_channels / groups, kernel...]`.
    pub const fn weight(&self) -> &Tensor {
        &self.weight
    }
    /// Returns optional bias shaped `[out_channels]`.
    pub const fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    /// Upsamples to a requested spatial size, resolving stride-related ambiguity.
    ///
    /// Supply `D` spatial sizes or a full output shape. Each size must lie
    /// between the zero-output-padding size and that size plus `stride - 1`.
    /// Full-shape batch/channel entries are ignored, as they do not select
    /// output padding. The configured output padding is overridden.
    ///
    /// ```
    /// use rusttorch::{Device, Kind, Tensor, nn::{ConvTransposeConfig, VarStore}};
    /// let store = VarStore::new(Device::Cpu);
    /// let up = ConvTransposeConfig::new(1, 1, [3]).stride([2]).build(&store.root())?;
    /// let x = Tensor::ones([1, 1, 3], (Kind::Float, Device::Cpu));
    /// assert_eq!(up.forward_with_output_size(&x, &[8])?.size(), [1, 1, 8]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn forward_with_output_size(&self, input: &Tensor, output_size: &[i64]) -> Result<Tensor> {
        let shape = spatial_input::<D>(input)?;
        let sizes = if output_size.len() == shape.len() {
            &output_size[shape.len() - D..]
        } else {
            output_size
        };
        if sizes.len() != D {
            return Err(invalid(
                "output_size",
                "must contain spatial sizes or the full output shape",
            ));
        }
        let mut output_padding = [0; D];
        for axis in 0..D {
            // Use i128 for size arithmetic so malformed/extreme options return
            // errors instead of overflowing before the backend can validate them.
            let minimum = (shape[shape.len() - D + axis] as i128 - 1)
                * self.config.stride[axis] as i128
                - 2 * self.config.padding[axis] as i128
                + self.config.dilation[axis] as i128 * (self.config.kernel_size[axis] as i128 - 1)
                + 1;
            let extra = sizes[axis] as i128 - minimum;
            if extra < 0 || extra >= self.config.stride[axis] as i128 {
                return Err(invalid(
                    "output_size",
                    "is outside the range allowed by kernel, stride, padding, and dilation",
                ));
            }
            output_padding[axis] = extra as i64;
        }
        self.apply(input, &output_padding)
    }
    fn apply(&self, input: &Tensor, output_padding: &[i64; D]) -> Result<Tensor> {
        let shape = spatial_input::<D>(input)?;
        ensure_device(
            "transposed convolution weight",
            &self.weight,
            input.device(),
        )?;
        if shape[shape.len() - D - 1] != self.config.in_channels {
            return Err(invalid(
                "input channels",
                "must match the transposed convolution configuration",
            ));
        }
        if let Some(bias) = &self.bias {
            ensure_device("transposed convolution bias", bias, input.device())?;
        }
        let c = self.config;
        Ok(match D {
            1 => input.f_conv_transpose1d(
                &self.weight,
                self.bias.as_ref(),
                c.stride.as_slice(),
                c.padding.as_slice(),
                output_padding.as_slice(),
                c.groups,
                c.dilation.as_slice(),
            )?,
            2 => input.f_conv_transpose2d(
                &self.weight,
                self.bias.as_ref(),
                c.stride.as_slice(),
                c.padding.as_slice(),
                output_padding.as_slice(),
                c.groups,
                c.dilation.as_slice(),
            )?,
            3 => input.f_conv_transpose3d(
                &self.weight,
                self.bias.as_ref(),
                c.stride.as_slice(),
                c.padding.as_slice(),
                output_padding.as_slice(),
                c.groups,
                c.dilation.as_slice(),
            )?,
            _ => {
                return Err(invalid(
                    "convolution dimensions",
                    "must be one, two, or three",
                ));
            }
        })
    }
}

impl<const D: usize> Module for ConvTranspose<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.apply(input, &self.config.output_padding)
    }
}

#[derive(Debug, Clone, Copy)]
struct PoolOptions<const D: usize> {
    kernel: [i64; D],
    stride: [i64; D],
    padding: [i64; D],
    ceil_mode: bool,
}

impl<const D: usize> PoolOptions<D> {
    fn new(kernel: [i64; D]) -> Result<Self> {
        let options = Self {
            kernel,
            stride: kernel,
            padding: [0; D],
            ceil_mode: false,
        };
        options.validate()?;
        Ok(options)
    }
    fn validate(&self) -> Result<()> {
        if !(1..=3).contains(&D) {
            return Err(invalid("pool dimensions", "must be one, two, or three"));
        }
        if self
            .kernel
            .iter()
            .chain(self.stride.iter())
            .any(|v| *v <= 0)
        {
            return Err(invalid("pool kernel/stride", "must be positive per axis"));
        }
        if self
            .padding
            .iter()
            .zip(self.kernel)
            .any(|(p, k)| *p < 0 || *p > k / 2)
        {
            return Err(invalid(
                "pool padding",
                "must be nonnegative and at most half the kernel size per axis",
            ));
        }
        Ok(())
    }
}

/// Keeps the largest activation in each spatial window.
///
/// Useful after image convolutions to downsample while preserving strong
/// features. `D` is `1..=3`. Defaults are stride equal to kernel size, zero
/// padding, dilation one, and floor output sizes. Padding represents negative
/// infinity. Use [`Self::forward_with_indices`] when selected locations matter.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{MaxPool2d, Module}};
/// let pool = MaxPool2d::new([2, 2])?;
/// let image = Tensor::randn([2, 8, 16, 16], (Kind::Float, Device::Cpu));
/// assert_eq!(pool.forward(&image)?.size(), [2, 8, 8, 8]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MaxPool<const D: usize> {
    options: PoolOptions<D>,
    dilation: [i64; D],
}

/// Averages activations in each spatial window to downsample features.
///
/// `D` is `1..=3`. Defaults are stride equal to kernel size, zero padding,
/// floor output sizes, and counting padded zeros in the divisor.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{AvgPool1d, Module}};
/// let pool = AvgPool1d::new([2])?;
/// let sequence = Tensor::ones([1, 3, 8], (Kind::Float, Device::Cpu));
/// assert_eq!(pool.forward(&sequence)?.size(), [1, 3, 4]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AvgPool<const D: usize> {
    options: PoolOptions<D>,
    count_include_pad: bool,
    divisor_override: Option<i64>,
}

macro_rules! pool_setters {
    ($pool:ident) => {
        impl<const D: usize> $pool<D> {
            /// Sets a positive step per axis; validated when forwarding input.
            #[must_use]
            pub const fn stride(mut self, stride: [i64; D]) -> Self {
                self.options.stride = stride;
                self
            }
            /// Sets nonnegative padding at most half the kernel size per axis;
            /// validated when forwarding input.
            #[must_use]
            pub const fn padding(mut self, padding: [i64; D]) -> Self {
                self.options.padding = padding;
                self
            }
            /// Uses ceiling output sizes, retaining partial windows that start inside the input.
            #[must_use]
            pub const fn ceil_mode(mut self, ceil: bool) -> Self {
                self.options.ceil_mode = ceil;
                self
            }
        }
    };
}
pool_setters!(MaxPool);
pool_setters!(AvgPool);

impl<const D: usize> MaxPool<D> {
    /// Creates non-overlapping max pooling; rejects invalid dimensions/kernel sizes.
    pub fn new(kernel_size: [i64; D]) -> Result<Self> {
        Ok(Self {
            options: PoolOptions::new(kernel_size)?,
            dilation: [1; D],
        })
    }
    /// Sets positive spacing between window elements; validated on forward.
    #[must_use]
    pub const fn dilation(mut self, dilation: [i64; D]) -> Self {
        self.dilation = dilation;
        self
    }
    /// Returns pooled values and flattened spatial indices of their source elements.
    ///
    /// ```
    /// use rusttorch::{Tensor, nn::MaxPool1d};
    /// let input = Tensor::from_slice(&[1_f32, 3., 2., 4.]).reshape([1, 1, 4]);
    /// let (values, indices) = MaxPool1d::new([2])?.forward_with_indices(&input)?;
    /// assert_eq!(Vec::<i64>::try_from(&indices.flatten(0, -1))?, [1, 3]);
    /// # Ok::<(), rusttorch::RustTorchError>(())
    /// ```
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        spatial_input::<D>(input)?;
        self.options.validate()?;
        if self.dilation.iter().any(|d| *d <= 0) {
            return Err(invalid("pool dilation", "must be positive per axis"));
        }
        let o = self.options;
        Ok(match D {
            1 => input.f_max_pool1d_with_indices(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                self.dilation.as_slice(),
                o.ceil_mode,
            )?,
            2 => input.f_max_pool2d_with_indices(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                self.dilation.as_slice(),
                o.ceil_mode,
            )?,
            3 => input.f_max_pool3d_with_indices(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                self.dilation.as_slice(),
                o.ceil_mode,
            )?,
            _ => return Err(invalid("pool dimensions", "must be one, two, or three")),
        })
    }
}

impl<const D: usize> Module for MaxPool<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_indices(input)?.0)
    }
}

impl<const D: usize> AvgPool<D> {
    /// Creates non-overlapping average pooling; rejects invalid dimensions/kernel sizes.
    pub fn new(kernel_size: [i64; D]) -> Result<Self> {
        Ok(Self {
            options: PoolOptions::new(kernel_size)?,
            count_include_pad: true,
            divisor_override: None,
        })
    }
    /// Includes padded zeros in the average denominator when enabled (the default).
    #[must_use]
    pub const fn count_include_pad(mut self, count: bool) -> Self {
        self.count_include_pad = count;
        self
    }
    /// Uses a fixed nonzero divisor for 2-D/3-D pooling. `None` restores the
    /// window-element count. Setting a divisor for 1-D pooling returns an error
    /// on forward because that operation has no divisor override.
    #[must_use]
    pub const fn divisor_override(mut self, divisor: Option<i64>) -> Self {
        self.divisor_override = divisor;
        self
    }
}

impl<const D: usize> Module for AvgPool<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        spatial_input::<D>(input)?;
        self.options.validate()?;
        if self.divisor_override == Some(0) || D == 1 && self.divisor_override.is_some() {
            return Err(invalid(
                "divisor_override",
                "must be nonzero and is supported only for 2-D/3-D average pooling",
            ));
        }
        let o = self.options;
        Ok(match D {
            1 => input.f_avg_pool1d(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                o.ceil_mode,
                self.count_include_pad,
            )?,
            2 => input.f_avg_pool2d(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                o.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )?,
            3 => input.f_avg_pool3d(
                o.kernel.as_slice(),
                o.stride.as_slice(),
                o.padding.as_slice(),
                o.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )?,
            _ => return Err(invalid("pool dimensions", "must be one, two, or three")),
        })
    }
}

/// Averages adaptive windows to produce a fixed positive spatial size.
///
/// Use output size `[1, 1]` for global image pooling before a classifier, even
/// when image dimensions vary. `D` is `1..=3`; each target axis must be positive.
/// Resolve any axis you want to retain from `input.size()` before construction.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{AdaptiveAvgPool2d, Module}};
/// let global = AdaptiveAvgPool2d::new([1, 1])?;
/// let features = Tensor::ones([4, 16, 9, 7], (Kind::Float, Device::Cpu));
/// assert_eq!(global.forward(&features)?.size(), [4, 16, 1, 1]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveAvgPool<const D: usize> {
    output_size: [i64; D],
}

/// Keeps maxima from adaptive windows to produce a fixed positive spatial size.
///
/// Useful for converting variable-resolution feature maps to a fixed grid.
/// `D` is `1..=3`; [`Self::forward_with_indices`] also returns source locations.
///
/// ```
/// use rusttorch::{Device, Kind, Tensor, nn::{AdaptiveMaxPool3d, Module}};
/// let pool = AdaptiveMaxPool3d::new([2, 2, 2])?;
/// let volume = Tensor::randn([1, 4, 5, 7, 9], (Kind::Float, Device::Cpu));
/// assert_eq!(pool.forward(&volume)?.size(), [1, 4, 2, 2, 2]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveMaxPool<const D: usize> {
    output_size: [i64; D],
}

macro_rules! adaptive_constructor {
    ($pool:ident) => {
        impl<const D: usize> $pool<D> {
            /// Sets one positive output size per spatial axis; rejects dimensions outside `1..=3`.
            pub fn new(output_size: [i64; D]) -> Result<Self> {
                if !(1..=3).contains(&D) || output_size.iter().any(|s| *s <= 0) {
                    return Err(invalid(
                        "adaptive output_size",
                        "must contain one, two, or three positive spatial sizes",
                    ));
                }
                Ok(Self { output_size })
            }
        }
    };
}
adaptive_constructor!(AdaptiveAvgPool);
adaptive_constructor!(AdaptiveMaxPool);

impl<const D: usize> Module for AdaptiveAvgPool<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        spatial_input::<D>(input)?;
        Ok(match D {
            1 => input.f_adaptive_avg_pool1d(self.output_size.as_slice())?,
            2 => input.f_adaptive_avg_pool2d(self.output_size.as_slice())?,
            3 => input.f_adaptive_avg_pool3d(self.output_size.as_slice())?,
            _ => return Err(invalid("pool dimensions", "must be one, two, or three")),
        })
    }
}

impl<const D: usize> AdaptiveMaxPool<D> {
    /// Returns adaptive maxima and their flattened spatial source indices.
    /// The output and index tensors both have the configured spatial size.
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        spatial_input::<D>(input)?;
        Ok(match D {
            1 => input.f_adaptive_max_pool1d(self.output_size.as_slice())?,
            2 => input.f_adaptive_max_pool2d(self.output_size.as_slice())?,
            3 => input.f_adaptive_max_pool3d(self.output_size.as_slice())?,
            _ => return Err(invalid("pool dimensions", "must be one, two, or three")),
        })
    }
}

impl<const D: usize> Module for AdaptiveMaxPool<D> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_indices(input)?.0)
    }
}

macro_rules! spatial_aliases {
    ($layer:ident, $one:ident, $two:ident, $three:ident) => {
        #[doc = concat!("Sequence variant of [`", stringify!($layer), "`]. See its example for construction and use.")]
        pub type $one = $layer<1>;
        #[doc = concat!("Image variant of [`", stringify!($layer), "`]. See its example for construction and use.")]
        pub type $two = $layer<2>;
        #[doc = concat!("Volume variant of [`", stringify!($layer), "`]. See its example for construction and use.")]
        pub type $three = $layer<3>;
    };
}
spatial_aliases!(
    ConvTranspose,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d
);
spatial_aliases!(MaxPool, MaxPool1d, MaxPool2d, MaxPool3d);
spatial_aliases!(AvgPool, AvgPool1d, AvgPool2d, AvgPool3d);
spatial_aliases!(
    AdaptiveAvgPool,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d
);
spatial_aliases!(
    AdaptiveMaxPool,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d
);
