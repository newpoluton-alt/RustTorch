//! Stateless nonlinearities and probability transformations.

use super::Module;
use crate::{Result, RustTorchError};
use tch::Tensor;

fn defined(input: &Tensor) -> Result<()> {
    if input.defined() {
        Ok(())
    } else {
        Err(RustTorchError::InvalidConfiguration {
            field: "activation input",
            reason: "must be a defined tensor".to_owned(),
        })
    }
}

fn floating(input: &Tensor) -> Result<()> {
    defined(input)?;
    if input.f_is_floating_point()? {
        Ok(())
    } else {
        Err(RustTorchError::InvalidConfiguration {
            field: "activation dtype",
            reason: "requires a real floating-point tensor".to_owned(),
        })
    }
}

macro_rules! activation {
    ($name:ident, $operation:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// ```
        /// use rusttorch::{Tensor, nn::Module};
        #[doc = concat!("let output = rusttorch::nn::", stringify!($name), ".forward(&Tensor::from_slice(&[-1_f32, 0., 1.]))?;")]
        /// assert_eq!(output.size(), [3]);
        /// # Ok::<(), rusttorch::RustTorchError>(())
        /// ```
        #[derive(Debug, Default, Clone, Copy)]
        pub struct $name;
        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Result<Tensor> { defined(input)?; Ok(input.$operation()?) }
        }
    };
}
activation!(
    Sigmoid,
    f_sigmoid,
    "Maps logits into `[0, 1]` with `1 / (1 + exp(-x))`; useful for binary predictions."
);
activation!(
    Tanh,
    f_tanh,
    "Maps real features into `[-1, 1]`; useful in bounded predictions and recurrent models."
);
activation!(
    SiLU,
    f_silu,
    "Applies `x * sigmoid(x)`, a smooth nonlinearity for hidden image or token features."
);

macro_rules! softmax {
    ($name:ident, $operation:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// Negative dimensions count from the end. Input shape is preserved;
        /// an invalid dimension or unsupported dtype returns an error.
        ///
        /// ```
        /// use rusttorch::{Tensor, nn::Module};
        #[doc = concat!("let layer = rusttorch::nn::", stringify!($name), "::new(-1);")]
        /// let scores = Tensor::from_slice(&[1_f32, 2., 3.]).reshape([1, 3]);
        /// assert_eq!(layer.forward(&scores)?.size(), [1, 3]);
        /// # Ok::<(), rusttorch::RustTorchError>(())
        /// ```
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            dim: i64,
        }
        impl $name {
            /// Selects the class/feature axis. See the layer example.
            pub const fn new(dim: i64) -> Self {
                Self { dim }
            }
        }
        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Result<Tensor> {
                defined(input)?;
                Ok(input.$operation(self.dim, None)?)
            }
        }
    };
}
softmax!(
    Softmax,
    f_softmax,
    "Converts scores to probabilities that sum to one along the selected axis."
);
softmax!(
    LogSoftmax,
    f_log_softmax,
    "Converts scores to stable log probabilities for likelihood-based models."
);

/// Preserves positive features and multiplies nonpositive features by a slope.
///
/// Useful when negative activations should retain a gradient. The default
/// negative slope is `0.01`; custom slopes must be finite.
///
/// ```
/// use rusttorch::{Tensor, nn::{LeakyReLU, Module}};
/// let layer = LeakyReLU::new(0.2)?;
/// let output = layer.forward(&Tensor::from_slice(&[-1_f32, 2.]))?;
/// assert_eq!(Vec::<f32>::try_from(&output)?, [-0.2, 2.]);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl LeakyReLU {
    /// Creates a leaky rectifier; rejects non-finite slopes.
    pub fn new(negative_slope: f64) -> Result<Self> {
        if !negative_slope.is_finite() {
            return Err(RustTorchError::InvalidConfiguration {
                field: "negative_slope",
                reason: "must be finite".to_owned(),
            });
        }
        Ok(Self { negative_slope })
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        floating(input)?;
        Ok(input.f_where_self(&input.f_gt(0.0)?, &input.f_mul_scalar(self.negative_slope)?)?)
    }
}

/// Applies `x` to positive values and `alpha * (exp(x) - 1)` otherwise.
///
/// Use for hidden features with a smooth, saturating negative branch. Alpha
/// defaults to one and must be finite. Output preserves the input shape.
///
/// ```
/// use rusttorch::{Tensor, nn::{ELU, Module}};
/// let output = ELU::new(1.2)?.forward(&Tensor::from_slice(&[-1_f32, 0., 2.]))?;
/// assert!(output.double_value(&[0]) < 0.0);
/// assert_eq!(output.double_value(&[2]), 2.0);
/// # Ok::<(), rusttorch::RustTorchError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    alpha: f64,
}

impl Default for ELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl ELU {
    /// Creates an exponential linear unit; rejects non-finite alpha.
    pub fn new(alpha: f64) -> Result<Self> {
        if !alpha.is_finite() {
            return Err(RustTorchError::InvalidConfiguration {
                field: "alpha",
                reason: "must be finite".to_owned(),
            });
        }
        Ok(Self { alpha })
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        floating(input)?;
        // Clamp before exponentiation so large positive inputs cannot create
        // infinities and NaN gradients in the unselected negative branch.
        let negative = input
            .f_clamp_max(0.0)?
            .f_expm1()?
            .f_mul_scalar(self.alpha)?;
        Ok(input.f_where_self(&input.f_gt(0.0)?, &negative)?)
    }
}
