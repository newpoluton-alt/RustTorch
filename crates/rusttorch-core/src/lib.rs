//! Tensors, automatic differentiation, device selection, and errors for RustTorch.
//!
//! Use this crate when you need tensor calculations without the higher-level
//! model and data-loading APIs. Applications that train neural networks can
//! instead use [`rusttorch`](https://docs.rs/rusttorch), which re-exports these
//! types alongside its layers and optimizers.
//!
//! [`Tensor`] is backed by LibTorch and supports numerical operations, views,
//! reductions, and gradient recording. Fallible methods use the `f_` prefix;
//! their errors convert into the crate's [`Result`] type.
//!
//! # Calculate a gradient
//!
//! Mark leaf tensors as requiring gradients before constructing a calculation.
//! Backpropagating a scalar result fills each leaf tensor's gradient.
//!
//! ```
//! use rusttorch_core::{Kind, Result, Tensor};
//!
//! # fn main() -> Result<()> {
//! let values = Tensor::f_from_slice(&[2_f32, 3.])?.set_requires_grad(true);
//! let squared_sum = values.f_mul(&values)?.f_sum(Kind::Float)?;
//! squared_sum.f_backward()?;
//!
//! let gradient = values.grad();
//! assert_eq!(gradient.double_value(&[0]), 4.0);
//! assert_eq!(gradient.double_value(&[1]), 6.0);
//! # Ok(())
//! # }
//! ```
//!
//! Use [`no_grad`] or [`no_grad_guard`] for calculations that do not need a
//! gradient graph, such as prediction and manual parameter updates.
//!
//! # Select a compute device
//!
//! Keep tensors participating in an operation on the same device.
//! [`DeviceSpec::Auto`] selects CUDA, then MPS, then CPU according to runtime
//! availability. An explicit request returns an error if the device is not
//! available; [`available_devices`] reports the usable backends.
//!
//! ```
//! use rusttorch_core::{DeviceSpec, Kind, Result, Tensor, resolve_device};
//!
//! # fn main() -> Result<()> {
//! let device = resolve_device(DeviceSpec::Cpu)?;
//! let features = Tensor::f_zeros([8, 4], (Kind::Float, device))?;
//! assert_eq!(features.device(), device);
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

pub mod device;
pub mod error;

pub use device::{DeviceCapabilities, DeviceSpec, available_devices, resolve_device};
pub use error::{Result, RustTorchError};
pub use tch::{Device, Kind, Reduction, Tensor, no_grad, no_grad_guard};

/// Seeds LibTorch's random number generator.
pub fn manual_seed(seed: i64) {
    tch::manual_seed(seed);
}
