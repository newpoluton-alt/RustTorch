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
pub use tch::{Device, Kind, Reduction, Tensor, no_grad_guard};

/// Runs a closure with reverse-mode gradient recording disabled on this thread.
///
/// Use this for inference on a tracked input. Nested calls preserve the
/// surrounding mode. The previous mode is restored when the closure returns
/// or unwinds, so later training on this thread can record gradients again.
///
/// ```
/// use rusttorch_core::{Tensor, no_grad};
///
/// let input = Tensor::from(3.0).set_requires_grad(true);
/// let prediction = no_grad(|| input.f_square())?;
/// assert_eq!(prediction.double_value(&[]), 9.0);
/// assert!(!prediction.requires_grad());
/// assert!(input.f_square()?.requires_grad());
/// # Ok::<(), rusttorch_core::RustTorchError>(())
/// ```
pub fn no_grad<T, F: FnOnce() -> T>(f: F) -> T {
    let _guard = no_grad_guard();
    f()
}

/// Enables reverse-mode recording for a calculation on this thread.
///
/// Use this inside [`no_grad`] to compute an input sensitivity while the rest
/// of an evaluation loop runs without a gradient graph. Inputs whose gradients
/// you need must already require gradients. On return or unwind, the calling
/// thread resumes its previous recording mode.
///
/// ```
/// use rusttorch_core::{Result, Tensor, no_grad, with_grad};
///
/// let input = Tensor::from(3.0).set_requires_grad(true);
/// no_grad(|| -> Result<()> {
///     let score = with_grad(|| input.f_square())?;
///     score.f_backward()?;
///     assert_eq!(input.grad().double_value(&[]), 6.0);
///     assert!(!input.f_square()?.requires_grad());
///     Ok(())
/// })?;
/// # Ok::<(), rusttorch_core::RustTorchError>(())
/// ```
pub fn with_grad<T, F: FnOnce() -> T>(f: F) -> T {
    // tch::with_grad restores its mode only on normal return. This guard
    // remembers the caller's mode and restores it on either return path.
    let _restore = no_grad_guard();
    tch::with_grad(f)
}

/// Seeds LibTorch's shared random generator before drawing tensors.
///
/// Use this to reproduce the order of draws on one runtime/device. Other threads
/// share the generator, and matching seeds do not guarantee identical streams on
/// different backends. Dataset samplers have their own explicit seed controls.
///
/// ```
/// use rusttorch_core::{Device, Kind, Tensor, manual_seed};
/// manual_seed(42);
/// let first = Tensor::randn([3], (Kind::Float, Device::Cpu));
/// manual_seed(42);
/// let repeated = Tensor::randn([3], (Kind::Float, Device::Cpu));
/// assert!(first.equal(&repeated));
/// ```
pub fn manual_seed(seed: i64) {
    tch::manual_seed(seed);
}
