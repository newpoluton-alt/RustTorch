//! An unofficial, eager-first Rust frontend for LibTorch.
//!
//! RustTorch provides fallible neural-network modules, optimizer builders,
//! SafeTensors state exchange, and an optional inspectable graph API. Numerical
//! operations and automatic differentiation remain delegated to [`tch`] and
//! the linked LibTorch runtime.
//!
//! # Quick start
//!
//! ```no_run
//! use rusttorch::nn::Sequential;
//! use rusttorch::{DeviceSpec, Kind, Result, Tensor};
//!
//! fn main() -> Result<()> {
//!     let model = Sequential::builder()
//!         .linear(2, 4)
//!         .relu()
//!         .linear(4, 1)
//!         .build(DeviceSpec::Auto)?;
//!     let input = Tensor::f_zeros([8, 2], (Kind::Float, model.device()))?;
//!     let output = model.forward(&input)?;
//!     assert_eq!(output.size(), [8, 1]);
//!     Ok(())
//! }
//! ```
//!
//! Explicit backend requests fail when unavailable; use [`DeviceSpec::Auto`]
//! only when selecting the best available backend is desired. Model weights
//! are exchanged through `.safetensors` files rather than Python pickle files.

#![warn(missing_docs)]

pub mod device;
pub mod error;
pub mod graph;
pub mod interop;
pub mod nn;
pub mod optim;

pub use device::{DeviceCapabilities, DeviceSpec, available_devices, resolve_device};
pub use error::{Result, RustTorchError};
pub use tch::{Device, Kind, Reduction, Tensor, no_grad, no_grad_guard};

/// Seeds LibTorch's random number generator.
pub fn manual_seed(seed: i64) {
    tch::manual_seed(seed);
}
