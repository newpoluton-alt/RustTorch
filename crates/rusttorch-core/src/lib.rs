//! Shared runtime contracts for the RustTorch workspace.

#![deny(missing_docs)]
#![doc = include_str!("../COMPATIBILITY.md")]

pub mod device;
pub mod error;

pub use device::{DeviceCapabilities, DeviceSpec, available_devices, resolve_device};
pub use error::{Result, RustTorchError};
pub use tch::{Device, Kind, Reduction, Tensor, no_grad, no_grad_guard};

/// Seeds LibTorch's random number generator.
pub fn manual_seed(seed: i64) {
    tch::manual_seed(seed);
}
