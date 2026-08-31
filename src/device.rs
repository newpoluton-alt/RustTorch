//! Runtime device discovery and strict device selection.

pub use rusttorch_core::device::{
    DeviceCapabilities, DeviceSpec, available_devices, ensure_device, resolve_device,
};
