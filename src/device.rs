//! Runtime device discovery and strict device selection.

use std::sync::OnceLock;

use tch::{Cuda, Device, Kind, Tensor};

use crate::{Result, RustTorchError};

/// A device request that is resolved against the linked LibTorch build.
///
/// New backend requests may be added in future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceSpec {
    /// Selects CUDA device 0, then MPS, then CPU, in that order of availability.
    Auto,
    /// Selects the CPU backend.
    Cpu,
    /// Selects the CUDA device at the given zero-based index.
    Cuda(usize),
    /// Selects Apple's Metal Performance Shaders backend.
    Mps,
}

/// Backends that can actually be used by the current process.
///
/// Obtain this value from [`available_devices`]; future releases may report
/// additional capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceCapabilities {
    /// Whether the CPU backend is available. This is always `true`.
    pub cpu: bool,
    /// Whether at least one CUDA device is usable.
    pub cuda: bool,
    /// Number of usable CUDA devices exposed by LibTorch.
    pub cuda_device_count: usize,
    /// Whether cuDNN is available with the CUDA backend.
    pub cudnn: bool,
    /// Whether an MPS tensor operation succeeds in this process.
    pub mps: bool,
}

/// Probes the linked LibTorch backends without consulting Python at runtime.
pub fn available_devices() -> DeviceCapabilities {
    let cuda = Cuda::is_available();
    let cuda_device_count = if cuda {
        Cuda::device_count().max(0) as usize
    } else {
        0
    };
    DeviceCapabilities {
        cpu: true,
        cuda,
        cuda_device_count,
        cudnn: cuda && Cuda::cudnn_is_available(),
        mps: mps_is_available(),
    }
}

/// Resolves an explicit or automatic device request.
pub fn resolve_device(spec: DeviceSpec) -> Result<Device> {
    let capabilities = available_devices();
    match spec {
        DeviceSpec::Auto if capabilities.cuda => Ok(Device::Cuda(0)),
        DeviceSpec::Auto if capabilities.mps => Ok(Device::Mps),
        DeviceSpec::Auto | DeviceSpec::Cpu => Ok(Device::Cpu),
        DeviceSpec::Cuda(index) if index < capabilities.cuda_device_count => {
            Ok(Device::Cuda(index))
        }
        DeviceSpec::Cuda(index) => Err(RustTorchError::BackendUnavailable {
            backend: "CUDA",
            reason: format!(
                "device {index} was requested, but the linked LibTorch build exposes {} usable CUDA device(s)",
                capabilities.cuda_device_count
            ),
        }),
        DeviceSpec::Mps if capabilities.mps => Ok(Device::Mps),
        DeviceSpec::Mps => Err(RustTorchError::BackendUnavailable {
            backend: "MPS",
            reason:
                "the linked LibTorch build or current machine does not provide a usable MPS device"
                    .to_owned(),
        }),
    }
}

fn mps_is_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Tensor::f_zeros([1], (Kind::Float, Device::Mps))
            .and_then(|tensor| tensor.f_add_scalar(1.0))
            .is_ok()
    })
}

pub(crate) fn ensure_device(
    context: impl Into<String>,
    tensor: &Tensor,
    expected: Device,
) -> Result<()> {
    let actual = tensor.device();
    if actual == expected {
        Ok(())
    } else {
        Err(RustTorchError::DeviceMismatch {
            context: context.into(),
            expected,
            actual,
        })
    }
}
