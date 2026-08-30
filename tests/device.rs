use rusttorch::{Device, DeviceSpec, RustTorchError, available_devices, resolve_device};

#[test]
fn cpu_resolves_explicitly() {
    assert_eq!(
        resolve_device(DeviceSpec::Cpu).expect("CPU must always be available"),
        Device::Cpu,
    );
}

#[test]
fn reported_capabilities_match_resolved_devices() {
    let capabilities = available_devices();

    assert!(capabilities.cpu);
    assert!(!capabilities.cudnn || capabilities.cuda);
    if capabilities.cuda {
        assert!(capabilities.cuda_device_count > 0);
        assert_eq!(
            resolve_device(DeviceSpec::Cuda(0)).expect("reported CUDA device must resolve"),
            Device::Cuda(0),
        );
    } else {
        assert_eq!(capabilities.cuda_device_count, 0);
        assert!(!capabilities.cudnn);
    }

    if capabilities.mps {
        assert_eq!(
            resolve_device(DeviceSpec::Mps).expect("reported MPS device must resolve"),
            Device::Mps,
        );
    }

    let expected_auto = if capabilities.cuda {
        Device::Cuda(0)
    } else if capabilities.mps {
        Device::Mps
    } else {
        Device::Cpu
    };
    assert_eq!(
        resolve_device(DeviceSpec::Auto).expect("automatic device selection must resolve"),
        expected_auto,
    );
}

#[test]
fn unavailable_or_out_of_range_cuda_is_an_error() {
    let capabilities = available_devices();
    let invalid_index = capabilities.cuda_device_count;

    match resolve_device(DeviceSpec::Cuda(invalid_index)) {
        Err(RustTorchError::BackendUnavailable { backend, reason }) => {
            assert_eq!(backend, "CUDA");
            assert!(!reason.is_empty());
        }
        other => panic!(
            "CUDA index {invalid_index} must be rejected when only {} device(s) are reported, got {other:?}",
            capabilities.cuda_device_count,
        ),
    }
}

#[test]
fn explicit_mps_is_resolved_or_rejected_without_fallback() {
    let capabilities = available_devices();

    match (capabilities.mps, resolve_device(DeviceSpec::Mps)) {
        (true, Ok(Device::Mps)) => {}
        (false, Err(RustTorchError::BackendUnavailable { backend, reason })) => {
            assert_eq!(backend, "MPS");
            assert!(!reason.is_empty());
        }
        (_, other) => panic!("MPS resolution contradicted reported capabilities: {other:?}"),
    }
}
