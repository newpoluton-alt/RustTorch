use rusttorch_core::{Device, DeviceSpec, Kind, Result, Tensor, resolve_device};

#[test]
fn core_exports_the_shared_runtime_contract() -> Result<()> {
    let device = resolve_device(DeviceSpec::Cpu)?;
    assert_eq!(device, Device::Cpu);
    let tensor = Tensor::f_zeros([2], (Kind::Float, device))?;
    assert_eq!(tensor.size(), [2]);
    Ok(())
}
