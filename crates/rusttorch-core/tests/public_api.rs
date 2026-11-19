use rusttorch_core::{Device, DeviceSpec, Kind, Result, Tensor, resolve_device};

#[test]
fn core_exports_the_shared_runtime_contract() -> Result<()> {
    let device = resolve_device(DeviceSpec::Cpu)?;
    assert_eq!(device, Device::Cpu);
    let tensor = Tensor::f_zeros([2], (Kind::Float, device))?;
    assert_eq!(tensor.size(), [2]);
    Ok(())
}

#[test]
fn no_grad_restores_the_calling_threads_mode_after_unwinding() {
    use rusttorch_core::no_grad;
    use std::panic::catch_unwind;

    fn records_gradients() -> bool {
        Tensor::from(2.0)
            .set_requires_grad(true)
            .square()
            .requires_grad()
    }

    assert!(records_gradients());
    let result = catch_unwind(|| {
        no_grad(|| {
            assert!(!records_gradients());
            assert!(std::thread::spawn(records_gradients).join().unwrap());
            let nested = catch_unwind(|| no_grad(|| panic!("nested inference failed")));
            assert_eq!(
                nested.unwrap_err().downcast_ref::<&str>(),
                Some(&"nested inference failed")
            );
            assert!(!records_gradients());
            panic!("inference failed");
        })
    });
    assert_eq!(
        result.unwrap_err().downcast_ref::<&str>(),
        Some(&"inference failed")
    );
    assert!(records_gradients());
}

#[test]
fn with_grad_records_inside_inference_and_restores_after_error_or_panic() {
    use rusttorch_core::{no_grad, with_grad};
    use std::panic::catch_unwind;

    let input = Tensor::from(3.0).set_requires_grad(true);
    no_grad(|| {
        let score = with_grad(|| input.square());
        score.backward();
        assert_eq!(input.grad().double_value(&[]), 6.0);
        assert!(!input.square().requires_grad());
        assert_eq!(with_grad(|| Err::<(), _>("error")), Err("error"));
        assert!(!input.square().requires_grad());

        let unwind = catch_unwind(|| with_grad(|| panic!("gradient calculation failed")));
        assert!(unwind.is_err());
        assert!(!input.square().requires_grad());
    });
    assert!(input.square().requires_grad());
}
