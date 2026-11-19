use rusttorch::{Result, Tensor};

#[test]
fn dense_dispatch_and_frontend_calls() -> Result<()> {
    let a = Tensor::from_slice(&[-2_f32, 1.]);
    let b = Tensor::from_slice(&[1_f32, 2.]);
    let sum = tch::Tensor::f_add(&a, &b)?;
    let shifted = tch::Tensor::f_add_scalar(&sum, 2.)?;
    assert_eq!(Vec::<f32>::try_from(&shifted)?, [1., 5.]);
    let matrix = tch::Tensor::f_transpose(&shifted.reshape([1, 2]), 0, 1)?;
    assert_eq!(matrix.size(), [2, 1]);

    let relu = rusttorch::nn::functional::relu(&a)?;
    let weights = b.reshape([1, 2]);
    let score = rusttorch::nn::functional::linear(&relu, &weights, None)?;
    assert_eq!(score.double_value(&[0]), 2.);
    Ok(())
}

#[test]
fn runtime_value_encodings() {
    // These values only exercise the Rust representation, including on CPU-only hosts.
    let kinds = [
        tch::Kind::BFloat16,
        tch::Kind::Bool,
        tch::Kind::ComplexDouble,
        tch::Kind::ComplexHalf,
        tch::Kind::ComplexFloat,
        tch::Kind::Half,
        tch::Kind::Float,
        tch::Kind::Double,
        tch::Kind::Float8e4m3fn,
        tch::Kind::Float8e4m3fnuz,
        tch::Kind::Float8e5m2,
        tch::Kind::Float8e5m2fnuz,
        tch::Kind::Int16,
        tch::Kind::Int,
        tch::Kind::Int64,
        tch::Kind::Int8,
        tch::Kind::QInt32,
        tch::Kind::QInt8,
        tch::Kind::QUInt8,
        tch::Kind::Uint8,
    ];
    assert!(kinds.iter().all(|kind| kind.elt_size_in_bytes() > 0));

    let layouts = [
        tch::Layout::Strided,
        tch::Layout::Sparse,
        tch::Layout::SparseCsr,
        tch::Layout::Mkldnn,
        tch::Layout::SparseCsc,
        tch::Layout::SparseBsr,
        tch::Layout::SparseBsc,
    ];
    assert_eq!(layouts.map(|layout| layout.to_i8()), [0, 1, 2, 3, 4, 5, 6]);

    let devices = [
        tch::Device::Cpu,
        tch::Device::Cuda(0),
        tch::Device::Mps,
        tch::Device::Vulkan,
    ];
    assert_eq!(
        devices.map(tch::Device::is_cuda),
        [false, true, false, false]
    );
}
