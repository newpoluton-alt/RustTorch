use std::{collections::BTreeMap, fs, path::PathBuf};

use rusttorch::{Device, Kind, Result, Tensor, tensor::*};

fn doubles(values: &[f64], shape: &[i64]) -> Result<Tensor> {
    Ok(Tensor::f_from_slice(values)?.f_reshape(shape)?)
}

fn close(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.size(), expected.size());
    assert!(
        actual.allclose(expected, 1e-9, 1e-10, false),
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn indexing_updates_preserve_inputs_and_accumulate_gradients() -> Result<()> {
    let x = doubles(&[1., 2., 3., 4., 5., 6.], &[3, 2])?.set_requires_grad(true);
    let ids = Tensor::f_from_slice(&[2_i64, 0, 2])?;
    let selected = select_rows(&x, &ids)?;
    close(&selected, &doubles(&[5., 6., 1., 2., 5., 6.], &[3, 2])?);
    selected.f_sum(Kind::Double)?.f_backward()?;
    close(&x.grad(), &doubles(&[1., 1., 0., 0., 2., 2.], &[3, 2])?);
    close(
        &gather_rows(&x, &Tensor::f_from_slice(&[1_i64, 0, 1])?)?,
        &doubles(&[2., 3., 6.], &[3])?,
    );
    close(
        &scatter_add_rows(3, &ids, &x)?,
        &doubles(&[3., 4., 0., 0., 6., 8.], &[3, 2])?,
    );
    let replaced = replace_masked(&x, &x.f_gt(3.)?, &x.f_zeros_like()?)?;
    close(&replaced, &doubles(&[1., 2., 3., 0., 0., 0.], &[3, 2])?);
    close(&x, &doubles(&[1., 2., 3., 4., 5., 6.], &[3, 2])?);
    Ok(())
}

#[test]
fn views_copies_broadcasts_and_standardization() -> Result<()> {
    let x = Tensor::f_arange(24, (Kind::Double, Device::Cpu))?.f_reshape([2, 3, 4])?;
    let mut view = flatten_features(&x, false)?;
    let copied = flatten_features(&x, true)?;
    let _ = view.f_fill_(7.)?;
    assert_eq!(x.double_value(&[0, 0, 0]), 7.);
    assert_eq!(copied.double_value(&[0, 0]), 0.);
    let transpose = copied.f_reshape([2, 3, 4])?.f_transpose(1, 2)?;
    assert!(flatten_features(&transpose, false).is_err());
    close(
        &flatten_features(&transpose, true)?,
        &transpose.f_reshape([2, 12])?,
    );
    assert_eq!(
        flatten_features(
            &Tensor::f_zeros([0, 2, 3], (Kind::Float, Device::Cpu))?,
            false
        )?
        .size(),
        [0, 6]
    );
    let bias = doubles(&[1., 2.], &[1, 2])?;
    let broadcast = bias.f_expand([3, 2], false)?;
    assert_eq!(broadcast.stride(), [0, 1]);
    close(
        &standardize(&doubles(&[1., 4., 3., 4.], &[2, 2])?, 0, 1e-8)?,
        &doubles(&[-1., 0., 1., 0.], &[2, 2])?,
    );
    Ok(())
}

#[test]
fn least_squares_handles_rank_deficiency_and_native_linalg() -> Result<()> {
    let design = doubles(&[1., 0., 1., 1., 1., 2.], &[3, 2])?.set_requires_grad(true);
    let targets = doubles(&[1., 3., 5.], &[3, 1])?;
    let fit = least_squares(&design, &targets, None)?;
    close(&fit.solution, &doubles(&[1., 2.], &[2, 1])?);
    assert_eq!(fit.rank.int64_value(&[]), 2);
    fit.solution.f_sum(Kind::Double)?.f_backward()?;
    assert!(design.grad().defined());
    let deficient = doubles(&[1., 1., 2., 2., 3., 3.], &[3, 2])?;
    let fit = least_squares(&deficient, &targets, None)?;
    assert_eq!(fit.rank.int64_value(&[]), 1);
    assert_eq!(fit.residuals.numel(), 0);
    let a = doubles(&[4., 1., 1., 3.], &[2, 2])?;
    let b = doubles(&[1., 2.], &[2, 1])?;
    close(&a.f_matmul(&Tensor::f_linalg_solve(&a, &b, true)?)?, &b);
    let (u, s, v) = a.f_svd(true, true)?;
    close(
        &u.f_matmul(&s.f_diagflat(0)?)?
            .f_matmul(&v.f_transpose(0, 1)?)?,
        &a,
    );
    let (eigenvalues, eigenvectors) = a.f_linalg_eigh("L")?;
    close(
        &a.f_matmul(&eigenvectors)?,
        &eigenvectors.f_mul(&eigenvalues)?,
    );
    assert!((Tensor::f_linalg_det(&a)?.double_value(&[]) - 11.).abs() < 1e-10);
    Ok(())
}

#[test]
fn fft_and_special_functions_keep_native_gradients() -> Result<()> {
    for n in [5, 6] {
        let x = Tensor::f_arange(n, (Kind::Double, Device::Cpu))?.set_requires_grad(true);
        let gain =
            Tensor::f_ones([n / 2 + 1], (Kind::Double, Device::Cpu))?.set_requires_grad(true);
        let filtered = spectral_filter(&x, &gain)?;
        close(&filtered, &x);
        filtered.f_square()?.f_sum(Kind::Double)?.f_backward()?;
        close(&x.grad(), &x.f_mul_scalar(2.)?);
        assert!(gain.grad().defined());
        let complex = Tensor::f_complex(&x, &x.f_mul_scalar(0.5)?)?;
        close(
            &complex
                .f_fft_fft(None, -1, "ortho")?
                .f_fft_ifft(None, -1, "ortho")?,
            &complex,
        );
    }
    let x = doubles(&[-2., 0., 2.], &[3])?.set_requires_grad(true);
    let cdf = x.f_special_ndtr()?;
    cdf.f_sum(Kind::Double)?.f_backward()?;
    let expected = x
        .f_square()?
        .f_mul_scalar(-0.5)?
        .f_exp()?
        .f_div_scalar((2. * std::f64::consts::PI).sqrt())?;
    close(&x.grad(), &expected);
    assert!(
        doubles(&[-40.], &[1])?
            .f_special_log_ndtr()?
            .double_value(&[0])
            .is_finite()
    );
    close(
        &doubles(&[1., 2., 3.], &[3])?.f_special_gammaln()?,
        &doubles(&[0., 0., 2_f64.ln()], &[3])?,
    );
    Ok(())
}

#[test]
fn sparse_coo_csr_and_quantized_boundaries() -> Result<()> {
    let ids = Tensor::f_from_slice(&[0_i64, 0, 1, 0, 0, 1])?.f_reshape([2, 3])?;
    let values = doubles(&[1., 2., 4.], &[3])?;
    let sparse = sparse_coo_matrix(&ids, &values, [2, 2])?;
    let dense = doubles(&[2., 3., 5., 7.], &[2, 2])?.set_requires_grad(true);
    let expected = doubles(&[6., 9., 20., 28.], &[2, 2])?;
    let output = sparse_matmul(&sparse, &dense)?;
    close(&output, &expected);
    output.f_sum(Kind::Double)?.f_backward()?;
    close(&dense.grad(), &doubles(&[3., 3., 4., 4.], &[2, 2])?);
    let dense_source = doubles(&[3., 0., 0., 4.], &[2, 2])?.set_requires_grad(true);
    let differentiable = dense_source.f_to_sparse_sparse_dim(2)?;
    sparse_matmul(&differentiable, &dense)?
        .f_sum(Kind::Double)?
        .f_backward()?;
    close(&dense_source.grad(), &doubles(&[5., 0., 0., 12.], &[2, 2])?);
    let csr = sparse.f_to_sparse_csr(None)?;
    close(&sparse_matmul(&csr, &dense)?, &expected);
    close(
        &csr.f_to_dense(None, false)?,
        &doubles(&[3., 0., 0., 4.], &[2, 2])?,
    );
    let x = Tensor::f_from_slice(&[-100_f32, -0.25, 0., 0.25, 100.])?;
    let quantized = quantize_per_tensor(&x, 0.1, 0, Kind::QInt8)?;
    assert_eq!(
        Vec::<i8>::try_from(quantized.f_int_repr()?)?,
        [-128, -2, 0, 2, 127]
    );
    close(
        &quantized.f_dequantize()?.f_to_kind(Kind::Double)?,
        &doubles(
            &[
                -12.8_f32 as f64,
                -0.2_f32 as f64,
                0.,
                0.2_f32 as f64,
                12.7_f32 as f64,
            ],
            &[5],
        )?,
    );
    Ok(())
}

#[test]
fn malformed_inputs_return_errors_without_mutation() -> Result<()> {
    let undefined = Tensor::new();
    let x = doubles(&[1., 2., 3., 4.], &[2, 2])?;
    let ids = Tensor::f_from_slice(&[0_i64, 1])?;
    for value in [&undefined, &Tensor::from(1_f64)] {
        assert!(select_rows(value, &ids).is_err());
        assert!(gather_rows(value, &ids).is_err());
        assert!(flatten_features(value, false).is_err());
        assert!(standardize(value, 0, 1e-8).is_err());
        assert!(least_squares(value, &x, None).is_err());
        assert!(spectral_filter(value, &x).is_err());
        assert!(sparse_matmul(value, &x).is_err());
    }
    assert!(select_rows(&x, &undefined).is_err());
    assert!(select_rows(&x, &Tensor::f_from_slice(&[-1_i64])?).is_err());
    assert!(select_rows(&x, &Tensor::f_from_slice(&[2_i64])?).is_err());
    assert!(select_rows(&x, &Tensor::f_from_slice(&[0_f32])?).is_err());
    assert!(scatter_add_rows(-1, &ids, &x).is_err());
    assert!(scatter_add_rows(2, &undefined, &x).is_err());
    assert!(scatter_add_rows(2, &ids, &undefined).is_err());
    assert!(replace_masked(&x, &undefined, &x).is_err());
    assert!(replace_masked(&x, &x, &x).is_err());
    assert!(replace_masked(&x, &x.f_gt(0.)?, &undefined).is_err());
    assert!(standardize(&x, 2, 1e-8).is_err());
    assert!(standardize(&x, 0, f64::NAN).is_err());
    assert!(least_squares(&x, &x, Some(f64::NAN)).is_err());
    assert!(least_squares(&x.f_mul_scalar(f64::INFINITY)?, &x, None).is_err());
    assert!(sparse_coo_matrix(&undefined, &ids, [2, 2]).is_err());
    assert!(sparse_coo_matrix(&x.f_to_kind(Kind::Int64)?, &undefined, [2, 2]).is_err());
    assert!(sparse_matmul(&x, &x).is_err());
    let coordinates = Tensor::f_from_slice(&[0_i64, 0])?.f_reshape([2, 1])?;
    assert!(
        sparse_coo_matrix(
            &coordinates,
            &doubles(&[1.], &[1])?.set_requires_grad(true),
            [1, 1]
        )
        .is_err()
    );
    let bad_coordinates = Tensor::f_from_slice(&[2_i64, 0])?.f_reshape([2, 1])?;
    assert!(sparse_coo_matrix(&bad_coordinates, &doubles(&[1.], &[1])?, [1, 1]).is_err());
    let malformed_sparse = Tensor::f_sparse_coo_tensor_indices_size(
        &bad_coordinates,
        &doubles(&[1.], &[1])?,
        [1, 1],
        (Kind::Double, Device::Cpu),
        false,
    )?;
    assert!(sparse_matmul(&malformed_sparse, &doubles(&[1.], &[1, 1])?).is_err());
    assert!(quantize_per_tensor(&undefined, 0.1, 0, Kind::QInt8).is_err());
    let xf = x.f_to_kind(Kind::Float)?;
    for (scale, zero, kind) in [
        (0., 0, Kind::QInt8),
        (f64::NAN, 0, Kind::QInt8),
        (1., 128, Kind::QInt8),
        (1., -1, Kind::QUInt8),
        (1., 0, Kind::Int8),
    ] {
        assert!(quantize_per_tensor(&xf, scale, zero, kind).is_err());
    }
    assert!(quantize_per_tensor(&xf.set_requires_grad(true), 0.1, 0, Kind::QInt8).is_err());
    close(&x, &doubles(&[1., 2., 3., 4.], &[2, 2])?);
    Ok(())
}

fn parity_outputs() -> Result<BTreeMap<&'static str, Tensor>> {
    let mut out = BTreeMap::new();
    let x = doubles(&[1., 2., 3., 4., 5., 6.], &[3, 2])?.set_requires_grad(true);
    let ids = Tensor::f_from_slice(&[2_i64, 0, 2])?;
    out.insert("select", select_rows(&x, &ids)?);
    out.insert(
        "gather",
        gather_rows(&x, &Tensor::f_from_slice(&[1_i64, 0, 1])?)?,
    );
    out.insert("scatter", scatter_add_rows(3, &ids, &x)?);
    out.insert(
        "masked",
        replace_masked(&x, &x.f_gt(3.)?, &x.f_zeros_like()?)?,
    );
    out.insert("standardize", standardize(&x, 0, 1e-8)?);
    out.insert(
        "flatten",
        flatten_features(&x.f_reshape([1, 3, 2])?.f_transpose(1, 2)?, true)?,
    );
    out["select"]
        .f_square()?
        .f_sum(Kind::Double)?
        .f_backward()?;
    out.insert("select_grad", x.grad());
    let a = doubles(&[4., 1., 1., 3.], &[2, 2])?;
    out.insert(
        "solve",
        Tensor::f_linalg_solve(&a, &doubles(&[1., 2.], &[2, 1])?, true)?,
    );
    out.insert("det", Tensor::f_linalg_det(&a)?);
    out.insert("svd_values", a.f_svd(true, true)?.1);
    out.insert("eigenvalues", a.f_linalg_eigh("L")?.0);
    out.insert("norm", a.f_linalg_norm(2., [0, 1].as_slice(), false, None)?);
    let design = doubles(&[1., 0., 1., 1., 1., 2.], &[3, 2])?;
    let fit = least_squares(&design, &doubles(&[1., 2.5, 5.5], &[3, 1])?, None)?;
    out.insert("lstsq", fit.solution);
    out.insert("lstsq_residuals", fit.residuals);
    out.insert("lstsq_rank", fit.rank);
    out.insert("lstsq_singular_values", fit.singular_values);
    let signal = doubles(&[1., 2., -1., 0., 3.], &[5])?.set_requires_grad(true);
    let gain = doubles(&[1., 0.5, 0.], &[3])?;
    out.insert(
        "rfft",
        signal.f_fft_rfft(None, -1, "backward")?.f_view_as_real()?,
    );
    let filtered = spectral_filter(&signal, &gain)?;
    filtered.f_square()?.f_sum(Kind::Double)?.f_backward()?;
    out.insert("filtered", filtered);
    out.insert("filter_grad", signal.grad());
    let complex = Tensor::f_complex(&signal, &signal.f_mul_scalar(0.5)?)?;
    let fft = complex.f_fft_fft(None, -1, "ortho")?;
    out.insert("fft", fft.f_view_as_real()?);
    out.insert("ifft", fft.f_fft_ifft(None, -1, "ortho")?.f_view_as_real()?);
    let z = doubles(&[-3., 0., 2.], &[3])?;
    out.insert("normal_cdf", z.f_special_ndtr()?);
    out.insert("log_normal_cdf", z.f_special_log_ndtr()?);
    out.insert(
        "log_gamma",
        doubles(&[0.5, 1., 3.], &[3])?.f_special_gammaln()?,
    );
    let coords = Tensor::f_from_slice(&[0_i64, 0, 1, 0, 0, 1])?.f_reshape([2, 3])?;
    let values = doubles(&[1., 2., 4.], &[3])?;
    let sparse = sparse_coo_matrix(&coords, &values, [2, 2])?;
    let differentiable_source = doubles(&[3., 0., 0., 4.], &[2, 2])?.set_requires_grad(true);
    let differentiable = differentiable_source.f_to_sparse_sparse_dim(2)?;
    let sparse_output = sparse_matmul(&differentiable, &a)?;
    sparse_output
        .f_square()?
        .f_sum(Kind::Double)?
        .f_backward()?;
    out.insert("sparse_mm", sparse_output);
    out.insert("sparse_grad", differentiable_source.grad());
    out.insert("csr_mm", sparse_matmul(&sparse.f_to_sparse_csr(None)?, &a)?);
    out.insert("sparse_dense", sparse.f_to_dense(None, false)?);
    let q = quantize_per_tensor(
        &Tensor::f_from_slice(&[-100_f32, -0.25, 0., 0.25, 100.])?,
        0.1,
        0,
        Kind::QInt8,
    )?;
    out.insert("quantized", q.f_int_repr()?);
    out.insert("dequantized", q.f_dequantize()?);
    Ok(out)
}

#[test]
#[ignore = "run through scripts/run-python-parity.sh"]
fn tensor_workflows_python_parity() -> Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("RUSTTORCH_PYTHON_REFERENCE_DIR")
            .expect("run through scripts/run-python-parity.sh"),
    );
    let reference: BTreeMap<String, Vec<f64>> = serde_json::from_slice(
        &fs::read(directory.join("tensor_workflows.json")).expect("read tensor fixture"),
    )
    .expect("parse tensor fixture");
    let outputs = parity_outputs()?;
    assert_eq!(outputs.len(), reference.len());
    for (name, tensor) in outputs {
        let expected = Tensor::f_from_slice(&reference[name])?;
        close(&tensor.f_to_kind(Kind::Double)?.f_reshape([-1])?, &expected);
    }
    Ok(())
}

#[test]
fn empty_nested_and_copy_gradient_boundaries() -> Result<()> {
    let huge_empty_ids = Tensor::f_zeros([0], (Kind::Int64, Device::Cpu))?.f_as_strided(
        [i64::MAX, 2, 0],
        [0, 0, 0],
        None,
    )?;
    assert!(
        select_rows(
            &Tensor::f_ones([2], (Kind::Double, Device::Cpu))?,
            &huge_empty_ids
        )
        .is_err()
    );
    let constant = Tensor::f_ones([2], (Kind::Float, Device::Cpu))?;
    assert!(quantize_per_tensor(&constant, 1e-50, 0, Kind::QInt8).is_err());
    assert!(quantize_per_tensor(&constant, 1e300, 0, Kind::QInt8).is_err());
    let mut coordinate = Tensor::f_from_slice(&[0_i64, 0])?.f_reshape([2, 1])?;
    let mut value = doubles(&[2.], &[1])?;
    let independent_sparse = sparse_coo_matrix(&coordinate, &value, [1, 1])?;
    let _ = coordinate.f_fill_(100)?;
    let _ = value.f_fill_(7.)?;
    assert_eq!(
        independent_sparse
            .f_to_dense(None, false)?
            .double_value(&[0, 0]),
        2.
    );
    assert!(standardize(&constant, 0, 1e-50).is_err());
    assert!(standardize(&constant, 0, 1e100).is_err());
    assert!(
        standardize(&constant, 0, f64::from(f32::MIN_POSITIVE))?
            .f_isfinite()?
            .f_all()?
            .int64_value(&[])
            != 0
    );
    let empty = Tensor::f_zeros([0, 2], (Kind::Double, Device::Cpu))?;
    let no_ids = Tensor::f_zeros([0], (Kind::Int64, Device::Cpu))?;
    assert_eq!(select_rows(&empty, &no_ids)?.size(), [0, 2]);
    assert_eq!(scatter_add_rows(0, &no_ids, &empty)?.size(), [0, 2]);
    assert!(standardize(&empty, 0, 1e-8).is_err());
    let padded = Tensor::f_ones([2, 3, 2], (Kind::Double, Device::Cpu))?;
    let sizes = Tensor::f_from_slice(&[2_i64, 2, 3, 2])?.f_reshape([2, 2])?;
    let nested = Tensor::f_internal_nested_from_padded(&padded, &sizes, false)?;
    assert!(flatten_features(&nested, true).is_err());
    assert!(select_rows(&nested, &no_ids).is_err());
    assert!(standardize(&nested, 0, 1e-8).is_err());
    let copied_input = doubles(&[1., 2., 3., 4.], &[1, 2, 2])?.set_requires_grad(true);
    flatten_features(&copied_input, true)?
        .f_square()?
        .f_sum(Kind::Double)?
        .f_backward()?;
    close(&copied_input.grad(), &copied_input.f_mul_scalar(2.)?);
    for kind in [Kind::Float, Kind::Double] {
        let x = Tensor::f_from_slice(&[1_f32, 3., 5.])?.f_to_kind(kind)?;
        assert!(
            standardize(&x, 0, 1e-8)?
                .f_isfinite()?
                .f_all()?
                .int64_value(&[])
                != 0
        );
        let gains = Tensor::f_ones([2], (kind, Device::Cpu))?;
        assert!(spectral_filter(&x, &gains)?.allclose(&x, 1e-5, 1e-6, false));
        assert!((x.f_special_ndtr()?.double_value(&[0]) - 0.841344746).abs() < 1e-6);
    }
    let malformed_csr = Tensor::f_sparse_csr_tensor_crow_col_value_size(
        &Tensor::f_from_slice(&[0_i64, 1, 2])?,
        &Tensor::f_from_slice(&[0_i64, 3])?,
        &doubles(&[1., 1.], &[2])?,
        [2, 2],
        (Kind::Double, Device::Cpu),
    )?;
    assert!(sparse_matmul(&malformed_csr, &doubles(&[1., 1.], &[2, 1])?).is_err());
    Ok(())
}
