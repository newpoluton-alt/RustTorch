use rusttorch::{
    Device, Kind, Result, Tensor,
    autograd::{self, GradOptions},
};

fn values(t: &Tensor) -> Vec<f64> {
    Vec::try_from(t.reshape([-1])).unwrap()
}
fn close(t: &Tensor, expected: &[f64]) {
    assert_eq!(t.numel(), expected.len());
    for (a, b) in values(t).iter().zip(expected) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}

#[test]
fn gradients_preserve_leaf_buffers_and_support_higher_orders() -> Result<()> {
    let x = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
    x.square().sum(Kind::Double).backward();
    let before = x.grad().copy();
    let loss = x.f_pow_tensor_scalar(3)?.f_sum(Kind::Double)?;
    let first = autograd::grad(
        &loss,
        &[&x],
        GradOptions {
            create_graph: true,
            ..Default::default()
        },
    )?
    .remove(0);
    close(&first, &[12., 27.]);
    let second =
        autograd::grad(&first.f_sum(Kind::Double)?, &[&x], GradOptions::default())?.remove(0);
    close(&second, &[12., 18.]);
    assert!(x.grad().equal(&before));
    Ok(())
}

#[test]
fn vector_products_and_jacobian_hessian_match_analytic_derivatives() -> Result<()> {
    let x = Tensor::from_slice(&[2_f64, 3.]).set_requires_grad(true);
    let matrix = Tensor::from_slice(&[1_f64, 2., 3., 4.]).reshape([2, 2]);
    let function = |x: &Tensor| -> Result<Tensor> { Ok(matrix.f_matmul(&x.f_square()?)?) };
    let tangent = Tensor::from_slice(&[0.5_f64, -1.]);
    let (out, product) = autograd::jvp(function, &x, &tangent, true)?;
    close(&out, &[22., 48.]);
    close(&product, &[-10., -18.]);
    let higher =
        autograd::grad(&product.f_sum(Kind::Double)?, &[&x], GradOptions::default())?.remove(0);
    close(&higher, &[4., -12.]);
    close(
        &autograd::jacobian(function, &x, false)?,
        &[4., 12., 12., 24.],
    );
    let result = autograd::vjp(&function(&x)?, &x, &tangent, GradOptions::default())?;
    close(&result, &[-10., -18.]);
    let h = autograd::hessian(
        |x| Ok(x.f_pow_tensor_scalar(3)?.f_sum(Kind::Double)?),
        &x,
        false,
    )?;
    close(&h, &[12., 0., 0., 18.]);
    assert!(!h.requires_grad());
    Ok(())
}

#[test]
fn constant_unused_empty_and_multidimensional_derivatives_are_defined() -> Result<()> {
    let x = Tensor::ones([2, 2], (Kind::Double, Device::Cpu));
    let f = |_: &Tensor| Ok(Tensor::ones([3], (Kind::Double, Device::Cpu)));
    let j = autograd::jacobian(f, &x, false)?;
    assert_eq!(j.size(), [3, 2, 2]);
    assert_eq!(j.sum(Kind::Double).double_value(&[]), 0.);
    let (_, product) = autograd::jvp(f, &x, &x, false)?;
    close(&product, &[0., 0., 0.]);
    let h = autograd::hessian(|x| Ok(x.f_sum(Kind::Double)?), &x, false)?;
    assert_eq!(h.size(), [2, 2, 2, 2]);
    assert_eq!(h.sum(Kind::Double).double_value(&[]), 0.);
    let j = autograd::jacobian(|x| Ok(x.f_narrow(0, 0, 0)?), &x, false)?;
    assert_eq!(j.size(), [0, 2, 2, 2]);
    let external = Tensor::ones([1], (Kind::Double, Device::Cpu)).set_requires_grad(true);
    let j = autograd::jacobian(|_| Ok(external.f_square()?), &x, false)?;
    assert_eq!(j.sum(Kind::Double).double_value(&[]), 0.);
    Ok(())
}

#[test]
fn surrogate_derivatives_preserve_forward_values_and_validate_boundaries() -> Result<()> {
    let x = Tensor::from_slice(&[0.2_f64, 1.7]).set_requires_grad(true);
    let y = autograd::with_surrogate_gradient(&x.f_round()?, &x.f_pow_tensor_scalar(3)?)?;
    close(&y, &[0., 2.]);
    let g = autograd::grad(&y.f_sum(Kind::Double)?, &[&x], GradOptions::default())?.remove(0);
    close(&g, &[0.12, 8.67]);
    assert!(autograd::with_surrogate_gradient(&x, &x.f_mul_scalar(f64::INFINITY)?).is_err());
    assert!(autograd::jacobian(|x| Ok(x.copy()), &Tensor::new(), false).is_err());
    assert!(autograd::jacobian(|x| Ok(x.copy()), &Tensor::from_slice(&[1_i64]), false).is_err());
    assert!(
        autograd::jvp(
            |x| Ok(x.copy()),
            &x,
            &Tensor::zeros([1], (Kind::Double, Device::Cpu)),
            false
        )
        .is_err()
    );
    assert!(autograd::grad(&x.square(), &[&x], GradOptions::default()).is_err());
    assert!(autograd::grad(&x.sum(Kind::Double), &[], GradOptions::default()).is_err());
    let unused = Tensor::ones([2], (Kind::Double, Device::Cpu)).set_requires_grad(true);
    assert!(autograd::grad(&x.sum(Kind::Double), &[&unused], GradOptions::default()).is_err());
    Ok(())
}

#[test]
#[ignore = "requires scripts/run-python-parity.sh"]
fn functional_derivatives_match_pinned_python() -> Result<()> {
    let root = std::env::var("RUSTTORCH_PYTHON_REFERENCE_DIR").expect("run the parity script");
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::path::Path::new(&root).join("differentiation.json")).unwrap(),
    )
    .unwrap();
    fn flattened(value: &serde_json::Value, result: &mut Vec<f64>) {
        if let Some(a) = value.as_array() {
            for v in a {
                flattened(v, result);
            }
        } else {
            result.push(value.as_f64().unwrap());
        }
    }
    let compare = |name: &str, actual: Tensor| {
        let mut expected = vec![];
        flattened(&reference[name], &mut expected);
        close(&actual, &expected);
    };
    let x = Tensor::from_slice(&[0.2_f64, -0.7, 1.1]).set_requires_grad(true);
    let function = |x: &Tensor| -> Result<Tensor> {
        Ok(Tensor::f_stack(
            &[
                x.f_sin()?.f_sum(Kind::Double)?,
                x.f_square()?.f_mul(&x.f_exp()?)?.f_sum(Kind::Double)?,
            ],
            0,
        )?)
    };
    compare("jacobian", autograd::jacobian(function, &x, false)?);
    compare(
        "jvp",
        autograd::jvp(
            function,
            &x,
            &Tensor::from_slice(&[0.3_f64, 0.4, -0.2]),
            false,
        )?
        .1,
    );
    compare(
        "vjp",
        autograd::vjp(
            &function(&x)?,
            &x,
            &Tensor::from_slice(&[0.5_f64, -0.8]),
            GradOptions::default(),
        )?,
    );
    compare(
        "hessian",
        autograd::hessian(
            |x| {
                Ok(x.f_sin()?
                    .f_mul(&x.f_roll([1], [0])?)?
                    .f_sum(Kind::Double)?)
            },
            &x,
            false,
        )?,
    );
    let options = GradOptions {
        create_graph: true,
        ..Default::default()
    };
    let first = autograd::grad(
        &x.f_pow_tensor_scalar(4)?.f_sum(Kind::Double)?,
        &[&x],
        options,
    )?
    .remove(0);
    let second = autograd::grad(&first.f_sum(Kind::Double)?, &[&x], options)?.remove(0);
    compare(
        "third",
        autograd::grad(&second.f_sum(Kind::Double)?, &[&x], GradOptions::default())?.remove(0),
    );
    Ok(())
}
