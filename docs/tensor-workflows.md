# Tensor and numerical workflows

These recipes use the same `Tensor` as `tch`, backed by pinned PyTorch/LibTorch
2.13.0. Every block is a complete Rust program, executed as a documentation test.
Put a block in `src/main.rs` after following the
[installation guide](https://github.com/newpoluton-alt/RustTorch#installation).
The numerical evidence here is CPU-only.

The [`tensor` helpers](https://docs.rs/rusttorch/latest/rusttorch/tensor/index.html)
validate task-specific shapes, kinds, devices and undefined inputs. They return
`Result`, allocate outputs and preserve native gradients, except for explicitly
requested shared views, COO coordinate construction and quantization. Native `Tensor::f_*` methods return
LibTorch errors; unprefixed convenience methods can panic. These helpers cover
particular workflows, not the entire Python tensor namespace.

## Select class scores and aggregate contributions

A score matrix is `[samples, classes]`. `select_rows` selects whole samples;
`gather_rows` selects one class per sample. `scatter_add_rows` sums messages
into groups, including repeated IDs. IDs are int64, nonnegative, in bounds and
on the input device. `replace_masked` requires an exact-shape Boolean mask and
replacement tensor, preventing unintended broadcasting. None modify the input.
Gradients through repeated selections accumulate.

```rust
use rusttorch::{Kind, Result, Tensor};
use rusttorch::tensor::{gather_rows, replace_masked, scatter_add_rows, select_rows};

fn main() -> Result<()> {
    let scores = Tensor::f_from_slice(&[1_f64, 2., 3., 4., 5., 6.])?
        .f_reshape([3, 2])?.set_requires_grad(true);
    let chosen = gather_rows(&scores, &Tensor::f_from_slice(&[1_i64, 0, 1])?)?;
    assert_eq!(Vec::<f64>::try_from(chosen)?, [2., 3., 6.]);
    let ids = Tensor::f_from_slice(&[2_i64, 0, 2])?;
    select_rows(&scores, &ids)?.f_sum(Kind::Double)?.f_backward()?;
    assert_eq!(scores.grad().double_value(&[2, 0]), 2.);
    let messages = scatter_add_rows(3, &ids, &scores)?;
    assert_eq!(messages.double_value(&[0, 0]), 3.);
    let clean = replace_masked(&scores, &scores.f_gt(3.)?, &scores.f_zeros_like()?)?;
    assert_eq!(clean.double_value(&[2, 1]), 0.);
    assert_eq!(scores.double_value(&[2, 1]), 6.);
    assert!(select_rows(&scores, &Tensor::f_from_slice(&[-1_i64])?).is_err());
    Ok(())
}
```

Native `f_narrow`, `f_select`, `f_index_select`, `f_gather`, `f_scatter_add` and
`f_index_put` remain available for individual operations. `IndexOp` provides
`i(...)` convenience indexing; prefer fallible native methods for recoverable
errors. Python list indexing and assignment syntax are not emulated.

## Control views, copies and feature normalization

Views and transposes share storage. `reshape` may return a view or allocate,
so it is not an ownership promise. `flatten_features(x, false)` requires a view;
`true` always returns independent storage. Both preserve gradient connections.
`expand` broadcasts with zero strides and shares storage: copy before mutating
an expanded tensor that refers repeatedly to the same storage locations.

`standardize` uses population variance (`correction=0`), retains the reduction
axis for broadcasting, and clamps standard deviation below by epsilon, which
must remain finite and positive in the input dtype. Axis
zero normalizes each feature across samples. Constant features become zero.
Floating conversion is explicit; conversion to integers discards gradient
tracking. Device moves require an available backend and matching input devices.

```rust
use rusttorch::{Device, Kind, Result, Tensor};
use rusttorch::tensor::{flatten_features, standardize};

fn main() -> Result<()> {
    let images = Tensor::f_arange(24, (Kind::Float, Device::Cpu))?.f_reshape([2, 3, 4])?;
    let mut view = flatten_features(&images, false)?;
    let independent = flatten_features(&images, true)?;
    let _ = view.f_fill_(7.)?;
    assert_eq!(images.double_value(&[0, 0, 0]), 7.);
    assert_eq!(independent.double_value(&[0, 0]), 0.);
    let transposed = independent.f_reshape([2, 3, 4])?.f_transpose(1, 2)?;
    assert!(flatten_features(&transposed, false).is_err());
    assert_eq!(flatten_features(&transposed, true)?.size(), [2, 12]);
    let input = Tensor::f_from_slice(&[1_i64, 4, 3, 4])?.f_reshape([2, 2])?
        .f_to_kind(Kind::Double)?.f_to_device(Device::Cpu)?;
    let normalized = standardize(&input, 0, 1e-8)?;
    assert_eq!(normalized.double_value(&[0, 0]), -1.);
    assert_eq!(normalized.double_value(&[0, 1]), 0.);
    let bias = Tensor::f_from_slice(&[0.5_f64, -0.5])?;
    let expanded = bias.f_unsqueeze(0)?.f_expand([2, 2], false)?;
    assert_eq!(expanded.stride(), [0, 1]);
    assert_eq!(normalized.f_add(&expanded)?.f_mean(Kind::Double)?.double_value(&[]), 0.);
    Ok(())
}
```

## Fit coefficients and inspect matrix structure

Use `solve(A, B)` for square nonsingular systems instead of explicitly inverting
`A`. `least_squares` fits possibly rank-deficient CPU matrices with SVD-based
`gelsd`, returning coefficients, residuals, rank and singular values. Residuals
are empty unless the system is overdetermined with full rank. Set `rcond` for
a domain-specific rank cutoff. Batched and complex least squares remain native
APIs outside the helper's two-dimensional real CPU scope.

`eigh("L")` reads the lower triangle of a symmetric matrix, returning ascending
eigenvalues and eigenvectors in columns. The caller supplies a symmetric matrix;
it does not verify the unused triangle. SVD vector signs are not unique: check
reconstruction and singular values, not exact vector entries. The safe `tch`
CPU call is `f_svd(true, true)`, returning **V**, not **Vh**. Its generated
`f_linalg_svd` binding requires a string driver and cannot express PyTorch's
`driver=None`; CPU rejects a driver. This is a documented binding gap.

```rust
use rusttorch::{Result, Tensor};
use rusttorch::tensor::least_squares;

fn main() -> Result<()> {
    let design = Tensor::f_from_slice(&[1_f64, 0., 1., 1., 1., 2.])?.f_reshape([3, 2])?;
    let targets = Tensor::f_from_slice(&[1_f64, 3., 5.])?.f_reshape([3, 1])?;
    let fit = least_squares(&design, &targets, None)?;
    assert!((fit.solution.double_value(&[0, 0]) - 1.).abs() < 1e-10);
    assert!((fit.solution.double_value(&[1, 0]) - 2.).abs() < 1e-10);
    assert_eq!(fit.rank.int64_value(&[]), 2);
    let a = Tensor::f_from_slice(&[4_f64, 1., 1., 3.])?.f_reshape([2, 2])?;
    let b = Tensor::f_from_slice(&[1_f64, 2.])?.f_reshape([2, 1])?;
    let solution = Tensor::f_linalg_solve(&a, &b, true)?;
    assert!(a.f_matmul(&solution)?.allclose(&b, 1e-10, 1e-10, false));
    let (u, s, v) = a.f_svd(true, true)?;
    let reconstructed = u.f_matmul(&s.f_diagflat(0)?)?.f_matmul(&v.f_transpose(0, 1)?)?;
    assert!(reconstructed.allclose(&a, 1e-10, 1e-10, false));
    let (eigenvalues, eigenvectors) = a.f_linalg_eigh("L")?;
    assert!(a.f_matmul(&eigenvectors)?.allclose(
        &eigenvectors.f_mul(&eigenvalues)?, 1e-10, 1e-10, false));
    assert!((Tensor::f_linalg_det(&a)?.double_value(&[]) - 11.).abs() < 1e-10);
    let spectral_norm = a.f_linalg_norm(2., [0, 1].as_slice(), false, None)?;
    assert!((spectral_norm.double_value(&[]) - s.double_value(&[0])).abs() < 1e-10);
    Ok(())
}
```

Native operations preserve autograd where LibTorch implements it. Rank changes,
repeated singular/eigenvalues and singular systems retain native numerical and
gradient limitations; not every decomposition is differentiable at every input.

## Filter real signals and round-trip complex data

RFFT of real length `n` returns `n / 2 + 1` complex bins. IRFFT needs the
original length for odd inputs. `spectral_filter` preserves that length, applies
one real gain per bin, and supports leading batch axes. Filtering is periodic;
it does not pad for linear convolution. Choose gains for the sample rate and
frequency response your application requires.

```rust
use rusttorch::{Device, Kind, Result, Tensor};
use rusttorch::tensor::spectral_filter;

fn main() -> Result<()> {
    let signal = Tensor::f_from_slice(&[1_f64, 2., -1., 0., 3.])?.set_requires_grad(true);
    let identity = Tensor::f_ones([3], (Kind::Double, Device::Cpu))?;
    assert!(spectral_filter(&signal, &identity)?.allclose(&signal, 1e-10, 1e-10, false));
    let filtered = spectral_filter(&signal, &Tensor::f_from_slice(&[1_f64, 0.5, 0.])?)?;
    assert_eq!(filtered.size(), [5]);
    filtered.f_square()?.f_sum(Kind::Double)?.f_backward()?;
    assert!(signal.grad().defined());
    let complex = Tensor::f_complex(&signal, &signal.f_mul_scalar(0.5)?)?;
    let spectrum = complex.f_fft_fft(None, -1, "ortho")?;
    let recovered = spectrum.f_fft_ifft(None, -1, "ortho")?;
    assert!(recovered.allclose(&complex, 1e-10, 1e-10, false));
    assert_eq!(spectrum.f_view_as_real()?.size(), [5, 2]);
    Ok(())
}
```

## Compute tail probabilities and log factorials

`special_ndtr` is the standard normal CDF; `special_log_ndtr` computes its log
stably in the negative tail, where `log(ndtr(x))` can underflow. `gammaln(n + 1)`
computes log factorials for discrete likelihoods without huge factorial values.
Float32/float64 CPU values are in scope, with native domain/NaN/infinity behavior
and gradients.

```rust
use rusttorch::{Kind, Result, Tensor};

fn main() -> Result<()> {
    let z = Tensor::f_from_slice(&[-40_f64, 0., 2.])?.set_requires_grad(true);
    let log_probability = z.f_special_log_ndtr()?;
    assert!(log_probability.double_value(&[0]).is_finite());
    assert!((z.f_special_ndtr()?.double_value(&[1]) - 0.5).abs() < 1e-12);
    log_probability.f_sum(Kind::Double)?.f_backward()?;
    assert!(z.grad().defined());
    let count = Tensor::f_from_slice(&[0_f64, 1., 5.])?;
    let log_factorial = count.f_add_scalar(1.)?.f_special_gammaln()?;
    assert!((log_factorial.double_value(&[2]) - 120_f64.ln()).abs() < 1e-10);
    Ok(())
}
```

## Build sparse matrices and quantize inference data

COO coordinates are `[2, nonzeros]`: row IDs followed by column IDs.
`sparse_coo_matrix` checks bounds before construction and coalesces duplicate
coordinates by addition. Coordinates and values are copied so later input
mutation cannot invalidate the checked matrix. Native sparse constructors can
otherwise accept invalid
coordinates when invariant checking is disabled. Multiply by dense feature
columns for graph aggregation or sparse linear maps. CPU CSR conversion,
densification and multiplication are tested. The C++ coordinate constructor
disconnects value gradients, so the helper rejects values requiring gradients.
Native dense-to-COO conversion and sparse multiplication preserve tested
gradients to the dense source and dense feature inputs. General sparse operators and sparse optimizers remain
outside this scope.

Quantization stores affine 8-bit values for inference. Scale and zero point are
explicit calibration choices: a smaller scale gives finer precision and a
narrower range. Scale must remain finite and positive in float32, the
dequantized output dtype. Out-of-range values saturate. Detach deliberately before
quantizing tensors that require gradients, then dequantize for floating
operations. This helper does not convert models, implement quantized layers,
or provide quantization-aware training. These legacy quantized tensor
constructors are deprecated in the pinned PyTorch 2.13 release.

```rust
use rusttorch::{Kind, Result, Tensor};
use rusttorch::tensor::{quantize_per_tensor, sparse_coo_matrix, sparse_matmul};

fn main() -> Result<()> {
    let coordinates = Tensor::f_from_slice(&[0_i64, 0, 1, 0, 0, 1])?.f_reshape([2, 3])?;
    let values = Tensor::f_from_slice(&[1_f64, 2., 4.])?;
    let matrix = sparse_coo_matrix(&coordinates, &values, [2, 2])?;
    let features = Tensor::f_from_slice(&[2_f64, 3., 5., 7.])?.f_reshape([2, 2])?;
    let output = sparse_matmul(&matrix, &features)?;
    assert_eq!(output.double_value(&[0, 0]), 6.);
    let dense_source = matrix.f_to_dense(None, false)?.set_requires_grad(true);
    let differentiable = dense_source.f_to_sparse_sparse_dim(2)?;
    sparse_matmul(&differentiable, &features)?.f_sum(Kind::Double)?.f_backward()?;
    assert_eq!(dense_source.grad().double_value(&[0, 0]), 5.);
    let csr = matrix.f_to_sparse_csr(None)?;
    assert!(sparse_matmul(&csr, &features)?.allclose(&output, 1e-10, 1e-10, false));
    assert_eq!(csr.f_to_dense(None, false)?.double_value(&[0, 0]), 3.);
    let input = Tensor::f_from_slice(&[-1_f32, 0., 1.])?;
    let packed = quantize_per_tensor(&input, 0.1, 0, Kind::QInt8)?;
    assert_eq!(Vec::<i8>::try_from(packed.f_int_repr()?)?, [-10, 0, 10]);
    assert!(packed.f_dequantize()?.allclose(&input, 1e-6, 1e-6, false));
    Ok(())
}
```

Nested/jagged tensors remain deferred. Generated bindings expose some internal
nested operators but not a supported Rust construction and validation workflow
equivalent to `torch.nested`. Padding plus an explicit mask is the documented
sequence-data route; no unsafe bridge is introduced.

## Evidence and upstream scope

The [Rust checks](https://github.com/newpoluton-alt/RustTorch/blob/main/tests/tensor_workflows.rs)
exercise errors, aliasing, empty batches, odd/even FFT lengths, rank deficiency,
reconstruction, gradients and quantization saturation. The
[Python fixtures](https://github.com/newpoluton-alt/RustTorch/blob/main/tests/python_reference/tensor_workflows.py)
compare CPU outputs and selected gradients against pinned PyTorch 2.13.0. Run
`scripts/run-python-parity.sh` in the configured development environment.
No performance or accelerator parity claim is implied.

Pinned upstream references:
[ATen declarations](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/native_functions.yaml),
[linear algebra](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/linalg/__init__.py),
[FFT](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/fft/__init__.py),
[special functions](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/special/__init__.py),
[sparse](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/sparse/__init__.py)
at tag `v2.13.0`, commit prefix `cf30153`. Helpers and fixture programs are
original compositions of public operations, not copied upstream implementations.
The compatibility ledger records the full pinned revision.
