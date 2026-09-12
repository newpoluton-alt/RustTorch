//! Validated tensor workflows built from native LibTorch operations.
//!
//! Helpers allocate outputs and preserve native automatic differentiation; none
//! mutate an input. [`flatten_features`] is the explicit exception to independent
//! storage: its `copy` argument selects a view or an independent allocation.
//! COO coordinate construction and quantization are boundaries without gradients. For individual
//! operations, use the existing [`Tensor`] `f_*` methods directly. See the
//! [complete recipes](https://github.com/newpoluton-alt/RustTorch/blob/main/docs/tensor-workflows.md)
//! for linear algebra, broadcasting, complex FFTs, special functions and CSR.

use crate::{Device, Kind, Result, RustTorchError, Tensor};

fn invalid(field: &'static str, reason: impl Into<String>) -> RustTorchError {
    RustTorchError::InvalidConfiguration {
        field,
        reason: reason.into(),
    }
}

fn shape(value: &Tensor, field: &'static str) -> Result<Vec<i64>> {
    if !value.defined() {
        return Err(invalid(field, "tensor must be defined"));
    }
    // Unlike size(), this catches unsupported nested-tensor metadata as an error.
    Ok(Vec::<i64>::try_from(value.f_internal_shape_as_tensor()?)?)
}

fn same_device(left: &Tensor, right: &Tensor, field: &'static str) -> Result<()> {
    if left.device() != right.device() {
        return Err(RustTorchError::DeviceMismatch {
            context: field.into(),
            expected: left.device(),
            actual: right.device(),
        });
    }
    Ok(())
}

fn same_kind(left: &Tensor, right: &Tensor, field: &'static str) -> Result<()> {
    if left.f_kind()? != right.f_kind()? {
        return Err(RustTorchError::DtypeMismatch {
            name: field.into(),
            expected: left.f_kind()?,
            actual: right.f_kind()?,
        });
    }
    Ok(())
}

fn real(value: &Tensor, field: &'static str) -> Result<()> {
    if !matches!(value.f_kind()?, Kind::Float | Kind::Double) {
        return Err(invalid(field, "expected float32 or float64"));
    }
    Ok(())
}

fn indices(value: &Tensor, count: i64, field: &'static str) -> Result<Vec<i64>> {
    let size = shape(value, field)?;
    if value.f_kind()? != Kind::Int64 {
        return Err(invalid(field, "indices must have int64 dtype"));
    }
    if !size.contains(&0)
        && (value.f_min()?.int64_value(&[]) < 0 || value.f_max()?.int64_value(&[]) >= count)
    {
        return Err(invalid(field, format!("indices must be in [0, {count})")));
    }
    Ok(size)
}

/// Select samples from `[samples, ...]` in the order of a one-dimensional ID tensor.
///
/// Repeated IDs are allowed and their gradients accumulate. IDs must be int64,
/// in bounds, and on the input device; negative Python-style IDs are rejected.
/// The result owns new storage; changing it does not change `input`.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::select_rows};
/// # fn main() -> Result<()> {
/// let samples = Tensor::f_from_slice(&[10_f32, 20., 30.])?;
/// let selected = select_rows(&samples, &Tensor::f_from_slice(&[2_i64, 0, 2])?)?;
/// assert_eq!(Vec::<f32>::try_from(selected)?, [30., 10., 30.]);
/// # Ok(()) }
/// ```
pub fn select_rows(input: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let size = shape(input, "select_rows.input")?;
    if size.is_empty() {
        return Err(invalid("select_rows.input", "expected at least one axis"));
    }
    let id_size = indices(ids, size[0], "select_rows.ids")?;
    same_device(input, ids, "select_rows.ids")?;
    if id_size.len() != 1 {
        return Err(invalid("select_rows.ids", "expected one-dimensional IDs"));
    }
    Ok(input.f_index_select(0, ids)?)
}

/// Select one class score per row of a `[batch, classes]` tensor.
///
/// `classes` has shape `[batch]`, int64 dtype, and values in `[0, classes)`.
/// This is useful for action values or the log-probability of observed labels.
/// Gradients flow only to selected scores. No broadcasting or mutation occurs.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::gather_rows};
/// # fn main() -> Result<()> {
/// let scores = Tensor::f_from_slice(&[1_f32, 2., 3., 4.])?.f_reshape([2, 2])?;
/// let chosen = gather_rows(&scores, &Tensor::f_from_slice(&[1_i64, 0])?)?;
/// assert_eq!(Vec::<f32>::try_from(chosen)?, [2., 3.]);
/// # Ok(()) }
/// ```
pub fn gather_rows(input: &Tensor, classes: &Tensor) -> Result<Tensor> {
    let size = shape(input, "gather_rows.input")?;
    if size.len() != 2 {
        return Err(invalid("gather_rows.input", "expected [batch, classes]"));
    }
    let id_size = indices(classes, size[1], "gather_rows.classes")?;
    same_device(input, classes, "gather_rows.classes")?;
    if id_size != [size[0]] {
        return Err(invalid("gather_rows.classes", "expected one ID per row"));
    }
    Ok(input
        .f_gather(1, &classes.f_unsqueeze(1)?, false)?
        .f_squeeze_dim(1)?)
}

/// Sum `[items, ...]` contributions into `rows` output groups.
///
/// IDs have shape `[items]` and int64 dtype. Repeated IDs are added; missing
/// groups are zero. This out-of-place scatter is useful for graph-neighbor
/// aggregation, and preserves gradients to each contribution. Accelerator
/// reductions can have the backend's usual nondeterminism.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::scatter_add_rows};
/// # fn main() -> Result<()> {
/// let messages = Tensor::f_from_slice(&[2_f32, 3., 4.])?;
/// let totals = scatter_add_rows(3, &Tensor::f_from_slice(&[0_i64, 0, 2])?, &messages)?;
/// assert_eq!(Vec::<f32>::try_from(totals)?, [5., 0., 4.]);
/// # Ok(()) }
/// ```
pub fn scatter_add_rows(rows: i64, ids: &Tensor, values: &Tensor) -> Result<Tensor> {
    let mut size = shape(values, "scatter_add_rows.values")?;
    if rows < 0 || size.is_empty() {
        return Err(invalid(
            "scatter_add_rows",
            "rows must be nonnegative and values must have an item axis",
        ));
    }
    let id_size = indices(ids, rows, "scatter_add_rows.ids")?;
    same_device(values, ids, "scatter_add_rows.ids")?;
    if id_size != [size[0]] {
        return Err(invalid("scatter_add_rows.ids", "expected one ID per item"));
    }
    size[0] = rows;
    Ok(Tensor::f_zeros(&size, (values.f_kind()?, values.device()))?.f_index_add(0, ids, values)?)
}

/// Replace masked values without changing the original tensor.
///
/// All three inputs must have identical shapes and devices; `mask` must be
/// Boolean, and replacements must match the input dtype. Requiring exact shapes
/// avoids accidentally broadcasting a sample mask along the wrong axis.
/// Gradients follow the selected input or replacement at each element.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::replace_masked};
/// # fn main() -> Result<()> {
/// let x = Tensor::f_from_slice(&[-1_f32, 2.])?;
/// let clean = replace_masked(&x, &x.f_lt(0.)?, &x.f_zeros_like()?)?;
/// assert_eq!(Vec::<f32>::try_from(clean)?, [0., 2.]);
/// # Ok(()) }
/// ```
pub fn replace_masked(input: &Tensor, mask: &Tensor, replacements: &Tensor) -> Result<Tensor> {
    let size = shape(input, "replace_masked.input")?;
    let mask_size = shape(mask, "replace_masked.mask")?;
    let replacement_size = shape(replacements, "replace_masked.replacements")?;
    same_device(input, mask, "replace_masked.mask")?;
    same_device(input, replacements, "replace_masked.replacements")?;
    same_kind(input, replacements, "replace_masked.replacements")?;
    if size != mask_size || size != replacement_size || mask.f_kind()? != Kind::Bool {
        return Err(invalid(
            "replace_masked",
            "expected equal shapes and a Boolean mask",
        ));
    }
    Ok(replacements.f_where_self(mask, input)?)
}

/// Flatten `[batch, ...features]` into `[batch, features]` with explicit storage.
///
/// `copy = false` requires a view-compatible layout and shares storage; a
/// noncontiguous layout that cannot be viewed returns an error. `copy = true`
/// always allocates independent storage, including when the input is contiguous.
/// Both choices preserve gradients. Empty batches and zero-size feature axes
/// are supported because the feature count is computed without `-1` inference.
///
/// ```
/// use rusttorch::{Device, Kind, Result, Tensor, tensor::flatten_features};
/// # fn main() -> Result<()> {
/// let images = Tensor::f_zeros([2, 3, 4, 4], (Kind::Float, Device::Cpu))?;
/// assert_eq!(flatten_features(&images, false)?.size(), [2, 48]);
/// # Ok(()) }
/// ```
pub fn flatten_features(input: &Tensor, copy: bool) -> Result<Tensor> {
    let size = shape(input, "flatten_features.input")?;
    if size.len() < 2 {
        return Err(invalid(
            "flatten_features.input",
            "expected [batch, ...features]",
        ));
    }
    let features = size[1..]
        .iter()
        .try_fold(1_i64, |n, axis| n.checked_mul(*axis))
        .ok_or_else(|| invalid("flatten_features.input", "feature count overflows int64"))?;
    let target = [size[0], features];
    if copy {
        Ok(input.f_view_copy(target)?)
    } else {
        Ok(input.f_view(target)?)
    }
}

/// Standardize float32/float64 values along one axis using population variance.
///
/// Computes `(x - mean) / max(std, epsilon)` with retained reduction dimensions
/// for broadcasting. Constant features become zero; `epsilon` must be finite
/// and positive after conversion to the input dtype. The chosen axis must exist
/// and contain at least one value.
/// Negative axes are accepted, and gradients remain connected to the input.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::standardize};
/// # fn main() -> Result<()> {
/// let x = Tensor::f_from_slice(&[1_f64, 3.])?;
/// assert_eq!(Vec::<f64>::try_from(standardize(&x, 0, 1e-8)?)?, [-1., 1.]);
/// # Ok(()) }
/// ```
pub fn standardize(input: &Tensor, dim: i64, epsilon: f64) -> Result<Tensor> {
    let size = shape(input, "standardize.input")?;
    real(input, "standardize.input")?;
    let rank = size.len() as i64;
    let effective_epsilon = if input.f_kind()? == Kind::Float {
        f64::from(epsilon as f32)
    } else {
        epsilon
    };
    if !effective_epsilon.is_finite() || effective_epsilon <= 0. || dim < -rank || dim >= rank {
        return Err(invalid(
            "standardize",
            "expected positive finite epsilon representable in the input dtype and a valid axis",
        ));
    }
    let dim = if dim < 0 { dim + rank } else { dim };
    if size[dim as usize] == 0 {
        return Err(invalid(
            "standardize.input",
            "reduction axis must not be empty",
        ));
    }
    let mean = input.f_mean_dim([dim].as_slice(), true, None)?;
    let std = input
        .f_std_dim([dim].as_slice(), false, true)?
        .f_clamp_min(epsilon)?;
    Ok(input.f_sub(&mean)?.f_div(&std)?)
}

/// Fit diagnostics from [`least_squares`], matching the native `gelsd` outputs.
///
/// See [`least_squares`] for a complete fitting example. Empty residual tensors
/// are meaningful: rank-deficient and non-overdetermined systems have none.
#[derive(Debug)]
pub struct LeastSquares {
    /// Coefficients of shape `[features, targets]`.
    pub solution: Tensor,
    /// Squared residuals per target, or an empty tensor when unavailable.
    pub residuals: Tensor,
    /// Scalar numerical rank under the chosen singular-value cutoff.
    pub rank: Tensor,
    /// Singular values of the design matrix in decreasing order.
    pub singular_values: Tensor,
}

/// Fit a possibly rank-deficient CPU system using the SVD-based `gelsd` driver.
///
/// `design` is `[observations, features]`, `targets` is `[observations, targets]`;
/// both must be float32/float64, finite, and on CPU with matching dtype. `rcond`
/// is an optional finite nonnegative rank cutoff; `None` uses the native default.
/// Outputs retain native autograd. For batched/complex systems or GPU drivers,
/// call [`Tensor::f_linalg_lstsq`] directly and choose the backend driver.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::least_squares};
/// # fn main() -> Result<()> {
/// let design = Tensor::f_from_slice(&[1_f64, 0., 1., 1., 1., 2.])?.f_reshape([3, 2])?;
/// let targets = Tensor::f_from_slice(&[1_f64, 3., 5.])?.f_reshape([3, 1])?;
/// let fit = least_squares(&design, &targets, None)?;
/// assert!((fit.solution.double_value(&[1, 0]) - 2.).abs() < 1e-10);
/// # Ok(()) }
/// ```
pub fn least_squares(
    design: &Tensor,
    targets: &Tensor,
    rcond: Option<f64>,
) -> Result<LeastSquares> {
    let a = shape(design, "least_squares.design")?;
    let b = shape(targets, "least_squares.targets")?;
    real(design, "least_squares.design")?;
    same_kind(design, targets, "least_squares.targets")?;
    same_device(design, targets, "least_squares.targets")?;
    if a.len() != 2 || b.len() != 2 || a[0] != b[0] || a.contains(&0) || b.contains(&0) {
        return Err(invalid(
            "least_squares",
            "expected nonempty [observations, features] and [observations, targets]",
        ));
    }
    if design.device() != Device::Cpu || rcond.is_some_and(|r| !r.is_finite() || r < 0.) {
        return Err(invalid(
            "least_squares",
            "expected CPU inputs and a finite nonnegative rcond",
        ));
    }
    if design.f_isfinite()?.f_all()?.int64_value(&[]) == 0
        || targets.f_isfinite()?.f_all()?.int64_value(&[]) == 0
    {
        return Err(invalid("least_squares", "inputs must be finite"));
    }
    let (solution, residuals, rank, singular_values) =
        design.f_linalg_lstsq(targets, rcond, "gelsd")?;
    Ok(LeastSquares {
        solution,
        residuals,
        rank,
        singular_values,
    })
}

/// Filter real signals along their last axis with real frequency-bin gains.
///
/// Input shape is `[..., samples]`; gain shape is exactly `[samples / 2 + 1]`.
/// Both tensors use the same float32/float64 dtype and device. A gain of one
/// reconstructs the input, including odd sample counts; zero removes a bin.
/// Uses the native `backward` FFT normalization and preserves gradients to both
/// input and gains. This is a periodic spectral filter, not padded convolution.
///
/// ```
/// use rusttorch::{Device, Kind, Result, Tensor, tensor::spectral_filter};
/// # fn main() -> Result<()> {
/// let x = Tensor::f_from_slice(&[1_f64, 2., 3., 4., 5.])?;
/// let gains = Tensor::f_ones([3], (Kind::Double, Device::Cpu))?;
/// assert!(spectral_filter(&x, &gains)?.allclose(&x, 1e-10, 1e-10, false));
/// # Ok(()) }
/// ```
pub fn spectral_filter(input: &Tensor, gains: &Tensor) -> Result<Tensor> {
    let size = shape(input, "spectral_filter.input")?;
    let gain_size = shape(gains, "spectral_filter.gains")?;
    real(input, "spectral_filter.input")?;
    same_kind(input, gains, "spectral_filter.gains")?;
    same_device(input, gains, "spectral_filter.gains")?;
    let n = *size
        .last()
        .ok_or_else(|| invalid("spectral_filter.input", "expected a sample axis"))?;
    if n == 0 || gain_size != [n / 2 + 1] {
        return Err(invalid(
            "spectral_filter",
            "expected nonempty signals and [samples / 2 + 1] gains",
        ));
    }
    Ok(input
        .f_fft_rfft(None, -1, "backward")?
        .f_mul(gains)?
        .f_fft_irfft(n, -1, "backward")?)
}

/// Construct a coalesced sparse COO matrix from checked row/column coordinates.
///
/// Coordinates have shape `[2, nonzeros]` and int64 dtype; values have shape
/// `[nonzeros]` with float32/float64 dtype on the same device. Duplicate entries
/// are summed, coordinates must be in bounds, and dimensions may be zero.
/// Explicit uncoalesced construction followed by coalescing gives independent
/// coordinates and values, so later input mutation cannot invalidate the matrix.
/// The native C++ factory disconnects value gradients, so values requiring
/// gradients are rejected. For differentiable sparse construction, convert a
/// dense tensor with [`Tensor::f_to_sparse_sparse_dim`] instead.
///
/// ```
/// use rusttorch::{Result, Tensor, tensor::sparse_coo_matrix};
/// # fn main() -> Result<()> {
/// let ids = Tensor::f_from_slice(&[0_i64, 1, 0, 1])?.f_reshape([2, 2])?;
/// let matrix = sparse_coo_matrix(&ids, &Tensor::f_from_slice(&[2_f32, 3.])?, [2, 2])?;
/// assert!(matrix.is_sparse());
/// # Ok(()) }
/// ```
pub fn sparse_coo_matrix(coordinates: &Tensor, values: &Tensor, size: [i64; 2]) -> Result<Tensor> {
    let ids = shape(coordinates, "sparse_coo_matrix.coordinates")?;
    let data = shape(values, "sparse_coo_matrix.values")?;
    real(values, "sparse_coo_matrix.values")?;
    if values.requires_grad() {
        return Err(invalid(
            "sparse_coo_matrix.values",
            "native coordinate construction disconnects gradients; detach explicitly or convert a dense tensor to COO",
        ));
    }
    same_device(values, coordinates, "sparse_coo_matrix.coordinates")?;
    if size.iter().any(|n| *n < 0)
        || data.len() != 1
        || ids != [2, data[0]]
        || coordinates.f_kind()? != Kind::Int64
    {
        return Err(invalid(
            "sparse_coo_matrix",
            "expected nonnegative dimensions, int64 [2, nnz] coordinates and [nnz] values",
        ));
    }
    indices(
        &coordinates.f_select(0, 0)?,
        size[0],
        "sparse_coo_matrix.rows",
    )?;
    indices(
        &coordinates.f_select(0, 1)?,
        size[1],
        "sparse_coo_matrix.columns",
    )?;
    Ok(Tensor::f_sparse_coo_tensor_indices_size(
        coordinates,
        values,
        size,
        (values.f_kind()?, values.device()),
        false,
    )?
    .f_coalesce()?)
}

/// Multiply a two-dimensional sparse COO/CSR matrix by dense feature columns.
///
/// Input shapes are `[rows, columns]` and `[columns, features]`, with matching
/// float32/float64 dtype and device. Layout errors are returned by LibTorch.
/// COO converted from dense and dense-input gradients are covered by CPU tests; CSR forward
/// conversion/multiplication is covered without claiming full sparse autograd.
///
/// ```
/// use rusttorch::{Device, Kind, Result, Tensor, tensor::{sparse_coo_matrix, sparse_matmul}};
/// # fn main() -> Result<()> {
/// let ids = Tensor::f_from_slice(&[0_i64, 0])?.f_reshape([2, 1])?;
/// let matrix = sparse_coo_matrix(&ids, &Tensor::f_from_slice(&[2_f32])?, [1, 1])?;
/// let y = sparse_matmul(&matrix, &Tensor::f_ones([1, 2], (Kind::Float, Device::Cpu))?)?;
/// assert_eq!(y.size(), [1, 2]);
/// # Ok(()) }
/// ```
pub fn sparse_matmul(matrix: &Tensor, dense: &Tensor) -> Result<Tensor> {
    let a = shape(matrix, "sparse_matmul.matrix")?;
    let b = shape(dense, "sparse_matmul.dense")?;
    real(matrix, "sparse_matmul.matrix")?;
    same_kind(matrix, dense, "sparse_matmul.dense")?;
    same_device(matrix, dense, "sparse_matmul.dense")?;
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(invalid(
            "sparse_matmul",
            "expected [rows, columns] and [columns, features]",
        ));
    }
    if matrix.f_sparse_dim()? != 2 || dense.f_sparse_dim()? != 0 {
        return Err(invalid(
            "sparse_matmul",
            "expected a COO/CSR matrix and dense features",
        ));
    }
    if matrix.is_sparse() {
        let coordinates = matrix.f_internal_indices()?;
        let values = matrix.f_internal_values()?;
        let coordinate_size = shape(&coordinates, "sparse_matmul.coordinates")?;
        let value_size = shape(&values, "sparse_matmul.values")?;
        if value_size.len() != 1 || coordinate_size != [2, value_size[0]] {
            return Err(invalid(
                "sparse_matmul.matrix",
                "invalid COO coordinate/value shapes",
            ));
        }
        indices(&coordinates.f_select(0, 0)?, a[0], "sparse_matmul.rows")?;
        indices(&coordinates.f_select(0, 1)?, a[1], "sparse_matmul.columns")?;
    } else {
        Tensor::f_internal_validate_sparse_csr_tensor_args(
            &matrix.f_crow_indices()?,
            &matrix.f_col_indices()?,
            &matrix.f_values()?,
            &a,
            false,
        )?;
    }
    Ok(Tensor::f_internal_sparse_mm(matrix, dense)?)
}

/// Quantize finite CPU float32 data with explicit per-tensor affine parameters.
///
/// The output kind must be `QInt8` or `QUInt8`; scale must remain finite and
/// positive after float32 conversion (the dequantized output dtype),
/// and zero point must fit the selected integer range. Values saturate to that
/// range and use native rounding. Inputs requiring gradients are
/// rejected: explicitly detach before crossing this inference-only boundary.
/// This does not convert models or provide quantized training/operators. These
/// legacy quantized tensor constructors are deprecated by pinned PyTorch 2.13.
///
/// ```
/// use rusttorch::{Kind, Result, Tensor, tensor::quantize_per_tensor};
/// # fn main() -> Result<()> {
/// let x = Tensor::f_from_slice(&[-1_f32, 0., 1.])?;
/// let packed = quantize_per_tensor(&x, 0.1, 0, Kind::QInt8)?;
/// assert!(packed.f_dequantize()?.allclose(&x, 1e-6, 1e-6, false));
/// # Ok(()) }
/// ```
pub fn quantize_per_tensor(
    input: &Tensor,
    scale: f64,
    zero_point: i64,
    kind: Kind,
) -> Result<Tensor> {
    shape(input, "quantize_per_tensor.input")?;
    let bounds = match kind {
        Kind::QInt8 => -128..=127,
        Kind::QUInt8 => 0..=255,
        _ => {
            return Err(invalid(
                "quantize_per_tensor.kind",
                "expected QInt8 or QUInt8",
            ));
        }
    };
    if input.device() != Device::Cpu || input.f_kind()? != Kind::Float || input.requires_grad() {
        return Err(invalid(
            "quantize_per_tensor.input",
            "expected detached CPU float32 data",
        ));
    }
    let effective_scale = scale as f32;
    if !effective_scale.is_finite() || effective_scale <= 0. || !bounds.contains(&zero_point) {
        return Err(invalid(
            "quantize_per_tensor",
            "expected scale representable as positive finite float32 and a representable zero point",
        ));
    }
    if input.f_isfinite()?.f_all()?.int64_value(&[]) == 0 {
        return Err(invalid(
            "quantize_per_tensor.input",
            "values must be finite",
        ));
    }
    Ok(input.f_quantize_per_tensor(scale, zero_point, kind)?)
}
